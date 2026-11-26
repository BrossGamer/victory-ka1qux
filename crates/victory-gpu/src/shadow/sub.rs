//! Small copies of buffer views' own bytes ([`super::exact`]) as ranges of
//! shared slab buffers (`VICTORY_EXACT_SUB=1`; off by default).
//!
//! With the pool each copy of a view's own bytes took a buffer of 64 KiB at
//! the least, a committed resource of its own, so some 23,000 copies kept
//! 1.4 GiB of buffers where most of them need 256 bytes to 4 KiB (the
//! structured, raw and constant buffer views at +4, +8 or +12 bytes of
//! their memory). Here a copy of 16 KiB or less takes a range of a slab: a
//! buffer of [`SLAB_BYTES`] cut into ranges of one size class, each copy
//! owning one ([`Sub`]) from the day it is made to the day it is let go,
//! with its mapping and its GPU address at the range's offset, and the views
//! made on it counting their elements from there.
//!
//! The ranges lie on 768 bytes, not 256: a constant buffer view needs 256 and
//! a raw view 16, and a typed view of 12-byte elements (`R32G32B32`) needs
//! its first element, counted from the buffer's start, to be a whole
//! number of them, so every range starts on a multiple of 256 and of 12.
//!
//! Who holds a range is told by the reference count of its [`Sub`], not of the
//! slab's buffer, which every range shares: the copy holds one, and so does
//! each descriptor slot that views the range ([`Held`]); a list holds the
//! copy itself (an `Arc<Shadow>`, which holds the range). The range goes
//! back to its slab when the last of them lets go, and a copy is let go
//! from the map of copies only while nothing but the map holds it
//! (`super::exact_in`). A slab nothing holds any more is released after it
//! has been empty for [`IDLE`] askings.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

/// The most bytes (rounded up to 256) a copy may have to take a range of a
/// slab.
pub(crate) const MOST: u64 = 16 << 10;

/// The bytes of a range start on a multiple of this: 256 for constant
/// buffer views and 12 for typed views of three 32-bit components.
const UNIT: u64 = 768;

/// The size classes: bytes of the ranges of each class, multiples of
/// [`UNIT`] (the last holds [`MOST`]).
const CAPS: [u64; 6] = [UNIT, 2 * UNIT, 4 * UNIT, 8 * UNIT, 16 * UNIT, 24 * UNIT];

/// The bytes of a slab (`VICTORY_EXACT_SUB_SLAB_KB`, 4,096 by default).
const SLAB_BYTES: u64 = 4 << 20;

/// The askings an empty slab is kept for before it is released.
const IDLE: u64 = 4096;

/// How often (in askings) [`Pool::trim`] looks for slabs to release.
pub(crate) const TRIM_EVERY: u64 = 64;

/// Whether the copies of single views' bytes of [`MOST`] bytes or less are
/// ranges of slabs (`VICTORY_EXACT_SUB=1`; off by default; with the pool
/// only).
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if let Some(on) = FORCED.with(std::cell::Cell::get) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_EXACT_SUB").is_some_and(|v| v == "1"))
}

#[cfg(test)]
thread_local! {
    /// The switch for this thread's tests.
    static FORCED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Sets the switch for this thread (tests).
#[cfg(test)]
pub(crate) fn force(on: Option<bool>) {
    FORCED.with(|f| f.set(on));
}

fn slab_bytes() -> u64 {
    static BYTES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        let kb = std::env::var("VICTORY_EXACT_SUB_SLAB_KB").ok().and_then(|v| v.trim().parse::<u64>().ok());
        kb.map_or(SLAB_BYTES, |kb| kb << 10).max(CAPS[CAPS.len() - 1])
    })
}

/// Since the start: slabs made, slabs released, ranges cut from a slab's
/// unused part, ranges taken from those given back.
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The class whose ranges hold `bytes` (already rounded to 256), if the copy
/// is small enough for one.
pub(crate) fn class_of(bytes: u64) -> Option<usize> {
    if bytes > MOST {
        return None;
    }
    CAPS.iter().position(|&cap| cap >= bytes)
}

/// The ranges of one slab: those never given out ahead of `bump`, and those
/// given back.
struct Ranges {
    cap: u64,
    /// Where the last range that fits ends.
    limit: u64,
    bump: u64,
    free: Vec<u32>,
    /// Ranges given out and not given back.
    live: u32,
}

impl Ranges {
    fn new(bytes: u64, cap: u64) -> Ranges {
        Ranges { cap, limit: bytes / cap * cap, bump: 0, free: Vec::new(), live: 0 }
    }

    /// The offset of a range given out: the one last given back, or the next
    /// of the unused part; `None` when the slab is full. The second says
    /// whether it was given back before.
    fn take(&mut self) -> Option<(u64, bool)> {
        if let Some(base) = self.free.pop() {
            self.live += 1;
            return Some((u64::from(base), true));
        }
        if self.bump < self.limit {
            let base = self.bump;
            self.bump += self.cap;
            self.live += 1;
            return Some((base, false));
        }
        None
    }

    fn give(&mut self, base: u64) {
        debug_assert!(base < self.bump && base.is_multiple_of(self.cap), "a range of this slab");
        debug_assert!(self.live > 0, "given back once");
        self.free.push(base as u32);
        self.live -= 1;
    }
}

/// A buffer cut into ranges of one size class.
struct Slab {
    buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    bytes: u64,
    cap: u64,
    ranges: Mutex<Ranges>,
    /// The asking at which [`Pool::trim`] first found the slab empty
    /// (0: not found so).
    empty_since: AtomicU64,
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// whose ranges are written by the one thread that holds each, as the
// mapping of any copy is.
unsafe impl Send for Slab {}
// SAFETY: as above.
unsafe impl Sync for Slab {}

/// A range of a slab, given out to one copy: the memory the copy's bytes
/// are in. Given back when the last holder lets go.
pub(crate) struct Sub {
    slab: Arc<Slab>,
    base: u64,
}

impl Sub {
    /// The buffer the range is in (shared with the slab's other ranges).
    pub(crate) fn buffer(&self) -> &ID3D12Resource {
        &self.slab.buffer
    }

    /// The range's first byte, mapped.
    pub(crate) fn data(&self) -> *mut u8 {
        // SAFETY: inside the slab's mapping.
        unsafe { self.slab.data.add(self.base as usize) }
    }

    /// The range's first byte's GPU address.
    pub(crate) fn gpu(&self) -> u64 {
        self.slab.gpu + self.base
    }

    /// The range's offset in its buffer.
    pub(crate) fn base(&self) -> u64 {
        self.base
    }

    /// The bytes of the range.
    pub(crate) fn cap(&self) -> u64 {
        self.slab.cap
    }
}

impl Drop for Sub {
    fn drop(&mut self) {
        self.slab.ranges.lock().unwrap_or_else(|e| e.into_inner()).give(self.base);
    }
}

/// What a descriptor slot holds of the host resource it views: the resource,
/// and for a view of a range of a slab ([`Sub`]) the range, which nothing
/// else may be given while a slot shows it (shaders index heaps directly, so
/// a slot the title has not rewritten may be read long after).
#[derive(Clone)]
pub(crate) struct Held {
    resource: ID3D12Resource,
    lease: Option<Arc<Sub>>,
}

impl Held {
    /// A resource's view of a range of a slab (`lease`) or of all of a
    /// resource (none).
    pub(crate) fn new(resource: ID3D12Resource, lease: Option<Arc<Sub>>) -> Held {
        Held { resource, lease }
    }

    pub(crate) fn resource(&self) -> &ID3D12Resource {
        &self.resource
    }

    /// A number that tells what the slot shows from what another slot
    /// shows: the range's, or the resource's.
    pub(crate) fn raw(&self) -> usize {
        match &self.lease {
            Some(lease) => Arc::as_ptr(lease) as usize,
            None => windows::core::Interface::as_raw(&self.resource) as usize,
        }
    }

    /// Whether it is a view of the memory `copy` is in: its range, if it is a
    /// range of a slab; else its resource.
    pub(crate) fn is_of(&self, copy: &super::Shadow) -> bool {
        match (&self.lease, copy.sub()) {
            (Some(mine), Some(theirs)) => Arc::ptr_eq(mine, theirs),
            (None, None) => windows::core::Interface::as_raw(&self.resource) == windows::core::Interface::as_raw(&copy.buffer),
            _ => false,
        }
    }
}

impl From<ID3D12Resource> for Held {
    fn from(resource: ID3D12Resource) -> Held {
        Held::new(resource, None)
    }
}

/// What the slabs hold now (the log line).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Census {
    pub(crate) slabs: u64,
    pub(crate) slab_bytes: u64,
    /// Ranges given out (the copies alive) and their bytes.
    pub(crate) ranges: u64,
    pub(crate) range_bytes: u64,
    /// Ranges given back and waiting for another copy.
    pub(crate) free: u64,
}

/// The slabs, by size class.
pub(crate) struct Pool {
    classes: [Vec<Arc<Slab>>; CAPS.len()],
    /// The bytes of the slabs made from here (0: [`slab_bytes`]).
    bytes: u64,
}

impl Pool {
    pub(crate) const fn new() -> Pool {
        Pool { classes: [const { Vec::new() }; CAPS.len()], bytes: 0 }
    }

    /// A pool whose slabs have `bytes` bytes (tests).
    #[cfg(test)]
    pub(crate) fn with_slab_bytes(bytes: u64) -> Pool {
        Pool { bytes, ..Pool::new() }
    }

    /// A range for a copy of `bytes` (rounded to 256) bytes, from the oldest
    /// slab of its class with room, or a new slab; `None` if the copy is too
    /// big for a range or no slab can be made.
    pub(crate) fn take(&mut self, bytes: u64) -> Option<Arc<Sub>> {
        let class = class_of(bytes)?;
        for slab in &self.classes[class] {
            let taken = slab.ranges.lock().unwrap_or_else(|e| e.into_inner()).take();
            if let Some((base, again)) = taken {
                STATS[2 + usize::from(again)].fetch_add(1, Ordering::Relaxed);
                slab.empty_since.store(0, Ordering::Relaxed);
                return Some(Arc::new(Sub { slab: slab.clone(), base }));
            }
        }
        let size = if self.bytes == 0 { slab_bytes() } else { self.bytes };
        let (buffer, data) = crate::host::host()?.shared_buffer(size)?;
        crate::host::name(&buffer, || format!("title view slab of {}-byte ranges", CAPS[class]));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let mut ranges = Ranges::new(size, CAPS[class]);
        let (base, _) = ranges.take()?;
        let slab = Arc::new(Slab {
            buffer,
            data,
            gpu,
            bytes: size,
            cap: CAPS[class],
            ranges: Mutex::new(ranges),
            empty_since: AtomicU64::new(0),
        });
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[2].fetch_add(1, Ordering::Relaxed);
        self.classes[class].push(slab.clone());
        Some(Arc::new(Sub { slab, base }))
    }

    /// Releases the slabs found empty at two lookings `IDLE` askings apart
    /// (`now` is the asking), and returns how many went.
    pub(crate) fn trim(&mut self, now: u64) -> usize {
        let mut released = 0;
        for slabs in &mut self.classes {
            slabs.retain(|slab| {
                let empty = slab.ranges.lock().unwrap_or_else(|e| e.into_inner()).live == 0;
                if !empty {
                    slab.empty_since.store(0, Ordering::Relaxed);
                    return true;
                }
                let since = slab.empty_since.load(Ordering::Relaxed);
                if since == 0 {
                    slab.empty_since.store(now.max(1), Ordering::Relaxed);
                    return true;
                }
                // (Nothing holds a range, so nothing but the pool holds the slab.)
                let keep = now < since + IDLE || Arc::strong_count(slab) > 1;
                if !keep {
                    released += 1;
                    STATS[1].fetch_add(1, Ordering::Relaxed);
                }
                keep
            });
        }
        released
    }

    /// The slabs and their ranges now.
    pub(crate) fn census(&self) -> Census {
        let mut census = Census::default();
        for slab in self.classes.iter().flatten() {
            let ranges = slab.ranges.lock().unwrap_or_else(|e| e.into_inner());
            census.slabs += 1;
            census.slab_bytes += slab.bytes;
            census.ranges += u64::from(ranges.live);
            census.range_bytes += u64::from(ranges.live) * slab.cap;
            census.free += ranges.free.len() as u64;
        }
        census
    }
}

/// Slabs made, slabs released, ranges cut from unused parts and ranges
/// given back and taken again, since the start.
pub(crate) fn stats() -> [u64; 4] {
    [0, 1, 2, 3].map(|i| STATS[i].load(Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::core::Interface;

    /// The classes hold what they are for, on lines the views need, and no
    /// bigger than twice what is asked (but for the smallest).
    #[test]
    fn the_classes_hold_their_copies_on_the_lines_views_need() {
        assert_eq!(class_of(256), Some(0));
        assert_eq!(class_of(768), Some(0));
        assert_eq!(class_of(769 + 255), Some(1));
        assert_eq!(class_of(4096), Some(3));
        assert_eq!(class_of(MOST), Some(CAPS.len() - 1));
        assert_eq!(class_of(MOST + 256), None);
        for cap in CAPS {
            assert_eq!(cap % 256, 0, "a constant buffer view starts on 256");
            assert_eq!(cap % 12, 0, "a typed view of 12-byte elements starts on one");
            assert_eq!(cap % 16, 0, "a raw view starts on 16");
        }
        assert!(CAPS[CAPS.len() - 1] >= MOST);
        for bytes in (256..=MOST).step_by(256) {
            let cap = CAPS[class_of(bytes).unwrap()];
            assert!(cap >= bytes && (cap <= 2 * bytes || cap == UNIT), "{bytes} bytes in a range of {cap}");
        }
    }

    /// Ranges are cut one after the other, given back and taken again most
    /// recently given first, and never given out twice.
    #[test]
    fn ranges_are_given_out_once_and_taken_again_when_given_back() {
        let cap = 2 * UNIT;
        let mut ranges = Ranges::new(10 * cap + 5, cap);
        let mut got = Vec::new();
        while let Some((base, again)) = ranges.take() {
            assert!(!again);
            got.push(base);
        }
        assert_eq!(got, (0..10).map(|i| i * cap).collect::<Vec<_>>());
        assert_eq!(ranges.live, 10);
        ranges.give(3 * cap);
        ranges.give(7 * cap);
        assert_eq!(ranges.live, 8);
        assert_eq!(ranges.take(), Some((7 * cap, true)));
        assert_eq!(ranges.take(), Some((3 * cap, true)));
        assert_eq!(ranges.take(), None, "full");
        assert_eq!(ranges.live, 10);
    }

    fn device() -> bool {
        crate::host::host().is_some()
    }

    /// A pool gives ranges of a size class on one slab until it is full,
    /// then makes another; a range goes back to its slab when its lease is
    /// dropped and the next copy of the class takes it; each range has its
    /// own mapping and GPU address in the slab.
    #[test]
    fn a_pool_gives_ranges_in_classes_and_takes_them_back() {
        if !device() {
            return; // no host device to make buffers on
        }
        let slab = 64 << 10;
        let mut pool = Pool::with_slab_bytes(slab);
        // Class 0 (768 bytes): 85 ranges in a slab of 64 KiB.
        let per = slab / UNIT;
        let first: Vec<Arc<Sub>> = (0..per).map(|_| pool.take(256).unwrap()).collect();
        assert_eq!(pool.census().slabs, 1);
        let bases: std::collections::HashSet<u64> = first.iter().map(|s| s.base()).collect();
        assert_eq!(bases.len() as u64, per, "every range its own");
        assert!(first.iter().all(|s| s.base() % UNIT == 0 && s.cap() == UNIT && s.gpu() % 256 == 0));
        assert!(first.iter().all(|s| s.buffer().as_raw() == first[0].buffer().as_raw()));
        assert_eq!(first[1].gpu(), first[0].gpu() + UNIT);
        // SAFETY: both inside the slab's mapping.
        assert_eq!(unsafe { first[1].data().offset_from(first[0].data()) }, UNIT as isize);
        // The next goes to a new slab, a bigger copy to a slab of its class.
        let spill = pool.take(768).unwrap();
        let big = pool.take(4096).unwrap();
        assert_eq!(pool.census(), Census { slabs: 3, slab_bytes: 3 * slab, ranges: per + 2, range_bytes: per * UNIT + UNIT + 8 * UNIT, free: 0 });
        assert!(pool.take(MOST + 256).is_none(), "too big for a range");
        // Given back, a range is taken again before a slab is made.
        let (base, gpu) = (first[5].base(), first[5].gpu());
        let mut first = first;
        first.remove(5);
        let census = pool.census();
        assert_eq!((census.ranges, census.free), (per + 1, 1));
        let again = pool.take(500).unwrap();
        assert_eq!((again.base(), again.gpu()), (base, gpu));
        assert_eq!(pool.census().slabs, 3);
        drop((spill, big, again, first));
        let census = pool.census();
        assert_eq!((census.ranges, census.range_bytes), (0, 0));
        assert_eq!(census.free, per + 2);
    }

    /// An empty slab is released once it has been found empty for a while,
    /// a slab with a range given out never is, and one that is taken from
    /// again is not.
    #[test]
    fn empty_slabs_are_released_when_idle() {
        if !device() {
            return; // no host device to make buffers on
        }
        let mut pool = Pool::with_slab_bytes(64 << 10);
        let held = pool.take(256).unwrap();
        let other = pool.take(8192).unwrap();
        assert_eq!(pool.census().slabs, 2);
        drop(other);
        assert_eq!(pool.trim(10), 0, "first found empty");
        assert_eq!(pool.trim(10 + IDLE - 1), 0, "not yet");
        // Taken from again: no longer empty.
        let again = pool.take(8000).unwrap();
        assert_eq!(pool.trim(10 + IDLE), 0);
        drop(again);
        assert_eq!(pool.trim(20 + IDLE), 0, "found empty again");
        assert_eq!(pool.trim(20 + 2 * IDLE), 1, "released");
        assert_eq!(pool.census().slabs, 1);
        assert_eq!(pool.trim(1_000_000), 0, "the slab with a range given out stays");
        assert_eq!(pool.census().slabs, 1);
        drop(held);
    }

    use super::super::{Exact, Shadow, exact_in};

    /// One asking for the copy of 12 bytes at `base + i`, as a range of a
    /// slab, whose bytes are then written to say whose they are.
    fn ask(exact: &mut Exact, fast: bool, base: u64, i: u64) -> Arc<Shadow> {
        let copy = exact_in(exact, base + i, 12, true, fast, true).unwrap();
        assert!(copy.sub().is_some(), "a range of a slab");
        let mut name = [0u8; 12];
        name[..8].copy_from_slice(&i.to_le_bytes());
        name[8..].copy_from_slice(&(!(i as u32)).to_le_bytes());
        // SAFETY: the copy's mapped bytes, 12 of them, written by this
        // thread alone (nothing refreshes it).
        unsafe { std::ptr::copy_nonoverlapping(name.as_ptr(), copy.data, 12) };
        copy
    }

    /// The bytes `ask` wrote: its `i` and the complement of it.
    fn name_of(copy: &Shadow) -> (u64, u32) {
        // SAFETY: the copy's mapped bytes, 12 of them.
        let bytes = unsafe { std::slice::from_raw_parts(copy.data, 12) };
        (u64::from_le_bytes(bytes[..8].try_into().unwrap()), u32::from_le_bytes(bytes[8..].try_into().unwrap()))
    }

    /// The buffer and offset of a copy's range.
    fn range(copy: &Shadow) -> (usize, u64) {
        let sub = copy.sub().unwrap();
        (sub.buffer().as_raw() as usize, sub.base())
    }

    /// A range held by a list (the `Arc` of its copy) or by a slot (its
    /// [`Held`]) is never given to another copy, and no two copies alive
    /// share one; the idle ones go, their ranges are used again, and once
    /// nothing holds the others they go too (`fast`: the queue; else the
    /// scan).
    fn held_ranges_are_not_reused(fast: bool) {
        use std::collections::{HashMap, HashSet};
        if !device() {
            return; // no host device to make buffers on
        }
        let title = vec![0u8; 4096];
        let base = title.as_ptr() as u64;
        let key = |i: u64| (base + i, 12u64);
        let mut exact = Exact::new();
        exact.subs = Pool::with_slab_bytes(64 << 10);
        // Of each range, the copies that ever had it.
        let mut owners: HashMap<(usize, u64), HashSet<(u64, u64)>> = HashMap::new();
        let mut note = |copy: &Arc<Shadow>, i: u64| {
            owners.entry(range(copy)).or_default().insert(key(i));
        };
        let (mut listed, mut slot) = (None, None);
        for i in 0..300 {
            let copy = ask(&mut exact, fast, base, i);
            note(&copy, i);
            match i {
                0 => listed = Some(copy),
                1 => slot = Some(copy.held()),
                _ => {}
            }
        }
        // Many more askings than a copy stays for, copy 2 among them asked
        // for again now and then.
        for i in 300..1500 {
            if i % 50 == 0 {
                let copy = ask(&mut exact, fast, base, 2);
                note(&copy, 2);
            }
            let copy = ask(&mut exact, fast, base, i);
            note(&copy, i);
        }
        let listed_copy = listed.as_ref().unwrap();
        assert!(Arc::ptr_eq(&exact.map[&key(0)].0, listed_copy), "held by a list");
        let slotted = &exact.map[&key(1)].0;
        assert!(slot.as_ref().unwrap().is_of(slotted), "held by a slot, the copy of the map");
        assert!(exact.map.contains_key(&key(2)), "asked for lately");
        assert_eq!(name_of(listed_copy), (0, !0u32), "the list's copy kept its bytes");
        assert_eq!(name_of(slotted), (1, !1u32), "the slot's copy kept its bytes");
        assert_eq!(owners[&range(listed_copy)], HashSet::from([key(0)]), "no other copy was given the list's range");
        assert_eq!(owners[&range(slotted)], HashSet::from([key(1)]), "no other copy was given the slot's range");
        // The copies alive hold a range each, and their bytes are their own.
        let alive: HashSet<(usize, u64)> = exact.map.values().map(|(copy, _)| range(copy)).collect();
        assert_eq!(alive.len(), exact.map.len(), "no two copies share a range");
        for (copy, _) in exact.map.values() {
            let i = copy.span().0 - base;
            assert_eq!(name_of(copy), (i, !(i as u32)), "a copy's bytes are its own");
        }
        assert!((3..300).all(|i| !exact.map.contains_key(&key(i))), "the idle ones were let go");
        let census = exact.subs.census();
        assert_eq!(census.ranges, exact.map.len() as u64, "a range for each copy alive");
        assert!(census.slabs * 85 < 1500, "{} slabs for 1,500 copies: the ranges let go were not used again", census.slabs);
        assert!(stats()[3] > 0);
        // Nothing holds them now, and copy 2 is not asked for again.
        drop(listed);
        drop(slot);
        for i in 1500..3000 {
            ask(&mut exact, fast, base, i);
        }
        assert!((0..3).all(|i| !exact.map.contains_key(&key(i))), "gone once nothing holds them");
        assert_eq!(exact.subs.census().ranges, exact.map.len() as u64);
        drop(title);
    }

    #[test]
    fn with_the_queue_held_ranges_are_not_reused() {
        held_ranges_are_not_reused(true);
    }

    #[test]
    fn with_the_scan_held_ranges_are_not_reused() {
        held_ranges_are_not_reused(false);
    }

    /// A range goes back to its slab when the last holder lets go: the copy
    /// the map lets go, and the slot that shows it.
    #[test]
    fn a_range_goes_back_when_the_last_holder_lets_go() {
        if !device() {
            return; // no host device to make buffers on
        }
        let title = vec![0u8; 4096];
        let base = title.as_ptr() as u64;
        let mut exact = Exact::new();
        exact.subs = Pool::with_slab_bytes(64 << 10);
        let copy = ask(&mut exact, false, base, 0);
        let slot = copy.held();
        assert_eq!(Arc::strong_count(copy.sub().unwrap()), 2, "the copy and the slot");
        assert!(copy.held_apart(), "a slot shows it");
        assert!(slot.is_of(&copy));
        let other = ask(&mut exact, false, base, 1);
        assert!(!other.held_apart());
        assert!(!slot.is_of(&other), "another range of the same buffer");
        assert_eq!(slot.resource().as_raw(), other.buffer.as_raw(), "the same buffer");
        assert_ne!(slot.raw(), other.held().raw());
        // The map lets the copies go; the slot keeps its range.
        exact.map.clear();
        exact.queue.clear();
        drop((copy, other));
        let census = exact.subs.census();
        assert_eq!((census.ranges, census.free), (1, 1), "the slot's range is still out");
        let taken = exact.subs.take(256).unwrap();
        assert_ne!(taken.base(), 0, "not the slot's range");
        drop(taken);
        drop(slot);
        let census = exact.subs.census();
        assert_eq!((census.ranges, census.free), (0, 2), "given back with the slot");
        drop(title);
    }

    /// The card reads a copy that is a range of a slab where the range is:
    /// by a copy of the buffer at the copy's offset, and by a shader at its
    /// GPU address, which also writes into the copy and into none of its
    /// neighbours.
    #[test]
    fn the_card_reads_and_writes_a_copy_at_its_ranges_offset() {
        let Some(host) = crate::host::host() else {
            return; // no host device
        };
        let Ok(mut rig) = crate::inplace::Rig::new(&host.device) else {
            return;
        };
        let mut title: Vec<u8> = (0..4096u32).map(|i| (i * 7 + 3) as u8).collect();
        // Two dwords for the shader to add, at 2048.
        title[2048..2052].copy_from_slice(&1000u32.to_le_bytes());
        title[2052..2056].copy_from_slice(&234u32.to_le_bytes());
        let base = title.as_ptr() as u64;
        let mut exact = Exact::new();
        exact.subs = Pool::with_slab_bytes(64 << 10);
        // Copies of different sizes of one class: ranges of one slab.
        let wanted: [(u64, u64); 4] = [(0, 36), (100, 300), (1000, 700), (2048, 64)];
        let copies: Vec<_> = wanted
            .iter()
            .map(|&(at, len)| {
                let copy = exact_in(&mut exact, base + at, len, true, false, true).unwrap();
                assert!(copy.update((base + at) as usize, len as usize) > 0);
                copy
            })
            .collect();
        let bases: Vec<u64> = copies.iter().map(|c| c.sub().unwrap().base()).collect();
        assert_eq!(bases, [0, UNIT, 2 * UNIT, 3 * UNIT], "one slab, a range each");
        for (copy, &(at, len)) in copies.iter().zip(&wanted) {
            let start = base + at;
            let sub = copy.sub().unwrap();
            // SAFETY: a live buffer.
            let slab_gpu = unsafe { copy.buffer.GetGPUVirtualAddress() };
            assert_eq!((copy.offset(start), copy.gpu(start)), (sub.base(), slab_gpu + sub.base()));
            assert_eq!(copy.offset(start + 8), sub.base() + 8);
            assert_eq!(copy.gpu(start) % 256, 0);
            let got = rig.fetch(&copy.buffer, copy.offset(start), len as usize).unwrap();
            assert_eq!(got, &title[at as usize..(at + len) as usize], "what the card reads at the range of {at}+{len}");
        }
        // A shader adds the two dwords into the third (the copy's bytes 8 to
        // 12) at the copy's GPU address.
        let last = &copies[3];
        rig.shade(last.gpu(base + 2048)).unwrap();
        // SAFETY: the copy's mapping, which the card is done with.
        let sum = unsafe { std::ptr::read_unaligned(last.data.add(8).cast::<u32>()) };
        assert_eq!(sum, 1234, "written at the range");
        // The neighbours are as they were.
        for (copy, &(at, len)) in copies.iter().zip(&wanted).take(3) {
            let got = rig.fetch(&copy.buffer, copy.offset(base + at), len as usize).unwrap();
            assert_eq!(got, &title[at as usize..(at + len) as usize], "the neighbour at {at}+{len}");
        }
        drop(title);
    }
}
