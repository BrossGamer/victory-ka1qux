//! The title's writes of descriptor records into its shader-visible heap say
//! which slots they wrote, and the per-frame look at every changed heap page
//! goes (`VICTORY_PORT_DESCRIPTORS=1`, off by default; read in [`on`]).
//!
//! After avatarmanz/VIctory-7900XTX ("Renderer port 2", GPL-3.0), fitted to
//! what this tree has. What the layer did before, with the title's heap
//! placed in memory it owns (`descriptor::sync_heaps`): the write watch's
//! changed 64 KiB pages of the heap, every record on them read and compared
//! with the one last seen: some 290 pages and 600,000 slots a frame, 50 to
//! 60 ms of the sync here, a good part of it asking the watch.
//!
//! What the title does is told by the user's profile and not built in
//! (`VICTORY_DESCRIPTORS_PROFILE`; `apps/victory`'s `descriptor_profile`):
//! the heap is the memory of its frame allocator, which hands out the draws'
//! and dispatches' descriptor tables (and much else), and the tables are
//! filled one 32-byte record at a time by a few stores in two functions. Views
//! made by our own calls (`Create*View`, `CopyDescriptors*`) are mirrored where
//! they are made (`descriptor::mirror`).
//!
//! What is done here: those stores are patched (`victory_loader::descriptor_sites`)
//! to set a byte per slot and per 128 slots in maps of ours ([`MARKS`]), the
//! title's `memcpy` and `memmove` mark the view records they copy into the heap
//! ([`import`]), and our own mirroring marks too ([`mark`]). `sync_heaps` takes
//! the marked slots ([`take`]) in place of the changed pages and compares
//! and mirrors only those, as before: the batch of
//! [`crate::descriptor::host::batched`] reads their runs where they are
//! (`VICTORY_SYNC_IN_PLACE=1`) or through a copy, compares each record with the
//! one last seen (a slot written with the record it had costs nothing more)
//! and mirrors what changed.
//!
//! Correctness first, the fallback: every [`check_every`] frames the write watch
//! is asked, and a few of the pages it says were written ([`check_budget`] a
//! frame, round the heap) are looked at whole, as before. A slot found there
//! with a new view record that no mark told of is made (so it is right from
//! then on) and counted, with the place in the title's code that last faulted
//! on its 64 KiB block when the guards saw one ([`faulted`]): a writer the
//! patches miss shows in the log. Records that are not views (the allocator's
//! other data) are counted apart, and so are those a draw's table holds (the
//! draw makes what it reads).
//!
//! Assumption: a slot whose view record the title overwrote with other data
//! (not a view) and no mark told of keeps showing its old view on the host
//! until the fallback looks at its page; nothing reads such a slot as a
//! descriptor (the title wrote no descriptor there).

use std::sync::atomic::{
    AtomicPtr, AtomicU8, AtomicU64, AtomicUsize,
    Ordering::{AcqRel, Acquire, Relaxed, Release},
};

/// Whether this is on.
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // Read once, here: the switch.
    *ON.get_or_init(|| std::env::var("VICTORY_PORT_DESCRIPTORS").is_ok_and(|v| v.trim() == "1"))
}

/// What the patched stores read (`victory_loader::descriptor_sites::MARKS_LEN`
/// bytes): the heap's first byte, its length (0: none), the map of slots, the
/// map of chunks of [`CHUNK`] slots.
#[repr(C)]
pub struct Marks {
    start: AtomicU64,
    len: AtomicU64,
    slots: AtomicPtr<AtomicU8>,
    chunks: AtomicPtr<AtomicU8>,
}

const _: () = assert!(size_of::<Marks>() == 32);

pub static MARKS: Marks = Marks {
    start: AtomicU64::new(0),
    len: AtomicU64::new(0),
    slots: AtomicPtr::new(std::ptr::null_mut()),
    chunks: AtomicPtr::new(std::ptr::null_mut()),
};

/// Slots a byte of the map of chunks stands for (4 KiB of records).
const CHUNK: usize = 128;

/// Where the stores' cells are, for `descriptor_sites::apply`.
pub fn marks_address() -> usize {
    std::ptr::from_ref(&MARKS) as usize
}

/// The title's image (base, size) and how many of its stores were patched.
static IMAGE: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];

/// Told by the launcher: `patched` of the title's record stores in the image
/// at `base` (`size` bytes) mark their slots.
pub fn hooked(base: usize, size: usize, patched: usize) {
    IMAGE[0].store(base, Relaxed);
    IMAGE[1].store(size, Relaxed);
    IMAGE[2].store(patched, Relaxed);
}

/// Whether [`take`] stands for the heap's changed pages: the switch is on and
/// stores were patched.
fn working() -> bool {
    on() && IMAGE[2].load(Relaxed) > 0
}

/// The heap whose slots are marked: its slots, and the places in the title's
/// code that last faulted on each of its 64 KiB blocks.
struct Adopted {
    start: u64,
    count: usize,
    slots: &'static [AtomicU8],
    chunks: &'static [AtomicU8],
    writers: &'static [AtomicUsize],
}

static ADOPTED: std::sync::RwLock<Option<Adopted>> = std::sync::RwLock::new(None);

/// A placed shader-visible view heap of `count` records at `start`: the first
/// such is the one whose slots are marked. Whether it is.
pub(crate) fn adopt(start: u64, count: u32) -> bool {
    working() && adopt_now(start, count)
}

fn adopt_now(start: u64, count: u32) -> bool {
    let mut adopted = ADOPTED.write().unwrap_or_else(|e| e.into_inner());
    if adopted.is_some() {
        tracing::info!(start = format_args!("{start:#x}"), count, "descriptor marks: another placed view heap, looked at by its pages as before");
        return false;
    }
    let count = count as usize;
    // (Never freed: a store of the title's may be marking while the heap
    // goes; a few MB at most.)
    let leak = |n: usize| -> &'static [AtomicU8] { Box::leak((0..n).map(|_| AtomicU8::new(0)).collect()) };
    let slots = leak(count);
    let chunks = leak(count.div_ceil(CHUNK));
    let writers: &'static [AtomicUsize] = Box::leak((0..(count * 32).div_ceil(1 << 16)).map(|_| AtomicUsize::new(0)).collect());
    MARKS.slots.store(slots.as_ptr().cast_mut(), Release);
    MARKS.chunks.store(chunks.as_ptr().cast_mut(), Release);
    MARKS.start.store(start, Release);
    // Last: a store that sees a length sees the rest.
    MARKS.len.store(count as u64 * 32, Release);
    *adopted = Some(Adopted { start, count, slots, chunks, writers });
    tracing::info!(
        start = format_args!("{start:#x}"),
        count,
        stores = IMAGE[2].load(Relaxed),
        "descriptor marks: the title's view heap's slots are told by its record stores and our calls, not looked for on its pages (VICTORY_PORT_DESCRIPTORS)"
    );
    true
}

/// The heap at `start` goes: nothing more is marked for it.
pub(crate) fn let_go(start: u64) {
    let mut adopted = ADOPTED.write().unwrap_or_else(|e| e.into_inner());
    if adopted.as_ref().is_some_and(|a| a.start == start) {
        MARKS.len.store(0, Release);
        *adopted = None;
        tracing::info!(start = format_args!("{start:#x}"), "descriptor marks: the marked view heap went");
    }
}

/// Whether the heap at `start` is the one whose slots are marked.
pub(crate) fn adopted(start: u64) -> bool {
    ADOPTED.read().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|a| a.start == start)
}

/// Marks the slot at `handle` (a record our own calls wrote), if it is in the
/// marked heap.
#[inline]
pub(crate) fn mark(handle: u64) {
    let len = MARKS.len.load(Acquire);
    if len == 0 {
        return;
    }
    let at = handle.wrapping_sub(MARKS.start.load(Relaxed));
    if at >= len {
        return;
    }
    mark_slot((at / 32) as usize);
    MARKED_BY_US.fetch_add(1, Relaxed);
}

/// Marks slot `slot` of the marked heap (in it: checked by the caller).
#[inline]
fn mark_slot(slot: usize) {
    // SAFETY: the maps are leaked, as long as the length says.
    unsafe {
        (*MARKS.slots.load(Relaxed).add(slot)).store(1, Release);
        (*MARKS.chunks.load(Relaxed).add(slot / CHUNK)).store(1, Release);
    }
}

/// Whether slot `index`'s mark is set (told, not yet taken).
fn marked(a: &Adopted, index: u32) -> bool {
    a.slots.get(index as usize).is_some_and(|s| s.load(Acquire) != 0)
}

/// The marked slots of the heap at `start`, each mark cleared, as runs of
/// slots `(first, count)`; `None` if it is not the marked heap.
pub(crate) fn take(start: u64) -> Option<Vec<(u32, u32)>> {
    let began = std::time::Instant::now();
    let adopted = ADOPTED.read().unwrap_or_else(|e| e.into_inner());
    let a = adopted.as_ref().filter(|a| a.start == start)?;
    let mut runs: Vec<(u32, u32)> = Vec::new();
    let mut n = 0u64;
    for (c, chunk) in a.chunks.iter().enumerate() {
        if chunk.load(Relaxed) == 0 {
            continue;
        }
        // Cleared before its slots are read: a mark set meanwhile sets it again.
        chunk.swap(0, AcqRel);
        let first = c * CHUNK;
        for (i, slot) in a.slots[first..(first + CHUNK).min(a.count)].iter().enumerate() {
            if slot.load(Relaxed) == 0 || slot.swap(0, AcqRel) == 0 {
                continue;
            }
            let index = (first + i) as u32;
            n += 1;
            match runs.last_mut() {
                Some((at, len)) if *at + *len == index => *len += 1,
                _ => runs.push((index, 1)),
            }
        }
    }
    TAKEN[0].fetch_add(n, Relaxed);
    TAKEN[1].fetch_add(1, Relaxed);
    TAKEN[2].fetch_add(began.elapsed().as_nanos() as u64, Relaxed);
    Some(runs)
}

/// A write fault of the title's at `at` (from its instruction at `rip`), if in
/// the marked heap: who wrote its block last, for [`missed`]. Runs in the
/// fault handler: nothing here waits.
pub(crate) fn faulted(at: usize, rip: usize) {
    if MARKS.len.load(Relaxed) == 0 {
        return;
    }
    let Ok(adopted) = ADOPTED.try_read() else { return };
    if let Some(a) = adopted.as_ref() {
        let off = (at as u64).wrapping_sub(a.start);
        if let Some(w) = a.writers.get((off >> 16) as usize) {
            w.store(rip, Relaxed);
        }
    }
}

/// Frames between askings of the write watch for the fallback
/// (`VICTORY_PORT_DESCRIPTORS_EVERY=<frames>`, 8 if not given).
fn check_every() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_PORT_DESCRIPTORS_EVERY").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(8u64).max(1))
}

/// Pages the fallback looks at a frame (`VICTORY_PORT_DESCRIPTORS_CHECK=<pages>`,
/// 16 if not given; 0: none).
fn check_budget() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_PORT_DESCRIPTORS_CHECK").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(16usize))
}

/// The fallback's pages: those the watch said were written, not looked at yet;
/// the frame it last asked in, and the frame it last looked in.
#[derive(Default)]
struct Fallback {
    pages: std::collections::BTreeSet<usize>,
    asked: u64,
    looked: u64,
    /// The page it goes on from (round the heap).
    next: usize,
}

impl Fallback {
    /// Whether to ask the write watch in `frame`: not with no budget, and
    /// once in `every` frames.
    fn ask_due(&mut self, frame: u64, every: u64, budget: usize) -> bool {
        if budget == 0 || self.asked != 0 && frame < self.asked + every {
            return false;
        }
        self.asked = frame.max(1);
        true
    }

    /// What the watch said was written (byte ranges), for later.
    fn asked(&mut self, ranges: &[(usize, usize)]) {
        for &(from, to) in ranges {
            let mut page = from & !(crate::watch::PAGE - 1);
            while page < to {
                self.pages.insert(page);
                page += crate::watch::PAGE;
            }
        }
    }

    /// The pages to look at whole in `frame`, at most `budget` (once a frame;
    /// none after), going on round the heap from where it left off.
    fn pages_to_check(&mut self, frame: u64, budget: usize) -> Vec<usize> {
        if self.looked == frame {
            return Vec::new();
        }
        self.looked = frame;
        let mut out = Vec::new();
        while out.len() < budget {
            let from = self.next;
            let Some(&page) = self.pages.range(from..).next().or_else(|| self.pages.iter().next()) else {
                break;
            };
            self.pages.remove(&page);
            self.next = page + crate::watch::PAGE;
            out.push(page);
        }
        out
    }
}

static FALLBACK: std::sync::Mutex<Option<Fallback>> = std::sync::Mutex::new(None);

fn with_fallback<T>(f: impl FnOnce(&mut Fallback) -> T) -> T {
    f(FALLBACK.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Fallback::default))
}

/// Whether to ask the write watch of the marked heap this frame.
pub(crate) fn ask_due(frame: u64) -> bool {
    with_fallback(|f| f.ask_due(frame, check_every(), check_budget()))
}

/// What the watch said was written (byte ranges), for later.
pub(crate) fn asked(ranges: &[(usize, usize)]) {
    with_fallback(|f| f.asked(ranges));
}

/// The pages to look at whole this frame (once a frame; none after).
pub(crate) fn to_check(frame: u64) -> Vec<usize> {
    let pages = with_fallback(|f| f.pages_to_check(frame, check_budget()));
    CHECKED[0].fetch_add(pages.len() as u64, Relaxed);
    pages
}

/// The fallback found slot `index` (at `handle`) with a record other than the
/// one last seen. `view`: it is a view record; `tabled`: a draw read the slot
/// through a table lately (the draw makes it anyway). A view record that no
/// mark told of is counted and said.
pub(crate) fn missed(index: u32, handle: u64, view: bool, tabled: bool) {
    if !view {
        CHECKED[2].fetch_add(1, Relaxed);
        return;
    }
    if tabled {
        CHECKED[4].fetch_add(1, Relaxed);
        return;
    }
    let adopted = ADOPTED.read().unwrap_or_else(|e| e.into_inner());
    let Some(a) = adopted.as_ref() else { return };
    if marked(a, index) {
        // Told, not taken yet.
        CHECKED[3].fetch_add(1, Relaxed);
        return;
    }
    let n = CHECKED[1].fetch_add(1, Relaxed);
    MISSED_ALL.fetch_add(1, Relaxed);
    let rip = a.writers.get((handle.wrapping_sub(a.start) >> 16) as usize).map_or(0, |w| w.load(Relaxed));
    let (base, size) = (IMAGE[0].load(Relaxed), IMAGE[1].load(Relaxed));
    let writer = if rip >= base && rip < base + size {
        format!("title+{:#x}", rip - base)
    } else if rip != 0 {
        format!("{rip:#x} (not the title's)")
    } else {
        "not known (no fault on its block)".into()
    };
    *WRITERS.lock().unwrap_or_else(|e| e.into_inner()).entry(writer.clone()).or_default() += 1;
    if n < 20 || n.is_power_of_two() {
        tracing::warn!(
            slot = index,
            handle = format_args!("{handle:#x}"),
            %writer,
            "descriptor marks: a view record in the title's heap that no store told of (found by the fallback; made now)"
        );
    }
}

/// The runtime's own `memcpy` and `memmove` ([`import`]).
static REAL: [AtomicUsize; 2] = [const { AtomicUsize::new(0) }; 2];

/// Since the last line: the title's copies into the marked heap, records they marked.
static COPIES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// What the title's import of `name` (bound to `address`) is bound to: with
/// the switch, and stores patched, its `memcpy` and `memmove` mark the view
/// records they copy into the heap (the fallback found some hundreds of view
/// records a frame on its pages that the stores had not told of, the last
/// writers of their blocks the runtime's `memcpy` among them).
pub fn import(name: &str, address: usize) -> usize {
    let (which, ours) = match name {
        "memcpy" => (0, copy as *const () as usize),
        "memmove" => (1, shift as *const () as usize),
        _ => return address,
    };
    if !working() {
        return address;
    }
    REAL[which].store(address, Relaxed);
    ours
}

type Copy = unsafe extern "system-unwind" fn(*mut u8, *const u8, usize) -> *mut u8;

unsafe extern "system-unwind" fn copy(to: *mut u8, from: *const u8, n: usize) -> *mut u8 {
    // SAFETY: the title's copy, by the function it asked for.
    unsafe { copied(to, from, n, REAL[0].load(Relaxed)) }
}

unsafe extern "system-unwind" fn shift(to: *mut u8, from: *const u8, n: usize) -> *mut u8 {
    // SAFETY: as above.
    unsafe { copied(to, from, n, REAL[1].load(Relaxed)) }
}

/// # Safety
/// `real` copies as `memcpy` does; the rest are the title's.
#[inline]
unsafe fn copied(to: *mut u8, from: *const u8, n: usize, real: usize) -> *mut u8 {
    // SAFETY: the host's function, noted by `import` before the title ran.
    let real: Copy = unsafe { std::mem::transmute(real) };
    // SAFETY: by the caller.
    let out = unsafe { real(to, from, n) };
    let len = MARKS.len.load(Acquire);
    if len != 0 && n >= 4 {
        // SAFETY: just written by the copy.
        unsafe { marked_by_copy(to as u64, n as u64) };
    }
    out
}

/// Marks each slot of the marked heap whose first word the copy of `n` bytes
/// to `to` just wrote and is a view record's.
///
/// # Safety
/// The bytes `[to, to + n)` were just written, so those inside the heap are
/// readable.
unsafe fn marked_by_copy(to: u64, n: u64) {
    let (start, len) = (MARKS.start.load(Relaxed), MARKS.len.load(Relaxed));
    let (lo, hi) = (to.max(start), to.saturating_add(n).min(start + len));
    if lo >= hi {
        return;
    }
    COPIES[0].fetch_add(1, Relaxed);
    let mut slot = (lo - start).div_ceil(32);
    let mut marked = 0;
    while start + slot * 32 + 4 <= hi {
        let at = start + slot * 32;
        // SAFETY: by the caller, inside the heap.
        if unsafe { (at as *const u32).read_unaligned() } == crate::descriptor::VIEW_TAG {
            mark_slot(slot as usize);
            marked += 1;
        }
        slot += 1;
    }
    COPIES[1].fetch_add(marked, Relaxed);
}

/// Our own calls' marks since the last line.
static MARKED_BY_US: AtomicU64 = AtomicU64::new(0);
/// Since the last line: slots taken, takings, nanoseconds taking.
static TAKEN: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Since the last line, of the fallback: pages looked at whole, view records no
/// mark told of, other records changed (not views), view records told and not
/// taken yet, and view records of draws' tables.
pub(crate) static CHECKED: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Since the start: view records the fallback found that no mark told of.
static MISSED_ALL: AtomicU64 = AtomicU64::new(0);
/// Since the start: the missed records by the place that last faulted on their block.
static WRITERS: std::sync::Mutex<std::collections::BTreeMap<String, u64>> = std::sync::Mutex::new(std::collections::BTreeMap::new());
/// Since the start: slots taken, of them marked by our calls; pages checked.
static ALL: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// With the frame line: what this did per frame over `frames`.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let [taken, takings, ns] = [0, 1, 2].map(|i| TAKEN[i].swap(0, Relaxed));
    let ours = MARKED_BY_US.swap(0, Relaxed);
    let [pages, missed, other, pending, tabled] = [0, 1, 2, 3, 4].map(|i| CHECKED[i].swap(0, Relaxed));
    let copies = [0, 1].map(|i| COPIES[i].swap(0, Relaxed));
    ALL[0].fetch_add(taken, Relaxed);
    ALL[1].fetch_add(ours, Relaxed);
    ALL[2].fetch_add(pages, Relaxed);
    if !working() {
        tracing::info!("descriptor marks (VICTORY_PORT_DESCRIPTORS=1): not working, no store of the title's was patched (see VICTORY_DESCRIPTORS_PROFILE)");
        return;
    }
    let writers = {
        let by = WRITERS.lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<(&String, &u64)> = by.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        v.iter().take(6).map(|(w, n)| format!("{w} {n}")).collect::<Vec<_>>().join(", ")
    };
    tracing::info!(
        "descriptor marks (VICTORY_PORT_DESCRIPTORS=1) per frame: {:.0} slots told (of them {:.0} marked by our calls, the rest by the title's {} patched stores and its memcpy/memmove) in {:.1} takings, {:.3} ms taking; \
         of the title's memcpy/memmove calls into the heap {:.0}, {:.0} records they marked; fallback: {:.1} pages looked at whole, {:.2} view records no mark told of, \
         {:.2} view records of draws' tables, {:.0} other records changed, {:.1} told and not taken yet; \
         since the start: {} slots told ({} by our calls), {} pages looked at, {} view records missed; their writers: {}",
        taken as f64 / frames,
        ours as f64 / frames,
        IMAGE[2].load(Relaxed),
        takings as f64 / frames,
        ns as f64 / 1e6 / frames,
        copies[0] as f64 / frames,
        copies[1] as f64 / frames,
        pages as f64 / frames,
        missed as f64 / frames,
        tabled as f64 / frames,
        other as f64 / frames,
        pending as f64 / frames,
        ALL[0].load(Relaxed),
        ALL[1].load(Relaxed),
        ALL[2].load(Relaxed),
        MISSED_ALL.load(Relaxed),
        if writers.is_empty() { "none".into() } else { writers }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test at a time uses the statics.
    static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn marks_layout_is_what_the_stores_read() {
        // `victory_loader::descriptor_sites::MARKS_LEN` (no dependency on the loader).
        assert_eq!(size_of::<Marks>(), 32);
        assert_eq!(std::mem::offset_of!(Marks, start), 0);
        assert_eq!(std::mem::offset_of!(Marks, len), 8);
        assert_eq!(std::mem::offset_of!(Marks, slots), 16);
        assert_eq!(std::mem::offset_of!(Marks, chunks), 24);
    }

    /// Slots marked as the stores mark them, by our calls and by the
    /// title's copies come out once each, as runs; outside the heap nothing is
    /// marked, and after it goes nothing is taken.
    #[test]
    fn marked_slots_are_taken_once_as_runs() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let start = 0x1000_0000u64;
        assert!(adopt_now(start, 1000));
        assert!(!adopt_now(start + 0x1_0000, 10), "the first heap only");
        assert!(adopted(start) && !adopted(start + 32));
        let store = |slot: usize| {
            let a = ADOPTED.read().unwrap();
            let a = a.as_ref().unwrap();
            a.slots[slot].store(1, Release);
            a.chunks[slot / CHUNK].store(1, Release);
        };
        store(5);
        mark(start + 6 * 32 + 7);
        mark(start + 300 * 32);
        store(999);
        mark(start - 32);
        mark(start + 1000 * 32);
        assert_eq!(take(start).unwrap(), [(5, 2), (300, 1), (999, 1)]);
        assert_eq!(take(start).unwrap(), []);
        assert!(take(start + 32).is_none());
        let_go(start);
        mark(start + 5 * 32);
        assert!(take(start).is_none());
        assert_eq!(MARKS.len.load(Relaxed), 0);
    }

    /// The title's copy into the heap marks the slots whose first word it wrote
    /// and is a view record's; a copy outside the heap, or of other data, or
    /// across a slot's start only, marks none.
    #[test]
    fn a_copy_into_the_heap_marks_the_view_records_it_wrote() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        unsafe extern "system-unwind" fn plain(to: *mut u8, from: *const u8, n: usize) -> *mut u8 {
            // SAFETY: the test's own buffers, apart.
            unsafe { std::ptr::copy(from, to, n) };
            to
        }
        // A heap in memory of our own, 64 slots, 32-aligned.
        let mut memory = vec![0u8; 64 * 32 + 64];
        let start = (memory.as_mut_ptr() as usize).next_multiple_of(32) as u64;
        assert!(adopt_now(start, 64));
        REAL[0].store(plain as *const () as usize, Relaxed);
        let view = |k: u8| {
            let mut r = [k; 32];
            r[..4].copy_from_slice(&crate::descriptor::VIEW_TAG.to_le_bytes());
            r
        };
        // Records 3, 4 (views) and 5 (other data) in one copy of three; and
        // the copy starts at 3's first byte.
        let mut source = Vec::new();
        source.extend(view(1));
        source.extend(view(2));
        source.extend([7u8; 32]);
        // SAFETY: the heap and the source are ours.
        unsafe {
            copy((start + 3 * 32) as *mut u8, source.as_ptr(), source.len());
            assert_eq!(take(start).unwrap(), [(3, 2)]);
            // A copy that starts inside a record: its first word is not the copy's.
            copy((start + 10 * 32 + 8) as *mut u8, view(3).as_ptr(), 32);
            assert_eq!(take(start).unwrap(), []);
            // One that starts in the middle of a record and covers the first words of
            // two more, the last cut short (marked: its first word is written).
            let mut long = vec![0u8; 16];
            long.extend(view(2));
            long.extend(&view(3)[..20]);
            copy((start + 20 * 32 - 16) as *mut u8, long.as_ptr(), long.len());
            assert_eq!(take(start).unwrap(), [(20, 2)]);
            // Outside the heap, and short.
            let mut elsewhere = view(4).to_vec();
            copy(elsewhere.as_mut_ptr(), view(5).as_ptr(), 32);
            copy((start + 40 * 32) as *mut u8, view(5).as_ptr(), 3);
            assert_eq!(take(start).unwrap(), []);
        }
        let_go(start);
    }

    /// The fallback asks the watch once in a few frames, takes its pages
    /// round the heap a few a frame, and looks only once a frame.
    #[test]
    fn the_fallback_looks_at_a_few_pages_round_the_heap() {
        let page = crate::watch::PAGE;
        let mut f = Fallback::default();
        let (every, budget) = (4, 2);
        assert!(f.ask_due(10, every, budget));
        assert!(!f.ask_due(11, every, budget) && !f.ask_due(13, every, budget));
        assert!(f.ask_due(14, every, budget));
        assert!(!Fallback::default().ask_due(1, every, 0), "no budget, no asking");
        // Pages 5, 6 and 9 written, the range of the last cut in the middle.
        f.asked(&[(5 * page + 100, 7 * page), (9 * page + 8, 9 * page + 16)]);
        assert_eq!(f.pages_to_check(20, budget), [5 * page, 6 * page]);
        assert!(f.pages_to_check(20, budget).is_empty(), "once a frame");
        assert_eq!(f.pages_to_check(21, budget), [9 * page]);
        assert!(f.pages_to_check(22, budget).is_empty(), "none left");
        // New pages: it goes on from where it was, then round to the start.
        f.asked(&[(3 * page, 4 * page), (12 * page, 13 * page)]);
        assert_eq!(f.pages_to_check(23, budget), [12 * page, 3 * page]);
    }

    /// What no mark told of is told apart from the told, the other data and the
    /// tables.
    #[test]
    fn a_missed_record_is_counted_by_what_it_is() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let start = 0x2000_0000u64;
        assert!(adopt_now(start, 100));
        for c in &CHECKED {
            c.store(0, Relaxed);
        }
        let before = MISSED_ALL.load(Relaxed);
        missed(3, start + 3 * 32, false, false);
        missed(4, start + 4 * 32, true, true);
        {
            let a = ADOPTED.read().unwrap();
            a.as_ref().unwrap().slots[5].store(1, Release);
        }
        missed(5, start + 5 * 32, true, false);
        missed(6, start + 6 * 32, true, false);
        assert_eq!(CHECKED.each_ref().map(|c| c.load(Relaxed)), [0, 1, 1, 1, 1]);
        assert_eq!(MISSED_ALL.load(Relaxed), before + 1);
        let_go(start);
    }
}
