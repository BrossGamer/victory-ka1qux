//! Renderer port #1: the title's 32-bit graphics heap as the card's own
//! memory (`VICTORY_PORT_CONSTANTS=1`, off by default). After avatarmanz's
//! VIctory-7900XTX (GPL-3.0), "Renderer port 1" (`86dc109`, `4b82f6c`,
//! `9c23e8d`; the page-order half is in `victory-hle`'s `xmem/port.rs`);
//! built here on [`crate::inplace`]'s views and [`crate::inplace::open_stretch`]
//! instead of a table of its own.
//!
//! What the title does at its start: one reservation of graphics memory
//! below 4 GiB, a hundred and some MB, split into a few pieces of different
//! protection; then ONE `XMemAllocatePhysicalPages` for all their pages and
//! one `XMemMapPhysicalPages` for each piece, each from the next page of
//! that one array. The runtime makes one section of all the pages, so the
//! reservation is views of one section. With the page numbers given their
//! slices in the order of the addresses they are mapped at
//! (`xmem/port.rs`), the whole reservation is one stretch of one section:
//! views that follow one another in memory and in the section alike. One of
//! the title's buffers lies in it that is nearly all of it (a ring of
//! constants the CPU fills, and what the skinning passes write through one
//! view of all of it); it is never tile-mapped.
//!
//! What the layer did with it: a host copy in the card's memory, the
//! title's writes found and sent up (some 17 MB a frame in hundreds of
//! pieces), what shaders wrote fetched back and merged, and every constant
//! buffer read out of it either waited for flights or done on the card from
//! the copy (the "title memory read" settles of the frame line).
//!
//! What is done here instead: the stretch is mapped once more, at an
//! address of ours (a view of the same section: the same physical pages),
//! and opened as a host heap there (`OpenExistingHeapFromAddress1`); title
//! buffers and windows of title memory in it are buffers placed in that
//! heap ([`crate::shadow::Shadow`]'s in-place form): the card reads and
//! writes the title's bytes where the title wrote them. Nothing is tracked,
//! sent up, fetched back or compared for it, and no compare or hash pass of
//! ours ever runs over it (they are for copies in the card's memory: one
//! such pass over memory shared with the PC removed the device on another
//! tree). No tile mapping is ever made into this heap (that removes the
//! device).
//!
//! Why our own view: a heap opened from an address holds the pages it was
//! opened on: the title could not unmap, decommit or protect them while it
//! stands. Opened on our view, the title's own views stay free to change.
//! When the title maps other pages at such an address, what is in place
//! there is let go (`crate::inplace::title_memory_freeing`), the region
//! forgotten, and its view unmapped as soon as the host lets go of the heap
//! ([`crate::inplace::OwnView`]).
//!
//! What it assumes: the CPU steps of ours that read this memory at
//! submission (fills of constant buffers, the title's buffer copies done on
//! the CPU) read what the title's CPU wrote, not what a submission still in
//! the air writes there: in place nothing declares the flights' writes, so
//! nothing waits for them. The title's constants are its CPU's; how many
//! such steps there are is on the frame-rate line ([`log`]).
//!
//! Not with `VICTORY_PAGES_ORDERED=1`: that gives page numbers their slices
//! when they are first used, which leaves nothing to put in address order
//! (`xmem/port.rs` says so once and leaves the pages where they are).

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;

use crate::inplace::{self, OwnView, Placed};

const K64: u64 = 64 << 10;

/// The ported memory lies below this: the title's graphics memory for
/// 32-bit addresses, which only that heap is.
const LOW: u64 = 1 << 32;

/// The most bytes of one region: its pages are locked into the PC's memory
/// while the heap stands.
const MOST: u64 = 512 << 20;

/// The most regions open at once (the title has one such heap).
const MOST_REGIONS: usize = 8;

/// How long a stretch the host would not open is left alone.
const COOL: Duration = Duration::from_secs(2);

/// `VICTORY_PORT_CONSTANTS=1`: renderer port #1 on (title buffers and
/// windows of title memory in the title's 32-bit graphics memory, view
/// memory that is one stretch of a section, used in place); unset or
/// anything else: off, everything as before. Read once; the runtime asks
/// here too, for the page order.
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // Read once, here: the switch of renderer port #1.
    *ON.get_or_init(|| std::env::var("VICTORY_PORT_CONSTANTS").is_ok_and(|v| v.trim() == "1"))
}

/// A stretch of title memory opened as a heap through a view of ours. The
/// heap is dropped before the view (field order), and the view is unmapped
/// when the host lets the heap go (a heap that buffers of lists in the air
/// still hold stands a moment longer: [`inplace::retry_views`]).
struct Region {
    start: u64,
    end: u64,
    heap: ID3D12Heap,
    _view: OwnView,
}

// SAFETY: D3D12 heaps are free-threaded; the view is an address of ours.
unsafe impl Send for Region {}

static REGIONS: Mutex<Vec<Region>> = Mutex::new(Vec::new());

/// The lowest start and highest end of every region (both 0: none), for a
/// quick look on the paths that run thousands of times a frame.
static BOUNDS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Stretches the host would not open, until when: asked again each time
/// they are asked about, they would cost a view and a refusal each.
static COOLING: Mutex<Vec<(u64, u64, Instant)>> = Mutex::new(Vec::new());

// The statistics, since the start: regions opened and their bytes; title
// buffers and windows placed; refused because the memory is not one
// stretch of a section, not committed, refused by the host, no buffer could
// be placed, too many regions, not room in the host's shared memory;
// regions forgotten because the title mapped other pages there. Then, since
// the last line: fills read out of ported memory on the CPU and their
// bytes, buffer copies on the CPU that touch it and their bytes.
const OPENED: usize = 0;
const OPENED_BYTES: usize = 1;
const PLACED: usize = 2;
const NOT_STRETCH: usize = 3;
const NOT_COMMITTED: usize = 4;
const REFUSED: usize = 5;
const NO_BUFFER: usize = 6;
const FORGOTTEN: usize = 7;
const NO_ROOM: usize = 8;
const TOO_MANY: usize = 9;
const FILLS: usize = 10;
const FILL_BYTES: usize = 11;
const COPIES: usize = 12;
const COPY_BYTES: usize = 13;
static STATS: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];

/// Why a title buffer or window got no place in a region.
#[derive(Debug)]
pub(crate) enum Refusal {
    /// The memory is not one stretch of one section (views that follow one
    /// another in memory and in the section), or the stretch does not hold
    /// the range, or is over [`MOST`]: the words say which.
    NotStretch(String),
    /// A page of the stretch is not committed.
    NotCommitted,
    /// The host refused the view or the heap.
    Refused(String),
    /// The host refused the buffer in a region's heap.
    NoBuffer(String),
    /// [`MOST_REGIONS`] regions are open.
    TooMany,
    /// The host's shared GPU memory would be left with less than it keeps.
    NoRoom,
    /// The runtime was changing mappings in the stretch: the ask is tried
    /// again at the next.
    Changing,
    /// The host refused this stretch a moment ago ([`COOL`]).
    Cooling,
}

impl Refusal {
    /// The statistic that counts it, if any.
    fn stat(&self) -> Option<usize> {
        match self {
            Refusal::NotStretch(_) => Some(NOT_STRETCH),
            Refusal::NotCommitted => Some(NOT_COMMITTED),
            Refusal::Refused(_) => Some(REFUSED),
            Refusal::NoBuffer(_) => Some(NO_BUFFER),
            Refusal::TooMany => Some(TOO_MANY),
            Refusal::NoRoom => Some(NO_ROOM),
            Refusal::Changing | Refusal::Cooling => None,
        }
    }

    fn why(&self) -> String {
        match self {
            Refusal::NotStretch(why) | Refusal::Refused(why) | Refusal::NoBuffer(why) => why.clone(),
            Refusal::NotCommitted => "pages of the stretch are not committed".to_owned(),
            Refusal::TooMany => format!("{MOST_REGIONS} regions are open"),
            Refusal::NoRoom => "the host's shared GPU memory is nearly used".to_owned(),
            Refusal::Changing => "the runtime was changing mappings there".to_owned(),
            Refusal::Cooling => "refused a moment ago".to_owned(),
        }
    }
}

/// Whether `[at, at + len)` may be ported: in the title's 32-bit graphics
/// memory.
pub(crate) fn wanted(at: u64, len: u64) -> bool {
    on() && below_4_gib(at, len)
}

/// Whether the non-empty range `[at, at + len)` lies below 4 GiB.
fn below_4_gib(at: u64, len: u64) -> bool {
    at.checked_add(len).is_some_and(|end| end <= LOW && len > 0)
}

/// Whether a region is open that holds `[at, end)`: what the windows of
/// `crate::unified` leave to the port.
pub(crate) fn holds(at: u64, end: u64) -> bool {
    if !on() || end <= at || BOUNDS[1].load(Ordering::Relaxed) < end || BOUNDS[0].load(Ordering::Relaxed) > at {
        return false;
    }
    let regions = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    regions.iter().any(|r| r.start <= at && end <= r.end)
}

/// Title memory `[start, start + len)` in place, if [`wanted`] and the
/// memory is one stretch of a section: a buffer placed in the region's
/// heap. `None` leaves the caller to do what it did before.
pub(crate) fn place_for(start: u64, len: u64, window: bool) -> Option<Placed> {
    if !wanted(start, len) {
        return None;
    }
    let host = crate::host::host()?;
    match place_in(&host.device, start, len, &crate::unified::room) {
        Ok(placed) => {
            inplace::counted(window, len);
            let n = STATS[PLACED].fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 32 || n.is_power_of_two() {
                tracing::info!(
                    start = format_args!("{start:#x}"),
                    len,
                    window,
                    "renderer port 1: title memory in place, read and written by the card where the title has it"
                );
            }
            Some(placed)
        }
        Err(refusal) => {
            if let Some(stat) = refusal.stat() {
                let n = STATS[stat].fetch_add(1, Ordering::Relaxed) + 1;
                if n <= 8 || n.is_power_of_two() {
                    tracing::warn!(
                        start = format_args!("{start:#x}"),
                        len,
                        window,
                        why = refusal.why(),
                        "renderer port 1: title memory not in place, a copy as before"
                    );
                }
            }
            None
        }
    }
}

/// [`place_for`]'s work: the region holding the pages of `[start, start +
/// len)`, opened if none does (`room` says whether the host's shared GPU
/// memory has room for a heap of that many bytes), and a buffer placed in
/// it.
pub(crate) fn place_in(device: &ID3D12Device, start: u64, len: u64, room: &dyn Fn(u64) -> bool) -> Result<Placed, Refusal> {
    let at = start & !(K64 - 1);
    let end = (start + len.max(1)).next_multiple_of(K64);
    let mut regions = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    let found = spans(&regions).position(|(first, last)| first <= at && end <= last);
    let i = match found {
        Some(i) => i,
        None => {
            if regions.len() >= MOST_REGIONS {
                return Err(Refusal::TooMany);
            }
            let region = open_region(device, at, end, room)?;
            regions.push(region);
            let n = regions.len() - 1;
            let r = &regions[n];
            STATS[OPENED].fetch_add(1, Ordering::Relaxed);
            STATS[OPENED_BYTES].fetch_add(r.end - r.start, Ordering::Relaxed);
            bounds(&regions);
            n
        }
    };
    let (heap, first) = (regions[i].heap.clone(), regions[i].start);
    // The table of regions is free again for the host call below. A region
    // forgotten meanwhile leaves this buffer in a heap over the view's old
    // pages: the caller holds the lock of the table the buffer goes into
    // (`crate::shadow`), and what the runtime's remap lets go of takes that
    // lock after it forgot the region, so the buffer goes with the rest.
    drop(regions);
    let buffer = inplace::place(device, &heap, at - first, end - at).map_err(|e| Refusal::NoBuffer(format!("no buffer placed in the region: {e}")))?;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + (start - at);
    // The view is the region's: the buffer's heap reference is all a
    // placed copy holds.
    Ok(Placed { buffer, heap, gpu, bias: start - at, view: None })
}

/// The spans `(start, end)` of `regions`.
fn spans(regions: &[Region]) -> impl Iterator<Item = (u64, u64)> + '_ {
    regions.iter().map(|r| (r.start, r.end))
}

/// Opens the stretch around `[at, end)` (64 KiB aligned) as a heap through
/// a view of ours of the same pages.
fn open_region(device: &ID3D12Device, at: u64, end: u64, room: &dyn Fn(u64) -> bool) -> Result<Region, Refusal> {
    if cooling(at, end) {
        return Err(Refusal::Cooling);
    }
    let Some(run) = inplace::stretch(at as usize, (end - at) as usize) else {
        return Err(Refusal::NotStretch(format!("not one stretch of a section: {:?}", inplace::cover(at as usize, (end - at) as usize))));
    };
    // The whole stretch, kept to what may be ported: the next buffer in it
    // finds the region made.
    let (mut first, mut last, mut offset) = (run.start as u64, run.end as u64, run.offset);
    if first < LOW && last > LOW {
        last = LOW;
    }
    if last - first > MOST {
        // Only what was asked for.
        offset += at - first;
        (first, last) = (at, end);
    }
    if last - first > MOST {
        return Err(Refusal::NotStretch(format!("{:#x} bytes, over the most of a region", last - first)));
    }
    if first > at || last < end || !offset.is_multiple_of(K64) {
        return Err(Refusal::NotStretch(format!("the stretch {first:#x}+{:#x} does not hold it", last - first)));
    }
    let bytes = last - first;
    if !room(bytes) {
        return Err(Refusal::NoRoom);
    }
    // The runtime changes no mapping in the range while our view is made
    // and the heap opened on it: the pages are those the views showed when
    // the stretch was looked at (again, under the hold).
    let Some(_still) = crate::watch::title_memory_reading_range_now(first as usize, bytes as usize) else {
        return Err(Refusal::Changing);
    };
    if inplace::stretch(at as usize, (end - at) as usize) != Some(run) {
        return Err(Refusal::Changing);
    }
    match inplace::open_stretch(device, run.section, offset, bytes) {
        Ok((heap, view)) => {
            tracing::info!(
                start = format_args!("{first:#x}"),
                bytes,
                section = format_args!("{:#x}", run.section),
                offset = format_args!("{offset:#x}"),
                "renderer port 1: a stretch of the title's graphics memory opened as a host heap through a view of ours"
            );
            Ok(Region { start: first, end: last, heap, _view: view })
        }
        Err(e) => {
            let refusal = match e {
                inplace::StretchError::NotMapped(e) => Refusal::Refused(format!("no view of the stretch: {e}")),
                inplace::StretchError::NotCommitted => Refusal::NotCommitted,
                inplace::StretchError::Refused(e) => Refusal::Refused(format!("refused by the host: {e}")),
            };
            cool(at, end);
            Err(refusal)
        }
    }
}

/// Whether the host refused a stretch that holds `[at, end)` lately.
fn cooling(at: u64, end: u64) -> bool {
    let mut cool = COOLING.lock().unwrap_or_else(|e| e.into_inner());
    cool.retain(|&(_, _, until)| until > Instant::now());
    cool.iter().any(|&(from, to, _)| from <= at && end <= to)
}

/// Leaves `[at, end)` alone for [`COOL`].
fn cool(at: u64, end: u64) {
    let mut cool = COOLING.lock().unwrap_or_else(|e| e.into_inner());
    if cool.len() < 16 {
        cool.push((at, end, Instant::now() + COOL));
    }
}

fn bounds(regions: &[Region]) {
    let lo = spans(regions).map(|r| r.0).min().unwrap_or(0);
    let hi = spans(regions).map(|r| r.1).max().unwrap_or(0);
    BOUNDS[0].store(lo, Ordering::Relaxed);
    BOUNDS[1].store(hi, Ordering::Relaxed);
}

/// The title maps other pages at `[at, at + len)`: the regions over it are
/// forgotten (what is placed in them is let go by the caller, and their
/// views unmapped when the host lets go of the heaps).
pub(crate) fn forget(at: usize, len: usize) {
    let (at, end) = (at as u64, (at as u64).saturating_add(len as u64));
    if BOUNDS[1].load(Ordering::Relaxed) <= at || BOUNDS[0].load(Ordering::Relaxed) >= end {
        return;
    }
    let mut regions = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    let before = regions.len();
    regions.retain(|r| r.end <= at || r.start >= end);
    STATS[FORGOTTEN].fetch_add((before - regions.len()) as u64, Ordering::Relaxed);
    bounds(&regions);
}

/// Whether `[at, at + len)` touches `[lo, hi)` (0, 0: nothing).
fn touches_span(lo: u64, hi: u64, at: usize, len: usize) -> bool {
    hi != 0 && (at as u64) < hi && (at as u64).saturating_add(len as u64) > lo
}

/// Whether `[at, at + len)` touches ported memory (a quick look).
fn touches(at: usize, len: usize) -> bool {
    touches_span(BOUNDS[0].load(Ordering::Relaxed), BOUNDS[1].load(Ordering::Relaxed), at, len)
}

/// A fill of upload memory read `len` bytes at `src` on the CPU.
pub(crate) fn filled(src: usize, len: usize) {
    if touches(src, len) {
        STATS[FILLS].fetch_add(1, Ordering::Relaxed);
        STATS[FILL_BYTES].fetch_add(len as u64, Ordering::Relaxed);
    }
}

/// A title buffer copy was done on the CPU.
pub(crate) fn copied(dst: usize, src: usize, len: usize) {
    if touches(dst, len) || touches(src, len) {
        STATS[COPIES].fetch_add(1, Ordering::Relaxed);
        STATS[COPY_BYTES].fetch_add(len as u64, Ordering::Relaxed);
    }
}

/// For the frame-rate line: what is in place, and what of ours read or
/// wrote it on the CPU per frame.
pub(crate) fn log(frames: f64) {
    if !on() {
        return;
    }
    // Views of regions let go that a heap held a moment ago.
    inplace::retry_views();
    let s: Vec<u64> = STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
    let per: Vec<f64> = STATS[FILLS..].iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "renderer port 1 (VICTORY_PORT_CONSTANTS=1): {} regions of the title's graphics memory in place ({} MB), {} title buffers and windows placed in them \
         (no host copy, nothing sent up or fetched back for them); per frame on the CPU: {:.0} fills read out of them ({:.1} KB), {:.1} buffer copies \
         touching them ({:.1} KB); since the start refused: {} not one stretch of a section, {} not committed, {} by the host, {} no buffer, \
         {} too many regions, {} no room in the host's shared memory; {} regions forgotten (other pages mapped there)",
        s[OPENED],
        s[OPENED_BYTES] >> 20,
        s[PLACED],
        per[0],
        per[1] / 1024.0,
        per[2],
        per[3] / 1024.0,
        s[NOT_STRETCH],
        s[NOT_COMMITTED],
        s[REFUSED],
        s[NO_BUFFER],
        s[TOO_MANY],
        s[NO_ROOM],
        s[FORGOTTEN],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROOM: &dyn Fn(u64) -> bool = &|_| true;

    /// Only the title's 32-bit memory is asked for.
    #[test]
    fn only_memory_below_4_gib_is_wanted_with_the_switch() {
        // The switch is read once from the environment; tests do not set
        // it, so nothing is wanted.
        if !on() {
            assert!(!wanted(0x1000_0000, 0x1000));
        }
        assert!(below_4_gib(0x1000_0000, 0x800_0000));
        assert!(below_4_gib(LOW - 0x1000, 0x1000));
        assert!(!below_4_gib(LOW - 0x1000, 0x1001));
        assert!(!below_4_gib(0xffff_0000, 0x2_0000));
        assert!(!below_4_gib(0x1_0000_0000, 0x1000));
        assert!(!below_4_gib(u64::MAX, 2));
        assert!(!below_4_gib(0x1000, 0));
    }

    /// What the quick look at the bounds says touches ported memory.
    #[test]
    fn a_range_touches_ported_memory_only_inside_its_bounds() {
        assert!(!touches_span(0, 0, 0x1000, 0x100), "no region");
        assert!(touches_span(0x1_0000, 0x2_0000, 0x1_ff00, 0x200));
        assert!(touches_span(0x1_0000, 0x2_0000, 0x0_ff00, 0x200), "reaching in from below");
        assert!(!touches_span(0x1_0000, 0x2_0000, 0x0_ff00, 0x100), "ending where it starts");
        assert!(!touches_span(0x1_0000, 0x2_0000, 0x2_0000, 0x100), "starting where it ends");
        assert!(touches_span(0x1_0000, 0x2_0000, 0x0, usize::MAX), "no overflow");
    }

    /// A stretch of a reserved section, mapped as `xmem` maps physical
    /// pages (views of consecutive pages in turn, committed), is opened
    /// through a view of ours: the card reads the bytes the title's views
    /// show, at the buffer's offset, and its writes show in them; a second
    /// buffer in the stretch is placed in the same region (one view, one
    /// heap); the title's own views can be unmapped under the heap, and
    /// forgetting the region lets go of our view once the buffers are gone.
    #[test]
    fn warp_a_stretch_of_a_reserved_section_is_used_in_place_through_a_view_of_ours() {
        use windows_sys::Win32::System::Memory as mem;
        let _one = crate::address_space();
        let Some(device) = inplace::tests::warp() else { return };
        if windows::core::Interface::cast::<ID3D12Device13>(&device).is_err() {
            return;
        }
        const K: usize = 64 << 10;
        // SAFETY (the whole test): the section, placeholder and views made
        // here.
        unsafe {
            let process = windows_sys::Win32::System::Threading::GetCurrentProcess();
            let section = mem::CreateFileMappingW(
                windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE,
                std::ptr::null(),
                mem::PAGE_READWRITE | mem::SEC_RESERVE,
                0,
                8 * K as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let hole = mem::VirtualAlloc2(process, std::ptr::null(), 8 * K, mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER, mem::PAGE_NOACCESS, std::ptr::null_mut(), 0)
                .cast::<u8>();
            assert!(!hole.is_null());
            // Two views, pages 0..3 and 3..8 of the section, in turn.
            mem::VirtualFree(hole.cast(), 3 * K, mem::MEM_RELEASE | mem::MEM_PRESERVE_PLACEHOLDER);
            let mut views = Vec::new();
            for (page, n) in [(0usize, 3usize), (3, 5)] {
                let view = mem::MapViewOfFile3(section, process, hole.add(page * K).cast(), (page * K) as u64, n * K, mem::MEM_REPLACE_PLACEHOLDER, mem::PAGE_READWRITE, std::ptr::null_mut(), 0);
                assert!(!view.Value.is_null(), "{}", std::io::Error::last_os_error());
                assert!(!mem::VirtualAlloc(view.Value, n * K, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());
                inplace::title_memory_view(view.Value as usize, n * K, section as usize, (page * K) as u64);
                views.push(view);
            }
            let base = hole as u64;
            let (start, len) = (base + K as u64 + 0x40, 5 * K as u64);
            let ours_before = inplace::own_views_mapped();
            let placed = place_in(&device, start, len, ROOM).unwrap_or_else(|e| panic!("not placed: {e:?}"));
            assert_eq!(placed.bias, 0x40);
            assert!(placed.view.is_none(), "the view is the region's");
            assert!(
                REGIONS.lock().unwrap().iter().any(|r| (r.start, r.end) == (base, base + 8 * K as u64)),
                "the region is the whole stretch, not what was asked for"
            );
            let mut rig = inplace::Rig::new(&device).unwrap();
            for (i, b) in std::slice::from_raw_parts_mut(start as *mut u8, len as usize).iter_mut().enumerate() {
                *b = (i * 7 + 3) as u8;
            }
            let got = rig.fetch(&placed.buffer, placed.bias, len as usize).unwrap();
            assert!(got == std::slice::from_raw_parts(start as *const u8, len as usize), "the card reads the title's bytes, across both views");
            rig.store(&placed.buffer, placed.bias + 3 * K as u64, &[0x5a; 256]).unwrap();
            assert!(std::slice::from_raw_parts((start + 3 * K as u64) as *const u8, 256).iter().all(|&b| b == 0x5a), "the card's write shows in the title's view");
            // Another buffer in the same stretch: the same region.
            let (regions, views_of_ours) = (REGIONS.lock().unwrap().len(), inplace::own_views_mapped());
            assert_eq!(views_of_ours, ours_before + 1, "one view of ours for the region");
            let second = place_in(&device, base + 6 * K as u64, 100, ROOM).unwrap_or_else(|e| panic!("not placed: {e:?}"));
            assert_eq!(REGIONS.lock().unwrap().len(), regions, "one region for the stretch");
            assert_eq!(inplace::own_views_mapped(), views_of_ours, "no other view of ours for the second buffer");
            assert_eq!(second.bias, 0);
            // Both buffers are in one heap: what the card writes through
            // one is read through the other.
            rig.store(&second.buffer, 0, &[0x33; 64]).unwrap();
            let seen = rig.fetch(&placed.buffer, 5 * K as u64, 64).unwrap();
            assert!(seen.iter().all(|&b| b == 0x33), "one heap, one memory");
            // The title's views may change under the heap.
            drop((rig, placed, second));
            forget(base as usize, 8 * K);
            assert!(REGIONS.lock().unwrap().iter().all(|r| r.end <= base || r.start >= base + 8 * K as u64), "forgotten");
            for view in views {
                assert!(mem::UnmapViewOfFile2(process, view, mem::MEM_PRESERVE_PLACEHOLDER) != 0, "the title's own view can be unmapped");
            }
            inplace::views_gone(base as usize, 8 * K);
            // Our view goes once the host has let go of the heap.
            for _ in 0..500 {
                inplace::retry_views();
                if inplace::own_views_mapped() <= ours_before {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            assert!(inplace::own_views_mapped() <= ours_before, "the region's view is unmapped");
            mem::VirtualFree(hole.cast(), 0, mem::MEM_RELEASE);
            mem::VirtualFree(hole.add(3 * K).cast(), 0, mem::MEM_RELEASE);
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// Memory that is not one stretch of one section is refused with the
    /// reason, and the host is not asked: views from elsewhere in the
    /// section in turn (as the title maps its pieces before the runtime
    /// puts the pages in address order), a hole, a stretch the range
    /// leaves; a stretch there is no room for is not opened either.
    #[test]
    fn warp_memory_that_is_not_one_stretch_is_refused() {
        let _one = crate::address_space();
        let Some(device) = inplace::tests::warp() else { return };
        let at = 0x7_c000_0000u64;
        let k = K64 as usize;
        // Views of section 91, the second piece first in the section.
        inplace::title_memory_view(at as usize, 2 * k, 91, 4 * K64);
        inplace::title_memory_view(at as usize + 2 * k, 2 * k, 91, 0);
        let scattered = place_in(&device, at + 0x100, 3 * K64, ROOM).err();
        assert!(matches!(&scattered, Some(Refusal::NotStretch(why)) if why.contains("Scattered")), "{scattered:?}");
        let hole = place_in(&device, at + 3 * K64, 2 * K64, ROOM).err();
        assert!(matches!(&hole, Some(Refusal::NotStretch(why)) if why.contains("Holes")), "{hole:?}");
        let none = place_in(&device, at - 4 * K64, K64, ROOM).err();
        assert!(matches!(&none, Some(Refusal::NotStretch(why)) if why.contains("None")), "{none:?}");
        // Inside the first view alone it is one run, but there is no room:
        // not asked of the host.
        let crowded = place_in(&device, at, 2 * K64, &|_| false).err();
        assert!(matches!(crowded, Some(Refusal::NoRoom)), "{crowded:?}");
        inplace::views_gone(at as usize, 4 * k);
        assert!(REGIONS.lock().unwrap().iter().all(|r| r.end <= at || r.start >= at + 4 * K64));
    }

    /// Regions let go leave no view of ours behind and are forgotten only
    /// where the title mapped other pages.
    #[test]
    fn forgetting_keeps_the_regions_the_title_did_not_touch() {
        let _one = crate::address_space();
        let Some(device) = inplace::tests::warp() else { return };
        if windows::core::Interface::cast::<ID3D12Device13>(&device).is_err() {
            return;
        }
        use windows_sys::Win32::System::Memory as mem;
        const PAGES: usize = 8;
        // SAFETY: sections and views of the test's own.
        unsafe {
            let section = mem::CreateFileMappingW(
                windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE,
                std::ptr::null(),
                mem::PAGE_READWRITE | mem::SEC_RESERVE,
                0,
                (PAGES * K64 as usize) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let title = mem::MapViewOfFile(section, mem::FILE_MAP_WRITE, 0, 0, PAGES * K64 as usize).Value as usize;
            assert_ne!(title, 0);
            mem::VirtualAlloc(title as *const _, PAGES * K64 as usize, mem::MEM_COMMIT, mem::PAGE_READWRITE);
            inplace::title_memory_view(title, PAGES * K64 as usize, section as usize, 0);
            let ours_before = inplace::own_views_mapped();
            let placed = place_in(&device, title as u64, 4 * K64, ROOM).unwrap_or_else(|e| panic!("not placed: {e:?}"));
            let n = REGIONS.lock().unwrap().len();
            // Another range: the region stays.
            forget(title - 4 * K64 as usize, 4 * K64 as usize);
            forget(title + PAGES * K64 as usize, K64 as usize);
            assert_eq!(REGIONS.lock().unwrap().len(), n, "not touched");
            // One page of it: the region goes.
            forget(title + 3 * K64 as usize, 1);
            assert_eq!(REGIONS.lock().unwrap().len(), n - 1, "forgotten");
            drop(placed);
            inplace::views_gone(title, PAGES * K64 as usize);
            for _ in 0..500 {
                inplace::retry_views();
                if inplace::own_views_mapped() <= ours_before {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            assert!(inplace::own_views_mapped() <= ours_before, "the forgotten region's view is unmapped");
            mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: title as *mut _ });
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }
}
