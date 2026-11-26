//! The records the title copied into its heap as bytes, looked at all the
//! changed pages of a heap at once (`VICTORY_HEAP_SYNC_BATCH=1`; off by
//! default), for [`crate::descriptor::sync_heaps`].
//!
//! After Marvisak/VIctory pull request 3055 (by MeMako, "batch and
//! parallelize descriptor heap syncs"), fitted to what this tree has: the
//! pull request's base took each changed page alone, with three vectors
//! made for it and a lock of the views and of the slots for it, and made
//! every changed slot serially. Here the page's records already went
//! through one lock of the fingerprints for the page
//! ([`HostHeap::mirror_copied_page`], avatarmanz/VIctory-7900XTX), so what
//! remains of that pull request's first part is the reading and
//! fingerprinting of the pages themselves, which are many (some 170 a
//! frame, 350,000 records) and independent. With the switch on:
//!
//! 1. every changed page of the heap is read and fingerprinted at once, on
//!    the kept threads ([`crate::shadow::wide_run`]) and this one, a few
//!    pages to a job: no thread made for it, and a heap with few pages is
//!    done on this thread;
//! 2. the fingerprints are compared with those last seen under one lock of
//!    them for the whole heap, in page order, which gives the slots to
//!    mirror in the order the pages' one-by-one version has;
//! 3. those slots are mirrored one after the other on this thread, as
//!    [`HostHeap::mirror_as`] does for each, exactly as before.
//!
//! The slots mirrored, and what is made of each, are those of the pages'
//! version. What differs is when a page is read: all of them before the
//! first slot is mirrored, not each just before its own (a record the
//! title writes in between is mirrored as it is when its slot's turn comes,
//! and compared next time with the print the page read: it is then seen
//! as new again, and mirrored again, which makes the same slot).
//!
//! The pull request's second part (the slots of the records that changed,
//! made on several threads under one hold of the slots' lock) is not here.
//! There the mirroring of a slot was a few locks and one `Create*View`
//! call; here [`HostHeap::mirror_as`] does the books of the placed textures,
//! the spans, the queue of textures picked by index, the slots made stale
//! and the ranges the pass over buffer views takes, each under a lock of its
//! own and in the order the slots come, so that all that could be made on
//! another thread is the host's view of a buffer, which `world_frame_sync`
//! (the test, run with `--ignored`) measures at 0.17 microseconds
//! a view (the books of a mirrored slot, with a buffer view, 0.4 to 0.5).
//! The pull request's lookups of the host copies in one go are what
//! `VICTORY_SHADOW_CACHE=1` does for the lookups of a recording thread
//! here (`crate::shadow`'s memory of answers).
//!
//! What the switch takes off the pages' version, measured by that test
//! on 700,416 records of 342 changed pages (release, 20 logical processors),
//! medians of nine: 14.4 ms to 5.0 ms with none changed, 20.1 to 7.8 ms
//! with 6,000 changed, 23.3 to 12.6 ms with 14,000.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::{COPIED_TABLED, HostHeap, sync_makes, sync_tables, zero_record_print};
use crate::descriptor::changed_slots;

/// Whether the changed pages of a heap are looked at together
/// (`VICTORY_HEAP_SYNC_BATCH=1`; off by default: page by page, as before).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_HEAP_SYNC_BATCH").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "the changed pages of a descriptor heap are read and fingerprinted together, on the kept threads (VICTORY_HEAP_SYNC_BATCH=1)"
            );
        }
        on
    })
}

/// Pages to a job: fewer than twice that are done on the calling thread.
const JOB_PAGES: usize = 4;
/// The most jobs one heap's pages are cut into (the kept threads are at
/// most eight, and this one helps).
const JOBS_MOST: usize = 8;

/// Since the last frame line: heaps synced in a batch, pages they held,
/// jobs those were cut into, and the microseconds of the three parts (the
/// pages read and fingerprinted, the fingerprints compared, the slots
/// mirrored).
static COUNT: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// With the frame line: what the batches did per frame over `frames`.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let [syncs, pages, jobs, scan, compare, mirror] = [0, 1, 2, 3, 4, 5].map(|i| COUNT[i].swap(0, Ordering::Relaxed));
    if syncs == 0 {
        return;
    }
    let ms = |us: u64| us as f64 / 1000.0 / frames;
    tracing::info!(
        "heap syncs in a batch (VICTORY_HEAP_SYNC_BATCH=1) per frame: {:.1} heaps of {:.1} pages in all, read on {:.1} jobs: {:.2} ms \
         reading and fingerprinting, {:.2} ms comparing the fingerprints, {:.2} ms mirroring the slots that changed (this thread)",
        syncs as f64 / frames,
        pages as f64 / frames,
        jobs as f64 / frames,
        ms(scan),
        ms(compare),
        ms(mirror)
    );
}

/// A page's records, fingerprinted: the print of slot `first + i` is
/// `prints[i]`, 0 for a record that is not whole in the page's bytes (left
/// to the page that holds its first byte).
#[derive(Debug, PartialEq)]
struct Scanned {
    first: u64,
    prints: Vec<u64>,
}

/// What each job of [`scan_pages`] brings back, by job: its pages' scans.
type Parts = Vec<Option<Vec<Option<Scanned>>>>;

/// The title's address ranges of a heap's changed pages, as `(from, to)` of
/// one page each (a changed range cut at the 64 KiB pages it covers).
pub(crate) fn pages_of(ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let size = crate::watch::PAGE;
    let mut pages = Vec::new();
    for &(range_from, range_end) in ranges {
        let mut page = range_from & !(size - 1);
        while page < range_end {
            pages.push((page.max(range_from), (page + size).min(range_end)));
            page += size;
        }
    }
    pages
}

/// Reads the records of the page `[from, to)` of a heap whose slot `i` has
/// its record at `start + i * step`, and fingerprints them (what
/// [`HostHeap::mirror_copied_page`] and [`HostHeap::mirror_copied_prints`]
/// do before they compare). `None`: the page is not there (the title
/// unmapped it). `in_place`: the records are fingerprinted where they are
/// first (`VICTORY_SYNC_IN_PLACE=1`), as `sync_heaps` does; `bytes` and
/// `prints` are the caller's room for the page's copy and for the prints.
fn scan_page(
    (from, to): (usize, usize),
    (start, step): (u64, u64),
    mirrored_count: u32,
    in_place: bool,
    bytes: &mut Vec<u8>,
    prints: &mut Vec<u64>,
) -> Option<Scanned> {
    let slots = changed_slots(from, to, start as usize, step, mirrored_count);
    if in_place {
        // The records that lie whole in the page, as the page's version
        // below takes them.
        let first = slots.start.max(((from - start as usize) as u64).div_ceil(step));
        let last = slots.end.min((to - start as usize) as u64 / step);
        if first >= last {
            return Some(Scanned { first, prints: Vec::new() });
        }
        let at = (start + first * step) as usize;
        if crate::heapprint::prints(at, (last - first) as usize, prints) {
            return Some(Scanned { first, prints: std::mem::take(prints) });
        }
        // Not read in place (a page of it is not there, or faulted lately,
        // or a copy-back left it pending): through the kernel's copy.
    }
    bytes.resize(to - from, 0);
    if !crate::host::read_into(from as *const u8, bytes) {
        return None;
    }
    let mut out = Vec::with_capacity((slots.end - slots.start) as usize);
    for index in slots.clone() {
        let handle = start + index * step;
        // A record that straddles the page's ends is left to the page that
        // holds its first byte.
        let print = (handle as usize)
            .checked_sub(from)
            .and_then(|at| bytes.get(at..at + step as usize))
            .map_or(0, |record| super::record_print(record) | 1);
        out.push(print);
    }
    Some(Scanned { first: slots.start, prints: out })
}

impl HostHeap {
    /// [`HostHeap::mirror_copied_page`] for every page of a heap at once
    /// (module notes): `pages` are the title's byte ranges `(from, to)` of
    /// its changed pages in address order, one page each; slot `i` has its
    /// record at `start + i * step`; `mirrored_count` is the slots that have
    /// host counterparts. Returns the pages that were there, the records
    /// looked at and those mirrored.
    pub(crate) fn mirror_copied_batch(
        &self,
        pages: &[(usize, usize)],
        at: (u64, u64),
        mirrored_count: u32,
        in_place: bool,
    ) -> (u64, u64, u64) {
        let (read, looked, made, _) = self.mirror_copied_batch_noting(pages, at, mirrored_count, in_place, usize::MAX);
        (read, looked, made)
    }

    /// [`HostHeap::mirror_copied_batch`] that also says which records of the
    /// ranges from `noting` on (`VICTORY_PORT_DESCRIPTORS=1`: the first ranges
    /// are runs of slots that were marked, the rest pages the fallback looks
    /// at whole) were not the ones last seen: the slot's index, and whether a
    /// draw read it through a table lately (left to the draw, as for any
    /// range). A record that changed on a range the marks did not tell of is a
    /// writer the patches miss.
    pub(crate) fn mirror_copied_batch_noting(
        &self,
        pages: &[(usize, usize)],
        (start, step): (u64, u64),
        mirrored_count: u32,
        in_place: bool,
        noting: usize,
    ) -> (u64, u64, u64, Vec<(u32, bool)>) {
        if pages.is_empty() {
            return (0, 0, 0, Vec::new());
        }
        let began = Instant::now();
        let scans = scan_pages(pages, (start, step), mirrored_count, in_place);
        let scanned = Instant::now();
        let (frame, tables, zero) = (crate::frame::current(), sync_tables(), zero_record_print());
        let (mut read, mut looked) = (0u64, 0u64);
        let mut due: Vec<(u32, u64)> = Vec::new();
        let mut found: Vec<(u32, bool)> = Vec::new();
        // The scans are those of the ranges, in their order (a job that did not
        // come back is not told apart: then none is noted).
        let noting = if scans.len() == pages.len() { noting } else { usize::MAX };
        {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            if copied.is_empty() {
                copied.resize(self.count as usize, 0);
            }
            for (n, scan) in scans.iter().enumerate() {
                let Some(scan) = scan else { continue };
                read += 1;
                for (i, &print) in scan.prints.iter().enumerate() {
                    if print == 0 {
                        continue;
                    }
                    let index = scan.first + i as u64;
                    looked += 1;
                    // None past the host heap's slots.
                    let Some(last) = copied.get_mut(index as usize) else {
                        continue;
                    };
                    // The record last seen, or a slot never looked at that
                    // holds nothing: as it was.
                    if *last == print || (*last == 0 && print == zero) {
                        continue;
                    }
                    // A table's slot lately: left to its draws, and looked
                    // at again next time.
                    let tabled = (n >= noting || !tables) && self.tabled_lately(index as u32, frame);
                    if n >= noting {
                        found.push((index as u32, tabled));
                    }
                    if !tables && tabled {
                        COPIED_TABLED.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    *last = print;
                    due.push((index as u32, start + index * step));
                }
            }
        }
        let compared = Instant::now();
        let makes = sync_makes();
        for &(index, handle) in &due {
            self.mirror_as(index, handle, false, makes);
        }
        let us = |a: Instant, b: Instant| b.duration_since(a).as_micros() as u64;
        let jobs = pages.len().div_ceil(JOB_PAGES).min(JOBS_MOST);
        for (count, n) in COUNT.iter().zip([
            1,
            pages.len() as u64,
            if pages.len() >= 2 * JOB_PAGES { jobs as u64 } else { 0 },
            us(began, scanned),
            us(scanned, compared),
            us(compared, Instant::now()),
        ]) {
            count.fetch_add(n, Ordering::Relaxed);
        }
        (read, looked, due.len() as u64, found)
    }
}

/// Every page of `pages` read and fingerprinted ([`scan_page`]), in the
/// pages' order; `None` for a page that was not there. A heap with fewer
/// than twice [`JOB_PAGES`] pages is done on this thread; else the pages go
/// to jobs for the kept threads and this one ([`crate::shadow::wide_run`]),
/// [`JOB_PAGES`] to a job at least.
fn scan_pages(pages: &[(usize, usize)], at: (u64, u64), mirrored_count: u32, in_place: bool) -> Vec<Option<Scanned>> {
    let scan_all = |chunk: &[(usize, usize)]| -> Vec<Option<Scanned>> {
        // The copy and the prints of a job's pages are made once for them.
        let (mut bytes, mut prints) = (Vec::new(), Vec::new());
        chunk.iter().map(|&page| scan_page(page, at, mirrored_count, in_place, &mut bytes, &mut prints)).collect()
    };
    if pages.len() < 2 * JOB_PAGES {
        return scan_all(pages);
    }
    let jobs = pages.len().div_ceil(JOB_PAGES).min(JOBS_MOST);
    let each = pages.len().div_ceil(jobs);
    let parts: Arc<Mutex<Parts>> = Arc::new(Mutex::new((0..jobs).map(|_| None).collect()));
    let work: Vec<Box<dyn FnOnce() + Send>> = pages
        .chunks(each)
        .enumerate()
        .map(|(n, chunk)| {
            let (chunk, parts) = (chunk.to_vec(), parts.clone());
            Box::new(move || {
                let mut bytes = Vec::new();
                let mut prints = Vec::new();
                // A job that panicked would leave the caller waiting for it
                // for ever: its pages count as not there (looked at again
                // next time).
                let done = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    chunk
                        .iter()
                        .map(|&page| scan_page(page, at, mirrored_count, in_place, &mut bytes, &mut prints))
                        .collect::<Vec<_>>()
                }))
                .unwrap_or_else(|_| chunk.iter().map(|_| None).collect());
                parts.lock().unwrap_or_else(|e| e.into_inner())[n] = Some(done);
            }) as Box<dyn FnOnce() + Send>
        })
        .collect();
    crate::shadow::wide_run(work);
    let parts = std::mem::take(&mut *parts.lock().unwrap_or_else(|e| e.into_inner()));
    parts.into_iter().flatten().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{View, changed_slots, create_cbv, create_placed_srv};
    use windows::Win32::Graphics::Direct3D12::D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

    const RESOURCE: usize = 0x62_0071;
    const STEP: u64 = 32;
    /// Slots of the test heap, and the bytes of a page of it (128 slots):
    /// sixteen pages, enough for the jobs of the batch.
    const COUNT: u32 = 2048;
    const PAGE_BYTES: usize = 4096;

    /// Cutting changed ranges into their pages: ranges that start or end
    /// inside a page give partial pages, as the pages' version has them.
    #[test]
    fn ranges_are_cut_at_pages() {
        let page = crate::watch::PAGE;
        let base = 16 * page;
        assert_eq!(pages_of(&[]), Vec::<(usize, usize)>::new());
        assert_eq!(pages_of(&[(base, base + page)]), vec![(base, base + page)]);
        assert_eq!(
            pages_of(&[(base + 64, base + 2 * page + 32), (base + 5 * page, base + 6 * page)]),
            vec![
                (base + 64, base + page),
                (base + page, base + 2 * page),
                (base + 2 * page, base + 2 * page + 32),
                (base + 5 * page, base + 6 * page),
            ]
        );
    }

    /// Writes slot `i`'s record: a constant buffer view of the test buffer in
    /// one slot of three, a shader view of a range of it in the next, none
    /// in the third; `salt` moves the views.
    fn write_record(slot: u64, i: u32, buffer: (u64, u64), salt: u32) {
        let (start, len) = buffer;
        match i % 3 {
            0 => {
                let address = start + u64::from((i + salt) % 200) * 256;
                create_cbv(std::ptr::null(), &[address as u32, (address >> 32) as u32, 256, 0], slot);
            }
            1 => {
                let mut resource = [0u32; 14];
                resource[0] = 1; // a buffer
                resource[4] = len as u32;
                let mut desc = [0u32; 16];
                desc[1] = 1; // a buffer view
                desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                desc[4] = (i + salt) % 64 * 4;
                desc[6] = 16;
                desc[7] = 16;
                desc[10] = start as u32;
                desc[11] = (start >> 32) as u32;
                create_placed_srv(std::ptr::null(), &resource, &desc, slot);
            }
            _ => {}
        }
    }

    /// The pages `[from, to)` of the title's memory.
    fn pages(memory: &[[u32; 8]]) -> Vec<(usize, usize)> {
        pages_of_size(memory, PAGE_BYTES)
    }

    fn pages_of_size(memory: &[[u32; 8]], bytes: usize) -> Vec<(usize, usize)> {
        let (from, len) = (memory.as_ptr() as usize, memory.len() * 32);
        (0..len.div_ceil(bytes)).map(|n| (from + n * bytes, from + ((n + 1) * bytes).min(len))).collect()
    }

    /// What the pages' version does for the changed pages `pages` of a heap
    /// whose title memory is `memory`: [`HostHeap::mirror_copied_page`] page
    /// by page, as `sync_heaps` does. The pages read, the records looked
    /// at and those mirrored.
    fn one_by_one(heap: &HostHeap, memory: &[[u32; 8]], pages: &[(usize, usize)]) -> (u64, u64, u64) {
        let (start, count) = (memory.as_ptr() as u64, memory.len() as u32);
        let (mut read, mut looked, mut mirrored) = (0, 0, 0);
        let mut bytes = Vec::new();
        for &(from, to) in pages {
            bytes.resize(to - from, 0);
            assert!(crate::host::read_into(from as *const u8, &mut bytes));
            let slots = changed_slots(from, to, start as usize, STEP, count);
            let (seen, made) = heap.mirror_copied_page(slots, (start, STEP), from, &bytes);
            read += 1;
            looked += seen;
            mirrored += made;
        }
        (read, looked, mirrored)
    }

    /// The pages of a heap looked at together mirror the slots, and make
    /// what is made of them, as the pages one by one do, round after round:
    /// new records, the same again, some moved, some cleared.
    #[test]
    fn a_batch_mirrors_what_the_pages_one_by_one_do() {
        let (Some(each), Some(batch)) = (HostHeap::new(false, COUNT), HostHeap::new(false, COUNT)) else {
            return; // no host GPU
        };
        let buffer = vec![0u8; 256 << 10];
        let buffer = (buffer.as_ptr() as u64, buffer.len() as u64);
        crate::shadow::register(RESOURCE, buffer.0, buffer.1);
        let mut memory = vec![[0u32; 8]; COUNT as usize];
        let start = memory.as_ptr() as u64;
        let pages = pages(&memory);
        assert!(pages.len() >= 2 * JOB_PAGES, "the pages go to jobs");
        let both = |memory: &[[u32; 8]]| {
            let one = one_by_one(&each, memory, &pages);
            let all = batch.mirror_copied_batch(&pages, (start, STEP), COUNT, false);
            assert_eq!(one, all);
            for i in 0..COUNT {
                assert_eq!(each.made(i), batch.made(i), "slot {i} made");
                assert_eq!(each.kept_at(i).is_some(), batch.kept_at(i).is_some(), "slot {i} shows a host resource");
            }
            all
        };
        for i in 0..COUNT {
            write_record(start + u64::from(i) * STEP, i, buffer, 0);
        }
        let (read, looked, mirrored) = both(&memory);
        assert_eq!((read, looked), (pages.len() as u64, u64::from(COUNT)));
        assert!(mirrored >= u64::from(COUNT) * 2 / 3, "the records written: {mirrored}");
        assert!(batch.made(0) && batch.kept_at(0).is_some(), "a view was made");
        assert_eq!(both(&memory).2, 0, "all as last seen");
        for i in (0..COUNT).filter(|i| i % 5 == 0) {
            write_record(start + u64::from(i) * STEP, i, buffer, 7);
        }
        for i in (0..COUNT).filter(|i| i % 11 == 0) {
            memory[i as usize] = [0; 8];
        }
        assert!(both(&memory).2 > 0, "some moved, some cleared");
        assert_eq!(both(&memory).2, 0);
        crate::shadow::unregister(RESOURCE, buffer.0);
    }

    /// The ranges from `noting` on say which records changed on them: those of
    /// the marked runs before them are mirrored all the same and not told, and
    /// a record that did not change is told of by none.
    #[test]
    fn records_changed_on_the_noted_ranges_are_told() {
        for in_place in [false, true] {
            let Some(heap) = HostHeap::new(false, COUNT) else {
                eprintln!("no host GPU: skipped");
                return;
            };
            let buffer = vec![0u8; 256 << 10];
            let buffer = (buffer.as_ptr() as u64, buffer.len() as u64);
            crate::shadow::register(RESOURCE, buffer.0, buffer.1);
            let mut memory = vec![[0u32; 8]; COUNT as usize];
            let start = memory.as_mut_ptr() as u64;
            let at = |slot: u64| start + slot * STEP;
            for i in (0..COUNT).filter(|i| i % 3 == 0) {
                write_record(at(u64::from(i)), i, buffer, 0);
            }
            let pages = pages(&memory);
            // Everything seen once; nothing is told.
            let (_, _, _, found) = heap.mirror_copied_batch_noting(&pages, (start, STEP), COUNT, in_place, 0);
            assert!(!found.is_empty(), "the first look finds the records new");
            let (_, _, made, found) = heap.mirror_copied_batch_noting(&pages, (start, STEP), COUNT, in_place, 0);
            assert_eq!((made, found.len()), (0, 0), "nothing changed, nothing told");
            // A record of the run (slot 3) and one of the page after (slot 201), changed.
            write_record(at(3), 3, buffer, 7);
            write_record(at(201), 201, buffer, 7);
            let run = (at(3) as usize, at(4) as usize);
            let page = pages[1];
            assert!((page.0 as u64..page.1 as u64).contains(&at(201)));
            // Noted from the page: the run's record is mirrored, not told.
            let (read, looked, made, found) = heap.mirror_copied_batch_noting(&[run, page], (start, STEP), COUNT, in_place, 1);
            assert_eq!((read, made), (2, 2), "both records mirrored");
            assert!(looked > 128, "the run's record and the page's: {looked}");
            assert_eq!(found, [(201, false)], "in place: {in_place}");
            // Seen now: not told again, whichever ranges are noted.
            let (_, _, made, found) = heap.mirror_copied_batch_noting(&[run, page], (start, STEP), COUNT, in_place, 0);
            assert_eq!((made, found.len()), (0, 0));
            // Without noting, as `mirror_copied_batch`.
            write_record(at(204), 204, buffer, 9);
            assert_eq!(heap.mirror_copied_batch(&[page], (start, STEP), COUNT, in_place).2, 1);
            crate::shadow::unregister(RESOURCE, buffer.0);
        }
    }

    /// Pages read on jobs come back as the same pages read here, in order,
    /// with the pages not there left out; the fingerprints are those of the
    /// records.
    #[test]
    fn pages_read_on_jobs_are_those_read_here() {
        let mut memory = vec![[0u32; 8]; COUNT as usize];
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for word in memory.iter_mut().flatten() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *word = seed as u32;
        }
        let start = memory.as_ptr() as u64;
        let mut pages = pages(&memory);
        // A page that starts inside a record and ends inside another.
        pages.push((pages[0].0 + 8, pages[0].0 + 8 + 1000));
        for in_place in [false, true] {
            let by_jobs = scan_pages(&pages, (start, STEP), COUNT, in_place);
            let (mut bytes, mut prints) = (Vec::new(), Vec::new());
            let here: Vec<_> =
                pages.iter().map(|&page| scan_page(page, (start, STEP), COUNT, in_place, &mut bytes, &mut prints)).collect();
            assert_eq!(by_jobs, here, "in place: {in_place}");
            let whole = by_jobs[0].as_ref().expect("a page of the title's memory");
            assert_eq!(whole.first, 0);
            let record: Vec<u8> = memory[5].iter().flat_map(|w| w.to_le_bytes()).collect();
            assert_eq!(whole.prints[5], super::super::record_print(&record) | 1);
        }
        // The partial page, read through a copy: its first record starts
        // before it and its last ends after it, and both are left out (0).
        let (mut bytes, mut prints) = (Vec::new(), Vec::new());
        let cut = scan_page(*pages.last().unwrap(), (start, STEP), COUNT, false, &mut bytes, &mut prints).expect("there");
        assert_eq!((cut.first, cut.prints.first(), cut.prints.last()), (0, Some(&0), Some(&0)));
        assert!(cut.prints[1..cut.prints.len() - 1].iter().all(|&p| p != 0));
    }

    /// A heap of the world's size synced both ways, a changed page at a time
    /// and all together: the milliseconds of each for frames of 0, 6,000 and
    /// 14,000 records changed over 342 changed pages of 64 KiB (700,416
    /// slots looked at, as a frame of the open world has), and what a host
    /// view of a buffer costs to make alone, to say how much of the
    /// mirroring there is for more threads to take.
    /// `cargo test --release -p victory-gpu --lib world_frame -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark: run in release with --ignored --nocapture"]
    fn world_frame_sync() {
        const PAGES: usize = 342;
        const BYTES: usize = 64 << 10;
        const SLOTS: u32 = (PAGES * BYTES / 32) as u32;
        const ROUNDS: usize = 9;
        let (Some(each), Some(batch)) = (HostHeap::new(false, SLOTS), HostHeap::new(false, SLOTS)) else {
            eprintln!("no host GPU: skipped");
            return;
        };
        let buffer = vec![0u8; 4 << 20];
        let buffer = (buffer.as_ptr() as u64, buffer.len() as u64);
        crate::shadow::register(RESOURCE, buffer.0, buffer.1);
        let mut memory = vec![[0u32; 8]; SLOTS as usize];
        let start = memory.as_ptr() as u64;
        let pages = pages_of_size(&memory, BYTES);
        // Records of distinct views, two sets of them, as a heap rewritten
        // every frame holds.
        let mut scratch = [0u32; 8];
        let pool: Vec<Vec<[u32; 3]>> = (0..2)
            .map(|set| {
                (0..4096u32)
                    .map(|k| {
                        scratch = [0; 8];
                        write_record(scratch.as_mut_ptr() as u64, k, buffer, set * 13);
                        [scratch[0], scratch[1], scratch[2]]
                    })
                    .collect()
            })
            .collect();
        let write = |memory: &mut Vec<[u32; 8]>, slot: usize, record: [u32; 3]| memory[slot][..3].copy_from_slice(&record);
        // Seeded: some 200,000 slots hold a view, and both heaps know them.
        for page in 0..PAGES {
            for k in 0..600 {
                let slot = page * (BYTES / 32) + k;
                write(&mut memory, slot, pool[0][slot % 4096]);
            }
        }
        one_by_one(&each, &memory, &pages);
        batch.mirror_copied_batch(&pages, (start, STEP), SLOTS, false);
        let median = |mut times: Vec<f64>| {
            times.sort_by(f64::total_cmp);
            (times[times.len() / 2], times[0], times[times.len() - 1])
        };
        for changed in [0usize, 6_000, 14_000] {
            let (mut serial, mut together) = (Vec::new(), Vec::new());
            for round in 0..ROUNDS {
                let set = &pool[(round + 1) % 2];
                for k in 0..changed {
                    let slot = (k % PAGES) * (BYTES / 32) + (k / PAGES) % (BYTES / 32);
                    write(&mut memory, slot, set[k % 4096]);
                }
                let at = std::time::Instant::now();
                let one = one_by_one(&each, &memory, &pages);
                serial.push(at.elapsed().as_secs_f64() * 1e3);
                let at = std::time::Instant::now();
                let all = batch.mirror_copied_batch(&pages, (start, STEP), SLOTS, false);
                together.push(at.elapsed().as_secs_f64() * 1e3);
                assert_eq!(one, all);
            }
            let (a, b) = (median(serial), median(together));
            eprintln!(
                "{changed} changed: a page at a time {:.2} ms (best {:.2}, worst {:.2}), together {:.2} ms (best {:.2}, worst {:.2})",
                a.0, a.1, a.2, b.0, b.1, b.2
            );
        }
        // What making the host views of the records alone costs.
        let views: Vec<View> = pool[0]
            .iter()
            .filter(|record| record[0] == crate::descriptor::VIEW_TAG)
            .filter_map(|record| crate::descriptor::view(record.as_ptr() as u64))
            .collect();
        let at = std::time::Instant::now();
        let mut made = 0;
        for (k, view) in views.iter().enumerate().cycle().take(20_000) {
            made += usize::from(super::super::make(view, each.cpu((k % 1024) as u32)).is_some());
        }
        eprintln!(
            "{} views of buffers ({made} made of 20,000 makes): {:.2} us each to make alone",
            views.len(),
            at.elapsed().as_secs_f64() * 1e6 / 20_000.0
        );
        crate::shadow::unregister(RESOURCE, buffer.0);
    }
}
