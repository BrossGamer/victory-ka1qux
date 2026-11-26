//! The slots a frame's records changed, mirrored faster
//! (`VICTORY_MIRROR_FAST`; off by default).
//!
//! The pass over a heap's changed records ([`super::batched`],
//! [`HostHeap::mirror_copied_page`], [`HostHeap::mirror_copied_prints`])
//! finds which slots hold a record that is not the one last seen, cheaply
//! (0.7 ms reading and fingerprinting, 2.6 ms comparing, in a log of the
//! open world), and then mirrors them one after the other on the thread that
//! submits ([`HostHeap::mirror_as`]): some 27,700 a frame, 2.4 microseconds
//! each, 67 ms. What the switch takes:
//!
//! - `1` (or `heaps`): the heaps that have changed records are mirrored
//!   together ([`mirror_heaps`]);
//! - `short`: each batch of slots is mirrored the short way ([`Memory`], the
//!   rest of this note);
//! - `all`: both.
//!
//! # Where the time of a mirrored slot goes
//!
//! `mirror_bench::profile` (release, `--ignored --nocapture`) times each part
//! of [`HostHeap::mirror_as`] over a frame of the world's size and mix:
//! 1.2 to 1.4 microseconds a slot alone on this machine (2.4 in the game,
//! where the same few hundred buffers' locks and counts are asked for by the
//! recording threads at 2.4 million a second as well), in no one place. About
//! half is looking at things: the view a record names, its fingerprint, the
//! host copy of the memory a buffer view reads (found, or made on the view's
//! own bytes), the same host copy again for the range the pass over buffer
//! views is given, the host's description of the view, the host's call. The
//! other half is the slot's books under a lock each (the texture queue 0.09,
//! the spans 0.13, the sets of buffer view slots 0.09, the resource kept
//! 0.09, the fingerprint of what it shows 0.02), one slot after another in
//! the order the slots come.
//!
//! # The heaps together
//!
//! A frame of the open world changes the records of five heaps (190 pages in
//! all), and [`crate::descriptor::sync_heaps`] mirrored them one heap after
//! the other. A heap's books are its own (its `made`, `kept`, texture queue,
//! spans, sets of buffer view slots and fingerprints, each under a lock of
//! the heap), so with the switch the batches of the heaps run together, the
//! one with the most changed pages here and each of the others on a thread
//! of its own for the length of the batch. What they share is the title's
//! views and the host copies of its buffers, which the recording threads ask
//! for all the time. `mirror_bench::heaps` (the same pages mirrored one heap
//! after the other, on kept threads, and on a thread each): five heaps of
//! 5,500 records 29.0 ms, 7.3 ms on a thread each (3.9x; with three threads
//! asking for copies meanwhile 30.6 to 9.8 ms), one of 16,000 and four of
//! 2,900 31.9 to 15.2 ms, two of 13,800 27.6 to 13.6 ms, one heap of 27,700
//! no change. The time of a frame is that of the heap with the most changed
//! pages, not the sum. The kept threads of [`crate::shadow::wide_run`] gave
//! 1.1x to 1.8x on the five heaps, unevenly, and are not used. The frame
//! line says what the heaps came to.
//!
//! # The short way
//!
//! Mirroring the slots of a batch on several threads did not pay either: the
//! lookups of the host copies share a lock and the counts of the copies they
//! find, and what the threads found is cold in the cache of the thread that
//! goes on with it (12,000 slots took 2.6 ms to look at ahead and mirrored
//! in as long as before). The short way does less on this thread:
//!
//! - a buffer view that another slot of the batch had already (the same
//!   record: tables of draws name the same constants and structures over
//!   and over) takes what the first one's looking found: the view, its
//!   fingerprint, the host copy and the host's description of it, and the
//!   copy the pass is given ([`Ahead`]). The record is the view's name (a
//!   constant buffer view and a view of placed memory carry the view whole,
//!   a view of a resource object its table entry and generation), and what
//!   was found is used only if the title has registered or let go no buffer
//!   since ([`crate::shadow::buffer_generation`]), and, for a view of a
//!   resource object, if the view is still the one its entry names. The
//!   memory lives as long as the batch;
//! - the copies of buffers are asked for from this thread's memory of
//!   answers ([`crate::shadow::AnswersHere`], what `VICTORY_SHADOW_CACHE=1`
//!   does for every thread), not from the table under its lock each time;
//! - what a slot kept before goes to the retirement in one piece at the end
//!   of the batch, not one by one ([`retire`]): each was a box and a lock;
//! - the fingerprint of what the slot shows is put with the resource it
//!   keeps, in one hold of its lock, not two;
//! - a write of a slot in no span leaves the spans alone ([`super::spanned`],
//!   `VICTORY_SPANS_QUICK`).
//!
//! In the benchmark that is worth between -3% and +2% of a mix of the
//! world's (down to -29% for the kinds of view whose copies are of their own
//! bytes, when the records repeat as much as the benchmark's do, 83%), and
//! +7% to +11% on a small batch (the frame line counts the repeats the game
//! has).
//!
//! # What is the same
//!
//! The slots mirrored, and what is made of each, are those of the batch
//! without the switch: [`tests`] compares the books of two heaps given the
//! same records, round after round, and of heaps mirrored together with
//! those mirrored one after the other.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::{HostHeap, Plan, VIEW_TAG, View, buffer_copy, buffer_view_kind, plan_view, same_view_signature, sync_ranges, view_at};

/// What `VICTORY_MIRROR_FAST` asks for: the heaps together (`1`, `heaps`),
/// the slots the short way (`short`), or both (`all`).
#[derive(Clone, Copy)]
pub(crate) struct Mode {
    pub(crate) short: bool,
    pub(crate) heaps: bool,
}

/// [`Mode`] as the environment says (off by default).
pub(crate) fn mode() -> Mode {
    #[cfg(test)]
    if let Some(mode) = FORCED.with(Cell::get) {
        return mode;
    }
    static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match std::env::var("VICTORY_MIRROR_FAST").as_deref() {
            Ok("1" | "heaps") => Mode { short: false, heaps: true },
            Ok("short") => Mode { short: true, heaps: false },
            Ok("all") => Mode { short: true, heaps: true },
            _ => Mode { short: false, heaps: false },
        };
        if mode.short || mode.heaps {
            tracing::info!(
                short = mode.short,
                heaps = mode.heaps,
                "the slots a frame's records changed are mirrored {}{}{} (VICTORY_MIRROR_FAST)",
                if mode.short { "with their lookups shared and their locks spared" } else { "as before" },
                if mode.short && mode.heaps { ", " } else { "" },
                if mode.heaps { "and the heaps that have some together" } else { "" }
            );
        }
        if mode.heaps && !super::batched::on() {
            tracing::warn!(
                "VICTORY_MIRROR_FAST puts the heaps together only for the changed pages of a heap taken all at once: it needs VICTORY_HEAP_SYNC_BATCH=1"
            );
        }
        mode
    })
}

/// Whether the changed slots of a heap are mirrored the short way.
pub(crate) fn short() -> bool {
    mode().short
}

/// Whether the heaps that have changed records are mirrored together.
pub(crate) fn heaps() -> bool {
    mode().heaps
}

#[cfg(test)]
thread_local! {
    /// Tests: the switch as this thread sets it.
    pub(super) static FORCED: Cell<Option<Mode>> = const { Cell::new(None) };
}

thread_local! {
    /// Whether this thread is mirroring a batch of slots with the switch on.
    static IN_BATCH: Cell<bool> = const { Cell::new(false) };
    /// What the slots of the batch kept before, until the batch is done.
    static RETIRED: RefCell<Vec<crate::shadow::Held>> = const { RefCell::new(Vec::new()) };
}

/// Whether the calling thread is mirroring a batch of slots the short way.
pub(super) fn in_batch() -> bool {
    IN_BATCH.get()
}

/// What a slot kept before is done with: at the end of the batch, if a batch
/// is being mirrored here, else now ([`crate::host::retire`]).
pub(super) fn retire(old: crate::shadow::Held) {
    if in_batch() {
        RETIRED.with(|retired| retired.borrow_mut().push(old));
    } else {
        crate::host::retire(old);
    }
}

/// A batch being mirrored the short way: while it lives this thread asks for
/// copies of buffers from its memory of answers, and what slots let go is
/// kept for the end ([`retire`]).
struct Batch {
    _answers: crate::shadow::AnswersHere,
}

impl Batch {
    fn begin() -> Batch {
        IN_BATCH.set(true);
        Batch { _answers: crate::shadow::AnswersHere::begin() }
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        IN_BATCH.set(false);
        let retired = RETIRED.with(|retired| std::mem::take(&mut *retired.borrow_mut()));
        if !retired.is_empty() {
            crate::host::retire(retired);
        }
    }
}

/// Since the last frame line: batches mirrored the short way, the slots in
/// them, the buffer views among those that took what another slot of the
/// batch had found, and the microseconds the batches took; then the groups
/// of heaps mirrored together, the heaps in them, the microseconds they took
/// (waiting for them, this thread) and the microseconds of their heaps'
/// own batches, added.
static COUNT: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Tests: the counts since the last call (batches, slots, repeats,
/// microseconds).
#[cfg(test)]
pub(super) fn counts() -> [u64; 4] {
    [0, 1, 2, 3].map(|i| COUNT[i].swap(0, Ordering::Relaxed))
}

/// With the frame line: what the batches did per frame over `frames`.
pub(crate) fn line(frames: f64) {
    let [batches, slots, repeats, micros, groups, heaps, wall, sum] = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| COUNT[i].swap(0, Ordering::Relaxed));
    if batches > 0 {
        tracing::info!(
            "slots mirrored per frame the short way (VICTORY_MIRROR_FAST): {:.0} slots in {:.1} batches, {:.0} of them buffer views \
             that took what the batch had found for the same record already, {:.2} ms in the batches (all threads)",
            slots as f64 / frames,
            batches as f64 / frames,
            repeats as f64 / frames,
            micros as f64 / 1000.0 / frames
        );
    }
    if groups > 0 {
        tracing::info!(
            "heaps mirrored together per frame (VICTORY_MIRROR_FAST): {:.1} heaps with changed records, {:.2} ms for them all (this thread \
             waiting), {:.2} ms in their own batches added",
            heaps as f64 / frames,
            wall as f64 / 1000.0 / frames,
            sum as f64 / 1000.0 / frames
        );
    }
}

/// What was settled ahead for one slot, from the record it had then.
pub(super) struct Ahead {
    /// The first three words of the record: the tag and the view's name.
    pub(super) record: [u32; 3],
    /// The view the record names (`None`: no view of ours).
    pub(super) view: Option<View>,
    /// The fingerprint of the view ([`same_view_signature`]), if settled.
    pub(super) sig: Option<Option<u64>>,
    /// The buffer view settled as far as the host's call ([`plan_view`]), if
    /// the view is one that is made now.
    pub(super) plan: Option<Option<Plan>>,
    /// The host copy and range the pass over buffer views is given
    /// ([`buffer_copy`]), if the slot gives it one.
    pub(super) copy: Option<Option<(Arc<crate::shadow::Shadow>, u64, u64)>>,
    /// [`crate::shadow::buffer_generation`] when the host copies were looked
    /// for.
    pub(super) buffers: u64,
}

/// The settlings of the batch's buffer views by record, which another slot
/// with the same record takes.
#[derive(Default)]
struct Memory {
    found: HashMap<u64, Found, std::hash::BuildHasherDefault<Spread>>,
    repeats: u64,
}

/// What looking at a buffer view found, to be taken by the next slot with
/// its record.
struct Found {
    view: View,
    sig: Option<u64>,
    plan: Plan,
    copy: Option<(Arc<crate::shadow::Shadow>, u64, u64)>,
    buffers: u64,
}

/// A hasher for keys that are record words already (a view's index and
/// generation, or its address and size): one multiplication.
#[derive(Default)]
struct Spread(u64);

impl std::hash::Hasher for Spread {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }
    }

    fn write_u64(&mut self, n: u64) {
        let h = n.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 = h ^ (h >> 32);
    }
}

/// Settles what slot `index`, whose record is at `handle`, needs ahead
/// (`makes`: the buffer views are made now, [`super::sync_makes`]), taking
/// it from `memory` where another slot of the batch had the same record, and
/// leaving it there for the next. Looks at the title's views and buffers and
/// at nothing of the heap but the key its slot shows (a slot that shows its
/// record's view already is left as it is by [`HostHeap::mirror_as_with`],
/// and needs nothing).
fn settle(heap: &HostHeap, memory: &mut Memory, index: u32, handle: u64, makes: bool) -> Ahead {
    // SAFETY: the handle is a descriptor slot in title memory, as in
    // `mirror_as_with`, which reads the same words next.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    let mut ahead = Ahead { record: [tag, i, generation], view: None, sig: None, plan: None, copy: None, buffers: 0 };
    if index >= heap.count || tag != VIEW_TAG {
        return ahead;
    }
    let key = u64::from(generation) << 32 | u64::from(i);
    // What a slot that shows the view already needs: nothing, but the view.
    if heap.made_fast.get(index as usize).is_some_and(|fast| fast.load(Ordering::Acquire) == key) {
        ahead.view = view_at(i, generation);
        return ahead;
    }
    if let Some(found) = memory.found.get(&key)
        && found.buffers == crate::shadow::buffer_generation()
    {
        memory.repeats += 1;
        ahead.view = Some(found.view.clone());
        ahead.sig = Some(found.sig);
        ahead.plan = Some(Some(found.plan.clone()));
        ahead.copy = Some(found.copy.clone());
        ahead.buffers = found.buffers;
        return ahead;
    }
    let Some(view) = view_at(i, generation) else {
        return ahead;
    };
    ahead.sig = Some(if heap.sampler { None } else { same_view_signature(&view) });
    if makes && buffer_view_kind(&view) {
        ahead.buffers = crate::shadow::buffer_generation();
        ahead.plan = plan_view(&view);
        if sync_ranges() {
            ahead.copy = Some(buffer_copy(&view));
        }
        // What the next slot with this record takes: a view that is made
        // (not one the host cannot), and the range it gives.
        if let (Some(Some(plan)), Some(sig), Some(copy)) = (&ahead.plan, ahead.sig, &ahead.copy) {
            memory
                .found
                .insert(key, Found { view: view.clone(), sig, plan: plan.clone(), copy: copy.clone(), buffers: ahead.buffers });
        }
    }
    ahead.view = Some(view);
    ahead
}

impl HostHeap {
    /// Mirrors the slots `due` (title slot and its record's address, in slot
    /// order) as [`HostHeap::mirror_as`] does one after the other, the short
    /// way if `VICTORY_MIRROR_FAST=1` (module notes).
    pub(super) fn mirror_due(&self, due: &[(u32, u64)], makes: bool) {
        if !short() {
            for &(index, handle) in due {
                self.mirror_as(index, handle, false, makes);
            }
            return;
        }
        let began = Instant::now();
        let mut memory = Memory::default();
        {
            let _batch = Batch::begin();
            for &(index, handle) in due {
                let ahead = settle(self, &mut memory, index, handle, makes);
                self.mirror_as_with(index, handle, false, makes, Some(ahead));
            }
        }
        for (count, n) in COUNT.iter().zip([1, due.len() as u64, memory.repeats, began.elapsed().as_micros() as u64]) {
            count.fetch_add(n, Ordering::Relaxed);
        }
    }
}

/// Pages a heap has changed from which it gets a thread of its own in
/// [`mirror_heaps`] (the 64 KiB pages the title's writes are told by: 8 of
/// them are 16,000 records at most, some 17 ms; a thread to start takes
/// 50 to 100 microseconds).
const THREAD_PAGES: usize = 8;

/// A heap's changed pages for [`mirror_heaps`], as
/// [`HostHeap::mirror_copied_batch`] takes them.
pub(crate) struct Work {
    /// The heap: the caller's, held until [`mirror_heaps`] returns.
    pub(crate) host: *const HostHeap,
    pub(crate) pages: Vec<(usize, usize)>,
    pub(crate) at: (u64, u64),
    pub(crate) mirrored_count: u32,
    pub(crate) in_place: bool,
}

// SAFETY: a heap is shared by every thread of the title already; the
// pointer is read only while the caller waits for the job.
unsafe impl Send for Work {}

/// [`HostHeap::mirror_copied_batch`] for each of `work`'s heaps, together:
/// the heap with the most pages here, each of the others on a thread made
/// for it, which ends with it (the kept threads of [`crate::shadow::wide_run`]
/// were tried first, and scaled badly and unevenly in the benchmark: 1.1x to
/// 1.8x on five heaps where a thread each gave 3.9x; they also have the
/// scans of each heap's pages to do). The pages read, the records looked at
/// and the records mirrored, of all of them. A panic of a heap's batch
/// goes on here, as it would one after the other.
pub(crate) fn mirror_heaps(mut work: Vec<Work>) -> (u64, u64, u64) {
    let began = Instant::now();
    work.sort_by_key(|w| std::cmp::Reverse(w.pages.len()));
    let heaps = work.len() as u64;
    let run = |w: Work| -> ((u64, u64, u64), u64) {
        let at = Instant::now();
        // SAFETY: see `Work`.
        let host = unsafe { &*w.host };
        let done = host.mirror_copied_batch(&w.pages, w.at, w.mirrored_count, w.in_place);
        (done, at.elapsed().as_micros() as u64)
    };
    // The heaps with fewer pages than [`THREAD_PAGES`] are done here, after the
    // biggest: a thread costs more than they take.
    let mut work = work.into_iter();
    let Some(biggest) = work.next() else {
        return (0, 0, 0);
    };
    let (others, small): (Vec<Work>, Vec<Work>) = work.partition(|w| w.pages.len() >= THREAD_PAGES);
    let results: Vec<((u64, u64, u64), u64)> = if others.is_empty() {
        std::iter::once(biggest).chain(small).map(run).collect()
    } else {
        std::thread::scope(|scope| {
            let threads: Vec<_> = others
                .into_iter()
                .map(|w| {
                    let run = &run;
                    std::thread::Builder::new()
                        .name("victory-heap-sync".into())
                        .spawn_scoped(scope, move || {
                            // (The whole of it: it is the work that is `Send`, not its pointer.)
                            let w = w;
                            run(w)
                        })
                        .expect("a thread for a heap's changed records")
                })
                .collect();
            let mut results: Vec<_> = std::iter::once(biggest).chain(small).map(run).collect();
            for thread in threads {
                match thread.join() {
                    Ok(done) => results.push(done),
                    Err(panic) => std::panic::resume_unwind(panic),
                }
            }
            results
        })
    };
    let (mut read, mut looked, mut made, mut sum) = (0, 0, 0, 0);
    for ((r, l, m), us) in results {
        (read, looked, made, sum) = (read + r, looked + l, made + m, sum + us);
    }
    for (count, n) in COUNT[4..].iter().zip([1, heaps, began.elapsed().as_micros() as u64, sum]) {
        count.fetch_add(n, Ordering::Relaxed);
    }
    (read, looked, made)
}

#[cfg(test)]
mod tests;
