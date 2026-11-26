//! A list makes a view its tables hold once (`VICTORY_LIST_VIEWS=1`; off by
//! default).
//!
//! The title's culling and lighting dispatches each bind a table of their own
//! that differs in one constant buffer, and the same thirty views beside it.
//! A table made afresh for a draw (`crate::draw::State::tables`) made each of
//! its views anew: found the record's view, made its host descriptor, noted
//! what the slot kept. The same table with the same descriptors is made once
//! in a list already; here a view is made once in a list too, in a heap the
//! card does not see, and its descriptor is copied into the table's slot for
//! the next table that has it. What the descriptor is made from is the record
//! (`[VIEW_TAG, index, generation]`, which names one view for good), kept
//! with the host resource it shows and the write mark `make` leaves for it,
//! and what [`HostHeap::keep`] does for the slot is done for the copy as for
//! the first. Constant buffer views are not here (a draw's own copy of the
//! constants is made for each table).
//!
//! The made views are per thread and per recording of a list: a thread that
//! records another list, or the same list again after one reset, makes its
//! own; there are [`SLOTS`] of them, and a table that finds none left makes its
//! views as it did.
//!
//! After avatarmanz/VIctory-7900XTX 4e77a879 (`VICTORY_LIST_VIEWS`, there on by
//! default), without its notes of what a draw's views were brought up to date
//! at (`HostList::epoch`), which this tree does not have: every draw still
//! brings them up to date as before.

use super::*;

/// Views a thread keeps made for the recording it is at.
const SLOTS: u32 = 8192;

/// Whether a list makes a view its tables hold once (`VICTORY_LIST_VIEWS=1`).
pub(crate) fn list_views() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIST_VIEWS").is_some_and(|v| v == "1"))
}

/// What [`make`] left for the slot besides the descriptor: the mark table
/// entry of a view whose writes are marked ([`HostHeap::set_mark`]).
type Mark = Option<(usize, (u32, u32))>;

/// A hash for a record's words: the standard library's is several times what
/// this is, and it is run for every view of every table.
#[derive(Default)]
pub(crate) struct WordsHash(u64);

impl std::hash::Hasher for WordsHash {
    fn finish(&self) -> u64 {
        self.0 ^ (self.0 >> 32)
    }

    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(word));
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }

    fn write_u32(&mut self, n: u32) {
        self.write_u64(u64::from(n));
    }
}

struct ListView<V> {
    view: V,
    /// Its descriptor's slot in [`ListViews`].
    at: u32,
    kept: Kept,
    mark: Mark,
}

/// The descriptors a thread made for one recording of a list, by the record
/// they were made from, in a heap the card does not see.
struct ListViews<V> {
    recording: u64,
    made: std::collections::HashMap<[u32; 3], ListView<V>, std::hash::BuildHasherDefault<WordsHash>>,
}

impl<V> ListViews<V> {
    fn new() -> Self {
        ListViews { recording: 0, made: Default::default() }
    }

    /// The made view of `words` in `recording`; made by `make` into the slot it
    /// is given if this recording has none yet. `None` if `make` makes none or
    /// there is no slot left. A thread that comes to another recording forgets
    /// what it made.
    fn entry(
        &mut self,
        recording: u64,
        words: [u32; 3],
        make: impl FnOnce(u32) -> Option<(V, Kept, Mark)>,
    ) -> Option<&ListView<V>> {
        if self.recording != recording {
            self.recording = recording;
            self.made.clear();
        }
        if !self.made.contains_key(&words) {
            let at = self.made.len() as u32;
            if at >= SLOTS {
                return None;
            }
            let (view, kept, mark) = make(at)?;
            self.made.insert(words, ListView { view, at, kept, mark });
        }
        self.made.get(&words)
    }
}

/// The heap of the descriptors a thread made, apart from the views.
struct Slots {
    heap: ID3D12DescriptorHeap,
    step: usize,
    start: usize,
}

impl Slots {
    fn new() -> Option<Slots> {
        let host = host()?;
        // SAFETY: plain object creation.
        unsafe {
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: SLOTS,
                    ..Default::default()
                })
                .ok()?;
            let step = host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            let start = heap.GetCPUDescriptorHandleForHeapStart().ptr;
            Some(Slots { heap, step, start })
        }
    }

    fn slot(&self, i: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        debug_assert!(i < SLOTS && !windows::core::Interface::as_raw(&self.heap).is_null());
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: self.start + i as usize * self.step }
    }
}

thread_local! {
    static LIST_VIEWS: std::cell::RefCell<Option<(Slots, ListViews<View>)>> = const { std::cell::RefCell::new(None) };
}

/// Since the last frame line: views made, and views copied from one made
/// already (diagnostics builds).
#[cfg(feature = "diagnostics")]
static COUNTS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The frame line: made and copied views per frame.
#[cfg(feature = "diagnostics")]
pub(crate) fn line(frames: f64) -> Option<String> {
    use std::sync::atomic::Ordering::Relaxed;
    let [made, copied] = [0, 1].map(|i| COUNTS[i].swap(0, Relaxed) as f64 / frames);
    (made + copied > 0.0).then(|| {
        format!(
            "views of the tables, per frame (VICTORY_LIST_VIEWS=1): {made:.0} made, {copied:.0} copied from one made already in the list"
        )
    })
}

impl HostHeap {
    /// Makes slot `index` (a dynamic one) the view the record `words` names,
    /// from this thread's descriptor of it if `recording` has made one, else
    /// as [`HostHeap::make_at`] does, and notes what it shows
    /// ([`HostHeap::keep`]). The view, or `None` (the slot is not made): not
    /// a view for this ([`HostHeap::make_at`] is the way): a sampler heap, a
    /// constant buffer, a record that names none, no room left.
    pub(crate) fn make_listed(&self, recording: u64, words: [u32; 3], index: u32) -> Option<View> {
        // (A constant buffer's record is not looked at here at all.)
        if self.sampler || words[2] >> crate::descriptor::KIND_SHIFT == crate::descriptor::CONSTANT {
            return None;
        }
        let host = host()?;
        LIST_VIEWS.with(|local| {
            let mut local = local.borrow_mut();
            if local.is_none() {
                *local = Some((Slots::new()?, ListViews::new()));
            }
            let (slots, views) = local.as_mut()?;
            #[cfg(feature = "diagnostics")]
            let before = views.made.len();
            let made = views.entry(recording, words, |at| {
                let view = view_at(words[1], words[2])?;
                // (A constant buffer's bytes are copied for each table.)
                if matches!(view, View::Constant(..)) {
                    return None;
                }
                let kept = match make(&view, slots.slot(at)) {
                    Some(kept) => kept,
                    None => {
                        null_for(&view, slots.slot(at));
                        None
                    }
                };
                // What `make` left for the slot, taken now: the next view made
                // on this thread would take its place.
                let mark = MARK.with(std::cell::Cell::take);
                Some((view, kept, mark))
            })?;
            let (at, view, kept, mark) = (made.at, made.view.clone(), made.kept.clone(), made.mark);
            #[cfg(feature = "diagnostics")]
            COUNTS[usize::from(views.made.len() == before)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // SAFETY: a slot of our heap, and one of this thread's heap.
            unsafe {
                host.device.CopyDescriptorsSimple(
                    1,
                    self.cpu(index),
                    slots.slot(at),
                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                )
            };
            // As `make` left it for the first slot: what `keep` takes.
            MARK.with(|m| m.set(mark));
            self.keep(index, kept);
            Some(view)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tables of records that differ get hashes that differ (the key of a
    /// table made afresh, `VICTORY_RECORD_HASH=1`).
    #[test]
    fn records_that_differ_hash_apart() {
        use std::hash::{Hash, Hasher};
        let of = |records: &[[u32; 3]]| {
            let mut h = WordsHash::default();
            records.iter().for_each(|r| r.hash(&mut h));
            h.finish()
        };
        let mut seen = std::collections::HashSet::new();
        // One record, one word or two of it different; runs of zeros; order.
        for a in 0..64u32 {
            for b in 0..64u32 {
                for c in [0, 1, 0x8000_0000, u32::MAX] {
                    assert!(seen.insert(of(&[[a, b, c]])), "{a} {b} {c}");
                    assert!(seen.insert(of(&[[1, 2, 3], [a, b, c]])), "{a} {b} {c} after");
                    assert!(seen.insert(of(&[[a, b, c], [1, 2, 3], [0, 0, 0]])), "{a} {b} {c} before");
                }
            }
        }
        assert!(seen.insert(of(&[])));
        // (One record of zeros is among the above.)
        let zeros: Vec<u64> = (2..100).map(|n| of(&vec![[0; 3]; n])).collect();
        assert!(zeros.iter().all(|h| seen.insert(*h)));
    }

    /// A view is made once in a recording, and again in the next one; a
    /// record that gives no view is asked again; the slots run out.
    #[test]
    fn a_view_is_made_once_in_a_recording() {
        let mut views: ListViews<u32> = ListViews::new();
        let made = std::cell::RefCell::new(Vec::new());
        let ask = |views: &mut ListViews<u32>, recording: u64, words: [u32; 3], gives: bool| {
            views
                .entry(recording, words, |at| {
                    made.borrow_mut().push((recording, words, at));
                    gives.then_some((words[1], None, None))
                })
                .map(|v| (v.at, v.view))
        };
        let (a, b) = ([7, 1, 1], [7, 2, 1]);
        assert_eq!(ask(&mut views, 1, a, true), Some((0, 1)));
        assert_eq!(ask(&mut views, 1, b, true), Some((1, 2)));
        // Again: not made again, the same slot.
        assert_eq!(ask(&mut views, 1, a, true), Some((0, 1)));
        assert_eq!(made.borrow().len(), 2);
        // Another recording makes its own, from the first slot.
        assert_eq!(ask(&mut views, 2, b, true), Some((0, 2)));
        assert_eq!(made.borrow().len(), 3);
        // And the first recording's views are not kept: it makes them again.
        assert_eq!(ask(&mut views, 1, a, true), Some((0, 1)));
        assert_eq!(made.borrow().len(), 4);
        // A record that gives no view is asked each time and takes no slot.
        let none = [7, 9, 1];
        assert_eq!(ask(&mut views, 1, none, false), None);
        assert_eq!(ask(&mut views, 1, none, false), None);
        assert_eq!(made.borrow().iter().filter(|m| m.1 == none).count(), 2);
        assert_eq!(ask(&mut views, 1, b, true), Some((1, 2)));
        // Distinct records up to the slots there are; one more has none.
        for i in 2..SLOTS {
            assert!(ask(&mut views, 1, [7, 100 + i, 1], true).is_some());
        }
        assert_eq!(ask(&mut views, 1, [7, 99_999, 1], true), None);
        // Those made are still there.
        assert_eq!(ask(&mut views, 1, a, true), Some((0, 1)));
    }

    fn words_of(record: &[u32; 8]) -> [u32; 3] {
        [record[0], record[1], record[2]]
    }

    /// A slot made from a view made in the list shows what a slot made by
    /// [`HostHeap::make_at`] does, the second slot from the first one's
    /// descriptor; a constant buffer and a sampler heap are not made this way.
    /// (The view is of no resource: the host makes a null view for it, which
    /// is a call a device takes whatever is in the title's memory.)
    #[test]
    fn a_listed_slot_shows_what_make_at_shows() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let view = View::Shader(0, None);
        // The title's descriptor slot: the record of the view.
        let mut record = [0u32; 8];
        crate::descriptor::record(record.as_mut_ptr() as u64, view.clone());
        let words = crate::descriptor::record_at(record.as_ptr() as u64).expect("a record of a view");
        assert_eq!(words, words_of(&record));
        let first = heap.dynamic(3).expect("dynamic slots");
        heap.make_at(first, &view);
        let raw = |i: u32| heap.kept_at(i).map(|r| windows::core::Interface::as_raw(&r) as usize);
        // Twice in one recording: the same view comes back, the slots show what `make_at` showed.
        let a = heap.make_listed(1, words, first + 1).expect("a listed view");
        let b = heap.make_listed(1, words, first + 2).expect("the same, copied");
        assert!(matches!(a, View::Shader(0, None)) && matches!(b, View::Shader(0, None)));
        assert_eq!(raw(first + 1), raw(first));
        assert_eq!(raw(first + 2), raw(first));
        // Not a view of this kind: a sampler heap, and a constant buffer's record.
        let samplers = HostHeap::new(true, 8).expect("a sampler heap");
        assert!(samplers.make_listed(1, words, 0).is_none());
        let mut constant = [0u32; 8];
        crate::descriptor::record(constant.as_mut_ptr() as u64, View::Constant(0x1000, 256));
        let words = crate::descriptor::record_at(constant.as_ptr() as u64).expect("a record");
        assert!(heap.make_listed(1, words, first).is_none());
        heap.release(first, 3);
        // SAFETY: a plain query; the device took all of it.
        assert!(unsafe { host().expect("a host").device.GetDeviceRemovedReason() }.is_ok(), "the device was removed");
    }
}
