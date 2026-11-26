//! Which slots of a view heap may be in its placed texture spans
//! ([`super::Spans`]), told without the spans' lock
//! (`VICTORY_SPANS_QUICK=1`; off by default).
//!
//! After avatarmanz/VIctory-7900XTX `a284ca7` and `288af58` (GPL-3.0).
//! Every record the title writes into its heap goes through
//! `HostHeap::note_buffer`, which took the heap's one spans lock to forget
//! the slot there, whatever the record was: of the writes a frame most are of
//! no placed texture view, and the slot is in no span, so there is nothing
//! to forget. A bit per slot says "may be in the spans" (set where the spans
//! take the slot in, cleared where they let it go; a slot another view made
//! stale stays set until it is forgotten, which says only "may"): a write of
//! another kind to a slot with the bit clear leaves the spans alone.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Whether a write of no placed texture view leaves the spans alone where its
/// slot is not in them (`VICTORY_SPANS_QUICK=1`; otherwise the lock is taken
/// for every write, as before).
pub(crate) fn quick() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // (With `VICTORY_MIRROR_FAST` mirroring the short way too: the mirroring of changed records
    // wrote a slot of no span some 20,000 times a frame with the lock.)
    *ON.get_or_init(|| std::env::var_os("VICTORY_SPANS_QUICK").is_some_and(|v| v == "1")) || super::mirror_fast::short()
}

/// Since the last frame line: writes that left the spans alone, and writes
/// that took their lock.
pub(crate) static COUNT: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A bit for each slot of a heap, shared by the heap and its spans.
#[derive(Clone, Default)]
pub(super) struct Mask(Arc<[AtomicU64]>);

impl Mask {
    /// For `slots` slots, none set.
    pub(super) fn new(slots: usize) -> Mask {
        Mask((0..slots.div_ceil(64)).map(|_| AtomicU64::new(0)).collect())
    }

    /// Slot `index` is in the spans, or may be.
    pub(super) fn put(&self, index: u32) {
        if let Some(word) = self.0.get(index as usize / 64) {
            word.fetch_or(1 << (index % 64), Ordering::Release);
        }
    }

    /// Slot `index` is in no span.
    pub(super) fn remove(&self, index: u32) {
        if let Some(word) = self.0.get(index as usize / 64) {
            word.fetch_and(!(1 << (index % 64)), Ordering::Release);
        }
    }

    /// Whether slot `index` may be in the spans. A slot past the mask's
    /// room may.
    pub(super) fn may_hold(&self, index: u32) -> bool {
        self.0.get(index as usize / 64).is_none_or(|word| word.load(Ordering::Acquire) >> (index % 64) & 1 != 0)
    }

    /// The slots set, in order (tests).
    #[cfg(test)]
    pub(super) fn all(&self) -> Vec<u32> {
        (0..self.0.len() as u32 * 64).filter(|&i| self.may_hold(i)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Mask;

    #[test]
    fn a_mask_holds_what_was_put_in() {
        let mask = Mask::new(200);
        assert!(mask.all().is_empty());
        for slot in [0, 1, 63, 64, 199] {
            mask.put(slot);
        }
        assert_eq!(mask.all(), [0, 1, 63, 64, 199]);
        mask.remove(63);
        mask.remove(150);
        assert_eq!(mask.all(), [0, 1, 64, 199]);
        assert!(mask.may_hold(1) && !mask.may_hold(2));
        // Past its room (and with none made): may be there.
        assert!(mask.may_hold(100_000));
        assert!(Mask::default().may_hold(0));
    }
}
