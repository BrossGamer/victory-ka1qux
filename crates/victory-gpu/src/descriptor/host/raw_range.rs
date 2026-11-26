//! The range of a raw buffer view ([`super::HostHeap::raw_at`],
//! [`super::HostHeap::raw_uav_at`]) kept inside its buffer.
//!
//! After fxceless/victory-fork PR #1266 (lichenlang89). The title's byte
//! offset and length used to become a view without being checked against the
//! buffer. `CreateShaderResourceView` and `CreateUnorderedAccessView` return
//! nothing, so a description that runs past the buffer is refused with no
//! line from us, and what a refused creation leaves in the slot is undefined:
//! the slot is shader-visible, so a later draw or dispatch may read (wrong
//! bytes, zeros) or write (a fault) through it.
//!
//! Now the start is used exactly as asked, the length is cut at what the
//! buffer holds and at `u32::MAX`, and a request that lies outside the buffer
//! gets the null view the heap already has. On by default;
//! `VICTORY_RAW_RANGE=0` makes the views as before.

use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

/// Whether the range of a raw view is kept inside its buffer (on by default;
/// `VICTORY_RAW_RANGE=0`: the range as the title asked it, as before).
fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RAW_RANGE").is_none_or(|v| v != "0"))
}

/// Where a raw buffer view starts and how many dwords it covers.
///
/// `FirstElement` is 64-bit and `NumElements` 32-bit, so only the length is
/// cut at `u32::MAX`; the start is used as asked and never moved. Moving it is
/// what a clamp would do, and clamping the start of a request that lies
/// entirely past the buffer turns it into a view of the buffer's tail: 100
/// bytes of buffer with an offset of 112 would view bytes 96 to 99 as if they
/// had been asked for.
#[derive(Clone, Copy)]
pub(super) struct RawRange {
    pub(super) first: u64,
    pub(super) elements: u32,
}

/// Why a request lost part of its range, for the counters below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cut {
    /// The start is at or past the buffer's last dword: nothing to view.
    Past,
    /// The length was cut to what the buffer holds.
    Length,
    /// The title named less than a dword: one element, as before this guard.
    Short,
    /// The start is not on the 16 bytes a raw view needs.
    Unaligned,
}

/// The view of `len` bytes from `offset` of `buffer`: `None` when the request
/// has nothing in the buffer (the caller makes a null view). Callers pass
/// `offset` on 16 bytes (`draw.rs` biases by `offset & 15` and makes a null
/// view for a root SRV that is not on 16; `memory.rs` passes 0): one off it is
/// refused here too. `kind` names the view (SRV, UAV) in the counters.
pub(super) fn of(
    kind: &'static str,
    buffer: &ID3D12Resource,
    offset: u64,
    len: u64,
) -> Option<RawRange> {
    if !on() {
        return Some(RawRange {
            first: offset / 4,
            elements: (len / 4).clamp(1, u64::from(u32::MAX)) as u32,
        });
    }
    // SAFETY: a live buffer.
    let size = unsafe { buffer.GetDesc() }.Width;
    let (range, cut) = raw_range(size, offset, len);
    if let Some(cut) = cut {
        count(kind, cut, offset, len, size);
    }
    range
}

fn raw_range(size: u64, offset: u64, len: u64) -> (Option<RawRange>, Option<Cut>) {
    let words = size / 4;
    if !offset.is_multiple_of(16) {
        return (None, Some(Cut::Unaligned));
    }
    let asked = offset / 4;
    if asked >= words {
        return (None, Some(Cut::Past));
    }
    let avail = words - asked;
    // A length under a dword made one element before this guard. The floor is
    // kept because nothing was gathered about why the title would ask for less
    // than a dword; the buffer's end still caps it.
    let want = (len / 4).max(1);
    let elements = want.min(avail).min(u64::from(u32::MAX));
    let cut = if elements < want {
        Some(Cut::Length)
    } else if len < 4 {
        Some(Cut::Short)
    } else {
        None
    };
    (
        Some(RawRange {
            first: asked,
            elements: elements as u32,
        }),
        cut,
    )
}

/// Counts what [`raw_range`] refused or cut, by kind (SRV, UAV) and cause, so
/// that a run can say whether the guard applied at all. One atomic per cell,
/// said on the first and every 4096th: no lock, nothing per view beyond that.
/// A legal, uncut request touches no counter.
fn count(kind: &'static str, cut: Cut, offset: u64, len: u64, size: u64) {
    static SRV: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
    static UAV: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
    let cells = if kind == "SRV" { &SRV } else { &UAV };
    let cell = match cut {
        Cut::Past => &cells[0],
        Cut::Length => &cells[1],
        Cut::Short => &cells[2],
        Cut::Unaligned => &cells[3],
    };
    let n = cell.fetch_add(1, Ordering::Relaxed) + 1;
    if n == 1 || n % 4096 == 0 {
        tracing::warn!(
            kind,
            cut = ?cut,
            offset,
            len,
            size,
            seen = n,
            "a raw buffer view the title's range did not cover (VICTORY_RAW_RANGE=0 makes it as asked)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rr(size: u64, offset: u64, len: u64) -> (Option<(u64, u32)>, Option<Cut>) {
        let (range, cut) = raw_range(size, offset, len);
        (range.map(|v| (v.first, v.elements)), cut)
    }

    fn range(size: u64, offset: u64, len: u64) -> Option<(u64, u32)> {
        rr(size, offset, len).0
    }

    /// 100 bytes of buffer and an offset of 112 is the case a clamp of the
    /// start gets wrong: it viewed bytes 96 to 99, which the request had
    /// asked nothing about.
    #[test]
    fn the_range_stays_inside_the_buffer() {
        assert_eq!(range(1024, 0, 256), Some((0, 64)));
        assert_eq!(range(1024, 768, 512), Some((192, 64)));
        assert_eq!(range(1024, 1024, 16), None, "a start at the end");
        assert_eq!(range(1024, 1040, 16), None, "a start past the end");
        assert_eq!(range(100, 112, 16), None, "no intersection: never the tail");
        assert_eq!(range(100, 96, 16), Some((24, 1)), "the last dword");
        assert_eq!(range(100, 96, 4), Some((24, 1)));
        assert_eq!(range(0, 0, 16), None, "an empty buffer");
        assert_eq!(range(3, 0, 16), None, "no whole dword");
        assert_eq!(range(1024, 0, 0), Some((0, 1)), "the floor of one element stands");
        assert_eq!(range(1024, 0, 1), Some((0, 1)), "as does a length under a dword");
        assert_eq!(range(1024, 3, 16), None, "not on 16 bytes");
        assert_eq!(range(1024, u64::MAX, 16), None, "no wrap, no clamp back");
        assert_eq!(range(1024, 0, u64::MAX), Some((0, 256)), "the length is cut");
    }

    /// Why a request was refused or cut, which is what the counters report.
    #[test]
    fn the_cause_is_the_one_the_counters_would_report() {
        assert_eq!(rr(100, 112, 16), (None, Some(Cut::Past)));
        assert_eq!(rr(1024, 3, 16), (None, Some(Cut::Unaligned)));
        assert_eq!(rr(1024, 0, 0), (Some((0, 1)), Some(Cut::Short)));
        assert_eq!(rr(1024, 768, 512), (Some((192, 64)), Some(Cut::Length)));
        // Nothing cut, nothing refused: no counter is touched at all.
        assert_eq!(rr(1024, 0, 256), (Some((0, 64)), None));
        assert_eq!(rr(1024, 0, 1020), (Some((0, 255)), None), "255 of 256 dwords");
    }

    /// `FirstElement` is 64-bit: a start past the 32-bit element index is kept
    /// as it is, not narrowed to `u32::MAX`.
    #[test]
    fn a_start_past_32_bits_is_kept() {
        let size = 1u64 << 40;
        let offset = (u64::from(u32::MAX) + 10) * 16;
        let range = range(size, offset, 64).expect("inside the buffer");
        assert_eq!(range.0, offset / 4);
        assert!(range.0 > u64::from(u32::MAX));
        assert_eq!(range.1, 16);
    }

    /// `NumElements` is 32-bit: the length alone is cut at `u32::MAX`.
    #[test]
    fn the_length_is_cut_at_32_bits() {
        let size = 1u64 << 40;
        assert_eq!(range(size, 0, u64::MAX), Some((0, u32::MAX)));
    }

    /// An aligned start far past a buffer is still refused: alignment is not a
    /// reason to accept a request, and the check is not reached by the
    /// unaligned case first.
    #[test]
    fn an_aligned_start_past_the_buffer_is_refused() {
        let size = 1u64 << 40;
        assert_eq!(rr(size, 1u64 << 60, 64), (None, Some(Cut::Past)));
        assert_eq!(rr(size, size, 64), (None, Some(Cut::Past)));
        assert_eq!(rr(size, size - 16, 64), (Some((size / 4 - 4, 4)), Some(Cut::Length)));
    }

    /// Every accepted range holds the invariants the guard promises, and no
    /// request the input rules allow is turned away for no reason.
    #[test]
    fn accepted_ranges_hold_their_invariant() {
        let sizes = [0u64, 1, 3, 4, 100, 1024, 4096, 1 << 30];
        let offsets = [0u64, 4, 16, 48, 96, 112, 1024, 1040, 4096, u64::MAX];
        let lens = [0u64, 1, 3, 4, 16, 512, 1 << 20, u64::MAX];
        for size in sizes {
            for offset in offsets {
                for len in lens {
                    let (range, _) = raw_range(size, offset, len);
                    let aligned = offset.is_multiple_of(16);
                    let inside = offset / 4 < size / 4;
                    match range {
                        Some(range) => {
                            assert!(range.elements > 0, "{size} {offset} {len}");
                            assert_eq!(
                                range.first,
                                offset / 4,
                                "{size} {offset} {len}: the start moved"
                            );
                            assert_eq!(range.first % 4, 0, "a raw view starts on 16 bytes");
                            assert!(
                                range.first + u64::from(range.elements) <= size / 4,
                                "{size} {offset} {len}: {} + {} past the buffer",
                                range.first,
                                range.elements
                            );
                        }
                        None => assert!(
                            !(aligned && inside),
                            "{size} {offset} {len}: refused though the rules allow it"
                        ),
                    }
                }
            }
        }
    }
}
