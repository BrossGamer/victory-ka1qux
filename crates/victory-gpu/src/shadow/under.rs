//! The title's bytes put into the CPU side of a copy in the card's memory as
//! they are compared (`VICTORY_UNDER_FUSED=1`; off by default), for
//! [`super::Shadow::merge_under_writes`].
//!
//! After avatarmanz/VIctory-7900XTX `6f3ede5` (GPL-3.0): there each 32-byte
//! block in which a word differs takes the title's bytes while the page is
//! compared, instead of one copy per run afterwards (190,000 runs in 400
//! pages a frame in one of their sessions). Written afresh for this tree,
//! whose runs are byte-exact and each counts against the submission's run
//! allowance (`VICTORY_MERGE_RUNS`).

use std::sync::atomic::{AtomicU64, Ordering};

/// Whether the pages put in under flights' shader writes are compared and
/// put in in one pass (`VICTORY_UNDER_FUSED=1`). Only with no gap joined
/// over (`VICTORY_MERGE_GAP` 0, the default): joined runs would need the
/// runs first, and the allowance counts them joined.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UNDER_FUSED").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line: 0 nanoseconds the pages' compare and put
/// took (with or without [`on`]), 1 pages compared in one pass, 2 blocks of
/// 32 bytes put in, 3 pages left half done because the run allowance ran
/// out.
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Adds `n` to counter `i` of [`STATS`] (0 to 3).
pub(super) fn count(i: usize, n: u64) {
    STATS[i].fetch_add(n, Ordering::Relaxed);
}

/// [`STATS`], taken.
pub(crate) fn stats() -> [u64; 4] {
    std::array::from_fn(|i| STATS[i].swap(0, Ordering::Relaxed))
}

/// `title` compared with `held` (of one length), and every 32-byte block of
/// `held` in which a byte differs takes `title`'s bytes (the bytes of the
/// block that do not differ are equal, so the same bytes end up there as
/// with one copy per run). The runs of bytes that differ are put onto
/// `runs`, as (offset, bytes) in order, a run that goes on from the block
/// before in the same one: the same as `changed_runs`' for the page.
///
/// `left` is the runs the caller may still take, each run counted once: a
/// block whose new runs are more than that is left as it is and the result
/// is false (the blocks before it are put in, and their runs are in `runs`
/// and taken from `left`). True: all of the page is in.
pub(super) fn changed_runs_put(title: &[u8], held: &mut [u8], left: &mut usize, runs: &mut Vec<(usize, usize)>) -> bool {
    assert_eq!(title.len(), held.len());
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2, and the slices are of one length.
        return unsafe { put_avx2(title, held, left, runs) };
    }
    put_scalar(title, held, left, runs)
}

/// The block `[at, at + 32)` (or the rest, shorter) with bit `k` set where
/// byte `k` differs.
fn mask_of(title: &[u8], held: &[u8], at: usize) -> u32 {
    let end = (at + 32).min(title.len());
    let mut mask = 0u32;
    for (k, (t, h)) in title[at..end].iter().zip(&held[at..end]).enumerate() {
        mask |= u32::from(t != h) << k;
    }
    mask
}

/// The runs a block's `mask` (bit `k`: byte `base + k` differs) makes, put
/// onto `runs` (the first joined to the last if it goes on from it), if they
/// fit in `left`: the number taken from it. False, and nothing done, if
/// not.
fn note_block(mask: u32, base: usize, left: &mut usize, runs: &mut Vec<(usize, usize)>) -> bool {
    let mut new = (mask & !(mask << 1)).count_ones() as usize;
    let goes_on = mask & 1 != 0 && runs.last().is_some_and(|last| last.0 + last.1 == base);
    if goes_on {
        new -= 1;
    }
    if new > *left {
        return false;
    }
    *left -= new;
    let mut rest = mask;
    while rest != 0 {
        let first = rest.trailing_zeros() as usize;
        let bytes = (rest >> first).trailing_ones() as usize;
        match runs.last_mut() {
            Some(last) if last.0 + last.1 == base + first => last.1 += bytes,
            _ => runs.push((base + first, bytes)),
        }
        let end = first + bytes;
        rest = if end >= 32 { 0 } else { rest & (u32::MAX << end) };
    }
    true
}

/// [`changed_runs_put`] without vector instructions.
fn put_scalar(title: &[u8], held: &mut [u8], left: &mut usize, runs: &mut Vec<(usize, usize)>) -> bool {
    let mut at = 0;
    while at < title.len() {
        let end = (at + 32).min(title.len());
        let mask = mask_of(title, held, at);
        if mask != 0 {
            if !note_block(mask, at, left, runs) {
                return false;
            }
            held[at..end].copy_from_slice(&title[at..end]);
            count(2, 1);
        }
        at = end;
    }
    true
}

/// # Safety
/// AVX2, and two slices of one length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn put_avx2(title: &[u8], held: &mut [u8], left: &mut usize, runs: &mut Vec<(usize, usize)>) -> bool {
    use std::arch::x86_64::*;
    let n = title.len();
    let (t, h) = (title.as_ptr(), held.as_mut_ptr());
    let mut i = 0;
    // Four blocks at a time: most of a page is the same.
    while i + 128 <= n {
        // SAFETY: inside both slices (`i + 128 <= n`).
        let equal = unsafe {
            [
                _mm256_cmpeq_epi8(_mm256_loadu_si256(t.add(i).cast()), _mm256_loadu_si256(h.add(i).cast())),
                _mm256_cmpeq_epi8(_mm256_loadu_si256(t.add(i + 32).cast()), _mm256_loadu_si256(h.add(i + 32).cast())),
                _mm256_cmpeq_epi8(_mm256_loadu_si256(t.add(i + 64).cast()), _mm256_loadu_si256(h.add(i + 64).cast())),
                _mm256_cmpeq_epi8(_mm256_loadu_si256(t.add(i + 96).cast()), _mm256_loadu_si256(h.add(i + 96).cast())),
            ]
        };
        let all = _mm256_and_si256(_mm256_and_si256(equal[0], equal[1]), _mm256_and_si256(equal[2], equal[3]));
        if _mm256_movemask_epi8(all) as u32 == u32::MAX {
            i += 128;
            continue;
        }
        for (k, eq) in equal.into_iter().enumerate() {
            let mask = !(_mm256_movemask_epi8(eq) as u32);
            if mask == 0 {
                continue;
            }
            let at = i + 32 * k;
            if !note_block(mask, at, left, runs) {
                return false;
            }
            // SAFETY: inside both slices; `held` is ours to write.
            unsafe { _mm256_storeu_si256(h.add(at).cast(), _mm256_loadu_si256(t.add(at).cast())) };
            count(2, 1);
        }
        i += 128;
    }
    while i + 32 <= n {
        // SAFETY: inside both slices (`i + 32 <= n`).
        let (now, was) = unsafe { (_mm256_loadu_si256(t.add(i).cast()), _mm256_loadu_si256(h.add(i).cast())) };
        let mask = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(now, was)) as u32);
        if mask != 0 {
            if !note_block(mask, i, left, runs) {
                return false;
            }
            // SAFETY: as above.
            unsafe { _mm256_storeu_si256(h.add(i).cast(), now) };
            count(2, 1);
        }
        i += 32;
    }
    // The rest, less than a block.
    if i < n {
        let mask = mask_of(title, held, i);
        if mask != 0 {
            if !note_block(mask, i, left, runs) {
                return false;
            }
            held[i..].copy_from_slice(&title[i..]);
            count(2, 1);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::changed_runs_put;
    use crate::shadow::changed_runs;

    /// A small generator, the same every run.
    fn next(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    /// The runs are `changed_runs`', `held` is `title` after, whatever the
    /// lengths (whole blocks or not) and the changes (single bytes, whole
    /// blocks, stretches across blocks, nothing).
    #[test]
    fn puts_the_title_in_and_finds_the_runs_changed_runs_finds() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for round in 0..400 {
            let n = [65536, 4096, 256, 260, 4, 0, 1, 1000, 33, 127, 129, 31, 32][round % 13];
            let held: Vec<u8> = (0..n).map(|_| next(&mut seed) as u8).collect();
            let mut title = held.clone();
            for _ in 0..(next(&mut seed) % 400) as usize {
                if n == 0 {
                    break;
                }
                let at = next(&mut seed) as usize % n;
                let len = [1, 3, 4, 40, 300][(next(&mut seed) % 5) as usize].min(n - at);
                for b in &mut title[at..at + len] {
                    *b = b.wrapping_add(1 + (next(&mut seed) % 255) as u8);
                }
            }
            let want = changed_runs(&title, &held);
            let mut merged = held.clone();
            let mut left = usize::MAX;
            let mut runs = Vec::new();
            assert!(changed_runs_put(&title, &mut merged, &mut left, &mut runs), "round {round}");
            assert_eq!(runs, want, "round {round}, {n} bytes");
            assert_eq!(merged, title, "round {round}, {n} bytes");
            assert_eq!(usize::MAX - left, want.len(), "round {round}: runs taken from the allowance");
            // The scalar way, the same.
            let mut again = held.clone();
            let (mut left, mut by_hand) = (usize::MAX, Vec::new());
            assert!(super::put_scalar(&title, &mut again, &mut left, &mut by_hand), "round {round}");
            assert_eq!((by_hand, again), (want, title), "round {round}: scalar");
        }
    }

    /// With room for fewer runs than the page has, what comes before the
    /// block that does not fit is put in and counted, the rest is left as it
    /// was, and the result says so.
    #[test]
    fn stops_before_the_block_whose_runs_do_not_fit() {
        let held = vec![0u8; 512];
        let mut title = held.clone();
        // Runs: bytes 3, 40, 41 and 100 (blocks 0, 1 and 3), and 300.
        for at in [3, 40, 41, 100, 300] {
            title[at] = 9;
        }
        let mut merged = held.clone();
        let (mut left, mut runs) = (3, Vec::new());
        assert!(!changed_runs_put(&title, &mut merged, &mut left, &mut runs));
        assert_eq!(runs, vec![(3, 1), (40, 2), (100, 1)]);
        assert_eq!(left, 0);
        assert_eq!(&merged[..128], &title[..128], "the blocks that fit are in");
        assert_eq!(&merged[128..], &held[128..], "the rest is as it was");
        // Exactly enough: all in.
        let mut merged = held.clone();
        let (mut left, mut runs) = (4, Vec::new());
        assert!(changed_runs_put(&title, &mut merged, &mut left, &mut runs));
        assert_eq!((left, merged), (0, title));
    }

    /// A run that goes on across a block's edge is one run, and takes one of
    /// the allowance.
    #[test]
    fn a_run_across_blocks_is_one_run() {
        let held = vec![1u8; 200];
        let mut title = held.clone();
        title[30..70].fill(2);
        let mut merged = held.clone();
        let (mut left, mut runs) = (1, Vec::new());
        assert!(changed_runs_put(&title, &mut merged, &mut left, &mut runs));
        assert_eq!((runs, left, merged), (vec![(30, 40)], 0, title));
    }
}
