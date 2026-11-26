//! What shaders wrote, put into title view memory beside the guards
//! (`VICTORY_COPY_BACK_BESIDE=1`; off by default). After Marvisak/VIctory
//! pull request 3069 ("marked pages go back beside the guards"), written
//! afresh for this tree's guards.
//!
//! A copy-back of a page of view memory held its guards open
//! (`crate::watch::writing_held`: the page's blocks made writable, and the
//! same pages at every other view), had the kernel write the bytes, and gave
//! the hold back, after which the next check made the block read-only again
//! and the title's first write to it faulted once more: two changes of
//! protection a block and a fault, the 64 KiB page that the copy-back wrote
//! (the pull request measured 45 ms of its 60 ms copy-back in 2,250 pages a
//! frame). Here the bytes are stored through a view of our own on the same
//! section (`crate::inplace::write_own_runs`), which no guard covers: the
//! title's views stay read-only and show the new bytes (the same pages), and
//! what the opening would have told is told by hand: the guards
//! (`crate::watch::wrote_beside`: the blocks are not clean, their regions'
//! change count moves, armed blocks keep their bytes) and the journal
//! (`crate::watch::wrote_own`: every copy of the page, at any of its views,
//! takes it again at its next refresh).
//!
//! `VICTORY_COPY_BACK_BESIDE=2` also notes the page current for the copy that
//! wrote it when it was current before, as `VICTORY_BACK_OWN_VIEW=1` does
//! for the relay's merges: the copy holds what the title now holds there
//! (the copy-back put the same bytes into both), so its next refresh need
//! not take the page in again. Left out of `1`, where the page is copied
//! once more at the next refresh, as it was after a kernel write.
//!
//! Only runs that lie in view memory the runtime has shown us
//! (`write_own_runs` finds the section and page); any other run, and any run
//! on a page that is in more than [`crate::watch::MANY_VIEWS`] views or while
//! the runtime changes mappings, is written the old way, so a copy that is
//! not in view memory is untouched by the switch.

use std::sync::atomic::{AtomicU64, Ordering};

use super::Shadow;
use crate::watch::PAGE;

/// `VICTORY_COPY_BACK_BESIDE`: 0 off (unset), 1 written beside the guards, 2
/// also the copy that wrote notes the page current.
pub(super) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_COPY_BACK_BESIDE").as_deref().map(str::trim) {
        Ok("1") => 1,
        Ok("2") => 2,
        _ => 0,
    })
}

/// Since the start: pages written beside the guards, their bytes, pages left
/// to the old way, and pages noted current for the copy that wrote them.
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The runs `(offset, bytes)` of a stretch at title address `at`, cut at the
/// 64 KiB pages of title memory: `(page number, offset, bytes)` in order.
fn pieces(at: usize, runs: &[(usize, usize)]) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::with_capacity(runs.len());
    for &(off, len) in runs {
        let (mut from, end) = (off, off + len);
        while from < end {
            let page = (at + from) / PAGE;
            let next = ((page + 1) * PAGE - at).min(end);
            out.push((page, from, next - from));
            from = next;
        }
    }
    out
}

impl Shadow {
    /// Writes the `runs` (offset, bytes; `None`: all of it) of `bytes`,
    /// which lie at title address `at`, into view memory beside the guards,
    /// a page at a time. Returns the runs it could not write that way, for
    /// the caller to write as before (empty: all are in).
    pub(super) fn write_beside(&self, at: usize, bytes: &[u8], runs: Option<&[(usize, usize)]>) -> Vec<(usize, usize)> {
        let whole = [(0, bytes.len())];
        let pieces = pieces(at, runs.unwrap_or(&whole));
        let (mut left, mut i) = (Vec::new(), 0);
        while i < pieces.len() {
            let page = pieces[i].0;
            let j = i + pieces[i..].iter().take_while(|p| p.0 == page).count();
            let list: Vec<(usize, usize)> = pieces[i..j].iter().map(|&(_, off, len)| (off, len)).collect();
            if !self.write_page_beside(at, bytes, page, &list) {
                left.extend_from_slice(&list);
            }
            i = j;
        }
        left
    }

    /// [`Shadow::write_beside`] for the runs `list` of one page. False with
    /// the runs not written if view memory does not hold them.
    fn write_page_beside(&self, at: usize, bytes: &[u8], page: usize, list: &[(usize, usize)]) -> bool {
        let (first, end) = (list[0].0, list.last().map_or(0, |&(off, len)| off + len));
        let (low, span) = (at + first, end - first);
        // Told before the write, so that armed blocks keep their bytes from
        // before it, and after it, so that a copy that took the block
        // between does not keep the bytes from before.
        crate::watch::wrote_beside(low, span, true);
        let Some(shown) = crate::inplace::write_own_runs(at, bytes, list) else {
            STATS[2].fetch_add(1, Ordering::Relaxed);
            return false;
        };
        crate::watch::wrote_beside(low, span, false);
        let others: Vec<usize> = shown.iter().map(|&a| a / PAGE).filter(|&p| p != page).collect();
        let (before, now) = crate::watch::wrote_own(page, &others);
        // A placed texture there changed with no descriptor written (what
        // `crate::watch::writing_held` tells the texture table).
        crate::descriptor::host::remapped(low, span);
        if mode() >= 2 {
            let base = self.start as usize / PAGE;
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(taken) = page.checked_sub(base).and_then(|i| copied.get_mut(i))
                && *taken != 0
                && *taken >= before
            {
                *taken = (*taken).max(now);
                STATS[3].fetch_add(1, Ordering::Relaxed);
            }
        }
        STATS[1].fetch_add(list.iter().map(|&(_, len)| len as u64).sum(), Ordering::Relaxed);
        let n = STATS[0].fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_multiple_of(16384) {
            tracing::info!(
                "what shaders wrote, put into view memory beside the guards (VICTORY_COPY_BACK_BESIDE={}), since the start: {n} pages {} MB, \
                 {} pages noted current for the copy that wrote them (not copied in again), {} pages written the old way (not view memory, \
                 a page of very many views, or the runtime changing mappings)",
                mode(),
                STATS[1].load(Ordering::Relaxed) >> 20,
                STATS[3].load(Ordering::Relaxed),
                STATS[2].load(Ordering::Relaxed)
            );
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs are cut at the pages of title memory, in order, and a run that
    /// ends where a page does is not cut again.
    #[test]
    fn runs_are_cut_at_the_pages() {
        let at = 5 * PAGE + 100;
        assert_eq!(pieces(at, &[(0, 10), (20, 5)]), [(5, 0, 10), (5, 20, 5)]);
        // A run across a page boundary: 100 bytes to the next, then the rest.
        let across = PAGE - 100;
        assert_eq!(pieces(at, &[(across - 8, 20)]), [(5, across - 8, 8), (6, across, 12)]);
        // Ending exactly at the boundary, and the whole stretch of three pages.
        assert_eq!(pieces(at, &[(PAGE - 100 - 16, 16)]), [(5, PAGE - 100 - 16, 16)]);
        let all = pieces(PAGE, &[(0, 3 * PAGE)]);
        assert_eq!(all, [(1, 0, PAGE), (2, PAGE, PAGE), (3, 2 * PAGE, PAGE)]);
        assert!(pieces(at, &[]).is_empty());
    }
}
