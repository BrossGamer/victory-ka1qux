//! Windows of title memory outside buffers that shaders are declared to
//! write and never do (`VICTORY_QUIET_WINDOWS=1`; off by default).
//!
//! A shader that can store through a root view or a buffer view of a window
//! has the whole range from the view's address to the window's end declared
//! written (a root view has no size). Each submission then snapshots the
//! declared bytes before the card runs and compares them with the copy after
//! it, to find what the shaders wrote: 156 MB a frame in one run, and its
//! biggest windows, 4 MiB each, gave back 0.000 MB (the study's census of
//! `copystat::compared`).
//!
//! A window is **quiet** after it has been compared [`Rules::after`] times in
//! a row (`VICTORY_QUIET_AFTER`, 64) and no byte of it went back to title
//! memory. From then on a declaration of a range inside what those compares
//! covered is not made at all: no step is queued for the list, so nothing is
//! snapshotted, compared or put in the submission's rules of shader writes,
//! and nothing is lost by it but what a shader that never wrote does next.
//! Anything else brings the window back to learning:
//!
//! - a declaration of bytes outside the extent compared (the shader or the
//!   view is not the one that was looked at);
//! - an **audit**: one declaration in [`Rules::audit`] (`VICTORY_QUIET_AUDIT`,
//!   32) is let through, and the compare after it that finds a byte written
//!   ends the quiet.
//!
//! What it costs when a shader does write after all: the bytes it wrote
//! between two audits stay in the copy and never reach title memory (the
//! copy keeps them for the shaders after it, until a refresh finds the page
//! written by the title and takes the title's bytes over them). A shader
//! that stores rarely and depending on data (a flag a culling pass raises
//! when something is visible) is the case that loses. `VICTORY_QUIET_WINDOWS=check` runs the
//! same rules and holds nothing back: every declaration is made and
//! compared as without the switch, and the log counts what the switch would
//! have held back and the windows found writing after they turned quiet
//! (and how many bytes): the number to look at before trusting `1`.
//!
//! Only windows made as copies in the PC's memory, and only declarations of
//! root views and buffer views the draws and dispatches make
//! (`Shadow::gpu_writes_viewed`): the page tables of title memory by address
//! (`crate::memory`) declare every numbered copy written for the
//! structure builder, whose output is read by later dispatches through the
//! copy, and keep doing so. A declaration through marked views is never held
//! back (`crate::marks` arms those ranges itself).

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// What `VICTORY_QUIET_WINDOWS` says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    /// Not set: nothing is tracked.
    Off,
    /// `1`: declarations of quiet windows are held back.
    On,
    /// `check`: the same rules, nothing held back, what would be counted.
    Check,
}

/// The rules of [`Mode`] and the two counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Rules {
    pub(super) mode: Mode,
    /// Compares with no byte written in a row before a window is quiet.
    pub(super) after: u32,
    /// One declaration in this many of a quiet window is let through.
    pub(super) audit: u32,
}

/// The rules the environment sets.
pub(super) fn rules() -> Rules {
    static RULES: std::sync::OnceLock<Rules> = std::sync::OnceLock::new();
    *RULES.get_or_init(|| {
        let mode = match std::env::var("VICTORY_QUIET_WINDOWS").as_deref().map(str::trim) {
            Ok("1") => Mode::On,
            Ok("check") => Mode::Check,
            _ => Mode::Off,
        };
        let count = |name: &str, default: u32| {
            std::env::var(name).ok().and_then(|v| v.trim().parse::<u32>().ok()).filter(|&n| n > 0).unwrap_or(default)
        };
        let rules = Rules { mode, after: count("VICTORY_QUIET_AFTER", 64), audit: count("VICTORY_QUIET_AUDIT", 32) };
        if mode != Mode::Off {
            tracing::info!(
                ?mode,
                after = rules.after,
                audit = rules.audit,
                "windows of title memory that shaders are declared to write and never do: held back once quiet \
                 (VICTORY_QUIET_WINDOWS=1; check: counted only)"
            );
        }
        rules
    })
}

/// Whether a declaration of shader writes is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Verdict {
    /// Queued as usual.
    Declare,
    /// Not made: the window is quiet and the range is inside what was
    /// compared.
    Hold,
}

/// A window's record, in its `Shadow`.
pub(super) struct Quiet {
    quiet: AtomicBool,
    /// Compares in a row with no byte written (while learning).
    streak: AtomicU32,
    /// The lowest and highest address (end) those compares covered.
    low: AtomicUsize,
    high: AtomicUsize,
    /// Declarations of the quiet window since the last one let through.
    held: AtomicU32,
}

impl Quiet {
    pub(super) fn new() -> Quiet {
        Quiet {
            quiet: AtomicBool::new(false),
            streak: AtomicU32::new(0),
            low: AtomicUsize::new(usize::MAX),
            high: AtomicUsize::new(0),
            held: AtomicU32::new(0),
        }
    }

    pub(super) fn is_quiet(&self) -> bool {
        self.quiet.load(Ordering::Relaxed)
    }

    /// Back to learning, with nothing known of the window.
    fn forget(&self) {
        self.quiet.store(false, Ordering::Relaxed);
        self.streak.store(0, Ordering::Relaxed);
        self.low.store(usize::MAX, Ordering::Relaxed);
        self.high.store(0, Ordering::Relaxed);
        self.held.store(0, Ordering::Relaxed);
    }

    /// A shader may write `[from, to)` of the window (title addresses): made
    /// or held back. Where nearly every declaration goes: one load of a flag
    /// for a window that is not quiet.
    pub(super) fn declared(&self, rules: &Rules, from: usize, to: usize) -> Verdict {
        if rules.mode == Mode::Off || !self.quiet.load(Ordering::Relaxed) {
            return Verdict::Declare;
        }
        if from < self.low.load(Ordering::Relaxed) || to > self.high.load(Ordering::Relaxed) {
            // Bytes nobody has looked at: looked at again.
            self.forget();
            count(RANGE, 1, 0);
            return Verdict::Declare;
        }
        if self.held.fetch_add(1, Ordering::Relaxed) + 1 >= rules.audit {
            self.held.store(0, Ordering::Relaxed);
            count(AUDIT, 1, 0);
            return Verdict::Declare;
        }
        let bytes = to.saturating_sub(from) as u64;
        match rules.mode {
            Mode::On => {
                count(HELD, 1, bytes);
                Verdict::Hold
            }
            // Counted as what would have been held back, and made.
            _ => {
                count(WOULD, 1, bytes);
                Verdict::Declare
            }
        }
    }

    /// `[from, to)` was compared after a submission and `written` bytes of it
    /// went back to title memory.
    pub(super) fn compared(&self, rules: &Rules, from: usize, to: usize, written: usize) {
        if rules.mode == Mode::Off {
            return;
        }
        if written > 0 {
            if self.quiet.load(Ordering::Relaxed) {
                // Found writing after it turned quiet: with `check` every
                // declaration is compared, so this counts all such writes;
                // with `1` only the audits' (the rest are not compared).
                count(WROTE, 1, written as u64);
            }
            self.forget();
            return;
        }
        if self.quiet.load(Ordering::Relaxed) {
            return;
        }
        self.low.fetch_min(from, Ordering::Relaxed);
        self.high.fetch_max(to, Ordering::Relaxed);
        if self.streak.fetch_add(1, Ordering::Relaxed) + 1 >= rules.after {
            self.quiet.store(true, Ordering::Relaxed);
            self.held.store(0, Ordering::Relaxed);
            count(TURNED, 1, 0);
        }
    }
}

/// Since the last frame-rate line, by [`HELD`] to [`WROTE`]: (count, bytes).
static STATS: [[AtomicU64; 2]; 6] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; 6];
/// Windows that turned quiet.
const TURNED: usize = 0;
/// Declarations held back.
const HELD: usize = 1;
/// Declarations that check mode would have held back.
const WOULD: usize = 2;
/// Declarations let through to look again (audits).
const AUDIT: usize = 3;
/// Quiet windows brought back by a declaration outside what was compared.
const RANGE: usize = 4;
/// Compares of quiet windows that found bytes written: bytes.
const WROTE: usize = 5;

/// Since the start, kept for the line: windows that turned quiet and windows
/// found writing after they had.
static TOTAL: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

fn count(which: usize, n: u64, bytes: u64) {
    STATS[which][0].fetch_add(n, Ordering::Relaxed);
    STATS[which][1].fetch_add(bytes, Ordering::Relaxed);
    match which {
        TURNED => TOTAL[0].fetch_add(n, Ordering::Relaxed),
        WROTE => TOTAL[1].fetch_add(n, Ordering::Relaxed),
        _ => 0,
    };
}

/// The frame-rate line's part for the switch: what was held back (or would
/// have been) per frame, the audits, and what turned back. `quiet` is the
/// windows quiet now and their bytes.
pub(super) fn line(frames: f64, quiet: [u64; 2]) {
    let rules = rules();
    if rules.mode == Mode::Off {
        return;
    }
    let s: Vec<[u64; 2]> = STATS.iter().map(|c| [c[0].swap(0, Ordering::Relaxed), c[1].swap(0, Ordering::Relaxed)]).collect();
    let mb = |v: u64| v as f64 / (1 << 20) as f64 / frames;
    let per = |v: u64| v as f64 / frames;
    let (held, label) = if rules.mode == Mode::On { (s[HELD], "held back") } else { (s[WOULD], "would be held back (check: all made)") };
    tracing::info!(
        "quiet windows (VICTORY_QUIET_WINDOWS={}, after {} compares with nothing written, one declaration in {} let through): \
         {} windows quiet now ({:.0} MB); per frame {:.0} declarations {} ({:.1} MB not snapshotted or compared), {:.1} audits; \
         since the last line: {} windows turned quiet, {} turned back by a range not compared before, {} found writing after turning quiet ({} bytes); \
         since the start {} turned quiet, {} found writing after",
        if rules.mode == Mode::On { "1" } else { "check" },
        rules.after,
        rules.audit,
        quiet[0],
        quiet[1] as f64 / (1 << 20) as f64,
        per(held[0]),
        label,
        mb(held[1]),
        per(s[AUDIT][0]),
        s[TURNED][0],
        s[RANGE][0],
        s[WROTE][0],
        s[WROTE][1],
        TOTAL[0].load(Ordering::Relaxed),
        TOTAL[1].load(Ordering::Relaxed)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules_of(mode: Mode) -> Rules {
        Rules { mode, after: 4, audit: 3 }
    }

    /// Four compares with nothing written make a window quiet; until then
    /// every declaration is made.
    #[test]
    fn a_window_turns_quiet_after_so_many_compares_with_nothing_written() {
        let on = rules_of(Mode::On);
        let q = Quiet::new();
        for _ in 0..3 {
            assert_eq!(q.declared(&on, 0x1000, 0x2000), Verdict::Declare);
            q.compared(&on, 0x1000, 0x2000, 0);
            assert!(!q.is_quiet());
        }
        q.compared(&on, 0x1000, 0x3000, 0);
        assert!(q.is_quiet(), "the fourth compare in a row");
        // The extent is the union of what the compares covered.
        assert_eq!((q.low.load(Ordering::Relaxed), q.high.load(Ordering::Relaxed)), (0x1000, 0x3000));
    }

    /// A byte written in a compare starts the count again.
    #[test]
    fn a_write_in_a_compare_starts_the_count_again() {
        let on = rules_of(Mode::On);
        let q = Quiet::new();
        for _ in 0..3 {
            q.compared(&on, 0x1000, 0x2000, 0);
        }
        q.compared(&on, 0x1000, 0x2000, 16);
        q.compared(&on, 0x1000, 0x2000, 0);
        assert!(!q.is_quiet(), "one compare since the write");
        for _ in 0..3 {
            q.compared(&on, 0x1000, 0x2000, 0);
        }
        assert!(q.is_quiet());
    }

    /// A quiet window holds back declarations inside the extent, lets one
    /// in three through (the audit), and a declaration outside the extent
    /// brings it back to learning and is made.
    #[test]
    fn a_quiet_window_holds_back_what_it_looked_at_and_audits_now_and_then() {
        let on = rules_of(Mode::On);
        let q = Quiet::new();
        for _ in 0..4 {
            q.compared(&on, 0x1000, 0x3000, 0);
        }
        assert!(q.is_quiet());
        assert_eq!(q.declared(&on, 0x1800, 0x3000), Verdict::Hold);
        assert_eq!(q.declared(&on, 0x1000, 0x2000), Verdict::Hold);
        assert_eq!(q.declared(&on, 0x1000, 0x2000), Verdict::Declare, "the third is the audit");
        // The audit's compare finds nothing: still quiet, and the count of
        // held declarations starts again.
        q.compared(&on, 0x1000, 0x2000, 0);
        assert!(q.is_quiet());
        assert_eq!(q.declared(&on, 0x1000, 0x2000), Verdict::Hold);
        // Bytes nobody looked at.
        assert_eq!(q.declared(&on, 0x0800, 0x2000), Verdict::Declare);
        assert!(!q.is_quiet());
        assert_eq!((q.low.load(Ordering::Relaxed), q.high.load(Ordering::Relaxed)), (usize::MAX, 0));
        let again = Quiet::new();
        for _ in 0..4 {
            again.compared(&on, 0x1000, 0x3000, 0);
        }
        assert_eq!(again.declared(&on, 0x1000, 0x3001), Verdict::Declare, "one byte past the end");
        assert!(!again.is_quiet());
    }

    /// An audit that finds a byte written ends the quiet.
    #[test]
    fn an_audit_that_finds_a_write_ends_the_quiet() {
        let on = rules_of(Mode::On);
        let q = Quiet::new();
        for _ in 0..4 {
            q.compared(&on, 0x1000, 0x2000, 0);
        }
        assert_eq!(q.declared(&on, 0x1000, 0x2000), Verdict::Hold);
        assert_eq!(q.declared(&on, 0x1000, 0x2000), Verdict::Hold);
        assert_eq!(q.declared(&on, 0x1000, 0x2000), Verdict::Declare);
        q.compared(&on, 0x1000, 0x2000, 8);
        assert!(!q.is_quiet());
        assert_eq!(q.declared(&on, 0x1000, 0x2000), Verdict::Declare);
    }

    /// With `check` the rules run and nothing is held back; with the switch
    /// off nothing is tracked at all.
    #[test]
    fn check_holds_nothing_back_and_off_tracks_nothing() {
        let check = rules_of(Mode::Check);
        let q = Quiet::new();
        for _ in 0..4 {
            q.compared(&check, 0x1000, 0x2000, 0);
        }
        assert!(q.is_quiet(), "learned as with the switch on");
        assert_eq!(q.declared(&check, 0x1000, 0x2000), Verdict::Declare);
        assert_eq!(q.declared(&check, 0x1000, 0x2000), Verdict::Declare);
        let off = rules_of(Mode::Off);
        let q = Quiet::new();
        for _ in 0..10 {
            q.compared(&off, 0x1000, 0x2000, 0);
        }
        assert!(!q.is_quiet());
        assert_eq!(q.declared(&off, 0x1000, 0x2000), Verdict::Declare);
    }
}
