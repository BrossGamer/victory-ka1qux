//! Which lanes of a wave set a bit of the write bitmap (module notes of
//! `emit`, "Write marks"), a child module of `lower` so it shares the
//! emitter's state.
//!
//! The bitmap lives in system memory, because the host reads and clears it
//! after each submission, so each load of one of its dwords crosses the bus,
//! once per lane that does it. Most lanes of a wave write the same 64 KiB
//! page, or pages of one bitmap dword, so lanes that share what another lane
//! sets leave it to that lane.

use std::sync::OnceLock;

use super::{Attr, Bin, Lower, V, flags, pred};

/// How the lanes of a wave set the bit of a page they wrote:
/// `VICTORY_WAVE_MARKS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum WaveMarks {
    /// `0`: every lane loads the bitmap dword and, where its bit is clear,
    /// ORs it in.
    PerLane,
    /// `all-equal`: the lanes do as the first one if all of them set bits
    /// of one dword (one load, one OR of all their bits), else per lane.
    /// (After upstream VIctory's pull request #2755.)
    AllEqual,
    /// The default: the lanes of the first lane's dword leave the load and
    /// the OR of their bits to the first lane; lanes of other dwords set
    /// their own. (After upstream VIctory's pull request #2958 by
    /// moe-miau, which leaves to the first lane only the lanes with its own
    /// bit.)
    Leader,
}

/// The mode of this process; [`WaveMarks::Leader`] unless
/// `VICTORY_WAVE_MARKS` is `0` or `all-equal`. Part of the translation: the
/// shader cache keys on it (`victory-gpu`'s does, with every switch the
/// translator reads).
pub(in super::super) fn wave_marks() -> WaveMarks {
    #[cfg(test)]
    if let Some(mode) = TEST_WAVE_MARKS.get() {
        return mode;
    }
    static MODE: OnceLock<WaveMarks> = OnceLock::new();
    *MODE.get_or_init(|| parse(std::env::var("VICTORY_WAVE_MARKS").ok().as_deref()))
}

/// `VICTORY_WAVE_MARKS`'s value as a mode, for testing.
fn parse(value: Option<&str>) -> WaveMarks {
    match value.map(str::trim) {
        Some("0") => WaveMarks::PerLane,
        Some("all-equal") => WaveMarks::AllEqual,
        _ => WaveMarks::Leader,
    }
}

#[cfg(test)]
thread_local! {
    /// A test's choice for [`wave_marks`], for its thread only.
    pub(in super::super) static TEST_WAVE_MARKS: std::cell::Cell<Option<WaveMarks>> = const { std::cell::Cell::new(None) };
}

impl Lower<'_> {
    /// Sets bit `n` of the write bitmap for every lane that runs it (inside
    /// [`Self::mark_store`]'s guards: the lanes that wrote), the way
    /// [`wave_marks`] says. Where lanes share what they set, one of them
    /// ORs the bits of all of them: the dword is loaded first and the
    /// atomic skipped where all the bits are already set (bits only
    /// accumulate until the host reads them after the submission, so a
    /// stale load only costs an extra OR). Any set of lanes can share a
    /// dword, whatever the wave's size, and a console wave split over two
    /// host waves is two waves here. Helper lanes of a pixel shader set
    /// nothing (their stores are dropped, and whether they take part in
    /// wave operations depends on the host).
    pub(super) fn mark_page(&mut self, bitmap: V, n: V) {
        let c = self.cur;
        let [k5, k2, k31, one] = [5, 2, 31, 1].map(|v| self.s.m.i32(v));
        let word = self.s.m.bin(c, Bin::LShr, n, k5);
        let at = self.s.m.bin(c, Bin::Shl, word, k2);
        let shift = self.s.m.bin(c, Bin::And, n, k31);
        let bit = self.s.m.bin(c, Bin::Shl, one, shift);
        let (mask, writer) = match wave_marks() {
            WaveMarks::PerLane => (bit, None),
            WaveMarks::AllEqual => self.mark_all_equal(word, bit),
            WaveMarks::Leader => self.mark_leader(word, bit),
        };
        let elected = self.begin_guarded(writer);
        let c = self.cur;
        let old = self.raw_word(bitmap, at);
        let set = self.s.m.bin(c, Bin::And, old, mask);
        let missing = self.s.m.cmp(c, pred::NE, set, mask);
        let update = self.begin_guarded(Some(missing));
        let i32t = self.s.t.i32;
        let iu = self.s.m.undef(i32t);
        // Operation 2 is OR.
        self.s.op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &[bitmap, k2, at, iu, iu, mask]);
        self.end_guarded(update, &[]);
        self.end_guarded(elected, &[]);
    }

    /// The bits this lane ORs into the bitmap dword at `word`, and whether
    /// it does ([`WaveMarks::AllEqual`]): all of the active lanes' bits, by
    /// the first lane, when they all set bits of one dword; its own bit,
    /// by itself, when not.
    fn mark_all_equal(&mut self, word: V, bit: V) -> (V, Option<V>) {
        let c = self.cur;
        self.s.flags |= flags::WAVE_OPS;
        let (i1, i32t) = (self.s.t.i1, self.s.t.i32);
        let same = self.s.op(c, "waveActiveAllEqual.i32", Attr::NoUnwind, i1, 115, &[word]);
        let or = self.s.m.i8(1);
        let all = self.s.op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[bit, or]);
        let first = self.s.op(c, "waveIsFirstLane", Attr::NoUnwind, i1, 110, &[]);
        let mask = self.s.m.select(c, same, all, bit);
        let yes = self.s.m.bool(true);
        let mut writer = self.s.m.select(c, same, first, yes);
        if self.s.pixel {
            let helper = self.s.op(c, "isHelperLane.i1", Attr::ReadOnly, i1, 221, &[]);
            let no = self.s.m.bool(false);
            writer = self.s.m.select(c, helper, no, writer);
        }
        (mask, Some(writer))
    }

    /// The bits this lane ORs into the bitmap dword at `word`, and whether
    /// it does ([`WaveMarks::Leader`]). The leading lane is the first
    /// active one and, in a pixel shader, the first that is no helper (a
    /// helper's writes are dropped). Its dword's lanes leave it the OR of
    /// all their bits; a lane of another dword sets its own bit. So a wave
    /// of lanes that write nearby (one page, pages of one dword) costs one
    /// load and one OR, and one with a few lanes elsewhere costs one more
    /// per such lane, where [`WaveMarks::AllEqual`] falls back to one per
    /// lane as soon as one lane differs. (After upstream VIctory's pull
    /// request #2958 by moe-miau: the lanes of the same bit as the leading
    /// lane's leave the load and the atomic to it.)
    fn mark_leader(&mut self, word: V, bit: V) -> (V, Option<V>) {
        let c = self.cur;
        self.s.flags |= flags::WAVE_OPS;
        let (i1, i32t) = (self.s.t.i1, self.s.t.i32);
        let helper = self.s.pixel.then(|| self.s.op(c, "isHelperLane.i1", Attr::ReadOnly, i1, 221, &[]));
        let (lead_word, is_lead) = if let Some(helper) = helper {
            let lane = self.s.op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
            let none = self.s.m.i32(u32::MAX);
            let key = self.s.m.select(c, helper, none, lane);
            // Operation 2, unsigned: the minimum.
            let (min, unsigned) = (self.s.m.i8(2), self.s.m.i8(1));
            let lead = self.s.op(c, "waveActiveOp.i32", Attr::NoUnwind, i32t, 119, &[key, min, unsigned]);
            let at = self.s.op(c, "waveReadLaneAt.i32", Attr::NoUnwind, i32t, 117, &[word, lead]);
            (at, self.s.m.cmp(c, pred::EQ, lane, lead))
        } else {
            let first = self.s.op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[word]);
            (first, self.s.op(c, "waveIsFirstLane", Attr::NoUnwind, i1, 110, &[]))
        };
        let with_lead = self.s.m.cmp(c, pred::EQ, word, lead_word);
        let zero = self.s.m.i32(0);
        let mut give = self.s.m.select(c, with_lead, bit, zero);
        if let Some(helper) = helper {
            give = self.s.m.select(c, helper, zero, give);
        }
        let or = self.s.m.i8(1);
        let all = self.s.op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[give, or]);
        let mask = self.s.m.select(c, with_lead, all, bit);
        let yes = self.s.m.bool(true);
        let mut writer = self.s.m.select(c, with_lead, is_lead, yes);
        if let Some(helper) = helper {
            let no = self.s.m.bool(false);
            writer = self.s.m.select(c, helper, no, writer);
        }
        (mask, Some(writer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_picks_the_mode() {
        assert_eq!(parse(None), WaveMarks::Leader);
        assert_eq!(parse(Some("1")), WaveMarks::Leader);
        assert_eq!(parse(Some("")), WaveMarks::Leader);
        assert_eq!(parse(Some("0")), WaveMarks::PerLane);
        assert_eq!(parse(Some(" 0 ")), WaveMarks::PerLane);
        assert_eq!(parse(Some("all-equal")), WaveMarks::AllEqual);
    }
}
