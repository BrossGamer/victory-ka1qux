//! The console thread index of a thread of a group, for programs with
//! console lanes: where a host wave's threads stand among the group's.
//!
//! A console wave holds a wave-aligned run of the group's flat thread
//! ids; a host wave need not. NVIDIA (and RDNA in 32-lane waves) fill a 2-D
//! group from 16 x 2 tiles, and a group of 8 x 8 is filled in tiles too, so
//! the flat id of a host wave's first lane rounded down to the wave size is
//! not a base every wave has of its own: in a 32 x 2 group both host waves
//! round down to the same base, and half of the threads took the ids of the
//! other half (fork issue #1243: three 32 x 2 sky compute passes of the
//! title's pack).
//!
//! A program with console lanes in a group of several host waves takes its
//! threads from a group counter instead: one more LDS dword (`ticket`, or
//! `halves` for a console wave64 in host waves of 32), through which each
//! host wave takes the next block of its lanes ([`Lower::wave_base`]). After
//! fxceless/victory-fork PR #1261 (Garu667); `VICTORY_THREAD_TICKET=0` keeps
//! the old rounding for the programs that are not `halves`.

use super::{Attr, Bin, Lower, Rmw, V, flags, pred};

/// Whether programs with console lanes in groups of several host waves take
/// their console threads from a group counter (`ticket`, fork issue #1243).
/// `VICTORY_THREAD_TICKET=0`: the flat id of the wave's first lane rounded
/// down, as before.
pub(super) fn ticket_on() -> bool {
    std::env::var_os("VICTORY_THREAD_TICKET").is_none_or(|v| v != "0")
}

/// Whether the cross-lane operations that only read lanes of their own 32-lane half of a
/// console wave64 count the lanes by the console's (the rows of a DPP row write mask, the
/// half of a `ds_bpermute_b32`), and run in host waves of 32 without the halves meeting.
/// `VICTORY_HALF_LOCAL=0`: the host lane everywhere, a permute across the two host waves
/// refused, as before.
pub(super) fn half_local_on() -> bool {
    std::env::var_os("VICTORY_HALF_LOCAL").is_none_or(|v| v != "0")
}

/// The dwords of group-shared memory a compute shader may have (32 KiB).
const LDS_DWORDS_MOST: u32 = 8192;

/// Whether a program that already holds `lds_dwords` of group-shared memory
/// has room for the ticket's one more dword. One of the title's programs
/// holds 32,768 bytes, all there is: with the dword the host's validator
/// refuses it ("Total Thread Group Shared Memory storage is 32772"), so it
/// keeps the old rounding, as before.
pub(super) fn ticket_fits(lds_dwords: u32) -> bool {
    lds_dwords < LDS_DWORDS_MOST
}

impl<'a> Lower<'a> {
    /// With `halves` or `ticket`: the first console thread of this host wave (0, w, 2w, ... for
    /// host waves of w lanes). A GPU need not give a host wave consecutive thread indices
    /// (NVIDIA's 8 x 8 groups, the 16 x 2 tiles of 2-D groups, #1243), so each wave takes its
    /// threads from an LDS counter instead: every thread zeroes it, the group meets, and each
    /// wave's first active lane adds for the wave. A full wave takes the next block of w from the
    /// low 16 bits; a wave with fewer lanes (the group's last, partial one) takes its lanes'
    /// count from the high 16 bits, past the full waves' blocks, so that a partial wave arriving
    /// early does not shift the full waves off their alignment. Lanes are placed by their rank
    /// among the active lanes (`wave_rank`). Branch-free, in the entry block, which every
    /// thread runs.
    pub(super) fn wave_base(&mut self) -> Option<V> {
        let slot = self.halves.or(self.ticket)?;
        if let Some(v) = self.wave_base {
            return Some(v);
        }
        let (e, i32t) = (self.s.entry, self.s.t.i32);
        let w = self.s.wave_size.unwrap_or(32);
        let group: u32 = self.s.threads.map_or(w, |t| t.iter().product());
        let lds = self.s.lds();
        let k = self.s.m.i32(slot);
        let ptr = self.s.m.element(e, lds, k);
        let zero = self.s.m.i32(0);
        self.s.m.store(e, ptr, zero);
        self.group_sync(e);
        let one = self.s.m.i32(1);
        let (sum, unsigned) = (self.s.m.i8(0), self.s.m.i8(1));
        let count = self.s.op(e, "waveActiveOp.i32", Attr::NoUnwind, i32t, 119, &[one, sum, unsigned]);
        let rank = self.s.op(e, "wavePrefixOp.i32", Attr::NoUnwind, i32t, 121, &[one, sum, unsigned]);
        let kw = self.s.m.i32(w);
        let full = self.s.m.cmp(e, pred::EQ, count, kw);
        let first = self.s.m.cmp(e, pred::EQ, rank, zero);
        let sixteen = self.s.m.i32(16);
        let high = self.s.m.bin(e, Bin::Shl, count, sixteen);
        let take = self.s.m.select(e, full, kw, high);
        let add = self.s.m.select(e, first, take, zero);
        let ptr = self.s.m.element(e, lds, k);
        let old = self.s.m.atomic(e, Rmw::Add, ptr, add);
        let found = self.s.op(e, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[old]);
        let low_mask = self.s.m.i32(0xffff);
        let low = self.s.m.bin(e, Bin::And, found, low_mask);
        let past = self.s.m.bin(e, Bin::LShr, found, sixteen);
        let blocks = self.s.m.i32(group / w * w);
        let tail = self.s.m.bin(e, Bin::Add, blocks, past);
        let base = self.s.m.select(e, full, low, tail);
        self.s.flags |= flags::WAVE_OPS;
        self.wave_base = Some(base);
        self.wave_rank = Some(rank);
        Some(base)
    }

    /// This thread's console lane (0..63) in a console wave64: with `halves` (two host waves of
    /// 32 lanes) its console thread index in the wave, whose host wave holds the 32 lanes of one
    /// half; else the host lane, which is the console lane. The host lane of a half only tells
    /// the lane within the half, which is not enough where the half itself matters (the rows 0-3
    /// of a DPP row write mask, `VICTORY_HALF_LOCAL`).
    pub(super) fn console_lane_in_wave(&mut self) -> V {
        if self.halves.is_none() {
            return self.lane_index();
        }
        let e = self.s.entry;
        let f = self.console_thread();
        let k63 = self.s.m.i32(63);
        self.s.m.bin(e, Bin::And, f, k63)
    }

    /// The console thread index of this thread in the group: `wave_base` plus the lane's rank
    /// in its host wave, else the wave's first thread index rounded down plus the lane.
    pub(super) fn console_thread(&mut self) -> V {
        let (e, i32t) = (self.s.entry, self.s.t.i32);
        if let Some(base) = self.wave_base() {
            let rank = self.wave_rank.expect("made with the base");
            return self.s.m.bin(e, Bin::Add, base, rank);
        }
        let lane = self.s.op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let flat = self.s.op(e, "flattenedThreadIdInGroup.i32", Attr::ReadNone, i32t, 96, &[]);
        let first = self.s.op(e, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[flat]);
        let wave = self.s.wave_size.unwrap_or(32);
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        self.s.m.bin(e, Bin::Add, base, lane)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ticket_is_given_only_where_the_group_shared_memory_has_room_for_it() {
        assert!(ticket_fits(0));
        assert!(ticket_fits(8191));
        assert!(!ticket_fits(8192), "all 32 KiB held already");
        assert!(!ticket_fits(9000));
    }
}
