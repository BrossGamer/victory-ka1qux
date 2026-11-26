//! Values that are the same in every lane of a console wave, for the first-lane reads of a
//! console wave64 program run as two host waves of 32.
//!
//! The console compiler keeps wave-uniform data in VGPRs and fetches it with
//! `v_readfirstlane_b32`. The first lane of a console wave may be in the other host wave, so
//! such a read has the two halves exchange (group barriers), which no thread may skip: it
//! cannot stand where not every lane of the group runs. When every lane of the console wave
//! holds the same value, either host wave reads its own first lane and nothing is exchanged
//! (`Lower`'s `v_readfirstlane_b32`). This is the proof that they do.
//!
//! The ray-tracing structure builder's prefix scans read the first lane of a subtotal in a loop
//! whose exit differs from one console wave to the next; each console wave's lanes still agree
//! on the subtotal and on whether they leave. That is what [`lane_uniform`] follows now: a
//! condition the same in every lane of a console wave does not split the wave, so what it
//! guards stays the same in the wave. `VICTORY_WAVE_ALIKE=0` goes back to the proof that only
//! counted what every lane of the program runs ([`plain`]).
//!
//! After avatarmanz/VIctory-7900XTX#142 (gomeaw, backporting the fix of justrealyme and ryotdev,
//! fxceless/victory-fork#497, with the review of adriaespii in #36).

use std::collections::HashSet;

use super::lower::lane_regions;
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Program, Reg, Value};
use crate::native::{NativeProgram, StageIo};
use crate::structure::{Atom, Bool, LaneExit, Structured};

/// VGPR values that are the same in every lane of a console wave holding them (`all` is
/// `all_lane_blocks`): written lane by lane from scalars, constants and such values, by an
/// instruction that either every arriving lane runs or that runs under a condition itself the
/// same in every lane of the console wave (then only over an old value that is such a value
/// too: a wave the condition leaves out keeps that); loaded from LDS at such an address; or
/// the thread id's x shifted right past the console wave (`wave_tid`, ids `>> 6` or more,
/// where a console wave is one run of 64 ids along x). The compiler keeps wave-uniform data
/// in VGPRs and fetches it with `v_readfirstlane`.
/// A value joined by a phi counts where every predecessor is a block every lane of the console
/// wave reaches together: no branch on a lane-dependent condition can have split them (a
/// branch on such alike values does not). Loop-carried values count only when what the loop
/// carries is made of such values; the analysis runs to a fixed point from nothing, so a value
/// that needs itself to be alike never is.
pub(super) fn lane_uniform(
    p: &Program,
    st: &Structured,
    all: &[bool],
    wave_tid: Option<Value>,
) -> HashSet<Value> {
    if std::env::var_os("VICTORY_WAVE_ALIKE").is_some_and(|v| v == "0") {
        return plain(p, st, all);
    }
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let order = crate::ir::reverse_postorder(&p.blocks);
    let regions = lane_regions(st);
    let mut same: HashSet<Value> = HashSet::new();
    loop {
        // An atom of a lane condition that every lane of a console wave holds alike.
        let alike_atom = |same: &HashSet<Value>, a: &Atom| match *a {
            Atom::Uniform(_) => true,
            Atom::Lane(v) => same.contains(&v),
            _ => false,
        };
        let alike_bool = |same: &HashSet<Value>, b: Bool| {
            st.bools.support(b).iter().all(|a| alike_atom(same, a))
        };
        // Blocks every lane of a console wave reaches together: `all`, or reached only past
        // branches on alike conditions.
        let wave_all: Vec<bool> = match &regions {
            Some(regions) => {
                let mut w = vec![true; st.exits.len()];
                for &(d, ref region) in regions {
                    let LaneExit::Branch { when, .. } = st.exits[d] else {
                        continue;
                    };
                    if alike_bool(&same, when) {
                        continue;
                    }
                    for (b, r) in region.iter().enumerate() {
                        if *r {
                            w[b] = false;
                        }
                    }
                }
                w
            }
            None => all.to_vec(),
        };
        let mut next: HashSet<Value> = HashSet::new();
        for &b in &order {
            let block = &p.blocks[b];
            let whole = block
                .preds
                .iter()
                .all(|&q| wave_all.get(q).copied().unwrap_or(false));
            for phi in &block.phis {
                if matches!(phi.reg, Reg::V(_))
                    && whole
                    && phi
                        .args
                        .iter()
                        .all(|a| next.contains(a) || same.contains(a))
                {
                    next.insert(phi.value);
                }
            }
            let known = |next: &HashSet<Value>, x: &Value| next.contains(x) || same.contains(x);
            for (i, inst) in block.insts.iter().enumerate() {
                let name = inst.name();
                // under a condition: only one every lane of the console wave holds alike, and
                // over an old value that is alike too
                if let Some(g) = st.guards[b][i] {
                    if !alike_bool(&same, g) {
                        continue;
                    }
                    let old_alike = inst
                        .writes
                        .iter()
                        .filter(|w| matches!(w.0, Reg::V(_)))
                        .all(|w| inst.read(w.0).is_none_or(|old| known(&next, &old)));
                    if !old_alike {
                        continue;
                    }
                }
                // an LDS read at an alike address reads one value in every lane
                if let Fields::Ds {
                    addr: Some(Operand::Vector { reg, .. }),
                    dst: Some(_),
                    data,
                    gds: false,
                    ..
                } = &inst.fields
                    && name.starts_with("ds_read")
                    && !name.contains("_d16")
                    && data.is_empty()
                {
                    if inst.read(Reg::V(*reg)).is_some_and(|a| known(&next, &a)) {
                        next.extend(
                            inst.writes
                                .iter()
                                .filter(|w| matches!(w.0, Reg::V(_)))
                                .map(|w| w.1),
                        );
                    }
                    continue;
                }
                let ops = match &inst.fields {
                    Fields::Valu(v)
                        if v.dpp.is_none()
                            && v.sdwa.is_none()
                            && !name.starts_with("v_mbcnt")
                            && !name.starts_with("v_cmpx")
                            && !name.starts_with("v_movrel")
                            && !name.starts_with("v_swaprel")
                            && !name.starts_with("v_div_fmas")
                            && !name.contains("lane") =>
                    {
                        v.src.as_slice()
                    }
                    _ => continue,
                };
                // the thread id's x shifted past the console wave: one value per console wave
                // (the b32 instruction uses only the low five bits of the shift count)
                if let (
                    Some(tid),
                    "v_lshrrev_b32",
                    [
                        Operand::Const(k @ 128..=192),
                        Operand::Vector { reg, .. },
                        ..,
                    ],
                ) = (wave_tid, name, ops)
                    && ((k - 128) & 31) >= 6
                    && inst.read(Reg::V(*reg)) == Some(tid)
                {
                    next.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
                    continue;
                }
                // Implicit VGPR inputs include accumulators and the other side of a swap.
                let implicit_alike = !(name == "v_swap_b32" || name.contains("mac_"))
                    || inst
                        .reads
                        .iter()
                        .all(|(reg, value)| !matches!(reg, Reg::V(_)) || known(&next, value));
                let mask_alike = name != "v_cndmask_b32"
                    || matches!(ops.get(2), Some(Operand::Scalar { code, .. })
                        if inst.read(Reg::S(*code))
                            .and_then(|v| st.masks.get(&v))
                            .is_some_and(|&b| alike_bool(&same, b)));
                let operands_alike = ops.iter().all(|op| match *op {
                    Operand::Vector { reg, dwords } => (0..dwords).all(|i| {
                        reg.checked_add(i)
                            .and_then(|r| inst.read(Reg::V(r)))
                            .is_some_and(|v| known(&next, &v))
                    }),
                    Operand::Scalar { code, dwords } if code != NULL => (0..dwords).all(|i| {
                        code.checked_add(i)
                            .and_then(|r| inst.read(Reg::S(r)))
                            .is_some_and(|v| !is_mask(v) || known(&next, &v))
                    }),
                    // These sources depend on execution or an implicit register.
                    Operand::Const(251..=254) | Operand::Attr { .. } => false,
                    _ => true,
                });
                if implicit_alike && mask_alike && operands_alike {
                    next.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
                }
            }
        }
        if next.is_subset(&same) {
            return same;
        }
        same.extend(next);
    }
}

/// The thread id's x (`v0` on entry) of a console wave64 compute program whose console waves are
/// runs of 64 ids along x (a group 64-aligned along x, or one row): `id >> 6` and up is then one
/// value per console wave ([`lane_uniform`]).
pub(super) fn wave_tid(p: &Program, native: &NativeProgram) -> Option<Value> {
    let StageIo::Cs(cs) = &native.io else {
        return None;
    };
    let [tx, ty, tz] = cs.threads;
    if !native.wave64 || !(tx.is_multiple_of(64) || (ty, tz) == (1, 1)) {
        return None;
    }
    p.inputs
        .iter()
        .find(|(r, _)| *r == Reg::V(0))
        .map(|(_, v)| *v)
}

/// The proof as it was (`VICTORY_WAVE_ALIKE=0`): VGPR values written lane by lane from
/// scalars, constants and such values, by an instruction every arriving lane runs (no guard).
/// A value joined by a phi counts only where every predecessor is a block all lanes run (no
/// lane branch can have split the arriving lanes); loop-carried values never do.
fn plain(p: &Program, st: &Structured, all: &[bool]) -> HashSet<Value> {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let mut same: HashSet<Value> = HashSet::new();
    for b in crate::ir::reverse_postorder(&p.blocks) {
        let block = &p.blocks[b];
        let whole = block
            .preds
            .iter()
            .all(|&q| all.get(q).copied().unwrap_or(false));
        for phi in &block.phis {
            if matches!(phi.reg, Reg::V(_)) && whole && phi.args.iter().all(|a| same.contains(a)) {
                same.insert(phi.value);
            }
        }
        for (i, inst) in block.insts.iter().enumerate() {
            let name = inst.name();
            let ops: Vec<Operand> = match &inst.fields {
                Fields::Valu(v)
                    if v.dpp.is_none()
                        && v.sdwa.is_none()
                        && !name.starts_with("v_mbcnt")
                        && !name.starts_with("v_cmpx")
                        && !name.contains("lane") =>
                {
                    v.src.clone()
                }
                _ => continue,
            };
            if st.guards[b][i].is_some() {
                continue; // lanes outside EXEC keep another value
            }
            let alike = ops.iter().all(|op| match *op {
                Operand::Vector { reg, .. } => {
                    // every dword of the operand the instruction reads from that register on
                    let read: Vec<Value> = inst
                        .reads
                        .iter()
                        .filter(|r| matches!(r.0, Reg::V(x) if x >= reg))
                        .map(|r| r.1)
                        .collect();
                    !read.is_empty() && read.iter().all(|x| same.contains(x))
                }
                // a scalar is one value; a lane mask is not, unless a compare of such values made it
                Operand::Scalar { code, .. } if code != NULL => inst
                    .read(Reg::S(code))
                    .is_none_or(|x| !is_mask(x) || same.contains(&x)),
                _ => true,
            });
            if alike {
                same.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
            }
        }
    }
    same
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::{ir, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    #[test]
    fn shifted_thread_ids_are_uniform_only_when_every_lane_agrees() {
        // (the proof of before takes none of them: a test of the switch holds the same lock)
        let _one = crate::emit::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for shift in 0..=64u32 {
            // v_lshrrev_b32 v7, shift, v0; s_endpgm
            let words = [0x2C0E_0080 + shift, S_ENDPGM];
            let code: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
            let p = ir::build(&code, true, 8).unwrap();
            let st = structure::structure(&p).unwrap();
            let inst = &p.blocks[0].insts[0];
            let tid = inst.read(Reg::V(0)).unwrap();
            let shifted = inst.writes.iter().find(|w| w.0 == Reg::V(7)).unwrap().1;
            let uniform = lane_uniform(&p, &st, &[true], Some(tid));
            let expected = (0..64u32).all(|lane| lane.wrapping_shr(shift) == 0);
            assert_eq!(uniform.contains(&shifted), expected, "inline shift {shift}");
        }
    }

    #[test]
    fn a_swap_also_depends_on_its_old_destination() {
        let _one = crate::emit::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let words = [
            0x7E02_0280, // v_mov_b32 v1, 0
            0x7E00_CB01, // v_swap_b32 v0, v1
            S_ENDPGM,
        ];
        let code: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let p = ir::build(&code, true, 8).unwrap();
        let st = structure::structure(&p).unwrap();
        let swap = &p.blocks[0].insts[1];
        assert_eq!(swap.name(), "v_swap_b32");
        let old_v0 = swap.read(Reg::V(0)).unwrap();
        let new_v1 = swap.write(Reg::V(1)).unwrap();
        let uniform = lane_uniform(&p, &st, &[true], Some(old_v0));
        assert!(!uniform.contains(&new_v1));
    }
}
