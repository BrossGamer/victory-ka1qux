//! A console wave64 compute program on a host of 32-lane waves: a wave reduction inside a
//! per-lane branch, with that branch taken wave-wide ([`crate::structure::structure_wide`])
//! so that the two host waves can exchange their halves inside it. Lowered, validated by
//! `dxil.dll` when present, and run on the host GPU when it takes waves of 32.

use std::collections::HashSet;

use super::host::{Dispatch, Host, Root, View};
use super::valu_tests::{DS_READ_B32, DS_WRITE_B32, ds};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// 64 threads in one console wave. Each thread t holds the bit `1 << (t >> 1)`; the threads
/// below s0 (a root constant) OR theirs over the wave inside `if (t < s0) { ... }` and store
/// the result to dword 0 of the raw UAV.
/// `skip`: the branch past the body when no thread is below s0: `s_cbranch_execz`, or
/// `s_cbranch_scc0` on the SCC `s_and_saveexec` sets ("some lane is left").
fn program(skip: u32) -> NativeProgram {
    let mut words = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        skip | 23,   // the branch to the restore of EXEC
    ];
    words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    native(&words)
}

/// A wave64 compute program of 64 threads (one console wave; the thread index in v0) with 4
/// root constants in s0..s3 and a raw UAV's descriptor in s[8:11], then `body`.
fn native(body: &[u32]) -> NativeProgram {
    native_of(body, true, 64)
}

/// The same for a console of 32-lane waves when `wave64` is not set, with `threads` threads.
fn native_of(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `program()` for a host of 32-lane waves, with the branches closing `wide` taken wave-wide.
fn lowered(native: &NativeProgram, p: &ir::Program, wide: &HashSet<usize>) -> Result<HostShader, ShaderError> {
    lowered_on(native, p, wide, 32)
}

/// [`lowered`] for a host whose waves have at most `host_wave` lanes.
fn lowered_on(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
    host_wave: u32,
) -> Result<HostShader, ShaderError> {
    let st = structure::structure_wide(p, wide).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let lowered = lower::lower(&lower::Parts {
        native,
        p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        written_root_uavs: lowered.written_root_uavs,
        writes_guest_memory: lowered.writes_guest_memory,
        root_indexed_views: res.root_indexed_views,
        half_views: lowered.half_views,
    })
}

#[test]
fn a_sum_of_halves_over_the_wave_is_the_consoles_tree_bit_for_bit() {
    // thread t holds the 16-bit float with bits 0x3C00 + 37 t (1.0 and up, with odd
    // mantissas, so that additions round); all of them store the wave's sum
    let mut words = vec![
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    // (the wave OR's words with v_add_f16 for v_or_b32: opcode 50 for 28)
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == "v_add_f16").count(), 6);
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // the console's sum: lanes 2k and 2k+1, then pairs, ... each addition rounded to 16 bits
    // (to nearest, ties to even); values here are normal numbers
    let wide = |h: u16| f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13);
    let narrow = |f: f32| {
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let terms: Vec<u16> = (0..64u16).map(|t| 0x3C00 + 37 * t).collect();
    let mut tree = terms.clone();
    while tree.len() > 1 {
        tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
    }
    let in_a_row = terms.iter().copied().reduce(add).unwrap();
    assert_ne!(tree[0], in_a_row, "these terms tell the tree from another order");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    // (the high half is lane 0's term's: zero)
    assert_eq!(out[0][0], u32::from(tree[0]), "tree {:#x}, in a row {in_a_row:#x}", tree[0]);
    eprintln!("ran on the hardware device: the sum {:#x} of the console's tree", tree[0]);
}

/// The console's 16-bit wave sum where the lanes at or past s0 have left the program (a
/// thread past the end of the data): on the console they add +0, and after each step of the
/// tree they hold their run's partial sum all the same. On a host of 64-lane waves (AMD) the
/// lanes that are here read each partner run's sum from a lane of it that is here.
#[test]
fn a_sum_of_halves_where_lanes_have_left_is_the_consoles_tree_bit_for_bit() {
    let mut words = vec![
        0x7DA8_0000, // v_cmpx_gt_u32 exec, s0, v0: the lanes below s0 stay
        0, // s_cbranch_execz to the end (filled in below)
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    // past every word after it, to the program's end
    words[1] = 0xBF88_0000 | (words.len() as u32 - 2);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered_on(&native, &p, &HashSet::new(), 64).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let wide = |h: u16| {
        if h == 0 {
            0.0
        } else {
            f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13)
        }
    };
    let narrow = |f: f32| {
        if f == 0.0 {
            return 0;
        }
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = false;
    for stay in [64u32, 63, 40, 33, 32, 17, 5, 1] {
        let terms: Vec<u16> = (0..64u32)
            .map(|t| if t < stay { (0x3C00 + 37 * t) as u16 } else { 0 })
            .collect();
        let mut tree = terms.clone();
        while tree.len() > 1 {
            tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
        }
        let Some(got) = on_the_card(&shader, [stay, 0]) else {
            return;
        };
        ran = true;
        // (the high half is lane 0's term's: zero)
        assert_eq!(got, u32::from(tree[0]), "{stay} lanes stay: got {got:#x}, the console's tree {:#x}", tree[0]);
    }
    if ran {
        eprintln!("ran on the hardware device: the console's tree with lanes gone, in every case");
    }
}

/// One `v_add_f16` whose exact result lies between two halves: 1 + 3/1024 (0x3C03) plus 2
/// (0x4000) is 3 + 1.5 steps of that range, which the console rounds up to 0x4202. The
/// host's `f32tof16` may give 0x4201 (toward zero); with `VICTORY_F16_NEAREST=1` the
/// translation gives the console's whatever the host does.
#[test]
fn a_16_bit_sum_rounds_to_nearest_when_asked_to() {
    let words = [
        0x7E0E_02FF,
        0x0000_3C03, // v_mov_b32 v7, 0x3c03
        0x7E0C_02FF,
        0x0000_4000, // v_mov_b32 v6, 0x4000
        0x6400_0D07, // v_add_f16 v0, v7, v6
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert!(p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "v_add_f16"));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for nearest in [false, true] {
        let was = lower::TEST_F16_NEAREST.replace(Some(nearest));
        let shader = lowered(&native, &p, &HashSet::new());
        lower::TEST_F16_NEAREST.set(was);
        let shader = shader.expect("lower");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 hardware device; not run");
            return;
        };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            eprintln!("the hardware device does not take this program; not run");
            return;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        let got = out[0][0] & 0xFFFF;
        eprintln!("v_add_f16 0x3c03 + 0x4000 on the hardware device, nearest {nearest}: {got:#x}");
        if nearest {
            assert_eq!(got, 0x4202);
        } else {
            assert!(got == 0x4201 || got == 0x4202, "{got:#x}");
        }
    }
}

#[test]
fn a_wave_or_inside_a_lane_branch_needs_the_branch_wave_wide() {
    for (skip, name) in [(0xBF88_0000, "s_cbranch_execz"), (0xBF84_0000, "s_cbranch_scc0")] {
        a_wave_or_inside(skip, name);
    }
}

fn a_wave_or_inside(skip: u32, name: &str) {
    let native = program(skip);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == name))
        .expect("the branch");
    // per lane, the two host waves cannot meet inside the branch
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a wave reduction where not every lane runs"),
        "{:?}",
        per_lane.err()
    );
    // ... and the lowering names the block, around which that branch is the one to widen
    let needs = lower::take_needs();
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &needs), Some(vec![branch]));
    // (taken, past the body, when every lane's bit is clear)
    assert_eq!(st.wave_any[branch], Some(false));
    // wave-wide, every lane runs the reduction (EXEC stays a guard on what it does there)
    let wide = HashSet::from([branch]);
    let st = structure::structure_wide(&p, &wide).expect("structure");
    let inside = match p.blocks[branch].exit {
        ir::Exit::Branch { not_taken, .. } => not_taken,
        _ => unreachable!(),
    };
    assert!(st.guards[inside].iter().any(Option::is_some));
    let shader = lowered(&native, &p, &wide).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // On a GPU that takes waves of 32 (two host waves for the console's one): the OR is of
    // both halves, whichever half's lanes take the branch.
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    // (threads below s0 take the branch; thread t gives bit t >> 1)
    for (below, expected) in [(64u32, u32::MAX), (40, 0x000F_FFFF), (33, 0x0001_FFFF), (10, 0x1F), (1, 1)] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if let (RootType::Constants, 0, 0) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = below;
            }
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        assert_eq!(out[0][0], expected, "{name}: {below} threads in the branch");
    }
    eprintln!("ran on the hardware device ({name}): the OR of both halves in every case");
}

/// Runs `shader` on the hardware device with the root constants `s0`, `s1` and gives dword 0
/// of the UAV; None (and a line) without a device that takes it.
fn on_the_card(shader: &HostShader, constants: [u32; 2]) -> Option<u32> {
    whole_uav(shader, constants, 1).map(|out| out[0])
}

/// As [`on_the_card`], and gives the first `dwords` dwords of the UAV.
fn whole_uav(shader: &HostShader, constants: [u32; 2], dwords: u32) -> Option<Vec<u32>> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let Ok((root, pso)) = host.compute_pipeline(shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return None;
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(dwords)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0].clone())
}

/// A 32-lane console wave. A register holds a lane mask on one path (the lanes below s0)
/// and a number on the other, and is read as a number after the join, as the trace programs
/// keep their lanes' masks and their node numbers on one stack
/// (a tiled shadow compute program of the title, s46 at 0x104, 0x75c and 0x188). The
/// mask arrives as the ballot of its lanes.
#[test]
fn a_lane_mask_joined_with_a_number_is_read_as_the_number_it_is() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE94_036A, // s_mov_b32 s20, vcc_lo
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0001, // s_cbranch_scc1 over the next
        0xBE94_0385, // s_mov_b32 s20, 5
        0x7E00_0214, // v_mov_b32 v0, s20
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (below, other, expected) in [(32u32, 0u32, u32::MAX), (10, 0, 0x3FF), (0, 0, 0), (10, 1, 5)] {
        let Some(got) = on_the_card(&shader, [below, other]) else { return };
        assert_eq!(got, expected, "{below} lanes in the mask, s1 = {other}");
    }
    eprintln!("ran on the hardware device: the mask's bits, or the number");
}

/// A 32-lane console wave. `if (t < s0) { ... SCC = (s1 == 7) }` and a branch on SCC after
/// the join: the skip brings the SCC of `s_and_saveexec` ("some lane is left") along
/// (a shadow-trace compute program of the title, 0x360-0x39c). On the console
/// the skip is taken by the wave, so with some lane inside every lane sees the inner test.
/// A lane that skipped on its own would see "some lane is left" instead: refused per lane,
/// right with the branch taken wave-wide. Lane 31 stores what it saw.
#[test]
fn scc_of_a_mask_operation_joined_with_a_scalar_test_needs_the_skip_wave_wide() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE8D_3C6A, // s_and_saveexec_b32 s13, vcc_lo
        0xBF88_0001, // s_cbranch_execz over the next
        0xBF06_8701, // s_cmp_eq_u32 s1, 7
        0xBEFE_030D, // s_mov_b32 exec_lo, s13
        0xBF85_0002, // s_cbranch_scc1 to "seven"
        0x7E0A_0281, // v_mov_b32 v5, 1
        0xBF82_0001, // s_branch over the next
        0x7E0A_0282, // seven: v_mov_b32 v5, 2
        0x7DA4_009F, // v_cmpx_eq_u32 exec_lo, 31, v0
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 5 << 8, // buffer_store_dword v5, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz"))
        .expect("the skip");
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a lane mask read as data"),
        "{:?}",
        per_lane.err()
    );
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &lower::take_needs()), Some(vec![branch]));
    let shader = lowered(&native, &p, &HashSet::from([branch])).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // (lanes below s0 inside, s1) -> what lane 31 stores: 2 if SCC was set at the branch
    for (below, s1, expected) in [(0u32, 7u32, 1u32), (10, 7, 2), (10, 3, 1), (32, 3, 1), (32, 7, 2)] {
        let Some(got) = on_the_card(&shader, [below, s1]) else { return };
        assert_eq!(got, expected, "{below} lanes inside, s1 = {s1}");
    }
    eprintln!("ran on the hardware device: the inner test's SCC for every lane once some lane is inside");
}

/// Issue #111: `v_writelane_b32` writes its target lane whenever any lane of
/// the console wave is in EXEC, even if the target lane itself is outside EXEC;
/// if no lane is in EXEC (the wave skipped), the write must not happen.
#[test]
fn a_lane_write_reaches_its_lane_outside_exec() {
    // One console wave of 64: v9 = 7, s6 = 5, if (t > s0) { v_writelane_b32 v9, s6, 0 },
    // then every thread stores v_readlane_b32(v9, 0). v9 also has an ordinary use (v_mov_b32)
    // so ir::unspill keeps v_writelane_b32 / v_readlane_b32.
    let words = [
        0x7E12_0287, // v_mov_b32 v9, 7
        0xBE86_0385, // s_mov_b32 s6, 5
        0x7D82_0000, // v_cmp_lt_u32 vcc, s0, v0  (t > s0)
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0002, // s_cbranch_execz +2 (over the 2-dword v_writelane_b32)
        0xD761_0009,
        0x0001_0006, // v_writelane_b32 v9, s6, 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0xD760_0007,
        0x0001_0109, // v_readlane_b32 s7, v9, 0
        0x7E02_0207, // v_mov_b32 v1, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 1 << 8, // buffer_store_dword v1, off, s[8:11], 0
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let io = io::raise(&p, &native).expect("io");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("res");
    let empty_bytes = crate::dxbc::build(&[]);
    let empty = crate::Container::parse(&empty_bytes).unwrap();

    // On a 32-lane host, per-lane lowering refuses the skipped write so emit()
    // switches to the without_skips form where all 64 lanes reach the write.
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a lane write where not every lane runs"),
        "{:?}",
        per_lane.err()
    );
    let shader = lower::TEST_HOST_WAVE.with(|cell| {
        let was = cell.replace(Some(32));
        let r = super::emit(&empty, &native, &p, &st, &io, &res);
        cell.set(was);
        r
    })
    .expect("emit on 32-lane host");
    // Also check native 64-lane lowering (without_skips form).
    let no_skips = ir::without_skips(&p, false).expect("without_skips");
    let st_ns = structure::structure(&no_skips).expect("structure");
    let io_ns = io::raise(&no_skips, &native).expect("io");
    let res_ns = resource::track(&no_skips, &native.layout, native.stage, Some(&st_ns)).expect("res");
    let shader64 = lower::TEST_HOST_WAVE.with(|cell| {
        let was = cell.replace(Some(64));
        let r = super::emit(&empty, &native, &no_skips, &st_ns, &io_ns, &res_ns);
        cell.set(was);
        r
    })
    .expect("emit on 64-lane host");

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("32-lane DXIL validation failed");
        v.validate(&mut shader64.dxil.clone()).expect("64-lane DXIL validation failed");
    }

    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (s0, expected) in [(10u32, 5u32), (62, 5), (63, 7)] {
        let Some(got) = on_the_card(&shader, [s0, 0]) else { return };
        assert_eq!(got, expected, "s0 = {s0}");
    }
    eprintln!("ran on the hardware device: v_writelane_b32 reaches lane 0 outside EXEC");
}

/// A spilled 64-thread wave64 compute program dispatched with `DispatchX`
/// `USE_THREAD_DIMENSIONS` (`USER_DISPATCH_THREADS`): reads `s_bcnt1_i32_b64(exec)`
/// on entry, computes a whole-wave OR of `1 << (lane >> 1)`, and stores
/// `(popcnt, wave_or)` unconditionally at `uav[lane * 2..+2]`.
///
/// Verified for both host wave sizes:
/// - `host_max_wave = 64`: entry early-exit masks hardware `EXEC` before `B0`
///   so `s_bcnt1_i32_b64(exec)` and the wave OR see only lanes `< limit`, and
///   lanes `>= limit` never store.
/// - `host_max_wave = 32`: predicated mode keeps both 32-lane host waves alive
///   for `exchange_halves` barriers (no hang even at `limit = 1` where the
///   upper half has zero active lanes) while masking ballots, reductions, and
///   stores identically.
#[test]
fn spilled_compute_masks_partial_dispatch_threads_on_wave64_and_wave32() {
    let mut words = vec![
        // Read descriptor-table offset (user_data 16) from spilled table s[0:1] + 0.
        0x8703_FF01,
        0x0000_00FF, // s_and_b32 s3, s1, 0xff
        0xBE82_0300, // s_mov_b32 s2, s0
        0x3D << 26 | 4 << 6 | 1,
        125 << 25, // s_load_dword s4, s[2:3], 0
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xBE86_107E, // s_bcnt1_i32_b64 s6, exec
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
    ];
    words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x3402_0083, // v_lshlrev_b32 v1, 3, v0 (lane * 8 bytes)
        0x7E04_0206, // v_mov_b32 v2, s6 (popcnt(exec))
        0x7E06_0207, // v_mov_b32 v3, s7 (wave OR)
        0x38 << 26 | 0x1D << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 2 << 8 | 1, // buffer_store_dwordx2 v[2:3], v1, s[8:11], 0 offen
        S_ENDPGM,
    ]);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 16,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 16,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1, // s0..s1 spilled table pointer
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = crate::spill::build(&native).expect("spill::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");

    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let host = Host::new(false).ok();
    for host_wave in [64u32, 32] {
        let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
        let lowered = lower::lower(&lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        });
        lower::TEST_HOST_WAVE.set(was);
        let lowered = lowered.unwrap_or_else(|e| panic!("lower (host_wave={host_wave}): {e}"));
        assert!(
            lowered
                .user_data
                .iter()
                .any(|u| u.root == RootType::Other(super::USER_DISPATCH_THREADS) && u.dword == 0),
            "host_wave={host_wave}: missing USER_DISPATCH_THREADS(0)"
        );
        let shader = HostShader {
            stage: HwStage::Cs,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            root_uavs: lowered.root_uavs,
            written_root_uavs: lowered.written_root_uavs,
            writes_guest_memory: lowered.writes_guest_memory,
            root_indexed_views: res.root_indexed_views.clone(),
            half_views: Vec::new(),
        };
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy)
                .unwrap_or_else(|e| panic!("DXIL validation failed (host_wave={host_wave}): {e}"));
        }
        let Some(host) = host.as_ref() else { continue };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            eprintln!("device skipped host_wave={host_wave} pipeline");
            continue;
        };
        for (limit, active, expected_or) in [
            (45u32, 45usize, 0x007F_FFFFu32),
            (1, 1, 1),
            (0, 64, u32::MAX),
        ] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                if u.root == RootType::Other(super::USER_DISPATCH_THREADS) && u.dword == 0 {
                    ud[u.index as usize] = limit;
                }
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(128)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("dispatch failed (host_wave={host_wave}, limit={limit}): {e}"));
            for lane in 0..64 {
                let (got_cnt, got_or) = (out[0][lane * 2], out[0][lane * 2 + 1]);
                if lane < active {
                    assert_eq!(
                        (got_cnt, got_or),
                        (active as u32, expected_or),
                        "host_wave={host_wave} limit={limit} lane={lane}"
                    );
                } else {
                    assert_eq!(
                        (got_cnt, got_or),
                        (0, 0),
                        "host_wave={host_wave} limit={limit}: inactive lane {lane} stored!"
                    );
                }
            }
        }
        eprintln!("ran on the hardware device (host_wave={host_wave}): partial dispatch limits 45, 1, 0 exact");
    }
}

/// A console wave32 program with console lanes (`v_mbcnt`) in a 32 x 2 group: thread (x, y)
/// stores 1 at dword 32y + x. Every dword must be written once: the host's relabelled thread
/// ids are a bijection of the group. A 32-lane host need not give a host wave a 32-aligned run
/// of flat ids (2-D groups are filled from 16 x 2 tiles on NVIDIA and RDNA), so the first flat
/// id of a wave rounded down gave two waves the same ids (fork issue #1243); the waves now take
/// their threads from a group counter. With `VICTORY_THREAD_TICKET=0` (the old rounding) the
/// count of dwords written is printed, not checked: it depends on the card.
#[test]
fn the_console_threads_of_a_two_dimensional_group_are_each_given_once() {
    let words = [
        0xD765_0003, // v_mbcnt_lo_u32_b32 v3, exec_lo, 0
        0x0001_007E,
        0x3408_0285, // v_lshlrev_b32 v4, 5, v1
        0x4A08_0104, // v_add_nc_u32 v4, v4, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 5 << 8 | 4, // buffer_store_dword v5, v4, s[8:11], 0 offen
    ];
    let mut native = native_of(&words, false, 64);
    native.io = StageIo::Cs(ComputeIo {
        rsrc2: 5 << 1 | 1 << 11, // 5 user SGPRs, thread ids x and y in v0, v1
        threads: [32, 2, 1],
    });
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let lower_with = |ticket: bool| {
        // SAFETY: tests that read this switch hold ONE_AT_A_TIME
        unsafe { std::env::set_var("VICTORY_THREAD_TICKET", if ticket { "1" } else { "0" }) };
        let shader = lowered_on(&native, &p, &HashSet::new(), 32);
        unsafe { std::env::remove_var("VICTORY_THREAD_TICKET") };
        shader.expect("lower")
    };
    let shader = lower_with(true);
    let old = lower_with(false);
    assert_ne!(shader.dxil, old.dxil, "the counter is used for a group of two host waves");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let run = |shader: &HostShader| -> Option<Vec<u32>> {
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("the hardware device does not take this program: {e}; not run");
                return None;
            }
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        Some(out[0][..64].to_vec())
    };
    let Some(got) = run(&shader) else { return };
    let missing: Vec<usize> = (0..64).filter(|&i| got[i] != 1).collect();
    assert!(missing.is_empty(), "dwords not written once: {missing:?}");
    if let Some(before) = run(&old) {
        let written = before.iter().filter(|&&d| d == 1).count();
        eprintln!("the old rounding on this device: {written} of 64 dwords written");
    }
    eprintln!("ran on the hardware device: 64 of 64 console threads, each once");
}

/// A wave64 program of 128 threads (two console waves) that leaves early when no lane of its
/// wave is below s0, as the console's compiler does when the end of the program is too far for
/// a branch: `s_cbranch_execnz` over a far jump to the end. `extra` are words that go into the
/// far jump's block before it, and `work_at_end` puts an instruction between the jump's target
/// and the end of the program. Then the gate: every thread stores 1, or 2 when some lane of its
/// console wave that is still running has the thread number s1 (the lanes that fail a test).
/// (After the tiled cascade shadow programs, whose gate is a wave-wide test of that kind.)
fn far_early_out(extra: &[u32], work_at_end: bool) -> NativeProgram {
    let gate = [
        0x7D84_0001, // v_cmp_eq_u32 vcc, s1, v0
        0xBE90_106A, // s_bcnt1_i32_b64 s16, vcc
        0xBF06_8010, // s_cmp_eq_u32 s16, 0
        0x8511_8281, // s_cselect_b32 s17, 1, 2
        0x7E02_0211, // v_mov_b32 v1, s17
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 1 << 8 | 2, // buffer_store_dword v1, v2, s[8:11], 0 offen
    ];
    let jump = [
        0xBE8E_1F00, // s_getpc_b64 s[14:15]
        0x800E_FF0E,
        4 * (4 + gate.len() as u32), // s_add_u32 s14, s14, the way to the end
        0x820F_800F, // s_addc_u32 s15, s15, 0
        0xBE80_200E, // s_setpc_b64 s[14:15]
    ];
    let mut words = vec![
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF89_0000 | (extra.len() + jump.len()) as u32, // s_cbranch_execnz over the jump
    ];
    words.extend_from_slice(extra);
    words.extend_from_slice(&jump);
    words.extend_from_slice(&gate);
    if work_at_end {
        words.push(0x7E06_0281); // v_mov_b32 v3, 1
    }
    native_of(&words, true, 128)
}

fn branches_named(p: &ir::Program, name: &str) -> usize {
    p.blocks.iter().filter(|b| b.insts.last().is_some_and(|i| i.name() == name)).count()
}

/// What the gate of [`far_early_out`] stores in each of 128 threads.
fn gate_expected(below: u32, failing: u32) -> Vec<u32> {
    (0..128u32)
        .map(|t| {
            let some_fail = failing < below && failing / 64 == t / 64;
            match (t < below, some_fail) {
                (false, _) => 0,
                (true, false) => 1,
                (true, true) => 2,
            }
        })
        .collect()
}

/// The far early-out is left out like the near one; the gate then counts the lanes of both
/// host waves of each console wave. Before, the program stopped where not every lane runs.
#[test]
fn a_far_early_out_is_left_out_and_the_gate_counts_the_lanes_of_both_halves() {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let native = far_early_out(&[], false);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(branches_named(&p, "s_cbranch_execnz"), 1);
    lower::take_needs();
    let kept = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&kept, Err(ShaderError::Unsupported { what, .. }) if what.ends_with("where not every lane runs")),
        "{:?}",
        kept.err()
    );
    for (name, form) in [("early-outs", ir::without_early_outs(&p)), ("skips", ir::without_skips(&p, false))] {
        let form = form.unwrap_or_else(|| panic!("without {name}: the far early-out stays"));
        assert_eq!(branches_named(&form, "s_cbranch_execnz"), 0, "without {name}");
        // the far jump, which nothing reaches any more, is gone with it
        assert!(!form.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "s_setpc_b64"), "{name}");
    }
    let form = ir::without_early_outs(&p).expect("left out");
    let shader = lowered(&native, &form, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // (threads below s0 are in bounds; the one thread s1 fails the test) A failing thread of the
    // upper half gives 2 to the lower half's threads as well, and a failing thread out of
    // bounds counts for nothing
    let cases = [(128u32, 40u32), (128, 100), (128, 63), (40, 50), (40, 10), (33, 32), (1, 0), (0, 5)];
    for (below, failing) in cases {
        let Some(got) = whole_uav(&shader, [below, failing], 128) else { return };
        assert_eq!(got, gate_expected(below, failing), "{below} threads in bounds, thread {failing} fails");
    }
    eprintln!("ran on the hardware device: the gate's count over both halves of each console wave");
    // What the program got before, when no exact form was found: each host wave on its own,
    // counting the lanes of its half alone (not asserted: it is the approximation that was).
    let local = lower::with_local_waves_any(|| lowered(&native, &p, &HashSet::new()));
    if let Ok(local) = local {
        let wrong: usize = cases
            .iter()
            .filter_map(|&(below, failing)| {
                let got = whole_uav(&local, [below, failing], 128)?;
                Some(got.iter().zip(gate_expected(below, failing)).filter(|(g, e)| **g != *e).count())
            })
            .sum();
        eprintln!("before, each host wave on its own: {wrong} of {} thread results wrong", cases.len() * 128);
    }
}

#[test]
fn a_far_jump_that_does_work_is_not_an_early_out() {
    // work in the jump's block, and work between the jump's target and the end
    for (name, native) in [
        ("work in the jump", far_early_out(&[0xBE94_0385], false)), // s_mov_b32 s20, 5
        ("work at the target", far_early_out(&[], true)),
    ] {
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        assert_eq!(branches_named(&p, "s_cbranch_execnz"), 1, "{name}");
        assert!(ir::without_early_outs(&p).is_none(), "{name}");
        assert!(ir::without_skips(&p, false).is_none(), "{name}");
        assert!(ir::without_skips(&p, true).is_none(), "{name}");
    }
}

#[test]
fn the_far_early_out_stays_with_the_switch_off() {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let native = far_early_out(&[], false);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    // SAFETY: tests that read this switch hold ONE_AT_A_TIME
    unsafe { std::env::set_var("VICTORY_CASCADE_EXACT", "0") };
    let (early, skips) = (ir::without_early_outs(&p), ir::without_skips(&p, true));
    unsafe { std::env::remove_var("VICTORY_CASCADE_EXACT") };
    assert!(early.is_none() && skips.is_none());
}

/// `if (t < s0) { park s[20:21]; if (s1 == 0) {..} else {..}; take it back }` and a read of
/// s[20:21] after the join, as the console's compiler leaves a parked scalar across an inner
/// branch. `then` is the inner branch's first arm; the second one leaves the parked scalar alone.
fn parked_across_an_inner_branch(then: u32) -> ir::Program {
    let body = [
        0xBE96_0414, // s_mov_b64 s[22:23], s[20:21]
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0002, // s_cbranch_scc1 to the second arm
        then,
        0xBF82_0001, // s_branch to the join
        0xBE98_0382, // s_mov_b32 s24, 2
        0xBE94_0416, // s_mov_b64 s[20:21], s[22:23]
    ];
    let mut words = vec![
        0xBE94_0406, // s_mov_b64 s[20:21], s[6:7]
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0000 | body.len() as u32, // s_cbranch_execz past the body
    ];
    words.extend_from_slice(&body);
    words.extend_from_slice(&[
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x7E02_0214, // v_mov_b32 v1, s20
    ]);
    let native = native(&words);
    ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build")
}

#[test]
fn a_skip_whose_scalars_pass_an_inner_branch_unchanged_is_left_out() {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let p = parked_across_an_inner_branch(0xBE96_0314); // s_mov_b32 s22, s20
    assert_eq!(branches_named(&p, "s_cbranch_execz"), 1);
    // (s22 comes back through a phi of the inner join whose two arguments are copies of one
    // value: only `deep` follows it)
    assert!(ir::without_skips(&p, false).is_none());
    let deep = ir::without_skips(&p, true).expect("left out");
    assert_eq!(branches_named(&deep, "s_cbranch_execz"), 0);
}

#[test]
fn a_skip_whose_scalar_an_inner_branch_changes_is_kept() {
    let p = parked_across_an_inner_branch(0xBE96_0387); // s_mov_b32 s22, 7
    assert_eq!(branches_named(&p, "s_cbranch_execz"), 1);
    assert!(ir::without_skips(&p, false).is_none());
    assert!(ir::without_skips(&p, true).is_none());
}

/// A wave64 program of 64 threads, all lanes running, for a host of 32-lane waves: `v1` =
/// `0x1000_0001` + lane, `v2` = `0xDEAD_0000` + lane, `v3` = (lane + 1) * 4, then `op`, then v2
/// stored at the lane's dword.
fn lane_program(op: &[u32]) -> NativeProgram {
    let mut words = vec![
        0x4A02_00FF, // v_add_nc_u32 v1, lit, v0
        0x1000_0001,
        0x4A04_00FF, // v_add_nc_u32 v2, lit, v0
        0xDEAD_0000,
        0x4A06_0081, // v_add_nc_u32 v3, 1, v0
        0x3406_0682, // v_lshlrev_b32 v3, 2, v3
    ];
    words.extend_from_slice(op);
    words.extend_from_slice(&[
        0x340A_0082, // v_lshlrev_b32 v5, 2, v0
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 2 << 8 | 5, // buffer_store_dword v2, v5, s[8:11], 0 offen
    ]);
    native_of(&words, true, 64)
}

/// `v_mov_b32_dpp v2, v1` with a row control, the row and bank write masks, no bound control.
fn dpp_row(ctrl: u16, row_mask: u8) -> Vec<u32> {
    vec![0x7E04_02FA, u32::from(row_mask) << 28 | 0xF << 24 | u32::from(ctrl) << 8 | 1]
}

/// `v_mov_b32_dpp8 v2, v1` taking lane 7 - n of every group of 8 in lane n.
fn dpp8_reverse() -> Vec<u32> {
    let sel: u32 = (0..8).map(|n| (7 - n) << (3 * n)).sum();
    vec![0x7E04_02E9, 1 | sel << 8]
}

/// What the program of [`lane_program`] has in v1 of `lane`.
fn lane_v1(lane: u32) -> u32 {
    0x1000_0001 + lane
}

/// The wave64 program `op` translated for a host of `host_wave`-lane waves and run on the
/// hardware device (None without one that takes it; an error with the translation's refusal):
/// how many of its 64 lanes differ from `expected`.
fn wrong_lanes(op: &[u32], expected: &dyn Fn(u32) -> u32, host_wave: u32) -> Result<Option<usize>, String> {
    let native = lane_program(op);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered_on(&native, &p, &HashSet::new(), host_wave).map_err(|e| e.to_string())?;
    let Some(got) = whole_uav(&shader, [0, 0], 64) else { return Ok(None) };
    Ok(Some((0..64u32).filter(|&l| got[l as usize] != expected(l)).count()))
}

/// Cross-lane operations that only read lanes of their own 32-lane half of a wave64 (DPP rows,
/// DPP8, `ds_swizzle_b32` in its group-of-32 mode, `ds_bpermute_b32`) and DPP row write masks
/// that name the four rows of the wave, in a host of waves of 32 and of 64: the console's
/// results (RDNA2) in all 64 lanes, with no refusal for the lanes of the other half.
/// (After Marvisak/VIctory#3115, which refuses these where this translation, with its host
/// waves each one half of the console's, runs them.)
#[test]
fn half_local_cross_lane_ops_of_wave64_match_rdna2_in_waves_of_32_and_64() {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let v2 = |l: u32| 0xDEAD_0000 + l;
    // every row (the mask names rows 0-3 of the console's 64 lanes)
    let shr = |row_mask: u8| {
        move |l: u32| {
            if row_mask >> (l >> 4) & 1 != 0 && l & 15 != 0 { lane_v1(l - 1) } else { v2(l) }
        }
    };
    let cases: Vec<(&str, Vec<u32>, Box<dyn Fn(u32) -> u32>)> = vec![
        ("row_ror:4", dpp_row(0x124, 0xF), Box::new(|l| lane_v1(l & !15 | (l + 12) & 15))),
        ("row_mirror", dpp_row(0x140, 0xF), Box::new(|l| lane_v1(l ^ 15))),
        ("row_share:3", dpp_row(0x153, 0xF), Box::new(|l| lane_v1(l & !15 | 3))),
        ("dpp8 reversed", dpp8_reverse(), Box::new(|l| lane_v1(l ^ 7))),
        ("ds_swizzle_b32 SWAP(16)", ds_swizzle(2, 1, 0x401F).to_vec(), Box::new(|l| lane_v1(l ^ 16))),
        ("ds_bpermute_b32 lane + 1 in the half", ds(0xB3, 2, 3, 1).to_vec(), Box::new(|l| lane_v1(l & 32 | (l + 1) & 31))),
        ("row_shr:1 row_mask 0xF", dpp_row(0x111, 0xF), Box::new(shr(0xF))),
        ("row_shr:1 row_mask 0x5", dpp_row(0x111, 0x5), Box::new(shr(0x5))),
        ("row_shr:1 row_mask 0x3", dpp_row(0x111, 0x3), Box::new(shr(0x3))),
        ("row_shr:1 row_mask 0xC", dpp_row(0x111, 0xC), Box::new(shr(0xC))),
    ];
    for host_wave in [32, 64] {
        for (label, op, expected) in &cases {
            let got = wrong_lanes(op, expected, host_wave);
            assert!(matches!(got, Ok(Some(0)) | Ok(None)), "{label} in host waves of {host_wave}: {got:?}");
        }
        eprintln!("ran on the hardware device (host waves of {host_wave}): {} cases exact in all 64 lanes", cases.len());
    }
    // What the translation did before (not asserted).
    // SAFETY: tests that read this switch hold ONE_AT_A_TIME
    unsafe { std::env::set_var("VICTORY_HALF_LOCAL", "0") };
    for host_wave in [32, 64] {
        for (label, op, expected) in &cases {
            match wrong_lanes(op, expected, host_wave) {
                Ok(Some(n)) if n > 0 => eprintln!("before, host waves of {host_wave}: {label}: {n} of 64 lanes wrong"),
                Err(why) => eprintln!("before, host waves of {host_wave}: {label}: refused ({why})"),
                _ => {}
            }
        }
    }
    unsafe { std::env::remove_var("VICTORY_HALF_LOCAL") };
}

/// `ds_swizzle_b32 vdst, vaddr offset:offset`.
const fn ds_swizzle(vdst: u8, vaddr: u8, offset: u16) -> [u32; 2] {
    [0x36 << 26 | 0x35 << 18 | offset as u32, (vdst as u32) << 24 | vaddr as u32]
}

/// `v_writelane_b32 vN, sX, K`.
fn writelane(n: u32, x: u32, lane: u32) -> Vec<u32> {
    vec![0xD761_0000 | n, (128 + lane) << 9 | x]
}

/// `v_readlane_b32 sX, vN, K`.
fn readlane(x: u32, n: u32, lane: u32) -> Vec<u32> {
    vec![0xD760_0000 | x, (128 + lane) << 9 | (256 + n)]
}

fn parked(words: &[Vec<u32>]) -> ir::Program {
    let native = native(&words.concat());
    ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build")
}

fn instructions<'a>(p: &'a ir::Program, name: &str) -> Vec<&'a ir::Inst> {
    p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == name).collect()
}

/// (From avatarmanz/VIctory-7900XTX#82's tests, which this translation's `unspill` passes: a
/// pair parked in two lanes that another save or restore of one of them shares is not moved as
/// a pair. Its other test, a pair that keeps being a pair while one of its lanes holds another
/// scalar elsewhere in the program, needs the pairing by lifetime of that fork, which this
/// one does not have: there the pair comes back as two scalars.)
#[test]
fn a_parked_pair_another_save_or_restore_shares_stays_two_scalars() {
    let cases = [
        (
            "a lone restore of the lane",
            vec![
                writelane(10, 20, 4),
                writelane(10, 21, 5),
                readlane(22, 10, 4),
                readlane(23, 10, 5),
                readlane(24, 10, 4),
            ],
        ),
        (
            "another save of the lane on one path to the restore",
            vec![
                writelane(10, 20, 4),
                writelane(10, 21, 5),
                vec![0xBF06_8001, 0xBF85_0002], // s_cmp_eq_u32 s1, 0 and s_cbranch_scc1 over the save
                writelane(10, 30, 4),
                readlane(22, 10, 4),
                readlane(23, 10, 5),
            ],
        ),
    ];
    for (name, words) in cases {
        let p = parked(&words);
        assert!(instructions(&p, "s_mov_b64").is_empty(), "{name}");
        assert!(!instructions(&p, "s_mov_b32").is_empty(), "{name}");
    }
}

/// `if (t < s0) { if (s1 == 0) { s[22:23] = exec } else { <other> }; exec = s[22:23]; v1 = 0 }`:
/// the restore of EXEC reads what the inner join makes of the two arms.
fn exec_restored_after_an_inner_join(other: u32) -> ir::Program {
    let body = [
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0002, // s_cbranch_scc1 to the second arm
        0xBE96_047E, // s_mov_b64 s[22:23], exec
        0xBF82_0001, // s_branch to the join
        other,
        0xBEFE_0416, // s_mov_b64 exec, s[22:23]
        0x7E02_0280, // v_mov_b32 v1, 0
    ];
    let mut words = vec![
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0000 | body.len() as u32, // s_cbranch_execz past the body
    ];
    words.extend_from_slice(&body);
    words.push(0xBEFE_040C); // s_mov_b64 exec, s[12:13]
    let native = native(&words);
    ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build")
}

/// A mask the skipped region restores into EXEC cannot make a lane of the skip run when the
/// region saved it from its own (empty) EXEC on every path to the restore, however the inner
/// joins and copies carry it: the skip is left out. A path that brings a mask from outside the
/// region may revive lanes: the skip stays.
#[test]
fn a_mask_joined_from_masks_the_skipped_region_saved_is_as_empty() {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let both = exec_restored_after_an_inner_join(0xBE96_047E); // s_mov_b64 s[22:23], exec
    assert_eq!(branches_named(&both, "s_cbranch_execz"), 1);
    let form = ir::without_skips(&both, true).expect("left out");
    assert_eq!(branches_named(&form, "s_cbranch_execz"), 0);
    let from_outside = exec_restored_after_an_inner_join(0xBE96_0406); // s_mov_b64 s[22:23], s[6:7]
    assert!(ir::without_skips(&from_outside, true).is_none());
    // as before with the switch off: the masks saved in the region alone
    // SAFETY: tests that read this switch hold ONE_AT_A_TIME
    unsafe { std::env::set_var("VICTORY_CASCADE_EXACT", "0") };
    let before = ir::without_skips(&both, true);
    unsafe { std::env::remove_var("VICTORY_CASCADE_EXACT") };
    assert!(before.is_none());
}

/// The wave widths the hardware device takes: (min, max).
fn wave_widths(host: &Host) -> (u32, u32) {
    use windows::Win32::Graphics::Direct3D12::{D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1};
    let mut waves = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: the output has the layout and size required by OPTIONS1.
    unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            (&raw mut waves).cast(),
            size_of_val(&waves) as u32,
        )
    }
    .expect("query wave capabilities");
    (waves.WaveLaneCountMin, waves.WaveLaneCountMax)
}

/// The prefix of wave totals the console's structure builder computes: each console
/// wave of a 256-thread group writes its total (10 + its index) to LDS, then, under
/// `if (id >> 6 != 0)`, loops over the totals of the waves before it with lanes leaving at their
/// own wave's index, reading the running sum's first lane (`v_readfirstlane`) every iteration.
/// (The total is read from the wave's last lane first, as the builder's scan does: a read
/// across the two host waves, which makes them exchange.)
/// The condition and the sum are the same in every lane of a console wave, so each host wave
/// takes its own first lane; before, the read was refused ("a first-lane read where not every
/// lane runs"), which kept the builder off on 32-lane GPUs. Every thread stores the sum of the
/// waves before its own: 0, 10, 10 + 11, 10 + 11 + 12.
fn wave_prefix_program(lane_dependent: bool) -> NativeProgram {
    let lds_address = if lane_dependent {
        0x3410_0082 // v_lshlrev_b32 v8, 2, v0
    } else {
        0x7E10_0210 // v_mov_b32 v8, s16
    };
    let words = [
        0x2C0E_0086, // v_lshrrev_b32 v7, 6, v0             (the console wave's index)
        0x3410_0E82, // v_lshlrev_b32 v8, 2, v7
        0x4A12_0E8A, // v_add_nc_u32 v9, 10, v7
        0xD760_0013,
        0x0201_7F09, // v_readlane_b32 s19, v9, 63   (the wave's last lane: the other host wave)
        0x7E12_0213, // v_mov_b32 v9, s19
        0xD834_0000,
        0x0000_0908, // ds_write_b32 v8, v9
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBF8A_0000, // s_barrier
        0x7E10_0280, // v_mov_b32 v8, 0
        0xBE80_047E, // s_mov_b64 s[0:1], exec
        0x7DAA_0E80, // v_cmpx_ne_u32 0, v7
        0xBF88_0011, // s_cbranch_execz END
        0xBE86_047E, // s_mov_b64 s[6:7], exec
        0xBE92_0380, // s_mov_b32 s18, 0
        0xBE91_0380, // s_mov_b32 s17, 0
        0xBF82_0002, // s_branch BODY
        0x87FE_0614, // LOOPTOP: s_and_b64 exec, s[20:21], s[6:7]
        0x7E22_0508, // v_readfirstlane_b32 s17, v8
        0x8F10_8212, // BODY: s_lshl_b32 s16, s18, 2
        0x8112_8112, // s_add_i32 s18, s18, 1
        lds_address, // LDS address for the next subtotal
        0xD8D8_0000,
        0x0800_0008, // ds_read_b32 v8, v8
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x4A10_1011, // v_add_nc_u32 v8, s17, v8
        0xBE94_047E, // s_mov_b64 s[20:21], exec
        0x7DA4_0E12, // v_cmpx_eq_u32 s18, v7
        0x8A86_7E06, // s_andn2_b64 s[6:7], s[6:7], exec
        0xBF85_FFF3, // s_cbranch_scc1 LOOPTOP
        0xBEFE_0400, // END: s_mov_b64 exec, s[0:1]
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xE070_1000,
        0x8002_0801, // buffer_store_dword v8, v1, s[8:11], 0 offen
    ];
    let mut native = native_of(&words, true, 256);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 2 << 15; // 1 KiB covers the lane-varying address as well.
    }
    native
}

/// Runs the prefix program's translation on the hardware device: what its 256 threads store,
/// None where the device takes no wave of 32.
fn run_prefix_on_the_card(shader: &HostShader) -> Option<Vec<u32>> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let (min, max) = wave_widths(&host);
    if !(min..=max).contains(&32) {
        eprintln!("the device does not support wave32; hardware dispatch skipped");
        return None;
    }
    let (root, pso) = host.compute_pipeline(shader).expect("wave32 prefix pipeline");
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(256)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0][..256].to_vec())
}

#[test]
fn a_first_lane_read_of_a_wave_alike_sum_under_a_wave_alike_loop() {
    let native = wave_prefix_program(false);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert!(p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "v_readfirstlane_b32"));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
        eprintln!("DXIL validator accepted the wave32 prefix shader");
    }
    let Some(got) = run_prefix_on_the_card(&shader) else { return };
    let expected = |t: u32| (0..(t >> 6)).map(|w| 10 + w).sum::<u32>();
    for t in 0..256u32 {
        assert_eq!(got[t as usize], expected(t), "thread {t}: {got:?}");
    }
    eprintln!("ran on the hardware device: every console wave got the sum of the waves before it");
}

#[test]
fn a_lane_dependent_prefix_cannot_take_its_own_first_lane() {
    let native = wave_prefix_program(true);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let Err(error) = lowered(&native, &p, &HashSet::new()) else {
        panic!("the two host waves may read different subtotals");
    };
    assert!(error.to_string().contains("a first-lane read where not every lane runs"), "{error}");
}

#[test]
fn the_old_proof_refuses_the_prefix_with_the_switch_off() {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let native = wave_prefix_program(false);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    // SAFETY: tests that read this switch hold ONE_AT_A_TIME
    unsafe { std::env::set_var("VICTORY_WAVE_ALIKE", "0") };
    let kept = lowered(&native, &p, &HashSet::new());
    unsafe { std::env::remove_var("VICTORY_WAVE_ALIKE") };
    let Err(error) = kept else { panic!("the proof of before takes only what every lane runs") };
    assert!(error.to_string().contains("a first-lane read where not every lane runs"), "{error}");
}

/// One console wave of 64 threads with 512 bytes of LDS. The threads clear
/// the flag, then the threads below s0 raise it and read it back inside
/// their branch. Every thread stores the final flag plus one at its own
/// UAV offset. `second` adds another write in the branch after the read.
fn flag_program(second: Option<u32>) -> NativeProgram {
    let mut body = vec![
        0x7E12_0280, // v_mov_b32 v9, 0
    ];
    body.extend_from_slice(&ds(DS_WRITE_B32, 0, 9, 9)); // ds_write_b32 v9, v9
    let mut branch = vec![0x7E04_0281]; // v_mov_b32 v2, 1
    branch.extend_from_slice(&ds(DS_WRITE_B32, 0, 9, 2)); // ds_write_b32 v9, v2
    // (a scalar exit from the branch makes it blocks of their own, which only its lanes run)
    let mut rest = ds(DS_READ_B32, 3, 9, 0).to_vec(); // ds_read_b32 v3, v9
    if let Some(k) = second {
        rest.push(0x7E0C_02FF); // v_mov_b32 v6, lit
        rest.push(k);
        rest.extend_from_slice(&ds(DS_WRITE_B32, 0, 9, 6)); // ds_write_b32 v9, v6
    }
    branch.extend_from_slice(&[
        0xBF07_8001,                     // s_cmp_lg_u32 s1, 0
        0xBF85_0000 | rest.len() as u32, // s_cbranch_scc1 past the rest
    ]);
    branch.extend_from_slice(&rest);
    body.extend_from_slice(&[
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0000 | branch.len() as u32, // s_cbranch_execz past the branch
    ]);
    body.extend_from_slice(&branch);
    body.push(0xBEFE_040C); // s_mov_b64 exec, s[12:13]
    body.extend_from_slice(&ds(DS_READ_B32, 5, 9, 0)); // ds_read_b32 v5, v9
    body.extend_from_slice(&[
        0x4A0A_0A81, // v_add_nc_u32 v5, 1, v5
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 5 << 8 | 1, // buffer_store_dword v5, v1, s[8:11], 0 offen
    ]);
    let mut native = native(&body);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 15; // 512 bytes of LDS
    }
    native
}

/// The wave size `native` lowers to on a host of 32-lane waves.
fn flag_wave(native: &NativeProgram) -> Option<u32> {
    let p = crate::spill::build(native).expect("build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let lowered = lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    lowered.expect("lower").wave_size
}

#[test]
fn a_branch_that_only_raises_an_lds_flag_runs_in_waves_of_32() {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // a read that misses the other host wave's write changes nothing the branch leaves behind
    assert_eq!(flag_wave(&flag_program(None)), Some(32));
    // after the read the branch writes the flag again, with the same constant
    assert_eq!(flag_wave(&flag_program(Some(1))), Some(32));
}

#[test]
fn a_branch_that_writes_lds_two_different_values_keeps_64_lanes() {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // a second value after the read: what the flag ends up holding depends on what was read
    assert_eq!(flag_wave(&flag_program(Some(2))), Some(64));
}

#[test]
fn the_flag_branch_keeps_64_lanes_with_the_switch_off() {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: tests that read this switch hold ONE_AT_A_TIME
    unsafe { std::env::set_var("VICTORY_LDS_FLAGS", "0") };
    let wave = flag_wave(&flag_program(None));
    unsafe { std::env::remove_var("VICTORY_LDS_FLAGS") };
    assert_eq!(wave, Some(64));
}

/// Dispatches the flag program's translation on every count of active lanes, both sides of the
/// branch's scalar skip; the final flag plus one is stored by every thread. Gives the number of
/// dispatches that were checked, None where the device takes no wave of 32.
fn run_flag_program(shader: &HostShader) -> Option<usize> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware adapter, skipping flag dispatches");
        return None;
    };
    let (min, max) = wave_widths(&host);
    if !(min..=max).contains(&32) {
        eprintln!("the device does not support wave32; flag dispatches skipped");
        return None;
    }
    // (a 64-lane program, which a device of 32-lane waves does not take, is not run)
    let Ok((root, pso)) = host.compute_pipeline(shader) else {
        eprintln!("the device does not take this program; flag dispatches skipped");
        return None;
    };
    let mut checked = 0;
    for active in [0, 1, 31, 32, 33, 63, 64] {
        // Exercise both sides of the branch's scalar skip.
        for skip in [0, 1] {
            let mut ud = vec![0; shader.user_data_dwords() as usize];
            for u in &shader.user_data {
                if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
                    ud[u.index as usize] = [active, skip][d as usize];
                }
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(64)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .expect("dispatch the split wave pipeline");
            assert_eq!(out[0][..64], [1 + u32::from(active != 0); 64], "{active} active lanes, skip {skip}");
            checked += 1;
        }
    }
    Some(checked)
}

#[test]
fn both_host_waves_observe_the_lds_flag_after_the_join() {
    let native = flag_program(Some(1));
    let p = crate::spill::build(&native).expect("build");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut validator) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut dxil = shader.dxil.clone();
        validator.validate(&mut dxil).expect("validate the split wave shader");
    }
    assert_eq!(flag_wave(&native), Some(32));
    if let Some(n) = run_flag_program(&shader) {
        eprintln!("all {n} flag dispatches passed on a hardware device that takes waves of 32");
    }
    // What it was before: the same program as 64-lane waves, which a device that takes both
    // widths runs the same (and one of 32-lane waves refuses to build).
    // SAFETY: tests that read this switch hold ONE_AT_A_TIME
    unsafe { std::env::set_var("VICTORY_LDS_FLAGS", "0") };
    let before = lowered(&native, &p, &HashSet::new());
    let before_wave = flag_wave(&native);
    unsafe { std::env::remove_var("VICTORY_LDS_FLAGS") };
    let before = before.expect("lower");
    assert_eq!(before_wave, Some(64));
    if let Some(n) = run_flag_program(&before) {
        eprintln!("before (waves of 64): {n} flag dispatches passed too");
    }
}
