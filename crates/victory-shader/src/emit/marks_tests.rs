//! Write marks (module notes, "Write marks"): stores and atomics through
//! unordered buffer views of the heap set the bits of the title pages they
//! write, and programs without such writes come out as without marks.

use super::host::{Dispatch, Host, Root, View};
use super::lower::marks::{TEST_WAVE_MARKS, WaveMarks};
use super::{HostShader, SLOT_VIEW_MARKS, SLOT_VIEW_WRITES, VIEW_MARK_NONE, lower, validator};
use crate::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
/// Inline constant 0.
const ZERO: u8 = 0x80;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], offset`.
const fn s_load_dwordx4(dst: u8, base: u8, offset: u32) -> [u32; 2] {
    [0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1), 125 << 25 | offset]
}

/// VOP1 `v_mov_b32 vdst, s<src>`.
const fn v_mov_b32(vdst: u8, src: u8) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | 0x001 << 9 | src as u32
}

/// MUBUF `op` of `vdata` at `byte_off` through `s[srsrc:srsrc+3]`.
const fn mubuf(op: u32, vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | op << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}
const BUFFER_STORE_DWORD: u32 = 0x1C;
const BUFFER_STORE_DWORDX4: u32 = 0x1E;
const BUFFER_ATOMIC_ADD: u32 = 0x32;

/// MUBUF `op` of `vdata` through `s[srsrc:srsrc+3]` at the byte in `vaddr`.
const fn mubuf_offen(op: u32, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | op << 18 | 1 << 12,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// VOP2 `v_lshlrev_b32 vdst, s<shift>, v<src>`.
const fn v_lshlrev_b32(vdst: u8, shift: u8, src: u8) -> u32 {
    0x1A << 25 | (vdst as u32) << 17 | (src as u32) << 9 | shift as u32
}

const S_NOP: u32 = 0xBF80_0000;

/// VOP2 `op vdst, src0, v<vsrc1>`; `src0` is a 9-bit source: an SGPR's
/// number, 128 + n for the inline constant n, 256 + v for a VGPR.
const fn vop2(op: u32, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    op << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | src0 as u32
}
const V_LSHRREV_B32: u32 = 0x16;
const V_AND_B32: u32 = 0x1B;
const V_ADD_NC_U32: u32 = 0x25;

/// A one-thread compute program with 4 dwords of root constants in
/// `s0..s3` and a table of `views` raw UAVs at user data 4; `body` runs
/// after the descriptors are in `s8..` (four SGPRs each).
fn program(views: u32, body: &[[u32; 2]], marks: bool) -> HostShader {
    program_of(views, body, marks, 1)
}

/// [`program`] with `threads` threads in x; with more than one, v0 keeps
/// the thread's index.
fn program_of(views: u32, body: &[[u32; 2]], marks: bool, threads: u32) -> HostShader {
    let mut w = vec![s_mov_b32(5, ZERO)];
    for k in 0..views {
        w.extend(s_load_dwordx4(8 + 4 * k as u8, 4, 32 * k));
    }
    let first = u8::from(threads > 1);
    w.extend((first..4).map(|v| v_mov_b32(v, 0)));
    for words in body {
        w.extend(words);
    }
    w.push(S_ENDPGM);
    let param = |root, user_data, sgprs, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let uav = |k: u32| UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 8 * k,
        lower_bound: k,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![param(RootType::Constants, 0, 4, 0), param(RootType::Table, 4, 1, 1)],
        used: (0..views).map(uav).collect(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [threads, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let parts = lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    };
    let (l, whole) = if marks {
        lower::with_view_marks(|| lower::lower(&parts))
    } else {
        (lower::lower(&parts), true)
    };
    let l = l.expect("lower");
    assert!(whole, "every write marked");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        written_root_uavs: l.written_root_uavs,
        writes_guest_memory: l.writes_guest_memory,
        root_indexed_views: res.root_indexed_views,
        half_views: Vec::new(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    shader
}

/// A dword store and a 16-byte store through view 0 (first bit 100, its
/// first byte 16 bytes before a 64 KiB title page ends, so the second
/// store crosses into the next page), a store through view 1 (not marked)
/// and an atomic add through view 0 (dword 2, still zero). The writes
/// land; exactly bits 100 and 101 are set.
#[test]
fn writes_through_views_mark_their_pages() {
    let body = [
        mubuf(BUFFER_STORE_DWORD, 0, 8, 0),
        mubuf(BUFFER_STORE_DWORDX4, 0, 8, 12),
        mubuf(BUFFER_STORE_DWORD, 0, 12, 4),
        mubuf(BUFFER_ATOMIC_ADD, 0, 8, 8),
    ];
    let shader = program(2, &body, true);
    assert!(super::marks_writes(&shader), "the bitmap is bound");
    assert!(shader.root_srvs.contains(&SLOT_VIEW_MARKS) && shader.root_uavs.contains(&SLOT_VIEW_WRITES));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let value = 0x1234_5678;
    let marks = vec![100, 0xFFF0, VIEW_MARK_NONE, 0];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = if u.root == RootType::Constants { value } else { 0 };
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    // Root parameters: user data, then root views by slot
                    // and kind (the mark table, then the bitmap).
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (1, Root::View { srv: true, data: marks.clone() }),
                        (2, Root::Uav(8)),
                    ],
                    views: &[View::RawUav(8), View::RawUav(8)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(out[0], [value, 0, value, value, value, value, value, 0], "{label}: view 0");
        assert_eq!(out[1][1], value, "{label}: view 1");
        let marked: Vec<usize> = (0..out[2].len() * 32).filter(|&b| out[2][b / 32] >> (b % 32) & 1 != 0).collect();
        assert_eq!(marked, [100, 101], "{label}: pages");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// A program that writes through no view of the heap is the same with
/// marks asked for as without: no root views of the marks, the same code.
#[test]
fn programs_without_view_writes_are_unchanged() {
    let without = program(1, &[], false);
    let with = program(1, &[], true);
    assert!(!super::marks_writes(&with));
    assert_eq!(with.dxil, without.dxil);
    assert_eq!((with.root_srvs, with.root_uavs), (without.root_srvs, without.root_uavs));
}

/// 64 threads store their index at byte `index << s0` of view 0 (first
/// bit 100, from its page's start): a shift of 2 writes one page, 11 two
/// pages of one bitmap dword, 16 a page each over two dwords, 21 a dword
/// each. Lanes of one dword set their bits with one atomic between them
/// (wave operations, whatever the wave's size); every lane's page is
/// marked and no other, and the stores that fit the view land.
#[test]
fn lanes_of_a_wave_mark_every_page_they_write() {
    let body = [[v_lshlrev_b32(1, 0, 0), S_NOP], mubuf_offen(BUFFER_STORE_DWORD, 0, 1, 8)];
    let shader = program_of(1, &body, true, 64);
    assert!(super::marks_writes(&shader), "the bitmap is bound");
    let container = crate::dxbc::Container::parse(&shader.dxil).expect("container");
    let sfi = container.part(crate::dxbc::FourCC::SFI0).expect("SFI0").data;
    let features = u64::from_le_bytes(sfi.try_into().expect("8 bytes"));
    assert_ne!(features & 0x4000, 0, "wave operations declared");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let marks = vec![100, 0];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        for shift in [2u32, 11, 16, 21] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = if u.root == RootType::Constants { shift } else { 0 };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(ud)),
                            (1, Root::View { srv: true, data: marks.clone() }),
                            (2, Root::Uav(4096 / 32)),
                        ],
                        views: &[View::RawUav(64)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let mut data = vec![0u32; 64];
            let mut pages = Vec::new();
            for lane in 0..64u32 {
                let byte = lane << shift;
                if byte < 256 {
                    data[(byte / 4) as usize] = lane;
                }
                pages.push(100 + (byte >> 16) as usize);
            }
            pages.dedup();
            assert_eq!(out[0], data, "{label}: view 0, shift {shift}");
            let marked: Vec<usize> = (0..out[1].len() * 32).filter(|&b| out[1][b / 32] >> (b % 32) & 1 != 0).collect();
            assert_eq!(marked, pages, "{label}: pages, shift {shift}");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// [`program_of`] with the lanes of a wave marking the way `mode` says.
fn program_in(mode: WaveMarks, views: u32, body: &[[u32; 2]], threads: u32) -> HostShader {
    let was = TEST_WAVE_MARKS.replace(Some(mode));
    let shader = program_of(views, body, true, threads);
    TEST_WAVE_MARKS.set(was);
    shader
}

/// 64 threads store their index at byte `(((lane + s1) >> 6) << s0) +
/// ((lane & s2) << s3)` of view 0 (`s0` to `s3` the root constants):
/// `s0` and `s1` put the lanes whose `lane + s1` reaches 64 (the last lane
/// for 1, all but the first for 63, the upper half for 32) a page or more
/// from the rest, `s2` and `s3` spread the lanes among themselves.
fn spread_body() -> Vec<[u32; 2]> {
    let pair = |w: u32| [w, S_NOP];
    vec![
        // v1 = (lane + s1) >> 6, then << s0
        pair(vop2(V_ADD_NC_U32, 1, 1, 0)),
        pair(vop2(V_LSHRREV_B32, 1, 128 + 6, 1)),
        pair(v_lshlrev_b32(1, 0, 1)),
        // v2 = (lane & s2) << s3
        pair(vop2(V_AND_B32, 2, 2, 0)),
        pair(v_lshlrev_b32(2, 3, 2)),
        // the byte, and the store at it
        pair(vop2(V_ADD_NC_U32, 1, 256 + 1, 2)),
        mubuf_offen(BUFFER_STORE_DWORD, 0, 1, 8),
    ]
}

/// The byte of view 0 that `lane` stores at ([`spread_body`]).
fn spread_byte(lane: u32, [s0, s1, s2, s3]: [u32; 4]) -> u32 {
    (((lane + s1) >> 6) << s0).wrapping_add((lane & s2) << s3)
}

/// Whichever way the lanes of a wave set their marks (each its own, as one
/// when they all share a dword, as the leading lane's dword's lanes do),
/// the bits come out the same: exactly the pages of the lanes that wrote.
/// The spreads have the lanes of the leading lane's dword the most of the
/// wave, a few of it, or only that lane, with the others in the next dword,
/// in dwords of their own or a dword apart; the view's first page is the
/// first bit of a dword, inside one, or the last of one. The stores land
/// the same too.
#[test]
fn every_way_of_marking_by_wave_sets_the_same_bits() {
    let body = spread_body();
    let modes = [WaveMarks::PerLane, WaveMarks::AllEqual, WaveMarks::Leader];
    let shaders: Vec<_> = modes.iter().map(|&m| program_in(m, 1, &body, 64)).collect();
    let waves = |shader: &HostShader| {
        let container = crate::dxbc::Container::parse(&shader.dxil).expect("container");
        let sfi = container.part(crate::dxbc::FourCC::SFI0).expect("SFI0").data;
        u64::from_le_bytes(sfi.try_into().expect("8 bytes")) & 0x4000 != 0
    };
    assert!(!waves(&shaders[0]), "marks one by one need no wave operations");
    assert!(waves(&shaders[1]) && waves(&shaders[2]), "the others declare them");
    assert_ne!(shaders[1].dxil, shaders[2].dxil, "the leading lane's dword is another scheme");
    assert_ne!(shaders[0].dxil, shaders[2].dxil);
    // (s0, s1, s2, s3)
    let spreads: [[u32; 4]; 20] = [
        // every lane the same byte
        [2, 0, 0, 2],
        // one page, distinct dwords of it
        [16, 0, 63, 2],
        // the last lane a page on, in the leading lane's dword
        [16, 1, 0, 2],
        [16, 1, 63, 2],
        // all but the first lane a page on
        [16, 63, 0, 2],
        // the last lane a dword on
        [21, 1, 0, 2],
        [21, 1, 63, 2],
        // all but the first lane a dword on
        [21, 63, 0, 2],
        [21, 63, 63, 2],
        // the first two lanes, then the rest
        [21, 62, 0, 2],
        // halves, a page apart and a dword apart
        [16, 32, 0, 2],
        [21, 32, 0, 2],
        // a lane short of a third
        [21, 31, 0, 2],
        // two pages of one dword
        [16, 0, 63, 11],
        // a page each: 64 pages over three dwords
        [16, 0, 63, 16],
        [21, 1, 63, 16],
        [21, 63, 63, 16],
        // pages 32 apart, so a dword each
        [21, 0, 63, 21],
        // the last lane far off among near pages
        [22, 1, 7, 16],
        [22, 62, 3, 12],
    ];
    // The first bit of the view's first page: a dword's start, inside it,
    // its last.
    let firsts = [96u32, 100, 127];
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for (shader, mode) in shaders.iter().zip(modes) {
            let (root, pso) = host.compute_pipeline(shader).expect("pipeline");
            for spread in spreads {
                for first in firsts {
                    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
                    for u in &shader.user_data {
                        ud[u.index as usize] = if u.root == RootType::Constants { spread[u.dword as usize] } else { 0 };
                    }
                    let out = host
                        .dispatch(
                            &root,
                            &pso,
                            &Dispatch {
                                roots: vec![
                                    (0, Root::Constants(ud)),
                                    (1, Root::View { srv: true, data: vec![first, 0] }),
                                    (2, Root::Uav(4096 / 32)),
                                ],
                                views: &[View::RawUav(64)],
                                samplers: &[],
                                groups: [1, 1, 1],
                            },
                        )
                        .unwrap_or_else(|e| panic!("{label}, {mode:?}: {e}"));
                    let what = format!("{label}, {mode:?}, spread {spread:?}, first bit {first}");
                    let mut pages: Vec<usize> =
                        (0..64).map(|lane| first as usize + (spread_byte(lane, spread) >> 16) as usize).collect();
                    pages.sort_unstable();
                    pages.dedup();
                    let marked: Vec<usize> =
                        (0..out[1].len() * 32).filter(|&b| out[1][b / 32] >> (b % 32) & 1 != 0).collect();
                    assert_eq!(marked, pages, "{what}: pages");
                    // Each store that fits the view lands (where lanes share
                    // a byte, one of theirs does).
                    for at in 0..64u32 {
                        let writers: Vec<u32> = (0..64).filter(|&l| spread_byte(l, spread) == at * 4).collect();
                        match writers.as_slice() {
                            [] => assert_eq!(out[0][at as usize], 0, "{what}: dword {at}"),
                            [one] => assert_eq!(out[0][at as usize], *one, "{what}: dword {at}"),
                            many => assert!(many.contains(&out[0][at as usize]), "{what}: dword {at}"),
                        }
                    }
                }
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}
