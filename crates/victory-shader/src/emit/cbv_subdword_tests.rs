//! Byte and 16-bit buffer loads (`buffer_load_{ubyte,sbyte,ushort,sshort}`)
//! from a constant buffer: the value at the byte address, taken out of the
//! dword that holds it and extended as the instruction says. They gave the
//! whole dword before (so a value past the first byte of its dword came
//! with the bytes before it). After Marvisak/VIctory pull request 3008 by
//! RVZO6, whose test this follows.
use super::host::{Dispatch, Host, Root, View};
use super::intersect_tests::build_compute_shader;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};

/// A wave32 compute shader that reads a field of a constant buffer at byte
/// `lane * step` (`step` is 1 for byte loads, 2 for short loads) and stores
/// the 32-bit result at dword `lane` of the raw buffer its second table
/// holds. The constant buffer is a root view at `s[0:1]`, or a table entry
/// (`table_cbv`).
fn program(mubuf_op: u32, step_shift: u32, table_cbv: bool) -> Vec<u32> {
    let mut w = Vec::new();
    if table_cbv {
        w.extend([
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
        ]);
    } else {
        // The constant buffer's resource record, built around the root
        // view's 64-bit address in s[0:1].
        w.extend([
            0xBE88_0300, // s_mov_b32 s8, s0
            0xBE89_0301, // s_mov_b32 s9, s1
            0xBE8A_03C0, // s_mov_b32 s10, 64
            0xBE8B_03FF,
            0x0002_0000, // s_mov_b32 s11, 0x0002_0000
        ]);
    }
    w.extend([
        0xBE8C_0302, // s_mov_b32 s12, s2
        0xBE8D_0380, // s_mov_b32 s13, 0
        0xF408_0406,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    ]);
    // v1 = v0 << step_shift
    w.push(0x3402_0080 | step_shift);
    w.push(0xBF8C_C07F); // s_waitcnt lgkmcnt(0)
    // buffer_load_{ubyte,sbyte,ushort,sshort} v2, v1, s[8:11], 0 offen
    w.push(0xE000_1000 | (mubuf_op << 18));
    w.push(0x8002_0201);
    w.push(0xBF8C_3F70); // s_waitcnt vmcnt(0)
    w.push(0x3406_0082); // v_lshlrev_b32 v3, 2, v0
    w.push(0xE070_1000);
    w.push(0x8004_0203); // buffer_store_dword v2, v3, s[16:19], 0 offen
    w.push(0xBF81_0000); // s_endpgm
    w
}

pub(super) fn native(mubuf_op: u32, step_shift: u32, table_cbv: bool, threads: u32) -> NativeProgram {
    let cbv_param = RootParam {
        root: if table_cbv { RootType::Table } else { RootType::Descriptor },
        table_offset_dwords: 0,
        sgprs: if table_cbv { 1 } else { 2 },
        base_register: 0,
        user_data: 0,
        slot: 0,
    };
    let uav_param = RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 2,
        slot: 2,
    };
    let used = |kind, root, user_data, slot| UsedResource {
        kind,
        root,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data,
        slot,
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![cbv_param, uav_param],
        used: vec![
            used(ResourceType::Cbv, cbv_param.root, 0, 0),
            used(ResourceType::Uav, RootType::Table, 2, 2),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [threads, 1, 1],
        }),
        code: program(mubuf_op, step_shift, table_cbv)
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect(),
    }
}

/// The four loads translate to four different programs, in both ways of
/// holding the buffer.
#[test]
fn the_width_and_the_sign_of_a_constant_buffer_load_change_the_program() {
    for table_cbv in [true, false] {
        let dxil = |op, shift, threads| build_compute_shader(&native(op, shift, table_cbv, threads)).unwrap().0.dxil;
        let (ubyte, sbyte) = (dxil(0x008, 0, 16), dxil(0x009, 0, 16));
        let (ushort, sshort) = (dxil(0x00a, 1, 8), dxil(0x00b, 1, 8));
        assert_ne!(ubyte, sbyte, "table {table_cbv}: ubyte and sbyte");
        assert_ne!(ushort, sshort, "table {table_cbv}: ushort and sshort");
        assert_ne!(ubyte, ushort, "table {table_cbv}: ubyte and ushort");
        assert_ne!(sbyte, sshort, "table {table_cbv}: sbyte and sshort");
    }
}

/// The values come out right, on the hardware and on WARP.
#[test]
fn constant_buffer_byte_and_short_loads_take_their_own_bits() {
    let data: [u32; 4] = [0x80FF_7F01, 0xFE02_817E, 0xA55A_00FF, 0x7FFF_8000];
    let bytes: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
    // (name, opcode, step shift, threads, what each lane stores)
    let cases: [(&str, u32, u32, u32, Vec<u32>); 4] = [
        ("buffer_load_ubyte", 0x008, 0, 16, bytes.iter().map(|&b| u32::from(b)).collect()),
        ("buffer_load_sbyte", 0x009, 0, 16, bytes.iter().map(|&b| i32::from(b as i8) as u32).collect()),
        (
            "buffer_load_ushort",
            0x00a,
            1,
            8,
            bytes.chunks_exact(2).map(|c| u32::from(u16::from_le_bytes([c[0], c[1]]))).collect(),
        ),
        (
            "buffer_load_sshort",
            0x00b,
            1,
            8,
            bytes.chunks_exact(2).map(|c| i32::from(i16::from_le_bytes([c[0], c[1]])) as u32).collect(),
        ),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (name, warp) in [("WARP", true), ("hardware", false)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {name} device; skipped");
            continue;
        };
        for table_cbv in [true, false] {
            for (op_name, op, shift, threads, want) in &cases {
                let (shader, _) = build_compute_shader(&native(*op, *shift, table_cbv, *threads)).unwrap();
                let mut user_data = vec![0u32; shader.user_data_dwords().max(1) as usize];
                for u in &shader.user_data {
                    user_data[u.index as usize] = match (u.root, u.slot) {
                        (RootType::Table, 0) => 0,
                        // The raw buffer's view follows the constant
                        // buffer's, 32 bytes on, when both are in the heap.
                        (RootType::Table, 2) => u32::from(table_cbv) * 32,
                        other => panic!("unexpected user data {other:?}"),
                    };
                }
                let uav = View::RawUav(*threads);
                let cbv = View::Cbv(&data);
                let (roots, views): (Vec<(u32, Root)>, Vec<View>) = if table_cbv {
                    (vec![(0, Root::Constants(user_data))], vec![cbv, uav])
                } else {
                    (
                        vec![
                            (0, Root::Constants(user_data)),
                            (1, Root::View { srv: false, data: data.to_vec() }),
                        ],
                        vec![uav],
                    )
                };
                let (root, pso) = host.compute_pipeline(&shader).unwrap();
                let out = host
                    .dispatch(&root, &pso, &Dispatch { roots, views: &views, samplers: &[], groups: [1, 1, 1] })
                    .unwrap();
                assert_eq!(out[0], *want, "{name} {op_name} table {table_cbv}");
                ran += 1;
            }
        }
    }
    if ran == 0 {
        eprintln!("no device: the loads were not checked");
    }
}
