//! Constant buffers read as raw bytes (`VICTORY_PORT_CBV=1`, renderer port
//! 5: `raw_constants`). Built from avatarmanz/VIctory-7900XTX's test of the
//! same switch (GPL-3.0, `3920dc2`), which reads a table's constant buffer
//! through the raw twin of its slot; here the slot itself holds the raw view.
use super::cbv_subdword_tests::native;
use super::host::{Dispatch, Host, Root, Stride, View};
use super::intersect_tests::build_compute_shader;
use crate::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};

/// The translation with the switch on or off, for this thread alone.
fn built(program: &NativeProgram, raw: bool) -> super::HostShader {
    super::shader::CONSTANTS_RAW_HERE.set(Some(raw));
    let built = build_compute_shader(program);
    super::shader::CONSTANTS_RAW_HERE.set(None);
    built.unwrap().0
}

/// A program that reads a constant buffer at a constant offset and at one it
/// computes, and stores what it read to a raw UAV, with the constant buffer a
/// table's entry (the table's first, the UAV's the entry after it).
fn scalar_program() -> NativeProgram {
    let mut words: Vec<u32> = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0 (the constant buffer)
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20 (the UAV)
        0xF428_0404,
        0xFA00_0010, // s_buffer_load_dwordx4 s[16:19], s[8:11], 0x10
        0xF420_0504,
        0x0000_0000, // s_buffer_load_dword s20, s[8:11], s0
    ];
    for (out, s) in (16..21u32).enumerate() {
        // v_mov_b32 v1, s<s>; buffer_store_dword v1, s[12:15], offset 4 * out
        words.push(0x7E00_0000 | 1 << 17 | 0x001 << 9 | s);
        words.push(0x38 << 26 | 0x1C << 18 | (4 * out as u32));
        words.push(0x80 << 24 | (12 >> 2) << 16 | 1 << 8);
    }
    words.push(0xBF81_0000); // s_endpgm
    let table = |kind, at| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: at,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            RootParam { root: RootType::Constants, table_offset_dwords: 0, sgprs: 4, base_register: 0, user_data: 0, slot: 0 },
            RootParam { root: RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: 4, slot: 1 },
        ],
        used: vec![table(ResourceType::Cbv, 0), table(ResourceType::Uav, 8)],
        io: StageIo::Cs(ComputeIo { rsrc2: 5 << 1, threads: [1, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// A table's constant buffer, read at a constant and at a computed offset,
/// gives the same dwords as raw bytes (the slot a raw view) as it does as a
/// constant buffer; the translation is another one with the switch on.
#[test]
fn table_constant_buffers_read_as_raw_bytes_in_their_slots() {
    let native = scalar_program();
    let (before, after) = (built(&native, false), built(&native, true));
    assert_ne!(before.dxil, after.dxil, "the switch changes the translation");
    let data: Vec<u32> = (0..16u32).map(|i| 0x0101_0101u32.wrapping_mul(i + 3) ^ 0x4000_0000).collect();
    let s0 = 8u32;
    let want: Vec<u32> = vec![data[4], data[5], data[6], data[7], data[(s0 / 4) as usize]];
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (raw, shader) in [(false, before), (true, after)] {
        let Ok(host) = Host::new(true) else {
            eprintln!("no software adapter; the dispatch is skipped");
            return;
        };
        let (root, pso) = host.compute_pipeline(&shader).unwrap();
        // The constant buffer's slot, then the UAV's.
        let views = if raw {
            vec![View::Buffer { data: &data, stride: Stride::Raw }, View::RawUav(8)]
        } else {
            vec![View::Cbv(&data), View::RawUav(8)]
        };
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch { roots: vec![(0, Root::Constants(vec![s0, 0, 0, 0, 0]))], views: &views, samplers: &[], groups: [1, 1, 1] },
            )
            .unwrap();
        let got = out.iter().find(|o| o[..5].iter().any(|&w| w != 0)).map(|o| o[..5].to_vec());
        assert_eq!(got.as_deref(), Some(&want[..]), "switch {raw}: {out:x?}");
    }
}

/// A read past the end of the raw view gives zeros, as one past the end of a
/// constant buffer does: the view holds the first eight dwords of the
/// sixteen the program's computed read can reach.
#[test]
fn a_raw_constant_buffer_read_ends_where_its_view_does() {
    let native = scalar_program();
    let shader = built(&native, true);
    let data: Vec<u32> = (0..8u32).map(|i| 0x0202_0202u32.wrapping_mul(i + 1) ^ 0x1000_0000).collect();
    // The constant read at 0x10 lies inside; the computed one (at 0x28) lies past the end.
    let s0 = 0x28u32;
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(true) else {
        eprintln!("no software adapter; the dispatch is skipped");
        return;
    };
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    let views = vec![View::Buffer { data: &data, stride: Stride::Raw }, View::RawUav(8)];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch { roots: vec![(0, Root::Constants(vec![s0, 0, 0, 0, 0]))], views: &views, samplers: &[], groups: [1, 1, 1] },
        )
        .unwrap();
    let got = out.iter().find(|o| o[..5].iter().any(|&w| w != 0)).map(|o| o[..5].to_vec());
    assert_eq!(got.as_deref(), Some(&[data[4], data[5], data[6], data[7], 0][..]), "{out:x?}");
}

/// The byte and 16-bit loads of a constant buffer (a table's and a root one)
/// give the same values as raw bytes: a table's through the slot's raw view,
/// a root one as a root SRV with its bias.
#[test]
fn constant_buffer_byte_and_short_loads_read_raw_too() {
    let data: [u32; 4] = [0x80FF_7F01, 0xFE02_817E, 0xA55A_00FF, 0x7FFF_8000];
    let bytes: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
    // (opcode, step shift, threads, what each lane stores)
    let cases: [(u32, u32, u32, Vec<u32>); 4] = [
        (0x008, 0, 16, bytes.iter().map(|&b| u32::from(b)).collect()),
        (0x009, 0, 16, bytes.iter().map(|&b| i32::from(b as i8) as u32).collect()),
        (0x00a, 1, 8, bytes.chunks_exact(2).map(|c| u32::from(u16::from_le_bytes([c[0], c[1]]))).collect()),
        (0x00b, 1, 8, bytes.chunks_exact(2).map(|c| i32::from(i16::from_le_bytes([c[0], c[1]])) as u32).collect()),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(true) else {
        eprintln!("no software adapter; the dispatch is skipped");
        return;
    };
    for table_cbv in [true, false] {
        for (op, shift, threads, want) in &cases {
            let program = native(*op, *shift, table_cbv, *threads);
            let (off, shader) = (built(&program, false), built(&program, true));
            assert_ne!(off.dxil, shader.dxil, "op {op:#x} table {table_cbv}: the switch changes the translation");
            if !table_cbv {
                // The constant buffer a root SRV, with a bias word of the host's.
                assert!(shader.root_cbvs.is_empty(), "op {op:#x}: no root constant buffer view is left");
                assert_eq!(shader.root_srvs.len(), 1, "op {op:#x}: the root constant buffer is a root SRV");
            }
            let mut user_data = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                user_data[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Table, 0) => 0,
                    (RootType::Table, 2) => u32::from(table_cbv) * 32,
                    // The root view begins where its buffer does.
                    (RootType::Other(code), 0) if code == super::USER_VIEW_BIAS => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let uav = View::RawUav(*threads);
            let (roots, views): (Vec<(u32, Root)>, Vec<View>) = if table_cbv {
                (
                    vec![(0, Root::Constants(user_data))],
                    vec![View::Buffer { data: &data, stride: Stride::Raw }, uav],
                )
            } else {
                (
                    vec![(0, Root::Constants(user_data)), (1, Root::View { srv: true, data: data.to_vec() })],
                    vec![uav],
                )
            };
            let (root, pso) = host.compute_pipeline(&shader).unwrap();
            let out = host
                .dispatch(&root, &pso, &Dispatch { roots, views: &views, samplers: &[], groups: [1, 1, 1] })
                .unwrap();
            assert_eq!(out[0], *want, "op {op:#x} table {table_cbv}");
        }
    }
}
