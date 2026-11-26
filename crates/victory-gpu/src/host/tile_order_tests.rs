//! The tile mappings of a list reach the queue after the lists ahead of it
//! in the submission (fork pull request #784, after which the work that
//! gives a reserved texture its tiles is [`HostList::defer_tiles`]).

use super::*;

#[test]
fn tile_work_splits_a_submission_before_each_later_list_that_has_it() {
    // No work, or work on the first run only: one piece, with every
    // list in it (a trailing one for the card's compare among them).
    assert_eq!(tile_segments(&[], &[], 0), vec![(None, 0..0)]);
    assert_eq!(tile_segments(&[false, false], &[0, 2], 5), vec![(None, 0..5)]);
    assert_eq!(tile_segments(&[true, false], &[0, 1], 3), vec![(Some(0), 0..3)]);
    // Work on the second run: a piece before it.
    assert_eq!(tile_segments(&[false, true], &[0, 2], 4), vec![(None, 0..2), (Some(1), 2..4)]);
    // Runs 0 and 2 of three: the first two stay together.
    assert_eq!(
        tile_segments(&[true, false, true], &[0, 1, 3], 5),
        vec![(Some(0), 0..3), (Some(2), 3..5)]
    );
    // Runs with work one after the other each have a piece.
    assert_eq!(
        tile_segments(&[false, true, true], &[0, 1, 2], 4),
        vec![(None, 0..1), (Some(1), 1..2), (Some(2), 2..4)]
    );
}

/// With `defer_tiles` the work is not among the steps run ahead of the
/// submission, and is found where the list is reached.
#[test]
fn tile_work_is_not_run_with_the_steps_ahead_of_the_lists() {
    use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
    let Some(host) = host() else {
        return; // no host GPU
    };
    let list = host.new_list().unwrap();
    let ran = std::sync::Arc::new(AtomicU32::new(0));
    let seen = ran.clone();
    list.defer_tiles(move || {
        seen.fetch_add(1, Relaxed);
    });
    assert!(list.has_tile_steps());
    list.run_deferred(&mut Batch::default());
    assert_eq!(ran.load(Relaxed), 0, "run with the steps, ahead of every list of the submission");
    list.run_tile_steps();
    assert_eq!(ran.load(Relaxed), 1);
}

/// Multi-list submission `[List 0, List 1]` where `List 0` reads an unmapped
/// tiled texture and `List 1` maps a recycled dirty tile onto it and overwrites
/// it. `List 1`'s `defer_tiles` `UpdateTileMappings` must run AFTER `List 0`
/// executes so `List 0` reads zeros rather than stale previous-owner VRAM.
#[test]
fn deferred_tile_mappings_on_second_list_do_not_expose_stale_vram_to_first_list() {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    let Some(host) = super::host() else {
        return;
    };
    // (Says which adapter this ran on.)
    eprintln!("tile mapping order: on {}", if host.software() { "a software adapter" } else { "the card" });
    let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: querying device tiled-resources tier.
    let ok = unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            (&raw mut opts).cast(),
            size_of_val(&opts) as u32,
        )
    };
    if ok.is_err() || opts.TiledResourcesTier.0 < 2 {
        return; // requires Tier 2 null-tile zero reads
    }

    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: 256,
        Height: 256,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0u64;
    // SAFETY: querying copy footprint for a valid 256x256 R8 texture.
    unsafe {
        host.device.GetCopyableFootprints(&desc, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
    }

    // SAFETY: creating a 64 KiB tile heap and two reserved textures on a live device.
    let (heap, donor, target): (ID3D12Heap, ID3D12Resource, ID3D12Resource) = unsafe {
        let mut heap: Option<ID3D12Heap> = None;
        host.device
            .CreateHeap(
                &D3D12_HEAP_DESC {
                    SizeInBytes: 65536,
                    Properties: D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    Alignment: 65536,
                    Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
                },
                &mut heap,
            )
            .unwrap();
        let mut donor: Option<ID3D12Resource> = None;
        host.device
            .CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut donor)
            .unwrap();
        let mut target: Option<ID3D12Resource> = None;
        host.device
            .CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COPY_SOURCE, None, &mut target)
            .unwrap();
        (heap.unwrap(), donor.unwrap(), target.unwrap())
    };

    let map_tile = |queue: &ID3D12CommandQueue, tex: &ID3D12Resource, heap: Option<&ID3D12Heap>| {
        let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
        let size = D3D12_TILE_REGION_SIZE {
            NumTiles: 1,
            ..Default::default()
        };
        let flag = if heap.is_some() {
            D3D12_TILE_RANGE_FLAG_NONE
        } else {
            D3D12_TILE_RANGE_FLAG_NULL
        };
        let offset = 0u32;
        let count = 1u32;
        // SAFETY: single-tile mapping on a live 1-tile reserved resource.
        unsafe {
            queue.UpdateTileMappings(
                tex,
                1,
                Some(&coord),
                Some(&size),
                heap,
                1,
                Some(&flag),
                Some(&offset),
                Some(&count),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
    };

    let make_readback = || -> ID3D12Resource {
        let mut rb: Option<ID3D12Resource> = None;
        // SAFETY: creating a readback buffer on a live device.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_READBACK,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: total,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: DXGI_FORMAT_UNKNOWN,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut rb,
                )
                .unwrap();
        }
        rb.unwrap()
    };

    let copy_tex_to_buf = |list: &super::HostList, tex: &ID3D12Resource, buf: &ID3D12Resource| {
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed handle held alive by test scope.
            pResource: unsafe { std::mem::transmute_copy(buf) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
        };
        let src = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed handle held alive by test scope.
            pResource: unsafe { std::mem::transmute_copy(tex) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        // SAFETY: open command list and compatible texture/buffer footprints.
        unsafe { list.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
    };
    let copy_buf_to_tex = |list: &super::HostList, buf: &ID3D12Resource, tex: &ID3D12Resource| {
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed handle held alive by test scope.
            pResource: unsafe { std::mem::transmute_copy(tex) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let src = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed handle held alive by test scope.
            pResource: unsafe { std::mem::transmute_copy(buf) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
        };
        // SAFETY: open command list and compatible buffer/texture footprints.
        unsafe { list.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
    };

    // Step 1: Poison heap tile 0 with 0xa5 via donor texture, then unmap donor.
    let (up_poison, ptr_poison) = host.upload_buffer(total).unwrap();
    // SAFETY: writing to mapped upload buffer.
    unsafe { std::ptr::write_bytes(ptr_poison, 0xa5, total as usize) };
    let poison_list = host.new_list().unwrap();
    copy_buf_to_tex(&poison_list, &up_poison, &donor);
    poison_list.close();
    map_tile(host.queue(), &donor, Some(&heap));
    host.execute(&[&poison_list]);
    map_tile(host.queue(), &donor, None);

    // Step 2: Record [list1, list2] in one multi-list submission.
    // list1 reads `target` (currently unmapped -> must see 0x00).
    let rb_before = make_readback();
    let list1 = host.new_list().unwrap();
    copy_tex_to_buf(&list1, &target, &rb_before);
    list1.close();

    // list2 maps the dirty 0xa5 heap tile onto `target` via defer_tiles,
    // overwrites `target` with 0x3c, and reads back `target`.
    let (up_fresh, ptr_fresh) = host.upload_buffer(total).unwrap();
    // SAFETY: writing to mapped upload buffer.
    unsafe { std::ptr::write_bytes(ptr_fresh, 0x3c, total as usize) };
    let rb_after = make_readback();
    let list2 = host.new_list().unwrap();
    {
        let target_clone = target.clone();
        let heap_clone = heap.clone();
        list2.defer_tiles(move || {
            if let Some(h) = super::host() {
                let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
                let size = D3D12_TILE_REGION_SIZE {
                    NumTiles: 1,
                    ..Default::default()
                };
                let flag = D3D12_TILE_RANGE_FLAG_NONE;
                let offset = 0u32;
                let count = 1u32;
                // SAFETY: live reserved resource and heap on the host queue.
                unsafe {
                    h.queue().UpdateTileMappings(
                        &target_clone,
                        1,
                        Some(&coord),
                        Some(&size),
                        Some(&heap_clone),
                        1,
                        Some(&flag),
                        Some(&offset),
                        Some(&count),
                        D3D12_TILE_MAPPING_FLAG_NONE,
                    );
                }
            }
        });
    }
    let to_dest = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: borrowed handle held alive by test scope.
                pResource: unsafe { std::mem::transmute_copy(&target) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: D3D12_RESOURCE_STATE_COPY_SOURCE,
                StateAfter: D3D12_RESOURCE_STATE_COPY_DEST,
            }),
        },
    };
    let to_src = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: borrowed handle held alive by test scope.
                pResource: unsafe { std::mem::transmute_copy(&target) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
            }),
        },
    };
    // SAFETY: open command list; valid transitions on live resource.
    unsafe {
        list2.list.ResourceBarrier(&[to_dest]);
        copy_buf_to_tex(&list2, &up_fresh, &target);
        list2.list.ResourceBarrier(&[to_src]);
        copy_tex_to_buf(&list2, &target, &rb_after);
    }
    list2.close();

    host.execute(&[&list1, &list2]);

    let read_first = |rb: &ID3D12Resource| -> u8 {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: completed readback buffer.
        unsafe {
            rb.Map(0, None, Some(&mut ptr)).unwrap();
            let b = *ptr.cast::<u8>();
            rb.Unmap(0, None);
            b
        }
    };
    assert_eq!(
        read_first(&rb_before),
        0x00,
        "list1 must execute before list2's UpdateTileMappings binds recycled 0xa5 VRAM"
    );
    assert_eq!(read_first(&rb_after), 0x3c);
    map_tile(host.queue(), &target, None);
}
