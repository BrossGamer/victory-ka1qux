//! A probe, run by hand (`cargo test -p victory-gpu gpu_upload_probe --
//! --ignored --nocapture`, with no game running): whether the card can
//! take buffers in a GPU upload heap (the card's own memory, written by the
//! CPU through the resizable BAR), and what that is worth next to the heaps
//! the copies use now. Prints, for the first adapter:
//!
//! * the answer to `GPUUploadHeapSupported` (feature options 16);
//! * what a buffer in each kind of heap costs the process and the PC: the
//!   process's private bytes, the system's commit (the card's own memory
//!   counts in it here), and the card's local and shared memory in use;
//! * the CPU's write speed into each (one pass over the buffer, the first
//!   one and the best of five; and 64 KiB blocks in shuffled order, as a
//!   title's scattered pages come) and the CPU's read speed from each;
//! * the GPU's speed reading each, as a copy into a buffer of the card's
//!   (default heap) and as a copy from one, and the whole of the path
//!   the copies take today (the CPU writes an upload buffer, the GPU copies
//!   it into a default one) against writing the GPU upload buffer alone.
//!
//! `VICTORY_PROBE_MB` is the size of a buffer (256), `VICTORY_PROBE_PHASE`
//! is `commit`, `speed` or `all` (the default). Asserts nothing.

use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

fn size() -> u64 {
    std::env::var("VICTORY_PROBE_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(256) << 20
}

fn buffer(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

/// `[process private bytes, the system's commit, the card's local memory in
/// use by the process, its shared memory in use]`.
fn measure(adapter: &IDXGIAdapter3) -> [i64; 4] {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS { cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32, ..Default::default() };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    let mut info = PERFORMANCE_INFORMATION { cb: size_of::<PERFORMANCE_INFORMATION>() as u32, ..Default::default() };
    // SAFETY: a writable structure of the size given.
    unsafe { K32GetPerformanceInfo(&mut info, info.cb) };
    let usage = |group| {
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: a plain query into our structure.
        unsafe { adapter.QueryVideoMemoryInfo(0, group, &mut own) }.map_or(0, |()| own.CurrentUsage as i64)
    };
    [
        c.PagefileUsage as i64,
        (info.CommitTotal * info.PageSize) as i64,
        usage(DXGI_MEMORY_SEGMENT_GROUP_LOCAL),
        usage(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL),
    ]
}

fn mb(v: i64) -> i64 {
    v / (1 << 20)
}

fn delta(after: [i64; 4], before: [i64; 4]) -> String {
    format!(
        "private {:+} MB, system commit {:+} MB, card local {:+} MB, shared {:+} MB",
        mb(after[0] - before[0]),
        mb(after[1] - before[1]),
        mb(after[2] - before[2]),
        mb(after[3] - before[3]),
    )
}

fn gbps(bytes: u64, took: Duration) -> f64 {
    bytes as f64 / took.as_secs_f64() / 1e9
}

struct Gpu {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    value: u64,
}

impl Gpu {
    unsafe fn new(adapter: &IDXGIAdapter1) -> Option<Gpu> {
        // SAFETY: plain device creation and object creation.
        unsafe {
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
            let device = device?;
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).ok()?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).ok()?;
            list.Close().ok()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            Some(Gpu { device, queue, allocator, list, fence, value: 0 })
        }
    }

    /// Records, runs and waits; how long it took, `None` on a failure.
    unsafe fn run(&mut self, record: impl FnOnce(&ID3D12GraphicsCommandList)) -> Option<Duration> {
        // SAFETY: one list, run and waited for before the next.
        unsafe {
            self.allocator.Reset().ok()?;
            self.list.Reset(&self.allocator, None).ok()?;
            record(&self.list);
            self.list.Close().ok()?;
            let started = Instant::now();
            self.queue.ExecuteCommandLists(&[Some(self.list.cast().ok()?)]);
            self.value += 1;
            self.queue.Signal(&self.fence, self.value).ok()?;
            while self.fence.GetCompletedValue() < self.value {
                if started.elapsed().as_secs() > 30 {
                    return None;
                }
                std::thread::yield_now();
            }
            let took = started.elapsed();
            self.device.GetDeviceRemovedReason().ok()?;
            Some(took)
        }
    }

    /// A buffer of `size` bytes in a heap of `kind`; the error is the
    /// device's own.
    unsafe fn make(
        &self,
        kind: D3D12_HEAP_TYPE,
        size: u64,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> windows::core::Result<ID3D12Resource> {
        // SAFETY: plain creation.
        unsafe {
            let mut out: Option<ID3D12Resource> = None;
            self.device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &buffer(size, flags),
                state,
                None,
                &mut out,
            )?;
            out.ok_or_else(windows::core::Error::empty)
        }
    }
}

/// A buffer in a GPU upload heap, trying the states and flags the device
/// might want; says which worked.
unsafe fn make_gpu_upload(gpu: &Gpu, size: u64) -> Option<ID3D12Resource> {
    // SAFETY: plain creation.
    unsafe {
        for (what, flags, state) in [
            ("no flags, common", D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON),
            ("no flags, generic read", D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
            ("unordered access, common", D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
        ] {
            match gpu.make(D3D12_HEAP_TYPE_GPU_UPLOAD, size, flags, state) {
                Ok(buffer) => {
                    println!("  a GPU upload buffer of {} MB: made with {what}", mb(size as i64));
                    return Some(buffer);
                }
                Err(e) => println!("  a GPU upload buffer with {what}: refused: {e}"),
            }
        }
        None
    }
}

/// The CPU's write speeds into `dst` from `src`: the first pass over the
/// whole buffer, the best of five, and 64 KiB blocks in shuffled order
/// (GB/s).
unsafe fn cpu_write(dst: *mut u8, src: &[u8]) -> (f64, f64, f64) {
    let bytes = src.len() as u64;
    let (mut first, mut best) = (0.0f64, 0.0f64);
    for pass in 0..5 {
        let started = Instant::now();
        // SAFETY: `dst` is a mapped buffer of `src.len()` bytes.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        let speed = gbps(bytes, started.elapsed());
        if pass == 0 {
            first = speed;
        }
        best = best.max(speed);
    }
    const BLOCK: usize = 64 << 10;
    let blocks = src.len() / BLOCK;
    // A fixed shuffle (a multiplicative step coprime to the count).
    let step = (0..blocks).rev().find(|s| *s > 1 && gcd(*s, blocks) == 1).unwrap_or(1);
    let started = Instant::now();
    for i in 0..blocks {
        let at = i * step % blocks * BLOCK;
        // SAFETY: a block inside both the mapped buffer and the source.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr().add(at), dst.add(at), BLOCK) };
    }
    (first, best, gbps(bytes, started.elapsed()))
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The CPU's read speed from `src` into a vector, over `bytes` bytes (GB/s).
unsafe fn cpu_read(src: *const u8, bytes: usize) -> f64 {
    let mut sink = vec![0u8; bytes];
    let started = Instant::now();
    // SAFETY: `src` is a mapped buffer of at least `bytes` bytes.
    unsafe { std::ptr::copy_nonoverlapping(src, sink.as_mut_ptr(), bytes) };
    let speed = gbps(bytes as u64, started.elapsed());
    std::hint::black_box(&sink);
    speed
}

unsafe fn map(resource: &ID3D12Resource) -> Option<*mut u8> {
    let mut at: *mut std::ffi::c_void = std::ptr::null_mut();
    // SAFETY: a plain map of the whole buffer.
    unsafe { resource.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }), Some(&mut at)).ok()? };
    Some(at.cast())
}

#[test]
#[ignore = "a probe of the host device, run by hand with no game running"]
fn gpu_upload_probe() {
    // SAFETY: plain device, resource and command calls on objects made here.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory.EnumAdapters1(0).unwrap();
        let adapter3: IDXGIAdapter3 = adapter.cast().unwrap();
        let desc = adapter.GetDesc1().unwrap();
        let name = String::from_utf16_lossy(&desc.Description).trim_end_matches('\0').to_string();
        println!(
            "adapter 0: {name}, {} MB of its own memory, {} MB of system memory set aside, {} MB shared",
            desc.DedicatedVideoMemory >> 20,
            desc.DedicatedSystemMemory >> 20,
            desc.SharedSystemMemory >> 20
        );
        let Some(mut gpu) = Gpu::new(&adapter) else {
            println!("no host device");
            return;
        };
        let mut architecture = D3D12_FEATURE_DATA_ARCHITECTURE1::default();
        let asked = gpu.device.CheckFeatureSupport(
            D3D12_FEATURE_ARCHITECTURE1,
            std::ptr::from_mut(&mut architecture).cast(),
            size_of::<D3D12_FEATURE_DATA_ARCHITECTURE1>() as u32,
        );
        println!(
            "architecture: UMA {}, cache-coherent UMA {} (query {:?})",
            architecture.UMA.as_bool(),
            architecture.CacheCoherentUMA.as_bool(),
            asked.is_ok()
        );
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
        let asked = gpu.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS16,
            std::ptr::from_mut(&mut options).cast(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS16>() as u32,
        );
        let supported = asked.is_ok() && options.GPUUploadHeapSupported.as_bool();
        match &asked {
            Ok(()) => println!("GPUUploadHeapSupported: {}", options.GPUUploadHeapSupported.as_bool()),
            Err(e) => println!("GPUUploadHeapSupported: not answered ({e}): this runtime does not know feature options 16"),
        }
        let size = size();
        let phase = std::env::var("VICTORY_PROBE_PHASE").unwrap_or_else(|_| "all".into());

        if phase == "all" || phase == "commit" {
            println!("--- what a buffer of {} MB costs ---", mb(size as i64));
            let cases: [(&str, D3D12_HEAP_TYPE, D3D12_RESOURCE_STATES); 3] = [
                ("upload heap", D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ),
                ("GPU upload heap", D3D12_HEAP_TYPE_GPU_UPLOAD, D3D12_RESOURCE_STATE_COMMON),
                ("default heap", D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON),
            ];
            let only = std::env::var("VICTORY_PROBE_CASE").ok().and_then(|v| v.parse::<usize>().ok());
            for (i, (what, kind, state)) in cases.into_iter().enumerate() {
                if only.is_some_and(|o| o != i) || (kind == D3D12_HEAP_TYPE_GPU_UPLOAD && !supported) {
                    continue;
                }
                let before = measure(&adapter3);
                let made = if kind == D3D12_HEAP_TYPE_GPU_UPLOAD {
                    make_gpu_upload(&gpu, size)
                } else {
                    gpu.make(kind, size, D3D12_RESOURCE_FLAG_NONE, state).ok()
                };
                let Some(made) = made else {
                    println!("{i}: {what}: refused");
                    continue;
                };
                let mapped = map(&made);
                // Touch every page the way a first fill does.
                if let Some(at) = mapped.filter(|_| kind != D3D12_HEAP_TYPE_DEFAULT) {
                    std::ptr::write_bytes(at, 0x5a, size as usize);
                }
                let after = measure(&adapter3);
                drop(made);
                std::thread::sleep(Duration::from_millis(300));
                println!(
                    "{i}: {what}: made and filled: {}; released: {}",
                    delta(after, before),
                    delta(measure(&adapter3), before)
                );
            }
        }

        if phase == "all" || phase == "speed" {
            println!("--- speeds, buffers of {} MB ---", mb(size as i64));
            let src: Vec<u8> = (0..size as usize).map(|i| (i * 7 % 251) as u8).collect();
            let uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            let upload = gpu.make(D3D12_HEAP_TYPE_UPLOAD, size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ).ok();
            let card_upload = if supported { make_gpu_upload(&gpu, size) } else { None };
            let default_a = gpu.make(D3D12_HEAP_TYPE_DEFAULT, size, uav, D3D12_RESOURCE_STATE_COMMON).ok();
            let default_b = gpu.make(D3D12_HEAP_TYPE_DEFAULT, size, uav, D3D12_RESOURCE_STATE_COMMON).ok();
            let (Some(upload), Some(default_a), Some(default_b)) = (upload, default_a, default_b) else {
                println!("the plain buffers were not made");
                return;
            };
            let upload_at = map(&upload).unwrap();
            let before = measure(&adapter3);
            // CPU writing and reading.
            let (first, best, scattered) = cpu_write(upload_at, &src);
            println!(
                "CPU writes into the upload heap: first pass {first:.1} GB/s, best of five {best:.1}, 64 KiB blocks shuffled {scattered:.1}; reads {:.2} GB/s",
                cpu_read(upload_at, 16 << 20)
            );
            let mut card_at = None;
            if let Some(card) = &card_upload {
                let at = map(card).unwrap();
                card_at = Some(at);
                let (first, best, scattered) = cpu_write(at, &src);
                println!(
                    "CPU writes into the GPU upload heap: first pass {first:.1} GB/s, best of five {best:.1}, 64 KiB blocks shuffled {scattered:.1}; reads {:.2} GB/s",
                    cpu_read(at, 16 << 20)
                );
            }
            // The GPU reading, as a copy into the card's own memory (and
            // one inside it, for what the card can do at all).
            let rounds = 8u64;
            let total = rounds * size;
            let copy_from = |gpu: &mut Gpu, from: &ID3D12Resource, to: &ID3D12Resource| {
                gpu.run(|list| {
                    for _ in 0..rounds {
                        list.CopyBufferRegion(to, 0, from, 0, size);
                    }
                })
            };
            // One warm-up copy each, so that first-use costs are not timed.
            let _ = copy_from(&mut gpu, &upload, &default_a);
            let _ = copy_from(&mut gpu, &default_b, &default_a);
            let took_upload = copy_from(&mut gpu, &upload, &default_a);
            let took_default = copy_from(&mut gpu, &default_b, &default_a);
            println!(
                "GPU copies {} MB eight times into a default buffer: from the upload heap {} GB/s; from a default buffer {} GB/s",
                mb(size as i64),
                took_upload.map_or("failed".into(), |t| format!("{:.1}", gbps(total, t))),
                took_default.map_or("failed".into(), |t| format!("{:.1}", gbps(total, t))),
            );
            if let Some(card) = &card_upload {
                let _ = copy_from(&mut gpu, card, &default_a);
                let took_card = copy_from(&mut gpu, card, &default_a);
                let took_back = gpu.run(|list| {
                    for _ in 0..rounds {
                        list.CopyBufferRegion(card, 0, &default_b, 0, size);
                    }
                });
                println!(
                    "GPU copies from the GPU upload heap into a default buffer {} GB/s; from a default buffer into the GPU upload heap {} GB/s",
                    took_card.map_or("failed".into(), |t| format!("{:.1}", gbps(total, t))),
                    took_back.map_or("failed".into(), |t| format!("{:.1}", gbps(total, t))),
                );
                if let Some(at) = card_at {
                    println!(
                        "CPU reads of what the GPU wrote into the GPU upload heap: {:.2} GB/s (16 MB)",
                        cpu_read(at, 16 << 20)
                    );
                }
            }
            // What one refresh of `size` bytes costs the way copies go now
            // (the CPU writes an upload buffer, the GPU copies it into the
            // card's own buffer) and the way a GPU upload heap would (the
            // CPU writes the buffer the shaders read).
            let started = Instant::now();
            std::ptr::copy_nonoverlapping(src.as_ptr(), upload_at, src.len());
            let wrote = started.elapsed();
            let copied = gpu.run(|list| list.CopyBufferRegion(&default_a, 0, &upload, 0, size));
            println!(
                "one refresh of {} MB today: the CPU writes the upload buffer {:.1} ms + the GPU copies it {} ms = {:.1} ms",
                mb(size as i64),
                wrote.as_secs_f64() * 1e3,
                copied.map_or("failed".into(), |t| format!("{:.1}", t.as_secs_f64() * 1e3)),
                (wrote + copied.unwrap_or_default()).as_secs_f64() * 1e3,
            );
            if let Some(at) = card_at {
                let started = Instant::now();
                std::ptr::copy_nonoverlapping(src.as_ptr(), at, src.len());
                println!(
                    "one refresh with a GPU upload heap: the CPU writes the buffer the shaders read {:.1} ms",
                    started.elapsed().as_secs_f64() * 1e3
                );
            }
            println!(
                "memory held by the buffers: {}",
                delta(measure(&adapter3), before)
            );
        }
    }
}
