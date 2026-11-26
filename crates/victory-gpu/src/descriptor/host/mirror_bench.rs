//! What the mirroring of the slots a frame's records changed costs, by part,
//! and what [`HostHeap::mirror_copied_batch`] makes of them with
//! `VICTORY_MIRROR_FAST` off and on (the second half of this file's tests:
//! the same slots must end up showing the same views).
//!
//! `cargo test --release -p victory-gpu --lib mirror_bench -- --ignored --nocapture --test-threads=1`
//!
//! A frame of the open world had, in the log of a run, some 27,700 records
//! mirrored a frame (on 190 changed pages of 64 KiB), of which some 18,500
//! were views of buffers (constant buffer views, structured and typed views
//! of buffers, 11,600 asks a frame for a copy of a view's own bytes) and a
//! few thousand placed texture views; each cost 2.4 microseconds on the
//! thread that submits. The records here are written through the title's own
//! entry points (`create_cbv`, `create_placed_srv`, `create_placed_uav`), in
//! runs of a table's length, over more than a thousand title buffers.

use std::time::Instant;

use super::*;
use crate::descriptor::{create_cbv, create_placed_srv, create_placed_uav, create_srv};

pub(super) const STEP: u64 = 32;
/// Slots of the benchmark's heap (64 pages of 64 KiB).
pub(super) const SLOTS: u32 = 131_072;
const PAGE_BYTES: usize = 64 << 10;
/// Title buffers registered, each of [`BUFFER_LEN`] bytes on a 64 KiB stride.
const BUFFERS: usize = 1536;
const BUFFER_LEN: u64 = 60 << 10;
/// Of a hundred asks for a copy of a view's own bytes, those that are for one not made before (a
/// frame of the log: 785 made of 11,600 asked).
const FRESH: u64 = 7;
const RESOURCE: usize = 0x6a_0000;
/// Texture objects the title made, for [`Kind::TextureObject`].
const OBJECTS: u64 = 400;

/// Texture objects the title made (small ones placed at distinct addresses),
/// each with its host copy: made once for the process.
fn texture_objects() -> &'static [usize] {
    static POOL: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let resource = [3, 0, 0, 0, 64, 0, 64, 7 << 16 | 1, 71, 1, 0, 0, 0, 0];
        let mut pool = Vec::new();
        for k in 0..OBJECTS {
            let mut desc = [0u32; 16];
            desc[0] = 71;
            desc[1] = 4;
            desc[2] = windows::Win32::Graphics::Direct3D12::D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            desc[5] = 7;
            let address = 0x50_0000_0000u64 + k * (1 << 20);
            desc[10] = address as u32;
            desc[11] = (address >> 32) as u32;
            if let Some(View::Shader(object, _)) = crate::descriptor::host::placed_texture(&View::Placed(false, resource, desc))
                && crate::resource::host_resource(object as crate::object::This).is_some()
            {
                pool.push(object);
            }
        }
        pool
    })
}

/// What a record is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// A constant buffer view on 256 bytes of its buffer's copy.
    Cbv,
    /// A constant buffer view off 256 bytes of the copy: its own bytes.
    CbvOff,
    /// A structured view that becomes a raw view of the copy.
    Structured,
    /// A typed view off its copy's elements: a copy of its own bytes.
    TypedOwn,
    /// A raw unordered access view.
    UavRaw,
    /// A view of a placed texture: queued, and in the spans.
    Texture,
    /// A view of a texture object that has its host copy: made at once.
    TextureObject,
    /// A record cleared (a slot that was made, made null).
    Clear,
}

/// The title's memory and its heap.
pub(super) struct World {
    pub(super) heap: HostHeap,
    pub(super) slots: u32,
    pub(super) memory: Vec<[u32; 8]>,
    pub(super) start: u64,
    arena: Vec<u8>,
    base: u64,
    seed: u64,
}

impl World {
    pub(super) fn new() -> Option<World> {
        World::with_slots(SLOTS)
    }

    /// A world whose heap has `slots` slots.
    pub(super) fn with_slots(slots: u32) -> Option<World> {
        let Some(heap) = HostHeap::new(false, slots) else {
            eprintln!("no host heap (device removed: {:?})", device_removed());
            return None;
        };
        let memory = vec![[0u32; 8]; slots as usize];
        let start = memory.as_ptr() as u64;
        let arena = vec![0u8; (BUFFERS + 1) * (64 << 10)];
        let base = (arena.as_ptr() as u64).next_multiple_of(64 << 10);
        for b in 0..BUFFERS {
            crate::shadow::register(RESOURCE + b, base + b as u64 * (64 << 10), BUFFER_LEN);
        }
        Some(World { heap, slots, memory, start, arena, base, seed: 0x9e37_79b9_7f4a_7c15 })
    }

    fn random(&mut self) -> u64 {
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 7;
        self.seed ^= self.seed << 17;
        self.seed
    }

    pub(super) fn pages(&self) -> Vec<(usize, usize)> {
        let (from, len) = (self.start as usize, self.memory.len() * 32);
        (0..len.div_ceil(PAGE_BYTES)).map(|n| (from + n * PAGE_BYTES, from + ((n + 1) * PAGE_BYTES).min(len))).collect()
    }

    /// Writes slot `slot`'s record as `kind`; `n` moves it (another buffer,
    /// another place in it).
    pub(super) fn write(&mut self, slot: u32, kind: Kind, n: u64) {
        let at = self.start + u64::from(slot) * STEP;
        // Where in which buffer: the views that need a copy of their own bytes
        // come back to a few places, some new ones among them.
        let h = n.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let (b, place) = if (h >> 40) % 100 < FRESH {
            ((h >> 20) % BUFFERS as u64, 24 + (h >> 8) % 200)
        } else {
            ((h >> 20) % 48, (h >> 8) % 24)
        };
        let buffer = self.base + b * (64 << 10);
        match kind {
            Kind::Cbv => {
                let address = buffer + place * 256;
                create_cbv(std::ptr::null(), &[address as u32, (address >> 32) as u32, 256, 0], at);
            }
            Kind::CbvOff => {
                // Where the copy of the buffer has no 256-byte line: the
                // view is on 16 bytes only.
                let address = buffer + place * 256 + 16 * (1 + (h >> 4) % 5);
                create_cbv(std::ptr::null(), &[address as u32, (address >> 32) as u32, 192, 0], at);
            }
            Kind::Structured | Kind::TypedOwn | Kind::UavRaw => {
                let mut resource = [0u32; 14];
                resource[0] = 1; // a buffer
                resource[4] = BUFFER_LEN as u32;
                let mut desc = [0u32; 16];
                desc[1] = 1; // a buffer view
                if kind != Kind::UavRaw {
                    desc[2] = windows::Win32::Graphics::Direct3D12::D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                }
                // Where the view's memory starts: a buffer's first byte, or
                // 4 bytes in (off a typed view's elements).
                let address = buffer + if kind == Kind::TypedOwn { 4 } else { 0 };
                desc[10] = address as u32;
                desc[11] = (address >> 32) as u32;
                match kind {
                    Kind::Structured => {
                        desc[4] = (place * 4) as u32;
                        desc[6] = 16;
                        desc[7] = 64;
                    }
                    Kind::TypedOwn => {
                        desc[0] = 6; // R32G32B32_FLOAT
                        desc[4] = (place * 8) as u32;
                        desc[6] = 24;
                    }
                    _ => {
                        // An unordered access view's own layout: first element
                        // (two words), elements, stride, counter (two), flags.
                        desc[0] = 0x27; // R32_TYPELESS
                        desc[2] = (place * 16) as u32;
                        desc[4] = 64;
                        desc[8] = 0x1; // raw
                    }
                }
                if kind == Kind::UavRaw {
                    create_placed_uav(std::ptr::null(), &resource, &desc, at);
                } else {
                    create_placed_srv(std::ptr::null(), &resource, &desc, at);
                }
            }
            Kind::Texture => {
                let mut resource = [0u32; 14];
                resource[0] = 3; // a 2D texture
                resource[4] = 1024;
                resource[6] = 1024;
                resource[7] = 11 << 16 | 1;
                resource[8] = 71; // BC1_UNORM
                resource[9] = 1;
                let mut desc = [0u32; 16];
                desc[0] = 71;
                desc[1] = 4; // a 2D view
                desc[2] = windows::Win32::Graphics::Direct3D12::D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                desc[5] = 11;
                let address = 0x40_0000_0000u64 + (n % 3000) * (4 << 20);
                desc[10] = address as u32;
                desc[11] = (address >> 32) as u32;
                create_placed_srv(std::ptr::null(), &resource, &desc, at);
            }
            Kind::TextureObject => {
                let pool = texture_objects();
                let object = pool[(n % pool.len() as u64) as usize];
                let mut desc = [0u32; 10];
                desc[0] = 71;
                desc[1] = 4;
                desc[2] = windows::Win32::Graphics::Direct3D12::D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                desc[5] = 7;
                create_srv(std::ptr::null(), object as crate::object::This, &desc, at);
            }
            Kind::Clear => self.memory[slot as usize] = [0; 8],
        }
    }

    /// Slot `slot`'s record changed in a word that is no part of the name
    /// of its view: another record, the same view.
    pub(super) fn touch(&mut self, slot: u32, n: u32) {
        self.memory[slot as usize][5] = n;
    }

    /// The slots of runs of a table's length (8 to 71), `count` of them in
    /// all or a little less, at places the title's ring would write.
    pub(super) fn runs(&mut self, count: usize) -> Vec<u32> {
        let count = count.min(self.slots as usize / 2);
        let mut slots = Vec::with_capacity(count);
        let mut taken = std::collections::HashSet::new();
        while slots.len() < count {
            let len = 8 + (self.random() % 64) as u32;
            let first = (self.random() % u64::from(self.slots - len)) as u32;
            for slot in first..first + len {
                if taken.insert(slot) {
                    slots.push(slot);
                }
            }
        }
        slots.sort_unstable();
        slots
    }
}

impl Drop for World {
    fn drop(&mut self) {
        for b in 0..BUFFERS {
            crate::shadow::unregister(RESOURCE + b, self.base + b as u64 * (64 << 10));
        }
        let _ = &self.arena;
    }
}

/// Threads that ask for the host copies of the world's buffers as the
/// title's recording threads do (some 540,000 asks a frame of 230 ms in the
/// log, 2.4 million a second), until the value is dropped: what the
/// mirroring of a frame meets in the game and not alone.
pub(super) struct Recorders {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    threads: Vec<std::thread::JoinHandle<u64>>,
}

impl Recorders {
    pub(super) fn begin(threads: usize, base: u64) -> Recorders {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let threads = (0..threads)
            .map(|n| {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let mut seed = 0x2545_f491_4f6c_dd1du64 ^ (n as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                    let mut asks = 0u64;
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        // The same few hundred buffers every thread asks for.
                        let b = (seed >> 33) % 400;
                        let address = base + b * (64 << 10) + (seed >> 12) % 4096 * 8;
                        std::hint::black_box(crate::shadow::find(address, 64));
                        asks += 1;
                        for _ in 0..12 {
                            std::hint::spin_loop();
                        }
                    }
                    asks
                })
            })
            .collect();
        Recorders { stop, threads }
    }

    /// The asks made so far.
    pub(super) fn end(self) -> u64 {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        self.threads.into_iter().map(|t| t.join().unwrap_or(0)).sum()
    }
}

/// The time stamp counter's ticks in a nanosecond.
fn ticks_per_ns() -> f64 {
    let (began, at) = (Instant::now(), prof::now_ticks());
    while began.elapsed().as_millis() < 60 {
        std::hint::spin_loop();
    }
    (prof::now_ticks() - at) as f64 / began.elapsed().as_nanos() as f64
}

/// What the mix of a frame of the open world is taken to be, by kind: the
/// log's 18,500 buffer views and a few thousand textures of 27,700 records.
pub(super) const MIX: [(Kind, u32); 8] = [
    (Kind::Cbv, 22),
    (Kind::CbvOff, 10),
    (Kind::Structured, 17),
    (Kind::TypedOwn, 13),
    (Kind::UavRaw, 4),
    (Kind::Texture, 14),
    (Kind::TextureObject, 18),
    (Kind::Clear, 2),
];

/// Kind of the `k`th slot of a mix.
pub(super) fn kind_of(mix: &[(Kind, u32)], k: u64) -> Kind {
    let total: u32 = mix.iter().map(|&(_, w)| w).sum();
    let mut at = (k.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 33) as u32 % total;
    for &(kind, weight) in mix {
        if at < weight {
            return kind;
        }
        at -= weight;
    }
    mix[0].0
}

/// Rewrites `changed` slots by `mix`, in `round`'s places.
pub(super) fn rewrite(world: &mut World, mix: &[(Kind, u32)], changed: usize, round: u64) -> usize {
    let slots = world.runs(changed);
    for (k, &slot) in slots.iter().enumerate() {
        let n = round * 100_003 + k as u64 * 7 + u64::from(slot);
        let kind = kind_of(mix, k as u64 + round);
        world.write(slot, kind, n);
    }
    slots.len()
}

/// One configuration's rounds: the median milliseconds of a sync, and the
/// parts' share of the mirroring on this thread.
fn measure(mix: &[(Kind, u32)], changed: usize, rounds: u64, label: &str) -> f64 {
    measure_as(mix, changed, rounds, label, false, true, 0)
}

/// [`measure`] with the switch [`mirror_fast::short`] as `fast` says, and the
/// table of parts only if `parts`; the median microseconds a mirrored slot.
fn measure_as(
    mix: &[(Kind, u32)],
    changed: usize,
    rounds: u64,
    label: &str,
    fast: bool,
    parts: bool,
    recorders: usize,
) -> f64 {
    mirror_fast::FORCED.with(|forced| forced.set(Some(mirror_fast::Mode { short: fast, heaps: false })));
    let label = &format!("{label} (fast {}{})", if fast { "on" } else { "off" }, if recorders > 0 { ", recorders" } else { "" });
    let Some(mut world) = World::new() else {
        eprintln!("no host GPU: skipped");
        return 0.0;
    };
    let per_ns = ticks_per_ns();
    let pages = world.pages();
    // Everything a first time: every slot of the heap holds a record of the mix
    // (so that a later write replaces a view, as a ring's slots are).
    let all: Vec<u32> = (0..SLOTS).collect();
    for (k, &slot) in all.iter().enumerate() {
        let kind = kind_of(mix, k as u64);
        if kind != Kind::Clear {
            world.write(slot, kind, k as u64);
        }
    }
    let (read, looked, made) = world.heap.mirror_copied_batch(&pages, (world.start, STEP), SLOTS, false);
    eprintln!("{label}: seeded {looked} records on {read} pages, {made} mirrored");
    let _ = mirror_fast::counts();
    let threads = (recorders > 0).then(|| Recorders::begin(recorders, world.base));
    let mut times = Vec::new();
    let mut ticks = [0u64; prof::PARTS];
    let mut calls = [0u64; prof::PARTS];
    let mut mirrored = 0u64;
    for round in 1..=rounds {
        let n = rewrite(&mut world, mix, changed, round);
        prof::start();
        let at = Instant::now();
        let (_, _, made) = world.heap.mirror_copied_batch(&pages, (world.start, STEP), SLOTS, false);
        let took = at.elapsed();
        let (t, c) = prof::stop();
        // The first round warms what the first look of each buffer makes.
        if round == 1 {
            continue;
        }
        times.push(took.as_secs_f64() * 1e3);
        mirrored += made;
        for i in 0..prof::PARTS {
            ticks[i] += t[i];
            calls[i] += c[i];
        }
        assert!(made > 0 && n > 0);
    }
    if let Some(threads) = threads {
        let asks = threads.end();
        eprintln!("{label}: {recorders} recording threads asked {asks} times for copies meanwhile");
    }
    times.sort_by(f64::total_cmp);
    let median = times[times.len() / 2];
    let [batches, slots, repeats, micros] = mirror_fast::counts();
    if batches > 0 {
        eprintln!(
            "{label}: per sync {:.0} slots in the batch, {:.0} took what another slot of it had found, {:.1} ms in the batch",
            slots as f64 / batches as f64,
            repeats as f64 / batches as f64,
            micros as f64 / 1e3 / batches as f64
        );
    }
    let per_slot = mirrored as f64 / times.len() as f64;
    eprintln!(
        "{label}: {changed} slots rewritten a round, {per_slot:.0} mirrored: {median:.2} ms a sync (best {:.2}, worst {:.2}), {:.2} us a slot",
        times[0],
        times[times.len() - 1],
        median * 1e3 / per_slot
    );
    let total: u64 = ticks.iter().sum();
    let per = |n: u64| n as f64 / per_ns / 1e3 / mirrored as f64;
    if !parts {
        return median * 1e3 / per_slot;
    }
    for i in 0..prof::PARTS {
        if ticks[i] == 0 && calls[i] == 0 {
            continue;
        }
        eprintln!(
            "    {:<36} {:>7.3} us a slot  {:>5.1}%  ({:.2} calls a slot)",
            prof::NAMES[i],
            per(ticks[i]),
            ticks[i] as f64 * 100.0 / total as f64,
            calls[i] as f64 / mirrored as f64
        );
    }
    eprintln!("    {:<36} {:>7.3} us a slot", "all parts", per(total));
    median * 1e3 / per_slot
}

/// The switches of the game runs this benchmark follows (the log of run dA):
/// copies of views' own bytes in a queue of idle ones and on slabs. Read once
/// by the process, so before anything asks.
fn game_settings() {
    // SAFETY: before any other thread of this test reads the environment.
    unsafe {
        std::env::set_var("VICTORY_EXACT_FAST", "1");
        std::env::set_var("VICTORY_EXACT_SUB", "1");
    }
}

/// The same frames mirrored without and with the looking done ahead
/// (`VICTORY_MIRROR_FAST`), the microseconds a mirrored slot, by kind of
/// record and for the mix, at several sizes of frame.
/// `cargo test --release -p victory-gpu --lib mirror_bench::fast -- --ignored --nocapture --test-threads=1`
#[test]
#[ignore = "benchmark: run in release with --ignored --nocapture"]
fn fast() {
    game_settings();
    let mut rows = Vec::new();
    for kind in [Kind::Cbv, Kind::CbvOff, Kind::Structured, Kind::TypedOwn, Kind::UavRaw, Kind::Texture, Kind::TextureObject] {
        let mix = [(kind, 1)];
        let (off, on) = (
            measure_as(&mix, 12_000, 7, &format!("{kind:?}"), false, false, 0),
            measure_as(&mix, 12_000, 7, &format!("{kind:?}"), true, false, 0),
        );
        rows.push((format!("{kind:?}, 12,000"), off, on));
    }
    for changed in [1_000, 6_000, 27_700] {
        let (off, on) = (
            measure_as(&MIX, changed, 9, "the mix", false, changed == 27_700, 0),
            measure_as(&MIX, changed, 9, "the mix", true, changed == 27_700, 0),
        );
        rows.push((format!("the mix, {changed}"), off, on));
    }
    for recorders in [3, 6] {
        let (off, on) = (
            measure_as(&MIX, 27_700, 9, "the mix", false, false, recorders),
            measure_as(&MIX, 27_700, 9, "the mix", true, false, recorders),
        );
        rows.push((format!("the mix, 27700, {recorders} recorders"), off, on));
    }
    eprintln!("microseconds a mirrored slot, switch off and on:");
    for (what, off, on) in rows {
        eprintln!("  {what:<22} {off:>6.2} -> {on:>6.2}  ({:+.0}%)", (on / off - 1.0) * 100.0);
    }
}

/// Heaps that have changed records, mirrored one after the other (as
/// `sync_heaps` does) and together ([`mirror_fast::mirror_heaps`]): the
/// milliseconds a frame, and what they come to when recording threads ask
/// for copies meanwhile. `changed` is the records rewritten in each heap.
/// The mirrored heaps are the title's, and each is mirrored into a heap of
/// each kind from the same memory, one after the other in time.
fn heaps_round(mix: &[(Kind, u32)], changed: &[usize], recorders: usize, short: bool, rounds: u64) -> (f64, f64, f64) {
    let Some(first) = World::new() else {
        return (0.0, 0.0, 0.0);
    };
    let mut worlds = vec![first];
    for _ in 1..changed.len() {
        worlds.extend(World::new());
    }
    let together: Vec<HostHeap> = worlds.iter().filter_map(|_| HostHeap::new(false, SLOTS)).collect();
    assert_eq!(together.len(), worlds.len());
    let pages: Vec<Vec<(usize, usize)>> = worlds.iter().map(World::pages).collect();
    // Every slot of every heap a record first.
    for world in &mut worlds {
        for slot in 0..SLOTS {
            let kind = kind_of(mix, u64::from(slot));
            if kind != Kind::Clear {
                world.write(slot, kind, u64::from(slot));
            }
        }
    }
    let one_by_one_into = |worlds: &[World], heaps: &[HostHeap], pages: &[Vec<(usize, usize)>]| {
        for ((world, heap), pages) in worlds.iter().zip(heaps).zip(pages) {
            heap.mirror_copied_batch(pages, (world.start, STEP), SLOTS, false);
        }
    };
    let works = |worlds: &[World], heaps: &[HostHeap]| -> Vec<mirror_fast::Work> {
        worlds
            .iter()
            .zip(heaps)
            .zip(&pages)
            .map(|((world, heap), pages)| mirror_fast::Work {
                host: std::ptr::from_ref(heap),
                pages: pages.clone(),
                at: (world.start, STEP),
                mirrored_count: SLOTS,
                in_place: false,
            })
            .collect()
    };
    let one_by_one = |worlds: &[World]| -> (u64, u64, u64) {
        let mut all = (0, 0, 0);
        for (world, pages) in worlds.iter().zip(&pages) {
            let (r, l, m) = world.heap.mirror_copied_batch(pages, (world.start, STEP), SLOTS, false);
            all = (all.0 + r, all.1 + l, all.2 + m);
        }
        all
    };
    let set = |short, heaps| mirror_fast::FORCED.with(|forced| forced.set(Some(mirror_fast::Mode { short, heaps })));
    set(false, false);
    let seeded = one_by_one(&worlds);
    set(short, true);
    let (r, l, m) = mirror_fast::mirror_heaps(works(&worlds, &together));
    assert_eq!((r, l, m), seeded, "the seeding of both kinds");
    let threads = (recorders > 0).then(|| Recorders::begin(recorders, worlds[0].base));
    let apart: Vec<HostHeap> = worlds.iter().filter_map(|_| HostHeap::new(false, SLOTS)).collect();
    set(false, false);
    one_by_one_into(&worlds, &apart, &pages);
    let (mut serial, mut parallel, mut scoped) = (Vec::new(), Vec::new(), Vec::new());
    for round in 1..=rounds {
        for (world, &n) in worlds.iter_mut().zip(changed) {
            rewrite(world, mix, n, round);
        }
        set(false, false);
        let at = Instant::now();
        let one = one_by_one(&worlds);
        serial.push(at.elapsed().as_secs_f64() * 1e3);
        set(short, true);
        let at = Instant::now();
        let all = mirror_fast::mirror_heaps(works(&worlds, &together));
        parallel.push(at.elapsed().as_secs_f64() * 1e3);
        assert_eq!(one, all, "the same pages, records and mirrorings");
        // One plain thread to a heap.
        set(short, false);
        let at = Instant::now();
        let by_threads = std::thread::scope(|scope| {
            let jobs: Vec<_> = worlds
                .iter()
                .zip(&apart)
                .zip(&pages)
                .map(|((world, heap), pages)| scope.spawn(move || heap.mirror_copied_batch(pages, (world.start, STEP), SLOTS, false)))
                .collect();
            jobs.into_iter().map(|job| job.join().unwrap()).fold((0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
        });
        scoped.push(at.elapsed().as_secs_f64() * 1e3);
        assert_eq!(one, by_threads);
    }
    if let Some(threads) = threads {
        threads.end();
    }
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    // (The first round warms what the first look at each buffer makes.)
    (median(serial[1..].to_vec()), median(parallel[1..].to_vec()), median(scoped[1..].to_vec()))
}

/// `cargo test --release -p victory-gpu --lib mirror_bench::heaps -- --ignored --nocapture --test-threads=1`
#[test]
#[ignore = "benchmark: run in release with --ignored --nocapture"]
fn heaps() {
    game_settings();
    eprintln!("by kind of record, 5 heaps of 5,000 records, no recorders:");
    for kind in [Kind::Cbv, Kind::CbvOff, Kind::Structured, Kind::TypedOwn, Kind::UavRaw, Kind::Texture, Kind::TextureObject] {
        let (serial, together, scoped) = heaps_round(&[(kind, 1)], &[5_000; 5], 0, false, 7);
        let (_, both, _) = heaps_round(&[(kind, 1)], &[5_000; 5], 0, true, 7);
        eprintln!(
            "  {kind:?}: {serial:6.1} ms -> {together:6.1} ms together ({:.2}x; plain threads {scoped:.1} ms, {:.2}x), {both:6.1} ms with the short way too ({:.2}x)",
            serial / together,
            serial / scoped,
            serial / both
        );
    }
    eprintln!("milliseconds a frame, the heaps one after the other and together:");
    for (what, changed) in [
        ("5 heaps of 5,500", vec![5_500; 5]),
        ("one of 16,000 and four of 2,900", vec![16_000, 2_900, 2_900, 2_900, 2_900]),
        ("2 heaps of 13,800", vec![13_800; 2]),
        ("one heap of 27,700", vec![27_700]),
    ] {
        for recorders in [0, 3] {
            let (serial, together, scoped) = heaps_round(&MIX, &changed, recorders, false, 7);
            let (_, both, scoped_both) = heaps_round(&MIX, &changed, recorders, true, 7);
            eprintln!(
                "  {what}, {recorders} recorders: {serial:6.1} ms -> {together:6.1} ms together ({:.2}x; plain threads {scoped:.1}), {both:6.1} ms with the short way too ({:.2}x; plain threads {scoped_both:.1})",
                serial / together,
                serial / both
            );
        }
    }
}

/// The cost of a mirrored slot, by part, of each kind of record alone and of
/// the mix.
/// `cargo test --release -p victory-gpu --lib mirror_bench::profile -- --ignored --nocapture --test-threads=1`
#[test]
#[ignore = "benchmark: run in release with --ignored --nocapture"]
fn profile() {
    game_settings();
    for kind in [Kind::Cbv, Kind::CbvOff, Kind::Structured, Kind::TypedOwn, Kind::UavRaw, Kind::Texture, Kind::TextureObject] {
        measure(&[(kind, 1)], 12_000, 7, &format!("{kind:?}"));
    }
    measure(&MIX, 27_700, 9, "the mix");
}
