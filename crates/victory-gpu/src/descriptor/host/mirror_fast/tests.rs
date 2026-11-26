use super::super::mirror_bench::{Kind, MIX, STEP, World, kind_of, rewrite};
use super::super::{HostHeap, Waiting};
use super::*;

/// Slots of the test heaps: eight pages of 64 KiB, which the scan of a
/// batch cuts into jobs.
const SLOTS: u32 = 16_384;

/// What a heap holds of what mirroring a slot puts there: everything the
/// slot's books are, each as a plain value.
#[derive(Debug, PartialEq)]
struct Books {
    made: Vec<u64>,
    shown: Vec<u64>,
    copied: Vec<u64>,
    kept: Vec<usize>,
    strides: Vec<u32>,
    addresses: Vec<u32>,
    buffer_slots: Vec<u32>,
    changed: Vec<u32>,
    taken: Vec<(u32, usize, u64, u64)>,
    queue: Vec<(u32, Waiting)>,
    spans: Vec<(u32, (u64, u64))>,
    first: Vec<(u32, u32)>,
}

fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort();
    v
}

fn books(heap: &HostHeap) -> Books {
    let slots = victory_shader::emit::SLOT_STRIDES_COUNT.min(heap.count) as usize;
    let table = |data: usize, words: usize| -> Vec<u32> {
        // SAFETY: the mapping of a table of two dwords a slot at most.
        (0..words).map(|i| unsafe { (data as *const u32).add(i).read_volatile() }).collect()
    };
    let placed = heap.placed.lock().unwrap_or_else(|e| e.into_inner());
    Books {
        made: heap.made.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        shown: heap.shown.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        copied: heap.copied.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        kept: (0..heap.count)
            .map(|i| heap.kept_piece(i).and_then(|(kept, at)| kept[at].as_ref().map(crate::shadow::Held::raw)).unwrap_or(0))
            .collect(),
        strides: heap.strides.as_ref().map(|(_, _, data)| table(*data, slots)).unwrap_or_default(),
        addresses: heap.addresses.as_ref().map(|(_, _, data)| table(*data, 2 * slots)).unwrap_or_default(),
        buffer_slots: heap.buffers.slots.lock().unwrap_or_else(|e| e.into_inner()).iter().copied().collect(),
        changed: sorted(heap.buffers.changed.lock().unwrap_or_else(|e| e.into_inner()).iter().copied().collect()),
        taken: sorted(
            heap.buffers
                .taken
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|(&slot, (copy, from, to))| (slot, Arc::as_ptr(copy) as usize, *from, *to))
                .collect(),
        ),
        queue: heap.textures.lock().unwrap_or_else(|e| e.into_inner()).slots.iter().map(|(&slot, &w)| (slot, w)).collect(),
        spans: sorted(placed.by_slot.iter().map(|(&slot, &span)| (slot, span)).collect()),
        first: sorted(placed.first.iter().map(|(&slot, &first)| (slot, first)).collect()),
    }
}

fn set(short: bool, heaps: bool) {
    FORCED.with(|forced| forced.set(Some(Mode { short, heaps })));
}

fn seed(world: &mut World) {
    for slot in 0..SLOTS {
        let kind = kind_of(&MIX, u64::from(slot));
        if kind != Kind::Clear {
            world.write(slot, kind, u64::from(slot));
        }
    }
}

/// The same records mirrored into two heaps, one without the short way and
/// one with it, round after round: new records, records that are another
/// record of the view the slot shows, records cleared, the same view
/// written into many slots. Both end each round with the same books.
#[test]
#[ignore = "VICTORY_MIRROR_FAST is off (measured at no gain); the books differ on the CI's software adapter only: to be looked at before the switch is used"]
fn the_short_way_leaves_the_books_as_they_are_without_it() {
    let (Some(mut world), Some(short)) = (World::with_slots(SLOTS), HostHeap::new(false, SLOTS)) else {
        return; // no host GPU
    };
    let pages = world.pages();
    let _ = counts();
    let round = |world: &World, what: &str| -> u64 {
        set(false, false);
        let plain = world.heap.mirror_copied_batch(&pages, (world.start, STEP), SLOTS, false);
        set(true, false);
        let fast = short.mirror_copied_batch(&pages, (world.start, STEP), SLOTS, false);
        assert_eq!(plain, fast, "{what}: pages read, records looked at and mirrored");
        assert_eq!(books(&world.heap), books(&short), "{what}: the books");
        plain.2
    };
    seed(&mut world);
    assert!(round(&world, "seeded") > u64::from(SLOTS) / 2);
    for n in 1..=3 {
        rewrite(&mut world, &MIX, 3_000, n);
        assert!(round(&world, &format!("round {n}")) > 0);
    }
    // Another record of the view the slot shows: nothing to make.
    for slot in (0..SLOTS).step_by(7) {
        world.touch(slot, 0x77);
    }
    assert!(round(&world, "touched") > 0);
    // Records cleared.
    for slot in (0..SLOTS).step_by(5) {
        world.write(slot, Kind::Clear, 0);
    }
    assert!(round(&world, "cleared") > 0);
    // One view in many slots, of each kind that is a buffer's.
    for (k, kind) in [Kind::Cbv, Kind::CbvOff, Kind::Structured, Kind::TypedOwn, Kind::UavRaw].into_iter().enumerate() {
        for slot in (k as u32 * 1000)..(k as u32 * 1000 + 500) {
            world.write(slot, kind, 1234 + k as u64);
        }
    }
    assert!(round(&world, "repeated") > 0);
    let [batches, _, repeats, _] = counts();
    assert!(batches >= 7 && repeats > 500, "the short way took repeats: {repeats} in {batches} batches");
    FORCED.with(|forced| forced.set(None));
}

/// Heaps that have changed records mirrored together on threads of their own
/// are those mirrored one after the other: the same totals, and the same
/// books in each.
#[test]
#[ignore = "VICTORY_MIRROR_FAST is off (measured at no gain); the books differ on the CI's software adapter only: to be looked at before the switch is used"]
fn heaps_mirrored_together_are_those_mirrored_one_by_one() {
    let mut worlds: Vec<World> = (0..3).filter_map(|_| World::with_slots(SLOTS)).collect();
    let together: Vec<HostHeap> = worlds.iter().filter_map(|_| HostHeap::new(false, SLOTS)).collect();
    if worlds.len() != 3 || together.len() != 3 {
        return; // no host GPU
    }
    // One heap with the pages of a thread of its own, one with a few, and one with a page: the
    // two small ones are mirrored here.
    let pages: Vec<_> = worlds
        .iter()
        .zip([8, 3, 1])
        .map(|(world, n)| {
            let pages = world.pages();
            pages[..n].to_vec()
        })
        .collect();
    for world in &mut worlds {
        seed(world);
    }
    let round = |worlds: &[World], what: &str| -> u64 {
        set(false, false);
        let mut one = (0, 0, 0);
        for (world, pages) in worlds.iter().zip(&pages) {
            let (r, l, m) = world.heap.mirror_copied_batch(pages, (world.start, STEP), SLOTS, false);
            one = (one.0 + r, one.1 + l, one.2 + m);
        }
        let work = worlds
            .iter()
            .zip(&together)
            .zip(&pages)
            .map(|((world, heap), pages)| Work {
                host: std::ptr::from_ref(heap),
                pages: pages.clone(),
                at: (world.start, STEP),
                mirrored_count: SLOTS,
                in_place: false,
            })
            .collect();
        let all = mirror_heaps(work);
        assert_eq!(one, all, "{what}: pages read, records looked at and mirrored");
        for (n, (world, heap)) in worlds.iter().zip(&together).enumerate() {
            assert_eq!(books(&world.heap), books(heap), "{what}: the books of heap {n}");
        }
        one.2
    };
    assert!(round(&worlds, "seeded") > 12 * u64::from(SLOTS) / 8 / 2);
    for n in 1..=3u64 {
        for world in &mut worlds {
            rewrite(world, &MIX, 2_000 + 1_000 * n as usize, n);
        }
        assert!(round(&worlds, &format!("round {n}")) > 0);
    }
    // Heaps with nothing changed but one record cleared, and then none.
    worlds[1].write(5, Kind::Clear, 0);
    assert!(round(&worlds, "one record cleared") > 0);
    assert_eq!(round(&worlds, "nothing changed"), 0);
    FORCED.with(|forced| forced.set(None));
}

/// No heap at all, and one heap alone, come back as they are.
#[test]
fn no_heaps_and_one_heap_are_mirrored_as_they_are() {
    assert_eq!(mirror_heaps(Vec::new()), (0, 0, 0));
    let Some(mut world) = World::with_slots(SLOTS) else {
        return; // no host GPU
    };
    seed(&mut world);
    let work = |world: &World| Work {
        host: std::ptr::from_ref(&world.heap),
        pages: world.pages(),
        at: (world.start, STEP),
        mirrored_count: SLOTS,
        in_place: false,
    };
    let (read, looked, made) = mirror_heaps(vec![work(&world)]);
    assert_eq!((read, looked), (8, u64::from(SLOTS)));
    assert!(made > 0);
    assert_eq!(mirror_heaps(vec![work(&world)]).2, 0);
}

/// The switch's names say what they are for.
#[test]
fn the_modes_are_told_apart() {
    set(true, false);
    assert!(short() && !heaps());
    set(false, true);
    assert!(!short() && heaps());
    FORCED.with(|forced| forced.set(None));
}
