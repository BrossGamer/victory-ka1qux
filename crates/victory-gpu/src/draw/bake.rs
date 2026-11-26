//! The textures a draw of the title's decoration bake samples, uploaded
//! before it.
//!
//! After fxceless/victory-fork PR #1264 (UwUCammie), adapted to this tree's
//! queue of textures picked by index (`descriptor/host.rs`
//! `TextureQueue`) and to the pass before a bake that is here already
//! (`State::early_textures`, `bake_source_slots`).
//!
//! The decoration bake (the pass that composites characters' skin once while
//! a scene loads) binds no texture through a table. Each draw's constants and
//! its buffer of 304-byte records hold view heap indices, so everything it
//! samples has to be on the card already through that queue. A source still
//! waiting at that moment is baked wrong for the rest of the run: black,
//! blotched or missing skin.
//!
//! What the pass before a bake did not reach, and this does:
//! - It knew four bake pixel programs by hash; the title has 322 of them. A
//!   bake program is recognised here by what it reads: a pixel shader whose
//!   only table entries are a buffer of 304-byte records and a constant buffer
//!   in the same table ([`tables`]).
//! - It read the first 16 KiB of every buffer the draw names for dwords that
//!   look like texture slots. The draw's own constants say which records it
//!   walks, and those are read exactly ([`indices_named`]), however far into their
//!   buffer they lie.
//! - It served the slots that were due: not the ones whose turn had not come
//!   or that were held back for lack of room, and not a slot the spans took
//!   for stale when the title wrote a newer view of clashing memory while it
//!   still reads the old one ([`State::bake_sources`], `TextureQueue::queued`
//!   and `requeue`).
//! - It stopped at its budget: a bake's own sources are served whatever the
//!   budget, up to `VICTORY_BAKE_SOURCES_MB` (1024) a draw, and before a draw
//!   of a program the pass did not know, by a pass of their own that does
//!   nothing of the queue's upkeep ([`State::bake_sources_pass`]).
//!
//! The card's room is as before: a source is made past the line for new host
//! textures up to the bake's own, higher one (`resource::room_for_bake`).
//!
//! On by default; `VICTORY_BAKE_SOURCES=0` only counts, and the pass before
//! the four known programs is as it was.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::{ResourceType, RootType};

use super::{BAKE_SOURCES, PASS_BUDGET, Root, State};
use crate::descriptor::View;
use crate::host::HostList;
use crate::pipeline::host::HostPipeline;

thread_local! {
    /// While [`State::bake_sources_pass`] runs the texture pass: it serves
    /// the bake draw's own sources and does nothing of the queue's upkeep,
    /// which a pass of this frame did or will do.
    pub(super) static PASS_ONLY: Cell<bool> = const { Cell::new(false) };
    /// What the passes before a bake draw did with its own sources:
    /// uploaded, their bytes, left because their memory is not mapped, and
    /// left because no more of their memory is mapped than at their last
    /// upload.
    pub(super) static PASS_SOURCED: Cell<[u64; 4]> = const { Cell::new([0; 4]) };
    /// The tables a pixel shader reads as a bake does ([`tables`]), by the
    /// shader's hash: asked at every draw.
    static TABLES: RefCell<HashMap<String, Option<BakeTables>>> = RefCell::new(HashMap::new());
}

/// Where a pixel shader of the title's decoration bake finds what it
/// samples: the root parameter of its one table, and in that table the
/// dwords of its buffer of decoration records and of its constants.
///
/// The bake names no texture in a table. Its constants hold the view heap
/// indices of up to five textures (bytes 0x20, 0x30, 0x40, 0x50 and 0x60)
/// and the records it walks (first at 0x78, the one past the last at 0x7c,
/// of the count at 0); a record is 304 bytes and holds five more indices
/// (bytes 0, 0x10, 0x20, 0x30 and 0x40). Read from the programs' code, which
/// is the same in all of them but for the two table offsets, and checked at
/// every draw: an index counts only if the slot it names holds a texture
/// view.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct BakeTables {
    slot: u8,
    records: u32,
    constants: u32,
}

/// The stride of the bake's records.
const RECORD: u64 = 304;
/// Where its constants and each of its records hold view heap indices.
const CONSTANT_INDICES: [usize; 5] = [0x20, 0x30, 0x40, 0x50, 0x60];
const RECORD_INDICES: [usize; 5] = [0, 0x10, 0x20, 0x30, 0x40];

/// [`BakeTables`] of pipeline `p`, if its pixel shader reads one table with a
/// buffer of 304-byte records and constants and nothing else through tables.
pub(super) fn tables(p: &HostPipeline) -> Option<BakeTables> {
    let stage = p.stages.iter().find(|s| s.shader.stage == "pixel")?;
    let read = || {
        let mut tabled = stage.translated().used.iter().filter(|u| u.root == RootType::Table);
        let (a, b) = (tabled.next()?, tabled.next()?);
        if tabled.next().is_some() {
            return None;
        }
        let (records, constants) = if a.kind == ResourceType::Srv { (a, b) } else { (b, a) };
        (records.kind == ResourceType::Srv
            && u64::from(records.stride) == RECORD
            && constants.kind == ResourceType::Cbv
            && records.slot == constants.slot)
            .then_some(BakeTables {
                slot: records.slot,
                records: records.table_offset_dwords,
                constants: constants.table_offset_dwords,
            })
    };
    TABLES.with_borrow_mut(|known| match known.get(stage.shader.hash.as_str()) {
        Some(&tables) => tables,
        None => {
            let tables = read();
            known.insert(stage.shader.hash.clone(), tables);
            tables
        }
    })
}

/// The view heap indices a bake draw names, in order and each once, and the
/// records it walks with how many of them were read: from its constants, and
/// from the records `first..end` they give and the one after them, which the
/// program looks at as well; no more records than the buffer holds (`held`),
/// and no more than 256. `record` reads the start of one.
fn indices_named(
    constants: &[u8; 0x80],
    held: u32,
    mut record: impl FnMut(u32) -> Option<[u8; 0x44]>,
) -> (Vec<u32>, (u32, u32)) {
    let word = |bytes: &[u8], at: usize| u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    let mut named: Vec<u32> = CONSTANT_INDICES.iter().map(|&at| word(constants, at)).collect();
    let (first, end) = (word(constants, 0x78), word(constants, 0x7c));
    let mut read = 0;
    // (No record is looked at where none is walked.)
    let last = if end > first { end.saturating_add(1) } else { first };
    for i in first..last.min(held).min(first.saturating_add(256)) {
        let Some(record) = record(i) else {
            break;
        };
        named.extend(RECORD_INDICES.iter().map(|&at| word(&record, at)));
        read += 1;
    }
    named.sort_unstable();
    named.dedup();
    (named, (end.saturating_sub(first), read))
}

/// Whether the textures a bake draw samples are uploaded before it, whatever
/// the queue's order, the pass's budget and a slot's turn
/// (`VICTORY_BAKE_SOURCES=0`: they are only counted), and the megabytes one
/// draw may have uploaded so (`VICTORY_BAKE_SOURCES_MB`, 1024).
pub(super) fn forced() -> (bool, u64) {
    static ON: std::sync::OnceLock<(bool, u64)> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        (
            std::env::var_os("VICTORY_BAKE_SOURCES").is_none_or(|v| v != "0"),
            std::env::var("VICTORY_BAKE_SOURCES_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1024),
        )
    })
}

/// What a bake draw samples ([`State::bake_sources`]).
#[derive(Default, Debug)]
pub(super) struct BakeSources {
    /// The records the draw walks, and how many of them were read.
    records: (u32, u32),
    /// Slots named that hold a texture view, and their handles.
    textures: Vec<(u32, u64)>,
    /// Of them: with their host texture made from that view and nothing
    /// waiting.
    ready: u32,
    /// Waiting in the queue of textures picked by index.
    pub(super) queued: Vec<u32>,
    /// Neither, and a texture the title placed in memory that is there: the
    /// host slot shows nothing, and nothing would make it (the title wrote a
    /// newer view of memory this one clashes with, which made this one stale
    /// to the layer; the bake reads it all the same).
    pub(super) unmade: Vec<u32>,
    /// Neither, and something else: a texture the host itself writes.
    other: u32,
}

/// Since the start: bytes uploaded for bake draws as their own textures.
static SOURCED_BYTES: AtomicU64 = AtomicU64::new(0);

/// Adds `done` (see [`PASS_SOURCED`]) to the bytes counted since the start.
pub(super) fn sourced(done: [u64; 4]) {
    SOURCED_BYTES.fetch_add(done[1], Ordering::Relaxed);
}

/// The mapped bytes of the memory a placed texture view shows, of the mips
/// it shows (what lies behind them is another texture's), if `view` is one.
fn mapped(view: Option<View>) -> Option<u64> {
    let Some(View::Placed(false, resource, desc)) = view else {
        return None;
    };
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let len = crate::resource::placed_bytes_from(&resource, crate::resource::first_mip(&desc)).unwrap_or(1);
    Some(crate::texture::mapped_bytes(address as *const u8, len as usize))
}

impl State {
    /// The textures a draw of the title's decoration bake samples
    /// ([`BakeTables`]), by what the queue of textures picked by index holds
    /// of them. `None`: the draw's table does not hold what a bake's does.
    pub(super) fn bake_sources(
        &self,
        tables: BakeTables,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
    ) -> Option<BakeSources> {
        let Some(&Root::Table(table)) = self.roots_ref(false).get(usize::from(tables.slot)) else {
            return None;
        };
        let View::Constant(constants, size) = crate::descriptor::view(table + u64::from(tables.constants) * 4)? else {
            return None;
        };
        let mut words = [0u8; 0x80];
        if (size as usize) < words.len() || !crate::host::read_into(constants as *const u8, &mut words) {
            return None;
        }
        let (address, len) = crate::descriptor::view(table + u64::from(tables.records) * 4)
            .and_then(|view| crate::descriptor::host::buffer_range(&view))
            .unwrap_or((0, 0));
        let held = u32::try_from(len / RECORD).unwrap_or(u32::MAX);
        let (named, records) = indices_named(&words, held, |i| {
            let mut record = [0u8; 0x44];
            crate::host::read_into((address + u64::from(i) * RECORD) as *const u8, &mut record).then_some(record)
        });
        let mut sources = BakeSources { records, ..Default::default() };
        let mut textures = Vec::new();
        for index in named {
            let Some(handle) = heap.handle(index) else {
                continue;
            };
            if crate::descriptor::view(handle).is_some_and(|view| crate::descriptor::host::is_texture_view(&view)) {
                textures.push((index, handle));
            }
        }
        let indices: Vec<u32> = textures.iter().map(|t| t.0).collect();
        sources.queued = host.texture_slots_queued(&indices).into_iter().map(|q| q.0).collect();
        sources.textures = textures;
        for &(index, handle) in &sources.textures {
            if sources.queued.contains(&index) {
                continue;
            }
            if host.texture_slot_made_from(index, handle) {
                sources.ready += 1;
            } else if mapped(crate::descriptor::view(handle)).is_some_and(|mapped| mapped > 0) {
                sources.unmade.push(index);
            } else {
                sources.other += 1;
            }
        }
        Some(sources)
    }

    /// Of a bake draw's textures, the ones the draw samples wrong: placed by
    /// the title in memory that is there, with a host slot that shows nothing
    /// of them. Counted when the passes before the draw are done.
    pub(super) fn bake_missing(sources: &BakeSources, host: &crate::descriptor::host::HostHeap) -> u64 {
        let missing = |&&(index, handle): &&(u32, u64)| {
            !host.texture_slot_made_from(index, handle)
                && mapped(crate::descriptor::view(handle)).is_some_and(|mapped| mapped > 0)
        };
        sources.textures.iter().filter(missing).count() as u64
    }

    /// Before a draw of a program of the bake's shape that the pass before a
    /// bake does not know: its own sources, by a pass of their own. Nothing
    /// is done for a draw whose sources are all where they should be.
    pub(super) fn bake_sources_pass(
        &self,
        list: &HostList,
        p: &HostPipeline,
        tables: BakeTables,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
    ) {
        let Some(sources) = self.bake_sources(tables, heap, host) else {
            return;
        };
        let force = forced().0 && (!sources.queued.is_empty() || !sources.unmade.is_empty());
        if !force {
            Self::log_bake_sources(p, &sources, [0; 4], Self::bake_missing(&sources, host), false);
            return;
        }
        host.texture_slots_requeue(&sources.unmade, crate::frame::current());
        BAKE_SOURCES.set(sources.queued.iter().chain(&sources.unmade).copied().collect());
        PASS_BUDGET.set(Some((0, 0)));
        PASS_ONLY.set(true);
        PASS_SOURCED.set([0; 4]);
        self.heap_textures(list, heap, host);
        PASS_ONLY.set(false);
        PASS_BUDGET.set(None);
        BAKE_SOURCES.with_borrow_mut(Vec::clear);
        let done = PASS_SOURCED.get();
        sourced(done);
        Self::log_bake_sources(p, &sources, done, Self::bake_missing(&sources, host), true);
    }

    /// A line for a bake draw's sources: for the first draws, then at counts
    /// that double, and for up to 256 draws that sample some of theirs wrong
    /// (`missing`, [`State::bake_missing`]).
    pub(super) fn log_bake_sources(p: &HostPipeline, sources: &BakeSources, done: [u64; 4], missing: u64, forced: bool) {
        static DRAWS: AtomicU64 = AtomicU64::new(0);
        static SHORT: AtomicU64 = AtomicU64::new(0);
        static TOTALS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
        let n = DRAWS.fetch_add(1, Ordering::Relaxed) + 1;
        let waiting = sources.queued.len() as u64;
        let unmade = sources.unmade.len() as u64;
        let [uploaded, bytes, unmapped, as_before] = done;
        for (total, now) in
            TOTALS.iter().zip([waiting, uploaded, unmapped, u64::from(sources.other), unmade, as_before, missing])
        {
            total.fetch_add(now, Ordering::Relaxed);
        }
        if !(n.is_power_of_two() || (missing > 0 && SHORT.fetch_add(1, Ordering::Relaxed) < 256)) {
            return;
        }
        let pixel = p.stages.iter().find(|s| s.shader.stage == "pixel").map_or("", |s| s.shader.hash.as_str());
        let [waited, uploads, unmapped_so_far, others, unmade_so_far, as_before_so_far, missing_so_far] =
            TOTALS.each_ref().map(|t| t.load(Ordering::Relaxed));
        tracing::info!(
            frame = crate::frame::current(),
            draws = n,
            pixel,
            records = sources.records.0,
            records_read = sources.records.1,
            textures = sources.textures.len(),
            on_the_card = sources.ready,
            waiting,
            made_stale = unmade,
            the_host_s_own = sources.other,
            forced,
            uploaded,
            kilobytes = bytes >> 10,
            as_far_as_mapped_already = as_before,
            not_mapped = unmapped,
            sampled_wrong = missing,
            sampled_wrong_so_far = missing_so_far,
            megabytes_so_far = SOURCED_BYTES.load(Ordering::Relaxed) >> 20,
            waited_so_far = waited,
            uploaded_so_far = uploads,
            as_far_as_mapped_already_so_far = as_before_so_far,
            not_mapped_so_far = unmapped_so_far,
            made_stale_so_far = unmade_so_far,
            the_host_s_own_so_far = others,
            "a decoration bake draw's own textures, by its constants and records: the ones still waiting are uploaded before it (VICTORY_BAKE_SOURCES=0: only counted)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bake draw names the indices in its constants, and those in the
    /// records its constants give and in the one after them; not those of
    /// records past the buffer, nor of one that cannot be read.
    #[test]
    fn a_bake_draw_names_the_indices_of_its_constants_and_records() {
        let mut constants = [0u8; 0x80];
        let put = |bytes: &mut [u8], at: usize, value: u32| bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
        for (n, at) in CONSTANT_INDICES.into_iter().enumerate() {
            put(&mut constants, at, 100 + n as u32);
        }
        // Records 2 and 3, of 9 in all.
        put(&mut constants, 0, 9);
        put(&mut constants, 0x78, 2);
        put(&mut constants, 0x7c, 4);
        // Record `i` names 1000 + 10 i + n at its n-th place; record 3 the
        // index of the constants' first once more.
        let record = |i: u32| {
            let mut record = [0xffu8; 0x44];
            for (n, at) in RECORD_INDICES.into_iter().enumerate() {
                put(&mut record, at, 1000 + 10 * i + n as u32);
            }
            if i == 3 {
                put(&mut record, 0, 100);
            }
            Some(record)
        };
        let (names, records) = indices_named(&constants, 9, record);
        assert_eq!(records, (2, 3), "two records walked, and the one after them looked at");
        let mut expected: Vec<u32> = (100..105).collect();
        expected.extend((1020..1025).chain(1031..1035).chain(1040..1045));
        assert_eq!(names, expected);
        // The buffer ends with record 3: the one after is not there.
        assert_eq!(indices_named(&constants, 4, record).1, (2, 2));
        // A record that cannot be read ends the walk.
        let (names, records) = indices_named(&constants, 9, |i| if i == 3 { None } else { record(i) });
        assert_eq!(records, (2, 1));
        assert!(names.contains(&1024) && !names.contains(&1040));
        // No records at all: the constants' own.
        put(&mut constants, 0x7c, 2);
        assert_eq!(indices_named(&constants, 0, record), ((100..105).collect(), (0, 0)));
    }
}
