//! `VICTORY_BAKE_PROBE=1`: what the title's decoration bake reads and what
//! it leaves, in one run (diagnostics builds; counts, addresses, sizes,
//! formats, frame numbers and hashes in the log, nothing of the title's
//! contents).
//!
//! The title bakes its characters' skin and decorations into textures of
//! its own with mesh draws whose pixel shaders pick their source textures
//! by index from the view heap (`draw.rs` `BAKE_PIXEL_SHADERS`); in some
//! launches the result is right and in others a source is missing. The
//! probe separates the ways that can happen:
//!
//! - **The sources were not there** (`bake probe: ... a bake draw`): at the
//!   draw, the dwords of the buffers the bake's tables name (and its root
//!   constants) that are slots of the view heap holding a texture view are
//!   taken to be the textures it picks, and each is classed: its memory not
//!   mapped, only partly mapped, never uploaded, uploaded while some of its
//!   memory was still zero and filled since, changed since the upload with
//!   nothing due, upload pending, or as it was uploaded. A dword that only
//!   looks like a slot is a false candidate: read the counts, and the
//!   per-slot lines for the first draws.
//! - **The sources arrived after the bake** (`bake probe: ... after the
//!   bake`): the same slots are looked at again 2, 8 and 30 seconds after
//!   each of the first eight bake draws, and those whose memory differs from
//!   what the bake saw (or that were empty then and are filled now) are
//!   counted.
//! - **The bake's target is read through another copy** (`bake probe: a
//!   draw samples memory of a bake target ...`): every texture a draw reads
//!   through a table is checked against the memory of the targets the bake
//!   drew into; one that lies in it and is another title texture than the
//!   target is counted (read by a copy on the host, or by its own copy
//!   filled from title memory, which never held what the bake drew), and
//!   the first are described with both descriptions.
//! - **The result** (`bake probe: a bake target after the frame`): after
//!   the frames in which a bake ran, each target is read back and its
//!   256x256 blocks that hold any colour are counted, with a hash, to tell
//!   a good launch from a bad one by number. So is a texture a draw writes
//!   through an unordered access view that has the shape of a target (the
//!   compute pass after a bake draw reads a target and writes a texture of
//!   its own: the one materials may read).
//! - **Two copies of one memory** (`bake probe: census`): every 128 frames,
//!   the title textures that share their memory with a texture the host
//!   writes, and whether the host reads that one by a copy.
//! - **What a remap does to staged textures** (`texture::remapped`): the
//!   calls, the staged textures they marked for a new upload, how many of
//!   those had been found unchanged, and how many lie in a bake target or
//!   in the sources of an earlier bake.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};

use crate::descriptor::View;
use crate::object::This;

/// Whether the probe is on (`VICTORY_BAKE_PROBE=1`).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BAKE_PROBE").is_some_and(|v| v != "0"))
}

/// The ways a texture a bake names can stand at the draw, in the order
/// [`classify`] tries them.
const CLASSES: [&str; 9] = [
    "named",
    "no_host_slot",
    "unmapped",
    "partly_mapped",
    "shown_not_uploaded_by_the_pass",
    "uploaded_empty_now_filled",
    "changed_since_upload",
    "upload_pending",
    "as_uploaded",
];
const NO_HOST_SLOT: usize = 1;
const UNMAPPED: usize = 2;
const PARTLY: usize = 3;
const NOT_UPLOADED: usize = 4;
const EMPTY_NOW_FILLED: usize = 5;
const CHANGED: usize = 6;
const PENDING: usize = 7;
const AS_UPLOADED: usize = 8;

/// What is known of one texture a bake names, at the draw.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Facts {
    /// The host slot shows a host resource.
    pub(crate) shown: bool,
    /// Bytes of its memory that are mapped, and that it takes.
    pub(crate) mapped: u64,
    pub(crate) bytes: u64,
    /// What the pass noted when it uploaded the slot: the windows of its
    /// memory that held something then ([`windows_with_data`]) and the
    /// fingerprint of the memory.
    pub(crate) uploaded: Option<(u8, u64)>,
    /// The slot is in the queue and due (or a look found it changed).
    pub(crate) pending: bool,
    /// The same, now.
    pub(crate) bits_now: u8,
    pub(crate) print_now: u64,
}

/// Which of [`CLASSES`] a texture stands in.
pub(crate) fn classify(f: &Facts) -> usize {
    if f.mapped == 0 {
        return UNMAPPED;
    }
    if f.mapped < f.bytes {
        return PARTLY;
    }
    if !f.shown {
        return if f.pending { PENDING } else { NO_HOST_SLOT };
    }
    let Some((bits, print)) = f.uploaded else {
        return if f.pending { PENDING } else { NOT_UPLOADED };
    };
    if f.bits_now & !bits != 0 {
        return EMPTY_NOW_FILLED;
    }
    if f.print_now != print {
        return CHANGED;
    }
    if f.pending {
        return PENDING;
    }
    AS_UPLOADED
}

/// Which of the start, the middle and the end of `[at, at + len)` (4 KiB
/// each, bits 0, 1 and 2) hold any byte that is not zero; unmapped pages
/// read as zeros.
pub(crate) fn windows_with_data(at: u64, len: u64) -> u8 {
    const W: usize = 4096;
    if len == 0 {
        return 0;
    }
    let len = len as usize;
    let spots = [0usize, (len / 2) & !0xfff, len.saturating_sub(W)];
    let mut bits = 0u8;
    let mut buf = [0u8; W];
    for (i, &from) in spots.iter().enumerate() {
        let n = W.min(len - from);
        buf[..n].fill(0);
        crate::host::copy_mapped(buf.as_mut_ptr(), (at as usize + from) as *const u8, n, false);
        if buf[..n].iter().any(|&b| b != 0) {
            bits |= 1 << i;
        }
    }
    bits
}

/// What the pass did when it uploaded a slot (noted by [`uploaded`]).
#[derive(Clone, Copy, Debug)]
struct Upload {
    frame: u64,
    mapped_in_part: bool,
    tabled: bool,
    bits: u8,
    print: u64,
}

static UPLOADS: Mutex<Option<HashMap<u32, Upload>>> = Mutex::new(None);

/// The memory a texture view shows: address and bytes.
fn span_of(view: &View) -> Option<(u64, u64)> {
    match view {
        View::Placed(false, resource, desc) if resource[0] != 1 => {
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let bytes = crate::resource::placed_bytes_from(resource, crate::resource::first_mip(desc))?;
            Some((address, bytes))
        }
        View::Shader(r, _) => {
            let t = crate::resource::probe_texture(*r as This, false)?;
            Some((t.start, t.bytes))
        }
        _ => None,
    }
}

/// The texture pass uploaded (or, for a texture a table's draws read,
/// handed to them) the view at title slot `index` of `heap`: noted with the
/// state of its memory, to compare with at a bake. `partly`: not all of its
/// memory was mapped.
pub(crate) fn uploaded(heap: &crate::descriptor::Heap, index: u32, frame: u64, partly: bool, tabled: bool) {
    if !on() {
        return;
    }
    let Some(view) = heap.handle(index).and_then(crate::descriptor::view) else {
        return;
    };
    let Some((address, bytes)) = span_of(&view) else {
        return;
    };
    let note = Upload {
        frame,
        mapped_in_part: partly,
        tabled,
        bits: windows_with_data(address, bytes),
        print: crate::texture::fingerprint(address, bytes.max(1)),
    };
    UPLOADS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(index, note);
}

/// One source of a bake as it was at the draw.
#[derive(Clone, Copy, Debug)]
struct Source {
    slot: u32,
    address: u64,
    bytes: u64,
    print: u64,
    bits: u8,
}

/// A bake draw whose sources are looked at again later.
struct Bake {
    n: u64,
    at_ms: u64,
    sources: Vec<Source>,
    checks: u8,
}

/// The first bake draws kept for [`present`]'s second looks.
static BAKES: Mutex<Vec<Bake>> = Mutex::new(Vec::new());

/// How long after a bake each second look is made.
const AFTER_MS: [u64; 3] = [2_000, 8_000, 30_000];

/// The memory of the textures the bake draws into: start → (end, title
/// texture).
static TARGETS: Mutex<Option<std::collections::BTreeMap<u64, (u64, usize)>>> = Mutex::new(None);

/// Host copies of the bake's targets, held for the read-back at the frame's
/// present.
static NOTED: Mutex<Vec<(usize, Arc<crate::resource::HostTexture>)>> = Mutex::new(Vec::new());

/// The shapes (width, height, format) of the textures a bake draws into: a
/// texture a later draw writes through an unordered access view that has
/// one of them is taken for the bake's output ([`written`]): the bake's
/// compute pass reads a target and writes a texture of its own.
static SHAPES: Mutex<Vec<(u64, u32, u32)>> = Mutex::new(Vec::new());

static BAKE_COUNT: AtomicU64 = AtomicU64::new(0);
static BAKED_THIS_FRAME: AtomicBool = AtomicBool::new(false);
static READ_FRAMES: AtomicU64 = AtomicU64::new(0);
static CLASS_TOTALS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

/// A bake draw, as `draw.rs` found it.
pub(crate) struct BakeDraw<'a> {
    /// The title's frame, and the bake's pixel shader.
    pub(crate) frame: u64,
    pub(crate) pixel: &'a str,
    /// The bytes of the memory its tables' buffers and root views name.
    pub(crate) buffers: Vec<(u64, Vec<u8>)>,
    /// The title textures it draws into.
    pub(crate) targets: &'a [usize],
    pub(crate) heap: &'a crate::descriptor::Heap,
    pub(crate) host: &'a crate::descriptor::host::HostHeap,
    /// The queue of textures picked by index, as the pass last counted it.
    pub(crate) queue: [u64; 9],
}

/// A draw of the bake (after the texture pass that runs before it).
pub(crate) fn bake(d: &BakeDraw) {
    if !on() {
        return;
    }
    let n = BAKE_COUNT.fetch_add(1, Relaxed) + 1;
    BAKED_THIS_FRAME.store(true, Relaxed);
    note_targets(n, d);
    if n > 12 && !n.is_power_of_two() {
        return;
    }
    let mut slots: Vec<u32> = Vec::new();
    let mut dwords = 0u64;
    for (_, bytes) in &d.buffers {
        for c in bytes.chunks_exact(4) {
            let v = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            dwords += 1;
            // Slots below 16 are the title's own fixed ones; the rest of
            // small values are mostly counts and flags.
            if v < 16 {
                continue;
            }
            let Some(view) = d.heap.handle(v).and_then(crate::descriptor::view) else {
                continue;
            };
            if crate::descriptor::host::is_texture_view(&view) {
                slots.push(v);
            }
        }
    }
    slots.sort_unstable();
    slots.dedup();
    let uploads = UPLOADS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let now = crate::frame::current();
    let mut counts = [0u64; 9];
    let mut sources = Vec::new();
    let mut lines = Vec::new();
    for &slot in slots.iter().take(400) {
        let Some(handle) = d.heap.handle(slot) else {
            continue;
        };
        let Some(view) = crate::descriptor::view(handle) else {
            continue;
        };
        let Some((address, bytes)) = span_of(&view) else {
            continue;
        };
        if bytes == 0 {
            continue;
        }
        let probe = d.host.probe_slot(slot);
        let mapped = crate::texture::mapped_bytes(address as *const u8, bytes as usize);
        let bits_now = windows_with_data(address, bytes);
        let print_now = crate::texture::fingerprint(address, bytes);
        let up = uploads.as_ref().and_then(|m| m.get(&slot)).copied();
        // The host slot is made from the very view the title has now, and
        // holds a host resource (the fork's own question, `texture_slot_made_from`).
        let shown = d.host.texture_slot_made_from(slot, handle);
        let facts = Facts {
            shown,
            mapped,
            bytes,
            uploaded: up.map(|u| (u.bits, u.print)),
            pending: probe.queued && (probe.due <= d.frame || probe.stale),
            bits_now,
            print_now,
        };
        let class = classify(&facts);
        counts[class] += 1;
        counts[0] += 1;
        sources.push(Source { slot, address, bytes, print: print_now, bits: bits_now });
        if n <= 2 {
            let size = match &view {
                View::Placed(_, r, _) => format!("{}x{} f{} lv{:#x}", r[4], r[6], r[8], r[7]),
                _ => String::from("object"),
            };
            lines.push((class, slot, address, bytes, mapped, size, probe, up, bits_now, facts));
        }
    }
    for (c, total) in counts.iter().zip(&CLASS_TOTALS) {
        total.fetch_add(*c, Relaxed);
    }
    let [named, no_host_slot, unmapped, partly, not_uploaded, empty_now_filled, changed, pending, as_uploaded] =
        counts;
    tracing::info!(
        bake = n,
        frame = d.frame,
        pixel = d.pixel,
        buffers = d.buffers.len(),
        dwords,
        slots_named = slots.len(),
        named,
        no_host_slot,
        unmapped,
        partly_mapped = partly,
        shown_not_uploaded_by_the_pass = not_uploaded,
        uploaded_empty_now_filled = empty_now_filled,
        changed_since_upload = changed,
        upload_pending = pending,
        as_uploaded,
        card_has_room = crate::resource::room(),
        passes_without_room_so_far = crate::draw::HEAP_TEXTURES_FULL.load(Relaxed),
        queue = ?d.queue,
        "bake probe: a bake draw: the textures its data names, by how they stand now (queue: slots, due, waiting for memory, partly uploaded, oldest due, never uploaded, uploaded whole waiting for a look, found changed, looks due)"
    );
    // The ones that are not as uploaded first, then some that are.
    lines.sort_by_key(|l| l.0 == AS_UPLOADED);
    for (class, slot, address, bytes, mapped, size, p, up, bits_now, facts) in lines.into_iter().take(32) {
        tracing::info!(
            bake = n,
            slot,
            class = CLASSES[class],
            address = format_args!("{address:#x}"),
            bytes,
            mapped,
            size,
            made_from_the_view_now = facts.shown,
            holds_a_host_resource = p.shown,
            host = format_args!("{:#x}", p.host),
            queued = p.queued,
            due = p.due,
            written = p.written,
            looks = p.looks,
            changes = p.changes,
            stale = p.stale,
            mapped_at_the_last_partial_upload = p.mapped,
            looked_at_since = p.relook.map_or(0, |(_, since)| since),
            uploaded_at = up.map_or(0, |u| u.frame),
            frames_since = up.map_or(0, |u| now.saturating_sub(u.frame)),
            uploaded_in_part = up.is_some_and(|u| u.mapped_in_part),
            left_to_tables = up.is_some_and(|u| u.tabled),
            windows_at_upload = up.map_or(0, |u| u.bits),
            windows_now = bits_now,
            same_print = facts.uploaded.is_some_and(|(_, print)| print == facts.print_now),
            "bake probe: one texture a bake draw names"
        );
    }
    if n <= 8 && !sources.is_empty() {
        BAKES.lock().unwrap_or_else(|e| e.into_inner()).push(Bake {
            n,
            at_ms: crate::host::now_ms(),
            sources,
            checks: 0,
        });
    }
}

/// Notes the textures a bake draws into: their memory (for [`sampled`]),
/// their host copies (for the read-back at the present), and, for the first
/// bakes and each new target, their description.
fn note_targets(n: u64, d: &BakeDraw) {
    let mut targets = TARGETS.lock().unwrap_or_else(|e| e.into_inner());
    let map = targets.get_or_insert_with(Default::default);
    for &t in d.targets {
        let Some(info) = crate::resource::probe_texture(t as This, false) else {
            continue;
        };
        if info.bytes == 0 {
            continue;
        }
        {
            let mut shapes = SHAPES.lock().unwrap_or_else(|e| e.into_inner());
            let shape = (info.width, info.height, info.format);
            if !shapes.contains(&shape) && shapes.len() < 16 {
                shapes.push(shape);
            }
        }
        let new = map.insert(info.start, (info.start + info.bytes, t)).is_none_or(|(_, was)| was != t);
        if new || n <= 2 {
            tracing::info!(
                bake = n,
                frame = d.frame,
                target = format_args!("{t:#x}"),
                address = format_args!("{:#x}", info.start),
                bytes = info.bytes,
                width = info.width,
                height = info.height,
                layers = info.layers,
                mips = info.mips,
                format = info.format,
                flags = format_args!("{:#x}", info.flags),
                host = format_args!("{:#x}", info.host),
                "bake probe: a texture a bake draws into"
            );
        }
        let mut noted = NOTED.lock().unwrap_or_else(|e| e.into_inner());
        if noted.len() < 8
            && !noted.iter().any(|(r, _)| *r == t)
            && READ_FRAMES.load(Relaxed) < 12
            && let Some(copy) = crate::resource::host_copy(t as This)
        {
            noted.push((t, copy));
        }
    }
}

/// A draw or dispatch writes the texture `resource` through an unordered
/// access view (`readable`): if it has the shape of a bake's target and is
/// not one, it is noted for the read-back at the present as a texture the
/// bake leaves (the compute pass that follows a bake draw reads a target and
/// writes such a texture), and its description is logged once.
pub(crate) fn written(resource: This) {
    if !on() || BAKE_COUNT.load(Relaxed) == 0 || READ_FRAMES.load(Relaxed) >= 12 {
        return;
    }
    let Some(t) = crate::resource::probe_texture(resource, false) else {
        return;
    };
    let shape = (t.width, t.height, t.format);
    if !SHAPES.lock().unwrap_or_else(|e| e.into_inner()).contains(&shape) {
        return;
    }
    if TARGETS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|m| m.values().any(|&(_, r)| r == resource as usize))
    {
        return;
    }
    let mut noted = NOTED.lock().unwrap_or_else(|e| e.into_inner());
    if noted.len() >= 8 || noted.iter().any(|(r, _)| *r == resource as usize) {
        return;
    }
    if let Some(copy) = crate::resource::host_copy(resource) {
        noted.push((resource as usize, copy));
        tracing::info!(
            frame = crate::frame::current(),
            resource = format_args!("{:#x}", resource as usize),
            address = format_args!("{:#x}", t.start),
            width = t.width,
            height = t.height,
            format = t.format,
            flags = format_args!("{:#x}", t.flags),
            host = format_args!("{:#x}", t.host),
            "bake probe: a texture written through an unordered access view has the shape of a bake target: noted for the read-back"
        );
    }
}

/// How draws that sample memory of a bake's targets read it, since the
/// start: the target itself, a copy on the host from the target
/// (`resource::alias_of`), another title texture at the same start, another
/// that only overlaps.
static SAMPLES: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// A draw reads the texture `resource` through a table (`readable`);
/// `placed`: through a view the title placed in memory.
pub(crate) fn sampled(resource: This, placed: bool) {
    if !on() {
        return;
    }
    let targets = TARGETS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(map) = targets.as_ref().filter(|m| !m.is_empty()) else {
        return;
    };
    let Some(t) = crate::resource::probe_texture(resource, false) else {
        return;
    };
    if t.bytes == 0 {
        return;
    }
    if map.get(&t.start).is_some_and(|&(_, r)| r == resource as usize) {
        SAMPLES[0].fetch_add(1, Relaxed);
        return;
    }
    let end = t.start + t.bytes;
    let Some((&start, &(target_end, target))) =
        map.iter().find(|&(&s, &(e, r))| r != resource as usize && s < end && t.start < e)
    else {
        return;
    };
    let aliased = crate::resource::alias_of(resource).is_some();
    let kind = if aliased {
        1
    } else if start == t.start {
        2
    } else {
        3
    };
    let n = SAMPLES[kind].fetch_add(1, Relaxed) + 1;
    // The first of each kind, and then every power of two.
    if n <= 6 || n.is_power_of_two() {
        let target_info = crate::resource::probe_texture(target as This, true);
        tracing::info!(
            frame = crate::frame::current(),
            kind = ["", "read by a copy from the target", "another texture at the same start", "another texture that overlaps"][kind],
            n,
            through_a_placed_view = placed,
            viewed_is_filled_by_the_title = t.fills,
            viewed_is_a_placed_view_texture = t.placed,
            viewed_host_copy_was_written_by_the_host = t.host_wrote,
            viewed = ?t,
            target_start = format_args!("{start:#x}"),
            target_end = format_args!("{target_end:#x}"),
            target = ?target_info,
            "bake probe: a draw samples memory of a bake target through another title texture"
        );
    }
}

/// Counts of the remaps `texture::remapped` took (calls, staged textures
/// examined, marked for a new upload, of those found unchanged since their
/// upload, of those in a bake's target, of those in the sources of one of
/// the first bakes).
static REMAPS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// `texture::remapped` ran and looked at `examined` staged textures.
pub(crate) fn remap_call(examined: u64) {
    if !on() {
        return;
    }
    REMAPS[0].fetch_add(1, Relaxed);
    REMAPS[1].fetch_add(examined, Relaxed);
}

/// `texture::remapped` marked the staged texture over `[start, start +
/// bytes)` for a new upload; `was_clean`: it had been uploaded and found
/// unchanged since (without the mark, its next list would have read the old
/// texels).
pub(crate) fn remap_hit(start: u64, bytes: u64, was_clean: bool) {
    if !on() {
        return;
    }
    REMAPS[2].fetch_add(1, Relaxed);
    if was_clean {
        REMAPS[3].fetch_add(1, Relaxed);
    }
    let end = start.saturating_add(bytes);
    let in_target = TARGETS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|m| m.iter().any(|(&s, &(e, _))| s < end && start < e));
    if in_target {
        REMAPS[4].fetch_add(1, Relaxed);
    }
    let in_source = BAKES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .any(|b| b.sources.iter().any(|s| s.address < end && start < s.address + s.bytes));
    if in_source {
        REMAPS[5].fetch_add(1, Relaxed);
    }
}

/// At a frame's present (every submission has run).
pub(crate) fn present(frame: u64) {
    if !on() {
        return;
    }
    second_looks();
    if BAKED_THIS_FRAME.swap(false, Relaxed) {
        let read = READ_FRAMES.fetch_add(1, Relaxed) + 1;
        if read <= 12 || read.is_power_of_two() {
            read_targets(frame, read);
        }
        // Held for the first twelve frames only: the title may release the
        // targets and these copies would keep their memory.
        if read >= 12 {
            NOTED.lock().unwrap_or_else(|e| e.into_inner()).clear();
        }
    }
    static LAST: Mutex<[u64; 4]> = Mutex::new([0; 4]);
    let samples = SAMPLES.each_ref().map(|c| c.load(Relaxed));
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if *last != samples {
            *last = samples;
            tracing::info!(
                frame,
                the_target_itself = samples[0],
                read_by_a_copy_from_it = samples[1],
                other_texture_at_its_start = samples[2],
                other_texture_overlapping = samples[3],
                "bake probe: draws that sample memory of a bake's targets, since the start"
            );
        }
    }
    if frame.is_multiple_of(128) && BAKE_COUNT.load(Relaxed) > 0 {
        let c = crate::resource::probe_census();
        let class_totals: Vec<u64> = CLASS_TOTALS.iter().map(|c| c.load(Relaxed)).collect();
        let r = REMAPS.each_ref().map(|c| c.load(Relaxed));
        tracing::info!(
            frame,
            bakes = BAKE_COUNT.load(Relaxed),
            title_textures = c[0],
            textures_the_host_writes = c[1],
            view_textures_with_a_host_copy = c[2],
            memory_shared_by_a_target_and_a_view_texture = c[3],
            of_those_read_by_a_copy = c[4],
            of_those_whose_host_copy_was_written_by_the_host = c[5],
            overlapping = c[6],
            textures_written_by_host_copies = c[7],
            remap_calls = r[0],
            staged_examined = r[1],
            staged_marked = r[2],
            marked_found_unchanged = r[3],
            marked_in_a_bake_target = r[4],
            marked_in_a_bake_source = r[5],
            textures_uploaded_again_with_the_card_full = crate::draw::HEAP_TEXTURES_REUPLOADED_FULL.load(Relaxed),
            passes_without_room = crate::draw::HEAP_TEXTURES_FULL.load(Relaxed),
            slots_made_stale_by_a_newer_view = crate::descriptor::host::STALE[0].load(Relaxed),
            slots_made_due_by_remaps = crate::descriptor::host::STALE[3].load(Relaxed),
            made_slots_found_unmapped = crate::descriptor::host::STALE[4].load(Relaxed),
            unmaps_refused_for_a_newer_write = crate::descriptor::host::GONE_REFUSED.load(Relaxed),
            view_sweep_at_the_last_look = ?crate::descriptor::host::SWEPT.each_ref().map(|c| c.load(Relaxed)),
            ?class_totals,
            "bake probe: census"
        );
    }
}

/// The second looks at the sources of the first bakes ([`AFTER_MS`]).
fn second_looks() {
    let now = crate::host::now_ms();
    // The work is done outside the lock: the sources are read.
    let due: Vec<(u64, u8, Vec<Source>, u64)> = {
        let mut bakes = BAKES.lock().unwrap_or_else(|e| e.into_inner());
        bakes
            .iter_mut()
            .filter_map(|b| {
                let wait = *AFTER_MS.get(usize::from(b.checks))?;
                (now.saturating_sub(b.at_ms) >= wait).then(|| {
                    b.checks += 1;
                    (b.n, b.checks, b.sources.clone(), now - b.at_ms)
                })
            })
            .collect()
    };
    for (n, check, sources, after) in due {
        let (mut changed, mut filled, mut unmapped, mut empty_then) = (0u64, 0u64, 0u64, 0u64);
        let mut first_changed: Vec<u32> = Vec::new();
        for s in &sources {
            if s.bits != 0b111 {
                empty_then += 1;
            }
            if crate::texture::mapped_bytes(s.address as *const u8, s.bytes as usize) == 0 {
                unmapped += 1;
                continue;
            }
            if crate::texture::fingerprint(s.address, s.bytes) != s.print {
                changed += 1;
                if first_changed.len() < 8 {
                    first_changed.push(s.slot);
                }
            }
            if windows_with_data(s.address, s.bytes) & !s.bits != 0 {
                filled += 1;
            }
        }
        tracing::info!(
            bake = n,
            look = check,
            ms_after_the_bake = after,
            sources = sources.len(),
            changed_since_the_bake = changed,
            empty_at_the_bake_filled_since = filled,
            with_a_window_empty_at_the_bake = empty_then,
            unmapped_now = unmapped,
            ?first_changed,
            "bake probe: the sources of a bake draw, after the bake"
        );
    }
}

/// Reads the targets of the bakes back and counts their colour.
fn read_targets(frame: u64, read: u64) {
    let noted: Vec<(usize, Arc<crate::resource::HostTexture>)> =
        NOTED.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let Some(host) = crate::host::host() else {
        return;
    };
    for (i, (resource, copy)) in noted.iter().enumerate() {
        let stats = crate::capture::with_rows(host, copy, |rows, d, f| block_stats(rows, d.Format.0 as u32, f));
        let Some(Some(s)) = stats else {
            continue;
        };
        tracing::info!(
            frame,
            read,
            index = i,
            target = format_args!("{resource:#x}"),
            width = s.width,
            height = s.height,
            format = s.format,
            blocks_with_colour = s.blocks_with_colour,
            blocks = s.blocks,
            texels_with_colour = s.texels_with_colour,
            texels = u64::from(s.width) * u64::from(s.height),
            hash = format_args!("{:016x}", s.hash),
            "bake probe: a bake target after the frame (blocks are 256x256 texels; colour: any byte of the first three of a texel that is not zero)"
        );
    }
}

/// What [`block_stats`] counts in a read-back texture.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct BlockStats {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: u32,
    pub(crate) blocks_with_colour: u64,
    pub(crate) blocks: u64,
    pub(crate) texels_with_colour: u64,
    pub(crate) hash: u64,
}

/// Counts the texels of `rows` (pitch from `f`) that have colour, and the
/// 256x256 blocks that have any; `None` for formats that are not plain
/// texels of 1 to 16 bytes.
pub(crate) fn block_stats(
    rows: &[u8],
    format: u32,
    f: &windows::Win32::Graphics::Direct3D12::D3D12_SUBRESOURCE_FOOTPRINT,
) -> Option<BlockStats> {
    let (bytes, block) = crate::resource::element(format)?;
    if block != 1 {
        return None;
    }
    let (width, height, pitch) = (f.Width as usize, f.Height as usize, f.RowPitch as usize);
    let bytes = bytes as usize;
    let look = if bytes == 4 { 3 } else { bytes };
    if rows.len() < pitch * height || pitch < width * bytes {
        return None;
    }
    let (bw, bh) = (width.div_ceil(256), height.div_ceil(256));
    let mut blocks = vec![false; bw * bh];
    let mut texels = 0u64;
    for y in 0..height {
        let row = &rows[y * pitch..y * pitch + width * bytes];
        for (x, texel) in row.chunks_exact(bytes).enumerate() {
            if texel[..look].iter().any(|&b| b != 0) {
                texels += 1;
                blocks[(y / 256) * bw + x / 256] = true;
            }
        }
    }
    let hash = blake3::hash(&rows[..pitch * height]);
    let mut first = [0u8; 8];
    first.copy_from_slice(&hash.as_bytes()[..8]);
    Some(BlockStats {
        width: f.Width,
        height: f.Height,
        format,
        blocks_with_colour: blocks.iter().filter(|&&b| b).count() as u64,
        blocks: blocks.len() as u64,
        texels_with_colour: texels,
        hash: u64::from_le_bytes(first),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D12::D3D12_SUBRESOURCE_FOOTPRINT;

    fn facts() -> Facts {
        Facts {
            shown: true,
            mapped: 100,
            bytes: 100,
            uploaded: Some((0b111, 7)),
            pending: false,
            bits_now: 0b111,
            print_now: 7,
        }
    }

    #[test]
    fn a_texture_as_it_was_uploaded_is_classed_so() {
        assert_eq!(classify(&facts()), AS_UPLOADED);
    }

    #[test]
    fn memory_not_mapped_or_only_partly_mapped_comes_before_the_upload() {
        assert_eq!(classify(&Facts { mapped: 0, ..facts() }), UNMAPPED);
        assert_eq!(classify(&Facts { mapped: 40, ..facts() }), PARTLY);
    }

    #[test]
    fn a_window_empty_at_the_upload_and_filled_since_is_the_stale_read() {
        let stale = Facts { uploaded: Some((0b001, 7)), bits_now: 0b101, print_now: 9, ..facts() };
        assert_eq!(classify(&stale), EMPTY_NOW_FILLED);
        // Windows that were empty then and still are change nothing.
        let same = Facts { uploaded: Some((0b001, 7)), bits_now: 0b001, ..facts() };
        assert_eq!(classify(&same), AS_UPLOADED);
    }

    #[test]
    fn a_changed_fingerprint_with_nothing_due_is_a_texture_that_was_missed() {
        assert_eq!(classify(&Facts { print_now: 8, ..facts() }), CHANGED);
        // Due: the pass will upload it.
        assert_eq!(classify(&Facts { print_now: 8, pending: true, ..facts() }), CHANGED);
        assert_eq!(classify(&Facts { pending: true, ..facts() }), PENDING);
    }

    #[test]
    fn a_slot_never_uploaded_is_pending_or_not_shown() {
        let never = Facts { uploaded: None, ..facts() };
        assert_eq!(classify(&never), NOT_UPLOADED);
        assert_eq!(classify(&Facts { pending: true, ..never }), PENDING);
        assert_eq!(classify(&Facts { shown: false, ..never }), NO_HOST_SLOT);
        assert_eq!(classify(&Facts { shown: false, pending: true, ..never }), PENDING);
    }

    #[test]
    fn windows_with_data_tells_the_empty_from_the_filled() {
        let mut memory = vec![0u8; 3 * 4096 + 100];
        assert_eq!(windows_with_data(memory.as_ptr() as u64, memory.len() as u64), 0);
        memory[10] = 1;
        assert_eq!(windows_with_data(memory.as_ptr() as u64, memory.len() as u64), 0b001);
        let last = memory.len() - 1;
        memory[last] = 1;
        assert_eq!(windows_with_data(memory.as_ptr() as u64, memory.len() as u64), 0b101);
        memory[(last / 2) & !0xfff] = 1;
        assert_eq!(windows_with_data(memory.as_ptr() as u64, memory.len() as u64), 0b111);
    }

    #[test]
    fn block_stats_counts_the_blocks_that_hold_colour() {
        // 512x256 RGBA8 (format 28), pitch 2,048: the left block black with
        // full alpha, one coloured texel in the right one.
        let (w, h, pitch) = (512usize, 256usize, 2048usize);
        let mut rows = vec![0u8; pitch * h];
        for texel in rows.chunks_exact_mut(4) {
            texel[3] = 255;
        }
        let at = 100 * pitch + 300 * 4;
        rows[at] = 9;
        let footprint = D3D12_SUBRESOURCE_FOOTPRINT {
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(28),
            Width: w as u32,
            Height: h as u32,
            Depth: 1,
            RowPitch: pitch as u32,
        };
        let s = block_stats(&rows, 28, &footprint).expect("a plain format");
        assert_eq!((s.blocks, s.blocks_with_colour, s.texels_with_colour), (2, 1, 1));
        // Compressed formats are not counted.
        assert!(block_stats(&rows, 71, &footprint).is_none());
    }
}
