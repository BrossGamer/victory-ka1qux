//! Renderer port #5: the title's constant buffers read by the card where the
//! title has them (`VICTORY_PORT_CBV=1`, off by default; read in [`on`]).
//! After avatarmanz/VIctory-7900XTX (GPL-3.0), "Renderer port 5" (`3920dc2`),
//! worked out again for this tree (see below for where it differs).
//!
//! What the console allows: a constant buffer anywhere on 16 bytes. What
//! D3D12 asks: a constant buffer view on 256. So the layer made every
//! constant buffer a draw's tables name afresh at each submission: a copy of
//! its bytes (out of title memory, on the CPU) into upload memory of its own
//! on 256, and a view of that ([`crate::draw`]'s tables): some 22,000 a
//! frame of the open world, and as many steps of the submitting thread. Every
//! root one not on 256 bytes was copied too (64 KiB each). With renderer port
//! 1 (`VICTORY_PORT_CONSTANTS=1`) the memory they are copied out of is the
//! card's to read in place already.
//!
//! What is done here instead: shaders are translated to read constant buffers
//! as raw bytes (`victory_shader::emit::constants_raw`): a table's, and one
//! picked from the view heap, through the raw buffer view in its slot; a
//! root one as a root SRV. A raw view begins on 16 bytes, a root SRV on 4. So
//! a constant buffer of a table in title memory in place is a raw view of the
//! title's bytes, made in its slot, and nothing is copied or done at
//! submission; a root one is bound as root SRVs are, from 16 bytes below its
//! address with the bias the shader adds. Constant buffers anywhere else are
//! copied as before, and their slot is a raw view of the copy.
//!
//! Where it differs from avatarmanz's: his tree keeps the constant buffer
//! view in the slot and makes the raw view in a twin of the slot, in another
//! range of a view heap of 2^22 views (made for the buffers shaders pick by an
//! index, a change of its own: 128 MiB a heap, twice the null-filling at the
//! start). This tree has no twins and needs none for this: with the switch on
//! no shader reads a slot as a constant buffer (the table's and the heap's
//! are read raw, the root ones as root SRVs), so every slot that held a
//! constant buffer view holds a raw view of the same bytes instead, which has
//! an end where a raw read through a constant buffer view has none (AMD).
//!
//! What it assumes: what the card reads of a constant buffer at a draw is
//! what the title's CPU wrote before it submitted, as for every other title
//! buffer in place (renderer port 1). Before, the copy was taken at
//! submission whatever the title wrote after; now the card reads the ring
//! when the draw runs. The title writes its constants into a ring it waits on
//! its fences for.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether renderer port #5 is on: the shaders' side
/// (`victory_shader::emit::constants_raw`) and the host's views follow the
/// same switch. It needs `VICTORY_CONSTANT_MARKS` (on by default): without the
/// marks a slot the program picks cannot be told a constant buffer from a
/// texture's descriptor, and the switch is ignored.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = victory_shader::emit::constants_raw();
        if on {
            tracing::info!(
                "renderer port 5 (VICTORY_PORT_CBV=1): constant buffers are read as raw bytes; a table's constant buffer in title memory in place \
                 is given to the card where it is (VICTORY_PORT_CBV_PLACE=0: copied as before and read raw all the same)"
            );
        } else if std::env::var("VICTORY_PORT_CBV").is_ok_and(|v| v.trim() == "1") {
            tracing::warn!("VICTORY_PORT_CBV=1 is ignored: it needs VICTORY_CONSTANT_MARKS (on by default; it is 0 here)");
        }
        on
    })
}

/// Whether a table's constant buffer in title memory in place is given to the
/// card where it is (on with the port; `VICTORY_PORT_CBV_PLACE=0`: copied as
/// before, and read through the raw view of the copy all the same).
pub(crate) fn in_place_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    on() && *ON.get_or_init(|| std::env::var_os("VICTORY_PORT_CBV_PLACE").is_none_or(|v| v != "0"))
}

/// Whether a raw view can begin `offset` bytes into its buffer: on 16 bytes.
pub(crate) fn raw_ok(offset: u64) -> bool {
    offset.is_multiple_of(16)
}

/// The most bytes a raw view of a constant buffer of `size` bytes reads: what
/// the constant buffer view read, whole 256-byte rows up to 64 KiB (the view
/// of D3D12 is rounded so; the console's has no rounding but the bytes after
/// the size are the title's own, as they were).
pub(crate) fn view_bytes(size: u32) -> u64 {
    const LARGEST: u32 = 64 << 10;
    u64::from(size.min(LARGEST).next_multiple_of(256))
}

// Since the last line: tables' constant buffers given in place, their bytes;
// copied as before because their memory is not in place, because they are not
// on 16 bytes of it; title slots' constant buffer records made as raw views;
// root constant buffers bound as root SRVs; constant buffer records the
// mirroring met (a new record in a slot of the title's heap), and those of
// them that named the view the slot showed already.
const IN_PLACE: usize = 0;
const IN_PLACE_BYTES: usize = 1;
const COPIED: usize = 2;
const UNALIGNED: usize = 3;
const SLOTS: usize = 4;
const ROOTS: usize = 5;
const MIRRORED: usize = 6;
const SAME: usize = 7;
static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// A table's constant buffer of `size` bytes was given in place.
pub(crate) fn in_place(size: u32) {
    STATS[IN_PLACE].fetch_add(1, Relaxed);
    STATS[IN_PLACE_BYTES].fetch_add(u64::from(size), Relaxed);
}

/// A table's constant buffer was copied as before; `unaligned` when it is in
/// place but not on 16 bytes of its buffer.
pub(crate) fn copied(unaligned: bool) {
    STATS[if unaligned { UNALIGNED } else { COPIED }].fetch_add(1, Relaxed);
}

/// A title slot's constant buffer record was made a raw view.
pub(crate) fn slot_made() {
    STATS[SLOTS].fetch_add(1, Relaxed);
}

/// A root constant buffer was bound as a root SRV.
pub(crate) fn root_bound() {
    STATS[ROOTS].fetch_add(1, Relaxed);
}

/// The mirroring met a constant buffer record (a new one in a slot of the
/// heap).
pub(crate) fn mirrored() {
    STATS[MIRRORED].fetch_add(1, Relaxed);
}

/// ... and the slot showed the view it names already: nothing was made.
pub(crate) fn mirrored_same() {
    STATS[SAME].fetch_add(1, Relaxed);
}

/// For the frame-rate line (with the port on).
pub(crate) fn log(frames: f64) {
    if !on() {
        return;
    }
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Relaxed) as f64 / frames.max(1e-9)).collect();
    tracing::info!(
        "renderer port 5 (VICTORY_PORT_CBV=1) per frame: {:.0} tables' constant buffers read by the card in place ({:.1} KB not copied, no step at \
         submission); copied as before: {:.0} not in place, {:.0} in place but not on 16 bytes; {:.0} title slots' constant buffer records made raw \
         views; {:.0} root constant buffers bound as root SRVs; {:.0} constant buffer records mirrored into slots of the title's heap, \
         {:.0} of them the view the slot showed already",
        s[IN_PLACE],
        s[IN_PLACE_BYTES] / 1024.0,
        s[COPIED],
        s[UNALIGNED],
        s[SLOTS],
        s[ROOTS],
        s[MIRRORED],
        s[SAME],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_views_need_sixteen_bytes() {
        assert!(raw_ok(0x3500_0010));
        assert!(raw_ok(0x3500_0100));
        assert!(!raw_ok(0x3500_0008));
        assert!(!raw_ok(0x3500_0004));
    }

    #[test]
    fn a_view_reads_what_the_constant_buffer_view_did() {
        assert_eq!(view_bytes(480), 512);
        assert_eq!(view_bytes(64), 256);
        assert_eq!(view_bytes(64 << 10), 64 << 10);
        assert_eq!(view_bytes(u32::MAX), 64 << 10);
    }
}
