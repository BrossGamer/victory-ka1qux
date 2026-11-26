//! Windows of title memory outside buffers, as one buffer in a GPU upload
//! heap (`VICTORY_WINDOW_UPLOAD=1`; off by default).
//!
//! A window on the card (`VICTORY_CARD_WINDOWS_MB`) is two buffers: the CPU's
//! side in the PC's memory, which takes title memory, and the card's side,
//! which a list of ours brings up to date before each submission. With the
//! card's memory mapped by the CPU through a resizable BAR (a "GPU upload
//! heap": `GPUUploadHeapSupported`, feature options 16) a window the CPU
//! fills and shaders only read needs one buffer, in the card's memory, and
//! nothing sends it there: title memory is copied into it, and the card
//! reads it at the card's own speed instead of over the bus.
//!
//! The memory is write-combined: the CPU writes it at about 10 GB/s and reads
//! it at 0.04 GB/s (measured on the RX 7800 XT by `gpu_upload_probe`). So
//! only windows that nothing on the CPU reads are made there: windows asked
//! for by root constant buffer and shader resource views and by vertex and
//! index streams (reads only), never one a shader writes (what the CPU
//! compares, snapshots and copies back would be read from the card's
//! memory). A window in such a heap that is asked for by something that may
//! write is taken out of the table of windows, and a plain one takes its
//! place (`crate::shadow`, `loose`); and a write declared on one all the
//! same is counted and said once.
//!
//! The windows share the budget of `VICTORY_CARD_WINDOWS_MB` and are made
//! while the card has room (`crate::shadow`, `window_card_room`); with this
//! switch the texture queue keeps the part of that budget not used yet free
//! ([`crate::shadow::card_room_kept`]). The buffers of windows let go are
//! kept, a few of them, to be used again: making one costs commit that is
//! given back only when it is released (about 120 MB a heap in the probe).
//!
//! After fxceless/victory-fork 79cfdd96 (`VICTORY_WINDOW_UPLOAD`), with the
//! read-only rule, the pool and the room kept added here.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use crate::host::Host;

/// Whether the host device has GPU upload heaps (asked when it is made).
static SUPPORTED: AtomicBool = AtomicBool::new(false);

/// Whether `VICTORY_WINDOW_UPLOAD=1` was set.
fn wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOW_UPLOAD").is_some_and(|v| v == "1"))
}

/// Whether windows may go to GPU upload heaps: asked for, and the device has
/// them.
pub(crate) fn enabled() -> bool {
    wanted() && SUPPORTED.load(Ordering::Relaxed)
}

/// Asked when the host device is made: whether it has GPU upload heaps. Said
/// in the log when the switch is on.
pub(crate) fn note_device(device: &ID3D12Device) {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    // SAFETY: a plain feature query into a structure of its size.
    let asked = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS16,
            (&raw mut options).cast(),
            size_of_val(&options) as u32,
        )
    };
    let supported = asked.is_ok() && options.GPUUploadHeapSupported.as_bool();
    SUPPORTED.store(supported, Ordering::Relaxed);
    if wanted() {
        tracing::info!(
            supported,
            "GPU upload heaps (the card's memory written by the CPU through a resizable BAR; \
             VICTORY_WINDOW_UPLOAD=1 puts windows outside buffers that shaders only read there)"
        );
        crate::host::add_frame_line_source(line);
    }
}

/// Buffers kept for windows to come, by their size, up to a number of
/// bytes.
struct Pool<T> {
    kept: Vec<(u64, T)>,
    bytes: u64,
    most: u64,
}

impl<T> Pool<T> {
    const fn new(most: u64) -> Self {
        Pool { kept: Vec::new(), bytes: 0, most }
    }

    /// A kept buffer of exactly `size` bytes.
    fn take(&mut self, size: u64) -> Option<T> {
        let at = self.kept.iter().position(|(n, _)| *n == size)?;
        self.bytes -= size;
        Some(self.kept.swap_remove(at).1)
    }

    /// Keeps `what` of `size` bytes, if the pool has room for it; else hands
    /// it back.
    fn give(&mut self, size: u64, what: T) -> Result<(), T> {
        if self.bytes + size > self.most {
            return Err(what);
        }
        self.bytes += size;
        self.kept.push((size, what));
        Ok(())
    }
}

/// The most bytes of buffers kept for windows to come.
const POOL_MOST: u64 = 64 << 20;

/// A buffer, its mapping.
struct Kept(windows::Win32::Graphics::Direct3D12::ID3D12Resource, usize);

static POOL: Mutex<Pool<Kept>> = Mutex::new(Pool::new(POOL_MOST));

/// Since the last frame line: buffers made, reused from the pool, kept in
/// the pool and released; windows taken out of the table for an ask that may
/// write; writes declared on windows here (which should be none).
static COUNTS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
const MADE: usize = 0;
const REUSED: usize = 1;
const KEPT: usize = 2;
const RELEASED: usize = 3;
const SWAPPED: usize = 4;
const WRITTEN: usize = 5;

/// A kept buffer of `size` bytes, zeroed (it holds what the window before
/// it did; a window's pages that are not copied read as zeros), and where
/// it is mapped.
pub(crate) fn take(size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    let Kept(buffer, data) = POOL.lock().unwrap_or_else(|e| e.into_inner()).take(size)?;
    // SAFETY: the buffer's own mapping, `size` bytes of it, which nothing
    // else holds (it was given back by the one window that had it).
    unsafe { std::ptr::write_bytes(data as *mut u8, 0, size as usize) };
    COUNTS[REUSED].fetch_add(1, Ordering::Relaxed);
    Some((buffer, data as *mut u8))
}

/// A window's buffer is no longer wanted: kept for a window to come if
/// nothing else holds it (a descriptor slot that still shows it would show
/// the next window's bytes through it) and the pool has room. True if kept;
/// then its bytes stay counted with the windows' on the card.
pub(crate) fn give(buffer: &ID3D12Resource, data: *mut u8, size: u64) -> bool {
    if crate::host::references(buffer) != 1 {
        COUNTS[RELEASED].fetch_add(1, Ordering::Relaxed);
        return false;
    }
    let kept = POOL.lock().unwrap_or_else(|e| e.into_inner()).give(size, Kept(buffer.clone(), data as usize));
    COUNTS[if kept.is_ok() { KEPT } else { RELEASED }].fetch_add(1, Ordering::Relaxed);
    kept.is_ok()
}

/// A window in a GPU upload heap was taken out of the table of windows for
/// an ask that may write ([`crate::shadow`]).
pub(crate) fn swapped_for_a_plain_window() {
    COUNTS[SWAPPED].fetch_add(1, Ordering::Relaxed);
}

/// A write was declared on a window in a GPU upload heap: its snapshots and
/// compares would read the card's memory. Counted, and said once.
pub(crate) fn write_declared() {
    COUNTS[WRITTEN].fetch_add(1, Ordering::Relaxed);
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| {
        tracing::warn!(
            "shaders write a window that is in a GPU upload heap: what the CPU compares and copies back is read from the \
             card's memory, slowly (VICTORY_WINDOW_UPLOAD=1 makes such windows only for reads)"
        );
    });
}

/// The frame line: what the switch did since the last one.
fn line(frames: f64) -> Option<String> {
    let [made, reused, kept, released, swapped, written] = [0, 1, 2, 3, 4, 5].map(|i| COUNTS[i].swap(0, Ordering::Relaxed));
    if made + reused + kept + released + swapped + written == 0 {
        return None;
    }
    let pooled = POOL.lock().unwrap_or_else(|e| e.into_inner()).bytes;
    Some(format!(
        "windows in GPU upload heaps (VICTORY_WINDOW_UPLOAD=1), per frame: {:.2} buffers made, {:.2} made from a kept one, \
         {:.2} kept, {:.2} released, {:.2} swapped for a plain window for an ask that may write; shaders' writes declared on \
         them: {written}; {} MB kept now",
        made as f64 / frames,
        reused as f64 / frames,
        kept as f64 / frames,
        released as f64 / frames,
        swapped as f64 / frames,
        pooled >> 20
    ))
}

impl Host {
    /// A buffer in the card's own memory that the CPU maps and writes (a GPU
    /// upload heap) and shaders read: for windows the CPU fills and no
    /// shader writes. A kept one of the size first. `None` where the device
    /// has no such heaps or turns the buffer down (then the caller's usual
    /// buffer). Its flags and state are those of the buffers it stands in
    /// for (unordered access allowed, common): a view of that kind over a
    /// window must still be makeable.
    pub(crate) fn gpu_upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        if !enabled() {
            return None;
        }
        if let Some(kept) = take(size) {
            return Some(kept);
        }
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer; nothing
        // is read through the mapping (the CPU writes only).
        let made = unsafe {
            self.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }), Some(&mut data))
                })
        };
        match (made, buffer) {
            (Ok(()), Some(buffer)) if !data.is_null() => {
                let data = data.cast::<u8>();
                if !works(data, size) {
                    return None;
                }
                COUNTS[MADE].fetch_add(1, Ordering::Relaxed);
                Some((buffer, data))
            }
            (made, _) => {
                // Said once: every later one would say the same.
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(size, result = ?made, "a buffer in a GPU upload heap could not be made or mapped: windows stay in the PC's memory");
                });
                None
            }
        }
    }
}

/// Whether title memory reaches a buffer in a GPU upload heap and reads back
/// the same, tried on the first one made before any is used: a plain copy
/// and the layer's own copy (which goes through the kernel where a page is
/// not mapped) must both write it. If not, no window goes to such a heap in
/// this run.
fn works(data: *mut u8, size: u64) -> bool {
    static WORKS: OnceLock<bool> = OnceLock::new();
    *WORKS.get_or_init(|| {
        let n = (size as usize).min(8192);
        let pattern: Vec<u8> = (0..n).map(|i| (i * 31 + 7) as u8).collect();
        let half = n / 2;
        // SAFETY: the buffer's own mapping, `size` bytes of it, and a source
        // of ours; read back once, which is slow and short.
        let same = unsafe {
            std::ptr::copy_nonoverlapping(pattern.as_ptr(), data, half);
            crate::host::copy_mapped(data.add(half), pattern[half..].as_ptr(), n - half, false);
            std::slice::from_raw_parts(data, n) == pattern.as_slice()
        };
        // SAFETY: as above (left as a fresh buffer is: zeroed).
        unsafe { std::ptr::write_bytes(data, 0, n) };
        if same {
            tracing::info!("a buffer in a GPU upload heap takes a plain copy and the layer's copy and reads back the same: such heaps are used");
        } else {
            tracing::warn!("a buffer in a GPU upload heap did not read back what was written into it: such heaps are not used in this run");
        }
        same
    })
}

#[cfg(test)]
mod tests {
    use super::Pool;

    #[test]
    fn a_pool_keeps_buffers_by_size_up_to_its_bytes() {
        let mut pool: Pool<&str> = Pool::new(10);
        assert_eq!(pool.take(4), None);
        assert!(pool.give(4, "a").is_ok());
        assert!(pool.give(4, "b").is_ok());
        // No room for 4 more: handed back.
        assert_eq!(pool.give(4, "c"), Err("c"));
        // A buffer of another size is not a buffer of this one.
        assert_eq!(pool.take(2), None);
        assert!(matches!(pool.take(4), Some("a" | "b")));
        assert_eq!(pool.bytes, 4);
        // The room it left is room again.
        assert!(pool.give(4, "c").is_ok());
        assert!(pool.take(4).is_some() && pool.take(4).is_some());
        assert_eq!(pool.take(4), None);
        assert_eq!(pool.bytes, 0);
    }
}
