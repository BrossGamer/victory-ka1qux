//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees, `external` host
//! libraries recording into title lists. Planned: `present` (swap chain).

#[cfg(all(windows, feature = "diagnostics"))]
mod bake_probe;
#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
#[cfg(all(windows, feature = "diagnostics"))]
mod renderdoc;
#[cfg(windows)]
mod cardmeter;
#[cfg(windows)]
mod compose;
pub mod console;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod ambient;
#[cfg(windows)]
mod awake;
#[cfg(windows)]
mod drvwatch;
#[cfg(windows)]
mod fpsbox;
#[cfg(windows)]
mod spread;
#[cfg(windows)]
mod draw;
pub mod dxgi;
#[cfg(windows)]
pub mod external;
pub mod fence;
pub mod frame;
#[cfg(windows)]
pub mod framelog;
#[cfg(all(windows, feature = "diagnostics"))]
mod framegen;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
#[cfg(windows)]
mod lazy;
#[cfg(windows)]
mod inplace;
#[cfg(windows)]
mod unified;
#[cfg(windows)]
mod port;
#[cfg(windows)]
mod material_demand;
pub mod list;
pub mod object;
pub mod pipeline;
#[cfg(windows)]
mod cores;
#[cfg(windows)]
mod heapprint;
#[cfg(windows)]
mod plaincopy;
#[cfg(windows)]
mod primitive;
#[cfg(all(windows, feature = "diagnostics"))]
mod profile;
#[cfg(all(test, windows))]
mod unified_probe;
#[cfg(all(test, windows))]
mod card_commit_probe;
#[cfg(all(test, windows))]
mod gpu_upload_probe;
pub mod query;
pub mod queue;
#[cfg(windows)]
mod queue2;
pub mod resource;
pub mod root;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod marks;
#[cfg(windows)]
mod memory;
#[cfg(windows)]
mod shadow;
#[cfg(windows)]
mod window_upload;
#[cfg(windows)]
mod tiles;
pub mod signature;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
mod watch;

#[cfg(windows)]
pub use host::{add_frame_line_source, on_failure};
#[cfg(all(windows, feature = "diagnostics"))]
pub use bake_probe::{
    Writer as BakeProbeWriter, dstorage_tag as bake_probe_dstorage_tag, note as bake_probe_note,
    note_signal as bake_probe_dstorage_signal, probe_on as bake_probe_on,
};
#[cfg(all(windows, feature = "diagnostics"))]
pub use copystat::add_memory_lines;
#[cfg(windows)]
pub use inplace::{title_memory_freeing, title_memory_section, title_memory_view};
#[cfg(windows)]
pub use unified::{in_place_on, title_memory_change};
#[cfg(windows)]
pub use port::on as port_constants_on;
#[cfg(windows)]
pub use lazy::{kernel_will_read, lazy_back_on, title_memory_commit, title_memory_release, title_memory_reserved};
#[cfg(windows)]
pub use watch::{
    Writing, title_memory_reading, title_memory_reading_now, title_memory_reading_range,
    title_memory_remapped, title_memory_remapping, title_memory_remapping_range, title_memory_written,
    title_pages_changed, title_pages_told, writing_async as kernel_will_write_async, writing_held as kernel_will_write,
};

/// Tests that map, commit and free memory of their own, and look at tables
/// the layer keeps by address (write watching, windows of title memory, the
/// runs found in private memory), take this for their whole run: memory one
/// of them frees is memory another is given at the same address, with what
/// the first left in those tables still there, and the two tests found each
/// other's guards and windows when the whole suite ran at once.
#[cfg(all(test, windows))]
pub(crate) fn address_space() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}
