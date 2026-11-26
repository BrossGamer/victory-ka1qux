//! Stand-ins for texture slots that would read zeros (2D and 2D-array
//! slots; a stand-in of another dimension is a view mismatch the hardware
//! need not survive, so 1D, 3D and cube slots still read zeros).
//!
//! A slot the title wrote a texture view into, for which the host has no
//! texture (not uploaded yet, streamed out, made stale by a newer view,
//! refused), reads zeros: a surface that picks it is black. Two switches
//! change what it reads:
//!
//! - `VICTORY_DEBUG_NULL_VIEWS=1` (builds with diagnostics): a checkerboard
//!   of a colour and black, so that a surface whose texture is missing shows
//!   as a pattern where it would be black, and a surface that stays black
//!   is not a missing texture (it is lit black, or drawn black). The colour
//!   says why the slot has no texture:
//!   - magenta: the title wrote a texture view there and the host has no
//!     texture for it (not uploaded yet, refused, of a kind it cannot make);
//!   - yellow: the memory of the texture it showed was unmapped (streamed
//!     out), and the slot was made null for it;
//!   - orange: a newer view of other memory there made it stale (a texture
//!     streamed into the place of another: the slot is null from then on);
//!   - cyan: no view at all (never written, cleared, or past the host
//!     heap).
//! - `VICTORY_NEUTRAL_NULL_VIEWS=1` (every build; off by default): the first
//!   three read a mid-grey texel in place of zeros, until the texture is
//!   there. Slots with no view at all still read zeros. A hiding of the
//!   problem more than a fix, for a surface that waits for its texture.
//!
//! After victory-fork pull request 687 (`VICTORY_DEBUG_NULL_VIEWS`, by
//! abilliontokens), with the reasons told apart and the neutral texel.

use std::cell::Cell;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

/// Why a slot is made null, as the code that makes it says
/// ([`because`]); [`Why::Written`] when nothing does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Why {
    /// The title wrote a view the host has no texture for.
    Written = 0,
    /// The memory of the texture the slot showed was unmapped.
    Gone = 1,
    /// A newer view of other memory made the slot stale.
    Stale = 2,
}

thread_local! {
    static WHY: Cell<Why> = const { Cell::new(Why::Written) };
}

/// While it lives, null views made by this thread are for `why`.
pub(super) struct Because(Why);

impl Drop for Because {
    fn drop(&mut self) {
        WHY.set(self.0);
    }
}

/// Null views made by this thread until the result is dropped are for
/// `why` ([`Why`]).
pub(super) fn because(why: Why) -> Because {
    Because(WHY.replace(why))
}

/// Stand-ins used, by colour: magenta, yellow, orange, cyan; grey.
static USED: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

const NAMES: [&str; 5] = ["magenta", "yellow", "orange", "cyan", "grey"];

/// What the stand-ins are.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Checkerboards by reason (`VICTORY_DEBUG_NULL_VIEWS=1`).
    Checkers,
    /// One grey texel for every slot with a view (`VICTORY_NEUTRAL_NULL_VIEWS=1`).
    Grey,
}

struct Stands {
    mode: Mode,
    /// By [`Why`], then the one for slots with no view; the grey one is
    /// the first.
    textures: Vec<ID3D12Resource>,
}

// SAFETY: D3D12 resources are free-threaded; these are only read.
unsafe impl Send for Stands {}
// SAFETY: as above.
unsafe impl Sync for Stands {}

/// A texture of `side` times `side` texels of RGBA8, in memory the CPU
/// writes once and shaders read from then on: with `checkers`, `colour`
/// and black in turns, else `colour` alone.
fn texture(device: &ID3D12Device, colour: [u8; 4], side: u32, checkers: bool) -> windows::core::Result<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
        MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(side),
        Height: side,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    let texels: Vec<u8> = (0..side * side)
        .flat_map(|i| if !checkers || (i % side + i / side).is_multiple_of(2) { colour } else { [0, 0, 0, 255] })
        .collect();
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: creation of a CPU-writable texture, then one write of
    // `texels` (`side` rows of `side * 4` bytes) into its only subresource.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            None,
            &mut texture,
        )?;
        let texture = texture.ok_or_else(windows::core::Error::empty)?;
        texture.Map(0, None, None)?;
        let written = texture.WriteToSubresource(0, None, texels.as_ptr().cast(), side * 4, side * side * 4);
        texture.Unmap(0, None);
        written?;
        Ok(texture)
    }
}

/// The stand-ins, if a switch asks for them (made on first use).
fn get() -> Option<&'static Stands> {
    static STANDS: LazyLock<Option<Stands>> = LazyLock::new(|| {
        let on = |name: &str| std::env::var_os(name).is_some_and(|v| v == "1");
        let mode = if cfg!(feature = "diagnostics") && on("VICTORY_DEBUG_NULL_VIEWS") {
            Mode::Checkers
        } else if on("VICTORY_NEUTRAL_NULL_VIEWS") {
            Mode::Grey
        } else {
            return None;
        };
        let device = &host()?.device;
        let made: windows::core::Result<Vec<ID3D12Resource>> = match mode {
            Mode::Checkers => [[255, 0, 255, 255], [255, 255, 0, 255], [255, 128, 0, 255], [0, 255, 255, 255]]
                .into_iter()
                .map(|colour| texture(device, colour, 8, true))
                .collect(),
            Mode::Grey => texture(device, [128, 128, 128, 255], 1, false).map(|t| vec![t]),
        };
        let textures = match made {
            Ok(textures) => textures,
            Err(e) => {
                tracing::warn!(error = %e, "stand-in textures for slots without a texture: none made");
                return None;
            }
        };
        match mode {
            Mode::Checkers => tracing::info!(
                "VICTORY_DEBUG_NULL_VIEWS: texture slots (2D and 2D arrays) that would read zeros read a checkerboard: \
                 magenta where the title wrote a view and the host has no texture for it, yellow where the memory of \
                 the texture it showed was unmapped, orange where a newer view of other memory made it stale, cyan \
                 where there is no view at all; a surface that stays black is not a missing texture"
            ),
            Mode::Grey => tracing::info!(
                "VICTORY_NEUTRAL_NULL_VIEWS: texture slots (2D and 2D arrays) the title wrote a view into and the host \
                 has no texture for read a mid-grey texel until it has"
            ),
        }
        Some(Stands { mode, textures })
    });
    STANDS.as_ref()
}

/// Makes `at` a view of the stand-in for a slot with a view the host cannot
/// show (`known`, by the reason [`because`] set) or with none, if a switch
/// asks for it and `dimension` (the title's SRV dimension) is 2D or a 2D
/// array; whether it did.
pub(super) fn view(known: bool, dimension: u32, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> bool {
    let Some(stands) = get() else { return false };
    let (which, texture) = match stands.mode {
        Mode::Checkers => {
            let which = if known { WHY.get() as usize } else { 3 };
            (which, &stands.textures[which])
        }
        Mode::Grey if known => (4, &stands.textures[0]),
        Mode::Grey => return false,
    };
    let Some(host) = host() else { return false };
    let mut desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        ..Default::default()
    };
    match D3D12_SRV_DIMENSION(dimension as i32) {
        D3D12_SRV_DIMENSION_TEXTURE2D => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            desc.Anonymous.Texture2D = D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() };
        }
        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV { MipLevels: 1, ArraySize: 1, ..Default::default() };
        }
        _ => return false,
    }
    let n = USED[which].fetch_add(1, Ordering::Relaxed) + 1;
    if n == 1 || (n.is_power_of_two() && n >= 1024) {
        tracing::info!(colour = NAMES[which], slots_made_so_far = n, "slots made a stand-in texture");
    }
    // SAFETY: a slot of our heap and a live texture the view fits.
    unsafe { host.device.CreateShaderResourceView(texture, Some(&desc), at) };
    true
}
