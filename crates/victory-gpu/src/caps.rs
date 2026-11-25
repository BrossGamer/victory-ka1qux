//! What the host GPU can do, asked once and said in the log and by
//! `victory gpu-info`: what translated shaders and the world need (shader
//! model 6.6, mesh shaders, directly indexed heaps), what the translation
//! adapts to (wave widths, double precision), and the runtime the process
//! loaded ([`crate::agility`]).
//!
//! Intel Arc in particular runs waves of 8 or 16 lanes as well as 32
//! (`WaveLaneCountMin` 8 on Alchemist, 16 on Battlemage) and has no hardware
//! double-precision shader operations: the translation pins compute programs
//! that use wave operations to the console's width there
//! (`victory_shader::emit::set_host_min_wave`) and sums 64-bit floats across
//! waves in integer arithmetic (`victory_shader::emit::set_host_doubles`).

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

/// The host adapter and device's capabilities.
#[derive(Debug, Clone, Default)]
pub struct Caps {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    /// Dedicated video memory, bytes.
    pub video_memory: u64,
    /// The user-mode driver's version, as Windows shows it.
    pub driver: String,
    /// Highest shader model, as 0x66 for 6.6.
    pub shader_model: u32,
    pub resource_binding_tier: i32,
    /// Double-precision float operations in shaders (Intel Arc has none in hardware).
    pub doubles: bool,
    pub wave_min: u32,
    pub wave_max: u32,
    pub mesh_shader_tier: i32,
    pub raytracing_tier: i32,
    pub depth_bounds: bool,
    pub barycentrics: bool,
    pub atomic64_typed: bool,
    pub atomic64_heap: bool,
    pub enhanced_barriers: bool,
    /// The `D3D12Core.dll` loaded, if any.
    pub runtime: Option<String>,
}

impl Caps {
    /// The vendor by PCI id.
    pub fn vendor_name(&self) -> &'static str {
        match self.vendor {
            0x8086 => "Intel",
            0x1002 => "AMD",
            0x10de => "NVIDIA",
            0x1414 => "Microsoft (software)",
            _ => "unknown",
        }
    }

    /// What the title's world needs and the host lacks; empty when it has it all.
    pub fn missing(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if self.shader_model < 0x66 {
            missing.push(
                "shader model 6.6 (directly indexed heaps): run scripts\\fetch-agility.ps1 and update the driver",
            );
        }
        if self.mesh_shader_tier == 0 {
            missing.push("mesh shaders: run scripts\\fetch-agility.ps1 and update the driver");
        }
        if self.resource_binding_tier < 3 {
            missing.push("resource binding tier 3");
        }
        missing
    }

    /// The lines the log and `gpu-info` print.
    pub fn lines(&self) -> Vec<String> {
        let runtime = match &self.runtime {
            Some(path) if crate::agility::is_agility(path) => format!("Agility SDK ({path})"),
            Some(path) => format!(
                "the system's ({path}); scripts\\fetch-agility.ps1 gets the Agility SDK"
            ),
            None => "the system's (no D3D12Core.dll loaded)".to_owned(),
        };
        vec![
            format!(
                "adapter        {} ({}, {:04x}:{:04x})",
                self.name,
                self.vendor_name(),
                self.vendor,
                self.device
            ),
            format!("video memory   {} MB dedicated", self.video_memory >> 20),
            format!("driver         {}", self.driver),
            format!("d3d12 runtime  {runtime}"),
            format!(
                "shader model   {}.{}",
                self.shader_model >> 4,
                self.shader_model & 0xf
            ),
            format!("binding tier   {}", self.resource_binding_tier),
            format!("doubles        {}", self.doubles),
            format!("wave lanes     {}..{}", self.wave_min, self.wave_max),
            format!("mesh shaders   tier {}", self.mesh_shader_tier),
            format!("ray tracing    tier {:#x}", self.raytracing_tier),
            format!("depth bounds   {}", self.depth_bounds),
            format!("barycentrics   {}", self.barycentrics),
            format!(
                "64-bit atomics typed {}, heap {}",
                self.atomic64_typed, self.atomic64_heap
            ),
            format!("enh. barriers  {}", self.enhanced_barriers),
        ]
    }
}

/// Asks `device` (on `adapter`, if known) what it can do.
pub fn query(device: &ID3D12Device, adapter: Option<&IDXGIAdapter1>) -> Caps {
    let mut caps = Caps {
        runtime: crate::agility::runtime(),
        ..Caps::default()
    };
    if let Some(adapter) = adapter {
        // SAFETY: plain queries filling our structures.
        unsafe {
            if let Ok(desc) = adapter.GetDesc1() {
                let len = desc
                    .Description
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(desc.Description.len());
                caps.name = String::from_utf16_lossy(&desc.Description[..len]);
                caps.vendor = desc.VendorId;
                caps.device = desc.DeviceId;
                caps.video_memory = desc.DedicatedVideoMemory as u64;
            }
            if let Ok(v) = adapter.CheckInterfaceSupport(&IDXGIDevice::IID) {
                let v = v as u64;
                caps.driver = format!(
                    "{}.{}.{}.{}",
                    v >> 48,
                    (v >> 32) & 0xffff,
                    (v >> 16) & 0xffff,
                    v & 0xffff
                );
            }
        }
    }
    // SAFETY (each): `feature` writes `T` of its size.
    fn ask<T: Default>(device: &ID3D12Device, feature: D3D12_FEATURE) -> Option<T> {
        let mut data = T::default();
        unsafe {
            device.CheckFeatureSupport(feature, (&raw mut data).cast(), size_of::<T>() as u32)
        }
        .ok()
        .map(|()| data)
    }
    // The newest model first: an older runtime refuses models it does not know.
    for model in (0x60..=0x69).rev() {
        let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL(model),
        };
        // SAFETY: as above.
        let ok = unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut sm).cast(),
                size_of_val(&sm) as u32,
            )
        };
        if ok.is_ok() {
            caps.shader_model = sm.HighestShaderModel.0 as u32;
            break;
        }
    }
    if let Some(o) = ask::<D3D12_FEATURE_DATA_D3D12_OPTIONS>(device, D3D12_FEATURE_D3D12_OPTIONS) {
        caps.resource_binding_tier = o.ResourceBindingTier.0;
        caps.doubles = o.DoublePrecisionFloatShaderOps.as_bool();
    }
    if let Some(o) = ask::<D3D12_FEATURE_DATA_D3D12_OPTIONS1>(device, D3D12_FEATURE_D3D12_OPTIONS1)
    {
        caps.wave_min = o.WaveLaneCountMin;
        caps.wave_max = o.WaveLaneCountMax;
    }
    if let Some(o) = ask::<D3D12_FEATURE_DATA_D3D12_OPTIONS2>(device, D3D12_FEATURE_D3D12_OPTIONS2)
    {
        caps.depth_bounds = o.DepthBoundsTestSupported.as_bool();
    }
    if let Some(o) = ask::<D3D12_FEATURE_DATA_D3D12_OPTIONS3>(device, D3D12_FEATURE_D3D12_OPTIONS3)
    {
        caps.barycentrics = o.BarycentricsSupported.as_bool();
    }
    if let Some(o) = ask::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>(device, D3D12_FEATURE_D3D12_OPTIONS5)
    {
        caps.raytracing_tier = o.RaytracingTier.0;
    }
    if let Some(o) = ask::<D3D12_FEATURE_DATA_D3D12_OPTIONS7>(device, D3D12_FEATURE_D3D12_OPTIONS7)
    {
        caps.mesh_shader_tier = o.MeshShaderTier.0;
    }
    if let Some(o) = ask::<D3D12_FEATURE_DATA_D3D12_OPTIONS9>(device, D3D12_FEATURE_D3D12_OPTIONS9)
    {
        caps.atomic64_typed = o.AtomicInt64OnTypedResourceSupported.as_bool();
    }
    if let Some(o) =
        ask::<D3D12_FEATURE_DATA_D3D12_OPTIONS11>(device, D3D12_FEATURE_D3D12_OPTIONS11)
    {
        caps.atomic64_heap = o.AtomicInt64OnDescriptorHeapResourceSupported.as_bool();
    }
    if let Some(o) =
        ask::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>(device, D3D12_FEATURE_D3D12_OPTIONS12)
    {
        caps.enhanced_barriers = o.EnhancedBarriersSupported.as_bool();
    }
    caps
}

/// `VICTORY_ADAPTER`: the host adapter by its place in the high-performance
/// order (`0`, `1`, ...) or a piece of its name (`A770`, `B580`, `Arc`);
/// unset, the first hardware adapter in that order, so that a PC with
/// integrated graphics beside an Arc (or any other card) gets the card.
/// `None` when DXGI lists nothing that fits: D3D12 then picks as before.
pub fn pick_adapter() -> Option<IDXGIAdapter1> {
    let wanted = std::env::var("VICTORY_ADAPTER")
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty());
    // SAFETY: plain DXGI enumeration.
    unsafe {
        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        let mut hardware = Vec::new();
        for i in 0.. {
            let Ok(adapter) = factory
                .EnumAdapterByGpuPreference::<IDXGIAdapter1>(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
            else {
                break;
            };
            let Ok(desc) = adapter.GetDesc1() else {
                continue;
            };
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                continue;
            }
            let len = desc
                .Description
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(desc.Description.len());
            hardware.push((String::from_utf16_lossy(&desc.Description[..len]), adapter));
        }
        let picked = match &wanted {
            Some(w) => match w.parse::<usize>() {
                Ok(n) => hardware.into_iter().nth(n),
                Err(_) => {
                    let w = w.to_ascii_lowercase();
                    hardware
                        .into_iter()
                        .find(|(name, _)| name.to_ascii_lowercase().contains(&w))
                }
            },
            None => hardware.into_iter().next(),
        };
        match picked {
            Some((name, adapter)) => {
                tracing::info!(adapter = %name, asked = ?wanted, "host adapter");
                Some(adapter)
            }
            None => {
                tracing::warn!(
                    asked = ?wanted,
                    "no hardware adapter fits VICTORY_ADAPTER; the default one is used"
                );
                None
            }
        }
    }
}

/// Every adapter DXGI lists, in the high-performance order: for `gpu-info`.
pub fn adapters() -> Vec<String> {
    let mut out = Vec::new();
    // SAFETY: plain DXGI enumeration.
    unsafe {
        let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory6>(DXGI_CREATE_FACTORY_FLAGS(0)) else {
            return out;
        };
        for i in 0.. {
            let Ok(adapter) = factory
                .EnumAdapterByGpuPreference::<IDXGIAdapter1>(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
            else {
                break;
            };
            if let Ok(desc) = adapter.GetDesc1() {
                let len = desc
                    .Description
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(desc.Description.len());
                let soft = desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0;
                out.push(format!(
                    "{i}: {}{} ({} MB)",
                    String::from_utf16_lossy(&desc.Description[..len]),
                    if soft { " [software]" } else { "" },
                    desc.DedicatedVideoMemory >> 20
                ));
            }
        }
    }
    out
}

/// The host device's capabilities (made on first use), and the size of
/// its view heaps: what `victory gpu-info` prints.
pub fn report() -> Option<(Caps, u32)> {
    let host = crate::host::host()?;
    crate::descriptor::host::probe_view_limit();
    Some((
        host.caps.clone(),
        victory_shader::emit::view_heap_limit() + 1,
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_host_device_says_what_it_can_do() {
        let Some((caps, views)) = super::report() else {
            return;
        };
        for line in caps.lines() {
            eprintln!("{line}");
        }
        eprintln!("view heap      {views} views; missing: {:?}", caps.missing());
        assert!(caps.shader_model >= 0x60);
        assert!(caps.wave_max >= caps.wave_min);
    }
}
