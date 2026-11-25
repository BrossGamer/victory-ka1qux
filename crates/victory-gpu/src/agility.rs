//! The DirectX 12 Agility SDK: the two symbols by which the D3D12 loader
//! finds a newer runtime (`D3D12Core.dll`) in `D3D12\` beside the
//! executable instead of using the system's.
//!
//! On Windows 10 the system's runtime stops at shader model 6.5 and has no
//! mesh shaders; translated shaders need 6.6 (directly indexed heaps) and
//! the world mesh shaders. On an Intel Arc A770 with the system runtime,
//! every root signature with directly indexed heaps was refused
//! (`E_INVALIDARG`) and no mesh pipeline could be made.
//!
//! The runtime itself is Microsoft's redistributable and is never committed
//! (docs/asset-policy.md): `scripts\fetch-agility.ps1` fetches it into
//! `target\`. Without it the loader quietly keeps the system's runtime;
//! [`runtime`] says which one a process got.
//!
//! The build script exports both symbols from every executable it links
//! (`/EXPORT:...,DATA`); the `victory` program's own build script does the
//! same for it.

/// `atoi` at compile time.
const fn parse(text: &str) -> u32 {
    let bytes = text.as_bytes();
    let mut value = 0u32;
    let mut i = 0;
    while i < bytes.len() {
        value = value * 10 + (bytes[i] - b'0') as u32;
        i += 1;
    }
    value
}

/// The SDK version the runtime in `D3D12\` must be (the minor of 1.616.1).
#[allow(non_upper_case_globals)]
#[unsafe(no_mangle)]
#[used]
pub static D3D12SDKVersion: u32 = parse(env!("VICTORY_AGILITY_VERSION"));

/// Where the runtime is, relative to the executable: a `const char*`.
#[allow(non_upper_case_globals)]
#[unsafe(no_mangle)]
#[used]
pub static D3D12SDKPath: &[u8; 9] = b".\\D3D12\\\0";

/// Which D3D12 runtime this process loaded: the path of the
/// `D3D12Core.dll` it has (Agility when it is in our `D3D12\` folder), or
/// `None` before a device is made or with an old system runtime.
pub fn runtime() -> Option<String> {
    // Keep both exported statics referenced in the same object unit as `runtime()`.
    let _ = std::hint::black_box((
        std::ptr::addr_of!(D3D12SDKVersion),
        std::ptr::addr_of!(D3D12SDKPath),
    ));
    use windows_sys::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW};
    let name: Vec<u16> = "D3D12Core.dll\0".encode_utf16().collect();
    // SAFETY: a plain lookup of a loaded module; the buffer is ours.
    unsafe {
        let module = GetModuleHandleW(name.as_ptr());
        if module.is_null() {
            return None;
        }
        let mut path = [0u16; 520];
        let n = GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) as usize;
        Some(String::from_utf16_lossy(&path[..n]))
    }
}

/// Whether [`runtime`] is the Agility SDK's (in a `D3D12` folder beside
/// the executable).
pub fn is_agility(runtime: &str) -> bool {
    let lower = runtime.to_ascii_lowercase();
    !lower.contains("\\system32\\") && !lower.contains("\\syswow64\\")
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_version_is_a_number_and_the_path_a_c_string() {
        assert_eq!(super::parse("616"), 616);
        assert!(super::D3D12SDKVersion >= 600);
        assert_eq!(super::D3D12SDKPath.last(), Some(&0));
        assert!(super::is_agility(r"D:\x\target\debug\D3D12\D3D12Core.dll"));
        assert!(!super::is_agility(r"C:\Windows\System32\D3D12Core.dll"));
    }
}
