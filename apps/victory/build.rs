//! Exports the DirectX 12 Agility SDK's symbols (defined in
//! `victory_gpu::agility`) from the program, where the D3D12 loader looks.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg=/EXPORT:D3D12SDKVersion,DATA");
        println!("cargo:rustc-link-arg=/EXPORT:D3D12SDKPath,DATA");
    }
}
