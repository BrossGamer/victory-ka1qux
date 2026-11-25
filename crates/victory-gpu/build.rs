//! Links the DirectX 12 Agility SDK into the executables that make a host
//! device (`src/agility.rs`): the test executables of this crate here; the
//! `victory` program in its own build script.

fn main() {
    println!("cargo:rerun-if-env-changed=VICTORY_AGILITY_VERSION");
    let version = std::env::var("VICTORY_AGILITY_VERSION").unwrap_or_else(|_| "616".to_owned());
    assert!(
        version.trim().parse::<u32>().is_ok(),
        "VICTORY_AGILITY_VERSION must be the SDK's minor number (1.616.1 -> 616)"
    );
    println!("cargo:rustc-env=VICTORY_AGILITY_VERSION={}", version.trim());
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        // MSVC executables export nothing unless told: the D3D12 loader
        // looks these two up in the process's executable.
        println!("cargo:rustc-link-arg=/EXPORT:D3D12SDKVersion,DATA");
        println!("cargo:rustc-link-arg=/EXPORT:D3D12SDKPath,DATA");
    }
}
