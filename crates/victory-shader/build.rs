//! Fingerprint the inputs to the shader translator for its disk cache.

use std::path::{Path, PathBuf};
use std::process::Command;

const MANIFESTS: [&str; 4] = [
    "Cargo.toml",
    "build.rs",
    "../../Cargo.toml",
    "../../Cargo.lock",
];

fn main() {
    println!("cargo:rerun-if-changed=src");
    for path in MANIFESTS {
        println!("cargo:rerun-if-changed={path}");
    }
    println!("cargo:rerun-if-env-changed=RUSTC");

    // Cargo runs build scripts in the crate directory. Hash only relative
    // file names so identical builds can share a cache across checkouts.
    let mut files: Vec<_> = MANIFESTS.into_iter().map(PathBuf::from).collect();
    sources(Path::new("src"), &mut files);
    let mut files: Vec<_> = files
        .into_iter()
        .map(|p| (p.to_string_lossy().replace('\\', "/"), p))
        .collect();
    files.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let mut hash = blake3::Hasher::new_derive_key("VIctory shader translator build identity v1");
    let compiler = Command::new(std::env::var_os("RUSTC").expect("Cargo provides RUSTC"))
        .arg("-vV")
        .output()
        .expect("query rustc version");
    assert!(compiler.status.success(), "rustc -vV failed");
    field(&mut hash, "rustc", &compiler.stdout);
    for name in ["TARGET", "CARGO_CFG_TARGET_FEATURE"] {
        let value = std::env::var(name).unwrap_or_default();
        field(&mut hash, name, value.as_bytes());
    }
    let mut features: Vec<_> = std::env::vars_os()
        .filter_map(|(name, _)| {
            let name = name.into_string().ok()?;
            name.starts_with("CARGO_FEATURE_").then_some(name)
        })
        .collect();
    features.sort_unstable();
    for feature in features {
        field(&mut hash, "feature", feature.as_bytes());
    }
    let mut switches = std::collections::BTreeSet::new();
    for (name, path) in files {
        let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("read {name}: {error}"));
        field(&mut hash, &name, &bytes);
        if name.ends_with(".rs") {
            read_switches(&String::from_utf8_lossy(&bytes), &mut switches);
        }
    }
    println!(
        "cargo:rustc-env=VICTORY_SHADER_SOURCE_HASH={}",
        hash.finalize().to_hex()
    );
    // Every `VICTORY_*` switch the translator reads: their values join the
    // cache key at run time (`emit::translation_fingerprint`).
    let switches: Vec<_> = switches.into_iter().collect();
    println!(
        "cargo:rustc-env=VICTORY_SHADER_SWITCHES={}",
        switches.join(",")
    );
    agility();
}

/// The test executables run translated shaders on the host GPU: they load
/// the DirectX 12 Agility SDK as `victory-gpu`'s do (`src/agility_tests.rs`;
/// `scripts\fetch-agility.ps1` puts the runtime beside them). This crate
/// links no other executable.
fn agility() {
    println!("cargo:rerun-if-env-changed=VICTORY_AGILITY_VERSION");
    let version = std::env::var("VICTORY_AGILITY_VERSION").unwrap_or_else(|_| "616".to_owned());
    println!("cargo:rustc-env=VICTORY_AGILITY_VERSION={}", version.trim());
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg=/EXPORT:D3D12SDKVersion,DATA");
        println!("cargo:rustc-link-arg=/EXPORT:D3D12SDKPath,DATA");
    }
}

/// The quoted `"VICTORY_..."` names in a source file.
fn read_switches(text: &str, out: &mut std::collections::BTreeSet<String>) {
    let mut rest = text;
    while let Some(at) = rest.find("\"VICTORY_") {
        let tail = &rest[at + 1..];
        let end = tail
            .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
            .unwrap_or(tail.len());
        if tail[end..].starts_with('"') {
            out.insert(tail[..end].to_string());
        }
        rest = &tail[end..];
    }
}

fn field(hash: &mut blake3::Hasher, name: &str, value: &[u8]) {
    for bytes in [name.as_bytes(), value] {
        hash.update(&(bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
}

fn sources(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in
        std::fs::read_dir(dir).unwrap_or_else(|error| panic!("read {}: {error}", dir.display()))
    {
        let entry = entry.expect("read source entry");
        let path = entry.path();
        if entry.file_type().expect("read source file type").is_dir() {
            sources(&path, files);
        } else {
            files.push(path);
        }
    }
}
