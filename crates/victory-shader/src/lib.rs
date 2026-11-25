//! Xbox Series shaders to host shaders.
//!
//! An Xbox Series shader is a standard DXBC container ([`dxbc`]). Its `DXIL`
//! part ([`dxil`]) may carry Xbox additions around the LLVM bitcode: an
//! `XDAT` table before it indexing blobs after it, among them `xbop`
//! (compiler options) and the zlib-compressed GFX10 code. The bitcode may use
//! console-only intrinsics, so the GFX10 code is translated in general.

pub mod bvh;
pub mod dxbc;
pub mod dxil;
pub mod emit;
pub mod gfx10;
pub mod io;
pub mod ir;
pub mod native;
pub mod resource;
pub mod spill;
pub mod structure;

/// The DirectX 12 Agility SDK's symbols for this crate's test executables,
/// which make host devices of their own (as `victory_gpu::agility` does
/// for the program; only one of the two is ever linked into an executable).
#[cfg(all(test, windows))]
mod agility_tests {
    const fn parse(text: &str) -> u32 {
        let bytes = text.as_bytes();
        let (mut value, mut i) = (0u32, 0);
        while i < bytes.len() {
            value = value * 10 + (bytes[i] - b'0') as u32;
            i += 1;
        }
        value
    }
    #[allow(non_upper_case_globals)]
    #[unsafe(no_mangle)]
    #[used]
    pub static D3D12SDKVersion: u32 = parse(env!("VICTORY_AGILITY_VERSION"));
    #[allow(non_upper_case_globals)]
    #[unsafe(no_mangle)]
    #[used]
    pub static D3D12SDKPath: &[u8; 9] = b".\\D3D12\\\0";
}

pub use dxbc::{Container, FourCC, Part};
pub use dxil::{DxilProgram, ShaderKind, XboxExtras};
pub use native::{
    HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource, native_programs,
};

/// Build-time identity of the translator and its compile inputs.
pub const SOURCE_HASH: &str = env!("VICTORY_SHADER_SOURCE_HASH");

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ShaderError {
    #[error("not a DXBC container")]
    NotDxbc,
    #[error("malformed shader: {0}")]
    Malformed(&'static str),
    #[error("GFX10 code at {offset:#x}: {reason}")]
    Isa { offset: usize, reason: &'static str },
    /// Valid code the translator cannot handle yet: an opcode name or a construct.
    #[error("GFX10 code at {offset:#x}: {what} is not supported")]
    Unsupported { offset: usize, what: &'static str },
}
