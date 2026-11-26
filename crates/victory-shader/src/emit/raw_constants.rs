//! Constant buffers read as raw bytes (`VICTORY_PORT_CBV=1`, renderer port 5;
//! [`super::super::shader::constants_raw`]), a child module of `lower` so it
//! shares the emitter's state.
//!
//! The console takes a constant buffer anywhere on 16 bytes; a D3D12 constant
//! buffer view begins on 256. The layer therefore copied every constant
//! buffer a draw's tables name, at each submission, into upload memory of its
//! own (some 22,000 a frame in the open world), and every root one that was
//! not on 256 bytes (64 KiB each). A raw view of a buffer begins on 16 bytes
//! and a root SRV on 4, so a program that reads its constant buffers as raw
//! bytes lets the host give the card the title's bytes where they are.
//!
//! What is read where:
//! - a table's constant buffer, and one the program picks from the view heap
//!   at an index it computes (where the host's table of strides marks the
//!   slot as a constant buffer): through the raw buffer view the host makes
//!   in that very slot, at the byte of the read;
//! - a root one: as the root SRV `t<slot>` in `ROOT_CBV_SPACE`, which the host
//!   binds at the title's address like any root SRV (the slot's bias,
//!   `USER_VIEW_BIAS`, covers an address off 16 bytes).
//!
//! A raw read through a constant buffer view has no end on AMD (reads past the
//! view's size reach what lies after it, up to 4 GiB on), which is why the
//! host's slot holds a raw view, never a constant buffer view, while the
//! switch is on. Reads past a raw view's end give zeros, as a constant
//! buffer's do.
//!
//! After avatarmanz/VIctory-7900XTX (GPL-3.0), "Renderer port 5" (`3920dc2`),
//! which keeps the constant buffer view in the slot and the raw view in a
//! twin slot of another range of the heap; this tree has no twin slots and
//! needs none: nothing reads the slot as a constant buffer while the switch is
//! on.

use super::{Attr, Bin, Lower, ResourceType, RootType, Site, Source, V, Why};

impl Lower<'_> {
    /// The handle a constant buffer `src` is read through as raw bytes, or
    /// `None` to read it as a constant buffer, as before (the switch is off,
    /// or `src` is no table's, heap's or root view's).
    ///
    /// A root descriptor gives the root SRV of its parameter; a table's or
    /// the heap's entry the raw buffer view (resource kind 11) at the heap
    /// index it names. Anything else (a descriptor built otherwise, or chosen
    /// between kinds of parameter) stays a constant buffer.
    pub(super) fn constants_raw(&mut self, site: Site, src: &Source) -> Result<Option<V>, Why> {
        if !super::super::shader::constants_raw() {
            return Ok(None);
        }
        if let Source::Built { address, .. } = src {
            let Some(&rp) = self.x.native.layout.get(address.param) else {
                return Ok(None);
            };
            let kind = self.x.native.used.iter().find(|u| u.user_data == rp.user_data).map(|u| u.kind);
            if rp.root == RootType::Descriptor && matches!(kind, Some(ResourceType::Cbv) | None) {
                return Ok(Some(self.s.root_srv_handle(rp.slot)));
            }
            return Ok(None);
        }
        let x = self.x;
        let scalar = &self.scalar;
        let through_heap = src.leaves().iter().all(|leaf| match leaf {
            Source::Loaded { load, .. } | Source::Fetched { load, .. } => scalar
                .get(load)
                .and_then(|a| x.native.layout.get(a.param))
                .is_some_and(|p| matches!(p.root, RootType::Table | RootType::ResourceHeap)),
            _ => false,
        });
        if !through_heap {
            return Ok(None);
        }
        // (Raw buffer, no unordered access: as a table's SRV is read.)
        Ok(Some(self.handle(site, src, false, (11, 0))?))
    }

    /// The `count` dwords from byte `addr` of the raw buffer `h`, one load a
    /// dword: each is bounds-checked alone, so a read that begins in the view
    /// and ends past it gives zeros for the part past, as a constant buffer's
    /// does.
    pub(super) fn raw_dwords(&mut self, h: V, addr: V, count: u32) -> Vec<V> {
        let c = self.cur;
        (0..count)
            .map(|k| {
                let a = if k == 0 {
                    addr
                } else {
                    let off = self.s.m.i32(4 * k);
                    self.s.m.bin(c, Bin::Add, addr, off)
                };
                let mask = self.s.m.i8(1);
                let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
                let r = self.s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, self.s.t.res_ret, 139, &[h, a, u, mask, align]);
                let x = self.s.m.extract(c, r, 0);
                self.as_int(x)
            })
            .collect()
    }
}
