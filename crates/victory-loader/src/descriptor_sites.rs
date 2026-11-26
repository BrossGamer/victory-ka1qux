//! A title's own stores of descriptor records tell the layer which slots they
//! wrote (`VICTORY_PORT_DESCRIPTORS=1`, read in `victory-gpu`'s
//! `port_descriptors`).
//!
//! After avatarmanz/VIctory-7900XTX ("Renderer port 2", GPL-3.0), whose code
//! this is adapted from. A title that fills the descriptor tables of its draws
//! and dispatches itself does it in a few functions, one 32-byte record at a
//! time: `vmovups [base+index], ymm0`. Each such store is replaced by a jump
//! to code of ours that does the same store and then, if the address lies in
//! the descriptor heap the layer names in [`MARKS`](MARKS_LEN), sets the slot's
//! byte in a map of slots and its chunk's byte in a map of chunks, and jumps
//! back. The layer then looks at the marked slots in place of every slot of
//! every page the write watch says changed.
//!
//! Nothing about a title is built in. *Where* the stores are is told by the
//! user's profile (`apps/victory`'s `descriptor_profile`: byte patterns that
//! must each match exactly once); here only the shape of an instruction is
//! known. A store whose bytes are not the expected form is left alone.
//!
//! What the stub does and does not touch: the store first, as the title had it;
//! then `rax`, `rcx`, `rdx` and the flags are saved on the stack and put back
//! (the flags too, where the original port counted on the code after the store
//! to set them: here the stub does not need that to hold); nothing else.

use std::sync::atomic::Ordering::Relaxed;

use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use crate::sse4a::{STUBS_LEN, jump, near};

/// The cells the stubs read, `#[repr(C)]` in `victory-gpu`: the heap's first
/// byte, its length in bytes (0: none, nothing is marked), the map of slots
/// (a byte each), the map of chunks (a byte for 128 slots).
pub const MARKS_LEN: usize = 32;

/// A store of the title's to hook: its address and its length in bytes (at
/// least the 5 of the jump that replaces it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Store {
    pub at: usize,
    pub len: usize,
}

/// What became of a store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hooked {
    Done,
    /// Left alone, for this reason.
    Failed(&'static str),
}

/// `lea rax, [base+index*scale]` for a `vmovups [base+index*scale], ymm0` (a
/// VEX store, opcode 11h, ModRM 04h with a SIB byte and no displacement, the
/// stack pointer neither its base nor its index): `None` for another form.
fn lea_of(store: &[u8]) -> Option<[u8; 4]> {
    let (x, b, op_at) = match *store.first()? {
        0xc5 => (false, false, 2),
        0xc4 => {
            let p = *store.get(1)?;
            // Only the 0F map (the second byte's low five bits).
            if p & 0x1f != 1 {
                return None;
            }
            (p & 0x40 == 0, p & 0x20 == 0, 3)
        }
        _ => return None,
    };
    let (op, modrm, sib) = (
        *store.get(op_at)?,
        *store.get(op_at + 1)?,
        *store.get(op_at + 2)?,
    );
    if op != 0x11 || modrm != 0x04 || store.len() != op_at + 3 {
        return None;
    }
    // No base (mode 0, base 101: rbp and r13 would take a 32-bit
    // displacement), the stack pointer as base, or as index (index 100
    // without the extension bit is none).
    if sib & 7 == 5 || (sib & 7 == 4 && !b) || (sib >> 3 & 7 == 4 && !x) {
        return None;
    }
    Some([0x48 | u8::from(x) << 1 | u8::from(b), 0x8d, 0x04, sib])
}

/// The code at `to` for the store `store` at `at`: the store, the marks, back
/// to `at + store.len()`. `marks` is the address of the cells.
fn stub(store: &[u8], at: usize, to: usize, marks: usize) -> Option<Vec<u8>> {
    let lea = lea_of(store)?;
    let mut out = store.to_vec();
    out.push(0x50); // push rax
    out.extend(lea); // lea rax, [base+index]
    out.extend([0x52, 0x51, 0x9c]); // push rdx ; push rcx ; pushfq
    out.extend([0x48, 0xba]); // mov rdx, marks
    out.extend((marks as u64).to_le_bytes());
    out.extend([0x48, 0x2b, 0x02]); // sub rax, [rdx]: the offset in the heap
    out.extend([0x48, 0x3b, 0x42, 0x08]); // cmp rax, [rdx+8]
    out.extend([0x73, 0x18]); // jae past the marks (below the heap wraps round)
    out.extend([0x48, 0xc1, 0xe8, 0x05]); // shr rax, 5: the slot
    out.extend([0x48, 0x8b, 0x4a, 0x10]); // mov rcx, [rdx+16]
    out.extend([0xc6, 0x04, 0x01, 0x01]); // mov byte [rcx+rax], 1
    out.extend([0x48, 0xc1, 0xe8, 0x07]); // shr rax, 7: the chunk
    out.extend([0x48, 0x8b, 0x4a, 0x18]); // mov rcx, [rdx+24]
    out.extend([0xc6, 0x04, 0x01, 0x01]); // mov byte [rcx+rax], 1
    out.extend([0x9d, 0x59, 0x5a, 0x58]); // popfq ; pop rcx ; pop rdx ; pop rax
    let back = jump(to + out.len(), at + store.len())?;
    out.extend(back);
    Some(out)
}

/// Memory for the stubs, within reach of the code they serve: the block
/// and how much of it is used.
type Room = Option<(usize, usize)>;

/// The place in `room` for the stub of the store at `at`, a new block near
/// it if this one is full or out of its reach: the stub, the jump to it.
fn place(
    room: &mut Room,
    at: usize,
    store: &[u8],
    marks: usize,
) -> Option<(usize, Vec<u8>, [u8; 5])> {
    let fits = |&(block, used): &(usize, usize)| {
        let to = block + used;
        let body = stub(store, at, to, marks)?;
        let go = jump(at, to)?;
        (used + body.len() <= STUBS_LEN).then_some((to, body, go))
    };
    if room.as_ref().and_then(fits).is_none() {
        *room = near(at).map(|block| (block, 0));
    }
    let (to, body, go) = room.as_ref().and_then(fits)?;
    if let Some((_, used)) = room.as_mut() {
        *used += body.len().next_multiple_of(16);
    }
    Some((to, body, go))
}

/// Hooks `stores`; the stubs read the cells at `marks` ([`MARKS_LEN`] bytes).
/// What became of each, said in the log too. The code is changed as the
/// rewriting of wait instructions changes it: under the same lock, the pages
/// writable for the write alone and put back, a failed protection call
/// remembered for good.
///
/// # Safety
/// Each store is the bytes of an instruction in code that stays mapped and
/// that no thread runs yet (or, if one does, is not inside those bytes); no
/// other code jumps into the middle of a store; `marks` stays valid, and as
/// the cells say, for as long as the code runs.
pub unsafe fn apply(marks: usize, stores: &[Store]) -> Vec<Hooked> {
    let mut room: Room = None;
    let mut out = Vec::new();
    for store in stores {
        let done = (|| {
            if store.len < 5 || store.len > 16 {
                return Hooked::Failed("shorter than a jump, or not an instruction's length");
            }
            // SAFETY: the instruction's bytes, by the caller.
            let bytes =
                unsafe { std::slice::from_raw_parts(store.at as *const u8, store.len) }.to_vec();
            if lea_of(&bytes).is_none() {
                return Hooked::Failed("not a store of the expected form");
            }
            if crate::crash::PROTECT_FAILED.load(Relaxed) {
                return Hooked::Failed("Windows would not make code writable earlier");
            }
            let Some((to, body, go)) = place(&mut room, store.at, &bytes, marks) else {
                return Hooked::Failed("no room for code near the title's, or out of reach");
            };
            // One change of the title's code at a time.
            let _alone = crate::crash::CODE_PATCH
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let mut new = vec![0x90; store.len];
            new[..5].copy_from_slice(&go);
            let mut old = 0;
            // SAFETY: our own memory for the stub, which nothing runs yet; then
            // the store's bytes, made writable for the write alone and put
            // back, which nothing runs either (by the caller).
            unsafe {
                std::ptr::copy_nonoverlapping(body.as_ptr(), to as *mut u8, body.len());
                if VirtualProtect(
                    store.at as *const _,
                    store.len,
                    PAGE_EXECUTE_READWRITE,
                    &mut old,
                ) == 0
                {
                    crate::crash::PROTECT_FAILED.store(true, Relaxed);
                    return Hooked::Failed("Windows would not make the code writable");
                }
                std::ptr::copy_nonoverlapping(new.as_ptr(), store.at as *mut u8, store.len);
                VirtualProtect(store.at as *const _, store.len, old, &mut old);
                FlushInstructionCache(GetCurrentProcess(), store.at as *const _, store.len);
            }
            Hooked::Done
        })();
        let at = format_args!("{:#x}", store.at);
        match done {
            Hooked::Done => {
                tracing::info!(%at, "a descriptor record store of the title patched to tell the layer its slot (VICTORY_PORT_DESCRIPTORS)")
            }
            Hooked::Failed(why) => {
                tracing::warn!(%at, why, "a descriptor record store of the title not patched")
            }
        }
        out.push(done);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, VirtualAlloc};

    /// The marks `victory-gpu` keeps, for a test.
    #[repr(C)]
    struct Marks {
        start: u64,
        len: u64,
        slots: *mut u8,
        chunks: *mut u8,
    }

    /// What the generated function reads and writes: the table and index
    /// values the store's registers get, the record, the flags before it, and
    /// `rax`, `rcx`, `rdx` and the flags before and after the store.
    #[repr(C)]
    #[derive(Default)]
    struct Frame {
        table: u64,
        index: u64,
        record: u64,
        flags: u64,
        before: [u64; 4],
        after: [u64; 4],
    }

    const SENTINELS: [u64; 3] = [
        0x1111_2222_3333_4444,
        0x5555_6666_7777_8888,
        0x0123_4567_89ab_cdef,
    ];

    /// `vmovups [base+index*scale], ymm0`.
    fn vmovups_store(base: u8, index: u8, scale: u8) -> Vec<u8> {
        let sib = scale << 6 | (index & 7) << 3 | (base & 7);
        if base < 8 && index < 8 {
            vec![0xc5, 0xfc, 0x11, 0x04, sib]
        } else {
            let x = if index >= 8 { 0 } else { 0x40 };
            let b = if base >= 8 { 0 } else { 0x20 };
            vec![0xc4, 0x80 | x | b | 1, 0x7c, 0x11, 0x04, sib]
        }
    }

    /// `mov r64, imm64`.
    fn mov_imm(reg: u8, imm: u64) -> Vec<u8> {
        let mut v = vec![0x48 | (reg >> 3), 0xb8 | (reg & 7)];
        v.extend(imm.to_le_bytes());
        v
    }

    /// `mov to, [from+disp]` (`from` neither rsp nor r12: no SIB).
    fn load(to: u8, from: u8, disp: u8) -> Vec<u8> {
        vec![
            0x48 | (to >> 3) << 2 | (from >> 3),
            0x8b,
            0x40 | (to & 7) << 3 | (from & 7),
            disp,
        ]
    }

    /// `mov [to+disp], from`.
    fn save(to: u8, disp: u8, from: u8) -> Vec<u8> {
        vec![
            0x48 | (from >> 3) << 2 | (to >> 3),
            0x89,
            0x40 | (from & 7) << 3 | (to & 7),
            disp,
        ]
    }

    /// `push reg` or `pop reg`.
    fn stack(op: u8, reg: u8) -> Vec<u8> {
        if reg >= 8 {
            vec![0x41, op | (reg & 7)]
        } else {
            vec![op | reg]
        }
    }

    /// The registers of x86-64, by their numbers.
    const RAX: u8 = 0;
    const RCX: u8 = 1;
    const RDX: u8 = 2;

    /// A function `fn(frame)` that loads the record into ymm0, sets rax, rcx
    /// and rdx to sentinels and the store's registers to the frame's values,
    /// loads the flags, notes rax, rcx, rdx and the flags, runs `store`, notes
    /// them again and returns; and where the store is in it.
    fn function(base: u8, index: u8, store: &[u8]) -> (Vec<u8>, usize) {
        // The frame's register: not one the store, or the sentinels, use, and
        // not one that needs a SIB byte as a base.
        let frame = [3u8, 6, 7, 5, 8, 9, 10, 11, 13, 14, 15]
            .into_iter()
            .find(|&r| r != base && r != index)
            .unwrap();
        let mut code = Vec::new();
        for r in [3u8, 5, 6, 7, 12, 13, 14, 15] {
            code.extend(stack(0x50, r));
        }
        // mov frame, rcx
        code.extend([0x48 | (frame >> 3), 0x89, 0xc0 | RCX << 3 | (frame & 7)]);
        code.extend(load(RAX, frame, 16));
        code.extend([0xc5, 0xfc, 0x10, 0x00]); // vmovups ymm0, [rax]
        for (reg, value) in [RAX, RCX, RDX].into_iter().zip(SENTINELS) {
            code.extend(mov_imm(reg, value));
        }
        code.extend(load(base, frame, 0));
        code.extend(load(index, frame, 8));
        // push qword [frame+24] ; popfq
        let rex = if frame >= 8 { vec![0x41] } else { vec![] };
        code.extend(&rex);
        code.extend([0xff, 0x70 | (frame & 7), 24, 0x9d]);
        let note = |code: &mut Vec<u8>, at: u8| {
            code.extend(save(frame, at, RAX));
            code.extend(save(frame, at + 8, RCX));
            code.extend(save(frame, at + 16, RDX));
            code.extend([0x9c]); // pushfq
            code.extend(&rex);
            code.extend([0x8f, 0x40 | (frame & 7), at + 24]); // pop qword [frame+at+24]
        };
        note(&mut code, 32);
        let here = code.len();
        code.extend(store);
        note(&mut code, 64);
        for r in [15u8, 14, 13, 12, 7, 6, 5, 3] {
            code.extend(stack(0x58, r));
        }
        code.extend([0xc5, 0xf8, 0x77, 0xc3]); // vzeroupper ; ret
        (code, here)
    }

    /// Memory of our own to run code from and keep a heap in.
    fn memory(len: usize) -> *mut u8 {
        // SAFETY: fresh memory of our own, never freed (a test).
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_EXECUTE_READWRITE,
            )
        };
        assert!(!p.is_null());
        p.cast()
    }

    #[test]
    fn the_stores_registers_give_the_address_to_mark() {
        // The forms of VEX 2 and 3, with registers of the extended set, and a scale.
        let cases: [(u8, u8, u8, [u8; 4]); 5] = [
            (0, 2, 0, [0x48, 0x8d, 0x04, 0x10]),
            (11, 6, 0, [0x49, 0x8d, 0x04, 0x33]),
            (9, 10, 0, [0x4b, 0x8d, 0x04, 0x11]),
            (3, 13, 0, [0x4a, 0x8d, 0x04, 0x2b]),
            (14, 15, 1, [0x4b, 0x8d, 0x04, 0x7e]),
        ];
        for (base, index, scale, lea) in cases {
            let store = vmovups_store(base, index, scale);
            assert_eq!(lea_of(&store), Some(lea), "{store:02x?}");
        }
        // A displacement, the stack pointer as base or index, no index, a base of
        // rbp or r13 (a displacement), another opcode, a longer instruction:
        // not taken.
        for other in [
            &[0xc5, 0xfc, 0x11, 0x44, 0x31, 0x08][..],
            &[0xc5, 0xfc, 0x11, 0x04, 0x24],
            &[0xc5, 0xfc, 0x11, 0x04, 0x21],
            &[0xc5, 0xfc, 0x11, 0x04, 0x35],
            &[0xc4, 0xc1, 0x7c, 0x11, 0x04, 0x2d],
            &[0xc5, 0xfc, 0x10, 0x04, 0x31],
            &[0xc5, 0xfc, 0x11, 0x04, 0x31, 0x90],
            &[0xc4, 0xe2, 0x7c, 0x11, 0x04, 0x31],
            &[0x90],
        ] {
            assert_eq!(lea_of(other), None, "{other:02x?}");
        }
    }

    /// A store of each form, in a function that notes rax, rcx, rdx and the
    /// flags before and after it, run plain and hooked: the record is stored,
    /// what the function notes is what the plain one notes and what it was
    /// before, the slot and its chunk are marked when the address is in the
    /// heap and no others, and nothing is marked with no heap.
    #[test]
    fn a_hooked_store_stores_marks_and_keeps_the_registers_and_flags() {
        // (base, index, scale): VEX 2 and 3, the registers the stub borrows
        // among them, scales.
        let forms: [(u8, u8, u8); 14] = [
            (0, 1, 0),
            (1, 2, 0),
            (2, 0, 0),
            (3, 6, 0),
            (6, 7, 0),
            (7, 14, 0),
            (8, 9, 0),
            (11, 12, 0),
            (12, 3, 0),
            (3, 12, 0),
            (12, 6, 0),
            (1, 6, 1),
            (9, 8, 3),
            (15, 14, 2),
        ];
        let heap = memory(0x10000);
        let record: [u8; 32] = std::array::from_fn(|k| k as u8 ^ 0x5a);
        // In the heap at slot 0x40 + 3 of its start.
        let slot = 0x40 + 3;
        // SAFETY: the heap is ours, 0x10000 bytes.
        let target = unsafe { heap.add(0x2000 + slot * 32) };
        for (base, index, scale) in forms {
            let store = vmovups_store(base, index, scale);
            let (code, here) = function(base, index, &store);
            for flags in [0x202u64, 0xad7, 0x203, 0x802, 0x8d5] {
                for heap_len in [0x8000u64, 0] {
                    let (plain, hooked) = (memory(0x1000), memory(0x1000));
                    let mut slots = vec![0u8; 0x8000 / 32];
                    let mut chunks = vec![0u8; 0x8000 / 32 / 128 + 1];
                    let marks = Box::new(Marks {
                        start: heap as u64 + 0x2000,
                        len: heap_len,
                        slots: slots.as_mut_ptr(),
                        chunks: chunks.as_mut_ptr(),
                    });
                    // The address is base + index << scale: the index value is shifted back.
                    let (table, index_value) = (target as u64 - 0x60, 0x60u64 >> scale);
                    // SAFETY: memory of our own, written, one copy changed before it runs, then run.
                    unsafe {
                        std::ptr::copy_nonoverlapping(code.as_ptr(), plain, code.len());
                        std::ptr::copy_nonoverlapping(code.as_ptr(), hooked, code.len());
                        let stores = [Store {
                            at: hooked as usize + here,
                            len: store.len(),
                        }];
                        assert_eq!(
                            apply(&raw const *marks as usize, &stores),
                            [Hooked::Done],
                            "{store:02x?}"
                        );
                        assert_eq!(*hooked.add(here), 0xe9);
                        let run = |f: *mut u8| {
                            let mut frame = Frame {
                                table,
                                index: index_value,
                                record: record.as_ptr() as u64,
                                flags,
                                ..Default::default()
                            };
                            std::ptr::write_bytes(target, 0, 32);
                            let f: extern "system" fn(*mut Frame) = std::mem::transmute(f);
                            f(&mut frame);
                            assert_eq!(
                                std::slice::from_raw_parts(target, 32),
                                &record,
                                "stored: {store:02x?}"
                            );
                            frame
                        };
                        let (seen_plain, seen) = (run(plain), run(hooked));
                        assert_eq!(
                            seen.after, seen_plain.after,
                            "registers and flags as plain: {store:02x?}, flags {flags:#x}"
                        );
                        assert_eq!(
                            seen.before[..3],
                            seen.after[..3],
                            "rax, rcx, rdx kept: {store:02x?}"
                        );
                        assert_eq!(
                            seen.before[3] & 0x8d5,
                            seen.after[3] & 0x8d5,
                            "flags kept: {store:02x?}, {flags:#x}"
                        );
                    }
                    if heap_len == 0 {
                        assert!(
                            slots.iter().chain(&chunks).all(|&b| b == 0),
                            "no heap, nothing marked"
                        );
                    } else {
                        assert_eq!(slots[slot], 1, "{store:02x?}");
                        assert_eq!(slots.iter().filter(|&&b| b != 0).count(), 1);
                        assert_eq!(chunks[slot / 128], 1);
                        assert_eq!(chunks.iter().filter(|&&b| b != 0).count(), 1);
                    }
                    drop(marks);
                }
            }
        }
    }

    /// A store of the title's before the heap, or after it, is stored and not marked.
    #[test]
    fn a_store_outside_the_heap_is_not_marked() {
        let heap = memory(0x10000);
        let record: [u8; 32] = std::array::from_fn(|k| k as u8 ^ 0x33);
        let store = vmovups_store(1, 6, 0);
        let (code, here) = function(1, 6, &store);
        let hooked = memory(0x1000);
        let mut slots = vec![0u8; 0x1000 / 32];
        let mut chunks = vec![0u8; 0x1000 / 32 / 128 + 1];
        let marks = Box::new(Marks {
            start: heap as u64 + 0x4000,
            len: 0x1000,
            slots: slots.as_mut_ptr(),
            chunks: chunks.as_mut_ptr(),
        });
        // SAFETY: memory of our own, as above.
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), hooked, code.len());
            assert_eq!(
                apply(
                    &raw const *marks as usize,
                    &[Store {
                        at: hooked as usize + here,
                        len: store.len()
                    }]
                ),
                [Hooked::Done]
            );
            let f: extern "system" fn(*mut Frame) = std::mem::transmute(hooked);
            for target in [
                heap.add(0x3fe0),
                heap.add(0x5000),
                heap.add(0x6000),
                heap.add(0x7000),
            ] {
                let mut frame = Frame {
                    table: target as u64,
                    index: 0,
                    record: record.as_ptr() as u64,
                    flags: 0x202,
                    ..Default::default()
                };
                f(&mut frame);
                assert_eq!(std::slice::from_raw_parts(target, 32), &record);
                assert!(
                    slots.iter().chain(&chunks).all(|&b| b == 0),
                    "outside the heap: {target:p}"
                );
            }
            // The first and the last slots of the heap are in it.
            for (target, slot) in [
                (heap.add(0x4000), 0usize),
                (heap.add(0x4000 + 0x1000 - 32), 0x1000 / 32 - 1),
            ] {
                let mut frame = Frame {
                    table: target as u64,
                    index: 0,
                    record: record.as_ptr() as u64,
                    flags: 0x202,
                    ..Default::default()
                };
                f(&mut frame);
                assert_eq!(slots[slot], 1);
                assert_eq!(chunks[slot / 128], 1);
                slots[slot] = 0;
                chunks[slot / 128] = 0;
            }
        }
        drop(marks);
    }

    #[test]
    fn other_bytes_are_left_alone() {
        let place = memory(0x1000);
        let mut bytes = vmovups_store(1, 6, 0);
        bytes[3] = 0x44; // a displacement: another form
        bytes.push(0x08);
        let mut padded = bytes.clone();
        padded.resize(8, 0x90);
        // SAFETY: memory of our own; nothing runs it.
        unsafe {
            std::ptr::copy_nonoverlapping(padded.as_ptr(), place, padded.len());
            let at = place as usize;
            let out = apply(
                0,
                &[
                    Store { at, len: 6 },
                    Store { at, len: 4 },
                    Store { at, len: 40 },
                ],
            );
            assert!(
                out.iter().all(|h| matches!(h, Hooked::Failed(_))),
                "{out:?}"
            );
            assert_eq!(std::slice::from_raw_parts(place, padded.len()), &padded[..]);
        }
    }
}
