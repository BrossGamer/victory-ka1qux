//! SSE4a's `extrq` and `insertq` with their field in the instruction, as SSE2
//! (`VICTORY_REWRITE_SSE4A=1`; upstream VIctory #3042, by 101Leafy).
//!
//! Intel CPUs have no SSE4a, so each one faults and `crash` emulates it, about a microsecond
//! an exception, and titles run them in hot loops: hundreds of exceptions a second at a single
//! place. The first fault at a place replaces the instruction's first five bytes by a jump to
//! the same work in SSE2, in memory within reach of the title's code; the bytes left after the
//! jump are never run. The forms with their field in a register, and places where the jump
//! would cross 16 aligned bytes, stay emulated.
//!
//! The code is changed as `VICTORY_REWRITE_WAITS` changes the wait instructions (`crash`'s
//! `rewrite_wait`): one thread at a time (the same lock), the page made writable for the store
//! alone and put back, a failed protection call remembered for good. The address is noted
//! ([`crate::crash::note_patched`]) so that a thread that faulted on the old bytes and reads
//! the new ones is let go on instead of taken for a crash.

use std::sync::Mutex;

/// `extrq xmm, len, idx` (`66 [rex] 0f 78 /0 ib ib`, `dst` the r/m register) or
/// `insertq dst, src, len, idx` (`f2 [rex] 0f 78 /r ib ib`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Field {
    pub(crate) insert: bool,
    pub(crate) dst: u8,
    pub(crate) src: u8,
    /// Bits in the field, 1 to 64, and where it starts.
    pub(crate) len: u32,
    pub(crate) idx: u32,
    /// Bytes in the instruction.
    pub(crate) size: usize,
}

pub(crate) fn decode(code: &[u8]) -> Option<Field> {
    let prefix = *code.first()?;
    let (rex, at) = match *code.get(1)? {
        r @ 0x40..=0x4f => (r, 2),
        _ => (0, 1),
    };
    if !matches!(prefix, 0x66 | 0xf2) || code.get(at..at + 2)? != [0x0f, 0x78] {
        return None;
    }
    let &[modrm, len, idx] = code.get(at + 2..at + 5)? else {
        return None;
    };
    if modrm >> 6 != 3 {
        return None;
    }
    let reg = (modrm >> 3) & 7 | (rex & 4) << 1;
    let rm = modrm & 7 | (rex & 1) << 3;
    let insert = prefix == 0xf2;
    if !insert && reg != 0 {
        return None;
    }
    let len = match u32::from(len & 63) {
        0 => 64,
        n => n,
    };
    let (dst, src) = if insert { (reg, rm) } else { (rm, rm) };
    Some(Field { insert, dst, src, len, idx: u32::from(idx & 63), size: at + 5 })
}

/// `jmp rel32` at `at` to `to`, if it reaches.
pub(crate) fn jump(at: usize, to: usize) -> Option<[u8; 5]> {
    let rel = i32::try_from(to as i64 - (at as i64 + 5)).ok()?;
    let [a, b, c, d] = rel.to_le_bytes();
    Some([0xe9, a, b, c, d])
}

/// The instruction `f` at `at` as SSE2 code placed at `place`, entered 16 bytes in (the first
/// 16 are the bits `insertq` keeps, for its `pand`), then a jump back past it. Only the
/// destination changes, its high quadword zeroed as the emulation does with the switch on;
/// `insertq` borrows a third register, saved on the stack and put back, and changes no flag.
pub(crate) fn stub(f: Field, at: usize, place: usize) -> Option<Vec<u8>> {
    // `prefix [rex] 0f code` with registers `reg` and `rm`.
    fn op(prefix: u8, code: u8, reg: u8, rm: u8) -> Vec<u8> {
        let rex = (reg & 8) >> 1 | (rm & 8) >> 3;
        let mut v = vec![prefix];
        if rex != 0 {
            v.push(0x40 | rex);
        }
        v.extend([0x0f, code, 0xc0 | (reg & 7) << 3 | rm & 7]);
        v
    }
    // psrlq (`down`) or psllq of `x` by `by`, nothing for 0.
    fn shift(down: bool, x: u8, by: u32) -> Vec<u8> {
        if by == 0 {
            return Vec::new();
        }
        let mut v = op(0x66, 0x73, if down { 2 } else { 6 }, x);
        v.push(by as u8);
        v
    }
    let mut code = vec![0; 16];
    let x = f.dst;
    if !f.insert {
        code.extend(shift(true, x, f.idx));
        code.extend(shift(false, x, 64 - f.len));
        code.extend(shift(true, x, 64 - f.len));
        code.extend(op(0xf3, 0x7e, x, x)); // movq x, x
    } else {
        let ones = if f.len == 64 { u64::MAX } else { (1 << f.len) - 1 };
        code[..8].copy_from_slice(&(!(ones << f.idx)).to_le_bytes());
        let t = (0..3).find(|&t| t != f.dst && t != f.src)?;
        code.extend([0x48, 0x8d, 0x64, 0x24, 0xf0]); // lea rsp, [rsp - 16]
        code.extend([0xf3, 0x0f, 0x7f, t << 3 | 4, 0x24]); // movdqu [rsp], t
        code.extend(op(0xf3, 0x7e, t, f.src)); // movq t, src
        code.extend(shift(false, t, 64 - f.len));
        code.extend(shift(true, t, 64 - f.len));
        code.extend(shift(false, t, f.idx));
        // pand dst, [the kept bits]
        code.extend(if x >= 8 { &[0x66, 0x44][..] } else { &[0x66] });
        let disp = -i32::try_from(code.len() + 7).ok()?;
        code.extend([0x0f, 0xdb, (x & 7) << 3 | 5]);
        code.extend(disp.to_le_bytes());
        code.extend(op(0x66, 0xeb, x, t)); // por dst, t
        code.extend([0xf3, 0x0f, 0x6f, t << 3 | 4, 0x24]); // movdqu t, [rsp]
        code.extend([0x48, 0x8d, 0x64, 0x24, 0x10]); // lea rsp, [rsp + 16]
    }
    let end = place + code.len();
    code.extend(jump(end, at + f.size)?);
    Some(code)
}

pub(crate) const STUBS_LEN: usize = 64 << 10;

/// The memory new stubs go in, near the title's code, and the bytes used. Its lock is held
/// while a stub is made and the code changed: one thread at a time.
static STUBS: Mutex<Option<(usize, usize)>> = Mutex::new(None);

/// Replacements made since the start: `extrq`, `insertq`.
pub(crate) static REPLACED: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Memory for code within reach of a `jmp rel32` at `at`.
pub(crate) fn near(at: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc,
    };
    let at = at & !0xffff;
    let mut places = (0x10_0000..0x7000_0000)
        .step_by(0x10_0000)
        .flat_map(|d| [at.checked_sub(d), at.checked_add(d)])
        .flatten();
    places.find_map(|place| {
        let rights = PAGE_EXECUTE_READWRITE;
        // SAFETY: reserves free memory at that address or fails.
        let got = unsafe {
            VirtualAlloc(place as *const _, STUBS_LEN, MEM_COMMIT | MEM_RESERVE, rights)
        };
        (!got.is_null()).then_some(got as usize)
    })
}

/// After the instruction at `at` faulted and was emulated: replaces it by a jump to its
/// [`stub`], if it is an `extrq` or `insertq` with its field in the instruction and the jump
/// fits in its aligned 16 bytes. True if the bytes at `at` are the jump now.
///
/// # Safety
/// `at` is in code that stays mapped, at an instruction that just faulted.
pub(crate) unsafe fn replace(at: usize) -> bool {
    if (at & 15) + 5 > 16 || !std::is_x86_feature_detected!("cmpxchg16b") {
        return false;
    }
    let n = (0x1000 - (at & 0xfff)).min(7);
    // SAFETY: the instruction's bytes, which the CPU just fetched, none past its page.
    let code = unsafe { std::slice::from_raw_parts(at as *const u8, n) };
    if decode(code).is_none() {
        return false;
    }
    let mut stubs = STUBS.lock().unwrap_or_else(|e| e.into_inner());
    // Again under the lock: a second thread's fault at the same place finds the jump.
    // SAFETY: as above.
    let code = unsafe { std::slice::from_raw_parts(at as *const u8, n) };
    let Some(f) = decode(code) else {
        return false;
    };
    let fits = |&(place, used): &(usize, usize)| {
        let body = stub(f, at, place + used)?;
        let jump = jump(at, place + used + 16)?;
        (used + body.len() <= STUBS_LEN).then_some((body, jump))
    };
    // A full block, or one out of reach of this code, is kept for the stubs in it and a new
    // one made.
    if stubs.as_ref().and_then(fits).is_none() {
        *stubs = near(at).map(|place| (place, 0));
    }
    let Some(((body, jump), (place, used))) = stubs.as_mut().and_then(|s| Some((fits(&*s)?, s)))
    else {
        return false;
    };
    let at_stub = *place + *used;
    // SAFETY: memory made for the stubs, past those in use, which nothing runs yet.
    unsafe { std::ptr::copy_nonoverlapping(body.as_ptr(), at_stub as *mut u8, body.len()) };
    *used += body.len().next_multiple_of(16);
    let mut was = [0; 5];
    was.copy_from_slice(&code[..5]);
    // SAFETY: as for this function; the stub does what the instruction does.
    let done = unsafe { patch(at, was, jump) };
    if done {
        REPLACED[usize::from(f.insert)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        crate::crash::note_patched(at);
    }
    done
}

/// Replaces the 5 bytes at `at`, which are `was`, by `jump` in one 16-byte store, so that a
/// thread about to run them sees either all old or all new. True if they are the jump now.
///
/// # Safety
/// `at` is in code that stays mapped, the 5 bytes inside its aligned 16.
unsafe fn patch(at: usize, was: [u8; 5], jump: [u8; 5]) -> bool {
    use std::arch::x86_64::cmpxchg16b;
    use std::sync::atomic::Ordering::{Relaxed, SeqCst};
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    if crate::crash::PROTECT_FAILED.load(Relaxed) {
        return false;
    }
    // One thread changes the title's code at a time: two changing one page's protection would
    // put it back wrong (the wait instructions' rewrite takes the same lock).
    let _alone = crate::crash::CODE_PATCH.lock().unwrap_or_else(|e| e.into_inner());
    let word = (at & !15) as *mut u128;
    let page = (at & !0xfff) as *const core::ffi::c_void;
    let here = (at & 15)..(at & 15) + 5;
    let mut old = 0;
    let mut done = false;
    // SAFETY: the aligned 16 bytes lie in one page of mapped code, made writable for the store
    // and put back; the CPU has cmpxchg16b (the caller asked).
    unsafe {
        if VirtualProtect(page, 0x1000, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            crate::crash::PROTECT_FAILED.store(true, Relaxed);
            return false;
        }
        // What is there (zeros are replaced by zeros).
        let mut now = cmpxchg16b(word, 0, 0, SeqCst, SeqCst);
        loop {
            let mut bytes = now.to_le_bytes();
            if bytes[here.clone()] != was {
                break;
            }
            bytes[here.clone()].copy_from_slice(&jump);
            let there = cmpxchg16b(word, now, u128::from_le_bytes(bytes), SeqCst, SeqCst);
            if there == now {
                done = true;
                break;
            }
            now = there;
        }
        VirtualProtect(page, 0x1000, old, &mut old);
        if done {
            FlushInstructionCache(GetCurrentProcess(), word as *const _, 16);
        }
    }
    done
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Diagnostics::Debug::M128A;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc, VirtualFree,
    };

    /// What the emulation does to the registers `d_in` (in both halves of the destination) and
    /// `s_in` (in both halves of the source), with the high quadword zeroed as the stubs do:
    /// the destination's two halves.
    fn emulated(insn: &[u8], f: Field, d_in: u64, s_in: u64) -> [u64; 2] {
        let reg = |low: u64| M128A { Low: low, High: low as i64 };
        let mut xmm = [reg(0); 16];
        xmm[usize::from(f.dst)] = reg(d_in);
        if f.src != f.dst {
            xmm[usize::from(f.src)] = reg(s_in);
        }
        // `int3`s after the instruction, so that a decoder reading too far sees no valid bytes.
        let mut bytes = [0xcc_u8; 16];
        bytes[..insn.len()].copy_from_slice(insn);
        // SAFETY: the instruction's bytes, which are readable.
        let len = unsafe { crate::crash::sse4a(bytes.as_ptr(), &mut xmm, true) };
        assert_eq!(len, Some(insn.len()), "{f:?}");
        [xmm[usize::from(f.dst)].Low, xmm[usize::from(f.dst)].High as u64]
    }

    /// A function `fn(destination, source, out)` that runs `insn` at `offset` bytes into an
    /// aligned 16: xmm8 and xmm9 are kept for the caller, xmm0-2 all ones, the destination
    /// holds the first value in both halves, the source the second in both halves (the same
    /// register: the second), then the instruction runs, and the destination, the source and
    /// xmm0-2 are stored. Returns the code and where the instruction is in it.
    fn function(f: Field, insn: &[u8], offset: usize) -> (Vec<u8>, usize) {
        let (dst, src) = (f.dst, f.src);
        let (r, low) = (|x: u8| (x & 8) >> 1, |x: u8| (x & 7) << 3);
        let movq = |x: u8, gpr: u8| [0x66, 0x48 | r(x), 0x0f, 0x6e, 0xc0 | low(x) | gpr];
        let both = |x: u8| [0x66, 0x40 | r(x) | r(x) >> 2, 0x0f, 0x6c, 0xc0 | ((x & 7) * 9)];
        let store = |x: u8, at: u8| [0xf3, 0x41 | r(x), 0x0f, 0x7f, 0x40 | low(x), at];
        let mut code = vec![0x48, 0x8d, 0x64, 0x24, 0xd8];
        code.extend([0xf3, 0x44, 0x0f, 0x7f, 0x04, 0x24]);
        code.extend([0xf3, 0x44, 0x0f, 0x7f, 0x4c, 0x24, 0x10]);
        code.extend([0x66, 0x0f, 0x76, 0xc0, 0x66, 0x0f, 0x76, 0xc9, 0x66, 0x0f, 0x76, 0xd2]);
        code.extend(movq(dst, 1));
        code.extend(both(dst));
        if src != dst {
            code.extend(movq(src, 2));
            code.extend(both(src));
        }
        while code.len() % 16 != offset {
            code.push(0x90);
        }
        let here = code.len();
        code.extend(insn);
        code.extend(store(dst, 0));
        code.extend(store(src, 16));
        for x in 0..3 {
            code.extend(store(x, 32 + 16 * x));
        }
        code.extend([0xf3, 0x44, 0x0f, 0x6f, 0x04, 0x24]);
        code.extend([0xf3, 0x44, 0x0f, 0x6f, 0x4c, 0x24, 0x10]);
        code.extend([0x48, 0x8d, 0x64, 0x24, 0x28, 0xc3]);
        (code, here)
    }

    /// An executable page of our own, released at the end.
    struct Page(usize);

    impl Page {
        fn new() -> Page {
            // SAFETY: a fresh page of our own.
            let page = unsafe {
                VirtualAlloc(std::ptr::null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE)
            };
            assert!(!page.is_null());
            Page(page as usize)
        }
    }

    impl Drop for Page {
        fn drop(&mut self) {
            // SAFETY: the page of `new`.
            unsafe { VirtualFree(self.0 as *mut _, 0, MEM_RELEASE) };
        }
    }

    /// Runs `code` (a [`function`]) in `page` twice and checks both passes: the first faults on
    /// a CPU without SSE4a, is emulated and replaced, the second runs the SSE2. Both leave the
    /// destination as the emulation does, every other register as it was. Whether the bytes
    /// of the instruction then are a jump is `replaced`'s to say.
    fn run_twice(page: &Page, code: &[u8], here: usize, insn: &[u8], f: Field, inputs: (u64, u64), replaced: bool) {
        let (d_in, s_in) = inputs;
        let d_in = if f.src == f.dst { s_in } else { d_in };
        let expected = emulated(insn, f, d_in, s_in);
        // SAFETY: memory of our own, written, then run as the function it holds.
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), page.0 as *mut u8, code.len());
            let run: extern "system" fn(u64, u64, *mut [u64; 10]) = std::mem::transmute(page.0);
            for pass in 0..2 {
                let mut out = [0u64; 10];
                run(d_in, s_in, &mut out);
                assert_eq!(out[..2], expected, "{f:?}, pass {pass}");
                if f.src != f.dst {
                    assert_eq!(out[2..4], [s_in, s_in], "source kept: {f:?}, pass {pass}");
                }
                for x in 0..3u8 {
                    if x != f.dst && x != f.src {
                        let at = 4 + 2 * x as usize;
                        assert_eq!(out[at..at + 2], [u64::MAX; 2], "xmm{x} kept: {f:?}, pass {pass}");
                    }
                }
            }
            let now = std::slice::from_raw_parts((page.0 + here) as *const u8, insn.len());
            if replaced {
                assert_eq!(now[0], 0xe9, "a jump: {f:?}");
            } else {
                assert_eq!(now, insn, "left alone: {f:?}");
            }
        }
    }

    /// The instruction `66`/`f2 [rex] 0f 78 modrm len idx` of the registers asked for.
    fn instruction(insert: bool, dst: u8, src: u8, len: u8, idx: u8) -> Vec<u8> {
        let (reg, rm) = if insert { (dst, src) } else { (0, dst) };
        let rex = 0x40 | (reg & 8) >> 1 | (rm & 8) >> 3;
        let mut v = vec![if insert { 0xf2 } else { 0x66 }];
        if rex != 0x40 {
            v.push(rex);
        }
        v.extend([0x0f, 0x78, 0xc0 | (reg & 7) << 3 | rm & 7, len, idx]);
        v
    }

    fn prepare() -> bool {
        crate::crash::report_crashes();
        crate::crash::force_rewrite_sse4a(true);
        !std::is_x86_feature_detected!("sse4a")
    }

    /// A few forms at different places in their aligned 16: each run twice, and the bytes
    /// become a jump only where one fits.
    #[test]
    fn an_extrq_or_insertq_becomes_sse2() {
        let replaced = prepare();
        let (d_in, s_in) = (0x0123_4567_89ab_cdefu64, 0xfedc_ba98_7654_3210u64);
        // The bytes, where in its aligned 16 the instruction lies, and whether a jump fits.
        let cases: [(&[u8], usize, bool); 6] = [
            (&[0xf2, 0x0f, 0x78, 0xc1, 0x10, 0x10], 1, true),
            (&[0x66, 0x0f, 0x78, 0xc2, 0x08, 0x14], 7, true),
            (&[0xf2, 0x45, 0x0f, 0x78, 0xc8, 0x0c, 0x28], 11, true),
            (&[0x66, 0x41, 0x0f, 0x78, 0xc1, 0x00, 0x04], 3, true),
            (&[0xf2, 0x0f, 0x78, 0xc9, 0x05, 0x03], 0, true),
            (&[0xf2, 0x0f, 0x78, 0xc1, 0x10, 0x10], 12, false),
        ];
        let page = Page::new();
        for (insn, offset, fits) in cases {
            let f = decode(insn).unwrap();
            let (code, here) = function(f, insn, offset);
            run_twice(&page, &code, here, insn, f, (d_in, s_in), replaced && fits);
        }
        assert!(decode(&[0xf2, 0x0f, 0x79, 0xc1]).is_none(), "the field in a register");
        assert!(decode(&[0x66, 0x0f, 0x78, 0xca, 0x08, 0x14]).is_none(), "extrq with reg not 0");
    }

    /// Every length and index, for both: the stub gives what the emulation gives (the high
    /// quadword zeroed), with bit patterns that tell a shifted or masked field from another.
    #[test]
    fn the_stubs_do_what_the_emulation_does_for_every_length_and_index() {
        let replaced = prepare();
        let page = Page::new();
        let patterns = [
            (0x0123_4567_89ab_cdefu64, 0xfedc_ba98_7654_3210u64),
            (u64::MAX, 0),
            (0, u64::MAX),
            (0xaaaa_aaaa_aaaa_aaaa, 0x5555_5555_5555_5555),
        ];
        for insert in [false, true] {
            for len in 0..64u8 {
                for idx in 0..64u8 {
                    // The registers and the place vary with the field.
                    let n = usize::from(len) * 64 + usize::from(idx);
                    let regs = [(1, 2), (2, 1), (0, 1), (8, 9), (9, 3), (4, 5), (5, 8), (3, 3)];
                    let (dst, src) = regs[n % regs.len()];
                    let insn = instruction(insert, dst, if insert { src } else { dst }, len, idx);
                    let f = decode(&insn).unwrap();
                    let offset = n % 12;
                    let (code, here) = function(f, &insn, offset);
                    run_twice(&page, &code, here, &insn, f, patterns[n % patterns.len()], replaced);
                }
            }
        }
    }

    /// The same, for the registers the stubs borrow: the destination and the source take each
    /// of xmm0-2 in turn, and the others come back as they were.
    #[test]
    fn insertq_borrows_a_register_that_is_neither_of_its_own() {
        let replaced = prepare();
        let page = Page::new();
        for dst in 0..3u8 {
            for src in 0..3u8 {
                for (len, idx) in [(16, 16), (1, 63), (63, 1), (0, 0), (32, 40), (8, 0)] {
                    let insn = instruction(true, dst, src, len, idx);
                    let f = decode(&insn).unwrap();
                    let (code, here) = function(f, &insn, 2);
                    run_twice(&page, &code, here, &insn, f, (0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210), replaced);
                }
            }
        }
    }

    /// A thread that faulted on the old bytes, and finds the jump when its handler reads them,
    /// is let go on at the same place; at a place nobody changed the same exception is no one's
    /// to take.
    #[test]
    fn a_fault_at_a_replaced_place_is_let_go_on() {
        use windows_sys::Win32::Foundation::EXCEPTION_ILLEGAL_INSTRUCTION;
        use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, EXCEPTION_POINTERS, EXCEPTION_RECORD};
        // A jump where the title's instruction was (not run: only read).
        let mut bytes = [0xcc_u8; 16];
        bytes[..5].copy_from_slice(&[0xe9, 1, 2, 3, 4]);
        let at = bytes.as_ptr() as usize;
        let fault = || {
            // SAFETY: zeroed records are valid; the handler reads `bytes` and writes `context`.
            unsafe {
                let mut record: EXCEPTION_RECORD = std::mem::zeroed();
                record.ExceptionCode = EXCEPTION_ILLEGAL_INSTRUCTION;
                let mut context: CONTEXT = std::mem::zeroed();
                context.Rip = at as u64;
                let mut pointers = EXCEPTION_POINTERS { ExceptionRecord: &mut record, ContextRecord: &mut context };
                let taken = crate::crash::emulate_amd_only_for_test(&mut pointers);
                (taken, context.Rip == at as u64)
            }
        };
        assert_eq!(fault(), (false, true), "no one changed these bytes: not ours to take");
        crate::crash::note_patched(at);
        assert_eq!(fault(), (true, true), "the new bytes run from here");
    }
}
