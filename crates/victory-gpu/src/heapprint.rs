//! The records of the title's descriptor heap fingerprinted where they are,
//! each read once (`VICTORY_SYNC_IN_PLACE=1`; off by default), for
//! [`crate::descriptor::sync_heaps`].
//!
//! After avatarmanz/VIctory-7900XTX `15ab94e` and `8822ed4` (GPL-3.0): there
//! a changed heap page's records are read where they are, once, and compared
//! with those last seen as they are read, where before the page was copied
//! into a buffer and compared in a second pass (7.4 and 5.3 ms a frame of
//! their six threads for 35 MB). Here the pass keeps a 64-bit fingerprint of
//! each record, not its bytes ([`crate::descriptor::host`]'s `copied`), so
//! what is done in place is the fingerprint itself: one function of ours,
//! that uses no stack, reads each 32-byte record of title memory into
//! registers and puts the fingerprint `record_print(record) | 1` of it into
//! a table. A handler first in the line turns a read fault inside it (a page
//! the title unmapped meanwhile: it unmaps heap pages too) into its return
//! of false; the caller then goes the way it went before, through the
//! kernel's copy, which says which page is gone.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    ".globl victory_print_records",
    ".globl victory_print_records_at",
    ".globl victory_print_records_after",
    ".globl victory_print_records_fault",
    "victory_print_records:",
    // rcx: the first record, rdx: where the prints go (a u64 each), r8: how
    // many records of 32 bytes. Nothing on the stack; only volatile
    // registers. eax: 1 if all were read, 0 if a page was not there (the
    // prints before it are made).
    "    test r8, r8",
    "    jz 3f",
    "    mov r11, 0x9e3779b97f4a7c15",
    "victory_print_records_at:",
    "2:",
    // h = K ^ 32; for each of the four words: h = (h ^ w) * K; h ^= h >> 29.
    "    mov rax, r11",
    "    xor rax, 32",
    "    mov r9, qword ptr [rcx]",
    "    xor rax, r9",
    "    imul rax, r11",
    "    mov r9, rax",
    "    shr r9, 29",
    "    xor rax, r9",
    "    mov r9, qword ptr [rcx + 8]",
    "    xor rax, r9",
    "    imul rax, r11",
    "    mov r9, rax",
    "    shr r9, 29",
    "    xor rax, r9",
    "    mov r9, qword ptr [rcx + 16]",
    "    xor rax, r9",
    "    imul rax, r11",
    "    mov r9, rax",
    "    shr r9, 29",
    "    xor rax, r9",
    "    mov r9, qword ptr [rcx + 24]",
    "    xor rax, r9",
    "    imul rax, r11",
    "    mov r9, rax",
    "    shr r9, 29",
    "    xor rax, r9",
    // h ^ (h >> 32), never 0 (0 is "never looked at").
    "    mov r9, rax",
    "    shr r9, 32",
    "    xor rax, r9",
    "    or rax, 1",
    "    mov qword ptr [rdx], rax",
    "    add rcx, 32",
    "    add rdx, 8",
    "    dec r8",
    "    jnz 2b",
    "victory_print_records_after:",
    "3:",
    "    mov eax, 1",
    "    ret",
    "victory_print_records_fault:",
    "    xor eax, eax",
    "    ret",
);

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    fn victory_print_records(records: *const u8, out: *mut u64, count: usize) -> u32;
    static victory_print_records_at: u8;
    static victory_print_records_after: u8;
    static victory_print_records_fault: u8;
}

/// Whether the pass over the heap's changed pages reads their records in
/// place (`VICTORY_SYNC_IN_PLACE=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| cfg!(target_arch = "x86_64") && std::env::var_os("VICTORY_SYNC_IN_PLACE").is_some_and(|v| v == "1"))
}

/// Since the last frame line: pages whose records were fingerprinted in
/// place, their records, and pages that went the old way because a page of
/// them was not there (or had faulted lately, or a copy-back left it
/// pending).
static COUNT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The handler that turns a read fault inside [`victory_print_records`] into
/// its return of false; put first in the line, once.
#[cfg(target_arch = "x86_64")]
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, EXCEPTION_GUARD_PAGE, EXCEPTION_IN_PAGE_ERROR};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched and the
        // context it will continue with.
        let (r, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
        // Only inside the function's loop.
        let (from, to) = ((&raw const victory_print_records_at) as u64, (&raw const victory_print_records_after) as u64);
        if context.Rip < from || context.Rip >= to {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        const READ: usize = 0;
        let read = r.NumberParameters >= 1 && r.ExceptionInformation[0] == READ;
        let gone = (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION || r.ExceptionCode == EXCEPTION_IN_PAGE_ERROR) && read;
        if gone || r.ExceptionCode == EXCEPTION_GUARD_PAGE {
            context.Rip = (&raw const victory_print_records_fault) as u64;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        EXCEPTION_CONTINUE_SEARCH
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: a handler that reads the record it is given and nothing else.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_fault));
    });
}

/// The fingerprint (`record_print(record) | 1`) of each of the `count`
/// records of 32 bytes at title address `at`, into `out` (emptied first);
/// false, with `out` holding nothing to use, if a page of them is not there,
/// was not there lately ([`crate::plaincopy`]'s notes of it), or a copy-back
/// left it pending ([`crate::lazy`]): the caller reads the page the other
/// way.
pub(crate) fn prints(at: usize, count: usize, out: &mut Vec<u64>) -> bool {
    out.clear();
    #[cfg(target_arch = "x86_64")]
    {
        let bytes = count * 32;
        if !on() || count == 0 || crate::lazy::touches(at, bytes) || crate::plaincopy::faulted_lately(at, bytes) {
            COUNT[2].fetch_add(1, Relaxed);
            return false;
        }
        install();
        out.resize(count, 0);
        // SAFETY: `out` holds `count` prints; a source page that is not
        // there ends the pass with 0 instead of a fault ([`install`]).
        let whole = unsafe { victory_print_records(at as *const u8, out.as_mut_ptr(), count) != 0 };
        if whole {
            COUNT[0].fetch_add(1, Relaxed);
            COUNT[1].fetch_add(count as u64, Relaxed);
        } else {
            COUNT[3].fetch_add(1, Relaxed);
            crate::plaincopy::note_fault(at, bytes);
            out.clear();
        }
        whole
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (at, count);
        false
    }
}

/// With the frame line: what [`prints`] did per frame over `frames`; nothing
/// while it is off.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let [pages, records, left, gone] = [0, 1, 2, 3].map(|i| COUNT[i].swap(0, Relaxed) as f64 / frames);
    if pages + left + gone > 0.0 {
        tracing::info!(
            "heap records fingerprinted where they are (VICTORY_SYNC_IN_PLACE=1) per frame: {pages:.1} pages, {records:.0} records; \
             {left:.1} pages left to the copy first (pending, faulted lately), {gone:.2} met a page that was not there"
        );
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    /// What `descriptor::host::record_print` says, which the pass compares.
    fn reference(record: &[u8]) -> u64 {
        const K: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut h = K ^ record.len() as u64;
        for word in record.chunks_exact(8) {
            h = (h ^ u64::from_le_bytes(word.try_into().unwrap())).wrapping_mul(K);
            h ^= h >> 29;
        }
        (h ^ (h >> 32)) | 1
    }

    #[test]
    fn the_fingerprints_are_those_of_record_print() {
        install();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for count in [0usize, 1, 2, 63, 64, 2048] {
            let bytes: Vec<u8> = (0..count * 32).map(|_| ((next() % 3) as u8).wrapping_mul(next() as u8)).collect();
            let mut out = vec![7u64; 3];
            out.resize(count, 0);
            // SAFETY: `out` holds `count` prints; the bytes are ours.
            let whole = unsafe { victory_print_records(bytes.as_ptr(), out.as_mut_ptr(), count) != 0 };
            assert!(whole);
            let want: Vec<u64> = bytes.chunks_exact(32).map(reference).collect();
            assert_eq!(out[..count], want[..], "{count} records");
            assert!(out.iter().take(count).all(|p| p & 1 == 1), "never 0");
        }
    }

    /// Two pages reserved, the first committed: the records in the first are
    /// fingerprinted; records that run into the second end the pass with
    /// false, the prints before the hole made, and the thread goes on.
    #[test]
    fn a_page_that_is_not_there_ends_the_pass_without_a_fault() {
        let _one = crate::address_space();
        install();
        const PAGE: usize = 4096;
        // SAFETY: memory of the test's own.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as *mut u8;
        assert!(!base.is_null());
        // SAFETY: the first page of that reservation.
        assert!(!unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        // SAFETY: committed and writable.
        let page = unsafe { std::slice::from_raw_parts_mut(base, PAGE) };
        for (i, b) in page.iter_mut().enumerate() {
            *b = (i * 7 + 3) as u8;
        }
        let mut out = vec![0u64; 2 * PAGE / 32];
        let (n, whole) = (PAGE / 32, PAGE / 32 * 2);
        // SAFETY: as above; `out` holds the prints.
        assert!(unsafe { victory_print_records(base, out.as_mut_ptr(), n) } != 0);
        let want: Vec<u64> = page.chunks_exact(32).map(reference).collect();
        assert_eq!(out[..n], want[..]);
        out.fill(0);
        // SAFETY: as above; the second page is not there.
        assert_eq!(unsafe { victory_print_records(base, out.as_mut_ptr(), whole) }, 0);
        assert_eq!(out[..n], want[..], "the prints before the hole");
        assert!(out[n..].iter().all(|&p| p == 0), "none after it");
        // A start in the hole, and a page closed to reads.
        // SAFETY: as above.
        assert_eq!(unsafe { victory_print_records(base.wrapping_add(PAGE), out.as_mut_ptr(), 1) }, 0);
        // SAFETY: the third page of the reservation.
        assert!(!unsafe { VirtualAlloc(base.wrapping_add(2 * PAGE).cast(), PAGE, MEM_COMMIT, PAGE_NOACCESS) }.is_null());
        // SAFETY: as above.
        assert_eq!(unsafe { victory_print_records(base.wrapping_add(2 * PAGE), out.as_mut_ptr(), 1) }, 0);
        // And what the caller owns came back: this still works.
        // SAFETY: as above.
        assert!(unsafe { victory_print_records(base, out.as_mut_ptr(), n) } != 0);
        assert_eq!(out[..n], want[..]);
        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }
}
