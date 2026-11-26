//! Renderer port #1, the runtime's half (`VICTORY_PORT_CONSTANTS=1`, off by
//! default; the GPU side's is `victory_gpu`'s `port`). After avatarmanz's
//! VIctory-7900XTX (GPL-3.0), commits `4b82f6c` and `9c23e8d`.
//!
//! The title reserves its 32-bit graphics heap (a hundred and some MB below
//! 4 GiB) in one go, splits it into pieces, takes all the physical pages of
//! the pieces in ONE `XMemAllocatePhysicalPages` and maps them piece by
//! piece, each from the next pages of that one array and in another order
//! than the pieces' addresses. [`allocate_physical`] gives the array's page
//! numbers the slices of one section in the order the numbers were made in,
//! so the pieces, mapped in the other order, are views of one section in a
//! scattered order, and no one view of ours can show the heap. Here the page
//! numbers of a mapping into such a reservation are given the slices at
//! their address's offset in the reservation, when the reservation is all
//! of one section: the heap becomes one stretch of one section, views that
//! follow one another in memory and in the section alike, which the GPU
//! side opens as one host heap.
//!
//! Only fresh slices are exchanged: never mapped through any view of ours
//! and never written (no bit in the section's [`Section::committed`]), so
//! every page of them still reads as zeros and which slice a page number
//! names is not something the title can observe.
//!
//! The reservation is known by what the title asks with: graphics memory
//! below 4 GiB reserved with no commit (`flags & ADDRESS_32BIT`), of a size
//! worth a heap ([`LEAST`]); never by an address.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::{ADDRESS_32BIT, MEM_COMMIT, PAGE, physical};

/// A reservation smaller than this is not one worth a host heap of its own.
const LEAST: usize = 8 << 20;

/// The title's reservations of 32-bit graphics memory: base and bytes.
static LOW_RESERVATIONS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// The lowest base and highest end of [`LOW_RESERVATIONS`] (both 0: none),
/// for a quick look at each mapping call (the title makes hundreds a frame).
static BOUNDS: [AtomicUsize; 2] = [const { AtomicUsize::new(0) }; 2];

/// Pages given another place in their section so far by [`arrange`], and
/// the times it found a page that was not fresh.
static ARRANGED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether `kind` and `flags` ask for what [`reserved`] notes: a
/// reservation with no commit, of 32-bit graphics memory, of the size of a
/// heap.
fn is_low_reservation(size: usize, kind: u32, flags: u64) -> bool {
    size >= LEAST && kind & MEM_COMMIT == 0 && flags & ADDRESS_32BIT != 0
}

/// The title reserved `[at, at + size)` of address space at an address of
/// the host's choosing, with `kind` and `flags` as it asked: noted if it is
/// a reservation of 32-bit graphics memory ([`is_low_reservation`]).
pub(super) fn reserved(at: usize, size: usize, kind: u32, flags: u64) {
    if at != 0 && is_low_reservation(size, kind, flags) {
        note(at, size);
    }
}

/// `[at, at + size)` is a reservation of the title's 32-bit graphics memory.
/// One that overlaps an earlier one replaces it (the earlier was given up).
fn note(at: usize, size: usize) {
    let mut all = LOW_RESERVATIONS.lock().unwrap_or_else(|e| e.into_inner());
    all.retain(|&(from, bytes)| from + bytes <= at || from >= at + size);
    all.push((at, size));
    bounds(&all);
}

fn bounds(all: &[(usize, usize)]) {
    BOUNDS[0].store(all.iter().map(|r| r.0).min().unwrap_or(0), Ordering::Relaxed);
    BOUNDS[1].store(all.iter().map(|r| r.0 + r.1).max().unwrap_or(0), Ordering::Relaxed);
}

/// Before `pages` are mapped at `base`: if `base` lies in a 32-bit graphics
/// reservation ([`reserved`]) whose pages are all one section as big as it,
/// the page numbers are given the slices of that section in the
/// reservation's address order (each page at its offset in the reservation)
/// where both slices are fresh. Page numbers with no slice yet
/// (`VICTORY_PAGES_ORDERED=1` gives them when they are first used) are left
/// where they are, and a page mapped already stays.
pub(super) fn arrange(base: usize, pages: &[usize]) {
    let end = base.saturating_add(pages.len() * PAGE);
    if BOUNDS[1].load(Ordering::Relaxed) < end || BOUNDS[0].load(Ordering::Relaxed) > base {
        return;
    }
    let Some((first, bytes)) = LOW_RESERVATIONS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .copied()
        .find(|&(at, len)| at <= base && end <= at + len)
    else {
        return;
    };
    let mut table = physical();
    if pages.iter().any(|p| table.unbound.contains_key(p)) {
        say(base, pages.len(), "pages are given their slices when they are first used (VICTORY_PAGES_ORDERED=1), not in order of address");
        return;
    }
    let Some(section) = pages.first().and_then(|p| table.pages.get(p)).map(|s| s.section.clone()) else {
        return;
    };
    if section.slices * PAGE < bytes || !pages.iter().all(|p| table.pages.get(p).is_some_and(|s| Arc::ptr_eq(&s.section, &section))) {
        say(base, pages.len(), "pages mapped into the 32-bit heap are not of one section as big as it");
        return;
    }
    let fresh = |index: usize| section.committed.get(index / 64).is_some_and(|w| w.load(Ordering::Acquire) >> (index % 64) & 1 == 0);
    // Which number has which slice of the section.
    let mut owner: HashMap<usize, usize> =
        table.pages.iter().filter(|(_, s)| Arc::ptr_eq(&s.section, &section)).map(|(&number, s)| (s.index, number)).collect();
    let mut moved = 0u64;
    for (i, number) in pages.iter().enumerate() {
        let want = (base - first) / PAGE + i;
        let Some(have) = table.pages.get(number).map(|s| s.index) else {
            continue;
        };
        if have == want {
            continue;
        }
        if want >= section.slices || !fresh(have) || !fresh(want) {
            ARRANGED[1].fetch_add(1, Ordering::Relaxed);
            continue;
        }
        // The number that has the slice wanted takes this one's.
        if let Some(other) = owner.get(&want).copied()
            && let Some(slice) = table.pages.get_mut(&other)
        {
            slice.index = have;
            owner.insert(have, other);
        }
        if let Some(slice) = table.pages.get_mut(number) {
            slice.index = want;
        }
        owner.insert(want, *number);
        moved += 1;
    }
    if moved > 0 {
        let all = ARRANGED[0].fetch_add(moved, Ordering::Relaxed) + moved;
        tracing::info!(
            at = format_args!("{base:#x}"),
            pages = pages.len(),
            moved,
            all,
            not_fresh = ARRANGED[1].load(Ordering::Relaxed),
            "renderer port 1: the 32-bit graphics pages given their section's slices in address order"
        );
    }
}

/// Says once in a while why pages mapped into the 32-bit heap are left
/// where they are.
fn say(at: usize, pages: usize, why: &str) {
    static SAID: AtomicU64 = AtomicU64::new(0);
    if SAID.fetch_add(1, Ordering::Relaxed) < 4 {
        tracing::warn!(at = format_args!("{at:#x}"), pages, "renderer port 1: {why}: left where they are");
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;

    /// `n` pages of one new allocation.
    fn pages_of_one_call(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()), 1);
        assert_eq!(given, n);
        pages
    }

    /// A reservation as the title makes its 32-bit heap: reserve, 32-bit
    /// graphics memory, no 64 KiB page bits, no commit. Known to [`arrange`]
    /// as the call site notes it with the switch on (but small: the size
    /// floor is for the title).
    fn reserve_low(n: usize) -> *mut c_void {
        let kind = MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
        let base = XMemVirtualAlloc(std::ptr::null_mut(), n * PAGE, kind, GRAPHICS | ADDRESS_32BIT, PAGE_NOACCESS);
        assert!(!base.is_null() && (base as usize) < 1 << 32);
        note(base as usize, n * PAGE);
        base
    }

    fn forget(at: usize) {
        let mut all = LOW_RESERVATIONS.lock().unwrap();
        all.retain(|r| r.0 != at);
        bounds(&all);
    }

    fn index(number: usize) -> Option<usize> {
        physical().pages.get(&number).map(|s| s.index)
    }

    /// What is asked for makes a reservation known; nothing else does.
    #[test]
    fn a_reservation_is_known_by_what_the_title_asks() {
        let kind = MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
        assert!(is_low_reservation(LEAST, kind, GRAPHICS | ADDRESS_32BIT));
        assert!(!is_low_reservation(LEAST - 1, kind, GRAPHICS | ADDRESS_32BIT), "too small for a heap");
        assert!(!is_low_reservation(LEAST, kind | MEM_COMMIT, GRAPHICS | ADDRESS_32BIT), "committed at once");
        assert!(!is_low_reservation(LEAST, kind, GRAPHICS), "not 32-bit memory");
        // A later reservation at the same addresses replaces the earlier.
        let at = 0x3000_0000usize;
        reserved(at, LEAST, kind, GRAPHICS | ADDRESS_32BIT);
        reserved(at + PAGE, 2 * LEAST, kind, GRAPHICS | ADDRESS_32BIT);
        let all = LOW_RESERVATIONS.lock().unwrap().clone();
        assert!(!all.contains(&(at, LEAST)) && all.contains(&(at + PAGE, 2 * LEAST)), "{all:?}");
        forget(at + PAGE);
    }

    #[test]
    fn the_32_bit_heap_gets_its_slices_in_address_order() {
        // As the title's heap: a 32-bit reservation, all of its pages from
        // one call, mapped piece by piece in another order than the
        // pieces' addresses.
        let base = reserve_low(4);
        let pages = pages_of_one_call(4);
        let at = |page: usize| base.wrapping_byte_add(page * PAGE);
        // The last two pages' place first, from the array's start.
        arrange(at(2) as usize, &pages[..2]);
        assert_eq!(XMemMapPhysicalPages(at(2), 2, pages.as_ptr()), at(2));
        arrange(base as usize, &pages[2..]);
        assert_eq!(XMemMapPhysicalPages(base, 2, pages[2..].as_ptr()), base);
        assert_eq!([index(pages[0]), index(pages[1]), index(pages[2]), index(pages[3])], [Some(2), Some(3), Some(0), Some(1)]);
        // Mapped already: nothing moves any more.
        arrange(base as usize, &pages[..2]);
        assert_eq!(index(pages[0]), Some(2));
        // The views are one stretch of one section now, in address order.
        {
            let physical = physical();
            let (a, b) = (physical.views.get(&(base as usize)), physical.views.get(&(at(2) as usize)));
            let (a, b) = (a.expect("a view at the base"), b.expect("a view of the last pages"));
            assert!(Arc::ptr_eq(&a.1.section, &b.1.section));
            assert_eq!((a.1.index, b.1.index), (0, 2), "address order in the section");
        }
        // SAFETY: mapped read-write.
        unsafe { at(3).cast::<u8>().write(7) };
        assert_eq!(XMemMapPhysicalPages(base, 4, std::ptr::null()), base);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
        forget(base as usize);
    }

    /// A page that was mapped, or written through a view of ours, keeps its
    /// slice: its bytes would show somewhere else if it moved.
    #[test]
    fn a_page_that_was_touched_keeps_its_slice() {
        let base = reserve_low(4);
        let pages = pages_of_one_call(4);
        // DirectStorage writes page 0 (slice 0) before it is mapped.
        assert!(write_physical(&pages[..1], 0, &[5]));
        // Its address would have it take slice 2: it does not (slice 0 has
        // been written), and the page that has slice 2 stays too.
        arrange(base as usize + 2 * PAGE, &pages[..1]);
        assert_eq!([index(pages[0]), index(pages[2])], [Some(0), Some(2)]);
        // A page that wants a touched slice stays as well.
        arrange(base as usize, &pages[2..3]);
        assert_eq!([index(pages[2]), index(pages[0])], [Some(2), Some(0)]);
        // Two untouched slices exchange: slice 3 for the page that wants
        // it, and slice 1 to the number that had it.
        arrange(base as usize + 3 * PAGE, &pages[1..2]);
        assert_eq!([index(pages[1]), index(pages[3])], [Some(3), Some(1)]);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
        forget(base as usize);
    }

    /// Pages of another reservation, of a section too small for one, or
    /// with no slice yet, are left alone.
    #[test]
    fn pages_outside_a_known_reservation_stay() {
        let pages = pages_of_one_call(2);
        let before = [index(pages[0]), index(pages[1])];
        // No reservation known at that address.
        arrange(0x1234_0000, &pages);
        assert_eq!([index(pages[0]), index(pages[1])], before);
        // A reservation one section does not fill.
        let base = 0x2345_0000usize;
        note(base, 64 * PAGE);
        arrange(base + PAGE, &pages);
        assert_eq!([index(pages[0]), index(pages[1])], before, "the section is smaller than the reservation");
        forget(base);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }
}
