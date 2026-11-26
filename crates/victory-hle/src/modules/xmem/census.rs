//! Diagnostics: where the title's physical pages lie and how the title
//! maps them, counted in diagnostics builds (the code is always built, so
//! that its tests run everywhere) and logged with the memory lines
//! ([`lines`], every few seconds, each prefixed `memory, physical pages,`).
//!
//! The question it answers: the host GPU reads title memory in place only
//! where the pages shown at consecutive addresses are consecutive slices of
//! one section (`victory_gpu`'s in-place windows), and the log of the in-place
//! path counts how often they are not. This says why:
//!
//! * *allocations*: how the title asks for pages (the allocator here hands
//!   out page numbers in ascending order, as one section a request, and
//!   never gives a number twice, so what it makes is in order);
//! * *mapped now*: the views the title holds, as runs of consecutive pages
//!   and by how many pages lie in runs of each length;
//! * *mapping calls*, for each of `XMemMapPhysicalPages`, its scatter form
//!   and the GPU's page-mapping copies: of the entries side by side at
//!   consecutive addresses, how many show the next page of a section, and
//!   for those that do not, what they show instead; and how many of the
//!   pages were never used before (their slice is still free to choose),
//!   with the run lengths there would be if those had been placed in the
//!   order of the mapping (`VICTORY_PAGES_ORDERED=1`).

#![cfg_attr(not(feature = "diagnostics"), allow(dead_code))]

use std::collections::HashSet;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use super::{MIB, PAGE, Physical};

/// Run lengths, in pages, in classes: 1, 2, 3, 4 to 7, 8 to 15, 16 to 63,
/// 64 or more. The in-place path wants 4 pages (256 KiB) at least.
const CLASSES: usize = 7;
const CLASS_NAMES: [&str; CLASSES] = ["1", "2", "3", "4-7", "8-15", "16-63", "64+"];

fn class(pages: usize) -> usize {
    match pages {
        0..=1 => 0,
        2 => 1,
        3 => 2,
        4..=7 => 3,
        8..=15 => 4,
        16..=63 => 5,
        _ => 6,
    }
}

/// The 4 MiB cells the host's windows are made of.
const CELL: usize = 4 * MIB;

/// Cells that lie wholly in `[start, end)`.
fn whole_cells(start: usize, end: usize) -> u64 {
    (end / CELL).saturating_sub(start.div_ceil(CELL)) as u64
}

/// Which way the title maps pages.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Kind {
    /// `XMemMapPhysicalPages`: an array of pages at consecutive addresses.
    Map,
    /// `XMemMapPhysicalPagesScatter`: a page and an address each.
    Scatter,
    /// The GPU's page-mapping copies: pages that are shown somewhere, shown
    /// at consecutive addresses.
    Alias,
}

impl Kind {
    const ALL: [Kind; 3] = [Kind::Map, Kind::Scatter, Kind::Alias];

    fn text(self) -> &'static str {
        match self {
            Kind::Map => "XMemMapPhysicalPages",
            Kind::Scatter => "the scatter form",
            Kind::Alias => "page-mapping copies by the GPU",
        }
    }
}

/// One entry of an array the title maps.
#[derive(Clone, Copy, Debug)]
pub(super) struct Entry {
    pub(super) address: usize,
    /// The section (its address, as an identity) and the index of the slice
    /// the page shows; `None` for an entry that clears an address or names
    /// no page.
    pub(super) slice: Option<(usize, usize)>,
    /// A page no mapping or write has used before.
    pub(super) new: bool,
}

/// What the entries of one kind of call came to, since the start.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
struct Calls {
    calls: u64,
    entries: u64,
    /// Entries that show no page.
    no_page: u64,
    /// Neighbours: entries at consecutive addresses, both with a page.
    pairs: u64,
    /// ...of which the second shows the page after the first's, in the same
    /// section;
    next: u64,
    /// ...a page of the same section further on (and of those, up to 16
    /// pages on);
    ahead: u64,
    ahead_near: u64,
    /// ...an earlier page of the same section;
    behind: u64,
    /// ...a page of another section.
    section: u64,
    /// Entries for pages never used before.
    new_entries: u64,
    /// Neighbours both new, and of those the ones that are not consecutive
    /// (what binding slices in the order of mapping would join).
    new_pairs: u64,
    new_apart: u64,
    /// Entries by the length of the run they lie in (pages in runs of each
    /// class): as the slices are, and as they would be if new pages were
    /// placed in the order of mapping.
    runs: [u64; CLASSES],
    joined: [u64; CLASSES],
}

/// Counts the entries of one call.
fn analyse(entries: &[Entry], c: &mut Calls) {
    c.calls += 1;
    let (mut actual, mut joined) = (0usize, 0usize);
    let flush = |len: &mut usize, by: &mut [u64; CLASSES]| {
        if *len > 0 {
            by[class(*len)] += *len as u64;
            *len = 0;
        }
    };
    for (i, e) in entries.iter().enumerate() {
        c.entries += 1;
        let Some(slice) = e.slice else {
            c.no_page += 1;
            flush(&mut actual, &mut c.runs);
            flush(&mut joined, &mut c.joined);
            continue;
        };
        if e.new {
            c.new_entries += 1;
        }
        let before = i.checked_sub(1).map(|j| entries[j]).filter(|p| p.slice.is_some() && e.address == p.address.wrapping_add(PAGE));
        let Some((previous, last)) = before.and_then(|p| p.slice.map(|s| (p, s))) else {
            // Not next to the entry before it, or that names no page.
            flush(&mut actual, &mut c.runs);
            flush(&mut joined, &mut c.joined);
            (actual, joined) = (1, 1);
            continue;
        };
        c.pairs += 1;
        let consecutive = slice.0 == last.0 && slice.1 == last.1 + 1;
        if consecutive {
            c.next += 1;
        } else if slice.0 != last.0 {
            c.section += 1;
        } else if slice.1 > last.1 {
            c.ahead += 1;
            if slice.1 - last.1 <= 16 {
                c.ahead_near += 1;
            }
        } else {
            c.behind += 1;
        }
        let both_new = previous.new && e.new;
        if both_new {
            c.new_pairs += 1;
            if !consecutive {
                c.new_apart += 1;
            }
        }
        if consecutive {
            actual += 1;
        } else {
            flush(&mut actual, &mut c.runs);
            actual = 1;
        }
        if consecutive || both_new {
            joined += 1;
        } else {
            flush(&mut joined, &mut c.joined);
            joined = 1;
        }
    }
    flush(&mut actual, &mut c.runs);
    flush(&mut joined, &mut c.joined);
}

/// The views the title holds, as runs.
#[derive(Default, Debug, PartialEq, Eq)]
struct Snapshot {
    views: u64,
    pages: u64,
    /// Runs of consecutive slices of one section at consecutive addresses;
    /// and runs of consecutive addresses alone.
    runs: u64,
    address_runs: u64,
    /// Pages in the runs of each length class.
    by_class: [u64; CLASSES],
    /// Where neighbours at consecutive addresses do not show consecutive
    /// pages: a page of the same section further on, an earlier one, a page
    /// of another section.
    ahead: u64,
    behind: u64,
    section: u64,
    /// 4 MiB cells with any view, wholly mapped (consecutive addresses all
    /// through), and wholly one run.
    cells_touched: u64,
    cells_mapped: u64,
    cells_one_run: u64,
}

/// The walk of [`snapshot`]: the slice run and the address run it is in
/// (where each starts, how many pages the slice run has, where the address
/// run ends) and what has been counted.
#[derive(Default)]
struct Walk {
    s: Snapshot,
    run_start: usize,
    run_pages: usize,
    address_start: usize,
    address_end: usize,
    open: bool,
}

impl Walk {
    fn end_run(&mut self) {
        if self.run_pages > 0 {
            self.s.runs += 1;
            self.s.by_class[class(self.run_pages)] += self.run_pages as u64;
            self.s.cells_one_run += whole_cells(self.run_start, self.run_start + self.run_pages * PAGE);
            self.run_pages = 0;
        }
    }

    fn end_addresses(&mut self) {
        if self.open {
            self.s.address_runs += 1;
            self.s.cells_mapped += whole_cells(self.address_start, self.address_end);
            self.open = false;
        }
    }
}

/// The views in address order, each as `(address, pages, section, index of
/// its first slice)`.
fn snapshot(views: impl Iterator<Item = (usize, usize, usize, usize)>) -> Snapshot {
    let mut w = Walk::default();
    // The slice that would continue the slice run, and the first cell no
    // view has been counted in yet.
    let mut next: (usize, usize) = (0, 0);
    let mut cell_next = 0usize;
    for (base, pages, section, index) in views {
        if pages == 0 {
            continue;
        }
        w.s.views += 1;
        w.s.pages += pages as u64;
        let end = base + pages * PAGE;
        let (first, last) = (base / CELL, (end - 1) / CELL);
        let from = first.max(cell_next);
        if last >= from {
            w.s.cells_touched += (last - from + 1) as u64;
            cell_next = last + 1;
        }
        if w.open && base == w.address_end {
            if (section, index) == next {
                w.run_pages += pages;
            } else {
                if section != next.0 {
                    w.s.section += 1;
                } else if index > next.1 {
                    w.s.ahead += 1;
                } else {
                    w.s.behind += 1;
                }
                w.end_run();
                (w.run_start, w.run_pages) = (base, pages);
            }
            w.address_end = end;
        } else {
            w.end_run();
            w.end_addresses();
            (w.run_start, w.run_pages, w.address_start, w.address_end, w.open) = (base, pages, base, end, true);
        }
        next = (section, index + pages);
    }
    w.end_run();
    w.end_addresses();
    w.s
}

#[derive(Default)]
struct Census {
    /// Requests for pages, the pages in them, the sections made, and the
    /// requests by size (1 page, 2 to 15, 16 to 255, 256 to 4095, more).
    requests: u64,
    pages: u64,
    sections: u64,
    sizes: [u64; 5],
    freed: u64,
    calls: [Calls; 3],
    /// Page numbers that a mapping or a write has used.
    used: HashSet<usize>,
}

static CENSUS: LazyLock<Mutex<Census>> = LazyLock::new(|| Mutex::new(Census::default()));

fn census() -> MutexGuard<'static, Census> {
    CENSUS.lock().unwrap_or_else(|e| e.into_inner())
}

fn counting() -> bool {
    cfg!(feature = "diagnostics")
}

/// The title asked for `pages` pages, made as `sections` sections.
pub(super) fn allocated(pages: usize, sections: usize) {
    if !counting() {
        return;
    }
    let mut c = census();
    c.requests += 1;
    c.pages += pages as u64;
    c.sections += sections as u64;
    let size = match pages {
        0..=1 => 0,
        2..=15 => 1,
        16..=255 => 2,
        256..=4095 => 3,
        _ => 4,
    };
    c.sizes[size] += 1;
}

pub(super) fn freed(pages: usize) {
    if counting() {
        census().freed += pages as u64;
    }
}

/// A write (DirectStorage's) used these pages.
pub(super) fn touched(pages: &[usize]) {
    if counting() {
        census().used.extend(pages.iter().copied());
    }
}

/// The title maps `entries` (address, page number; number 0 clears the
/// address). The pages have their slices.
pub(super) fn mapped(physical: &Physical, kind: Kind, entries: impl Iterator<Item = (usize, usize)>) {
    if !counting() {
        return;
    }
    let mut c = census();
    let list: Vec<Entry> = entries
        .map(|(address, page)| Entry {
            address,
            slice: physical.pages.get(&page).map(|s| (Arc::as_ptr(&s.section) as usize, s.index)),
            new: page != 0 && c.used.insert(page),
        })
        .collect();
    let at = Kind::ALL.iter().position(|&k| k == kind).unwrap_or(0);
    analyse(&list, &mut c.calls[at]);
}

/// The GPU's page-mapping copy shows these slices (section identity and
/// index) at consecutive addresses.
pub(super) fn aliased(slices: &[(usize, usize)]) {
    if !counting() {
        return;
    }
    let list: Vec<Entry> = slices
        .iter()
        .enumerate()
        .map(|(i, &slice)| Entry { address: i * PAGE, slice: Some(slice), new: false })
        .collect();
    let mut c = census();
    analyse(&list, &mut c.calls[2]);
}

/// Registers [`lines`] with the memory report.
#[cfg(feature = "diagnostics")]
pub(super) fn register() {
    victory_gpu::add_memory_lines(lines);
}

fn pct(part: u64, whole: u64) -> String {
    if whole == 0 { "-".to_string() } else { format!("{:.0}%", part as f64 * 100.0 / whole as f64) }
}

fn shares(by: &[u64; CLASSES]) -> String {
    let total: u64 = by.iter().sum();
    CLASS_NAMES.iter().zip(by).map(|(name, &n)| format!("{name}: {}", pct(n, total))).collect::<Vec<_>>().join(" ")
}

fn calls_lines(kind: Kind, c: &Calls) -> [String; 2] {
    [
        format!(
            "physical pages, {}: {} calls, {} entries ({} with no page), {} neighbours at consecutive addresses, of which the next page of the              same section {} ({}), a page of the same section further on {} ({} within 16 pages), an earlier one {}, a page of another              section {}; pages new to the title {} ({}), neighbours that are both new and not consecutive {} of {}",
            kind.text(),
            c.calls,
            c.entries,
            c.no_page,
            c.pairs,
            c.next,
            pct(c.next, c.pairs),
            c.ahead,
            c.ahead_near,
            c.behind,
            c.section,
            c.new_entries,
            pct(c.new_entries, c.entries - c.no_page),
            c.new_apart,
            c.new_pairs,
        ),
        format!(
            "physical pages, {}, share of entries by the length in pages of the run they lie in: as placed {}; if new pages were placed in              the order of mapping {}",
            kind.text(),
            shares(&c.runs),
            shares(&c.joined),
        ),
    ]
}

/// The census as lines of the memory report (without `memory, `).
pub(super) fn lines() -> Vec<String> {
    let (snap, twice, known, unbound) = {
        let physical = super::physical();
        let views = physical.views.iter().map(|(&base, (len, slice))| (base, len.div_ceil(PAGE), Arc::as_ptr(&slice.section) as usize, slice.index));
        let snap = snapshot(views);
        let twice = physical.shown.values().filter(|at| at.len() > 1).count();
        (snap, twice, physical.pages.len() + physical.unbound.len(), physical.unbound.len())
    };
    let c = census();
    let mb = |pages: u64| pages * PAGE as u64 / MIB as u64;
    let mut out = vec![
        format!(
            "physical pages, requests: {} for {} pages ({} MB) made as {} sections; sizes in pages 1: {} 2-15: {} 16-255: {} 256-4095: {} \
             4096+: {}; {} pages freed; {} page numbers known now, {} of them with no slice yet (VICTORY_PAGES_ORDERED={}); numbers are given \
             out in ascending order and never again, one section a request unless VICTORY_SECTION_MB",
            c.requests,
            c.pages,
            mb(c.pages),
            c.sections,
            c.sizes[0],
            c.sizes[1],
            c.sizes[2],
            c.sizes[3],
            c.sizes[4],
            c.freed,
            known,
            unbound,
            u8::from(super::pages_ordered()),
        ),
        format!(
            "physical pages, mapped now: {} views, {} pages ({} MB) in {} runs of consecutive pages of one section at consecutive addresses \
             ({} runs by address alone); share of pages by run length in pages {}; pages shown at two or more addresses {}; neighbours at \
             consecutive addresses that do not show consecutive pages: a page of the same section further on {}, an earlier one {}, a page of \
             another section {}; 4 MiB cells with a view {}, wholly mapped {}, wholly one run {}",
            snap.views,
            snap.pages,
            mb(snap.pages),
            snap.runs,
            snap.address_runs,
            shares(&snap.by_class),
            twice,
            snap.ahead,
            snap.behind,
            snap.section,
            snap.cells_touched,
            snap.cells_mapped,
            snap.cells_one_run,
        ),
    ];
    for (&kind, calls) in Kind::ALL.iter().zip(&c.calls) {
        out.extend(calls_lines(kind, calls));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A page of section 1 at slice `index`, shown at page address `slot`,
    /// new or not.
    fn at(slot: usize, index: usize, new: bool) -> Entry {
        Entry { address: 0x1000_0000 + slot * PAGE, slice: Some((1, index)), new }
    }

    #[test]
    fn consecutive_pages_are_one_run() {
        let entries: Vec<Entry> = (0..8).map(|i| at(i, 10 + i, false)).collect();
        let mut c = Calls::default();
        analyse(&entries, &mut c);
        assert_eq!((c.calls, c.entries, c.pairs, c.next), (1, 8, 7, 7));
        assert_eq!(c.runs, [0, 0, 0, 0, 8, 0, 0], "eight pages in a run of 8 to 15");
        assert_eq!(c.runs, c.joined, "nothing to join");
    }

    #[test]
    fn scattered_pages_are_runs_of_one_and_new_ones_would_join() {
        // Four pages side by side, listed out of order, none used before.
        let entries = [at(0, 9, true), at(1, 2, true), at(2, 3, true), at(3, 0, true)];
        let mut c = Calls::default();
        analyse(&entries, &mut c);
        assert_eq!(c.pairs, 3);
        assert_eq!(c.next, 1, "only slice 2 then 3 follows on");
        assert_eq!((c.behind, c.ahead, c.section), (2, 0, 0), "9 then 2 and 3 then 0 go back");
        assert_eq!(c.runs, [2, 2, 0, 0, 0, 0, 0], "a page alone, a run of two, a page alone");
        assert_eq!((c.new_entries, c.new_pairs, c.new_apart), (4, 3, 2));
        assert_eq!(c.joined, [0, 0, 0, 4, 0, 0, 0], "placed in order of mapping they would be one run of four");
    }

    #[test]
    fn pages_used_before_keep_their_runs_apart() {
        // The middle page is old: the neighbours cannot be joined to it.
        let entries = [at(0, 5, true), at(1, 40, false), at(2, 6, true)];
        let mut c = Calls::default();
        analyse(&entries, &mut c);
        assert_eq!(c.new_pairs, 0);
        assert_eq!(c.joined, c.runs, "one new page next to an old one joins nothing");
        assert_eq!(c.ahead, 1);
        assert_eq!(c.behind, 1);
        assert_eq!(c.ahead_near, 0, "40 is 35 slices on");
    }

    #[test]
    fn neighbours_need_consecutive_addresses_and_a_page() {
        let mut gap = at(2, 11, false);
        gap.address += PAGE; // a page of address between
        let none = Entry { address: 0x1000_0000 + 3 * PAGE, slice: None, new: false };
        let entries = [at(0, 10, false), at(1, 11, false), gap, none, at(4, 12, false)];
        let mut c = Calls::default();
        analyse(&entries, &mut c);
        assert_eq!(c.entries, 5);
        assert_eq!(c.no_page, 1);
        assert_eq!(c.pairs, 1, "only the first two are neighbours");
        assert_eq!(c.runs, [2, 2, 0, 0, 0, 0, 0], "two pages alone, one run of two");
    }

    #[test]
    fn another_section_ends_a_run() {
        let mut other = at(1, 11, false);
        other.slice = Some((2, 11));
        let mut c = Calls::default();
        analyse(&[at(0, 10, false), other], &mut c);
        assert_eq!((c.next, c.section), (0, 1));
    }

    #[test]
    fn runs_are_counted_across_views_and_cells() {
        const P: usize = 64; // pages in a cell
        let cell = CELL;
        let views = [
            // A whole cell of one section in two views that go on from one another.
            (cell, 40, 1usize, 0usize),
            (cell + 40 * PAGE, 24, 1, 40),
            // Right after it, a view that does not (an earlier slice of the same section).
            (2 * cell, 2, 1, 7),
            // A gap, then three pages of another section and two after them that go on.
            (5 * cell, 1, 2, 100),
            (5 * cell + PAGE, 2, 2, 101),
            // Address-adjacent, another section.
            (5 * cell + 3 * PAGE, 1, 3, 0),
        ];
        let s = snapshot(views.into_iter());
        assert_eq!((s.views, s.pages), (6, 40 + 24 + 2 + 1 + 2 + 1));
        assert_eq!(s.runs, 4, "the cell, the 2 pages, the 3 pages, the last one");
        assert_eq!(s.address_runs, 2, "cell and the two pages after it, then the four pages after the gap");
        assert_eq!(s.by_class, [1, 2, 3, 0, 0, 0, P as u64]);
        assert_eq!((s.ahead, s.behind, s.section), (0, 1, 1));
        assert_eq!(s.cells_touched, 3, "the cell, the cell of the two pages, the cell after the gap");
        assert_eq!(s.cells_mapped, 1, "the first cell is wholly mapped");
        assert_eq!(s.cells_one_run, 1);
    }

    /// With counting on (diagnostics builds): a mapping is counted, and the
    /// report is eight lines, each starting as the log's readers look for.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn the_report_counts_a_mapping_and_has_its_lines() {
        use std::ffi::c_void;
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(super::super::XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let at = super::super::XMemVirtualAlloc(
            std::ptr::null_mut(),
            3 * PAGE,
            super::super::MEM_RESERVE | super::super::PAGES_64K,
            0,
            super::super::PAGE_READWRITE,
        );
        assert!(!at.is_null());
        assert_eq!(super::super::XMemMapPhysicalPages(at, 3, pages.as_ptr()), at);
        let lines = lines();
        for line in &lines {
            println!("memory, {line}");
        }
        assert_eq!(lines.len(), 8);
        let starts = [
            "physical pages, requests: ",
            "physical pages, mapped now: ",
            "physical pages, XMemMapPhysicalPages: ",
            "physical pages, XMemMapPhysicalPages, share of entries",
            "physical pages, the scatter form: ",
            "physical pages, the scatter form, share of entries",
            "physical pages, page-mapping copies by the GPU: ",
            "physical pages, page-mapping copies by the GPU, share of entries",
        ];
        for (line, start) in lines.iter().zip(starts) {
            assert!(line.starts_with(start), "{line}");
        }
        let calls: u64 = lines[2][starts[2].len()..].split(' ').next().unwrap().parse().unwrap();
        assert!(calls >= 1, "{}", lines[2]);
        let none: *const usize = std::ptr::null();
        assert_eq!(super::super::XMemMapPhysicalPages(at, 3, none), at);
        let _: *mut c_void = at;
        assert_eq!(super::super::XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn an_empty_table_is_all_zero() {
        assert_eq!(snapshot(std::iter::empty()), Snapshot::default());
    }

    #[test]
    fn percentages_and_shares_read_plainly() {
        assert_eq!(pct(1, 4), "25%");
        assert_eq!(pct(0, 0), "-");
        assert_eq!(shares(&[0, 0, 0, 4, 0, 0, 0]), "1: 0% 2: 0% 3: 0% 4-7: 100% 8-15: 0% 16-63: 0% 64+: 0%");
    }
}
