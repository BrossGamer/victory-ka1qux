//! Which cores our busy threads run on, on a processor with two kinds of
//! cores (`VICTORY_P_CORES=1` and `VICTORY_LANE_CORE=1`; both off by default).
//!
//! An Intel hybrid has fast cores (P: hyperthreaded) and efficiency cores
//! (E) that run a thread some 40 % slower. The console has eight cores alike,
//! and the title pins its own threads as if so; ours are unpinned, and the
//! scheduler parks some of them on an E core (run 229: the submit thread was
//! found on one in 1,529 looks of 24,021, and the frame waits for that
//! thread). `framelog::keep_on_fast_cores` already asks for the P cores for
//! the threads that called it. What is added here, for the threads that
//! register with [`prefer`]:
//!
//! - with `VICTORY_P_CORES=1`, our submission thread and the thread that
//!   puts the copy-backs in order each get a P core of their own (both its
//!   logical processors, and no other of our busy threads there: the
//!   siblings of a core share its units), and every other busy thread of
//!   ours (copy and compare workers, fills, the title's own thread that
//!   submits lists) gets the other P cores as a preference;
//! - with `VICTORY_LANE_CORE=1` (after avatarmanz/VIctory-7900XTX 11511b8,
//!   GPL-3.0), the submission thread has a core to itself as far as this
//!   process goes: the threads of the process that have no choice of their
//!   own are kept off its two logical processors too
//!   (`SetProcessDefaultCpuSets`).
//!
//! A set of processors selected for a thread is a preference the scheduler
//! keeps unless nothing of it is free (`SetThreadSelectedCpuSets`), not a
//! hard affinity: nothing starves. With one kind of core nothing is done,
//! and the log says so once. The last P cores are taken (the title's pinned
//! threads are on the first ones).

use std::sync::Mutex;

/// What a thread that asks is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// The submission thread: the one a frame waits for.
    Lane,
    /// A thread of ours that works all the time and that others wait for
    /// (the copy-back relay): a core of its own if there is one.
    Busy,
    /// A thread that works beside the rest, in turns (workers of a pool, a
    /// title thread that submits).
    Pool,
}

/// A processor set of the system: its id, its logical processor, its
/// efficiency class (higher is faster) and its core (which two of them may
/// share).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CpuSet {
    pub id: u32,
    pub processor: u8,
    pub class: u8,
    pub core: u8,
}

/// Whether `VICTORY_P_CORES=1`.
pub(crate) fn p_cores() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_P_CORES").is_some_and(|v| v == "1"))
}

/// Whether `VICTORY_LANE_CORE=1`.
pub(crate) fn lane_core() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LANE_CORE").is_some_and(|v| v == "1"))
}

/// The most P cores the threads that ask for a core of their own take
/// (`VICTORY_P_CORES_OWN`, 2: the submission thread and the relay), always
/// leaving four for the rest.
fn own_most() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_P_CORES_OWN").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2usize).min(4))
}

/// The system's processor sets of group 0.
fn cpu_sets() -> Vec<CpuSet> {
    use windows_sys::Win32::System::SystemInformation::{GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut bytes = 0u32;
    // SAFETY: a size query, then the call into a buffer of that size.
    unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above; the buffer holds `bytes` bytes, aligned for the records.
    let ok = unsafe { GetSystemCpuSetInformation(buffer.as_mut_ptr().cast(), bytes, &mut bytes, GetCurrentProcess(), 0) };
    let mut sets = Vec::new();
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_CPU_SET_INFORMATION>()) };
        if info.Size == 0 {
            break;
        }
        // SAFETY: the record's type is CpuSet (the only one there is).
        let set = unsafe { &info.Anonymous.CpuSet };
        if set.Group == 0 {
            sets.push(CpuSet { id: set.Id, processor: set.LogicalProcessorIndex, class: set.EfficiencyClass, core: set.CoreIndex });
        }
        at += info.Size as usize;
    }
    sets
}

/// Where each kind of thread goes (ids of processor sets).
#[derive(Debug, Default, PartialEq, Eq)]
struct Plan {
    /// The submission thread's core (empty: none of its own).
    lane: Vec<u32>,
    /// A core of its own for each busy thread that asks, in turn.
    busy: Vec<Vec<u32>>,
    /// The other P cores, for the rest (all of them if none are left).
    pool: Vec<u32>,
    /// With the lane core on: every set but the lane core's, which the
    /// threads of the process with no choice of their own get.
    default: Vec<u32>,
}

/// The plan for `sets`; `None` where all are of one class. `lane`: the
/// submission thread has a core of its own; `own`: the most such cores for
/// it and busy threads together (four P cores stay for the rest, and a lane
/// core needs six).
fn plan(sets: &[CpuSet], lane: bool, own: usize) -> Option<Plan> {
    let best = sets.iter().map(|c| c.class).max()?;
    let fast: Vec<&CpuSet> = sets.iter().filter(|c| c.class == best).collect();
    if fast.len() == sets.len() {
        return None;
    }
    let mut cores: Vec<u8> = fast.iter().map(|c| c.core).collect();
    cores.sort_unstable();
    cores.dedup();
    let own = if cores.len() >= 6 { own.min(cores.len() - 4) } else { 0 };
    // From the last core down.
    let taken: Vec<u8> = cores.iter().rev().take(own).copied().collect();
    let of = |core: u8| -> Vec<u32> { fast.iter().filter(|c| c.core == core).map(|c| c.id).collect() };
    let mut plan = Plan::default();
    for (n, &core) in taken.iter().enumerate() {
        if n == 0 && lane {
            plan.lane = of(core);
        } else {
            plan.busy.push(of(core));
        }
    }
    let rest: Vec<u32> = fast.iter().filter(|c| !taken.contains(&c.core)).map(|c| c.id).collect();
    plan.pool = if rest.is_empty() { fast.iter().map(|c| c.id).collect() } else { rest };
    if !plan.lane.is_empty() {
        plan.default = sets.iter().filter(|c| !plan.lane.contains(&c.id)).map(|c| c.id).collect();
    }
    Some(plan)
}

/// A thread that asked: a handle of it kept open, what it is, its kind.
struct Asked {
    thread: usize,
    what: String,
    kind: Kind,
}

static ASKED: Mutex<Vec<Asked>> = Mutex::new(Vec::new());

/// The ids a thread of `kind` is given, the `n`th of the busy ones if it is
/// one.
fn ids_for<'a>(plan: &'a Plan, kind: Kind, busy_before: usize) -> (&'a [u32], &'static str) {
    match kind {
        Kind::Lane if !plan.lane.is_empty() => (&plan.lane, "a core of its own"),
        Kind::Busy if busy_before < plan.busy.len() => (&plan.busy[busy_before], "a core of its own"),
        _ => (&plan.pool, "the P cores the others leave"),
    }
}

/// The calling thread asks for the cores of its kind, if `VICTORY_P_CORES=1`
/// (any kind) or `VICTORY_LANE_CORE=1` (the submission thread, and the
/// threads that called `framelog::keep_on_fast_cores` before: `legacy`, which
/// are kept off the lane core). True if it was done; false if neither is
/// asked for or all cores are alike (said once), so the caller does what it
/// did before.
pub(crate) fn prefer(what: &str, kind: Kind, legacy: bool) -> bool {
    use windows_sys::Win32::System::Threading::{GetCurrentThreadId, OpenThread, THREAD_SET_LIMITED_INFORMATION};
    let (p, lane) = (p_cores(), lane_core());
    if !p && !(lane && (kind == Kind::Lane || legacy)) {
        return false;
    }
    if plan_now().is_none() {
        return false;
    }
    // SAFETY: our own thread; the handle stays open for the thread's life.
    let thread = unsafe { OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, GetCurrentThreadId()) };
    if thread.is_null() {
        tracing::warn!(what, error = %std::io::Error::last_os_error(), "the thread could not be given cores (VICTORY_P_CORES)");
        return false;
    }
    ASKED.lock().unwrap_or_else(|e| e.into_inner()).push(Asked { thread: thread as usize, what: what.to_owned(), kind });
    settle(Some(what));
    true
}

/// The plan, and said once what the processor is and what is done.
fn plan_now() -> Option<&'static Plan> {
    static PLAN: std::sync::OnceLock<Option<(Vec<CpuSet>, Plan)>> = std::sync::OnceLock::new();
    PLAN.get_or_init(|| {
        let sets = cpu_sets();
        let (p, lane) = (p_cores(), lane_core());
        let own = usize::from(lane || p) * (if p { own_most() } else { 1 });
        let made = plan(&sets, lane || p, own);
        let mut classes: Vec<u8> = sets.iter().map(|c| c.class).collect();
        classes.sort_unstable();
        classes.dedup();
        match &made {
            None => tracing::info!(
                logical = sets.len(),
                "VICTORY_P_CORES / VICTORY_LANE_CORE: the {} logical processors are of one efficiency class: nothing is done",
                sets.len()
            ),
            Some(plan) => {
                let line: Vec<String> = classes
                    .iter()
                    .rev()
                    .map(|&class| {
                        let of: Vec<&CpuSet> = sets.iter().filter(|c| c.class == class).collect();
                        let mut cores: Vec<u8> = of.iter().map(|c| c.core).collect();
                        cores.sort_unstable();
                        cores.dedup();
                        format!("class {class}: {} logical processors on {} cores", of.len(), cores.len())
                    })
                    .collect();
                tracing::info!(
                    lane = plan.lane.len(),
                    own_for_busy = plan.busy.len(),
                    pool = plan.pool.len(),
                    "processor classes found ({}); the highest class is taken for P cores (VICTORY_P_CORES={}, VICTORY_LANE_CORE={})",
                    line.join("; "),
                    u8::from(p),
                    u8::from(lane)
                );
            }
        }
        made.map(|plan| (sets, plan))
    })
    .as_ref()
    .map(|(_, plan)| plan)
}

/// Tells every thread that asked, and the process, what the plan gives them
/// now; `said`: the one just registered, whose place is logged.
fn settle(said: Option<&str>) {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetProcessDefaultCpuSets, SetThreadSelectedCpuSets};
    let Some(plan) = plan_now() else { return };
    let asked = ASKED.lock().unwrap_or_else(|e| e.into_inner());
    let mut busy_before = 0;
    for a in asked.iter() {
        let (ids, place) = ids_for(plan, a.kind, busy_before);
        busy_before += usize::from(a.kind == Kind::Busy);
        // SAFETY: a handle kept open and the system's own set ids.
        let ok = unsafe { SetThreadSelectedCpuSets(a.thread as _, ids.as_ptr(), ids.len() as u32) } != 0;
        if said == Some(a.what.as_str()) {
            if ok {
                tracing::info!(thread = a.what.as_str(), kind = ?a.kind, sets = ids.len(), "the thread prefers {place} (VICTORY_P_CORES / VICTORY_LANE_CORE)");
            } else {
                tracing::warn!(thread = a.what.as_str(), error = %std::io::Error::last_os_error(), "the thread could not be given cores (VICTORY_P_CORES / VICTORY_LANE_CORE)");
            }
        }
    }
    if !plan.default.is_empty() && lane_core() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            // SAFETY: our own process and the system's own set ids.
            let ok = unsafe { SetProcessDefaultCpuSets(GetCurrentProcess(), plan.default.as_ptr(), plan.default.len() as u32) } != 0;
            tracing::info!(ok, sets = plan.default.len(), core = ?plan.lane, "the submission thread has a core of its own: the process's other threads are kept off it (VICTORY_LANE_CORE=1)");
        });
    }
}

/// For the title's thread that submits lists: asks once, on the first call
/// of this thread.
pub(crate) fn this_title_thread() {
    thread_local! {
        static DONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    if !p_cores() || DONE.get() {
        return;
    }
    DONE.set(true);
    prefer("the title's submitting thread", Kind::Pool, false);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An Intel hybrid of six P cores (two logical processors each, ids 256
    /// on) and eight E cores.
    fn hybrid() -> Vec<CpuSet> {
        (0..12u8)
            .map(|n| CpuSet { id: 256 + u32::from(n), processor: n, class: 1, core: n / 2 })
            .chain((12..20u8).map(|n| CpuSet { id: 256 + u32::from(n), processor: n, class: 0, core: n - 6 }))
            .collect()
    }

    #[test]
    fn the_last_p_cores_are_taken_and_the_rest_is_the_pool() {
        let plan = plan(&hybrid(), true, 2).expect("a hybrid");
        assert_eq!(plan.lane, [266, 267], "the last P core, both its logical processors");
        assert_eq!(plan.busy, [[264, 265]], "the one before it, for the relay");
        assert_eq!(plan.pool, (256..264).collect::<Vec<u32>>(), "the other four P cores");
        assert_eq!(plan.default.len(), 18, "every set but the lane core's");
        assert!(!plan.default.contains(&266) && !plan.default.contains(&267));
        assert!(plan.pool.iter().all(|id| !plan.lane.contains(id) && !plan.busy[0].contains(id)));
    }

    #[test]
    fn with_the_lane_core_alone_one_core_is_taken() {
        let plan = plan(&hybrid(), true, 1).unwrap();
        assert_eq!((plan.lane.len(), plan.busy.len(), plan.pool.len()), (2, 0, 10));
    }

    #[test]
    fn busy_threads_without_a_lane_take_the_first_core_taken() {
        let plan = plan(&hybrid(), false, 2).unwrap();
        assert!(plan.lane.is_empty() && plan.default.is_empty());
        assert_eq!(plan.busy, [vec![266, 267], vec![264, 265]]);
    }

    #[test]
    fn one_class_or_few_p_cores_do_nothing_or_take_no_core() {
        let alike: Vec<CpuSet> = (0..8u8).map(|n| CpuSet { id: u32::from(n), processor: n, class: 0, core: n }).collect();
        assert!(plan(&alike, true, 2).is_none());
        assert!(plan(&[], true, 2).is_none());
        // Four P cores and some E: all P cores are the pool.
        let small: Vec<CpuSet> = (0..8u8)
            .map(|n| CpuSet { id: u32::from(n), processor: n, class: 1, core: n / 2 })
            .chain((8..12u8).map(|n| CpuSet { id: u32::from(n), processor: n, class: 0, core: n }))
            .collect();
        let plan = plan(&small, true, 2).unwrap();
        assert!(plan.lane.is_empty() && plan.busy.is_empty());
        assert_eq!(plan.pool, (0..8).collect::<Vec<u32>>());
    }

    #[test]
    fn a_thread_gets_its_place_in_the_order_it_asked() {
        let plan = plan(&hybrid(), true, 2).unwrap();
        assert_eq!(ids_for(&plan, Kind::Lane, 0).0, [266, 267]);
        assert_eq!(ids_for(&plan, Kind::Busy, 0).0, [264, 265]);
        // A second busy thread has no core left of its own.
        assert_eq!(ids_for(&plan, Kind::Busy, 1).0, plan.pool);
        assert_eq!(ids_for(&plan, Kind::Pool, 0).0, plan.pool);
    }

    #[test]
    fn this_machines_sets_can_be_read() {
        let sets = cpu_sets();
        assert!(!sets.is_empty());
        // Said for whoever runs it with `--nocapture`: the classes found and
        // what a plan makes of them here.
        let classes: std::collections::BTreeSet<u8> = sets.iter().map(|c| c.class).collect();
        eprintln!("{} logical processors, classes {classes:?}: plan {:?}", sets.len(), plan(&sets, true, 2));
        // A hybrid has a plan with the P cores apart from the pool's.
        if classes.len() > 1 {
            let plan = plan(&sets, true, 2).expect("two classes");
            assert!(plan.pool.iter().all(|id| !plan.lane.contains(id)) || plan.lane.is_empty());
        }
    }
}
