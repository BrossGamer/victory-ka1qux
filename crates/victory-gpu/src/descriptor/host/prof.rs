//! Section timers for the mirroring of a slot ([`super::HostHeap::mirror_as`]
//! and what it calls), for the benchmark of `mirror_bench` only.
//!
//! [`part!`] wraps an expression and, in a test build while a thread has the
//! timers on ([`start`]), adds the time spent in it to its [`Part`]; the time
//! of a part nested in another is the inner one's alone, so the parts add up
//! to the whole. Outside tests it is the expression and nothing else.

/// The sections, as the benchmark names them.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum Part {
    /// What is in no other part: the reading of the record, the fast path's
    /// test, the epochs of material records, the branches between the parts.
    /// (The part the timers start in, never entered by name.)
    #[allow(dead_code)]
    Other,
    /// The lock of `made`, taken.
    MadeLock,
    /// The view the record names, found (the table's lock, the copy of its words).
    ViewAt,
    /// The fingerprint of a buffer view (`same_view_signature`).
    Signature,
    /// The test whether the slot shows the view already (`shows_same`).
    ShowsSame,
    /// The books of a write (`note_buffer`): the queue of textures picked by index.
    BooksQueue,
    /// The books: the spans of placed textures (their lock and tree).
    BooksSpans,
    /// The books: the stride and address tables shaders read (mapped upload memory).
    BooksTables,
    /// The books: the sets of buffer view slots the pass over them reads.
    BooksBuffers,
    /// The slots a newer view made stale, let go.
    Stale,
    /// A view of a texture with no host copy yet waits for it.
    Waiting,
    /// `make` outside the parts below: the match, the checks of the view.
    Make,
    /// The host copy of the memory a view reads, found (`shadow::find`).
    MakeFind,
    /// The host copy of a view's own bytes (`shadow::exact`).
    MakeExact,
    /// The host's `Create*View` and the hold of the copy it views.
    MakeCreate,
    /// The slot's resource kept (`keep`: the mark table, the fingerprint, the piece's lock, the old one retired).
    Keep,
    /// The levels-on-demand books of the slot (`note_tiled`).
    Tiled,
    /// The slot's fingerprint of the view shown (`note_shown`).
    Shown,
    /// The range the pass over buffer views is given (`buffer_copy` and `take_range`).
    Range,
    /// A null view made for a slot the host cannot show (`null_for`, `make_null`).
    Null,
}

#[cfg(test)]
pub(crate) const PARTS: usize = 20;
#[cfg(test)]
pub(crate) const NAMES: [&str; PARTS] = [
    "other (record read, branches)",
    "made lock",
    "view_at (views table)",
    "signature (hash of view)",
    "shows_same test",
    "books: texture queue",
    "books: placed spans",
    "books: stride/address tables",
    "books: buffer view sets",
    "stale slots let go",
    "waiting texture",
    "make (checks)",
    "make: shadow::find",
    "make: shadow::exact",
    "make: Create*View + hold",
    "keep (mark, shown, kept piece)",
    "tiled books",
    "note_shown",
    "range (buffer_copy + take_range)",
    "null views",
];

#[cfg(test)]
thread_local! {
    static ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static CURRENT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static LAST: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static TICKS: [std::cell::Cell<u64>; PARTS] = const { [const { std::cell::Cell::new(0) }; PARTS] };
    static CALLS: [std::cell::Cell<u64>; PARTS] = const { [const { std::cell::Cell::new(0) }; PARTS] };
}

#[cfg(test)]
#[allow(unused_unsafe)]
pub(crate) fn now_ticks() -> u64 {
    // SAFETY: reads the time stamp counter.
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// The calling thread's timers on, from zero, in [`Part::Other`].
#[cfg(test)]
pub(crate) fn start() {
    TICKS.with(|t| t.iter().for_each(|c| c.set(0)));
    CALLS.with(|t| t.iter().for_each(|c| c.set(0)));
    CURRENT.set(0);
    LAST.set(now_ticks());
    ON.set(true);
}

/// The timers off; the ticks and calls of each part since [`start`].
#[cfg(test)]
pub(crate) fn stop() -> ([u64; PARTS], [u64; PARTS]) {
    if ON.get() {
        let at = now_ticks();
        TICKS.with(|t| t[CURRENT.get()].set(t[CURRENT.get()].get() + at - LAST.get()));
    }
    ON.set(false);
    (TICKS.with(|t| t.each_ref().map(std::cell::Cell::get)), CALLS.with(|t| t.each_ref().map(std::cell::Cell::get)))
}

/// Entered `part`: the part left, for [`leave`]; `usize::MAX` with the timers off.
#[cfg(test)]
pub(crate) fn enter(part: Part) -> usize {
    if !ON.get() {
        return usize::MAX;
    }
    let at = now_ticks();
    let was = CURRENT.get();
    TICKS.with(|t| t[was].set(t[was].get() + at - LAST.get()));
    CALLS.with(|c| c[part as usize].set(c[part as usize].get() + 1));
    CURRENT.set(part as usize);
    LAST.set(at);
    was
}

/// Back in the part `enter` returned.
#[cfg(test)]
pub(crate) fn leave(was: usize) {
    if was == usize::MAX {
        return;
    }
    let at = now_ticks();
    let part = CURRENT.get();
    TICKS.with(|t| t[part].set(t[part].get() + at - LAST.get()));
    CURRENT.set(was);
    LAST.set(at);
}

/// `part!(prof::Part::X, expression)`: the expression, its time counted in
/// `X` in a test build. The expression must not return or `?` out of the
/// function.
macro_rules! part {
    ($part:expr, $e:expr) => {{
        #[cfg(test)]
        let __was = $crate::descriptor::host::prof::enter($part);
        let __value = $e;
        #[cfg(test)]
        $crate::descriptor::host::prof::leave(__was);
        __value
    }};
}
pub(crate) use part;
