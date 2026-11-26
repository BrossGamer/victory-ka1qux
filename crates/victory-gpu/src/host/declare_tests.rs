//! A range shaders write is declared once until the list next writes title
//! memory (`VICTORY_DECLARE_ONCE=1`, [`Deferred::declare`]).

use super::*;

#[test]
fn a_range_is_declared_once_until_the_list_writes_title_memory() {
    let mut deferred = Deferred::default();
    let range = (0x10, 0x4000, 256, false);
    // The first declaration is recorded, the repeats are not.
    assert!(deferred.declare(range, true));
    assert!(!deferred.declare(range, true));
    assert!(!deferred.declare(range, true));
    // Another copy, another range, another length, or the same range
    // written through marked views: each is a declaration of its own.
    assert!(deferred.declare((0x20, 0x4000, 256, false), true));
    assert!(deferred.declare((0x10, 0x4100, 256, false), true));
    assert!(deferred.declare((0x10, 0x4000, 512, false), true));
    assert!(deferred.declare((0x10, 0x4000, 256, true), true));
    // Title memory written in the list (a copy, or work that may write it):
    // what the shaders write is declared again after it, in its order.
    deferred.title_written();
    assert!(deferred.declare(range, true));
    assert!(!deferred.declare(range, true));
}

#[test]
fn every_declaration_is_recorded_without_the_switch() {
    let mut deferred = Deferred::default();
    let range = (0x10, 0x4000, 256, false);
    for _ in 0..4 {
        assert!(deferred.declare(range, false));
    }
    // And nothing was kept to leave a later declaration out.
    assert!(deferred.gpu_written.is_empty());
}

#[test]
fn a_reset_forgets_what_was_declared() {
    let mut deferred = Deferred::default();
    let range = (0x10, 0x4000, 256, false);
    assert!(deferred.declare(range, true));
    deferred.clear();
    assert!(deferred.declare(range, true));
}
