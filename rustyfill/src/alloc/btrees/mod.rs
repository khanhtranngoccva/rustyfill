//! Fallible B-tree map entry operations.
//!
//! This module re-exports three traits covering the full API surface:
//!
//! - [`TryBTreeMap`] — a single-call `try_insert(key, value)` on
//!   `BTreeMap<K, V>` (plain insert semantics). Routes through the standard
//!   entry API; only the split cascade can fail, reported as a [`Result`].
//! - [`TryBTreeMapEntry`] — fallible `try_or_insert`-family methods on
//!   the standard `BTreeMap` Entry API (`Entry<'_, K, V>`).
//! - [`TryBTreeMapVacantEntry`] — fallible `try_insert` on the standard
//!   `VacantEntry<'_, K, V>`.
//!
//! All three manipulate the internal B-tree structure directly using mirrored
//! types from [`rustyfill_sys`], handling OOM by returning [`Result`] instead
//! of panicking. The key lookup performed by the standard `BTreeMap::entry()`
//! is allocation-free (a pure pointer descent), so all three traits are safe to
//! call under an intermittently failing allocator — only the split cascade can
//! fail, and it does so gracefully.
//!
//! An earlier revision of this crate shipped `catch_unwind`-based `TryBTreeMap`
//! and `TryBTreeSet` wrappers (the `panic` feature). Those have been removed:
//! `catch_unwind` cannot intercept genuine out-of-memory conditions (`handle_alloc_error`
//! calls `abort()` rather than panicking), and even when it does catch an allocation
//! panic, std's `BTreeMap`/`BTreeSet` make no data-integrity guarantees across a
//! panicked mutation — e.g. elements stranded mid-promotion during a split can be
//! silently lost.

mod entry;
mod entry_try_extend;
/// Fallible `TryDebug` / `TryDisplay` impls for `BTreeMap<K, V>` and
/// `BTreeSet<T>`.
mod fmt_;

// Re-export the public API of the (private) `entry` module so callers can use
// the short paths `crate::alloc::btrees::{...}` without reaching into a
// submodule. The implementation lives in `entry`; these are plain aliases.
pub use entry::{
    TryBTreeMap, TryBTreeMapEntry, TryBTreeMapEntryWithError, TryBTreeMapVacantEntry, TryBTreeSet,
    TryBTreeWithCloneError,
};
