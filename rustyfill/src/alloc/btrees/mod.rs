//! Fallible B-tree map entry operations.
//!
//! The [`entry`] submodule provides three traits covering the full API surface:
//!
//! - [`entry::TryBTreeMap`] — a single-call `try_insert(key, value)` on
//!   `BTreeMap<K, V>` (plain insert semantics). Routes through the standard
//!   entry API; only the split cascade can fail, reported as a [`Result`].
//! - [`entry::TryBTreeMapEntry`] — fallible `try_or_insert`-family methods on
//!   the standard `BTreeMap` Entry API (`Entry<'_, K, V>`).
//! - [`entry::TryBTreeMapVacantEntry`] — fallible `try_insert` on the standard
//!   `VacantEntry<'_, K, V>`.
//!
//! All three manipulate the internal B-tree structure directly using mirrored
//! types from [`rustyfill_sys`], handling OOM by returning [`Result`] instead
//! of panicking. The key lookup performed by the standard `BTreeMap::entry()`
//! is allocation-free (a pure pointer descent), so all three traits are safe to
//! call under an intermittently failing allocator — only the split cascade can
//! fail, and it does so gracefully. Requires the `btree-entry` feature.
//!
//! An earlier revision of this crate shipped `catch_unwind`-based `TryBTreeMap`
//! and `TryBTreeSet` wrappers (the `panic` feature). Those have been removed:
//! `catch_unwind` cannot intercept genuine out-of-memory conditions (`handle_alloc_error`
//! calls `abort()` rather than panicking), and even when it does catch an allocation
//! panic, std's `BTreeMap`/`BTreeSet` make no data-integrity guarantees across a
//! panicked mutation — e.g. elements stranded mid-promotion during a split can be
//! silently lost.

#[cfg(feature = "btree-entry")]
pub mod entry;
#[cfg(feature = "btree-entry")]
mod entry_try_extend;
