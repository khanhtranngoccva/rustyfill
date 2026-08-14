//! Fallible B-tree map and set operations.
//!
//! # Deprecated
//!
//! This module is deprecated. The `catch_unwind`-based approach cannot survive
//! genuine out-of-memory conditions: `handle_alloc_error` calls `abort()` rather
//! than panicking, so `catch_unwind` never gets a chance to catch anything.
//! Additionally, Rust's `-Z oom=panic` flag was removed in 1.94.0 due to
//! reentrancy soundness concerns, eliminating the only mechanism that would have
//! made this work.
//!
//! Use the [`scapegoat`](https://crates.io/crates/scapegoat) crate instead, which
//! provides a fully fallible scapegoat tree implementation with proper OOM handling.
//! Or if you still want to use a version that is similar to std (but still not ABI compatible!),
//! refer to [`fallible_collections`](https://crates.io/crates/fallible_collections)
//!
//! If you must use this anyway, you will have to use [`::lang_alloc::alloc::set_alloc_error_hook`].
//!
//! # Entry API (non-deprecated)
//!
//! The [`entry`] submodule provides [`TryBTreeMapEntry`], a trait that adds
//! `try_insert_entry` to `BTreeMap<K, V>` via direct manipulation of internal
//! node allocations. Unlike the deprecated `TryBTreeMap`, this approach properly
//! handles OOM by returning [`Result`] instead of relying on `catch_unwind`.
//! It requires the `btree-entry` feature and depends on [`rustyfill-sys`] bindings.

#[cfg(feature = "panic")]
mod btreemap_;
#[cfg(feature = "panic")]
mod btreeset_;

#[cfg(feature = "btree-entry")]
pub mod entry;

#[cfg(feature = "panic")]
#[deprecated(
    since = "0.2.0",
    note = "catch_unwind cannot intercept OOM aborts; use the `scapegoat` crate for fallible trees"
)]
pub use btreemap_::{TryBTreeMap, TryBTreeMapError};

#[cfg(feature = "panic")]
#[deprecated(
    since = "0.2.0",
    note = "catch_unwind cannot intercept OOM aborts; use the `scapegoat` crate for fallible trees"
)]
pub use btreeset_::{TryBTreeSet, TryBTreeSetError};
