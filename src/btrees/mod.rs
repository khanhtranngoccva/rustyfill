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
//!
//! If you must use this anyway, you will have to use [`std::alloc::set_alloc_error_hook`].
mod btreemap_;
mod btreeset_;

#[deprecated(
    since = "0.2.0",
    note = "catch_unwind cannot intercept OOM aborts; use the `scapegoat` crate for fallible trees"
)]
pub use btreemap_::{TryBTreeMap, TryBTreeMapError};

#[deprecated(
    since = "0.2.0",
    note = "catch_unwind cannot intercept OOM aborts; use the `scapegoat` crate for fallible trees"
)]
pub use btreeset_::{TryBTreeSet, TryBTreeSetError};
