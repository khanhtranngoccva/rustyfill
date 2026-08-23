//! Generic fallible-extension traits for collections.
//!
//! These traits factor out the "fallibly extend a collection" contract that is
//! shared by several concrete collection types (`Vec`, `VecDeque`, `HashMap`,
//! `HashSet`, and, under the `unstable` feature, `DashMap`/`DashSet`). They let
//! callers write generic code over any fallible-extensible collection without
//! depending on a specific per-collection trait.
//!
//! Per-collection implementations live in each collection's own module:
//!
//! | Collection          | Implementation module                          | Feature     |
//! |---------------------|------------------------------------------------|-------------|
//! | `Vec<T>`           | [`crate::alloc::vec`]                          | (always)    |
//! | `VecDeque<T>`      | [`crate::alloc::vecdeque`]                     | (always)    |
//! | `HashMap<K,V,S>`   | [`crate::std::hashmap`]                        | `std`       |
//! | `HashSet<T, S>`    | [`crate::std::hashset`]                        | `std`       |
//! | `BTreeMap<K,V>`    | [`crate::alloc::btrees`]                       | `std`         |
//! | `BTreeSet<T>`      | [`crate::alloc::btrees`]                       | `std`         |
//! | `DashMap<K,V,S>`   | [`crate::dashmap`]                             | `unstable`  |
//! | `DashSet<T, S>`    | [`crate::dashmap`]                             | `unstable`  |
//!
//! # [`TryExtend`]
//!
//! Fallibly extend a collection from an iterator source. On failure the error
//! carries a [`Resumable`] wrapping the remainder of
//! the source (plus any consumed-but-uncommitted element), which can be passed
//! straight back into another `try_extend` call to retry. Because both plain
//! iterators and [`Resumable`] wrappers implement
//! [`ResumableSource`](crate::recovery::ResumableSource) with the same inner
//! type, the error type stays identical across retries — it never grows.
//!
//! ```rust,ignore
//! use rustyfill::prelude::*;
//!
//! let mut v: Vec<i32> = Vec::new();
//! let items = 0..10_000;
//!
//! // First attempt — fails on OOM.
//! let remaining = match v.try_extend(items) {
//!     Ok(()) => return,
//!     Err((resumable, _err)) => resumable.into_remainder(),
//! };
//!
//! // Retry with the remainder wrapped in a Resumable.
//! let remaining = match v.try_extend(Resumable::from_remainder(remaining)) {
//!     Ok(()) => return,
//!     Err((resumable, _err)) => resumable.into_remainder(),
//! };
//! ```
//!
//! # [`TryExtendFromSlice`]
//!
//! Fallibly extend a collection by cloning elements from a slice. The error
//! payload always contains the *remainder* — the unprocessed tail of the input
//! slice — alongside the failure reason. On a capacity failure the remainder is
//! the full input (nothing was consumed); on a mid-way clone failure it is the
//! tail starting at the failing index.
//!
//! Implementors do not rollback on failure.
//!
//! ## Implementors
//!
//! | Collection          | Item        | Error                                  | Remainder               | Feature     |
//! |---------------------|-------------|----------------------------------------|-------------------------|-------------|
//! | `Vec<T>`           | `T`         | `TryVecWithCloneError`                 | `&'s [T]`                | (always)    |
//! | `VecDeque<T>`      | `T`         | `TryVecDequeWithCloneError`            | `&'s [T]`                | (always)    |
//! | `HashMap<K,V,S>`   | `(K, V)`    | `TryHashMapWithCloneError`             | `&'s [(K, V)]`           | `std`       |
//! | `HashSet<T, S>`    | `T`         | `TryHashSetWithCloneError`             | `&'s [T]`                | `std`       |
//! | `BTreeMap<K,V>`    | `(K, V)`    | `TryBTreeWithCloneError`               | `&'s [(K, V)]`           | `std`       |
//! | `BTreeSet<T>`      | `T`         | `TryBTreeWithCloneError`               | `&'s [T]`                | `std`       |
//! | `DashMap<K,V,S>`   | `(K, V)`    | `TryDashMapWithCloneError`             | `&'s [(K, V)]`           | `unstable`  |
//! | `DashSet<T, S>`    | `T`         | `TryDashSetWithCloneError`             | `&'s [T]`                | `unstable`  |
//!
//! ## [`TryExtend`] errors
//!
//! Iterator-based extension moves items into the collection, so only capacity
//! reservation can fail. All implementations use `TryReserveError` directly
//! (except BTreeMap/BTreeSet which use `AllocError`, since their insertion
//! allocates nodes without a separate reserve phase).

use crate::recovery::Resumable;

/// Fallibly extend a collection from an iterator source.
///
/// The source may be any [`ResumableSource`](crate::recovery::ResumableSource):
/// a plain `IntoIterator`, or a [`Resumable`] wrapper carrying a stranded head
/// element plus a remainder iterator from a previous failed attempt. Passing a
/// [`Resumable`] back in preserves the same inner iterator type, so the error
/// type is stable across retries.
///
/// See the [module docs](self) for an example.
pub trait TryExtend<Item>: Sized {
    /// The error returned on failure, paired with a [`Resumable`] over the
    /// source's inner iterator so the caller can retry.
    type Error;

    /// Fallibly extend `self` with all items produced by `source`.
    fn try_extend<S>(&mut self, source: S) -> Result<(), (Resumable<S::Inner>, Self::Error)>
    where
        S: crate::recovery::ResumableSource<Item = Item>;

    /// Alias for [`Self::try_extend`].
    fn fallible_extend<S>(&mut self, source: S) -> Result<(), (Resumable<S::Inner>, Self::Error)>
    where
        S: crate::recovery::ResumableSource<Item = Item>,
    {
        Self::try_extend(self, source)
    }
}

/// Fallibly extend a collection by cloning elements from a slice.
///
/// On failure the error is a tuple of the **remainder** (the unprocessed tail
/// of the input slice) and the underlying error. Callers can retry with just
/// the remainder once memory pressure has eased.
///
/// A mid-way clone failure does not trigger a rollback.
pub trait TryExtendFromSlice<'s, Item>: Sized {
    /// The error type accompanying the remainder slice.
    type Error;

    /// Fallibly extend `self` by cloning each element of `other`.
    fn try_extend_from_slice(&mut self, other: &'s [Item])
    -> Result<(), (&'s [Item], Self::Error)>;

    /// Alias for [`Self::try_extend_from_slice`].
    fn fallible_extend_from_slice(
        &mut self,
        other: &'s [Item],
    ) -> Result<(), (&'s [Item], Self::Error)> {
        Self::try_extend_from_slice(self, other)
    }
}
