//! Fallible dash map operations.
//!
//! Provides the [`TryDashMap`] trait with methods that mirror common `DashMap`
//! constructors and mutating operations but return [`Result`] to handle allocation
//! failures gracefully.
//!
//! # Design
//!
//! `TryDashMap` is implemented for `dashmap::DashMap<K, V, S>`. Unlike `HashMap`,
//! `DashMap` is a concurrent hash map whose mutation methods take `&self` (not
//! `&mut self`) and internally lock individual shards. The fallible methods
//! reserve capacity on each shard before inserting so that the subsequent
//! operation cannot panic on out-of-memory.
//!
//! The `raw-api` feature of `dashmap` is used to access per-shard internals
//! (`shards_mut()`, `determine_map()`) so that insertion can be done on a
//! single locked shard rather than acquiring all locks simultaneously.

use hashbrown::raw::RawTable;

use crate::alloc::{AllocError, TryReserveError};
use crate::prelude::TryDefault;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_fmt::{TryDebug, helpers::FormatterExt};
use lang_core::cmp::Eq;
use lang_core::fmt;
use lang_core::mem;
use lang_core::ptr;
use lang_std::hash::{BuildHasher, Hash, RandomState};

type DashMap<K, V, S = RandomState> = dashmap::DashMap<K, V, S>;

use crate::dashmap::mapref::{Entry, OccupiedEntry, VacantEntry};

// ── Error types ───────────────────────────────────────────────────────────────

/// Error returned by blocking [`TryDashMap`] operations.
///
/// Wraps the ways a DashMap operation can fail: a reserve failure from
/// [`DashMap::try_reserve`](dashmap::DashMap::try_reserve) or a clone failure
/// when an element's `try_clone` cannot allocate its internal buffers.
#[derive(Debug)]
pub enum TryDashMapError {
    /// A raw heap allocation failed (no collection involved).
    Alloc(AllocError),
    /// A capacity reservation on the DashMap failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An element clone failed during a method that requires [`TryClone`].
    Clone(TryCloneError),
    /// An arithmetic overflow occurred while computing required capacity.
    Overflow,
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Display for TryDashMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "dash map operation failed: heap allocation error"),
            Self::Reserve(e) => write!(f, "dash map operation failed: {}", e),
            Self::Clone(e) => write!(f, "dash map operation failed: {}", e),
            Self::Overflow => {
                write!(
                    f,
                    "dash map operation failed: capacity calculation overflowed"
                )
            }
            Self::Other(msg) => write!(f, "dash map operation failed: {}", msg),
        }
    }
}

impl From<AllocError> for TryDashMapError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

impl From<dashmap::TryReserveError> for TryDashMapError {
    fn from(e: dashmap::TryReserveError) -> Self {
        Self::Reserve(TryReserveError::from(e))
    }
}

impl From<TryCloneError> for TryDashMapError {
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

impl From<crate::try_default::TryDefaultError> for TryDashMapError {
    fn from(err: crate::try_default::TryDefaultError) -> Self {
        match err {
            crate::try_default::TryDefaultError::Alloc(e) => Self::Alloc(e),
            crate::try_default::TryDefaultError::Reserve(e) => Self::Reserve(e),
            crate::try_default::TryDefaultError::Overflow => Self::Overflow,
            crate::try_default::TryDefaultError::Other(msg) => Self::Other(msg),
        }
    }
}

impl TryDebug for TryDashMapError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => f
                .try_debug_struct("TryDashMapError::Alloc")
                .field("0", e)
                .finish(),
            Self::Reserve(e) => f
                .try_debug_struct("TryDashMapError::Reserve")
                .field("0", e)
                .finish(),
            Self::Clone(e) => f
                .try_debug_struct("TryDashMapError::Clone")
                .field("0", e)
                .finish(),
            Self::Overflow => f.write_str("TryDashMapError::Overflow"),
            Self::Other(msg) => f
                .try_debug_struct("TryDashMapError::Other")
                .field("0", msg)
                .finish(),
        }
    }
}

/// Error returned by non-blocking [`TryDashMap`] operations.
///
/// Extends [`TryDashMapError`] with a [`Locked`](Self::Locked) variant for when
/// the target shard is held by another writer and the caller chose not to block.
#[derive(Debug)]
pub enum TryDashMapNonblockError {
    /// A raw heap allocation failed (no collection involved).
    Alloc(AllocError),
    /// A capacity reservation on the DashMap failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An element clone failed during a method that requires [`TryClone`].
    Clone(TryCloneError),
    /// An arithmetic overflow occurred while computing required capacity.
    Overflow,
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
    /// The target shard is currently locked by another reader or writer.
    Locked,
}

impl fmt::Display for TryDashMapNonblockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "dash map operation failed: heap allocation error"),
            Self::Reserve(e) => write!(f, "dash map operation failed: {}", e),
            Self::Clone(e) => write!(f, "dash map operation failed: {}", e),
            Self::Overflow => {
                write!(
                    f,
                    "dash map operation failed: capacity calculation overflowed"
                )
            }
            Self::Other(msg) => write!(f, "dash map operation failed: {}", msg),
            Self::Locked => write!(f, "dash map operation failed: shard locked"),
        }
    }
}

impl From<TryDashMapError> for TryDashMapNonblockError {
    fn from(err: TryDashMapError) -> Self {
        match err {
            TryDashMapError::Alloc(e) => Self::Alloc(e),
            TryDashMapError::Reserve(r) => Self::Reserve(r),
            TryDashMapError::Clone(c) => Self::Clone(c),
            TryDashMapError::Overflow => Self::Overflow,
            TryDashMapError::Other(m) => Self::Other(m),
        }
    }
}

impl TryDebug for TryDashMapNonblockError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => f
                .try_debug_struct("TryDashMapNonblockError::Alloc")
                .field("0", e)
                .finish(),
            Self::Reserve(e) => f
                .try_debug_struct("TryDashMapNonblockError::Reserve")
                .field("0", e)
                .finish(),
            Self::Clone(e) => f
                .try_debug_struct("TryDashMapNonblockError::Clone")
                .field("0", e)
                .finish(),
            Self::Overflow => f.write_str("TryDashMapNonblockError::Overflow"),
            Self::Other(msg) => f
                .try_debug_struct("TryDashMapNonblockError::Other")
                .field("0", msg)
                .finish(),
            Self::Locked => f.write_str("TryDashMapNonblockError::Locked"),
        }
    }
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// A trait for fallible DashMap operations.
///
/// Implemented for `dashmap::DashMap<K, V, S>`. Mirrors the most commonly-used
/// `DashMap` methods that can fail due to allocation pressure, returning
/// [`Result`] values that propagate [`TryDashMapError`] on failure.
///
/// Unlike `std::collections::HashMap`, `DashMap` is concurrent: mutation
/// methods take `&self` and lock individual shards at runtime. These fallible
/// wrappers pre-reserve capacity on the relevant shard(s) so that the actual
/// insert does not panic on OOM.
pub trait TryDashMap<K, V, S = RandomState>: Sized {
    // ── Construction ────────────────────────────────────────────────────────

    /// Fallibly construct an empty `DashMap` with a default-constructed hasher.
    ///
    /// Unlike [`Self::try_with_capacity`], which hardcodes [`RandomState`] and
    /// may panic on first use in a new thread (due to thread-local seeding),
    /// this method uses [`TryDefault`] to construct the hasher fallibly. If
    /// hasher construction fails (e.g. `RandomState` panics during seed
    /// initialization), the error is returned rather than unwinding.
    ///
    /// Requires `S: [`TryDefault`]` so that hasher creation is safe even when
    /// it involves runtime allocation or thread-local state.
    fn try_new() -> Result<DashMap<K, V, S>, TryDashMapError>
    where
        S: TryDefault;

    /// Fallibly construct an empty `DashMap` with at least enough capacity for
    /// `capacity` elements.
    ///
    /// Constructs the hasher via [`TryDefault`] (same as [`Self::try_new`]),
    /// then reserves capacity for `capacity` elements. Returns
    /// [`TryDashMapError::Reserve`] if the capacity reservation fails, or
    /// [`TryDashMapError::Other`] if hasher construction panics.
    ///
    /// Requires `S: [`TryDefault`]` so that hasher creation is safe even when
    /// it involves runtime allocation or thread-local state.
    fn try_with_capacity(capacity: usize) -> Result<DashMap<K, V, S>, TryDashMapError>
    where
        S: TryDefault;

    /// Fallibly construct an empty `DashMap` with at least enough capacity for
    /// `capacity` elements, using the provided hash builder.
    ///
    /// Returns [`TryDashMapError::Reserve`] if the initial allocation fails.
    /// Equivalent to [`DashMap::with_capacity_and_hasher`](dashmap::DashMap::with_capacity_and_hasher)
    /// but fallible.
    fn try_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<DashMap<K, V, S>, TryDashMapError>;

    // ── Insertion ───────────────────────────────────────────────────────────

    /// Fallibly insert a key-value pair into the map, always replacing any
    /// existing value for the same key.
    ///
    /// Reserves capacity before inserting so this method never panics on
    /// out-of-memory. Returns [`TryDashMapError::Reserve`] if the capacity
    /// reservation fails.
    ///
    /// Returns `None` if the key was not previously present, or
    /// `Some(old_value)` if the key existed and was replaced.
    fn try_insert(&self, key: K, value: V) -> Result<Option<V>, TryDashMapError>
    where
        K: Eq + Hash;

    /// Like [`Self::try_insert`] but returns ownership of `key` and `value`
    /// back on allocation failure.
    fn try_insert_give_back(&self, key: K, value: V) -> Result<Option<V>, (K, V, TryDashMapError)>
    where
        K: Eq + Hash;

    /// Fallibly insert a key-value pair only if the key is not already present.
    ///
    /// Returns `Ok(())` if the key was newly inserted. Returns
    /// `Err((key, value, error))` if the insertion failed, giving ownership of
    /// both `key` and `value` back to the caller. The error is
    /// [`TryDashMapError::Reserve`] on allocation failure or
    /// [`TryDashMapError::Other`] if the key already exists.
    fn try_insert_unique(&self, key: K, value: V) -> Result<(), (K, V, TryDashMapError)>
    where
        K: Eq + Hash;

    /// Non-blocking variant of [`Self::try_insert`].
    ///
    /// Returns `Ok(Some(result))` on success, `Ok(None)` if the shard is locked.
    fn try_insert_nonblock(
        &self,
        key: K,
        value: V,
    ) -> Result<Option<Option<V>>, TryDashMapNonblockError>
    where
        K: Eq + Hash;

    /// Non-blocking variant of [`Self::try_insert_give_back`].
    fn try_insert_give_back_nonblock(
        &self,
        key: K,
        value: V,
    ) -> Result<Option<Option<V>>, (K, V, TryDashMapNonblockError)>
    where
        K: Eq + Hash;

    /// Non-blocking variant of [`Self::try_insert_unique`].
    fn try_insert_unique_nonblock(
        &self,
        key: K,
        value: V,
    ) -> Result<Option<()>, (K, V, TryDashMapNonblockError)>
    where
        K: Eq + Hash;

    /// Fallibly obtain an [`Entry`] for a key.
    ///
    /// Acquires an exclusive lock on the target shard, reserves one slot via
    /// `try_reserve`, then calls `find_or_find_insert_slot` — all within a
    /// single lock acquisition. This guarantees that subsequent insertion into
    /// the returned entry cannot panic on out-of-memory.
    ///
    /// Blocks waiting for the shard lock. Returns
    /// [`TryDashMapError::Reserve`] if the shard cannot grow.
    fn try_entry<'a>(&'a self, key: K) -> Result<Entry<'a, K, V>, TryDashMapError>
    where
        K: Eq + Hash;

    /// Like [`Self::try_entry`] but returns `None` instead of blocking if the
    /// shard is currently locked by another writer.
    ///
    /// Returns `Ok(Some(entry))` on success, `Ok(None)` if the shard was
    /// contended, or `Err(TryDashMapNonblockError::Locked)` if the shard could not
    /// be acquired.
    fn try_entry_nonblock<'a>(
        &'a self,
        key: K,
    ) -> Result<Option<Entry<'a, K, V>>, TryDashMapNonblockError>
    where
        K: Eq + Hash;

    /// Like [`Self::try_entry`] but returns ownership of `key` back on error.
    ///
    /// Use this when you need the key available in the error path (e.g. for
    /// give-back insertion variants) so as to not require `K: Clone`.
    fn try_entry_give_back<'a>(&'a self, key: K) -> Result<Entry<'a, K, V>, (K, TryDashMapError)>
    where
        K: Eq + Hash;

    /// Non-blocking variant of [`Self::try_entry_give_back`].
    fn try_entry_give_back_nonblock<'a>(
        &'a self,
        key: K,
    ) -> Result<Option<Entry<'a, K, V>>, (K, TryDashMapNonblockError)>
    where
        K: Eq + Hash;

    /// Like [`Self::try_entry`] but takes a reference to the key and only clones
    /// it *after* capacity has been reserved on the target shard.
    ///
    /// This avoids wasting a clone if reservation fails. The key is cloned
    /// unconditionally once reservation succeeds (both [`Entry::Occupied`] and
    /// [`Entry::Vacant`] require an owned key internally).
    fn try_entry_ref<'a>(&'a self, key: &K) -> Result<Entry<'a, K, V>, TryDashMapError>
    where
        K: TryClone + Eq + Hash;

    // ── Aliases with `fallible_` prefix ────────────────────────────────────

    /// Alias for [`Self::try_entry_ref`].
    fn fallible_entry_ref<'a>(&'a self, key: &K) -> Result<Entry<'a, K, V>, TryDashMapError>
    where
        K: TryClone + Eq + Hash,
    {
        Self::try_entry_ref(self, key)
    }

    /// Alias for [`Self::try_new`].
    fn fallible_new() -> Result<DashMap<K, V, S>, TryDashMapError>
    where
        S: TryDefault,
    {
        Self::try_new()
    }

    /// Alias for [`Self::try_with_capacity`].
    fn fallible_with_capacity(capacity: usize) -> Result<DashMap<K, V, S>, TryDashMapError>
    where
        S: TryDefault,
    {
        Self::try_with_capacity(capacity)
    }

    /// Alias for [`Self::try_with_capacity_and_hasher`].
    fn fallible_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<DashMap<K, V, S>, TryDashMapError> {
        Self::try_with_capacity_and_hasher(capacity, hasher)
    }

    /// Alias for [`Self::try_insert`].
    fn fallible_insert(&self, key: K, value: V) -> Result<Option<V>, TryDashMapError>
    where
        K: Eq + Hash,
    {
        Self::try_insert(self, key, value)
    }

    /// Alias for [`Self::try_insert_give_back`].
    fn fallible_insert_give_back(
        &self,
        key: K,
        value: V,
    ) -> Result<Option<V>, (K, V, TryDashMapError)>
    where
        K: Eq + Hash,
    {
        Self::try_insert_give_back(self, key, value)
    }

    /// Alias for [`Self::try_insert_unique`].
    fn fallible_insert_unique(&self, key: K, value: V) -> Result<(), (K, V, TryDashMapError)>
    where
        K: Eq + Hash,
    {
        Self::try_insert_unique(self, key, value)
    }

    /// Alias for [`Self::try_insert_nonblock`].
    fn fallible_insert_nonblock(
        &self,
        key: K,
        value: V,
    ) -> Result<Option<Option<V>>, TryDashMapNonblockError>
    where
        K: Eq + Hash,
    {
        Self::try_insert_nonblock(self, key, value)
    }

    /// Alias for [`Self::try_insert_give_back_nonblock`].
    fn fallible_insert_give_back_nonblock(
        &self,
        key: K,
        value: V,
    ) -> Result<Option<Option<V>>, (K, V, TryDashMapNonblockError)>
    where
        K: Eq + Hash,
    {
        Self::try_insert_give_back_nonblock(self, key, value)
    }

    /// Alias for [`Self::try_insert_unique_nonblock`].
    fn fallible_insert_unique_nonblock(
        &self,
        key: K,
        value: V,
    ) -> Result<Option<()>, (K, V, TryDashMapNonblockError)>
    where
        K: Eq + Hash,
    {
        Self::try_insert_unique_nonblock(self, key, value)
    }

    /// Alias for [`Self::try_entry`].
    fn fallible_entry<'a>(&'a self, key: K) -> Result<Entry<'a, K, V>, TryDashMapError>
    where
        K: Eq + Hash,
    {
        Self::try_entry(self, key)
    }

    /// Alias for [`Self::try_entry_nonblock`].
    fn fallible_entry_nonblock<'a>(
        &'a self,
        key: K,
    ) -> Result<Option<Entry<'a, K, V>>, TryDashMapNonblockError>
    where
        K: Eq + Hash,
    {
        Self::try_entry_nonblock(self, key)
    }

    /// Alias for [`Self::try_entry_give_back`].
    fn fallible_entry_give_back<'a>(
        &'a self,
        key: K,
    ) -> Result<Entry<'a, K, V>, (K, TryDashMapError)>
    where
        K: Eq + Hash,
    {
        Self::try_entry_give_back(self, key)
    }

    /// Alias for [`Self::try_entry_give_back_nonblock`].
    fn fallible_entry_give_back_nonblock<'a>(
        &'a self,
        key: K,
    ) -> Result<Option<Entry<'a, K, V>>, (K, TryDashMapNonblockError)>
    where
        K: Eq + Hash,
    {
        Self::try_entry_give_back_nonblock(self, key)
    }

    // ── Capacity / shrink ───────────────────────────────────────────────────

    /// Fallibly shrink the capacity of this DashMap to match its length.
    ///
    /// **Deprecated:** This method name conflicts with the unstable inherent
    /// [`DashMap::try_shrink_to_fit`](dashmap::DashMap::try_shrink_to_fit).
    /// Use [`Self::fallible_shrink_to_fit`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with unstable DashMap::try_shrink_to_fit; use fallible_shrink_to_fit"
    )]
    fn try_shrink_to_fit(&self) -> Result<(), TryDashMapError>;

    /// Fallibly shrink the capacity of this DashMap to match its length.
    ///
    /// Rebuilds the internal table so that it holds approximately `len` elements.
    /// Returns [`TryDashMapError::Reserve`] if the allocation for the rebuilt
    /// table fails. Equivalent to [`DashMap::shrink_to_fit`](dashmap::DashMap::shrink_to_fit)
    /// but fallible.
    ///
    /// This method replaces the deprecated [`Self::try_shrink_to_fit`] which
    /// shares its name with the unstable inherent [`DashMap::try_shrink_to_fit`](dashmap::DashMap::try_shrink_to_fit).
    #[allow(deprecated)]
    fn fallible_shrink_to_fit(&self) -> Result<(), TryDashMapError> {
        Self::try_shrink_to_fit(self)
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    /// Fallibly collect an iterator of `(K, V)` pairs into a `DashMap`.
    ///
    /// Constructs the hasher via [`TryDefault`] and uses the iterator's size
    /// hint to pre-allocate when possible.
    fn try_collect<I: IntoIterator<Item = (K, V)>>(
        iter: I,
    ) -> Result<DashMap<K, V, S>, TryDashMapError>
    where
        S: TryDefault;

    /// Fallibly create a `DashMap` from an iterator using the provided hasher.
    fn try_collect_with_hasher<I: IntoIterator<Item = (K, V)>>(
        iter: I,
        hasher: S,
    ) -> Result<DashMap<K, V, S>, TryDashMapError>;

    /// Alias for [`Self::try_collect`].
    fn fallible_collect<I: IntoIterator<Item = (K, V)>>(
        iter: I,
    ) -> Result<DashMap<K, V, S>, TryDashMapError>
    where
        S: TryDefault,
    {
        Self::try_collect(iter)
    }

    /// Alias for [`Self::try_collect_with_hasher`].
    fn fallible_collect_with_hasher<I: IntoIterator<Item = (K, V)>>(
        iter: I,
        hasher: S,
    ) -> Result<DashMap<K, V, S>, TryDashMapError> {
        Self::try_collect_with_hasher(iter, hasher)
    }
}

// ── Implementation ────────────────────────────────────────────────────────────

impl<K: Eq + Hash, V, S: BuildHasher + TryClone> TryDashMap<K, V, S> for DashMap<K, V, S> {
    // ── Construction ────────────────────────────────────────────────────────

    fn try_new() -> Result<DashMap<K, V, S>, TryDashMapError>
    where
        S: TryDefault,
    {
        let hasher = S::try_default()?;
        Ok(DashMap::with_hasher(hasher))
    }

    fn try_with_capacity(capacity: usize) -> Result<DashMap<K, V, S>, TryDashMapError>
    where
        S: TryDefault,
    {
        let mut map = Self::try_new()?;
        if capacity > 0 {
            map.try_reserve(capacity)
                .map_err(|_| TryDashMapError::Reserve(TryReserveError::Other))?;
        }
        Ok(map)
    }

    fn try_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<DashMap<K, V, S>, TryDashMapError> {
        let mut map = DashMap::with_hasher(hasher);
        if capacity > 0 {
            map.try_reserve(capacity)
                .map_err(|_| TryDashMapError::Reserve(TryReserveError::Other))?;
        }
        Ok(map)
    }

    // ── Insertion ───────────────────────────────────────────────────────────

    fn try_insert(&self, key: K, value: V) -> Result<Option<V>, TryDashMapError>
    where
        K: Eq + Hash,
    {
        let entry = TryDashMap::try_entry(self, key)?;
        match entry {
            Entry::Occupied(mut e) => {
                let old = e.insert(value);
                Ok(Some(old))
            }
            Entry::Vacant(e) => {
                e.insert(value);
                Ok(None)
            }
        }
    }

    fn try_insert_give_back(&self, key: K, value: V) -> Result<Option<V>, (K, V, TryDashMapError)>
    where
        K: Eq + Hash,
    {
        let entry = match TryDashMap::try_entry_give_back(self, key) {
            Ok(e) => e,
            Err((k, err)) => return Err((k, value, err)),
        };
        match entry {
            Entry::Occupied(mut e) => {
                let old = e.insert(value);
                Ok(Some(old))
            }
            Entry::Vacant(e) => {
                e.insert(value);
                Ok(None)
            }
        }
    }

    fn try_insert_unique(&self, key: K, value: V) -> Result<(), (K, V, TryDashMapError)>
    where
        K: Eq + Hash,
    {
        let entry = match TryDashMap::try_entry_give_back(self, key) {
            Ok(e) => e,
            Err((k, err)) => return Err((k, value, err)),
        };
        match entry {
            Entry::Occupied(e) => {
                let returned_key = e.into_key();
                Err((
                    returned_key,
                    value,
                    TryDashMapError::Other("key already exists"),
                ))
            }
            Entry::Vacant(e) => {
                e.insert(value);
                Ok(())
            }
        }
    }

    fn try_insert_nonblock(
        &self,
        key: K,
        value: V,
    ) -> Result<Option<Option<V>>, TryDashMapNonblockError>
    where
        K: Eq + Hash,
    {
        let entry = match TryDashMap::try_entry_nonblock(self, key)? {
            Some(e) => e,
            None => return Ok(None),
        };
        match entry {
            Entry::Occupied(mut e) => {
                let old = e.insert(value);
                Ok(Some(Some(old)))
            }
            Entry::Vacant(e) => {
                e.insert(value);
                Ok(Some(None))
            }
        }
    }

    fn try_insert_give_back_nonblock(
        &self,
        key: K,
        value: V,
    ) -> Result<Option<Option<V>>, (K, V, TryDashMapNonblockError)>
    where
        K: Eq + Hash,
    {
        let entry = match TryDashMap::try_entry_give_back_nonblock(self, key) {
            Ok(Some(e)) => e,
            Ok(None) => return Ok(None),
            Err((k, err)) => return Err((k, value, err)),
        };
        match entry {
            Entry::Occupied(mut e) => {
                let old = e.insert(value);
                Ok(Some(Some(old)))
            }
            Entry::Vacant(e) => {
                e.insert(value);
                Ok(Some(None))
            }
        }
    }

    fn try_insert_unique_nonblock(
        &self,
        key: K,
        value: V,
    ) -> Result<Option<()>, (K, V, TryDashMapNonblockError)>
    where
        K: Eq + Hash,
    {
        let entry = match TryDashMap::try_entry_give_back_nonblock(self, key) {
            Ok(Some(e)) => e,
            Ok(None) => return Ok(None),
            Err((k, err)) => return Err((k, value, err)),
        };
        match entry {
            Entry::Occupied(e) => {
                let returned_key = e.into_key();
                Err((
                    returned_key,
                    value,
                    TryDashMapNonblockError::Other("key already exists"),
                ))
            }
            Entry::Vacant(e) => {
                e.insert(value);
                Ok(Some(()))
            }
        }
    }

    fn try_entry<'a>(&'a self, key: K) -> Result<Entry<'a, K, V>, TryDashMapError>
    where
        K: Eq + Hash,
    {
        let hash = compute_hash(self, &key);
        let shard_idx = self.determine_shard(hash as usize);
        let hf = self.hasher().clone();

        let mut shard = self.shards()[shard_idx].write();

        // Reserve one slot first so that find_or_find_insert_slot cannot panic.
        shard
            .try_reserve(1, |(k, _v): &ShardEntry<K, V>| hf.hash_one(k))
            .map_err(|_| TryDashMapError::Reserve(TryReserveError::Other))?;

        // Build the entry while still holding the lock — no re-acquisition needed.
        match shard.find_or_find_insert_slot(
            hash,
            |(k, _v): &ShardEntry<K, V>| k == &key,
            |(k, _v): &ShardEntry<K, V>| hf.hash_one(k),
        ) {
            Ok(bucket) => Ok(Entry::Occupied(unsafe {
                OccupiedEntry::new(shard, key, bucket)
            })),
            Err(slot) => Ok(Entry::Vacant(unsafe {
                VacantEntry::new(shard, key, hash, slot)
            })),
        }
    }

    fn try_entry_give_back<'a>(&'a self, key: K) -> Result<Entry<'a, K, V>, (K, TryDashMapError)>
    where
        K: Eq + Hash,
    {
        let hash = compute_hash(self, &key);
        let shard_idx = self.determine_shard(hash as usize);
        let hf = self.hasher().clone();

        let mut shard = self.shards()[shard_idx].write();
        match shard.try_reserve(1, |(k, _v): &ShardEntry<K, V>| hf.hash_one(k)) {
            Ok(()) => {}
            Err(_) => return Err((key, TryDashMapError::Reserve(TryReserveError::Other))),
        }

        match shard.find_or_find_insert_slot(
            hash,
            |(k, _v): &ShardEntry<K, V>| k == &key,
            |(k, _v): &ShardEntry<K, V>| hf.hash_one(k),
        ) {
            Ok(bucket) => Ok(Entry::Occupied(unsafe {
                OccupiedEntry::new(shard, key, bucket)
            })),
            Err(slot) => Ok(Entry::Vacant(unsafe {
                VacantEntry::new(shard, key, hash, slot)
            })),
        }
    }

    fn try_entry_nonblock<'a>(
        &'a self,
        key: K,
    ) -> Result<Option<Entry<'a, K, V>>, TryDashMapNonblockError>
    where
        K: Eq + Hash,
    {
        let hash = compute_hash(self, &key);
        let shard_idx = self.determine_shard(hash as usize);
        let hf = self.hasher().clone();

        let Some(mut shard) = self.shards()[shard_idx].try_write() else {
            return Err(TryDashMapNonblockError::Locked);
        };

        // Reserve one slot so insertion cannot panic.
        shard
            .try_reserve(1, |(k, _v): &ShardEntry<K, V>| hf.hash_one(k))
            .map_err(|_| TryDashMapNonblockError::Reserve(TryReserveError::Other))?;

        match shard.find_or_find_insert_slot(
            hash,
            |(k, _v): &ShardEntry<K, V>| k == &key,
            |(k, _v): &ShardEntry<K, V>| hf.hash_one(k),
        ) {
            Ok(bucket) => Ok(Some(Entry::Occupied(unsafe {
                OccupiedEntry::new(shard, key, bucket)
            }))),
            Err(slot) => Ok(Some(Entry::Vacant(unsafe {
                VacantEntry::new(shard, key, hash, slot)
            }))),
        }
    }

    fn try_entry_give_back_nonblock<'a>(
        &'a self,
        key: K,
    ) -> Result<Option<Entry<'a, K, V>>, (K, TryDashMapNonblockError)>
    where
        K: Eq + Hash,
    {
        let hash = compute_hash(self, &key);
        let shard_idx = self.determine_shard(hash as usize);
        let hf = self.hasher().clone();

        let Some(mut shard) = self.shards()[shard_idx].try_write() else {
            return Err((key, TryDashMapNonblockError::Locked));
        };

        match shard.try_reserve(1, |(k, _v): &ShardEntry<K, V>| hf.hash_one(k)) {
            Ok(()) => {}
            Err(_) => {
                return Err((
                    key,
                    TryDashMapNonblockError::Reserve(TryReserveError::Other),
                ));
            }
        }

        match shard.find_or_find_insert_slot(
            hash,
            |(k, _v): &ShardEntry<K, V>| k == &key,
            |(k, _v): &ShardEntry<K, V>| hf.hash_one(k),
        ) {
            Ok(bucket) => Ok(Some(Entry::Occupied(unsafe {
                OccupiedEntry::new(shard, key, bucket)
            }))),
            Err(slot) => Ok(Some(Entry::Vacant(unsafe {
                VacantEntry::new(shard, key, hash, slot)
            }))),
        }
    }

    fn try_entry_ref<'a>(&'a self, key: &K) -> Result<Entry<'a, K, V>, TryDashMapError>
    where
        K: TryClone + Eq + Hash,
    {
        let hash = compute_hash(self, key);
        let shard_idx = self.determine_shard(hash as usize);
        let hf = self.hasher().clone();

        let mut shard = self.shards()[shard_idx].write();

        // Reserve one slot first so that find_or_find_insert_slot cannot panic.
        shard
            .try_reserve(1, |(k, _v): &ShardEntry<K, V>| hf.hash_one(k))
            .map_err(|_| TryDashMapError::Reserve(TryReserveError::Other))?;

        // Clone the key only after reservation succeeded.
        let key = key.try_clone()?;

        // Build the entry while still holding the lock to prevent the reservation from being stolen.
        match shard.find_or_find_insert_slot(
            hash,
            |(k, _v): &ShardEntry<K, V>| k == &key,
            |(k, _v): &ShardEntry<K, V>| hf.hash_one(k),
        ) {
            Ok(bucket) => Ok(Entry::Occupied(unsafe {
                OccupiedEntry::new(shard, key, bucket)
            })),
            Err(slot) => Ok(Entry::Vacant(unsafe {
                VacantEntry::new(shard, key, hash, slot)
            })),
        }
    }

    // ── Capacity / shrink ───────────────────────────────────────────────────

    fn try_shrink_to_fit(&self) -> Result<(), TryDashMapError> {
        use lang_core::mem::ManuallyDrop;

        let hf = self.hasher().clone();

        for s in self.shards().iter() {
            let mut shard = s.write();
            let count = shard.len();

            // Allocate a new empty dehydrated table and reserve exact capacity — fallible.
            let mut new_table: RawTable<ManuallyDrop<ShardEntry<K, V>>> = RawTable::default();
            new_table
                .try_reserve(count, |e: &ManuallyDrop<ShardEntry<K, V>>| {
                    hf.hash_one(&e.0)
                })
                .map_err(|_| TryDashMapError::Reserve(TryReserveError::Other))?;

            // Iterate the old table and write each entry into the new one via raw
            // pointer reads + insert. The old table's control bytes still mark these
            // slots as occupied, but the values have been logically moved.
            let mut copy_ok = true;
            unsafe {
                for bucket in shard.iter() {
                    let ptr = bucket.as_ptr();
                    // SAFETY: `ptr` is valid for reads — it points to a live entry
                    // inside the old table that we hold exclusive access to.
                    let hash = hf.hash_one(&(*ptr).0);
                    let entry: ManuallyDrop<ShardEntry<K, V>> = ptr::read(ptr as *mut _); // takes ownership
                    match new_table.try_insert_no_grow(hash, entry) {
                        Ok(_) => {}
                        Err(_) => {
                            copy_ok = false;
                            break;
                        }
                    }
                }
            }

            if !copy_ok {
                // Do not perform hydration - because we are using ManuallyDrop, no element dropping occurs, and the old table stays intact.
                return Err(TryDashMapError::Other("shrink copy failed"));
            }

            // Hydrate the new table into the shard slot.
            let old_table = mem::replace(&mut *shard, unsafe {
                mem::transmute::<RawTable<ManuallyDrop<ShardEntry<K, V>>>, RawTable<ShardEntry<K, V>>>(
                    new_table,
                )
            });
            // Values were moved out of the old table above, so we need to dehydrate the table.
            let _dehydrated_table = unsafe {
                mem::transmute::<RawTable<ShardEntry<K, V>>, RawTable<ManuallyDrop<ShardEntry<K, V>>>>(
                    old_table,
                )
            };
        }

        Ok(())
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    fn try_collect<I: IntoIterator<Item = (K, V)>>(
        iter: I,
    ) -> Result<DashMap<K, V, S>, TryDashMapError>
    where
        S: TryDefault,
    {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower);
        let mut map = Self::try_new()?;
        if capacity > 0 {
            map.try_reserve(capacity)
                .map_err(|_| TryDashMapError::Reserve(TryReserveError::Other))?;
        }
        for (key, value) in iter {
            <DashMap<K, V, S> as TryDashMap<K, V, S>>::try_insert(&map, key, value)?;
        }
        Ok(map)
    }

    fn try_collect_with_hasher<I: IntoIterator<Item = (K, V)>>(
        iter: I,
        hasher: S,
    ) -> Result<DashMap<K, V, S>, TryDashMapError> {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower);
        let mut map = DashMap::with_hasher(hasher);
        if capacity > 0 {
            map.try_reserve(capacity)
                .map_err(|_| TryDashMapError::Reserve(TryReserveError::Other))?;
        }
        for (key, value) in iter {
            Self::try_insert(&map, key, value)?;
        }
        Ok(map)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

type ShardEntry<K, V> = (K, dashmap::SharedValue<V>);

/// Compute the u64 hash of a key using the map's hasher factory.
fn compute_hash<K: Eq + Hash, V, S: BuildHasher + TryClone>(
    map: &DashMap<K, V, S>,
    key: &K,
) -> u64 {
    map.hasher().hash_one(key)
}

// ── TryClone for DashMap<K, V, S> ──────────────────────────────────────────────

/// Implements [`TryClone`] for `DashMap<K, V, S>`
/// when keys and values are cloneable and the hasher factory can be cloned.
///
/// Iterates over the map, clones each key-value pair fallibly, and inserts
/// into a new map. If any clone fails, the partially-built map is dropped
/// and the error is returned.
impl<K, V, S> crate::try_clone::TryClone for DashMap<K, V, S>
where
    K: Eq + Hash + crate::try_clone::TryClone,
    V: crate::try_clone::TryClone,
    S: BuildHasher + TryClone,
{
    fn try_clone(&self) -> Result<Self, crate::try_clone::TryCloneError> {
        let hasher = self.hasher().clone();
        let out = DashMap::with_hasher(hasher);
        // We cannot reserve everything immediately because shards may be uneven.
        for ref_cell in self.iter() {
            let entry = TryDashMap::try_entry_ref(&out, ref_cell.key()).map_err(|e| match e {
                TryDashMapError::Alloc(a) => TryCloneError::Alloc(a),
                TryDashMapError::Reserve(r) => TryCloneError::Reserve(r),
                TryDashMapError::Clone(c) => c,
                TryDashMapError::Overflow => TryCloneError::Overflow,
                TryDashMapError::Other(m) => TryCloneError::Other(m),
            })?;
            // Cloning happens only after reservation succeeded.
            let value_cloned = ref_cell.value().try_clone()?;
            entry.insert(value_cloned);
        }
        Ok(out)
    }
}

// ── TryDefault for DashMap<K, V, S> ────────────────────────────────────────────

impl<K: Eq + Hash, V, S> crate::try_default::TryDefault for DashMap<K, V, S>
where
    S: BuildHasher + TryDefault + TryClone,
{
    fn try_default() -> Result<Self, crate::try_default::TryDefaultError> {
        // An empty DashMap requires no allocation.
        Ok(DashMap::with_hasher(S::try_default()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::try_default::TryDefault as _;
    use lang_alloc::string::String;
    use lang_alloc::string::ToString;
    use lang_alloc::vec;
    use lang_alloc::vec::Vec;
    use lang_std::iter;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn try_with_capacity_zero() {
        let map: DashMap<i32, i32> = DashMap::<i32, i32>::try_with_capacity(0).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn try_with_capacity_nonzero() {
        let map: DashMap<i32, String> = DashMap::<i32, String>::try_with_capacity(10).unwrap();
        assert!(map.is_empty());
        assert!(map.capacity() >= 10);
    }

    #[test]
    fn try_with_capacity_and_hasher() {
        use lang_std::collections::hash_map::DefaultHasher;
        use lang_std::hash::BuildHasherDefault;

        let hasher = BuildHasherDefault::<DefaultHasher>::default();
        let map: DashMap<&str, i32, _> = DashMap::try_with_capacity_and_hasher(5, hasher).unwrap();
        assert!(map.is_empty());
        assert!(map.capacity() >= 5);
    }

    // ── Insertion ────────────────────────────────────────────────────────────

    #[test]
    fn try_insert_single() {
        let map: DashMap<i32, &str> = DashMap::new();
        let old = map.try_insert(1, "one").unwrap();
        assert_eq!(old, None);
        assert_eq!(*map.get(&1).unwrap(), "one");
    }

    #[test]
    fn try_insert_replaces_existing() {
        let map: DashMap<i32, &str> = DashMap::new();
        map.try_insert(1, "one").unwrap();
        let old = map.try_insert(1, "ONE").unwrap();
        assert_eq!(old, Some("one"));
        assert_eq!(*map.get(&1).unwrap(), "ONE");
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn try_insert_multiple_keys() {
        let map: DashMap<&str, i32> = DashMap::new();
        map.try_insert("a", 1).unwrap();
        map.try_insert("b", 2).unwrap();
        map.try_insert("c", 3).unwrap();
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn fallible_insert_matches_try_insert() {
        let map: DashMap<i32, &str> = DashMap::new();
        map.fallible_insert(42, "hi").unwrap();
        assert_eq!(*map.get(&42).unwrap(), "hi");
    }

    // ── Give-back variants ───────────────────────────────────────────────────

    #[test]
    fn try_insert_give_back_success() {
        let map: DashMap<String, Vec<u8>> = DashMap::new();
        map.try_insert_give_back("k".to_string(), vec![1]).unwrap();
        assert_eq!(*map.get("k").unwrap(), vec![1]);
    }

    #[test]
    fn try_insert_give_back_error_type_shape() {
        let map: DashMap<i32, i32> = DashMap::new();
        let result: Result<Option<i32>, (i32, i32, TryDashMapError)> =
            map.try_insert_give_back(1, 2);
        assert!(result.is_ok());
    }

    // ── Unique insertion ─────────────────────────────────────────────────────

    #[test]
    fn try_insert_unique_new_key() {
        let map: DashMap<i32, &str> = DashMap::new();
        map.try_insert_unique(1, "one").unwrap();
        assert_eq!(*map.get(&1).unwrap(), "one");
    }

    #[test]
    fn try_insert_unique_duplicate_rejected() {
        let map: DashMap<i32, &str> = DashMap::new();
        map.try_insert_unique(1, "one").unwrap();
        let result = map.try_insert_unique(1, "TWO");
        let (returned_key, returned_val, err) = result.unwrap_err();
        assert_eq!(returned_key, 1);
        assert_eq!(returned_val, "TWO");
        matches!(err, TryDashMapError::Other(_));
        assert_eq!(*map.get(&1).unwrap(), "one");
        assert_eq!(map.len(), 1);
    }

    // ── Entry API ────────────────────────────────────────────────────────────

    use crate::dashmap::mapref::Entry;

    /// Disambiguates our `TryDashMap::try_entry` from `DashMap::try_entry`.
    fn __try_entry<'a, K, V, S>(
        m: &'a DashMap<K, V, S>,
        key: K,
    ) -> Result<Entry<'a, K, V>, TryDashMapError>
    where
        K: Eq + Hash,
        S: BuildHasher + TryClone,
    {
        TryDashMap::try_entry(m, key)
    }

    /// Disambiguates our `TryDashMap::try_entry_nonblock` from `DashMap::try_entry`.
    fn __try_entry_nonblock<'a, K, V, S>(
        m: &'a DashMap<K, V, S>,
        key: K,
    ) -> Result<Option<Entry<'a, K, V>>, TryDashMapNonblockError>
    where
        K: Eq + Hash,
        S: BuildHasher + TryClone,
    {
        TryDashMap::try_entry_nonblock(m, key)
    }

    #[test]
    fn try_entry_vacant_or_insert() {
        let map: DashMap<String, i32> = DashMap::new();
        __try_entry(&map, "hello".to_string())
            .unwrap()
            .or_insert(42);
        assert_eq!(*map.get("hello").unwrap(), 42);
    }

    #[test]
    fn try_entry_occupied_or_insert_noop() {
        let map: DashMap<String, i32> = DashMap::new();
        map.try_insert("key".to_string(), 1).unwrap();
        __try_entry(&map, "key".to_string()).unwrap().or_insert(99);
        assert_eq!(*map.get("key").unwrap(), 1);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn try_entry_or_fallible_default() {
        let map: DashMap<String, Vec<i32>> = DashMap::new();
        __try_entry(&map, "a".to_string())
            .unwrap()
            .or_try_default()
            .unwrap();
        assert!(map.get("a").unwrap().is_empty());
    }

    #[test]
    fn try_entry_or_insert_with() {
        let map: DashMap<String, String> = DashMap::new();
        __try_entry(&map, "x".to_string())
            .unwrap()
            .or_insert_with(|| "computed".to_string());
        assert_eq!(*map.get("x").unwrap(), "computed");
    }

    #[test]
    fn try_entry_or_fallible_insert_with_ok() {
        let map: DashMap<i32, String> = DashMap::new();
        let val = __try_entry(&map, 1)
            .unwrap()
            .or_try_insert_with(|| Ok::<_, ()>("ok".to_string()))
            .unwrap();
        assert_eq!(&**val, "ok");
    }

    #[test]
    fn try_entry_or_fallible_insert_with_err_propagates() {
        let map: DashMap<i32, String> = DashMap::new();
        let result = __try_entry(&map, 1)
            .unwrap()
            .or_try_insert_with(|| Err::<String, i32>(42));
        match result {
            Err(e) => assert_eq!(e, 42),
            Ok(_) => panic!("expected error"),
        }
        assert!(map.is_empty());
    }

    #[test]
    fn try_entry_insert_overwrites() {
        let map: DashMap<String, i32> = DashMap::new();
        map.try_insert("k".to_string(), 1).unwrap();
        let ref_mut = __try_entry(&map, "k".to_string()).unwrap().insert(99);
        assert_eq!(*ref_mut, 99);
        drop(ref_mut);
        assert_eq!(*map.get("k").unwrap(), 99);
    }

    #[test]
    fn try_entry_insert_entry_vacant() {
        let map: DashMap<String, i32> = DashMap::new();
        let occupied = __try_entry(&map, "a".to_string()).unwrap().insert_entry(10);
        assert_eq!(*occupied.get(), 10);
        drop(occupied);
        assert_eq!(*map.get("a").unwrap(), 10);
    }

    #[test]
    fn try_entry_insert_entry_occupied() {
        let map: DashMap<String, i32> = DashMap::new();
        map.try_insert("a".to_string(), 1).unwrap();
        let occupied = __try_entry(&map, "a".to_string()).unwrap().insert_entry(20);
        assert_eq!(*occupied.get(), 20);
        drop(occupied);
        assert_eq!(*map.get("a").unwrap(), 20);
    }

    #[test]
    fn try_entry_key_and_into_key() {
        let map: DashMap<String, i32> = DashMap::new();
        let entry = __try_entry(&map, "abc".to_string()).unwrap();
        assert_eq!(entry.key(), "abc");
        drop(entry);
    }

    #[test]
    fn try_entry_and_modify_occupied() {
        let map: DashMap<String, i32> = DashMap::new();
        map.try_insert("n".to_string(), 5).unwrap();
        __try_entry(&map, "n".to_string())
            .unwrap()
            .and_modify(|v| *v *= 2);
        assert_eq!(*map.get("n").unwrap(), 10);
    }

    #[test]
    fn try_entry_and_modify_vacant_noop() {
        let map: DashMap<String, i32> = DashMap::new();
        __try_entry(&map, "missing".to_string())
            .unwrap()
            .and_modify(|v| *v = 999);
        assert!(map.is_empty());
    }

    // ── OccupiedEntry methods ────────────────────────────────────────────────

    #[test]
    fn occupied_get_and_get_mut() {
        let map: DashMap<String, i32> = DashMap::new();
        map.try_insert("k".to_string(), 7).unwrap();
        match __try_entry(&map, "k".to_string()).unwrap() {
            Entry::Occupied(mut e) => {
                assert_eq!(*e.get(), 7);
                *e.get_mut() = 14;
            }
            _ => panic!("expected occupied"),
        }
        assert_eq!(*map.get("k").unwrap(), 14);
    }

    #[test]
    fn occupied_insert_replaces() {
        let map: DashMap<String, i32> = DashMap::new();
        map.try_insert("k".to_string(), 1).unwrap();
        match __try_entry(&map, "k".to_string()).unwrap() {
            Entry::Occupied(mut e) => {
                let old = e.insert(2);
                assert_eq!(old, 1);
            }
            _ => panic!("expected occupied"),
        }
        assert_eq!(*map.get("k").unwrap(), 2);
    }

    #[test]
    fn occupied_into_ref() {
        let map: DashMap<String, i32> = DashMap::new();
        map.try_insert("k".to_string(), 42).unwrap();
        match __try_entry(&map, "k".to_string()).unwrap() {
            Entry::Occupied(e) => {
                let mut ref_mut = e.into_ref();
                assert_eq!(*ref_mut, 42);
                *ref_mut = 100;
            }
            _ => panic!("expected occupied"),
        }
        assert_eq!(*map.get("k").unwrap(), 100);
    }

    #[test]
    fn occupied_remove() {
        let map: DashMap<String, i32> = DashMap::new();
        map.try_insert("k".to_string(), 42).unwrap();
        match __try_entry(&map, "k".to_string()).unwrap() {
            Entry::Occupied(e) => {
                let val = e.remove();
                assert_eq!(val, 42);
            }
            _ => panic!("expected occupied"),
        }
        assert!(map.is_empty());
    }

    #[test]
    fn occupied_remove_entry() {
        let map: DashMap<String, i32> = DashMap::new();
        map.try_insert("k".to_string(), 42).unwrap();
        match __try_entry(&map, "k".to_string()).unwrap() {
            Entry::Occupied(e) => {
                let (k, v) = e.remove_entry();
                assert_eq!(k, "k");
                assert_eq!(v, 42);
            }
            _ => panic!("expected occupied"),
        }
        assert!(map.is_empty());
    }

    #[test]
    fn occupied_replace_entry() {
        let map: DashMap<String, i32> = DashMap::new();
        map.try_insert("k".to_string(), 1).unwrap();
        match __try_entry(&map, "k".to_string()).unwrap() {
            Entry::Occupied(e) => {
                let (old_k, old_v) = e.replace_entry(99);
                assert_eq!(old_k, "k");
                assert_eq!(old_v, 1);
            }
            _ => panic!("expected occupied"),
        }
        assert_eq!(*map.get("k").unwrap(), 99);
    }

    #[test]
    fn occupied_into_key() {
        let map: DashMap<String, i32> = DashMap::new();
        map.try_insert("k".to_string(), 1).unwrap();
        match __try_entry(&map, "k".to_string()).unwrap() {
            Entry::Occupied(e) => {
                let key = e.into_key();
                assert_eq!(key, "k");
            }
            _ => panic!("expected occupied"),
        }
    }

    // ── VacantEntry methods ──────────────────────────────────────────────────

    #[test]
    fn vacant_insert() {
        let map: DashMap<String, i32> = DashMap::new();
        match __try_entry(&map, "new".to_string()).unwrap() {
            Entry::Vacant(e) => {
                let ref_mut = e.insert(42);
                assert_eq!(*ref_mut, 42);
            }
            _ => panic!("expected vacant"),
        }
        assert_eq!(*map.get("new").unwrap(), 42);
    }

    #[test]
    fn vacant_insert_entry() {
        let map: DashMap<String, i32> = DashMap::new();
        match __try_entry(&map, "new".to_string()).unwrap() {
            Entry::Vacant(e) => {
                let occ = e.insert_entry(42);
                assert_eq!(*occ.get(), 42);
            }
            _ => panic!("expected vacant"),
        }
        assert_eq!(*map.get("new").unwrap(), 42);
    }

    #[test]
    fn vacant_key_and_into_key() {
        let map: DashMap<String, i32> = DashMap::new();
        let entry = __try_entry(&map, "abc".to_string()).unwrap();
        match entry {
            Entry::Vacant(e) => {
                assert_eq!(e.key(), "abc");
                let consumed = e.into_key();
                assert_eq!(consumed, "abc");
            }
            _ => panic!("expected vacant"),
        }
    }

    // ── try_entry_ref ────────────────────────────────────────────────────────

    /// Disambiguates our `TryDashMap::try_entry_ref` from any inherent methods.
    fn __try_entry_ref<'a, K, V, S>(
        m: &'a DashMap<K, V, S>,
        key: &K,
    ) -> Result<Entry<'a, K, V>, TryDashMapError>
    where
        K: TryClone + Eq + Hash,
        S: BuildHasher + TryClone,
    {
        TryDashMap::try_entry_ref(m, key)
    }

    #[test]
    fn try_entry_ref_vacant_inserts() {
        let map: DashMap<String, i32> = DashMap::new();
        let key = "hello".to_string();
        __try_entry_ref(&map, &key).unwrap().or_insert(42);
        assert_eq!(*map.get("hello").unwrap(), 42);
    }

    #[test]
    fn try_entry_ref_occupied_no_replace() {
        let map: DashMap<String, i32> = DashMap::new();
        map.try_insert("key".to_string(), 1).unwrap();
        let key = "key".to_string();
        __try_entry_ref(&map, &key).unwrap().or_insert(99);
        assert_eq!(*map.get("key").unwrap(), 1);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn try_entry_ref_occupied_modify() {
        let map: DashMap<String, i32> = DashMap::new();
        map.try_insert("n".to_string(), 5).unwrap();
        let key = "n".to_string();
        __try_entry_ref(&map, &key).unwrap().and_modify(|v| *v *= 2);
        assert_eq!(*map.get("n").unwrap(), 10);
    }

    #[test]
    fn try_entry_ref_insert_overwrites() {
        let map: DashMap<String, i32> = DashMap::new();
        map.try_insert("k".to_string(), 1).unwrap();
        let key = "k".to_string();
        let ref_mut = __try_entry_ref(&map, &key).unwrap().insert(99);
        assert_eq!(*ref_mut, 99);
        drop(ref_mut);
        assert_eq!(*map.get("k").unwrap(), 99);
    }

    #[test]
    fn try_entry_ref_key_access() {
        let map: DashMap<String, i32> = DashMap::new();
        let key = "abc".to_string();
        let entry = __try_entry_ref(&map, &key).unwrap();
        assert_eq!(entry.key(), "abc");
        drop(entry);
    }

    #[test]
    fn try_entry_ref_clone_after_reservation() {
        // Verifies that try_entry_ref works with a key type that implements TryClone.
        // The key is cloned after reservation succeeds, not before.
        let map: DashMap<Vec<u8>, String> = DashMap::new();
        let key = vec![1, 2, 3];
        __try_entry_ref(&map, &key)
            .unwrap()
            .or_insert_with(|| "data".to_string());
        assert_eq!(*map.get(&key).unwrap(), "data");
    }

    // ── Non-blocking entry ───────────────────────────────────────────────────

    #[test]
    fn try_entry_nonblock_returns_some() {
        let map: DashMap<String, i32> = DashMap::new();
        let entry = __try_entry_nonblock(&map, "a".to_string()).unwrap();
        assert!(entry.is_some());
        entry.unwrap().or_insert(7);
        assert_eq!(*map.get("a").unwrap(), 7);
    }

    #[test]
    fn try_entry_nonblock_existing_key() {
        let map: DashMap<String, i32> = DashMap::new();
        map.try_insert("b".to_string(), 1).unwrap();
        let entry = __try_entry_nonblock(&map, "b".to_string()).unwrap();
        assert!(entry.is_some());
        entry.unwrap().and_modify(|v| *v += 1);
        assert_eq!(*map.get("b").unwrap(), 2);
    }

    #[test]
    fn fallible_entry_matches_try_entry() {
        let map: DashMap<String, i32> = DashMap::new();
        map.fallible_entry("x".to_string()).unwrap().or_insert(10);
        assert_eq!(*map.get("x").unwrap(), 10);
    }

    #[test]
    fn fallible_entry_nonblock_alias_works() {
        let map: DashMap<String, i32> = DashMap::new();
        let entry = map.fallible_entry_nonblock("c".to_string()).unwrap();
        assert!(entry.is_some());
        entry.unwrap().or_insert(42);
        assert_eq!(*map.get("c").unwrap(), 42);
    }

    // ── Extension ────────────────────────────────────────────────────────────

    #[test]
    fn try_extend_from_iterator() {
        use crate::try_extend::TryExtend;
        let mut map: DashMap<i32, &str> = DashMap::new();
        <_ as TryExtend<(i32, &str)>>::try_extend(&mut map, [(1, "one"), (2, "two")]).unwrap();
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn try_extend_empty() {
        use crate::try_extend::TryExtend;
        let mut map: DashMap<i32, i32> = DashMap::new();
        <_ as TryExtend<(i32, i32)>>::try_extend(&mut map, iter::empty::<(i32, i32)>()).unwrap();
        assert!(map.is_empty());
    }

    // ── Shrink ────────────────────────────────────────────────────────────────

    #[test]
    fn try_shrink_to_fit_preserves_data() {
        let map: DashMap<i32, String> = DashMap::new();
        map.try_insert(1, "hello".to_string()).unwrap();
        map.try_insert(2, "world".to_string()).unwrap();
        map.try_shrink_to_fit().unwrap();
        assert_eq!(*map.get(&1).unwrap(), "hello");
        assert_eq!(*map.get(&2).unwrap(), "world");
        assert_eq!(map.len(), 2);
    }

    // ── Bulk construction ────────────────────────────────────────────────────

    #[test]
    fn try_collect_range() {
        let map: DashMap<i32, i32> =
            <DashMap<i32, i32> as TryDashMap<_, _, RandomState>>::try_collect(
                (0..5).map(|i| (i, i * 10)),
            )
            .unwrap();
        assert_eq!(map.len(), 5);
        assert_eq!(*map.get(&3).unwrap(), 30);
    }

    #[test]
    fn try_collect_empty() {
        let map: DashMap<i32, i32> =
            <DashMap<i32, i32> as TryDashMap<_, _, RandomState>>::try_collect(iter::empty::<(
                i32,
                i32,
            )>())
            .unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn try_collect_with_hasher() {
        use lang_std::collections::hash_map::DefaultHasher;
        use lang_std::hash::BuildHasherDefault;

        let hasher = BuildHasherDefault::<DefaultHasher>::default();
        let map: DashMap<i32, i32, _> =
            DashMap::try_collect_with_hasher([(1, 10), (2, 20)], hasher).unwrap();
        assert_eq!(*map.get(&1).unwrap(), 10);
        assert_eq!(*map.get(&2).unwrap(), 20);
    }

    // ── TryClone ─────────────────────────────────────────────────────────────

    #[test]
    fn try_clone_empty_map() {
        let map: DashMap<i32, i32> = DashMap::new();
        let c = map.try_clone().unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn try_clone_populated_map() {
        let map: DashMap<i32, String> = DashMap::new();
        map.insert(1, "hello".to_string());
        map.insert(2, "world".to_string());
        let c = map.try_clone().unwrap();
        assert_eq!(*c.get(&1).unwrap(), "hello");
        assert_eq!(*c.get(&2).unwrap(), "world");
    }

    // ── TryDefault ───────────────────────────────────────────────────────────

    #[test]
    fn try_default_empty_map() {
        let map: DashMap<i32, i32> = DashMap::try_default().unwrap();
        assert!(map.is_empty());
    }

    // ── Combined workflows ───────────────────────────────────────────────────

    #[test]
    fn build_insert_clone_default() {
        let map: DashMap<String, i32> = DashMap::try_default().unwrap();
        map.fallible_insert("alpha".to_string(), 1).unwrap();
        map.fallible_insert("beta".to_string(), 2).unwrap();
        let c = map.try_clone().unwrap();
        assert_eq!(*c.get("alpha").unwrap(), 1);
        assert_eq!(*c.get("beta").unwrap(), 2);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn collect_then_extend() {
        use crate::try_extend::TryExtend;
        let mut map: DashMap<i32, &str> =
            <DashMap<i32, &str> as TryDashMap<_, _, RandomState>>::try_collect([
                (1, "one"),
                (2, "two"),
            ])
            .unwrap();
        <_ as TryExtend<(i32, &str)>>::try_extend(&mut map, [(3, "three")]).unwrap();
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn fallible_aliases_match_try_methods() {
        let m1: DashMap<i32, i32> =
            <DashMap<i32, i32> as TryDashMap<_, _, RandomState>>::fallible_with_capacity(5)
                .unwrap();
        let m2: DashMap<i32, i32> =
            <DashMap<i32, i32> as TryDashMap<_, _, RandomState>>::try_with_capacity(5).unwrap();
        assert!(m1.is_empty());
        assert!(m2.is_empty());
    }

    // ── OOM tests ─────────────────────────────────────────────────────
    use rustyfill_test_allocator::{FailPolicy, with_policy};

    #[test]
    fn dashmap_try_insert_fails_on_oom() {
        let map: DashMap<u32, u32> = DashMap::new();
        let r = with_policy(FailPolicy::fail_next_alloc(), || map.try_insert(1, 10));
        assert!(r.is_err());
    }

    #[test]
    fn dashmap_nth_alloc_fail_targets_correct_call() {
        // Each `try_insert` performs exactly one heap allocation (the shard's
        // initial bucket-table growth from the empty singleton), so failing
        // the 2nd allocation makes the first insert succeed and the second
        // fail. This requires that no *other* allocation occurs in between.
        //
        // With too few shards, hashbrown's growth strategy can trigger an
        // internal rehash (capacity 3 → 6) during the *second* insert's slot
        // search, consuming the 2nd allocation before `try_reserve` ever sees
        // it — making the second insert unexpectedly succeed. Using 32 shards
        // keeps each shard sparse enough that this never happens within two
        // inserts, independent of the host's CPU count.
        let map: DashMap<u32, u32> = DashMap::with_shard_amount(32);
        let (r1_ok, r2_err) = with_policy(FailPolicy::fail_nth_alloc(2), || {
            let r1 = map.try_insert(1, 10);
            let r2 = map.try_insert(2, 20);
            (r1.is_ok(), r2.is_err())
        });
        assert!(r1_ok, "first insert should succeed");
        assert!(r2_err, "second insert should fail");
    }

    #[test]
    fn dashmap_oom_restores_allocation_afterwards() {
        let map: DashMap<u32, u32> = DashMap::new();
        let r = with_policy(FailPolicy::fail_next_alloc(), || map.try_insert(1, 10));
        assert!(r.is_err());
        assert!(map.try_insert(1, 10).is_ok());
    }
}
