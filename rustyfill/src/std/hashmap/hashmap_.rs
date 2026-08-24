//! Fallible hash map operations.
//!
//! Provides the [`TryHashMap`] trait with methods that mirror common `HashMap`
//! constructors and mutating operations but return [`Result`] to handle allocation
//! failures gracefully, using [`::lang_std::collections::TryReserveError`] as the primary
//! error type.
//!
//! # Design
//!
//! `TryHashMap` is implemented for `HashMap<K, V, S>`. Methods that may grow the
//! internal table (`insert`, `extend`, etc.) return a `Result` instead of panicking
//! on out-of-memory. Read-only accessors delegate directly to `HashMap`.
//!
//! The trait also implements [`TryClone`](crate::try_clone::TryClone) and
//! [`TryDefault`](crate::try_default::TryDefault) for `HashMap<K, V, S>` when
//! `K` and `V` satisfy the respective bounds.

use crate::alloc::TryReserveError;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::{TryDebug, TryDisplay};
use lang_core::cmp;
use lang_core::cmp::Eq;
use lang_core::fmt;
use lang_std::collections::HashMap;
use lang_std::collections::hash_map;
use lang_std::hash::{BuildHasher, Hash, RandomState};

// ── Error types ───────────────────────────────────────────────────────────────

/// Error returned by [`TryHashMap`] constructors.
///
/// Construction can only fail via a capacity reservation or hasher creation;
/// it never clones elements.
pub enum TryHashMapConstructionError {
    /// A capacity reservation on the hash map failed (overflow or OOM).
    Reserve(TryReserveError),
    /// Hasher construction failed via [`TryDefault`].
    Default(TryDefaultError),
}

impl fmt::Debug for TryHashMapConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(self, f)
    }
}

impl fmt::Display for TryHashMapConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }
}

impl From<TryReserveError> for TryHashMapConstructionError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<TryDefaultError> for TryHashMapConstructionError {
    fn from(err: TryDefaultError) -> Self {
        Self::Default(err)
    }
}

impl TryDebug for TryHashMapConstructionError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::debug_field(f, "TryHashMapConstructionError::Reserve", e),
            Self::Default(e) => u::debug_field(f, "TryHashMapConstructionError::Default", e),
        }
    }
}

impl TryDisplay for TryHashMapConstructionError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::display_delegated(f, "hash map", e),
            Self::Default(e) => u::display_delegated(f, "hash map", e),
        }
    }
}

/// Error for fallible hash map operations whose failure modes are limited to a
/// capacity reservation ([`TryReserveError`]) or an element clone failure
/// ([`TryCloneError`]).
///
/// Covers `try_extend_from_slice`, `try_shrink_to(_fit)`, and the
/// [`TryExtend`](crate::try_extend::TryExtend) /
/// [`TryExtendFromSlice`](crate::try_extend::TryExtendFromSlice) impls — any
/// operation that can only fail by reserving capacity or cloning elements.
pub enum TryHashMapWithCloneError {
    /// A capacity reservation on the hash map failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An element clone failed during a method that requires [`TryClone`].
    Clone(TryCloneError),
}

impl fmt::Debug for TryHashMapWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(self, f)
    }
}

impl fmt::Display for TryHashMapWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }
}

impl From<TryReserveError> for TryHashMapWithCloneError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<TryCloneError> for TryHashMapWithCloneError {
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

impl TryDebug for TryHashMapWithCloneError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::debug_field(f, "TryHashMapWithCloneError::Reserve", e),
            Self::Clone(e) => u::debug_field(f, "TryHashMapWithCloneError::Clone", e),
        }
    }
}

impl TryDisplay for TryHashMapWithCloneError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::display_delegated(f, "hash map", e),
            Self::Clone(e) => u::display_delegated(f, "hash map", e),
        }
    }
}

/// Error for fallible hash map insertion when the key already exists or a
/// capacity reservation fails.
///
/// Used by [`TryHashMap::try_insert_unique`]. The given-back key and value
/// travel alongside this error as a tuple: `Result<(), (K, V, TryHashMapInsertUniqueError)>`.
pub enum TryHashMapInsertUniqueError {
    /// A capacity reservation failed.
    Reserve(TryReserveError),
    /// The key was already present in the map.
    KeyAlreadyExists,
}

impl fmt::Debug for TryHashMapInsertUniqueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(self, f)
    }
}

impl fmt::Display for TryHashMapInsertUniqueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }
}

impl TryDebug for TryHashMapInsertUniqueError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::debug_field(f, "TryHashMapInsertUniqueError::Reserve", e),
            Self::KeyAlreadyExists => {
                u::debug_unit(f, "TryHashMapInsertUniqueError::KeyAlreadyExists")
            }
        }
    }
}

impl TryDisplay for TryHashMapInsertUniqueError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::display_delegated(f, "hash map", e),
            Self::KeyAlreadyExists => u::display_fixed(f, "hash map", "key already exists"),
        }
    }
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// A trait for fallible hash map operations.
///
/// Implemented for `HashMap<K, V, S>`. Mirrors the most commonly-used `HashMap`
/// methods that can fail due to allocation pressure, returning [`Result`] values
/// that propagate [`TryReserveError`] or an operation-specific error on failure.
///
/// # Note on `try_insert`
///
/// The inherent [`HashMap::try_insert`](lang_std::collections::HashMap::try_insert) on
/// stable Rust returns `Err(old_value)` when a key already exists, but may *panic*
/// on allocation failure. Our [`Self::try_insert`] reserves capacity first so it
/// never panics on OOM — it returns [`TryReserveError`] instead, but it does not
/// return the old value on key collision.
pub trait TryHashMap<K, V, S = RandomState>: Sized {
    // ── Construction ────────────────────────────────────────────────────────

    /// Fallibly construct an empty `HashMap` with a default-constructed hasher.
    ///
    /// Unlike [`Self::try_with_capacity`], which hardcodes [`RandomState`] and
    /// may panic on first use in a new thread (due to thread-local seeding),
    /// this method uses [`TryDefault`] to construct the hasher fallibly. If
    /// hasher construction fails (e.g. `RandomState` panics during seed
    /// initialization), the error is returned rather than unwinding.
    ///
    /// Requires `S: [`TryDefault`]` so that hasher creation is safe even when
    /// it involves runtime allocation or thread-local state.
    fn try_new() -> Result<HashMap<K, V, S>, TryHashMapConstructionError>
    where
        S: TryDefault;

    /// Fallibly construct an empty `HashMap` with at least enough capacity for
    /// `capacity` elements.
    ///
    /// Constructs the hasher via [`TryDefault`] (same as [`Self::try_new`]),
    /// then reserves capacity for `capacity` elements. Returns
    /// [`TryHashMapConstructionError::Reserve`] if the capacity reservation
    /// fails, or [`TryHashMapConstructionError::Default`] if hasher
    /// construction fails.
    ///
    /// Requires `S: [`TryDefault`]` so that hasher creation is safe even when
    /// it involves runtime allocation or thread-local state.
    fn try_with_capacity(capacity: usize) -> Result<HashMap<K, V, S>, TryHashMapConstructionError>
    where
        S: TryDefault;

    /// Fallibly construct an empty `HashMap` with at least enough capacity for
    /// `capacity` elements, using the provided hash builder.
    ///
    /// Returns [`TryReserveError`] if the initial allocation fails.
    /// Equivalent to [`HashMap::with_capacity_and_hasher`] but fallible.
    fn try_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<HashMap<K, V, S>, TryReserveError>;

    // ── Insertion ───────────────────────────────────────────────────────────

    /// Fallibly insert a key-value pair into the map, always replacing any
    /// existing value for the same key.
    ///
    /// Reserves capacity for one additional element before inserting, so this
    /// method never panics on out-of-memory. Returns [`TryReserveError`]
    /// if the capacity reservation fails.
    ///
    /// Returns `Ok(None)` if the key was not previously present, or
    /// `Ok(Some(old_value))` if the key existed and was replaced.
    ///
    /// Unlike the original [`HashMap::try_insert`], key collisions cause the old value to be evicted.
    /// See [`Self::try_insert_unique`] for the fallible version of the original behavior.
    ///
    /// **Deprecated:** This method name conflicts with the inherent
    /// [`HashMap::try_insert`]. Use [`Self::fallible_insert`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with inherent HashMap::try_insert; use fallible_insert"
    )]
    fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, TryReserveError>
    where
        K: Eq + Hash;

    /// Like [`Self::try_insert`] or [`Self::fallible_insert`] but returns ownership of `key` and `value` back on allocation failure.
    ///
    /// Unlike the original [`HashMap::try_insert`], key collisions cause the old value to be evicted. See [`Self::try_insert_unique`] for the variant that fails on key collisions.
    fn try_insert_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<Option<V>, (K, V, TryReserveError)>
    where
        K: Eq + Hash;

    /// Fallibly insert a key-value pair only if the key is not already present.
    ///
    /// Reserves capacity for one additional element before inserting, so this
    /// method never panics on out-of-memory.
    ///
    /// Returns `Ok(())` if the key was newly inserted. Returns
    /// `Err((key, value, TryHashMapInsertUniqueError))` if the insertion failed, giving ownership of
    /// both `key` and `value` back to the caller. The error is
    /// [`TryHashMapInsertUniqueError::Reserve`] on allocation failure or
    /// [`TryHashMapInsertUniqueError::KeyAlreadyExists`] if the key already exists.
    fn try_insert_unique(
        &mut self,
        key: K,
        value: V,
    ) -> Result<(), (K, V, TryHashMapInsertUniqueError)>
    where
        K: Eq + Hash;

    /// Fallibly obtain an [`Entry`] for a key, reserving capacity first.
    ///
    /// Reserves space for exactly one additional element so that subsequent
    /// operations on the returned [`Entry`] (such as [`Entry::or_insert`] or
    /// [`Entry::and_modify`]) cannot panic on out-of-memory. Returns
    /// [`TryReserveError`] if the capacity reservation fails.
    ///
    /// Unlike the inherent [`HashMap::entry`], this method guarantees that
    /// inserting through the entry will not allocate again.
    ///
    /// [`Entry`]: lang_std::collections::hash_map::Entry
    /// [`Entry::or_insert`]: lang_std::collections::hash_map::Entry::or_insert
    /// [`Entry::and_modify`]: lang_std::collections::hash_map::Entry::and_modify
    fn try_entry<'a>(
        &'a mut self,
        key: K,
    ) -> Result<hash_map::Entry<'a, K, V>, TryReserveError>
    where
        K: Eq + Hash;

    // ── Aliases with `fallible_` prefix ────────────────────────────────────

    /// Alias for [`Self::try_new`].
    fn fallible_new() -> Result<HashMap<K, V, S>, TryHashMapConstructionError>
    where
        S: TryDefault,
    {
        Self::try_new()
    }

    /// Alias for [`Self::try_with_capacity`].
    fn fallible_with_capacity(
        capacity: usize,
    ) -> Result<HashMap<K, V, S>, TryHashMapConstructionError>
    where
        S: TryDefault,
    {
        Self::try_with_capacity(capacity)
    }

    /// Alias for [`Self::try_with_capacity_and_hasher`].
    fn fallible_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<HashMap<K, V, S>, TryReserveError> {
        Self::try_with_capacity_and_hasher(capacity, hasher)
    }

    /// Fallibly insert a key-value pair into the map, always replacing any
    /// existing value for the same key.
    ///
    /// Reserves capacity for one additional element before inserting, so this
    /// method never panics on out-of-memory. Returns [`TryReserveError`]
    /// if the capacity reservation fails.
    ///
    /// Returns `Ok(None)` if the key was not previously present, or
    /// `Ok(Some(old_value))` if the key existed and was replaced.
    ///
    /// Unlike the original [`HashMap::try_insert`], key collisions cause the old value to be evicted. See [`Self::fallible_insert_unique`] for the original behavior.
    ///
    /// This method replaces the deprecated [`Self::try_insert`] which shares its
    /// name with the inherent [`HashMap::try_insert`].
    #[allow(deprecated)]
    fn fallible_insert(&mut self, key: K, value: V) -> Result<Option<V>, TryReserveError>
    where
        K: Eq + Hash,
    {
        Self::try_insert(self, key, value)
    }

    /// Like [`Self::fallible_insert`] but returns ownership of `key` and `value`
    /// back on allocation failure.
    ///
    /// Unlike the original [`HashMap::try_insert`], key collisions cause the old value to evicted. See [`Self::fallible_insert_unique`] for the variant that fails on key collisions.
    fn fallible_insert_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<Option<V>, (K, V, TryReserveError)>
    where
        K: Eq + Hash,
    {
        Self::try_insert_give_back(self, key, value)
    }

    /// Alias for [`Self::try_insert_unique`].
    fn fallible_insert_unique(
        &mut self,
        key: K,
        value: V,
    ) -> Result<(), (K, V, TryHashMapInsertUniqueError)>
    where
        K: Eq + Hash,
    {
        Self::try_insert_unique(self, key, value)
    }

    /// Alias for [`Self::try_entry`].
    fn fallible_entry<'a>(
        &'a mut self,
        key: K,
    ) -> Result<hash_map::Entry<'a, K, V>, TryReserveError>
    where
        K: Eq + Hash,
    {
        Self::try_entry(self, key)
    }

    // ── Extension ───────────────────────────────────────────────────────────

    /// Fallibly shrink the capacity of this hash map to match its length.
    ///
    /// Rebuilds the internal table so that it holds approximately `len` elements.
    /// Requires `S: TryClone` so the hasher can be safely duplicated for the new
    /// table without risking a panic. Returns [`TryHashMapWithCloneError::Reserve`]
    /// if the allocation for the rebuilt table fails, or
    /// [`TryHashMapWithCloneError::Clone`] if duplicating the hasher factory fails.
    /// Equivalent to [`HashMap::shrink_to_fit`] but fallible.
    fn try_shrink_to_fit(&mut self) -> Result<(), TryHashMapWithCloneError>
    where
        S: TryClone;

    /// Fallibly shrink the capacity of this hash map to hold at least
    /// `min_capacity` elements.
    ///
    /// If the current capacity is already less than or equal to `min_capacity`,
    /// does nothing and returns `Ok(())`. Otherwise rebuilds the table with the
    /// target capacity. Requires `S: TryClone` so the hasher can be safely
    /// duplicated. Returns [`TryHashMapWithCloneError::Reserve`] if the allocation
    /// fails, or [`TryHashMapWithCloneError::Clone`] if duplicating the hasher
    /// factory fails. Equivalent to [`HashMap::shrink_to`] but fallible.
    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryHashMapWithCloneError>
    where
        S: TryClone;

    /// Fallibly shrink the capacity of this hash map to match its length.
    ///
    /// Alias for [`Self::try_shrink_to_fit`].
    fn fallible_shrink_to_fit(&mut self) -> Result<(), TryHashMapWithCloneError>
    where
        S: TryClone,
    {
        Self::try_shrink_to_fit(self)
    }

    /// Fallibly shrink the capacity of this hash map to hold at least
    /// `min_capacity` elements.
    ///
    /// Alias for [`Self::fallible_shrink_to_fit`].
    fn fallible_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryHashMapWithCloneError>
    where
        S: TryClone,
    {
        Self::try_shrink_to(self, min_capacity)
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    /// Fallibly collect an iterator of `(K, V)` pairs into a `HashMap`.
    ///
    /// Constructs the hasher via [`TryDefault`] and uses the iterator's size
    /// hint to pre-allocate when possible. Returns
    /// [`TryHashMapConstructionError::Reserve`] if a capacity reservation fails,
    /// or [`TryHashMapConstructionError::Default`] if hasher construction fails.
    fn try_collect<I: IntoIterator<Item = (K, V)>>(
        iter: I,
    ) -> Result<HashMap<K, V, S>, TryHashMapConstructionError>
    where
        S: TryDefault;

    /// Fallibly create a `HashMap` from an iterator using the provided hasher.
    fn try_collect_with_hasher<I: IntoIterator<Item = (K, V)>>(
        iter: I,
        hasher: S,
    ) -> Result<HashMap<K, V, S>, TryReserveError>;

    /// Alias for [`Self::try_collect`].
    fn fallible_collect<I: IntoIterator<Item = (K, V)>>(
        iter: I,
    ) -> Result<HashMap<K, V, S>, TryHashMapConstructionError>
    where
        S: TryDefault,
    {
        Self::try_collect(iter)
    }

    /// Alias for [`Self::try_collect_with_hasher`].
    fn fallible_collect_with_hasher<I: IntoIterator<Item = (K, V)>>(
        iter: I,
        hasher: S,
    ) -> Result<HashMap<K, V, S>, TryReserveError> {
        Self::try_collect_with_hasher(iter, hasher)
    }
}

// ── Implementation ────────────────────────────────────────────────────────────

impl<K: Eq + Hash, V, S: BuildHasher> TryHashMap<K, V, S> for HashMap<K, V, S> {
    // ── Construction ────────────────────────────────────────────────────────

    fn try_new() -> Result<HashMap<K, V, S>, TryHashMapConstructionError>
    where
        S: TryDefault,
    {
        let hasher = S::try_default()?;
        Ok(HashMap::with_hasher(hasher))
    }

    fn try_with_capacity(capacity: usize) -> Result<HashMap<K, V, S>, TryHashMapConstructionError>
    where
        S: TryDefault,
    {
        let mut map = Self::try_new()?;
        if capacity > 0 {
            map.try_reserve(capacity)?;
        }
        Ok(map)
    }

    fn try_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<HashMap<K, V, S>, TryReserveError> {
        let mut map = HashMap::with_hasher(hasher);
        if capacity > 0 {
            map.try_reserve(capacity)?;
        }
        Ok(map)
    }

    // ── Insertion ───────────────────────────────────────────────────────────

    fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, TryReserveError>
    where
        K: Eq + Hash,
    {
        match self.try_entry(key)? {
            hash_map::Entry::Occupied(mut e) => Ok(Some(e.insert(value))),
            hash_map::Entry::Vacant(e) => {
                let _v = e.insert(value);
                Ok(None)
            }
        }
    }

    fn try_insert_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<Option<V>, (K, V, TryReserveError)>
    where
        K: Eq + Hash,
    {
        // Reserve first so we can return the key on failure without moving it.
        if let Err(e) = self.try_reserve(1) {
            return Err((key, value, e));
        }
        match self.entry(key) {
            hash_map::Entry::Occupied(mut e) => Ok(Some(e.insert(value))),
            hash_map::Entry::Vacant(e) => {
                let _v = e.insert(value);
                Ok(None)
            }
        }
    }

    fn try_insert_unique(
        &mut self,
        key: K,
        value: V,
    ) -> Result<(), (K, V, TryHashMapInsertUniqueError)>
    where
        K: Eq + Hash,
    {
        // Check existence first so we can return the key on duplicate without moving it.
        if self.contains_key(&key) {
            return Err((key, value, TryHashMapInsertUniqueError::KeyAlreadyExists));
        }
        if let Err(e) = self.try_reserve(1) {
            return Err((key, value, TryHashMapInsertUniqueError::Reserve(e)));
        }
        match self.entry(key) {
            hash_map::Entry::Occupied(_) => {
                // Unreachable: we checked above and no concurrent mutation is possible.
                unreachable!("key appeared between contains_key and entry")
            }
            hash_map::Entry::Vacant(e) => {
                let _v = e.insert(value);
                Ok(())
            }
        }
    }

    fn try_entry<'a>(
        &'a mut self,
        key: K,
    ) -> Result<hash_map::Entry<'a, K, V>, TryReserveError>
    where
        K: Eq + Hash,
    {
        self.try_reserve(1)?;
        Ok(self.entry(key))
    }

    // ── Capacity / shrink ───────────────────────────────────────────────────

    fn try_shrink_to_fit(&mut self) -> Result<(), TryHashMapWithCloneError>
    where
        S: TryClone,
    {
        Self::try_shrink_to(self, self.len())
    }

    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryHashMapWithCloneError>
    where
        S: TryClone,
    {
        let target = cmp::max(self.len(), min_capacity);
        if self.capacity() <= target {
            return Ok(());
        }
        let hasher = self
            .hasher()
            .try_clone()
            .map_err(TryHashMapWithCloneError::from)?;
        // Apparently, the hashbrown library also reallocates a new entire hash table for the shrink and moves items to the new table, so complexity wise, this should not be worse than the library.
        let mut new_map = HashMap::with_capacity_and_hasher(0, hasher);
        new_map
            .try_reserve(target)
            .map_err(TryHashMapWithCloneError::Reserve)?;
        for (k, v) in self.drain() {
            new_map.insert(k, v);
        }
        *self = new_map;
        Ok(())
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    fn try_collect<I: IntoIterator<Item = (K, V)>>(
        iter: I,
    ) -> Result<HashMap<K, V, S>, TryHashMapConstructionError>
    where
        S: TryDefault,
    {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower);
        let mut map = Self::try_new()?;
        if capacity > 0 {
            map.try_reserve(capacity)?;
        }
        for (key, value) in iter {
            if map.len() == map.capacity() {
                map.try_reserve(1)?;
            }
            map.insert(key, value);
        }
        Ok(map)
    }

    fn try_collect_with_hasher<I: IntoIterator<Item = (K, V)>>(
        iter: I,
        hasher: S,
    ) -> Result<HashMap<K, V, S>, TryReserveError> {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower);
        let mut map = HashMap::with_hasher(hasher);
        if capacity > 0 {
            map.try_reserve(capacity)?;
        }
        for (key, value) in iter {
            if map.len() == map.capacity() {
                map.try_reserve(1)?;
            }
            map.insert(key, value);
        }
        Ok(map)
    }
}

// ── TryClone for HashMap<K, V, S> ──────────────────────────────────────────────

/// Implements [`TryClone`] for `HashMap<K, V, S>` when keys and values are
/// cloneable and the hasher factory implements [`TryClone`].
impl<K, V, S> TryClone for HashMap<K, V, S>
where
    K: Eq + Hash + TryClone,
    V: TryClone,
    S: BuildHasher + TryClone,
{
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let hasher = self.hasher().try_clone()?;
        let mut out = HashMap::with_hasher(hasher);
        if !self.is_empty() {
            // Reserve space first so allocation failures are caught early.
            out.try_reserve(self.len())
                .map_err(TryCloneError::Reserve)?;
        }
        for (key, value) in self.iter() {
            match (key.try_clone(), value.try_clone()) {
                (Ok(k), Ok(v)) => {
                    out.insert(k, v);
                }
                (Err(e), _) | (_, Err(e)) => {
                    drop(out);
                    return Err(e);
                }
            }
        }
        Ok(out)
    }
}

// ── TryDefault for HashMap<K, V, S> ────────────────────────────────────────────

impl<K, V, S: BuildHasher + TryDefault> TryDefault for HashMap<K, V, S> {
    fn try_default() -> Result<Self, TryDefaultError> {
        let hasher = S::try_default()?;
        Ok(HashMap::with_hasher(hasher))
    }
}

// ── TryDebug for HashMap<K, V, S> ──────────────────────────────────────────────

impl<K: crate::try_fmt::TryDebug, V: crate::try_fmt::TryDebug, S> crate::try_fmt::TryDebug
    for HashMap<K, V, S>
{
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::try_fmt::helpers::FormatterExt;
        f.try_debug_map().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::TryReserveErrorExt;
    use lang_alloc::format;
    use lang_alloc::string::String;
    use lang_alloc::string::ToString;
    use lang_alloc::vec;
    use lang_alloc::vec::Vec;
    use lang_core::fmt::Write as _;
    use lang_std::collections::hash_map::RandomState;
    use lang_std::iter;

    /// A `TryReserveError` instance for exercising the `Reserve` arm.
    fn reserve_err() -> TryReserveError {
        TryReserveError::new_capacity_overflow()
    }

    /// Formats a value via its `Display` impl into a fresh String.
    fn render_display(e: &impl fmt::Display) -> String {
        let mut s = String::new();
        // Our error Display impls only call `write!` on literals/wrapped values,
        // so this cannot fail in practice; ignore the infallible-in-practice result.
        let _ = write!(&mut s, "{e}");
        s
    }

    /// Captures the `TryDebug` rendering of a value.
    fn render_trydebug(e: &impl TryDebug) -> String {
        struct Cap<'a>(&'a dyn TryDebug);
        impl fmt::Debug for Cap<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.try_fmt(f)
            }
        }
        format!("{:?}", Cap(e))
    }

    /// Captures the `TryDisplay` rendering of a value (should match `Display`).
    fn render_trydisplay(e: &impl TryDisplay) -> String {
        struct Cap<'a>(&'a dyn TryDisplay);
        impl fmt::Display for Cap<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.try_fmt(f)
            }
        }
        let mut s = String::new();
        let _ = write!(&mut s, "{}", Cap(e));
        s
    }

    /// Exercises every variant of `TryHashMapConstructionError` through all
    /// three impls (moved from errors::uniform).
    #[test]
    fn hashmap_construction_error_covers_all_variants() {
        let default_err = || TryDefaultError::Alloc(crate::alloc::AllocError);
        let errs = [
            TryHashMapConstructionError::Reserve(reserve_err()),
            TryHashMapConstructionError::Default(default_err()),
        ];
        for err in errs.iter() {
            let disp = render_display(err);
            assert!(
                disp.starts_with("hash map operation failed:"),
                "got {disp:?}"
            );
            let tdisp = render_trydisplay(err);
            assert_eq!(tdisp, disp, "TryDisplay must match Display");
            let dbg = render_trydebug(err);
            assert!(
                dbg.contains("TryHashMapConstructionError::"),
                "got {dbg:?}"
            );
        }
    }

    /// Error shape returned by [`TryExtendFromSlice::try_extend_from_slice`]:
    /// the unconsumed tail of the source slice paired with the failure reason.
    type ExtendErr<'a, K, V> = Result<(), (&'a [(K, V)], TryHashMapWithCloneError)>;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn try_with_capacity_zero() {
        let map: HashMap<i32, i32> = HashMap::<i32, i32>::try_with_capacity(0).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn try_with_capacity_nonzero() {
        let map: HashMap<i32, String> = HashMap::<i32, String>::try_with_capacity(10).unwrap();
        assert!(map.is_empty());
        assert!(map.capacity() >= 10);
    }

    #[test]
    fn try_with_capacity_and_hasher() {
        use lang_std::collections::hash_map::DefaultHasher;
        use lang_std::hash::BuildHasherDefault;

        let hasher = BuildHasherDefault::<DefaultHasher>::default();
        let map: HashMap<&str, i32, _> = HashMap::try_with_capacity_and_hasher(5, hasher).unwrap();
        assert!(map.is_empty());
        assert!(map.capacity() >= 5);
    }

    // ── Insertion ────────────────────────────────────────────────────────────

    #[test]
    fn fallible_insert_single() {
        let mut map: HashMap<i32, &str> = HashMap::new();
        let old = map.fallible_insert(1, "one").unwrap();
        assert_eq!(old, None);
        assert_eq!(map.get(&1), Some(&"one"));
    }

    #[test]
    fn fallible_insert_evicts_old_value() {
        let mut map: HashMap<i32, &str> = HashMap::new();
        map.fallible_insert(1, "one").unwrap();
        let old = map.fallible_insert(1, "ONE").unwrap();
        assert_eq!(old, Some("one"));
        assert_eq!(map.get(&1), Some(&"ONE"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn fallible_insert_multiple_keys() {
        let mut map: HashMap<&str, i32> = HashMap::new();
        map.fallible_insert("a", 1).unwrap();
        map.fallible_insert("b", 2).unwrap();
        map.fallible_insert("c", 3).unwrap();
        assert_eq!(map.len(), 3);
        assert_eq!(map["a"], 1);
        assert_eq!(map["b"], 2);
        assert_eq!(map["c"], 3);
    }

    #[test]
    fn fallible_insert_complex_values() {
        let mut map: HashMap<String, Vec<u8>> = HashMap::new();
        map.fallible_insert("key".to_string(), vec![1, 2, 3])
            .unwrap();
        assert_eq!(map["key"], vec![1, 2, 3]);
    }

    // ── Unique insertion ─────────────────────────────────────────────────────

    #[test]
    fn try_insert_unique_new_key() {
        let mut map: HashMap<i32, &str> = HashMap::new();
        map.try_insert_unique(1, "one").unwrap();
        assert_eq!(map.get(&1), Some(&"one"));
    }

    #[test]
    fn try_insert_unique_duplicate_rejected() {
        let mut map: HashMap<i32, &str> = HashMap::new();
        map.try_insert_unique(1, "one").unwrap();
        let result = map.try_insert_unique(1, "TWO");
        let (returned_key, returned_val, base_err) = result.unwrap_err();
        assert_eq!(returned_key, 1);
        assert_eq!(returned_val, "TWO");
        assert!(matches!(
            base_err,
            TryHashMapInsertUniqueError::KeyAlreadyExists
        ));
        assert_eq!(map.get(&1), Some(&"one"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn fallible_insert_unique_alias_works() {
        let mut map: HashMap<i32, String> = HashMap::new();
        map.fallible_insert_unique(42, "hi".to_string()).unwrap();
        assert_eq!(map[&42], "hi");
    }

    // ── Extension ────────────────────────────────────────────────────────────

    #[test]
    fn try_extend_from_iterator() {
        use crate::try_extend::TryExtend;
        let mut map: HashMap<i32, &str> = HashMap::new();
        <_ as TryExtend<(i32, &str)>>::try_extend(&mut map, [(1, "one"), (2, "two")]).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map[&1], "one");
    }

    #[test]
    fn try_extend_empty() {
        use crate::try_extend::TryExtend;
        let mut map: HashMap<i32, i32> = HashMap::new();
        <_ as TryExtend<(i32, i32)>>::try_extend(&mut map, iter::empty::<(i32, i32)>()).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn try_extend_existing() {
        use crate::try_extend::TryExtend;
        let mut map: HashMap<i32, &str> = HashMap::new();
        map.fallible_insert(1, "one").unwrap();
        <_ as TryExtend<(i32, &str)>>::try_extend(&mut map, [(2, "two"), (3, "three")]).unwrap();
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn try_extend_from_slice_clones() {
        use crate::try_extend::TryExtendFromSlice;
        let mut map: HashMap<String, Vec<u8>> = HashMap::new();
        map.fallible_insert("a".to_string(), vec![1]).unwrap();
        let slice: &[(String, Vec<u8>)] = &[("b".to_string(), vec![2, 3])];
        <_ as TryExtendFromSlice<'_, (String, Vec<u8>)>>::try_extend_from_slice(&mut map, slice)
            .unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map["b"], vec![2, 3]);
    }

    #[test]
    fn try_extend_from_slice_empty() {
        use crate::try_extend::TryExtendFromSlice;
        let mut map: HashMap<i32, i32> = HashMap::new();
        <_ as TryExtendFromSlice<'_, (i32, i32)>>::try_extend_from_slice(&mut map, &[]).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn try_extend_from_slice_later_entry_overwrites_earlier_key() {
        use crate::try_extend::TryExtendFromSlice;
        // A later entry in the source overwrites an earlier one for the same key.
        // The final value must be the last occurrence, matching `Extend` semantics.
        let mut map: HashMap<&str, i32> = HashMap::new();
        let slice: &[(&str, i32)] = &[("k", 1), ("j", 9), ("k", 2)];
        <_ as TryExtendFromSlice<'_, (&str, i32)>>::try_extend_from_slice(&mut map, slice).unwrap();
        assert_eq!(map["k"], 2);
        assert_eq!(map["j"], 9);
        assert_eq!(map.len(), 2);
    }

    // ── Shrink ────────────────────────────────────────────────────────────────

    #[test]
    fn try_shrink_to_fit_preserves_data() {
        let mut map: HashMap<i32, String> = HashMap::new();
        map.fallible_insert(1, "hello".to_string()).unwrap();
        map.fallible_insert(2, "world".to_string()).unwrap();
        map.fallible_shrink_to_fit().unwrap();
        assert_eq!(map[&1], "hello");
        assert_eq!(map[&2], "world");
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn try_shrink_to_fit_reduces_excess() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        map.try_reserve(1024).unwrap();
        map.fallible_insert(1, 100).unwrap();
        let cap_before = map.capacity();
        assert!(cap_before >= 1024);
        map.fallible_shrink_to_fit().unwrap();
        assert!(map.capacity() < cap_before || map.capacity() >= 1);
        assert_eq!(map[&1], 100);
    }

    #[test]
    fn try_shrink_to_above_len() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        map.try_reserve(256).unwrap();
        map.fallible_insert(42, 99).unwrap();
        let _cap_before = map.capacity();
        map.fallible_shrink_to(32).unwrap();
        assert!(map.capacity() >= 32);
        assert_eq!(map[&42], 99);
    }

    #[test]
    fn try_shrink_to_noop_when_already_small() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        map.fallible_insert(1, 1).unwrap();
        let _cap_before = map.capacity();
        map.fallible_shrink_to(16).unwrap();
        // capacity may stay the same since it's already small.
        assert_eq!(map[&1], 1);
    }

    // ── Bulk construction ────────────────────────────────────────────────────

    #[test]
    fn try_collect_range() {
        let map: HashMap<i32, i32> =
            <HashMap<i32, i32> as TryHashMap<_, _, RandomState>>::try_collect(
                (0..5).map(|i| (i, i * 10)),
            )
            .unwrap();
        assert_eq!(map.len(), 5);
        assert_eq!(map[&3], 30);
    }

    #[test]
    fn try_collect_empty() {
        let map: HashMap<i32, i32> =
            <HashMap<i32, i32> as TryHashMap<_, _, RandomState>>::try_collect(iter::empty::<(
                i32,
                i32,
            )>())
            .unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn try_collect_strings() {
        let pairs = vec![("a".to_string(), 1), ("b".to_string(), 2)];
        let map: HashMap<String, i32> =
            <HashMap<String, i32> as TryHashMap<_, _, RandomState>>::try_collect(pairs).unwrap();
        assert_eq!(map["a"], 1);
        assert_eq!(map["b"], 2);
    }

    #[test]
    fn try_collect_with_hasher() {
        use lang_std::collections::hash_map::DefaultHasher;
        use lang_std::hash::BuildHasherDefault;

        let hasher = BuildHasherDefault::<DefaultHasher>::default();
        let map: HashMap<i32, i32, _> =
            HashMap::try_collect_with_hasher([(1, 10), (2, 20)], hasher).unwrap();
        assert_eq!(map[&1], 10);
        assert_eq!(map[&2], 20);
    }

    // ── Give-back variants ───────────────────────────────────────────────────

    #[test]
    fn fallible_insert_give_back_success() {
        let mut map: HashMap<String, Vec<u8>> = HashMap::new();
        map.fallible_insert_give_back("k".to_string(), vec![1])
            .unwrap();
        assert_eq!(map["k"], vec![1]);
    }

    #[test]
    fn fallible_insert_give_back_error_type_shape() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        let result: Result<Option<i32>, (i32, i32, TryReserveError)> =
            map.fallible_insert_give_back(1, 2);
        assert!(result.is_ok());
    }

    // ── Entry API ────────────────────────────────────────────────────────────

    #[test]
    fn try_entry_or_insert_new_key() {
        let mut map: HashMap<String, i32> = HashMap::new();
        map.try_entry("hello".to_string()).unwrap().or_insert(42);
        assert_eq!(map["hello"], 42);
    }

    #[test]
    fn try_entry_or_insert_existing_key() {
        let mut map: HashMap<String, i32> = HashMap::new();
        map.fallible_insert("key".to_string(), 1).unwrap();
        map.try_entry("key".to_string()).unwrap().or_insert(99);
        assert_eq!(map["key"], 1);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn try_entry_and_modify() {
        let mut map: HashMap<String, i32> = HashMap::new();
        map.fallible_insert("count".to_string(), 5).unwrap();
        map.try_entry("count".to_string())
            .unwrap()
            .and_modify(|v| *v += 1);
        assert_eq!(map["count"], 6);
    }

    #[test]
    fn try_entry_or_insert_with_fn() {
        let mut map: HashMap<String, Vec<u8>> = HashMap::new();
        map.try_entry("data".to_string())
            .unwrap()
            .or_insert_with(|| vec![1, 2, 3]);
        assert_eq!(map["data"], vec![1, 2, 3]);
    }

    #[test]
    fn try_entry_vacant_inserts() {
        let mut map: HashMap<i32, String> = HashMap::new();
        map.fallible_insert(1, "one".to_string()).unwrap();
        map.try_entry(99).unwrap().or_insert_with(String::new);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn fallible_entry_matches_try_entry() {
        let mut map: HashMap<String, i32> = HashMap::new();
        map.fallible_entry("x".to_string()).unwrap().or_insert(10);
        assert_eq!(map["x"], 10);
    }

    // ── TryClone ─────────────────────────────────────────────────────────────

    #[test]
    fn try_clone_empty_map() {
        let map: HashMap<i32, i32> = HashMap::new();
        let c = map.try_clone().unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn try_clone_populated_map() {
        let mut map: HashMap<i32, String> = HashMap::new();
        map.insert(1, "hello".to_string());
        map.insert(2, "world".to_string());
        let c = map.try_clone().unwrap();
        assert_eq!(c[&1], "hello");
        assert_eq!(c[&2], "world");
    }

    #[test]
    fn try_clone_nested_values() {
        let mut map: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        map.insert("nested".to_string(), vec![vec![1, 2], vec![3]]);
        let c = map.try_clone().unwrap();
        assert_eq!(c["nested"], vec![vec![1, 2], vec![3]]);
    }

    // ── TryDefault ───────────────────────────────────────────────────────────

    #[test]
    fn try_default_empty_map() {
        let map: HashMap<i32, i32> = HashMap::try_default().unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn try_default_map_with_custom_hasher() {
        use lang_std::collections::hash_map::DefaultHasher;
        use lang_std::hash::BuildHasherDefault;

        let map: HashMap<i32, i32, BuildHasherDefault<DefaultHasher>> =
            HashMap::try_default().unwrap();
        assert!(map.is_empty());
    }

    // ── Combined workflows ───────────────────────────────────────────────────

    #[test]
    fn build_insert_clone_default() {
        let mut map: HashMap<String, i32> = HashMap::try_default().unwrap();
        map.fallible_insert("alpha".to_string(), 1).unwrap();
        map.fallible_insert("beta".to_string(), 2).unwrap();
        let c = map.try_clone().unwrap();
        assert_eq!(c["alpha"], 1);
        assert_eq!(c["beta"], 2);
        // Original still intact
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn collect_then_extend() {
        use crate::try_extend::TryExtend;
        let mut a: HashMap<i32, &str> =
            <HashMap<i32, &str> as TryHashMap<_, _, RandomState>>::try_collect([
                (1, "one"),
                (2, "two"),
            ])
            .unwrap();
        <_ as TryExtend<(i32, &str)>>::try_extend(&mut a, [(3, "three")]).unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(a[&3], "three");
    }

    #[test]
    fn extend_from_slice_success_error_type_shape() {
        use crate::try_extend::TryExtendFromSlice;
        // Verify the success path and the (remaining_subslice, error) tuple shape.
        let mut map: HashMap<String, Vec<u8>> = HashMap::new();
        let slice: &[(String, Vec<u8>)] = &[("x".to_string(), vec![1])];
        let result: ExtendErr<'_, String, Vec<u8>> =
            <_ as TryExtendFromSlice<'_, (String, Vec<u8>)>>::try_extend_from_slice(
                &mut map, slice,
            );
        assert!(result.is_ok());
        assert_eq!(map["x"], vec![1]);
    }

    #[test]
    fn fallible_aliases_match_try_methods() {
        let m1: HashMap<i32, i32> =
            <HashMap<i32, i32> as TryHashMap<_, _, RandomState>>::fallible_with_capacity(5)
                .unwrap();
        let m2: HashMap<i32, i32> =
            <HashMap<i32, i32> as TryHashMap<_, _, RandomState>>::try_with_capacity(5).unwrap();
        assert!(m1.is_empty());
        assert!(m2.is_empty());
    }

    // ── OOM tests ─────────────────────────────────────────────────────────────
    use rustyfill_test_allocator::{FailPolicy, with_policy};

    #[test]
    fn hashmap_try_with_capacity_fails_on_oom() {
        let r: Result<HashMap<u32, u32>, TryHashMapConstructionError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <HashMap<u32, u32> as TryHashMap<u32, u32, RandomState>>::try_with_capacity(10)
            });
        assert!(r.is_err());
    }

    #[test]
    fn hashmap_try_with_capacity_zero_succeeds_under_oom() {
        let r: Result<HashMap<u32, u32>, TryHashMapConstructionError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <HashMap<u32, u32> as TryHashMap<u32, u32, RandomState>>::try_with_capacity(0)
            });
        assert!(r.is_ok());
    }

    #[test]
    fn hashmap_try_insert_fails_on_oom() {
        let mut map: HashMap<u32, u32> = HashMap::new();
        map.try_shrink_to_fit().unwrap();
        let r = with_policy(FailPolicy::fail_next_alloc(), || map.fallible_insert(1, 2));
        assert!(r.is_err());
    }

    #[test]
    fn hashmap_try_clone_fails_on_oom() {
        let orig: HashMap<u32, u32> = HashMap::from([(1, 2), (3, 4)]);
        let r: Result<HashMap<u32, u32>, TryCloneError> =
            with_policy(FailPolicy::fail_next_alloc(), || orig.try_clone());
        assert!(r.is_err());
    }

    #[test]
    fn hashmap_try_clone_empty_succeeds_under_oom() {
        let orig: HashMap<u32, u32> = HashMap::new();
        let r: Result<HashMap<u32, u32>, TryCloneError> =
            with_policy(FailPolicy::fail_next_alloc(), || orig.try_clone());
        assert!(r.is_ok());
    }

    #[test]
    fn hashmap_try_collect_fails_on_oom() {
        let pairs = [(1u32, 2u32), (3u32, 4u32)];
        let r: Result<HashMap<u32, u32>, TryHashMapConstructionError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                HashMap::try_collect(pairs.iter().copied())
            });
        assert!(r.is_err());
    }

    #[test]
    fn hashmap_oom_restores_allocation_afterwards() {
        let r: Result<HashMap<u32, u32>, TryHashMapConstructionError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <HashMap<u32, u32> as TryHashMap<u32, u32, RandomState>>::try_with_capacity(10)
            });
        assert!(r.is_err());
        // Allocation works again after guard scope ends.
        let r: Result<HashMap<u32, u32>, TryHashMapConstructionError> =
            <HashMap<u32, u32> as TryHashMap<u32, u32, RandomState>>::try_with_capacity(10);
        assert!(r.is_ok());
    }

    #[test]
    fn hashmap_nth_alloc_fail_targets_correct_call() {
        type HM = HashMap<u32, u32, RandomState>;
        let (r1_ok, r2_err, r3_ok) = with_policy(FailPolicy::fail_nth_alloc(2), || {
            let r1: Result<HM, TryHashMapConstructionError> =
                <HM as TryHashMap<u32, u32, RandomState>>::try_with_capacity(1);
            let r2: Result<HM, TryHashMapConstructionError> =
                <HM as TryHashMap<u32, u32, RandomState>>::try_with_capacity(1);
            let r3: Result<HM, TryHashMapConstructionError> =
                <HM as TryHashMap<u32, u32, RandomState>>::try_with_capacity(1);
            (r1.is_ok(), r2.is_err(), r3.is_ok())
        });
        assert!(r1_ok, "first alloc should succeed");
        assert!(r2_err, "second alloc should fail");
        assert!(r3_ok, "third alloc should succeed");
    }

    // ── Mid-operation clone failure: no rollback, remainder returned ─────────

    #[test]
    fn extend_from_slice_returns_remaining_subslice_on_clone_failure() {
        // A mid-way clone failure must NOT roll back already-inserted entries
        // (keys may have been overwritten by later source entries, so removing
        // them would resurrect stale values). Instead, it returns the
        // unprocessed tail of the slice alongside the error.
        use lang_alloc::string::String;

        let source: Vec<(String, String)> = vec![
            ("key0".into(), "val0".into()),
            ("key1".into(), "val1".into()),
            ("key2".into(), "val2".into()),
            ("key3".into(), "val3".into()),
            ("key4".into(), "val4".into()),
            ("key5".into(), "val5".into()),
            ("key6".into(), "val6".into()),
            ("key7".into(), "val7".into()),
            ("key8".into(), "val8".into()),
            ("key9".into(), "val9".into()),
        ];

        let mut map: HashMap<String, String> = HashMap::new();

        use crate::try_extend::TryExtendFromSlice;
        let r: ExtendErr<'_, String, String> = with_policy(FailPolicy::fail_nth_alloc(2), || {
            <HashMap<String, String> as TryExtendFromSlice<'_, (String, String)>>::try_extend_from_slice(&mut map, &source)
        });

        match r {
            Err((remaining, err)) => {
                matches!(err, TryHashMapWithCloneError::Clone(_));
                // The returned subslice must be a contiguous tail of `source`.
                assert!(!remaining.is_empty());
                let fail_idx = source.len() - remaining.len();
                assert_eq!(remaining, &source[fail_idx..]);
                // Every entry before the failing index was inserted (no rollback).
                for i in 0..fail_idx {
                    assert_eq!(map[&source[i].0], source[i].1);
                }
                // Entries at or after the failing index were never inserted.
                for source in source.iter().skip(fail_idx) {
                    assert!(
                        !map.contains_key(&source.0),
                        "entry at failing index or beyond should not be present"
                    );
                }
            }
            Ok(()) => {
                // If no allocation failed, everything landed.
                assert_eq!(map.len(), source.len());
            }
        }
    }

    #[test]
    fn extend_from_slice_no_rollback_preserves_overwritten_keys() {
        // The critical scenario motivating "no rollback": the source contains
        // duplicate keys where a LATER entry overwrites an EARLIER one. Both
        // "dup" entries are placed at the front so they are always processed
        // before any mid-slice clone failure can occur. If we had drained
        // already-inserted entries on that later failure, "dup" would have been
        // removed entirely (resurrecting nothing, but losing the valid last
        // value). With no-rollback, "dup" must retain its LAST value "second".
        use lang_alloc::string::String;

        let source: Vec<(String, String)> = vec![
            ("dup".into(), "first".into()),  // index 0: dup -> "first"
            ("dup".into(), "second".into()), // index 1: dup -> "second" (overwrite)
            ("a".into(), "va".into()),
            ("b".into(), "vb".into()),
            ("c".into(), "vc".into()),
            ("d".into(), "vd".into()),
            ("e".into(), "ve".into()),
            ("f".into(), "vf".into()),
            ("g".into(), "vg".into()),
            ("h".into(), "vh".into()),
            ("i".into(), "vi".into()),
            ("j".into(), "vj".into()),
        ];

        let mut map: HashMap<String, String> = HashMap::new();

        use crate::try_extend::TryExtendFromSlice;
        // Fail a clone well past indices 0..2 so both "dup" entries are
        // guaranteed to be committed before the failure fires.
        let r: ExtendErr<'_, String, String> = with_policy(FailPolicy::fail_nth_alloc(8), || {
            <HashMap<String, String> as TryExtendFromSlice<'_, (String, String)>>::try_extend_from_slice(&mut map, &source)
        });

        match r {
            Err((remaining, err)) => {
                matches!(err, TryHashMapWithCloneError::Clone(_));
                // The failure must have occurred strictly after both "dup"
                // entries were processed, i.e. the remaining tail starts at
                // some index >= 2.
                let fail_idx = source.len() - remaining.len();
                assert!(
                    fail_idx >= 2,
                    "failure should land past the two 'dup' entries, got index {}",
                    fail_idx
                );
                // No-rollback: "dup" survived with its LAST (overwriting) value.
                assert_eq!(
                    map.get("dup"),
                    Some(&"second".to_string()),
                    "overwritten key must keep its LAST value, not the earlier stale one"
                );
            }
            Ok(()) => {
                // No allocation failed — everything landed, including the overwrite.
                assert_eq!(map.get("dup"), Some(&"second".to_string()));
            }
        }
    }
}
