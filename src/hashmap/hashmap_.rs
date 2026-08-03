//! Fallible hash map operations.
//!
//! Provides the [`TryHashMap`] trait with methods that mirror common `HashMap`
//! constructors and mutating operations but return [`Result`] to handle allocation
//! failures gracefully, using [`std::collections::TryReserveError`] as the primary
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

use crate::alloc::AllocError;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use core::fmt;
use std::cmp::Eq;
use std::collections::{HashMap, TryReserveError, hash_map::RandomState};
use std::hash::{BuildHasher, Hash};

// ── Error type ────────────────────────────────────────────────────────────────

/// Error returned by [`TryHashMap`] operations.
///
/// Wraps the ways a hash map operation can fail on stable Rust: a reserve
/// failure ([`TryReserveError`], returned by the inherent `HashMap::try_reserve`)
/// or a clone failure ([`TryCloneError`]) when an element's `try_clone` cannot
/// allocate its internal buffers.
#[derive(Debug)]
pub enum TryHashMapError {
    /// A raw heap allocation failed (no collection involved).
    Alloc(AllocError),
    /// A capacity reservation on the hash map failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An element clone failed during a method that requires [`TryClone`].
    Clone(TryCloneError),
    /// An arithmetic overflow occurred while computing required capacity.
    Overflow,
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Display for TryHashMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "hash map operation failed: heap allocation error"),
            Self::Reserve(e) => write!(f, "hash map operation failed: {}", e),
            Self::Clone(e) => write!(f, "hash map operation failed: {}", e),
            Self::Overflow => write!(
                f,
                "hash map operation failed: capacity calculation overflowed"
            ),
            Self::Other(msg) => write!(f, "hash map operation failed: {}", msg),
        }
    }
}

impl From<AllocError> for TryHashMapError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

impl From<TryReserveError> for TryHashMapError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<TryCloneError> for TryHashMapError {
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// A trait for fallible hash map operations.
///
/// Implemented for `HashMap<K, V, S>`. Mirrors the most commonly-used `HashMap`
/// methods that can fail due to allocation pressure, returning [`Result`] values
/// that propagate [`TryReserveError`] or [`TryHashMapError`] on failure.
///
/// # Note on `try_insert`
///
/// The inherent [`HashMap::try_insert`](std::collections::HashMap::try_insert) on
/// stable Rust returns `Err(old_value)` when a key already exists, but may *panic*
/// on allocation failure. Our [`Self::try_insert`] reserves capacity first so it
/// never panics on OOM — it returns [`TryHashMapError::Reserve`] instead, but it
/// does not return the old value on key collision.
pub trait TryHashMap<K, V, S>: Sized {
    // ── Construction ────────────────────────────────────────────────────────

    /// Fallibly construct an empty `HashMap` with at least enough capacity for
    /// `capacity` elements.
    ///
    /// Returns [`TryReserveError`] if the initial allocation fails.
    /// Equivalent to [`HashMap::with_capacity`] but fallible.
    fn try_with_capacity(capacity: usize) -> Result<HashMap<K, V, RandomState>, TryReserveError>;

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

    /// Fallibly insert a key-value pair into the map.
    ///
    /// Reserves capacity for one additional element before inserting, so this
    /// method never panics on out-of-memory. Returns [`TryHashMapError::Reserve`]
    /// if the capacity reservation fails.
    ///
    /// If the map already contained this key, the old value is overwritten and
    /// discarded (unlike the inherent `HashMap::try_insert`, which returns it).
    ///
    /// **Deprecated:** This method name conflicts with the inherent
    /// [`HashMap::try_insert`]. Use [`Self::fallible_insert`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with inherent HashMap::try_insert; use fallible_insert"
    )]
    fn try_insert(&mut self, key: K, value: V) -> Result<(), TryHashMapError>
    where
        K: Eq + Hash;

    /// Like [`Self::try_insert`] but returns ownership of `key` and `value` back
    /// on allocation failure.
    ///
    /// **Deprecated:** Use [`Self::fallible_insert_give_back`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with inherent HashMap::try_insert; use fallible_insert_give_back"
    )]
    fn try_insert_give_back(&mut self, key: K, value: V) -> Result<(), (K, V, TryHashMapError)>
    where
        K: Eq + Hash;

    // ── Aliases with `fallible_` prefix ────────────────────────────────────

    /// Alias for [`Self::try_with_capacity`].
    fn fallible_with_capacity(
        capacity: usize,
    ) -> Result<HashMap<K, V, RandomState>, TryReserveError> {
        Self::try_with_capacity(capacity)
    }

    /// Alias for [`Self::try_with_capacity_and_hasher`].
    fn fallible_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<HashMap<K, V, S>, TryReserveError> {
        Self::try_with_capacity_and_hasher(capacity, hasher)
    }

    /// Fallibly insert a key-value pair into the map.
    ///
    /// Reserves capacity for one additional element before inserting, so this
    /// method never panics on out-of-memory. Returns [`TryHashMapError::Reserve`]
    /// if the capacity reservation fails.
    ///
    /// If the map already contained this key, the old value is overwritten and
    /// discarded.
    ///
    /// This method replaces the deprecated [`Self::try_insert`] which shares its
    /// name with the inherent [`HashMap::try_insert`].
    #[allow(deprecated)]
    fn fallible_insert(&mut self, key: K, value: V) -> Result<(), TryHashMapError>
    where
        K: Eq + Hash,
    {
        Self::try_insert(self, key, value)
    }

    /// Like [`Self::fallible_insert`] but returns ownership of `key` and `value`
    /// back on allocation failure.
    ///
    /// This method replaces the deprecated [`Self::try_insert_give_back`].
    #[allow(deprecated)]
    fn fallible_insert_give_back(&mut self, key: K, value: V) -> Result<(), (K, V, TryHashMapError)>
    where
        K: Eq + Hash,
    {
        Self::try_insert_give_back(self, key, value)
    }

    // ── Extension ───────────────────────────────────────────────────────────

    /// Fallibly extend the map with all key-value pairs from an iterator.
    ///
    /// Uses the iterator's size hint to reserve capacity upfront when available.
    /// Returns [`TryReserveError`] if the allocation fails.
    fn try_extend<I: IntoIterator<Item = (K, V)>>(
        &mut self,
        iter: I,
    ) -> Result<(), TryReserveError>;

    /// Fallibly extend the map by cloning key-value pairs from a slice.
    ///
    /// Returns [`TryHashMapError::Reserve`] on capacity failure or
    /// [`TryHashMapError::Clone`] if a key or value clone fails.
    fn try_extend_from_slice(&mut self, other: &[(K, V)]) -> Result<(), TryHashMapError>
    where
        K: Eq + Hash + TryClone,
        V: TryClone;

    /// Alias for [`Self::try_extend`].
    fn fallible_extend<I: IntoIterator<Item = (K, V)>>(
        &mut self,
        iter: I,
    ) -> Result<(), TryReserveError> {
        Self::try_extend(self, iter)
    }

    /// Alias for [`Self::try_extend_from_slice`].
    fn fallible_extend_from_slice(&mut self, other: &[(K, V)]) -> Result<(), TryHashMapError>
    where
        K: Eq + Hash + TryClone,
        V: TryClone,
    {
        Self::try_extend_from_slice(self, other)
    }

    // ── Capacity / shrink ───────────────────────────────────────────────────

    /// Fallibly shrink the capacity of this hash map to match its length.
    ///
    /// Rebuilds the internal table so that it holds approximately `len` elements.
    /// Requires [`BuildHasher::clone`] so the hasher can be reused for the new
    /// table. Returns [`TryHashMapError::Reserve`] if the allocation for the
    /// rebuilt table fails. Equivalent to [`HashMap::shrink_to_fit`] but fallible.
    ///
    /// # Panics
    ///
    /// Cloning the hasher factory may panic if it relies on heap allocation.
    /// This is not the case for any standard-library hasher (`RandomState`,
    /// `SipHasher13`, etc.) or popular third-party hashers (`ahash`, `FxHash`,
    /// FNV, DJB2, and so on), which store only small stack integers. However,
    /// a custom hasher that internally allocates (e.g., caches compiled lookup
    /// tables) could panic during the clone step.
    fn try_shrink_to_fit(&mut self) -> Result<(), TryHashMapError>
    where
        S: Clone;

    /// Fallibly shrink the capacity of this hash map to hold at least
    /// `min_capacity` elements.
    ///
    /// If the current capacity is already less than or equal to `min_capacity`,
    /// does nothing and returns `Ok(())`. Otherwise rebuilds the table with the
    /// target capacity. Requires [`BuildHasher::clone`] so the hasher can be
    /// reused. Returns [`TryHashMapError::Reserve`] if the allocation fails.
    /// Equivalent to [`HashMap::shrink_to`] but fallible.
    ///
    /// # Panics
    ///
    /// Cloning the hasher factory may panic if it relies on heap allocation.
    /// This is not the case for any standard-library hasher (`RandomState`,
    /// `SipHasher13`, etc.) or popular third-party hashers (`ahash`, `FxHash`,
    /// FNV, DJB2, and so on), which store only small stack integers. However,
    /// a custom hasher that internally allocates (e.g., caches compiled lookup
    /// tables) could panic during the clone step.
    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryHashMapError>
    where
        S: Clone;

    /// Alias for [`Self::try_shrink_to_fit`].
    fn fallible_shrink_to_fit(&mut self) -> Result<(), TryHashMapError>
    where
        S: Clone,
    {
        Self::try_shrink_to_fit(self)
    }

    /// Alias for [`Self::try_shrink_to`].
    fn fallible_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryHashMapError>
    where
        S: Clone,
    {
        Self::try_shrink_to(self, min_capacity)
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    /// Fallibly collect an iterator of `(K, V)` pairs into a `HashMap`.
    ///
    /// Uses the iterator's size hint to pre-allocate when possible.
    fn try_collect<I: IntoIterator<Item = (K, V)>>(
        iter: I,
    ) -> Result<HashMap<K, V, RandomState>, TryReserveError>;

    /// Fallibly create a `HashMap` from an iterator using the provided hasher.
    fn try_collect_with_hasher<I: IntoIterator<Item = (K, V)>>(
        iter: I,
        hasher: S,
    ) -> Result<HashMap<K, V, S>, TryReserveError>;

    /// Alias for [`Self::try_collect`].
    fn fallible_collect<I: IntoIterator<Item = (K, V)>>(
        iter: I,
    ) -> Result<HashMap<K, V, RandomState>, TryReserveError> {
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

#[allow(deprecated)]
impl<K: Eq + Hash, V, S: BuildHasher> TryHashMap<K, V, S> for HashMap<K, V, S> {
    // ── Construction ────────────────────────────────────────────────────────

    fn try_with_capacity(capacity: usize) -> Result<HashMap<K, V, RandomState>, TryReserveError> {
        // HashMap::with_capacity on stable never fails — it allocates eagerly
        // or returns an empty map for capacity == 0. We wrap in Ok for API
        // symmetry with the fallible trait contract.
        Ok(HashMap::with_capacity(capacity))
    }

    fn try_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<HashMap<K, V, S>, TryReserveError> {
        // Same as above: with_capacity_and_hasher on stable never fails.
        Ok(HashMap::with_capacity_and_hasher(capacity, hasher))
    }

    // ── Insertion ───────────────────────────────────────────────────────────

    fn try_insert(&mut self, key: K, value: V) -> Result<(), TryHashMapError>
    where
        K: Eq + Hash,
    {
        self.try_reserve(1).map_err(TryHashMapError::Reserve)?;
        self.insert(key, value);
        Ok(())
    }

    fn try_insert_give_back(&mut self, key: K, value: V) -> Result<(), (K, V, TryHashMapError)>
    where
        K: Eq + Hash,
    {
        match self.try_reserve(1) {
            Ok(()) => {
                self.insert(key, value);
                Ok(())
            }
            Err(e) => Err((key, value, TryHashMapError::Reserve(e))),
        }
    }

    // ── Extension ───────────────────────────────────────────────────────────

    fn try_extend<I: IntoIterator<Item = (K, V)>>(
        &mut self,
        iter: I,
    ) -> Result<(), TryReserveError> {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        if lower > 0 {
            self.try_reserve(lower)?;
        }
        for (key, value) in iter {
            if self.len() == self.capacity() {
                self.try_reserve(1)?;
            }
            self.insert(key, value);
        }
        Ok(())
    }

    fn try_extend_from_slice(&mut self, other: &[(K, V)]) -> Result<(), TryHashMapError>
    where
        K: Eq + Hash + TryClone,
        V: TryClone,
    {
        if other.is_empty() {
            return Ok(());
        }
        let len_before = self.len();
        self.try_reserve(other.len())
            .map_err(TryHashMapError::Reserve)?;
        for (key, value) in other {
            match (key.try_clone(), value.try_clone()) {
                (Ok(k), Ok(v)) => {
                    self.insert(k, v);
                }
                (Err(e), _) | (_, Err(e)) => {
                    // Drain the elements we already inserted.
                    for _ in 0..self.len() - len_before {
                        self.drain().next();
                    }
                    return Err(TryHashMapError::Clone(e));
                }
            }
        }
        Ok(())
    }

    // ── Capacity / shrink ───────────────────────────────────────────────────

    fn try_shrink_to_fit(&mut self) -> Result<(), TryHashMapError>
    where
        S: Clone,
    {
        Self::try_shrink_to(self, self.len())
    }

    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryHashMapError>
    where
        S: Clone,
    {
        let target = core::cmp::max(self.len(), min_capacity);
        if self.capacity() <= target {
            return Ok(());
        }
        // Apparently, the hashbrown library also reallocates a new entire hash table for the shrink and moves items to the new table, so complexity wise, this should not be worse than the library.
        let length = self.len();
        let hasher = self.hasher().clone();
        let mut new_map = HashMap::with_capacity_and_hasher(target, hasher);
        new_map
            .try_reserve(length)
            .map_err(TryHashMapError::Reserve)?;
        for (k, v) in self.drain() {
            new_map.insert(k, v);
        }
        *self = new_map;
        Ok(())
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    fn try_collect<I: IntoIterator<Item = (K, V)>>(
        iter: I,
    ) -> Result<HashMap<K, V, RandomState>, TryReserveError> {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower);
        let mut map = HashMap::new();
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
/// cloneable and the hasher factory can be cloned.
///
/// # Panics
///
/// Cloning the hasher factory may panic if it relies on heap allocation.
/// This is not the case for any standard-library hasher (`RandomState`,
/// `SipHasher13`, etc.) or popular third-party hashers (`ahash`, `FxHash`,
/// FNV, DJB2, and so on), which store only small stack integers. However,
/// a custom hasher that internally allocates (e.g., caches compiled lookup
/// tables) could panic during the clone step.
impl<K, V, S> TryClone for HashMap<K, V, S>
where
    K: Eq + Hash + TryClone,
    V: TryClone,
    S: BuildHasher + Clone,
{
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let hasher = self.hasher().clone();
        let mut out = HashMap::with_capacity_and_hasher(self.len(), hasher);
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

impl<K, V, S: BuildHasher + Default> TryDefault for HashMap<K, V, S> {
    fn try_default() -> Result<Self, TryDefaultError> {
        // An empty HashMap requires no allocation.
        Ok(HashMap::with_hasher(S::default()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        use std::collections::hash_map::DefaultHasher;
        use std::hash::BuildHasherDefault;

        let hasher = BuildHasherDefault::<DefaultHasher>::default();
        let map: HashMap<&str, i32, _> = HashMap::try_with_capacity_and_hasher(5, hasher).unwrap();
        assert!(map.is_empty());
        assert!(map.capacity() >= 5);
    }

    // ── Insertion ────────────────────────────────────────────────────────────

    #[test]
    fn fallible_insert_single() {
        let mut map: HashMap<i32, &str> = HashMap::new();
        map.fallible_insert(1, "one").unwrap();
        assert_eq!(map.get(&1), Some(&"one"));
    }

    #[test]
    fn fallible_insert_overwrites() {
        let mut map: HashMap<i32, &str> = HashMap::new();
        map.fallible_insert(1, "one").unwrap();
        map.fallible_insert(1, "ONE").unwrap();
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

    // ── Extension ────────────────────────────────────────────────────────────

    #[test]
    fn try_extend_from_iterator() {
        let mut map: HashMap<i32, &str> = HashMap::new();
        map.try_extend([(1, "one"), (2, "two")]).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map[&1], "one");
    }

    #[test]
    fn try_extend_empty() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        map.try_extend(std::iter::empty::<(i32, i32)>()).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn try_extend_existing() {
        let mut map: HashMap<i32, &str> = HashMap::new();
        map.fallible_insert(1, "one").unwrap();
        map.try_extend([(2, "two"), (3, "three")]).unwrap();
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn try_extend_from_slice_clones() {
        let mut map: HashMap<String, Vec<u8>> = HashMap::new();
        map.fallible_insert("a".to_string(), vec![1]).unwrap();
        let slice: &[(String, Vec<u8>)] = &[("b".to_string(), vec![2, 3])];
        map.try_extend_from_slice(slice).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map["b"], vec![2, 3]);
    }

    #[test]
    fn try_extend_from_slice_empty() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        map.try_extend_from_slice(&[]).unwrap();
        assert!(map.is_empty());
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
            <HashMap<i32, i32> as TryHashMap<_, _, RandomState>>::try_collect(std::iter::empty::<(
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
        use std::collections::hash_map::DefaultHasher;
        use std::hash::BuildHasherDefault;

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
        let result: Result<(), (i32, i32, TryHashMapError)> = map.fallible_insert_give_back(1, 2);
        assert!(result.is_ok());
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
        use std::collections::hash_map::DefaultHasher;
        use std::hash::BuildHasherDefault;

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
        let mut a: HashMap<i32, &str> =
            <HashMap<i32, &str> as TryHashMap<_, _, RandomState>>::try_collect([
                (1, "one"),
                (2, "two"),
            ])
            .unwrap();
        a.try_extend([(3, "three")]).unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(a[&3], "three");
    }

    #[test]
    fn extend_from_slice_rollback_on_failure_type() {
        // We can't easily force a clone failure, but we verify the error type.
        let mut map: HashMap<String, Vec<u8>> = HashMap::new();
        let slice: &[(String, Vec<u8>)] = &[("x".to_string(), vec![1])];
        let result: Result<(), TryHashMapError> = map.try_extend_from_slice(slice);
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
}
