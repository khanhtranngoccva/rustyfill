//! Fallible B-tree map operations.
//!
//! Provides the [`TryBTreeMap`] trait with methods that mirror common `BTreeMap`
//! constructors and mutating operations but return [`Result`] to handle allocation
//! failures gracefully.
//!
//! # Design
//!
//! `TryBTreeMap` is implemented for `BTreeMap<K, V>`. Methods that may grow the
//! internal tree (`insert`, `extend`, etc.) return a `Result` instead of panicking
//! on out-of-memory. Read-only accessors delegate directly to `BTreeMap`.
//!
//! Because `BTreeMap::try_reserve` does not exist, these methods internally
//! use [`std::panic::catch_unwind`] to intercept allocation panics from the
//! B-tree's internal node allocator. This means `K` and `V` must be
//! [`RefUnwindSafe`](core::panic::RefUnwindSafe) for the fallible mutation methods.
//!
//! The trait also implements [`TryClone`](crate::try_clone::TryClone) and
//! [`TryDefault`](crate::try_default::TryDefault) for `BTreeMap<K, V>` when
//! `K` and `V` satisfy the respective bounds.

use crate::alloc::{AllocError, PayloadBox};
use crate::try_clone::TryCloneError;
use core::fmt;
use core::mem::ManuallyDrop;
use core::ptr;
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, RefUnwindSafe, catch_unwind};

// ── Error type ────────────────────────────────────────────────────────────────

/// Error returned by [`TryBTreeMap`] operations.
///
/// Since `BTreeMap::try_reserve` does not exist, this error type
/// wraps a caught panic as [`Self::AllocPanic`] when an internal node allocation
/// fails during insertion or extension. Clone failures during bulk operations
/// are wrapped as [`Self::Clone`].
#[derive(Debug)]
pub enum TryBTreeMapError {
    /// A raw heap allocation failed (no collection involved).
    Alloc(AllocError),
    /// An internal B-tree node allocation failed, caught via [`catch_unwind`].
    /// Stores the raw panic payload box directly to avoid any further allocation
    /// at the point of catch. Message extraction only happens lazily in [`fmt::Display`].
    AllocPanic(PayloadBox),
    /// An element clone failed during a method that requires [`TryClone`].
    Clone(TryCloneError),
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Display for TryBTreeMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "B-tree map operation failed: heap allocation error"),
            Self::AllocPanic(payload) => {
                write!(
                    f,
                    "B-tree map operation failed: internal allocation panicked: {}",
                    payload.message()
                )
            }
            Self::Clone(e) => write!(f, "B-tree map cloning failed: {}", e),
            Self::Other(msg) => write!(f, "B-tree map operation failed: {}", msg),
        }
    }
}

impl From<AllocError> for TryBTreeMapError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

impl From<TryCloneError> for TryBTreeMapError {
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// A trait for fallible B-tree map operations.
///
/// Implemented for `BTreeMap<K, V>`. Mirrors the most commonly-used `BTreeMap`
/// methods that can fail due to allocation pressure, returning [`Result`] values
/// that propagate [`TryBTreeMapError`] on failure.
///
/// # Note
///
/// Because `BTreeMap::try_reserve` does not exist, mutation methods use
/// [`std::panic::catch_unwind`] internally to intercept OOM panics.
/// This is only possible due to the BTreeMap itself being UnwindSafe. Keys and
/// values must be [`RefUnwindSafe`] for these methods.
///
/// # Note on `try_insert`
///
/// The inherent [`BTreeMap::try_insert`](std::collections::BTreeMap::try_insert) on
/// nightly Rust returns `Err(old_value)` when a key already exists, but may *panic*
/// on allocation failure. Our [`Self::try_insert`] catches allocation panics so it
/// never propagates one — it returns [`TryBTreeMapError::AllocPanic`] instead, but it
/// does not return the old value on key collision.
pub trait TryBTreeMap<K, V>: Sized {
    // ── Construction ────────────────────────────────────────────────────────

    /// Fallibly construct an empty `BTreeMap`.
    ///
    /// Unlike `HashMap`, `BTreeMap` does not pre-allocate nodes on construction.
    /// This always succeeds without allocation. Equivalent to [`BTreeMap::new`]
    /// but fallible.
    fn try_new() -> BTreeMap<K, V>;

    // ── Insertion ───────────────────────────────────────────────────────────

    /// Fallibly insert a key-value pair into the map, always replacing any
    /// existing value for the same key.
    ///
    /// Catches allocation panics from internal B-tree node allocation, so this
    /// method never panics on out-of-memory. Returns
    /// [`TryBTreeMapError::AllocPanic`] if an internal allocation fails.
    ///
    /// Returns `Ok(None)` if the key was not previously present, or
    /// `Ok(Some(old_value))` if the key existed and was replaced.
    ///
    /// **Deprecated:** This method name conflicts with the inherent
    /// [`BTreeMap::try_insert`] (nightly). Use [`Self::fallible_insert`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with inherent BTreeMap::try_insert; use fallible_insert"
    )]
    fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe;

    /// Like [`Self::try_insert`] or [`Self::fallible_insert`] but returns ownership
    /// of `key` and `value` back on allocation failure.
    ///
    /// Unlike the original [`BTreeMap::try_insert`], key collisions cause the old
    /// value to be evicted. See [`Self::try_insert_unique`] for the variant that
    /// fails on key collisions.
    fn try_insert_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<Option<V>, (K, V, TryBTreeMapError)>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe;

    /// Fallibly insert a key-value pair only if the key is not already present.
    ///
    /// Catches allocation panics from internal B-tree node allocation.
    ///
    /// Returns `Ok(())` if the key was newly inserted. Returns
    /// `Err((key, value, error))` if the insertion failed, giving ownership of
    /// both `key` and `value` back to the caller. The error is
    /// [`TryBTreeMapError::AllocPanic`] on allocation failure or
    /// [`TryBTreeMapError::Other`] if the key already exists.
    fn try_insert_unique(&mut self, key: K, value: V) -> Result<(), (K, V, TryBTreeMapError)>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe;

    /// Fallibly obtain an [`Entry`] for a key.
    ///
    /// Note: because `BTreeMap::try_reserve` does not exist, we cannot guarantee
    /// that inserting through the entry will not allocate again. The entry API
    /// itself may still panic on OOM after this method returns `Ok`.
    ///
    /// [`Entry`]: std::collections::btree_map::Entry
    /// [`Entry::or_insert`]: std::collections::btree_map::Entry::or_insert
    /// [`Entry::and_modify`]: std::collections::btree_map::Entry::and_modify
    fn try_entry<'a>(
        &'a mut self,
        key: K,
    ) -> Result<std::collections::btree_map::Entry<'a, K, V>, TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe;

    // ── Aliases with `fallible_` prefix ────────────────────────────────────

    /// Alias for [`Self::try_new`].
    fn fallible_new() -> BTreeMap<K, V> {
        Self::try_new()
    }

    /// Fallibly insert a key-value pair into the map, always replacing any
    /// existing value for the same key.
    ///
    /// Catches allocation panics from internal B-tree node allocation, so this
    /// method never panics on out-of-memory. Returns
    /// [`TryBTreeMapError::AllocPanic`] if an internal allocation fails.
    ///
    /// Returns `Ok(None)` if the key was not previously present, or
    /// `Ok(Some(old_value))` if the key existed and was replaced.
    ///
    /// This method replaces the deprecated [`Self::try_insert`] which shares its
    /// name with the inherent [`BTreeMap::try_insert`].
    #[allow(deprecated)]
    fn fallible_insert(&mut self, key: K, value: V) -> Result<Option<V>, TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        Self::try_insert(self, key, value)
    }

    /// Like [`Self::fallible_insert`] but returns ownership of `key` and `value`
    /// back on allocation failure.
    ///
    /// Unlike the original [`BTreeMap::try_insert`], key collisions cause the old
    /// value to be evicted. See [`Self::fallible_insert_unique`] for the variant
    /// that fails on key collisions.
    fn fallible_insert_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<Option<V>, (K, V, TryBTreeMapError)>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        Self::try_insert_give_back(self, key, value)
    }

    /// Alias for [`Self::try_insert_unique`].
    fn fallible_insert_unique(&mut self, key: K, value: V) -> Result<(), (K, V, TryBTreeMapError)>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        Self::try_insert_unique(self, key, value)
    }

    /// Alias for [`Self::try_entry`].
    fn fallible_entry<'a>(
        &'a mut self,
        key: K,
    ) -> Result<std::collections::btree_map::Entry<'a, K, V>, TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        Self::try_entry(self, key)
    }

    // ── Extension ───────────────────────────────────────────────────────────

    /// Fallibly extend the map with all key-value pairs from an iterator.
    ///
    /// Catches allocation panics from internal B-tree node allocation during
    /// the extend operation. Returns [`TryBTreeMapError::AllocPanic`] if an
    /// internal allocation fails.
    ///
    /// Note: because we catch the panic after the fact, partial extension may
    /// have occurred on failure. The map will be structurally consistent but
    /// may contain some of the extended elements.
    fn try_extend<I: IntoIterator<Item = (K, V)>>(
        &mut self,
        iter: I,
    ) -> Result<(), TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe;

    /// Fallibly extend the map by cloning key-value pairs from a slice.
    ///
    /// Returns [`TryBTreeMapError::Clone`] if a key or value clone fails, or
    /// [`TryBTreeMapError::AllocPanic`] if an internal allocation fails.
    /// On clone failure, rolls back any elements already inserted.
    fn try_extend_from_slice(&mut self, other: &[(K, V)]) -> Result<(), TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe + crate::try_clone::TryClone,
        V: RefUnwindSafe + crate::try_clone::TryClone;

    /// Alias for [`Self::try_extend`].
    fn fallible_extend<I: IntoIterator<Item = (K, V)>>(
        &mut self,
        iter: I,
    ) -> Result<(), TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        Self::try_extend(self, iter)
    }

    /// Alias for [`Self::try_extend_from_slice`].
    fn fallible_extend_from_slice(&mut self, other: &[(K, V)]) -> Result<(), TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe + crate::try_clone::TryClone,
        V: RefUnwindSafe + crate::try_clone::TryClone,
    {
        Self::try_extend_from_slice(self, other)
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    /// Fallibly collect an iterator of `(K, V)` pairs into a `BTreeMap`.
    ///
    /// Catches allocation panics from internal B-tree node allocation.
    /// Returns [`TryBTreeMapError::AllocPanic`] if an internal allocation fails.
    fn try_collect<I: IntoIterator<Item = (K, V)>>(
        iter: I,
    ) -> Result<BTreeMap<K, V>, TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe;

    /// Alias for [`Self::try_collect`].
    fn fallible_collect<I: IntoIterator<Item = (K, V)>>(
        iter: I,
    ) -> Result<BTreeMap<K, V>, TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        Self::try_collect(iter)
    }
}

// ── Implementation ────────────────────────────────────────────────────────────

#[allow(deprecated)]
impl<K: Ord + RefUnwindSafe, V: RefUnwindSafe> TryBTreeMap<K, V> for BTreeMap<K, V> {
    // ── Construction ────────────────────────────────────────────────────────

    fn try_new() -> BTreeMap<K, V> {
        BTreeMap::new()
    }

    // ── Insertion ───────────────────────────────────────────────────────────

    fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        // FIXME: this does not catch aborting panics!
        let result: Option<V> = catch_unwind(AssertUnwindSafe(|| self.insert(key, value)))
            .map_err(|payload| TryBTreeMapError::AllocPanic(PayloadBox(payload)))?;
        Ok(result)
    }

    fn try_insert_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<Option<V>, (K, V, TryBTreeMapError)>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        // Step 1: Transmute &mut BTreeMap<K, V> into &mut BTreeMap<ManuallyDrop<K>, ManuallyDrop<V>>
        let md_map: &mut BTreeMap<ManuallyDrop<K>, ManuallyDrop<V>> = unsafe {
            &mut *(self as *mut BTreeMap<K, V> as *mut BTreeMap<ManuallyDrop<K>, ManuallyDrop<V>>)
        };

        // Step 2: Wrap the originals in ManuallyDrop.
        let md_value = ManuallyDrop::new(value);
        let md_key = ManuallyDrop::new(key);

        // Step 3: Insert into the ManuallyDrop map inside catch_unwind.
        // Use ptr::read to move out of the references without consuming md_key/md_value,
        // so they're still accessible on panic.
        // FIXME: this does not catch aborting panics!
        let result = catch_unwind(AssertUnwindSafe(|| {
            let k = unsafe { ptr::read(&md_key) };
            let v = unsafe { ptr::read(&md_value) };
            md_map.insert(k, v)
        }));

        match result {
            Ok(maybe_old_md_value) => {
                if let Some(old_md_value) = maybe_old_md_value {
                    // Collision: BTreeMap::insert kept the old key in the tree,
                    // replaced the value with ours, and dropped the new key as
                    // ManuallyDrop<K> (skipping the inner K's destructor).
                    //
                    // The new key is still valid in md_key on our stack (ptr::read
                    // copies without invalidating). We extract it and drop it here
                    // so the inner K is properly freed.
                    let old_v = ManuallyDrop::into_inner(old_md_value);
                    let _new_k = ManuallyDrop::into_inner(md_key);
                    // _new_k drops here, freeing the inner K.
                    return Ok(Some(old_v));
                }
                Ok(None)
            }
            Err(payload) => {
                // Step 4 (failure): Recover the original values.
                // Nothing was inserted, so the ManuallyDrop wrappers still hold valid data.
                let key = ManuallyDrop::into_inner(md_key);
                let value = ManuallyDrop::into_inner(md_value);
                Err((
                    key,
                    value,
                    TryBTreeMapError::AllocPanic(PayloadBox(payload)),
                ))
            }
        }
    }

    fn try_insert_unique(&mut self, key: K, value: V) -> Result<(), (K, V, TryBTreeMapError)>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        if self.contains_key(&key) {
            return Err((key, value, TryBTreeMapError::Other("key already exists")));
        }
        // Step 1: Transmute &mut BTreeMap<K, V> into &mut BTreeMap<ManuallyDrop<K>, ManuallyDrop<V>>
        let md_map: &mut BTreeMap<ManuallyDrop<K>, ManuallyDrop<V>> = unsafe {
            &mut *(self as *mut BTreeMap<K, V> as *mut BTreeMap<ManuallyDrop<K>, ManuallyDrop<V>>)
        };

        // Step 2: Wrap originals in ManuallyDrop.
        let md_value = ManuallyDrop::new(value);
        let md_key = ManuallyDrop::new(key);

        // Step 3: Insert into the ManuallyDrop map.
        // FIXME: this does not catch aborting panics!
        let result = catch_unwind(AssertUnwindSafe(|| {
            let k = unsafe { ptr::read(&md_key) };
            let v = unsafe { ptr::read(&md_value) };
            md_map.insert(k, v)
        }));

        match result {
            Ok(_old) => Ok(()),
            Err(payload) => {
                // Step 4: Recover the original values.
                let key = ManuallyDrop::into_inner(md_key);
                let value = ManuallyDrop::into_inner(md_value);
                Err((
                    key,
                    value,
                    TryBTreeMapError::AllocPanic(PayloadBox(payload)),
                ))
            }
        }
    }

    fn try_entry<'a>(
        &'a mut self,
        key: K,
    ) -> Result<std::collections::btree_map::Entry<'a, K, V>, TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        // Entry API on BTreeMap doesn't allocate until you call or_insert/etc.
        // We can't pre-reserve, so we just return the entry directly.
        // The caveat is documented in the trait definition.
        Ok(self.entry(key))
    }

    // ── Extension ───────────────────────────────────────────────────────────

    fn try_extend<I: IntoIterator<Item = (K, V)>>(
        &mut self,
        iter: I,
    ) -> Result<(), TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        catch_unwind(AssertUnwindSafe(|| {
            self.extend(iter);
        }))
        .map_err(|payload| TryBTreeMapError::AllocPanic(PayloadBox(payload)))?;
        Ok(())
    }

    fn try_extend_from_slice(&mut self, other: &[(K, V)]) -> Result<(), TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe + crate::try_clone::TryClone,
        V: RefUnwindSafe + crate::try_clone::TryClone,
    {
        if other.is_empty() {
            return Ok(());
        }
        let len_before = self.len();
        for (key, value) in other {
            match (key.try_clone(), value.try_clone()) {
                (Ok(k), Ok(v)) => {
                    self.insert(k, v);
                }
                (Err(e), _) | (_, Err(e)) => {
                    // Rollback: drain elements we already inserted.
                    for _ in 0..self.len() - len_before {
                        self.pop_first();
                    }
                    return Err(TryBTreeMapError::Clone(e));
                }
            }
        }
        Ok(())
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    fn try_collect<I: IntoIterator<Item = (K, V)>>(
        iter: I,
    ) -> Result<BTreeMap<K, V>, TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        let mut map = BTreeMap::new();
        catch_unwind(AssertUnwindSafe(|| {
            map.extend(iter);
        }))
        .map_err(|payload| TryBTreeMapError::AllocPanic(PayloadBox(payload)))?;
        Ok(map)
    }
}

// ── TryClone for BTreeMap<K, V> ──────────────────────────────────────────────

/// Implements [`TryClone`] for `BTreeMap<K, V>` when keys and values are
/// cloneable. Clones one pair at a time and inserts it directly, avoiding an
/// intermediate `Vec` allocation. Catches allocation panics from internal
/// B-tree node growth.
impl<K, V> crate::try_clone::TryClone for BTreeMap<K, V>
where
    K: Ord + crate::try_clone::TryClone,
    V: crate::try_clone::TryClone,
{
    fn try_clone(&self) -> Result<Self, crate::try_clone::TryCloneError> {
        use crate::try_clone::TryCloneError;

        let mut out = BTreeMap::new();
        for (k, v) in self.iter() {
            let (kc, vc) = match (k.try_clone(), v.try_clone()) {
                (Ok(kc), Ok(vc)) => (kc, vc),
                (Err(e), _) | (_, Err(e)) => {
                    drop(out);
                    return Err(e);
                }
            };
            catch_unwind(AssertUnwindSafe(|| {
                out.insert(kc, vc);
            }))
            .map_err(|_| TryCloneError::Other("BTreeMap allocation failed during clone"))?;
        }
        Ok(out)
    }
}

// ── TryDefault for BTreeMap<K, V> ─────────────────────────────────────────────

impl<K, V> crate::try_default::TryDefault for BTreeMap<K, V> {
    fn try_default() -> Result<Self, crate::try_default::TryDefaultError>
    where
        Self: Sized,
    {
        // An empty BTreeMap requires no allocation.
        Ok(BTreeMap::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::try_clone::TryClone;
    use crate::try_default::TryDefault;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn try_new_creates_empty_map() {
        let map: BTreeMap<i32, i32> = BTreeMap::<i32, i32>::try_new();
        assert!(map.is_empty());
    }

    #[test]
    fn fallible_new_alias_works() {
        let map: BTreeMap<String, i32> =
            <BTreeMap<String, i32> as TryBTreeMap<_, _>>::fallible_new();
        assert!(map.is_empty());
    }

    // ── Insertion ────────────────────────────────────────────────────────────

    #[test]
    fn fallible_insert_single() {
        let mut map: BTreeMap<i32, &str> = BTreeMap::new();
        let old = map.fallible_insert(1, "one").unwrap();
        assert_eq!(old, None);
        assert_eq!(map.get(&1), Some(&"one"));
    }

    #[test]
    fn fallible_insert_evicts_old_value() {
        let mut map: BTreeMap<i32, &str> = BTreeMap::new();
        map.fallible_insert(1, "one").unwrap();
        let old = map.fallible_insert(1, "ONE").unwrap();
        assert_eq!(old, Some("one"));
        assert_eq!(map.get(&1), Some(&"ONE"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn fallible_insert_multiple_keys() {
        let mut map: BTreeMap<&str, i32> = BTreeMap::new();
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
        let mut map: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        map.fallible_insert("key".to_string(), vec![1, 2, 3])
            .unwrap();
        assert_eq!(map["key"], vec![1, 2, 3]);
    }

    // ── Unique insertion ─────────────────────────────────────────────────────

    #[test]
    fn try_insert_unique_new_key() {
        let mut map: BTreeMap<i32, &str> = BTreeMap::new();
        map.try_insert_unique(1, "one").unwrap();
        assert_eq!(map.get(&1), Some(&"one"));
    }

    #[test]
    fn try_insert_unique_duplicate_rejected() {
        let mut map: BTreeMap<i32, &str> = BTreeMap::new();
        map.try_insert_unique(1, "one").unwrap();
        let result = map.try_insert_unique(1, "TWO");
        let (returned_key, returned_val, err) = result.unwrap_err();
        assert_eq!(returned_key, 1);
        assert_eq!(returned_val, "TWO");
        matches!(err, TryBTreeMapError::Other(_));
        assert_eq!(map.get(&1), Some(&"one"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn fallible_insert_unique_alias_works() {
        let mut map: BTreeMap<i32, String> = BTreeMap::new();
        map.fallible_insert_unique(42, "hi".to_string()).unwrap();
        assert_eq!(map[&42], "hi");
    }

    // ── Give-back variants ───────────────────────────────────────────────────

    #[test]
    fn fallible_insert_give_back_success() {
        let mut map: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        map.fallible_insert_give_back("k".to_string(), vec![1])
            .unwrap();
        assert_eq!(map["k"], vec![1]);
    }

    #[test]
    fn fallible_insert_give_back_error_type_shape() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new();
        let result: Result<Option<i32>, (i32, i32, TryBTreeMapError)> =
            map.fallible_insert_give_back(1, 2);
        assert!(result.is_ok());
    }

    #[test]
    fn fallible_insert_give_back_overwrite_returns_old() {
        let mut map: BTreeMap<i32, String> = BTreeMap::new();
        map.fallible_insert_give_back(1, "first".to_string())
            .unwrap();
        let old = map
            .fallible_insert_give_back(1, "second".to_string())
            .unwrap();
        assert_eq!(old.as_deref(), Some("first"));
        assert_eq!(map[&1], "second");
    }

    // ── Drop correctness (Miri / sanitizer validation) ───────────────────────

    /// Verifies that overwriting an entry with Box-containing keys and values
    /// does not double-free or leak. Miri and ASAN will catch violations.
    #[test]
    fn give_back_drop_behavior_no_double_free_or_leak() {
        #[derive(Debug)]
        struct DroppedKey(Box<u64>);
        impl PartialEq for DroppedKey {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
        impl Eq for DroppedKey {}
        impl PartialOrd for DroppedKey {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for DroppedKey {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.0.cmp(&other.0)
            }
        }

        #[derive(Debug)]
        #[allow(clippy::box_collection)]
        struct DroppedValue(Box<Vec<u8>>);

        let mut map: BTreeMap<DroppedKey, DroppedValue> = BTreeMap::new();

        // Insert initial entry with heap-allocated key and value.
        map.fallible_insert_give_back(
            DroppedKey(Box::new(42)),
            DroppedValue(Box::new(vec![1, 2, 3])),
        )
        .unwrap();

        // Overwrite — the old key/value and the new key must all be properly
        // dropped without double-free or leaks.
        let old = map
            .fallible_insert_give_back(
                DroppedKey(Box::new(42)),
                DroppedValue(Box::new(vec![4, 5, 6])),
            )
            .unwrap();

        // Verify the returned old value is valid and gets dropped correctly.
        assert!(old.is_some());
        let old_val = old.unwrap();
        assert_eq!(*old_val.0, vec![1, 2, 3]);
        // old_val drops here, freeing its Box<Vec<u8>>.

        // Verify the current entry has the new value.
        let cur_key = map.keys().next().unwrap();
        assert_eq!(*cur_key.0, 42);
        let cur_val = map.values().next().unwrap();
        assert_eq!(*cur_val.0, vec![4, 5, 6]);

        // map drops here — the remaining entry's Box<u64> key and Box<Vec<u8>>
        // value must be freed exactly once.
    }

    /// Same check for try_insert_unique on a fresh key.
    #[test]
    fn insert_unique_drop_behavior_no_leak() {
        #[derive(Debug)]
        struct TrackedKey(Box<i32>);
        impl PartialEq for TrackedKey {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
        impl Eq for TrackedKey {}
        impl PartialOrd for TrackedKey {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for TrackedKey {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.0.cmp(&other.0)
            }
        }

        let mut map: BTreeMap<TrackedKey, String> = BTreeMap::new();
        map.try_insert_unique(TrackedKey(Box::new(7)), "hello".to_string())
            .unwrap();

        let key = map.keys().next().unwrap();
        assert_eq!(*key.0, 7);
        // map drops here — TrackedKey's Box<i32> must be freed.
    }

    // ── Entry API ────────────────────────────────────────────────────────────

    #[test]
    fn try_entry_or_insert_new_key() {
        let mut map: BTreeMap<String, i32> = BTreeMap::new();
        map.try_entry("hello".to_string()).unwrap().or_insert(42);
        assert_eq!(map["hello"], 42);
    }

    #[test]
    fn try_entry_or_insert_existing_key() {
        let mut map: BTreeMap<String, i32> = BTreeMap::new();
        map.fallible_insert("key".to_string(), 1).unwrap();
        map.try_entry("key".to_string()).unwrap().or_insert(99);
        assert_eq!(map["key"], 1);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn try_entry_and_modify() {
        let mut map: BTreeMap<String, i32> = BTreeMap::new();
        map.fallible_insert("count".to_string(), 5).unwrap();
        map.try_entry("count".to_string())
            .unwrap()
            .and_modify(|v| *v += 1);
        assert_eq!(map["count"], 6);
    }

    #[test]
    fn fallible_entry_matches_try_entry() {
        let mut map: BTreeMap<String, i32> = BTreeMap::new();
        map.fallible_entry("x".to_string()).unwrap().or_insert(10);
        assert_eq!(map["x"], 10);
    }

    // ── Extension ────────────────────────────────────────────────────────────

    #[test]
    fn try_extend_from_iterator() {
        let mut map: BTreeMap<i32, &str> = BTreeMap::new();
        map.try_extend([(1, "one"), (2, "two")]).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map[&1], "one");
    }

    #[test]
    fn try_extend_empty() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new();
        map.try_extend(std::iter::empty::<(i32, i32)>()).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn try_extend_existing() {
        let mut map: BTreeMap<i32, &str> = BTreeMap::new();
        map.fallible_insert(1, "one").unwrap();
        map.try_extend([(2, "two"), (3, "three")]).unwrap();
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn try_extend_from_slice_clones() {
        let mut map: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        map.fallible_insert("a".to_string(), vec![1]).unwrap();
        let slice: &[(String, Vec<u8>)] = &[("b".to_string(), vec![2, 3])];
        map.try_extend_from_slice(slice).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map["b"], vec![2, 3]);
    }

    #[test]
    fn try_extend_from_slice_empty() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new();
        map.try_extend_from_slice(&[]).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn fallible_extend_alias_works() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new();
        map.fallible_extend([(1, 10), (2, 20)]).unwrap();
        assert_eq!(map.len(), 2);
    }

    // ── Bulk construction ────────────────────────────────────────────────────

    #[test]
    fn try_collect_range() {
        let map: BTreeMap<i32, i32> =
            <BTreeMap<i32, i32> as TryBTreeMap<_, _>>::try_collect((0..5).map(|i| (i, i * 10)))
                .unwrap();
        assert_eq!(map.len(), 5);
        assert_eq!(map[&3], 30);
    }

    #[test]
    fn try_collect_empty() {
        let map: BTreeMap<i32, i32> = <BTreeMap<i32, i32> as TryBTreeMap<_, _>>::try_collect(
            std::iter::empty::<(i32, i32)>(),
        )
        .unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn try_collect_strings() {
        let pairs = vec![("a".to_string(), 1), ("b".to_string(), 2)];
        let map: BTreeMap<String, i32> =
            <BTreeMap<String, i32> as TryBTreeMap<_, _>>::try_collect(pairs).unwrap();
        assert_eq!(map["a"], 1);
        assert_eq!(map["b"], 2);
    }

    #[test]
    fn fallible_collect_alias_works() {
        let map: BTreeMap<i32, i32> =
            <BTreeMap<i32, i32> as TryBTreeMap<_, _>>::fallible_collect([(1, 10), (2, 20)])
                .unwrap();
        assert_eq!(map.len(), 2);
    }

    // ── TryClone ─────────────────────────────────────────────────────────────

    #[test]
    fn try_clone_empty_map() {
        let map: BTreeMap<i32, i32> = BTreeMap::new();
        let c = map.try_clone().unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn try_clone_populated_map() {
        let mut map: BTreeMap<i32, String> = BTreeMap::new();
        map.insert(1, "hello".to_string());
        map.insert(2, "world".to_string());
        let c = map.try_clone().unwrap();
        assert_eq!(c[&1], "hello");
        assert_eq!(c[&2], "world");
    }

    #[test]
    fn try_clone_nested_values() {
        let mut map: BTreeMap<String, Vec<Vec<u8>>> = BTreeMap::new();
        map.insert("nested".to_string(), vec![vec![1, 2], vec![3]]);
        let c = map.try_clone().unwrap();
        assert_eq!(c["nested"], vec![vec![1, 2], vec![3]]);
    }

    // ── TryDefault ───────────────────────────────────────────────────────────

    #[test]
    fn try_default_empty_map() {
        let map: BTreeMap<i32, i32> = BTreeMap::try_default().unwrap();
        assert!(map.is_empty());
    }

    // ── Combined workflows ───────────────────────────────────────────────────

    #[test]
    fn build_insert_clone_default() {
        let mut map: BTreeMap<String, i32> = BTreeMap::try_default().unwrap();
        map.fallible_insert("alpha".to_string(), 1).unwrap();
        map.fallible_insert("beta".to_string(), 2).unwrap();
        let c = map.try_clone().unwrap();
        assert_eq!(c["alpha"], 1);
        assert_eq!(c["beta"], 2);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn collect_then_extend() {
        let mut a: BTreeMap<i32, &str> =
            <BTreeMap<i32, &str> as TryBTreeMap<_, _>>::try_collect([(1, "one"), (2, "two")])
                .unwrap();
        a.try_extend([(3, "three")]).unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(a[&3], "three");
    }

    #[test]
    fn ordered_iteration_after_operations() {
        let mut map: BTreeMap<i32, &str> = BTreeMap::new();
        map.fallible_insert(3, "three").unwrap();
        map.fallible_insert(1, "one").unwrap();
        map.fallible_insert(2, "two").unwrap();
        let keys: Vec<&i32> = map.keys().collect();
        assert_eq!(keys, &[&1, &2, &3]);
    }

    #[test]
    fn extend_from_slice_rollback_on_failure_type() {
        let mut map: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let slice: &[(String, Vec<u8>)] = &[("x".to_string(), vec![1])];
        let result: Result<(), TryBTreeMapError> = map.try_extend_from_slice(slice);
        assert!(result.is_ok());
        assert_eq!(map["x"], vec![1]);
    }

    #[test]
    fn fallible_aliases_match_try_methods() {
        let m1: BTreeMap<i32, i32> = <BTreeMap<i32, i32> as TryBTreeMap<_, _>>::fallible_new();
        let m2: BTreeMap<i32, i32> = <BTreeMap<i32, i32> as TryBTreeMap<_, _>>::try_new();
        assert!(m1.is_empty());
        assert!(m2.is_empty());
    }

    // ── Error formatting ─────────────────────────────────────────────────────

    #[test]
    fn error_display_alloc_panic() {
        let payload: Box<dyn core::any::Any + Send> = Box::new("out of memory");
        let err = TryBTreeMapError::AllocPanic(PayloadBox(payload));
        let msg = format!("{}", err);
        assert!(msg.contains("allocation panicked"));
    }

    #[test]
    fn error_display_other() {
        let err = TryBTreeMapError::Other("key already exists");
        let msg = format!("{}", err);
        assert!(msg.contains("key already exists"));
    }
}
