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
//! use [`lang_std::panic::catch_unwind`] to intercept allocation panics from the
//! B-tree's internal node allocator. This means `K` and `V` must be
//! [`RefUnwindSafe`] (panic::RefUnwindSafe) for the fallible mutation methods.
//!
//! The trait also implements [`TryClone`](crate::try_clone::TryClone) and
//! [`TryDefault`](crate::try_default::TryDefault) for `BTreeMap<K, V>` when
//! `K` and `V` satisfy the respective bounds.

use crate::alloc::{AllocError, PayloadBox};
use crate::try_clone::TryCloneError;
use crate::try_fmt::{TryDebug, helpers::FormatterExt};
use lang_alloc::collections::BTreeMap;
use lang_alloc::collections::btree_map;
use lang_core::fmt;
#[cfg(not(feature = "btree-entry"))]
use lang_core::mem::ManuallyDrop;
use lang_core::panic::{AssertUnwindSafe, RefUnwindSafe};
#[cfg(not(feature = "btree-entry"))]
use lang_core::ptr;
use lang_std::panic::catch_unwind;

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
    /// An element clone failed during a method that requires [`crate::try_clone::TryClone`].
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

impl TryDebug for TryBTreeMapError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => f
                .try_debug_struct("TryBTreeMapError::Alloc")
                .field("0", e)
                .finish(),
            Self::AllocPanic(p) => f
                .try_debug_struct("TryBTreeMapError::AllocPanic")
                .field("0", p)
                .finish(),
            Self::Clone(e) => f
                .try_debug_struct("TryBTreeMapError::Clone")
                .field("0", e)
                .finish(),
            Self::Other(msg) => f
                .try_debug_struct("TryBTreeMapError::Other")
                .field("0", msg)
                .finish(),
        }
    }
}

/// A trait for fallible B-tree map operations.
///
/// Implemented for `BTreeMap<K, V>`. Mirrors the most commonly-used `BTreeMap`
/// methods that can fail due to allocation pressure, returning [`Result`] values
/// that propagate [`TryBTreeMapError`] on failure.
///
/// # Note
///
/// Because `BTreeMap::try_reserve` does not exist, mutation methods use
/// [`lang_alloc::panic::catch_unwind`] internally to intercept OOM panics.
/// This is only possible due to the BTreeMap itself being UnwindSafe. Keys and
/// values must be [`RefUnwindSafe`] for these methods.
///
/// # Note on `try_insert`
///
/// The inherent [`BTreeMap::try_insert`](lang_alloc::collections::BTreeMap::try_insert) on
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
    /// **Note:** This method name conflicts with the inherent
    /// [`BTreeMap::try_insert`] (nightly). Prefer [`Self::fallible_insert`] instead.
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
    /// [`Entry`]: lang_alloc::collections::btree_map::Entry
    /// [`Entry::or_insert`]: lang_alloc::collections::btree_map::Entry::or_insert
    /// [`Entry::and_modify`]: lang_alloc::collections::btree_map::Entry::and_modify
    fn try_entry<'a>(&'a mut self, key: K) -> Result<btree_map::Entry<'a, K, V>, TryBTreeMapError>
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
    /// This method replaces the [`Self::try_insert`] which shares its
    /// name with the inherent [`BTreeMap::try_insert`].
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
    ) -> Result<btree_map::Entry<'a, K, V>, TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        Self::try_entry(self, key)
    }

    // ── Extension ───────────────────────────────────────────────────────────

    /// Fallibly extend the map with all key-value pairs from an iterator source.
    ///
    /// Accepts anything that implements [`ResumableSource`](crate::recovery::ResumableSource).
    /// Uses [`Self::try_insert_give_back`] so that on allocation failure the
    /// consumed-but-uncommitted pair is returned in a [`Resumable`](crate::recovery::Resumable).
    ///
    /// Note: elements already inserted before the failure are not rolled back.
    fn try_extend<S>(
        &mut self,
        source: S,
    ) -> Result<(), (TryBTreeMapError, crate::recovery::Resumable<S::Inner>)>
    where
        S: crate::recovery::ResumableSource<Item = (K, V)>,
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
    fn fallible_extend<S>(
        &mut self,
        source: S,
    ) -> Result<(), (TryBTreeMapError, crate::recovery::Resumable<S::Inner>)>
    where
        S: crate::recovery::ResumableSource<Item = (K, V)>,
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        Self::try_extend(self, source)
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

impl<K: Ord + RefUnwindSafe, V: RefUnwindSafe> TryBTreeMap<K, V> for BTreeMap<K, V> {
    // ── Construction ────────────────────────────────────────────────────────

    fn try_new() -> BTreeMap<K, V> {
        BTreeMap::new()
    }

    // ── Insertion ───────────────────────────────────────────────────────────

    #[cfg(feature = "btree-entry")]
    fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        use crate::alloc::btrees::entry::VacantEntryExt;
        match self.entry(key) {
            lang_alloc::collections::btree_map::Entry::Occupied(mut occ) => {
                let old_val = core::mem::replace(occ.get_mut(), value);
                Ok(Some(old_val))
            }
            lang_alloc::collections::btree_map::Entry::Vacant(vac) => {
                match vac.try_insert(value) {
                    Ok(_) => Ok(None),
                    Err((_, _, e)) => Err(TryBTreeMapError::Alloc(*e.alloc_error())),
                }
            }
        }
    }

    #[cfg(not(feature = "btree-entry"))]
    fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        let result: Option<V> = catch_unwind(AssertUnwindSafe(|| self.insert(key, value)))
            .map_err(|payload| TryBTreeMapError::AllocPanic(PayloadBox(payload)))?;
        Ok(result)
    }

    #[cfg(feature = "btree-entry")]
    fn try_insert_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<Option<V>, (K, V, TryBTreeMapError)>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        use crate::alloc::btrees::entry::VacantEntryExt;
        // Use the entry API: if occupied, replace and return old value.
        // If vacant, fallible-insert. On alloc failure, return (key, value, err).
        match self.entry(key) {
            lang_alloc::collections::btree_map::Entry::Occupied(mut occ) => {
                // Replace: take old value out, insert new one.
                let old = occ.get_mut();
                let old_val = core::mem::replace(old, value);
                Ok(Some(old_val))
            }
            lang_alloc::collections::btree_map::Entry::Vacant(vac) => {
                match vac.try_insert(value) {
                    Ok(_) => Ok(None),
                    Err((k, v, e)) => Err((k, v, TryBTreeMapError::Alloc(*e.alloc_error()))),
                }
            }
        }
    }

    #[cfg(not(feature = "btree-entry"))]
    fn try_insert_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<Option<V>, (K, V, TryBTreeMapError)>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        let md_map: &mut BTreeMap<ManuallyDrop<K>, ManuallyDrop<V>> = unsafe {
            &mut *(self as *mut BTreeMap<K, V> as *mut BTreeMap<ManuallyDrop<K>, ManuallyDrop<V>>)
        };
        let md_value = ManuallyDrop::new(value);
        let md_key = ManuallyDrop::new(key);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let k = unsafe { ptr::read(&md_key) };
            let v = unsafe { ptr::read(&md_value) };
            md_map.insert(k, v)
        }));
        match result {
            Ok(maybe_old_md_value) => {
                if let Some(old_md_value) = maybe_old_md_value {
                    let old_v = ManuallyDrop::into_inner(old_md_value);
                    let _new_k = ManuallyDrop::into_inner(md_key);
                    return Ok(Some(old_v));
                }
                Ok(None)
            }
            Err(payload) => {
                let key = ManuallyDrop::into_inner(md_key);
                let value = ManuallyDrop::into_inner(md_value);
                Err((key, value, TryBTreeMapError::AllocPanic(PayloadBox(payload))))
            }
        }
    }

    #[cfg(feature = "btree-entry")]
    fn try_insert_unique(&mut self, key: K, value: V) -> Result<(), (K, V, TryBTreeMapError)>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        use crate::alloc::btrees::entry::VacantEntryExt;
        if self.contains_key(&key) {
            return Err((key, value, TryBTreeMapError::Other("key already exists")));
        }
        match self.entry(key) {
            lang_alloc::collections::btree_map::Entry::Vacant(vac) => {
                match vac.try_insert(value) {
                    Ok(_) => Ok(()),
                    Err((k, v, e)) => Err((k, v, TryBTreeMapError::Alloc(*e.alloc_error()))),
                }
            }
            lang_alloc::collections::btree_map::Entry::Occupied(_) => {
                unreachable!("contains_key check above guarantees vacancy")
            }
        }
    }

    #[cfg(not(feature = "btree-entry"))]
    fn try_insert_unique(&mut self, key: K, value: V) -> Result<(), (K, V, TryBTreeMapError)>
    where
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        if self.contains_key(&key) {
            return Err((key, value, TryBTreeMapError::Other("key already exists")));
        }
        let md_map: &mut BTreeMap<ManuallyDrop<K>, ManuallyDrop<V>> = unsafe {
            &mut *(self as *mut BTreeMap<K, V> as *mut BTreeMap<ManuallyDrop<K>, ManuallyDrop<V>>)
        };
        let md_value = ManuallyDrop::new(value);
        let md_key = ManuallyDrop::new(key);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let k = unsafe { ptr::read(&md_key) };
            let v = unsafe { ptr::read(&md_value) };
            md_map.insert(k, v)
        }));
        match result {
            Ok(_old) => Ok(()),
            Err(payload) => {
                let key = ManuallyDrop::into_inner(md_key);
                let value = ManuallyDrop::into_inner(md_value);
                Err((key, value, TryBTreeMapError::AllocPanic(PayloadBox(payload))))
            }
        }
    }


    fn try_entry<'a>(&'a mut self, key: K) -> Result<btree_map::Entry<'a, K, V>, TryBTreeMapError>
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

    fn try_extend<S>(
        &mut self,
        source: S,
    ) -> Result<(), (TryBTreeMapError, crate::recovery::Resumable<S::Inner>)>
    where
        S: crate::recovery::ResumableSource<Item = (K, V)>,
        K: Ord + RefUnwindSafe,
        V: RefUnwindSafe,
    {
        use crate::recovery::Resumable;

        let (head, mut iter) = source.safe_into_iter();

        if let Some(pair) = head
            && let Err((k, v, e)) = Self::try_insert_give_back(self, pair.0, pair.1)
        {
            return Err((e, Resumable::new((k, v), iter)));
        }

        while let Some(pair) = iter.next() {
            match Self::try_insert_give_back(self, pair.0, pair.1) {
                Ok(_) => {}
                Err((k, v, e)) => {
                    return Err((e, Resumable::new((k, v), iter)));
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "btree-entry")]
    fn try_extend_from_slice(&mut self, other: &[(K, V)]) -> Result<(), TryBTreeMapError>
    where
        K: Ord + RefUnwindSafe + crate::try_clone::TryClone,
        V: RefUnwindSafe + crate::try_clone::TryClone,
    {
        use crate::alloc::btrees::entry::VacantEntryExt;
        if other.is_empty() {
            return Ok(());
        }
        let len_before = self.len();
        for (key, value) in other {
            match (key.try_clone(), value.try_clone()) {
                (Ok(k), Ok(v)) => {
                    match self.entry(k) {
                        lang_alloc::collections::btree_map::Entry::Occupied(mut occ) => {
                            *occ.get_mut() = v;
                        }
                        lang_alloc::collections::btree_map::Entry::Vacant(vac) => {
                            if let Err((_, _, e)) = vac.try_insert(v) {
                                // Alloc failure: rollback and propagate.
                                for _ in 0..self.len() - len_before {
                                    self.pop_first();
                                }
                                return Err(TryBTreeMapError::Alloc(*e.alloc_error()));
                            }
                        }
                    }
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

    #[cfg(not(feature = "btree-entry"))]
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

/// Implements [`crate::try_clone::TryClone`] for `BTreeMap<K, V>` when keys and values are
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
    use lang_alloc::boxed::Box;
    use lang_alloc::format;
    use lang_alloc::string::String;
    use lang_alloc::string::ToString;
    use lang_alloc::vec;
    use lang_alloc::vec::Vec;
    use lang_core::any::Any;
    use lang_core::cmp;
    use lang_core::iter;
    #[cfg(feature = "std")]
    use rustyfill_test_allocator::{FailPolicy, with_policy};

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
            fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for DroppedKey {
            fn cmp(&self, other: &Self) -> cmp::Ordering {
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
        assert_eq!(*old_val.0, [1u8, 2, 3]);
        // old_val drops here, freeing its Box<Vec<u8>>.

        // Verify the current entry has the new value.
        let cur_key = map.keys().next().unwrap();
        assert_eq!(*cur_key.0, 42);
        let cur_val = map.values().next().unwrap();
        assert_eq!(*cur_val.0, [4u8, 5, 6]);

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
            fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for TrackedKey {
            fn cmp(&self, other: &Self) -> cmp::Ordering {
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
        map.try_extend(iter::empty::<(i32, i32)>()).unwrap();
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
        let map: BTreeMap<i32, i32> =
            <BTreeMap<i32, i32> as TryBTreeMap<_, _>>::try_collect(iter::empty::<(i32, i32)>())
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
        assert_eq!(c["nested"].len(), 2);
        assert_eq!(c["nested"][0], [1u8, 2]);
        assert_eq!(c["nested"][1], [3u8]);
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
        let payload: Box<dyn Any + Send> = Box::new("out of memory");
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

    // ── Additional edge cases ────────────────────────────────────────────────

    #[test]
    fn fallible_insert_reverse_order() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new();
        for i in (0..15).rev() {
            map.fallible_insert(i, i * 100).unwrap();
        }
        assert_eq!(map.len(), 15);
        for i in 0..15 {
            assert_eq!(map[&i], i * 100);
        }
    }

    #[test]
    fn fallible_insert_large_values() {
        let big: String = "x".repeat(1024);
        let mut map: BTreeMap<i32, String> = BTreeMap::new();
        map.fallible_insert(1, big.clone()).unwrap();
        assert_eq!(map[&1].len(), 1024);
        map.fallible_insert(2, big.clone()).unwrap();
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn try_collect_deduplicates_keys() {
        // When duplicate keys appear in the iterator, later values win.
        let pairs = vec![(1, "first"), (2, "two"), (1, "second")];
        let map: BTreeMap<i32, &str> =
            <BTreeMap<i32, &str> as TryBTreeMap<_, _>>::try_collect(pairs).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map[&1], "second");
        assert_eq!(map[&2], "two");
    }

    #[test]
    fn try_extend_overwrites_existing_keys() {
        let mut map: BTreeMap<i32, String> = BTreeMap::new();
        map.fallible_insert(1, "original".to_string()).unwrap();
        map.try_extend([(1, "overwritten".to_string())]).unwrap();
        assert_eq!(map[&1], "overwritten");
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn try_extend_from_slice_overwrites_and_clones() {
        let mut map: BTreeMap<String, String> = BTreeMap::new();
        map.fallible_insert("a".to_string(), "one".to_string()).unwrap();
        let slice: &[(String, String)] = &[("a".to_string(), "updated".to_string())];
        map.try_extend_from_slice(slice).unwrap();
        assert_eq!(map["a"], "updated");
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn try_insert_unique_after_many_entries() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new();
        for i in 0..20 {
            map.fallible_insert(i, i).unwrap();
        }
        map.try_insert_unique(21, 21).unwrap();
        assert_eq!(map.len(), 21);
        let result = map.try_insert_unique(10, 999);
        assert!(result.is_err());
        let (k, v, err) = result.unwrap_err();
        assert_eq!(k, 10);
        assert_eq!(v, 999);
        matches!(err, TryBTreeMapError::Other(_));
    }

    #[test]
    fn try_clone_then_mutate_independent() {
        let mut orig: BTreeMap<i32, String> = BTreeMap::new();
        orig.insert(1, "hello".to_string());
        let c = orig.try_clone().unwrap();
        orig.insert(2, "world".to_string());
        assert_eq!(orig.len(), 2);
        assert_eq!(c.len(), 1);
        assert_eq!(c[&1], "hello");
    }

    #[test]
    fn try_insert_negative_keys() {
        let mut map: BTreeMap<i64, &str> = BTreeMap::new();
        map.fallible_insert(-5, "neg five").unwrap();
        map.fallible_insert(-1, "neg one").unwrap();
        map.fallible_insert(0, "zero").unwrap();
        map.fallible_insert(1, "pos one").unwrap();
        assert_eq!(map[&-5], "neg five");
        assert_eq!(map[&-1], "neg one");
        assert_eq!(map[&0], "zero");
        assert_eq!(map[&1], "pos one");
    }

    #[test]
    fn try_insert_entry_api_chain() {
        let mut map: BTreeMap<String, i32> = BTreeMap::new();
        // Chain entry API operations
        map.try_entry("a".to_string()).unwrap().or_insert(1);
        map.try_entry("b".to_string()).unwrap().or_insert(2);
        map.try_entry("a".to_string())
            .unwrap()
            .and_modify(|v| *v += 10);
        assert_eq!(map["a"], 11);
        assert_eq!(map["b"], 2);
    }

    #[test]
    fn try_extend_chained_operations() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new();
        map.try_extend([(1, 10), (2, 20)]).unwrap();
        map.try_extend([(3, 30), (4, 40)]).unwrap();
        assert_eq!(map.len(), 4);
        assert_eq!(map[&1], 10);
        assert_eq!(map[&4], 40);
    }

    #[test]
    fn try_collect_strings_ordered_iteration() {
        let pairs = vec![
            ("zebra".to_string(), 1),
            ("apple".to_string(), 2),
            ("mango".to_string(), 3),
        ];
        let map: BTreeMap<String, i32> =
            <BTreeMap<String, i32> as TryBTreeMap<_, _>>::try_collect(pairs).unwrap();
        let keys: Vec<&String> = map.keys().collect();
        assert_eq!(keys, &[&"apple".to_string(), &"mango".to_string(), &"zebra".to_string()]);
    }

    #[test]
    fn try_insert_give_back_with_boxed_key_value() {
        use lang_alloc::vec::Vec;
        let mut map: BTreeMap<Box<i32>, Box<Vec<u8>>> = BTreeMap::new();
        map.fallible_insert_give_back(Box::new(1), Box::new(vec![1, 2]))
            .unwrap();
        assert_eq!(*map[&Box::new(1)], vec![1, 2]);
    }

    #[test]
    fn try_default_then_extend() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::try_default().unwrap();
        map.try_extend([(5, 50), (10, 100)]).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map[&5], 50);
    }

    // ── Error formatting additional tests ────────────────────────────────────

    #[test]
    fn error_display_alloc_variant() {
        let l = unsafe { lang_core::alloc::Layout::from_size_align_unchecked(1, 1) };
        let err = TryBTreeMapError::Alloc(AllocError { layout: l });
        let msg = format!("{}", err);
        assert!(msg.contains("allocation"), "error message: {}", msg);
    }

    #[test]
    fn error_from_alloc_error_conversion() {
        let l = unsafe { lang_core::alloc::Layout::from_size_align_unchecked(1, 1) };
        let ae = AllocError { layout: l };
        let err: TryBTreeMapError = ae.into();
        matches!(err, TryBTreeMapError::Alloc(_));
    }

    // ── Stress test ──────────────────────────────────────────────────────────

    #[test]
    fn fallible_insert_many_entries_stress() {
        let mut map: BTreeMap<usize, usize> = BTreeMap::new();
        for i in 0..200 {
            map.fallible_insert(i, i.wrapping_mul(7)).unwrap();
        }
        assert_eq!(map.len(), 200);
        for i in 0..200 {
            assert_eq!(map[&i], i.wrapping_mul(7));
        }
    }

    #[test]
    fn fallible_insert_out_of_order_then_verify_sorted_keys() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new();
        let keys = [15, 3, 22, 1, 10, 8, 20, 5, 12, 2, 18, 7, 25, 0, 11, 6, 14, 9, 16, 4, 13, 19, 21, 23, 24];
        for k in &keys {
            map.fallible_insert(*k, k * 100).unwrap();
        }
        let collected: Vec<i32> = map.keys().copied().collect();
        let mut sorted = keys.to_vec();
        sorted.sort();
        assert_eq!(collected.as_slice(), &sorted);
    }

    // ── Explicit rollback tests (mid-operation clone failure) ───────────────

    #[test]
    #[cfg(feature = "std")]
    fn extend_from_slice_rollback_on_mid_way_clone_failure() {
        // try_extend_from_slice on BTreeMap<String, String> clones each pair
        // and inserts it. A mid-way clone failure must pop all elements
        // already inserted during this call via pop_first().
        let source: Vec<(String, String)> = vec![
            ("key0".into(), "val0".into()), ("key1".into(), "val1".into()),
            ("key2".into(), "val2".into()), ("key3".into(), "val3".into()),
            ("key4".into(), "val4".into()), ("key5".into(), "val5".into()),
            ("key6".into(), "val6".into()), ("key7".into(), "val7".into()),
            ("key8".into(), "val8".into()), ("key9".into(), "val9".into()),
        ];
        let len_source = source.len();

        let mut map: BTreeMap<String, String> = BTreeMap::from([
            ("pre_k0".into(), "pre_v0".into()),
            ("pre_k1".into(), "pre_v1".into()),
            ("pre_k2".into(), "pre_v2".into()),
        ]);
        let len_before = map.len();

        let r: Result<(), TryBTreeMapError> =
            with_policy(FailPolicy::fail_nth_alloc(2), || {
                <BTreeMap<String, String> as TryBTreeMap<String, String>>::try_extend_from_slice(&mut map, &source)
            });

        match r {
            Err(TryBTreeMapError::Clone(_)) => {
                assert_eq!(
                    map.len(),
                    len_before,
                    "pop_first rollback did not restore length: expected {}, got {}",
                    len_before,
                    map.len()
                );
                // Pre-existing entries must be intact.
                assert_eq!(map["pre_k0"], "pre_v0");
                assert_eq!(map["pre_k1"], "pre_v1");
                assert_eq!(map["pre_k2"], "pre_v2");
                // No source keys should appear.
                for (sk, _) in &source {
                    assert!(
                        !map.contains_key(sk.as_str()),
                        "source key found in map after rollback"
                    );
                }
            }
            Ok(()) => {
                assert_eq!(map.len(), len_before + len_source);
            }
            Err(other) => {
                panic!("unexpected error variant: {:?}", other);
            }
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn extend_from_slice_rollback_empty_start() {
        let source: Vec<(String, String)> = vec![
            ("k0".into(), "v0".into()), ("k1".into(), "v1".into()),
            ("k2".into(), "v2".into()), ("k3".into(), "v3".into()),
            ("k4".into(), "v4".into()), ("k5".into(), "v5".into()),
            ("k6".into(), "v6".into()), ("k7".into(), "v7".into()),
            ("k8".into(), "v8".into()), ("k9".into(), "v9".into()),
        ];

        let mut map: BTreeMap<String, String> = BTreeMap::new();

        let _: Result<(), TryBTreeMapError> =
            with_policy(FailPolicy::fail_nth_alloc(3), || {
                <BTreeMap<String, String> as TryBTreeMap<String, String>>::try_extend_from_slice(&mut map, &source)
            });

        assert!(
            map.is_empty(),
            "map should be empty after full rollback, but has {} entries",
            map.len()
        );
    }
}
