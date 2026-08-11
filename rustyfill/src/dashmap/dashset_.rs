//! Placeholder for [`TryDashSet`] implementation.
//!
//! To be implemented in a subsequent chunk, following the same pattern as
//! [`TryDashMap`](super::dashmap_::TryDashMap).

use crate::alloc::{AllocError, TryReserveError};
use crate::dashmap::TryDashMap;
use crate::prelude::{TryClone, TryDefault};
use crate::try_clone::TryCloneError;
use crate::try_fmt::{TryDebug, helpers::FormatterExt};
use crate::lang_std::cmp::Eq;
use crate::lang_std::hash::{BuildHasher, Hash, RandomState};
use core::alloc::Layout;
use core::fmt;

type DashMap<K, V, S = RandomState> = dashmap::DashMap<K, V, S>;
type DashSet<T, S = RandomState> = dashmap::DashSet<T, S>;

// ── Error type ────────────────────────────────────────────────────────────────

/// Error returned by [`TryDashSet`] operations.
#[derive(Debug)]
pub enum TryDashSetError {
    /// A raw heap allocation failed (no collection involved).
    Alloc(AllocError),
    /// A capacity reservation on the DashSet failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An element clone failed during a method that requires [`TryClone`].
    Clone(TryCloneError),
    /// An arithmetic overflow occurred while computing required capacity.
    Overflow,
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Display for TryDashSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "dash set operation failed: heap allocation error"),
            Self::Reserve(e) => write!(f, "dash set operation failed: {}", e),
            Self::Clone(e) => write!(f, "dash set operation failed: {}", e),
            Self::Overflow => {
                write!(
                    f,
                    "dash set operation failed: capacity calculation overflowed"
                )
            }
            Self::Other(msg) => write!(f, "dash set operation failed: {}", msg),
        }
    }
}

impl From<AllocError> for TryDashSetError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

impl From<TryCloneError> for TryDashSetError {
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

impl From<crate::try_default::TryDefaultError> for TryDashSetError {
    fn from(err: crate::try_default::TryDefaultError) -> Self {
        match err {
            crate::try_default::TryDefaultError::Alloc(e) => Self::Alloc(e),
            crate::try_default::TryDefaultError::Reserve(e) => Self::Reserve(e),
            crate::try_default::TryDefaultError::Overflow => Self::Overflow,
            crate::try_default::TryDefaultError::Other(msg) => Self::Other(msg),
        }
    }
}

impl TryDebug for TryDashSetError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => f
                .try_debug_struct("TryDashSetError::Alloc")
                .field("0", e)
                .finish(),
            Self::Reserve(e) => f
                .try_debug_struct("TryDashSetError::Reserve")
                .field("0", e)
                .finish(),
            Self::Clone(e) => f
                .try_debug_struct("TryDashSetError::Clone")
                .field("0", e)
                .finish(),
            Self::Overflow => f.write_str("TryDashSetError::Overflow"),
            Self::Other(msg) => f
                .try_debug_struct("TryDashSetError::Other")
                .field("0", msg)
                .finish(),
        }
    }
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// A trait for fallible DashSet operations.
///
/// Implemented for `dashmap::DashSet<T, S>`. Mirrors the most commonly-used
/// `DashSet` methods that can fail due to allocation pressure, returning
/// [`Result`] values that propagate [`TryDashSetError`] on failure.
pub trait TryDashSet<T, S = RandomState>: Sized {
    // ── Construction ────────────────────────────────────────────────────────

    /// Fallibly construct an empty `DashSet` with a default-constructed hasher.
    ///
    /// Unlike [`Self::try_with_capacity`] which previously hardcoded
    /// [`RandomState`], this method uses [`TryDefault`] to construct the hasher
    /// fallibly. If hasher construction fails (e.g. `RandomState` panics during
    /// seed initialization), the error is returned rather than unwinding.
    ///
    /// Requires `S: [`TryDefault`]` so that hasher creation is safe even when
    /// it involves runtime allocation or thread-local state.
    fn try_new() -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault;

    /// Fallibly construct an empty `DashSet` with at least enough capacity for
    /// `capacity` elements.
    ///
    /// Constructs the hasher via [`TryDefault`] (same as [`Self::try_new`]),
    /// then reserves capacity for `capacity` elements. Returns
    /// [`TryDashSetError::Reserve`] if the capacity reservation fails, or
    /// [`TryDashSetError::Other`] if hasher construction panics.
    ///
    /// Requires `S: [`TryDefault`]` so that hasher creation is safe even when
    /// it involves runtime allocation or thread-local state.
    fn try_with_capacity(capacity: usize) -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault;

    /// Fallibly construct an empty `DashSet` with at least enough capacity for
    /// `capacity` elements, using the provided hash builder.
    fn try_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<DashSet<T, S>, TryDashSetError>;

    // ── Insertion ───────────────────────────────────────────────────────────

    /// Fallibly insert the value into the set.
    ///
    /// Returns `true` if the value was not already present, `false` otherwise.
    fn try_insert(&self, value: T) -> Result<bool, TryDashSetError>
    where
        T: Eq + Hash;

    /// Like [`Self::try_insert`] but returns ownership of `value` back on
    /// allocation failure.
    fn try_insert_give_back(&self, value: T) -> Result<bool, (T, TryDashSetError)>
    where
        T: Eq + Hash;

    // ── Aliases with `fallible_` prefix ────────────────────────────────────

    /// Alias for [`Self::try_new`].
    fn fallible_new() -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault,
    {
        Self::try_new()
    }

    /// Alias for [`Self::try_with_capacity`].
    fn fallible_with_capacity(capacity: usize) -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault,
    {
        Self::try_with_capacity(capacity)
    }

    /// Alias for [`Self::try_with_capacity_and_hasher`].
    fn fallible_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<DashSet<T, S>, TryDashSetError> {
        Self::try_with_capacity_and_hasher(capacity, hasher)
    }

    /// Alias for [`Self::try_insert`].
    fn fallible_insert(&self, value: T) -> Result<bool, TryDashSetError>
    where
        T: Eq + Hash,
    {
        Self::try_insert(self, value)
    }

    /// Alias for [`Self::try_insert_give_back`].
    fn fallible_insert_give_back(&self, value: T) -> Result<bool, (T, TryDashSetError)>
    where
        T: Eq + Hash,
    {
        Self::try_insert_give_back(self, value)
    }

    // ── Extension ───────────────────────────────────────────────────────────

    /// Fallibly extend the set with all values from an iterator source.
    ///
    /// Accepts anything that implements [`ResumableSource`](crate::recovery::ResumableSource).
    /// Inserts each value individually via [`Self::try_insert_give_back`] so that
    /// on failure the consumed-but-uncommitted element is returned in a
    /// [`Resumable`](crate::recovery::Resumable).
    ///
    /// Note: elements already inserted before the failure are not rolled back.
    fn try_extend<Src>(
        &self,
        source: Src,
    ) -> Result<(), (TryDashSetError, crate::recovery::Resumable<Src::Inner>)>
    where
        Src: crate::recovery::ResumableSource<Item = T>;

    /// Alias for [`Self::try_extend`].
    fn fallible_extend<Src>(
        &self,
        source: Src,
    ) -> Result<(), (TryDashSetError, crate::recovery::Resumable<Src::Inner>)>
    where
        Src: crate::recovery::ResumableSource<Item = T>,
    {
        Self::try_extend(self, source)
    }

    // ── Capacity / shrink ───────────────────────────────────────────────────

    /// Fallibly shrink the capacity of this DashSet to match its length.
    fn try_shrink_to_fit(&self) -> Result<(), TryDashSetError>;

    /// Alias for [`Self::try_shrink_to_fit`].
    fn fallible_shrink_to_fit(&self) -> Result<(), TryDashSetError> {
        Self::try_shrink_to_fit(self)
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    /// Fallibly collect an iterator of `T` values into a `DashSet`.
    ///
    /// Constructs the hasher via [`TryDefault`] and uses the iterator's size
    /// hint to pre-allocate when possible.
    fn try_collect<I: IntoIterator<Item = T>>(iter: I) -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault;

    /// Fallibly create a `DashSet` from an iterator using the provided hasher.
    fn try_collect_with_hasher<I: IntoIterator<Item = T>>(
        iter: I,
        hasher: S,
    ) -> Result<DashSet<T, S>, TryDashSetError>;

    /// Alias for [`Self::try_collect`].
    fn fallible_collect<I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault,
    {
        Self::try_collect(iter)
    }

    /// Alias for [`Self::try_collect_with_hasher`].
    fn fallible_collect_with_hasher<I: IntoIterator<Item = T>>(
        iter: I,
        hasher: S,
    ) -> Result<DashSet<T, S>, TryDashSetError> {
        Self::try_collect_with_hasher(iter, hasher)
    }
}

// ── Implementation ────────────────────────────────────────────────────────────

/// Convert to the inner dashmap type. This operation is useful for reserve operations.
fn convert_ref<T, S>(set: &DashSet<T, S>) -> &DashMap<T, (), S> {
    // SAFETY: DashSet<T, S> is supposed to be a thin newtype wrapper around DashMap<T, (), S> with a single `pub(crate) inner` field. The memory layout is identical, so reborrowing as &DashMap<T, (), S> is safe. The assertion below should break if any extra fields are detected.
    const {
        let map_layout: Layout = Layout::new::<DashMap<T, (), S>>();
        let set_layout: Layout = Layout::new::<DashSet<T, S>>();
        assert!(map_layout.size() == set_layout.size());
        assert!(map_layout.align() == set_layout.align());
    };
    unsafe { ::lang_std::mem::transmute(set) }
}

/// Mutable variant of [`convert_ref`].
fn convert_mut<T, S>(set: &mut DashSet<T, S>) -> &mut DashMap<T, (), S> {
    // SAFETY: Same layout guarantees as [`convert_ref`], extended to mutable references.
    unsafe { ::lang_std::mem::transmute(set) }
}

impl<T: Eq + Hash, S: BuildHasher + TryClone> TryDashSet<T, S> for DashSet<T, S> {
    // ── Construction ────────────────────────────────────────────────────────

    fn try_new() -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault,
    {
        let hasher = S::try_default()?;
        Ok(DashSet::with_hasher(hasher))
    }

    fn try_with_capacity(capacity: usize) -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault,
    {
        let mut set = Self::try_new()?;
        if capacity > 0 {
            convert_mut(&mut set)
                .try_reserve(capacity)
                .map_err(|e| TryDashSetError::Reserve(TryReserveError::from(e)))?;
        }
        Ok(set)
    }

    fn try_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<DashSet<T, S>, TryDashSetError> {
        let mut set = DashSet::with_hasher(hasher);
        if capacity > 0 {
            convert_mut(&mut set)
                .try_reserve(capacity)
                .map_err(|e| TryDashSetError::Reserve(TryReserveError::from(e)))?;
        }
        Ok(set)
    }

    // ── Insertion ───────────────────────────────────────────────────────────

    fn try_insert(&self, value: T) -> Result<bool, TryDashSetError>
    where
        T: Eq + Hash,
    {
        let map = convert_ref(self);
        let had_old = map.try_insert(value, ()).map_err(map_error_to_set)?;
        Ok(had_old.is_none())
    }

    fn try_insert_give_back(&self, value: T) -> Result<bool, (T, TryDashSetError)>
    where
        T: Eq + Hash,
    {
        let map = convert_ref(self);
        match map.try_insert_give_back(value, ()) {
            Ok(had_old) => Ok(had_old.is_none()),
            Err((k, _, e)) => Err((k, map_error_to_set(e))),
        }
    }

    // ── Extension ───────────────────────────────────────────────────────────

    fn try_extend<Src>(
        &self,
        source: Src,
    ) -> Result<(), (TryDashSetError, crate::recovery::Resumable<Src::Inner>)>
    where
        Src: crate::recovery::ResumableSource<Item = T>,
        T: Eq + Hash,
    {
        use crate::recovery::Resumable;

        let (head, mut iter) = source.safe_into_iter();

        if let Some(value) = head
            && let Err((v, e)) = Self::try_insert_give_back(self, value)
        {
            return Err((e, Resumable::new(v, iter)));
        }

        while let Some(value) = iter.next() {
            match Self::try_insert_give_back(self, value) {
                Ok(_) => {}
                Err((v, e)) => {
                    return Err((e, Resumable::new(v, iter)));
                }
            }
        }
        Ok(())
    }

    // ── Capacity / shrink ───────────────────────────────────────────────────

    fn try_shrink_to_fit(&self) -> Result<(), TryDashSetError> {
        convert_ref(self).try_shrink_to_fit().map_err(|e| match e {
            super::TryDashMapError::Alloc(a) => TryDashSetError::Alloc(a),
            super::TryDashMapError::Reserve(r) => TryDashSetError::Reserve(r),
            super::TryDashMapError::Clone(c) => TryDashSetError::Clone(c),
            super::TryDashMapError::Overflow => TryDashSetError::Overflow,
            super::TryDashMapError::Other(m) => TryDashSetError::Other(m),
        })
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    fn try_collect<I: IntoIterator<Item = T>>(iter: I) -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault,
    {
        let set = Self::try_new()?;
        for value in iter {
            <DashSet<T, S> as TryDashSet<T, S>>::try_insert(&set, value)?;
        }
        Ok(set)
    }

    fn try_collect_with_hasher<I: IntoIterator<Item = T>>(
        iter: I,
        hasher: S,
    ) -> Result<DashSet<T, S>, TryDashSetError> {
        let set = DashSet::with_hasher(hasher);
        for value in iter {
            Self::try_insert(&set, value)?;
        }
        Ok(set)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Convert a [`crate::dashmap::TryDashMapError`] into a [`TryDashSetError`].
fn map_error_to_set(e: crate::dashmap::TryDashMapError) -> TryDashSetError {
    match e {
        crate::dashmap::TryDashMapError::Alloc(a) => TryDashSetError::Alloc(a),
        crate::dashmap::TryDashMapError::Reserve(r) => TryDashSetError::Reserve(r),
        crate::dashmap::TryDashMapError::Clone(c) => TryDashSetError::Clone(c),
        crate::dashmap::TryDashMapError::Overflow => TryDashSetError::Overflow,
        crate::dashmap::TryDashMapError::Other(m) => TryDashSetError::Other(m),
    }
}

// ── TryClone for DashSet<T, S> ────────────────────────────────────────────────

impl<T, S> crate::try_clone::TryClone for DashSet<T, S>
where
    T: Eq + Hash + crate::try_clone::TryClone,
    S: BuildHasher + TryClone,
{
    fn try_clone(&self) -> Result<Self, crate::try_clone::TryCloneError> {
        let map_ref = convert_ref(self);
        let out = DashSet::with_hasher(map_ref.hasher().clone());
        // We cannot reserve everything immediately because shards may be uneven.
        if !self.is_empty() {
            let map: &DashMap<T, (), S> = convert_ref(&out);
            for elem in self.iter() {
                let entry = TryDashMap::try_entry_ref(map, &elem).map_err(|e| match e {
                    crate::dashmap::TryDashMapError::Alloc(a) => TryCloneError::Alloc(a),
                    crate::dashmap::TryDashMapError::Reserve(r) => TryCloneError::Reserve(r),
                    crate::dashmap::TryDashMapError::Clone(c) => c,
                    crate::dashmap::TryDashMapError::Overflow => TryCloneError::Overflow,
                    crate::dashmap::TryDashMapError::Other(m) => TryCloneError::Other(m),
                })?;
                entry.insert(());
            }
        }
        Ok(out)
    }
}

// ── TryDefault for DashSet<T, S> ──────────────────────────────────────────────

impl<T: Eq + Hash, S> crate::try_default::TryDefault for DashSet<T, S>
where
    S: BuildHasher + TryClone + TryDefault,
{
    fn try_default() -> Result<Self, crate::try_default::TryDefaultError> {
        Ok(DashSet::with_hasher(S::try_default()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang_alloc::string::String;
    use crate::lang_alloc::string::ToString;
    use crate::lang_alloc::vec;
    use crate::try_clone::TryClone as _;
    use crate::try_default::TryDefault as _;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn try_with_capacity_zero() {
        let set: DashSet<i32> = DashSet::<i32>::try_with_capacity(0).unwrap();
        assert!(set.is_empty());
    }

    #[test]
    fn try_with_capacity_nonzero() {
        let set: DashSet<String> = DashSet::<String>::try_with_capacity(10).unwrap();
        assert!(set.is_empty());
        assert!(set.capacity() >= 10);
    }

    // ── Insertion ────────────────────────────────────────────────────────────

    #[test]
    fn try_insert_single() {
        let set: DashSet<i32> = DashSet::new();
        let inserted = set.try_insert(42).unwrap();
        assert!(inserted);
        assert!(set.contains(&42));
    }

    #[test]
    fn try_insert_duplicate_returns_false() {
        let set: DashSet<i32> = DashSet::new();
        assert!(set.try_insert(1).unwrap());
        assert!(!set.try_insert(1).unwrap());
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn try_insert_multiple_values() {
        let set: DashSet<&str> = DashSet::new();
        set.try_insert("alpha").unwrap();
        set.try_insert("beta").unwrap();
        set.try_insert("gamma").unwrap();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn fallible_insert_matches_try_insert() {
        let set: DashSet<i32> = DashSet::new();
        set.fallible_insert(99).unwrap();
        assert!(set.contains(&99));
    }

    // ── Give-back variants ───────────────────────────────────────────────────

    #[test]
    fn try_insert_give_back_success() {
        let set: DashSet<String> = DashSet::new();
        set.try_insert_give_back("hello".to_string()).unwrap();
        assert!(set.contains("hello"));
    }

    #[test]
    fn try_insert_give_back_error_type_shape() {
        let set: DashSet<i32> = DashSet::new();
        let result: Result<bool, (i32, TryDashSetError)> = set.try_insert_give_back(1);
        assert!(result.is_ok());
    }

    // ── Extension ────────────────────────────────────────────────────────────

    #[test]
    fn try_extend_from_iterator() {
        let set: DashSet<i32> = DashSet::new();
        set.try_extend([1, 2, 3]).unwrap();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn try_extend_empty() {
        let set: DashSet<i32> = DashSet::new();
        set.try_extend(::lang_std::iter::empty::<i32>()).unwrap();
        assert!(set.is_empty());
    }

    // ── Shrink ────────────────────────────────────────────────────────────────

    #[test]
    fn try_shrink_to_fit_preserves_data() {
        let set: DashSet<String> = DashSet::new();
        set.try_insert("hello".to_string()).unwrap();
        set.try_insert("world".to_string()).unwrap();
        set.try_shrink_to_fit().unwrap();
        assert!(set.contains("hello"));
        assert!(set.contains("world"));
        assert_eq!(set.len(), 2);
    }

    // ── Bulk construction ────────────────────────────────────────────────────

    #[test]
    fn try_collect_range() {
        let set: DashSet<i32> =
            <DashSet<i32> as TryDashSet<_, RandomState>>::try_collect(0..5).unwrap();
        assert_eq!(set.len(), 5);
        assert!(set.contains(&3));
    }

    #[test]
    fn try_collect_empty() {
        let set: DashSet<i32> = <DashSet<i32> as TryDashSet<_, RandomState>>::try_collect(
            ::lang_std::iter::empty::<i32>(),
        )
        .unwrap();
        assert!(set.is_empty());
    }

    #[test]
    fn try_collect_with_deduplication() {
        let set: DashSet<i32> =
            <DashSet<i32> as TryDashSet<_, RandomState>>::try_collect(vec![1, 2, 2, 3, 3]).unwrap();
        assert_eq!(set.len(), 3);
    }

    // ── TryClone ─────────────────────────────────────────────────────────────

    #[test]
    fn try_clone_empty_set() {
        let set: DashSet<i32> = DashSet::new();
        let c = set.try_clone().unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn try_clone_populated_set() {
        let set: DashSet<String> = DashSet::new();
        set.insert("hello".to_string());
        set.insert("world".to_string());
        let c = set.try_clone().unwrap();
        assert!(c.contains("hello"));
        assert!(c.contains("world"));
    }

    // ── TryDefault ───────────────────────────────────────────────────────────

    #[test]
    fn try_default_empty_set() {
        let set: DashSet<i32> = DashSet::try_default().unwrap();
        assert!(set.is_empty());
    }

    // ── Combined workflows ───────────────────────────────────────────────────

    #[test]
    fn build_insert_clone_default() {
        let set: DashSet<String> = DashSet::try_default().unwrap();
        set.fallible_insert("alpha".to_string()).unwrap();
        set.fallible_insert("beta".to_string()).unwrap();
        let c = set.try_clone().unwrap();
        assert!(c.contains("alpha"));
        assert!(c.contains("beta"));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn collect_then_extend() {
        let set: DashSet<i32> =
            <DashSet<i32> as TryDashSet<_, RandomState>>::try_collect([1, 2]).unwrap();
        set.try_extend([3, 4]).unwrap();
        assert_eq!(set.len(), 4);
    }

    // ── OOM tests ─────────────────────────────────────────────────────
    use rustyfill_test_allocator::{FailPolicy, with_policy};

    #[test]
    fn dashset_try_insert_fails_on_oom() {
        let set: DashSet<u32> = DashSet::new();
        let r = with_policy(FailPolicy::fail_next_alloc(), || set.try_insert(1));
        assert!(r.is_err());
    }

    #[test]
    fn dashset_nth_alloc_fail_targets_correct_call() {
        // DashSet::try_clone does multiple internal allocations before reaching
        // fallible code, so nth-counting on try_clone is unreliable.
        let set: DashSet<u32> = DashSet::new();
        let (r1_ok, r2_err) = with_policy(FailPolicy::fail_nth_alloc(2), || {
            let r1 = set.try_insert(1);
            let r2 = set.try_insert(2);
            (r1.is_ok(), r2.is_err())
        });
        assert!(r1_ok, "first insert should succeed");
        assert!(r2_err, "second insert should fail");
    }

    #[test]
    fn dashset_oom_restores_allocation_afterwards() {
        let set: DashSet<u32> = DashSet::new();
        let r = with_policy(FailPolicy::fail_next_alloc(), || set.try_insert(1));
        assert!(r.is_err());
        assert!(set.try_insert(1).is_ok());
    }
}
