//! Fallible hash set operations.
//!
//! Provides the [`TryHashSet`] trait with methods that mirror common `HashSet`
//! constructors and mutating operations but return [`Result`] to handle allocation
//! failures gracefully, using [`::lang_std::collections::TryReserveError`] as the primary
//! error type.
//!
//! # Design
//!
//! `TryHashSet` is implemented for `HashSet<T, S>`. Methods that may grow the
//! internal table (`insert`, `extend`, etc.) return a `Result` instead of panicking
//! on out-of-memory. Read-only accessors delegate directly to `HashSet`.
//!
//! The trait also implements [`TryClone`](crate::try_clone::TryClone) and
//! [`TryDefault`](crate::try_default::TryDefault) for `HashSet<T, S>` when
//! `T` satisfies the respective bounds.

use crate::alloc::AllocError;
use crate::alloc::TryReserveError;
use lang_core::cmp;
use lang_core::fmt;
use lang_std::cmp::Eq;
use lang_std::collections::{HashSet, TryReserveError as StdTryReserveError};
use lang_std::hash::{BuildHasher, Hash, RandomState};
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::{TryDebug, helpers::FormatterExt};

// ── Error type ────────────────────────────────────────────────────────────────

/// Error returned by [`TryHashSet`] operations.
///
/// Wraps the ways a hash set operation can fail on stable Rust: a reserve
/// failure ([`TryReserveError`], returned by the inherent `HashSet::try_reserve`)
/// or a clone failure ([`TryCloneError`]) when an element's `try_clone` cannot
/// allocate its internal buffers.
#[derive(Debug)]
pub enum TryHashSetError {
    /// A raw heap allocation failed (no collection involved).
    Alloc(AllocError),
    /// A capacity reservation on the hash set failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An element clone failed during a method that requires [`TryClone`].
    Clone(TryCloneError),
    /// An arithmetic overflow occurred while computing required capacity.
    Overflow,
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Display for TryHashSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "hash set operation failed: heap allocation error"),
            Self::Reserve(e) => write!(f, "hash set operation failed: {}", e),
            Self::Clone(e) => write!(f, "hash set operation failed: {}", e),
            Self::Overflow => write!(
                f,
                "hash set operation failed: capacity calculation overflowed"
            ),
            Self::Other(msg) => write!(f, "hash set operation failed: {}", msg),
        }
    }
}

impl From<AllocError> for TryHashSetError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

impl From<TryReserveError> for TryHashSetError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<TryCloneError> for TryHashSetError {
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

impl TryDebug for TryHashSetError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => f
                .try_debug_struct("TryHashSetError::Alloc")
                .field("0", e)
                .finish(),
            Self::Reserve(e) => f
                .try_debug_struct("TryHashSetError::Reserve")
                .field("0", e)
                .finish(),
            Self::Clone(e) => f
                .try_debug_struct("TryHashSetError::Clone")
                .field("0", e)
                .finish(),
            Self::Overflow => f.write_str("TryHashSetError::Overflow"),
            Self::Other(msg) => f
                .try_debug_struct("TryHashSetError::Other")
                .field("0", msg)
                .finish(),
        }
    }
}

impl From<TryDefaultError> for TryHashSetError {
    fn from(err: TryDefaultError) -> Self {
        match err {
            TryDefaultError::Alloc(e) => Self::Alloc(e),
            TryDefaultError::Reserve(e) => Self::Reserve(e),
            TryDefaultError::Overflow => Self::Overflow,
            TryDefaultError::Other(msg) => Self::Other(msg),
        }
    }
}

impl From<StdTryReserveError> for TryHashSetError {
    fn from(e: StdTryReserveError) -> Self {
        Self::Reserve(TryReserveError::from(e))
    }
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// A trait for fallible hash set operations.
///
/// Implemented for `HashSet<T, S>`. Mirrors the most commonly-used `HashSet`
/// methods that can fail due to allocation pressure, returning [`Result`] values
/// that propagate [`TryReserveError`] or [`TryHashSetError`] on failure.
pub trait TryHashSet<T, S = RandomState>: Sized {
    // ── Construction ────────────────────────────────────────────────────────

    /// Fallibly construct an empty `HashSet` with a default-constructed hasher.
    ///
    /// Unlike [`Self::try_with_capacity`], which hardcodes [`RandomState`] and
    /// may panic on first use in a new thread (due to thread-local seeding),
    /// this method uses [`TryDefault`] to construct the hasher fallibly. If
    /// hasher construction fails (e.g. `RandomState` panics during seed
    /// initialization), the error is returned rather than unwinding.
    ///
    /// Requires `S: [`TryDefault`]` so that hasher creation is safe even when
    /// it involves runtime allocation or thread-local state.
    fn try_new() -> Result<HashSet<T, S>, TryHashSetError>
    where
        S: TryDefault;

    /// Fallibly construct an empty `HashSet` with at least enough capacity for
    /// `capacity` elements.
    ///
    /// Constructs the hasher via [`TryDefault`] (same as [`Self::try_new`]),
    /// then reserves capacity for `capacity` elements. Returns
    /// [`TryHashSetError::Reserve`] if the capacity reservation fails, or
    /// [`TryHashSetError::Other`] if hasher construction panics.
    ///
    /// Requires `S: [`TryDefault`]` so that hasher creation is safe even when
    /// it involves runtime allocation or thread-local state.
    fn try_with_capacity(capacity: usize) -> Result<HashSet<T, S>, TryHashSetError>
    where
        S: TryDefault;

    /// Fallibly construct an empty `HashSet` with at least enough capacity for
    /// `capacity` elements, using the provided hash builder.
    ///
    /// Returns [`TryReserveError`] if the initial allocation fails.
    /// Equivalent to [`HashSet::with_capacity_and_hasher`] but fallible.
    fn try_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<HashSet<T, S>, TryReserveError>;

    // ── Insertion ───────────────────────────────────────────────────────────

    /// Fallibly insert the value into the set.
    ///
    /// Reserves capacity for one additional element before inserting, so this
    /// method never panics on out-of-memory. Returns [`TryHashSetError::Reserve`]
    /// if the capacity reservation fails.
    ///
    /// Returns `true` if the value was not already present in the set, `false`
    /// otherwise (in which case it is not modified).
    fn try_insert(&mut self, value: T) -> Result<bool, TryHashSetError>
    where
        T: Eq + Hash;

    /// Like [`Self::try_insert`] but returns ownership of `value` back on
    /// allocation failure.
    fn try_insert_give_back(&mut self, value: T) -> Result<bool, (T, TryHashSetError)>
    where
        T: Eq + Hash;

    // ── Aliases with `fallible_` prefix ────────────────────────────────────

    /// Alias for [`Self::try_new`].
    fn fallible_new() -> Result<HashSet<T, S>, TryHashSetError>
    where
        S: TryDefault,
    {
        Self::try_new()
    }

    /// Alias for [`Self::try_with_capacity`].
    fn fallible_with_capacity(capacity: usize) -> Result<HashSet<T, S>, TryHashSetError>
    where
        S: TryDefault,
    {
        Self::try_with_capacity(capacity)
    }

    /// Alias for [`Self::try_with_capacity_and_hasher`].
    fn fallible_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<HashSet<T, S>, TryReserveError> {
        Self::try_with_capacity_and_hasher(capacity, hasher)
    }

    /// Alias for [`Self::try_insert`].
    fn fallible_insert(&mut self, value: T) -> Result<bool, TryHashSetError>
    where
        T: Eq + Hash,
    {
        Self::try_insert(self, value)
    }

    /// Alias for [`Self::try_insert_give_back`].
    fn fallible_insert_give_back(&mut self, value: T) -> Result<bool, (T, TryHashSetError)>
    where
        T: Eq + Hash,
    {
        Self::try_insert_give_back(self, value)
    }

    // ── Extension ───────────────────────────────────────────────────────────

    /// Fallibly extend the set with all values from an iterator source.
    ///
    /// Accepts anything that implements [`ResumableSource`](crate::recovery::ResumableSource).
    /// On reserve failure, returns a [`Resumable`](crate::recovery::Resumable)
    /// containing any consumed-but-uncommitted element and the remainder of the
    /// iterator, which the caller can pass right back in.
    fn try_extend<Src>(
        &mut self,
        source: Src,
    ) -> Result<(), (TryReserveError, crate::recovery::Resumable<Src::Inner>)>
    where
        Src: crate::recovery::ResumableSource<Item = T>;

    /// Fallibly extend the set by cloning elements from a slice.
    ///
    /// Returns [`TryHashSetError::Reserve`] on capacity failure or
    /// [`TryHashSetError::Clone`] if an element clone fails.
    fn try_extend_from_slice(&mut self, other: &[T]) -> Result<(), TryHashSetError>
    where
        T: Eq + Hash + TryClone;

    /// Alias for [`Self::try_extend`].
    fn fallible_extend<Src>(
        &mut self,
        source: Src,
    ) -> Result<(), (TryReserveError, crate::recovery::Resumable<Src::Inner>)>
    where
        Src: crate::recovery::ResumableSource<Item = T>,
    {
        Self::try_extend(self, source)
    }

    /// Alias for [`Self::try_extend_from_slice`].
    fn fallible_extend_from_slice(&mut self, other: &[T]) -> Result<(), TryHashSetError>
    where
        T: Eq + Hash + TryClone,
    {
        Self::try_extend_from_slice(self, other)
    }

    // ── Capacity / shrink ───────────────────────────────────────────────────

    /// Fallibly shrink the capacity of this hash set to match its length.
    ///
    /// Rebuilds the internal table so that it holds approximately `len` elements.
    /// Requires `S: TryClone` so the hasher can be safely duplicated for the new
    /// table without risking a panic. Returns [`TryHashSetError::Reserve`] if the
    /// allocation for the rebuilt table fails, or [`TryHashSetError::Clone`] if
    /// duplicating the hasher factory fails. Equivalent to
    /// [`HashSet::shrink_to_fit`] but fallible.
    fn try_shrink_to_fit(&mut self) -> Result<(), TryHashSetError>
    where
        S: TryClone;

    /// Fallibly shrink the capacity of this hash set to hold at least
    /// `min_capacity` elements.
    ///
    /// If the current capacity is already less than or equal to `min_capacity`,
    /// does nothing and returns `Ok(())`. Otherwise rebuilds the table with the
    /// target capacity. Requires `S: TryClone` so the hasher can be safely
    /// duplicated. Returns [`TryHashSetError::Reserve`] if the allocation fails,
    /// or [`TryHashSetError::Clone`] if duplicating the hasher factory fails.
    /// Equivalent to [`HashSet::shrink_to`] but fallible.
    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryHashSetError>
    where
        S: TryClone;

    /// Alias for [`Self::try_shrink_to_fit`].
    fn fallible_shrink_to_fit(&mut self) -> Result<(), TryHashSetError>
    where
        S: TryClone,
    {
        Self::try_shrink_to_fit(self)
    }

    /// Alias for [`Self::try_shrink_to`].
    fn fallible_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryHashSetError>
    where
        S: TryClone,
    {
        Self::try_shrink_to(self, min_capacity)
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    /// Fallibly collect an iterator of `T` values into a `HashSet`.
    ///
    /// Constructs the hasher via [`TryDefault`] and uses the iterator's size
    /// hint to pre-allocate when possible.
    fn try_collect<I: IntoIterator<Item = T>>(iter: I) -> Result<HashSet<T, S>, TryHashSetError>
    where
        S: TryDefault;

    /// Fallibly create a `HashSet` from an iterator using the provided hasher.
    fn try_collect_with_hasher<I: IntoIterator<Item = T>>(
        iter: I,
        hasher: S,
    ) -> Result<HashSet<T, S>, TryReserveError>;

    /// Alias for [`Self::try_collect`].
    fn fallible_collect<I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<HashSet<T, S>, TryHashSetError>
    where
        S: TryDefault,
    {
        Self::try_collect(iter)
    }

    /// Alias for [`Self::try_collect_with_hasher`].
    fn fallible_collect_with_hasher<I: IntoIterator<Item = T>>(
        iter: I,
        hasher: S,
    ) -> Result<HashSet<T, S>, TryReserveError> {
        Self::try_collect_with_hasher(iter, hasher)
    }
}

// ── Implementation ────────────────────────────────────────────────────────────

impl<T: Eq + Hash, S: BuildHasher> TryHashSet<T, S> for HashSet<T, S> {
    // ── Construction ────────────────────────────────────────────────────────

    fn try_new() -> Result<HashSet<T, S>, TryHashSetError>
    where
        S: TryDefault,
    {
        let hasher = S::try_default()?;
        Ok(HashSet::with_hasher(hasher))
    }

    fn try_with_capacity(capacity: usize) -> Result<HashSet<T, S>, TryHashSetError>
    where
        S: TryDefault,
    {
        let mut set = Self::try_new()?;
        if capacity > 0 {
            set.try_reserve(capacity).map_err(TryHashSetError::from)?;
        }
        Ok(set)
    }

    fn try_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<HashSet<T, S>, TryReserveError> {
        let mut set = HashSet::with_hasher(hasher);
        if capacity > 0 {
            set.try_reserve(capacity)?;
        }
        Ok(set)
    }

    // ── Insertion ───────────────────────────────────────────────────────────

    fn try_insert(&mut self, value: T) -> Result<bool, TryHashSetError>
    where
        T: Eq + Hash,
    {
        self.try_reserve(1)
            .map_err(|e| TryHashSetError::Reserve(e.into()))?;
        Ok(self.insert(value))
    }

    fn try_insert_give_back(&mut self, value: T) -> Result<bool, (T, TryHashSetError)>
    where
        T: Eq + Hash,
    {
        match self.try_reserve(1) {
            Ok(()) => Ok(self.insert(value)),
            Err(e) => Err((value, TryHashSetError::Reserve(e.into()))),
        }
    }

    // ── Extension ───────────────────────────────────────────────────────────

    fn try_extend<Src>(
        &mut self,
        source: Src,
    ) -> Result<(), (TryReserveError, crate::recovery::Resumable<Src::Inner>)>
    where
        Src: crate::recovery::ResumableSource<Item = T>,
    {
        use crate::recovery::Resumable;

        let (head, mut iter) = source.safe_into_iter();

        if let Some(value) = head {
            if self.len() == self.capacity()
                && let Err(e) = self.try_reserve(1)
            {
                return Err((e.into(), Resumable::new(value, iter)));
            }
            self.insert(value);
        }

        let (lower, _) = iter.size_hint();
        if lower > 0
            && let Err(e) = self.try_reserve(lower)
        {
            return Err((e.into(), Resumable::from_remainder(iter)));
        }
        while let Some(value) = iter.next() {
            if self.len() == self.capacity()
                && let Err(e) = self.try_reserve(1)
            {
                return Err((e.into(), Resumable::new(value, iter)));
            }
            self.insert(value);
        }
        Ok(())
    }

    fn try_extend_from_slice(&mut self, other: &[T]) -> Result<(), TryHashSetError>
    where
        T: Eq + Hash + TryClone,
    {
        if other.is_empty() {
            return Ok(());
        }
        let len_before = self.len();
        self.try_reserve(other.len())
            .map_err(|e| TryHashSetError::Reserve(e.into()))?;
        for elem in other {
            match elem.try_clone() {
                Ok(cloned) => {
                    self.insert(cloned);
                }
                Err(e) => {
                    // Drain the elements we already inserted.
                    for _ in 0..self.len() - len_before {
                        self.drain().next();
                    }
                    return Err(TryHashSetError::Clone(e));
                }
            }
        }
        Ok(())
    }

    // ── Capacity / shrink ───────────────────────────────────────────────────

    fn try_shrink_to_fit(&mut self) -> Result<(), TryHashSetError>
    where
        S: TryClone,
    {
        Self::try_shrink_to(self, self.len())
    }

    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryHashSetError>
    where
        S: TryClone,
    {
        let target = cmp::max(self.len(), min_capacity);
        if self.capacity() <= target {
            return Ok(());
        }
        let hasher = self.hasher().try_clone().map_err(TryHashSetError::from)?;
        let mut new_set = HashSet::with_capacity_and_hasher(0, hasher);
        new_set
            .try_reserve(target)
            .map_err(|e| TryHashSetError::Reserve(e.into()))?;
        for v in self.drain() {
            new_set.insert(v);
        }
        *self = new_set;
        Ok(())
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    fn try_collect<I: IntoIterator<Item = T>>(iter: I) -> Result<HashSet<T, S>, TryHashSetError>
    where
        S: TryDefault,
    {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower);
        let mut set = Self::try_new()?;
        if capacity > 0 {
            set.try_reserve(capacity).map_err(TryHashSetError::from)?;
        }
        for value in iter {
            if set.len() == set.capacity() {
                set.try_reserve(1)?;
            }
            set.insert(value);
        }
        Ok(set)
    }

    fn try_collect_with_hasher<I: IntoIterator<Item = T>>(
        iter: I,
        hasher: S,
    ) -> Result<HashSet<T, S>, TryReserveError> {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower);
        let mut set = HashSet::with_hasher(hasher);
        if capacity > 0 {
            set.try_reserve(capacity)?;
        }
        for value in iter {
            if set.len() == set.capacity() {
                set.try_reserve(1)?;
            }
            set.insert(value);
        }
        Ok(set)
    }
}

// ── TryClone for HashSet<T, S> ────────────────────────────────────────────────

/// Implements [`TryClone`] for `HashSet<T, S>` when elements are cloneable and
/// the hasher factory implements [`TryClone`].
impl<T, S> TryClone for HashSet<T, S>
where
    T: Eq + Hash + TryClone,
    S: BuildHasher + TryClone,
{
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let hasher = self.hasher().try_clone()?;
        let mut out = HashSet::with_hasher(hasher);
        if !self.is_empty() {
            out.try_reserve(self.len())
                .map_err(|e| TryCloneError::Reserve(e.into()))?;
        }
        for elem in self.iter() {
            match elem.try_clone() {
                Ok(cloned) => {
                    out.insert(cloned);
                }
                Err(e) => {
                    drop(out);
                    return Err(e);
                }
            }
        }
        Ok(out)
    }
}

// ── TryDefault for HashSet<T, S> ──────────────────────────────────────────────

impl<T, S: BuildHasher + TryDefault> TryDefault for HashSet<T, S> {
    fn try_default() -> Result<Self, TryDefaultError> {
        let hasher = S::try_default()?;
        Ok(HashSet::with_hasher(hasher))
    }
}

// ── TryDebug for HashSet<T, S> ────────────────────────────────────────────────

impl<T: crate::try_fmt::TryDebug, S> crate::try_fmt::TryDebug for HashSet<T, S> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::try_fmt::helpers::FormatterExt;
        f.try_debug_set().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang_alloc::string::String;
    use lang_alloc::string::ToString;
    use lang_alloc::vec;
    use lang_alloc::vec::Vec;
    use lang_std::hash::RandomState;
    use lang_std::iter;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn try_with_capacity_zero() {
        let set: HashSet<i32> = HashSet::<i32>::try_with_capacity(0).unwrap();
        assert!(set.is_empty());
    }

    #[test]
    fn try_with_capacity_nonzero() {
        let set: HashSet<String> = HashSet::<String>::try_with_capacity(10).unwrap();
        assert!(set.is_empty());
        assert!(set.capacity() >= 10);
    }

    #[test]
    fn try_with_capacity_and_hasher() {
        use lang_std::collections::hash_map::DefaultHasher;
        use lang_std::hash::BuildHasherDefault;

        let hasher = BuildHasherDefault::<DefaultHasher>::default();
        let set: HashSet<&str, _> = HashSet::try_with_capacity_and_hasher(5, hasher).unwrap();
        assert!(set.is_empty());
        assert!(set.capacity() >= 5);
    }

    // ── Insertion ────────────────────────────────────────────────────────────

    #[test]
    fn fallible_insert_single() {
        let mut set: HashSet<i32> = HashSet::new();
        let inserted = set.fallible_insert(42).unwrap();
        assert!(inserted);
        assert!(set.contains(&42));
    }

    #[test]
    fn fallible_insert_duplicate_returns_false() {
        let mut set: HashSet<i32> = HashSet::new();
        assert!(set.fallible_insert(1).unwrap());
        assert!(!set.fallible_insert(1).unwrap());
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn fallible_insert_multiple_values() {
        let mut set: HashSet<&str> = HashSet::new();
        set.fallible_insert("alpha").unwrap();
        set.fallible_insert("beta").unwrap();
        set.fallible_insert("gamma").unwrap();
        assert_eq!(set.len(), 3);
        assert!(set.contains("alpha"));
        assert!(set.contains("beta"));
        assert!(set.contains("gamma"));
    }

    #[test]
    fn fallible_insert_complex_values() {
        let mut set: HashSet<Vec<u8>> = HashSet::new();
        set.fallible_insert(vec![1, 2, 3]).unwrap();
        assert!(set.contains(&vec![1, 2, 3]));
    }

    // ── Give-back variants ───────────────────────────────────────────────────

    #[test]
    fn fallible_insert_give_back_success() {
        let mut set: HashSet<String> = HashSet::new();
        set.fallible_insert_give_back("hello".to_string()).unwrap();
        assert!(set.contains("hello"));
    }

    #[test]
    fn fallible_insert_give_back_error_type_shape() {
        let mut set: HashSet<i32> = HashSet::new();
        let result: Result<bool, (i32, TryHashSetError)> = set.fallible_insert_give_back(1);
        assert!(result.is_ok());
    }

    // ── Extension ────────────────────────────────────────────────────────────

    #[test]
    fn try_extend_from_iterator() {
        let mut set: HashSet<i32> = HashSet::new();
        set.try_extend([1, 2, 3]).unwrap();
        assert_eq!(set.len(), 3);
        assert!(set.contains(&1));
        assert!(set.contains(&3));
    }

    #[test]
    fn try_extend_empty() {
        let mut set: HashSet<i32> = HashSet::new();
        set.try_extend(iter::empty::<i32>()).unwrap();
        assert!(set.is_empty());
    }

    #[test]
    fn try_extend_existing() {
        let mut set: HashSet<i32> = HashSet::new();
        set.fallible_insert(1).unwrap();
        set.try_extend([2, 3]).unwrap();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn try_extend_from_slice_clones() {
        let mut set: HashSet<Vec<u8>> = HashSet::new();
        set.fallible_insert(vec![1]).unwrap();
        let slice: &[Vec<u8>] = &[vec![2, 3]];
        set.try_extend_from_slice(slice).unwrap();
        assert_eq!(set.len(), 2);
        assert!(set.contains(&vec![2, 3]));
    }

    #[test]
    fn try_extend_from_slice_empty() {
        let mut set: HashSet<i32> = HashSet::new();
        set.try_extend_from_slice(&[]).unwrap();
        assert!(set.is_empty());
    }

    // ── Shrink ────────────────────────────────────────────────────────────────

    #[test]
    fn try_shrink_to_fit_preserves_data() {
        let mut set: HashSet<String> = HashSet::new();
        set.fallible_insert("hello".to_string()).unwrap();
        set.fallible_insert("world".to_string()).unwrap();
        set.fallible_shrink_to_fit().unwrap();
        assert!(set.contains("hello"));
        assert!(set.contains("world"));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn try_shrink_to_fit_reduces_excess() {
        let mut set: HashSet<i32> = HashSet::new();
        set.try_reserve(1024).unwrap();
        set.fallible_insert(1).unwrap();
        let cap_before = set.capacity();
        assert!(cap_before >= 1024);
        set.fallible_shrink_to_fit().unwrap();
        assert!(set.capacity() < cap_before || set.capacity() >= 1);
        assert!(set.contains(&1));
    }

    #[test]
    fn try_shrink_to_above_len() {
        let mut set: HashSet<i32> = HashSet::new();
        set.try_reserve(256).unwrap();
        set.fallible_insert(42).unwrap();
        set.fallible_shrink_to(32).unwrap();
        assert!(set.capacity() >= 32);
        assert!(set.contains(&42));
    }

    #[test]
    fn try_shrink_to_noop_when_already_small() {
        let mut set: HashSet<i32> = HashSet::new();
        set.fallible_insert(1).unwrap();
        set.fallible_shrink_to(16).unwrap();
        assert!(set.contains(&1));
    }

    // ── Bulk construction ────────────────────────────────────────────────────

    #[test]
    fn try_collect_range() {
        let set: HashSet<i32> =
            <HashSet<i32> as TryHashSet<_, RandomState>>::try_collect(0..5).unwrap();
        assert_eq!(set.len(), 5);
        assert!(set.contains(&3));
    }

    #[test]
    fn try_collect_empty() {
        let set: HashSet<i32> = <HashSet<i32> as TryHashSet<_, RandomState>>::try_collect(
            iter::empty::<i32>(),
        )
        .unwrap();
        assert!(set.is_empty());
    }

    #[test]
    fn try_collect_strings() {
        let vals = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let set: HashSet<String> =
            <HashSet<String> as TryHashSet<_, RandomState>>::try_collect(vals).unwrap();
        assert_eq!(set.len(), 3);
        assert!(set.contains("a"));
    }

    #[test]
    fn try_collect_with_deduplication() {
        let set: HashSet<i32> =
            <HashSet<i32> as TryHashSet<_, RandomState>>::try_collect(vec![1, 2, 2, 3, 3]).unwrap();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn try_collect_with_hasher() {
        use lang_std::collections::hash_map::DefaultHasher;
        use lang_std::hash::BuildHasherDefault;

        let hasher = BuildHasherDefault::<DefaultHasher>::default();
        let set: HashSet<i32, _> = HashSet::try_collect_with_hasher([1, 2, 3], hasher).unwrap();
        assert_eq!(set.len(), 3);
    }

    // ── TryClone ─────────────────────────────────────────────────────────────

    #[test]
    fn try_clone_empty_set() {
        let set: HashSet<i32> = HashSet::new();
        let c = set.try_clone().unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn try_clone_populated_set() {
        let mut set: HashSet<String> = HashSet::new();
        set.insert("hello".to_string());
        set.insert("world".to_string());
        let c = set.try_clone().unwrap();
        assert!(c.contains("hello"));
        assert!(c.contains("world"));
    }

    #[test]
    fn try_clone_nested_values() {
        let mut set: HashSet<Vec<Vec<u8>>> = HashSet::new();
        set.insert(vec![vec![1, 2], vec![3]]);
        let c = set.try_clone().unwrap();
        assert!(c.contains(&vec![vec![1, 2], vec![3]]));
    }

    // ── TryDefault ───────────────────────────────────────────────────────────

    #[test]
    fn try_default_empty_set() {
        let set: HashSet<i32> = HashSet::try_default().unwrap();
        assert!(set.is_empty());
    }

    #[test]
    fn try_default_set_with_custom_hasher() {
        use lang_std::collections::hash_map::DefaultHasher;
        use lang_std::hash::BuildHasherDefault;

        let set: HashSet<i32, BuildHasherDefault<DefaultHasher>> = HashSet::try_default().unwrap();
        assert!(set.is_empty());
    }

    // ── Combined workflows ───────────────────────────────────────────────────

    #[test]
    fn build_insert_clone_default() {
        let mut set: HashSet<String> = HashSet::try_default().unwrap();
        set.fallible_insert("alpha".to_string()).unwrap();
        set.fallible_insert("beta".to_string()).unwrap();
        let c = set.try_clone().unwrap();
        assert!(c.contains("alpha"));
        assert!(c.contains("beta"));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn collect_then_extend() {
        let mut a: HashSet<i32> =
            <HashSet<i32> as TryHashSet<_, RandomState>>::try_collect([1, 2]).unwrap();
        a.try_extend([3, 4]).unwrap();
        assert_eq!(a.len(), 4);
        assert!(a.contains(&4));
    }

    #[test]
    fn extend_from_slice_rollback_on_failure_type() {
        let mut set: HashSet<Vec<u8>> = HashSet::new();
        let slice: &[Vec<u8>] = &[vec![1]];
        let result: Result<(), TryHashSetError> = set.try_extend_from_slice(slice);
        assert!(result.is_ok());
        assert!(set.contains(&vec![1]));
    }

    #[test]
    fn fallible_aliases_match_try_methods() {
        let s1: HashSet<i32> =
            <HashSet<i32> as TryHashSet<_, RandomState>>::fallible_with_capacity(5).unwrap();
        let s2: HashSet<i32> =
            <HashSet<i32> as TryHashSet<_, RandomState>>::try_with_capacity(5).unwrap();
        assert!(s1.is_empty());
        assert!(s2.is_empty());
    }

    // ── OOM tests ─────────────────────────────────────────────────────────────
    use rustyfill_test_allocator::{FailPolicy, with_policy};

    #[test]
    fn hashset_try_with_capacity_fails_on_oom() {
        let r: Result<HashSet<u32>, TryHashSetError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <HashSet<u32> as TryHashSet<u32, RandomState>>::try_with_capacity(10)
            });
        assert!(r.is_err());
    }

    #[test]
    fn hashset_try_with_capacity_zero_succeeds_under_oom() {
        let r: Result<HashSet<u32>, TryHashSetError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <HashSet<u32> as TryHashSet<u32, RandomState>>::try_with_capacity(0)
            });
        assert!(r.is_ok());
    }

    #[test]
    fn hashset_try_insert_fails_on_oom() {
        let mut set: HashSet<u32> = HashSet::new();
        set.try_shrink_to_fit().unwrap();
        let r = with_policy(FailPolicy::fail_next_alloc(), || set.try_insert(42));
        assert!(r.is_err());
    }

    #[test]
    fn hashset_try_clone_fails_on_oom() {
        let orig: HashSet<u32> = HashSet::from([1, 2, 3]);
        let r: Result<HashSet<u32>, TryCloneError> =
            with_policy(FailPolicy::fail_next_alloc(), || orig.try_clone());
        assert!(r.is_err());
    }

    #[test]
    fn hashset_try_clone_empty_succeeds_under_oom() {
        let orig: HashSet<u32> = HashSet::new();
        let r: Result<HashSet<u32>, TryCloneError> =
            with_policy(FailPolicy::fail_next_alloc(), || orig.try_clone());
        assert!(r.is_ok());
    }

    #[test]
    fn hashset_try_collect_fails_on_oom() {
        let items = [1u32, 2u32, 3u32];
        let r: Result<HashSet<u32>, TryHashSetError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                HashSet::try_collect(items.iter().copied())
            });
        assert!(r.is_err());
    }

    #[test]
    fn hashset_oom_restores_allocation_afterwards() {
        let r: Result<HashSet<u32>, TryHashSetError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <HashSet<u32> as TryHashSet<u32, RandomState>>::try_with_capacity(10)
            });
        assert!(r.is_err());
        // Allocation works again after guard scope ends.
        let r: Result<HashSet<u32>, TryHashSetError> =
            <HashSet<u32> as TryHashSet<u32, RandomState>>::try_with_capacity(10);
        assert!(r.is_ok());
    }

    #[test]
    fn hashset_nth_alloc_fail_targets_correct_call() {
        type HS = HashSet<u32, RandomState>;
        let (r1_ok, r2_err, r3_ok) = with_policy(FailPolicy::fail_nth_alloc(2), || {
            let r1: Result<HS, TryHashSetError> =
                <HS as TryHashSet<u32, RandomState>>::try_with_capacity(1);
            let r2: Result<HS, TryHashSetError> =
                <HS as TryHashSet<u32, RandomState>>::try_with_capacity(1);
            let r3: Result<HS, TryHashSetError> =
                <HS as TryHashSet<u32, RandomState>>::try_with_capacity(1);
            (r1.is_ok(), r2.is_err(), r3.is_ok())
        });
        assert!(r1_ok, "first alloc should succeed");
        assert!(r2_err, "second alloc should fail");
        assert!(r3_ok, "third alloc should succeed");
    }
}
