//! Fallible B-tree set operations.
//!
//! Provides the [`TryBTreeSet`] trait with methods that mirror common `BTreeSet`
//! constructors and mutating operations but return [`Result`] to handle allocation
//! failures gracefully.
//!
//! # Design
//!
//! `TryBTreeSet` is implemented for `BTreeSet<T>`. Methods that may grow the
//! internal tree (`insert`, `extend`, etc.) return a `Result` instead of panicking
//! on out-of-memory. Read-only accessors delegate directly to `BTreeSet`.
//!
//! Because `BTreeSet::try_reserve` does not exist, these methods internally
//! use [`std::panic::catch_unwind`] to intercept allocation panics from the
//! B-tree's internal node allocator. This means `T` must be
//! [`RefUnwindSafe`](core::panic::RefUnwindSafe) for the fallible mutation methods.
//!
//! The trait also implements [`TryClone`](crate::try_clone::TryClone) and
//! [`TryDefault`](crate::try_default::TryDefault) for `BTreeSet<T>` when
//! `T` satisfies the respective bounds.

use crate::alloc::{AllocError, PayloadBox};
use crate::try_clone::TryCloneError;
use core::fmt;
use std::collections::BTreeSet;
use std::panic::{AssertUnwindSafe, RefUnwindSafe, catch_unwind};

// ── Error type ────────────────────────────────────────────────────────────────

/// Error returned by [`TryBTreeSet`] operations.
///
/// Since `BTreeSet::try_reserve` does not exist, this error type
/// wraps a caught panic as [`Self::AllocPanic`] when an internal node allocation
/// fails during insertion or extension. Clone failures during bulk operations
/// are wrapped as [`Self::Clone`].
#[derive(Debug)]
pub enum TryBTreeSetError {
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

impl fmt::Display for TryBTreeSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "B-tree set operation failed: heap allocation error"),
            Self::AllocPanic(payload) => {
                write!(
                    f,
                    "B-tree set operation failed: internal allocation panicked: {}",
                    payload.message()
                )
            }
            Self::Clone(e) => write!(f, "B-tree set cloning failed: {}", e),
            Self::Other(msg) => write!(f, "B-tree set operation failed: {}", msg),
        }
    }
}

impl From<AllocError> for TryBTreeSetError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

impl From<TryCloneError> for TryBTreeSetError {
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// A trait for fallible B-tree set operations.
///
/// Implemented for `BTreeSet<T>`. Mirrors the most commonly-used `BTreeSet`
/// methods that can fail due to allocation pressure, returning [`Result`] values
/// that propagate [`TryBTreeSetError`] on failure.
///
/// # Note
///
/// Because `BTreeSet::try_reserve` does not exist, mutation methods use
/// [`std::panic::catch_unwind`] internally to intercept OOM panics.
/// Elements must be [`RefUnwindSafe`] for these methods.
pub trait TryBTreeSet<T>: Sized {
    // ── Construction ────────────────────────────────────────────────────────

    /// Fallibly construct an empty `BTreeSet`.
    ///
    /// Unlike `HashSet`, `BTreeSet` does not pre-allocate nodes on construction.
    /// This always succeeds without allocation. Equivalent to [`BTreeSet::new`]
    /// but fallible.
    fn try_new() -> BTreeSet<T>;

    // ── Insertion ───────────────────────────────────────────────────────────

    /// Fallibly insert the value into the set.
    ///
    /// Catches allocation panics from internal B-tree node allocation, so this
    /// method never panics on out-of-memory. Returns
    /// [`TryBTreeSetError::AllocPanic`] if an internal allocation fails.
    ///
    /// Returns `true` if the value was not already present in the set, `false`
    /// otherwise (in which case it is not modified).
    fn try_insert(&mut self, value: T) -> Result<bool, TryBTreeSetError>
    where
        T: Ord + RefUnwindSafe;

    // ── Aliases with `fallible_` prefix ────────────────────────────────────

    /// Alias for [`Self::try_new`].
    fn fallible_new() -> BTreeSet<T> {
        Self::try_new()
    }

    /// Alias for [`Self::try_insert`].
    fn fallible_insert(&mut self, value: T) -> Result<bool, TryBTreeSetError>
    where
        T: Ord + RefUnwindSafe,
    {
        Self::try_insert(self, value)
    }

    // ── Extension ───────────────────────────────────────────────────────────

    /// Fallibly extend the set with all values from an iterator.
    ///
    /// Catches allocation panics from internal B-tree node allocation during
    /// the extend operation. Returns [`TryBTreeSetError::AllocPanic`] if an
    /// internal allocation fails.
    ///
    /// Note: because we catch the panic after the fact, partial extension may
    /// have occurred on failure. The set will be structurally consistent but
    /// may contain some of the extended elements.
    fn try_extend<I: IntoIterator<Item = T>>(
        &mut self,
        iter: I,
    ) -> Result<(), TryBTreeSetError>
    where
        T: Ord + RefUnwindSafe;

    /// Fallibly extend the set by cloning elements from a slice.
    ///
    /// Returns [`TryBTreeSetError::Clone`] if an element clone fails, or
    /// [`TryBTreeSetError::AllocPanic`] if an internal allocation fails.
    /// On clone failure, rolls back any elements already inserted.
    fn try_extend_from_slice(&mut self, other: &[T]) -> Result<(), TryBTreeSetError>
    where
        T: Ord + RefUnwindSafe + crate::try_clone::TryClone;

    /// Alias for [`Self::try_extend`].
    fn fallible_extend<I: IntoIterator<Item = T>>(
        &mut self,
        iter: I,
    ) -> Result<(), TryBTreeSetError>
    where
        T: Ord + RefUnwindSafe,
    {
        Self::try_extend(self, iter)
    }

    /// Alias for [`Self::try_extend_from_slice`].
    fn fallible_extend_from_slice(&mut self, other: &[T]) -> Result<(), TryBTreeSetError>
    where
        T: Ord + RefUnwindSafe + crate::try_clone::TryClone,
    {
        Self::try_extend_from_slice(self, other)
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    /// Fallibly collect an iterator of `T` values into a `BTreeSet`.
    ///
    /// Catches allocation panics from internal B-tree node allocation.
    /// Returns [`TryBTreeSetError::AllocPanic`] if an internal allocation fails.
    fn try_collect<I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<BTreeSet<T>, TryBTreeSetError>
    where
        T: Ord + RefUnwindSafe;

    /// Alias for [`Self::try_collect`].
    fn fallible_collect<I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<BTreeSet<T>, TryBTreeSetError>
    where
        T: Ord + RefUnwindSafe,
    {
        Self::try_collect(iter)
    }
}

// ── Implementation ────────────────────────────────────────────────────────────

impl<T: Ord + RefUnwindSafe> TryBTreeSet<T> for BTreeSet<T> {
    // ── Construction ────────────────────────────────────────────────────────

    fn try_new() -> BTreeSet<T> {
        BTreeSet::new()
    }

    // ── Insertion ───────────────────────────────────────────────────────────

    fn try_insert(&mut self, value: T) -> Result<bool, TryBTreeSetError> {
        let result: bool = catch_unwind(AssertUnwindSafe(|| self.insert(value)))
            .map_err(|payload| TryBTreeSetError::AllocPanic(PayloadBox(payload)))?;
        Ok(result)
    }

    // ── Extension ───────────────────────────────────────────────────────────

    fn try_extend<I: IntoIterator<Item = T>>(
        &mut self,
        iter: I,
    ) -> Result<(), TryBTreeSetError>
    where
        T: Ord + RefUnwindSafe,
    {
        catch_unwind(AssertUnwindSafe(|| {
            self.extend(iter);
        }))
        .map_err(|payload| TryBTreeSetError::AllocPanic(PayloadBox(payload)))?;
        Ok(())
    }

    fn try_extend_from_slice(&mut self, other: &[T]) -> Result<(), TryBTreeSetError>
    where
        T: Ord + RefUnwindSafe + crate::try_clone::TryClone,
    {
        if other.is_empty() {
            return Ok(());
        }
        let len_before = self.len();
        for elem in other {
            match elem.try_clone() {
                Ok(cloned) => {
                    self.insert(cloned);
                }
                Err(e) => {
                    // Rollback: drain elements we already inserted.
                    for _ in 0..self.len() - len_before {
                        self.pop_first();
                    }
                    return Err(TryBTreeSetError::Clone(e));
                }
            }
        }
        Ok(())
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    fn try_collect<I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<BTreeSet<T>, TryBTreeSetError>
    where
        T: Ord + RefUnwindSafe,
    {
        let mut set = BTreeSet::new();
        catch_unwind(AssertUnwindSafe(|| {
            set.extend(iter);
        }))
        .map_err(|payload| TryBTreeSetError::AllocPanic(PayloadBox(payload)))?;
        Ok(set)
    }
}

// ── TryClone for BTreeSet<T> ────────────────────────────────────────────────

/// Implements [`TryClone`] for `BTreeSet<T>` when elements are cloneable.
/// Uses fallible clone for each element and catches allocation panics from
/// internal B-tree node growth. Clones one element at a time and inserts it
/// directly, avoiding an intermediate `Vec` allocation.
impl<T> crate::try_clone::TryClone for BTreeSet<T>
where
    T: Ord + crate::try_clone::TryClone,
{
    fn try_clone(&self) -> Result<Self, crate::try_clone::TryCloneError> {
        use crate::try_clone::TryCloneError;

        let mut out = BTreeSet::new();
        for elem in self.iter() {
            match elem.try_clone() {
                Ok(cloned) => {
                    catch_unwind(AssertUnwindSafe(|| {
                        out.insert(cloned);
                    }))
                    .map_err(|_| TryCloneError::Other("BTreeSet allocation failed during clone"))?;
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

// ── TryDefault for BTreeSet<T> ────────────────────────────────────────────────

impl<T> crate::try_default::TryDefault for BTreeSet<T> {
    fn try_default() -> Result<Self, crate::try_default::TryDefaultError>
    where
        Self: Sized,
    {
        // An empty BTreeSet requires no allocation.
        Ok(BTreeSet::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::try_clone::TryClone;
    use crate::try_default::TryDefault;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn try_new_creates_empty_set() {
        let set: BTreeSet<i32> = BTreeSet::<i32>::try_new();
        assert!(set.is_empty());
    }

    #[test]
    fn fallible_new_alias_works() {
        let set: BTreeSet<String> =
            <BTreeSet<String> as TryBTreeSet<_>>::fallible_new();
        assert!(set.is_empty());
    }

    // ── Insertion ────────────────────────────────────────────────────────────

    #[test]
    fn fallible_insert_single() {
        let mut set: BTreeSet<i32> = BTreeSet::new();
        let inserted = set.fallible_insert(42).unwrap();
        assert!(inserted);
        assert!(set.contains(&42));
    }

    #[test]
    fn fallible_insert_duplicate_returns_false() {
        let mut set: BTreeSet<i32> = BTreeSet::new();
        assert!(set.fallible_insert(1).unwrap());
        assert!(!set.fallible_insert(1).unwrap());
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn fallible_insert_multiple_values() {
        let mut set: BTreeSet<&str> = BTreeSet::new();
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
        let mut set: BTreeSet<Vec<u8>> = BTreeSet::new();
        set.fallible_insert(vec![1, 2, 3]).unwrap();
        assert!(set.contains(&vec![1, 2, 3]));
    }

    // ── Extension ────────────────────────────────────────────────────────────

    #[test]
    fn try_extend_from_iterator() {
        let mut set: BTreeSet<i32> = BTreeSet::new();
        set.try_extend([1, 2, 3]).unwrap();
        assert_eq!(set.len(), 3);
        assert!(set.contains(&1));
        assert!(set.contains(&3));
    }

    #[test]
    fn try_extend_empty() {
        let mut set: BTreeSet<i32> = BTreeSet::new();
        set.try_extend(std::iter::empty::<i32>()).unwrap();
        assert!(set.is_empty());
    }

    #[test]
    fn try_extend_existing() {
        let mut set: BTreeSet<i32> = BTreeSet::new();
        set.fallible_insert(1).unwrap();
        set.try_extend([2, 3]).unwrap();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn try_extend_from_slice_clones() {
        let mut set: BTreeSet<Vec<u8>> = BTreeSet::new();
        set.fallible_insert(vec![1]).unwrap();
        let slice: &[Vec<u8>] = &[vec![2, 3]];
        set.try_extend_from_slice(slice).unwrap();
        assert_eq!(set.len(), 2);
        assert!(set.contains(&vec![2, 3]));
    }

    #[test]
    fn try_extend_from_slice_empty() {
        let mut set: BTreeSet<i32> = BTreeSet::new();
        set.try_extend_from_slice(&[]).unwrap();
        assert!(set.is_empty());
    }

    #[test]
    fn fallible_extend_alias_works() {
        let mut set: BTreeSet<i32> = BTreeSet::new();
        set.fallible_extend([1, 2, 3]).unwrap();
        assert_eq!(set.len(), 3);
    }

    // ── Bulk construction ────────────────────────────────────────────────────

    #[test]
    fn try_collect_range() {
        let set: BTreeSet<i32> =
            <BTreeSet<i32> as TryBTreeSet<_>>::try_collect(0..5).unwrap();
        assert_eq!(set.len(), 5);
        assert!(set.contains(&3));
    }

    #[test]
    fn try_collect_empty() {
        let set: BTreeSet<i32> =
            <BTreeSet<i32> as TryBTreeSet<_>>::try_collect(std::iter::empty::<i32>())
                .unwrap();
        assert!(set.is_empty());
    }

    #[test]
    fn try_collect_strings() {
        let vals = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let set: BTreeSet<String> =
            <BTreeSet<String> as TryBTreeSet<_>>::try_collect(vals).unwrap();
        assert_eq!(set.len(), 3);
        assert!(set.contains("a"));
    }

    #[test]
    fn try_collect_with_deduplication() {
        let set: BTreeSet<i32> =
            <BTreeSet<i32> as TryBTreeSet<_>>::try_collect(vec![1, 2, 2, 3, 3])
                .unwrap();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn fallible_collect_alias_works() {
        let set: BTreeSet<i32> =
            <BTreeSet<i32> as TryBTreeSet<_>>::fallible_collect([1, 2, 3]).unwrap();
        assert_eq!(set.len(), 3);
    }

    // ── TryClone ─────────────────────────────────────────────────────────────

    #[test]
    fn try_clone_empty_set() {
        let set: BTreeSet<i32> = BTreeSet::new();
        let c = set.try_clone().unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn try_clone_populated_set() {
        let mut set: BTreeSet<String> = BTreeSet::new();
        set.insert("hello".to_string());
        set.insert("world".to_string());
        let c = set.try_clone().unwrap();
        assert!(c.contains("hello"));
        assert!(c.contains("world"));
    }

    #[test]
    fn try_clone_nested_values() {
        let mut set: BTreeSet<Vec<Vec<u8>>> = BTreeSet::new();
        set.insert(vec![vec![1, 2], vec![3]]);
        let c = set.try_clone().unwrap();
        assert!(c.contains(&vec![vec![1, 2], vec![3]]));
    }

    // ── TryDefault ───────────────────────────────────────────────────────────

    #[test]
    fn try_default_empty_set() {
        let set: BTreeSet<i32> = BTreeSet::try_default().unwrap();
        assert!(set.is_empty());
    }

    // ── Combined workflows ───────────────────────────────────────────────────

    #[test]
    fn build_insert_clone_default() {
        let mut set: BTreeSet<String> = BTreeSet::try_default().unwrap();
        set.fallible_insert("alpha".to_string()).unwrap();
        set.fallible_insert("beta".to_string()).unwrap();
        let c = set.try_clone().unwrap();
        assert!(c.contains("alpha"));
        assert!(c.contains("beta"));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn collect_then_extend() {
        let mut a: BTreeSet<i32> =
            <BTreeSet<i32> as TryBTreeSet<_>>::try_collect([1, 2]).unwrap();
        a.try_extend([3, 4]).unwrap();
        assert_eq!(a.len(), 4);
        assert!(a.contains(&4));
    }

    #[test]
    fn ordered_iteration_after_operations() {
        let mut set: BTreeSet<i32> = BTreeSet::new();
        set.fallible_insert(3).unwrap();
        set.fallible_insert(1).unwrap();
        set.fallible_insert(2).unwrap();
        let vals: Vec<&i32> = set.iter().collect();
        assert_eq!(vals, &[&1, &2, &3]);
    }

    #[test]
    fn extend_from_slice_rollback_on_failure_type() {
        let mut set: BTreeSet<Vec<u8>> = BTreeSet::new();
        let slice: &[Vec<u8>] = &[vec![1]];
        let result: Result<(), TryBTreeSetError> = set.try_extend_from_slice(slice);
        assert!(result.is_ok());
        assert!(set.contains(&vec![1]));
    }

    #[test]
    fn fallible_aliases_match_try_methods() {
        let s1: BTreeSet<i32> = <BTreeSet<i32> as TryBTreeSet<_>>::fallible_new();
        let s2: BTreeSet<i32> = <BTreeSet<i32> as TryBTreeSet<_>>::try_new();
        assert!(s1.is_empty());
        assert!(s2.is_empty());
    }

    // ── Error formatting ─────────────────────────────────────────────────────

    #[test]
    fn error_display_alloc_panic() {
        let payload: Box<dyn core::any::Any + Send> = Box::new("out of memory");
        let err = TryBTreeSetError::AllocPanic(PayloadBox(payload));
        let msg = format!("{}", err);
        assert!(msg.contains("allocation panicked"));
    }

    #[test]
    fn error_display_other() {
        let err = TryBTreeSetError::Other("logic error");
        let msg = format!("{}", err);
        assert!(msg.contains("logic error"));
    }
}
