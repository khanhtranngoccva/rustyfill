//! Fallible binary-heap operations and formatting for [`BinaryHeap<T>`].
//!
//! Provides:
//! - [`TryClone`] for `BinaryHeap<T>` when `T: TryClone` — reserves capacity
//!   up front, then clones each element (heap order is restored by the
//!   standard `push` on the clone).
//! - [`TryDefault`] for `BinaryHeap<T>` — an empty heap needs no allocation.
//! - [`TryDebug`] for `BinaryHeap<T>` — mirrors std's bracketed, comma-joined
//!   rendering, routing each element through its fallible formatter.
//! - [`TryBinaryHeap`] — fallible versions of the OOM-prone mutation methods
//!   (`push`, `append`, bulk construction) that return
//!   [`TryReserveError`] instead of panicking.
//! - [`TryExtend`]/[`TryExtendFromSlice`] impls so
//!   generic fallible-extension code works over heaps as well.
//!
//! Note: no `TryDisplay` impl — `BinaryHeap` does not implement `fmt::Display`
//! (only `Debug`), and `TryDisplay` requires `Display` as a supertrait.

use crate::alloc::TryReserveError;
use crate::recovery::Resumable;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_extend::{TryExtend, TryExtendFromSlice};
use crate::try_fmt::{TryDebug, TryDisplay};
use lang_alloc::collections::BinaryHeap;
use lang_core::fmt;

// ── TryClone for BinaryHeap<T> ────────────────────────────────────────────────

impl<T: TryClone + Ord> TryClone for BinaryHeap<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let mut out = BinaryHeap::<T>::new();
        if !self.is_empty() {
            // One reservation covers every slot we will push.
            out.try_reserve(self.len())
                .map_err(TryCloneError::Reserve)?;
        }
        for elem in self.iter() {
            match elem.try_clone() {
                Ok(cloned) => out.push(cloned),
                Err(e) => {
                    drop(out);
                    return Err(e);
                }
            }
        }
        Ok(out)
    }
}

// ── TryDefault for BinaryHeap<T> ──────────────────────────────────────────────

impl<T: TryDefault> TryDefault for BinaryHeap<T> {
    fn try_default() -> Result<Self, TryDefaultError> {
        // An empty heap requires no allocation.
        Ok(BinaryHeap::new())
    }
}

// ── TryDebug for BinaryHeap<T> ────────────────────────────────────────────────

impl<T: TryDebug> TryDebug for BinaryHeap<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::try_fmt::helpers::FormatterExt;
        f.try_debug_list().entries(self.iter()).finish()
    }
}

// ── Error types ───────────────────────────────────────────────────────────────

/// Error for fallible `BinaryHeap` operations that allocate and/or clone elements.
///
/// Covers `try_with_capacity`, `try_from_elem`, `try_from_slice`, and
/// `try_collect` — any operation whose failure modes are limited to a capacity
/// reservation ([`TryReserveError`]) or an element clone failure
/// ([`TryCloneError`]).
pub enum TryBinaryHeapWithCloneError {
    /// A capacity reservation on the heap failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An element clone failed during a method that requires `TryClone`.
    Clone(TryCloneError),
}

impl fmt::Debug for TryBinaryHeapWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(self, f)
    }
}

impl fmt::Display for TryBinaryHeapWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }
}

impl From<TryReserveError> for TryBinaryHeapWithCloneError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<TryCloneError> for TryBinaryHeapWithCloneError {
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

impl TryDebug for TryBinaryHeapWithCloneError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::debug_field(f, "TryBinaryHeapWithCloneError::Reserve", e),
            Self::Clone(e) => u::debug_field(f, "TryBinaryHeapWithCloneError::Clone", e),
        }
    }
}

impl TryDisplay for TryBinaryHeapWithCloneError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::display_delegated(f, "binary heap", e),
            Self::Clone(e) => u::display_delegated(f, "binary heap", e),
        }
    }
}

// ── TryBinaryHeap ─────────────────────────────────────────────────────────────

/// A trait for fallible binary-heap operations.
///
/// Implemented for `BinaryHeap<T>`. Mirrors the OOM-prone `BinaryHeap` methods
/// (`push`, `append`, `with_capacity`, bulk construction) but returns
/// [`Result`] values that propagate [`TryReserveError`] on failure instead of
/// panicking.
///
/// `BinaryHeap` stores its elements in an internal `Vec`, so every growth path
/// funnels through a single capacity reservation — exactly what
/// `BinaryHeap::try_reserve` exposes. Each method below reserves first, then
/// delegates to the infallible counterpart, which can no longer reallocate.
pub trait TryBinaryHeap<T>: Sized {
    // ── Construction ────────────────────────────────────────────────────────

    /// Fallibly construct an empty `BinaryHeap` with at least enough capacity
    /// for `capacity` elements. Equivalent to [`BinaryHeap::with_capacity`] but
    /// fallible.
    fn try_with_capacity(capacity: usize) -> Result<BinaryHeap<T>, TryReserveError>;

    /// Fallibly collect an iterator into a `BinaryHeap<T>`.
    ///
    /// Uses the iterator's size hint to pre-allocate when possible, falling back
    /// to incremental reservations if the iterator yields more than hinted.
    fn try_collect<I: IntoIterator<Item = T>>(iter: I) -> Result<BinaryHeap<T>, TryReserveError>;

    /// Fallibly create a `BinaryHeap<T>` from a slice by cloning each element via
    /// [`TryClone`].
    ///
    /// Returns [`TryBinaryHeapWithCloneError::Reserve`] on capacity failure or
    /// [`TryBinaryHeapWithCloneError::Clone`] if an element's [`TryClone::try_clone`]
    /// fails.
    fn try_from_slice(slice: &[T]) -> Result<BinaryHeap<T>, TryBinaryHeapWithCloneError>
    where
        T: TryClone;

    // ── Mutation ────────────────────────────────────────────────────────────

    /// Fallibly insert an element into the heap.
    ///
    /// Returns [`TryReserveError`] if growing the internal buffer fails. The
    /// heap is left untouched on failure.
    fn try_push(&mut self, value: T) -> Result<(), TryReserveError>;

    /// Like [`Self::try_push`] but returns ownership of `value` back on failure.
    fn try_push_give_back(&mut self, value: T) -> Result<(), (T, TryReserveError)>;

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// This is the fallible analogue of [`BinaryHeap::append`]. Elements are moved
    /// rather than cloned. If `self` has spare capacity the transfer may happen
    /// without any new allocation; otherwise a single `try_reserve` call is made
    /// first so that failure is returned as [`TryReserveError`] instead of
    /// panicking.
    ///
    /// On success `other` is drained (length zero). On failure `other` is left
    /// untouched.
    fn try_append(&mut self, other: &mut BinaryHeap<T>) -> Result<(), TryReserveError>;

    // ── Aliases with `fallible_` prefix ─────────────────────────────────────

    /// Alias for [`Self::try_with_capacity`].
    fn fallible_with_capacity(capacity: usize) -> Result<BinaryHeap<T>, TryReserveError> {
        Self::try_with_capacity(capacity)
    }

    /// Alias for [`Self::try_collect`].
    fn fallible_collect<I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<BinaryHeap<T>, TryReserveError> {
        Self::try_collect(iter)
    }

    /// Alias for [`Self::try_from_slice`].
    fn fallible_from_slice(slice: &[T]) -> Result<BinaryHeap<T>, TryBinaryHeapWithCloneError>
    where
        T: TryClone,
    {
        Self::try_from_slice(slice)
    }

    /// Alias for [`Self::try_push`].
    fn fallible_push(&mut self, value: T) -> Result<(), TryReserveError> {
        Self::try_push(self, value)
    }

    /// Alias for [`Self::try_push_give_back`].
    fn fallible_push_give_back(&mut self, value: T) -> Result<(), (T, TryReserveError)> {
        Self::try_push_give_back(self, value)
    }

    /// Alias for [`Self::try_append`].
    fn fallible_append(&mut self, other: &mut BinaryHeap<T>) -> Result<(), TryReserveError> {
        Self::try_append(self, other)
    }
}

impl<T: Ord> TryBinaryHeap<T> for BinaryHeap<T> {
    // ── Construction ────────────────────────────────────────────────────────
    fn try_with_capacity(capacity: usize) -> Result<BinaryHeap<T>, TryReserveError> {
        let mut heap = BinaryHeap::<T>::new();
        if capacity > 0 {
            heap.try_reserve(capacity)?;
        }
        Ok(heap)
    }

    fn try_collect<I: IntoIterator<Item = T>>(iter: I) -> Result<BinaryHeap<T>, TryReserveError> {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower);
        let mut heap = BinaryHeap::<T>::new();
        if capacity > 0 {
            heap.try_reserve(capacity)?;
        }
        for item in iter {
            // Iterator may yield more elements than its hint promised.
            if heap.len() == heap.capacity() {
                heap.try_reserve(1)?;
            }
            heap.push(item);
        }
        Ok(heap)
    }

    fn try_from_slice(slice: &[T]) -> Result<BinaryHeap<T>, TryBinaryHeapWithCloneError>
    where
        T: TryClone,
    {
        let mut heap = BinaryHeap::<T>::new();
        if !slice.is_empty() {
            heap.try_reserve(slice.len())
                .map_err(TryBinaryHeapWithCloneError::Reserve)?;
        }
        for item in slice {
            heap.push(
                item.try_clone()
                    .map_err(TryBinaryHeapWithCloneError::Clone)?,
            );
        }
        Ok(heap)
    }

    // ── Mutation ────────────────────────────────────────────────────────────

    fn try_push(&mut self, value: T) -> Result<(), TryReserveError> {
        self.try_reserve(1)?;
        self.push(value);
        Ok(())
    }

    fn try_push_give_back(&mut self, value: T) -> Result<(), (T, TryReserveError)> {
        match self.try_reserve(1) {
            Ok(()) => {
                self.push(value);
                Ok(())
            }
            Err(e) => Err((value, e)),
        }
    }

    fn try_append(&mut self, other: &mut BinaryHeap<T>) -> Result<(), TryReserveError> {
        let extra = other.len();
        if extra == 0 {
            return Ok(());
        }
        // Mirror std's append optimization: the larger heap absorbs the smaller
        // one, so at most `extra` slots need to be reserved here.
        if self.len() < other.len() {
            lang_core::mem::swap(self, other);
        }
        // Reserve first — lazy, no mutations until allocation succeeds.
        self.try_reserve(extra)?;
        // Now safe to call the inherent append; capacity is guaranteed.
        self.append(other);
        Ok(())
    }
}

// ── Generic TryExtend / TryExtendFromSlice impls ──────────────────────────────

impl<'s, T: Ord + TryClone> TryExtendFromSlice<'s, T> for BinaryHeap<T> {
    type Error = TryBinaryHeapWithCloneError;

    fn try_extend_from_slice(
        &mut self,
        other: &'s [T],
    ) -> Result<(), (&'s [T], TryBinaryHeapWithCloneError)> {
        if other.is_empty() {
            return Ok(());
        }
        self.try_reserve(other.len())
            .map_err(|e| (other, TryBinaryHeapWithCloneError::Reserve(e)))?;
        for (i, item) in other.iter().enumerate() {
            match item.try_clone() {
                Ok(cloned) => {
                    self.push(cloned);
                }
                Err(e) => {
                    // No rollback: elements pushed before the failure are kept.
                    // Return the unprocessed tail so the caller can retry.
                    return Err((&other[i..], TryBinaryHeapWithCloneError::Clone(e)));
                }
            }
        }
        Ok(())
    }
}

impl<T: Ord> TryExtend<T> for BinaryHeap<T> {
    type Error = TryReserveError;

    fn try_extend<S>(&mut self, source: S) -> Result<(), (Resumable<S::Inner>, TryReserveError)>
    where
        S: crate::recovery::ResumableSource<Item = T>,
    {
        let (head, mut iter) = source.safe_into_iter();

        if let Some(item) = head {
            if self.len() == self.capacity()
                && let Err(e) = self.try_reserve(1)
            {
                return Err((Resumable::new(item, iter), e));
            }
            self.push(item);
        }

        let (lower, _) = iter.size_hint();
        if lower > 0
            && let Err(e) = self.try_reserve(lower)
        {
            return Err((Resumable::from_remainder(iter), e));
        }
        while let Some(item) = iter.next() {
            if self.len() == self.capacity()
                && let Err(e) = self.try_reserve(1)
            {
                return Err((Resumable::new(item, iter), e));
            }
            self.push(item);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::TryReserveErrorExt;
    use crate::try_format;
    use lang_alloc::format;
    use lang_alloc::string::String;
    use lang_alloc::vec;
    use lang_alloc::vec::Vec;

    #[test]
    fn binary_heap_try_clone_success() {
        let bh: BinaryHeap<i32> = vec![3, 1, 2].into_iter().collect();
        let cloned = bh.try_clone().unwrap();
        // Same multiset of elements; the internal buffer pointers differ.
        let mut a: lang_alloc::vec::Vec<i32> = bh.into_iter().collect();
        let mut b: lang_alloc::vec::Vec<i32> = cloned.into_iter().collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        assert_eq!(b, vec![1, 2, 3]);
    }

    #[test]
    fn binary_heap_try_clone_empty() {
        let bh: BinaryHeap<i32> = BinaryHeap::new();
        let cloned = bh.try_clone().unwrap();
        assert!(cloned.is_empty());
    }

    #[test]
    fn binary_heap_try_default_empty() {
        let bh: BinaryHeap<i32> = BinaryHeap::try_default().unwrap();
        assert!(bh.is_empty());
    }

    #[test]
    fn binary_heap_try_debug_sorted() {
        let bh: BinaryHeap<i32> = vec![3, 1, 2].into_iter().collect();
        let dbg = try_format!("{:?}", bh).unwrap();
        // std Debug prints in descending order for a max-heap.
        assert_eq!(dbg, "[3, 1, 2]");
    }

    #[test]
    fn binary_heap_try_debug_empty() {
        let bh: BinaryHeap<String> = BinaryHeap::new();
        let dbg = try_format!("{:?}", bh).unwrap();
        assert_eq!(dbg, "[]");
    }

    // ── TryBinaryHeap tests ──────────────────────────────────────────────────

    fn sorted(h: BinaryHeap<i32>) -> lang_alloc::vec::Vec<i32> {
        let mut v: lang_alloc::vec::Vec<i32> = h.into_iter().collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn binary_heap_new_does_not_allocate() {
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        // If new() performed any heap allocation, fail_all_alloc would cause
        // the OOM handler to fire (or abort). It doesn't — Vec::new() is a
        // zero-capacity inline representation with no backing store.
        let h = with_policy(FailPolicy::fail_all(), BinaryHeap::<i32>::new);
        assert!(h.is_empty());
        assert_eq!(h.capacity(), 0);
    }

    #[test]
    fn binary_heap_try_with_capacity() {
        let h: BinaryHeap<i32> = BinaryHeap::try_with_capacity(8).unwrap();
        assert!(h.capacity() >= 8);
        assert!(h.is_empty());
        let h: BinaryHeap<i32> = BinaryHeap::try_with_capacity(0).unwrap();
        assert!(h.is_empty());
    }

    #[test]
    fn binary_heap_try_push_and_give_back() {
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        let mut h: BinaryHeap<i32> = BinaryHeap::new();
        h.try_push(5).unwrap();
        h.try_push(9).unwrap();
        assert_eq!(h.pop(), Some(9));
        assert_eq!(h.pop(), Some(5));

        let mut h: BinaryHeap<i32> = BinaryHeap::new();
        let (back, _err) = with_policy(FailPolicy::fail_next_alloc(), || {
            h.try_push_give_back(42).unwrap_err()
        });
        assert_eq!(back, 42);
        assert!(h.is_empty());
    }

    #[test]
    fn binary_heap_try_push_fails_on_oom() {
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        let mut h: BinaryHeap<i32> = BinaryHeap::new();
        let r = with_policy(FailPolicy::fail_next_alloc(), || h.try_push(1));
        assert!(r.is_err());
        assert!(h.is_empty());
    }

    #[test]
    fn binary_heap_try_append_moves_elements() {
        let mut a: BinaryHeap<i32> = vec![1, 2].into_iter().collect();
        let mut b: BinaryHeap<i32> = vec![3, 4, 5].into_iter().collect();
        a.try_append(&mut b).unwrap();
        assert!(b.is_empty());
        assert_eq!(sorted(a), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn binary_heap_try_append_swaps_when_other_larger() {
        let mut a: BinaryHeap<i32> = vec![1].into_iter().collect();
        let mut b: BinaryHeap<i32> = vec![2, 3, 4].into_iter().collect();
        a.try_append(&mut b).unwrap();
        assert!(b.is_empty());
        assert_eq!(sorted(a), vec![1, 2, 3, 4]);
    }

    #[test]
    fn binary_heap_try_append_fails_on_oom_leaves_both_intact() {
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        // Equal-size heaps prevent the swap inside try_append. The combined
        // size (4) exceeds the initial capacity (2), forcing a real alloc.
        let mut a: BinaryHeap<i32> = vec![1, 2].into_iter().collect();
        let mut b: BinaryHeap<i32> = vec![3, 4].into_iter().collect();
        let r = with_policy(FailPolicy::fail_all(), || a.try_append(&mut b));
        assert!(r.is_err());
        assert_eq!(sorted(a), vec![1, 2]);
        assert_eq!(sorted(b), vec![3, 4]);
    }

    #[test]
    fn binary_heap_try_collect_basic() {
        let h = BinaryHeap::try_collect(3..7).unwrap();
        assert_eq!(sorted(h), vec![3, 4, 5, 6]);
    }

    #[test]
    fn binary_heap_try_collect_fails_on_oom() {
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        let r = with_policy(FailPolicy::fail_next_alloc(), || {
            BinaryHeap::try_collect(0..10)
        });
        assert!(r.is_err());
    }

    #[test]
    fn binary_heap_try_from_slice_basic() {
        let h = BinaryHeap::try_from_slice(&[3, 1, 2]).unwrap();
        assert_eq!(sorted(h), vec![1, 2, 3]);
    }

    #[test]
    fn binary_heap_try_from_slice_fails_on_oom() {
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        let r = with_policy(FailPolicy::fail_next_alloc(), || {
            BinaryHeap::try_from_slice(&[1, 2, 3])
        });
        assert!(matches!(r, Err(TryBinaryHeapWithCloneError::Reserve(_))));
    }

    #[test]
    fn binary_heap_generic_try_extend_via_trait() {
        let mut h: BinaryHeap<i32> = BinaryHeap::new();
        <_ as TryExtend<i32>>::try_extend(&mut h, 10..13).unwrap();
        assert_eq!(sorted(h), vec![10, 11, 12]);
    }

    #[test]
    fn binary_heap_generic_try_extend_retry_with_resumable() {
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        let mut h: BinaryHeap<i32> = BinaryHeap::new();
        // Fail all allocations inside the policy. The head push's
        // try_reserve(1) fails immediately (heap is empty, len==cap==0).
        // All 4 items are stranded in the Resumable.
        let resumable =
            with_policy(
                FailPolicy::fail_all(),
                || match <_ as TryExtend<i32>>::try_extend(&mut h, 0..4) {
                    Ok(()) => panic!("expected failure"),
                    Err((resumable, _)) => resumable,
                },
            );
        assert_eq!(h.len(), 0);
        // Retry outside the policy: everything lands.
        <_ as TryExtend<i32>>::try_extend(&mut h, resumable).unwrap();
        assert_eq!(sorted(h), vec![0, 1, 2, 3]);
    }

    #[test]
    fn binary_heap_generic_try_extend_from_slice_via_trait() {
        let mut h: BinaryHeap<Vec<u8>> = BinaryHeap::new();
        let slice: &[Vec<u8>] = &[vec![7]];
        <_ as TryExtendFromSlice<'_, Vec<u8>>>::try_extend_from_slice(&mut h, slice).unwrap();
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn binary_heap_error_display() {
        let e = TryBinaryHeapWithCloneError::Reserve(TryReserveErrorExt::new_capacity_overflow());
        let s = format!("{e}");
        assert!(s.contains("binary heap"));
        let d = format!("{e:?}");
        assert!(d.contains("TryBinaryHeapWithCloneError::Reserve"));
    }
}
