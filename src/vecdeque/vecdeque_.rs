//! Fallible double-ended queue operations.
//!
//! Provides the [`TryVecDeque`] trait with methods that mirror common `VecDeque`
//! constructors and mutating operations but return [`Result`] to handle allocation
//! failures gracefully, using [`std::collections::TryReserveError`] as the primary
//! error type.
//!
//! # Design
//!
//! `TryVecDeque` is implemented for `VecDeque<T>`. Methods that may grow internal
//! capacity (`push_back`, `push_front`, `insert`, `extend`, etc.) return a `Result`
//! instead of panicking on out-of-memory. Read-only accessors delegate directly to
//! `VecDeque`.
//!
//! The trait also implements [`TryClone`](crate::try_clone::TryClone) and
//! [`TryDefault`](crate::try_default::TryDefault) for `VecDeque<T>` when `T`
//! satisfies the respective bounds.

use crate::alloc::AllocError;
use crate::alloc::TryReserveError;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use core::fmt;
use std::collections::VecDeque;

/// Error returned by [`TryVecDeque`] operations.
#[derive(Debug)]
pub enum TryVecDequeError {
    Alloc(AllocError),
    Reserve(TryReserveError),
    Clone(TryCloneError),
    Overflow,
    Other(&'static str),
}

impl fmt::Display for TryVecDequeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "deque operation failed: heap allocation error"),
            Self::Reserve(e) => write!(f, "deque operation failed: {}", e),
            Self::Clone(e) => write!(f, "deque operation failed: {}", e),
            Self::Overflow => {
                write!(f, "deque operation failed: capacity calculation overflowed")
            }
            Self::Other(msg) => write!(f, "deque operation failed: {}", msg),
        }
    }
}

impl From<AllocError> for TryVecDequeError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

impl From<TryReserveError> for TryVecDequeError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<TryCloneError> for TryVecDequeError {
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

/// A trait for fallible VecDeque operations.
pub trait TryVecDeque<T>: Sized {
    // ── Construction ────────────────────────────────────────────────────────

    /// Fallibly construct an empty `VecDeque` with at least enough capacity for
    /// `capacity` elements. Equivalent to [`VecDeque::with_capacity`] but fallible.
    fn try_with_capacity(capacity: usize) -> Result<VecDeque<T>, TryReserveError>;

    /// Fallibly construct a `VecDeque<T>` containing `value` cloned `n` times.
    fn try_from_elem(value: &T, n: usize) -> Result<VecDeque<T>, TryVecDequeError>
    where
        T: TryClone;

    /// Like [`Self::try_from_elem`] but takes ownership of `value` and returns
    /// it on failure so the caller is not left empty-handed.
    fn try_from_elem_give_back(value: T, n: usize) -> Result<VecDeque<T>, (T, TryVecDequeError)>
    where
        T: TryClone;

    /// Fallibly collect an iterator into a `VecDeque<T>`.
    fn try_collect<I: IntoIterator<Item = T>>(iter: I) -> Result<VecDeque<T>, TryReserveError>;

    /// Fallibly create a `VecDeque<T>` from a slice by cloning each element via
    /// [`TryClone`].
    fn try_from_slice(slice: &[T]) -> Result<VecDeque<T>, TryVecDequeError>
    where
        T: TryClone;

    // ── Mutation: push / pop ────────────────────────────────────────────────

    /// Fallibly append an element to the back of the deque.
    fn try_push_back(&mut self, value: T) -> Result<(), TryReserveError>;

    /// Like [`Self::try_push_back`] but returns ownership of `value` back on failure.
    fn try_push_back_give_back(&mut self, value: T) -> Result<(), (T, TryReserveError)>;

    /// Fallibly prepend an element to the front of the deque.
    fn try_push_front(&mut self, value: T) -> Result<(), TryReserveError>;

    /// Like [`Self::try_push_front`] but returns ownership of `value` back on failure.
    fn try_push_front_give_back(&mut self, value: T) -> Result<(), (T, TryReserveError)>;

    /// Remove and return the last element from the back, or `None` if empty.
    fn try_pop_back(&mut self) -> Option<T>;

    // ── Mutation: insert / remove / extend ──────────────────────────────────

    /// Fallibly insert an element at position `index`.
    fn try_insert(&mut self, index: usize, value: T) -> Result<(), TryVecDequeError>;

    /// Like [`Self::try_insert`] but returns ownership of `value` back on failure.
    fn try_insert_give_back(&mut self, index: usize, value: T)
    -> Result<(), (T, TryVecDequeError)>;

    /// Remove and return the element at `index`, shifting all elements after it.
    ///
    /// Returns [`TryVecDequeError::Other`] if `index >= len`.
    fn try_remove(&mut self, index: usize) -> Result<Option<T>, TryVecDequeError>;

    /// Fallibly extend the deque with all elements from an iterator.
    fn try_extend<I: IntoIterator<Item = T>>(&mut self, iter: I) -> Result<(), TryReserveError>;

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    fn try_append(&mut self, other: &mut VecDeque<T>) -> Result<(), TryReserveError>;

    // ── Mutation: resize / shrink / clear ───────────────────────────────────

    /// Resizes the deque so that `len` equals `new_len`.
    fn try_resize_with<F>(&mut self, new_len: usize, f: F) -> Result<(), TryReserveError>
    where
        F: FnMut() -> T;

    /// Fallibly shrink the capacity of this deque to match its length.
    fn fallible_shrink_to_fit(&mut self) -> Result<(), TryVecDequeError>;

    /// Fallibly shrink the capacity of this deque to at least `min_capacity`.
    fn fallible_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryVecDequeError>;

    /// Clears the deque, removing all values. This operation never allocates.
    fn try_clear(&mut self);

    // ── Aliases with `fallible_` prefix ─────────────────────────────────────

    /// Alias for [`Self::try_with_capacity`].
    fn fallible_with_capacity(capacity: usize) -> Result<VecDeque<T>, TryReserveError> {
        Self::try_with_capacity(capacity)
    }

    /// Alias for [`Self::try_from_elem`].
    fn fallible_from_elem(value: &T, n: usize) -> Result<VecDeque<T>, TryVecDequeError>
    where
        T: TryClone,
    {
        Self::try_from_elem(value, n)
    }

    /// Alias for [`Self::try_push_back`].
    fn fallible_push_back(&mut self, value: T) -> Result<(), TryReserveError> {
        Self::try_push_back(self, value)
    }

    /// Alias for [`Self::try_push_front`].
    fn fallible_push_front(&mut self, value: T) -> Result<(), TryReserveError> {
        Self::try_push_front(self, value)
    }

    /// Alias for [`Self::try_extend`].
    fn fallible_extend<I: IntoIterator<Item = T>>(
        &mut self,
        iter: I,
    ) -> Result<(), TryReserveError> {
        Self::try_extend(self, iter)
    }

    /// Alias for [`Self::try_collect`].
    fn fallible_collect<I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<VecDeque<T>, TryReserveError> {
        Self::try_collect(iter)
    }
}

impl<T> TryVecDeque<T> for VecDeque<T> {
    // ── Construction ────────────────────────────────────────────────────────

    fn try_with_capacity(capacity: usize) -> Result<VecDeque<T>, TryReserveError> {
        let mut deque = VecDeque::new();
        if capacity > 0 {
            deque.try_reserve(capacity)?;
        }
        Ok(deque)
    }

    fn try_from_elem(value: &T, n: usize) -> Result<VecDeque<T>, TryVecDequeError>
    where
        T: TryClone,
    {
        let mut deque = VecDeque::<T>::new();
        if n > 0 {
            deque
                .try_reserve(n)
                .map_err(|e| TryVecDequeError::Reserve(e.into()))?;
        }
        for _ in 0..n {
            deque.push_back(value.try_clone().map_err(TryVecDequeError::Clone)?);
        }
        Ok(deque)
    }

    fn try_from_elem_give_back(value: T, n: usize) -> Result<VecDeque<T>, (T, TryVecDequeError)>
    where
        T: TryClone,
    {
        match Self::try_from_elem(&value, n) {
            Ok(dq) => Ok(dq),
            Err(e) => Err((value, e)),
        }
    }

    fn try_collect<I: IntoIterator<Item = T>>(iter: I) -> Result<VecDeque<T>, TryReserveError> {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower);
        let mut deque = VecDeque::<T>::new();
        if capacity > 0 {
            deque.try_reserve(capacity)?;
        }
        for item in iter {
            if deque.len() == deque.capacity() {
                deque.try_reserve(1)?;
            }
            deque.push_back(item);
        }
        Ok(deque)
    }

    fn try_from_slice(slice: &[T]) -> Result<VecDeque<T>, TryVecDequeError>
    where
        T: TryClone,
    {
        let mut deque = VecDeque::<T>::new();
        if !slice.is_empty() {
            deque
                .try_reserve(slice.len())
                .map_err(|e| TryVecDequeError::Reserve(e.into()))?;
        }
        for item in slice {
            deque.push_back(item.try_clone().map_err(TryVecDequeError::Clone)?);
        }
        Ok(deque)
    }

    // ── Mutation: push / pop ────────────────────────────────────────────────

    fn try_push_back(&mut self, value: T) -> Result<(), TryReserveError> {
        self.try_reserve(1)?;
        self.push_back(value);
        Ok(())
    }

    fn try_push_back_give_back(&mut self, value: T) -> Result<(), (T, TryReserveError)> {
        match self.try_reserve(1) {
            Ok(()) => {
                self.push_back(value);
                Ok(())
            }
            Err(e) => Err((value, e.into())),
        }
    }

    fn try_push_front(&mut self, value: T) -> Result<(), TryReserveError> {
        self.try_reserve(1)?;
        self.push_front(value);
        Ok(())
    }

    fn try_push_front_give_back(&mut self, value: T) -> Result<(), (T, TryReserveError)> {
        match self.try_reserve(1) {
            Ok(()) => {
                self.push_front(value);
                Ok(())
            }
            Err(e) => Err((value, e.into())),
        }
    }

    fn try_pop_back(&mut self) -> Option<T> {
        self.pop_back()
    }

    // ── Mutation: insert / remove / extend ──────────────────────────────────

    fn try_insert(&mut self, index: usize, value: T) -> Result<(), TryVecDequeError> {
        if index > self.len() {
            return Err(TryVecDequeError::Other("insert index out of bounds"));
        }
        self.try_reserve(1)
            .map_err(|e| TryVecDequeError::Reserve(e.into()))?;
        self.insert(index, value);
        Ok(())
    }

    fn try_insert_give_back(
        &mut self,
        index: usize,
        value: T,
    ) -> Result<(), (T, TryVecDequeError)> {
        if index > self.len() {
            return Err((value, TryVecDequeError::Other("insert index out of bounds")));
        }
        match self.try_reserve(1) {
            Ok(()) => {
                self.insert(index, value);
                Ok(())
            }
            Err(e) => Err((value, TryVecDequeError::Reserve(e.into()))),
        }
    }

    fn try_remove(&mut self, index: usize) -> Result<Option<T>, TryVecDequeError> {
        if index >= self.len() {
            return Err(TryVecDequeError::Other("remove index out of bounds"));
        }
        // VecDeque::remove returns Option<T>. Since we validated bounds above,
        // this will always be Some, but we pass through the Option for safety.
        let val = self.remove(index);
        Ok(val)
    }

    fn try_extend<I: IntoIterator<Item = T>>(&mut self, iter: I) -> Result<(), TryReserveError> {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        if lower > 0 {
            self.try_reserve(lower)?;
        }
        for item in iter {
            if self.len() == self.capacity() {
                self.try_reserve(1)?;
            }
            self.push_back(item);
        }
        Ok(())
    }

    fn try_append(&mut self, other: &mut VecDeque<T>) -> Result<(), TryReserveError> {
        let extra = other.len();
        if extra == 0 {
            return Ok(());
        }
        self.try_reserve(extra)?;
        self.append(other);
        Ok(())
    }

    // ── Mutation: resize / shrink / clear ───────────────────────────────────

    fn try_resize_with<F>(&mut self, new_len: usize, mut f: F) -> Result<(), TryReserveError>
    where
        F: FnMut() -> T,
    {
        let current = self.len();
        if new_len <= current {
            self.truncate(new_len);
            return Ok(());
        }
        let extra = new_len - current;
        self.try_reserve(extra)?;
        for _ in 0..extra {
            self.push_back(f());
        }
        Ok(())
    }

    fn fallible_shrink_to_fit(&mut self) -> Result<(), TryVecDequeError> {
        <Self as TryVecDeque<T>>::fallible_shrink_to(self, self.len())
    }

    fn fallible_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryVecDequeError> {
        let target = core::cmp::max(self.len(), min_capacity);
        if self.capacity() <= target {
            return Ok(());
        }
        let mut spare = VecDeque::<T>::with_capacity(target);
        let len = self.len();
        std::mem::swap(self, &mut spare);
        if !self.is_empty() {
            self.try_reserve(len)
                .map_err(|e| TryVecDequeError::Reserve(e.into()))?;
        }
        for item in spare.drain(..) {
            self.push_back(item);
        }
        Ok(())
    }

    fn try_clear(&mut self) {
        self.clear();
    }
}

// ── TryClone for VecDeque<T> ─────────────────────────────────────────────────

impl<T: TryClone> TryClone for VecDeque<T> {
    fn try_clone(&self) -> Result<Self, crate::try_clone::TryCloneError> {
        let mut out = VecDeque::<T>::new();
        if !self.is_empty() {
            out.try_reserve(self.len())
                .map_err(|e| crate::try_clone::TryCloneError::Reserve(e.into()))?;
        }
        for elem in self.iter() {
            match elem.try_clone() {
                Ok(cloned) => out.push_back(cloned),
                Err(e) => {
                    drop(out);
                    return Err(e);
                }
            }
        }
        Ok(out)
    }
}

// ── TryDefault for VecDeque<T> ────────────────────────────────────────────────

impl<T: TryDefault> TryDefault for VecDeque<T> {
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(VecDeque::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_with_capacity_zero() {
        let dq: VecDeque<i32> = <VecDeque<i32> as TryVecDeque<i32>>::try_with_capacity(0).unwrap();
        assert!(dq.is_empty());
    }

    #[test]
    fn try_with_capacity_nonzero() {
        let dq: VecDeque<i32> = <VecDeque<i32> as TryVecDeque<i32>>::try_with_capacity(10).unwrap();
        assert!(dq.is_empty());
        assert!(dq.capacity() >= 10);
    }

    #[test]
    fn try_from_elem_single() {
        let dq: VecDeque<i32> = <VecDeque<i32> as TryVecDeque<i32>>::try_from_elem(&42, 1).unwrap();
        assert_eq!(dq.len(), 1);
        assert_eq!(dq[0], 42);
    }

    #[test]
    fn try_from_elem_multiple() {
        let elem = vec![1u8, 2];
        let dq: VecDeque<Vec<u8>> =
            <VecDeque<Vec<u8>> as TryVecDeque<Vec<u8>>>::try_from_elem(&elem, 3).unwrap();
        assert_eq!(dq.len(), 3);
    }

    #[test]
    fn try_from_elem_zero() {
        let dq: VecDeque<i32> = <VecDeque<i32> as TryVecDeque<i32>>::try_from_elem(&99, 0).unwrap();
        assert!(dq.is_empty());
    }

    #[test]
    fn try_push_back_appends() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(1).unwrap();
        dq.try_push_back(2).unwrap();
        assert_eq!(dq[0], 1);
        assert_eq!(dq[1], 2);
    }

    #[test]
    fn try_push_front_prepends() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_front(2).unwrap();
        dq.try_push_front(1).unwrap();
        assert_eq!(dq[0], 1);
        assert_eq!(dq[1], 2);
    }

    #[test]
    fn try_pop_back_returns_last() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(10).unwrap();
        dq.try_push_back(20).unwrap();
        assert_eq!(dq.try_pop_back(), Some(20));
        assert_eq!(dq.try_pop_back(), Some(10));
        assert_eq!(dq.try_pop_back(), None);
    }

    #[test]
    fn push_back_give_back_success() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back_give_back(42).unwrap();
        assert_eq!(dq[0], 42);
    }

    #[test]
    fn push_front_give_back_success() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_front_give_back(42).unwrap();
        assert_eq!(dq[0], 42);
    }

    #[test]
    fn try_insert_at_start() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(2).unwrap();
        dq.try_insert(0, 1).unwrap();
        assert_eq!(dq[0], 1);
        assert_eq!(dq[1], 2);
    }

    #[test]
    fn try_insert_out_of_bounds() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(1).unwrap();
        let err = dq.try_insert(5, 99).unwrap_err();
        matches!(err, TryVecDequeError::Other(_));
    }

    #[test]
    fn try_remove_middle() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(1).unwrap();
        dq.try_push_back(2).unwrap();
        dq.try_push_back(3).unwrap();
        assert_eq!(dq.try_remove(1).unwrap(), Some(2));
        assert_eq!(dq[0], 1);
        assert_eq!(dq[1], 3);
    }

    #[test]
    fn try_remove_out_of_bounds() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(1).unwrap();
        let err = dq.try_remove(5).unwrap_err();
        matches!(err, TryVecDequeError::Other(_));
    }

    #[test]
    fn try_extend_from_range() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_extend(0..5).unwrap();
        assert_eq!(dq.len(), 5);
    }

    #[test]
    fn try_extend_empty() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_extend(std::iter::empty::<i32>()).unwrap();
        assert!(dq.is_empty());
    }

    #[test]
    fn try_append_moves_elements() {
        let mut a: VecDeque<i32> = VecDeque::new();
        a.try_push_back(1).unwrap();
        let mut b: VecDeque<i32> = VecDeque::new();
        b.try_push_back(2).unwrap();
        b.try_push_back(3).unwrap();
        a.try_append(&mut b).unwrap();
        assert_eq!(a.len(), 3);
        assert!(b.is_empty());
    }

    #[test]
    fn try_append_both_empty() {
        let mut a: VecDeque<i32> = VecDeque::new();
        let mut b: VecDeque<i32> = VecDeque::new();
        a.try_append(&mut b).unwrap();
        assert!(a.is_empty());
    }

    #[test]
    fn try_resize_with_grow() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(1).unwrap();
        let mut counter = 10;
        dq.try_resize_with(4, || {
            counter += 1;
            counter
        })
        .unwrap();
        assert_eq!(dq.len(), 4);
        assert_eq!(dq[0], 1);
        assert_eq!(dq[1], 11);
    }

    #[test]
    fn try_resize_with_shrink() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(1).unwrap();
        dq.try_push_back(2).unwrap();
        dq.try_push_back(3).unwrap();
        dq.try_resize_with(1, || 99).unwrap();
        assert_eq!(dq.len(), 1);
        assert_eq!(dq[0], 1);
    }

    #[test]
    fn fallible_shrink_to_fit_reduces_excess() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_reserve(1024).unwrap();
        dq.try_push_back(1).unwrap();
        let cap_before = dq.capacity();
        assert!(cap_before >= 1024);
        dq.fallible_shrink_to_fit().unwrap();
        assert!(dq.capacity() < cap_before);
        assert_eq!(dq[0], 1);
    }

    #[test]
    fn try_clear_removes_all() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(1).unwrap();
        dq.try_push_back(2).unwrap();
        dq.try_clear();
        assert!(dq.is_empty());
    }

    #[test]
    fn try_collect_range() {
        let dq: VecDeque<u8> = <VecDeque<u8> as TryVecDeque<u8>>::try_collect(0..3).unwrap();
        assert_eq!(dq[0], 0);
        assert_eq!(dq[1], 1);
        assert_eq!(dq[2], 2);
    }

    #[test]
    fn try_collect_empty() {
        let dq: VecDeque<i32> =
            <VecDeque<i32> as TryVecDeque<i32>>::try_collect(std::iter::empty::<i32>()).unwrap();
        assert!(dq.is_empty());
    }

    #[test]
    fn try_from_slice_clones() {
        let slice: &[Vec<u8>] = &[vec![10], vec![20]];
        let dq: VecDeque<Vec<u8>> =
            <VecDeque<Vec<u8>> as TryVecDeque<Vec<u8>>>::try_from_slice(slice).unwrap();
        assert_eq!(dq[0], vec![10]);
        assert_eq!(dq[1], vec![20]);
    }

    #[test]
    fn try_from_slice_empty() {
        let dq: VecDeque<i32> = <VecDeque<i32> as TryVecDeque<i32>>::try_from_slice(&[]).unwrap();
        assert!(dq.is_empty());
    }

    #[test]
    fn try_clone_empty_deque() {
        let dq: VecDeque<i32> = VecDeque::new();
        assert!(
            crate::try_clone::TryClone::try_clone(&dq)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn try_clone_populated_deque() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.push_back(1);
        dq.push_back(2);
        dq.push_back(3);
        let c = crate::try_clone::TryClone::try_clone(&dq).unwrap();
        assert_eq!(c[0], 1);
        assert_eq!(c[1], 2);
        assert_eq!(c[2], 3);
    }

    #[test]
    fn try_default_empty_deque() {
        let dq: VecDeque<i32> = <VecDeque<i32> as TryDefault>::try_default().unwrap();
        assert!(dq.is_empty());
    }

    #[test]
    fn build_then_clone_then_default() {
        let mut dq: VecDeque<u32> = <VecDeque<u32> as TryDefault>::try_default().unwrap();
        dq.try_push_back(10).unwrap();
        dq.try_push_front(5).unwrap();
        let c = crate::try_clone::TryClone::try_clone(&dq).unwrap();
        assert_eq!(c[0], 5);
        assert_eq!(c[1], 10);
    }

    #[test]
    fn collect_then_append() {
        let a: VecDeque<i32> = <VecDeque<i32> as TryVecDeque<i32>>::try_collect(1..=3).unwrap();
        let mut b: VecDeque<i32> = <VecDeque<i32> as TryVecDeque<i32>>::try_collect(4..=6).unwrap();
        let mut combined = a;
        combined.try_append(&mut b).unwrap();
        assert_eq!(combined.len(), 6);
        assert!(b.is_empty());
    }
}

#[test]
fn try_with_capacity_nonzero() {
    let dq: VecDeque<i32> = <VecDeque<i32> as TryVecDeque<i32>>::try_with_capacity(10).unwrap();
    assert!(dq.is_empty());
    assert!(dq.capacity() >= 10);
}

#[test]
fn try_from_elem_single() {
    let dq: VecDeque<i32> = <VecDeque<i32> as TryVecDeque<i32>>::try_from_elem(&42, 1).unwrap();
    assert_eq!(dq.len(), 1);
    assert_eq!(dq[0], 42);
}

#[test]
fn try_from_elem_multiple() {
    let elem = vec![1u8, 2];
    let dq: VecDeque<Vec<u8>> =
        <VecDeque<Vec<u8>> as TryVecDeque<Vec<u8>>>::try_from_elem(&elem, 3).unwrap();
    assert_eq!(dq.len(), 3);
}

#[test]
fn try_from_elem_zero() {
    let dq: VecDeque<i32> = <VecDeque<i32> as TryVecDeque<i32>>::try_from_elem(&99, 0).unwrap();
    assert!(dq.is_empty());
}

// ── Push / Pop ───────────────────────────────────────────────────────────

#[test]
fn try_push_back_appends() {
    let mut dq: VecDeque<i32> = VecDeque::new();
    dq.try_push_back(1).unwrap();
    dq.try_push_back(2).unwrap();
    assert_eq!(dq[0], 1);
    assert_eq!(dq[1], 2);
}

#[test]
fn try_push_front_prepends() {
    let mut dq: VecDeque<i32> = VecDeque::new();
    dq.try_push_front(2).unwrap();
    dq.try_push_front(1).unwrap();
    assert_eq!(dq[0], 1);
    assert_eq!(dq[1], 2);
}

#[test]
fn try_pop_back_returns_last() {
    let mut dq: VecDeque<i32> = VecDeque::new();
    dq.try_push_back(10).unwrap();
    dq.try_push_back(20).unwrap();
    assert_eq!(dq.try_pop_back(), Some(20));
    assert_eq!(dq.try_pop_back(), Some(10));
    assert_eq!(dq.try_pop_back(), None);
}

#[test]
fn push_back_give_back_success() {
    let mut dq: VecDeque<i32> = VecDeque::new();
    dq.try_push_back_give_back(42).unwrap();
    assert_eq!(dq[0], 42);
}

#[test]
fn push_front_give_back_success() {
    let mut dq: VecDeque<i32> = VecDeque::new();
    dq.try_push_front_give_back(42).unwrap();
    assert_eq!(dq[0], 42);
}

// ── Insert / Remove ──────────────────────────────────────────────────────

#[test]
fn try_insert_at_start() {
    let mut dq: VecDeque<i32> = VecDeque::new();
    dq.try_push_back(2).unwrap();
    dq.try_insert(0, 1).unwrap();
    assert_eq!(dq[0], 1);
    assert_eq!(dq[1], 2);
}

#[test]
fn try_insert_at_end() {
    let mut dq: VecDeque<i32> = VecDeque::new();
    dq.try_push_back(1).unwrap();
    dq.try_insert(1, 2).unwrap();
    assert_eq!(dq[0], 1);
    assert_eq!(dq[1], 2);
}

#[test]
fn try_insert_out_of_bounds() {
    let mut dq: VecDeque<i32> = VecDeque::new();
    dq.try_push_back(1).unwrap();
    let err = dq.try_insert(5, 99).unwrap_err();
    matches!(err, TryVecDequeError::Other(_));
}

// ── Extend / Append ──────────────────────────────────────────────────────

#[test]
fn try_extend_from_range() {
    let mut dq: VecDeque<i32> = VecDeque::new();
    dq.try_extend(0..5).unwrap();
    assert_eq!(dq.len(), 5);
}

#[test]
fn try_extend_empty() {
    let mut dq: VecDeque<i32> = VecDeque::new();
    dq.try_extend(std::iter::empty::<i32>()).unwrap();
    assert!(dq.is_empty());
}

#[test]
fn try_append_moves_elements() {
    let mut a: VecDeque<i32> = VecDeque::new();
    a.try_push_back(1).unwrap();
    let mut b: VecDeque<i32> = VecDeque::new();
    b.try_push_back(2).unwrap();
    b.try_push_back(3).unwrap();
    a.try_append(&mut b).unwrap();
    assert_eq!(a.len(), 3);
    assert!(b.is_empty());
}

#[test]
fn try_append_both_empty() {
    let mut a: VecDeque<i32> = VecDeque::new();
    let mut b: VecDeque<i32> = VecDeque::new();
    a.try_append(&mut b).unwrap();
    assert!(a.is_empty());
}

// ── Resize / Shrink / Clear ──────────────────────────────────────────────

#[test]
fn try_resize_with_grow() {
    let mut dq: VecDeque<i32> = VecDeque::new();
    dq.try_push_back(1).unwrap();
    let mut counter = 10;
    dq.try_resize_with(4, || {
        counter += 1;
        counter
    })
    .unwrap();
    assert_eq!(dq.len(), 4);
    assert_eq!(dq[0], 1);
    assert_eq!(dq[1], 11);
}

#[test]
fn try_resize_with_shrink() {
    let mut dq: VecDeque<i32> = VecDeque::new();
    dq.try_push_back(1).unwrap();
    dq.try_push_back(2).unwrap();
    dq.try_push_back(3).unwrap();
    dq.try_resize_with(1, || 99).unwrap();
    assert_eq!(dq.len(), 1);
    assert_eq!(dq[0], 1);
}

#[test]
fn fallible_shrink_to_fit_reduces_excess() {
    let mut dq: VecDeque<i32> = VecDeque::new();
    dq.try_reserve(1024).unwrap();
    dq.try_push_back(1).unwrap();
    let cap_before = dq.capacity();
    assert!(cap_before >= 1024);
    dq.fallible_shrink_to_fit().unwrap();
    assert!(dq.capacity() < cap_before);
    assert_eq!(dq[0], 1);
}

#[test]
fn try_clear_removes_all() {
    let mut dq: VecDeque<i32> = VecDeque::new();
    dq.try_push_back(1).unwrap();
    dq.try_push_back(2).unwrap();
    dq.try_clear();
    assert!(dq.is_empty());
}
#[test]
fn try_remove_middle() {
    let mut dq: VecDeque<i32> = VecDeque::new();
    dq.try_push_back(1).unwrap();
    dq.try_push_back(2).unwrap();
    dq.try_push_back(3).unwrap();
    assert_eq!(dq.try_remove(1).unwrap(), Some(2));
    assert_eq!(dq[0], 1);
    assert_eq!(dq[1], 3);
}

#[test]
fn try_remove_out_of_bounds() {
    let mut dq: VecDeque<i32> = VecDeque::new();
    dq.try_push_back(1).unwrap();
    let err = dq.try_remove(5).unwrap_err();
    matches!(err, TryVecDequeError::Other(_));
}
