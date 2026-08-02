//! Fallible vector operations.
//!
//! Provides the [`TryVec`] trait with methods that mirror common `Vec` constructors
//! and mutating operations but return [`Result`] to handle allocation failures
//! gracefully, using [`std::collections::TryReserveError`] as the primary error type.
//!
//! # Design
//!
//! `TryVec` is implemented for `Vec<T>`. Methods that may grow internal capacity
//! (`push`, `insert`, `extend`, etc.) return a `Result` instead of panicking on
//! out-of-memory. Read-only accessors delegate directly to `Vec`.
//!
//! The trait also implements [`TryClone`](crate::try_clone::TryClone) and
//! [`TryDefault`](crate::try_default::TryDefault) for `Vec<T>` when `T` satisfies
//! the respective bounds.

use crate::alloc::AllocError;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use core::fmt;
use std::collections::TryReserveError;

/// Error returned by [`TryVec`] operations.
///
/// Wraps the two ways a vector operation can fail on stable Rust: a reserve
/// failure ([`TryReserveError`], returned by the inherent `Vec::try_reserve`)
/// or a clone failure ([`TryCloneError`]) when an element's `try_clone` cannot
/// allocate its internal buffers.
#[derive(Debug)]
pub enum TryVecError {
    /// A raw heap allocation failed (no collection involved).
    Alloc(AllocError),
    /// A capacity reservation on the vector failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An element clone failed during a method that requires `TryClone`.
    Clone(TryCloneError),
    /// An arithmetic overflow occurred while computing required capacity.
    Overflow,
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Display for TryVecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "vector operation failed: heap allocation error"),
            Self::Reserve(e) => write!(f, "vector operation failed: {}", e),
            Self::Clone(e) => write!(f, "vector operation failed: {}", e),
            Self::Overflow => write!(
                f,
                "vector operation failed: capacity calculation overflowed"
            ),
            Self::Other(msg) => write!(f, "vector operation failed: {}", msg),
        }
    }
}

impl From<AllocError> for TryVecError {
    fn from(_: AllocError) -> Self {
        Self::Alloc(AllocError)
    }
}

impl From<TryReserveError> for TryVecError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<TryCloneError> for TryVecError {
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

/// A trait for fallible vector operations.
///
/// Implemented for `Vec<T>`. Mirrors the most commonly-used `Vec` methods that can
/// fail due to allocation pressure, returning [`Result`] values that propagate
/// [`TryReserveError`] on failure.
pub trait TryVec<T>: Sized {
    // ── Construction ────────────────────────────────────────────────────────

    /// Fallibly construct a `Vec<T>` containing `value` cloned `capacity` times.
    ///
    /// Returns [`TryVecError::Reserve`] if the capacity allocation fails, or
    /// [`TryVecError::Clone`] if an element's [`TryClone::try_clone`] fails.
    /// Equivalent to `vec![value; capacity]` but fully fallible.
    fn try_from_elem(value: &T, capacity: usize) -> Result<Vec<T>, TryVecError>
    where
        T: TryClone;

    /// Like [`Self::try_from_elem`] but takes ownership of `value` and returns
    /// it on failure so the caller is not left empty-handed.
    fn try_from_elem_give_back(value: T, capacity: usize) -> Result<Vec<T>, (T, TryVecError)>
    where
        T: TryClone;

    // ── Mutation ────────────────────────────────────────────────────────────

    /// Fallibly append an element to the back of the vector.
    ///
    /// Returns [`TryReserveError`] if growing the internal buffer fails.
    fn try_push(&mut self, value: T) -> Result<(), TryReserveError>;

    /// Like [`Self::try_push`] but returns ownership of `value` back on failure.
    fn try_push_give_back(&mut self, value: T) -> Result<(), (T, TryReserveError)>;

    /// Fallibly insert an element at position `index`.
    ///
    /// Returns [`TryVecError::Reserve`] if growing the internal buffer fails, or
    /// [`TryVecError::Other`] if `index > len`.
    fn try_insert(&mut self, index: usize, value: T) -> Result<(), TryVecError>;

    /// Like [`Self::try_insert`] but returns ownership of `value` back on failure.
    fn try_insert_give_back(&mut self, index: usize, value: T) -> Result<(), (T, TryVecError)>;

    /// Fallibly extend the vector with all elements from an iterator.
    ///
    /// Uses the iterator's upper bound (if available) to reserve capacity upfront.
    /// Returns [`TryReserveError`] if the allocation fails.
    fn try_extend<I: IntoIterator<Item = T>>(&mut self, iter: I) -> Result<(), TryReserveError>;

    /// Fallibly append all elements from another slice by cloning each one.
    ///
    /// Returns [`TryVecError::Reserve`] on capacity failure or
    /// [`TryVecError::Clone`] if an element's [`TryClone::try_clone`] fails.
    fn try_extend_from_slice(&mut self, other: &[T]) -> Result<(), TryVecError>
    where
        T: TryClone;

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// This is the fallible analogue of [`Vec::append`]. Elements are moved from
    /// `other` rather than cloned. If `self` has spare capacity the transfer may
    /// happen without any new allocation; otherwise a single [`try_reserve`] call
    /// is made first so that failure is returned as [`TryReserveError`] instead
    /// of panicking.
    ///
    /// On success `other` is drained (length zero). On failure `other` is left
    /// untouched.
    ///
    /// [`try_reserve`]: https://doc.rust-lang.org/std/vec/struct.Vec.html#method.try_reserve
    fn try_append(&mut self, other: &mut Vec<T>) -> Result<(), TryReserveError>;

    /// Copies elements within the vector itself according to the given range.
    ///
    /// This is the fallible analogue of [`Vec::extend_from_within`]. The range
    /// `start..end` is copied into the back of the vector. Reserves capacity
    /// first so that allocation failures return [`TryVecError::Reserve`] instead
    /// of panicking. Uses [`TryClone`] for each copy so clone-time failures
    /// return [`TryVecError::Clone`] and the vector is rolled back to its
    /// pre-call state.
    ///
    /// Returns [`TryVecError::Other`] if the range is out of bounds.
    fn try_extend_from_within<R: std::ops::RangeBounds<usize>>(
        &mut self,
        range: R,
    ) -> Result<(), TryVecError>
    where
        T: TryClone;

    /// Resizes the vector in place so that `len` equals `new_len`.
    ///
    /// If `new_len` is greater than `len`, the vector is extended by cloning
    /// `value` via [`TryClone`]. If `new_len` is less than `len`, the vector
    /// is truncated. Returns [`TryVecError::Reserve`] on allocation failure or
    /// [`TryVecError::Clone`] if an element clone fails.
    fn try_resize(&mut self, value: &T, new_len: usize) -> Result<(), TryVecError>
    where
        T: TryClone;

    /// Like [`Self::try_resize`] but uses a closure to produce new elements.
    ///
    /// Reserve is attempted first so that allocation failures are returned
    /// cleanly. The closure is called only after capacity is secured.
    fn try_resize_with<F>(&mut self, new_len: usize, f: F) -> Result<(), TryReserveError>
    where
        F: FnMut() -> T;

    // ── Bulk construction ───────────────────────────────────────────────────

    /// Fallibly collect an iterator into a `Vec<T>`.
    ///
    /// Uses the iterator's size hint to pre-allocate when possible.
    fn try_collect<I: IntoIterator<Item = T>>(iter: I) -> Result<Vec<T>, TryReserveError>;

    /// Fallibly create a `Vec<T>` from a slice by cloning each element via
    /// [`TryClone`].
    ///
    /// Returns [`TryVecError::Reserve`] on capacity failure or
    /// [`TryVecError::Clone`] if an element's [`TryClone::try_clone`] fails.
    fn try_from_slice(slice: &[T]) -> Result<Vec<T>, TryVecError>
    where
        T: TryClone;
}

impl<T> TryVec<T> for Vec<T> {
    fn try_from_elem(value: &T, capacity: usize) -> Result<Vec<T>, TryVecError>
    where
        T: TryClone,
    {
        let mut vec = Vec::<T>::new();
        vec.try_reserve(capacity).map_err(TryVecError::Reserve)?;
        for _ in 0..capacity {
            vec.push(value.try_clone().map_err(TryVecError::Clone)?);
        }
        Ok(vec)
    }

    fn try_from_elem_give_back(value: T, capacity: usize) -> Result<Vec<T>, (T, TryVecError)>
    where
        T: TryClone,
    {
        match Self::try_from_elem(&value, capacity) {
            Ok(v) => Ok(v),
            Err(e) => Err((value, e)),
        }
    }

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

    fn try_insert(&mut self, index: usize, value: T) -> Result<(), TryVecError> {
        if index > self.len() {
            return Err(TryVecError::Other("insert index out of bounds"));
        }
        self.try_reserve(1).map_err(TryVecError::Reserve)?;
        self.insert(index, value);
        Ok(())
    }

    fn try_insert_give_back(&mut self, index: usize, value: T) -> Result<(), (T, TryVecError)> {
        if index > self.len() {
            return Err((value, TryVecError::Other("insert index out of bounds")));
        }
        match self.try_reserve(1) {
            Ok(()) => {
                self.insert(index, value);
                Ok(())
            }
            Err(e) => Err((value, TryVecError::Reserve(e))),
        }
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
            self.push(item);
        }
        Ok(())
    }

    fn try_extend_from_slice(&mut self, other: &[T]) -> Result<(), TryVecError>
    where
        T: TryClone,
    {
        if other.is_empty() {
            return Ok(());
        }
        let len_before = self.len();
        self.try_reserve(other.len())
            .map_err(TryVecError::Reserve)?;
        for item in other {
            match item.try_clone() {
                Ok(cloned) => self.push(cloned),
                Err(e) => {
                    self.truncate(len_before);
                    return Err(TryVecError::Clone(e));
                }
            }
        }
        Ok(())
    }

    fn try_append(&mut self, other: &mut Vec<T>) -> Result<(), TryReserveError> {
        let extra = other.len();
        if extra == 0 {
            return Ok(());
        }
        // Reserve first — lazy, no mutations until allocation succeeds.
        self.try_reserve(extra)?;
        // Now safe to call the inherent append; capacity is guaranteed.
        self.append(other);
        Ok(())
    }

    fn try_extend_from_within<R: std::ops::RangeBounds<usize>>(
        &mut self,
        range: R,
    ) -> Result<(), TryVecError>
    where
        T: TryClone,
    {
        use std::ops::Bound;

        let start = match range.start_bound() {
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i.checked_add(1).ok_or(TryVecError::Overflow)?,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&i) => i.checked_add(1).ok_or(TryVecError::Overflow)?,
            Bound::Excluded(&i) => i,
            Bound::Unbounded => self.len(),
        };

        if start >= end {
            return Ok(());
        }

        // Validate bounds before any mutation.
        if end > self.len() || start > self.len() {
            return Err(TryVecError::Other(
                "extend_from_within: range out of bounds",
            ));
        }

        let count = end - start;
        let len_before = self.len();
        // Reserve first — lazy, no element copies until allocation succeeds.
        self.try_reserve(count).map_err(TryVecError::Reserve)?;
        for i in start..end {
            match self[i].try_clone() {
                Ok(cloned) => self.push(cloned),
                Err(e) => {
                    self.truncate(len_before);
                    return Err(TryVecError::Clone(e));
                }
            }
        }
        Ok(())
    }

    fn try_resize(&mut self, value: &T, new_len: usize) -> Result<(), TryVecError>
    where
        T: TryClone,
    {
        let current = self.len();
        if new_len <= current {
            // Truncation never allocates.
            self.truncate(new_len);
            return Ok(());
        }
        let extra = new_len - current;
        // Reserve first — lazy.
        self.try_reserve(extra).map_err(TryVecError::Reserve)?;
        // Clone elements one-by-one into pre-reserved space.
        for _ in 0..extra {
            match value.try_clone() {
                Ok(cloned) => self.push(cloned),
                Err(e) => {
                    self.truncate(current);
                    return Err(TryVecError::Clone(e));
                }
            }
        }
        Ok(())
    }

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
        // Reserve first — lazy, closure not called until allocation succeeds.
        self.try_reserve(extra)?;
        for _ in 0..extra {
            self.push(f());
        }
        Ok(())
    }

    fn try_collect<I: IntoIterator<Item = T>>(iter: I) -> Result<Vec<T>, TryReserveError> {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower);
        let mut vec = Vec::<T>::new();
        if capacity > 0 {
            vec.try_reserve(capacity)?;
        }
        for item in iter {
            // Iterator may yield more elements than its hint promised.
            if vec.len() == vec.capacity() {
                vec.try_reserve(1)?;
            }
            vec.push(item);
        }
        Ok(vec)
    }

    fn try_from_slice(slice: &[T]) -> Result<Vec<T>, TryVecError>
    where
        T: TryClone,
    {
        let mut vec = Vec::<T>::new();
        vec.try_reserve(slice.len()).map_err(TryVecError::Reserve)?;
        for item in slice {
            vec.push(item.try_clone().map_err(TryVecError::Clone)?);
        }
        Ok(vec)
    }
}

// ── TryClone for Vec<T> ──────────────────────────────────────────────────────

impl<T: TryClone> TryClone for Vec<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let mut out = Vec::<T>::new();
        if !self.is_empty() {
            out.try_reserve(self.len())
                .map_err(TryCloneError::Reserve)?;
        }
        for elem in self.iter() {
            match elem.try_clone() {
                Ok(cloned) => {
                    out.push(cloned);
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

// ── TryDefault for Vec<T> ────────────────────────────────────────────────────

impl<T: TryDefault> TryDefault for Vec<T> {
    fn try_default() -> Result<Self, TryDefaultError> {
        // An empty Vec requires no allocation.
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn try_from_elem_single() {
        let v = Vec::<u8>::try_from_elem(&42, 1).unwrap();
        assert_eq!(v, [42]);
    }

    #[test]
    fn try_from_elem_multiple() {
        let elem = vec![1u8, 2];
        let v = Vec::<Vec<u8>>::try_from_elem(&elem, 3).unwrap();
        assert_eq!(v, vec![vec![1, 2], vec![1, 2], vec![1, 2]]);
    }

    #[test]
    fn try_from_elem_zero() {
        let v: Vec<i32> = Vec::<i32>::try_from_elem(&99, 0).unwrap();
        assert!(v.is_empty());
    }

    // ── Mutation ─────────────────────────────────────────────────────────────

    #[test]
    fn try_push_appends_element() {
        let mut v: Vec<i32> = Vec::new();
        v.try_push(1).unwrap();
        v.try_push(2).unwrap();
        assert_eq!(v, [1, 2]);
    }

    #[test]
    fn try_insert_at_start() {
        let mut v: Vec<i32> = Vec::new();
        v.try_push(2).unwrap();
        v.try_insert(0, 1).unwrap();
        assert_eq!(v, [1, 2]);
    }

    #[test]
    fn try_insert_at_end() {
        let mut v: Vec<i32> = Vec::new();
        v.try_push(1).unwrap();
        v.try_insert(1, 2).unwrap();
        assert_eq!(v, [1, 2]);
    }

    #[test]
    fn try_insert_middle() {
        let mut v: Vec<i32> = Vec::new();
        v.try_push(1).unwrap();
        v.try_push(3).unwrap();
        v.try_insert(1, 2).unwrap();
        assert_eq!(v, [1, 2, 3]);
    }

    #[test]
    fn try_extend_from_iterator() {
        let mut v: Vec<i32> = Vec::new();
        v.try_extend(0..5).unwrap();
        assert_eq!(v, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn try_extend_empty() {
        let mut v: Vec<i32> = Vec::new();
        v.try_extend(std::iter::empty::<i32>()).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn try_extend_existing() {
        let mut v: Vec<i32> = Vec::new();
        v.try_push(1).unwrap();
        v.try_extend([2, 3]).unwrap();
        assert_eq!(v, [1, 2, 3]);
    }

    #[test]
    fn try_extend_from_slice_clones_elements() {
        let mut v: Vec<Vec<u8>> = Vec::new();
        v.try_push(vec![1]).unwrap();
        let slice: &[Vec<u8>] = &[vec![2], vec![3]];
        v.try_extend_from_slice(slice).unwrap();
        assert_eq!(v, [vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn try_extend_from_slice_empty() {
        let mut v: Vec<i32> = Vec::new();
        v.try_extend_from_slice(&[]).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn try_append_moves_elements() {
        let mut v: Vec<i32> = Vec::new();
        v.try_push(1).unwrap();
        let mut other: Vec<i32> = Vec::new();
        other.try_push(2).unwrap();
        other.try_push(3).unwrap();
        v.try_append(&mut other).unwrap();
        assert_eq!(v, [1, 2, 3]);
        assert!(other.is_empty());
    }

    #[test]
    fn try_append_into_empty() {
        let mut v: Vec<i32> = Vec::new();
        let mut other: Vec<i32> = Vec::new();
        other.try_push(42).unwrap();
        v.try_append(&mut other).unwrap();
        assert_eq!(v, [42]);
        assert!(other.is_empty());
    }

    #[test]
    fn try_append_both_empty() {
        let mut v: Vec<i32> = Vec::new();
        let mut other: Vec<i32> = Vec::new();
        v.try_append(&mut other).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn try_extend_from_within_full() {
        let mut v: Vec<Vec<u8>> = vec![vec![10], vec![20], vec![30]];
        v.try_extend_from_within(0..2).unwrap();
        assert_eq!(v, [vec![10], vec![20], vec![30], vec![10], vec![20]]);
    }

    #[test]
    fn try_extend_from_within_single() {
        let mut v: Vec<Option<u32>> = vec![Some(5)];
        v.try_extend_from_within(..).unwrap();
        assert_eq!(v, [Some(5), Some(5)]);
    }

    #[test]
    fn try_extend_from_within_empty_range() {
        let mut v: Vec<i32> = vec![1, 2, 3];
        v.try_extend_from_within(1..1).unwrap();
        assert_eq!(v, [1, 2, 3]);
    }

    #[test]
    fn try_extend_from_within_unbounded() {
        let mut v: Vec<Option<u8>> = vec![Some(1), Some(2)];
        v.try_extend_from_within(..).unwrap();
        assert_eq!(v, [Some(1), Some(2), Some(1), Some(2)]);
    }

    #[test]
    fn try_resize_shrink() {
        let mut v: Vec<i32> = vec![1, 2, 3, 4];
        v.try_resize(&0, 2).unwrap();
        assert_eq!(v, [1, 2]);
    }

    #[test]
    fn try_resize_grow() {
        let elem = vec![9u8];
        let mut v: Vec<Vec<u8>> = vec![vec![1]];
        v.try_resize(&elem, 3).unwrap();
        assert_eq!(v, [vec![1], vec![9], vec![9]]);
    }

    #[test]
    fn try_resize_noop() {
        let mut v: Vec<i32> = vec![1, 2];
        v.try_resize(&0, 2).unwrap();
        assert_eq!(v, [1, 2]);
    }

    #[test]
    fn try_resize_with_grow() {
        let mut v: Vec<i32> = vec![1];
        let mut counter = 10;
        v.try_resize_with(4, || {
            counter += 1;
            counter
        })
        .unwrap();
        assert_eq!(v, [1, 11, 12, 13]);
    }

    #[test]
    fn try_resize_with_shrink() {
        let mut v: Vec<i32> = vec![1, 2, 3];
        v.try_resize_with(1, || 99).unwrap();
        assert_eq!(v, [1]);
    }

    // ── Give-back variants ───────────────────────────────────────────────────

    #[test]
    fn try_push_give_back_success_returns_unit() {
        let mut v: Vec<i32> = Vec::new();
        v.try_push_give_back(42).unwrap();
        assert_eq!(v, [42]);
    }

    #[test]
    fn try_insert_give_back_success() {
        let mut v: Vec<i32> = Vec::new();
        v.try_push(2).unwrap();
        v.try_insert_give_back(0, 1).unwrap();
        assert_eq!(v, [1, 2]);
    }

    #[test]
    fn try_from_elem_give_back_success() {
        let elem = vec![5u8];
        let v = Vec::<Vec<u8>>::try_from_elem_give_back(elem.clone(), 2).unwrap();
        assert_eq!(v, [vec![5], vec![5]]);
    }

    #[test]
    fn try_from_elem_give_back_returns_value_on_err_type() {
        // We can't easily force an OOM, but we can verify the error type shape.
        let elem = vec![1u8, 2];
        let result: Result<Vec<Vec<u8>>, (Vec<u8>, TryVecError)> =
            Vec::try_from_elem_give_back(elem.clone(), 0);
        assert!(result.is_ok());
    }

    #[test]
    fn try_append_give_back_success() {
        let mut v: Vec<i32> = Vec::new();
        v.try_push(1).unwrap();
        let mut other: Vec<i32> = Vec::new();
        other.try_push(2).unwrap();
        v.try_append(&mut other).unwrap();
        assert_eq!(v, [1, 2]);
    }

    #[test]
    fn try_append_error_type_shape() {
        let mut v: Vec<i32> = Vec::new();
        let mut other: Vec<i32> = vec![99];
        let result: Result<(), TryReserveError> = v.try_append(&mut other);
        assert!(result.is_ok());
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    #[test]
    fn try_collect_range() {
        let v: Vec<u8> = Vec::try_collect(0..3).unwrap();
        assert_eq!(v, [0, 1, 2]);
    }

    #[test]
    fn try_collect_empty() {
        let v: Vec<i32> = Vec::try_collect(std::iter::empty::<i32>()).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn try_collect_vecs() {
        let items = vec![vec![1u8], vec![2]];
        let v: Vec<Vec<u8>> = Vec::try_collect(items).unwrap();
        assert_eq!(v, [vec![1], vec![2]]);
    }

    #[test]
    fn try_from_slice_clones() {
        let slice: &[Vec<u8>] = &[vec![10], vec![20]];
        let v: Vec<Vec<u8>> = Vec::try_from_slice(slice).unwrap();
        assert_eq!(v, [vec![10], vec![20]]);
    }

    #[test]
    fn try_from_slice_empty() {
        let v: Vec<i32> = Vec::try_from_slice(&[]).unwrap();
        assert!(v.is_empty());
    }

    // ── TryClone ─────────────────────────────────────────────────────────────

    #[test]
    fn try_clone_empty_vec() {
        let v: Vec<i32> = Vec::new();
        assert!(v.try_clone().unwrap().is_empty());
    }

    #[test]
    fn try_clone_populated_vec() {
        let v = vec![1i32, 2, 3];
        assert_eq!(v.try_clone().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn try_clone_with_options() {
        let v: Vec<Option<u32>> = vec![Some(1), Some(2)];
        let c = v.try_clone().unwrap();
        assert_eq!(c, [Some(1), Some(2)]);
    }

    #[test]
    fn try_clone_nested_vecs() {
        let v: Vec<Vec<u8>> = vec![vec![1, 2], vec![3, 4]];
        let c = v.try_clone().unwrap();
        assert_eq!(c, vec![vec![1, 2], vec![3, 4]]);
    }

    // ── TryDefault ───────────────────────────────────────────────────────────

    #[test]
    fn try_default_empty_vec() {
        let v: Vec<i32> = Vec::try_default().unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn try_default_vec_of_options() {
        let v: Vec<Option<i32>> = Vec::try_default().unwrap();
        assert!(v.is_empty());
    }

    // ── Combined workflows ───────────────────────────────────────────────────

    #[test]
    fn build_then_clone_then_default() {
        let mut v: Vec<u32> = Vec::try_default().unwrap();
        v.try_push(10).unwrap();
        v.try_push(20).unwrap();
        let c = v.try_clone().unwrap();
        assert_eq!(c, [10, 20]);
    }

    #[test]
    fn collect_then_append() {
        let a: Vec<i32> = Vec::try_collect(1..=3).unwrap();
        let mut b: Vec<i32> = Vec::try_collect(4..=6).unwrap();
        let mut combined = a;
        combined.try_append(&mut b).unwrap();
        assert_eq!(combined, [1, 2, 3, 4, 5, 6]);
        assert!(b.is_empty());
    }
}
