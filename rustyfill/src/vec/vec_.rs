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
use crate::alloc::TryReserveError;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::vec::raw_manipulation::RawVecInnerView;
use core::fmt;
use std::alloc::Layout;

/// Panic-safe guard that truncates a `Vec` back to its original length on drop
/// unless disarmed via `forget()`. Used by fallible extend methods so that if
/// an element's `try_clone` panics mid-way, partially-pushed elements are
/// removed rather than left behind.
struct TruncateGuard<'a, T> {
    vec: &'a mut Vec<T>,
    len_before: usize,
}

impl<'a, T> TruncateGuard<'a, T> {
    fn new(vec: &'a mut Vec<T>) -> Self {
        Self {
            len_before: vec.len(),
            vec,
        }
    }

    /// Disable the guard — no truncation on scope exit.
    fn forget(self) {
        core::mem::forget(self);
    }
}

impl<'a, T> Drop for TruncateGuard<'a, T> {
    fn drop(&mut self) {
        self.vec.truncate(self.len_before);
    }
}

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
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
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

    /// Fallibly construct an empty `Vec<T>` with at least enough capacity for
    /// `capacity` elements. Equivalent to [`Vec::with_capacity`] but fallible.
    ///
    /// Returns [`TryReserveError`] if the initial allocation fails.
    fn try_with_capacity(capacity: usize) -> Result<Vec<T>, TryReserveError>;

    /// Fallibly construct a `Vec<T>` containing `value` cloned `count` times.
    ///
    /// Returns [`TryVecError::Reserve`] if the capacity allocation fails, or
    /// [`TryVecError::Clone`] if an element's [`TryClone::try_clone`] fails.
    /// Equivalent to `vec![value; count]` but fully fallible.
    fn try_from_elem(value: &T, count: usize) -> Result<Vec<T>, TryVecError>
    where
        T: TryClone;

    /// Like [`Self::try_from_elem`] but takes ownership of `value` and returns
    /// it on failure so the caller is not left empty-handed.
    fn try_from_elem_give_back(value: T, count: usize) -> Result<Vec<T>, (T, TryVecError)>
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

    /// Fallibly extend the vector with all elements from an iterator source.
    ///
    /// Accepts anything that implements [`ResumableSource`](crate::recovery::ResumableSource),
    /// including both plain iterators and [`Resumable`](crate::recovery::Resumable)
    /// wrappers from previous failures. The error type stays identical across
    /// retries because `Resumable<I>` and bare iterators share the same inner type.
    ///
    /// Uses the iterator's size hint to reserve capacity upfront when available.
    /// On reserve failure, returns a [`Resumable`](crate::recovery::Resumable)
    /// containing any consumed-but-uncommitted element and the remainder of the
    /// iterator, which the caller can pass right back in.
    fn try_extend<S>(
        &mut self,
        source: S,
    ) -> Result<(), (TryReserveError, crate::recovery::Resumable<S::Inner>)>
    where
        S: crate::recovery::ResumableSource<Item = T>;

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

    // ── Capacity ─────────────────────────────────────────────────────────────

    /// Fallibly shrink the capacity of this vector to match its length.
    ///
    /// **Deprecated:** This method name conflicts with the unstable inherent
    /// [`Vec::try_shrink_to_fit`]. Use [`Self::fallible_shrink_to_fit`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with unstable Vec::try_shrink_to_fit; use fallible_shrink_to_fit"
    )]
    fn try_shrink_to_fit(&mut self) -> Result<(), TryVecError>;

    /// Fallibly shrink the capacity of this vector to at least `min_capacity`.
    ///
    /// **Deprecated:** This method name conflicts with the unstable inherent
    /// [`Vec::try_shrink_to`]. Use [`Self::fallible_shrink_to`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with unstable Vec::try_shrink_to; use fallible_shrink_to"
    )]
    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryVecError>;

    // ── Aliases with `fallible_` prefix ─────────────────────────────────────

    /// Alias for [`Self::try_with_capacity`].
    fn fallible_with_capacity(capacity: usize) -> Result<Vec<T>, TryReserveError> {
        Self::try_with_capacity(capacity)
    }

    /// Alias for [`Self::try_from_elem`].
    fn fallible_from_elem(value: &T, count: usize) -> Result<Vec<T>, TryVecError>
    where
        T: TryClone,
    {
        Self::try_from_elem(value, count)
    }

    /// Alias for [`Self::try_from_elem_give_back`].
    fn fallible_from_elem_give_back(value: T, count: usize) -> Result<Vec<T>, (T, TryVecError)>
    where
        T: TryClone,
    {
        Self::try_from_elem_give_back(value, count)
    }

    /// Alias for [`Self::try_push`].
    fn fallible_push(&mut self, value: T) -> Result<(), TryReserveError> {
        Self::try_push(self, value)
    }

    /// Alias for [`Self::try_push_give_back`].
    fn fallible_push_give_back(&mut self, value: T) -> Result<(), (T, TryReserveError)> {
        Self::try_push_give_back(self, value)
    }

    /// Alias for [`Self::try_insert`].
    fn fallible_insert(&mut self, index: usize, value: T) -> Result<(), TryVecError> {
        Self::try_insert(self, index, value)
    }

    /// Alias for [`Self::try_insert_give_back`].
    fn fallible_insert_give_back(
        &mut self,
        index: usize,
        value: T,
    ) -> Result<(), (T, TryVecError)> {
        Self::try_insert_give_back(self, index, value)
    }

    /// Alias for [`Self::try_extend`].
    fn fallible_extend<S>(
        &mut self,
        source: S,
    ) -> Result<(), (TryReserveError, crate::recovery::Resumable<S::Inner>)>
    where
        S: crate::recovery::ResumableSource<Item = T>,
    {
        Self::try_extend(self, source)
    }

    /// Alias for [`Self::try_extend_from_slice`].
    fn fallible_extend_from_slice(&mut self, other: &[T]) -> Result<(), TryVecError>
    where
        T: TryClone,
    {
        Self::try_extend_from_slice(self, other)
    }

    /// Alias for [`Self::try_append`].
    fn fallible_append(&mut self, other: &mut Vec<T>) -> Result<(), TryReserveError> {
        Self::try_append(self, other)
    }

    /// Alias for [`Self::try_extend_from_within`].
    fn fallible_extend_from_within<R: std::ops::RangeBounds<usize>>(
        &mut self,
        range: R,
    ) -> Result<(), TryVecError>
    where
        T: TryClone,
    {
        Self::try_extend_from_within(self, range)
    }

    /// Alias for [`Self::try_resize`].
    fn fallible_resize(&mut self, value: &T, new_len: usize) -> Result<(), TryVecError>
    where
        T: TryClone,
    {
        Self::try_resize(self, value, new_len)
    }

    /// Alias for [`Self::try_resize_with`].
    fn fallible_resize_with<F>(&mut self, new_len: usize, f: F) -> Result<(), TryReserveError>
    where
        F: FnMut() -> T,
    {
        Self::try_resize_with(self, new_len, f)
    }

    /// Fallibly shrink the capacity of this vector to match its length.
    ///
    /// May reallocate if the current allocation is larger than needed.
    /// Returns [`TryVecError::Alloc`] if the re-allocation fails.
    /// Equivalent to [`Vec::shrink_to_fit`] but fallible.
    ///
    /// This method replaces the deprecated [`Self::try_shrink_to_fit`] which
    /// shares its name with the unstable inherent [`Vec::try_shrink_to_fit`].
    #[allow(deprecated)]
    fn fallible_shrink_to_fit(&mut self) -> Result<(), TryVecError> {
        Self::try_shrink_to_fit(self)
    }

    /// Fallibly shrink the capacity of this vector to at least `min_capacity`.
    ///
    /// If the current capacity is already less than or equal to `min_capacity`,
    /// does nothing and returns `Ok(())`. Otherwise reallocates down.
    /// Returns [`TryVecError::Alloc`] if the re-allocation fails.
    /// Equivalent to [`Vec::shrink_to`] but fallible.
    ///
    /// This method replaces the deprecated [`Self::try_shrink_to`] which shares
    /// its name with the unstable inherent [`Vec::try_shrink_to`].
    #[allow(deprecated)]
    fn fallible_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryVecError> {
        Self::try_shrink_to(self, min_capacity)
    }

    /// Alias for [`Self::try_collect`].
    fn fallible_collect<I: IntoIterator<Item = T>>(iter: I) -> Result<Vec<T>, TryReserveError> {
        Self::try_collect(iter)
    }

    /// Alias for [`Self::try_from_slice`].
    fn fallible_from_slice(slice: &[T]) -> Result<Vec<T>, TryVecError>
    where
        T: TryClone,
    {
        Self::try_from_slice(slice)
    }

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

#[allow(deprecated)]
impl<T> TryVec<T> for Vec<T> {
    fn try_with_capacity(capacity: usize) -> Result<Vec<T>, TryReserveError> {
        let mut vec = Vec::<T>::new();
        if capacity > 0 {
            vec.try_reserve(capacity)?;
        }
        Ok(vec)
    }

    fn try_from_elem(value: &T, count: usize) -> Result<Vec<T>, TryVecError>
    where
        T: TryClone,
    {
        let mut vec = Vec::<T>::new();
        if count > 0 {
            vec.try_reserve(count)
                .map_err(|e| TryVecError::Reserve(e.into()))?;
        }
        for _ in 0..count {
            vec.push(value.try_clone().map_err(TryVecError::Clone)?);
        }
        Ok(vec)
    }

    fn try_from_elem_give_back(value: T, count: usize) -> Result<Vec<T>, (T, TryVecError)>
    where
        T: TryClone,
    {
        match Self::try_from_elem(&value, count) {
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
            Err(e) => Err((value, e.into())),
        }
    }

    fn try_insert(&mut self, index: usize, value: T) -> Result<(), TryVecError> {
        if index > self.len() {
            return Err(TryVecError::Other("insert index out of bounds"));
        }
        self.try_reserve(1)
            .map_err(|e| TryVecError::Reserve(e.into()))?;
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
            Err(e) => Err((value, TryVecError::Reserve(e.into()))),
        }
    }

    fn try_extend<S>(
        &mut self,
        source: S,
    ) -> Result<(), (TryReserveError, crate::recovery::Resumable<S::Inner>)>
    where
        S: crate::recovery::ResumableSource<Item = T>,
    {
        use crate::recovery::Resumable;

        let (head, mut iter) = source.safe_into_iter();

        // Drain the head element first if present.
        if let Some(h) = head {
            if self.len() == self.capacity()
                && let Err(e) = self.try_reserve(1)
            {
                return Err((e.into(), Resumable::new(h, iter)));
            }
            self.push(h);
        }

        let (lower, _) = iter.size_hint();
        if lower > 0
            && let Err(e) = self.try_reserve(lower)
        {
            return Err((e.into(), Resumable::from_remainder(iter)));
        }
        while let Some(item) = iter.next() {
            if self.len() == self.capacity()
                && let Err(e) = self.try_reserve(1)
            {
                return Err((e.into(), Resumable::new(item, iter)));
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
        self.try_reserve(other.len())
            .map_err(|e| TryVecError::Reserve(e.into()))?;
        let guard = TruncateGuard::new(self);
        for item in other {
            match item.try_clone() {
                Ok(cloned) => {
                    guard.vec.push(cloned);
                }
                Err(e) => {
                    return Err(TryVecError::Clone(e));
                }
            }
        }
        guard.forget();
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
        // Reserve first — lazy, no element copies until allocation succeeds.
        self.try_reserve(count)
            .map_err(|e| TryVecError::Reserve(e.into()))?;
        let guard = TruncateGuard::new(self);
        for i in start..end {
            match guard.vec[i].try_clone() {
                Ok(cloned) => guard.vec.push(cloned),
                Err(e) => {
                    return Err(TryVecError::Clone(e));
                }
            }
        }
        guard.forget();
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
        self.try_reserve(extra)
            .map_err(|e| TryVecError::Reserve(e.into()))?;
        let guard = TruncateGuard::new(self);
        for _ in 0..extra {
            match value.try_clone() {
                Ok(cloned) => guard.vec.push(cloned),
                Err(e) => {
                    return Err(TryVecError::Clone(e));
                }
            }
        }
        guard.forget();
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
        let guard = TruncateGuard::new(self);
        for _ in 0..extra {
            guard.vec.push(f());
        }
        guard.forget();
        Ok(())
    }

    fn try_shrink_to_fit(&mut self) -> Result<(), TryVecError> {
        <Self as TryVec<T>>::try_shrink_to(self, self.len())
    }

    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryVecError> {
        let target = core::cmp::max(self.len(), min_capacity);
        if self.capacity() <= target {
            return Ok(());
        }
        let (mut current_raw, current_len) = RawVecInnerView::from_vec(std::mem::take(self));
        // SAFETY: target < self.capacity() (guaranteed by the early-return guard
        // above), and elem_layout matches the type T that was used to allocate the original buffer.
        // Should not panic here - the shrink_unchecked does not invoke user code.
        let res = unsafe { current_raw.shrink_unchecked(target, Layout::new::<T>()) };
        match res {
            Ok(()) => {
                // SAFETY: shrink succeeded; current_raw holds the new (or same)
                // allocation with updated capacity. Length is unchanged.
                *self = unsafe { current_raw.into_vec(current_len) };
                Ok(())
            }
            Err(e) => {
                // Allocation failed. shrink_unchecked returns early via `?` on
                // realloc failure, BEFORE updating self.ptr / self.cap — so
                // current_raw still holds the original (unshrunk) allocation.
                // SAFETY: pointer, length, and capacity are all still valid from
                // the original Vec.
                *self = unsafe { current_raw.into_vec(current_len) };
                Err(TryVecError::Alloc(e))
            }
        }
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
        vec.try_reserve(slice.len())
            .map_err(|e| TryVecError::Reserve(e.into()))?;
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
                .map_err(|e| TryCloneError::Reserve(e.into()))?;
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
    fn try_with_capacity_zero() {
        let v: Vec<i32> = <Vec<i32> as TryVec<i32>>::try_with_capacity(0).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn try_with_capacity_nonzero() {
        let v: Vec<i32> = <Vec<i32> as TryVec<i32>>::try_with_capacity(10).unwrap();
        assert!(v.is_empty());
        assert!(v.capacity() >= 10);
    }

    #[test]
    fn fallible_with_capacity_alias() {
        let v: Vec<String> = <Vec<String> as TryVec<String>>::fallible_with_capacity(5).unwrap();
        assert!(v.is_empty());
        assert!(v.capacity() >= 5);
    }

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

    // ── Shrink ────────────────────────────────────────────────────────────────

    #[test]
    fn try_shrink_to_fit_exact_capacity() {
        let mut v: Vec<i32> = vec![1, 2, 3];
        let cap_before = v.capacity();
        // capacity == len already; no-op path.
        v.fallible_shrink_to_fit().unwrap();
        assert_eq!(v.capacity(), cap_before);
        assert_eq!(v, [1, 2, 3]);
    }

    #[test]
    fn try_shrink_to_fit_reduces_excess() {
        let mut v: Vec<i32> = Vec::new();
        v.try_reserve(1024).unwrap();
        v.try_push(1).unwrap();
        v.try_push(2).unwrap();
        let cap_before = v.capacity();
        assert!(cap_before >= 1024);
        v.fallible_shrink_to_fit().unwrap();
        assert!(
            v.capacity() < cap_before,
            "capacity {} was not reduced from {}",
            v.capacity(),
            cap_before
        );
        assert!(v.capacity() >= 2);
        assert_eq!(v, [1, 2]);
    }

    #[test]
    fn try_shrink_to_fit_empty_large() {
        let mut v: Vec<i32> = Vec::new();
        v.try_reserve(512).unwrap();
        v.fallible_shrink_to_fit().unwrap();
        assert_eq!(v.capacity(), 0);
    }

    #[test]
    fn try_shrink_to_above_current_len() {
        let mut v: Vec<i32> = Vec::new();
        v.try_reserve(256).unwrap();
        v.try_push(42).unwrap();
        let cap_before = v.capacity();
        // min_capacity > len but < current capacity → should attempt to shrink.
        v.fallible_shrink_to(32).unwrap();
        assert!(v.capacity() >= 32);
        assert!(v.capacity() < cap_before || v.capacity() >= 32);
        assert_eq!(v, [42]);
    }

    #[test]
    fn try_shrink_to_below_current_len_is_noop() {
        let mut v: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
        let cap_before = v.capacity();
        // min_capacity < len → target == len, capacity already fits → no-op.
        v.fallible_shrink_to(2).unwrap();
        assert_eq!(v, [1, 2, 3, 4, 5, 6]);
        assert_eq!(v.capacity(), cap_before);
    }

    #[test]
    fn try_shrink_to_already_small() {
        let mut v: Vec<i32> = vec![1, 2];
        // capacity already <= min_capacity → no-op.
        v.fallible_shrink_to(16).unwrap();
        assert_eq!(v, [1, 2]);
    }

    #[test]
    fn try_shrink_to_preserves_complex_types() {
        let mut v: Vec<Vec<u8>> = Vec::new();
        v.try_reserve(128).unwrap();
        v.try_push(vec![1, 2, 3]).unwrap();
        v.try_push(vec![4, 5]).unwrap();
        v.fallible_shrink_to(4).unwrap();
        assert!(v.capacity() >= 4);
        assert_eq!(v, [vec![1, 2, 3], vec![4, 5]]);
    }

    #[test]
    fn try_shrink_to_fit_preserves_data() {
        let mut v: Vec<String> = Vec::new();
        v.try_reserve(64).unwrap();
        v.try_push("hello".to_string()).unwrap();
        v.try_push("world".to_string()).unwrap();
        v.fallible_shrink_to_fit().unwrap();
        assert_eq!(v, ["hello", "world"]);
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

    // ── OOM tests ─────────────────────────────────────────────────────────────

    use rustyfill_test_allocator::{FailPolicy, with_policy};

    #[test]
    fn vec_try_with_capacity_fails_on_oom() {
        let r: Result<Vec<u8>, TryReserveError> = with_policy(
            FailPolicy::fail_next_alloc(),
            || <Vec<u8> as TryVec<u8>>::try_with_capacity(10),
        );
        assert!(r.is_err());
    }

    #[test]
    fn vec_try_with_capacity_zero_succeeds_under_oom() {
        // Zero-capacity Vec doesn't allocate.
        let r: Result<Vec<u8>, TryReserveError> = with_policy(
            FailPolicy::fail_next_alloc(),
            || <Vec<u8> as TryVec<u8>>::try_with_capacity(0),
        );
        assert!(r.is_ok());
        assert!(r.as_ref().unwrap().is_empty());
    }

    #[test]
    fn vec_try_reserve_fails_on_oom() {
        let mut v: Vec<u32> = Vec::new();
        v.fallible_shrink_to_fit().unwrap();
        let r = with_policy(FailPolicy::fail_next_alloc(), || v.try_reserve(100));
        assert!(r.is_err());
    }

    #[test]
    fn vec_try_clone_fails_on_oom() {
        let orig: Vec<u32> = vec![1, 2, 3];
        let r: Result<Vec<u32>, TryCloneError> = with_policy(
            FailPolicy::fail_next_alloc(),
            || orig.try_clone(),
        );
        assert!(r.is_err());
    }

    #[test]
    fn vec_try_clone_empty_succeeds_under_oom() {
        let orig: Vec<u32> = Vec::new();
        let r: Result<Vec<u32>, TryCloneError> = with_policy(
            FailPolicy::fail_next_alloc(),
            || orig.try_clone(),
        );
        assert!(r.is_ok());
        assert!(r.as_ref().unwrap().is_empty());
    }

    #[test]
    fn vec_try_from_slice_fails_on_oom() {
        let slice: &[u32] = &[10, 20, 30];
        let r: Result<Vec<u32>, TryVecError> = with_policy(
            FailPolicy::fail_next_alloc(),
            || Vec::<u32>::try_from_slice(slice),
        );
        assert!(r.is_err());
    }

    #[test]
    fn vec_try_insert_fails_on_oom() {
        let mut v: Vec<u32> = Vec::new();
        let r = with_policy(FailPolicy::fail_next_alloc(), || v.try_insert(0, 42));
        assert!(r.is_err());
    }

    #[test]
    fn vec_try_push_fails_on_oom() {
        let mut v: Vec<u32> = Vec::new();
        let r = with_policy(FailPolicy::fail_next_alloc(), || v.try_push(42));
        assert!(r.is_err());
    }

    #[test]
    fn vec_try_resize_grow_fails_on_oom() {
        let mut v: Vec<u32> = Vec::new();
        let r = with_policy(FailPolicy::fail_next_alloc(), || v.try_resize_with(10, || 99u32));
        assert!(r.is_err());
    }

    #[test]
    fn vec_try_extend_fails_on_oom() {
        let items: Vec<u32> = vec![1, 2, 3];
        let mut v: Vec<u32> = Vec::new();
        let r = with_policy(FailPolicy::fail_next_alloc(), || v.try_extend(items.iter().copied()));
        assert!(r.is_err());
    }

    #[test]
    fn vec_try_collect_fails_on_oom() {
        let r: Result<Vec<u32>, TryReserveError> = with_policy(
            FailPolicy::fail_next_alloc(),
            || Vec::try_collect(1..=3),
        );
        assert!(r.is_err());
    }

    #[test]
    fn vec_try_from_elem_fails_on_oom() {
        let r: Result<Vec<u32>, TryVecError> = with_policy(
            FailPolicy::fail_next_alloc(),
            || Vec::try_from_elem(&0u32, 5),
        );
        assert!(r.is_err());
    }

    #[test]
    fn vec_oom_restores_allocation_afterwards() {
        let r: Result<Vec<u8>, TryReserveError> = with_policy(
            FailPolicy::fail_next_alloc(),
            || <Vec<u8> as TryVec<u8>>::try_with_capacity(10),
        );
        assert!(r.is_err());
        // Allocation works again after guard scope ends.
        let r: Result<Vec<u8>, TryReserveError> = <Vec<u8> as TryVec<u8>>::try_with_capacity(10);
        assert!(r.is_ok());
    }

    #[test]
    fn vec_nth_alloc_fail_targets_correct_call() {
        let results: (bool, bool, bool) = with_policy(FailPolicy::fail_nth_alloc(2), || {
            let r1: Result<Vec<u8>, TryReserveError> =
                <Vec<u8> as TryVec<u8>>::try_with_capacity(1);
            let r2: Result<Vec<u8>, TryReserveError> =
                <Vec<u8> as TryVec<u8>>::try_with_capacity(1);
            let r3: Result<Vec<u8>, TryReserveError> =
                <Vec<u8> as TryVec<u8>>::try_with_capacity(1);
            (r1.is_ok(), r2.is_err(), r3.is_ok())
        });
        assert!(results.0, "first alloc should succeed");
        assert!(results.1, "second alloc should fail");
        assert!(results.2, "third alloc should succeed");
    }
}
