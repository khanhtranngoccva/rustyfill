//! Fallible vector operations.
//!
//! Provides the [`TryVec`] trait with methods that mirror common `Vec` constructors
//! and mutating operations but return [`Result`] to handle allocation failures
//! gracefully, using [`::lang_alloc::collections::TryReserveError`] as the primary error type.
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

use super::raw_manipulation::RawVecInnerView;
use crate::alloc::TryReserveError;
use crate::alloc::TryReserveErrorExt;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::{TryDebug, TryDisplay};
use lang_alloc::boxed::Box;
use lang_alloc::vec::Vec;
use lang_core::alloc::Layout;
use lang_core::cmp;
use lang_core::fmt;
use lang_core::mem;
use lang_core::ops::{Bound, RangeBounds};

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
        mem::forget(self);
    }
}

impl<T> Drop for TruncateGuard<'_, T> {
    fn drop(&mut self) {
        self.vec.truncate(self.len_before);
    }
}

/// Error for fallible vector operations that can allocate and clone elements.
///
/// Covers `try_from_elem`, `try_from_slice`, `try_resize`,
/// `try_extend_from_slice_with_rollback`, and `try_shrink_to(_fit)` — any
/// operation whose failure modes are limited to a capacity reservation
/// ([`TryReserveError`]) or an element clone failure ([`TryCloneError`]).
pub enum TryVecWithCloneError {
    /// A capacity reservation on the vector failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An element clone failed during a method that requires `TryClone`.
    Clone(TryCloneError),
}

/// Error for fallible vector `extend_from_within` operations.
///
/// Can fail due to a capacity reservation failure, an out-of-bounds range,
/// or an element clone failure.
pub enum TryVecExtendFromWithinError {
    /// A capacity reservation failed.
    Reserve(TryReserveError),
    /// The provided range exceeded the vector's bounds.
    OutOfBounds,
    /// An element clone failed during copying.
    Clone(TryCloneError),
}

/// Error for fallible vector insert operations that can fail due to either a
/// capacity reservation failure or an out-of-bounds index.
///
/// Used by [`TryVec::try_insert`], [`TryVec::try_insert_give_back`] and their aliases.
/// In the give-back variant, the value travels alongside this error as a
/// tuple: `Result<(), (T, TryVecInsertError)>`.
pub enum TryVecInsertError {
    /// A capacity reservation failed.
    Reserve(TryReserveError),
    /// The provided index exceeded the vector's length.
    OutOfBounds,
}

impl fmt::Debug for TryVecInsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(self, f)
    }
}

impl fmt::Display for TryVecInsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }
}

impl TryDebug for TryVecInsertError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::debug_field(f, "TryVecInsertError::Reserve", e),
            Self::OutOfBounds => u::debug_unit(f, "TryVecInsertError::OutOfBounds"),
        }
    }
}

impl TryDisplay for TryVecInsertError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::display_delegated(f, "vector", e),
            Self::OutOfBounds => u::display_fixed(f, "vector", "insert index out of bounds"),
        }
    }
}

impl fmt::Debug for TryVecWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(self, f)
    }
}

impl fmt::Display for TryVecWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }
}

impl From<TryReserveError> for TryVecWithCloneError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<TryCloneError> for TryVecWithCloneError {
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

impl TryDebug for TryVecWithCloneError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::debug_field(f, "TryVecWithCloneError::Reserve", e),
            Self::Clone(e) => u::debug_field(f, "TryVecWithCloneError::Clone", e),
        }
    }
}

impl TryDisplay for TryVecWithCloneError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::display_delegated(f, "vector", e),
            Self::Clone(e) => u::display_delegated(f, "vector", e),
        }
    }
}

impl fmt::Debug for TryVecExtendFromWithinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(self, f)
    }
}

impl fmt::Display for TryVecExtendFromWithinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }
}

impl TryDebug for TryVecExtendFromWithinError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::debug_field(f, "TryVecExtendFromWithinError::Reserve", e),
            Self::OutOfBounds => u::debug_unit(f, "TryVecExtendFromWithinError::OutOfBounds"),
            Self::Clone(e) => u::debug_field(f, "TryVecExtendFromWithinError::Clone", e),
        }
    }
}

impl TryDisplay for TryVecExtendFromWithinError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::display_delegated(f, "vector", e),
            Self::OutOfBounds => u::display_fixed(f, "vector", "range out of bounds"),
            Self::Clone(e) => u::display_delegated(f, "vector", e),
        }
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
    /// Returns [`TryVecWithCloneError::Reserve`] if the capacity allocation fails,
    /// or [`TryVecWithCloneError::Clone`] if an element's [`TryClone::try_clone`]
    /// fails. Equivalent to `vec![value; count]` but fully fallible.
    fn try_from_elem(value: &T, count: usize) -> Result<Vec<T>, TryVecWithCloneError>
    where
        T: TryClone;

    /// Like [`Self::try_from_elem`] but takes ownership of `value` and returns
    /// it on failure so the caller is not left empty-handed.
    fn try_from_elem_give_back(value: T, count: usize) -> Result<Vec<T>, (T, TryVecWithCloneError)>
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
    /// Returns [`TryVecInsertError::Reserve`] if growing the internal buffer
    /// fails, or [`TryVecInsertError::OutOfBounds`] if `index > len`.
    fn try_insert(&mut self, index: usize, value: T) -> Result<(), TryVecInsertError>;

    /// Like [`Self::try_insert`] but returns ownership of `value` back on failure.
    fn try_insert_give_back(
        &mut self,
        index: usize,
        value: T,
    ) -> Result<(), (T, TryVecInsertError)>;

    /// Fallibly extend the vector with all elements from an iterator source.
    /// Like [`Self::try_extend_from_slice_with_rollback`] but without rollback.
    ///
    /// If a clone fails mid-way, the vector is truncated back to its length at
    /// the start of the call so that no partially-appended elements remain.
    /// The error does not carry a remainder since the collection state is
    /// restored to exactly what it was before the call.
    fn try_extend_from_slice_with_rollback(
        &mut self,
        other: &[T],
    ) -> Result<(), TryVecWithCloneError>
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
    /// first so that allocation failures return
    /// [`TryVecExtendFromWithinError::Reserve`] instead of panicking. Uses
    /// [`TryClone`] for each copy so clone-time failures return
    /// [`TryVecExtendFromWithinError::Clone`] and the vector is rolled back to
    /// its pre-call state.
    ///
    /// Returns [`TryVecExtendFromWithinError::OutOfBounds`] if the range is
    /// out of bounds.
    fn try_extend_from_within<R: RangeBounds<usize>>(
        &mut self,
        range: R,
    ) -> Result<(), TryVecExtendFromWithinError>
    where
        T: TryClone;

    /// Resizes the vector in place so that `len` equals `new_len`.
    ///
    /// If `new_len` is greater than `len`, the vector is extended by cloning
    /// `value` via [`TryClone`]. If `new_len` is less than `len`, the vector
    /// is truncated. Returns [`TryVecWithCloneError::Reserve`] on allocation failure or
    /// [`TryVecWithCloneError::Clone`] if an element clone fails.
    fn try_resize(&mut self, value: &T, new_len: usize) -> Result<(), TryVecWithCloneError>
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
    fn try_shrink_to_fit(&mut self) -> Result<(), TryReserveError>;

    /// Fallibly shrink the capacity of this vector to at least `min_capacity`.
    ///
    /// **Deprecated:** This method name conflicts with the unstable inherent
    /// [`Vec::try_shrink_to`]. Use [`Self::fallible_shrink_to`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with unstable Vec::try_shrink_to; use fallible_shrink_to"
    )]
    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryReserveError>;

    // ── Aliases with `fallible_` prefix ─────────────────────────────────────

    /// Alias for [`Self::try_with_capacity`].
    fn fallible_with_capacity(capacity: usize) -> Result<Vec<T>, TryReserveError> {
        Self::try_with_capacity(capacity)
    }

    /// Alias for [`Self::try_from_elem`].
    fn fallible_from_elem(value: &T, count: usize) -> Result<Vec<T>, TryVecWithCloneError>
    where
        T: TryClone,
    {
        Self::try_from_elem(value, count)
    }

    /// Alias for [`Self::try_from_elem_give_back`].
    fn fallible_from_elem_give_back(
        value: T,
        count: usize,
    ) -> Result<Vec<T>, (T, TryVecWithCloneError)>
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
    fn fallible_insert(&mut self, index: usize, value: T) -> Result<(), TryVecInsertError> {
        Self::try_insert(self, index, value)
    }

    /// Alias for [`Self::try_insert_give_back`].
    fn fallible_insert_give_back(
        &mut self,
        index: usize,
        value: T,
    ) -> Result<(), (T, TryVecInsertError)> {
        Self::try_insert_give_back(self, index, value)
    }

    /// Alias for [`Self::try_extend_from_slice_with_rollback`].
    fn fallible_extend_from_slice_with_rollback(
        &mut self,
        other: &[T],
    ) -> Result<(), TryVecWithCloneError>
    where
        T: TryClone,
    {
        Self::try_extend_from_slice_with_rollback(self, other)
    }

    /// Alias for [`Self::try_append`].
    fn fallible_append(&mut self, other: &mut Vec<T>) -> Result<(), TryReserveError> {
        Self::try_append(self, other)
    }

    /// Alias for [`Self::try_extend_from_within`].
    fn fallible_extend_from_within<R: RangeBounds<usize>>(
        &mut self,
        range: R,
    ) -> Result<(), TryVecExtendFromWithinError>
    where
        T: TryClone,
    {
        Self::try_extend_from_within(self, range)
    }

    /// Alias for [`Self::try_resize`].
    fn fallible_resize(&mut self, value: &T, new_len: usize) -> Result<(), TryVecWithCloneError>
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
    /// Shrink never clones elements, so the only failure mode is a failed
    /// re-allocation ([`TryReserveError`]).
    /// Equivalent to [`Vec::shrink_to_fit`] but fallible.
    ///
    /// This method replaces the deprecated [`Self::try_shrink_to_fit`] which
    /// shares its name with the unstable inherent [`Vec::try_shrink_to_fit`].
    #[allow(deprecated)]
    fn fallible_shrink_to_fit(&mut self) -> Result<(), TryReserveError> {
        Self::try_shrink_to_fit(self)
    }

    /// Fallibly shrink the capacity of this vector to at least `min_capacity`.
    ///
    /// If the current capacity is already less than or equal to `min_capacity`,
    /// does nothing and returns `Ok(())`. Otherwise reallocates down.
    /// Shrink never clones elements, so the only failure mode is a failed
    /// re-allocation ([`TryReserveError`]).
    /// Equivalent to [`Vec::shrink_to`] but fallible.
    ///
    /// This method replaces the deprecated [`Self::try_shrink_to`] which shares
    /// its name with the unstable inherent [`Vec::try_shrink_to`].
    #[allow(deprecated)]
    fn fallible_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryReserveError> {
        Self::try_shrink_to(self, min_capacity)
    }

    /// Alias for [`Self::try_collect`].
    fn fallible_collect<I: IntoIterator<Item = T>>(iter: I) -> Result<Vec<T>, TryReserveError> {
        Self::try_collect(iter)
    }

    /// Alias for [`Self::try_from_slice`].
    fn fallible_from_slice(slice: &[T]) -> Result<Vec<T>, TryVecWithCloneError>
    where
        T: TryClone,
    {
        Self::try_from_slice(slice)
    }

    /// Alias for [`Self::try_into_boxed_slice`].
    fn fallible_into_boxed_slice(self) -> Result<Box<[T]>, TryReserveError> {
        Self::try_into_boxed_slice(self)
    }

    /// Alias for [`Self::try_into_boxed_slice_give_back`].
    fn fallible_into_boxed_slice_give_back(self) -> Result<Box<[T]>, (Vec<T>, TryReserveError)> {
        Self::try_into_boxed_slice_give_back(self)
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    /// Fallibly collect an iterator into a `Vec<T>`.
    ///
    /// Uses the iterator's size hint to pre-allocate when possible.
    fn try_collect<I: IntoIterator<Item = T>>(iter: I) -> Result<Vec<T>, TryReserveError>;

    /// Fallibly create a `Vec<T>` from a slice by cloning each element via
    /// [`TryClone`].
    ///
    /// Returns [`TryVecWithCloneError::Reserve`] on capacity failure or
    /// [`TryVecWithCloneError::Clone`] if an element's [`TryClone::try_clone`] fails.
    fn try_from_slice(slice: &[T]) -> Result<Vec<T>, TryVecWithCloneError>
    where
        T: TryClone;

    // ── Conversion to boxed types ─────────────────────────────────────────────

    /// Fallibly convert this vector into a `Box<[T]>`.
    ///
    /// This is the fallible analogue of [`Vec::into_boxed_slice`]. The resulting
    /// box has exactly `len()` elements and no excess capacity. No elements are
    /// ever cloned: when the current allocation has spare capacity it is
    /// shrunk in place via `realloc`, and on success the buffer is handed
    /// straight to the box.
    ///
    /// Returns [`TryReserveError`] if the shrink reallocation fails. Note that
    /// unlike the give-back variant, the vector is consumed either way — on
    /// failure the caller does not get the elements back.
    ///
    /// For empty vectors, this returns an empty boxed slice without allocating.
    fn try_into_boxed_slice(self) -> Result<Box<[T]>, TryReserveError>;

    /// Like [`Self::try_into_boxed_slice`] but returns ownership of the vector
    /// back on failure so the caller is not left empty-handed.
    fn try_into_boxed_slice_give_back(self) -> Result<Box<[T]>, (Vec<T>, TryReserveError)>;
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

    fn try_from_elem(value: &T, count: usize) -> Result<Vec<T>, TryVecWithCloneError>
    where
        T: TryClone,
    {
        let mut vec = Vec::<T>::new();
        if count > 0 {
            vec.try_reserve(count)
                .map_err(TryVecWithCloneError::Reserve)?;
        }
        for _ in 0..count {
            vec.push(value.try_clone().map_err(TryVecWithCloneError::Clone)?);
        }
        Ok(vec)
    }

    fn try_from_elem_give_back(value: T, count: usize) -> Result<Vec<T>, (T, TryVecWithCloneError)>
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
            Err(e) => Err((value, e)),
        }
    }

    fn try_insert(&mut self, index: usize, value: T) -> Result<(), TryVecInsertError> {
        if index > self.len() {
            return Err(TryVecInsertError::OutOfBounds);
        }
        self.try_reserve(1).map_err(TryVecInsertError::Reserve)?;
        self.insert(index, value);
        Ok(())
    }

    fn try_insert_give_back(
        &mut self,
        index: usize,
        value: T,
    ) -> Result<(), (T, TryVecInsertError)> {
        if index > self.len() {
            return Err((value, TryVecInsertError::OutOfBounds));
        }
        match self.try_reserve(1) {
            Ok(()) => {
                self.insert(index, value);
                Ok(())
            }
            Err(e) => Err((value, TryVecInsertError::Reserve(e))),
        }
    }

    fn try_extend_from_slice_with_rollback(
        &mut self,
        other: &[T],
    ) -> Result<(), TryVecWithCloneError>
    where
        T: TryClone,
    {
        if other.is_empty() {
            return Ok(());
        }
        self.try_reserve(other.len())
            .map_err(TryVecWithCloneError::Reserve)?;
        let guard = TruncateGuard::new(self);
        for item in other {
            match item.try_clone() {
                Ok(cloned) => {
                    guard.vec.push(cloned);
                }
                Err(e) => {
                    return Err(TryVecWithCloneError::Clone(e));
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

    fn try_extend_from_within<R: RangeBounds<usize>>(
        &mut self,
        range: R,
    ) -> Result<(), TryVecExtendFromWithinError>
    where
        T: TryClone,
    {
        let start = match range.start_bound() {
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i.checked_add(1).ok_or_else(|| {
                TryVecExtendFromWithinError::Reserve(TryReserveErrorExt::new_capacity_overflow())
            })?,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&i) => i.checked_add(1).ok_or_else(|| {
                TryVecExtendFromWithinError::Reserve(TryReserveErrorExt::new_capacity_overflow())
            })?,
            Bound::Excluded(&i) => i,
            Bound::Unbounded => self.len(),
        };

        if start >= end {
            return Ok(());
        }

        // Validate bounds before any mutation.
        if end > self.len() || start > self.len() {
            return Err(TryVecExtendFromWithinError::OutOfBounds);
        }

        let count = end.saturating_sub(start);
        // Reserve first — lazy, no element copies until allocation succeeds.
        self.try_reserve(count)
            .map_err(TryVecExtendFromWithinError::Reserve)?;
        let guard = TruncateGuard::new(self);
        for i in start..end {
            match guard.vec[i].try_clone() {
                Ok(cloned) => guard.vec.push(cloned),
                Err(e) => {
                    return Err(TryVecExtendFromWithinError::Clone(e));
                }
            }
        }
        guard.forget();
        Ok(())
    }

    fn try_resize(&mut self, value: &T, new_len: usize) -> Result<(), TryVecWithCloneError>
    where
        T: TryClone,
    {
        let current = self.len();
        if new_len <= current {
            // Truncation never allocates.
            self.truncate(new_len);
            return Ok(());
        }
        let extra = new_len.saturating_sub(current);
        // Reserve first — lazy.
        self.try_reserve(extra)
            .map_err(TryVecWithCloneError::Reserve)?;
        let guard = TruncateGuard::new(self);
        for _ in 0..extra {
            match value.try_clone() {
                Ok(cloned) => guard.vec.push(cloned),
                Err(e) => {
                    return Err(TryVecWithCloneError::Clone(e));
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
        let extra = new_len.saturating_sub(current);
        // Reserve first — lazy, closure not called until allocation succeeds.
        self.try_reserve(extra)?;
        let guard = TruncateGuard::new(self);
        for _ in 0..extra {
            guard.vec.push(f());
        }
        guard.forget();
        Ok(())
    }

    fn try_shrink_to_fit(&mut self) -> Result<(), TryReserveError> {
        <Self as TryVec<T>>::try_shrink_to(self, self.len())
    }

    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryReserveError> {
        let target = cmp::max(self.len(), min_capacity);
        if self.capacity() <= target {
            return Ok(());
        }
        let (mut current_raw, current_len) = RawVecInnerView::from_vec(mem::take(self));
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
            Err(_) => {
                // Allocation failed. shrink_unchecked returns early via `?` on
                // realloc failure, BEFORE updating self.ptr / self.cap — so
                // current_raw still holds the original (unshrunk) allocation.
                // SAFETY: pointer, length, and capacity are all still valid from
                // the original Vec.
                *self = unsafe { current_raw.into_vec(current_len) };
                Err(TryReserveErrorExt::new_alloc(Layout::new::<T>()))
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

    fn try_from_slice(slice: &[T]) -> Result<Vec<T>, TryVecWithCloneError>
    where
        T: TryClone,
    {
        let mut vec = Vec::<T>::new();
        vec.try_reserve(slice.len())
            .map_err(TryVecWithCloneError::Reserve)?;
        for item in slice {
            vec.push(item.try_clone().map_err(TryVecWithCloneError::Clone)?);
        }
        Ok(vec)
    }

    fn try_into_boxed_slice(self) -> Result<Box<[T]>, TryReserveError> {
        // Shrink first so the buffer has exactly len() capacity; this is a
        // no-op when there's no spare capacity. On failure the elements are
        // dropped along with `self` — use try_into_boxed_slice_give_back to
        // recover them instead.
        let mut vec = self;
        <Vec<T> as TryVec<T>>::try_shrink_to_fit(&mut vec)?;
        Ok(vec.into_boxed_slice())
    }

    fn try_into_boxed_slice_give_back(self) -> Result<Box<[T]>, (Vec<T>, TryReserveError)> {
        // Shrink first. If the shrink fails, return the original vec so no
        // data is lost.
        let mut vec = self;
        match <Vec<T> as TryVec<T>>::try_shrink_to_fit(&mut vec) {
            Ok(()) => Ok(vec.into_boxed_slice()),
            Err(e) => Err((vec, e)),
        }
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

// ── TryDebug for Vec<T> ──────────────────────────────────────────────────────

impl<T: crate::try_fmt::TryDebug> crate::try_fmt::TryDebug for Vec<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::try_fmt::helpers::FormatterExt;
        f.try_debug_list().entries(self.iter()).finish()
    }
}

// NOTE: no `TryDisplay` impl for `Vec<T>` — `Vec` does not implement
// `fmt::Display` (only `Debug`), and `TryDisplay` requires `Display` as a
// supertrait. Use `TryDebug` for list rendering.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::TryReserveErrorExt;
    use lang_alloc::format;
    use lang_alloc::string::String;
    use lang_alloc::string::ToString;
    use lang_alloc::vec;
    use lang_core::fmt::Write as _;
    use lang_core::iter;

    /// A `TryReserveError` instance for exercising the `Reserve` arm.
    fn reserve_err() -> TryReserveError {
        TryReserveError::new_capacity_overflow()
    }

    /// Formats a value via its `Display` impl into a fresh String.
    fn render_display(e: &impl fmt::Display) -> String {
        let mut s = String::new();
        // Our error Display impls only call `write!` on literals/wrapped values,
        // so this cannot fail in practice; ignore the infallible-in-practice result.
        let _ = write!(&mut s, "{e}");
        s
    }

    /// Captures the `TryDebug` rendering of a value.
    fn render_trydebug(e: &impl TryDebug) -> String {
        struct Cap<'a>(&'a dyn TryDebug);
        impl fmt::Debug for Cap<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.try_fmt(f)
            }
        }
        format!("{:?}", Cap(e))
    }

    /// Captures the `TryDisplay` rendering of a value (should match `Display`).
    fn render_trydisplay(e: &impl TryDisplay) -> String {
        struct Cap<'a>(&'a dyn TryDisplay);
        impl fmt::Display for Cap<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.try_fmt(f)
            }
        }
        let mut s = String::new();
        let _ = write!(&mut s, "{}", Cap(e));
        s
    }

    // ── Error enum formatting (moved from errors::uniform) ────────────────────
    //
    // NOTE: we must BORROW each variant when formatting. Iterating by value moves
    // the error out, which does NOT execute its `Display::fmt` / `TryDebug::try_fmt`
    // — coverage tracks the formatter call, not the construction.

    #[test]
    fn vec_error_display_covers_all_variants() {
        let cases: [(TryVecWithCloneError, &str); 2] = [
            (
                TryVecWithCloneError::Reserve(reserve_err()),
                "vector operation failed:",
            ),
            (
                TryVecWithCloneError::Clone(TryCloneError::Reserve(reserve_err())),
                "vector operation failed:",
            ),
        ];
        for &(ref err, expected_prefix) in cases.iter() {
            let got = render_display(err);
            assert!(
                got.starts_with(expected_prefix),
                "expected prefix {expected_prefix:?}, got {got:?}"
            );
        }
    }

    #[test]
    fn vec_error_trydebug_covers_all_variants() {
        let errs = [
            TryVecWithCloneError::Reserve(reserve_err()),
            TryVecWithCloneError::Clone(TryCloneError::Reserve(reserve_err())),
        ];
        for err in errs.iter() {
            let got = render_trydebug(err);
            assert!(
                got.contains("TryVecWithCloneError::"),
                "missing type tag in {got:?}"
            );
        }
    }

    /// Drives the `TryDisplay` impl across every variant; it must match `Display`.
    #[test]
    fn vec_error_trydisplay_covers_all_variants() {
        let errs = [
            TryVecWithCloneError::Reserve(reserve_err()),
            TryVecWithCloneError::Clone(TryCloneError::Reserve(reserve_err())),
        ];
        for err in errs.iter() {
            let tdisp = render_trydisplay(err);
            assert_eq!(tdisp, render_display(err), "TryDisplay must match Display");
        }
    }

    /// Byte-identity guard: the delegated `Display` arm must produce exactly
    /// `"{prefix} operation failed: {wrapped}"`, where `{wrapped}` is the inner
    /// error's own rendering. This pins the helper output to the original
    /// hand-written format string character-for-character.
    #[test]
    fn display_delegated_is_byte_identical_to_original_format() {
        let reserve = reserve_err();
        let inner_text = render_display(&reserve); // std's own wording
        let expected = format!("vector operation failed: {inner_text}");
        let actual = render_display(&TryVecWithCloneError::Reserve(reserve));
        assert_eq!(
            actual, expected,
            "delegated Display drifted from original format"
        );

        // Same check for the Clone arm, whose detail is a TryCloneError.
        let clone_src = TryCloneError::Reserve(reserve_err());
        let clone_inner = render_display(&clone_src);
        let expected_clone = format!("vector operation failed: {clone_inner}");
        let actual_clone = render_display(&TryVecWithCloneError::Clone(clone_src));
        assert_eq!(actual_clone, expected_clone);
    }

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
        use crate::try_extend::TryExtend;
        let mut v: Vec<i32> = Vec::new();
        <_ as TryExtend<i32>>::try_extend(&mut v, 0..5).unwrap();
        assert_eq!(v, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn try_extend_empty() {
        use crate::try_extend::TryExtend;
        let mut v: Vec<i32> = Vec::new();
        <_ as TryExtend<i32>>::try_extend(&mut v, iter::empty::<i32>()).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn try_extend_existing() {
        use crate::try_extend::TryExtend;
        let mut v: Vec<i32> = Vec::new();
        v.try_push(1).unwrap();
        <_ as TryExtend<i32>>::try_extend(&mut v, [2, 3]).unwrap();
        assert_eq!(v, [1, 2, 3]);
    }

    #[test]
    fn try_extend_from_slice_clones_elements() {
        use crate::try_extend::TryExtendFromSlice;
        let mut v: Vec<Vec<u8>> = Vec::new();
        v.try_push(vec![1]).unwrap();
        let slice: &[Vec<u8>] = &[vec![2], vec![3]];
        <_ as TryExtendFromSlice<'_, Vec<u8>>>::try_extend_from_slice(&mut v, slice).unwrap();
        assert_eq!(v, [vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn try_extend_from_slice_empty() {
        use crate::try_extend::TryExtendFromSlice;
        let mut v: Vec<i32> = Vec::new();
        <_ as TryExtendFromSlice<'_, i32>>::try_extend_from_slice(&mut v, &[]).unwrap();
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
        let result: Result<Vec<Vec<u8>>, (Vec<u8>, TryVecWithCloneError)> =
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
        let v: Vec<i32> = Vec::try_collect(iter::empty::<i32>()).unwrap();
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

    // ── Conversion to boxed types ─────────────────────────────────────────────

    #[test]
    fn try_into_boxed_slice_empty() {
        let v: Vec<i32> = Vec::new();
        let boxed: Box<[i32]> = v.try_into_boxed_slice().unwrap();
        assert!(boxed.is_empty());
    }

    #[test]
    fn try_into_boxed_slice_exact_capacity() {
        let v: Vec<i32> = vec![1, 2, 3];
        let boxed: Box<[i32]> = v.try_into_boxed_slice().unwrap();
        assert_eq!(*boxed, [1, 2, 3]);
    }

    #[test]
    fn try_into_boxed_slice_with_spare_capacity() {
        let mut v: Vec<i32> = Vec::new();
        v.try_reserve(1024).unwrap();
        v.try_push(1).unwrap();
        v.try_push(2).unwrap();
        let boxed: Box<[i32]> = v.try_into_boxed_slice().unwrap();
        assert_eq!(*boxed, [1, 2]);
    }

    #[test]
    fn try_into_boxed_slice_preserves_data() {
        let v: Vec<String> = vec!["hello".to_string(), "world".to_string()];
        let boxed: Box<[String]> = v.try_into_boxed_slice().unwrap();
        assert_eq!(&*boxed, &["hello", "world"][..]);
    }

    #[test]
    fn try_into_boxed_slice_give_back_success() {
        let mut v: Vec<i32> = Vec::new();
        v.try_reserve(64).unwrap();
        v.try_push(7).unwrap();
        let boxed: Box<[i32]> = v.try_into_boxed_slice_give_back().unwrap();
        assert_eq!(*boxed, [7]);
    }

    #[test]
    fn try_into_boxed_slice_give_back_empty() {
        let v: Vec<i32> = Vec::new();
        let boxed: Box<[i32]> = v.try_into_boxed_slice_give_back().unwrap();
        assert!(boxed.is_empty());
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

    #[cfg(feature = "std")]
    mod oom {
        use super::*;
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        #[test]
        fn vec_try_with_capacity_fails_on_oom() {
            let r: Result<Vec<u8>, TryReserveError> =
                with_policy(FailPolicy::fail_next_alloc(), || {
                    <Vec<u8> as TryVec<u8>>::try_with_capacity(10)
                });
            assert!(r.is_err());
        }

        #[test]
        fn vec_try_with_capacity_zero_succeeds_under_oom() {
            // Zero-capacity Vec doesn't allocate.
            let r: Result<Vec<u8>, TryReserveError> =
                with_policy(FailPolicy::fail_next_alloc(), || {
                    <Vec<u8> as TryVec<u8>>::try_with_capacity(0)
                });
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
            let r: Result<Vec<u32>, TryCloneError> =
                with_policy(FailPolicy::fail_next_alloc(), || orig.try_clone());
            assert!(r.is_err());
        }

        #[test]
        fn vec_try_clone_empty_succeeds_under_oom() {
            let orig: Vec<u32> = Vec::new();
            let r: Result<Vec<u32>, TryCloneError> =
                with_policy(FailPolicy::fail_next_alloc(), || orig.try_clone());
            assert!(r.is_ok());
            assert!(r.as_ref().unwrap().is_empty());
        }

        #[test]
        fn vec_try_from_slice_fails_on_oom() {
            let slice: &[u32] = &[10, 20, 30];
            let r: Result<Vec<u32>, TryVecWithCloneError> =
                with_policy(FailPolicy::fail_next_alloc(), || {
                    Vec::<u32>::try_from_slice(slice)
                });
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
            let r = with_policy(FailPolicy::fail_next_alloc(), || {
                v.try_resize_with(10, || 99u32)
            });
            assert!(r.is_err());
        }

        #[test]
        fn vec_try_extend_fails_on_oom() {
            use crate::try_extend::TryExtend;
            let items: Vec<u32> = vec![1, 2, 3];
            let mut v: Vec<u32> = Vec::new();
            let r = with_policy(FailPolicy::fail_next_alloc(), || {
                <_ as TryExtend<u32>>::try_extend(&mut v, items.iter().copied())
            });
            assert!(r.is_err());
        }

        #[test]
        fn vec_try_collect_fails_on_oom() {
            let r: Result<Vec<u32>, TryReserveError> =
                with_policy(FailPolicy::fail_next_alloc(), || Vec::try_collect(1..=3));
            assert!(r.is_err());
        }

        #[test]
        fn vec_try_from_elem_fails_on_oom() {
            let r: Result<Vec<u32>, TryVecWithCloneError> =
                with_policy(FailPolicy::fail_next_alloc(), || {
                    Vec::try_from_elem(&0u32, 5)
                });
            assert!(r.is_err());
        }

        #[test]
        fn vec_oom_restores_allocation_afterwards() {
            let r: Result<Vec<u8>, TryReserveError> =
                with_policy(FailPolicy::fail_next_alloc(), || {
                    <Vec<u8> as TryVec<u8>>::try_with_capacity(10)
                });
            assert!(r.is_err());
            // Allocation works again after guard scope ends.
            let r: Result<Vec<u8>, TryReserveError> =
                <Vec<u8> as TryVec<u8>>::try_with_capacity(10);
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

        // ── Explicit rollback / TruncateGuard tests ─────────────────────────────

        #[test]
        fn extend_from_slice_with_rollback_on_mid_way_clone_failure() {
            // try_extend_from_slice_with_rollback on Vec<String> reserves capacity
            // upfront, then clones each element via String::try_clone(). By failing
            // the Nth allocation we can make a clone fail mid-way through the loop.
            // The TruncateGuard must drop all elements pushed before the failure.
            use lang_alloc::string::String;

            let source: Vec<String> = vec![
                "item0".into(),
                "item1".into(),
                "item2".into(),
                "item3".into(),
                "item4".into(),
                "item5".into(),
                "item6".into(),
                "item7".into(),
                "item8".into(),
                "item9".into(),
            ];
            let len_source = source.len();

            // Start with 3 pre-existing elements so we can verify they survive.
            let mut vec: Vec<String> = vec!["pre0".into(), "pre1".into(), "pre2".into()];
            let len_before = vec.len();

            // Fail an allocation somewhere in the middle of the clone loop.
            // try_reserve already succeeded (outside the policy scope for the
            // capacity reservation), so the first alloc inside with_policy will
            // be from the first or later String::try_clone() call.
            let r: Result<(), TryVecWithCloneError> =
                with_policy(FailPolicy::fail_nth_alloc(2), || {
                    <Vec<String> as TryVec<String>>::try_extend_from_slice_with_rollback(
                        &mut vec, &source,
                    )
                });

            match r {
                Err(TryVecWithCloneError::Clone(_)) => {
                    // Clone failed mid-way — TruncateGuard must have rolled back.
                    assert_eq!(
                        vec.len(),
                        len_before,
                        "TruncateGuard did not roll back: expected {} elements, got {}",
                        len_before,
                        vec.len()
                    );
                    // Pre-existing elements must be intact.
                    assert_eq!(vec[0], "pre0");
                    assert_eq!(vec[1], "pre1");
                    assert_eq!(vec[2], "pre2");
                }
                Ok(()) => {
                    // If it succeeded (failure didn't hit a clone alloc point),
                    // all elements were appended.
                    assert_eq!(vec.len(), len_before + len_source);
                }
                Err(other) => {
                    panic!("unexpected error variant: {:?}", other);
                }
            }
        }

        #[test]
        fn extend_from_slice_with_rollback_no_partial_elements_after_failure() {
            // Verify that after a mid-way clone failure of the _with_rollback
            // variant, the vec contains zero elements from the source slice — not
            // even the ones cloned before the failure.
            use lang_alloc::string::String;

            let source: Vec<String> = vec![
                "src0xxxxxxxx".into(),
                "src1xxxxxxxx".into(),
                "src2xxxxxxxx".into(),
                "src3xxxxxxxx".into(),
                "src4xxxxxxxx".into(),
                "src5xxxxxxxx".into(),
                "src6xxxxxxxx".into(),
                "src7xxxxxxxx".into(),
                "src8xxxxxxxx".into(),
                "src9xxxxxxxx".into(),
            ];
            let mut vec: Vec<String> = vec!["anchor".into()];

            let _: Result<(), TryVecWithCloneError> =
                with_policy(FailPolicy::fail_nth_alloc(3), || {
                    <Vec<String> as TryVec<String>>::try_extend_from_slice_with_rollback(
                        &mut vec, &source,
                    )
                });

            // Whatever happened, no source strings should appear in vec.
            for elem in vec.iter() {
                assert!(
                    !elem.starts_with("src"),
                    "found a source element in vec after supposed rollback"
                );
            }
        }

        #[test]
        fn extend_from_slice_no_rollback_returns_remainder_and_keeps_prefix() {
            // The standard try_extend_from_slice keeps already-cloned elements and
            // returns the unprocessed tail alongside the error.
            use lang_alloc::string::String;

            let source: Vec<String> = vec![
                "item0".into(),
                "item1".into(),
                "item2".into(),
                "item3".into(),
                "item4".into(),
                "item5".into(),
                "item6".into(),
                "item7".into(),
                "item8".into(),
                "item9".into(),
            ];
            let len_source = source.len();

            let mut vec: Vec<String> = vec!["pre".into()];
            let len_before = vec.len();

            use crate::try_extend::TryExtendFromSlice;
            let r: Result<(), (&[String], TryVecWithCloneError)> =
                with_policy(FailPolicy::fail_nth_alloc(2), || {
                    <Vec<String> as TryExtendFromSlice<'_, String>>::try_extend_from_slice(
                        &mut vec, &source,
                    )
                });

            match r {
                Err((remaining, err)) => {
                    matches!(err, TryVecWithCloneError::Clone(_));
                    // Returned subslice must be a contiguous tail of `source`.
                    assert!(!remaining.is_empty());
                    let fail_idx = len_source - remaining.len();
                    assert_eq!(remaining, &source[fail_idx..]);
                    // No-rollback: every element before the failing index was pushed.
                    assert_eq!(vec.len(), len_before + fail_idx);
                    for i in 0..fail_idx {
                        assert_eq!(vec[len_before + i], source[i]);
                    }
                    // Pre-existing elements are intact.
                    assert_eq!(vec[0], "pre");
                }
                Ok(()) => {
                    // If it succeeded (failure didn't hit a clone alloc point),
                    // all elements were appended.
                    assert_eq!(vec.len(), len_before + len_source);
                }
            }
        }

        #[test]
        fn extend_from_within_rollback_on_mid_way_clone_failure() {
            // try_extend_from_within clones elements from within the same vec.
            // A mid-way clone failure should trigger TruncateGuard to remove
            // the elements already pushed.
            use lang_alloc::string::String;

            let mut vec: Vec<String> =
                vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()];
            let len_before = vec.len();

            // Extend from [0..3], which clones 3 strings. Fail one mid-way.
            let r: Result<(), TryVecExtendFromWithinError> =
                with_policy(FailPolicy::fail_nth_alloc(2), || {
                    <Vec<String> as TryVec<String>>::try_extend_from_within(&mut vec, 0..3)
                });

            match r {
                Err(TryVecExtendFromWithinError::Clone(_)) => {
                    assert_eq!(
                        vec.len(),
                        len_before,
                        "TruncateGuard failed to roll back extend_from_within"
                    );
                    assert_eq!(vec[0], "a");
                    assert_eq!(vec[4], "e");
                }
                Ok(()) => {
                    assert_eq!(vec.len(), len_before + 3);
                }
                Err(other) => {
                    panic!("unexpected error: {:?}", other);
                }
            }
        }

        #[test]
        fn resize_with_clone_rollback_on_mid_way_failure() {
            // try_resize_with clones a value repeatedly to fill new slots.
            // Mid-way failure must truncate back to original length.
            use lang_alloc::string::String;

            let val: String = "repeated".into();
            let mut vec: Vec<String> = vec!["original".into()];
            let len_before = vec.len();

            // Resize to 15 — needs 14 clones. Fail one mid-way.
            let r: Result<(), TryVecWithCloneError> =
                with_policy(FailPolicy::fail_nth_alloc(3), || {
                    <Vec<String> as TryVec<String>>::try_resize(&mut vec, &val, 15)
                });

            match r {
                Err(TryVecWithCloneError::Clone(_)) => {
                    assert_eq!(
                        vec.len(),
                        len_before,
                        "resize rollback failed: expected {}, got {}",
                        len_before,
                        vec.len()
                    );
                    assert_eq!(vec[0], "original");
                }
                Ok(()) => {
                    assert_eq!(vec.len(), 15);
                }
                Err(other) => {
                    panic!("unexpected error: {:?}", other);
                }
            }
        }

        /// A panicking `try_clone` mid-extension must still trigger the
        /// [`TruncateGuard`] to roll back all elements appended during the
        /// call — unconditional rollback even on unwind.
        #[test]
        fn extend_from_slice_with_rollback_panic_safe() {
            use crate::try_clone::TryCloneError;
            use lang_std::panic;

            #[derive(Clone)]
            struct Panicky(u8);
            impl TryClone for Panicky {
                fn try_clone(&self) -> Result<Self, TryCloneError> {
                    if self.0 == 40 {
                        panic!("simulated clone panic");
                    }
                    Ok(Self(self.0))
                }
            }

            let mut vec: Vec<Panicky> = vec![Panicky(10), Panicky(20)];

            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                <Vec<Panicky> as TryVec<Panicky>>::try_extend_from_slice_with_rollback(
                    &mut vec,
                    &[Panicky(30), Panicky(40)],
                )
            }));
            assert!(result.is_err(), "expected a panic from element 40");
            // The guard truncated the appended 30 during unwinding; only the
            // pre-existing elements remain.
            assert_eq!(vec.len(), 2);
            assert_eq!(vec[0].0, 10);
            assert_eq!(vec[1].0, 20);
        }
    }
}
