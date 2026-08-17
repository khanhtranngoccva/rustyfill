//! Fallible double-ended queue operations.
//!
//! Provides the [`TryVecDeque`] trait with methods that mirror common `VecDeque`
//! constructors and mutating operations but return [`Result`] to handle allocation
//! failures gracefully, using [`::lang_alloc::collections::TryReserveError`] as the primary
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
use crate::alloc::vec::{TryVec, TryVecError};
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::{TryDebug, helpers::FormatterExt};
use lang_alloc::collections::VecDeque;
use lang_alloc::vec::Vec;
use lang_core::cmp;
use lang_core::fmt;
use lang_core::mem;
use lang_core::ops::{Bound, RangeBounds};

/// Panic-safe guard that truncates a `VecDeque` back to its original length on drop
/// unless disarmed via `forget()`. Used by fallible extend/resize methods so that if
/// an element's `try_clone` or a closure panics mid-way, partially-pushed elements are
/// removed rather than left behind.
struct TruncateGuard<'a, T> {
    deque: &'a mut VecDeque<T>,
    len_before: usize,
}

impl<'a, T> TruncateGuard<'a, T> {
    fn new(deque: &'a mut VecDeque<T>) -> Self {
        let len = deque.len();
        Self {
            deque,
            len_before: len,
        }
    }

    /// Disable the guard — no truncation on scope exit.
    fn forget(mut self) {
        self.len_before = self.deque.len();
        mem::forget(self);
    }
}

impl<'a, T> Drop for TruncateGuard<'a, T> {
    fn drop(&mut self) {
        self.deque.truncate(self.len_before);
    }
}

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

impl TryDebug for TryVecDequeError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => f
                .try_debug_struct("TryVecDequeError::Alloc")
                .field("0", e)
                .finish(),
            Self::Reserve(e) => f
                .try_debug_struct("TryVecDequeError::Reserve")
                .field("0", e)
                .finish(),
            Self::Clone(e) => f
                .try_debug_struct("TryVecDequeError::Clone")
                .field("0", e)
                .finish(),
            Self::Overflow => f.write_str("TryVecDequeError::Overflow"),
            Self::Other(msg) => f
                .try_debug_struct("TryVecDequeError::Other")
                .field("0", msg)
                .finish(),
        }
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

    /// Fallibly extend the deque with all elements from an iterator source.
    ///
    /// Accepts anything that implements [`ResumableSource`](crate::recovery::ResumableSource).
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
    fn try_extend_from_slice(&mut self, other: &[T]) -> Result<(), TryVecDequeError>
    where
        T: TryClone;

    /// Copies elements within the deque itself according to the given range.
    fn try_extend_from_within<R: RangeBounds<usize>>(
        &mut self,
        range: R,
    ) -> Result<(), TryVecDequeError>
    where
        T: TryClone;

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    fn try_append(&mut self, other: &mut VecDeque<T>) -> Result<(), TryReserveError>;

    // ── Mutation: resize / shrink / clear ───────────────────────────────────

    /// Resizes the deque so that `len` equals `new_len`.
    ///
    /// If `new_len` is greater than `len`, the deque is extended by cloning
    /// `value` via [`TryClone`]. If `new_len` is less than `len`, the deque
    /// is truncated. Returns [`TryVecDequeError::Reserve`] on allocation failure or
    /// [`TryVecDequeError::Clone`] if an element clone fails.
    fn try_resize(&mut self, value: &T, new_len: usize) -> Result<(), TryVecDequeError>
    where
        T: TryClone;

    /// Like [`Self::try_resize`] but uses a closure to produce new elements.
    fn try_resize_with<F>(&mut self, new_len: usize, f: F) -> Result<(), TryReserveError>
    where
        F: FnMut() -> T;

    /// Fallibly shrink the capacity of this deque to match its length.
    ///
    /// Converts the deque into a contiguous [`Vec`], shrinks the vector's buffer,
    /// and converts back. This is necessary because `VecDeque`'s internal ring
    /// buffer layout is opaque — we cannot directly reallocate its storage.
    ///
    /// **Deprecated:** This method name conflicts with the unstable inherent
    /// [`VecDeque::try_shrink_to_fit`](lang_alloc::collections::vec_deque::VecDeque::try_shrink_to_fit).
    /// Use [`Self::fallible_shrink_to_fit`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with unstable VecDeque::try_shrink_to_fit; use fallible_shrink_to_fit"
    )]
    fn try_shrink_to_fit(&mut self) -> Result<(), TryVecDequeError>;

    /// Fallibly shrink the capacity of this deque to at least `min_capacity`.
    ///
    /// Converts the deque into a contiguous [`Vec`], shrinks the vector's buffer,
    /// and converts back. The effective minimum capacity is `max(len, min_capacity)`.
    ///
    /// **Deprecated:** This method name conflicts with the unstable inherent
    /// [`VecDeque::try_shrink_to`](lang_alloc::collections::vec_deque::VecDeque::try_shrink_to).
    /// Use [`Self::fallible_shrink_to`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with unstable VecDeque::try_shrink_to; use fallible_shrink_to"
    )]
    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryVecDequeError>;

    /// Fallibly shrink the capacity of this deque to match its length.
    ///
    /// Converts the deque into a contiguous [`Vec`], shrinks the vector's buffer,
    /// and converts back. This is necessary because `VecDeque`'s internal ring
    /// buffer layout is opaque — we cannot directly reallocate its storage.
    ///
    /// This method replaces the deprecated [`Self::try_shrink_to_fit`] which
    /// shares its name with the unstable inherent [`VecDeque::try_shrink_to_fit`](lang_alloc::collections::vec_deque::VecDeque::try_shrink_to_fit).
    #[allow(deprecated)]
    fn fallible_shrink_to_fit(&mut self) -> Result<(), TryVecDequeError> {
        Self::try_shrink_to_fit(self)
    }

    /// Fallibly shrink the capacity of this deque to at least `min_capacity`.
    ///
    /// Converts the deque into a contiguous [`Vec`], shrinks the vector's buffer,
    /// and converts back. The effective minimum capacity is `max(len, min_capacity)`.
    ///
    /// This method replaces the deprecated [`Self::try_shrink_to`] which shares
    /// its name with the unstable inherent [`VecDeque::try_shrink_to`](lang_alloc::collections::vec_deque::VecDeque::try_shrink_to).
    #[allow(deprecated)]
    fn fallible_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryVecDequeError> {
        Self::try_shrink_to(self, min_capacity)
    }

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
    fn fallible_extend<S>(
        &mut self,
        source: S,
    ) -> Result<(), (TryReserveError, crate::recovery::Resumable<S::Inner>)>
    where
        S: crate::recovery::ResumableSource<Item = T>,
    {
        Self::try_extend(self, source)
    }

    /// Alias for [`Self::try_collect`].
    fn fallible_collect<I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<VecDeque<T>, TryReserveError> {
        Self::try_collect(iter)
    }
}

#[allow(deprecated)]
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

    fn try_extend<S>(
        &mut self,
        source: S,
    ) -> Result<(), (TryReserveError, crate::recovery::Resumable<S::Inner>)>
    where
        S: crate::recovery::ResumableSource<Item = T>,
    {
        use crate::recovery::Resumable;

        let (head, mut iter) = source.safe_into_iter();

        if let Some(item) = head {
            if self.len() == self.capacity()
                && let Err(e) = self.try_reserve(1)
            {
                return Err((e.into(), Resumable::new(item, iter)));
            }
            self.push_back(item);
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

    fn try_extend_from_slice(&mut self, other: &[T]) -> Result<(), TryVecDequeError>
    where
        T: TryClone,
    {
        if other.is_empty() {
            return Ok(());
        }
        self.try_reserve(other.len())
            .map_err(|e| TryVecDequeError::Reserve(e.into()))?;
        let guard = TruncateGuard::new(self);
        for item in other {
            match item.try_clone() {
                Ok(cloned) => guard.deque.push_back(cloned),
                Err(e) => {
                    return Err(TryVecDequeError::Clone(e));
                }
            }
        }
        guard.forget();
        Ok(())
    }

    fn try_extend_from_within<R: RangeBounds<usize>>(
        &mut self,
        range: R,
    ) -> Result<(), TryVecDequeError>
    where
        T: TryClone,
    {
        let start = match range.start_bound() {
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i.checked_add(1).ok_or(TryVecDequeError::Overflow)?,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&i) => i.checked_add(1).ok_or(TryVecDequeError::Overflow)?,
            Bound::Excluded(&i) => i,
            Bound::Unbounded => self.len(),
        };

        if start >= end {
            return Ok(());
        }

        if end > self.len() || start > self.len() {
            return Err(TryVecDequeError::Other(
                "extend_from_within: range out of bounds",
            ));
        }

        let count = end - start;
        self.try_reserve(count)
            .map_err(|e| TryVecDequeError::Reserve(e.into()))?;
        let guard = TruncateGuard::new(self);
        for i in start..end {
            match guard.deque[i].try_clone() {
                Ok(cloned) => guard.deque.push_back(cloned),
                Err(e) => {
                    return Err(TryVecDequeError::Clone(e));
                }
            }
        }
        guard.forget();
        Ok(())
    }

    // ── Mutation: resize / shrink / clear ───────────────────────────────────

    fn try_resize(&mut self, value: &T, new_len: usize) -> Result<(), TryVecDequeError>
    where
        T: TryClone,
    {
        let current = self.len();
        if new_len <= current {
            self.truncate(new_len);
            return Ok(());
        }
        let extra = new_len - current;
        self.try_reserve(extra)
            .map_err(|e| TryVecDequeError::Reserve(e.into()))?;
        let guard = TruncateGuard::new(self);
        for _ in 0..extra {
            match value.try_clone() {
                Ok(cloned) => guard.deque.push_back(cloned),
                Err(e) => {
                    return Err(TryVecDequeError::Clone(e));
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
        self.try_reserve(extra)?;
        let guard = TruncateGuard::new(self);
        for _ in 0..extra {
            guard.deque.push_back(f());
        }
        guard.forget();
        Ok(())
    }

    fn try_shrink_to_fit(&mut self) -> Result<(), TryVecDequeError> {
        <Self as TryVecDeque<T>>::try_shrink_to(self, self.len())
    }

    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryVecDequeError> {
        let target = cmp::max(self.len(), min_capacity);
        if self.capacity() <= target {
            return Ok(());
        }
        // Convert the deque into a contiguous Vec, shrink the Vec's buffer,
        // then convert back. We can't access VecDeque's internal ring-buffer
        // pointers directly, so this round-trip is the safest approach.
        let mut vec: Vec<T> = mem::take(self).into();
        let result = <Vec<T> as TryVec<T>>::fallible_shrink_to(&mut vec, target);
        // Recover the deque before error handling so that a shrink failure
        // does not silently discard the original data.
        *self = vec.into();
        result.map_err(|e| match e {
            TryVecError::Alloc(e) => TryVecDequeError::Alloc(e),
            TryVecError::Reserve(e) => TryVecDequeError::Reserve(e),
            TryVecError::Clone(_) => unreachable!("shrink does not clone"),
            TryVecError::Overflow => TryVecDequeError::Overflow,
            TryVecError::Other(msg) => TryVecDequeError::Other(msg),
        })
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

// ── TryDebug for VecDeque<T> ─────────────────────────────────────────────────

impl<T: crate::try_fmt::TryDebug> crate::try_fmt::TryDebug for VecDeque<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::try_fmt::helpers::FormatterExt;
        f.try_debug_list().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang_alloc::vec;
    use lang_alloc::vec::Vec;
    use lang_core::iter;

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
    fn try_insert_at_end() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(1).unwrap();
        dq.try_insert(1, 2).unwrap();
        assert_eq!(dq[0], 1);
        assert_eq!(dq[1], 2);
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
        dq.try_extend(iter::empty::<i32>()).unwrap();
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
    fn try_shrink_to_fit_reduces_capacity() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_reserve(512).unwrap();
        dq.try_push_back(42).unwrap();
        let cap_before = dq.capacity();
        assert!(cap_before >= 512);
        dq.try_shrink_to_fit().unwrap();
        assert!(dq.capacity() < cap_before);
        assert_eq!(dq.len(), 1);
        assert_eq!(dq[0], 42);
    }

    #[test]
    fn try_shrink_to_above_len_clamps_to_len() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_reserve(256).unwrap();
        dq.try_push_back(1).unwrap();
        dq.try_push_back(2).unwrap();
        // min_capacity > len is clamped to len
        dq.try_shrink_to(100).unwrap();
        assert_eq!(dq.len(), 2);
        assert!(dq.capacity() <= 256);
        assert_eq!(dq[0], 1);
        assert_eq!(dq[1], 2);
    }

    #[test]
    fn try_shrink_to_below_len_is_noop() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_reserve(64).unwrap();
        for i in 0..10 {
            dq.try_push_back(i).unwrap();
        }
        dq.try_shrink_to(2).unwrap();
        assert_eq!(dq.len(), 10);
        // capacity shouldn't go below len
        assert!(dq.capacity() >= 10);
    }

    #[test]
    fn try_shrink_to_already_small_is_noop() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_push_back(1).unwrap();
        dq.try_shrink_to(64).unwrap();
        assert_eq!(dq.len(), 1);
        assert_eq!(dq[0], 1);
    }

    #[test]
    fn try_shrink_preserves_order_after_pop_front() {
        let mut dq: VecDeque<i32> = VecDeque::new();
        dq.try_reserve(128).unwrap();
        for i in 0..5 {
            dq.try_push_back(i).unwrap();
        }
        dq.pop_front(); // remove 0, ring buffer is now split
        dq.try_shrink_to_fit().unwrap();
        assert_eq!(dq.len(), 4);
        assert_eq!(dq[0], 1);
        assert_eq!(dq[3], 4);
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
            <VecDeque<i32> as TryVecDeque<i32>>::try_collect(iter::empty::<i32>()).unwrap();
        assert!(dq.is_empty());
    }

    #[test]
    fn try_from_slice_clones() {
        let slice: &[Vec<u8>] = &[vec![10], vec![20]];
        let dq: VecDeque<Vec<u8>> =
            <VecDeque<Vec<u8>> as TryVecDeque<Vec<u8>>>::try_from_slice(slice).unwrap();
        assert_eq!(dq[0], [10u8]);
        assert_eq!(dq[1], [20u8]);
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

    // ── OOM tests ─────────────────────────────────────────────────────────────
    #[cfg(test)]
    use rustyfill_test_allocator::{FailPolicy, with_policy};

    #[test]
    fn vecdeque_try_with_capacity_fails_on_oom() {
        let r: Result<VecDeque<u32>, TryReserveError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <VecDeque<u32> as TryVecDeque<u32>>::try_with_capacity(10)
            });
        assert!(r.is_err());
    }

    #[test]
    fn vecdeque_try_with_capacity_zero_succeeds_under_oom() {
        let r: Result<VecDeque<u32>, TryReserveError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <VecDeque<u32> as TryVecDeque<u32>>::try_with_capacity(0)
            });
        assert!(r.is_ok());
    }

    #[test]
    fn vecdeque_try_push_back_fails_on_oom() {
        let mut dq: VecDeque<u32> = VecDeque::new();
        dq.try_shrink_to_fit().unwrap();
        let r = with_policy(FailPolicy::fail_next_alloc(), || dq.fallible_push_back(1));
        assert!(r.is_err());
    }

    #[test]
    fn vecdeque_try_push_front_fails_on_oom() {
        let mut dq: VecDeque<u32> = VecDeque::new();
        dq.try_shrink_to_fit().unwrap();
        let r = with_policy(FailPolicy::fail_next_alloc(), || dq.fallible_push_front(1));
        assert!(r.is_err());
    }

    #[test]
    fn vecdeque_try_clone_fails_on_oom() {
        let orig: VecDeque<u32> = VecDeque::from([1, 2, 3]);
        let r: Result<VecDeque<u32>, TryCloneError> =
            with_policy(FailPolicy::fail_next_alloc(), || orig.try_clone());
        assert!(r.is_err());
    }

    #[test]
    fn vecdeque_try_clone_empty_succeeds_under_oom() {
        let orig: VecDeque<u32> = VecDeque::new();
        let r: Result<VecDeque<u32>, TryCloneError> =
            with_policy(FailPolicy::fail_next_alloc(), || orig.try_clone());
        assert!(r.is_ok());
    }

    #[test]
    fn vecdeque_try_collect_fails_on_oom() {
        let items = [1u32, 2u32, 3u32];
        let r: Result<VecDeque<u32>, TryReserveError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                VecDeque::try_collect(items.iter().copied())
            });
        assert!(r.is_err());
    }

    #[test]
    fn vecdeque_try_from_slice_fails_on_oom() {
        let slice = &[1u32, 2u32];
        let r: Result<VecDeque<u32>, TryVecDequeError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <VecDeque<u32> as TryVecDeque<u32>>::try_from_slice(slice)
            });
        assert!(r.is_err());
    }

    #[test]
    fn vecdeque_try_from_elem_fails_on_oom() {
        let val = 42u32;
        let r: Result<VecDeque<u32>, TryVecDequeError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <VecDeque<u32> as TryVecDeque<u32>>::try_from_elem(&val, 5)
            });
        assert!(r.is_err());
    }

    #[test]
    fn vecdeque_oom_restores_allocation_afterwards() {
        let r: Result<VecDeque<u32>, TryReserveError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <VecDeque<u32> as TryVecDeque<u32>>::try_with_capacity(10)
            });
        assert!(r.is_err());
        let r: Result<VecDeque<u32>, TryReserveError> =
            <VecDeque<u32> as TryVecDeque<u32>>::try_with_capacity(10);
        assert!(r.is_ok());
    }

    #[test]
    fn vecdeque_nth_alloc_fail_targets_correct_call() {
        let (r1_ok, r2_err, r3_ok) = with_policy(FailPolicy::fail_nth_alloc(2), || {
            let r1: Result<VecDeque<u8>, TryReserveError> =
                <VecDeque<u8> as TryVecDeque<u8>>::try_with_capacity(1);
            let r2: Result<VecDeque<u8>, TryReserveError> =
                <VecDeque<u8> as TryVecDeque<u8>>::try_with_capacity(1);
            let r3: Result<VecDeque<u8>, TryReserveError> =
                <VecDeque<u8> as TryVecDeque<u8>>::try_with_capacity(1);
            (r1.is_ok(), r2.is_err(), r3.is_ok())
        });
        assert!(r1_ok, "first alloc should succeed");
        assert!(r2_err, "second alloc should fail");
        assert!(r3_ok, "third alloc should succeed");
    }

    // ── Explicit rollback / TruncateGuard tests ─────────────────────────────

    #[test]
    fn extend_from_slice_rollback_on_mid_way_clone_failure() {
        // try_extend_from_slice on VecDeque<String> reserves capacity upfront,
        // then clones each element. A mid-way clone failure must trigger the
        // TruncateGuard to drop all elements pushed before the failure.
        use lang_alloc::string::String;

        let source: Vec<String> = vec![
            "item0".into(), "item1".into(), "item2".into(), "item3".into(),
            "item4".into(), "item5".into(), "item6".into(), "item7".into(),
            "item8".into(), "item9".into(),
        ];
        let len_source = source.len();

        let mut deque: VecDeque<String> = VecDeque::from([
            "pre0".into(), "pre1".into(), "pre2".into(),
        ]);
        let len_before = deque.len();

        let r: Result<(), TryVecDequeError> =
            with_policy(FailPolicy::fail_nth_alloc(2), || {
                <VecDeque<String> as TryVecDeque<String>>::try_extend_from_slice(&mut deque, &source)
            });

        match r {
            Err(TryVecDequeError::Clone(_)) => {
                assert_eq!(
                    deque.len(),
                    len_before,
                    "TruncateGuard did not roll back: expected {} elements, got {}",
                    len_before,
                    deque.len()
                );
                assert_eq!(deque[0], "pre0");
                assert_eq!(deque[1], "pre1");
                assert_eq!(deque[2], "pre2");
            }
            Ok(()) => {
                assert_eq!(deque.len(), len_before + len_source);
            }
            Err(other) => {
                panic!("unexpected error variant: {:?}", other);
            }
        }
    }

    #[test]
    fn extend_from_within_rollback_on_mid_way_clone_failure() {
        // try_extend_from_within clones from within the same deque.
        // Mid-way failure must truncate back to original length.
        use lang_alloc::string::String;

        let mut deque: VecDeque<String> = VecDeque::from([
            "a".into(), "b".into(), "c".into(), "d".into(), "e".into(),
        ]);
        let len_before = deque.len();

        let r: Result<(), TryVecDequeError> =
            with_policy(FailPolicy::fail_nth_alloc(2), || {
                <VecDeque<String> as TryVecDeque<String>>::try_extend_from_within(&mut deque, 0..3)
            });

        match r {
            Err(TryVecDequeError::Clone(_)) => {
                assert_eq!(deque.len(), len_before,
                    "TruncateGuard failed to roll back extend_from_within");
                assert_eq!(deque[0], "a");
                assert_eq!(deque[4], "e");
            }
            Ok(()) => {
                assert_eq!(deque.len(), len_before + 3);
            }
            Err(other) => {
                panic!("unexpected error: {:?}", other);
            }
        }
    }

    #[test]
    fn resize_with_clone_rollback_on_mid_way_failure() {
        // try_resize_with clones a value repeatedly. Mid-way failure must
        // truncate back to original length via TruncateGuard.
        use lang_alloc::string::String;

        let val: String = "repeated".into();
        let mut deque: VecDeque<String> = VecDeque::from(["original".into()]);
        let len_before = deque.len();

        let r: Result<(), TryVecDequeError> =
            with_policy(FailPolicy::fail_nth_alloc(3), || {
                <VecDeque<String> as TryVecDeque<String>>::try_resize(&mut deque, &val, 15)
            });

        match r {
            Err(TryVecDequeError::Clone(_)) => {
                assert_eq!(deque.len(), len_before,
                    "resize rollback failed: expected {}, got {}", len_before, deque.len());
                assert_eq!(deque[0], "original");
            }
            Ok(()) => {
                assert_eq!(deque.len(), 15);
            }
            Err(other) => {
                panic!("unexpected error: {:?}", other);
            }
        }
    }
}
