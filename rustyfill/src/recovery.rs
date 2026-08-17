//! Iterator recovery for fallible extension operations.
//!
//! When a [`try_extend`](crate::try_extend::TryExtend::try_extend)-style operation fails,
//! elements from the iterator may have been consumed but not yet committed to
//! the collection. This module provides [`Resumable`] so that callers can
//! re-package a stranded element alongside the remainder and pass it back in.
//!
//! ## Stable types across retries
//!
//! The trait [`ResumableSource`] separates the *source* of items (which may carry
//! a head) from the *inner iterator* that produces them:
//!
//! - A plain `IntoIterator` implements [`ResumableSource`] trivially — no head,
//!   inner type is its own `IntoIter`.
//! - A [`Resumable<I>`] carries an optional head and wraps the same inner
//!   iterator `I`. Its `Inner` associated type is also `I`.
//!
//! Because `try_extend` accepts anything implementing [`ResumableSource`], both
//! raw sources and [`Resumable`] wrappers satisfy the same bound, and the error
//! type is always parameterized over the stable inner iterator:
//!
//! ```text
//! First call:  try_extend(range)           -> Err(Resumable<Range<i32>>)
//! Retry:      try_extend(retryable)        -> Err(Resumable<Range<i32>>)
//! Third call: same shape again              -> Err(Resumable<Range<i32>>)
//! ```
//!
//! No new generic parameters are introduced on retry — the type never grows.
//!
//! Two scenarios produce a [`Resumable`]:
//!
//! 1. **Initial reserve failure** — no elements were consumed. The [`Resumable`]
//!    has no head; only the full remainder iterator is present.
//!
//! 2. **Mid-iteration failure** — one element was popped but could not be
//!    inserted. The [`Resumable`] holds that stranded element as the head,
//!    plus the unconsumed remainder.

use crate::try_fmt::{TryDebug, helpers::FormatterExt};
use lang_core::fmt;

/// Trait for fallible iterators that may stall on allocation errors.
///
/// By default, iterators that implement this trait *stall*: when an internal
/// allocation fails they emit an `Err` item and hold the pending work so that
/// retrying [`Iterator::next`] re-attempts the failed operation. This preserves
/// correctness — no data is silently lost.
///
/// Callers that prefer progress over completeness can opt out:
///
/// - **Automatic**: call [`with_auto_unstall`](Stallable::with_auto_unstall) to get
///   back the same iterator configured so that each error is emitted once and the
///   pending item is automatically discarded. Composable since it consumes and
///   returns `Self`.
/// - **Manual**: call [`unstall`](Stallable::unstall) after seeing an `Err` to
///   discard the stalled item and move on.
///
/// Stalling is the default because it never loses data. Skipping is opt-in.
pub trait Stallable: Iterator {
    /// Consume a pending/stalled item, if any, so that iteration can proceed past
    /// the point of failure.
    ///
    /// Returns `true` if there was a pending item that was discarded.
    fn unstall(&mut self) -> bool;

    /// Return `self` with automatic unstalling toggled.
    ///
    /// When `auto` is `true`, each error is emitted once and the pending item is
    /// automatically discarded so iteration continues. When `auto` is `false`
    /// (the default), errors are repeated until the underlying operation succeeds
    /// or [`Self::unstall`] is called manually.
    fn with_auto_unstall(self, auto: bool) -> Self
    where
        Self: Sized,
    {
        let mut s = self;
        s.set_auto_unstall(auto);
        s
    }

    /// Set whether the iterator automatically discards pending items after emitting
    /// an error.
    fn set_auto_unstall(&mut self, auto: bool);
}

/// A source of items that decomposes into an optional leading element and an
/// inner iterator.
///
/// Any `IntoIterator` implements this via blanket — no head, inner is itself.
/// [`Resumable<I>`] implements it explicitly, carrying an optional head while
/// still exposing the same inner iterator `I`.
pub trait ResumableSource {
    /// The item type produced by this source.
    type Item;
    /// The stable inner iterator type. For a plain iterator this is `Self`;
    /// for a [`Resumable<I>`] this is `I`.
    type Inner: Iterator<Item = Self::Item>;

    /// Decompose into an optional leading element and the inner iterator.
    fn safe_into_iter(self) -> (Option<Self::Item>, Self::Inner);
}

impl<I: IntoIterator> ResumableSource for I {
    type Item = I::Item;
    type Inner = I::IntoIter;

    fn safe_into_iter(self) -> (Option<Self::Item>, Self::Inner) {
        (None, self.into_iter())
    }
}

/// Wraps an optional stranded element alongside a remainder iterator, allowing
/// the caller to pass both back into a fallible extend operation.
///
/// Constructed either directly by the caller or returned inside the error of
/// a failed `try_extend`. Implements [`ResumableSource`] with `Inner = I`, so
/// passing it back into `try_extend` preserves the same error type.
///
/// # Example
///
/// ```rust,ignore
/// use rustyfill::prelude::*;
///
/// let mut vec = Vec::<i32>::new();
/// let items = 0..10_000;
///
/// // First call — `remaining` is Range<i32>.
/// let remaining = match vec.try_extend(items) {
///     Ok(()) => return,
///     Err((_err, resumable)) => {
///         // Back off...
///         resumable.into_remainder()
///     }
/// };
///
/// // Retry — construct a Resumable from whatever we have left.
/// let remaining = match vec.try_extend(Resumable::from_remainder(remaining)) {
///     Ok(()) => return,
///     Err((_err, resumable)) => resumable.into_remainder(),
/// };
/// ```
///
/// Requires the `std` feature (enabled by default).
pub struct Resumable<I>
where
    I: Iterator,
{
    head: Option<I::Item>,
    remainder: I,
}

impl<I> Resumable<I>
where
    I: Iterator,
{
    /// Create a [`Resumable`] with a stranded element and the remainder.
    pub fn new(head: I::Item, remainder: I) -> Self {
        Self {
            head: Some(head),
            remainder,
        }
    }

    /// Create a [`Resumable`] with no stranded element — only the remainder.
    pub fn from_remainder(remainder: I) -> Self {
        Self {
            head: None,
            remainder,
        }
    }

    /// Returns `true` if there is a stranded head element.
    pub fn has_head(&self) -> bool {
        self.head.is_some()
    }

    /// Returns a reference to the stranded element, or `None`.
    pub fn head(&self) -> Option<&I::Item> {
        self.head.as_ref()
    }

    /// Returns a mutable reference to the stranded element, or `None`.
    pub fn head_mut(&mut self) -> Option<&mut I::Item> {
        self.head.as_mut()
    }

    /// Returns a reference to the remainder iterator.
    pub fn remainder(&self) -> &I {
        &self.remainder
    }

    /// Returns a mutable reference to the remainder iterator.
    pub fn remainder_mut(&mut self) -> &mut I {
        &mut self.remainder
    }

    /// Consumes this value, returning the remainder iterator. Drops the head.
    pub fn into_remainder(self) -> I {
        self.remainder
    }

    /// Consumes this value, returning both parts.
    pub fn into_parts(self) -> (Option<I::Item>, I) {
        (self.head, self.remainder)
    }
}

impl<I> ResumableSource for Resumable<I>
where
    I: Iterator,
{
    type Item = I::Item;
    type Inner = I;

    fn safe_into_iter(self) -> (Option<Self::Item>, Self::Inner) {
        (self.head, self.remainder)
    }
}

impl<I> fmt::Debug for Resumable<I>
where
    I: Iterator + fmt::Debug,
    I::Item: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Resumable")
            .field("head", &self.head)
            .field("remainder", &self.remainder)
            .finish()
    }
}

impl<I> TryDebug for Resumable<I>
where
    I: Iterator + TryDebug,
    I::Item: TryDebug,
{
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.try_debug_struct("Resumable")
            .field("head", &self.head)
            .field("remainder", &self.remainder)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang_alloc::format;
    use lang_alloc::vec;
    use lang_alloc::vec::Vec;
    use lang_core::ops;

    #[test]
    fn retryable_with_head() {
        let r = Resumable::new(42, 3..6);
        assert!(r.has_head());
        assert_eq!(*r.head().unwrap(), 42);
        assert_eq!(r.remainder().size_hint(), (3, Some(3)));
    }

    #[test]
    fn retryable_without_head() {
        let r = Resumable::from_remainder(0..5);
        assert!(!r.has_head());
        assert!(r.head().is_none());
        assert_eq!(r.remainder().size_hint(), (5, Some(5)));
    }

    #[test]
    fn safe_into_iter_yields_head_then_remainder() {
        let r = Resumable::new(0, 1..4);
        let (head, mut iter) = r.safe_into_iter();
        assert_eq!(head, Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn safe_into_iter_no_head() {
        let r = Resumable::from_remainder(0..4);
        let (head, iter) = r.safe_into_iter();
        assert!(head.is_none());
        let expected: Vec<i32> = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(iter.collect::<Vec<_>>(), expected);
    }

    #[test]
    fn into_remainder_drops_head() {
        let r = Resumable::new(7, 1..4);
        let rem = r.into_remainder();
        let expected: Vec<i32> = [1, 2, 3].into_iter().collect();
        assert_eq!(rem.collect::<Vec<_>>(), expected);
    }

    #[test]
    fn into_parts() {
        let r = Resumable::new(99, 10..12);
        let (h, i) = r.into_parts();
        assert_eq!(h, Some(99));
        let expected: Vec<i32> = [10, 11].into_iter().collect();
        assert_eq!(i.collect::<Vec<_>>(), expected);
    }

    #[test]
    fn stable_type_across_retries() {
        type Base = ops::Range<i32>;

        // Simulate first failure producing Resumable<Base>.
        let r1: Resumable<Base> = Resumable::new(0, 1..4);
        let (_head, inner): (_, Base) = r1.safe_into_iter();

        // User constructs second Resumable from the same base type.
        let r2: Resumable<Base> = Resumable::from_remainder(inner);
        let (_head2, mut inner2): (_, Base) = r2.safe_into_iter();

        // Still Base, never Resumable<Resumable<Base>>.
        assert_eq!(inner2.next(), Some(1));
    }

    #[test]
    fn blanket_safe_iterable_for_range() {
        let range = 10..13;
        let (head, inner): (Option<i32>, _) = range.safe_into_iter();
        assert!(head.is_none());
        let expected: Vec<i32> = [10, 11, 12].into_iter().collect();
        assert_eq!(inner.collect::<Vec<_>>(), expected);
    }

    #[test]
    fn blanket_safe_iterable_for_vec() {
        let v = vec![1, 2, 3];
        let (head, inner): (Option<i32>, _) = v.safe_into_iter();
        assert!(head.is_none());
        assert_eq!(inner.collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn debug_output() {
        let r = Resumable::new(42, 1..3);
        let s = format!("{r:?}");
        assert!(s.contains("Resumable"));
        assert!(s.contains("Some(42)"));
    }
}
