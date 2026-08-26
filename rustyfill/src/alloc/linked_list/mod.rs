//! Fallible linked-list operations and formatting for [`LinkedList<T>`].
//!
//! Provides:
//! - [`TryClone`] for `LinkedList<T>` when `T: TryClone` — clones each element
//!   in order, pushing onto a fresh list.
//! - [`TryDefault`] for `LinkedList<T>` — an empty list needs no allocation.
//! - [`TryDebug`] for `LinkedList<T>` — mirrors std's bracketed, comma-joined
//!   rendering, routing each element through its fallible formatter.
//! - [`TryLinkedList`] — fallible versions of the OOM-prone mutation methods
//!   (`push_front`, `push_back`, bulk construction) that return
//!   [`AllocError`] instead of panicking.
//! - [`TryExtend`]/[`TryExtendFromSlice`] impls so
//!   generic fallible-extension code works over lists as well.
//!
//! # Implementation strategy
//!
//! Unlike `Vec` or `BinaryHeap`, `LinkedList` allocates one node at a time with
//! no reserve API. The canonical fallible primitives are
//! [`TryLinkedList::try_push_front_mut_give_back`] and
//! [`TryLinkedList::try_push_back_mut_give_back`], which allocate a fully
//! initialized [`sys::Node<T>`] via [`TryBox::try_new_give_back`], convert it
//! to a raw pointer, and splice it into the list by operating on the
//! transmutated [`sys::LinkedList<T>`] representation. All other fallible
//! methods derive from these two.
//!
//! Because we control the node allocation ourselves (rather than delegating to
//! the infallible `push_front`/`push_back` which would perform a second,
//! uncontrolled allocation), every operation performs exactly one heap
//! allocation per element — the same as std.

use crate::alloc::AllocError;
use crate::alloc::boxed::TryBox;
use crate::recovery::Resumable;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_extend::{TryExtend, TryExtendFromSlice};
use crate::try_fmt::{TryDebug, TryDisplay};
use lang_alloc::boxed::Box;
use lang_alloc::collections::LinkedList;
use lang_core::fmt;
use lang_core::mem;
use lang_core::ptr::NonNull;

mod sys {
    pub use rustyfill_sys::std::collections::linked_list::{LinkedList as SysLinkedList, Node};
}

// ── Low-level node helpers ────────────────────────────────────────────────────

/// Fallibly allocate a fully-initialized `Node<T>` on the heap.
///
/// Uses [`TryBox::try_new_give_back`] so that on allocation failure the
/// element is handed back untouched. On success the boxed node is converted
/// to a raw pointer via [`Box::into_raw`] — from that moment the list owns
/// the node, exactly like std's `push_front`/`push_back`.
fn try_alloc_node<T>(element: T) -> Result<NonNull<sys::Node<T>>, (T, AllocError)> {
    let boxed: Box<sys::Node<T>> =
        match <Box<sys::Node<T>> as TryBox<_>>::try_new_give_back(sys::Node {
            next: None,
            prev: None,
            element,
        }) {
            Ok(b) => b,
            Err((node, e)) => {
                // Recover the element from the failed node so we can hand it
                // back to the caller. The node itself is freed by the Box drop.
                return Err((node.element, e));
            }
        };
    Ok(unsafe { NonNull::new_unchecked(Box::into_raw(boxed)) })
}

/// Transmute `&mut LinkedList<U>` to `&mut sys::SysLinkedList<U>`.
///
/// # Safety
/// The layout of `lang_alloc::collections::LinkedList<U>` (with default
/// allocator) is guaranteed identical to `sys::SysLinkedList<U>` by the
/// compile-time binding check in `rustyfill-sys`. Both have fields
/// `{ head: Option<NonNull<Node<U>>>, tail: Option<NonNull<Node<U>>>, len: usize,
/// alloc: (), marker: PhantomData<Box<Node<U>, ()>> }`.
unsafe fn as_sys_mut<U>(list: &mut LinkedList<U>) -> &mut sys::SysLinkedList<U> {
    unsafe { &mut *(list as *mut LinkedList<U> as *mut sys::SysLinkedList<U>) }
}

/// Panic-safe guard that pops a `LinkedList` back down to its original length
/// on drop unless disarmed via `forget()`. Used by fallible bulk-append methods
/// so that if an element's `try_clone` or node allocation fails (or panics)
/// mid-way, partially-appended elements are removed rather than left behind.
struct TruncateGuard<'a, T> {
    list: &'a mut LinkedList<T>,
    len_before: usize,
}

impl<'a, T> TruncateGuard<'a, T> {
    fn new(list: &'a mut LinkedList<T>) -> Self {
        Self {
            len_before: list.len(),
            list,
        }
    }

    /// Disable the guard — no rollback on scope exit.
    fn forget(self) {
        mem::forget(self);
    }
}

impl<T> Drop for TruncateGuard<'_, T> {
    fn drop(&mut self) {
        while self.list.len() > self.len_before {
            self.list.pop_back();
        }
    }
}

// ── TryClone for LinkedList<T> ────────────────────────────────────────────────

impl<T: TryClone> TryClone for LinkedList<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let mut out = LinkedList::<T>::new();
        for elem in self.iter() {
            match elem.try_clone() {
                Ok(cloned) => out
                    .try_push_back(cloned)
                    .map_err(|_| TryCloneError::Other("allocation failed"))?,
                Err(e) => {
                    drop(out);
                    return Err(e);
                }
            }
        }
        Ok(out)
    }
}

// ── TryDefault for LinkedList<T> ──────────────────────────────────────────────

impl<T: TryDefault> TryDefault for LinkedList<T> {
    fn try_default() -> Result<Self, TryDefaultError> {
        // An empty list requires no allocation.
        Ok(LinkedList::new())
    }
}

// ── TryDebug for LinkedList<T> ────────────────────────────────────────────────

impl<T: TryDebug> TryDebug for LinkedList<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::try_fmt::helpers::FormatterExt;
        f.try_debug_list().entries(self.iter()).finish()
    }
}

// NOTE: no `TryDisplay` impl — `LinkedList` does not implement `fmt::Display`.

// ── Error types ───────────────────────────────────────────────────────────────

/// Error for fallible `LinkedList` operations that allocate and/or clone elements.
///
/// Covers `try_from_slice` and slice-based extension — any operation whose
/// failure modes are limited to a node allocation
/// ([`AllocError`]) or an element clone failure
/// ([`TryCloneError`]).
pub enum TryLinkedListWithCloneError {
    /// A node allocation failed (OOM).
    Alloc(AllocError),
    /// An element clone failed during a method that requires `TryClone`.
    Clone(TryCloneError),
}

impl fmt::Debug for TryLinkedListWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(self, f)
    }
}

impl fmt::Display for TryLinkedListWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }
}

impl From<AllocError> for TryLinkedListWithCloneError {
    fn from(err: AllocError) -> Self {
        Self::Alloc(err)
    }
}

impl From<TryCloneError> for TryLinkedListWithCloneError {
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

impl TryDebug for TryLinkedListWithCloneError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Alloc(e) => u::debug_field(f, "TryLinkedListWithCloneError::Alloc", e),
            Self::Clone(e) => u::debug_field(f, "TryLinkedListWithCloneError::Clone", e),
        }
    }
}

impl TryDisplay for TryLinkedListWithCloneError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Alloc(e) => u::display_delegated(f, "linked list", e),
            Self::Clone(e) => u::display_delegated(f, "linked list", e),
        }
    }
}

// ── TryLinkedList ─────────────────────────────────────────────────────────────

/// A trait for fallible linked-list operations.
///
/// Implemented for `LinkedList<T>`. Mirrors the OOM-prone `LinkedList` methods
/// (`push_front`, `push_back`, bulk construction) but returns [`Result`] values
/// that propagate [`AllocError`] on failure instead of panicking.
///
/// The canonical primitives are [`Self::try_push_front_mut_give_back`] and
/// [`Self::try_push_back_mut_give_back`]: they allocate a single fully
/// initialized `Node<T>` via [`TryBox::try_new_give_back`] and splice it into
/// the list through raw pointer surgery on the mirrored struct fields. All
/// other fallible methods compose these.
///
/// Because `LinkedList` allocates one node per element (there is no bulk
/// reserve API), multi-node operations can fail mid-way. Those methods keep
/// whatever was already inserted and hand back the stranded element plus the
/// remainder so the caller can retry.
pub trait TryLinkedList<T>: Sized {
    // ── Canonical primitives ─────────────────────────────────────────────────

    /// Fallibly prepend an element to the front of the list.
    ///
    /// Allocates a single `Node<T>`, initializes it with `value`, and splices it
    /// before the current head. Returns `(T, AllocError)` if the node
    /// allocation fails, giving ownership of `value` back to the caller.
    ///
    /// This is the canonical fallible push primitive; all other front-insertion
    /// methods derive from it.
    fn try_push_front_mut_give_back(&mut self, value: T) -> Result<(), (T, AllocError)>;

    /// Fallibly append an element to the back of the list.
    ///
    /// Allocates a single `Node<T>`, initializes it with `value`, and splices it
    /// after the current tail. Returns `(T, AllocError)` if the node
    /// allocation fails, giving ownership of `value` back to the caller.
    ///
    /// This is the canonical fallible push primitive; all other back-insertion
    /// methods derive from it.
    fn try_push_back_mut_give_back(&mut self, value: T) -> Result<(), (T, AllocError)>;

    // ── Derived push methods ─────────────────────────────────────────────────

    /// Fallibly prepend an element to the front of the list.
    ///
    /// Returns [`AllocError`] if the new node cannot be allocated. The list is
    /// left untouched on failure.
    fn try_push_front(&mut self, value: T) -> Result<(), AllocError> {
        self.try_push_front_mut_give_back(value).map_err(|(_, e)| e)
    }

    /// Like [`Self::try_push_front`] but returns ownership of `value` back on
    /// failure.
    fn try_push_front_give_back(&mut self, value: T) -> Result<(), (T, AllocError)> {
        self.try_push_front_mut_give_back(value)
    }

    /// Fallibly append an element to the back of the list.
    ///
    /// Returns [`AllocError`] if the new node cannot be allocated. The list is
    /// left untouched on failure.
    fn try_push_back(&mut self, value: T) -> Result<(), AllocError> {
        self.try_push_back_mut_give_back(value).map_err(|(_, e)| e)
    }

    /// Like [`Self::try_push_back`] but returns ownership of `value` back on
    /// failure.
    fn try_push_back_give_back(&mut self, value: T) -> Result<(), (T, AllocError)> {
        self.try_push_back_mut_give_back(value)
    }

    // ── Construction ────────────────────────────────────────────────────────

    /// Fallibly collect an iterator into a `LinkedList<T>`.
    ///
    /// Nodes are allocated one at a time; on a node-allocation failure the
    /// partially-built list is dropped, and the stranded element plus the
    /// unconsumed remainder are handed back alongside the error so the caller
    /// can retry.
    fn try_collect<I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<LinkedList<T>, (Option<T>, I::IntoIter, AllocError)>;

    /// Fallibly create a `LinkedList<T>` from a slice by cloning each element
    /// via [`TryClone`].
    ///
    /// Returns [`TryLinkedListWithCloneError::Alloc`] if a node allocation
    /// fails, or [`TryLinkedListWithCloneError::Clone`] if an element's
    /// [`TryClone::try_clone`] fails. On either failure the partially-built
    /// list is discarded.
    fn try_from_slice(slice: &[T]) -> Result<LinkedList<T>, TryLinkedListWithCloneError>
    where
        T: TryClone;

    // ── Bulk extension ───────────────────────────────────────────────────────

    /// Fallibly append all elements from another slice by cloning each one.
    ///
    /// Rolls back to the pre-call state on any failure (allocation or clone),
    /// so no partially-appended elements remain. Because `LinkedList` appends
    /// in order and never reorders, undoing this call is simply popping the
    /// elements added during it off the back — no side buffer needed.
    fn try_extend_from_slice_with_rollback(
        &mut self,
        other: &[T],
    ) -> Result<(), TryLinkedListWithCloneError>
    where
        T: TryClone;

    // ── Aliases with `fallible_` prefix ─────────────────────────────────────

    /// Alias for [`Self::try_push_front_mut_give_back`].
    fn fallible_push_front_mut_give_back(&mut self, value: T) -> Result<(), (T, AllocError)> {
        self.try_push_front_mut_give_back(value)
    }

    /// Alias for [`Self::try_push_back_mut_give_back`].
    fn fallible_push_back_mut_give_back(&mut self, value: T) -> Result<(), (T, AllocError)> {
        self.try_push_back_mut_give_back(value)
    }

    /// Alias for [`Self::try_push_front`].
    fn fallible_push_front(&mut self, value: T) -> Result<(), AllocError> {
        self.try_push_front(value)
    }

    /// Alias for [`Self::try_push_front_give_back`].
    fn fallible_push_front_give_back(&mut self, value: T) -> Result<(), (T, AllocError)> {
        self.try_push_front_give_back(value)
    }

    /// Alias for [`Self::try_push_back`].
    fn fallible_push_back(&mut self, value: T) -> Result<(), AllocError> {
        self.try_push_back(value)
    }

    /// Alias for [`Self::try_push_back_give_back`].
    fn fallible_push_back_give_back(&mut self, value: T) -> Result<(), (T, AllocError)> {
        self.try_push_back_give_back(value)
    }

    /// Alias for [`Self::try_collect`].
    fn fallible_collect<I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<LinkedList<T>, (Option<T>, I::IntoIter, AllocError)> {
        Self::try_collect(iter)
    }

    /// Alias for [`Self::try_from_slice`].
    fn fallible_from_slice(slice: &[T]) -> Result<LinkedList<T>, TryLinkedListWithCloneError>
    where
        T: TryClone,
    {
        Self::try_from_slice(slice)
    }

    /// Alias for [`Self::try_extend_from_slice_with_rollback`].
    fn fallible_extend_from_slice_with_rollback(
        &mut self,
        other: &[T],
    ) -> Result<(), TryLinkedListWithCloneError>
    where
        T: TryClone,
    {
        Self::try_extend_from_slice_with_rollback(self, other)
    }
}

impl<T> TryLinkedList<T> for LinkedList<T> {
    // ── Canonical primitives ─────────────────────────────────────────────────
    fn try_push_front_mut_give_back(&mut self, value: T) -> Result<(), (T, AllocError)> {
        // Phase 1: allocate a fully-initialized node (the only fallible step).
        let mut node = match try_alloc_node(value) {
            Ok(n) => n,
            Err((v, e)) => return Err((v, e)),
        };
        // Phase 2: splice into the list (infallible pointer surgery).
        // SAFETY: `self` is a valid `LinkedList<T>` whose layout matches
        // `sys::SysLinkedList<T>` (enforced at compile time by rustyfill-sys).
        let sys_self = unsafe { as_sys_mut(self) };
        unsafe {
            // Mirror std's push_front_node. We use `as_mut()` on individual
            // nodes in sequence (never two simultaneously) to avoid creating
            // overlapping mutable references to whole nodes.
            node.as_mut().next = sys_self.head;
            node.as_mut().prev = None;

            let node_opt = Some(node);
            match sys_self.head {
                None => sys_self.tail = node_opt,
                Some(mut head) => head.as_mut().prev = node_opt,
            }
            sys_self.head = node_opt;
            // Overflow is unreachable: each node holds two pointers plus the
            // element, so a list of usize::MAX nodes would need >= 2 * ptr_size * usize::MAX
            // bytes — far beyond any addressable memory.
            sys_self.len = sys_self
                .len
                .checked_add(1)
                .expect("linked list length overflow");
        }
        Ok(())
    }

    fn try_push_back_mut_give_back(&mut self, value: T) -> Result<(), (T, AllocError)> {
        // Phase 1: allocate a fully-initialized node (the only fallible step).
        let mut node = match try_alloc_node(value) {
            Ok(n) => n,
            Err((v, e)) => return Err((v, e)),
        };
        // Phase 2: splice into the list (infallible pointer surgery).
        // SAFETY: same layout guarantee as above.
        let sys_self = unsafe { as_sys_mut(self) };
        unsafe {
            // Mirror std's push_back_node. Same discipline: one `as_mut()` at
            // a time, never overlapping.
            node.as_mut().next = None;
            node.as_mut().prev = sys_self.tail;

            let node_opt = Some(node);
            match sys_self.tail {
                None => sys_self.head = node_opt,
                Some(mut tail) => tail.as_mut().next = node_opt,
            }
            sys_self.tail = node_opt;
            // Overflow is unreachable: each node holds two pointers plus the
            // element, so a list of usize::MAX nodes would need >= 2 * ptr_size * usize::MAX
            // bytes — far beyond any addressable memory.
            sys_self.len = sys_self
                .len
                .checked_add(1)
                .expect("linked list length overflow");
        }
        Ok(())
    }

    // ── Construction ────────────────────────────────────────────────────────

    fn try_collect<I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<LinkedList<T>, (Option<T>, I::IntoIter, AllocError)> {
        let mut iter = iter.into_iter();
        let mut list = LinkedList::<T>::new();
        while let Some(item) = iter.next() {
            if let Err((item, e)) = list.try_push_back_mut_give_back(item) {
                return Err((Some(item), iter, e));
            }
        }
        Ok(list)
    }

    fn try_from_slice(slice: &[T]) -> Result<LinkedList<T>, TryLinkedListWithCloneError>
    where
        T: TryClone,
    {
        let mut list = LinkedList::<T>::new();
        for item in slice {
            let cloned = item
                .try_clone()
                .map_err(TryLinkedListWithCloneError::Clone)?;
            if let Err((_, e)) = list.try_push_back_mut_give_back(cloned) {
                return Err(TryLinkedListWithCloneError::Alloc(e));
            }
        }
        Ok(list)
    }

    // ── Bulk extension ───────────────────────────────────────────────────────

    fn try_extend_from_slice_with_rollback(
        &mut self,
        other: &[T],
    ) -> Result<(), TryLinkedListWithCloneError>
    where
        T: TryClone,
    {
        if other.is_empty() {
            return Ok(());
        }
        let guard = TruncateGuard::new(self);
        for item in other {
            let cloned = item
                .try_clone()
                .map_err(TryLinkedListWithCloneError::Clone)?;
            if let Err(e) = guard.list.try_push_back(cloned) {
                return Err(TryLinkedListWithCloneError::Alloc(e));
            }
        }
        guard.forget();
        Ok(())
    }
}

// ── Generic TryExtend / TryExtendFromSlice impls ──────────────────────────────

impl<T> TryExtend<T> for LinkedList<T> {
    type Error = AllocError;

    fn try_extend<S>(&mut self, source: S) -> Result<(), (Resumable<S::Inner>, AllocError)>
    where
        S: crate::recovery::ResumableSource<Item = T>,
    {
        let (head, mut iter) = source.safe_into_iter();

        if let Some(item) = head {
            if let Err((item, e)) = self.try_push_back_mut_give_back(item) {
                return Err((Resumable::new(item, iter), e));
            }
        }

        while let Some(item) = iter.next() {
            if let Err((item, e)) = self.try_push_back_mut_give_back(item) {
                return Err((Resumable::new(item, iter), e));
            }
        }
        Ok(())
    }
}

impl<'s, T: TryClone> TryExtendFromSlice<'s, T> for LinkedList<T> {
    type Error = TryLinkedListWithCloneError;

    fn try_extend_from_slice(
        &mut self,
        other: &'s [T],
    ) -> Result<(), (&'s [T], TryLinkedListWithCloneError)> {
        for (i, item) in other.iter().enumerate() {
            let cloned = match item.try_clone() {
                Ok(c) => c,
                Err(e) => return Err((&other[i..], TryLinkedListWithCloneError::Clone(e))),
            };
            if let Err((_, e)) = self.try_push_back_mut_give_back(cloned) {
                return Err((&other[i..], TryLinkedListWithCloneError::Alloc(e)));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::try_format;
    use lang_alloc::format;
    use lang_alloc::string::String;
    use lang_alloc::vec;
    use lang_alloc::vec::Vec;

    #[test]
    fn linked_list_try_clone_success() {
        let ll: LinkedList<i32> = vec![1, 2, 3].into_iter().collect();
        let cloned = ll.try_clone().unwrap();
        assert_eq!(
            cloned.iter().copied().collect::<lang_alloc::vec::Vec<_>>(),
            [1, 2, 3]
        );
    }

    #[test]
    fn linked_list_try_clone_empty() {
        let ll: LinkedList<i32> = LinkedList::new();
        let cloned = ll.try_clone().unwrap();
        assert!(cloned.is_empty());
    }

    #[test]
    fn linked_list_try_clone_string() {
        let ll: LinkedList<String> = vec!["a", "b"].into_iter().map(String::from).collect();
        let cloned = ll.try_clone().unwrap();
        assert_eq!(
            cloned
                .iter()
                .map(|s| s.as_str())
                .collect::<lang_alloc::vec::Vec<_>>(),
            ["a", "b"]
        );
    }

    #[test]
    fn linked_list_try_default_empty() {
        let ll: LinkedList<i32> = LinkedList::try_default().unwrap();
        assert!(ll.is_empty());
    }

    #[test]
    fn linked_list_try_debug() {
        let ll: LinkedList<i32> = vec![1, 2].into_iter().collect();
        let dbg = try_format!("{:?}", ll).unwrap();
        assert_eq!(dbg, "[1, 2]");
    }

    #[test]
    fn linked_list_try_debug_empty() {
        let ll: LinkedList<i32> = LinkedList::new();
        let dbg = try_format!("{:?}", ll).unwrap();
        assert_eq!(dbg, "[]");
    }

    // ── TryLinkedList tests ──────────────────────────────────────────────────

    fn contents(l: &LinkedList<i32>) -> lang_alloc::vec::Vec<i32> {
        l.iter().copied().collect()
    }

    #[test]
    fn linked_list_try_push_back_and_front() {
        let mut l: LinkedList<i32> = LinkedList::new();
        l.try_push_back(2).unwrap();
        l.try_push_back(3).unwrap();
        l.try_push_front(1).unwrap();
        assert_eq!(contents(&l), vec![1, 2, 3]);
    }

    #[test]
    fn linked_list_try_collect_basic() {
        let l = LinkedList::try_collect(1..4).unwrap();
        assert_eq!(contents(&l), vec![1, 2, 3]);
    }

    #[test]
    fn linked_list_try_from_slice_basic() {
        let l = LinkedList::try_from_slice(&[5, 6]).unwrap();
        assert_eq!(contents(&l), vec![5, 6]);
    }

    #[test]
    fn linked_list_generic_try_extend_via_trait() {
        let mut l: LinkedList<i32> = LinkedList::new();
        <_ as TryExtend<i32>>::try_extend(&mut l, 10..13).unwrap();
        assert_eq!(contents(&l), vec![10, 11, 12]);
    }

    #[test]
    fn linked_list_generic_try_extend_from_slice_via_trait() {
        let mut l: LinkedList<Vec<u8>> = LinkedList::new();
        let slice: &[Vec<u8>] = &[vec![7]];
        <_ as TryExtendFromSlice<'_, Vec<u8>>>::try_extend_from_slice(&mut l, slice).unwrap();
        assert_eq!(l.len(), 1);
    }

    // ── try_extend_from_slice_with_rollback tests ────────────────────────────

    #[test]
    fn linked_list_try_extend_from_slice_with_rollback_success() {
        let mut l: LinkedList<i32> = vec![1, 2].into_iter().collect();
        l.try_extend_from_slice_with_rollback(&[3, 4]).unwrap();
        assert_eq!(contents(&l), vec![1, 2, 3, 4]);
    }

    #[test]
    fn linked_list_try_extend_from_slice_with_rollback_empty_source() {
        let mut l: LinkedList<i32> = vec![1].into_iter().collect();
        l.try_extend_from_slice_with_rollback(&[]).unwrap();
        assert_eq!(contents(&l), vec![1]);
    }

    #[test]
    fn linked_list_try_extend_from_slice_with_rollback_alias() {
        let mut l: LinkedList<i32> = LinkedList::new();
        l.fallible_extend_from_slice_with_rollback(&[9]).unwrap();
        assert_eq!(contents(&l), vec![9]);
    }

    /// Isolated diagnostic: does `Box::try_new(Vec<u8>)` work under Miri?
    #[test]
    fn linked_list_error_display() {
        let e = TryLinkedListWithCloneError::Clone(TryCloneError::Other("boom"));
        let s = format!("{e}");
        assert!(s.contains("linked list"));
        let d = format!("{e:?}");
        assert!(d.contains("TryLinkedListWithCloneError::Clone"));
    }

    #[test]
    fn linked_list_interleaved_ops_maintain_invariants() {
        // Exercise bidirectional linking under mixed front/back pushes.
        let mut l: LinkedList<i32> = LinkedList::new();
        l.try_push_back(1).unwrap();
        l.try_push_back(2).unwrap();
        l.try_push_front(0).unwrap();
        l.try_push_back(3).unwrap();
        l.try_push_front(-1).unwrap();
        assert_eq!(contents(&l), vec![-1, 0, 1, 2, 3]);

        // Pop from both ends to verify prev/next links are correct.
        assert_eq!(l.pop_front(), Some(-1));
        assert_eq!(l.pop_back(), Some(3));
        assert_eq!(contents(&l), vec![0, 1, 2]);
    }

    // ── OOM tests ─────────────────────────────────────────────────────────────
    // Require `std`: the failing-allocator hooks are thread-local (std-only).
    #[cfg(feature = "std")]
    mod oom {
        use super::*;
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        #[test]
        fn linked_list_try_push_fails_on_oom() {
            let mut l: LinkedList<i32> = LinkedList::new();
            let r = with_policy(FailPolicy::fail_next_alloc(), || l.try_push_back(1));
            assert!(r.is_err());
            assert!(l.is_empty());

            let r = with_policy(FailPolicy::fail_next_alloc(), || l.try_push_front(1));
            assert!(r.is_err());
            assert!(l.is_empty());
        }

        #[test]
        fn linked_list_try_push_give_back_returns_value() {
            let mut l: LinkedList<i32> = LinkedList::new();
            let (back, _err) = with_policy(FailPolicy::fail_next_alloc(), || {
                l.try_push_back_give_back(42).unwrap_err()
            });
            assert_eq!(back, 42);
            assert!(l.is_empty());

            let (back, _err) = with_policy(FailPolicy::fail_next_alloc(), || {
                l.try_push_front_give_back(7).unwrap_err()
            });
            assert_eq!(back, 7);
            assert!(l.is_empty());
        }

        #[test]
        fn linked_list_try_push_canonical_mut_give_back() {
            let mut l: LinkedList<i32> = LinkedList::new();
            l.try_push_front_mut_give_back(1).unwrap();
            l.try_push_back_mut_give_back(2).unwrap();
            assert_eq!(contents(&l), vec![1, 2]);

            // Failure path: value is given back.
            let (back, _err) = with_policy(FailPolicy::fail_next_alloc(), || {
                l.try_push_back_mut_give_back(99).unwrap_err()
            });
            assert_eq!(back, 99);
            assert_eq!(contents(&l), vec![1, 2]);
        }

        #[test]
        fn linked_list_try_collect_fails_midway_keeps_partial_list() {
            // Each element costs exactly ONE allocation now (we control the node).
            // Values 0, 1 succeed (allocs 1, 2). Value 2 → alloc 3 fails.
            let (stranded, rest) =
                with_policy(
                    FailPolicy::fail_nth_alloc(3),
                    || match LinkedList::try_collect(0..5) {
                        Ok(_) => panic!("expected failure"),
                        Err((stranded, rest, _e)) => (stranded, rest),
                    },
                );
            assert_eq!(stranded, Some(2));
            assert_eq!(rest.collect::<lang_alloc::vec::Vec<_>>(), vec![3, 4]);
        }

        #[test]
        fn linked_list_try_from_slice_fails_on_oom() {
            let r = with_policy(FailPolicy::fail_next_alloc(), || {
                LinkedList::try_from_slice(&[1, 2])
            });
            assert!(matches!(r, Err(TryLinkedListWithCloneError::Alloc(_))));
        }

        /// Failure partway through the slice: the first new element appends
        /// fine; the second element fails, so the first must be rolled back
        /// by the [`TruncateGuard`] while the pre-existing elements survive.
        /// Covers both error variants:
        ///
        /// - **Alloc**: `i32` clones are `Copy` (no allocation), so each
        ///   element costs exactly one allocation — its node. With a
        ///   stack-allocated source slice and an empty list inside the
        ///   policy window, element 30's node is alloc 1 and element 40's
        ///   node is alloc 2 → no incidental heap traffic can interfere.
        /// - **Clone**: a failing `TryClone` type keeps the test
        ///   deterministic regardless of incidental heap traffic in the
        ///   harness (e.g. lazy TLS initialization).
        #[test]
        fn linked_list_try_extend_from_slice_with_rollback_restores_pre_call_state() {
            use crate::try_clone::TryCloneError;

            // ── Alloc variant ────────────────────────────────────────────────
            let src: [i32; 2] = [30, 40];
            let mut l: LinkedList<i32> = LinkedList::new();
            let r = with_policy(FailPolicy::fail_nth_alloc(2), || {
                l.try_extend_from_slice_with_rollback(&src)
            });
            assert!(matches!(r, Err(TryLinkedListWithCloneError::Alloc(_))));
            // The appended 30 was rolled back by the guard; list is empty.
            assert!(l.is_empty());

            // ── Clone variant ────────────────────────────────────────────────
            #[derive(Clone)]
            struct Sim(u8);
            impl TryClone for Sim {
                fn try_clone(&self) -> Result<Self, TryCloneError> {
                    if self.0 == 40 {
                        Err(TryCloneError::Other("simulated OOM"))
                    } else {
                        Ok(Self(self.0))
                    }
                }
            }

            let mut l: LinkedList<Sim> = LinkedList::new();
            l.try_push_back(Sim(10)).unwrap();
            l.try_push_back(Sim(20)).unwrap();
            // Element 30 clones + appends fine. Element 40's clone fails.
            let r = l.try_extend_from_slice_with_rollback(&[Sim(30), Sim(40), Sim(50)]);
            assert!(matches!(
                r,
                Err(TryLinkedListWithCloneError::Clone(TryCloneError::Other(_)))
            ));
            // The appended 30 was rolled back; pre-existing elements intact.
            assert_eq!(
                l.iter().map(|e| e.0).collect::<lang_alloc::vec::Vec<_>>(),
                vec![10, 20]
            );

            // The list is still fully usable after a rollback.
            l.try_push_back(Sim(99)).unwrap();
            assert_eq!(
                l.iter().map(|e| e.0).collect::<lang_alloc::vec::Vec<_>>(),
                vec![10, 20, 99]
            );
        }

        #[test]
        fn linked_list_try_extend_from_slice_with_rollback_first_element_fails() {
            let mut l: LinkedList<i32> = LinkedList::new();
            l.try_push_back(1).unwrap();
            let r = with_policy(FailPolicy::fail_next_alloc(), || {
                l.try_extend_from_slice_with_rollback(&[2, 3])
            });
            assert!(matches!(r, Err(TryLinkedListWithCloneError::Alloc(_))));
            assert_eq!(contents(&l), vec![1]);
        }

        #[test]
        fn linked_list_generic_try_extend_retry_with_resumable() {
            let mut l: LinkedList<i32> = LinkedList::new();
            // Each item costs exactly 1 alloc. Items 0, 1 succeed (allocs 1, 2).
            // Item 2 → alloc 3 fails. Resumable carries head=2, remainder=[3].
            let resumable =
                with_policy(FailPolicy::fail_nth_alloc(3), || {
                    match <_ as TryExtend<i32>>::try_extend(&mut l, 0..4) {
                        Ok(()) => panic!("expected failure"),
                        Err((resumable, _)) => resumable,
                    }
                });
            assert_eq!(l.len(), 2);
            // Retry outside the policy: pass the full Resumable (head + remainder).
            <_ as TryExtend<i32>>::try_extend(&mut l, resumable).unwrap();
            assert_eq!(contents(&l), vec![0, 1, 2, 3]);
        }

        /// A panicking `try_clone` mid-extension must still trigger the
        /// [`TruncateGuard`] to roll back all elements appended during the
        /// call — unconditional rollback even on unwind.
        #[test]
        fn linked_list_try_extend_from_slice_with_rollback_panic_safe() {
            use crate::try_clone::TryCloneError;

            use lang_core::sync::atomic::{AtomicU8, Ordering};
            use lang_std::panic;

            static PANIC_AT: AtomicU8 = AtomicU8::new(40);

            #[derive(Clone)]
            struct Panicky(u8);
            impl TryClone for Panicky {
                fn try_clone(&self) -> Result<Self, TryCloneError> {
                    if self.0 == PANIC_AT.load(Ordering::Relaxed) {
                        panic!("simulated clone panic");
                    }
                    Ok(Self(self.0))
                }
            }

            let mut l: LinkedList<Panicky> = LinkedList::new();
            l.try_push_back(Panicky(10)).unwrap();
            l.try_push_back(Panicky(20)).unwrap();

            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                l.try_extend_from_slice_with_rollback(&[Panicky(30), Panicky(40)])
            }));
            assert!(result.is_err(), "expected a panic from element 40");
            // The guard popped the appended 30 during unwinding; only the
            // pre-existing elements remain.
            assert_eq!(
                l.iter().map(|e| e.0).collect::<lang_alloc::vec::Vec<_>>(),
                vec![10, 20]
            );
        }
    }
}
