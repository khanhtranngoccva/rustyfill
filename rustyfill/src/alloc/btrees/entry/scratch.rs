//! Thread-local scratch buffers for the B-tree entry insert path.
//!
//! Both the probe phase (`Vec<CommitInternal<K,V>>`) and the reserve phase
//! (`Vec<Box<InternalNode<K,V>>>`) allocate small heap vectors on every split.
//! On hot insert loops these are pure overhead. We cache the *backing
//! allocations* in thread-locals using type-erased equivalents:
//!
//! - Reserve: `Vec<Box<()>>` — same size/align as `Vec<Box<any_T>>`
//!   (all `Box<T>` are one pointer word regardless of T).
//! - Probe: `Vec<ErasedCommitInternal>` — `#[repr(C)]` struct matching
//!   `CommitInternal<K,V>` layout.
//!
//! Soundness rests on two pillars:
//!   1. `CommitInternal<K,V>` derives `Copy` + `#[repr(C)]` — no drop glue,
//!      pinned field layout.
//!   2. The transmute uses `Vec::from_raw_parts` with len=0 and matching
//!      capacity — no length arithmetic, no byte-size calculations. At the
//!      moment of transmute, no elements are logically alive.

use super::{CommitInternal, alloc_error};
use crate::alloc::AllocError;
use crate::alloc::vec::TryVec;
use lang_alloc::boxed::Box;
use lang_core::mem;
use lang_core::ptr::NonNull;
use lang_std::vec::Vec;

// ── Type-erased element types ─────────────────────────────────────────────────

/// Type-erased version of [`CommitInternal`] for thread-local caching.
/// Must maintain identical `#[repr(C)]` layout to `CommitInternal<K,V>`.
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct ErasedCommitInternal {
    pub(super) ptr: NonNull<()>,
    pub(super) child_idx: usize,
    pub(super) will_split: bool,
    pub(super) sp_idx: Option<usize>,
}

/// Compile-time assertions that `ErasedCommitInternal` matches `CommitInternal<K,V>`.
const _: () = {
    assert!(
        core::mem::size_of::<ErasedCommitInternal>()
            == core::mem::size_of::<CommitInternal<(), ()>>(),
        "ErasedCommitInternal size mismatch"
    );
    assert!(
        core::mem::align_of::<ErasedCommitInternal>()
            == core::mem::align_of::<CommitInternal<(), ()>>(),
        "ErasedCommitInternal align mismatch"
    );
};

// ── Raw buffer conversions (len=0, capacity-preserving) ───────────────────────

/// Transmute a cleared `Vec<CommitInternal<K,V>>` into `Vec<ErasedCommitInternal>`.
/// SAFETY: `#[repr(C)]` guarantees field order and packing; `NonNull<T>` is
/// always one pointer word regardless of T. Elements are `Copy` so no drop glue.
#[inline]
unsafe fn commit_internal_vec_to_erased<K, V>(
    v: Vec<CommitInternal<K, V>>,
) -> Vec<ErasedCommitInternal> {
    debug_assert_eq!(v.len(), 0);
    let cap = v.capacity();
    let ptr = v.as_ptr() as *mut ErasedCommitInternal;
    let result = unsafe { Vec::from_raw_parts(ptr, 0, cap) };
    mem::forget(v);
    result
}

/// Restore a `Vec<CommitInternal<K,V>>` from a cached `Vec<ErasedCommitInternal>`.
/// SAFETY: Same repr(C) + NonNull uniformity + Copy argument.
#[inline]
unsafe fn erased_to_commit_internal_vec<K, V>(
    cached: Vec<ErasedCommitInternal>,
) -> Vec<CommitInternal<K, V>> {
    let cap = cached.capacity();
    let ptr = cached.as_ptr() as *mut CommitInternal<K, V>;
    let result = unsafe { Vec::from_raw_parts(ptr, 0, cap) };
    mem::forget(cached);
    result
}

/// Transmute a cleared `Vec<Box<T>>` into `Vec<Box<()>>` for caching.
/// SAFETY: `T: Sized` ensures `Box<T>` is a thin pointer (one word), matching
/// `Box<()>` and the vector is empty.
#[inline]
#[allow(clippy::vec_box)]
unsafe fn box_vec_to_erased<T: lang_core::marker::Sized>(v: Vec<Box<T>>) -> Vec<Box<()>> {
    debug_assert_eq!(v.len(), 0);
    let cap = v.capacity();
    let ptr = v.as_ptr() as *mut Box<()>;
    let result = unsafe { Vec::from_raw_parts(ptr, 0, cap) };
    mem::forget(v);
    result
}

/// Restore an empty `Vec<Box<T>>` from a cached `Vec<Box<()>>`.
/// SAFETY: `T: Sized` ensures `Box<T>` is a thin pointer (one word), matching
/// `Box<()>`.
#[inline]
#[allow(clippy::vec_box)]
unsafe fn erased_to_box_vec<T: lang_core::marker::Sized>(cached: Vec<Box<()>>) -> Vec<Box<T>> {
    let cap = cached.capacity();
    let ptr = cached.as_ptr() as *mut Box<T>;
    let result = unsafe { Vec::from_raw_parts(ptr, 0, cap) };
    mem::forget(cached);
    result
}

// ── Thread-local storage ───────────────────────────────────────────────────────

#[cfg(feature = "std")]
mod storage {
    use super::ErasedCommitInternal;
    use lang_alloc::boxed::Box;
    use lang_core::cell::Cell;
    use lang_std::thread_local;
    use lang_std::vec::Vec;

    thread_local! {
        #[allow(clippy::vec_box)]
        pub(super) static RESERVED_BUF: Cell<Option<Vec<Box<()>>>> = const { Cell::new(None) };
    }

    thread_local! {
        pub(super) static PROBE_BUF: Cell<Option<Vec<ErasedCommitInternal>>> = const { Cell::new(None) };
    }
}

#[cfg(not(feature = "std"))]
mod storage {}

// ── Newtype: CachedProbeBuffer ─────────────────────────────────────────────────
//
// Wraps the probe-phase `Vec<CommitInternal<K,V>>` inside `CommitPlan`. On drop,
// it clears the vec and returns the backing allocation to the thread-local cache.
// Because `CommitInternal` is `Copy`, clearing is a no-op at the machine level.

pub(super) struct CachedProbeBuffer<K, V> {
    inner: Vec<CommitInternal<K, V>>,
}

impl<K, V> CachedProbeBuffer<K, V> {
    /// Create a new probe buffer, recycling from the thread-local if possible.
    pub(super) fn try_new(capacity: usize) -> Result<Self, AllocError> {
        #[cfg(feature = "std")]
        {
            if let Some(cached) = storage::PROBE_BUF.with(|b| b.replace(None))
                && cached.capacity() >= capacity
            {
                let typed = unsafe { erased_to_commit_internal_vec::<K, V>(cached) };
                return Ok(Self { inner: typed });
            }
            // Capacity insufficient; fall through to fresh allocation.
        }
        let inner = <Vec<CommitInternal<K, V>> as TryVec<_>>::try_with_capacity(capacity)
            .map_err(|_| alloc_error())?;
        Ok(Self { inner })
    }

    /// Get a mutable reference to the underlying vec for filling.
    pub(super) fn as_mut(&mut self) -> &mut Vec<CommitInternal<K, V>> {
        &mut self.inner
    }

    /// Immutable iterator over the contained elements.
    pub(super) fn iter(&self) -> impl Iterator<Item = &CommitInternal<K, V>> {
        self.inner.iter()
    }
}

impl<K, V> Drop for CachedProbeBuffer<K, V> {
    fn drop(&mut self) {
        self.inner.clear();
        #[cfg(feature = "std")]
        {
            let drained = mem::take(&mut self.inner);
            let erased = unsafe { commit_internal_vec_to_erased::<K, V>(drained) };
            storage::PROBE_BUF.with(|b| b.set(Some(erased)));
        }
    }
}

// ── Newtype: CachedReserveBuffer ───────────────────────────────────────────────
//
// Owns the reserved internal-node boxes during the reserve phase.
//
// - **Rollback** (allocation failure mid-reserve): dropping the buffer drops
//   each remaining `Box<InternalNode<K,V>>`, freeing the node automatically.
// - **Commit success**: the caller calls `drain_to_pointers()` which converts
//   each `Box` to a raw pointer via `Box::into_raw` (leaking it into the tree).
//   The emptied vec's backing allocation is recycled on drop.

pub(super) struct CachedReserveBuffer<K, V> {
    inner: Vec<Box<super::sys::InternalNode<K, V>>>,
}

impl<K, V> CachedReserveBuffer<K, V> {
    /// Create a new reserve buffer, recycling from the thread-local if possible.
    pub(super) fn try_new(capacity: usize) -> Result<Self, AllocError> {
        #[cfg(feature = "std")]
        {
            if let Some(cached) = storage::RESERVED_BUF.with(|b| b.replace(None))
                && cached.capacity() >= capacity
            {
                let typed = unsafe { erased_to_box_vec::<super::sys::InternalNode<K, V>>(cached) };
                return Ok(Self { inner: typed });
            }
            // Capacity insufficient; fall through to fresh allocation.
        }
        let inner =
            <Vec<Box<super::sys::InternalNode<K, V>>> as TryVec<_>>::try_with_capacity(capacity)
                .map_err(|_| alloc_error())?;
        Ok(Self { inner })
    }

    /// Push a freshly allocated internal node box into the buffer.
    pub(super) fn push(&mut self, node: Box<super::sys::InternalNode<K, V>>) {
        self.inner.push(node);
    }

    /// Drain all boxes, converting each to a raw `NonNull<LeafNode>` pointer
    /// (matching the sys `BoxedNode` alias convention).
    ///
    /// After this call the buffer is empty (len=0). The returned pointers are
    /// owned by the caller — the commit wires them into the tree. The buffer's
    /// backing allocation is recycled when the buffer is dropped.
    pub(super) fn drain_to_pointers(&mut self) -> Vec<NonNull<super::sys::LeafNode<K, V>>> {
        let mut ptrs = Vec::with_capacity(self.inner.len());
        for box_node in self.inner.drain(..) {
            let raw: *mut super::sys::InternalNode<K, V> = Box::into_raw(box_node);
            // Cast to LeafNode pointer (InternalNode's first field is LeafNode,
            // so the addresses are identical). Matches the sys BoxedNode alias.
            ptrs.push(unsafe { NonNull::new_unchecked(raw as *mut super::sys::LeafNode<K, V>) });
        }
        ptrs
    }
}

impl<K, V> Drop for CachedReserveBuffer<K, V> {
    fn drop(&mut self) {
        // Any remaining boxes are dropped here (rollback path), freeing their
        // nodes. On the success path, drain_to_pointers already emptied the vec.
        // Recycle the backing Vec allocation to the thread-local cache.
        self.inner.clear();
        #[cfg(feature = "std")]
        {
            let drained = mem::take(&mut self.inner);
            let erased = unsafe { box_vec_to_erased::<super::sys::InternalNode<K, V>>(drained) };
            storage::RESERVED_BUF.with(|b| b.set(Some(erased)));
        }
    }
}

// ── Newtype: PendingLeaf ────────────────────────────────────────────────────────
//
// Owns the freshly allocated right-half leaf node as a `Box<LeafNode<K,V>>`.
// If the insert succeeds, the commit calls `into_raw()` to transfer ownership
// to the tree. If anything fails before that, dropping the `PendingLeaf` frees
// the node automatically (the Box destructor handles it).

pub(super) struct PendingLeaf<K, V> {
    box_: Option<Box<super::sys::LeafNode<K, V>>>,
}

impl<K, V> PendingLeaf<K, V> {
    /// Wrap a freshly allocated leaf box.
    pub(super) fn new(box_: Box<super::sys::LeafNode<K, V>>) -> Self {
        Self { box_: Some(box_) }
    }

    /// Transfer ownership to the tree. Returns the raw pointer.
    /// Consumes self; after this call the node is owned by the tree.
    pub(super) fn into_raw(mut self) -> NonNull<super::sys::LeafNode<K, V>> {
        let b = self.box_.take().expect("leaf already taken");
        let raw = Box::into_raw(b);
        // Forget self so Drop doesn't run on the now-empty Option (harmless,
        // but avoids any future confusion if the struct gains fields).
        mem::forget(self);
        unsafe { NonNull::new_unchecked(raw) }
    }
}
