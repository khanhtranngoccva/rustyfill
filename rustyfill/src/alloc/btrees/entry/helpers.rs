//! Low-level helpers for B-tree entry operations.
//!
//! These operate on raw pointers and mirrored [`rustyfill_sys`] types and are
//! used by the reserve-and-commit architecture in [`super`](crate::alloc::btrees::entry).
//! They are deliberately split into two families:
//!
//! * **Allocation / cleanup** — fallible node allocation ([`try_new_leaf`],
//!   [`try_new_internal`]) and manual teardown of nodes that were reserved but
//!   never wired into the tree ([`drop_leaf_node`], [`drop_internal_node`]).
//! * **Infallible mutation** — pure pointer surgery that moves already-live data
//!   between nodes ([`leaf_slice_insert`], [`internal_insert_fit`],
//!   [`copy_right_half`], parent-link maintenance). None of these allocate, so
//!   they can never fail; this is what makes the commit phase infallible.
//!
//! # Safety
//!
//! Nearly every function here is `unsafe`: it dereferences raw pointers into
//! live B-tree nodes. Callers must uphold the B-tree invariants described in
//! [`super`](crate::alloc::btrees::entry) and guarantee that any pointer passed
//! in refers to a valid, suitably-aligned node owned by the map being mutated.

use crate::alloc::boxed::TryBox;
use lang_alloc::boxed::Box;
use lang_core::marker::PhantomData;
use lang_core::{mem::MaybeUninit, ptr, ptr::NonNull};

mod sys {
    pub use rustyfill_sys::std::collections::btree::node::marker::*;
    pub use rustyfill_sys::std::collections::btree::node::{
        CAPACITY, EDGE_IDX_LEFT_OF_CENTER, EDGE_IDX_RIGHT_OF_CENTER, Handle, InternalNode,
        KV_IDX_CENTER, LeafNode, NodeRef,
    };
}

// ── Split geometry ────────────────────────────────────────────────────────────

/// Which side of a full node an incoming key lands on once the node splits.
#[derive(Clone, Copy, Debug)]
pub(crate) enum InsertionSide {
    /// The new key/value goes into the left half at slot `idx`.
    Left(usize),
    /// The new key/value goes into the freshly allocated right half at slot `idx`.
    Right(usize),
}

/// Given the edge index where an insertion descends into a *full* node, compute
/// the centre key/index that gets promoted, and which half (left or the new
/// right node) receives the new element and at what local index.
///
/// Mirrors std's `splitpoint`. `edge_idx` ranges over `0..=CAPACITY`.
pub(crate) fn splitpoint(edge_idx: usize) -> (usize, InsertionSide) {
    debug_assert!(edge_idx <= sys::CAPACITY);
    match edge_idx {
        0..sys::EDGE_IDX_LEFT_OF_CENTER => (
            sys::KV_IDX_CENTER - 1,
            InsertionSide::Left(edge_idx),
        ),
        sys::EDGE_IDX_LEFT_OF_CENTER => (sys::KV_IDX_CENTER, InsertionSide::Left(edge_idx)),
        sys::EDGE_IDX_RIGHT_OF_CENTER => (sys::KV_IDX_CENTER, InsertionSide::Right(0)),
        _ => (
            sys::KV_IDX_CENTER + 1,
            InsertionSide::Right(edge_idx - (sys::KV_IDX_CENTER + 1 + 1)),
        ),
    }
}

// ── Fallible node allocation ──────────────────────────────────────────────────

/// Allocate a fresh, empty leaf node on the heap.
///
/// Returns the raw pointer as a `NonNull`. On success the node has `parent ==
/// None`, `len == 0`, and uninitialised key/value slots. The caller takes full
/// ownership: either wire it into the tree or release it via [`drop_leaf_node`].
pub(crate) fn try_new_leaf<K, V>() -> Result<NonNull<sys::LeafNode<K, V>>, AllocError_> {
    let mut leaf_box: Box<MaybeUninit<sys::LeafNode<K, V>>> =
        <Box<sys::LeafNode<K, V>> as TryBox<_>>::fallible_new_uninit()?;

    unsafe {
        let ptr = leaf_box.as_mut_ptr();
        (*ptr).parent = None;
        (*ptr).len = 0;
    }

    let boxed: Box<sys::LeafNode<K, V>> = unsafe { leaf_box.assume_init() };
    let raw = Box::into_raw(boxed);
    Ok(unsafe { NonNull::new_unchecked(raw) })
}

/// Allocate a fresh, empty internal node on the heap.
///
/// As with [`try_new_leaf`], the returned pointer carries full ownership. Note
/// the pointer is typed as `NonNull<LeafNode>` (matching the sys `BoxedNode`
/// alias) and must be cast back to `InternalNode` before touching `data`/`edges`.
pub(crate) fn try_new_internal<K, V>() -> Result<NonNull<sys::LeafNode<K, V>>, AllocError_> {
    let mut node_box: Box<MaybeUninit<sys::InternalNode<K, V>>> =
        <Box<sys::InternalNode<K, V>> as TryBox<_>>::fallible_new_uninit()?;

    unsafe {
        let ptr = node_box.as_mut_ptr();
        (*ptr).data.parent = None;
        (*ptr).data.len = 0;
    }

    let boxed: Box<sys::InternalNode<K, V>> = unsafe { node_box.assume_init() };
    let raw = Box::into_raw(boxed);
    Ok(unsafe { NonNull::new_unchecked(raw).cast() })
}

/// Re-export of the allocation error surfaced by the fallible allocators above.
pub(crate) type AllocError_ = crate::alloc::AllocError;

// ── Manual node teardown (rollback path) ──────────────────────────────────────

/// Drop a heap-allocated leaf node that was reserved but never linked into the
/// tree, dropping any initialised keys/values first. Used during rollback when a
/// later reservation fails after an earlier one succeeded.
pub(crate) unsafe fn drop_leaf_node<K, V>(ptr: NonNull<sys::LeafNode<K, V>>) {
    unsafe {
        let leaf = ptr.as_ptr();
        let len = (*leaf).len as usize;
        for i in 0..len {
            (*leaf).keys[i].assume_init_drop();
            (*leaf).vals[i].assume_init_drop();
        }
        let _ = Box::from_raw(leaf);
    }
}

/// Drop a heap-allocated internal node that was reserved but never linked into
/// the tree, dropping any initialised keys/values first. Child edges are not
/// dropped: by construction a rolled-back internal node only ever references
/// nodes that remain reachable from the (unchanged) original tree, so freeing
/// them here would double-free.
pub(crate) unsafe fn drop_internal_node<K, V>(ptr: NonNull<sys::LeafNode<K, V>>) {
    unsafe {
        let internal = ptr.as_ptr() as *mut sys::InternalNode<K, V>;
        let len = (*internal).data.len as usize;
        for i in 0..len {
            (*internal).data.keys[i].assume_init_drop();
            (*internal).data.vals[i].assume_init_drop();
        }
        let _ = Box::from_raw(internal);
    }
}

// ── Infallible leaf mutation ──────────────────────────────────────────────────

/// Shift elements right and insert `(key, val)` at `idx` within a leaf's arrays.
///
/// Does **not** update `len`; the caller updates it. The caller must ensure the
/// resulting length stays within `CAPACITY`.
pub(crate) unsafe fn leaf_slice_insert<K, V>(
    leaf: *mut sys::LeafNode<K, V>,
    idx: usize,
    key: K,
    val: V,
) {
    unsafe {
        let len = (*leaf).len as usize;
        if len > idx {
            // Shift-right within the same array. Derive BOTH src and dst from a
            // single base pointer so that under stacked borrows neither read nor
            // write retags against the other (a separate as_ptr()/as_mut_ptr()
            // pair would create a SharedReadOnly then a Unique tag whose overlap
            // invalidates the former → UB on the copy's read of src).
            let keys_base = (*leaf).keys.as_mut_ptr();
            ptr::copy(keys_base.add(idx), keys_base.add(idx + 1), len - idx);
            let vals_base = (*leaf).vals.as_mut_ptr();
            ptr::copy(vals_base.add(idx), vals_base.add(idx + 1), len - idx);
        }
        (*leaf).keys[idx].write(key);
        (*leaf).vals[idx].write(val);
    }
}

/// Move the upper portion of a full leaf (`source`) into a fresh, empty leaf
/// (`right`).
///
/// After the call, `source` holds exactly `sp_idx` entries (slots `0..sp_idx`)
/// and `right` holds `old_len - sp_idx - 1` entries (the former slots
/// `sp_idx+1..old_len`), with `right.len` set accordingly. The promoted centre
/// entry at `sp_idx` is intentionally left untouched in `source`; the caller
/// reads it out separately.
pub(crate) unsafe fn copy_right_half_leaf<K, V>(
    source: *mut sys::LeafNode<K, V>,
    right: *mut sys::LeafNode<K, V>,
    sp_idx: usize,
) {
    unsafe {
        let old_len = (*source).len as usize;
        let new_right_len = old_len - sp_idx - 1;

        (*right).len = new_right_len as u16;
        if new_right_len > 0 {
            ptr::copy_nonoverlapping(
                (*source).keys.as_ptr().add(sp_idx + 1),
                (*right).keys.as_mut_ptr(),
                new_right_len,
            );
            ptr::copy_nonoverlapping(
                (*source).vals.as_ptr().add(sp_idx + 1),
                (*right).vals.as_mut_ptr(),
                new_right_len,
            );
        }
        (*source).len = sp_idx as u16;
    }
}

// ── Infallible internal-node mutation ─────────────────────────────────────────

/// Insert a separator `(key, val)` and a child edge into an internal node at
/// `edge_idx`, shifting subsequent separators and edges up by one.
///
/// Assumes the node is **not** full (so there is room for one more separator).
/// Updates `len` itself.
pub(crate) unsafe fn internal_insert_fit<K, V>(
    internal: *mut sys::InternalNode<K, V>,
    edge_idx: usize,
    key: K,
    val: V,
    child: NonNull<sys::LeafNode<K, V>>,
) {
    unsafe {
        // Mirrors std's internal-node `insert_fit`, which calls `slice_insert`
        // on three views of different lengths:
        //   * keys/vals over a slice of length `new_len = len + 1` → shifts
        //     `(len + 1) - idx - 1 = len - idx` initialised elements;
        //   * edges over a slice of length `new_len + 1 = len + 2` (one more
        //     than the key view, because an internal node with `n` separators
        //     has `n + 1` edges) → also shifts `len - idx` elements.
        // Both shifts therefore move exactly `len - edge_idx` elements. The
        // arrays are oversized (keys/vals hold `CAPACITY`, edges hold `2*B`),
        // so when `edge_idx == len` the "shift" copies from the one trailing
        // uninitialised slot into another uninitialised slot — harmless, since
        // nothing ever reads that slot afterwards.
        //
        // Operate through raw pointers only: taking `&mut` sub-slices here would
        // create fresh borrow tags that stack against any caller-held reference
        // to this node (stacked-borrows UB, caught by Miri).
        let len = (*internal).data.len as usize;
        let shift = len - edge_idx;
        if shift > 0 {
            let keys_base = (*internal).data.keys.as_mut_ptr();
            ptr::copy(keys_base.add(edge_idx), keys_base.add(edge_idx + 1), shift);
            let vals_base = (*internal).data.vals.as_mut_ptr();
            ptr::copy(vals_base.add(edge_idx), vals_base.add(edge_idx + 1), shift);
            let edges_base = (*internal).edges.as_mut_ptr();
            ptr::copy(edges_base.add(edge_idx + 1), edges_base.add(edge_idx + 2), shift);
        }
        (*internal).data.keys[edge_idx].write(key);
        (*internal).data.vals[edge_idx].write(val);
        (*internal).edges[edge_idx + 1].write(child);
        (*internal).data.len = (len + 1) as u16;
    }
}

/// Move the upper portion of a full internal node (`source`) into a fresh,
/// empty internal node (`right`), covering both the separator arrays and the
/// child-edge array.
///
/// After the call, `source` holds `sp_idx` separators and `sp_idx + 1` edges;
/// `right` holds `old_len - sp_idx - 1` separators and `old_len - sp_idx` edges
/// (former edges `sp_idx+1..=old_len`), with `right.data.len` set accordingly.
/// The promoted centre separator at `sp_idx` is left untouched in `source`.
///
/// NOTE: the moved edges still point at their original children whose parent
/// links now reference `source`; the caller must repair those links via
/// [`correct_parent_links`].
pub(crate) unsafe fn copy_right_half_internal<K, V>(
    source: *mut sys::InternalNode<K, V>,
    right: *mut sys::InternalNode<K, V>,
    sp_idx: usize,
) {
    unsafe {
        let old_len = (*source).data.len as usize;
        let new_right_len = old_len - sp_idx - 1;

        (*right).data.len = new_right_len as u16;
        if new_right_len > 0 {
            // Distinct allocations → non-overlapping. Raw pointers avoid any
            // borrow retag on source that stacked borrows would invalidate.
            let src_keys = (*source).data.keys.as_ptr().add(sp_idx + 1);
            let dst_keys = (*right).data.keys.as_mut_ptr();
            ptr::copy_nonoverlapping(src_keys, dst_keys, new_right_len);
            let src_vals = (*source).data.vals.as_ptr().add(sp_idx + 1);
            let dst_vals = (*right).data.vals.as_mut_ptr();
            ptr::copy_nonoverlapping(src_vals, dst_vals, new_right_len);
        }
        // Edges: the right node always inherits at least one edge — former
        // edge `old_len` (the last child of `source`) becomes its sole edge
        // even when `new_right_len == 0`. Without this the freshly allocated
        // node would be wired into the tree with an uninitialised edge slot,
        // leaving a dangling/unset child pointer that panics on drop.
        ptr::copy_nonoverlapping(
            (*source).edges.as_ptr().add(sp_idx + 1),
            (*right).edges.as_mut_ptr(),
            new_right_len + 1,
        );
        (*source).data.len = sp_idx as u16;
    }
}

// ── Parent-link maintenance ───────────────────────────────────────────────────

/// Point `child`'s stored parent link at `parent`, recording its position as
/// `idx` among the parent's edges.
pub(crate) unsafe fn set_parent_link<K, V>(
    child_ptr: *mut sys::LeafNode<K, V>,
    parent_ptr: NonNull<sys::LeafNode<K, V>>,
    idx: usize,
) {
    unsafe {
        (*child_ptr).parent = Some(parent_ptr.cast());
        (*child_ptr).parent_idx.write(idx as u16);
    }
}

/// Rewrite the parent links of the children referenced by `internal`'s edges
/// `start..=end` so they point back at `internal`. Used after moving a run of
/// edges into a freshly allocated node.
pub(crate) unsafe fn correct_parent_links<K, V>(
    internal: NonNull<sys::LeafNode<K, V>>,
    start: usize,
    end: usize,
) {
    unsafe {
        // Read edges through a raw pointer (no as_mut retag) so we don't create
        // a second Unique borrow on this node that would invalidate any
        // caller-held reference to it under stacked borrows.
        let iptr: *mut sys::InternalNode<K, V> = internal.cast().as_ptr();
        for i in start..=end {
            let child = (*iptr).edges[i].assume_init_read();
            set_parent_link(child.as_ptr(), internal, i);
        }
    }
}

// ── Ascending the tree ────────────────────────────────────────────────────────

/// How far a promotion climbs before it stops needing to recurse upward.
pub(crate) enum AscendResult<'a, K, V> {
    /// There is a parent internal node; `idx` is the edge position of the node
    /// we ascended from.
    Parent(sys::Handle<sys::NodeRef<sys::Mut<'a>, K, V, sys::Internal>, sys::Edge>),
    /// We reached the root; the whole tree grows by one level.
    Root(sys::NodeRef<sys::Mut<'a>, K, V, sys::LeafOrInternal>),
}

/// Follow a node's stored parent link to its parent, or report that it is the
/// root. Pure read of the (already-valid) parent link — no allocation.
pub(crate) fn ascend<'a, K, V>(
    node: sys::NodeRef<sys::Mut<'a>, K, V, sys::LeafOrInternal>,
) -> AscendResult<'a, K, V> {
    unsafe {
        let leaf = node.node.as_ptr();
        match (*leaf).parent {
            Some(parent_ptr) => {
                let parent_idx = usize::from((*leaf).parent_idx.assume_init());
                let parent_node_ref = sys::NodeRef::<sys::Mut<'a>, K, V, sys::Internal> {
                    height: node.height + 1,
                    node: parent_ptr.cast(),
                    _marker: PhantomData,
                };
                AscendResult::Parent(sys::Handle {
                    node: parent_node_ref,
                    idx: parent_idx,
                    _marker: PhantomData,
                })
            }
            None => AscendResult::Root(node),
        }
    }
}

// ── Type coercion helper ──────────────────────────────────────────────────────

pub(crate) trait ForgetTypeLoI<'a, K, V> {
    fn forget_type(self) -> sys::NodeRef<sys::Mut<'a>, K, V, sys::LeafOrInternal>;
}

impl<'a, K, V> ForgetTypeLoI<'a, K, V> for sys::NodeRef<sys::Mut<'a>, K, V, sys::Leaf> {
    fn forget_type(self) -> sys::NodeRef<sys::Mut<'a>, K, V, sys::LeafOrInternal> {
        sys::NodeRef {
            height: self.height,
            node: self.node,
            _marker: PhantomData,
        }
    }
}

impl<'a, K, V> ForgetTypeLoI<'a, K, V> for sys::NodeRef<sys::Mut<'a>, K, V, sys::Internal> {
    fn forget_type(self) -> sys::NodeRef<sys::Mut<'a>, K, V, sys::LeafOrInternal> {
        sys::NodeRef {
            height: self.height,
            node: self.node,
            _marker: PhantomData,
        }
    }
}

impl<K, V> ForgetTypeLoI<'_, K, V> for sys::NodeRef<sys::Owned, K, V, sys::Leaf> {
    fn forget_type(self) -> sys::NodeRef<sys::Mut<'static>, K, V, sys::LeafOrInternal> {
        sys::NodeRef {
            height: self.height,
            node: self.node,
            _marker: PhantomData,
        }
    }
}
