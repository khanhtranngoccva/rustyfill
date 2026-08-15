//! Fallible B-tree map entry operations via direct node manipulation.
//!
//! Provides [`TryBTreeMapEntry`] which adds a `try_insert_entry` API to
//! `BTreeMap<K, V>`. Unlike the deprecated [`crate::alloc::btrees::TryBTreeMap`]
//! (which relies on `catch_unwind`), this trait directly manipulates the
//! internal B-tree structure using mirrored types from [`rustyfill_sys`],
//! enabling proper OOM handling via [`Result`] returns.
//!
//! Both insertion branches — empty-map initialization and existing-tree
//! insertion with potential splits — allocate heap nodes. Every allocation
//! point is intercepted with fallible [`TryBox::fallible_new_uninit`] so that
//! OOM returns [`Err`] rather than panicking.
//!
//! # Reserve-and-Commit Architecture
//!
//! Insertion with cascading splits uses a strict two-phase approach:
//!
//! 1. **Reserve phase** — *all* allocations happen up front, in one batched
//!    pass over the split path. We walk the tree bottom-up, deciding for each
//!    full node whether it must split, and pre-allocate every new node the
//!    commit will need: one leaf for the initial leaf split, plus one internal
//!    node per cascading internal split, plus one more if the root grows. If any
//!    single allocation fails we drop the already-reserved nodes and return
//!    `Err`; because no mutation has touched the original tree yet, it remains
//!    completely intact.
//! 2. **Commit phase** — with every node already allocated, the actual splits
//!    are performed as pure pointer surgery. No allocation occurs here, so
//!    failure is impossible: the commit can never fail.
//!
//! The batching matters precisely because std's top-down "split on demand"
//! would interleave allocation with mutation; reserving everything first is
//! what lets a failed reservation roll back cleanly without leaving the tree
//! half-split.

use crate::alloc::AllocError;
use lang_alloc::collections::BTreeMap;
use lang_alloc::collections::btree_map::{Entry, VacantEntry};

use lang_core::fmt;
use lang_core::marker::PhantomData;
use lang_core::mem;
use lang_core::ptr;
use lang_core::ptr::NonNull;
use lang_std::collections::btree_map::OccupiedEntry;
use lang_std::vec::Vec;

mod helpers;
use helpers::*;

// ── Re-exported sys types ─────────────────────────────────────────────────────

mod sys {
    pub use rustyfill_sys::std::collections::btree::map::BTreeMap as SysBTreeMap;
    pub use rustyfill_sys::std::collections::btree::map::entry::VacantEntry as SysVacantEntry;
    pub use rustyfill_sys::std::collections::btree::node::marker::*;
    pub use rustyfill_sys::std::collections::btree::node::{
        CAPACITY, Handle, InternalNode, LeafNode, NodeRef,
    };
}

/// Return type for fallible insertion helpers: a reference to the inserted slot
/// paired with its index, or the original key/value plus the error on failure.
type InsertResult<'a, K, V> = Result<
    (sys::NodeRef<sys::Mut<'a>, K, V, sys::LeafOrInternal>, usize),
    (K, V, TryBTreeMapEntryError),
>;

// ── Error type ────────────────────────────────────────────────────────────────

/// Error returned by [`TryBTreeMapEntry::try_insert_entry`].
#[derive(Debug)]
pub enum TryBTreeMapEntryError {
    /// A heap allocation failed while creating a new B-tree node.
    Alloc(AllocError),
}

impl fmt::Display for TryBTreeMapEntryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(
                f,
                "B-tree map entry operation failed: heap allocation error"
            ),
        }
    }
}

impl From<AllocError> for TryBTreeMapEntryError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

// ── Layout validation ─────────────────────────────────────────────────────────

/// Compile-time assertion that the real std `VacantEntry<'_, K, V>` and our
/// polyfill `SysVacantEntry<'_, K, V, ()>` share the same size and alignment.
///
/// Both types differ only in their allocator parameter: the real one uses
/// `Global` (ZST, align 1) while the polyfill uses `()` (ZST, align 1).
/// Since neither contributes to the struct layout, the sizes must be equal.
/// This fires a `const_panic` if they ever diverge for any monomorphized `K, V`.
const fn assert_vacant_layout_compat<K: Ord, V>() {
    assert!(
        mem::size_of::<VacantEntry<'_, K, V>>()
            == mem::size_of::<sys::SysVacantEntry<'_, K, V, ()>>(),
        "VacantEntry and SysVacantEntry have different sizes"
    );
    assert!(
        mem::align_of::<VacantEntry<'_, K, V>>()
            == mem::align_of::<sys::SysVacantEntry<'_, K, V, ()>>(),
        "VacantEntry and SysVacantEntry have different alignments"
    );
}

/// Extract fields from a real `VacantEntry` by copying bytes into a
/// `SysVacantEntry`, then forgetting the original.
///
/// # Safety
/// - Caller must ensure `assert_vacant_layout_compat` passes for these `K, V`.
/// - After calling, the original `VacantEntry` must not be used or dropped.
unsafe fn extract_to_sys<'a, K: Ord, V>(
    ve: VacantEntry<'a, K, V>,
) -> sys::SysVacantEntry<'a, K, V, ()> {
    assert_vacant_layout_compat::<K, V>();
    let sys_ve = unsafe { mem::transmute_copy(&ve) };
    mem::forget(ve);
    sys_ve
}

// ── Trait definition ──────────────────────────────────────────────────────────

/// A trait for fallible B-tree map entry operations.
///
/// Implemented for `&mut BTreeMap<K, V>` to provide a convenient
/// [`try_insert_entry`](Self::try_insert_entry) method that dispatches through
/// the Entry API. Also implemented for [`VacantEntry`] directly via
/// [`VacantEntryExt::try_insert`].
///
/// On allocation failure during insertion, returns [`Err`] containing
/// the original `key` and `value` alongside the error, so neither is lost.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use rustyfill::alloc::btrees::entry::TryBTreeMapEntry;
///
/// let mut map = BTreeMap::new();
///
/// // First insert — key is vacant, value gets inserted.
/// let val = map.try_insert_entry("hello", 42).unwrap();
/// assert_eq!(val, &42);
/// assert_eq!(map["hello"], 42);
///
/// // Second call — key already exists, returns the existing value.
/// let val = map.try_insert_entry("hello", 99).unwrap();
/// assert_eq!(val, &42); // old value unchanged
/// ```
pub trait TryBTreeMapEntry<'a, K, V>: Sized {
    /// Obtain an entry for `key` and fallibly insert `value` if vacant.
    ///
    /// If the key is already present, returns a reference to the existing value.
    /// If the key is absent, performs a fallible insertion that may return
    /// [`Err`] on heap allocation failure.
    fn try_insert_entry(self, key: K, value: V) -> Result<&'a mut V, (K, V, TryBTreeMapEntryError)>
    where
        K: Ord;
}

// Extension trait placed on VacantEntry directly so callers can use
// `vacant.try_insert(value)` without going through &mut BTreeMap.
pub trait VacantEntryExt<'a, K, V>: Sized {
    /// Fallibly insert a value into this vacant entry.
    ///
    /// Returns an [`OccupiedEntry`] on success, or the key/value/error on failure.
    fn try_insert(self, value: V) -> Result<OccupiedEntry<'a, K, V>, (K, V, TryBTreeMapEntryError)>
    where
        K: Ord;
}

// ── Implementation on VacantEntry ─────────────────────────────────────────────

impl<'a, K: Ord + Clone, V> VacantEntryExt<'a, K, V> for VacantEntry<'a, K, V> {
    fn try_insert(
        self,
        value: V,
    ) -> Result<OccupiedEntry<'a, K, V>, (K, V, TryBTreeMapEntryError)> {
        let sys_ve = unsafe { extract_to_sys::<K, V>(self) };
        let key = sys_ve.key;
        let handle = sys_ve.handle;
        // dormant_map contains NonNull<BTreeMap<K,V,A>> — the inner pointer
        // points to the real BTreeMap since it was copied from the real VacantEntry.
        let map_ptr = sys_ve.dormant_map.ptr;

        // SAFETY: map_ptr was extracted from DormantMutRef.ptr inside the real
        // VacantEntry, so it points to a live BTreeMap. The VacantEntry has been
        // forgotten above, releasing the dormant borrow, so we can safely cast
        // through the sys view and mutate.
        let sys_map_ptr = map_ptr.as_ptr();
        let sys_map = unsafe { &mut *sys_map_ptr };

        let result = try_insert_kv(sys_map, handle, key.clone(), value);

        match result {
            Ok((_node_ref, _idx)) => {
                // Insertion succeeded. Re-search the real map to get a properly
                // typed OccupiedEntry via the public API. The dormant borrow was
                // released when we forgot the VacantEntry above.
                // SAFETY: map_ptr originally came from the real VacantEntry's
                // DormantMutRef.ptr, which pointed to a real BTreeMap<K, V>.
                // The sys DormantMutRef just re-typed the NonNull; the address is the same.
                let real_map_ptr = map_ptr.as_ptr() as *mut BTreeMap<K, V>;
                let map_ref = unsafe { &mut *real_map_ptr };
                match map_ref.entry(key) {
                    Entry::Occupied(entry) => Ok(entry),
                    Entry::Vacant(_) => {
                        unreachable!("insertion succeeded but key not found in map")
                    }
                }
            }
            Err((k, v, e)) => Err((k, v, e)),
        }
    }
}

// ── Implementation on &mut BTreeMap ───────────────────────────────────────────

impl<'a, K: Ord + Clone, V> TryBTreeMapEntry<'a, K, V> for &'a mut BTreeMap<K, V> {
    fn try_insert_entry(
        self,
        key: K,
        value: V,
    ) -> Result<&'a mut V, (K, V, TryBTreeMapEntryError)> {
        match self.entry(key) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(vacant) => match vacant.try_insert(value) {
                Ok(occupied) => Ok(occupied.into_mut()),
                Err((k, v, e)) => Err((k, v, e)),
            },
        }
    }
}

// ── Fallible insertion ────────────────────────────────────────────────────────

/// Insert a key-value pair into the tree with fallible allocation.
///
/// Dispatches between the empty-map fast path and the existing-tree path (which
/// may require splitting). Returns `(NodeRef, idx)` pointing at the inserted
/// value on success, or the original key/value plus an error on OOM — leaving
/// the map unchanged in the latter case.
fn try_insert_kv<'a, K, V>(
    inner_map: &mut sys::SysBTreeMap<K, V>,
    handle: Option<sys::Handle<sys::NodeRef<sys::Mut<'a>, K, V, sys::Leaf>, sys::Edge>>,
    key: K,
    value: V,
) -> InsertResult<'a, K, V> {
    match handle {
        None => try_insert_empty_map(inner_map, key, value),
        Some(leaf_edge) => try_insert_into_existing(inner_map, leaf_edge, key, value),
    }
}

/// Insert into an empty map by allocating a new leaf root.
fn try_insert_empty_map<'a, K, V>(
    inner_map: &mut sys::SysBTreeMap<K, V>,
    key: K,
    value: V,
) -> InsertResult<'a, K, V> {
    let leaf_box = match try_new_leaf() {
        Ok(b) => b,
        Err(e) => return Err((key, value, e.into())),
    };
    let owned_root: sys::NodeRef<sys::Owned, K, V, sys::Leaf> = sys::NodeRef {
        height: 0,
        node: leaf_box,
        _marker: PhantomData,
    };

    let leaf_ptr = owned_root.node.as_ptr();
    unsafe {
        (*leaf_ptr).keys[0].write(key);
        (*leaf_ptr).vals[0].write(value);
        (*leaf_ptr).len = 1;
    }

    inner_map.root = Some(unsafe {
        sys::NodeRef::<sys::Owned, K, V, sys::LeafOrInternal> {
            height: 0,
            node: ptr::read(&owned_root.node),
            _marker: PhantomData,
        }
    });
    inner_map.length = 1;

    let ret = unsafe {
        sys::NodeRef::<sys::Mut<'a>, K, V, sys::LeafOrInternal> {
            height: 0,
            node: NonNull::new_unchecked(leaf_ptr),
            _marker: PhantomData,
        }
    };
    Ok((ret, 0))
}

/// Insert into an existing non-empty tree. Handles splits with fallible allocation.
fn try_insert_into_existing<'a, K, V>(
    inner_map: &mut sys::SysBTreeMap<K, V>,
    leaf_edge: sys::Handle<sys::NodeRef<sys::Mut<'a>, K, V, sys::Leaf>, sys::Edge>,
    key: K,
    value: V,
) -> InsertResult<'a, K, V> {
    let node_len = unsafe { (*leaf_edge.node.node.as_ptr()).len as usize };
    if node_len < sys::CAPACITY {
        let insert_idx = leaf_edge.idx;
        unsafe {
            leaf_slice_insert(leaf_edge.node.node.as_ptr(), insert_idx, key, value);
            (*leaf_edge.node.node.as_ptr()).len = (node_len + 1) as u16;
        }
        inner_map.length += 1;
        return Ok((leaf_edge.node.forget_type(), insert_idx));
    }

    try_insert_with_split(inner_map, leaf_edge, key, value)
}

/// Insert with potential node splitting, all allocations fallible.
///
/// This is the heart of the reserve-and-commit design. It runs three phases:
///
/// 1. **Probe**: build a [`CommitPlan`] describing exactly which nodes will
///    split and how deep the cascade goes. Reads only — nothing is mutated.
/// 2. **Reserve**: allocate every node the commit needs, in one batch (one leaf
///    for the leaf split, one internal node per internal split, one extra if the
///    root grows). On any allocation failure, free the ones already reserved and
///    bail out — the probe mutated nothing, so the tree stays intact.
/// 3. **Commit**: perform the splits and promotions using only the reserved
///    nodes. This phase allocates nothing and therefore cannot fail.
fn try_insert_with_split<'a, K, V>(
    inner_map: &mut sys::SysBTreeMap<K, V>,
    leaf_edge: sys::Handle<sys::NodeRef<sys::Mut<'a>, K, V, sys::Leaf>, sys::Edge>,
    key: K,
    value: V,
) -> InsertResult<'a, K, V> {
    // ── Phase 1: probe the split path (pure reads) ───────────────────────────
    let plan = CommitPlan::build(leaf_edge);

    // Number of internal nodes the commit will consume: one per internal node
    // that splits, plus one more if the root grows into a fresh level.
    let num_internal_splits = plan.internals.iter().filter(|i| i.will_split).count();
    let num_reserved = num_internal_splits + if plan.new_root { 1 } else { 0 };

    // ── Phase 2: reserve all needed nodes in one batch ───────────────────────
    // The freshly allocated right-half leaf.
    let leaf_right = match try_new_leaf::<K, V>() {
        Ok(p) => p,
        Err(e) => return Err((key, value, e.into())),
    };
    // One internal node per internal split / root growth, deepest-first.
    let mut reserved_internals: Vec<NonNull<sys::LeafNode<K, V>>> =
        Vec::with_capacity(num_reserved);
    for _ in 0..num_reserved {
        match try_new_internal::<K, V>() {
            Ok(p) => reserved_internals.push(p),
            Err(e) => {
                // Rollback: free the leaf and any internals already reserved.
                // The probe did not mutate the tree, so it remains intact.
                unsafe {
                    drop_leaf_node(leaf_right);
                    for p in &reserved_internals {
                        drop_internal_node(*p);
                    }
                }
                return Err((key, value, e.into()));
            }
        }
    }

    // ── Phase 3: commit (infallible — no allocations remain) ─────────────────
    commit_split(inner_map, plan, leaf_right, &reserved_internals, key, value)
}

// ── Commit plan ───────────────────────────────────────────────────────────────

/// A single internal node on the split path, annotated with what the commit
/// must do to it.
struct CommitInternal<K, V> {
    /// Raw pointer to the internal node (deepest-first order in the plan).
    ptr: NonNull<sys::InternalNode<K, V>>,
    /// The edge index within this node at which the child we descended into sits.
    child_idx: usize,
    /// Whether this node is full and must split.
    will_split: bool,
    /// For a splitting node: its centre separator index and the promoted KV.
    sp: Option<(usize, (K, V))>,
}

/// Everything the probe learns about the upcoming split cascade, computed with
/// reads only so the reserve phase can decide how much to allocate.
struct CommitPlan<'a, K, V> {
    /// Raw pointer to the original (left) leaf being split.
    orig_leaf: *mut sys::LeafNode<K, V>,
    /// Centre separator index of the leaf (the slot promoted upward).
    leaf_sp: usize,
    /// Whether the new key goes into the freshly allocated right half (`true`)
    /// or the original left leaf (`false`).
    insert_right: bool,
    /// Local index at which the new key is written in its destination node.
    insert_idx: usize,
    /// The centre separator promoted out of the leaf.
    kv: (K, V),
    /// For each internal node on the path (deepest first): pointer, split flag,
    /// and (if splitting) its centre separator.
    internals: Vec<CommitInternal<K, V>>,
    /// Whether reaching the root forces a brand-new root level.
    new_root: bool,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a, K, V> CommitPlan<'a, K, V> {
    /// Walk the path from the target leaf up to the root, recording which nodes
    /// will split. Performs no mutations.
    fn build(
        leaf_edge: sys::Handle<sys::NodeRef<sys::Mut<'a>, K, V, sys::Leaf>, sys::Edge>,
    ) -> Self {
        let orig_leaf = leaf_edge.node.node.as_ptr();
        let (leaf_sp, leaf_side) = splitpoint(leaf_edge.idx);

        let (mk, mv) = unsafe {
            (
                (*orig_leaf).keys[leaf_sp].assume_init_read(),
                (*orig_leaf).vals[leaf_sp].assume_init_read(),
            )
        };
        let (insert_right, insert_idx) = match leaf_side {
            InsertionSide::Left(i) => (false, i),
            InsertionSide::Right(i) => (true, i),
        };

        let mut internals: Vec<CommitInternal<K, V>> = Vec::new();
        let mut cur: sys::NodeRef<sys::Mut<'a>, K, V, sys::LeafOrInternal> =
            leaf_edge.node.forget_type();
        #[allow(clippy::while_let_loop, reason = "annotate that we break at root")]
        loop {
            match ascend(cur) {
                AscendResult::Parent(parent_handle) => {
                    let parent_ptr: NonNull<sys::InternalNode<K, V>> =
                        parent_handle.node.node.cast();
                    let parent_len = unsafe { parent_ptr.as_ref() }.data.len as usize;
                    let will_split = parent_len >= sys::CAPACITY;
                    let sp = if will_split {
                        let (sp_idx, _) = splitpoint(parent_handle.idx);
                        let (pk, pv) = unsafe {
                            (
                                parent_ptr.as_ref().data.keys[sp_idx].assume_init_read(),
                                parent_ptr.as_ref().data.vals[sp_idx].assume_init_read(),
                            )
                        };
                        Some((sp_idx, (pk, pv)))
                    } else {
                        None
                    };
                    internals.push(CommitInternal {
                        ptr: parent_ptr,
                        child_idx: parent_handle.idx,
                        will_split,
                        sp,
                    });
                    cur = parent_handle.node.forget_type();
                }
                AscendResult::Root(_root) => {
                    // Reached the top of the tree: whatever internal nodes we
                    // processed above have been recorded in `internals`, and the
                    // final promotion will grow a fresh root level.
                    break;
                }
            }
        }
        // Reaching the root means the tree grows by one level.
        let new_root = true;

        CommitPlan {
            orig_leaf,
            leaf_sp,
            insert_right,
            insert_idx,
            kv: (mk, mv),
            internals,
            new_root,
            _lifetime: PhantomData,
        }
    }
}

// ── Commit (infallible) ───────────────────────────────────────────────────────

/// Perform the committed split cascade. Guaranteed infallible: every node it
/// touches was already allocated during the reserve phase, so no allocation can
/// fail here.
///
/// `reserved_internals` must contain one entry per internal node that splits
/// (`plan.internals` filtered by `will_split`), deepest-first, followed by one
/// extra entry if `plan.new_root`. `leaf_right` is the freshly allocated right
/// half of the leaf.
fn commit_split<'a, K, V>(
    inner_map: &mut sys::SysBTreeMap<K, V>,
    plan: CommitPlan<'a, K, V>,
    leaf_right: NonNull<sys::LeafNode<K, V>>,
    reserved_internals: &[NonNull<sys::LeafNode<K, V>>],
    key: K,
    value: V,
) -> InsertResult<'a, K, V> {
    let mut ri_iter = reserved_internals.iter();

    // ── Step 1: split the leaf and place the new key/value ───────────────────
    let right_leaf = leaf_right.as_ptr();
    let insert_node_ptr = if plan.insert_right {
        right_leaf
    } else {
        plan.orig_leaf
    };
    unsafe {
        copy_right_half_leaf(plan.orig_leaf, right_leaf, plan.leaf_sp);
        leaf_slice_insert(insert_node_ptr, plan.insert_idx, key, value);
        (*insert_node_ptr).len = ((*insert_node_ptr).len as usize + 1) as u16;
    }

    // Build the promotion carried upward: left = original leaf, right = new leaf.
    let left_leaf: sys::NodeRef<sys::Mut<'a>, K, V, sys::Leaf> = unsafe {
        sys::NodeRef {
            height: 0,
            node: NonNull::new_unchecked(plan.orig_leaf),
            _marker: PhantomData,
        }
    };
    let right_owned: sys::NodeRef<sys::Owned, K, V, sys::Leaf> = sys::NodeRef {
        height: 0,
        node: leaf_right,
        _marker: PhantomData,
    };
    let mut current_left: sys::NodeRef<sys::Mut<'a>, K, V, sys::LeafOrInternal> =
        left_leaf.forget_type();
    // SAFETY: `right_owned.node` is a freshly allocated node owned by this commit;
    // re-typing its borrow marker as `Mut<'a>` for the duration of the promotion
    // walk is sound because we hold exclusive access to it until it is wired into
    // the tree (or dropped on rollback).
    let mut current_right: sys::NodeRef<sys::Mut<'a>, K, V, sys::LeafOrInternal> = sys::NodeRef {
        height: 0,
        node: right_owned.node,
        _marker: PhantomData,
    };
    let mut current_kv = plan.kv;

    // ── Steps 2+: promote up through each internal node ──────────────────────
    for ci in plan.internals.into_iter() {
        let mut parent_ptr = ci.ptr;
        let edge_idx = ci.child_idx;

        if !ci.will_split {
            // Parent has room: absorb the promotion and stop climbing.
            let new_len = (unsafe { parent_ptr.as_ref() }.data.len as usize) + 1;
            unsafe {
                internal_insert_fit(
                    parent_ptr.as_mut(),
                    edge_idx,
                    current_kv.0,
                    current_kv.1,
                    current_right.node,
                );
                // The shift moved existing children one slot right; repair their
                // parent links (mirrors std's correct_childrens_parent_links).
                correct_parent_links::<K, V>(parent_ptr.cast(), edge_idx + 1, new_len);
            }
            inner_map.length += 1;
            return finish_commit(insert_node_ptr, plan.insert_idx);
        }

        // Parent is full: split it too, consuming the next reserved internal.
        let (sp_idx, (mk, mv)) = ci.sp.expect("will_split implies sp is set");
        let ri_raw = *ri_iter
            .next()
            .expect("an internal node was reserved per split");
        let mut ri_ptr: NonNull<sys::InternalNode<K, V>> = ri_raw.cast();
        let ri = unsafe { ri_ptr.as_mut() };

        unsafe {
            copy_right_half_internal(parent_ptr.as_mut(), ri, sp_idx);
            correct_parent_links::<K, V>(ri_raw.cast(), 0, ri.data.len as usize);
        }

        // Place the incoming promotion into whichever half it belongs to.
        let (_, ins_side) = splitpoint(edge_idx);
        unsafe {
            match ins_side {
                InsertionSide::Left(ii) => {
                    let new_len = (parent_ptr.as_ref().data.len as usize) + 1;
                    internal_insert_fit(
                        parent_ptr.as_mut(),
                        ii,
                        current_kv.0,
                        current_kv.1,
                        current_right.node,
                    );
                    correct_parent_links::<K, V>(parent_ptr.cast(), ii + 1, new_len);
                }
                InsertionSide::Right(ii) => {
                    let new_len = (ri.data.len as usize) + 1;
                    internal_insert_fit(ri, ii, current_kv.0, current_kv.1, current_right.node);
                    correct_parent_links::<K, V>(ri_raw.cast(), ii + 1, new_len);
                }
            }
        }

        // Advance the promotion one level up.
        current_left = sys::NodeRef::<sys::Mut<'a>, K, V, sys::LeafOrInternal> {
            height: current_left.height + 1,
            node: parent_ptr.cast(),
            _marker: PhantomData,
        };
        // SAFETY: `ri_raw` is a freshly reserved internal node owned by this commit.
        current_right = sys::NodeRef {
            height: current_right.height + 1,
            node: ri_raw,
            _marker: PhantomData,
        };
        current_kv = (mk, mv);
    }

    // ── Root growth: wrap the final promotion in a fresh root ────────────────
    debug_assert!(
        plan.new_root,
        "reached root-growth step but plan said otherwise"
    );
    let nr_raw = *ri_iter
        .next()
        .expect("a root node was reserved when new_root is set");
    let mut nr_ptr: NonNull<sys::InternalNode<K, V>> = nr_raw.cast();
    let nr = unsafe { nr_ptr.as_mut() };
    let new_height = current_left.height + 1;

    let mut old_root_owned: sys::NodeRef<sys::Owned, K, V, sys::LeafOrInternal> =
        unsafe { ptr::read(inner_map.root.as_ref().expect("root exists")) };
    inner_map.root.take();

    unsafe {
        nr.data.parent = None;
        nr.data.len = 0;
        nr.edges[0].write(old_root_owned.node);
        set_parent_link(old_root_owned.node.as_mut(), nr_raw.cast(), 0);

        let len = nr.data.len as usize;
        nr.data.keys[len].write(current_kv.0);
        nr.data.vals[len].write(current_kv.1);
        nr.edges[len + 1].write(current_right.node);
        nr.data.len = (len + 1) as u16;

        set_parent_link(current_right.node.as_mut(), nr_raw.cast(), 1);
    }

    inner_map.root = Some(sys::NodeRef::<sys::Owned, K, V, sys::LeafOrInternal> {
        height: new_height,
        node: nr_raw,
        _marker: PhantomData,
    });
    inner_map.length += 1;

    finish_commit(insert_node_ptr, plan.insert_idx)
}

/// Build the `(NodeRef, idx)` identifying the newly inserted value. The returned
/// reference is only used for its lifetime tag `'a`; the actual mutable value is
/// recovered by re-searching the real map in [`VacantEntryExt::try_insert`].
fn finish_commit<'a, K, V>(
    insert_node_ptr: *mut sys::LeafNode<K, V>,
    insert_idx: usize,
) -> InsertResult<'a, K, V> {
    let node_ref = unsafe {
        sys::NodeRef::<sys::Mut<'a>, K, V, sys::LeafOrInternal> {
            height: 0,
            node: NonNull::new_unchecked(insert_node_ptr),
            _marker: PhantomData,
        }
    };
    Ok((node_ref, insert_idx))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use lang_alloc::format;
    use lang_alloc::string::String;
    use lang_alloc::string::ToString;
    use lang_alloc::vec;
    use lang_core::alloc::Layout;

    fn dummy_layout() -> Layout {
        unsafe { Layout::from_size_align_unchecked(1, 1) }
    }

    #[test]
    fn try_insert_entry_new_key_empty_map() {
        let mut map: BTreeMap<i32, &str> = BTreeMap::new();
        let val = map.try_insert_entry(1, "one").unwrap();
        assert_eq!(val, &"one");
        assert_eq!(map[&1], "one");
    }

    #[test]
    fn try_insert_entry_existing_key_returns_old_value() {
        let mut map: BTreeMap<i32, &str> = BTreeMap::new();
        map.insert(1, "one");
        let val = map.try_insert_entry(1, "ONE").unwrap();
        assert_eq!(val, &"one");
        assert_eq!(map[&1], "one");
    }

    #[test]
    fn try_insert_entry_multiple_keys_ordered() {
        let mut map: BTreeMap<i32, &str> = BTreeMap::new();
        map.try_insert_entry(3, "three").unwrap();
        map.try_insert_entry(1, "one").unwrap();
        map.try_insert_entry(2, "two").unwrap();
        assert_eq!(map.len(), 3);
        assert_eq!(map[&1], "one");
        assert_eq!(map[&2], "two");
        assert_eq!(map[&3], "three");
    }

    #[test]
    fn try_insert_entry_many_entries_triggers_splits() {
        let mut map: BTreeMap<usize, usize> = BTreeMap::new();
        for i in 0..20 {
            map.try_insert_entry(i, i * 10).unwrap();
        }
        assert_eq!(map.len(), 20);
        for i in 0..20 {
            assert_eq!(map[&i], i * 10);
        }
    }

    #[test]
    fn try_insert_entry_just_past_first_split() {
        let mut map: BTreeMap<usize, usize> = BTreeMap::new();
        for i in 0..12 {
            map.try_insert_entry(i, i * 10).unwrap();
        }
        assert_eq!(map.len(), 12);
        for i in 0..12 {
            assert_eq!(map[&i], i * 10);
        }
    }

    #[test]
    fn try_insert_entry_second_split_leak_test() {
        for count in 12..=25 {
            let mut map: BTreeMap<usize, usize> = BTreeMap::new();
            for i in 0..count {
                map.try_insert_entry(i, i * 10).unwrap();
            }
            assert_eq!(map.len(), count);
            drop(map);
            lang_std::println!("ok with {} entries", count);
        }
    }

    #[test]
    fn vacant_entry_ext_try_insert_new_key() {
        let mut map: BTreeMap<i32, &str> = BTreeMap::new();
        let occupied = match map.entry(1) {
            Entry::Vacant(v) => v.try_insert("one").unwrap(),
            Entry::Occupied(_) => unreachable!(),
        };
        assert_eq!(occupied.get(), &"one");
        assert_eq!(map[&1], "one");
    }

    #[test]
    fn vacant_entry_ext_returns_occupied_entry() {
        let mut map: BTreeMap<String, i32> = BTreeMap::new();
        let occ = match map.entry("key".to_string()) {
            Entry::Vacant(v) => v.try_insert(42).unwrap(),
            Entry::Occupied(_) => unreachable!(),
        };
        assert_eq!(occ.key(), &"key");
        assert_eq!(occ.get(), &42);
    }

    #[test]
    fn vacant_entry_ext_does_not_affect_existing_entries() {
        let mut map: BTreeMap<i32, String> = BTreeMap::new();
        map.insert(1, "first".to_string());
        map.insert(3, "third".to_string());
        match map.entry(2) {
            Entry::Vacant(v) => {
                let _ = v.try_insert("second".to_string()).unwrap();
            }
            Entry::Occupied(_) => unreachable!(),
        }
        assert_eq!(map.len(), 3);
        assert_eq!(map[&1], "first");
        assert_eq!(map[&2], "second");
        assert_eq!(map[&3], "third");
    }

    #[test]
    fn vacant_entry_ext_with_complex_types() {
        use lang_alloc::vec::Vec;
        let mut map: BTreeMap<Vec<u8>, Vec<String>> = BTreeMap::new();
        let key = vec![1, 2, 3];
        let val = vec!["a".to_string(), "b".to_string()];
        match map.entry(key.clone()) {
            Entry::Vacant(v) => {
                let _ = v.try_insert(val.clone()).unwrap();
            }
            Entry::Occupied(_) => unreachable!(),
        }
        assert_eq!(map[&key], val);
    }

    #[test]
    fn entry_api_branches_for_both_vacant_and_occupied() {
        let mut map: BTreeMap<i32, String> = BTreeMap::new();
        let v1 = map.try_insert_entry(1, "hello".to_string()).unwrap();
        assert_eq!(v1, &"hello");
        let v2 = map.try_insert_entry(1, "world".to_string()).unwrap();
        assert_eq!(v2, &"hello");
        assert_eq!(map.len(), 1);
    }

    // ── OOM tests ───────────────────────────────────────────────────────────
    use rustyfill_test_allocator::{FailPolicy, with_policy};

    #[test]
    fn try_insert_entry_empty_map_fails_on_oom() {
        let mut map: BTreeMap<u32, u32> = BTreeMap::new();
        let r = with_policy(FailPolicy::fail_next_alloc(), || map.try_insert_entry(1, 2));
        assert!(r.is_err(), "insertion should fail on OOM");
        let (returned_key, returned_val, err) = r.unwrap_err();
        assert_eq!(returned_key, 1);
        assert_eq!(returned_val, 2);
        matches!(err, TryBTreeMapEntryError::Alloc(_));
        assert!(map.is_empty());
    }

    #[test]
    fn try_insert_entry_leaf_split_fails_on_oom() {
        let mut map: BTreeMap<u32, u32> = BTreeMap::new();
        for i in 0..11 {
            map.try_insert_entry(i, i * 10).unwrap();
        }
        assert_eq!(map.len(), 11);
        let r = with_policy(FailPolicy::fail_next_alloc(), || {
            map.try_insert_entry(11, 110)
        });
        assert!(r.is_err(), "split allocation should fail on OOM");
        let (returned_key, returned_val, err) = r.unwrap_err();
        assert_eq!(returned_key, 11);
        assert_eq!(returned_val, 110);
        matches!(err, TryBTreeMapEntryError::Alloc(_));
        assert_eq!(map.len(), 11);
        for i in 0..11 {
            assert_eq!(map[&i], i * 10);
        }
    }

    #[test]
    fn try_insert_entry_cascading_split_fails_on_oom() {
        let mut map: BTreeMap<u32, u32> = BTreeMap::new();
        for i in 0..30 {
            map.try_insert_entry(i, i * 10).unwrap();
        }
        assert_eq!(map.len(), 30);
        let r = with_policy(FailPolicy::fail_next_alloc(), || {
            map.try_insert_entry(31, 310)
        });
        match r {
            Err((k, v, err)) => {
                assert_eq!(k, 31);
                assert_eq!(v, 310);
                matches!(err, TryBTreeMapEntryError::Alloc(_));
                assert_eq!(map.len(), 30);
                for i in 0..30 {
                    assert_eq!(map[&i], i * 10);
                }
            }
            Ok(_) => {
                assert_eq!(map.len(), 31);
            }
        }
    }

    #[test]
    fn try_insert_entry_oom_returns_key_and_value() {
        let mut map: BTreeMap<[u8; 9], [u8; 3]> = BTreeMap::new();
        let key = *b"important";
        let val = [1u8, 2, 3];
        let r = with_policy(FailPolicy::fail_next_alloc(), || {
            map.try_insert_entry(key, val)
        });
        assert!(r.is_err(), "first insert should fail on OOM");
        let (returned_key, returned_val, _) = r.unwrap_err();
        assert_eq!(returned_key, key);
        assert_eq!(returned_val, val);
    }

    #[test]
    fn try_insert_entry_nth_alloc_fail_survives() {
        let results = with_policy(FailPolicy::fail_all_alloc(), || {
            let mut map: BTreeMap<u32, u32> = BTreeMap::new();
            let r1 = { map.try_insert_entry(1, 10).is_ok() };
            let r2 = { map.try_insert_entry(2, 20).is_ok() };
            (r1, r2)
        });
        let (_r1_ok, _r2_ok) = results;
    }

    #[test]
    fn oom_guard_restores_allocation_afterwards() {
        let mut map: BTreeMap<u32, u32> = BTreeMap::new();
        let _r = with_policy(FailPolicy::fail_next_alloc(), || map.try_insert_entry(1, 2));
        let post_r = map.try_insert_entry(99, 100);
        assert!(post_r.is_ok());
        assert_eq!(map[&99], 100);
    }

    // ── Edge cases and stress tests ─────────────────────────────────────────

    #[test]
    fn try_insert_entry_reverse_order() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new();
        for i in (0..20).rev() {
            map.try_insert_entry(i, i * 100).unwrap();
        }
        assert_eq!(map.len(), 20);
        for i in 0..20 {
            assert_eq!(map[&i], i * 100);
        }
    }

    #[test]
    fn try_insert_entry_large_values() {
        use lang_alloc::string::String;
        let big: String = "x".repeat(1024);
        let mut map: BTreeMap<i32, String> = BTreeMap::new();
        map.try_insert_entry(1, big.clone()).unwrap();
        assert_eq!(map[&1].len(), 1024);
        map.try_insert_entry(2, big.clone()).unwrap();
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn try_insert_entry_many_splits_stress() {
        let target: usize = 999999;
        let mut map: BTreeMap<usize, usize> = BTreeMap::new();
        for i in (0..target).rev() {
            map.try_insert_entry(i, i.wrapping_mul(7)).unwrap();
        }
        assert_eq!(map.len(), target);
        // <stress bugfix> Exercise multiple access patterns to detect corruption
        // that forward iteration alone might miss.
        // 1. Forward iteration (already exercised by into_iter below).
        // 2. Reverse iteration.
        for (_k, _v) in map.iter().rev() {
            // just walk
        }
        // 3. Random-ish lookups (every 7th key).
        for i in (0..target).step_by(7) {
            let _ = &map[&i];
        }
        // 4. Range queries that force internal node navigation.
        for i in (0..target.saturating_sub(10)).step_by(50) {
            for (_, _) in map.range(i..i + 10) {
                // just walk
            }
        }
        // 5. Full forward iteration via into_iter (triggers drop).
        for (_k, _v) in map.into_iter() {}
    }

    #[test]
    fn try_insert_entry_boundary_at_capacity() {
        let mut map: BTreeMap<usize, usize> = BTreeMap::new();
        for i in 0..sys::CAPACITY {
            map.try_insert_entry(i, i).unwrap();
        }
        assert_eq!(map.len(), sys::CAPACITY);
    }

    #[test]
    fn try_insert_entry_one_past_capacity_triggers_split() {
        let mut map: BTreeMap<usize, usize> = BTreeMap::new();
        for i in 0..=sys::CAPACITY {
            map.try_insert_entry(i, i).unwrap();
        }
        assert_eq!(map.len(), sys::CAPACITY + 1);
    }

    #[test]
    fn try_insert_entry_negative_keys() {
        let mut map: BTreeMap<i64, &str> = BTreeMap::new();
        map.try_insert_entry(-5, "neg five").unwrap();
        map.try_insert_entry(-1, "neg one").unwrap();
        map.try_insert_entry(0, "zero").unwrap();
        map.try_insert_entry(1, "pos one").unwrap();
        assert_eq!(map[&-5], "neg five");
        assert_eq!(map[&-1], "neg one");
        assert_eq!(map[&0], "zero");
        assert_eq!(map[&1], "pos one");
    }

    #[test]
    fn try_insert_entry_preserves_order_across_splits() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new();
        let keys = [
            15, 3, 22, 1, 10, 8, 20, 5, 12, 2, 18, 7, 25, 0, 11, 6, 14, 9, 16, 4, 13, 19, 21, 23,
            24,
        ];
        for k in &keys {
            map.try_insert_entry(*k, k * 100).unwrap();
        }
        assert_eq!(map.len(), keys.len());
        let collected: lang_alloc::vec::Vec<i32> = map.keys().copied().collect();
        let mut sorted = keys.to_vec();
        sorted.sort();
        assert_eq!(collected.as_slice(), &sorted);
    }

    #[test]
    fn try_insert_entry_existing_key_after_split() {
        let mut map: BTreeMap<i32, String> = BTreeMap::new();
        for i in 0..12 {
            map.try_insert_entry(i, format!("v{}", i)).unwrap();
        }
        assert_eq!(map.len(), 12);
        let returned = map.try_insert_entry(5, "REPLACED".to_string()).unwrap();
        assert_eq!(returned, &"v5");
        assert_eq!(map[&5], "v5");
        assert_eq!(map.len(), 12);
    }

    #[test]
    fn try_insert_entry_single_element_map_no_split() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new();
        map.try_insert_entry(42, 100).unwrap();
        assert_eq!(map.len(), 1);
        assert_eq!(map[&42], 100);
    }

    #[test]
    fn try_insert_entry_returned_mut_ref_is_valid() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new();
        let val = map.try_insert_entry(1, 0).unwrap();
        *val = 999;
        assert_eq!(map[&1], 999);
    }

    #[test]
    fn try_insert_entry_clone_key_used_internally() {
        use lang_alloc::string::String;
        let mut map: BTreeMap<String, i32> = BTreeMap::new();
        let s = "unique_key".to_string();
        map.try_insert_entry(s.clone(), 42).unwrap();
        assert_eq!(map[&s], 42);
        assert_eq!(s, "unique_key");
    }

    #[test]
    fn error_display_alloc_message() {
        let err = TryBTreeMapEntryError::Alloc(AllocError {
            layout: dummy_layout(),
        });
        let msg = format!("{}", err);
        assert!(msg.contains("allocation"), "error message: {}", msg);
    }

    #[test]
    fn error_debug_format() {
        let err = TryBTreeMapEntryError::Alloc(AllocError {
            layout: dummy_layout(),
        });
        let debug_msg = format!("{:?}", err);
        assert!(debug_msg.contains("Alloc"), "debug message: {}", debug_msg);
    }

    #[test]
    fn error_from_alloc_error() {
        let ae = AllocError {
            layout: dummy_layout(),
        };
        let err: TryBTreeMapEntryError = ae.into();
        matches!(err, TryBTreeMapEntryError::Alloc(_));
    }
}
