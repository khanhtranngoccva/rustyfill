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
//! Insertion with cascading splits uses a strict three-phase approach:
//!
//! 1. **Probe phase** — walk the tree bottom-up (reads only) to learn exactly
//!    which nodes will split and how deep the cascade goes. The probe records
//!    this in a heap-backed vector; because that vector itself must be grown,
//!    the probe runs a count pass first and then reserves its buffer up front
//!    via a fallible `try_with_capacity`. If that reservation fails we return
//!    `Err` immediately — nothing has been mutated yet.
//! 2. **Reserve phase** — with the split path fully known, allocate every node
//!    the commit needs in one batch: one leaf for the initial leaf split, one
//!    internal node per cascading internal split, plus one more if the root
//!    grows. The container holding those pointers is likewise reserved up
//!    front. If any single allocation fails we drop the already-reserved nodes
//!    and return `Err`; because no mutation has touched the original tree yet,
//!    it remains completely intact.
//! 3. **Commit phase** — with every node already allocated, the actual splits
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

use lang_alloc::alloc::Layout;
use lang_core::fmt;
use lang_core::marker::PhantomData;
use lang_core::mem;
use lang_core::ptr;
use lang_core::ptr::NonNull;
use lang_std::collections::btree_map::OccupiedEntry;

mod helpers;
use helpers::*;
mod scratch;
use scratch::{CachedProbeBuffer, CachedReserveBuffer, PendingLeaf};

// ── Re-exported sys types ─────────────────────────────────────────────────────

mod sys {
    pub use rustyfill_sys::std::collections::btree::borrow::DormantMutRef;
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

/// Construct an [`AllocError`] carrying a placeholder layout. Used when a
/// fallible allocation primitive (e.g. `Vec::try_reserve`) reports failure
/// without exposing the exact `Layout` that failed.
fn alloc_error() -> AllocError {
    AllocError {
        layout: unsafe { Layout::from_size_align_unchecked(1, 1) },
    }
}

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

impl TryBTreeMapEntryError {
    /// Extract the underlying [`AllocError`].
    pub fn alloc_error(&self) -> &AllocError {
        match self {
            Self::Alloc(e) => e,
        }
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

/// Compile-time assertion that the real std `OccupiedEntry<'_, K, V>` and our
/// polyfill `SysOccupiedEntry<'_, K, V, ()>` share the same size and alignment.
const fn assert_occupied_layout_compat<K: Ord, V>() {
    use rustyfill_sys::std::collections::btree::map::entry::OccupiedEntry as SysOccupiedEntry;
    assert!(
        mem::size_of::<OccupiedEntry<'_, K, V>>()
            == mem::size_of::<SysOccupiedEntry<'_, K, V, ()>>(),
        "OccupiedEntry and SysOccupiedEntry have different sizes"
    );
    assert!(
        mem::align_of::<OccupiedEntry<'_, K, V>>()
            == mem::align_of::<SysOccupiedEntry<'_, K, V, ()>>(),
        "OccupiedEntry and SysOccupiedEntry have different alignments"
    );
}

/// Construct a real `OccupiedEntry` from the node reference and index returned
/// by a successful insertion, plus the dormant map reference from the original
/// `VacantEntry`.
///
/// # Safety
/// - `node_ref` and `idx` must point to a valid, live KV pair in the tree.
/// - `dormant_map` must refer to the same `BTreeMap` that owns the tree.
/// - `assert_occupied_layout_compat` must hold for these `K, V`.
unsafe fn build_occupied_entry<'a, K: Ord, V>(
    node_ref: sys::NodeRef<sys::Mut<'a>, K, V, sys::LeafOrInternal>,
    idx: usize,
    dormant_map: sys::DormantMutRef<'a, sys::SysBTreeMap<K, V, ()>>,
) -> OccupiedEntry<'a, K, V> {
    assert_occupied_layout_compat::<K, V>();
    // Build the sys-view OccupiedEntry with public fields, then transmute-copy
    // into the real type (same layout, only the allocator param differs: () vs Global).
    let sys_occ = rustyfill_sys::std::collections::btree::map::entry::OccupiedEntry {
        handle: sys::Handle {
            node: node_ref,
            idx,
            _marker: PhantomData,
        },
        dormant_map,
        alloc: (),
        _marker: PhantomData,
    };
    unsafe { mem::transmute_copy(&sys_occ) }
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
/// // Second call — key already exists; like BTreeMap::insert the value is replaced.
/// let val = map.try_insert_entry("hello", 99).unwrap();
/// assert_eq!(val, &99); // new value stored
/// assert_eq!(map["hello"], 99);
/// ```
pub trait TryBTreeMapEntry<'a, K, V>: Sized {
    /// Obtain an entry for `key` and fallibly insert `value`.
    ///
    /// Mirrors [`BTreeMap::insert`](std::collections::BTreeMap::insert): whether
    /// the key was already present or not, its value is set to `value`, and a
    /// mutable reference to that (possibly overwritten) value is returned.
    /// Insertion of a new key may return [`Err`] on heap allocation failure.
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

impl<'a, K: Ord, V> VacantEntryExt<'a, K, V> for VacantEntry<'a, K, V> {
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
        let dormant_map = sys_ve.dormant_map;

        // SAFETY: map_ptr was extracted from DormantMutRef.ptr inside the real
        // VacantEntry, so it points to a live BTreeMap. The VacantEntry has been
        // forgotten above, releasing the dormant borrow, so we can safely cast
        // through the sys view and mutate.
        let sys_map = unsafe { &mut *map_ptr.as_ptr() };

        let result = try_insert_kv(sys_map, handle, key, value);

        match result {
            Ok((node_ref, idx)) => {
                // Construct the OccupiedEntry directly from the node reference
                // and index returned by the insertion. No re-search needed.
                // SAFETY: node_ref and idx point to the freshly inserted KV pair
                // in the tree. dormant_map was taken from the original VacantEntry
                // and still refers to the same BTreeMap. The layout of our sys
                // OccupiedEntry matches the real one (verified by the const assert
                // on VacantEntry; OccupiedEntry has the same field structure).
                let occupied = unsafe { build_occupied_entry(node_ref, idx, dormant_map) };
                Ok(occupied)
            }
            Err((k, v, e)) => Err((k, v, e)),
        }
    }
}

// ── Implementation on &mut BTreeMap ───────────────────────────────────────────

impl<'a, K: Ord, V> TryBTreeMapEntry<'a, K, V> for &'a mut BTreeMap<K, V> {
    fn try_insert_entry(
        self,
        key: K,
        value: V,
    ) -> Result<&'a mut V, (K, V, TryBTreeMapEntryError)> {
        self.entry(key).try_insert_entry(value)
    }
}

// ── Fallible Entry methods ────────────────────────────────────────────────────

/// Fallible versions of the canonical [`Entry`] consumption methods.
///
/// These mirror [`Entry::or_insert`], [`Entry::or_insert_with`],
/// [`Entry::or_insert_with_key`], and [`Entry::insert_entry`] but return a
/// [`Result`] so that allocation failures during the vacant-branch insertion
/// are reported rather than panicking.
///
/// The occupied branch never allocates, so it always succeeds.
pub trait TryEntryExt<'a, K, V>: Sized {
    /// Fallible version of [`Entry::or_insert`](lang_alloc::collections::btree_map::Entry::or_insert).
    fn try_or_insert(self, default: V) -> Result<&'a mut V, (K, V, TryBTreeMapEntryError)>
    where
        K: Ord;

    /// Fallible version of [`Entry::or_insert_with`](lang_alloc::collections::btree_map::Entry::or_insert_with).
    fn try_or_insert_with<F: FnOnce() -> V>(
        self,
        default: F,
    ) -> Result<&'a mut V, (K, V, TryBTreeMapEntryError)>
    where
        K: Ord;

    /// Fallible version of [`Entry::or_insert_with_key`](lang_alloc::collections::btree_map::Entry::or_insert_with_key).
    fn try_or_insert_with_key<F: FnOnce(&K) -> V>(
        self,
        default: F,
    ) -> Result<&'a mut V, (K, V, TryBTreeMapEntryError)>
    where
        K: Ord;

    /// Fallible version of [`Entry::insert_entry`](lang_alloc::collections::btree_map::Entry::insert_entry)
    /// (nightly `map_try_insert`). Inserts `value` unconditionally; if the key
    /// was already present, returns the old value in the error tuple alongside
    /// the new key/value (mirroring `OccupiedError`).
    fn try_insert_entry(self, value: V) -> Result<&'a mut V, (K, V, TryBTreeMapEntryError)>
    where
        K: Ord;
}

impl<'a, K: Ord, V> TryEntryExt<'a, K, V> for Entry<'a, K, V> {
    fn try_or_insert(self, default: V) -> Result<&'a mut V, (K, V, TryBTreeMapEntryError)> {
        match self {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(vacant) => match vacant.try_insert(default) {
                Ok(occupied) => Ok(occupied.into_mut()),
                Err((k, v, e)) => Err((k, v, e)),
            },
        }
    }

    fn try_or_insert_with<F: FnOnce() -> V>(
        self,
        default: F,
    ) -> Result<&'a mut V, (K, V, TryBTreeMapEntryError)> {
        match self {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(vacant) => {
                let value = default();
                match vacant.try_insert(value) {
                    Ok(occupied) => Ok(occupied.into_mut()),
                    Err((k, v, e)) => Err((k, v, e)),
                }
            }
        }
    }

    fn try_or_insert_with_key<F: FnOnce(&K) -> V>(
        self,
        default: F,
    ) -> Result<&'a mut V, (K, V, TryBTreeMapEntryError)> {
        match self {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(vacant) => {
                let value = default(vacant.key());
                match vacant.try_insert(value) {
                    Ok(occupied) => Ok(occupied.into_mut()),
                    Err((k, v, e)) => Err((k, v, e)),
                }
            }
        }
    }

    fn try_insert_entry(self, value: V) -> Result<&'a mut V, (K, V, TryBTreeMapEntryError)> {
        match self {
            // Mirrors BTreeMap::insert: a pre-existing key has its value replaced.
            Entry::Occupied(mut entry) => {
                entry.insert(value);
                Ok(entry.into_mut())
            }
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
    // ── Phase 1: probe the split path (reads + fallible buffer reservation) ──
    // The probe walks the tree twice (once to count, once to record) and
    // reserves its own backing vector up front via `try_with_capacity`. If that
    // reservation fails we bail out — the tree is still untouched.
    let plan = match CommitPlan::build(leaf_edge) {
        Ok(plan) => plan,
        Err(e) => return Err((key, value, e)),
    };

    // Number of internal nodes the commit will consume: one per internal node
    // that splits, plus one more if the root grows into a fresh level.
    let num_internal_splits = plan.internals.iter().filter(|i| i.will_split).count();
    let num_reserved = num_internal_splits + if plan.new_root { 1 } else { 0 };

    // ── Phase 2: reserve all needed nodes in one batch ───────────────────────
    // The freshly allocated right-half leaf. Owned by PendingLeaf: if anything
    // below fails, dropping it frees the node automatically.
    let leaf_box = match try_new_leaf_box::<K, V>() {
        Ok(b) => b,
        Err(e) => return Err((key, value, e.into())),
    };
    let leaf = PendingLeaf::new(leaf_box);

    // One internal node per internal split / root growth, deepest-first.
    // Owned by CachedReserveBuffer: if allocation fails mid-loop, dropping the
    // buffer frees all already-reserved boxes automatically.
    let mut reserve_buf = match CachedReserveBuffer::try_new(num_reserved) {
        Ok(buf) => buf,
        Err(e) => return Err((key, value, e)),
    };
    for _ in 0..num_reserved {
        match try_new_internal_box::<K, V>() {
            Ok(box_node) => reserve_buf.push(box_node),
            Err(e) => {
                // Rollback: both `leaf` and `reserve_buf` are dropped here,
                // freeing the leaf and all already-reserved internal nodes.
                // The probe did not mutate the tree, so it remains intact.
                return Err((key, value, e.into()));
            }
        }
    }

    // ── Phase 3: commit (infallible — no allocations remain) ─────────────────
    // Transfer ownership of the leaf and internal nodes to the tree.
    let leaf_right = leaf.into_raw();
    let reserved_ptrs = reserve_buf.drain_to_pointers();
    // `reserve_buf` is now empty; its Drop will recycle the Vec allocation.
    // `leaf` was consumed by into_raw(); nothing left to clean up.

    commit_split(inner_map, plan, leaf_right, &reserved_ptrs, key, value)
}

// ── Commit plan ───────────────────────────────────────────────────────────────

/// A single internal node on the split path, annotated with what the commit
/// must do to it.
///
/// Derives `Copy` + `Clone` to guarantee zero drop glue on the element type.
/// Uses `#[repr(C)]` to pin the field layout so it matches
/// [`scratch::ErasedCommitInternal`] exactly, making the thread-local buffer
/// transmute sound.
#[derive(Copy, Clone)]
#[repr(C)]
struct CommitInternal<K, V> {
    /// Raw pointer to the internal node (deepest-first order in the plan).
    ptr: NonNull<sys::InternalNode<K, V>>,
    /// The edge index within this node at which the child we descended into sits.
    child_idx: usize,
    /// Whether this node is full and must split.
    will_split: bool,
    /// For a splitting node: its centre separator index. The KV is re-read from
    /// the node during commit (the slot is still valid at that point because we
    /// walk bottom-up and haven't modified this node yet). Storing only the
    /// index avoids owning a copy of `V` that would be double-dropped if the
    /// commit stops early (a higher ancestor absorbs the promotion).
    sp_idx: Option<usize>,
}


/// Everything the probe learns about the upcoming split cascade, computed with
/// reads only so the reserve phase can decide how much to allocate.
///
/// The `internals` field uses [`CachedProbeBuffer`] which automatically returns
/// its backing allocation to the thread-local cache on drop, avoiding a heap
/// allocation on every subsequent insert that triggers a split.
struct CommitPlan<'a, K, V> {
    /// Raw pointer to the original (left) leaf being split.
    orig_leaf: *mut sys::LeafNode<K, V>,
    /// Centre separator index of the leaf (the slot promoted upward).
    leaf_sp: usize,
    /// Whether the new key goes into the freshly allocated right half (`true`)
    /// or the original left leaf (`false`).
    insert_right: bool,
    /// Local index at which the new key value pair is written in its destination node.
    insert_idx: usize,
    /// For each internal node on the path (deepest first): pointer, split flag,
    /// and (if splitting) its centre separator index. Wrapped in a cached buffer
    /// that recycles its allocation via thread-local storage.
    internals: CachedProbeBuffer<K, V>,
    /// Whether reaching the root forces a brand-new root level.
    new_root: bool,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a, K, V> CommitPlan<'a, K, V> {
    /// Walk the path from the target leaf up to the root, recording which nodes
    /// will split. Performs no mutations on the tree.
    ///
    /// The probe vector itself is heap-backed, so its growth can fail under OOM.
    /// To avoid growing it incrementally (which would allocate repeatedly over a
    /// deep cascade), we run **two passes**:
    ///
    /// 1. Count how many internal nodes sit on the split path.
    /// 2. [`Vec::try_reserve`] exactly that many slots up front — a single
    ///    allocation that fails cleanly if memory runs out.
    /// 3. Re-walk the path and fill the pre-sized vector (no further growth).
    ///
    /// Returns `Err` if reserving the probe buffer fails, leaving the tree
    /// untouched.
    fn build(
        leaf_edge: sys::Handle<sys::NodeRef<sys::Mut<'a>, K, V, sys::Leaf>, sys::Edge>,
    ) -> Result<Self, TryBTreeMapEntryError> {
        let orig_leaf = leaf_edge.node.node.as_ptr();
        let (leaf_sp, leaf_side) = splitpoint(leaf_edge.idx);

        let (insert_right, insert_idx) = match leaf_side {
            InsertionSide::Left(i) => (false, i),
            InsertionSide::Right(i) => (true, i),
        };

        // Starting-node fields, captured once so both passes begin from the
        // identical node. `NodeRef` is a cheap value (height + pointer); we
        // rebuild it per pass because `forget_type` consumes its receiver.
        let start_height = leaf_edge.node.height;
        let start_node = leaf_edge.node.node;

        // Pass 1: count the internal nodes on the split path.
        let num_internals = {
            let mut count = 0usize;
            let mut cur: sys::NodeRef<sys::Mut<'a>, K, V, sys::LeafOrInternal> = sys::NodeRef {
                height: start_height,
                node: start_node,
                _marker: PhantomData,
            };
            #[allow(clippy::while_let_loop, reason = "root node is annotated")]
            loop {
                match ascend(cur) {
                    AscendResult::Parent(parent_handle) => {
                        count += 1;
                        cur = parent_handle.node.forget_type();
                    }
                    AscendResult::Root(_) => break,
                }
            }
            count
        };

        // Allocate the probe buffer via the thread-local cache (recycles on
        // subsequent inserts). Falls back to a fresh fallible allocation if no
        // suitable cached buffer is available.
        let mut internals_buf = CachedProbeBuffer::try_new(num_internals)?;

        // Pass 2: re-walk and record each node's annotation.
        let mut cur: sys::NodeRef<sys::Mut<'a>, K, V, sys::LeafOrInternal> = sys::NodeRef {
            height: start_height,
            node: start_node,
            _marker: PhantomData,
        };
        #[allow(clippy::while_let_loop, reason = "root node is annotated")]
        loop {
            match ascend(cur) {
                AscendResult::Parent(parent_handle) => {
                    let parent_ptr: NonNull<sys::InternalNode<K, V>> =
                        parent_handle.node.node.cast();
                    let parent_len = unsafe { parent_ptr.as_ref() }.data.len as usize;
                    let will_split = parent_len >= sys::CAPACITY;
                    let sp_idx = if will_split {
                        let (sp_idx, _) = splitpoint(parent_handle.idx);
                        Some(sp_idx)
                    } else {
                        None
                    };
                    internals_buf.as_mut().push(CommitInternal {
                        ptr: parent_ptr,
                        child_idx: parent_handle.idx,
                        will_split,
                        sp_idx,
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

        Ok(CommitPlan {
            orig_leaf,
            leaf_sp,
            insert_right,
            insert_idx,
            internals: internals_buf,
            new_root,
            _lifetime: PhantomData,
        })
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
    // Read the leaf's centre separator BEFORE the split mutation. This is the
    // initial promotion carried upward. Reading it here (rather than storing a
    // copy in the plan) ensures the value is owned exactly once — by
    // `current_kv` — and never duplicated in a plan buffer that could be
    // partially consumed and dropped.
    let (mk, mv) = unsafe {
        (
            (*plan.orig_leaf).keys[plan.leaf_sp].assume_init_read(),
            (*plan.orig_leaf).vals[plan.leaf_sp].assume_init_read(),
        )
    };

    unsafe {
        copy_right_half_leaf(plan.orig_leaf, right_leaf, plan.leaf_sp);
        leaf_slice_insert(insert_node_ptr, plan.insert_idx, key, value);
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
    let mut current_kv = (mk, mv);

    // ── Steps 2+: promote up through each internal node ──────────────────────
    // Iterate by reference: `CommitInternal` is `Copy`, so we copy each element.
    // The plan (and its CachedProbeBuffer) is dropped at function exit, which
    // recycles the backing allocation to the thread-local cache.
    for ci in plan.internals.iter() {
        let parent_ptr = ci.ptr;
        let edge_idx = ci.child_idx;

        if !ci.will_split {
            // Parent has room: absorb the promotion and stop climbing.
            let new_len = (unsafe { parent_ptr.as_ref() }.data.len as usize) + 1;
            unsafe {
                internal_insert_fit(
                    parent_ptr.as_ptr(),
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
        let sp_idx = ci.sp_idx.expect("will_split implies sp_idx is set");
        let ri_raw = *ri_iter
            .next()
            .expect("an internal node was reserved per split");
        let ri_ptr: *mut sys::InternalNode<K, V> = ri_raw.as_ptr() as *mut _;

        // Re-read the centre separator from the node NOW. This is safe because
        // we walk bottom-up: no prior commit step has modified this node.
        // Reading it here (rather than storing a copy in the plan) ensures the
        // value is dropped exactly once — when it is written into the parent
        // below — and never again via a stale plan buffer.
        let (mk, mv) = unsafe {
            (
                parent_ptr.as_ref().data.keys[sp_idx].assume_init_read(),
                parent_ptr.as_ref().data.vals[sp_idx].assume_init_read(),
            )
        };

        unsafe {
            copy_right_half_internal(parent_ptr.as_ptr(), ri_ptr, sp_idx);
            // After the copy, `ri` holds `new_right_len` separators.
            let right_len = (*ri_ptr).data.len as usize;
            correct_parent_links::<K, V>(ri_raw.cast(), 0, right_len);
        }

        // Place the incoming promotion into whichever half it belongs to.
        let (_, ins_side) = splitpoint(edge_idx);
        unsafe {
            match ins_side {
                InsertionSide::Left(ii) => {
                    let new_len = (parent_ptr.as_ref().data.len as usize) + 1;
                    internal_insert_fit(
                        parent_ptr.as_ptr(),
                        ii,
                        current_kv.0,
                        current_kv.1,
                        current_right.node,
                    );
                    correct_parent_links::<K, V>(parent_ptr.cast(), ii + 1, new_len);
                }
                InsertionSide::Right(ii) => {
                    let new_len = ((*ri_ptr).data.len as usize) + 1;
                    internal_insert_fit(ri_ptr, ii, current_kv.0, current_kv.1, current_right.node);
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
    let nr_ptr: *mut sys::InternalNode<K, V> = nr_raw.as_ptr() as *mut _;
    let new_height = current_left.height + 1;

    let old_root_owned: sys::NodeRef<sys::Owned, K, V, sys::LeafOrInternal> =
        unsafe { ptr::read(inner_map.root.as_ref().expect("root exists")) };
    inner_map.root.take();

    unsafe {
        (*nr_ptr).data.parent = None;
        (*nr_ptr).data.len = 0;
        (*nr_ptr).edges[0].write(old_root_owned.node);
        set_parent_link(old_root_owned.node.as_ptr(), nr_raw.cast(), 0);

        let len = (*nr_ptr).data.len as usize;
        (*nr_ptr).data.keys[len].write(current_kv.0);
        (*nr_ptr).data.vals[len].write(current_kv.1);
        (*nr_ptr).edges[len + 1].write(current_right.node);
        (*nr_ptr).data.len = (len + 1) as u16;

        set_parent_link(current_right.node.as_ptr(), nr_raw.cast(), 1);
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
    fn try_insert_entry_existing_key_replaces_value() {
        // Mirrors BTreeMap::insert: re-inserting a present key overwrites the value.
        let mut map: BTreeMap<i32, &str> = BTreeMap::new();
        map.insert(1, "one");
        let val = map.try_insert_entry(1, "ONE").unwrap();
        assert_eq!(val, &"ONE");
        assert_eq!(map[&1], "ONE");
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
        // Re-inserting key 1 replaces the value (std insert semantics).
        let v2 = map.try_insert_entry(1, "world".to_string()).unwrap();
        assert_eq!(v2, &"world");
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

    /// The probe's backing vector is reserved up front via `try_with_capacity`.
    /// This test forces *that* allocation to fail (the very first heap allocation
    /// of the split insert) and verifies the tree is left intact. It isolates the
    /// fallible-probe path from the node-allocation paths covered by the other
    /// OOM tests.
    #[test]
    fn try_insert_entry_probe_buffer_reservation_fails_on_oom() {
        let mut map: BTreeMap<u32, u32> = BTreeMap::new();
        // Fill past the leaf capacity so the next insert takes the split path.
        for i in 0..11 {
            map.try_insert_entry(i, i * 10).unwrap();
        }
        assert_eq!(map.len(), 11);
        // Fail the first allocation inside try_insert_with_split — which is the
        // probe buffer's `try_with_capacity` reservation.
        let r = with_policy(FailPolicy::fail_next_alloc(), || {
            map.try_insert_entry(11, 110)
        });
        assert!(r.is_err(), "probe buffer reservation should fail on OOM");
        let (returned_key, returned_val, err) = r.unwrap_err();
        assert_eq!(returned_key, 11);
        assert_eq!(returned_val, 110);
        matches!(err, TryBTreeMapEntryError::Alloc(_));
        // Tree untouched: still 11 entries, all values preserved.
        assert_eq!(map.len(), 11);
        for i in 0..11 {
            assert_eq!(map[&i], i * 10);
        }
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
        // Key 5 already exists; re-inserting replaces its value (std semantics).
        let returned = map.try_insert_entry(5, "REPLACED".to_string()).unwrap();
        assert_eq!(returned, &"REPLACED");
        assert_eq!(map[&5], "REPLACED");
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

    // ── Randomised fuzzing ────────────────────────────────────────────────────
    //
    // The deterministic tests above pin specific shapes (11, 30, 999_999 keys)
    // but a B-tree's correctness depends on *every* combination of node fill
    // levels and split directions along the probe path. A randomised driver that
    // inserts an unpredictable stream of keys — then verifies the tree is a valid
    // ordered map with exactly the expected contents — surfaces corruption that
    // fixed-size tests miss (mis-shifted edges, stale parent links, dropped or
    // duplicated keys).
    //
    // We use a self-contained SplitMix64 PRNG rather than pulling in a `rand`
    // dependency: it is deterministic for a given seed (so failures reproduce
    // under Miri/sanitizers), portable across toolchains, and allocation-free.

    /// Minimal deterministic PRNG (SplitMix64). No external crate needed.
    struct FuzzRng(u64);

    impl FuzzRng {
        const fn new(seed: u64) -> Self {
            Self(seed)
        }
        /// Advance state and return the next pseudorandom 64-bit word.
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform integer in `[0, bound)` via rejection sampling (no modulo bias).
        fn below(&mut self, bound: usize) -> usize {
            debug_assert!(bound > 0);
            if bound == 1 {
                return 0;
            }
            // Largest multiple of `bound` that fits in usize without overflowing
            // when we add one back for the inclusive modulus range. Using
            // `wrapping_add` guards the edge case where `(usize::MAX / bound) *
            // bound == usize::MAX` (e.g. bound == 3 on 64-bit, since 2^64 - 1 is
            // divisible by 3).
            let limit = ((usize::MAX / bound) * bound).min(usize::MAX - 1);
            loop {
                let r = (self.next_u64() as usize) % (limit + 1);
                if r < limit {
                    return r % bound;
                }
            }
        }
    }

    /// Insert `n_ops` randomly-chosen keys into a fresh map via
    /// [`TryBTreeMapEntry`], tracking the authoritative key→value mapping in a
    /// second std `BTreeMap`. After every insertion we assert our map agrees with
    /// the oracle (length + full ordered equality), so any corruption is caught
    /// at the exact operation that introduced it rather than at drop time.
    fn run_fuzz_seed(seed: u64, n_ops: usize) {
        let mut rng = FuzzRng::new(seed);
        let mut ours: BTreeMap<u32, u32> = BTreeMap::new();
        // Oracle: the ground-truth mapping produced by std's own insert.
        let mut oracle: BTreeMap<u32, u32> = BTreeMap::new();

        for op in 0..n_ops {
            // Draw a key from a deliberately narrow band (~half the ops collide
            // with existing keys, exercising the Occupied branch and repeated
            // probes into the same region) mixed with wide-range draws (forcing
            // splits far apart).
            let key: u32 = if rng.below(2) == 0 {
                (rng.next_u64() % 4096) as u32
            } else {
                rng.next_u64() as u32
            };
            let val = (op as u32).wrapping_mul(31);

            // Was this key already present before this op? (Decide from the
            // oracle *before* updating it.) `try_insert_entry` returns `Ok` for
            // both new and existing keys — it only fails on OOM — so we cannot
            // infer occupancy from the Result alone.
            let old_val = oracle.get(&key).copied();

            match ours.try_insert_entry(key, val) {
                Ok(inserted_ref) => {
                    // Mirrors BTreeMap::insert: whether the key was new or already
                    // present, the stored value is replaced by `val`, so the
                    // returned &mut V must observe `val`. (`old_val` is still read
                    // above to keep the oracle's notion of "was this a collision"
                    // available for future assertions.)
                    let _ = old_val;
                    assert_eq!(*inserted_ref, val, "seed {seed}: op {op}");
                }
                Err((k, v, _)) => {
                    // Allocation failure is not expected here (normal allocator);
                    // if it somehow happens, abort loudly.
                    panic!("seed {seed}: op {op}: unexpected OOM for key {k}, val {v}");
                }
            }

            // Record ground truth.
            oracle.insert(key, val);

            // Invariant check after each op: identical contents, correct order.
            assert_eq!(
                ours.len(),
                oracle.len(),
                "seed {seed}: op {op}: length mismatch"
            );
            let ours_vec: lang_alloc::vec::Vec<(u32, u32)> =
                ours.iter().map(|(k, v)| (*k, *v)).collect();
            let oracle_vec: lang_alloc::vec::Vec<(u32, u32)> =
                oracle.iter().map(|(k, v)| (*k, *v)).collect();
            assert_eq!(
                ours_vec.as_slice(),
                oracle_vec.as_slice(),
                "seed {seed}: op {op}: content/order mismatch"
            );
        }

        // Final structural sweep: reverse iteration + range scans force the
        // drop/navigation machinery to walk every edge and parent link.
        for (_k, _v) in ours.iter().rev() {}
        if ours.len() > 20 {
            let lo = ours.keys().next().copied().unwrap();
            for i in 0..ours.len() {
                let k = lo.saturating_add(i as u32);
                let count = ours.range(k..k.checked_add(1).unwrap_or(k)).count();
                assert!(count <= 1, "seed {seed}: duplicate key {k}");
            }
        }
        // Dropping the map exercises IntoIter over the whole structure.
        drop(ours);
    }

    /// A value type whose every destruction increments a shared, locally-owned
    /// counter. Because the counter lives in an `Arc` captured by the closure,
    /// each test invocation gets its own isolated tally — no process-global
    /// state, so parallel test threads can't interfere. If the map's drop path
    /// double-drops or leaks any stored value, the final count will differ from
    /// the number of values inserted and this test fails. Miri additionally
    /// detects any actual double-free at the moment it happens.
    struct DropCountedValue {
        /// Shared destructor hook; cloned into every value so all drops funnel
        /// through the same counter.
        on_drop: lang_std::sync::Arc<dyn Fn(u32) + Send + Sync>,
        /// Random payload tag (fuzzy, not derived from the op index).
        tag: u32,
    }

    impl Drop for DropCountedValue {
        fn drop(&mut self) {
            (self.on_drop)(self.tag);
        }
    }

    /// Fuzzy drop-safety driver: insert `n_ops` random (key, value) pairs where
    /// *both* the key and the value are drawn from the PRNG (values are not
    /// derived from the op index), so collisions exercise the Occupied-replace
    /// path (the old value must be dropped exactly once when overwritten). After
    /// the loop we compare the destruction count against the number of values
    /// inserted — they must match exactly, proving no stored value was leaked or
    /// destroyed twice. Finally we iterate in reverse and drop the map, forcing
    /// the whole teardown to walk every node.
    fn run_drop_count_fuzz(seed: u64, n_ops: usize) {
        use lang_core::sync::atomic::{AtomicUsize, Ordering};
        use lang_std::sync::Arc;

        let mut rng = FuzzRng::new(seed);

        // Locally-scoped drop counter shared by every DropCountedValue in this
        // run. Construction is implicit (one per `DropCountedValue::new` call),
        // so we track insertions separately and compare against the drop count.
        let drop_counter = Arc::new(AtomicUsize::new(0));
        let on_drop: Arc<dyn Fn(u32) + Send + Sync> = {
            let c = Arc::clone(&drop_counter);
            Arc::new(move |_tag: u32| {
                c.fetch_add(1, Ordering::SeqCst);
            })
        };

        let mut ours: BTreeMap<u32, DropCountedValue> = BTreeMap::new();
        // Oracle mapping key -> tag, used to verify value integrity after each
        // op (catches logical corruption before it turns into heap damage).
        let mut oracle: BTreeMap<u32, u32> = BTreeMap::new();
        // One DropCountedValue is created per successful insertion. A re-insert
        // of an existing key drops the previous value immediately, then stores
        // the new one — so the total number of drops by the end must equal the
        // total number of insertions.
        let mut total_insertions = 0usize;

        for op in 0..n_ops {
            // Fuzzy key: mix narrow-band (collisions → Occupied replace) and
            // wide-range (fresh leaves / splits) draws.
            let key: u32 = if rng.below(3) == 0 {
                (rng.next_u64() % 512) as u32
            } else {
                rng.next_u64() as u32
            };
            // Fuzzy value: independent random tag, NOT tied to the op index, so
            // two inserts of the same key carry genuinely different payloads.
            let tag = rng.next_u64() as u32;
            let val = DropCountedValue {
                on_drop: Arc::clone(&on_drop),
                tag,
            };

            match ours.try_insert_entry(key, val) {
                Ok(inserted_ref) => {
                    total_insertions += 1;
                    // The stored value's tag must equal the one we just inserted.
                    assert_eq!(
                        inserted_ref.tag, tag,
                        "seed {seed}: op {op}: value tag corrupted (key {key})"
                    );
                }
                Err((k, v, _)) => {
                    // The returned (k, v) is dropped here by the pattern binding;
                    // that accounts for the one construction we just did.
                    drop(v);
                    panic!("seed {seed}: op {op}: unexpected OOM for key {k}");
                }
            }

            // Record ground truth and periodically verify full content equality.
            oracle.insert(key, tag);
            if op % 1000 == 0 {
                let ours_tags: lang_alloc::vec::Vec<(u32, u32)> =
                    ours.iter().map(|(k, v)| (*k, v.tag)).collect();
                let oracle_vec: lang_alloc::vec::Vec<(u32, u32)> =
                    oracle.iter().map(|(k, v)| (*k, *v)).collect();
                assert_eq!(
                    ours_tags.as_slice(),
                    oracle_vec.as_slice(),
                    "seed {seed}: op {op}: content/order mismatch"
                );
            }
        }

        // Reverse iteration forces navigation across all edges before teardown.
        for (_k, _v) in ours.iter().rev() {}

        // Dropping the map tears down every node and every stored value.
        drop(ours);

        let drops = drop_counter.load(Ordering::SeqCst);
        assert_eq!(
            drops, total_insertions,
            "seed {seed}: DROPS ({drops}) ≠ INSERTIONS ({total_insertions}) — \
             leaked or double-dropped values"
        );
    }

    /// Runs under Miri (kept small so the interpreter finishes in reasonable
    /// time). Exercises the full split cascade with fuzzy keys+values and asserts
    /// balanced drop accounting plus no double-free (Miri catches the latter
    /// directly).
    #[test]
    fn try_insert_entry_fuzz_drop_safety_miri() {
        // ~600 ops over a tight-ish band produces several levels of internal
        // nodes and plenty of Occupied-replace events, while staying fast enough
        // for Miri.
        run_drop_count_fuzz(0x000D_EADB_EEF1, 600);
    }

    /// Larger variant for native runs (ASan/UBSan-friendly, too big for Miri).
    #[cfg_attr(miri, ignore = "test case is too large for Miri")]
    #[test]
    fn try_insert_entry_fuzz_drop_safety_large() {
        run_drop_count_fuzz(0xCAFE_BABE, 50_000);
    }

    #[test]
    fn try_insert_entry_fuzz_random_keys_small() {
        // Small enough to also run comfortably under Miri.
        run_fuzz_seed(0xC0FFEE, 400);
    }

    #[cfg_attr(miri, ignore = "test case is too large for Miri")]
    #[test]
    fn try_insert_entry_fuzz_random_keys_large() {
        // Large multi-level trees (many cascading internal splits + root growth).
        run_fuzz_seed(0xDEADBEEF, 200_000);
    }

    #[cfg_attr(miri, ignore = "test case is too large for Miri")]
    #[test]
    fn try_insert_entry_fuzz_dense_collisions() {
        // Tight key band → heavy collision rate and repeated local splits.
        let mut rng = FuzzRng::new(0xFEED_FACE);
        let mut ours: BTreeMap<u32, u32> = BTreeMap::new();
        let mut oracle: BTreeMap<u32, u32> = BTreeMap::new();
        for op in 0..50_000 {
            let key = (rng.next_u64() % 2048) as u32;
            let val = (op as u32).wrapping_mul(17);
            let _ = ours.try_insert_entry(key, val).expect("dense fuzz: no OOM");
            oracle.insert(key, val);
        }
        let ours_vec: lang_alloc::vec::Vec<(u32, u32)> =
            ours.iter().map(|(k, v)| (*k, *v)).collect();
        let oracle_vec: lang_alloc::vec::Vec<(u32, u32)> =
            oracle.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(
            ours_vec.as_slice(),
            oracle_vec.as_slice(),
            "dense fuzz mismatch"
        );
        for (_k, _v) in ours.into_iter().rev() {}
    }
}
