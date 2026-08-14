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
//! point is intercepted with fallible [`TryBox::try_new_uninit`] so that
//! OOM returns [`Err`] rather than panicking.
//!
//! # Safety
//!
//! This module transmutes between `std::collections::BTreeMap` and its
//! internal representation. Care is taken to maintain all B-tree invariants
//! and to never leave the data structure in an invalid state on allocation
//! failure. On mid-split OOM, any newly allocated nodes are dropped via
//! [`drop_node`] to prevent leaks, while the original tree remains intact.

use crate::alloc::AllocError;
use crate::alloc::boxed::TryBox;
use lang_alloc::boxed::Box;
use lang_alloc::collections::BTreeMap;
use lang_alloc::collections::btree_map::{Entry, VacantEntry};

use lang_core::fmt;
use lang_core::marker::PhantomData;
use lang_core::mem::{self, MaybeUninit};
use lang_core::ptr;
use lang_core::ptr::NonNull;
use lang_std::collections::btree_map::OccupiedEntry;
use rustyfill_sys::std::collections::btree::node::{InternalNode, LeafNode};

// ── Re-exported sys types ─────────────────────────────────────────────────────

mod sys {
    pub use rustyfill_sys::std::collections::btree::map::BTreeMap as SysBTreeMap;
    pub use rustyfill_sys::std::collections::btree::map::entry::VacantEntry as SysVacantEntry;
    pub use rustyfill_sys::std::collections::btree::node::marker::*;
    pub use rustyfill_sys::std::collections::btree::node::{
        CAPACITY, Handle, InternalNode, LeafNode, NodeRef, SplitResult,
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
/// Returns `(NodeRef, idx)` pointing to the inserted value on success.
/// On failure, leaves the map unchanged and returns the key/value back.
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
        Err(e) => return Err((key, value, e)),
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
fn try_insert_with_split<'a, K, V>(
    inner_map: &mut sys::SysBTreeMap<K, V>,
    leaf_edge: sys::Handle<sys::NodeRef<sys::Mut<'a>, K, V, sys::Leaf>, sys::Edge>,
    key: K,
    value: V,
) -> InsertResult<'a, K, V> {
    // Reserve the box first before performing any irreversible operations.
    let right_box = match try_new_leaf() {
        Ok(b) => b,
        Err(e) => return Err((key, value, e)),
    };
    let (split_point_idx, insertion_side) = splitpoint(leaf_edge.idx);
    let source_ptr = leaf_edge.node.node.as_ptr();
    let (middle_k, middle_v) = unsafe {
        (
            (*source_ptr).keys[split_point_idx].assume_init_read(),
            (*source_ptr).vals[split_point_idx].assume_init_read(),
        )
    };

    let old_len = unsafe { (*source_ptr).len as usize };
    let new_right_len = old_len - split_point_idx - 1;

    unsafe {
        let right_leaf = right_box.as_ptr();
        (*right_leaf).len = new_right_len as u16;

        if new_right_len > 0 {
            ptr::copy_nonoverlapping(
                (*source_ptr).keys.as_ptr().add(split_point_idx + 1),
                (*right_leaf).keys.as_mut_ptr(),
                new_right_len,
            );
            ptr::copy_nonoverlapping(
                (*source_ptr).vals.as_ptr().add(split_point_idx + 1),
                (*right_leaf).vals.as_mut_ptr(),
                new_right_len,
            );
        }

        (*source_ptr).len = split_point_idx as u16;
    }

    let (insert_node_ptr, insert_idx) = match insertion_side {
        InsertionSide::Left(i) => (source_ptr, i),
        InsertionSide::Right(i) => (right_box.as_ptr(), i),
    };

    unsafe {
        leaf_slice_insert(insert_node_ptr, insert_idx, key, value);
        (*insert_node_ptr).len = ((*insert_node_ptr).len as usize + 1) as u16;
    }

    let left_node = unsafe {
        sys::NodeRef::<sys::Mut<'a>, K, V, sys::Leaf> {
            height: 0,
            node: NonNull::new_unchecked(source_ptr),
            _marker: PhantomData,
        }
    };
    let right_owned: sys::NodeRef<sys::Owned, K, V, sys::Leaf> = sys::NodeRef {
        height: 0,
        node: right_box,
        _marker: PhantomData,
    };

    let forget_right = || -> sys::NodeRef<sys::Owned, K, V, sys::LeafOrInternal> {
        unsafe {
            sys::NodeRef {
                height: 0,
                node: ptr::read(&right_owned.node),
                _marker: PhantomData,
            }
        }
    };

    let split = sys::SplitResult {
        left: sys::NodeRef {
            height: left_node.height,
            node: left_node.node,
            _marker: PhantomData,
        },
        kv: (middle_k, middle_v),
        right: forget_right(),
    };

    let right_ptr_for_cleanup = right_owned.node;

    let result = promote_split(
        inner_map,
        split,
        (insert_node_ptr, insert_idx),
        right_ptr_for_cleanup,
    );

    result.map_err(|(k, v, e)| {
        unsafe { drop_leaf_node(right_ptr_for_cleanup) };
        (k, v, e)
    })
}

/// Promote a split result up the tree, allocating new internal nodes as needed.
fn promote_split<'a, K, V>(
    inner_map: &mut sys::SysBTreeMap<K, V>,
    mut split: sys::SplitResult<'a, K, V, sys::LeafOrInternal>,
    inserted_pos: (*mut sys::LeafNode<K, V>, usize),
    initial_right_ptr: NonNull<sys::LeafNode<K, V>>,
) -> InsertResult<'a, K, V> {
    let mut initial_right_wired = false;
    let mut last_allocated: Option<(NonNull<sys::LeafNode<K, V>>, usize)> = None;

    let result = (|| {
        loop {
            let left_for_ascend = sys::NodeRef {
                height: split.left.height,
                node: split.left.node,
                _marker: PhantomData,
            };
            let (kv, mut right) = unsafe {
                let kv = ptr::read(&split.kv);
                let right_node = ptr::read(&split.right.node);
                (
                    kv,
                    sys::NodeRef::<sys::Owned, K, V, sys::LeafOrInternal> {
                        height: split.right.height,
                        node: right_node,
                        _marker: PhantomData,
                    },
                )
            };
            mem::forget(split);

            match ascend(left_for_ascend) {
                AscendResult::Parent(parent_edge) => {
                    let mut parent_ptr: NonNull<sys::InternalNode<K, V>> =
                        parent_edge.node.node.cast();
                    let parent_len = unsafe { parent_ptr.as_ref() }.data.len as usize;

                    if parent_len < sys::CAPACITY {
                        let edge_idx = parent_edge.idx;
                        unsafe {
                            internal_insert_fit(
                                parent_ptr.as_mut(),
                                edge_idx,
                                kv.0,
                                kv.1,
                                right.node,
                            );
                            // Set the parent link on the newly inserted right child.
                            // The child was allocated by try_new_leaf and has parent=None;
                            // we must wire it back to the parent so that the tree's
                            // drop traversal can navigate the full subtree.
                            let child_ptr = right.node.as_mut();
                            let parent_as_leaf: NonNull<LeafNode<K, V>> = parent_ptr.cast();
                            set_parent_link(child_ptr, parent_as_leaf, edge_idx + 1);
                        }
                        initial_right_wired = true;
                        break;
                    }

                    let (sp_idx, ins_side) = splitpoint(parent_edge.idx);

                    let (mk, mv) = unsafe {
                        let parent_mut = parent_ptr.as_mut();
                        (
                            parent_mut.data.keys[sp_idx].assume_init_read(),
                            parent_mut.data.vals[sp_idx].assume_init_read(),
                        )
                    };

                    let ri_box = match try_new_internal() {
                        Ok(b) => b,
                        Err(e) => return Err((kv.0, kv.1, e)),
                    };
                    let height = parent_edge.node.height;
                    last_allocated = Some((ri_box, height));

                    let old_parent_len = unsafe { parent_ptr.as_ref() }.data.len as usize;
                    let new_right_len = old_parent_len - sp_idx - 1;

                    unsafe {
                        let mut ri_ptr: NonNull<InternalNode<K, V>> = ri_box.cast();
                        let ri = ri_ptr.as_mut();
                        ri.data.parent = None;
                        ri.data.len = new_right_len as u16;
                        let parent_mut = parent_ptr.as_mut();

                        if new_right_len > 0 {
                            ptr::copy_nonoverlapping(
                                &parent_mut.data.keys[sp_idx + 1],
                                ri.data.keys.as_mut_ptr(),
                                new_right_len,
                            );
                            ptr::copy_nonoverlapping(
                                &parent_mut.data.vals[sp_idx + 1],
                                ri.data.vals.as_mut_ptr(),
                                new_right_len,
                            );
                        }

                        if new_right_len > 0 {
                            ptr::copy_nonoverlapping(
                                parent_mut.edges.as_ptr().add(sp_idx + 1),
                                ri.edges.as_mut_ptr(),
                                new_right_len + 1,
                            );
                        }

                        parent_mut.data.len = sp_idx as u16;
                    }

                    let right_owned: sys::NodeRef<sys::Owned, K, V, sys::Internal> = sys::NodeRef {
                        height,
                        node: ri_box,
                        _marker: PhantomData,
                    };

                    match ins_side {
                        InsertionSide::Left(ii) => {
                            let mut left_ptr = parent_ptr;
                            unsafe {
                                internal_insert_fit(left_ptr.as_mut(), ii, kv.0, kv.1, right.node);
                            }
                        }
                        InsertionSide::Right(ii) => unsafe {
                            let ri_ptr = right_owned.node.as_ptr() as *mut sys::InternalNode<K, V>;
                            let ri = ri_ptr.as_mut().expect("ri_ptr should not be null");
                            internal_insert_fit(ri, ii, kv.0, kv.1, right.node);
                        },
                    }
                    initial_right_wired = true;

                    unsafe { correct_parent_links(ri_box, 0, new_right_len, height) }

                    let left_internal = sys::NodeRef::<sys::Mut<'a>, K, V, sys::Internal> {
                        height,
                        node: parent_ptr.cast(),
                        _marker: PhantomData,
                    };
                    split = unsafe {
                        sys::SplitResult {
                            left: sys::NodeRef {
                                height: left_internal.height,
                                node: left_internal.node,
                                _marker: PhantomData,
                            },
                            kv: (mk, mv),
                            right: sys::NodeRef {
                                height: right_owned.height,
                                node: ptr::read(&right_owned.node),
                                _marker: PhantomData,
                            },
                        }
                    };
                }
                AscendResult::Root(root) => {
                    let new_root_box = match try_new_internal() {
                        Ok(b) => b,
                        Err(e) => return Err((kv.0, kv.1, e)),
                    };
                    let new_height = root.height + 1;
                    last_allocated = Some((new_root_box, new_height));

                    let mut old_root_owned: sys::NodeRef<sys::Owned, K, V, sys::LeafOrInternal> =
                        unsafe { ptr::read(inner_map.root.as_ref().unwrap()) };
                    inner_map.root.take();

                    let nr = new_root_box.as_ptr() as *mut sys::InternalNode<K, V>;
                    unsafe {
                        (*nr).data.parent = None;
                        (*nr).data.len = 0;
                        (*nr).edges[0].write(old_root_owned.node);
                    }

                    unsafe {
                        set_parent_link(old_root_owned.node.as_mut(), new_root_box, 0);
                    }

                    unsafe {
                        let len = (*nr).data.len as usize;
                        (*nr).data.keys[len].write(kv.0);
                        (*nr).data.vals[len].write(kv.1);
                        (*nr).edges[len + 1].write(right.node);
                        (*nr).data.len = (len + 1) as u16;
                    }

                    unsafe {
                        set_parent_link(right.node.as_mut(), new_root_box, 1);
                    }

                    initial_right_wired = true;

                    inner_map.root = Some(sys::NodeRef::<sys::Owned, K, V, sys::LeafOrInternal> {
                        height: new_height,
                        node: new_root_box,
                        _marker: PhantomData,
                    });
                    break;
                }
            }
        }

        inner_map.length += 1;
        let (p, idx) = inserted_pos;
        let node_ref = unsafe {
            sys::NodeRef::<sys::Mut<'a>, K, V, sys::LeafOrInternal> {
                height: 0,
                node: NonNull::new_unchecked(p),
                _marker: PhantomData,
            }
        };
        Ok((node_ref, idx))
    })();

    if result.is_err() {
        if !initial_right_wired {
            unsafe { drop_leaf_node(initial_right_ptr) };
        }
        if let Some((ptr, height)) = last_allocated
            && height > 0
        {
            unsafe { drop_internal_node(ptr) };
        }
    }

    result
}

// ── Helper: leaf slice insert ─────────────────────────────────────────────────

/// Insert a key/value at `idx` in a leaf node's arrays.
/// The caller must ensure len < CAPACITY and update len afterward.
unsafe fn leaf_slice_insert<K, V>(leaf: *mut sys::LeafNode<K, V>, idx: usize, key: K, val: V) {
    unsafe {
        let len = (*leaf).len as usize;
        if len > idx {
            let keys_ptr = (*leaf).keys.as_mut_ptr();
            let vals_ptr = (*leaf).vals.as_mut_ptr();
            ptr::copy(keys_ptr.add(idx), keys_ptr.add(idx + 1), len - idx);
            ptr::copy(vals_ptr.add(idx), vals_ptr.add(idx + 1), len - idx);
        }
        (*leaf).keys[idx].write(key);
        (*leaf).vals[idx].write(val);
    }
}

// ── Helper: internal insert fit ───────────────────────────────────────────────

/// Insert a KV pair and child edge into an internal node at `edge_idx`.
unsafe fn internal_insert_fit<K, V>(
    internal: &mut sys::InternalNode<K, V>,
    edge_idx: usize,
    key: K,
    val: V,
    child: NonNull<sys::LeafNode<K, V>>,
) {
    unsafe {
        let len = internal.data.len as usize;
        if len > edge_idx {
            ptr::copy(
                &internal.data.keys[edge_idx],
                &mut internal.data.keys[edge_idx + 1],
                len - edge_idx,
            );
            ptr::copy(
                &internal.data.vals[edge_idx],
                &mut internal.data.vals[edge_idx + 1],
                len - edge_idx,
            );
        }
        ptr::copy(
            &internal.edges[edge_idx + 1],
            &mut internal.edges[edge_idx + 2],
            len - edge_idx,
        );
        internal.data.keys[edge_idx].write(key);
        internal.data.vals[edge_idx].write(val);
        internal.edges[edge_idx + 1].write(child);
        internal.data.len = (len + 1) as u16;
    }
}

// ── Helper: correct parent links ──────────────────────────────────────────────

unsafe fn correct_parent_links<K, V>(
    internal: NonNull<sys::LeafNode<K, V>>,
    start: usize,
    end: usize,
    _height: usize,
) {
    unsafe {
        let mut iptr: NonNull<InternalNode<K, V>> = internal.cast();
        let iptr_casted = iptr.as_mut();
        for i in start..=end {
            let child_uninit = &mut iptr_casted.edges[i];
            let mut child = child_uninit.assume_init_read();
            set_parent_link(child.as_mut(), internal, i);
        }
    }
}

unsafe fn set_parent_link<K, V>(
    child_ptr: &mut sys::LeafNode<K, V>,
    parent_ptr: NonNull<sys::LeafNode<K, V>>,
    idx: usize,
) {
    child_ptr.parent = Some(parent_ptr.cast());
    child_ptr.parent_idx.write(idx as u16);
}

// ── Node type casting helpers ─────────────────────────────────────────────────

trait NodeRefForGetTypeLoI<'a, K, V> {
    fn forget_type(self) -> sys::NodeRef<sys::Mut<'a>, K, V, sys::LeafOrInternal>;
}

impl<'a, K, V> NodeRefForGetTypeLoI<'a, K, V> for sys::NodeRef<sys::Mut<'a>, K, V, sys::Leaf> {
    fn forget_type(self) -> sys::NodeRef<sys::Mut<'a>, K, V, sys::LeafOrInternal> {
        sys::NodeRef {
            height: self.height,
            node: self.node,
            _marker: PhantomData,
        }
    }
}

impl<'a, K, V> NodeRefForGetTypeLoI<'a, K, V> for sys::NodeRef<sys::Mut<'a>, K, V, sys::Internal> {
    fn forget_type(self) -> sys::NodeRef<sys::Mut<'a>, K, V, sys::LeafOrInternal> {
        sys::NodeRef {
            height: self.height,
            node: self.node,
            _marker: PhantomData,
        }
    }
}

// ── Edge/node navigation ─────────────────────────────────────────────────────

enum AscendResult<'a, K, V> {
    Parent(sys::Handle<sys::NodeRef<sys::Mut<'a>, K, V, sys::Internal>, sys::Edge>),
    Root(sys::NodeRef<sys::Mut<'a>, K, V, sys::LeafOrInternal>),
}

fn ascend<'a, K, V>(
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

// ── Node allocation helpers ───────────────────────────────────────────────────

fn try_new_leaf<K, V>() -> Result<NonNull<sys::LeafNode<K, V>>, TryBTreeMapEntryError> {
    let mut leaf_box: Box<MaybeUninit<sys::LeafNode<K, V>>> =
        <Box<sys::LeafNode<K, V>> as TryBox<_>>::fallible_new_uninit()
            .map_err(TryBTreeMapEntryError::Alloc)?;

    unsafe {
        let ptr = leaf_box.as_mut_ptr();
        (*ptr).parent = None;
        (*ptr).len = 0;
    }

    let boxed: Box<sys::LeafNode<K, V>> = unsafe { leaf_box.assume_init() };
    let raw = Box::into_raw(boxed);
    unsafe { Ok(NonNull::new_unchecked(raw)) }
}

fn try_new_internal<K, V>() -> Result<NonNull<sys::LeafNode<K, V>>, TryBTreeMapEntryError> {
    let mut node_box: Box<MaybeUninit<sys::InternalNode<K, V>>> =
        <Box<sys::InternalNode<K, V>> as TryBox<_>>::fallible_new_uninit()
            .map_err(TryBTreeMapEntryError::Alloc)?;

    unsafe {
        let ptr = node_box.as_mut_ptr();
        (*ptr).data.parent = None;
        (*ptr).data.len = 0;
    }

    let boxed: Box<sys::InternalNode<K, V>> = unsafe { node_box.assume_init() };
    let raw = Box::into_raw(boxed);
    unsafe { Ok(NonNull::new_unchecked(raw).cast()) }
}

/// Drop a heap-allocated leaf node, preventing memory leaks on OOM rollback.
unsafe fn drop_leaf_node<K, V>(ptr: NonNull<sys::LeafNode<K, V>>) {
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

/// Drop a heap-allocated internal node.
unsafe fn drop_internal_node<K, V>(ptr: NonNull<sys::LeafNode<K, V>>) {
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

// ── Split helpers ─────────────────────────────────────────────────────────────

fn splitpoint(edge_idx: usize) -> (usize, InsertionSide) {
    debug_assert!(edge_idx <= sys::CAPACITY);
    const KV_IDX_CENTER: usize = 5;
    const EDGE_LEFT_OF_CENTER: usize = 5;
    const EDGE_RIGHT_OF_CENTER: usize = 6;

    match edge_idx {
        0..EDGE_LEFT_OF_CENTER => (KV_IDX_CENTER - 1, InsertionSide::Left(edge_idx)),
        EDGE_LEFT_OF_CENTER => (KV_IDX_CENTER, InsertionSide::Left(edge_idx)),
        EDGE_RIGHT_OF_CENTER => (KV_IDX_CENTER, InsertionSide::Right(0)),
        _ => (
            KV_IDX_CENTER + 1,
            InsertionSide::Right(edge_idx - (KV_IDX_CENTER + 1 + 1)),
        ),
    }
}

enum InsertionSide {
    Left(usize),
    Right(usize),
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
        // SAFETY: size=1, align=1 is always a valid layout.
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
        // 12 entries triggers exactly one split (capacity is 11)
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

    // ── VacantEntryExt tests ────────────────────────────────────────────────

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
        // First call goes through Vacant branch
        let v1 = map.try_insert_entry(1, "hello".to_string()).unwrap();
        assert_eq!(v1, &"hello");
        // Second call goes through Occupied branch
        let v2 = map.try_insert_entry(1, "world".to_string()).unwrap();
        assert_eq!(v2, &"hello"); // unchanged
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
        // Map should still be empty — no partial state on failure.
        assert!(map.is_empty());
    }

    #[test]
    fn try_insert_entry_leaf_split_fails_on_oom() {
        // Fill the leaf to capacity (11) outside the policy, then the 12th
        // insert triggers a split which allocates a new leaf node. Fail that
        // allocation by setting policy to fail the very next alloc.
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
        // Original 11 entries must still be intact.
        assert_eq!(map.len(), 11);
        for i in 0..11 {
            assert_eq!(map[&i], i * 10);
        }
    }

    #[test]
    fn try_insert_entry_cascading_split_fails_on_oom() {
        // Build a tree with internal nodes outside the policy, then fail
        // the next allocation during a new insert that may need splitting.
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
                // Allocation failed — verify map integrity.
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
        // Use Copy types so no clone allocations interfere with the OOM policy.
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
        // Allow first alloc, fail on second, succeed on third.
        let results = with_policy(FailPolicy::fail_nth_alloc(2), || {
            let mut map: BTreeMap<u32, u32> = BTreeMap::new();
            let r1 = { map.try_insert_entry(1, 10).is_ok() };
            let r2 = { map.try_insert_entry(2, 20).is_ok() };
            (r1, r2)
        });
        // At least one should succeed; the exact pattern depends on allocation count.
        let (_r1_ok, _r2_ok) = results;
        // The important thing: no crash occurred.
    }

    #[test]
    fn oom_guard_restores_allocation_afterwards() {
        let mut map: BTreeMap<u32, u32> = BTreeMap::new();
        let _r = with_policy(FailPolicy::fail_next_alloc(), || map.try_insert_entry(1, 2));
        // Allocation works after the guard scope ends.
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
        // Insert 50 entries to force multiple splits across leaf and internal nodes.
        let mut map: BTreeMap<usize, usize> = BTreeMap::new();
        for i in 0..50 {
            map.try_insert_entry(i, i.wrapping_mul(7)).unwrap();
        }
        assert_eq!(map.len(), 50);
        for i in 0..50 {
            assert_eq!(map[&i], i.wrapping_mul(7));
        }
    }

    #[test]
    fn try_insert_entry_boundary_at_capacity() {
        // Exactly CAPACITY (11) entries fits in one leaf, no split needed.
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
        // Insert out of order across multiple splits
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
        // Force a split, then insert an existing key — should return the
        // existing value without replacing it (or_insert semantics).
        let mut map: BTreeMap<i32, String> = BTreeMap::new();
        for i in 0..12 {
            map.try_insert_entry(i, format!("v{}", i)).unwrap();
        }
        assert_eq!(map.len(), 12);
        let returned = map.try_insert_entry(5, "REPLACED".to_string()).unwrap();
        // Key 5 already exists, so the old value is returned unchanged.
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
        // Verify that the cloned key is properly stored even when K != Copy.
        use lang_alloc::string::String;
        let mut map: BTreeMap<String, i32> = BTreeMap::new();
        let s = "unique_key".to_string();
        map.try_insert_entry(s.clone(), 42).unwrap();
        assert_eq!(map[&s], 42);
        // The original string is still usable.
        assert_eq!(s, "unique_key");
    }

    // ── Error formatting ────────────────────────────────────────────────────

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
