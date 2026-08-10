//! Iterator types for [`ConcurrentHashMap`].
//!
//! Each iterator lazily initializes an [`Arc`] to a shard lock guard on first
//! access to each shard. The Arc is cloned into every yielded [`RefMulti`] /
//! [`RefMutMulti`] so that both the iterator and all outstanding guards keep
//! the lock alive simultaneously.
//!
//! Because `Arc::new` (initialization) and `Arc::clone` (cloning into yielded
//! items) are fallible, iteration returns [`Result`] items.

use core::hash::{BuildHasher, Hash};
use std::sync::Arc;

use hashbrown::raw::{RawIter, RawTable};
use parking_lot::{RwLockReadGuard, RwLockWriteGuard};

use crate::alloc::AllocError;
use crate::arc::TryArc;
use crate::try_clone::{TryClone, TryCloneError};

use super::map::ConcurrentHashMap;
use super::refs::{RefMulti, RefMutMulti};

// ── Error type ──────────────────────────────────────────────────────────────────

/// Error returned by [`Iter::next`] or [`IterMut::next`].
#[derive(Debug)]
pub enum IterError {
    /// Failed to allocate the `Arc` wrapper around a shard guard.
    Alloc(AllocError),
    /// Failed to clone the `Arc` guard when yielding an item (refcount overflow).
    Clone(TryCloneError),
}

impl core::fmt::Display for IterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "iteration failed: heap allocation error"),
            Self::Clone(e) => write!(f, "iteration failed: {}", e),
        }
    }
}

struct LockedIter<'a, K, V>(Arc<RwLockReadGuard<'a, RawTable<(K, V)>>>, RawIter<(K, V)>);
struct LockedIterMut<'a, K, V>(Arc<RwLockWriteGuard<'a, RawTable<(K, V)>>>, RawIter<(K, V)>);

// ── Iter (immutable) ────────────────────────────────────────────────────────────

/// An immutable iterator over a [`ConcurrentHashMap`].
///
/// Produced by [`ConcurrentHashMap::iter`]. Lazily acquires a read lock per
/// shard via an `Arc`, then streams through buckets cloning the Arc into each
/// yielded [`RefMulti`].
pub struct Iter<'a, K, V, S = std::hash::RandomState> {
    map: &'a ConcurrentHashMap<K, V, S>,
    /// Index of the shard we are currently draining.
    shard_idx: usize,
    /// Lazily-initialized: `(Arc<read_guard>, raw_table_iterator)`.
    /// `None` means we haven't locked this shard yet, or it was exhausted.
    current: Option<LockedIter<'a, K, V>>,
}

impl<'a, K, V, S> Iter<'a, K, V, S> {
    pub(crate) fn new(map: &'a ConcurrentHashMap<K, V, S>) -> Self {
        Self {
            map,
            shard_idx: 0,
            current: None,
        }
    }
}

impl<'a, K, V, S: BuildHasher> Iterator for Iter<'a, K, V, S>
where
    K: Eq + Hash + 'static,
    V: 'static,
{
    type Item = Result<RefMulti<'a, K, V>, IterError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Lazy init: bounds check + lock acquisition only when moving to a new shard.
            if self.current.is_none() {
                if self.shard_idx >= self.map.shard_count() {
                    return None;
                }
                let shard = &self.map.get_shards()[self.shard_idx];
                let guard = shard.read_table();
                if guard.is_empty() {
                    drop(guard);
                    self.shard_idx += 1;
                    continue;
                }
                let arc_guard =
                    match <Arc<RwLockReadGuard<'a, RawTable<(K, V)>>> as TryArc<_>>::fallible_new(
                        guard,
                    ) {
                        Ok(g) => g,
                        Err(e) => {
                            self.shard_idx += 1;
                            return Some(Err(IterError::Alloc(e)));
                        }
                    };
                // SAFETY: arc_guard holds the read lock so the table is stable.
                let iter = unsafe { arc_guard.iter() };
                self.current = Some(LockedIter(arc_guard, iter));
            }

            // Try to yield the next bucket from the current shard.
            if let Some(LockedIter(arc_guard, raw_iter)) = &mut self.current {
                if let Some(bucket) = raw_iter.next() {
                    let arc_for_item = match arc_guard.try_clone() {
                        Ok(g) => g,
                        Err(e) => return Some(Err(IterError::Clone(e))),
                    };
                    let kv = unsafe { bucket.as_ref() };
                    return Some(Ok(unsafe {
                        RefMulti::new(arc_for_item, &kv.0 as *const K, &kv.1 as *const V)
                    }));
                } else {
                    // Shard exhausted.
                    self.current = None;
                    self.shard_idx += 1;
                }
            }
        }
    }
}

// ── IterMut (mutable) ───────────────────────────────────────────────────────────

/// A mutable iterator over a [`ConcurrentHashMap`].
///
/// Produced by [`ConcurrentHashMap::iter_mut`]. Lazily acquires a write lock
/// per shard via an `Arc`, then streams through buckets cloning the Arc into
/// each yielded [`RefMutMulti`].
pub struct IterMut<'a, K, V, S = std::hash::RandomState> {
    map: &'a ConcurrentHashMap<K, V, S>,
    /// Index of the shard we are currently draining.
    shard_idx: usize,
    /// Lazily-initialized: `(Arc<write_guard>, raw_table_iterator)`.
    current: Option<LockedIterMut<'a, K, V>>,
}

impl<'a, K, V, S> IterMut<'a, K, V, S> {
    pub(crate) fn new(map: &'a ConcurrentHashMap<K, V, S>) -> Self {
        Self {
            map,
            shard_idx: 0,
            current: None,
        }
    }
}

impl<'a, K, V, S: BuildHasher> Iterator for IterMut<'a, K, V, S>
where
    K: Eq + Hash + 'static,
    V: 'static,
{
    type Item = Result<RefMutMulti<'a, K, V>, IterError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Lazy init: bounds check + lock acquisition only when moving to a new shard.
            if self.current.is_none() {
                if self.shard_idx >= self.map.shard_count() {
                    return None;
                }
                let shard = &self.map.get_shards()[self.shard_idx];
                let guard = shard.write_table();
                if guard.is_empty() {
                    drop(guard);
                    self.shard_idx += 1;
                    continue;
                }
                let arc_guard =
                    match <Arc<RwLockWriteGuard<'a, RawTable<(K, V)>>> as TryArc<_>>::fallible_new(
                        guard,
                    ) {
                        Ok(g) => g,
                        Err(e) => {
                            self.shard_idx += 1;
                            return Some(Err(IterError::Alloc(e)));
                        }
                    };
                // SAFETY: arc_guard holds the write lock so the table is stable.
                let iter = unsafe { arc_guard.iter() };
                self.current = Some(LockedIterMut(arc_guard, iter));
            }

            // Try to yield the next bucket from the current shard.
            if let Some(LockedIterMut(arc_guard, raw_iter)) = &mut self.current {
                if let Some(bucket) = raw_iter.next() {
                    let arc_for_item = match arc_guard.try_clone() {
                        Ok(g) => g,
                        Err(e) => return Some(Err(IterError::Clone(e))),
                    };
                    let kv = unsafe { bucket.as_mut() };
                    return Some(Ok(unsafe {
                        RefMutMulti::new(arc_for_item, &kv.0, &mut kv.1)
                    }));
                } else {
                    // Shard exhausted.
                    self.current = None;
                    self.shard_idx += 1;
                }
            }
        }
    }
}
