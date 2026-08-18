//! Iterator types for [`ConcurrentHashMap`].
//!
//! Each iterator lazily initializes an [`Arc`] to a shard lock guard on first
//! access to each shard. The Arc is cloned into every yielded [`RefMulti`] /
//! [`RefMutMulti`] so that both the iterator and all outstanding guards keep
//! the lock alive simultaneously.
//!
//! Because `Arc::new` (initialization) and `Arc::clone` (cloning into yielded
//! items) are fallible, iteration returns [`Result`] items.

use lang_core::fmt;
use lang_core::hash::{BuildHasher, Hash};
use lang_std::hash::RandomState;
use lang_std::sync::Arc;

use hashbrown::raw::{Bucket, RawIter, RawTable};
use parking_lot::{RwLockReadGuard, RwLockWriteGuard};

use crate::alloc::AllocError;
use crate::recovery::Stallable;
use crate::std::arc::TryArc;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_fmt::{TryDebug, helpers::FormatterExt};

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

impl fmt::Display for IterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "iteration failed: heap allocation error"),
            Self::Clone(e) => write!(f, "iteration failed: {}", e),
        }
    }
}

impl TryDebug for IterError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => f
                .try_debug_struct("IterError::Alloc")
                .field("0", e)
                .finish(),
            Self::Clone(e) => f
                .try_debug_struct("IterError::Clone")
                .field("0", e)
                .finish(),
        }
    }
}

struct LockedIter<'a, K, V> {
    arc_guard: Arc<RwLockReadGuard<'a, RawTable<(K, V)>>>,
    raw_iter: RawIter<(K, V)>,
    /// Bucket already popped from `raw_iter` but not yet yielded because
    /// `Arc::try_clone` failed. Stored here so retrying `next()` re-attempts
    /// the clone instead of advancing past this entry.
    pending_bucket: Option<Bucket<(K, V)>>,
}
struct LockedIterMut<'a, K, V> {
    arc_guard: Arc<RwLockWriteGuard<'a, RawTable<(K, V)>>>,
    raw_iter: RawIter<(K, V)>,
    /// Same as [`LockedIter::pending_bucket`] but for mutable iteration.
    pending_bucket: Option<Bucket<(K, V)>>,
}

// ── Iter (immutable) ────────────────────────────────────────────────────────────

/// An immutable iterator over a [`ConcurrentHashMap`].
///
/// Produced by [`ConcurrentHashMap::iter`]. Lazily acquires a read lock per
/// shard via an `Arc`, then streams through buckets cloning the Arc into each
/// yielded [`RefMulti`].
pub struct Iter<'a, K, V, S = RandomState> {
    map: &'a ConcurrentHashMap<K, V, S>,
    /// Index of the shard we are currently draining.
    shard_idx: usize,
    /// Lazily-initialized: `(Arc<read_guard>, raw_table_iterator)`.
    /// `None` means we haven't locked this shard yet, or it was exhausted.
    current: Option<LockedIter<'a, K, V>>,
    /// When true, automatically discard pending items after emitting an error.
    auto_unstall: bool,
}

impl<'a, K: Eq + Hash, V, S: BuildHasher> Iter<'a, K, V, S> {
    pub(crate) fn new(map: &'a ConcurrentHashMap<K, V, S>) -> Self {
        Self {
            map,
            shard_idx: 0,
            current: None,
            auto_unstall: false,
        }
    }

    /// Advance to the next shard. Safe: `shard_idx` is always kept below the shard count.
    fn advance_shard(&mut self) {
        debug_assert!(self.shard_idx < self.map.shard_count());
        let idx = self
            .shard_idx
            .checked_add(1)
            .expect("shard_idx below shard count");
        self.shard_idx = idx;
    }
}

impl<'a, K, V, S: BuildHasher> Iterator for Iter<'a, K, V, S>
where
    K: Eq + Hash,
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
                    self.advance_shard();
                    continue;
                }
                let arc_guard =
                    match <Arc<RwLockReadGuard<'a, RawTable<(K, V)>>> as TryArc<_>>::fallible_new(
                        guard,
                    ) {
                        Ok(g) => g,
                        Err(e) => {
                            if self.auto_unstall {
                                // Skip this shard — it couldn't be locked, move on.
                                self.advance_shard();
                                continue;
                            }
                            // Stall: do not advance shard_idx so that retrying next()
                            // re-attempts this same shard. The guard is dropped here
                            // and will be reacquired on the next call.
                            return Some(Err(IterError::Alloc(e)));
                        }
                    };
                // SAFETY: arc_guard holds the read lock so the table is stable.
                let iter = unsafe { arc_guard.iter() };
                self.current = Some(LockedIter {
                    arc_guard,
                    raw_iter: iter,
                    pending_bucket: None,
                });
            }

            // Try to yield the next bucket from the current shard.
            if let Some(LockedIter {
                arc_guard,
                raw_iter,
                pending_bucket,
            }) = &mut self.current
            {
                let bucket = match pending_bucket.take() {
                    Some(b) => b,
                    None => match raw_iter.next() {
                        Some(b) => b,
                        None => {
                            // Shard exhausted.
                            self.current = None;
                            self.advance_shard();
                            continue;
                        }
                    },
                };
                let arc_for_item = match arc_guard.try_clone() {
                    Ok(g) => g,
                    Err(e) => {
                        if self.auto_unstall {
                            // Discard this bucket and advance to the next entry.
                            continue;
                        }
                        // Stall: stash the bucket so retrying next() re-attempts
                        // the clone instead of advancing past this entry.
                        *pending_bucket = Some(bucket);
                        return Some(Err(IterError::Clone(e)));
                    }
                };
                let kv = unsafe { bucket.as_ref() };
                return Some(Ok(unsafe {
                    RefMulti::new(arc_for_item, &kv.0 as *const K, &kv.1 as *const V)
                }));
            }
        }
    }
}

impl<'a, K, V, S: BuildHasher> Stallable for Iter<'a, K, V, S>
where
    K: Eq + Hash,
{
    fn unstall(&mut self) -> bool {
        if let Some(ref mut li) = self.current {
            li.pending_bucket.take().is_some()
        } else {
            // Stalled at shard-level Arc allocation — "unstalling" means skipping
            // this shard so iteration can proceed. We can't tell for sure whether
            // we're actually stalled here (we might just be between shards), but
            // advancing is harmless in that case.
            if self.shard_idx < self.map.shard_count() {
                self.advance_shard();
                true
            } else {
                false
            }
        }
    }

    fn set_auto_unstall(&mut self, auto: bool) {
        self.auto_unstall = auto;
    }
}

// ── IterMut (mutable) ───────────────────────────────────────────────────────────

/// A mutable iterator over a [`ConcurrentHashMap`].
///
/// Produced by [`ConcurrentHashMap::iter_mut`]. Lazily acquires a write lock
/// per shard via an `Arc`, then streams through buckets cloning the Arc into
/// each yielded [`RefMutMulti`].
pub struct IterMut<'a, K, V, S = RandomState> {
    map: &'a ConcurrentHashMap<K, V, S>,
    /// Index of the shard we are currently draining.
    shard_idx: usize,
    /// Lazily-initialized: `(Arc<write_guard>, raw_table_iterator)`.
    current: Option<LockedIterMut<'a, K, V>>,
    /// When true, automatically discard pending items after emitting an error.
    auto_unstall: bool,
}

impl<'a, K: Eq + Hash, V, S: BuildHasher> IterMut<'a, K, V, S> {
    pub(crate) fn new(map: &'a ConcurrentHashMap<K, V, S>) -> Self {
        Self {
            map,
            shard_idx: 0,
            current: None,
            auto_unstall: false,
        }
    }

    /// Advance to the next shard. Safe: `shard_idx` is always kept below the shard count.
    fn advance_shard(&mut self) {
        debug_assert!(self.shard_idx < self.map.shard_count());
        let idx = self
            .shard_idx
            .checked_add(1)
            .expect("shard_idx below shard count");
        self.shard_idx = idx;
    }
}

impl<'a, K, V, S: BuildHasher> Iterator for IterMut<'a, K, V, S>
where
    K: Eq + Hash,
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
                    self.advance_shard();
                    continue;
                }
                let arc_guard =
                    match <Arc<RwLockWriteGuard<'a, RawTable<(K, V)>>> as TryArc<_>>::fallible_new(
                        guard,
                    ) {
                        Ok(g) => g,
                        Err(e) => {
                            if self.auto_unstall {
                                // Skip this shard — it couldn't be locked, move on.
                                self.advance_shard();
                                continue;
                            }
                            // Stall: do not advance shard_idx so that retrying next()
                            // re-attempts this same shard. The guard is dropped here
                            // and will be reacquired on the next call.
                            return Some(Err(IterError::Alloc(e)));
                        }
                    };
                // SAFETY: arc_guard holds the write lock so the table is stable.
                let iter = unsafe { arc_guard.iter() };
                self.current = Some(LockedIterMut {
                    arc_guard,
                    raw_iter: iter,
                    pending_bucket: None,
                });
            }

            // Try to yield the next bucket from the current shard.
            if let Some(LockedIterMut {
                arc_guard,
                raw_iter,
                pending_bucket,
            }) = &mut self.current
            {
                let bucket = match pending_bucket.take() {
                    Some(b) => b,
                    None => match raw_iter.next() {
                        Some(b) => b,
                        None => {
                            // Shard exhausted.
                            self.current = None;
                            self.advance_shard();
                            continue;
                        }
                    },
                };
                let arc_for_item = match arc_guard.try_clone() {
                    Ok(g) => g,
                    Err(e) => {
                        if self.auto_unstall {
                            // Discard this bucket and advance to the next entry.
                            continue;
                        }
                        // Stall: stash the bucket so retrying next() re-attempts
                        // the clone instead of advancing past this entry.
                        *pending_bucket = Some(bucket);
                        return Some(Err(IterError::Clone(e)));
                    }
                };
                let kv = unsafe { bucket.as_mut() };
                return Some(Ok(unsafe {
                    RefMutMulti::new(arc_for_item, &kv.0, &mut kv.1)
                }));
            }
        }
    }
}

impl<'a, K, V, S: BuildHasher> Stallable for IterMut<'a, K, V, S>
where
    K: Eq + Hash,
{
    fn unstall(&mut self) -> bool {
        if let Some(ref mut li) = self.current {
            li.pending_bucket.take().is_some()
        } else {
            if self.shard_idx < self.map.shard_count() {
                self.advance_shard();
                true
            } else {
                false
            }
        }
    }

    fn set_auto_unstall(&mut self, auto: bool) {
        self.auto_unstall = auto;
    }
}
