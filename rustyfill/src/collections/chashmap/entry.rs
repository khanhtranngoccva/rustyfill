//! Entry API for [`ConcurrentHashMap`](super::ConcurrentHashMap).

use hashbrown::raw::{Bucket, InsertSlot, RawTable};
use lang_core::hash::Hash;
use lang_core::mem;
use parking_lot::RwLockWriteGuard;

use super::refs::RefMut;

// ── Entry ──────────────────────────────────────────────────────────────────────

/// An entry in the concurrent hash map, either occupied or vacant.
pub enum Entry<'a, K, V> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K: Eq + Hash + 'a, V> Entry<'a, K, V> {
    /// Get the key.
    pub fn key(&self) -> &K {
        match self {
            Self::Occupied(e) => e.key(),
            Self::Vacant(e) => e.key(),
        }
    }

    /// Insert the value regardless of occupancy, returning a guard that keeps the lock alive.
    pub fn insert(self, value: V) -> RefMut<'a, K, V> {
        match self {
            Self::Occupied(mut e) => {
                e.insert(value);
                e.into_ref()
            }
            Self::Vacant(e) => e.insert(value),
        }
    }

    /// Insert if vacant, otherwise return a guard to the existing value.
    pub fn or_insert(self, value: V) -> RefMut<'a, K, V> {
        match self {
            Self::Occupied(e) => e.into_ref(),
            Self::Vacant(e) => e.insert(value),
        }
    }

    /// Insert using a closure if vacant.
    pub fn or_insert_with(self, f: impl FnOnce() -> V) -> RefMut<'a, K, V> {
        match self {
            Self::Occupied(e) => e.into_ref(),
            Self::Vacant(e) => e.insert(f()),
        }
    }

    /// Apply a function to the value if occupied.
    pub fn and_modify(self, f: impl FnOnce(&mut V)) -> Self {
        match self {
            Self::Occupied(mut e) => {
                f(e.get_mut());
                Self::Occupied(e)
            }
            Self::Vacant(e) => Self::Vacant(e),
        }
    }
}

// ── OccupiedEntry ──────────────────────────────────────────────────────────────

/// An occupied entry — the key already exists.
pub struct OccupiedEntry<'a, K, V> {
    pub(crate) guard: RwLockWriteGuard<'a, RawTable<(K, V)>>,
    pub(crate) bucket: Bucket<(K, V)>,
}

unsafe impl<K: Sync, V: Sync> Send for OccupiedEntry<'_, K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for OccupiedEntry<'_, K, V> {}

impl<'a, K: Eq + Hash + 'a, V> OccupiedEntry<'a, K, V> {
    /// Returns a reference to the key.
    pub fn key(&self) -> &K {
        unsafe { &self.bucket.as_ref().0 }
    }

    /// Returns a reference to the value.
    pub fn get(&self) -> &V {
        unsafe { &self.bucket.as_ref().1 }
    }

    /// Returns a mutable reference to the value.
    pub fn get_mut(&mut self) -> &mut V {
        unsafe { &mut self.bucket.as_mut().1 }
    }

    /// Replaces the value and returns the old one.
    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }

    /// Removes the entry and returns the value.
    pub fn remove(mut self) -> V {
        let ((_, v), _) = unsafe { self.guard.remove(self.bucket) };
        v
    }

    /// Removes the entry and returns the key-value pair.
    pub fn remove_entry(mut self) -> (K, V) {
        let ((k, v), _) = unsafe { self.guard.remove(self.bucket) };
        (k, v)
    }

    /// Consumes the entry and returns a guard that keeps the shard lock alive.
    pub fn into_ref(self) -> RefMut<'a, K, V> {
        unsafe {
            let kv = self.bucket.as_mut();
            RefMut::new(self.guard, &kv.0, &mut kv.1)
        }
    }
}

// ── VacantEntry ────────────────────────────────────────────────────────────────

/// A vacant entry — the key does not yet exist.
pub struct VacantEntry<'a, K, V> {
    pub(crate) guard: RwLockWriteGuard<'a, RawTable<(K, V)>>,
    pub(crate) key: K,
    pub(crate) hash: u64,
    pub(crate) slot: InsertSlot,
}

unsafe impl<K: Sync, V: Sync> Send for VacantEntry<'_, K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for VacantEntry<'_, K, V> {}

impl<'a, K: Eq + Hash + 'a, V> VacantEntry<'a, K, V> {
    /// Inserts the value and returns a guard that keeps the shard lock alive.
    pub fn insert(mut self, value: V) -> RefMut<'a, K, V> {
        let bucket = unsafe {
            self.guard
                .insert_in_slot(self.hash, self.slot, (self.key, value))
        };
        unsafe {
            let kv = bucket.as_mut();
            RefMut::new(self.guard, &kv.0, &mut kv.1)
        }
    }

    /// Returns a reference to the key.
    pub fn key(&self) -> &K {
        &self.key
    }
}
