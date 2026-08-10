//! Single-owner guard types that keep shard locks alive while providing
//! references into the map.
//!
//! These are used by [`get`](super::ConcurrentHashMap::get),
//! [`get_mut`](super::ConcurrentHashMap::get_mut), and the Entry API —
//! scenarios where exactly one consumer holds the lock.

use core::ops::{Deref, DerefMut};
use hashbrown::raw::RawTable;
use parking_lot::{RwLockReadGuard, RwLockWriteGuard};

// ── Ref (immutable) ─────────────────────────────────────────────────────────────

/// An immutable reference guard to a key-value pair.
pub struct Ref<'a, K, V> {
    _guard: RwLockReadGuard<'a, RawTable<(K, V)>>,
    k: *const K,
    v: *const V,
}

unsafe impl<K: Sync, V: Sync> Send for Ref<'_, K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for Ref<'_, K, V> {}

impl<'a, K, V> Ref<'a, K, V> {
    /// Create a new `Ref` from a read guard and raw pointers.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `k` and `v` point to valid data inside the
    /// `RawTable` protected by `guard`, and that the guard outlives this struct.
    pub(crate) unsafe fn new(
        guard: RwLockReadGuard<'a, RawTable<(K, V)>>,
        k: *const K,
        v: *const V,
    ) -> Self {
        Self {
            _guard: guard,
            k,
            v,
        }
    }

    /// Returns a reference to the key.
    pub fn key(&self) -> &K {
        unsafe { &*self.k }
    }

    /// Returns a reference to the value.
    pub fn value(&self) -> &V {
        unsafe { &*self.v }
    }
}

impl<K, V> Deref for Ref<'_, K, V> {
    type Target = V;

    fn deref(&self) -> &V {
        self.value()
    }
}

// ── RefMut (mutable) ────────────────────────────────────────────────────────────

/// A mutable reference guard to a key-value pair.
pub struct RefMut<'a, K, V> {
    #[allow(dead_code)]
    guard: RwLockWriteGuard<'a, RawTable<(K, V)>>,
    k: *const K,
    v: *mut V,
}

unsafe impl<K: Sync, V: Sync> Send for RefMut<'_, K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for RefMut<'_, K, V> {}

impl<'a, K, V> RefMut<'a, K, V> {
    /// Create a new `RefMut` from a lock guard and raw pointers.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `k` and `v` point to valid data inside the
    /// `RawTable` protected by `guard`, and that the guard outlives this struct.
    pub(crate) unsafe fn new(
        guard: RwLockWriteGuard<'a, RawTable<(K, V)>>,
        k: *const K,
        v: *mut V,
    ) -> Self {
        Self { guard, k, v }
    }

    /// Returns a reference to the key.
    pub fn key(&self) -> &K {
        unsafe { &*self.k }
    }

    /// Returns a reference to the value.
    pub fn value(&self) -> &V {
        unsafe { &*self.v }
    }

    /// Returns a mutable reference to the value.
    pub fn value_mut(&mut self) -> &mut V {
        unsafe { &mut *self.v }
    }

    /// Returns references to both key and value.
    pub fn pair(&self) -> (&K, &V) {
        (self.key(), self.value())
    }

    /// Returns references to both key and a mutable value.
    pub fn pair_mut(&mut self) -> (&K, &mut V) {
        unsafe { (&*self.k, &mut *self.v) }
    }

    /// Downgrades this exclusive guard to a shared guard, consuming `self`.
    ///
    /// This releases the write lock and replaces it with a read lock, allowing
    /// other readers (and writers once all readers drop) to proceed concurrently.
    pub fn downgrade(self) -> Ref<'a, K, V>
    where
        (K, V): Send,
    {
        Ref {
            _guard: RwLockWriteGuard::downgrade(self.guard),
            k: self.k,
            v: self.v,
        }
    }
}

impl<K, V> Deref for RefMut<'_, K, V> {
    type Target = V;

    fn deref(&self) -> &V {
        self.value()
    }
}

impl<K, V> DerefMut for RefMut<'_, K, V> {
    fn deref_mut(&mut self) -> &mut V {
        self.value_mut()
    }
}
