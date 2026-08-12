//! Multi-owner guard types that keep shard locks alive via [`Arc`] references.
//!
//! Used by the iterator module: the iterator holds one `Arc` to the guard and
//! clones it into each yielded [`RefMulti`] / [`RefMutMulti`]. The lock is held
//! as long as any holder exists.

use lang_core::ops::{Deref, DerefMut};
use lang_std::sync::Arc;

use hashbrown::raw::RawTable;
use parking_lot::{RwLockReadGuard, RwLockWriteGuard};

// ── RefMulti (immutable, shared-ownership) ──────────────────────────────────────

/// An immutable reference guard backed by a shared-counted lock.
///
/// Multiple `RefMulti` instances (plus the producing iterator) can coexist,
/// each holding an `Arc` clone of the same read guard.
pub struct RefMulti<'a, K, V> {
    _guard: Arc<RwLockReadGuard<'a, RawTable<(K, V)>>>,
    k: *const K,
    v: *const V,
}

unsafe impl<K: Sync, V: Sync> Send for RefMulti<'_, K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for RefMulti<'_, K, V> {}

impl<'a, K, V> RefMulti<'a, K, V> {
    /// Create a new `RefMulti` from an `Arc`-wrapped read guard and raw pointers.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `k` and `v` point to valid data inside the
    /// `RawTable` protected by `guard`, and that the guard outlives this struct.
    pub(crate) unsafe fn new(
        guard: Arc<RwLockReadGuard<'a, RawTable<(K, V)>>>,
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

impl<K, V> Deref for RefMulti<'_, K, V> {
    type Target = V;

    fn deref(&self) -> &V {
        self.value()
    }
}

// ── RefMutMulti (mutable, shared-ownership) ─────────────────────────────────────

/// A mutable reference guard backed by a shared-counted write lock.
///
/// Multiple `RefMutMulti` instances (plus the producing iterator) can coexist,
/// each holding an `Arc` clone of the same write guard.
pub struct RefMutMulti<'a, K, V> {
    #[allow(dead_code)]
    guard: Arc<RwLockWriteGuard<'a, RawTable<(K, V)>>>,
    k: *const K,
    v: *mut V,
}

unsafe impl<K: Sync, V: Sync> Send for RefMutMulti<'_, K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for RefMutMulti<'_, K, V> {}

impl<'a, K, V> RefMutMulti<'a, K, V> {
    /// Create a new `RefMutMulti` from an `Arc`-wrapped write guard and raw pointers.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `k` and `v` point to valid data inside the
    /// `RawTable` protected by `guard`, and that the guard outlives this struct.
    pub(crate) unsafe fn new(
        guard: Arc<RwLockWriteGuard<'a, RawTable<(K, V)>>>,
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
}

impl<K, V> Deref for RefMutMulti<'_, K, V> {
    type Target = V;

    fn deref(&self) -> &V {
        self.value()
    }
}

impl<K, V> DerefMut for RefMutMulti<'_, K, V> {
    fn deref_mut(&mut self) -> &mut V {
        self.value_mut()
    }
}
