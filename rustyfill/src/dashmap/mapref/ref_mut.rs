//! Mutable reference guard backed by an exclusive shard lock.

use crate::lang_core::fmt;
use crate::lang_core::hash::Hash;
use crate::lang_core::ops::{Deref, DerefMut};

type SharedValue<T> = dashmap::SharedValue<T>;
type RwLockWriteGuard<'a, K, V> =
    dashmap::RwLockWriteGuard<'a, hashbrown::raw::RawTable<(K, SharedValue<V>)>>;

/// A mutable reference to a key-value pair in a `DashMap`, backed by an
/// exclusive shard lock. Returned by [`VacantEntry::insert`](super::entry::VacantEntry::insert),
/// [`OccupiedEntry::into_ref`](super::entry::OccupiedEntry::into_ref), and
/// [`Entry::insert`](super::entry::Entry::insert).
pub struct RefMut<'a, K, V> {
    #[allow(dead_code)]
    guard: RwLockWriteGuard<'a, K, V>,
    k: *const K,
    v: *mut V,
}

unsafe impl<K: Eq + Hash + Sync, V: Sync> Send for RefMut<'_, K, V> {}
unsafe impl<K: Eq + Hash + Sync, V: Sync> Sync for RefMut<'_, K, V> {}

impl<'a, K: Eq + Hash, V> RefMut<'a, K, V> {
    /// # Safety
    ///
    /// The caller must ensure that `k` points to a live key inside the locked
    /// shard and `v` points to a live value wrapped in `SharedValue`.
    pub(crate) unsafe fn new(guard: RwLockWriteGuard<'a, K, V>, k: *const K, v: *mut V) -> Self {
        Self { guard, k, v }
    }

    /// Downgrades this exclusive lock to a shared read lock.
    pub fn downgrade(self) -> crate::dashmap::mapref::Ref<'a, K, V> {
        unsafe {
            crate::dashmap::mapref::Ref::new(
                RwLockWriteGuard::downgrade(self.guard),
                self.k,
                self.v,
            )
        }
    }

    /// Returns a reference to the key.
    pub fn key(&self) -> &K {
        self.pair().0
    }

    /// Returns a reference to the value.
    pub fn value(&self) -> &V {
        self.pair().1
    }

    /// Returns a mutable reference to the value.
    pub fn value_mut(&mut self) -> &mut V {
        self.pair_mut().1
    }

    /// Returns references to both the key and value.
    pub fn pair(&self) -> (&K, &V) {
        unsafe { (&*self.k, &*self.v) }
    }

    /// Returns references to both the key and a mutable reference to the value.
    pub fn pair_mut(&mut self) -> (&K, &mut V) {
        unsafe { (&*self.k, &mut *self.v) }
    }
}

impl<K: Eq + Hash, V> Deref for RefMut<'_, K, V> {
    type Target = V;

    fn deref(&self) -> &V {
        self.value()
    }
}

impl<K: Eq + Hash, V> DerefMut for RefMut<'_, K, V> {
    fn deref_mut(&mut self) -> &mut V {
        self.value_mut()
    }
}

impl<K: Eq + Hash + fmt::Debug, V: fmt::Debug> fmt::Debug for RefMut<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RefMut")
            .field("k", &self.key())
            .field("v", &self.value())
            .finish()
    }
}
