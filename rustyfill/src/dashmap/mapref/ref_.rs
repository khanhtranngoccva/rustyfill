//! Shared (read-only) reference guard backed by a shared shard lock.

use crate::lang_core::fmt;
use crate::lang_core::hash::Hash;
use crate::lang_core::ops::Deref;

type SharedValue<T> = dashmap::SharedValue<T>;
type RwLockReadGuard<'a, K, V> =
    dashmap::RwLockReadGuard<'a, hashbrown::raw::RawTable<(K, SharedValue<V>)>>;

/// An immutable reference to a key-value pair in a `DashMap`, backed by a
/// shared shard lock. Returned by [`RefMut::downgrade`](super::RefMut::downgrade).
pub struct Ref<'a, K, V> {
    #[allow(dead_code)]
    _guard: RwLockReadGuard<'a, K, V>,
    k: *const K,
    v: *const V,
}

unsafe impl<K: Eq + Hash + Sync, V: Sync> Send for Ref<'_, K, V> {}
unsafe impl<K: Eq + Hash + Sync, V: Sync> Sync for Ref<'_, K, V> {}

impl<'a, K: Eq + Hash, V> Ref<'a, K, V> {
    /// # Safety
    ///
    /// The caller must ensure that `k` points to a live key inside the locked
    /// shard and `v` points to a live value wrapped in `SharedValue`.
    pub(crate) unsafe fn new(guard: RwLockReadGuard<'a, K, V>, k: *const K, v: *const V) -> Self {
        Self {
            _guard: guard,
            k,
            v,
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

    /// Returns references to both the key and value.
    pub fn pair(&self) -> (&K, &V) {
        unsafe { (&*self.k, &*self.v) }
    }
}

impl<K: Eq + Hash + fmt::Debug, V: fmt::Debug> fmt::Debug for Ref<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ref")
            .field("k", &self.key())
            .field("v", &self.value())
            .finish()
    }
}

impl<K: Eq + Hash, V> Deref for Ref<'_, K, V> {
    type Target = V;

    fn deref(&self) -> &V {
        self.value()
    }
}
