//! One shard of the concurrent hash map.

use crossbeam_utils::CachePadded;
use hashbrown::raw::RawTable;
use parking_lot::RwLock;

/// One shard of the concurrent hash map, cache-line padded to prevent false sharing.
pub struct Shard<K, V> {
    // NOTE: dashmap had a soundness bug (issue #10) where `&T` was transmuted to
    // `&mut T`, violating Stacked Borrows even under exclusive locks. Our guards
    // avoid this: `RefMut` derives its `*mut V` from `bucket.as_mut()` under a
    // write lock, which provides genuine mutable provenance.
    inner: CachePadded<RwLock<RawTable<(K, V)>>>,
}

impl<K, V> Default for Shard<K, V> {
    fn default() -> Self {
        Self {
            inner: CachePadded::new(RwLock::new(RawTable::new())),
        }
    }
}

impl<K, V> Shard<K, V> {
    /// Create a new empty shard. Does not allocate.
    pub const fn new() -> Self {
        Self {
            inner: CachePadded::new(RwLock::new(RawTable::new())),
        }
    }
}

// Allow deref-like access to RwLock methods through the CachePadded wrapper
impl<K, V> Shard<K, V> {
    #[inline]
    pub(crate) fn read_table(&self) -> parking_lot::RwLockReadGuard<'_, RawTable<(K, V)>> {
        self.inner.read()
    }

    #[inline]
    pub(crate) fn write_table(&self) -> parking_lot::RwLockWriteGuard<'_, RawTable<(K, V)>> {
        self.inner.write()
    }

    #[inline]
    pub(crate) fn try_write_table(
        &self,
    ) -> Option<parking_lot::RwLockWriteGuard<'_, RawTable<(K, V)>>> {
        self.inner.try_write()
    }
}
