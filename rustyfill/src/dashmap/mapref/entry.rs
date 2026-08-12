//! Entry API for fallible DashMap operations.
//!
//! Mirrors the API of `dashmap::mapref::entry::{Entry, OccupiedEntry, VacantEntry}`
//! but is constructed by [`TryDashMap`](super::super::TryDashMap) so that capacity is
//! reserved *before* insertion, guaranteeing that entry methods like
//! [`VacantEntry::insert`] cannot panic on out-of-memory.

use lang_core::hash::Hash;
use lang_core::mem;

type SharedValue<T> = dashmap::SharedValue<T>;
type RwLockWriteGuard<'a, K, V> =
    dashmap::RwLockWriteGuard<'a, hashbrown::raw::RawTable<(K, SharedValue<V>)>>;
type ShardEntry<K, V> = (K, SharedValue<V>);

use super::RefMut;
use crate::try_default::{TryDefault, TryDefaultError};

/// An owned read-write accessor to a slot in a `DashMap`.
///
/// This mirrors [`dashmap::mapref::entry::Entry`] and is produced by
/// [`TryDashMap::try_entry`](super::super::TryDashMap::try_entry).
pub enum Entry<'a, K, V> {
    /// The entry is occupied.
    Occupied(OccupiedEntry<'a, K, V>),
    /// The entry is vacant.
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K: Eq + Hash, V> Entry<'a, K, V> {
    /// Apply a function to the stored value if it exists.
    pub fn and_modify(self, f: impl FnOnce(&mut V)) -> Self {
        match self {
            Entry::Occupied(mut entry) => {
                f(entry.get_mut());
                Entry::Occupied(entry)
            }
            Entry::Vacant(entry) => Entry::Vacant(entry),
        }
    }

    /// Get the key of the entry.
    pub fn key(&self) -> &K {
        match *self {
            Entry::Occupied(ref entry) => entry.key(),
            Entry::Vacant(ref entry) => entry.key(),
        }
    }

    /// Consume the entry and return the key.
    pub fn into_key(self) -> K {
        match self {
            Entry::Occupied(entry) => entry.into_key(),
            Entry::Vacant(entry) => entry.into_key(),
        }
    }

    /// Return a mutable reference to the element if it exists,
    /// otherwise insert the fallibly-constructed default value and return a mutable
    /// reference to that.
    ///
    /// Returns an error if constructing the default value fails (e.g. due to
    /// allocation failure). Unlike [`Default::default()`], which may panic on OOM,
    /// this method uses [`TryDefault::try_default()`] to propagate the error.
    pub fn or_try_default(self) -> Result<RefMut<'a, K, V>, TryDefaultError>
    where
        V: TryDefault,
    {
        match self {
            Entry::Occupied(entry) => Ok(entry.into_ref()),
            Entry::Vacant(entry) => Ok(entry.insert(V::try_default()?)),
        }
    }

    /// Return a mutable reference to the element if it exists,
    /// otherwise insert the provided value and return a mutable reference to that.
    pub fn or_insert(self, value: V) -> RefMut<'a, K, V> {
        match self {
            Entry::Occupied(entry) => entry.into_ref(),
            Entry::Vacant(entry) => entry.insert(value),
        }
    }

    /// Return a mutable reference to the element if it exists,
    /// otherwise insert the result of the closure and return a mutable reference.
    pub fn or_insert_with(self, value: impl FnOnce() -> V) -> RefMut<'a, K, V> {
        match self {
            Entry::Occupied(entry) => entry.into_ref(),
            Entry::Vacant(entry) => entry.insert(value()),
        }
    }

    /// Like [`Self::or_insert_with`] but the closure can fail.
    pub fn or_try_insert_with<E>(
        self,
        value: impl FnOnce() -> Result<V, E>,
    ) -> Result<RefMut<'a, K, V>, E> {
        match self {
            Entry::Occupied(entry) => Ok(entry.into_ref()),
            Entry::Vacant(entry) => Ok(entry.insert(value()?)),
        }
    }

    /// Sets the value of the entry regardless of whether it was already occupied,
    /// and returns a mutable reference to the inserted value.
    pub fn insert(self, value: V) -> RefMut<'a, K, V> {
        match self {
            Entry::Occupied(mut entry) => {
                entry.insert(value);
                entry.into_ref()
            }
            Entry::Vacant(entry) => entry.insert(value),
        }
    }

    /// Sets the value of the entry, and returns an occupied entry.
    pub fn insert_entry(self, value: V) -> OccupiedEntry<'a, K, V>
    where
        K: Clone,
    {
        match self {
            Entry::Occupied(mut entry) => {
                entry.insert(value);
                entry
            }
            Entry::Vacant(entry) => entry.insert_entry(value),
        }
    }
}

// ── VacantEntry ───────────────────────────────────────────────────────────────

/// A vacant entry – the key does not yet exist in the map.
pub struct VacantEntry<'a, K, V> {
    shard: RwLockWriteGuard<'a, K, V>,
    key: K,
    hash: u64,
    slot: hashbrown::raw::InsertSlot,
}

unsafe impl<K: Eq + Hash + Sync, V: Sync> Send for VacantEntry<'_, K, V> {}
unsafe impl<K: Eq + Hash + Sync, V: Sync> Sync for VacantEntry<'_, K, V> {}

impl<'a, K: Eq + Hash, V> VacantEntry<'a, K, V> {
    /// # Safety
    ///
    /// `slot` must be a valid `InsertSlot` returned by `find_or_find_insert_slot`
    /// on the locked `shard`.
    pub(crate) unsafe fn new(
        shard: RwLockWriteGuard<'a, K, V>,
        key: K,
        hash: u64,
        slot: hashbrown::raw::InsertSlot,
    ) -> Self {
        Self {
            shard,
            key,
            hash,
            slot,
        }
    }

    /// Inserts the value and returns a mutable reference to it.
    pub fn insert(mut self, value: V) -> RefMut<'a, K, V> {
        unsafe {
            let bucket = self.shard.insert_in_slot(
                self.hash,
                self.slot,
                (self.key, SharedValue::new(value)),
            );
            let (k, sv) = bucket.as_mut();
            let v = sv.get_mut() as *mut V;
            RefMut::new(self.shard, k, v)
        }
    }

    /// Inserts the value and returns an occupied entry with the same key.
    pub fn insert_entry(mut self, value: V) -> OccupiedEntry<'a, K, V>
    where
        K: Clone,
    {
        unsafe {
            let bucket = self.shard.insert_in_slot(
                self.hash,
                self.slot,
                (self.key.clone(), SharedValue::new(value)),
            );
            OccupiedEntry::new(self.shard, self.key, bucket)
        }
    }

    /// Consumes the entry and returns the key.
    pub fn into_key(self) -> K {
        self.key
    }

    /// Returns a reference to the key.
    pub fn key(&self) -> &K {
        &self.key
    }
}

// ── OccupiedEntry ─────────────────────────────────────────────────────────────

/// An occupied entry – the key already exists in the map.
pub struct OccupiedEntry<'a, K, V> {
    shard: RwLockWriteGuard<'a, K, V>,
    bucket: hashbrown::raw::Bucket<ShardEntry<K, V>>,
    key: K,
}

unsafe impl<K: Eq + Hash + Sync, V: Sync> Send for OccupiedEntry<'_, K, V> {}
unsafe impl<K: Eq + Hash + Sync, V: Sync> Sync for OccupiedEntry<'_, K, V> {}

impl<'a, K: Eq + Hash, V> OccupiedEntry<'a, K, V> {
    /// # Safety
    ///
    /// `bucket` must point to a live element inside the locked `shard`.
    pub(crate) unsafe fn new(
        shard: RwLockWriteGuard<'a, K, V>,
        key: K,
        bucket: hashbrown::raw::Bucket<ShardEntry<K, V>>,
    ) -> Self {
        Self { shard, bucket, key }
    }

    /// Returns a reference to the value.
    pub fn get(&self) -> &V {
        unsafe { self.bucket.as_ref().1.get() }
    }

    /// Returns a mutable reference to the value.
    pub fn get_mut(&mut self) -> &mut V {
        unsafe { self.bucket.as_mut().1.get_mut() }
    }

    /// Replaces the value and returns the old one.
    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }

    /// Consumes the entry and returns a mutable reference to the value,
    /// keeping the shard lock active.
    pub fn into_ref(self) -> RefMut<'a, K, V> {
        unsafe {
            let (k, sv) = self.bucket.as_mut();
            let v = sv.get_mut() as *mut V;
            RefMut::new(self.shard, k, v)
        }
    }

    /// Consumes the entry and returns the key.
    pub fn into_key(self) -> K {
        self.key
    }

    /// Returns a reference to the key.
    pub fn key(&self) -> &K {
        unsafe { &self.bucket.as_ref().0 }
    }

    /// Removes the entry and returns the value.
    pub fn remove(mut self) -> V {
        let ((_k, v), _) = unsafe { self.shard.remove(self.bucket) };
        v.into_inner()
    }

    /// Removes the entry and returns the key-value pair.
    pub fn remove_entry(mut self) -> (K, V) {
        let ((k, v), _) = unsafe { self.shard.remove(self.bucket) };
        (k, v.into_inner())
    }

    /// Replaces the value in-place and returns the old key-value pair.
    pub fn replace_entry(self, value: V) -> (K, V) {
        let (k, v) = mem::replace(
            unsafe { self.bucket.as_mut() },
            (self.key, SharedValue::new(value)),
        );
        (k, v.into_inner())
    }
}
