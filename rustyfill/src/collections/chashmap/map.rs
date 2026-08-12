//! Core [`ConcurrentHashMap`] implementation.

use crate::alloc::vec::SliceInitGuard;
use crate::alloc::{AllocError, TryReserveError};
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::{TryDebug, helpers::FormatterExt};
use lang_alloc;
use lang_alloc::boxed::Box;
use lang_core::borrow::Borrow;
use lang_core::fmt;
use lang_core::hash::{BuildHasher, Hash};
use lang_core::mem::{self, MaybeUninit};
use lang_core::ptr;
use lang_std::hash::RandomState;

use super::entry::{Entry, OccupiedEntry, VacantEntry};
use super::refs::{Ref, RefMut};
use super::shard::Shard;

// ── Storage enum ───────────────────────────────────────────────────────────────

/// Owns the backing array of shards, either on the heap or as a raw pointer
/// into a static allocation. The `Static` variant stores a raw pointer to sidestep
/// the `'static` bound that a `&'static mut [Shard]` would impose on the entire
/// struct, letting non-static `K`/`V` types flow through the type system freely.
pub(crate) enum ShardsStorage<K, V> {
    Heap(Box<[Shard<K, V>]>),
    Static(*mut [Shard<K, V>]),
}

impl<K, V> ShardsStorage<K, V> {
    fn get_shard(&self, idx: usize) -> &Shard<K, V> {
        &self.as_slice()[idx]
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn as_slice(&self) -> &[Shard<K, V>] {
        match self {
            Self::Heap(boxed) => boxed.as_ref(),
            Self::Static(slice_ptr) => unsafe { &**slice_ptr },
        }
    }
}

// ── Error types ────────────────────────────────────────────────────────────────

/// Error returned by blocking [`ConcurrentHashMap`] operations.
#[derive(Debug)]
pub enum ConcurrentHashMapError {
    Alloc(AllocError),
    Reserve(TryReserveError),
    Clone(TryCloneError),
    Overflow,
    Other(&'static str),
}

impl fmt::Display for ConcurrentHashMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(
                f,
                "concurrent hash map operation failed: heap allocation error"
            ),
            Self::Reserve(e) => write!(f, "concurrent hash map operation failed: {}", e),
            Self::Clone(e) => write!(f, "concurrent hash map operation failed: {}", e),
            Self::Overflow => write!(
                f,
                "concurrent hash map operation failed: capacity calculation overflowed"
            ),
            Self::Other(msg) => write!(f, "concurrent hash map operation failed: {}", msg),
        }
    }
}

impl From<AllocError> for ConcurrentHashMapError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

impl From<TryReserveError> for ConcurrentHashMapError {
    fn from(e: TryReserveError) -> Self {
        Self::Reserve(e)
    }
}

impl From<TryCloneError> for ConcurrentHashMapError {
    fn from(e: TryCloneError) -> Self {
        Self::Clone(e)
    }
}

impl From<TryDefaultError> for ConcurrentHashMapError {
    fn from(e: TryDefaultError) -> Self {
        match e {
            TryDefaultError::Alloc(a) => Self::Alloc(a),
            TryDefaultError::Reserve(r) => Self::Reserve(r),
            TryDefaultError::Overflow => Self::Overflow,
            TryDefaultError::Other(m) => Self::Other(m),
        }
    }
}

impl TryDebug for ConcurrentHashMapError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => f
                .try_debug_struct("ConcurrentHashMapError::Alloc")
                .field("0", e)
                .finish(),
            Self::Reserve(e) => f
                .try_debug_struct("ConcurrentHashMapError::Reserve")
                .field("0", e)
                .finish(),
            Self::Clone(e) => f
                .try_debug_struct("ConcurrentHashMapError::Clone")
                .field("0", e)
                .finish(),
            Self::Overflow => f.write_str("ConcurrentHashMapError::Overflow"),
            Self::Other(msg) => f
                .try_debug_struct("ConcurrentHashMapError::Other")
                .field("0", msg)
                .finish(),
        }
    }
}

/// Error returned by non-blocking [`ConcurrentHashMap`] operations.
#[derive(Debug)]
pub enum ConcurrentHashMapNonblockError {
    Alloc(AllocError),
    Reserve(TryReserveError),
    Clone(TryCloneError),
    Overflow,
    Other(&'static str),
    Locked,
}

impl fmt::Display for ConcurrentHashMapNonblockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(
                f,
                "concurrent hash map operation failed: heap allocation error"
            ),
            Self::Reserve(e) => write!(f, "concurrent hash map operation failed: {}", e),
            Self::Clone(e) => write!(f, "concurrent hash map operation failed: {}", e),
            Self::Overflow => write!(
                f,
                "concurrent hash map operation failed: capacity calculation overflowed"
            ),
            Self::Other(msg) => write!(f, "concurrent hash map operation failed: {}", msg),
            Self::Locked => write!(f, "concurrent hash map operation failed: shard locked"),
        }
    }
}

impl From<ConcurrentHashMapError> for ConcurrentHashMapNonblockError {
    fn from(e: ConcurrentHashMapError) -> Self {
        match e {
            ConcurrentHashMapError::Alloc(a) => Self::Alloc(a),
            ConcurrentHashMapError::Reserve(r) => Self::Reserve(r),
            ConcurrentHashMapError::Clone(c) => Self::Clone(c),
            ConcurrentHashMapError::Overflow => Self::Overflow,
            ConcurrentHashMapError::Other(m) => Self::Other(m),
        }
    }
}

impl TryDebug for ConcurrentHashMapNonblockError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => f
                .try_debug_struct("ConcurrentHashMapNonblockError::Alloc")
                .field("0", e)
                .finish(),
            Self::Reserve(e) => f
                .try_debug_struct("ConcurrentHashMapNonblockError::Reserve")
                .field("0", e)
                .finish(),
            Self::Clone(e) => f
                .try_debug_struct("ConcurrentHashMapNonblockError::Clone")
                .field("0", e)
                .finish(),
            Self::Overflow => f.write_str("ConcurrentHashMapNonblockError::Overflow"),
            Self::Other(msg) => f
                .try_debug_struct("ConcurrentHashMapNonblockError::Other")
                .field("0", msg)
                .finish(),
            Self::Locked => f.write_str("ConcurrentHashMapNonblockError::Locked"),
        }
    }
}

// ── ConcurrentHashMap ─────────────────────────────────────────────────────────

/// A concurrent hash map backed by sharded `RwLock<RawTable>` instances.
///
/// The backing store is either a `Box<[Shard]>` (heap-allocated) or a
/// `&'static mut [Shard]`, allowing static initialization with zero allocations.
///
/// All mutating and constructing operations are fallible: they return [`Result`] instead of panicking
/// on out-of-memory.
///
/// The shard count must be a power of two. This allows the hash-to-shard mapping
/// to use a fast bit-shift instead of division.
pub struct ConcurrentHashMap<K, V, S = RandomState> {
    shards: ShardsStorage<K, V>,
    hasher: S,
    /// Bit shift for fast shard index computation: `(hash << 7) >> shift`.
    shift: u32,
}

unsafe impl<K: Eq + Hash + Send, V: Send, S: Send + Sync> Send for ConcurrentHashMap<K, V, S> {}
unsafe impl<K: Eq + Hash + Send + Sync, V: Send + Sync, S: Send + Sync> Sync
    for ConcurrentHashMap<K, V, S>
{
}

// ── Construction impl blocks matching dashmap's API matrix ─────────────────────
// dashmap exposes: new(), with_capacity(), with_hasher(), with_capacity_and_hasher()
// We mirror this pattern plus the shard-count variants.

impl<K: Eq + Hash, V> ConcurrentHashMap<K, V, RandomState> {
    /// Fallibly construct an empty map with a default hasher and 32 shards.
    pub fn try_new() -> Result<Self, ConcurrentHashMapError> {
        Self::try_with_shards(32)
    }

    /// Fallibly construct with capacity spread across 32 shards.
    pub fn try_with_capacity(capacity: usize) -> Result<Self, ConcurrentHashMapError> {
        Self::try_with_capacity_and_shards(capacity, 32)
    }
}

impl<K: Eq + Hash, V, S: BuildHasher> ConcurrentHashMap<K, V, S> {
    // ── Construction with custom hasher (no capacity, no explicit shards) ──

    /// Construct an empty map with the provided hasher and 32 shards.
    ///
    /// Equivalent to `DashMap::with_hasher`.
    pub fn try_with_hasher(hasher: S) -> Result<Self, ConcurrentHashMapError> {
        Self::try_with_capacity_and_hasher_and_shards(0, hasher, 32)
    }

    // ── Construction with capacity and custom hasher ──

    /// Construct with capacity and the provided hasher, using 32 shards.
    ///
    /// Equivalent to `DashMap::with_capacity_and_hasher`.
    pub fn try_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<Self, ConcurrentHashMapError> {
        Self::try_with_capacity_and_hasher_and_shards(capacity, hasher, 32)
    }

    // ── Construction with custom shard count ──

    /// Fallibly construct with a custom number of shards and a default hasher.
    pub fn try_with_shards(shard_count: usize) -> Result<Self, ConcurrentHashMapError>
    where
        S: TryDefault,
    {
        let hasher = S::try_default()?;
        Self::try_with_capacity_and_hasher_and_shards(0, hasher, shard_count)
    }

    /// Fallibly construct with capacity spread evenly across `shard_count` shards.
    pub fn try_with_capacity_and_shards(
        capacity: usize,
        shard_count: usize,
    ) -> Result<Self, ConcurrentHashMapError>
    where
        S: TryDefault,
    {
        let hasher = S::try_default()?;
        Self::try_with_capacity_and_hasher_and_shards(capacity, hasher, shard_count)
    }

    /// Fallibly construct with a provided hasher and a custom number of shards.
    pub fn try_with_hasher_and_shards(
        hasher: S,
        shard_count: usize,
    ) -> Result<Self, ConcurrentHashMapError> {
        Self::try_with_capacity_and_hasher_and_shards(0, hasher, shard_count)
    }

    /// Fallibly constructs a map with the given capacity, hasher, and shard count.
    pub fn try_with_capacity_and_hasher_and_shards(
        capacity: usize,
        hasher: S,
        shard_count: usize,
    ) -> Result<Self, ConcurrentHashMapError> {
        if shard_count < 2 || (shard_count & (shard_count - 1)) != 0 {
            return Err(ConcurrentHashMapError::Other(
                "shard count must be a power of two and >= 2",
            ));
        }
        let shift = usize::BITS - shard_count.trailing_zeros();
        let layout = lang_alloc::alloc::Layout::array::<Shard<K, V>>(shard_count)
            .map_err(|_| ConcurrentHashMapError::Overflow)?;
        let ptr = unsafe { lang_alloc::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(ConcurrentHashMapError::Alloc(AllocError { layout }));
        }

        // Wrap immediately in a Box so Drop cleans up the allocation on panic.
        // SAFETY: layout matches `shard_count` elements of MaybeUninit<Shard<K,V>>,
        // which has the same size and alignment as Shard<K,V>.
        let mut uninit_shards: Box<[MaybeUninit<Shard<K, V>>]> =
            unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(ptr.cast(), shard_count)) };
        let mut guard = SliceInitGuard::new(&mut uninit_shards);

        for slot in guard.slots.iter_mut() {
            unsafe {
                ptr::write(slot.as_mut_ptr(), Shard::<K, V>::new());
            }
            guard.count += 1;
        }

        guard.forget();

        // SAFETY: all `shard_count` slots were written successfully above.
        // Box<[MaybeUninit<Shard<K,V>>]> and Box<[Shard<K,V>]> have identical layouts.
        let boxed: Box<[Shard<K, V>]> = unsafe { mem::transmute(uninit_shards) };

        let map = Self {
            shards: ShardsStorage::Heap(boxed),
            hasher,
            shift,
        };
        if capacity > 0 {
            map.try_reserve(capacity)?;
        }
        Ok(map)
    }

    /// Construct from a static slice of shards. No heap allocation occurs.
    ///
    /// Requires `K: 'static + Eq + Hash` and `V: 'static` because the shards live
    /// in a `static` item and must outlive any borrow.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that this function is called exactly once for the
    /// given `shards` slice, and that no other references to the static data exist
    /// concurrently. After this call, the [`ConcurrentHashMap`] owns exclusive access
    /// to the shard array. A typical pattern is to wrap this in an `OnceLock`:
    ///
    /// ```ignore
    /// use ::std::sync::OnceLock;
    /// static MAP: OnceLock<ConcurrentHashMap<Key, Val>> = OnceLock::new();
    ///
    /// MAP.get_or_init(|| unsafe {
    ///     ConcurrentHashMap::from_static(SLICE_OF_SHARDS, hasher)
    /// });
    /// ```
    ///
    /// Prefer [`declare_concurrent_hash_map!`](super::declare_concurrent_hash_map!) for
    /// safe, compile-time static declaration.
    pub unsafe fn from_static(shards: &'static mut [Shard<K, V>], hasher: S) -> Self
    where
        K: 'static + Eq + Hash,
        V: 'static,
    {
        assert!(
            shards.len() >= 2 && (shards.len() & (shards.len() - 1)) == 0,
            "static shard count must be a power of two and >= 2"
        );
        let shift = usize::BITS - shards.len().trailing_zeros();
        let ptr = shards.as_mut_ptr();
        let len = shards.len();
        Self {
            shards: ShardsStorage::Static(ptr::slice_from_raw_parts_mut(ptr, len)),
            hasher,
            shift,
        }
    }

    // ── Query ─────────────────────────────────────────────────────────────

    /// Returns the number of entries by scanning all shards.
    pub fn len(&self) -> usize {
        let mut count: usize = 0;
        for i in 0..self.shard_count() {
            let shard = self.shards.get_shard(i);
            let table = shard.read_table();
            count += table.len();
        }
        count
    }

    /// Returns true if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of shards.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Returns the hasher factory.
    pub fn hasher(&self) -> &S {
        &self.hasher
    }

    /// Returns the total capacity across all shards.
    pub fn capacity(&self) -> usize {
        let mut cap = 0;
        for i in 0..self.shard_count() {
            let shard = self.shards.get_shard(i);
            let table = shard.read_table();
            cap += table.capacity();
        }
        cap
    }

    // ── Insertion ─────────────────────────────────────────────────────────

    /// Fallibly insert a key-value pair.
    pub fn try_insert(&self, key: K, value: V) -> Result<Option<V>, ConcurrentHashMapError>
    where
        K: Eq + Hash,
    {
        let entry = self.try_entry(key)?;
        match entry {
            Entry::Occupied(mut e) => Ok(Some(e.insert(value))),
            Entry::Vacant(e) => {
                let _val = e.insert(value);
                Ok(None)
            }
        }
    }

    /// Like try_insert but returns key+value back on failure.
    pub fn try_insert_give_back(
        &self,
        key: K,
        value: V,
    ) -> Result<Option<V>, (K, V, ConcurrentHashMapError)>
    where
        K: Eq + Hash,
    {
        let entry = match self.try_entry_give_back(key) {
            Ok(e) => e,
            Err((k, err)) => return Err((k, value, err)),
        };
        match entry {
            Entry::Occupied(mut e) => Ok(Some(e.insert(value))),
            Entry::Vacant(e) => {
                let _val = e.insert(value);
                Ok(None)
            }
        }
    }

    /// Fallibly insert only if the key doesn't exist.
    pub fn try_insert_unique(&self, key: K, value: V) -> Result<(), (K, V, ConcurrentHashMapError)>
    where
        K: Eq + Hash + Clone,
    {
        let entry = match self.try_entry_give_back(key) {
            Ok(e) => e,
            Err((k, err)) => return Err((k, value, err)),
        };
        match entry {
            Entry::Occupied(e) => {
                let k = e.key().clone();
                Err((
                    k,
                    value,
                    ConcurrentHashMapError::Other("key already exists"),
                ))
            }
            Entry::Vacant(e) => {
                let _val = e.insert(value);
                Ok(())
            }
        }
    }

    // ── Lookup ────────────────────────────────────────────────────────────

    /// Return a reference guard to the value for the given key, if present.
    pub fn get<Q: ?Sized + Hash + Eq>(&self, key: &Q) -> Option<Ref<'_, K, V>>
    where
        K: Borrow<Q>,
    {
        let hash = self.hasher.hash_one(key);
        let idx = self.shard_index_internal(hash);
        let shard = self.shards.get_shard(idx);
        let guard = shard.read_table();
        guard
            .find(hash, |(k, _v)| k.borrow() == key)
            .map(|bucket| unsafe {
                let kv = bucket.as_ref();
                Ref::new(guard, &kv.0, &kv.1)
            })
    }

    /// Return a mutable reference guard to the value for the given key, if present.
    pub fn get_mut<Q: ?Sized + Hash + Eq>(&self, key: &Q) -> Option<RefMut<'_, K, V>>
    where
        K: Borrow<Q>,
    {
        let hash = self.hasher.hash_one(key);
        let idx = self.shard_index_internal(hash);
        let shard = self.shards.get_shard(idx);
        let guard = shard.write_table();
        guard
            .find(hash, |(k, _v)| k.borrow() == key)
            .map(|bucket| unsafe {
                let kv = bucket.as_mut();
                RefMut::new(guard, &kv.0, &mut kv.1)
            })
    }

    /// Returns true if the map contains a value for the given key.
    pub fn contains_key<Q: ?Sized + Hash + Eq>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
    {
        self.get(key).is_some()
    }

    // ── Removal ───────────────────────────────────────────────────────────

    /// Remove and return the value for the given key, if present.
    pub fn try_remove(&self, key: K) -> Result<Option<V>, ConcurrentHashMapError>
    where
        K: Eq + Hash,
    {
        let hash = self.compute_hash_internal(&key);
        let idx = self.shard_index_internal(hash);
        let shard = self.shards.get_shard(idx);
        let mut guard = shard.write_table();

        match guard.find(hash, |(k, _v)| *k == key) {
            Some(bucket) => {
                let ((_, v), _) = unsafe { guard.remove(bucket) };
                Ok(Some(v))
            }
            None => Ok(None),
        }
    }

    /// Remove and return the key-value pair for the given key, if present.
    pub fn try_remove_entry(&self, key: K) -> Result<Option<(K, V)>, ConcurrentHashMapError>
    where
        K: Eq + Hash,
    {
        let hash = self.compute_hash_internal(&key);
        let idx = self.shard_index_internal(hash);
        let shard = self.shards.get_shard(idx);
        let mut guard = shard.write_table();

        match guard.find(hash, |(k, _v)| *k == key) {
            Some(bucket) => {
                let ((k, v), _) = unsafe { guard.remove(bucket) };
                Ok(Some((k, v)))
            }
            None => Ok(None),
        }
    }

    // ── Entry API ─────────────────────────────────────────────────────────

    /// Fallibly obtain an entry for the given key.
    pub fn try_entry(&self, key: K) -> Result<Entry<'_, K, V>, ConcurrentHashMapError>
    where
        K: Eq + Hash,
    {
        self.do_entry(key)
    }

    /// Non-blocking variant of try_entry.
    pub fn try_entry_nonblock(
        &self,
        key: K,
    ) -> Result<Option<Entry<'_, K, V>>, ConcurrentHashMapNonblockError>
    where
        K: Eq + Hash,
    {
        self.do_entry_nonblock(key)
    }

    /// Like try_entry but returns the key back on error.
    pub fn try_entry_give_back(
        &self,
        key: K,
    ) -> Result<Entry<'_, K, V>, (K, ConcurrentHashMapError)>
    where
        K: Eq + Hash,
    {
        self.do_entry_give_back(key)
    }

    // ── Capacity / reserve ────────────────────────────────────────────────

    /// Fallibly reserve capacity distributed evenly across all shards.
    pub fn try_reserve(&self, additional: usize) -> Result<(), ConcurrentHashMapError> {
        let per_shard = additional.div_ceil(self.shard_count());
        if per_shard == 0 {
            return Ok(());
        }
        for i in 0..self.shard_count() {
            let shard = self.shards.get_shard(i);
            let mut table = shard.write_table();
            table
                .try_reserve(per_shard, |(k, _v): &(K, V)| self.hasher.hash_one(k))
                .map_err(|_| ConcurrentHashMapError::Reserve(TryReserveError::Other))?;
        }
        Ok(())
    }

    // ── Iteration ─────────────────────────────────────────────────────────

    /// Execute a closure on every key-value pair in the map.
    ///
    /// Acquires read locks on each shard sequentially. The closure receives
    /// immutable references to the key and value.
    pub fn visit_all<F>(&self, mut f: F)
    where
        F: FnMut(&K, &V),
    {
        for i in 0..self.shard_count() {
            let shard = self.shards.get_shard(i);
            let table = shard.read_table();
            // SAFETY: guard holds read lock, preventing mutation during iteration
            for bucket in unsafe { table.iter() } {
                let kv = unsafe { bucket.as_ref() };
                f(&kv.0, &kv.1);
            }
        }
    }

    /// Returns an immutable iterator over all key-value pairs in the map.
    ///
    /// Lazily acquires a read lock per shard via an `Arc` and yields fallible
    /// `RefMulti` guards. Errors occur if `Arc` allocation or cloning fails.
    pub fn iter(&self) -> super::iter::Iter<'_, K, V, S> {
        super::iter::Iter::new(self)
    }

    /// Returns a mutable iterator over all key-value pairs in the map.
    ///
    /// Lazily acquires a write lock per shard via an `Arc` and yields fallible
    /// `RefMutMulti` guards. Errors occur if `Arc` allocation or cloning fails.
    pub fn iter_mut(&self) -> super::iter::IterMut<'_, K, V, S> {
        super::iter::IterMut::new(self)
    }

    // ── Fast shard index ──────────────────────────────────────────────────

    /// Compute shard index using bit rotation.
    fn fast_shard_index(&self, hash: usize) -> usize {
        (hash << 7) >> self.shift
    }

    // ── Internal helpers for interner ─────────────────────────────────────

    /// Compute hash for any hashable type. Used by the interner module.
    pub(crate) fn compute_hash_internal<Q: ?Sized + Hash>(&self, key: &Q) -> u64 {
        self.hasher.hash_one(key)
    }

    /// Compute shard index from a pre-computed hash.
    pub(crate) fn shard_index_internal(&self, hash: u64) -> usize {
        // Fold upper 32 bits into lower half before truncating to usize,
        // so that 32-bit targets still benefit from full 64-bit entropy.
        let folded = hash ^ (hash >> 32);
        self.fast_shard_index(folded as usize)
    }

    /// Get a reference to the shards slice. Used by the interner module.
    pub(crate) fn get_shards(&self) -> &[Shard<K, V>] {
        self.shards.as_slice()
    }

    // ── Private entry implementations ─────────────────────────────────────

    fn do_entry(&self, key: K) -> Result<Entry<'_, K, V>, ConcurrentHashMapError>
    where
        K: Eq + Hash,
    {
        let hash = self.compute_hash_internal(&key);
        let idx = self.shard_index_internal(hash);
        let shard = self.shards.get_shard(idx);
        let mut guard = shard.write_table();

        guard
            .try_reserve(1, |(k, _v): &(K, V)| self.hasher.hash_one(k))
            .map_err(|_| ConcurrentHashMapError::Reserve(TryReserveError::Other))?;

        match guard.find_or_find_insert_slot(
            hash,
            |(k, _v)| *k == key,
            |(k, _v): &(K, V)| self.hasher.hash_one(k),
        ) {
            Ok(bucket) => Ok(Entry::Occupied(OccupiedEntry { guard, bucket })),
            Err(slot) => Ok(Entry::Vacant(VacantEntry {
                guard,
                key,
                hash,
                slot,
            })),
        }
    }

    fn do_entry_nonblock(
        &self,
        key: K,
    ) -> Result<Option<Entry<'_, K, V>>, ConcurrentHashMapNonblockError>
    where
        K: Eq + Hash,
    {
        let hash = self.compute_hash_internal(&key);
        let idx = self.shard_index_internal(hash);
        let shard = self.shards.get_shard(idx);
        let mut guard = shard
            .try_write_table()
            .ok_or(ConcurrentHashMapNonblockError::Locked)?;

        guard
            .try_reserve(1, |(k, _v): &(K, V)| self.hasher.hash_one(k))
            .map_err(|_| ConcurrentHashMapNonblockError::Reserve(TryReserveError::Other))?;

        match guard.find_or_find_insert_slot(
            hash,
            |(k, _v)| *k == key,
            |(k, _v): &(K, V)| self.hasher.hash_one(k),
        ) {
            Ok(bucket) => Ok(Some(Entry::Occupied(OccupiedEntry { guard, bucket }))),
            Err(slot) => Ok(Some(Entry::Vacant(VacantEntry {
                guard,
                key,
                hash,
                slot,
            }))),
        }
    }

    fn do_entry_give_back(&self, key: K) -> Result<Entry<'_, K, V>, (K, ConcurrentHashMapError)>
    where
        K: Eq + Hash,
    {
        let hash = self.compute_hash_internal(&key);
        let idx = self.shard_index_internal(hash);
        let shard = self.shards.get_shard(idx);
        let mut guard = shard.write_table();
        if guard
            .try_reserve(1, |(k, _v): &(K, V)| self.hasher.hash_one(k))
            .is_err()
        {
            return Err((key, ConcurrentHashMapError::Reserve(TryReserveError::Other)));
        }

        match guard.find_or_find_insert_slot(
            hash,
            |(k, _v)| *k == key,
            |(k, _v): &(K, V)| self.hasher.hash_one(k),
        ) {
            Ok(bucket) => Ok(Entry::Occupied(OccupiedEntry { guard, bucket })),
            Err(slot) => Ok(Entry::Vacant(VacantEntry {
                guard,
                key,
                hash,
                slot,
            })),
        }
    }
}

// ── Debug for ConcurrentHashMap ────────────────────────────────────────────────

impl<K, V, S> fmt::Debug for ConcurrentHashMap<K, V, S>
where
    K: fmt::Debug + Eq + Hash,
    V: fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();
        self.visit_all(|k, v| {
            map.entry(k, v);
        });
        map.finish()
    }
}

// ── Clone for ConcurrentHashMap ────────────────────────────────────────────────

impl<K, V, S> Clone for ConcurrentHashMap<K, V, S>
where
    K: Eq + Hash + Clone,
    V: Clone,
    S: BuildHasher + Clone,
{
    fn clone(&self) -> Self {
        let hasher = self.hasher.clone();
        let shard_count = self.shard_count();
        let out = Self::try_with_hasher_and_shards(hasher, shard_count)
            .expect("infallible Clone implementation of ConcurrentHashMap should not fail");
        self.visit_all(|k, v| {
            out.try_insert(k.clone(), v.clone())
                .expect("infallible Clone implementation of ConcurrentHashMap should not fail");
        });
        out
    }
}

// ── TryDebug for ConcurrentHashMap ────────────────────────────────────────────

impl<K, V, S> TryDebug for ConcurrentHashMap<K, V, S>
where
    K: TryDebug + Eq + Hash,
    V: TryDebug,
    S: BuildHasher,
{
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ConcurrentHashMap { ")?;
        let mut first = true;
        for i in 0..self.shard_count() {
            let shard = self.shards.get_shard(i);
            let table = shard.read_table();
            for bucket in unsafe { table.iter() } {
                if !first {
                    f.write_str(", ")?;
                }
                first = false;
                let kv = unsafe { bucket.as_ref() };
                TryDebug::try_fmt(&kv.0, f)?;
                f.write_str(": ")?;
                TryDebug::try_fmt(&kv.1, f)?;
            }
        }
        f.write_str(" }")
    }
}

// ── TryClone for ConcurrentHashMap ────────────────────────────────────────────

impl<K, V, S> TryClone for ConcurrentHashMap<K, V, S>
where
    K: Eq + Hash + TryClone,
    V: TryClone,
    S: BuildHasher + TryClone,
{
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let hasher = self.hasher.try_clone()?;
        let shard_count = self.shard_count();
        let out = Self::try_with_hasher_and_shards(hasher, shard_count).map_err(|e| match e {
            ConcurrentHashMapError::Alloc(a) => TryCloneError::Alloc(a),
            ConcurrentHashMapError::Reserve(r) => TryCloneError::Reserve(r),
            ConcurrentHashMapError::Clone(c) => c,
            ConcurrentHashMapError::Overflow => TryCloneError::Overflow,
            ConcurrentHashMapError::Other(m) => TryCloneError::Other(m),
        })?;

        for i in 0..self.shard_count() {
            let shard = self.shards.get_shard(i);
            let table = shard.read_table();
            // SAFETY: guard holds read lock, preventing mutation during iteration
            for bucket in unsafe { table.iter() } {
                let kv = unsafe { bucket.as_ref() };
                let k = kv.0.try_clone()?;
                let v = kv.1.try_clone()?;
                out.try_insert(k, v).map_err(|e| match e {
                    ConcurrentHashMapError::Alloc(a) => TryCloneError::Alloc(a),
                    ConcurrentHashMapError::Reserve(r) => TryCloneError::Reserve(r),
                    ConcurrentHashMapError::Clone(c) => c,
                    ConcurrentHashMapError::Overflow => TryCloneError::Overflow,
                    ConcurrentHashMapError::Other(m) => TryCloneError::Other(m),
                })?;
            }
        }
        Ok(out)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use lang_alloc::string::String;
    use lang_alloc::string::ToString;
    use lang_alloc::vec;
    use lang_alloc::vec::Vec;
    use lang_std::cell::Cell;
    use lang_std::sync::Arc;
    use lang_std::thread;

    #[test]
    fn try_new_creates_map() {
        let map: ConcurrentHashMap<u32, String> = ConcurrentHashMap::try_new().unwrap();
        assert!(map.is_empty());
        assert!(map.shard_count() > 0);
    }

    #[test]
    fn try_insert_and_get() {
        let map: ConcurrentHashMap<&str, i32> = ConcurrentHashMap::try_new().unwrap();
        map.try_insert("alpha", 1).unwrap();
        map.try_insert("beta", 2).unwrap();
        assert_eq!(*map.get(&"alpha").unwrap(), 1);
        assert_eq!(*map.get(&"beta").unwrap(), 2);
        assert!(map.get(&"gamma").is_none());
    }

    #[test]
    fn try_insert_overwrites() {
        let map: ConcurrentHashMap<&str, i32> = ConcurrentHashMap::try_new().unwrap();
        let prev = map.try_insert("key", 10).unwrap();
        assert!(prev.is_none());
        let prev = map.try_insert("key", 20).unwrap();
        assert_eq!(prev, Some(10));
        assert_eq!(*map.get(&"key").unwrap(), 20);
    }

    #[test]
    fn try_remove() {
        let map: ConcurrentHashMap<&str, i32> = ConcurrentHashMap::try_new().unwrap();
        map.try_insert("x", 42).unwrap();
        let val = map.try_remove("x").unwrap();
        assert_eq!(val, Some(42));
        assert!(map.get(&"x").is_none());
        let val = map.try_remove("x").unwrap();
        assert!(val.is_none());
    }

    #[test]
    fn try_remove_entry() {
        let map: ConcurrentHashMap<String, i32> = ConcurrentHashMap::try_new().unwrap();
        map.try_insert("hello".to_string(), 7).unwrap();
        let entry = map.try_remove_entry("hello".to_string()).unwrap();
        assert_eq!(entry, Some(("hello".to_string(), 7)));
    }

    #[test]
    fn contains_key() {
        let map: ConcurrentHashMap<i32, &str> = ConcurrentHashMap::try_new().unwrap();
        map.try_insert(1, "one").unwrap();
        assert!(map.contains_key(&1));
        assert!(!map.contains_key(&99));
    }

    #[test]
    fn entry_vacant_insert() {
        let map: ConcurrentHashMap<&str, u64> = ConcurrentHashMap::try_new().unwrap();
        {
            let entry = map.try_entry("new").unwrap();
            match entry {
                Entry::Vacant(e) => {
                    assert_eq!(*e.key(), "new");
                    e.insert(42);
                }
                Entry::Occupied(_) => panic!("expected vacant"),
            }
        }
        assert_eq!(*map.get(&"new").unwrap(), 42);
    }

    #[test]
    fn entry_occupied_modify() {
        let map: ConcurrentHashMap<&str, u64> = ConcurrentHashMap::try_new().unwrap();
        map.try_insert("existing", 100).unwrap();
        {
            let entry = map.try_entry("existing").unwrap();
            match entry {
                Entry::Occupied(mut e) => {
                    assert_eq!(*e.get(), 100);
                    e.insert(200);
                }
                Entry::Vacant(_) => panic!("expected occupied"),
            }
        }
        assert_eq!(*map.get(&"existing").unwrap(), 200);
    }

    #[test]
    fn entry_or_insert() {
        let map: ConcurrentHashMap<&str, Vec<i32>> = ConcurrentHashMap::try_new().unwrap();
        {
            let v = map.try_entry("a").unwrap().or_insert(vec![1]);
            assert_eq!(&*v, &[1]);
        }
        {
            let v = map.try_entry("a").unwrap().or_insert(vec![2, 3]);
            assert_eq!(&*v, &[1]);
        }
    }

    #[test]
    fn entry_or_insert_with() {
        let map: ConcurrentHashMap<&str, usize> = ConcurrentHashMap::try_new().unwrap();
        let called = Cell::new(false);
        {
            map.try_entry("b").unwrap().or_insert_with(|| {
                called.set(true);
                99
            });
        }
        assert!(called.get());
        called.set(false);
        {
            map.try_entry("b")
                .unwrap()
                .or_insert_with(|| panic!("should not be called"));
        }
        assert!(!called.get());
    }

    #[test]
    fn entry_and_modify() {
        let map: ConcurrentHashMap<&str, String> = ConcurrentHashMap::try_new().unwrap();
        map.try_insert("greet", "hi".to_string()).unwrap();
        map.try_entry("greet")
            .unwrap()
            .and_modify(|s: &mut String| s.push('!'));
        assert_eq!(*map.get(&"greet").unwrap(), "hi!".to_string());
        map.try_entry("missing")
            .unwrap()
            .and_modify(|_| panic!("should not run"));
    }

    #[test]
    fn occupied_entry_remove() {
        let map: ConcurrentHashMap<&str, i32> = ConcurrentHashMap::try_new().unwrap();
        map.try_insert("k", 55).unwrap();
        let val = map.try_entry("k").unwrap();
        if let Entry::Occupied(e) = val {
            assert_eq!(e.remove(), 55);
        } else {
            panic!("expected occupied");
        }
        assert!(map.get(&"k").is_none());
    }

    #[test]
    fn occupied_entry_remove_entry() {
        let map: ConcurrentHashMap<String, i32> = ConcurrentHashMap::try_new().unwrap();
        map.try_insert("item".to_string(), 3).unwrap();
        let entry = map.try_entry("item".to_string()).unwrap();
        if let Entry::Occupied(e) = entry {
            let (k, v): (String, i32) = e.remove_entry();
            assert_eq!(k, "item");
            assert_eq!(v, 3);
        }
        assert!(!map.contains_key(&"item".to_string()));
    }

    #[test]
    fn try_insert_give_back_on_success() {
        let map: ConcurrentHashMap<&str, i32> = ConcurrentHashMap::try_new().unwrap();
        let result = map.try_insert_give_back("x", 1).unwrap();
        assert!(result.is_none());
        assert_eq!(*map.get(&"x").unwrap(), 1);
    }

    #[test]
    fn try_insert_unique() {
        let map: ConcurrentHashMap<String, i32> = ConcurrentHashMap::try_new().unwrap();
        map.try_insert_unique("unique".to_string(), 1).unwrap();
        let result = map.try_insert_unique("unique".to_string(), 2);
        assert!(result.is_err());
    }

    #[test]
    fn try_reserve() {
        let map: ConcurrentHashMap<i32, i32> = ConcurrentHashMap::try_new().unwrap();
        map.try_reserve(1000).unwrap();
    }

    #[test]
    fn len_accuracy() {
        let map: ConcurrentHashMap<i32, i32> = ConcurrentHashMap::try_new().unwrap();
        assert_eq!(map.len(), 0);
        for i in 0..50 {
            map.try_insert(i, i * 10).unwrap();
        }
        assert_eq!(map.len(), 50);
    }

    #[test]
    fn concurrent_insert_and_read() {
        let map: Arc<ConcurrentHashMap<i32, i32>> = Arc::new(ConcurrentHashMap::try_new().unwrap());
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let map = Arc::clone(&map);
                thread::spawn(move || {
                    for j in 0..100 {
                        let key = i * 100 + j;
                        map.try_insert(key, key * 2).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(map.len(), 800);
        for i in 0..8 {
            for j in 0..100 {
                let key = i * 100 + j;
                assert_eq!(*map.get(&key).unwrap(), key * 2);
            }
        }
    }

    #[test]
    fn with_custom_shard_count() {
        let map: ConcurrentHashMap<i32, i32> = ConcurrentHashMap::try_with_shards(8).unwrap();
        assert_eq!(map.shard_count(), 8);
    }

    #[test]
    fn zero_shards_rejected() {
        let result: Result<ConcurrentHashMap<i32, i32>, _> = ConcurrentHashMap::try_with_shards(0);
        assert!(result.is_err());
    }

    #[test]
    fn non_power_of_two_shards_rejected() {
        let result: Result<ConcurrentHashMap<i32, i32>, _> = ConcurrentHashMap::try_with_shards(7);
        assert!(result.is_err());
    }

    #[test]
    fn get_mut_works() {
        let map: ConcurrentHashMap<&str, i32> = ConcurrentHashMap::try_new().unwrap();
        map.try_insert("counter", 0).unwrap();
        {
            let mut val = map.get_mut(&"counter").unwrap();
            *val += 1;
        }
        assert_eq!(*map.get(&"counter").unwrap(), 1);
    }

    #[test]
    fn hasher_accessible() {
        let map: ConcurrentHashMap<i32, i32> = ConcurrentHashMap::try_new().unwrap();
        let _hasher = map.hasher();
    }

    #[test]
    fn static_map_via_macro() {
        let map = &*super::super::__test_static_map::TEST_CHASHMAP_STATIC;
        map.try_insert(9999, "macro_test".to_string()).unwrap();
        assert_eq!(*map.get(&9999).unwrap(), "macro_test".to_string());
    }

    #[test]
    fn ref_guard_derefs_correctly() {
        let map: ConcurrentHashMap<&str, i32> = ConcurrentHashMap::try_new().unwrap();
        map.try_insert("x", 77).unwrap();
        let ref_ = map.get(&"x").unwrap();
        assert_eq!(*ref_, 77);
        assert_eq!(*ref_.value(), 77);
    }

    #[test]
    fn refmut_guard_derefs_and_key() {
        let map: ConcurrentHashMap<&str, String> = ConcurrentHashMap::try_new().unwrap();
        {
            let mut refmut = map.try_entry("k").unwrap().or_insert("initial".to_string());
            assert_eq!(refmut.key(), &"k");
            assert_eq!(&*refmut, "initial");
            refmut.push_str("_modified");
        }
        assert_eq!(*map.get(&"k").unwrap(), "initial_modified");
    }

    #[test]
    fn refmut_pair_and_pair_mut() {
        let map: ConcurrentHashMap<u32, Vec<u8>> = ConcurrentHashMap::try_new().unwrap();
        {
            let mut refmut = map.try_entry(42).unwrap().or_insert(vec![1, 2]);
            let (k, v) = refmut.pair();
            assert_eq!(*k, 42);
            assert_eq!(v, &[1, 2]);
            let (k2, v2) = refmut.pair_mut();
            assert_eq!(*k2, 42);
            v2.push(3);
        }
        assert_eq!(*map.get(&42).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn occupied_into_ref_returns_refmut() {
        let map: ConcurrentHashMap<&str, u64> = ConcurrentHashMap::try_new().unwrap();
        map.try_insert("a", 10).unwrap();
        {
            let entry = map.try_entry("a").unwrap();
            if let Entry::Occupied(e) = entry {
                let mut refmut = e.into_ref();
                assert_eq!(*refmut, 10);
                *refmut = 20;
            }
        }
        assert_eq!(*map.get(&"a").unwrap(), 20);
    }

    #[test]
    fn entry_insert_on_occupied_returns_refmut() {
        let map: ConcurrentHashMap<i32, i32> = ConcurrentHashMap::try_new().unwrap();
        map.try_insert(1, 100).unwrap();
        {
            let refmut = map.try_entry(1).unwrap().insert(200);
            assert_eq!(*refmut, 200);
        }
        assert_eq!(*map.get(&1).unwrap(), 200);
    }

    #[test]
    fn single_shard_rejected() {
        let result: Result<ConcurrentHashMap<i32, i32>, _> = ConcurrentHashMap::try_with_shards(1);
        assert!(result.is_err());
    }

    #[test]
    fn two_shard_map() {
        let map: ConcurrentHashMap<i32, i32> = ConcurrentHashMap::try_with_shards(2).unwrap();
        map.try_insert(1, 10).unwrap();
        map.try_insert(2, 20).unwrap();
        assert_eq!(*map.get(&1).unwrap(), 10);
        assert_eq!(*map.get(&2).unwrap(), 20);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn capacity_method_works() {
        let map: ConcurrentHashMap<i32, i32> = ConcurrentHashMap::try_new().unwrap();
        let initial_cap = map.capacity();
        map.try_reserve(100).unwrap();
        assert!(map.capacity() >= initial_cap);
    }

    #[test]
    fn with_hasher_api() {
        use lang_std::collections::hash_map::DefaultHasher;
        use lang_std::hash::BuildHasherDefault;
        let hasher = BuildHasherDefault::<DefaultHasher>::default();
        let map: ConcurrentHashMap<&str, i32, _> =
            ConcurrentHashMap::try_with_hasher(hasher).unwrap();
        map.try_insert("key", 42).unwrap();
        assert_eq!(*map.get(&"key").unwrap(), 42);
    }

    #[test]
    fn with_capacity_and_hasher_api() {
        use lang_std::collections::hash_map::DefaultHasher;
        use lang_std::hash::BuildHasherDefault;
        let hasher = BuildHasherDefault::<DefaultHasher>::default();
        let map: ConcurrentHashMap<&str, i32, _> =
            ConcurrentHashMap::try_with_capacity_and_hasher(50, hasher).unwrap();
        assert!(map.capacity() >= 50);
    }

    #[test]
    fn try_clone_basic() {
        let map: ConcurrentHashMap<&str, i32> = ConcurrentHashMap::try_new().unwrap();
        map.try_insert("a", 1).unwrap();
        map.try_insert("b", 2).unwrap();
        let cloned = map.try_clone().unwrap();
        assert_eq!(cloned.len(), 2);
        assert_eq!(*cloned.get(&"a").unwrap(), 1);
        assert_eq!(*cloned.get(&"b").unwrap(), 2);
    }

    #[test]
    fn try_clone_empty() {
        let map: ConcurrentHashMap<i32, i32> = ConcurrentHashMap::try_new().unwrap();
        let cloned = map.try_clone().unwrap();
        assert!(cloned.is_empty());
    }

    #[test]
    fn iter_yields_all_entries() {
        let map: ConcurrentHashMap<i32, i32> = ConcurrentHashMap::try_new().unwrap();
        for i in 0..10 {
            map.try_insert(i, i * 10).unwrap();
        }
        let mut seen = Vec::new();
        for result in map.iter() {
            let ref_ = result.unwrap();
            seen.push((*ref_.key(), *ref_.value()));
        }
        seen.sort_by_key(|&(k, _)| k);
        assert_eq!(seen.len(), 10);
        for (i, (k, v)) in seen.into_iter().enumerate() {
            assert_eq!(k, i as i32);
            assert_eq!(v, i as i32 * 10);
        }
    }

    #[test]
    fn iter_mut_yields_and_allows_mutation() {
        let map: ConcurrentHashMap<i32, i32> = ConcurrentHashMap::try_new().unwrap();
        map.try_insert(1, 10).unwrap();
        map.try_insert(2, 20).unwrap();
        for result in map.iter_mut() {
            let mut refmut = result.unwrap();
            *refmut *= 2;
        }
        assert_eq!(*map.get(&1).unwrap(), 20);
        assert_eq!(*map.get(&2).unwrap(), 40);
    }

    #[test]
    fn iter_empty_map() {
        let map: ConcurrentHashMap<i32, i32> = ConcurrentHashMap::try_new().unwrap();
        assert!(map.iter().next().is_none());
        assert!(map.iter_mut().next().is_none());
    }
}
