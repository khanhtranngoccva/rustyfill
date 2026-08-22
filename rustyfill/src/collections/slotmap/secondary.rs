//! Secondary map — associate extra data with slot-map keys.
//!
//! A [`SecondaryMap`] stores additional information keyed by handles from a
//! [`SlotMap`](crate::collections::slotmap::SlotMap). Unlike a `HashMap`,
//! it uses direct indexing: the key's slot index is used as the array offset,
//! so lookups are O(1) without hashing.

use crate::alloc::AllocError;
use crate::alloc::TryReserveError;
use crate::alloc::vec::TryVec;
use crate::collections::slotmap::key::{Key, KeyData, MAX_SLOTS_LEN};
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::{TryDebug, TryDisplay};
use crate::try_fmt::helpers::FormatterExt;
use lang_alloc::vec::Vec;
use lang_core::fmt::{self, Debug};
use lang_core::iter::{Enumerate, Extend, FromIterator, FusedIterator};
use lang_core::marker::PhantomData;
use lang_core::mem::replace;
use lang_core::num::NonZeroU32;
use lang_core::ops::{Index, IndexMut};

/// Returns true if `a` is an older version than `b`, accounting for wraparound.
fn is_older_version(a: u32, b: u32) -> bool {
    let diff = a.wrapping_sub(b);
    diff >= (1 << 31)
}

// ── Internal slot ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Slot<T> {
    Occupied { value: T, version: NonZeroU32 },
    Vacant,
}

use self::Slot::{Occupied, Vacant};

impl<T> Slot<T> {
    fn new_occupied(version: u32, value: T) -> Self {
        Occupied {
            value,
            version: unsafe { NonZeroU32::new_unchecked(version | 1u32) },
        }
    }

    fn new_vacant() -> Self {
        Vacant
    }

    #[inline(always)]
    fn version(&self) -> u32 {
        match self {
            Occupied { version, .. } => version.get(),
            Vacant => 0,
        }
    }

    pub(crate) unsafe fn get_unchecked(&self) -> &T {
        unsafe {
            match self {
                Occupied { value, .. } => value,
                Vacant => lang_core::hint::unreachable_unchecked(),
            }
        }
    }

    pub(crate) unsafe fn get_unchecked_mut(&mut self) -> &mut T {
        unsafe {
            match self {
                Occupied { value, .. } => value,
                Vacant => lang_core::hint::unreachable_unchecked(),
            }
        }
    }

    fn into_option(self) -> Option<T> {
        match self {
            Occupied { value, .. } => Some(value),
            Vacant => None,
        }
    }
}

impl<T: TryClone> TryClone for Slot<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(match self {
            Occupied { value, version } => Occupied {
                value: value.try_clone()?,
                version: *version,
            },
            Vacant => Vacant,
        })
    }
}

// ── Error type ──────────────────────────────────────────────────────────────────

/// Error returned by [`SecondaryMap`] operations.
pub enum SecondaryMapError {
    /// A raw heap allocation failed.
    Alloc(AllocError),
    /// Capacity reservation failed.
    Reserve(TryReserveError),
    /// The secondary map has grown too large.
    Overflow,
}

impl fmt::Debug for SecondaryMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(self, f)
    }
}

impl fmt::Display for SecondaryMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }
}

impl TryDebug for SecondaryMapError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => f
                .try_debug_tuple("SecondaryMapError::Alloc")
                .field(e)
                .finish(),
            Self::Reserve(e) => f
                .try_debug_tuple("SecondaryMapError::Reserve")
                .field(e)
                .finish(),
            Self::Overflow => f.write_str("SecondaryMapError::Overflow"),
        }
    }
}

impl TryDisplay for SecondaryMapError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => write!(f, "secondary map allocation failed: {}", e),
            Self::Reserve(e) => write!(f, "secondary map capacity reservation failed: {}", e),
            Self::Overflow => f.write_str("secondary map overflow"),
        }
    }
}

impl From<AllocError> for SecondaryMapError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

impl From<TryReserveError> for SecondaryMapError {
    fn from(e: TryReserveError) -> Self {
        Self::Reserve(e)
    }
}

// ── SecondaryMap ────────────────────────────────────────────────────────────────

/// Secondary map that associates extra data with keys from a [`SlotMap`](crate::collections::slotmap::SlotMap).
///
/// Uses direct indexing (no hashing) so lookups are O(1). Outdated entries are
/// cleaned up lazily when their slot is reused in the primary map.
#[derive(Debug, Clone)]
pub struct SecondaryMap<K: Key, V> {
    slots: Vec<Slot<V>>,
    num_elems: usize,
    _k: PhantomData<fn(K) -> K>,
}

impl<K: Key, V> SecondaryMap<K, V> {
    /// Constructs a new, empty [`SecondaryMap`].
    ///
    /// Fallible because even an empty map allocates space for the sentinel slot.
    pub fn try_new() -> Result<Self, SecondaryMapError> {
        Self::try_with_capacity(0)
    }

    /// Creates a [`SecondaryMap`] with the given capacity.
    ///
    /// Returns [`SecondaryMapError::Overflow`] if `capacity` would exceed the
    /// maximum number of usable slots (`MAX_SLOTS_LEN` – 1), accounting for
    /// the sentinel that always occupies index 0.
    pub fn try_with_capacity(capacity: usize) -> Result<Self, SecondaryMapError> {
        if capacity >= MAX_SLOTS_LEN.saturating_sub(1) {
            return Err(SecondaryMapError::Overflow);
        }
        // Safe: `capacity < MAX_SLOTS_LEN - 1 <= usize::MAX`, so `+1` cannot overflow.
        let mut slots = Vec::<Slot<V>>::fallible_with_capacity(
            capacity
                .checked_add(1)
                .expect("capacity below MAX_SLOTS_LEN"),
        )
        .map_err(SecondaryMapError::from)?;
        slots
            .try_push(Slot::new_vacant())
            .map_err(SecondaryMapError::from)?;
        Ok(Self {
            slots,
            num_elems: 0,
            _k: PhantomData,
        })
    }

    /// Returns the number of elements in the secondary map.
    pub fn len(&self) -> usize {
        self.num_elems
    }

    /// Returns `true` if the secondary map contains no elements.
    pub fn is_empty(&self) -> bool {
        self.num_elems == 0
    }

    /// Returns the number of elements the secondary map can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.slots.capacity().saturating_sub(1)
    }

    /// Tries to set the capacity to at least `new_capacity`.
    ///
    /// Returns [`SecondaryMapError::Overflow`] if `new_capacity` would exceed
    /// the maximum number of usable slots (`MAX_SLOTS_LEN` – 1).
    pub fn try_set_capacity(&mut self, new_capacity: usize) -> Result<(), SecondaryMapError> {
        if new_capacity >= MAX_SLOTS_LEN.saturating_sub(1) {
            return Err(SecondaryMapError::Overflow);
        }
        // Safe: `new_capacity < MAX_SLOTS_LEN - 1 <= usize::MAX`, so `+1` cannot overflow.
        let target = new_capacity
            .checked_add(1)
            .expect("capacity below MAX_SLOTS_LEN"); // sentinel
        if target > self.slots.capacity() {
            let needed = target.saturating_sub(self.slots.len());
            self.slots
                .try_reserve(needed)
                .map_err(SecondaryMapError::from)?;
        }
        Ok(())
    }

    /// Returns `true` if the secondary map contains the given key.
    pub fn contains_key(&self, key: K) -> bool {
        let kd = key.data();
        self.slots
            .get(kd.idx() as usize)
            .is_some_and(|slot| slot.version() == kd.version_raw())
    }

    /// Inserts a value at the given key.
    ///
    /// Returns `Err` if the underlying allocation fails or the index exceeds the
    /// storage limit. On failure, `value` is dropped. For a variant that gives
    /// `value` back, use [`Self::try_insert_give_back`].
    ///
    /// Returns `Ok(None)` if this is a new entry, `Ok(Some(old_value))` if the
    /// key was already present. Silently returns `Ok(None)` if the key was
    /// removed from the originating slot map and its slot has been reused with a
    /// newer version—in that case `value` is dropped as the insert was logically
    /// successful (a no-op).
    pub fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, SecondaryMapError> {
        match self.try_insert_give_back(key, value) {
            Ok(v) => Ok(v),
            Err((_, e)) => Err(e),
        }
    }

    /// Like [`Self::try_insert`] but returns ownership of `value` back on failure.
    ///
    /// Returns `Err((value, error))` if the underlying allocation fails or the
    /// index exceeds the storage limit, giving the unconsumed `value` back to the
    /// caller. Returns `Ok(None)` if this is a new entry, `Ok(Some(old_value))`
    /// if the key was already present.
    ///
    /// Silently returns `Ok(None)` if the key was removed from the originating
    /// slot map and its slot has been reused with a newer version. In this case
    /// `value` is dropped, as the insert was logically successful (a no-op).
    pub fn try_insert_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<Option<V>, (V, SecondaryMapError)> {
        match self.try_entry(key) {
            Ok(None) => {
                // Key was removed from the primary map and its slot reused.
                // Drop the value — this is a logical no-op, not an error.
                Ok(None)
            }
            Ok(Some(Entry::Occupied(mut entry))) => Ok(Some(entry.insert(value))),
            Ok(Some(Entry::Vacant(entry))) => {
                entry.insert(value);
                Ok(None)
            }
            Err(e) => Err((value, e)),
        }
    }

    /// Removes a key, returning the value if present.
    pub fn remove(&mut self, key: K) -> Option<V> {
        let kd = key.data();
        if let Some(slot) = self.slots.get_mut(kd.idx() as usize)
            && slot.version() == kd.version_raw()
        {
            // Safe: the key exists, so at least one element is present.
            let num_elems = self
                .num_elems
                .checked_sub(1)
                .expect("at least one element present");
            self.num_elems = num_elems;
            return replace(slot, Slot::new_vacant()).into_option();
        }
        None
    }

    /// Retains only elements satisfying the predicate.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(K, &mut V) -> bool,
    {
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if let Occupied { value, version } = slot {
                let key = KeyData::new(i as u32, version.get()).into();
                if !f(key, value) {
                    // Safe: the slot is occupied, so at least one element is present.
                    let num_elems = self
                        .num_elems
                        .checked_sub(1)
                        .expect("at least one element present");
                    self.num_elems = num_elems;
                    *slot = Slot::new_vacant();
                }
            }
        }
    }

    /// Clears the secondary map, keeping allocated memory.
    pub fn clear(&mut self) {
        self.drain();
    }

    /// Drains all elements, returning them as an iterator.
    pub fn drain(&mut self) -> Drain<'_, K, V> {
        Drain { cur: 1, sm: self }
    }

    // ── Accessors ──────────────────────────────────────────────────────────────

    /// Returns a reference to the value for the given key.
    pub fn get(&self, key: K) -> Option<&V> {
        let kd = key.data();
        self.slots
            .get(kd.idx() as usize)
            .filter(|slot| slot.version() == kd.version_raw())
            .map(|slot| unsafe { slot.get_unchecked() })
    }

    /// Returns a mutable reference to the value for the given key.
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        let kd = key.data();
        self.slots
            .get_mut(kd.idx() as usize)
            .filter(|slot| slot.version() == kd.version_raw())
            .map(|slot| unsafe { slot.get_unchecked_mut() })
    }

    /// Unchecked access.
    ///
    /// # Safety
    /// Caller must ensure `contains_key(key)` is true.
    pub unsafe fn get_unchecked(&self, key: K) -> &V {
        debug_assert!(self.contains_key(key));
        unsafe {
            self.slots
                .get_unchecked(key.data().idx() as usize)
                .get_unchecked()
        }
    }

    /// Unchecked mutable access.
    ///
    /// # Safety
    /// Caller must ensure `contains_key(key)` is true.
    pub unsafe fn get_unchecked_mut(&mut self, key: K) -> &mut V {
        debug_assert!(self.contains_key(key));
        unsafe {
            self.slots
                .get_unchecked_mut(key.data().idx() as usize)
                .get_unchecked_mut()
        }
    }

    /// Fallible entry API — returns `None` if the key was removed from the
    /// originating map, or `Err` if the underlying allocation fails.
    pub fn try_entry(&mut self, key: K) -> Result<Option<Entry<'_, K, V>>, SecondaryMapError> {
        if key.is_null() {
            return Ok(None);
        }

        let kd = key.data();
        let idx = kd.idx() as usize;

        // Guard against indices that would exceed our storage limit.
        if idx >= MAX_SLOTS_LEN.saturating_sub(1) {
            return Err(SecondaryMapError::Overflow);
        }

        // Ensure slot exists, growing fallibly in a single allocation.
        // Safe: `idx < MAX_SLOTS_LEN - 1 <= usize::MAX`, so `+1` cannot overflow.
        let target_len = idx.checked_add(1).expect("slot index below MAX_SLOTS_LEN");
        if self.slots.len() < target_len {
            self.slots
                .try_resize_with(target_len, Slot::new_vacant)
                .map_err(SecondaryMapError::from)?;
        }

        let slot = &self.slots[idx];
        if kd.version_raw() == slot.version() {
            Ok(Some(Entry::Occupied(OccupiedEntry {
                map: self,
                kd,
                _k: PhantomData,
            })))
        } else if is_older_version(kd.version_raw(), slot.version()) {
            Ok(None)
        } else {
            Ok(Some(Entry::Vacant(VacantEntry {
                map: self,
                kd,
                _k: PhantomData,
            })))
        }
    }

    // ── Aliases with `fallible_` prefix ────────────────────────────────────────

    /// Alias for [`Self::try_new`].
    pub fn fallible_new() -> Result<Self, SecondaryMapError> {
        Self::try_new()
    }

    /// Alias for [`Self::try_with_capacity`].
    pub fn fallible_with_capacity(capacity: usize) -> Result<Self, SecondaryMapError> {
        Self::try_with_capacity(capacity)
    }

    /// Alias for [`Self::try_set_capacity`].
    pub fn fallible_set_capacity(&mut self, new_capacity: usize) -> Result<(), SecondaryMapError> {
        Self::try_set_capacity(self, new_capacity)
    }

    /// Alias for [`Self::try_insert`].
    pub fn fallible_insert(&mut self, key: K, value: V) -> Result<Option<V>, SecondaryMapError> {
        Self::try_insert(self, key, value)
    }

    /// Alias for [`Self::try_insert_give_back`].
    pub fn fallible_insert_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<Option<V>, (V, SecondaryMapError)> {
        Self::try_insert_give_back(self, key, value)
    }

    /// Alias for [`Self::try_entry`].
    pub fn fallible_entry(&mut self, key: K) -> Result<Option<Entry<'_, K, V>>, SecondaryMapError> {
        Self::try_entry(self, key)
    }

    // ── Iterators ──────────────────────────────────────────────────────────────

    /// Immutable iterator over all key-value pairs.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            num_left: self.num_elems,
            slots: self.slots.iter().enumerate(),
            _k: PhantomData,
        }
    }

    /// Mutable iterator over all key-value pairs.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            num_left: self.num_elems,
            slots: self.slots.iter_mut().enumerate(),
            _k: PhantomData,
        }
    }

    /// Iterator over all keys.
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { inner: self.iter() }
    }

    /// Iterator over all values (immutable).
    pub fn values(&self) -> Values<'_, K, V> {
        Values { inner: self.iter() }
    }

    /// Iterator over all values (mutable).
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut {
            inner: self.iter_mut(),
        }
    }
}

impl<K: Key, V> Default for SecondaryMap<K, V> {
    /// Constructs an empty [`SecondaryMap`].
    ///
    /// # Panics
    ///
    /// Panics if the sentinel slot allocation fails. This is only reachable on
    /// catastrophic OOM and is consistent with Rust's `Default` contract, which
    /// cannot return errors. Prefer [`Self::try_new`] or [`Self::try_with_capacity`]
    /// for fallible construction.
    fn default() -> Self {
        Self::try_with_capacity(0)
            .expect("SecondaryMap::default panicked: failed to allocate sentinel slot")
    }
}

impl<K: Key, V> TryClone for SecondaryMap<K, V>
where
    V: TryClone,
{
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let slots = self.slots.try_clone()?;
        Ok(Self {
            slots,
            num_elems: self.num_elems,
            _k: PhantomData,
        })
    }
}

impl<K: Key, V> TryDefault for SecondaryMap<K, V> {
    fn try_default() -> Result<Self, TryDefaultError> {
        Self::try_with_capacity(0).map_err(|e| match e {
            SecondaryMapError::Alloc(a) => TryDefaultError::Alloc(a),
            SecondaryMapError::Reserve(r) => TryDefaultError::Reserve(r),
            SecondaryMapError::Overflow => TryDefaultError::Overflow,
        })
    }
}

impl<K: Key + TryDebug, V: TryDebug> TryDebug for SecondaryMap<K, V> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::try_fmt::helpers::FormatterExt;
        let mut map = f.try_debug_map();
        for (k, v) in self.iter() {
            map.entry(&k, v);
        }
        map.finish()
    }
}

impl<K: Key, V> Index<K> for SecondaryMap<K, V> {
    type Output = V;

    fn index(&self, key: K) -> &V {
        match self.get(key) {
            Some(r) => r,
            None => panic!("invalid SecondaryMap key used"),
        }
    }
}

impl<K: Key, V> IndexMut<K> for SecondaryMap<K, V> {
    fn index_mut(&mut self, key: K) -> &mut V {
        match self.get_mut(key) {
            Some(r) => r,
            None => panic!("invalid SecondaryMap key used"),
        }
    }
}

impl<K: Key, V: PartialEq> PartialEq for SecondaryMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter()
            .all(|(key, value)| other.get(key).is_some_and(|ov| *value == *ov))
    }
}

impl<K: Key, V: Eq> Eq for SecondaryMap<K, V> {}

impl<K: Key, V> FromIterator<(K, V)> for SecondaryMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        // SAFETY: constructing an empty map only allocates a sentinel slot.
        // Panics here are consistent with Rust's infallible trait contract.
        let mut sec = Self::try_new()
            .expect("SecondaryMap::from_iter panicked: failed to allocate sentinel slot");
        sec.extend(iter);
        sec
    }
}

impl<K: Key, V> Extend<(K, V)> for SecondaryMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            let _ = self.try_insert(k, v);
        }
    }
}

impl<'a, K: Key, V: 'a + Copy> Extend<(K, &'a V)> for SecondaryMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, &'a V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            let _ = self.try_insert(k, *v);
        }
    }
}

// ── Entry API ───────────────────────────────────────────────────────────────────

/// View into an occupied entry in a [`SecondaryMap`].
#[derive(Debug)]
pub struct OccupiedEntry<'a, K: Key, V> {
    map: &'a mut SecondaryMap<K, V>,
    kd: KeyData,
    _k: PhantomData<fn(K) -> K>,
}

/// View into a vacant entry in a [`SecondaryMap`].
#[derive(Debug)]
pub struct VacantEntry<'a, K: Key, V> {
    map: &'a mut SecondaryMap<K, V>,
    kd: KeyData,
    _k: PhantomData<fn(K) -> K>,
}

/// Entry enum for in-place manipulation.
#[derive(Debug)]
pub enum Entry<'a, K: Key, V> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K: Key, V> Entry<'a, K, V> {
    pub fn or_insert(self, default: V) -> &'a mut V {
        self.or_insert_with(|| default)
    }

    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(x) => x.into_mut(),
            Entry::Vacant(x) => x.insert(default()),
        }
    }

    pub fn key(&self) -> K {
        match self {
            Entry::Occupied(e) => e.kd.into(),
            Entry::Vacant(e) => e.kd.into(),
        }
    }

    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        match self {
            Entry::Occupied(mut e) => {
                f(e.get_mut());
                Entry::Occupied(e)
            }
            Entry::Vacant(e) => Entry::Vacant(e),
        }
    }
}

impl<'a, K: Key, V: Default> Entry<'a, K, V> {
    pub fn or_default(self) -> &'a mut V {
        self.or_insert_with(Default::default)
    }
}

impl<'a, K: Key, V> OccupiedEntry<'a, K, V> {
    pub fn key(&self) -> K {
        self.kd.into()
    }

    pub fn remove_entry(self) -> (K, V) {
        (self.kd.into(), self.remove())
    }

    pub fn get(&self) -> &V {
        unsafe { self.map.get_unchecked(self.kd.into()) }
    }

    pub fn get_mut(&mut self) -> &mut V {
        unsafe { self.map.get_unchecked_mut(self.kd.into()) }
    }

    pub fn into_mut(self) -> &'a mut V {
        unsafe { self.map.get_unchecked_mut(self.kd.into()) }
    }

    pub fn insert(&mut self, value: V) -> V {
        replace(self.get_mut(), value)
    }

    pub fn remove(self) -> V {
        let slot = unsafe { self.map.slots.get_unchecked_mut(self.kd.idx() as usize) };
        // Safe: removing an occupied entry implies at least one element is present.
        let num_elems = self
            .map
            .num_elems
            .checked_sub(1)
            .expect("at least one element present");
        self.map.num_elems = num_elems;
        unsafe {
            match replace(slot, Slot::new_vacant()) {
                Occupied { value, .. } => value,
                Vacant => lang_core::hint::unreachable_unchecked(),
            }
        }
    }
}

impl<'a, K: Key, V> VacantEntry<'a, K, V> {
    pub fn key(&self) -> K {
        self.kd.into()
    }

    pub fn insert(self, value: V) -> &'a mut V {
        let slot = unsafe { self.map.slots.get_unchecked_mut(self.kd.idx() as usize) };
        match replace(slot, Slot::new_occupied(self.kd.version_raw(), value)) {
            Occupied { .. } => {}
            Vacant => {
                // Safe: the entry index is bounded by `MAX_SLOTS_LEN`, so the count
                // cannot exceed `usize::MAX`.
                let num_elems = self
                    .map
                    .num_elems
                    .checked_add(1)
                    .expect("element count below MAX_SLOTS_LEN");
                self.map.num_elems = num_elems;
            }
        }
        unsafe { slot.get_unchecked_mut() }
    }
}

// ── Iterators ───────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct Drain<'a, K: Key + 'a, V: 'a> {
    sm: &'a mut SecondaryMap<K, V>,
    cur: usize,
}

#[derive(Debug)]
pub struct IntoIter<K: Key, V> {
    num_left: usize,
    slots: Enumerate<lang_alloc::vec::IntoIter<Slot<V>>>,
    _k: PhantomData<fn(K) -> K>,
}

#[derive(Debug)]
pub struct Iter<'a, K: Key + 'a, V: 'a> {
    num_left: usize,
    slots: Enumerate<lang_core::slice::Iter<'a, Slot<V>>>,
    _k: PhantomData<fn(K) -> K>,
}

impl<'a, K: 'a + Key, V: 'a> Clone for Iter<'a, K, V> {
    fn clone(&self) -> Self {
        Iter {
            num_left: self.num_left,
            slots: self.slots.clone(),
            _k: self._k,
        }
    }
}

#[derive(Debug)]
pub struct IterMut<'a, K: Key + 'a, V: 'a> {
    num_left: usize,
    slots: Enumerate<lang_core::slice::IterMut<'a, Slot<V>>>,
    _k: PhantomData<fn(K) -> K>,
}

#[derive(Debug)]
pub struct Keys<'a, K: Key + 'a, V: 'a> {
    inner: Iter<'a, K, V>,
}

impl<'a, K: 'a + Key, V: 'a> Clone for Keys<'a, K, V> {
    fn clone(&self) -> Self {
        Keys {
            inner: self.inner.clone(),
        }
    }
}

#[derive(Debug)]
pub struct Values<'a, K: Key + 'a, V: 'a> {
    inner: Iter<'a, K, V>,
}

impl<'a, K: 'a + Key, V: 'a> Clone for Values<'a, K, V> {
    fn clone(&self) -> Self {
        Values {
            inner: self.inner.clone(),
        }
    }
}

#[derive(Debug)]
pub struct ValuesMut<'a, K: Key + 'a, V: 'a> {
    inner: IterMut<'a, K, V>,
}

impl<'a, K: Key, V> Iterator for Drain<'a, K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        while let Some(slot) = self.sm.slots.get_mut(self.cur) {
            let idx = self.cur;
            // Safe: `cur` is a valid slot index, so it is below the slots length.
            let next_cur = self.cur.checked_add(1).expect("cursor below slot length");
            self.cur = next_cur;
            if let Occupied { value, version } = replace(slot, Slot::new_vacant()) {
                // Safe: the slot is occupied, so at least one element is present.
                let num_elems = self
                    .sm
                    .num_elems
                    .checked_sub(1)
                    .expect("at least one element present");
                self.sm.num_elems = num_elems;
                let key = KeyData::new(idx as u32, version.get()).into();
                return Some((key, value));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.sm.len(), Some(self.sm.len()))
    }
}

impl<'a, K: Key, V> Drop for Drain<'a, K, V> {
    fn drop(&mut self) {
        self.for_each(|_| {});
    }
}

impl<K: Key, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        for (idx, mut slot) in self.slots.by_ref() {
            if let Occupied { value, version } = replace(&mut slot, Slot::new_vacant()) {
                // Safe: `num_left` counts remaining elements and is decremented only on yield.
                let num_left = self
                    .num_left
                    .checked_sub(1)
                    .expect("remaining count positive");
                self.num_left = num_left;
                let key = KeyData::new(idx as u32, version.get()).into();
                return Some((key, value));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.num_left, Some(self.num_left))
    }
}

impl<'a, K: Key, V> Iterator for Iter<'a, K, V> {
    type Item = (K, &'a V);

    fn next(&mut self) -> Option<(K, &'a V)> {
        for (idx, slot) in self.slots.by_ref() {
            if let Occupied { value, version } = slot {
                // Safe: `num_left` counts remaining elements and is decremented only on yield.
                let num_left = self
                    .num_left
                    .checked_sub(1)
                    .expect("remaining count positive");
                self.num_left = num_left;
                let key = KeyData::new(idx as u32, version.get()).into();
                return Some((key, value));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.num_left, Some(self.num_left))
    }
}

impl<'a, K: Key, V> Iterator for IterMut<'a, K, V> {
    type Item = (K, &'a mut V);

    fn next(&mut self) -> Option<(K, &'a mut V)> {
        for (idx, slot) in self.slots.by_ref() {
            if let Occupied { value, version } = slot {
                let key = KeyData::new(idx as u32, version.get()).into();
                // Safe: `num_left` counts remaining elements and is decremented only on yield.
                let num_left = self
                    .num_left
                    .checked_sub(1)
                    .expect("remaining count positive");
                self.num_left = num_left;
                return Some((key, value));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.num_left, Some(self.num_left))
    }
}

impl<'a, K: Key, V> Iterator for Keys<'a, K, V> {
    type Item = K;

    fn next(&mut self) -> Option<K> {
        self.inner.next().map(|(key, _)| key)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K: Key, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<&'a V> {
        self.inner.next().map(|(_, value)| value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K: Key, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    fn next(&mut self) -> Option<&'a mut V> {
        self.inner.next().map(|(_, value)| value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K: Key, V> IntoIterator for &'a SecondaryMap<K, V> {
    type Item = (K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Key, V> IntoIterator for &'a mut SecondaryMap<K, V> {
    type Item = (K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K: Key, V> IntoIterator for SecondaryMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        let len = self.len();
        let mut it = self.slots.into_iter().enumerate();
        it.next(); // skip sentinel
        IntoIter {
            num_left: len,
            slots: it,
            _k: PhantomData,
        }
    }
}

impl<'a, K: Key, V> FusedIterator for Iter<'a, K, V> {}
impl<'a, K: Key, V> FusedIterator for IterMut<'a, K, V> {}
impl<'a, K: Key, V> FusedIterator for Keys<'a, K, V> {}
impl<'a, K: Key, V> FusedIterator for Values<'a, K, V> {}
impl<'a, K: Key, V> FusedIterator for ValuesMut<'a, K, V> {}
impl<'a, K: Key, V> FusedIterator for Drain<'a, K, V> {}
impl<K: Key, V> FusedIterator for IntoIter<K, V> {}

impl<'a, K: Key, V> ExactSizeIterator for Iter<'a, K, V> {}
impl<'a, K: Key, V> ExactSizeIterator for IterMut<'a, K, V> {}
impl<'a, K: Key, V> ExactSizeIterator for Keys<'a, K, V> {}
impl<'a, K: Key, V> ExactSizeIterator for Values<'a, K, V> {}
impl<'a, K: Key, V> ExactSizeIterator for ValuesMut<'a, K, V> {}
impl<'a, K: Key, V> ExactSizeIterator for Drain<'a, K, V> {}
impl<K: Key, V> ExactSizeIterator for IntoIter<K, V> {}

// ── Tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collections::slotmap::DefaultKey;
    use crate::collections::slotmap::SlotMap;

    #[test]
    fn basic_insert_get_remove() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::try_new().unwrap();
        let k = sm.try_insert(42).unwrap();
        let mut sec: SecondaryMap<DefaultKey, i32> = SecondaryMap::try_new().unwrap();
        sec.try_insert(k, 100).unwrap();
        assert_eq!(*sec.get(k).unwrap(), 100);
        assert_eq!(sec.remove(k), Some(100));
        assert_eq!(sec.get(k), None);
    }

    #[test]
    fn entry_api() {
        let mut sm: SlotMap<DefaultKey, ()> = SlotMap::try_with_key().unwrap();
        let k = sm.try_insert(()).unwrap();
        let mut sec: SecondaryMap<DefaultKey, i32> = SecondaryMap::try_new().unwrap();
        let v = sec.try_entry(k).unwrap().unwrap().or_insert(42);
        assert_eq!(*v, 42);
        *sec.try_entry(k).unwrap().unwrap().or_insert(0) *= 2;
        assert_eq!(sec[k], 84);
    }

    #[test]
    fn outdated_key_not_overwritten() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::try_new().unwrap();
        let k1 = sm.try_insert(1).unwrap();
        let mut sec: SecondaryMap<DefaultKey, i32> = SecondaryMap::try_new().unwrap();
        sec.try_insert(k1, 100).unwrap();
        // Remove and reinsert — same slot index, new version.
        sm.remove(k1);
        let k2 = sm.try_insert(2).unwrap();
        assert_eq!(k1.data().idx(), k2.data().idx());
        // Inserting k2 into the secondary map evicts the stale k1 entry
        // because k2's version is newer.
        sec.try_insert(k2, 200).unwrap();
        // After eviction, old key is no longer visible.
        assert_eq!(sec.get(k1), None);
        assert_eq!(*sec.get(k2).unwrap(), 200);
    }

    #[test]
    fn iteration_works() {
        let mut sm: SlotMap<DefaultKey, ()> = SlotMap::try_with_key().unwrap();
        let k1 = sm.try_insert(()).unwrap();
        let k2 = sm.try_insert(()).unwrap();
        let mut sec: SecondaryMap<DefaultKey, i32> = SecondaryMap::try_new().unwrap();
        sec.try_insert(k1, 10).unwrap();
        sec.try_insert(k2, 20).unwrap();
        let vals: Vec<_> = sec.values().copied().collect();
        assert_eq!(vals.len(), 2);
        assert!(vals.contains(&10));
        assert!(vals.contains(&20));
    }

    #[test]
    fn from_iterator() {
        let mut sm: SlotMap<DefaultKey, ()> = SlotMap::try_with_key().unwrap();
        let k1 = sm.try_insert(()).unwrap();
        let k2 = sm.try_insert(()).unwrap();
        let sec: SecondaryMap<DefaultKey, i32> = [(k1, 10), (k2, 20)].into_iter().collect();
        assert_eq!(*sec.get(k1).unwrap(), 10);
        assert_eq!(*sec.get(k2).unwrap(), 20);
    }
}
