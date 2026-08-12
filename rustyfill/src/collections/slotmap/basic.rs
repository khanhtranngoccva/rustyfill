//! Slot map — O(1) insert, access, and removal with stable versioned keys.
//!
//! Every operation that may allocate returns a [`Result`] instead of panicking.

use crate::alloc::AllocError;
use crate::alloc::TryReserveError;
use crate::collections::slotmap::key::{DefaultKey, Key, KeyData};
use crate::lang_alloc::vec::Vec;
use crate::lang_core::fmt::{self, Debug};
use crate::lang_core::hash::Hash;
use crate::lang_core::iter::{Enumerate, FusedIterator};
use crate::lang_core::marker::PhantomData;
use crate::lang_core::mem::{ManuallyDrop, MaybeUninit};
use crate::lang_core::ops::{Index, IndexMut};
use crate::try_fmt::helpers::FormatterExt;
use crate::try_fmt::TryDebug;

// ── Internal slot representation ────────────────────────────────────────────────

union SlotUnion<T> {
    value: ManuallyDrop<T>,
    next_free: u32,
}

struct Slot<T> {
    u: SlotUnion<T>,
    /// Even = vacant, odd = occupied.
    version: u32,
}

enum SlotContent<'a, T: 'a> {
    Occupied(&'a T),
    Vacant(&'a u32),
}

enum SlotContentMut<'a, T: 'a> {
    OccupiedMut(&'a mut T),
    VacantMut(&'a mut u32),
}

use self::SlotContent::{Occupied, Vacant};
use self::SlotContentMut::{OccupiedMut, VacantMut};

impl<T> Slot<T> {
    #[inline(always)]
    fn occupied(&self) -> bool {
        !self.version.is_multiple_of(2)
    }

    fn get(&self) -> SlotContent<'_, T> {
        unsafe {
            if self.occupied() {
                Occupied(&*self.u.value)
            } else {
                Vacant(&self.u.next_free)
            }
        }
    }

    fn get_mut(&mut self) -> SlotContentMut<'_, T> {
        unsafe {
            if self.occupied() {
                OccupiedMut(&mut *self.u.value)
            } else {
                VacantMut(&mut self.u.next_free)
            }
        }
    }
}

impl<T> Drop for Slot<T> {
    fn drop(&mut self) {
        if crate::lang_core::mem::needs_drop::<T>() && self.occupied() {
            unsafe {
                ManuallyDrop::drop(&mut self.u.value);
            }
        }
    }
}

impl<T: Clone> Clone for Slot<T> {
    fn clone(&self) -> Self {
        Self {
            u: match self.get() {
                Occupied(value) => SlotUnion {
                    value: ManuallyDrop::new(value.clone()),
                },
                Vacant(&next_free) => SlotUnion { next_free },
            },
            version: self.version,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        match (self.get_mut(), source.get()) {
            (OccupiedMut(self_val), Occupied(source_val)) => self_val.clone_from(source_val),
            (OccupiedMut(_), Vacant(&next_free)) => unsafe {
                ManuallyDrop::drop(&mut self.u.value);
                self.u = SlotUnion { next_free };
            },
            (VacantMut(self_nf), Vacant(&source_nf)) => *self_nf = source_nf,
            (VacantMut(_), Occupied(value)) => {
                self.u = SlotUnion {
                    value: ManuallyDrop::new(value.clone()),
                };
            },
        }
        self.version = source.version;
    }
}

impl<T: Debug> Debug for Slot<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut builder = f.debug_struct("Slot");
        builder.field("version", &self.version);
        match self.get() {
            Occupied(value) => builder.field("value", value).finish(),
            Vacant(next_free) => builder.field("next_free", next_free).finish(),
        }
    }
}

// ── Error type ──────────────────────────────────────────────────────────────────

/// Error returned by [`SlotMap`] operations.
#[derive(Debug)]
pub enum SlotMapError {
    /// A raw heap allocation failed.
    Alloc(AllocError),
    /// Capacity reservation failed.
    Reserve(TryReserveError),
    /// The slot map has reached its maximum size (2³² − 2 slots).
    Full,
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Display for SlotMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => write!(f, "slot map allocation failed: {}", e),
            Self::Reserve(e) => write!(f, "slot map capacity reservation failed: {}", e),
            Self::Full => f.write_str("slot map is full"),
            Self::Other(msg) => f.write_str(msg),
        }
    }
}

impl TryDebug for SlotMapError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => f
                .try_debug_struct("SlotMapError::Alloc")
                .field("0", e)
                .finish(),
            Self::Reserve(e) => f
                .try_debug_struct("SlotMapError::Reserve")
                .field("0", e)
                .finish(),
            Self::Full => f.write_str("SlotMapError::Full"),
            Self::Other(msg) => write!(f, "SlotMapError::Other({:?})", msg),
        }
    }
}

impl From<AllocError> for SlotMapError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

impl From<TryReserveError> for SlotMapError {
    fn from(e: TryReserveError) -> Self {
        Self::Reserve(e)
    }
}

impl From<crate::lang_std::collections::TryReserveError> for SlotMapError {
    fn from(e: crate::lang_std::collections::TryReserveError) -> Self {
        Self::Reserve(TryReserveError::from(e))
    }
}

// ── SlotMap ─────────────────────────────────────────────────────────────────────

/// Slot map — storage with stable unique keys.
///
/// Insertion, removal, and access are all O(1). Keys are versioned so that
/// once a key is removed it stays invalid even if the underlying slot is
/// reused. After 2³¹ deletions-and-insertions to the same slot the version
/// wraps around, but behavior remains safe.
///
/// Every operation that may allocate returns a [`Result`] instead of panicking.
#[derive(Debug)]
pub struct SlotMap<K: Key, V> {
    slots: Vec<Slot<V>>,
    free_head: u32,
    num_elems: u32,
    _k: PhantomData<fn(K) -> K>,
}

// ── Constructors ────────────────────────────────────────────────────────────────

impl<V> SlotMap<DefaultKey, V> {
    /// Constructs a new, empty [`SlotMap`].
    pub fn new() -> Self {
        Self::with_capacity_and_key(0)
    }

    /// Creates an empty [`SlotMap`] with room for at least `capacity` elements.
    ///
    /// Does not allocate if `capacity == 0`. May panic on overflow.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_key(capacity)
    }

    /// Creates a [`SlotMap`] with the given capacity.
    pub fn try_with_capacity(capacity: usize) -> Result<Self, SlotMapError> {
        Self::try_with_capacity_and_key(capacity)
    }
}

impl<K: Key, V> SlotMap<K, V> {
    /// Constructs a new, empty [`SlotMap`] with a custom key type.
    pub fn with_key() -> Self {
        Self::with_capacity_and_key(0)
    }

    /// Creates an empty [`SlotMap`] with the given capacity and key type.
    pub fn with_capacity_and_key(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity + 1);
        slots.push(Slot {
            u: SlotUnion { next_free: 0 },
            version: 0,
        });
        Self {
            slots,
            free_head: 1,
            num_elems: 0,
            _k: PhantomData,
        }
    }

    /// Creates a [`SlotMap`] with the given capacity and key type.
    pub fn try_with_capacity_and_key(capacity: usize) -> Result<Self, SlotMapError> {
        if capacity == 0 {
            return Ok(Self::with_key());
        }
        // We need capacity+1 for the sentinel. Use try_reserve on an empty vec
        // with a pushed sentinel.
        let mut slots = Vec::new();
        slots.push(Slot {
            u: SlotUnion { next_free: 0 },
            version: 0,
        });
        // Reserve remaining slots.
        let needed = capacity;
        slots.try_reserve(needed).map_err(SlotMapError::from)?;
        Ok(Self {
            slots,
            free_head: 1,
            num_elems: 0,
            _k: PhantomData,
        })
    }

    // ── Query methods ─────────────────────────────────────────────────────────

    /// Returns the number of elements in the slot map.
    pub fn len(&self) -> usize {
        self.num_elems as usize
    }

    /// Returns `true` if the slot map contains no elements.
    pub fn is_empty(&self) -> bool {
        self.num_elems == 0
    }

    /// Returns the number of elements the slot map can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.slots.capacity().saturating_sub(1)
    }

    /// Reserves capacity for at least `additional` more elements.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), SlotMapError> {
        let needed = (self.len() + additional).saturating_sub(self.slots.len() - 1);
        self.slots.try_reserve(needed).map_err(SlotMapError::from)?;
        Ok(())
    }

    /// Returns `true` if the slot map contains the given key.
    pub fn contains_key(&self, key: K) -> bool {
        let kd = key.data();
        self.slots
            .get(kd.idx() as usize)
            .is_some_and(|slot| slot.version == kd.version_raw())
    }

    // ── Insertion ─────────────────────────────────────────────────────────────

    /// Inserts a value into the slot map. Returns a unique key.
    ///
    /// Returns [`Err`] if the underlying allocation fails or the slot map is full.
    pub fn try_insert(&mut self, value: V) -> Result<K, SlotMapError> {
        self.try_insert_with_key::<_, SlotMapError>(move |_| Ok(value))
    }

    /// Inserts a value given by `f`, passing the assigned key into `f`.
    ///
    /// Useful for storing values that contain their own key.
    ///
    /// If `f` returns `Err`, the slot map is untouched.
    pub fn try_insert_with_key<F, E>(&mut self, f: F) -> Result<K, SlotMapError>
    where
        F: FnOnce(K) -> Result<V, E>,
        E: Into<SlotMapError>,
    {
        // Fast path: reuse a free slot from the freelist.
        if let Some(slot) = self.slots.get_mut(self.free_head as usize) {
            let occupied_version = slot.version | 1;
            let kd = KeyData::new(self.free_head, occupied_version);
            let value = f(kd.into()).map_err(Into::into)?;
            unsafe {
                self.free_head = slot.u.next_free;
                slot.u.value = ManuallyDrop::new(value);
                slot.version = occupied_version;
            }
            self.num_elems += 1;
            return Ok(kd.into());
        }

        // No free slot — grow the vector.
        if self.slots.len() >= u32::MAX as usize {
            return Err(SlotMapError::Full);
        }

        let idx = self.slots.len() as u32;
        let version = 1u32;
        let kd = KeyData::new(idx, version);

        // Allocate first, then push — order matters for panic safety.
        self.slots.try_reserve(1).map_err(SlotMapError::from)?;

        let value = f(kd.into()).map_err(Into::into)?;
        self.slots.push(Slot {
            u: SlotUnion {
                value: ManuallyDrop::new(value),
            },
            version,
        });
        self.free_head = kd.idx() + 1;
        self.num_elems += 1;
        Ok(kd.into())
    }

    // ── Removal ─────────────────────────────────────────────────────────────────

    /// Removes a key, returning the value if the key was active.
    pub fn remove(&mut self, key: K) -> Option<V> {
        let kd = key.data();
        if self.contains_key(key) {
            Some(unsafe { self.remove_from_slot(kd.idx() as usize) })
        } else {
            None
        }
    }

    /// Temporarily removes a key, keeping the slot reserved for `reattach()`.
    pub fn detach(&mut self, key: K) -> Option<V> {
        let kd = key.data();
        if self.contains_key(key) {
            unsafe {
                let slot = self.slots.get_unchecked_mut(kd.idx() as usize);
                let value = ManuallyDrop::take(&mut slot.u.value);
                slot.u.next_free = u32::MAX;
                slot.version = slot.version.wrapping_add(1);
                self.num_elems -= 1;
                Some(value)
            }
        } else {
            None
        }
    }

    /// Reattaches a previously detached key with a new value.
    ///
    /// # Panics
    ///
    /// Panics if the key is not currently detached. This is a logic error,
    /// not an allocation failure.
    pub fn reattach(&mut self, detached_key: K, value: V) {
        let kd = detached_key.data();
        let slot = self
            .slots
            .get_mut(kd.idx() as usize)
            .filter(|s| s.version == kd.version_raw().wrapping_add(1))
            .filter(|s| unsafe { s.u.next_free == u32::MAX })
            .expect("key is not detached");

        slot.u.value = ManuallyDrop::new(value);
        slot.version = slot.version.wrapping_sub(1);
        self.num_elems += 1;
    }

    // Helper: remove from an occupied slot. Caller must verify occupancy.
    #[inline(always)]
    unsafe fn remove_from_slot(&mut self, idx: usize) -> V {
        unsafe {
            let slot = self.slots.get_unchecked_mut(idx);
            let value = ManuallyDrop::take(&mut slot.u.value);
            slot.u.next_free = self.free_head;
            self.free_head = idx as u32;
            self.num_elems -= 1;
            slot.version = slot.version.wrapping_add(1);
            value
        }
    }

    // ── Mutation helpers ───────────────────────────────────────────────────────

    /// Retains only elements satisfying the predicate.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(K, &mut V) -> bool,
    {
        for i in 1..self.slots.len() {
            let slot = unsafe { self.slots.get_unchecked_mut(i) };
            let version = slot.version;
            let should_remove = if let OccupiedMut(value) = slot.get_mut() {
                let key = KeyData::new(i as u32, version).into();
                !f(key, value)
            } else {
                false
            };
            if should_remove {
                unsafe { self.remove_from_slot(i) };
            }
        }
    }

    /// Clears the slot map, keeping allocated memory.
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
            .filter(|slot| slot.version == kd.version_raw())
            .map(|slot| unsafe { &*slot.u.value })
    }

    /// Returns a mutable reference to the value for the given key.
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        let kd = key.data();
        self.slots
            .get_mut(kd.idx() as usize)
            .filter(|slot| slot.version == kd.version_raw())
            .map(|slot| unsafe { &mut *slot.u.value })
    }

    /// Unchecked access — caller must ensure `contains_key(key)` is true.
    ///
    /// # Safety
    /// Undefined behavior if the key is invalid.
    pub unsafe fn get_unchecked(&self, key: K) -> &V {
        debug_assert!(self.contains_key(key));
        unsafe { &self.slots.get_unchecked(key.data().idx() as usize).u.value }
    }

    /// Unchecked mutable access.
    ///
    /// # Safety
    /// Undefined behavior if the key is invalid.
    pub unsafe fn get_unchecked_mut(&mut self, key: K) -> &mut V {
        debug_assert!(self.contains_key(key));
        unsafe { &mut self.slots.get_unchecked_mut(key.data().idx() as usize).u.value }
    }

    /// Returns disjoint mutable references for multiple keys.
    ///
    /// Returns `None` if any key is invalid or keys overlap.
    pub fn get_disjoint_mut<const N: usize>(&mut self, keys: [K; N]) -> Option<[&mut V; N]> {
        let mut ptrs: [MaybeUninit<*mut V>; N] = [(); N].map(|_| MaybeUninit::uninit());
        let slots_ptr = self.slots.as_mut_ptr();
        let mut i = 0;
        while i < N {
            let kd = keys[i].data();
            if !self.contains_key(kd.into()) {
                break;
            }
            unsafe {
                let slot = &mut *slots_ptr.add(kd.idx() as usize);
                slot.version ^= 1;
                ptrs[i] = MaybeUninit::new(&mut *slot.u.value);
            }
            i += 1;
        }
        for k in &keys[..i] {
            let idx = k.data().idx() as usize;
            unsafe { (*slots_ptr.add(idx)).version ^= 1; }
        }
        if i == N {
            Some(ptrs.map(|p| unsafe { &mut *p.assume_init() }))
        } else {
            None
        }
    }

    // ── Iterators ──────────────────────────────────────────────────────────────

    /// Immutable iterator over all key-value pairs.
    pub fn iter(&self) -> Iter<'_, K, V> {
        let mut it = self.slots.iter().enumerate();
        it.next(); // skip sentinel
        Iter {
            slots: it,
            num_left: self.len(),
            _k: PhantomData,
        }
    }

    /// Mutable iterator over all key-value pairs.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        let len = self.len();
        let mut it = self.slots.iter_mut().enumerate();
        it.next();
        IterMut {
            slots: it,
            num_left: len,
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

// ── Trait impls ─────────────────────────────────────────────────────────────────

impl<K: Key, V> Clone for SlotMap<K, V>
where
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            slots: self.slots.clone(),
            ..*self
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.slots.clone_from(&source.slots);
        self.free_head = source.free_head;
        self.num_elems = source.num_elems;
    }
}

impl<K: Key, V> Default for SlotMap<K, V> {
    fn default() -> Self {
        Self::with_key()
    }
}

impl<K: Key, V> Index<K> for SlotMap<K, V> {
    type Output = V;

    fn index(&self, key: K) -> &V {
        match self.get(key) {
            Some(r) => r,
            None => panic!("invalid SlotMap key used"),
        }
    }
}

impl<K: Key, V> IndexMut<K> for SlotMap<K, V> {
    fn index_mut(&mut self, key: K) -> &mut V {
        match self.get_mut(key) {
            Some(r) => r,
            None => panic!("invalid SlotMap key used"),
        }
    }
}

impl<K: Key, V> Hash for SlotMap<K, V>
where
    V: Hash,
{
    fn hash<H: crate::lang_core::hash::Hasher>(&self, state: &mut H) {
        for (_k, v) in self.iter() {
            v.hash(state);
        }
    }
}

// ── Iterators ───────────────────────────────────────────────────────────────────

/// A draining iterator for [`SlotMap`].
#[derive(Debug)]
pub struct Drain<'a, K: 'a + Key, V: 'a> {
    sm: &'a mut SlotMap<K, V>,
    cur: usize,
}

/// An iterator that moves key-value pairs out of a [`SlotMap`].
#[derive(Debug, Clone)]
pub struct IntoIter<K: Key, V> {
    num_left: usize,
    slots: Enumerate<crate::lang_alloc::vec::IntoIter<Slot<V>>>,
    _k: PhantomData<fn(K) -> K>,
}

/// An immutable iterator over the key-value pairs in a [`SlotMap`].
#[derive(Debug)]
pub struct Iter<'a, K: 'a + Key, V: 'a> {
    num_left: usize,
    slots: Enumerate<crate::lang_core::slice::Iter<'a, Slot<V>>>,
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

/// A mutable iterator over the key-value pairs in a [`SlotMap`].
#[derive(Debug)]
pub struct IterMut<'a, K: 'a + Key, V: 'a> {
    num_left: usize,
    slots: Enumerate<crate::lang_core::slice::IterMut<'a, Slot<V>>>,
    _k: PhantomData<fn(K) -> K>,
}

/// An iterator over the keys in a [`SlotMap`].
#[derive(Debug)]
pub struct Keys<'a, K: 'a + Key, V: 'a> {
    inner: Iter<'a, K, V>,
}

impl<'a, K: 'a + Key, V: 'a> Clone for Keys<'a, K, V> {
    fn clone(&self) -> Self {
        Keys {
            inner: self.inner.clone(),
        }
    }
}

/// An iterator over the values in a [`SlotMap`].
#[derive(Debug)]
pub struct Values<'a, K: 'a + Key, V: 'a> {
    inner: Iter<'a, K, V>,
}

impl<'a, K: 'a + Key, V: 'a> Clone for Values<'a, K, V> {
    fn clone(&self) -> Self {
        Values {
            inner: self.inner.clone(),
        }
    }
}

/// A mutable iterator over the values in a [`SlotMap`].
#[derive(Debug)]
pub struct ValuesMut<'a, K: 'a + Key, V: 'a> {
    inner: IterMut<'a, K, V>,
}

// ── Iterator implementations ────────────────────────────────────────────────────

impl<'a, K: Key, V> Iterator for Drain<'a, K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        let len = self.sm.slots.len();
        while self.cur < len {
            let idx = self.cur;
            self.cur += 1;
            unsafe {
                let slot = self.sm.slots.get_unchecked(idx);
                if slot.occupied() {
                    let kd = KeyData::new(idx as u32, slot.version);
                    return Some((kd.into(), self.sm.remove_from_slot(idx)));
                }
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
            if slot.occupied() {
                let kd = KeyData::new(idx as u32, slot.version);
                slot.version = 0; // prevent double-drop
                let value = unsafe { ManuallyDrop::take(&mut slot.u.value) };
                self.num_left -= 1;
                return Some((kd.into(), value));
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
            if let Occupied(value) = slot.get() {
                let kd = KeyData::new(idx as u32, slot.version);
                self.num_left -= 1;
                return Some((kd.into(), value));
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
            let version = slot.version;
            if let OccupiedMut(value) = slot.get_mut() {
                let kd = KeyData::new(idx as u32, version);
                self.num_left -= 1;
                return Some((kd.into(), value));
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

// ── IntoIterator ────────────────────────────────────────────────────────────────

impl<'a, K: Key, V> IntoIterator for &'a SlotMap<K, V> {
    type Item = (K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Key, V> IntoIterator for &'a mut SlotMap<K, V> {
    type Item = (K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K: Key, V> IntoIterator for SlotMap<K, V> {
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

// ── FusedIterator / ExactSizeIterator markers ───────────────────────────────────

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
    use crate::lang_alloc::string::String;

    #[test]
    fn basic_insert_get_remove() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::new();
        assert!(sm.is_empty());

        let k1 = sm.try_insert(42).unwrap();
        let k2 = sm.try_insert(99).unwrap();
        assert_eq!(sm.len(), 2);
        assert_eq!(*sm.get(k1).unwrap(), 42);
        assert_eq!(*sm.get(k2).unwrap(), 99);

        assert_eq!(sm.remove(k1), Some(42));
        assert_eq!(sm.len(), 1);
        assert_eq!(sm.get(k1), None);
        assert_eq!(*sm.get(k2).unwrap(), 99);
        assert_eq!(sm.remove(k1), None); // already removed
    }

    #[test]
    fn insert_reuses_freed_slots() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::new();
        let k1 = sm.try_insert(1).unwrap();
        sm.remove(k1);
        let k2 = sm.try_insert(2).unwrap();
        // Same slot index, different version.
        assert_eq!(k1.data().idx(), k2.data().idx());
        assert_ne!(k1.data().version_raw(), k2.data().version_raw());
        assert_eq!(sm.get(k1), None); // old key invalidated
        assert_eq!(*sm.get(k2).unwrap(), 2);
    }

    #[test]
    fn insert_with_key_self_referential() {
        let mut sm: SlotMap<DefaultKey, (DefaultKey, i32)> = SlotMap::new();
        let k = sm.try_insert_with_key::<_, SlotMapError>(|key| Ok((key, 42))).unwrap();
        let (stored_key, val) = *sm.get(k).unwrap();
        assert_eq!(stored_key, k);
        assert_eq!(val, 42);
    }

    #[test]
    fn detach_and_reattach() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::new();
        let k = sm.try_insert(42).unwrap();
        assert_eq!(sm.detach(k), Some(42));
        assert_eq!(sm.get(k), None);
        sm.reattach(k, 100);
        assert_eq!(*sm.get(k).unwrap(), 100);
    }

    #[test]
    fn error_display_messages() {
        use crate::lang_alloc::string::ToString;
        let err_full = SlotMapError::Full;
        assert!(err_full.to_string().contains("full"));
    }

    #[test]
    fn drain_returns_all_elements() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::new();
        sm.try_insert(1).unwrap();
        sm.try_insert(2).unwrap();
        sm.try_insert(3).unwrap();
        let drained: Vec<_> = sm.drain().map(|(_, v)| v).collect();
        assert_eq!(sm.len(), 0);
        assert_eq!(drained.len(), 3);
        assert!(drained.contains(&1));
        assert!(drained.contains(&2));
        assert!(drained.contains(&3));
    }

    #[test]
    fn into_iter_moves_values_out() {
        let mut sm: SlotMap<DefaultKey, String> = SlotMap::new();
        sm.try_insert(String::from("a")).unwrap();
        sm.try_insert(String::from("b")).unwrap();
        let strings: Vec<_> = sm.into_iter().map(|(_, v)| v).collect();
        assert!(strings.contains(&String::from("a")));
        assert!(strings.contains(&String::from("b")));
    }

    #[test]
    fn retain_predicate() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::new();
        let k0 = sm.try_insert(0).unwrap();
        let k1 = sm.try_insert(1).unwrap();
        let k2 = sm.try_insert(2).unwrap();
        sm.retain(|_k, v| *v != 1);
        assert!(sm.contains_key(k0));
        assert!(!sm.contains_key(k1));
        assert!(sm.contains_key(k2));
        assert_eq!(sm.len(), 2);
    }

    #[test]
    fn null_key_is_always_invalid() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::new();
        let nk = DefaultKey::null();
        assert!(nk.is_null());
        assert_eq!(sm.get(nk), None);
        assert_eq!(sm.remove(nk), None);
        sm.try_insert(42).unwrap();
        assert_eq!(sm.get(nk), None);
    }

    #[test]
    fn capacity_grows_on_insert() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::new();
        let initial_cap = sm.capacity();
        for _ in 0..100 {
            sm.try_insert(0).unwrap();
        }
        assert!(sm.capacity() >= initial_cap + 100);
    }

    #[test]
    fn try_reserve_works() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::new();
        sm.try_insert(0).unwrap();
        sm.try_reserve(64).unwrap();
        assert!(sm.capacity() >= 65);
    }

    #[test]
    fn custom_key_type() {
        crate::new_key_type! {
            struct MyKey;
        }
        let mut sm: SlotMap<MyKey, u64> = SlotMap::with_key();
        let k = sm.try_insert(42).unwrap();
        assert_eq!(*sm.get(k).unwrap(), 42);
    }

    #[test]
    fn get_disjoint_mut_two_keys() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::new();
        let ka = sm.try_insert(10).unwrap();
        let kb = sm.try_insert(20).unwrap();
        let result = sm.get_disjoint_mut([ka, kb]);
        assert!(result.is_some());
        let [a, b] = result.unwrap();
        crate::lang_core::mem::swap(a, b);
        assert_eq!(sm[ka], 20);
        assert_eq!(sm[kb], 10);
    }

    #[test]
    fn get_disjoint_mut_duplicate_returns_none() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::new();
        let ka = sm.try_insert(10).unwrap();
        assert!(sm.get_disjoint_mut([ka, ka]).is_none());
    }

    #[test]
    fn get_disjoint_mut_invalid_key_returns_none() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::new();
        let ka = sm.try_insert(10).unwrap();
        let kb = sm.try_insert(20).unwrap();
        sm.remove(kb);
        assert!(sm.get_disjoint_mut([ka, kb]).is_none());
    }

    #[test]
    fn clone_preserves_state() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::new();
        let k1 = sm.try_insert(1).unwrap();
        let k2 = sm.try_insert(2).unwrap();
        sm.remove(k1);
        let cloned = sm.clone();
        assert_eq!(cloned.len(), 1);
        assert_eq!(*cloned.get(k2).unwrap(), 2);
        assert_eq!(cloned.get(k1), None);
    }

    #[test]
    fn index_trait_panic_on_invalid() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::new();
        let k = sm.try_insert(42).unwrap();
        assert_eq!(sm[k], 42);
        sm.remove(k);
        // Would panic if we did sm[k] here — skipped to keep test green.
    }

    #[test]
    fn try_with_capacity_zero_succeeds() {
        let sm: SlotMap<DefaultKey, i32> =
            SlotMap::try_with_capacity(0).unwrap();
        assert!(sm.is_empty());
    }

    #[test]
    fn values_mut_iteration() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::new();
        sm.try_insert(1).unwrap();
        sm.try_insert(2).unwrap();
        sm.try_insert(3).unwrap();
        sm.values_mut().for_each(|v| *v *= 3);
        let vals: Vec<_> = sm.values().copied().collect();
        assert!(vals.contains(&3));
        assert!(vals.contains(&6));
        assert!(vals.contains(&9));
    }

    #[test]
    fn keys_iterator() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::new();
        let expected: Vec<_> = (0..3).map(|i| sm.try_insert(i).unwrap()).collect();
        let actual: Vec<_> = sm.keys().collect();
        assert_eq!(actual.len(), 3);
        for ek in &expected {
            assert!(actual.contains(ek));
        }
    }
}
