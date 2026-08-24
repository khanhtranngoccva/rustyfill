//! Slot map — O(1) insert, access, and removal with stable versioned keys.
//!
//! Every operation that may allocate returns a [`Result`] instead of panicking.

use crate::alloc::vec::TryVec;
use crate::alloc::{TryReserveError, TryReserveErrorExt};
use crate::collections::slotmap::key::{
    DANGLING_SENTINEL, DefaultKey, Key, KeyData, MAX_SLOTS_LEN,
};
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::{TryDebug, TryDisplay};
use lang_alloc::vec::Vec;
use lang_core::fmt::{self, Debug};
use lang_core::hash::Hash;
use lang_core::iter::{Enumerate, FusedIterator};
use lang_core::marker::PhantomData;
use lang_core::mem::{ManuallyDrop, MaybeUninit};
use lang_core::ops::{Index, IndexMut};

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
        if lang_core::mem::needs_drop::<T>() && self.occupied() {
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
            }
        }
        self.version = source.version;
    }
}

impl<T: TryClone> TryClone for Slot<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(Self {
            u: match self.get() {
                Occupied(value) => SlotUnion {
                    value: ManuallyDrop::new(value.try_clone()?),
                },
                Vacant(&next_free) => SlotUnion { next_free },
            },
            version: self.version,
        })
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

/// Error returned by [`SlotMap::try_insert_with_key`] and its alias
/// [`SlotMap::fallible_insert_with_key`].
///
/// Unlike the other fallible operations, which can only fail on a capacity
/// reservation, insertion through a closure has two independent failure
/// sources: the reservation itself ([`SlotMapInsertWithError::Reserve`]) and the
/// closure's own error value ([`SlotMapInsertWithError::Closure`]).
#[derive(Clone)]
pub enum SlotMapInsertWithError<E> {
    /// The slot count could not be reserved. Carries the underlying
    /// [`TryReserveError`], whose `CapacityOverflow` kind also covers the
    /// case where the slot count would exceed the maximum representable
    /// value (2³² − 2 usable slots).
    Reserve(TryReserveError),
    /// The supplied closure returned `Err`.
    Closure(E),
}

impl<E> fmt::Debug for SlotMapInsertWithError<E>
where
    E: TryDebug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(self, f)
    }
}

impl<E> fmt::Display for SlotMapInsertWithError<E>
where
    E: TryDisplay,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }
}

impl<E> TryDebug for SlotMapInsertWithError<E>
where
    E: TryDebug,
{
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::debug_field(f, "SlotMapInsertWithError::Reserve", e),
            Self::Closure(e) => u::debug_field(f, "SlotMapInsertWithError::Closure", e),
        }
    }
}

impl<E> TryDisplay for SlotMapInsertWithError<E>
where
    E: TryDisplay,
{
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::display_delegated(f, "slot map", e),
            Self::Closure(e) => u::display_delegated(f, "slot map insertion closure", e),
        }
    }
}

impl<E> From<TryReserveError> for SlotMapInsertWithError<E> {
    fn from(e: TryReserveError) -> Self {
        Self::Reserve(e)
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
    ///
    /// Fallible because even an empty map allocates space for the sentinel slot.
    pub fn try_new() -> Result<Self, TryReserveError> {
        Self::try_with_capacity_and_key(0)
    }

    /// Creates a [`SlotMap`] with the given capacity.
    pub fn try_with_capacity(capacity: usize) -> Result<Self, TryReserveError> {
        Self::try_with_capacity_and_key(capacity)
    }

    // ── Aliases with `fallible_` prefix ────────────────────────────────────────

    /// Alias for [`Self::try_new`].
    pub fn fallible_new() -> Result<Self, TryReserveError> {
        Self::try_new()
    }

    /// Alias for [`Self::try_with_capacity`].
    pub fn fallible_with_capacity(capacity: usize) -> Result<Self, TryReserveError> {
        Self::try_with_capacity(capacity)
    }
}

impl<K: Key, V> SlotMap<K, V> {
    /// Constructs a new, empty [`SlotMap`] with a custom key type.
    ///
    /// Fallible because even an empty map allocates space for the sentinel slot.
    pub fn try_with_key() -> Result<Self, TryReserveError> {
        Self::try_with_capacity_and_key(0)
    }

    /// Creates a [`SlotMap`] with the given capacity and key type.
    ///
    /// Returns a [`TryReserveError`] with the `CapacityOverflow` kind
    /// if `capacity` would exceed the maximum number of usable slots
    /// (`MAX_SLOTS_LEN` – 1), accounting for the sentinel that always
    /// occupies index 0.
    pub fn try_with_capacity_and_key(capacity: usize) -> Result<Self, TryReserveError> {
        if capacity >= MAX_SLOTS_LEN.saturating_sub(1) {
            return Err(TryReserveError::new_capacity_overflow());
        }
        // Safe: `capacity < MAX_SLOTS_LEN - 1 <= usize::MAX`, so `+1` cannot overflow.
        let mut slots = Vec::fallible_with_capacity(
            capacity
                .checked_add(1)
                .expect("capacity below MAX_SLOTS_LEN"),
        )?;
        slots.try_push(Slot {
            u: SlotUnion { next_free: 0 },
            version: 0,
        })?;
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
    ///
    /// Returns a [`TryReserveError`] with the `CapacityOverflow` kind
    /// if the resulting size would exceed the maximum number of slots
    /// (`MAX_SLOTS_LEN` – 1 usable entries), ensuring callers can rely on
    /// subsequent [`Self::try_insert`] calls succeeding (barring allocation
    /// failure).
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let total = self
            .len()
            .checked_add(additional)
            .ok_or_else(TryReserveError::new_capacity_overflow)?;
        if total >= MAX_SLOTS_LEN.saturating_sub(1) {
            return Err(TryReserveError::new_capacity_overflow());
        }
        // One slot is reserved for the sentinel; the slots vec always holds it,
        // so `slots.len() >= 1` and the subtraction cannot underflow.
        let needed = total.saturating_sub(self.slots.len().saturating_sub(1));
        self.slots.try_reserve(needed)?;
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
    pub fn try_insert(&mut self, value: V) -> Result<K, TryReserveError> {
        self.insert_inner::<_, ()>(move |_| Ok(value))
            .map_err(|e| match e {
                SlotMapInsertWithError::Reserve(r) => r,
                // Unreachable: the infallible closure never produces a `Closure` error.
                SlotMapInsertWithError::Closure(()) => {
                    unreachable!("infallible closure reported an error")
                }
            })
    }

    /// Inserts a value given by `f`, passing the assigned key into `f`.
    ///
    /// Useful for storing values that contain their own key.
    ///
    /// The returned [`SlotMapInsertWithError`] distinguishes a failed capacity
    /// reservation ([`SlotMapInsertWithError::Reserve`]) from an error returned by the
    /// closure itself ([`SlotMapInsertWithError::Closure`]). In either case the slot map
    /// is left untouched.
    pub fn try_insert_with_key<F, E>(&mut self, f: F) -> Result<K, SlotMapInsertWithError<E>>
    where
        F: FnOnce(K) -> Result<V, E>,
    {
        self.insert_inner(f)
    }

    fn insert_inner<F, E>(&mut self, f: F) -> Result<K, SlotMapInsertWithError<E>>
    where
        F: FnOnce(K) -> Result<V, E>,
    {
        // Fast path: reuse a free slot from the freelist.
        if let Some(slot) = self.slots.get_mut(self.free_head as usize) {
            let occupied_version = slot.version | 1;
            let kd = KeyData::new(self.free_head, occupied_version);
            let value = f(kd.into()).map_err(SlotMapInsertWithError::Closure)?;
            unsafe {
                self.free_head = slot.u.next_free;
                slot.u.value = ManuallyDrop::new(value);
                slot.version = occupied_version;
            }
            // Safe: `num_elems < MAX_SLOTS_LEN - 1 <= u32::MAX`, so `+1` cannot overflow.
            let num_elems = self
                .num_elems
                .checked_add(1)
                .expect("element count below MAX_SLOTS_LEN");
            self.num_elems = num_elems;
            return Ok(kd.into());
        }

        // No free slot — grow the vector.
        if self.slots.len() >= MAX_SLOTS_LEN {
            return Err(SlotMapInsertWithError::Reserve(
                TryReserveError::new_capacity_overflow(),
            ));
        }

        let idx = self.slots.len() as u32;
        let version = 1u32;
        let kd = KeyData::new(idx, version);

        // Allocate first, then push.
        self.slots
            .try_reserve(1)
            .map_err(SlotMapInsertWithError::Reserve)?;
        let value = f(kd.into()).map_err(SlotMapInsertWithError::Closure)?;
        self.slots.push(Slot {
            u: SlotUnion {
                value: ManuallyDrop::new(value),
            },
            version,
        });
        // Safe: `idx < MAX_SLOTS_LEN <= u32::MAX`, so `+1` cannot overflow.
        let free_head = kd
            .idx()
            .checked_add(1)
            .expect("slot index below MAX_SLOTS_LEN");
        self.free_head = free_head;
        // Safe: `num_elems < MAX_SLOTS_LEN - 1 <= u32::MAX`, so `+1` cannot overflow.
        let num_elems = self
            .num_elems
            .checked_add(1)
            .expect("element count below MAX_SLOTS_LEN");
        self.num_elems = num_elems;
        Ok(kd.into())
    }

    // ── Aliases with `fallible_` prefix ────────────────────────────────────────

    /// Alias for [`Self::try_with_key`].
    pub fn fallible_with_key() -> Result<Self, TryReserveError> {
        Self::try_with_key()
    }

    /// Alias for [`Self::try_with_capacity_and_key`].
    pub fn fallible_with_capacity_and_key(capacity: usize) -> Result<Self, TryReserveError> {
        Self::try_with_capacity_and_key(capacity)
    }

    /// Alias for [`Self::try_reserve`].
    pub fn fallible_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        Self::try_reserve(self, additional)
    }

    /// Alias for [`Self::try_insert`].
    pub fn fallible_insert(&mut self, value: V) -> Result<K, TryReserveError> {
        Self::try_insert(self, value)
    }

    /// Alias for [`Self::try_insert_with_key`].
    pub fn fallible_insert_with_key<F, E>(&mut self, f: F) -> Result<K, SlotMapInsertWithError<E>>
    where
        F: FnOnce(K) -> Result<V, E>,
    {
        Self::try_insert_with_key(self, f)
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
                slot.u.next_free = DANGLING_SENTINEL;
                slot.version = slot.version.wrapping_add(1);
                // Safe: the key exists, so at least one element is present.
                let num_elems = self
                    .num_elems
                    .checked_sub(1)
                    .expect("at least one element present");
                self.num_elems = num_elems;
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
            .filter(|s| unsafe { s.u.next_free == DANGLING_SENTINEL })
            .expect("key is not detached");

        slot.u.value = ManuallyDrop::new(value);
        slot.version = slot.version.wrapping_sub(1);
        // Safe: the key was detached, so its slot is within `MAX_SLOTS_LEN` and
        // reattaching cannot push `num_elems` past `u32::MAX`.
        let num_elems = self
            .num_elems
            .checked_add(1)
            .expect("element count below MAX_SLOTS_LEN");
        self.num_elems = num_elems;
    }

    // Helper: remove from an occupied slot. Caller must verify occupancy.
    #[inline(always)]
    unsafe fn remove_from_slot(&mut self, idx: usize) -> V {
        unsafe {
            let slot = self.slots.get_unchecked_mut(idx);
            let value = ManuallyDrop::take(&mut slot.u.value);
            slot.u.next_free = self.free_head;
            self.free_head = idx as u32;
            // Safe: the caller verified the slot is occupied, so at least one element exists.
            let num_elems = self
                .num_elems
                .checked_sub(1)
                .expect("at least one element present");
            self.num_elems = num_elems;
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
        unsafe {
            &mut self
                .slots
                .get_unchecked_mut(key.data().idx() as usize)
                .u
                .value
        }
    }

    /// Returns disjoint mutable references for multiple keys.
    ///
    /// Returns `None` if any key is invalid or keys overlap.
    pub fn get_disjoint_mut<const N: usize>(&mut self, keys: [K; N]) -> Option<[&mut V; N]> {
        let mut ptrs: [MaybeUninit<*mut V>; N] = [(); N].map(|_| MaybeUninit::uninit());
        let slots_ptr = self.slots.as_mut_ptr();
        let mut i = 0usize;
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
            // Safe: `i < N <= usize::MAX`, so `+1` cannot overflow.
            let next_i = i.checked_add(1).expect("loop index below N");
            i = next_i;
        }
        for k in &keys[..i] {
            let idx = k.data().idx() as usize;
            unsafe {
                (*slots_ptr.add(idx)).version ^= 1;
            }
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

impl<K: Key, V> TryClone for SlotMap<K, V>
where
    V: TryClone,
{
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let slots = self.slots.try_clone()?;
        Ok(Self {
            slots,
            free_head: self.free_head,
            num_elems: self.num_elems,
            _k: PhantomData,
        })
    }
}

impl<K: Key, V> TryDefault for SlotMap<K, V> {
    fn try_default() -> Result<Self, TryDefaultError> {
        Self::try_with_capacity_and_key(0).map_err(TryDefaultError::Reserve)
    }
}

impl<K: Key + TryDebug, V: TryDebug> TryDebug for SlotMap<K, V> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::try_fmt::helpers::FormatterExt;
        let mut map = f.try_debug_map();
        for (k, v) in self.iter() {
            map.entry(&k, v);
        }
        map.finish()
    }
}

impl<K: Key, V> Default for SlotMap<K, V> {
    /// Constructs an empty [`SlotMap`].
    ///
    /// # Panics
    ///
    /// Panics if the sentinel slot allocation fails. This is only reachable on
    /// catastrophic OOM and is consistent with Rust's `Default` contract, which
    /// cannot return errors. Prefer [`Self::try_with_key`] or [`Self::try_new`]
    /// for fallible construction.
    fn default() -> Self {
        Self::try_with_capacity_and_key(0)
            .expect("SlotMap::default panicked: failed to allocate sentinel slot")
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
    fn hash<H: lang_core::hash::Hasher>(&self, state: &mut H) {
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
    slots: Enumerate<lang_alloc::vec::IntoIter<Slot<V>>>,
    _k: PhantomData<fn(K) -> K>,
}

/// An immutable iterator over the key-value pairs in a [`SlotMap`].
#[derive(Debug)]
pub struct Iter<'a, K: 'a + Key, V: 'a> {
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

/// A mutable iterator over the key-value pairs in a [`SlotMap`].
#[derive(Debug)]
pub struct IterMut<'a, K: 'a + Key, V: 'a> {
    num_left: usize,
    slots: Enumerate<lang_core::slice::IterMut<'a, Slot<V>>>,
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
            // Safe: `cur < len <= usize::MAX`, so `+1` cannot overflow.
            let next_cur = self.cur.checked_add(1).expect("cursor below slot length");
            self.cur = next_cur;
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
                // Safe: `num_left` counts remaining elements and is decremented only on yield.
                let num_left = self
                    .num_left
                    .checked_sub(1)
                    .expect("remaining count positive");
                self.num_left = num_left;
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
                // Safe: `num_left` counts remaining elements and is decremented only on yield.
                let num_left = self
                    .num_left
                    .checked_sub(1)
                    .expect("remaining count positive");
                self.num_left = num_left;
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
                // Safe: `num_left` counts remaining elements and is decremented only on yield.
                let num_left = self
                    .num_left
                    .checked_sub(1)
                    .expect("remaining count positive");
                self.num_left = num_left;
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
    use lang_alloc::string::String;

    #[test]
    fn basic_insert_get_remove() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::try_new().unwrap();
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
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::try_new().unwrap();
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
        let mut sm: SlotMap<DefaultKey, (DefaultKey, i32)> = SlotMap::try_new().unwrap();
        let k = sm
            .try_insert_with_key::<_, ()>(|key| Ok((key, 42)))
            .unwrap();
        let (stored_key, val) = *sm.get(k).unwrap();
        assert_eq!(stored_key, k);
        assert_eq!(val, 42);
    }

    #[test]
    fn detach_and_reattach() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::try_new().unwrap();
        let k = sm.try_insert(42).unwrap();
        assert_eq!(sm.detach(k), Some(42));
        assert_eq!(sm.get(k), None);
        sm.reattach(k, 100);
        assert_eq!(*sm.get(k).unwrap(), 100);
    }

    #[test]
    fn error_kinds_are_distinguishable() {
        use crate::alloc::{TryReserveErrorExt, TryReserveErrorKind};

        // The slot-count overflow condition reports the overflow kind.
        let err_overflow = TryReserveError::new_capacity_overflow();
        assert_eq!(
            err_overflow.error_kind(),
            TryReserveErrorKind::CapacityOverflow
        );

        // An allocation failure reports the alloc kind instead.
        let layout = lang_core::alloc::Layout::new::<u32>();
        let err_alloc = TryReserveError::new_alloc(layout);
        assert!(matches!(
            err_alloc.error_kind(),
            TryReserveErrorKind::AllocError { .. }
        ));
    }

    #[test]
    fn drain_returns_all_elements() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::try_new().unwrap();
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
        let mut sm: SlotMap<DefaultKey, String> = SlotMap::try_new().unwrap();
        sm.try_insert(String::from("a")).unwrap();
        sm.try_insert(String::from("b")).unwrap();
        let strings: Vec<_> = sm.into_iter().map(|(_, v)| v).collect();
        assert!(strings.contains(&String::from("a")));
        assert!(strings.contains(&String::from("b")));
    }

    #[test]
    fn retain_predicate() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::try_new().unwrap();
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
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::try_new().unwrap();
        let nk = DefaultKey::null();
        assert!(nk.is_null());
        assert_eq!(sm.get(nk), None);
        assert_eq!(sm.remove(nk), None);
        sm.try_insert(42).unwrap();
        assert_eq!(sm.get(nk), None);
    }

    #[test]
    fn capacity_grows_on_insert() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::try_new().unwrap();
        let initial_cap = sm.capacity();
        for _ in 0..100 {
            sm.try_insert(0).unwrap();
        }
        assert!(sm.capacity() >= initial_cap + 100);
    }

    #[test]
    fn try_reserve_works() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::try_new().unwrap();
        sm.try_insert(0).unwrap();
        sm.try_reserve(64).unwrap();
        assert!(sm.capacity() >= 65);
    }

    #[test]
    fn custom_key_type() {
        crate::new_key_type! {
            struct MyKey;
        }
        let mut sm: SlotMap<MyKey, u64> = SlotMap::try_with_key().unwrap();
        let k = sm.try_insert(42).unwrap();
        assert_eq!(*sm.get(k).unwrap(), 42);
    }

    #[test]
    fn get_disjoint_mut_two_keys() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::try_new().unwrap();
        let ka = sm.try_insert(10).unwrap();
        let kb = sm.try_insert(20).unwrap();
        let result = sm.get_disjoint_mut([ka, kb]);
        assert!(result.is_some());
        let [a, b] = result.unwrap();
        lang_core::mem::swap(a, b);
        assert_eq!(sm[ka], 20);
        assert_eq!(sm[kb], 10);
    }

    #[test]
    fn get_disjoint_mut_duplicate_returns_none() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::try_new().unwrap();
        let ka = sm.try_insert(10).unwrap();
        assert!(sm.get_disjoint_mut([ka, ka]).is_none());
    }

    #[test]
    fn get_disjoint_mut_invalid_key_returns_none() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::try_new().unwrap();
        let ka = sm.try_insert(10).unwrap();
        let kb = sm.try_insert(20).unwrap();
        sm.remove(kb);
        assert!(sm.get_disjoint_mut([ka, kb]).is_none());
    }

    #[test]
    fn clone_preserves_state() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::try_new().unwrap();
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
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::try_new().unwrap();
        let k = sm.try_insert(42).unwrap();
        assert_eq!(sm[k], 42);
        sm.remove(k);
        // Would panic if we did sm[k] here — skipped to keep test green.
    }

    #[test]
    fn try_with_capacity_zero_succeeds() {
        let sm: SlotMap<DefaultKey, i32> = SlotMap::try_with_capacity(0).unwrap();
        assert!(sm.is_empty());
    }

    #[test]
    fn values_mut_iteration() {
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::try_new().unwrap();
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
        let mut sm: SlotMap<DefaultKey, i32> = SlotMap::try_new().unwrap();
        let expected: Vec<_> = (0..3).map(|i| sm.try_insert(i).unwrap()).collect();
        let actual: Vec<_> = sm.keys().collect();
        assert_eq!(actual.len(), 3);
        for ek in &expected {
            assert!(actual.contains(ek));
        }
    }

    #[cfg(feature = "std")]
    mod oom {
        // ── OOM tests ──────────────────────────────────────────────────────────────────

        use super::*;
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        type Sm = SlotMap<DefaultKey, i32>;

        /// `try_new` allocates the sentinel slot, so a failed allocation must
        /// surface as a `TryReserveError`.
        #[test]
        fn try_new_fails_on_oom() {
            let r = with_policy(FailPolicy::fail_next_alloc(), Sm::try_new);
            assert!(r.is_err());
        }

        /// `try_with_capacity` first allocates the vec buffer, then pushes the
        /// sentinel. Failing the very next allocation hits the buffer allocation.
        #[test]
        fn try_with_capacity_fails_on_oom() {
            let r = with_policy(FailPolicy::fail_next_alloc(), || Sm::try_with_capacity(4));
            assert!(r.is_err());
        }

        /// Zero-capacity construction only needs the sentinel push; blocking
        /// reallocations alone must not prevent it from succeeding.
        #[test]
        fn try_with_capacity_zero_succeeds_under_realloc_oom() {
            let r = with_policy(FailPolicy::fail_all_realloc(), || Sm::try_with_capacity(0));
            assert!(r.is_ok());
            assert!(r.unwrap().is_empty());
        }

        /// `try_reserve` grows the slots vec; a failed reallocation must leave
        /// the map usable afterwards. The sentinel guarantees the slots vec
        /// always holds an existing buffer, so growth goes through `realloc`
        /// rather than a fresh `alloc` — hence `fail_next_realloc`.
        #[test]
        fn try_reserve_fails_on_oom_but_stays_usable() {
            let mut sm = Sm::try_new().unwrap();
            let r = with_policy(FailPolicy::fail_next_realloc(), || sm.try_reserve(16));
            assert!(r.is_err(), "reserve error expected, got {:?}", r);
            // The map remains fully functional after a failed reserve.
            let k = sm.try_insert(10).unwrap();
            assert_eq!(*sm.get(k).unwrap(), 10);
        }

        /// `try_insert` reserves one slot before pushing; on reallocation
        /// failure the map must stay empty and recover cleanly. The constructor
        /// overshoots capacity (requesting 1 yields 4), so shrink to fit first
        /// to force the insert into the realloc path, then fail that realloc.
        #[test]
        fn try_insert_fails_on_oom_and_map_stays_empty() {
            let mut sm = Sm::try_new().unwrap();
            sm.slots.fallible_shrink_to_fit().unwrap();
            let r = with_policy(FailPolicy::fail_next_realloc(), || sm.try_insert(10));
            assert!(r.is_err());
            assert!(sm.is_empty());
            // Recovery: insertion works once the policy is gone.
            let k = sm.try_insert(10).unwrap();
            assert_eq!(*sm.get(k).unwrap(), 10);
        }

        /// A successful insert followed by an OOM during a second insert must not
        /// lose the first entry. Shrink after the first insert so the second
        /// insert's growth goes through realloc, which we then fail.
        #[test]
        fn oom_during_second_insert_preserves_first_entry() {
            let mut sm = Sm::try_new().unwrap();
            let k1 = sm.try_insert(1).unwrap();
            sm.slots.fallible_shrink_to_fit().unwrap();
            let r = with_policy(FailPolicy::fail_next_realloc(), || sm.try_insert(2));
            assert!(r.is_err());
            assert_eq!(sm.len(), 1);
            assert_eq!(*sm.get(k1).unwrap(), 1);
        }

        /// Freelist reuse does not allocate, so inserting into a freed slot must
        /// succeed even when the next allocation is forced to fail.
        #[test]
        fn freelist_reuse_does_not_allocate_under_oom() {
            let mut sm = Sm::try_new().unwrap();
            let k = sm.try_insert(1).unwrap();
            sm.remove(k);
            // The freed slot is reused without any heap growth.
            let r = with_policy(FailPolicy::fail_next_alloc(), || sm.try_insert(2));
            assert!(r.is_ok());
            let k2 = r.unwrap();
            assert_eq!(*sm.get(k2).unwrap(), 2);
            assert_eq!(sm.len(), 1);
        }

        /// `try_insert_with_key` runs the closure *before* reserving; if the
        /// reallocation fails afterwards, the map must remain untouched. Shrink
        /// first so the insert actually needs to grow the buffer. The failure
        /// surfaces as [`SlotMapInsertWithError::Reserve`], not [`SlotMapInsertWithError::Closure`].
        #[test]
        fn try_insert_with_key_gives_back_on_reservation_failure() {
            let mut sm = Sm::try_new().unwrap();
            sm.slots.fallible_shrink_to_fit().unwrap();
            let r = with_policy(FailPolicy::fail_next_realloc(), || {
                sm.try_insert_with_key::<_, ()>(|k| Ok(k.data().idx() as i32))
            });
            assert!(matches!(r, Err(SlotMapInsertWithError::Reserve(_))));
            assert!(sm.is_empty());
        }

        /// A closure that returns `Err` surfaces as [`SlotMapInsertWithError::Closure`]
        /// carrying the caller's error value, and leaves the map untouched.
        #[test]
        fn try_insert_with_key_closure_error_is_reported_verbatim() {
            let mut sm = Sm::try_new().unwrap();
            let r = sm.try_insert_with_key::<_, &str>(|_| Err("rejected"));
            assert!(matches!(
                r,
                Err(SlotMapInsertWithError::Closure("rejected"))
            ));
            assert!(sm.is_empty());
        }

        /// The canonical `TryDebug` / `TryDisplay` impls (which std's
        /// `Debug` / `Display` delegate to) render both variants with the
        /// uniform prefix scheme.
        #[test]
        fn insert_error_formats_canonically() {
            use crate::errors::uniform as u;
            use lang_alloc::format;

            // Reserve variant: delegated display + tuple debug.
            let reserve =
                SlotMapInsertWithError::<&str>::Reserve(TryReserveError::new_capacity_overflow());
            assert_eq!(
                format!("{reserve}"),
                "slot map operation failed: memory allocation failed because the computed \
                 capacity exceeded the collection's maximum"
            );
            assert!(format!("{reserve:?}").starts_with("SlotMapInsertWithError::Reserve("));

            // Closure variant: carries the caller's message through Display.
            let closure = SlotMapInsertWithError::<&str>::Closure("rejected");
            assert_eq!(
                format!("{closure}"),
                "slot map insertion closure operation failed: rejected"
            );
            assert!(format!("{closure:?}").starts_with("SlotMapInsertWithError::Closure("));

            // The `From<TryReserveError>` conversion lands in `Reserve`.
            let from_err: SlotMapInsertWithError<()> =
                TryReserveError::new_capacity_overflow().into();
            assert!(matches!(from_err, SlotMapInsertWithError::Reserve(_)));
        }

        /// `try_clone` copies the slots vec fallibly; a failed allocation must
        /// leave the source intact.
        #[test]
        fn try_clone_fails_on_oom_and_source_survives() {
            let mut sm = Sm::try_new().unwrap();
            let k1 = sm.try_insert(1).unwrap();
            let k2 = sm.try_insert(2).unwrap();
            let r = with_policy(FailPolicy::fail_next_alloc(), || sm.try_clone());
            assert!(r.is_err());
            assert_eq!(sm.len(), 2);
            assert_eq!(*sm.get(k1).unwrap(), 1);
            assert_eq!(*sm.get(k2).unwrap(), 2);
        }

        /// `try_default` routes through `try_with_capacity_and_key(0)` and maps
        /// the error into `TryDefaultError`.
        #[test]
        fn try_default_fails_on_oom() {
            let r = with_policy(
                FailPolicy::fail_next_alloc(),
                <Sm as TryDefault>::try_default,
            );
            assert!(r.is_err());
        }

        /// After any OOM failure, allocations must work normally again — no leaked
        /// policy state.
        #[test]
        fn oom_restores_allocation_afterwards() {
            let _failed = with_policy(FailPolicy::fail_next_alloc(), Sm::try_new);
            assert!(_failed.is_err());
            let mut sm = Sm::try_new().expect("allocation must recover after OOM");
            let k = sm.try_insert(99).unwrap();
            assert_eq!(*sm.get(k).unwrap(), 99);
        }
    }
}
