#![allow(unstable_name_collisions)]
use crate::alloc::AllocError;
use crate::boxed::TryBox;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};

use core::mem::{ManuallyDrop, MaybeUninit, offset_of};
use core::pin::Pin;
use core::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
/// Internal representation of an Arc allocation.
///
/// Layout matches `std::sync::Arc`: two atomic counters followed by the data.
/// `#[repr(C)]` is required so the compiler does not reorder fields — std's Arc
/// computes counter offsets relative to the data pointer and expects this exact
/// ordering. `align(2)` ensures that `Weak::new()`'s dangling sentinel
/// (`usize::MAX`) can never be a valid payload address, since all real allocations
/// are aligned to at least 2.
#[repr(C, align(2))]
struct ArcInner<T> {
    strong: AtomicUsize,
    weak: AtomicUsize,
    data: T,
}

/// A trait for fallibly constructing an [`Arc`].
///
/// Implemented for `Arc<T>`. Mirrors the [`TryBox`](crate::boxed::TryBox) pattern:
/// only the allocating constructors are fallible; all other Arc behaviour
/// (cloning, downgrading, dropping) delegates to the standard library.
///
/// # Construction strategy
///
/// Allocation is delegated to [`TryBox::try_new_uninit`] via a boxed
/// `MaybeUninit<ArcInner<T>>`. After initialising the strong/weak counters
/// and the data in place, ownership transfers to std's `Arc` through
/// [`Arc::from_raw`] — no second allocation is performed.
pub trait TryArc<T>: Sized {
    /// The uninitialized variant of this arc.
    type Uninit: Sized;

    /// Fallibly allocate a new `Arc<T>`.
    ///
    /// Returns [`AllocError`] if the heap allocation fails. Unlike
    /// [`Arc::new`], this never panics on out-of-memory.
    fn try_new(value: T) -> Result<Self, AllocError>;

    /// Fallibly allocate an uninitialised `Arc<MaybeUninit<T>>`.
    ///
    /// Returns an `Arc` wrapping `MaybeUninit<T>` that can be initialised
    /// in place via [`MaybeUninit::write`] and converted to an `Arc<T>` using
    /// [`Arc::into_inner`] + [`MaybeUninit::assume_init`].
    fn try_new_uninit() -> Result<Self::Uninit, AllocError>;

    /// Fallibly allocate zero-initialised memory as an `Arc<MaybeUninit<T>>`.
    ///
    /// Returns an `Arc` wrapping `MaybeUninit<T>` whose underlying bytes are
    /// all set to zero. Safe to call [`MaybeUninit::assume_init`] on types
    /// whose all-zeros bitpattern is valid (e.g. numeric primitives, `bool`,
    /// `[T; N]` where `T` is also zeroable).
    fn try_new_zeroed() -> Result<Self::Uninit, AllocError>;

    /// Like [`Self::try_new`] but returns ownership of `value` back on failure
    /// so it can be reused or dropped cleanly.
    fn try_new_give_back(value: T) -> Result<Self, (T, AllocError)>;

    /// Unwraps the value if this is the only strong reference, otherwise fallibly
    /// clones the inner data.
    ///
    /// This is a panic-free analogue of [`Arc::unwrap_or_clone`]. When there are
    /// other strong references, the inner value is cloned via [`TryClone`] rather
    /// than [`Clone`], so allocation failures during cloning (e.g. cloning a
    /// [`String`]) return an error instead of panicking.
    ///
    /// On failure, returns the original `Arc` alongside the clone error so the
    /// caller retains access to the shared data.
    fn unwrap_or_try_clone(self) -> Result<T, (Self, TryCloneError)>
    where
        T: Clone + crate::try_clone::TryClone;

    /// Fallibly allocate `value` on the heap and pin it in place.
    ///
    /// Returns a [`Pin<Arc<T>>`] so that if `T` does not implement [`Unpin`],
    /// the value is immovable after allocation. This is the fallible analogue
    /// of [`Arc::pin`].
    fn try_pin(value: T) -> Result<Pin<Self>, AllocError>;

    // ── Aliases with `fallible_` prefix to avoid name collisions ────────────

    /// Alias for [`Self::try_new`].
    fn fallible_new(value: T) -> Result<Self, AllocError> {
        Self::try_new(value)
    }

    /// Alias for [`Self::try_new_uninit`].
    fn fallible_new_uninit() -> Result<Self::Uninit, AllocError> {
        Self::try_new_uninit()
    }

    /// Alias for [`Self::try_new_zeroed`].
    fn fallible_new_zeroed() -> Result<Self::Uninit, AllocError> {
        Self::try_new_zeroed()
    }

    /// Alias for [`Self::try_new_give_back`].
    fn fallible_new_give_back(value: T) -> Result<Self, (T, AllocError)> {
        Self::try_new_give_back(value)
    }

    /// Alias for [`Self::try_pin`].
    fn fallible_pin(value: T) -> Result<Pin<Self>, AllocError> {
        Self::try_pin(value)
    }
}

impl<T> TryArc<T> for Arc<T> {
    type Uninit = Arc<MaybeUninit<T>>;

    fn try_new(value: T) -> Result<Self, AllocError> {
        let mut slot =
            ManuallyDrop::new(<Box<ArcInner<T>> as TryBox<ArcInner<T>>>::try_new_uninit()?);
        let inner = slot.as_mut_ptr();
        unsafe {
            ptr::write(&mut (*inner).strong, AtomicUsize::new(1));
            ptr::write(&mut (*inner).weak, AtomicUsize::new(1));
            ptr::write(&mut (*inner).data, value);
        }
        let data_ptr = unsafe { &raw const (*inner).data };
        Ok(unsafe { Arc::from_raw(data_ptr) })
    }

    fn try_new_uninit() -> Result<Self::Uninit, AllocError> {
        let mut slot =
            ManuallyDrop::new(<Box<ArcInner<T>> as TryBox<ArcInner<T>>>::try_new_uninit()?);
        let inner = slot.as_mut_ptr();
        unsafe {
            ptr::write(&mut (*inner).strong, AtomicUsize::new(1));
            ptr::write(&mut (*inner).weak, AtomicUsize::new(1));
            // data field stays uninitialised
        }
        let data_ptr = unsafe { &raw const (*inner).data as *const MaybeUninit<T> };
        Ok(unsafe { Arc::from_raw(data_ptr) })
    }

    fn try_new_zeroed() -> Result<Self::Uninit, AllocError> {
        let mut slot =
            ManuallyDrop::new(<Box<ArcInner<T>> as TryBox<ArcInner<T>>>::try_new_zeroed()?);
        let inner = slot.as_mut_ptr();
        // The entire allocation is zeroed — including the data region.
        // Fix up the refcount headers; leave data as zeroes.
        unsafe {
            ptr::write(&mut (*inner).strong, AtomicUsize::new(1));
            ptr::write(&mut (*inner).weak, AtomicUsize::new(1));
        }
        let data_ptr = unsafe { &raw const (*inner).data as *const MaybeUninit<T> };
        Ok(unsafe { Arc::from_raw(data_ptr) })
    }

    fn try_new_give_back(value: T) -> Result<Self, (T, AllocError)> {
        match <Box<ArcInner<T>> as TryBox<ArcInner<T>>>::try_new_uninit() {
            Ok(raw_slot) => {
                let mut slot = ManuallyDrop::new(raw_slot);
                let inner = slot.as_mut_ptr();
                unsafe {
                    ptr::write(&mut (*inner).strong, AtomicUsize::new(1));
                    ptr::write(&mut (*inner).weak, AtomicUsize::new(1));
                    ptr::write(&mut (*inner).data, value);
                }
                let data_ptr = unsafe { &raw const (*inner).data };
                Ok(unsafe { Arc::from_raw(data_ptr) })
            }
            Err(e) => Err((value, e)),
        }
    }

    fn unwrap_or_try_clone(self) -> Result<T, (Self, TryCloneError)>
    where
        T: crate::try_clone::TryClone,
    {
        match Arc::try_unwrap(self) {
            Ok(val) => Ok(val),
            Err(arc) => match (*arc).try_clone() {
                Ok(cloned) => Ok(cloned),
                Err(e) => Err((arc, e)),
            },
        }
    }

    fn try_pin(value: T) -> Result<Pin<Self>, AllocError> {
        let arc = Self::try_new(value)?;
        Ok(unsafe { Pin::new_unchecked(arc) })
    }
}

/// Maximum reference count, matching std's `Arc::MAX_REFCOUNT`.
/// Beyond this threshold, clones are rejected to avoid counter overflow.
const MAX_REFCOUNT: usize = (isize::MAX) as usize;

/// Byte offset from the data pointer back to the start of ArcInner.
/// Computed with a concrete inner type (u8) — the header size is identical for
/// every T since [strong][weak] always precedes [data].
const HEADER_SIZE: usize = offset_of!(ArcInner<u8>, data);

/// Byte offset from the start of ArcInner to the `weak` atomic.
const WEAK_OFFSET: usize = offset_of!(ArcInner<u8>, weak);

impl<T: ?Sized> TryClone for Arc<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let data_ptr: *const T = self.as_ref();
        let strong_ptr = unsafe { data_ptr.byte_sub(HEADER_SIZE) }.cast::<AtomicUsize>();

        let ok = unsafe {
            (*strong_ptr).fetch_update(Ordering::Acquire, Ordering::Relaxed, |cur| {
                if cur >= MAX_REFCOUNT {
                    None
                } else {
                    Some(cur + 1)
                }
            })
        };

        match ok {
            Ok(_) => Ok(unsafe { core::ptr::read(self) }),
            Err(_) => Err(TryCloneError::Other("Arc strong refcount exceeded")),
        }
    }
}

impl<T: ?Sized> TryClone for Weak<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Check for the dangling sentinel used by Weak::new() — no allocation exists,
        // so cloning is just a pointer copy that can never fail.
        let data_ptr = self.as_ptr();
        if data_ptr.addr() == usize::MAX {
            return Ok(unsafe { core::ptr::read(self) });
        }

        // Walk from the data pointer back to the `weak` atomic inside ArcInner.
        // Layout: [strong][weak][data]. The distance from data back to weak is
        // HEADER_SIZE - WEAK_OFFSET, which is just sizeof(AtomicUsize).
        let weak_ptr =
            unsafe { data_ptr.byte_sub(HEADER_SIZE - WEAK_OFFSET) }.cast::<AtomicUsize>();

        let ok = unsafe {
            (*weak_ptr).fetch_update(Ordering::Acquire, Ordering::Relaxed, |cur| {
                if cur >= MAX_REFCOUNT {
                    None
                } else {
                    Some(cur + 1)
                }
            })
        };

        match ok {
            Ok(_) => Ok(unsafe { core::ptr::read(self) }),
            Err(_) => Err(TryCloneError::Other("Arc weak refcount exceeded")),
        }
    }
}

impl<T: TryDefault> TryDefault for Arc<T> {
    fn try_default() -> Result<Self, TryDefaultError> {
        // Allocate first so that if allocation fails we never touch T::try_default().
        let mut uninit = <Arc<T> as TryArc<T>>::try_new_uninit().map_err(TryDefaultError::Alloc)?;
        match T::try_default() {
            Ok(val) => {
                // We have sole ownership (freshly allocated, strong count == 1).
                Arc::get_mut(&mut uninit).unwrap().write(val);
                Ok(unsafe { uninit.assume_init() })
            }
            Err(e) => Err(e),
        }
    }
}

impl<T> TryDefault for Weak<T> {
    fn try_default() -> Result<Self, TryDefaultError> {
        // Weak::new() creates a dangling weak pointer — no allocation needed.
        Ok(Weak::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_basic() {
        let arc = Arc::<i32>::try_new(42).unwrap();
        assert_eq!(*arc, 42);
    }

    #[test]
    fn try_new_zst() {
        let arc = Arc::<()>::try_new(()).unwrap();
        assert_eq!(*arc, ());
    }

    #[test]
    fn clone_increments_strong() {
        let arc = Arc::<String>::try_new("hello".to_string()).unwrap();
        assert_eq!(Arc::strong_count(&arc), 1);
        let arc2 = arc.clone();
        assert_eq!(Arc::strong_count(&arc), 2);
        assert_eq!(Arc::strong_count(&arc2), 2);
        drop(arc2);
        assert_eq!(Arc::strong_count(&arc), 1);
    }

    #[test]
    fn strong_and_weak_counts() {
        let arc = Arc::<i32>::try_new(99).unwrap();
        assert_eq!(Arc::strong_count(&arc), 1);
        assert_eq!(Arc::weak_count(&arc), 0);

        let weak = Arc::downgrade(&arc);
        assert_eq!(Arc::weak_count(&arc), 1);

        let _weak2 = weak.clone();
        assert_eq!(Arc::weak_count(&arc), 2);

        drop(weak);
        assert_eq!(Arc::weak_count(&arc), 1);
    }

    #[test]
    fn downgrade_upgrade() {
        let arc = Arc::<i32>::try_new(42).unwrap();
        let weak = Arc::downgrade(&arc);

        // Upgrade while arc exists
        {
            let upgraded = weak.upgrade().unwrap();
            assert_eq!(*upgraded, 42);
        }

        drop(arc);
        // After last strong ref is gone, upgrade fails
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn try_unwrap_unique() {
        let arc = Arc::<i32>::try_new(42).unwrap();
        assert_eq!(Arc::try_unwrap(arc), Ok(42));
    }

    #[test]
    fn try_unwrap_not_unique() {
        let arc = Arc::<i32>::try_new(42).unwrap();
        let _arc2 = arc.clone();
        assert_eq!(*Arc::try_unwrap(arc).unwrap_err(), 42);
    }

    #[test]
    fn into_inner_race_simulation() {
        let arc = Arc::<i32>::try_new(42).unwrap();
        let arc2 = arc.clone();
        let r1 = Arc::into_inner(arc);
        let r2 = Arc::into_inner(arc2);
        assert!(r1.is_some() ^ r2.is_some());
        assert!(r1 == Some(42) || r2 == Some(42));
    }

    #[test]
    fn ptr_eq_same_allocation() {
        let arc = Arc::<i32>::try_new(42).unwrap();
        let arc2 = arc.clone();
        assert!(Arc::ptr_eq(&arc, &arc2));
        let arc3 = Arc::<i32>::try_new(42).unwrap();
        assert!(!Arc::ptr_eq(&arc, &arc3));
    }

    #[test]
    fn into_raw_from_raw() {
        let arc = Arc::<i32>::try_new(42).unwrap();
        let ptr = Arc::into_raw(arc);
        assert_eq!(unsafe { *ptr }, 42);
        let arc = unsafe { Arc::from_raw(ptr) };
        assert_eq!(*arc, 42);
    }

    #[test]
    fn get_mut_unique() {
        let mut arc = Arc::<i32>::try_new(42).unwrap();
        let val = Arc::get_mut(&mut arc);
        assert!(val.is_some());
        *val.unwrap() = 99;
        assert_eq!(*arc, 99);
    }

    #[test]
    fn get_mut_not_unique() {
        let mut arc = Arc::<i32>::try_new(42).unwrap();
        let _arc2 = arc.clone();
        assert!(Arc::get_mut(&mut arc).is_none());
    }

    #[test]
    fn debug_output() {
        let arc = Arc::<i32>::try_new(42).unwrap();
        let s = format!("{:?}", arc);
        assert_eq!(s, "42");
    }

    #[test]
    fn display_output() {
        let arc = Arc::<i32>::try_new(42).unwrap();
        let s = format!("{}", arc);
        assert_eq!(s, "42");
    }

    #[test]
    fn partial_eq() {
        let a = Arc::<i32>::try_new(42).unwrap();
        let b = Arc::<i32>::try_new(42).unwrap();
        let c = Arc::<i32>::try_new(43).unwrap();
        assert!(a == b);
        assert!(a != c);
    }

    // ── try_new_uninit tests ────────────────────────────────────────────────

    #[test]
    fn try_new_uninit_then_write() {
        let mut uninit: Arc<MaybeUninit<i32>> = Arc::<i32>::try_new_uninit().unwrap();
        // Write value in place — we have sole ownership so get_mut succeeds.
        Arc::get_mut(&mut uninit).unwrap().write(77);
        // Extract the MaybeUninit, assume init, and wrap in a fresh Arc.
        let owned = Arc::try_unwrap(uninit).unwrap();
        let val = Arc::new(unsafe { owned.assume_init() });
        assert_eq!(*val, 77);
    }

    #[test]
    fn try_new_uninit_zst() {
        let _uninit: Arc<MaybeUninit<()>> = Arc::<()>::try_new_uninit().unwrap();
    }

    // ── try_new_zeroed tests ────────────────────────────────────────────────

    #[test]
    fn try_new_zeroed_returns_zeros() {
        let uninit: Arc<MaybeUninit<i32>> = Arc::<i32>::try_new_zeroed().unwrap();
        let owned = Arc::try_unwrap(uninit).unwrap();
        let val = unsafe { owned.assume_init() };
        assert_eq!(val, 0);
    }

    #[test]
    fn try_new_zeroed_f64_is_positive_zero() {
        let uninit: Arc<MaybeUninit<f64>> = Arc::<f64>::try_new_zeroed().unwrap();
        let owned = Arc::try_unwrap(uninit).unwrap();
        let val = unsafe { owned.assume_init() };
        assert_eq!(val.to_bits(), 0);
    }

    #[test]
    fn try_new_zeroed_array() {
        let uninit: Arc<MaybeUninit<[u8; 4]>> = Arc::<[u8; 4]>::try_new_zeroed().unwrap();
        let owned = Arc::try_unwrap(uninit).unwrap();
        let arr = unsafe { owned.assume_init() };
        assert_eq!(arr, [0, 0, 0, 0]);
    }

    #[test]
    fn try_new_zeroed_zst() {
        let _uninit: Arc<MaybeUninit<()>> = Arc::<()>::try_new_zeroed().unwrap();
    }

    // ── try_new_give_back tests ─────────────────────────────────────────────

    #[test]
    fn try_new_give_back_success() {
        let arc = Arc::<String>::try_new_give_back("hello".to_string()).unwrap();
        assert_eq!(arc.as_str(), "hello");
    }

    #[test]
    fn try_new_give_back_signature() {
        let val = vec![1, 2, 3];
        let result: Result<Arc<Vec<i32>>, (Vec<i32>, AllocError)> =
            Arc::<Vec<i32>>::try_new_give_back(val);
        let _arc_val = result.unwrap();
    }

    // ── fallible_ alias tests ───────────────────────────────────────────────

    #[test]
    fn fallible_new_returns_value() {
        let arc = Arc::<i32>::fallible_new(42).unwrap();
        assert_eq!(*arc, 42);
    }

    #[test]
    fn fallible_new_uninit_works() {
        let _uninit: Arc<MaybeUninit<i32>> = Arc::<i32>::fallible_new_uninit().unwrap();
    }

    #[test]
    fn fallible_new_zeroed_works() {
        let uninit: Arc<MaybeUninit<u64>> = Arc::<u64>::fallible_new_zeroed().unwrap();
        let owned = Arc::try_unwrap(uninit).unwrap();
        let val = unsafe { owned.assume_init() };
        assert_eq!(val, 0);
    }

    #[test]
    fn fallible_new_give_back_success() {
        let arc = Arc::<String>::fallible_new_give_back("world".to_string()).unwrap();
        assert_eq!(arc.as_str(), "world");
    }

    // ── unwrap_or_try_clone tests ──────────────────────────────────────────────

    #[test]
    fn unwrap_or_try_clone_unique_returns_inner() {
        let arc = Arc::<[u8; 4]>::try_new(*b"hi!!").unwrap();
        let val = Arc::<[u8; 4]>::unwrap_or_try_clone(arc).unwrap();
        assert_eq!(val, *b"hi!!");
    }

    #[test]
    fn unwrap_or_try_clone_shared_clones() {
        let arc = Arc::<i32>::try_new(42).unwrap();
        let _arc2 = arc.clone();
        let val = Arc::<i32>::unwrap_or_try_clone(arc).unwrap();
        assert_eq!(val, 42);
    }

    #[test]
    fn unwrap_or_try_clone_with_option() {
        let arc = Arc::<Option<i32>>::try_new(Some(99)).unwrap();
        let _arc2 = arc.clone();
        let val = Arc::<Option<i32>>::unwrap_or_try_clone(arc).unwrap();
        assert_eq!(val, Some(99));
    }

    // ── TryClone tests ────────────────────────────────────────────────────────

    #[test]
    fn try_clone_increments_strong() {
        use crate::try_clone::TryClone;
        let arc = Arc::<i32>::try_new(42).unwrap();
        assert_eq!(Arc::strong_count(&arc), 1);
        let arc2 = arc.try_clone().unwrap();
        assert_eq!(Arc::strong_count(&arc), 2);
        assert_eq!(Arc::strong_count(&arc2), 2);
        assert!(Arc::ptr_eq(&arc, &arc2));
    }

    #[test]
    fn try_clone_preserves_value() {
        use crate::try_clone::TryClone;
        let arc = Arc::<String>::try_new("hello".to_string()).unwrap();
        let arc2 = arc.try_clone().unwrap();
        assert_eq!(arc2.as_str(), "hello");
    }

    #[test]
    fn try_clone_multiple() {
        use crate::try_clone::TryClone;
        let arc = Arc::<u64>::try_new(999).unwrap();
        let mut clones = Vec::new();
        for i in 0..100 {
            clones.push(arc.try_clone().unwrap());
            assert_eq!(Arc::strong_count(&arc), i as usize + 2);
        }
        // Drop all clones and verify count returns to 1.
        drop(clones);
        assert_eq!(Arc::strong_count(&arc), 1);
    }

    #[test]
    fn try_clone_rejects_at_max_refcount() {
        use crate::try_clone::TryClone;
        // Manually set the strong count to MAX_REFCOUNT so the closure rejects.
        let arc = Arc::<i32>::try_new(0).unwrap();
        let data_ptr: *const i32 = arc.as_ref();
        let header_size = offset_of!(ArcInner<i32>, data);
        let strong_ptr = unsafe { data_ptr.byte_sub(header_size) }.cast::<AtomicUsize>();
        unsafe { (*strong_ptr).store(MAX_REFCOUNT, Ordering::Relaxed) };
        assert!(arc.try_clone().is_err());
        // Restore so the Arc drops cleanly.
        unsafe { (*strong_ptr).store(1, Ordering::Relaxed) };
    }

    #[test]
    fn try_clone_rejects_above_max_refcount() {
        use crate::try_clone::TryClone;
        let arc = Arc::<i32>::try_new(0).unwrap();
        let data_ptr: *const i32 = arc.as_ref();
        let header_size = offset_of!(ArcInner<i32>, data);
        let strong_ptr = unsafe { data_ptr.byte_sub(header_size) }.cast::<AtomicUsize>();
        unsafe { (*strong_ptr).store(MAX_REFCOUNT + 1, Ordering::Relaxed) };
        assert!(arc.try_clone().is_err());
        unsafe { (*strong_ptr).store(1, Ordering::Relaxed) };
    }

    // ── Weak TryClone tests ───────────────────────────────────────────────────

    #[test]
    fn weak_try_clone_increments_weak_count() {
        use crate::try_clone::TryClone;
        let arc = Arc::<i32>::try_new(42).unwrap();
        let weak = Arc::downgrade(&arc);
        assert_eq!(Arc::weak_count(&arc), 1);
        let weak2 = weak.try_clone().unwrap();
        assert_eq!(Arc::weak_count(&arc), 2);
        drop(weak2);
        assert_eq!(Arc::weak_count(&arc), 1);
    }

    #[test]
    fn weak_try_clone_dangling_succeeds() {
        use crate::try_clone::TryClone;
        let weak: Weak<i32> = Weak::new();
        let weak2 = weak.try_clone().unwrap();
        assert!(weak.upgrade().is_none());
        assert!(weak2.upgrade().is_none());
    }

    #[test]
    fn weak_try_clone_multiple() {
        use crate::try_clone::TryClone;
        let arc = Arc::<u64>::try_new(0).unwrap();
        let weak = Arc::downgrade(&arc);
        let mut clones = Vec::new();
        for i in 0..100 {
            clones.push(weak.try_clone().unwrap());
            assert_eq!(Arc::weak_count(&arc), i as usize + 2);
        }
        drop(clones);
        assert_eq!(Arc::weak_count(&arc), 1);
    }

    #[test]
    fn weak_try_clone_rejects_at_max_refcount() {
        use crate::try_clone::TryClone;
        let arc = Arc::<i32>::try_new(0).unwrap();
        let weak = Arc::downgrade(&arc);
        let data_ptr: *const i32 = arc.as_ref();
        let header_size = offset_of!(ArcInner<i32>, data);
        let weak_offset = offset_of!(ArcInner<i32>, weak);
        let weak_ptr =
            unsafe { data_ptr.byte_sub(header_size - weak_offset) }.cast::<AtomicUsize>();
        // Overwrite after downgrade so we don't trigger std's own overflow check.
        unsafe { (*weak_ptr).store(MAX_REFCOUNT, Ordering::Relaxed) };
        assert!(weak.try_clone().is_err());
        // Restore so everything drops cleanly.
        unsafe { (*weak_ptr).store(1, Ordering::Relaxed) };
        drop(arc);
    }

    #[test]
    fn weak_try_clone_rejects_above_max_refcount() {
        use crate::try_clone::TryClone;
        let arc = Arc::<i32>::try_new(0).unwrap();
        let weak = Arc::downgrade(&arc);
        let data_ptr: *const i32 = arc.as_ref();
        let header_size = offset_of!(ArcInner<i32>, data);
        let weak_offset = offset_of!(ArcInner<i32>, weak);
        let weak_ptr =
            unsafe { data_ptr.byte_sub(header_size - weak_offset) }.cast::<AtomicUsize>();
        // Overwrite after downgrade so we don't trigger std's own overflow check.
        unsafe { (*weak_ptr).store(MAX_REFCOUNT + 1, Ordering::Relaxed) };
        assert!(weak.try_clone().is_err());
        unsafe { (*weak_ptr).store(1, Ordering::Relaxed) };
        drop(arc);
    }

    // ── Unsized Arc TryClone tests ──────────────────────────────────────────────

    #[test]
    fn arc_slice_try_clone() {
        use crate::try_clone::TryClone;
        let slice: &[i32] = &[1, 2, 3, 4, 5];
        let arc: Arc<[i32]> = Arc::from(slice);
        let arc2 = arc.try_clone().unwrap();
        assert_eq!(arc2.as_ref(), [1, 2, 3, 4, 5]);
        assert_eq!(Arc::strong_count(&arc), 2);
        assert!(Arc::ptr_eq(&arc, &arc2));
    }

    #[test]
    fn arc_dyn_trait_try_clone() {
        use crate::try_clone::TryClone;
        struct Greeter(String);
        impl std::fmt::Display for Greeter {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "Hello, {}!", self.0)
            }
        }
        let arc: Arc<dyn std::fmt::Display> = Arc::new(Greeter("world".into()));
        let arc2 = arc.try_clone().unwrap();
        assert_eq!(Arc::strong_count(&arc), 2);
        assert!(Arc::ptr_eq(&arc, &arc2));
    }

    #[test]
    fn arc_dyn_trait_try_clone_multiple() {
        use crate::try_clone::TryClone;
        let arc: Arc<dyn std::fmt::Debug> = Arc::new(42i32);
        let mut clones = Vec::new();
        for i in 0..50 {
            clones.push(arc.try_clone().unwrap());
            assert_eq!(Arc::strong_count(&arc), i as usize + 2);
        }
        drop(clones);
        assert_eq!(Arc::strong_count(&arc), 1);
    }

    // ── Unsized Weak TryClone tests ─────────────────────────────────────────────

    #[test]
    fn weak_slice_try_clone() {
        use crate::try_clone::TryClone;
        let arc: Arc<[i32]> = Arc::from(&[10, 20, 30][..]);
        let weak = Arc::downgrade(&arc);
        assert_eq!(Arc::weak_count(&arc), 1);
        let weak2 = weak.try_clone().unwrap();
        assert_eq!(Arc::weak_count(&arc), 2);
        drop(weak2);
        assert_eq!(Arc::weak_count(&arc), 1);
    }

    #[test]
    fn weak_dyn_trait_try_clone() {
        use crate::try_clone::TryClone;
        let arc: Arc<dyn std::fmt::Display> = Arc::new(99i64);
        let weak = Arc::downgrade(&arc);
        assert_eq!(Arc::weak_count(&arc), 1);
        let weak2 = weak.try_clone().unwrap();
        assert_eq!(Arc::weak_count(&arc), 2);
        drop(weak2);
        assert_eq!(Arc::weak_count(&arc), 1);
    }

    #[test]
    fn weak_slice_try_clone_multiple() {
        use crate::try_clone::TryClone;
        let arc: Arc<[u8]> = Arc::from([0u8, 1, 2, 3]);
        let weak = Arc::downgrade(&arc);
        let mut clones = Vec::new();
        for i in 0..50 {
            clones.push(weak.try_clone().unwrap());
            assert_eq!(Arc::weak_count(&arc), i as usize + 2);
        }
        drop(clones);
        assert_eq!(Arc::weak_count(&arc), 1);
    }

    // ── try_pin tests ─────────────────────────────────────────────────────────

    #[test]
    fn arc_try_pin_returns_pinned() {
        let _pinned: Pin<Arc<i32>> = Arc::<i32>::try_pin(42).unwrap();
    }

    #[test]
    fn arc_try_pin_value_accessible() {
        let pinned: Pin<Arc<u64>> = Arc::<u64>::try_pin(999).unwrap();
        let val: &u64 = &pinned;
        assert_eq!(*val, 999);
    }

    #[test]
    fn arc_try_pin_zst() {
        let _pinned: Pin<Arc<()>> = Arc::<()>::try_pin(()).unwrap();
    }

    // ── fallible_ alias tests for pin ─────────────────────────────────────────

    #[test]
    fn fallible_pin_works() {
        let _pinned: Pin<Arc<i32>> = Arc::<i32>::fallible_pin(42).unwrap();
    }
}
