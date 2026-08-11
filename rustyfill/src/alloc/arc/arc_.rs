use crate::alloc::AllocError;
use crate::alloc::boxed::TryBox;
use crate::lang_alloc::boxed::Box;
use crate::lang_core::fmt;
use crate::lang_core::mem::{ManuallyDrop, MaybeUninit};
use crate::lang_core::pin::Pin;
use crate::lang_core::ptr;
use crate::lang_core::sync::atomic::{AtomicUsize, Ordering};
use crate::lang_std::sync::{Arc, Weak};
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};

/// Internal representation of an Arc allocation.
///
/// Layout matches `::std::sync::Arc`: two atomic counters followed by the data.
/// `#[repr(C)]` is required so the compiler does not reorder fields — std's Arc
/// computes counter offsets relative to the data pointer and expects this exact
/// ordering. `align(2)` ensures that `Weak::new()`'s dangling sentinel
/// (`usize::MAX`) can never be a valid payload address, since all real allocations
/// are aligned to at least 2.
#[repr(C, align(2))]
struct ArcInner<T: ?Sized> {
    strong: AtomicUsize,
    weak: AtomicUsize,
    data: T,
}

/// A trait for fallibly constructing an [`Arc`].
///
/// Implemented for `Arc<T>`. Mirrors the [`TryBox`](crate::alloc::boxed::TryBox) pattern:
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
    /// **Deprecated:** This method name conflicts with the unstable inherent
    /// [`Arc::try_new`]. Use [`Self::fallible_new`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with unstable Arc::try_new; use fallible_new"
    )]
    fn try_new(value: T) -> Result<Self, AllocError>;

    /// Fallibly allocate an uninitialised `Arc<MaybeUninit<T>>`.
    ///
    /// **Deprecated:** This method name conflicts with the unstable inherent
    /// [`Arc::try_new_uninit`]. Use [`Self::fallible_new_uninit`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with unstable Arc::try_new_uninit; use fallible_new_uninit"
    )]
    fn try_new_uninit() -> Result<Self::Uninit, AllocError>;

    /// Fallibly allocate zero-initialised memory as an `Arc<MaybeUninit<T>>`.
    ///
    /// **Deprecated:** This method name conflicts with the unstable inherent
    /// [`Arc::try_new_zeroed`]. Use [`Self::fallible_new_zeroed`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with unstable Arc::try_new_zeroed; use fallible_new_zeroed"
    )]
    fn try_new_zeroed() -> Result<Self::Uninit, AllocError>;

    /// Like [`Self::try_new`] but returns ownership of `value` back on failure.
    ///
    /// On success, returns the newly allocated `Arc<T>`. On allocation failure,
    /// returns the original `value` alongside the [`AllocError`] so the caller
    /// can reuse or drop it cleanly rather than losing it to an OOM panic.
    fn try_new_give_back(value: T) -> Result<Self, (T, AllocError)>;

    /// Unwraps the value if this is the only strong reference, otherwise fallibly
    /// clones the inner data.
    ///
    /// This is a panic-free analogue of [`Arc::unwrap_or_clone`]. When there are
    /// other strong references, the inner value is cloned via [`TryClone`] rather
    /// than [`Clone`], so allocation failures during cloning (e.g. cloning a
    /// [`crate::lang_alloc::string::String`]) return an error instead of panicking.
    ///
    /// On failure, returns the original `Arc` alongside the clone error so the
    /// caller retains access to the shared data.
    fn unwrap_or_try_clone(self) -> Result<T, (Self, TryCloneError)>
    where
        T: Clone + crate::try_clone::TryClone;

    /// Fallibly allocate `value` on the heap and pin it in place.
    ///
    /// **Deprecated:** This method name conflicts with the unstable inherent
    /// [`Arc::try_pin`]. Use [`Self::fallible_pin`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with unstable Arc::try_pin; use fallible_pin"
    )]
    fn try_pin(value: T) -> Result<Pin<Self>, AllocError>;

    /// Like [`Self::try_pin`] but returns ownership of `value` back on failure.
    ///
    /// On success, returns the pinned `Arc<T>`. On allocation failure, returns
    /// the original `value` alongside the [`AllocError`] so the caller retains
    /// access to the shared data.
    fn try_pin_give_back(value: T) -> Result<Pin<Self>, (T, AllocError)>;

    // ── Aliases with `fallible_` prefix to avoid name collisions ────────────

    /// Fallibly allocate a new `Arc<T>`.
    ///
    /// Returns [`AllocError`] if the heap allocation fails. Unlike
    /// [`Arc::new`], this never panics on out-of-memory.
    ///
    /// Allocation is performed by first allocating a boxed
    /// `MaybeUninit<ArcInner<T>>` via [`TryBox::fallible_new_uninit`], then
    /// transferring ownership to std's `Arc` through [`Arc::from_raw`] — no
    /// second allocation is performed.
    ///
    /// This method replaces the deprecated [`Self::try_new`] which shares its
    /// name with the unstable inherent [`Arc::try_new`].
    #[allow(deprecated)]
    fn fallible_new(value: T) -> Result<Self, AllocError> {
        Self::try_new(value)
    }

    /// Fallibly allocate an uninitialised `Arc<MaybeUninit<T>>`.
    ///
    /// Returns an `Arc` wrapping `MaybeUninit<T>` that can be initialised
    /// in place via [`MaybeUninit::write`] and converted to an `Arc<T>` using
    /// [`Arc::into_inner`] + [`MaybeUninit::assume_init`].
    ///
    /// This method replaces the deprecated [`Self::try_new_uninit`] which shares
    /// its name with the unstable inherent [`Arc::try_new_uninit`].
    #[allow(deprecated)]
    fn fallible_new_uninit() -> Result<Self::Uninit, AllocError> {
        Self::try_new_uninit()
    }

    /// Fallibly allocate zero-initialised memory as an `Arc<MaybeUninit<T>>`.
    ///
    /// Returns an `Arc` wrapping `MaybeUninit<T>` whose underlying bytes are
    /// all set to zero. Safe to call [`MaybeUninit::assume_init`] on types
    /// whose all-zeros bitpattern is valid (e.g. numeric primitives, `bool`,
    /// `[T; N]` where `T` is also zeroable).
    ///
    /// This method replaces the deprecated [`Self::try_new_zeroed`] which shares
    /// its name with the unstable inherent [`Arc::try_new_zeroed`].
    #[allow(deprecated)]
    fn fallible_new_zeroed() -> Result<Self::Uninit, AllocError> {
        Self::try_new_zeroed()
    }

    /// Alias for [`Self::try_new_give_back`].
    fn fallible_new_give_back(value: T) -> Result<Self, (T, AllocError)> {
        Self::try_new_give_back(value)
    }

    /// Unwraps the value if this is the only strong reference, otherwise fallibly
    /// clones the inner data.
    ///
    /// This is a panic-free analogue of [`Arc::unwrap_or_clone`] using
    /// [`TryClone`] instead of [`Clone`]. On failure, returns the original `Arc`
    /// alongside the clone error so the caller retains access to the shared data.
    ///
    /// This method replaces [`Self::unwrap_or_try_clone`] under a name that
    /// won't collide with future std additions.
    fn unwrap_or_fallible_clone(self) -> Result<T, (Self, TryCloneError)>
    where
        T: Clone + crate::try_clone::TryClone,
    {
        Self::unwrap_or_try_clone(self)
    }

    /// Fallibly allocate `value` on the heap and pin it in place.
    ///
    /// Returns a [`Pin<Arc<T>>`] so that if `T` does not implement [`Unpin`],
    /// the value is immovable after allocation. This is the fallible analogue
    /// of [`Arc::pin`].
    ///
    /// This method replaces the deprecated [`Self::try_pin`] which shares its
    /// name with the unstable inherent [`Arc::try_pin`].
    #[allow(deprecated)]
    fn fallible_pin(value: T) -> Result<Pin<Self>, AllocError> {
        Self::try_pin(value)
    }

    /// Alias for [`Self::try_pin_give_back`].
    fn fallible_pin_give_back(value: T) -> Result<Pin<Self>, (T, AllocError)> {
        Self::try_pin_give_back(value)
    }
}

#[allow(deprecated)]
impl<T> TryArc<T> for Arc<T> {
    type Uninit = Arc<MaybeUninit<T>>;

    fn try_new(value: T) -> Result<Self, AllocError> {
        let mut slot =
            ManuallyDrop::new(<Box<ArcInner<T>> as TryBox<ArcInner<T>>>::try_new_uninit()?);
        let inner = slot.as_mut_ptr();
        // The pointer is always valid, cannot panic.
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
        // The pointer is always valid, cannot panic.
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
        // The pointer is always valid, cannot panic.
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
                // The pointer is always valid, cannot panic.
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
        let arc = <Self as TryArc<T>>::try_new(value)?;
        Ok(unsafe { Pin::new_unchecked(arc) })
    }

    fn try_pin_give_back(value: T) -> Result<Pin<Self>, (T, AllocError)> {
        match Self::try_new_give_back(value) {
            Ok(arc) => Ok(unsafe { Pin::new_unchecked(arc) }),
            Err((v, e)) => Err((v, e)),
        }
    }
}

/// Maximum reference count, matching std's `Arc::MAX_REFCOUNT`.
/// Beyond this threshold, clones are rejected to avoid counter overflow.
const MAX_REFCOUNT: usize = (isize::MAX) as usize;

/// Helper: obtain a reference to the `ArcInner<T>` stored inside an `Arc<T>`.
///
/// `Arc<T>`'s first field is `ptr: NonNull<ArcInner<T>>`. By casting `&Arc<T>`
/// to a raw pointer and reinterpreting it as `*const NonNull<ArcInner<T>>`, we
/// read out the inner pointer and dereference it into a `&ArcInner<T>`. This
/// avoids any byte-offset arithmetic and works correctly for all types including
/// DSTs with custom alignment (e.g. `#[repr(align(512))]`).
fn arc_inner<T: ?Sized>(arc: &Arc<T>) -> &ArcInner<T> {
    // SAFETY: Arc's first field is `ptr: NonNull<ArcInner<T>>`. Casting `&Arc<T>`
    // to `*const NonNull<ArcInner<T>>` is valid because NonNull has the same
    // layout as a raw pointer and the first field of a struct is at offset 0.
    // The NonNull points into a live heap allocation owned by the Arc, so
    // dereferencing yields a valid &ArcInner<T>.
    unsafe {
        let ptr_field: *const ptr::NonNull<ArcInner<T>> = ptr::from_ref(arc).cast();
        let non_null: ptr::NonNull<ArcInner<T>> = *ptr_field;
        non_null.as_ref()
    }
}

/// Helper: obtain a reference to the `ArcInner<T>` stored inside a `Weak<T>`.
///
/// Same approach as [`arc_inner`] but for `Weak`. `Weak<T>` also stores
/// `ptr: NonNull<ArcInner<T>>` as its first field. For the dangling sentinel
/// (`Weak::new()`), the pointer address is `usize::MAX` and must be checked
/// by the caller before invoking this function.
fn weak_inner<T: ?Sized>(weak: &Weak<T>) -> &ArcInner<T> {
    // SAFETY: Identical reasoning to arc_inner. Weak's first field is
    // `ptr: NonNull<ArcInner<T>>`. Caller must ensure this isn't a dangling
    // Weak (address != usize::MAX) before calling.
    unsafe {
        let ptr_field: *const ptr::NonNull<ArcInner<T>> = ptr::from_ref(weak).cast();
        let non_null: ptr::NonNull<ArcInner<T>> = *ptr_field;
        non_null.as_ref()
    }
}

/// Returns true if this `Weak` is a dangling sentinel (created via `Weak::new()`).
fn weak_is_dangling<T: ?Sized>(weak: &Weak<T>) -> bool {
    weak.as_ptr().addr() == usize::MAX
}

impl<T: ?Sized> TryClone for Arc<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let inner = arc_inner(self);

        let ok = inner
            .strong
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                if cur >= MAX_REFCOUNT {
                    None
                } else {
                    Some(cur + 1)
                }
            });

        match ok {
            Ok(_) => Ok(unsafe { ptr::read(self) }),
            Err(_) => Err(TryCloneError::Other("Arc strong refcount exceeded")),
        }
    }
}

impl<T: ?Sized> TryClone for Weak<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Check for the dangling sentinel used by Weak::new() — no allocation exists,
        // so cloning is just a pointer copy that can never fail.
        if weak_is_dangling(self) {
            return Ok(unsafe { ptr::read(self) });
        }

        let inner = weak_inner(self);

        let ok = inner
            .weak
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                if cur >= MAX_REFCOUNT {
                    None
                } else {
                    Some(cur + 1)
                }
            });

        match ok {
            Ok(_) => Ok(unsafe { ptr::read(self) }),
            Err(_) => Err(TryCloneError::Other("Arc weak refcount exceeded")),
        }
    }
}

/// Returned when a fallible upgrade of [`Weak`] to [`Arc`] fails due to refcount overflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TryUpgradeError;

impl fmt::Display for TryUpgradeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "upgrade failed: strong refcount exceeded")
    }
}

impl crate::try_fmt::TryDebug for TryUpgradeError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TryUpgradeError")
    }
}

/// Fallible operations on [`Weak`] pointers.
///
/// Implemented for `Weak<T>`. Provides [`try_upgrade`](Self::try_upgrade), which
/// upgrades a weak reference back into a strong [`Arc`] without panicking on
/// refcount overflow (std's [`Weak::upgrade`] asserts in that scenario).
pub trait TryWeak<T: ?Sized> {
    /// Attempts to upgrade this `Weak` pointer to an [`Arc`].
    ///
    /// Returns:
    /// - `Some(Ok(arc))` on successful upgrade.
    /// - `None` if the strong count is zero (data dropped) or this `Weak` is dangling — matching [`Weak::upgrade`].
    /// - `Some(Err(TryUpgradeError))` if the strong refcount is at or above the maximum.
    ///
    /// Uses `(Acquire, Relaxed)` ordering to synchronise with [`Arc::new_cyclic`] initialisation.
    fn try_upgrade(&self) -> Option<Result<Arc<T>, TryUpgradeError>>;

    // ── Aliases with `fallible_` prefix ─────────────────────────────────────

    /// Alias for [`Self::try_upgrade`].
    fn fallible_upgrade(&self) -> Option<Result<Arc<T>, TryUpgradeError>> {
        Self::try_upgrade(self)
    }
}

impl<T: ?Sized> TryWeak<T> for Weak<T> {
    fn try_upgrade(&self) -> Option<Result<Arc<T>, TryUpgradeError>> {
        // Dangling sentinel — no allocation exists. Matches Weak::upgrade returning None.
        if weak_is_dangling(self) {
            return None;
        }

        let inner = weak_inner(self);

        // Acquire on success synchronises with Arc::new_cyclic's Release store
        // so we observe the fully initialised value. Relaxed on failure since
        // we have no expectations about the new state.
        let ok = inner
            .strong
            .try_update(Ordering::Acquire, Ordering::Relaxed, |cur| {
                if cur == 0 {
                    // Data has been dropped; don't increment.
                    return None;
                } else if cur >= MAX_REFCOUNT {
                    // Would overflow on next increment.
                    return None;
                }
                Some(cur + 1)
            });

        match ok {
            Ok(_) => {
                // Strong count was successfully incremented from a non-zero
                // value below MAX_REFCOUNT. Allocation is still alive.
                // SAFETY: pointer is valid, strong count > 0 after increment.
                Some(Ok(unsafe {
                    ptr::read(&*(self as *const Weak<T> as *const Arc<T>))
                }))
            }
            Err(prev) => {
                if prev == 0 {
                    // Data dropped — matches Weak::upgrade returning None.
                    None
                } else {
                    // At max refcount — our exclusive failure mode.
                    Some(Err(TryUpgradeError))
                }
            }
        }
    }
}

#[allow(deprecated)]
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

// ── TryDebug for Arc<T> ──────────────────────────────────────────────────────

impl<T: ?Sized + crate::try_fmt::TryDebug> crate::try_fmt::TryDebug for Arc<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).try_fmt(f)
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::lang_alloc::string::String;
    use crate::lang_alloc::string::ToString;
    use crate::lang_alloc::vec;
    use crate::lang_alloc::vec::Vec;
    use crate::lang_std::format;

    type PinArcResult<T> = Result<Pin<Arc<T>>, (T, AllocError)>;

    #[test]
    fn try_new_basic() {
        let arc = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
        assert_eq!(*arc, 42);
    }

    #[test]
    fn try_new_zst() {
        let arc = <Arc<()> as TryArc<()>>::try_new(()).unwrap();
        assert_eq!(*arc, ());
    }

    #[test]
    fn clone_increments_strong() {
        let arc = <Arc<String> as TryArc<String>>::try_new("hello".to_string()).unwrap();
        assert_eq!(Arc::strong_count(&arc), 1);
        let arc2 = arc.clone();
        assert_eq!(Arc::strong_count(&arc), 2);
        assert_eq!(Arc::strong_count(&arc2), 2);
        drop(arc2);
        assert_eq!(Arc::strong_count(&arc), 1);
    }

    #[test]
    fn strong_and_weak_counts() {
        let arc = <Arc<i32> as TryArc<i32>>::try_new(99).unwrap();
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
        let arc = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
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
        let arc = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
        assert_eq!(Arc::try_unwrap(arc), Ok(42));
    }

    #[test]
    fn try_unwrap_not_unique() {
        let arc = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
        let _arc2 = arc.clone();
        assert_eq!(*Arc::try_unwrap(arc).unwrap_err(), 42);
    }

    #[test]
    fn into_inner_race_simulation() {
        let arc = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
        let arc2 = arc.clone();
        let r1 = Arc::into_inner(arc);
        let r2 = Arc::into_inner(arc2);
        assert!(r1.is_some() ^ r2.is_some());
        assert!(r1 == Some(42) || r2 == Some(42));
    }

    #[test]
    fn ptr_eq_same_allocation() {
        let arc = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
        let arc2 = arc.clone();
        assert!(Arc::ptr_eq(&arc, &arc2));
        let arc3 = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
        assert!(!Arc::ptr_eq(&arc, &arc3));
    }

    #[test]
    fn into_raw_from_raw() {
        let arc = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
        let ptr = Arc::into_raw(arc);
        assert_eq!(unsafe { *ptr }, 42);
        let arc = unsafe { Arc::from_raw(ptr) };
        assert_eq!(*arc, 42);
    }

    #[test]
    fn get_mut_unique() {
        let mut arc = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
        let val = Arc::get_mut(&mut arc);
        assert!(val.is_some());
        *val.unwrap() = 99;
        assert_eq!(*arc, 99);
    }

    #[test]
    fn get_mut_not_unique() {
        let mut arc = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
        let _arc2 = arc.clone();
        assert!(Arc::get_mut(&mut arc).is_none());
    }

    #[test]
    fn debug_output() {
        let arc = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
        let s = format!("{:?}", arc);
        assert_eq!(s, "42");
    }

    #[test]
    fn display_output() {
        let arc = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
        let s = format!("{}", arc);
        assert_eq!(s, "42");
    }

    #[test]
    fn partial_eq() {
        let a = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
        let b = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
        let c = <Arc<i32> as TryArc<i32>>::try_new(43).unwrap();
        assert!(a == b);
        assert!(a != c);
    }

    // ── try_new_uninit tests ────────────────────────────────────────────────

    #[test]
    fn try_new_uninit_then_write() {
        let mut uninit: Arc<MaybeUninit<i32>> =
            <Arc<i32> as TryArc<i32>>::try_new_uninit().unwrap();
        // Write value in place — we have sole ownership so get_mut succeeds.
        Arc::get_mut(&mut uninit).unwrap().write(77);
        // Extract the MaybeUninit, assume init, and wrap in a fresh Arc.
        let owned = Arc::try_unwrap(uninit).unwrap();
        let val = Arc::new(unsafe { owned.assume_init() });
        assert_eq!(*val, 77);
    }

    #[test]
    fn try_new_uninit_zst() {
        let _uninit: Arc<MaybeUninit<()>> = <Arc<()> as TryArc<()>>::try_new_uninit().unwrap();
    }

    // ── try_new_zeroed tests ────────────────────────────────────────────────

    #[test]
    fn try_new_zeroed_returns_zeros() {
        let uninit: Arc<MaybeUninit<i32>> = <Arc<i32> as TryArc<i32>>::try_new_zeroed().unwrap();
        let owned = Arc::try_unwrap(uninit).unwrap();
        let val = unsafe { owned.assume_init() };
        assert_eq!(val, 0);
    }

    #[test]
    fn try_new_zeroed_f64_is_positive_zero() {
        let uninit: Arc<MaybeUninit<f64>> = <Arc<f64> as TryArc<f64>>::try_new_zeroed().unwrap();
        let owned = Arc::try_unwrap(uninit).unwrap();
        let val = unsafe { owned.assume_init() };
        assert_eq!(val.to_bits(), 0);
    }

    #[test]
    fn try_new_zeroed_array() {
        let uninit: Arc<MaybeUninit<[u8; 4]>> =
            <Arc<[u8; 4]> as TryArc<[u8; 4]>>::try_new_zeroed().unwrap();
        let owned = Arc::try_unwrap(uninit).unwrap();
        let arr = unsafe { owned.assume_init() };
        assert_eq!(arr, [0, 0, 0, 0]);
    }

    #[test]
    fn try_new_zeroed_zst() {
        let _uninit: Arc<MaybeUninit<()>> = <Arc<()> as TryArc<()>>::try_new_zeroed().unwrap();
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
        let arc = <Arc<[u8; 4]> as TryArc<[u8; 4]>>::try_new(*b"hi!!").unwrap();
        let val = Arc::<[u8; 4]>::unwrap_or_try_clone(arc).unwrap();
        assert_eq!(val, *b"hi!!");
    }

    #[test]
    fn unwrap_or_try_clone_shared_clones() {
        let arc = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
        let _arc2 = arc.clone();
        let val = Arc::<i32>::unwrap_or_try_clone(arc).unwrap();
        assert_eq!(val, 42);
    }

    #[test]
    fn unwrap_or_try_clone_with_option() {
        let arc = <Arc<Option<i32>> as TryArc<Option<i32>>>::try_new(Some(99)).unwrap();
        let _arc2 = arc.clone();
        let val = Arc::<Option<i32>>::unwrap_or_try_clone(arc).unwrap();
        assert_eq!(val, Some(99));
    }

    // ── TryClone tests ────────────────────────────────────────────────────────

    #[test]
    fn try_clone_increments_strong() {
        use crate::try_clone::TryClone;
        let arc = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
        assert_eq!(Arc::strong_count(&arc), 1);
        let arc2 = arc.try_clone().unwrap();
        assert_eq!(Arc::strong_count(&arc), 2);
        assert_eq!(Arc::strong_count(&arc2), 2);
        assert!(Arc::ptr_eq(&arc, &arc2));
    }

    #[test]
    fn try_clone_preserves_value() {
        use crate::try_clone::TryClone;
        let arc = <Arc<String> as TryArc<String>>::try_new(String::from("hello")).unwrap();
        let arc2 = arc.try_clone().unwrap();
        assert_eq!(arc2.as_str(), "hello");
    }

    #[test]
    fn try_clone_multiple() {
        use crate::try_clone::TryClone;
        let arc = <Arc<u64> as TryArc<u64>>::try_new(999).unwrap();
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
        let arc = <Arc<i32> as TryArc<i32>>::try_new(0).unwrap();
        let inner = arc_inner(&arc);
        inner.strong.store(MAX_REFCOUNT, Ordering::Relaxed);
        assert!(arc.try_clone().is_err());
        // Restore so the Arc drops cleanly.
        inner.strong.store(1, Ordering::Relaxed);
    }

    #[test]
    fn try_clone_rejects_above_max_refcount() {
        use crate::try_clone::TryClone;
        let arc = <Arc<i32> as TryArc<i32>>::try_new(0).unwrap();
        let inner = arc_inner(&arc);
        inner.strong.store(MAX_REFCOUNT + 1, Ordering::Relaxed);
        assert!(arc.try_clone().is_err());
        inner.strong.store(1, Ordering::Relaxed);
    }

    // ── Weak TryClone tests ───────────────────────────────────────────────────

    #[test]
    fn weak_try_clone_increments_weak_count() {
        use crate::try_clone::TryClone;
        let arc = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
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
        let arc = <Arc<u64> as TryArc<u64>>::try_new(0).unwrap();
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
        let arc = <Arc<i32> as TryArc<i32>>::try_new(0).unwrap();
        let weak = Arc::downgrade(&arc);
        let inner = weak_inner(&weak);
        // Overwrite after downgrade so we don't trigger std's own overflow check.
        inner.weak.store(MAX_REFCOUNT, Ordering::Relaxed);
        assert!(weak.try_clone().is_err());
        // The original refcount includes 1 for all strong refcounts and 1 for one weak refcount.
        inner.weak.store(2, Ordering::Relaxed);
        drop(weak);
        drop(arc);
    }

    #[test]
    fn weak_try_clone_rejects_above_max_refcount() {
        use crate::try_clone::TryClone;
        let arc = <Arc<i32> as TryArc<i32>>::try_new(0).unwrap();
        let weak = Arc::downgrade(&arc);
        let inner = weak_inner(&weak);
        // Overwrite after downgrade so we don't trigger std's own overflow check.
        inner.weak.store(MAX_REFCOUNT + 1, Ordering::Relaxed);
        assert!(weak.try_clone().is_err());
        // The original refcount includes 1 for all strong refcounts and 1 for one weak refcount.
        inner.weak.store(2, Ordering::Relaxed);
        drop(weak);
        drop(arc);
    }

    // ── try_upgrade tests ───────────────────────────────────────────────────────

    #[test]
    fn weak_try_upgrade_success() {
        use super::TryWeak;
        let arc = <Arc<i32> as TryArc<i32>>::try_new(42).unwrap();
        let weak = Arc::downgrade(&arc);
        assert_eq!(Arc::strong_count(&arc), 1);
        let arc2 = weak.try_upgrade().unwrap().unwrap();
        assert_eq!(*arc2, 42);
        assert_eq!(Arc::strong_count(&arc), 2);
        assert!(Arc::ptr_eq(&arc, &arc2));
    }

    #[test]
    fn weak_try_upgrade_dangling_returns_none() {
        let weak: Weak<i32> = Weak::new();
        assert!(weak.try_upgrade().is_none());
    }

    #[test]
    fn weak_try_upgrade_after_drop_returns_none() {
        let weak = {
            let arc = <Arc<i32> as TryArc<i32>>::try_new(99).unwrap();
            Arc::downgrade(&arc)
        };
        // arc has been dropped; strong count is zero.
        assert!(weak.try_upgrade().is_none());
    }

    #[test]
    fn weak_try_upgrade_overflow_rejects() {
        let arc = <Arc<i32> as TryArc<i32>>::try_new(0).unwrap();
        let weak = Arc::downgrade(&arc);
        let inner = weak_inner(&weak);
        inner.strong.store(MAX_REFCOUNT, Ordering::Relaxed);
        let result = weak.try_upgrade();
        assert!(matches!(result, Some(Err(TryUpgradeError))));
        // Restore so arc drops cleanly.
        inner.strong.store(1, Ordering::Relaxed);
    }

    #[test]
    fn weak_try_upgrade_multiple_roundtrip() {
        use crate::std::arc::TryWeak;
        let arc = <Arc<String> as TryArc<String>>::try_new("hello".into()).unwrap();
        let weak = Arc::downgrade(&arc);
        for _ in 0..50 {
            let upgraded = weak.try_upgrade().unwrap().unwrap();
            assert_eq!(upgraded.as_str(), "hello");
            assert_eq!(Arc::strong_count(&arc), 2);
            drop(upgraded);
            assert_eq!(Arc::strong_count(&arc), 1);
        }
    }

    #[test]
    fn weak_try_upgrade_dyn_trait() {
        let arc: Arc<dyn ::lang_std::fmt::Debug> = Arc::new(42i32);
        let weak = Arc::downgrade(&arc);
        let upgraded = weak.try_upgrade().unwrap().unwrap();
        assert_eq!(Arc::strong_count(&arc), 2);
        assert!(Arc::ptr_eq(&arc, &upgraded));
    }

    #[test]
    fn weak_try_upgrade_error_display() {
        assert!(format!("{}", TryUpgradeError).contains("exceeded"));
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
        impl ::lang_std::fmt::Display for Greeter {
            fn fmt(&self, f: &mut ::lang_std::fmt::Formatter<'_>) -> ::lang_std::fmt::Result {
                write!(f, "Hello, {}!", self.0)
            }
        }
        let arc: Arc<dyn ::lang_std::fmt::Display> = Arc::new(Greeter("world".into()));
        let arc2 = arc.try_clone().unwrap();
        assert_eq!(Arc::strong_count(&arc), 2);
        assert!(Arc::ptr_eq(&arc, &arc2));
    }

    #[test]
    fn arc_dyn_trait_try_clone_multiple() {
        use crate::try_clone::TryClone;
        let arc: Arc<dyn ::lang_std::fmt::Debug> = Arc::new(42i32);
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
        let arc: Arc<dyn ::lang_std::fmt::Display> = Arc::new(99i64);
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
        let _pinned: Pin<Arc<i32>> = <Arc<i32> as TryArc<i32>>::try_pin(42).unwrap();
    }

    #[test]
    fn arc_try_pin_value_accessible() {
        let pinned: Pin<Arc<u64>> = <Arc<u64> as TryArc<u64>>::try_pin(999).unwrap();
        let val: &u64 = &pinned;
        assert_eq!(*val, 999);
    }

    #[test]
    fn arc_try_pin_zst() {
        let _pinned: Pin<Arc<()>> = <Arc<()> as TryArc<()>>::try_pin(()).unwrap();
    }

    // ── fallible_ alias tests for pin ─────────────────────────────────────────

    #[test]
    fn fallible_pin_works() {
        let _pinned: Pin<Arc<i32>> = Arc::<i32>::fallible_pin(42).unwrap();
    }

    // ── try_pin_give_back tests ─────────────────────────────────────────────

    #[test]
    fn arc_try_pin_give_back_success() {
        let pinned: Pin<Arc<String>> =
            <Arc<String> as TryArc<String>>::try_pin_give_back(String::from("hello")).unwrap();
        assert_eq!(pinned.as_str(), "hello");
    }

    #[test]
    fn arc_try_pin_give_back_signature() {
        let val = vec![1, 2, 3];
        let result: PinArcResult<Vec<i32>> =
            <Arc<Vec<i32>> as TryArc<Vec<i32>>>::try_pin_give_back(val);
        let _pinned = result.unwrap();
    }

    #[test]
    fn arc_try_pin_give_back_zst() {
        let _pinned: Pin<Arc<()>> = Arc::<()>::try_pin_give_back(()).unwrap();
    }

    #[test]
    fn fallible_pin_give_back_works() {
        let _pinned: Pin<Arc<i32>> = Arc::<i32>::fallible_pin_give_back(42).unwrap();
    }

    // ── DST with custom alignment tests ────────────────────────────────────────

    #[test]
    fn arc_try_clone_align_512() {
        use crate::try_clone::TryClone;
        struct AlignedData([u8; 64]);
        unsafe impl Send for AlignedData {}
        unsafe impl Sync for AlignedData {}

        trait Process: Send + Sync {
            fn process(&self) -> usize;
        }

        impl Process for AlignedData {
            fn process(&self) -> usize {
                self.0.len()
            }
        }

        // Create an Arc<dyn Process> backed by a type that needs high alignment.
        // The old offset-based approach would miscalculate header offsets because
        // padding between [strong][weak] and data differs when align(T) > 2.
        let inner: AlignedData = AlignedData([42u8; 64]);
        let arc: Arc<dyn Process> = Arc::new(inner);
        assert_eq!(arc.process(), 64);

        let arc2 = arc.try_clone().unwrap();
        assert_eq!(Arc::strong_count(&arc), 2);
        assert!(Arc::ptr_eq(&arc, &arc2));
        assert_eq!(arc2.process(), 64);
    }

    #[test]
    fn weak_try_clone_align_512() {
        use crate::try_clone::TryClone;
        struct AlignedData([u8; 64]);
        unsafe impl Send for AlignedData {}
        unsafe impl Sync for AlignedData {}

        trait Process: Send + Sync {
            fn process(&self) -> usize;
        }

        impl Process for AlignedData {
            fn process(&self) -> usize {
                self.0.len()
            }
        }

        let inner: AlignedData = AlignedData([99u8; 64]);
        let arc: Arc<dyn Process> = Arc::new(inner);
        let weak = Arc::downgrade(&arc);
        assert_eq!(Arc::weak_count(&arc), 1);

        let weak2 = weak.try_clone().unwrap();
        assert_eq!(Arc::weak_count(&arc), 2);

        // Verify we can still upgrade.
        let upgraded = weak2.upgrade().unwrap();
        assert_eq!(upgraded.process(), 64);
    }

    #[test]
    fn arc_slice_try_clone_large_alignment() {
        use crate::try_clone::TryClone;
        // Slices don't have custom alignment but exercise the unsized path.
        let data: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let arc: Arc<[u64]> = Arc::from(&data[..]);
        let arc2 = arc.try_clone().unwrap();
        assert_eq!(arc2.as_ref(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(Arc::strong_count(&arc), 2);

        let weak = Arc::downgrade(&arc);
        let weak2 = weak.try_clone().unwrap();
        assert_eq!(Arc::weak_count(&arc), 2);
        drop(weak2);
        assert_eq!(Arc::weak_count(&arc), 1);
    }

    // ── OOM tests ─────────────────────────────────────────────────────────────
    use rustyfill_test_allocator::{FailPolicy, with_policy};

    #[test]
    fn arc_fallible_new_fails_on_oom() {
        let r: Result<Arc<i32>, AllocError> = with_policy(FailPolicy::fail_next_alloc(), || {
            <Arc<i32> as TryArc<i32>>::fallible_new(42)
        });
        assert!(r.is_err());
    }

    #[test]
    fn arc_fallible_new_give_back_returns_value_on_oom() {
        let r: Result<Arc<i32>, (i32, AllocError)> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <Arc<i32> as TryArc<i32>>::try_new_give_back(99)
            });
        assert!(r.is_err());
        if let Err((returned, _err)) = r {
            assert_eq!(returned, 99);
        }
    }

    #[test]
    fn arc_fallible_new_uninit_fails_on_oom() {
        let r: Result<Arc<::lang_std::mem::MaybeUninit<i32>>, AllocError> = with_policy(
            FailPolicy::fail_next_alloc(),
            <Arc<i32> as TryArc<i32>>::fallible_new_uninit,
        );
        assert!(r.is_err());
    }

    #[test]
    fn arc_fallible_new_zeroed_fails_on_oom() {
        let r: Result<Arc<::lang_std::mem::MaybeUninit<[u8; 16]>>, AllocError> = with_policy(
            FailPolicy::fail_next_alloc(),
            <Arc<[u8; 16]> as TryArc<[u8; 16]>>::fallible_new_zeroed,
        );
        assert!(r.is_err());
    }

    #[test]
    fn arc_fallible_pin_fails_on_oom() {
        let r: Result<Pin<Arc<i32>>, AllocError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <Arc<i32> as TryArc<i32>>::fallible_pin(42)
            });
        assert!(r.is_err());
    }

    #[test]
    fn arc_fallible_pin_give_back_returns_value_on_oom() {
        let r: Result<Pin<Arc<i64>>, (i64, AllocError)> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <Arc<i64> as TryArc<i64>>::try_pin_give_back(99)
            });
        assert!(r.is_err());
        if let Err((returned, _err)) = r {
            assert_eq!(returned, 99);
        }
    }

    #[test]
    fn arc_try_default_fails_on_oom() {
        let r: Result<Arc<i32>, TryDefaultError> = with_policy(
            FailPolicy::fail_next_alloc(),
            <Arc<i32> as TryDefault>::try_default,
        );
        assert!(r.is_err());
    }

    #[test]
    fn arc_try_clone_succeeds_under_oom() {
        // Arc::try_clone only increments atomic refcounts, no heap allocation.
        let arc = Arc::<i32>::new(42);
        let r: Result<Arc<i32>, TryCloneError> =
            with_policy(FailPolicy::fail_next_alloc(), || arc.try_clone());
        assert!(r.is_ok());
    }

    #[test]
    fn arc_nth_alloc_fail_targets_correct_call() {
        let (r1_ok, r2_err, r3_ok) = with_policy(FailPolicy::fail_nth_alloc(2), || {
            let r1: Result<Arc<u8>, AllocError> = <Arc<u8> as TryArc<u8>>::fallible_new(1);
            let r2: Result<Arc<u8>, AllocError> = <Arc<u8> as TryArc<u8>>::fallible_new(2);
            let r3: Result<Arc<u8>, AllocError> = <Arc<u8> as TryArc<u8>>::fallible_new(3);
            (r1.is_ok(), r2.is_err(), r3.is_ok())
        });
        assert!(r1_ok, "first alloc should succeed");
        assert!(r2_err, "second alloc should fail");
        assert!(r3_ok, "third alloc should succeed");
    }

    #[test]
    fn arc_oom_restores_allocation_afterwards() {
        let r: Result<Arc<i32>, AllocError> = with_policy(FailPolicy::fail_next_alloc(), || {
            <Arc<i32> as TryArc<i32>>::fallible_new(42)
        });
        assert!(r.is_err());
        let r: Result<Arc<i32>, AllocError> = <Arc<i32> as TryArc<i32>>::fallible_new(42);
        assert!(r.is_ok());
    }

    #[test]
    fn weak_try_upgrade_dangling_no_alloc_needed() {
        // Weak::try_upgrade on a dangling pointer doesn't allocate.
        let weak: Weak<i32> = Weak::new();
        let result = with_policy(FailPolicy::fail_next_alloc(), || weak.try_upgrade());
        assert!(result.is_none());
    }
}
