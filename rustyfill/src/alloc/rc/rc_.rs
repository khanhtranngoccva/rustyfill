use crate::alloc::AllocError;
use crate::alloc::boxed::TryBox;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use lang_alloc::boxed::Box;
use lang_alloc::rc::{Rc, Weak};
use lang_core::cell::Cell;
use lang_core::fmt;
use lang_core::mem::{ManuallyDrop, MaybeUninit};
use lang_core::pin::Pin;
use lang_core::ptr;

/// Internal representation of an Rc allocation.
///
/// Layout matches `::std::rc::Rc`: two usize counters followed by the data.
/// `#[repr(C)]` is required so the compiler does not reorder fields — std's Rc
/// computes counter offsets relative to the data pointer and expects this exact
/// ordering. `align(2)` ensures that `Weak::new()`'s dangling sentinel
/// (`usize::MAX`) can never be a valid payload address, since all real allocations
/// are aligned to at least 2.
#[repr(C, align(2))]
struct RcInner<T: ?Sized> {
    strong: Cell<usize>,
    weak: Cell<usize>,
    data: T,
}

/// A trait for fallibly constructing an [`Rc`].
///
/// Implemented for `Rc<T>`. Mirrors the [`TryBox`](crate::alloc::boxed::TryBox) pattern:
/// only the allocating constructors are fallible; all other Rc behaviour
/// (cloning, downgrading, dropping) delegates to the standard library.
///
/// # Construction strategy
///
/// Allocation is delegated to [`TryBox::try_new_uninit`] via a boxed
/// `MaybeUninit<RcInner<T>>`. After initialising the strong/weak counters
/// and the data in place, ownership transfers to std's `Rc` through
/// [`Rc::from_raw`] — no second allocation is performed.
pub trait TryRc<T>: Sized {
    /// The uninitialized variant of this rc.
    type Uninit: Sized;

    /// Fallibly allocate a new `Rc<T>`.
    ///
    /// **Deprecated:** This method name conflicts with the unstable inherent
    /// [`Rc::try_new`]. Use [`Self::fallible_new`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with unstable Rc::try_new; use fallible_new"
    )]
    fn try_new(value: T) -> Result<Self, AllocError>;

    /// Fallibly allocate an uninitialised `Rc<MaybeUninit<T>>`.
    ///
    /// **Deprecated:** This method name conflicts with the unstable inherent
    /// [`Rc::try_new_uninit`]. Use [`Self::fallible_new_uninit`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with unstable Rc::try_new_uninit; use fallible_new_uninit"
    )]
    fn try_new_uninit() -> Result<Self::Uninit, AllocError>;

    /// Fallibly allocate zero-initialised memory as an `Rc<MaybeUninit<T>>`.
    ///
    /// **Deprecated:** This method name conflicts with the unstable inherent
    /// [`Rc::try_new_zeroed`]. Use [`Self::fallible_new_zeroed`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with unstable Rc::try_new_zeroed; use fallible_new_zeroed"
    )]
    fn try_new_zeroed() -> Result<Self::Uninit, AllocError>;

    /// Like [`Self::try_new`] but returns ownership of `value` back on failure.
    ///
    /// On success, returns the newly allocated `Rc<T>`. On allocation failure,
    /// returns the original `value` alongside the [`AllocError`] so the caller
    /// can reuse or drop it cleanly rather than losing it to an OOM panic.
    fn try_new_give_back(value: T) -> Result<Self, (T, AllocError)>;

    /// Unwraps the value if this is the only strong reference, otherwise fallibly
    /// clones the inner data.
    ///
    /// This is a panic-free analogue of [`Rc::unwrap_or_clone`]. When there are
    /// other strong references, the inner value is cloned via [`TryClone`] rather
    /// than [`Clone`], so allocation failures during cloning (e.g. cloning a
    /// [`lang_alloc::string::String`]) return an error instead of panicking.
    ///
    /// On failure, returns the original `Rc` alongside the clone error so the
    /// caller retains access to the shared data.
    fn unwrap_or_try_clone(self) -> Result<T, (Self, TryCloneError)>
    where
        T: crate::try_clone::TryClone;

    /// Fallibly allocate `value` on the heap and pin it in place.
    ///
    /// **Deprecated:** This method name conflicts with the unstable inherent
    /// [`Rc::try_pin`]. Use [`Self::fallible_pin`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "conflicts with unstable Rc::try_pin; use fallible_pin"
    )]
    fn try_pin(value: T) -> Result<Pin<Self>, AllocError>;

    /// Like [`Self::try_pin`] but returns ownership of `value` back on failure.
    ///
    /// On success, returns the pinned `Rc<T>`. On allocation failure, returns
    /// the original `value` alongside the [`AllocError`] so the caller retains
    /// access to the shared data.
    fn try_pin_give_back(value: T) -> Result<Pin<Self>, (T, AllocError)>;

    // ── Aliases with `fallible_` prefix to avoid name collisions ────────────

    /// Fallibly allocate a new `Rc<T>`.
    ///
    /// Returns [`AllocError`] if the heap allocation fails. Unlike
    /// [`Rc::new`], this never panics on out-of-memory.
    ///
    /// Allocation is performed by first allocating a boxed
    /// `MaybeUninit<RcInner<T>>` via [`TryBox::fallible_new_uninit`], then
    /// transferring ownership to std's `Rc` through [`Rc::from_raw`] — no
    /// second allocation is performed.
    ///
    /// This method replaces the deprecated [`Self::try_new`] which shares its
    /// name with the unstable inherent [`Rc::try_new`].
    #[allow(deprecated)]
    fn fallible_new(value: T) -> Result<Self, AllocError> {
        Self::try_new(value)
    }

    /// Fallibly allocate an uninitialised `Rc<MaybeUninit<T>>`.
    ///
    /// Returns an `Rc` wrapping `MaybeUninit<T>` that can be initialised
    /// in place via [`MaybeUninit::write`] and converted to an `Rc<T>` using
    /// [`Rc::into_inner`] + [`MaybeUninit::assume_init`].
    ///
    /// This method replaces the deprecated [`Self::try_new_uninit`] which shares
    /// its name with the unstable inherent [`Rc::try_new_uninit`].
    #[allow(deprecated)]
    fn fallible_new_uninit() -> Result<Self::Uninit, AllocError> {
        Self::try_new_uninit()
    }

    /// Fallibly allocate zero-initialised memory as an `Rc<MaybeUninit<T>>`.
    ///
    /// Returns an `Rc` wrapping `MaybeUninit<T>` whose underlying bytes are
    /// all set to zero. Safe to call [`MaybeUninit::assume_init`] on types
    /// whose all-zeros bitpattern is valid (e.g. numeric primitives, `bool`,
    /// `[T; N]` where `T` is also zeroable).
    ///
    /// This method replaces the deprecated [`Self::try_new_zeroed`] which shares
    /// its name with the unstable inherent [`Rc::try_new_zeroed`].
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
    /// This is a panic-free analogue of [`Rc::unwrap_or_clone`] using
    /// [`TryClone`] instead of [`Clone`]. On failure, returns the original `Rc`
    /// alongside the clone error so the caller retains access to the shared data.
    ///
    /// This method mirrors [`Self::unwrap_or_try_clone`].
    fn unwrap_or_fallible_clone(self) -> Result<T, (Self, TryCloneError)>
    where
        T: crate::try_clone::TryClone,
    {
        Self::unwrap_or_try_clone(self)
    }

    /// Fallibly allocate `value` on the heap and pin it in place.
    ///
    /// Returns a [`Pin<Rc<T>>`] so that if `T` does not implement [`Unpin`],
    /// the value is immovable after allocation. This is the fallible analogue
    /// of [`Rc::pin`].
    ///
    /// This method replaces the deprecated [`Self::try_pin`] which shares its
    /// name with the unstable inherent [`Rc::try_pin`].
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
impl<T> TryRc<T> for Rc<T> {
    type Uninit = Rc<MaybeUninit<T>>;

    fn try_new(value: T) -> Result<Self, AllocError> {
        let mut slot =
            ManuallyDrop::new(<Box<RcInner<T>> as TryBox<RcInner<T>>>::try_new_uninit()?);
        let inner = slot.as_mut_ptr();
        // The pointer is always valid, cannot panic.
        unsafe {
            ptr::write(&mut (*inner).strong, Cell::new(1));
            ptr::write(&mut (*inner).weak, Cell::new(1));
            ptr::write(&mut (*inner).data, value);
        }
        let data_ptr = unsafe { &raw const (*inner).data };
        Ok(unsafe { Rc::from_raw(data_ptr) })
    }

    fn try_new_uninit() -> Result<Self::Uninit, AllocError> {
        let mut slot =
            ManuallyDrop::new(<Box<RcInner<T>> as TryBox<RcInner<T>>>::try_new_uninit()?);
        let inner = slot.as_mut_ptr();
        // The pointer is always valid, cannot panic.
        unsafe {
            ptr::write(&mut (*inner).strong, Cell::new(1));
            ptr::write(&mut (*inner).weak, Cell::new(1));
            // data field stays uninitialised
        }
        let data_ptr = unsafe { &raw const (*inner).data as *const MaybeUninit<T> };
        Ok(unsafe { Rc::from_raw(data_ptr) })
    }

    fn try_new_zeroed() -> Result<Self::Uninit, AllocError> {
        let mut slot =
            ManuallyDrop::new(<Box<RcInner<T>> as TryBox<RcInner<T>>>::try_new_zeroed()?);
        let inner = slot.as_mut_ptr();
        // The entire allocation is zeroed — including the data region.
        // Fix up the refcount headers; leave data as zeroes.
        // The pointer is always valid, cannot panic.
        unsafe {
            ptr::write(&mut (*inner).strong, Cell::new(1));
            ptr::write(&mut (*inner).weak, Cell::new(1));
        }
        let data_ptr = unsafe { &raw const (*inner).data as *const MaybeUninit<T> };
        Ok(unsafe { Rc::from_raw(data_ptr) })
    }

    fn try_new_give_back(value: T) -> Result<Self, (T, AllocError)> {
        match <Box<RcInner<T>> as TryBox<RcInner<T>>>::try_new_uninit() {
            Ok(raw_slot) => {
                let mut slot = ManuallyDrop::new(raw_slot);
                let inner = slot.as_mut_ptr();
                // The pointer is always valid, cannot panic.
                unsafe {
                    ptr::write(&mut (*inner).strong, Cell::new(1));
                    ptr::write(&mut (*inner).weak, Cell::new(1));
                    ptr::write(&mut (*inner).data, value);
                }
                let data_ptr = unsafe { &raw const (*inner).data };
                Ok(unsafe { Rc::from_raw(data_ptr) })
            }
            Err(e) => Err((value, e)),
        }
    }

    fn unwrap_or_try_clone(self) -> Result<T, (Self, TryCloneError)>
    where
        T: crate::try_clone::TryClone,
    {
        match Rc::try_unwrap(self) {
            Ok(val) => Ok(val),
            Err(rc) => match (*rc).try_clone() {
                Ok(cloned) => Ok(cloned),
                Err(e) => Err((rc, e)),
            },
        }
    }

    fn try_pin(value: T) -> Result<Pin<Self>, AllocError> {
        let rc = <Self as TryRc<T>>::try_new(value)?;
        Ok(unsafe { Pin::new_unchecked(rc) })
    }

    fn try_pin_give_back(value: T) -> Result<Pin<Self>, (T, AllocError)> {
        match Self::try_new_give_back(value) {
            Ok(rc) => Ok(unsafe { Pin::new_unchecked(rc) }),
            Err((v, e)) => Err((v, e)),
        }
    }
}

/// Maximum reference count, matching std's `Rc::MAX_REFCOUNT`.
/// Beyond this threshold, clones are rejected to avoid counter overflow.
const MAX_REFCOUNT: usize = (isize::MAX) as usize;

/// Helper: obtain a reference to the `RcInner<T>` stored inside an `Rc<T>`.
///
/// `Rc<T>`'s first field is `ptr: NonNull<RcInner<T>>`. By casting `&Rc<T>`
/// to a raw pointer and reinterpreting it as `*const NonNull<RcInner<T>>`, we
/// read out the inner pointer and dereference it into a `&RcInner<T>`. This
/// avoids any byte-offset arithmetic and works correctly for all types including
/// DSTs with custom alignment.
fn rc_inner<T: ?Sized>(rc: &Rc<T>) -> &RcInner<T> {
    unsafe {
        let ptr_field: *const ptr::NonNull<RcInner<T>> = ptr::from_ref(rc).cast();
        let non_null: ptr::NonNull<RcInner<T>> = *ptr_field;
        non_null.as_ref()
    }
}

/// Helper: obtain a reference to the `RcInner<T>` stored inside a `Weak<T>`.
///
/// Same approach as [`rc_inner`] but for `Weak`. `Weak<T>` also stores
/// `ptr: NonNull<RcInner<T>>` as its first field. For the dangling sentinel
/// (`Weak::new()`), the pointer address is `usize::MAX` and must be checked
/// by the caller before invoking this function.
fn weak_inner<T: ?Sized>(weak: &Weak<T>) -> &RcInner<T> {
    // SAFETY: Identical reasoning to rc_inner. Weak's first field is
    // `ptr: NonNull<RcInner<T>>`. Caller must ensure this isn't a dangling
    // Weak (address != usize::MAX) before calling.
    unsafe {
        let ptr_field: *const ptr::NonNull<RcInner<T>> = ptr::from_ref(weak).cast();
        let non_null: ptr::NonNull<RcInner<T>> = *ptr_field;
        non_null.as_ref()
    }
}

/// Returns true if this `Weak` is a dangling sentinel (created via `Weak::new()`).
fn weak_is_dangling<T: ?Sized>(weak: &Weak<T>) -> bool {
    weak.as_ptr().addr() == usize::MAX
}

impl<T: ?Sized> TryClone for Rc<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let inner = rc_inner(self);
        let strong = inner.strong.get();

        // Single-threaded: plain mutable access to the counter is fine as long
        // as no other thread touches it (guaranteed by Rc's !Sync bound).
        if strong >= MAX_REFCOUNT {
            return Err(TryCloneError::Other("Rc strong refcount exceeded"));
        }

        // We've confirmed strong < MAX_REFCOUNT <= isize::MAX above, so +1 cannot
        // overflow. In single-threaded context, no race.
        let strong = strong
            .checked_add(1)
            .expect("strong refcount below MAX_REFCOUNT");
        inner.strong.set(strong);

        Ok(unsafe { ptr::read(self) })
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
        let weak = inner.weak.get();

        if weak >= MAX_REFCOUNT {
            return Err(TryCloneError::Other("Rc weak refcount exceeded"));
        }

        // We've confirmed weak < MAX_REFCOUNT <= isize::MAX above, so +1 cannot
        // overflow. In single-threaded context, no race.
        let weak = weak
            .checked_add(1)
            .expect("weak refcount below MAX_REFCOUNT");
        inner.weak.set(weak);

        Ok(unsafe { ptr::read(self) })
    }
}

/// Returned when a fallible upgrade of [`Weak`] to [`Rc`] fails due to refcount overflow.
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
/// upgrades a weak reference back into a strong [`Rc`] without panicking on
/// refcount overflow (std's [`Weak::upgrade`] asserts in that scenario).
pub trait TryWeak<T: ?Sized> {
    /// Attempts to upgrade this `Weak` pointer to an [`Rc`].
    ///
    /// Returns:
    /// - `Some(Ok(rc))` on successful upgrade.
    /// - `None` if the strong count is zero (data dropped) or this `Weak` is dangling — matching [`Weak::upgrade`].
    /// - `Some(Err(TryUpgradeError))` if the strong refcount is at or above the maximum.
    fn try_upgrade(&self) -> Option<Result<Rc<T>, TryUpgradeError>>;

    // ── Aliases with `fallible_` prefix ─────────────────────────────────────

    /// Alias for [`Self::try_upgrade`].
    fn fallible_upgrade(&self) -> Option<Result<Rc<T>, TryUpgradeError>> {
        Self::try_upgrade(self)
    }
}

impl<T: ?Sized> TryWeak<T> for Weak<T> {
    fn try_upgrade(&self) -> Option<Result<Rc<T>, TryUpgradeError>> {
        // Dangling sentinel — no allocation exists. Matches Weak::upgrade returning None.
        if weak_is_dangling(self) {
            return None;
        }

        let inner = weak_inner(self);

        // Single-threaded: direct access to counters.
        let strong = inner.strong.get();
        if strong == 0 {
            // Data has been dropped; don't increment.
            return None;
        } else if strong >= MAX_REFCOUNT {
            // Would overflow on next increment.
            return Some(Err(TryUpgradeError));
        }

        // Increment strong count. We've confirmed strong < MAX_REFCOUNT <= isize::MAX
        // above, so +1 cannot overflow.
        let strong = strong
            .checked_add(1)
            .expect("strong refcount below MAX_REFCOUNT");
        inner.strong.set(strong);

        // Strong count was successfully incremented from a non-zero
        // value below MAX_REFCOUNT. Allocation is still alive.
        // SAFETY: pointer is valid, strong count > 0 after increment.
        Some(Ok(unsafe {
            ptr::read(&*(self as *const Weak<T> as *const Rc<T>))
        }))
    }
}

#[allow(deprecated)]
impl<T: TryDefault> TryDefault for Rc<T> {
    fn try_default() -> Result<Self, TryDefaultError> {
        // Allocate first so that if allocation fails we never touch T::try_default().
        let mut uninit = <Rc<T> as TryRc<T>>::try_new_uninit().map_err(TryDefaultError::Alloc)?;
        match T::try_default() {
            Ok(val) => {
                // We have sole ownership (freshly allocated, strong count == 1).
                Rc::get_mut(&mut uninit).unwrap().write(val);
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

// ── TryDebug for Rc<T> ──────────────────────────────────────────────────────

impl<T: ?Sized + crate::try_fmt::TryDebug> crate::try_fmt::TryDebug for Rc<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).try_fmt(f)
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use lang_alloc::fmt;
    use lang_alloc::format;
    use lang_alloc::string::String;
    use lang_alloc::string::ToString;
    use lang_alloc::vec;
    use lang_alloc::vec::Vec;

    type PinRcResult<T> = Result<Pin<Rc<T>>, (T, AllocError)>;

    #[test]
    fn try_new_basic() {
        let rc = <Rc<i32> as TryRc<i32>>::try_new(42).unwrap();
        assert_eq!(*rc, 42);
    }

    #[test]
    fn try_new_zst() {
        let rc = <Rc<()> as TryRc<()>>::try_new(()).unwrap();
        assert_eq!(*rc, ());
    }

    #[test]
    fn clone_increments_strong() {
        let rc = <Rc<String> as TryRc<String>>::try_new("hello".to_string()).unwrap();
        assert_eq!(Rc::strong_count(&rc), 1);
        let rc2 = rc.clone();
        assert_eq!(Rc::strong_count(&rc), 2);
        assert_eq!(Rc::strong_count(&rc2), 2);
        drop(rc2);
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[test]
    fn strong_and_weak_counts() {
        let rc = <Rc<i32> as TryRc<i32>>::try_new(99).unwrap();
        assert_eq!(Rc::strong_count(&rc), 1);
        assert_eq!(Rc::weak_count(&rc), 0);

        let weak = Rc::downgrade(&rc);
        assert_eq!(Rc::weak_count(&rc), 1);

        let _weak2 = weak.clone();
        assert_eq!(Rc::weak_count(&rc), 2);

        drop(weak);
        assert_eq!(Rc::weak_count(&rc), 1);
    }

    #[test]
    fn downgrade_upgrade() {
        let rc = <Rc<i32> as TryRc<i32>>::try_new(42).unwrap();
        let weak = Rc::downgrade(&rc);

        // Upgrade while rc exists
        {
            let upgraded = weak.upgrade().unwrap();
            assert_eq!(*upgraded, 42);
        }

        drop(rc);
        // After last strong ref is gone, upgrade fails
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn try_unwrap_unique() {
        let rc = <Rc<i32> as TryRc<i32>>::try_new(42).unwrap();
        assert_eq!(Rc::try_unwrap(rc), Ok(42));
    }

    #[test]
    fn try_unwrap_not_unique() {
        let rc = <Rc<i32> as TryRc<i32>>::try_new(42).unwrap();
        let _rc2 = rc.clone();
        assert_eq!(*Rc::try_unwrap(rc).unwrap_err(), 42);
    }

    #[test]
    fn into_inner_race_simulation() {
        let rc = <Rc<i32> as TryRc<i32>>::try_new(42).unwrap();
        let rc2 = rc.clone();
        let r1 = Rc::into_inner(rc);
        let r2 = Rc::into_inner(rc2);
        assert!(r1.is_some() ^ r2.is_some());
        assert!(r1 == Some(42) || r2 == Some(42));
    }

    #[test]
    fn ptr_eq_same_allocation() {
        let rc = <Rc<i32> as TryRc<i32>>::try_new(42).unwrap();
        let rc2 = rc.clone();
        assert!(Rc::ptr_eq(&rc, &rc2));
        let rc3 = <Rc<i32> as TryRc<i32>>::try_new(42).unwrap();
        assert!(!Rc::ptr_eq(&rc, &rc3));
    }

    #[test]
    fn into_raw_from_raw() {
        let rc = <Rc<i32> as TryRc<i32>>::try_new(42).unwrap();
        let ptr = Rc::into_raw(rc);
        assert_eq!(unsafe { *ptr }, 42);
        let rc = unsafe { Rc::from_raw(ptr) };
        assert_eq!(*rc, 42);
    }

    #[test]
    fn get_mut_unique() {
        let mut rc = <Rc<i32> as TryRc<i32>>::try_new(42).unwrap();
        let val = Rc::get_mut(&mut rc);
        assert!(val.is_some());
        *val.unwrap() = 99;
        assert_eq!(*rc, 99);
    }

    #[test]
    fn get_mut_not_unique() {
        let mut rc = <Rc<i32> as TryRc<i32>>::try_new(42).unwrap();
        let _rc2 = rc.clone();
        assert!(Rc::get_mut(&mut rc).is_none());
    }

    #[test]
    fn partial_eq() {
        let a = <Rc<i32> as TryRc<i32>>::try_new(42).unwrap();
        let b = <Rc<i32> as TryRc<i32>>::try_new(42).unwrap();
        let c = <Rc<i32> as TryRc<i32>>::try_new(43).unwrap();
        assert!(a == b);
        assert!(a != c);
    }

    // ── try_new_uninit tests ────────────────────────────────────────────────

    #[test]
    fn try_new_uninit_then_write() {
        let mut uninit: Rc<MaybeUninit<i32>> = <Rc<i32> as TryRc<i32>>::try_new_uninit().unwrap();
        Rc::get_mut(&mut uninit).unwrap().write(77);
        let owned = Rc::try_unwrap(uninit).unwrap();
        let val = Rc::new(unsafe { owned.assume_init() });
        assert_eq!(*val, 77);
    }

    #[test]
    fn try_new_uninit_zst() {
        let _uninit: Rc<MaybeUninit<()>> = <Rc<()> as TryRc<()>>::try_new_uninit().unwrap();
    }

    // ── try_new_zeroed tests ────────────────────────────────────────────────

    #[test]
    fn try_new_zeroed_returns_zeros() {
        let uninit: Rc<MaybeUninit<i32>> = <Rc<i32> as TryRc<i32>>::try_new_zeroed().unwrap();
        let owned = Rc::try_unwrap(uninit).unwrap();
        let val = unsafe { owned.assume_init() };
        assert_eq!(val, 0);
    }

    #[test]
    fn try_new_zeroed_array() {
        let uninit: Rc<MaybeUninit<[u8; 4]>> =
            <Rc<[u8; 4]> as TryRc<[u8; 4]>>::try_new_zeroed().unwrap();
        let owned = Rc::try_unwrap(uninit).unwrap();
        let arr = unsafe { owned.assume_init() };
        assert_eq!(arr, [0, 0, 0, 0]);
    }

    // ── try_new_give_back tests ─────────────────────────────────────────────

    #[test]
    fn try_new_give_back_success() {
        let rc = Rc::<String>::try_new_give_back("hello".to_string()).unwrap();
        assert_eq!(rc.as_str(), "hello");
    }

    #[test]
    fn try_new_give_back_signature() {
        let val = vec![1, 2, 3];
        let result: Result<Rc<Vec<i32>>, (Vec<i32>, AllocError)> =
            Rc::<Vec<i32>>::try_new_give_back(val);
        let _rc_val = result.unwrap();
    }

    // ── fallible_ alias tests ───────────────────────────────────────────────

    #[test]
    fn fallible_new_returns_value() {
        let rc = Rc::<i32>::fallible_new(42).unwrap();
        assert_eq!(*rc, 42);
    }

    #[test]
    fn fallible_new_uninit_works() {
        let _uninit: Rc<MaybeUninit<i32>> = Rc::<i32>::fallible_new_uninit().unwrap();
    }

    #[test]
    fn fallible_new_zeroed_works() {
        let uninit: Rc<MaybeUninit<u64>> = Rc::<u64>::fallible_new_zeroed().unwrap();
        let owned = Rc::try_unwrap(uninit).unwrap();
        let val = unsafe { owned.assume_init() };
        assert_eq!(val, 0);
    }

    #[test]
    fn fallible_new_give_back_success() {
        let rc = Rc::<String>::fallible_new_give_back("world".to_string()).unwrap();
        assert_eq!(rc.as_str(), "world");
    }

    // ── unwrap_or_try_clone tests ──────────────────────────────────────────────

    #[test]
    fn unwrap_or_try_clone_unique_returns_inner() {
        let rc = <Rc<[u8; 4]> as TryRc<[u8; 4]>>::try_new(*b"hi!!").unwrap();
        let val = Rc::<[u8; 4]>::unwrap_or_try_clone(rc).unwrap();
        assert_eq!(val, *b"hi!!");
    }

    #[test]
    fn unwrap_or_try_clone_shared_clones() {
        let rc = <Rc<i32> as TryRc<i32>>::try_new(42).unwrap();
        let _rc2 = rc.clone();
        let val = Rc::<i32>::unwrap_or_try_clone(rc).unwrap();
        assert_eq!(val, 42);
    }

    // ── TryClone tests ────────────────────────────────────────────────────────

    #[test]
    fn try_clone_increments_strong() {
        use crate::try_clone::TryClone;
        let rc = <Rc<i32> as TryRc<i32>>::try_new(42).unwrap();
        assert_eq!(Rc::strong_count(&rc), 1);
        let rc2 = rc.try_clone().unwrap();
        assert_eq!(Rc::strong_count(&rc), 2);
        assert_eq!(Rc::strong_count(&rc2), 2);
        assert!(Rc::ptr_eq(&rc, &rc2));
    }

    #[test]
    fn try_clone_preserves_value() {
        use crate::try_clone::TryClone;
        let rc = <Rc<String> as TryRc<String>>::try_new("hello".to_string()).unwrap();
        let rc2 = rc.try_clone().unwrap();
        assert_eq!(rc2.as_str(), "hello");
    }

    #[test]
    fn try_clone_multiple() {
        use crate::try_clone::TryClone;
        let rc = <Rc<u64> as TryRc<u64>>::try_new(999).unwrap();
        let mut clones = Vec::new();
        for i in 0..100 {
            clones.push(rc.try_clone().unwrap());
            assert_eq!(Rc::strong_count(&rc), i as usize + 2);
        }
        drop(clones);
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[test]
    fn try_clone_rejects_at_max_refcount() {
        use crate::try_clone::TryClone;
        let rc = <Rc<i32> as TryRc<i32>>::try_new(0).unwrap();
        let inner = rc_inner(&rc);
        inner.strong.set(MAX_REFCOUNT);
        assert!(rc.try_clone().is_err());
        inner.strong.set(1);
    }

    // ── Weak TryClone tests ───────────────────────────────────────────────────

    #[test]
    fn weak_try_clone_increments_weak_count() {
        use crate::try_clone::TryClone;
        let rc = <Rc<i32> as TryRc<i32>>::try_new(42).unwrap();
        let weak = Rc::downgrade(&rc);
        assert_eq!(Rc::weak_count(&rc), 1);
        let weak2 = weak.try_clone().unwrap();
        assert_eq!(Rc::weak_count(&rc), 2);
        drop(weak2);
        assert_eq!(Rc::weak_count(&rc), 1);
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
        let rc = <Rc<u64> as TryRc<u64>>::try_new(0).unwrap();
        let weak = Rc::downgrade(&rc);
        let mut clones = Vec::new();
        for i in 0..100 {
            clones.push(weak.try_clone().unwrap());
            assert_eq!(Rc::weak_count(&rc), i as usize + 2);
        }
        drop(clones);
        assert_eq!(Rc::weak_count(&rc), 1);
    }

    #[test]
    fn weak_try_clone_rejects_at_max_refcount() {
        use crate::try_clone::TryClone;
        let rc = <Rc<i32> as TryRc<i32>>::try_new(0).unwrap();
        let weak = Rc::downgrade(&rc);
        let inner = weak_inner(&weak);
        inner.weak.set(MAX_REFCOUNT);
        assert!(weak.try_clone().is_err());
        inner.weak.set(2);
        drop(weak);
        drop(rc);
    }

    // ── try_upgrade tests ───────────────────────────────────────────────────────

    #[test]
    fn weak_try_upgrade_success() {
        let rc = <Rc<i32> as TryRc<i32>>::try_new(42).unwrap();
        let weak = Rc::downgrade(&rc);
        assert_eq!(Rc::strong_count(&rc), 1);
        let rc2 = weak.try_upgrade().unwrap().unwrap();
        assert_eq!(*rc2, 42);
        assert_eq!(Rc::strong_count(&rc), 2);
        assert!(Rc::ptr_eq(&rc, &rc2));
    }

    #[test]
    fn weak_try_upgrade_dangling_returns_none() {
        let weak: Weak<i32> = Weak::new();
        assert!(weak.try_upgrade().is_none());
    }

    #[test]
    fn weak_try_upgrade_after_drop_returns_none() {
        let weak = {
            let rc = <Rc<i32> as TryRc<i32>>::try_new(99).unwrap();
            Rc::downgrade(&rc)
        };
        assert!(weak.try_upgrade().is_none());
    }

    #[test]
    fn weak_try_upgrade_overflow_rejects() {
        let rc = <Rc<i32> as TryRc<i32>>::try_new(0).unwrap();
        let weak = Rc::downgrade(&rc);
        let inner = weak_inner(&weak);
        inner.strong.set(MAX_REFCOUNT);
        let result = weak.try_upgrade();
        assert!(matches!(result, Some(Err(TryUpgradeError))));
        inner.strong.set(1);
    }

    #[test]
    fn weak_try_upgrade_multiple_roundtrip() {
        let rc = <Rc<String> as TryRc<String>>::try_new("hello".into()).unwrap();
        let weak = Rc::downgrade(&rc);
        for _ in 0..50 {
            let upgraded = weak.try_upgrade().unwrap().unwrap();
            assert_eq!(upgraded.as_str(), "hello");
            assert_eq!(Rc::strong_count(&rc), 2);
            drop(upgraded);
            assert_eq!(Rc::strong_count(&rc), 1);
        }
    }

    #[test]
    fn weak_try_upgrade_error_display() {
        assert!(format!("{}", TryUpgradeError).contains("exceeded"));
    }

    // ── Unsized Rc TryClone tests ──────────────────────────────────────────────

    #[test]
    fn rc_slice_try_clone() {
        use crate::try_clone::TryClone;
        let slice: &[i32] = &[1, 2, 3, 4, 5];
        let rc: Rc<[i32]> = Rc::from(slice);
        let rc2 = rc.try_clone().unwrap();
        assert_eq!(rc2.as_ref(), [1, 2, 3, 4, 5]);
        assert_eq!(Rc::strong_count(&rc), 2);
        assert!(Rc::ptr_eq(&rc, &rc2));
    }

    #[test]
    fn rc_dyn_trait_try_clone() {
        use crate::try_clone::TryClone;
        struct Greeter(String);
        impl fmt::Display for Greeter {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "Hello, {}!", self.0)
            }
        }
        let rc: Rc<dyn fmt::Display> = Rc::new(Greeter("world".into()));
        let rc2 = rc.try_clone().unwrap();
        assert_eq!(Rc::strong_count(&rc), 2);
        assert!(Rc::ptr_eq(&rc, &rc2));
    }

    // ── Unsized Weak TryClone tests ─────────────────────────────────────────────

    #[test]
    fn weak_slice_try_clone() {
        use crate::try_clone::TryClone;
        let rc: Rc<[i32]> = Rc::from(&[10, 20, 30][..]);
        let weak = Rc::downgrade(&rc);
        assert_eq!(Rc::weak_count(&rc), 1);
        let weak2 = weak.try_clone().unwrap();
        assert_eq!(Rc::weak_count(&rc), 2);
        drop(weak2);
        assert_eq!(Rc::weak_count(&rc), 1);
    }

    // ── try_pin tests ─────────────────────────────────────────────────────────

    #[test]
    fn rc_try_pin_returns_pinned() {
        let _pinned: Pin<Rc<i32>> = <Rc<i32> as TryRc<i32>>::try_pin(42).unwrap();
    }

    #[test]
    fn rc_try_pin_value_accessible() {
        let pinned: Pin<Rc<u64>> = <Rc<u64> as TryRc<u64>>::try_pin(999).unwrap();
        let val: &u64 = &pinned;
        assert_eq!(*val, 999);
    }

    #[test]
    fn rc_try_pin_zst() {
        let _pinned: Pin<Rc<()>> = <Rc<()> as TryRc<()>>::try_pin(()).unwrap();
    }

    // ── fallible_ alias tests for pin ─────────────────────────────────────────

    #[test]
    fn fallible_pin_works() {
        let _pinned: Pin<Rc<i32>> = Rc::<i32>::fallible_pin(42).unwrap();
    }

    // ── try_pin_give_back tests ─────────────────────────────────────────────

    #[test]
    fn rc_try_pin_give_back_success() {
        let pinned: Pin<Rc<String>> =
            <Rc<String> as TryRc<String>>::try_pin_give_back("hello".to_string()).unwrap();
        assert_eq!(pinned.as_str(), "hello");
    }

    #[test]
    fn rc_try_pin_give_back_signature() {
        let val = vec![1, 2, 3];
        let result: PinRcResult<Vec<i32>> =
            <Rc<Vec<i32>> as TryRc<Vec<i32>>>::try_pin_give_back(val);
        let _pinned = result.unwrap();
    }

    #[test]
    fn rc_try_pin_give_back_zst() {
        let _pinned: Pin<Rc<()>> = Rc::<()>::try_pin_give_back(()).unwrap();
    }

    #[test]
    fn fallible_pin_give_back_works() {
        let _pinned: Pin<Rc<i32>> = Rc::<i32>::fallible_pin_give_back(42).unwrap();
    }

    // ── DST with custom alignment tests ────────────────────────────────────────

    #[test]
    fn rc_try_clone_align_high() {
        use crate::try_clone::TryClone;
        struct AlignedData([u8; 64]);

        trait Process {
            fn process(&self) -> usize;
        }

        impl Process for AlignedData {
            fn process(&self) -> usize {
                self.0.len()
            }
        }

        let inner: AlignedData = AlignedData([42u8; 64]);
        let rc: Rc<dyn Process> = Rc::new(inner);
        assert_eq!(rc.process(), 64);

        let rc2 = rc.try_clone().unwrap();
        assert_eq!(Rc::strong_count(&rc), 2);
        assert!(Rc::ptr_eq(&rc, &rc2));
        assert_eq!(rc2.process(), 64);
    }

    // ── OOM tests ─────────────────────────────────────────────────────────────
    #[cfg(feature = "std")]
    mod oom {
        use super::*;
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        #[test]
        fn rc_fallible_new_fails_on_oom() {
            let r: Result<Rc<i32>, AllocError> = with_policy(FailPolicy::fail_next_alloc(), || {
                <Rc<i32> as TryRc<i32>>::fallible_new(42)
            });
            assert!(r.is_err());
        }

        #[test]
        fn rc_fallible_new_give_back_returns_value_on_oom() {
            let r: Result<Rc<i32>, (i32, AllocError)> =
                with_policy(FailPolicy::fail_next_alloc(), || {
                    <Rc<i32> as TryRc<i32>>::try_new_give_back(99)
                });
            assert!(r.is_err());
            if let Err((returned, _err)) = r {
                assert_eq!(returned, 99);
            }
        }

        #[test]
        fn rc_fallible_new_uninit_fails_on_oom() {
            let r: Result<Rc<MaybeUninit<i32>>, AllocError> = with_policy(
                FailPolicy::fail_next_alloc(),
                <Rc<i32> as TryRc<i32>>::fallible_new_uninit,
            );
            assert!(r.is_err());
        }

        #[test]
        fn rc_fallible_new_zeroed_fails_on_oom() {
            let r: Result<Rc<MaybeUninit<[u8; 16]>>, AllocError> = with_policy(
                FailPolicy::fail_next_alloc(),
                <Rc<[u8; 16]> as TryRc<[u8; 16]>>::fallible_new_zeroed,
            );
            assert!(r.is_err());
        }

        #[test]
        fn rc_fallible_pin_fails_on_oom() {
            let r: Result<Pin<Rc<i32>>, AllocError> =
                with_policy(FailPolicy::fail_next_alloc(), || {
                    <Rc<i32> as TryRc<i32>>::fallible_pin(42)
                });
            assert!(r.is_err());
        }

        #[test]
        fn rc_fallible_pin_give_back_returns_value_on_oom() {
            let r: Result<Pin<Rc<i64>>, (i64, AllocError)> =
                with_policy(FailPolicy::fail_next_alloc(), || {
                    <Rc<i64> as TryRc<i64>>::try_pin_give_back(99)
                });
            assert!(r.is_err());
            if let Err((returned, _err)) = r {
                assert_eq!(returned, 99);
            }
        }

        #[test]
        fn rc_try_default_fails_on_oom() {
            let r: Result<Rc<i32>, TryDefaultError> = with_policy(
                FailPolicy::fail_next_alloc(),
                <Rc<i32> as TryDefault>::try_default,
            );
            assert!(r.is_err());
        }

        #[test]
        fn rc_try_clone_succeeds_under_oom() {
            // Rc::try_clone only increments refcounts, no heap allocation.
            let rc = Rc::<i32>::new(42);
            let r: Result<Rc<i32>, TryCloneError> =
                with_policy(FailPolicy::fail_next_alloc(), || rc.try_clone());
            assert!(r.is_ok());
        }

        #[test]
        fn rc_nth_alloc_fail_targets_correct_call() {
            let (r1_ok, r2_err, r3_ok) = with_policy(FailPolicy::fail_nth_alloc(2), || {
                let r1: Result<Rc<u8>, AllocError> = <Rc<u8> as TryRc<u8>>::fallible_new(1);
                let r2: Result<Rc<u8>, AllocError> = <Rc<u8> as TryRc<u8>>::fallible_new(2);
                let r3: Result<Rc<u8>, AllocError> = <Rc<u8> as TryRc<u8>>::fallible_new(3);
                (r1.is_ok(), r2.is_err(), r3.is_ok())
            });
            assert!(r1_ok, "first alloc should succeed");
            assert!(r2_err, "second alloc should fail");
            assert!(r3_ok, "third alloc should succeed");
        }

        #[test]
        fn rc_oom_restores_allocation_afterwards() {
            let r: Result<Rc<i32>, AllocError> = with_policy(FailPolicy::fail_next_alloc(), || {
                <Rc<i32> as TryRc<i32>>::fallible_new(42)
            });
            assert!(r.is_err());
            let r: Result<Rc<i32>, AllocError> = <Rc<i32> as TryRc<i32>>::fallible_new(42);
            assert!(r.is_ok());
        }

        #[test]
        fn weak_try_upgrade_dangling_no_alloc_needed() {
            let weak: Weak<i32> = Weak::new();
            let result = with_policy(FailPolicy::fail_next_alloc(), || weak.try_upgrade());
            assert!(result.is_none());
        }
    }
}
