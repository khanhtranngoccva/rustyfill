#![allow(unstable_name_collisions)]
//! Fallible heap allocation for boxed values.
//!
//! Provides the [`TryBox`] trait with methods that mirror `Box` constructors
//! but return [`Result`] to handle allocation failures gracefully.

use crate::alloc::AllocError;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use core::alloc::Layout;
use core::mem::{self, MaybeUninit};
use core::pin::Pin;

/// A trait for fallibly allocating a value on the heap.
///
/// Implemented for `Box<T>`.
pub trait TryBox<T>: Sized {
    /// The uninitialized variant of this box.
    type Uninit: Sized;

    /// Fallibly allocate `value` on the heap.
    ///
    /// For zero-sized types this never fails and returns immediately.
    fn try_new(value: T) -> Result<Self, AllocError>;

    /// Fallibly allocate uninitialized memory on the heap.
    ///
    /// Returns a [`Box<MaybeUninit<T>>`][MaybeUninit] that can be initialized
    /// in place via [`MaybeUninit::write`] and converted to a `Box<T>` using
    /// the inherent [`Box<MaybeUninit<T>>::assume_init`].
    fn try_new_uninit() -> Result<Self::Uninit, AllocError>;

    /// Fallibly allocate zero-initialised memory on the heap.
    ///
    /// Returns a [`Box<MaybeUninit<T>>`][MaybeUninit] whose underlying bytes
    /// are all set to zero. Safe to call [`MaybeUninit::assume_init`] on types
    /// whose all-zeros bitpattern is valid (e.g. numeric primitives, `bool`,
    /// `[T; N]` where `T` is also zeroable).
    fn try_new_zeroed() -> Result<Self::Uninit, AllocError>;

    /// Like [`Self::try_new`] but returns ownership of `value` back on failure
    /// so it can be reused or dropped cleanly.
    fn try_new_give_back(value: T) -> Result<Self, (T, AllocError)>;

    /// Fallibly allocate `value` on the heap and pin it in place.
    ///
    /// Returns a [`Pin<Box<T>>`] so that if `T` does not implement [`Unpin`],
    /// the value is immovable after allocation. This is the fallible analogue
    /// of [`Box::pin`].
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

// Internal helper to allocate raw memory and wrap it in a Box<MaybeUninit<T>>.
// Kept in its own module so our TryBox trait is NOT in scope, avoiding any
// ambiguity with std's (unstable) Box::try_new_uninit inherent method.
mod alloc_inner {
    use super::*;

    pub(super) fn alloc_box_uninit<T>() -> Result<Box<MaybeUninit<T>>, AllocError> {
        // ZSTs never actually allocate — Box uses a dangling pointer internally.
        if mem::size_of::<T>() == 0 {
            return Ok(Box::new(MaybeUninit::uninit()));
        }

        let layout = Layout::new::<T>();
        let ptr = unsafe {
            let raw = std::alloc::alloc(layout);
            if raw.is_null() {
                return Err(AllocError);
            }
            raw.cast::<MaybeUninit<T>>()
        };
        Ok(unsafe { Box::from_raw(ptr) })
    }

    pub(super) fn alloc_box_zeroed<T>() -> Result<Box<MaybeUninit<T>>, AllocError> {
        if mem::size_of::<T>() == 0 {
            return Ok(Box::new(MaybeUninit::zeroed()));
        }

        let layout = Layout::new::<T>();
        let ptr = unsafe {
            let raw = std::alloc::alloc_zeroed(layout);
            if raw.is_null() {
                return Err(AllocError);
            }
            raw.cast::<MaybeUninit<T>>()
        };
        Ok(unsafe { Box::from_raw(ptr) })
    }
}

impl<T> TryBox<T> for Box<T> {
    type Uninit = Box<MaybeUninit<T>>;

    
    fn try_new(value: T) -> Result<Self, AllocError> {
        let mut slot = alloc_inner::alloc_box_uninit::<T>()?;
        slot.as_mut().write(value);
        // SAFETY: we just wrote a valid T into the slot above.
        Ok(unsafe { slot.assume_init() })
    }

    
    fn try_new_uninit() -> Result<Box<MaybeUninit<T>>, AllocError> {
        alloc_inner::alloc_box_uninit::<T>()
    }

    
    fn try_new_zeroed() -> Result<Box<MaybeUninit<T>>, AllocError> {
        alloc_inner::alloc_box_zeroed::<T>()
    }

    
    fn try_new_give_back(value: T) -> Result<Self, (T, AllocError)> {
        let mut slot = match alloc_inner::alloc_box_uninit::<T>() {
            Ok(s) => s,
            Err(e) => return Err((value, e)),
        };
        slot.as_mut().write(value);
        // SAFETY: we just wrote a valid T into the slot.
        Ok(unsafe { slot.assume_init() })
    }

    
    fn try_pin(value: T) -> Result<Pin<Self>, AllocError> {
        let boxed = Self::try_new(value)?;
        Ok(unsafe { Pin::new_unchecked(boxed) })
    }
}

impl<T: TryClone> TryClone for Box<T> {
    
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Allocate first so that if allocation fails we never touch T::try_clone().
        let mut slot = <Box<T> as TryBox<T>>::try_new_uninit().map_err(TryCloneError::Alloc)?;
        match (**self).try_clone() {
            Ok(cloned) => {
                slot.write(cloned);
                // SAFETY: we just wrote a valid T into the slot above.
                Ok(unsafe { slot.assume_init() })
            }
            Err(e) => Err(e),
        }
    }
}

impl<T: TryDefault> TryDefault for Box<T> {
    
    fn try_default() -> Result<Self, TryDefaultError> {
        // Allocate first so that if allocation fails we never touch T::try_default().
        let mut slot = <Box<T> as TryBox<T>>::try_new_uninit().map_err(TryDefaultError::Alloc)?;
        match T::try_default() {
            Ok(val) => {
                slot.write(val);
                // SAFETY: we just wrote a valid T into the slot above.
                Ok(unsafe { slot.assume_init() })
            }
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_returns_value() {
        let b = <Box<i32> as TryBox<i32>>::try_new(42).unwrap();
        assert_eq!(*b, 42);
    }

    #[test]
    fn try_new_zst() {
        let b = <Box<()> as TryBox<()>>::try_new(()).unwrap();
        assert_eq!(*b, ());
    }

    #[test]
    fn try_new_uninit_then_write_and_assume_init() {
        let uninit: Box<MaybeUninit<i32>> = <Box<i32> as TryBox<i32>>::try_new_uninit().unwrap();
        let init = unsafe {
            (*(&*uninit as *const MaybeUninit<i32> as *mut MaybeUninit<i32>)).write(99);
            uninit.assume_init()
        };
        assert_eq!(*init, 99);
    }

    #[test]
    fn try_new_uninit_zst() {
        let _uninit: Box<MaybeUninit<()>> = <Box<()> as TryBox<()>>::try_new_uninit().unwrap();
    }

    #[test]
    fn try_new_give_back_success() {
        let b = <Box<String> as TryBox<String>>::try_new_give_back("hello".to_string()).unwrap();
        assert_eq!(b.as_str(), "hello");
    }

    #[test]
    fn try_new_give_back_signature() {
        let val = vec![1, 2, 3];
        let result: Result<Box<Vec<i32>>, (Vec<i32>, AllocError)> =
            <Box<Vec<i32>> as TryBox<Vec<i32>>>::try_new_give_back(val);
        let _box_val = result.unwrap();
    }

    #[test]
    fn try_new_with_struct() {
        #[derive(Debug, PartialEq)]
        struct Point {
            x: i32,
            y: i32,
        }

        let p = <Box<Point> as TryBox<Point>>::try_new(Point { x: 10, y: 20 }).unwrap();
        assert_eq!(p.x, 10);
        assert_eq!(p.y, 20);
    }

    #[test]
    fn fallible_new_returns_value() {
        let b = Box::<i32>::fallible_new(42).unwrap();
        assert_eq!(*b, 42);
    }

    #[test]
    fn fallible_new_uninit_then_write_and_assume_init() {
        let uninit: Box<MaybeUninit<i32>> = Box::<i32>::fallible_new_uninit().unwrap();
        let init = unsafe {
            (*(&*uninit as *const MaybeUninit<i32> as *mut MaybeUninit<i32>)).write(99);
            uninit.assume_init()
        };
        assert_eq!(*init, 99);
    }

    #[test]
    fn fallible_new_give_back_success() {
        let b = Box::<String>::fallible_new_give_back("hello".to_string()).unwrap();
        assert_eq!(b.as_str(), "hello");
    }

    #[test]
    fn try_new_zeroed_returns_zeros() {
        let uninit: Box<MaybeUninit<i32>> = <Box<i32> as TryBox<i32>>::try_new_zeroed().unwrap();
        let val = unsafe { uninit.assume_init() };
        assert_eq!(*val, 0);
    }

    #[test]
    fn try_new_zeroed_array() {
        let uninit: Box<MaybeUninit<[u8; 4]>> =
            <Box<[u8; 4]> as TryBox<[u8; 4]>>::try_new_zeroed().unwrap();
        let arr = unsafe { uninit.assume_init() };
        assert_eq!(*arr, [0, 0, 0, 0]);
    }

    #[test]
    fn try_new_zeroed_zst() {
        let _uninit: Box<MaybeUninit<()>> = <Box<()> as TryBox<()>>::try_new_zeroed().unwrap();
    }

    #[test]
    fn fallible_new_zeroed_works() {
        let uninit: Box<MaybeUninit<f64>> = Box::<f64>::fallible_new_zeroed().unwrap();
        let val = unsafe { uninit.assume_init() };
        assert_eq!(val.to_bits(), 0);
    }

    // ── try_pin tests ─────────────────────────────────────────────────────────

    #[test]
    fn box_try_pin_returns_pinned() {
        let _pinned: Pin<Box<i32>> = <Box<i32> as TryBox<i32>>::try_pin(42).unwrap();
    }

    #[test]
    fn box_try_pin_value_accessible() {
        let pinned: Pin<Box<u64>> = Box::<u64>::try_pin(999).unwrap();
        let val: &u64 = &pinned;
        assert_eq!(*val, 999);
    }

    #[test]
    fn box_try_pin_zst() {
        let _pinned: Pin<Box<()>> = Box::<()>::try_pin(()).unwrap();
    }

    // ── fallible_ alias tests ─────────────────────────────────────────────────

    #[test]
    fn fallible_pin_works() {
        let _pinned: Pin<Box<i32>> = Box::<i32>::fallible_pin(42).unwrap();
    }
}
