//! Fallible slice-to-`Vec` conversions.
//!
//! Provides the [`TrySlice`] trait with methods that mirror allocating
//! `&[T]` constructors but return [`Result`] to handle allocation failures
//! gracefully. Uses [`TryClone`](crate::try_clone::TryClone) for each element
//! copy so that clone-time allocation failures are also caught.

use crate::alloc::AllocError;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_to_owned::{TryToOwned, TryToOwnedError};
use core::alloc::Layout;
use core::mem::{self, MaybeUninit};
use core::ptr::{self, NonNull};
use super::vec_::TryVecError;

/// A trait for fallibly converting a slice into a [`Vec`].
///
/// Implemented for `[T]`. Methods reserve capacity upfront and use
/// [`TryClone`] for each element, returning [`TryVecError`] on failure.
pub trait TrySlice<T> {
    /// Fallibly copy this slice into a new [`Vec`].
    ///
    /// This is the fallible analogue of [`<[T]>::to_vec`]. Reserves capacity
    /// for the full slice length before cloning any elements, so that if a
    /// clone fails midway the vector is truncated back to its original state
    /// (empty in this case, since it was just created).
    ///
    /// Returns [`TryVecError::Reserve`] on allocation failure or
    /// [`TryVecError::Clone`] if an element's [`TryClone::try_clone`] fails.
    fn try_to_vec(&self) -> Result<Vec<T>, TryVecError>
    where
        T: TryClone;

    /// Fallibly repeat this slice `n` times into a new [`Vec`] by cloning.
    ///
    /// Mirrors [`<[T]>::repeat`] but uses [`TryClone`] for each element copy so
    /// that clone-time allocation failures are caught rather than panicking.
    /// Note that std's `repeat` only requires `T: Copy`; this method requires
    /// `T: TryClone` (which implies `Clone`), hence the `_clone` suffix to make
    /// the stricter bound explicit in the name.
    ///
    /// Reserves capacity upfront for the total length (`self.len() * n`).
    /// Returns an empty `Vec` when `n == 0` or the slice is empty.
    ///
    /// Returns [`TryVecError::Reserve`] on allocation failure,
    /// [`TryVecError::Overflow`] if `self.len() * n` overflows, or
    /// [`TryVecError::Clone`] if an element's [`TryClone::try_clone`] fails.
    /// On clone failure midway, the partial vector is discarded.
    fn try_repeat_clone(&self, n: usize) -> Result<Vec<T>, TryVecError>
    where
        T: TryClone;

    // ── Aliases with `fallible_` prefix ─────────────────────────────────────

    /// Alias for [`Self::try_to_vec`].
    fn fallible_to_vec(&self) -> Result<Vec<T>, TryVecError>
    where
        T: TryClone,
    {
        Self::try_to_vec(self)
    }

    /// Alias for [`Self::try_repeat_clone`].
    fn fallible_repeat_clone(&self, n: usize) -> Result<Vec<T>, TryVecError>
    where
        T: TryClone,
    {
        Self::try_repeat_clone(self, n)
    }
}

impl<T> TrySlice<T> for [T] {
    fn try_to_vec(&self) -> Result<Vec<T>, TryVecError>
    where
        T: TryClone,
    {
        let mut out = Vec::<T>::new();
        if !self.is_empty() {
            out.try_reserve(self.len()).map_err(|e| TryVecError::Reserve(e.into()))?;
        }
        for elem in self.iter() {
            out.push(elem.try_clone().map_err(TryVecError::Clone)?);
        }
        Ok(out)
    }

    fn try_repeat_clone(&self, n: usize) -> Result<Vec<T>, TryVecError>
    where
        T: TryClone,
    {
        let len = self.len();
        if len == 0 || n == 0 {
            return Ok(Vec::new());
        }
        let total_len = len
            .checked_mul(n)
            .ok_or(TryVecError::Overflow)?;
        let mut out = Vec::<T>::new();
        out.try_reserve(total_len).map_err(|e| TryVecError::Reserve(e.into()))?;
        for _ in 0..n {
            for elem in self.iter() {
                out.push(elem.try_clone().map_err(TryVecError::Clone)?);
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// TryToOwned impl for slices
// ---------------------------------------------------------------------------

impl<T> TryToOwned for [T]
where
    T: TryClone,
{
    fn try_to_owned(&self) -> Result<Vec<T>, TryToOwnedError> {
        let mut out = Vec::<T>::new();
        if !self.is_empty() {
            out.try_reserve(self.len())?;
        }
        for elem in self.iter() {
            out.push(elem.try_clone().map_err(TryToOwnedError::from)?);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Boxed slice TryClone + TryDefault
// ---------------------------------------------------------------------------
// Box<[T]> owns a dynamically-sized slice on the heap. Cloning requires
// allocating a new slice and cloning each element via T::try_clone().
// We allocate exactly the right size upfront (no overshoot, no shrinking) and
// follow the same MaybeUninit + guard pattern used for array cloning.
// The allocation is wrapped in a Box immediately so its Drop handles cleanup
// on both explicit errors and panics during element cloning.

/// Panic-safe guard that drops any initialized elements in a `MaybeUninit` slice
/// if dropped before `forget()` is called (e.g. on panic or early return).
struct SliceInitGuard<'a, T> {
    slots: &'a mut [MaybeUninit<T>],
    count: usize,
}

impl<'a, T> SliceInitGuard<'a, T> {
    fn new(slots: &'a mut [MaybeUninit<T>]) -> Self {
        Self { slots, count: 0 }
    }

    /// Disable the guard's Drop so that it no longer cleans up on scope exit.
    /// Call this only after all slots have been successfully initialized.
    fn forget(mut self) {
        self.count = 0;
        mem::forget(self);
    }
}

impl<'a, T> Drop for SliceInitGuard<'a, T> {
    fn drop(&mut self) {
        unsafe {
            for slot in self.slots.iter_mut().take(self.count) {
                core::ptr::drop_in_place(slot.as_mut_ptr());
            }
        }
    }
}

impl<T: TryClone> TryClone for Box<[T]> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let len = self.len();

        // Handle empty and ZST cases — no allocation needed.
        if len == 0 || mem::size_of::<T>() == 0 {
            let ptr: NonNull<T> =
                NonNull::new(ptr::without_provenance_mut(Layout::new::<T>().align()))
                    .expect("alignment should not be zero");
            // SAFETY: for empty slices and ZSTs, a dangling aligned pointer is valid.
            // We cast through a fat pointer constructed from the raw parts.
            return Ok(unsafe {
                let fat: *mut [T] = core::ptr::slice_from_raw_parts_mut(ptr.as_ptr().cast(), len);
                Box::from_raw(fat)
            });
        }

        // Allocate exactly `len` elements — no excess capacity, no shrinking.
        let layout = Layout::array::<T>(len).map_err(|_| TryCloneError::Overflow)?;
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(TryCloneError::Alloc(AllocError { layout }));
        }

        // Wrap immediately in a Box so Drop cleans up the allocation on panic.
        // SAFETY: layout matches `len` elements of MaybeUninit<T>, which has
        // the same size and alignment as T.
        let mut out: Box<[MaybeUninit<T>]> =
            unsafe { Box::from_raw(core::ptr::slice_from_raw_parts_mut(ptr.cast(), len)) };
        let mut guard = SliceInitGuard::new(&mut out);

        for (slot, elem) in guard.slots.iter_mut().zip(self.iter()) {
            match elem.try_clone() {
                Ok(cloned) => {
                    unsafe {
                        core::ptr::write(slot.as_mut_ptr(), cloned);
                    }
                    guard.count += 1;
                }
                Err(e) => {
                    // Guard drops initialized elements; Box drops the allocation.
                    return Err(e);
                }
            }
        }

        guard.forget();

        // SAFETY: all `len` slots were written successfully above.
        // Box<[MaybeUninit<T>]> and Box<[T]> have identical memory layouts.
        Ok(unsafe { mem::transmute::<Box<[MaybeUninit<T>]>, Box<[T]>>(out) })
    }
}

// ── Boxed slice TryDefault ─────────────────────────────────────────────────────
// An empty boxed slice is the natural default — no allocation needed beyond
// a thin/dangling pointer for ZST-like empty slices.

impl<T> TryDefault for Box<[T]> {
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Box::new([]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_to_vec_empty() {
        let s: &[i32] = &[];
        let v: Vec<i32> = s.try_to_vec().unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn try_to_vec_single() {
        let s: &[u8] = &[42];
        let v: Vec<u8> = s.try_to_vec().unwrap();
        assert_eq!(v, [42]);
    }

    #[test]
    fn try_to_vec_multiple() {
        let s: &[i32] = &[1, 2, 3];
        let v: Vec<i32> = s.try_to_vec().unwrap();
        assert_eq!(v, [1, 2, 3]);
    }

    #[test]
    fn try_to_vec_with_nested_vecs() {
        let s: &[Vec<u8>] = &[vec![1, 2], vec![3]];
        let v: Vec<Vec<u8>> = s.try_to_vec().unwrap();
        assert_eq!(v, [vec![1, 2], vec![3]]);
    }

    #[test]
    fn try_to_vec_with_options() {
        let s: &[Option<u32>] = &[Some(1), None, Some(3)];
        let v: Vec<Option<u32>> = s.try_to_vec().unwrap();
        assert_eq!(v, [Some(1), None, Some(3)]);
    }

    #[test]
    fn try_to_vec_deeply_nested() {
        let s: &[Vec<Vec<u8>>] = &[vec![vec![1]], vec![vec![2], vec![3]]];
        let v: Vec<Vec<Vec<u8>>> = s.try_to_vec().unwrap();
        assert_eq!(v, [vec![vec![1]], vec![vec![2], vec![3]]]);
    }

    #[test]
    fn try_to_vec_preserves_order() {
        let s: &[i32] = &(0..100).collect::<Vec<_>>();
        let v: Vec<i32> = s.try_to_vec().unwrap();
        assert_eq!(v, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn try_repeat_clone_zero_times() {
        let s: &[u8] = &[1, 2];
        let v: Vec<u8> = s.try_repeat_clone(0).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn try_repeat_clone_one_time() {
        let s: &[u8] = &[1, 2, 3];
        let v: Vec<u8> = s.try_repeat_clone(1).unwrap();
        assert_eq!(v, [1, 2, 3]);
    }

    #[test]
    fn try_repeat_clone_multiple_times() {
        let s: &[u8] = &[1, 2];
        let v: Vec<u8> = s.try_repeat_clone(3).unwrap();
        assert_eq!(v, [1, 2, 1, 2, 1, 2]);
    }

    #[test]
    fn try_repeat_clone_empty_slice() {
        let s: &[u8] = &[];
        let v: Vec<u8> = s.try_repeat_clone(5).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn try_repeat_clone_single_element() {
        let s: &[i32] = &[42];
        let v: Vec<i32> = s.try_repeat_clone(4).unwrap();
        assert_eq!(v, [42, 42, 42, 42]);
    }

    #[test]
    fn try_repeat_clone_overflow() {
        let s: &[u8] = &[1, 2];
        let result: Result<Vec<u8>, TryVecError> = s.try_repeat_clone(usize::MAX);
        assert!(matches!(result, Err(TryVecError::Overflow)));
    }

    #[test]
    fn try_repeat_clone_with_nested_vecs() {
        let s: &[Vec<u8>] = &[vec![1], vec![2]];
        let v: Vec<Vec<u8>> = s.try_repeat_clone(2).unwrap();
        assert_eq!(v, [vec![1], vec![2], vec![1], vec![2]]);
    }

    #[test]
    fn try_repeat_clone_matches_std() {
        let s: &[i32] = &[10, 20, 30];
        let expected: Vec<i32> = s.repeat(4);
        let actual: Vec<i32> = s.try_repeat_clone(4).unwrap();
        assert_eq!(actual, expected);
    }

    // ── TryToOwned tests ──────────────────────────────────────────────

    #[test]
    fn try_to_owned_empty_slice() {
        let s: &[u8] = &[];
        let v: Vec<u8> = s.try_to_owned().unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn try_to_owned_single() {
        let s: &[i32] = &[42];
        let v: Vec<i32> = s.try_to_owned().unwrap();
        assert_eq!(v, [42]);
    }

    #[test]
    fn try_to_owned_multiple() {
        let s: &[i32] = &[1, 2, 3];
        let v: Vec<i32> = s.try_to_owned().unwrap();
        assert_eq!(v, [1, 2, 3]);
    }

    #[test]
    fn try_to_owned_nested_vecs() {
        let s: &[Vec<u8>] = &[vec![1, 2], vec![3]];
        let v: Vec<Vec<u8>> = s.try_to_owned().unwrap();
        assert_eq!(v, [vec![1, 2], vec![3]]);
    }

    #[test]
    fn try_to_owned_preserves_order() {
        let s: &[i32] = &(0..100).collect::<Vec<_>>();
        let v: Vec<i32> = s.try_to_owned().unwrap();
        assert_eq!(v, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn try_to_owned_implies_to_owned_bound() {
        let s: &[i32] = &[1, 2];
        let owned: Vec<i32> = <[i32] as ToOwned>::to_owned(s);
        assert_eq!(owned, [1, 2]);
    }
}
