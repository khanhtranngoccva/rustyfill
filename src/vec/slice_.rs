//! Fallible slice-to-`Vec` conversions.
//!
//! Provides the [`TrySlice`] trait with methods that mirror allocating
//! `&[T]` constructors but return [`Result`] to handle allocation failures
//! gracefully. Uses [`TryClone`](crate::try_clone::TryClone) for each element
//! copy so that clone-time allocation failures are also caught.

use crate::try_clone::TryClone;
use crate::try_to_owned::{TryToOwned, TryToOwnedError};
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
