//! [`TryExtend`] / [`TryExtendFromSlice`] implementations for `Vec<T>`.

use crate::alloc::vec::TryVecWithCloneError;
use crate::alloc::TryReserveError;
use crate::recovery::Resumable;
use crate::try_clone::TryClone;
use crate::try_extend::{TryExtend, TryExtendFromSlice};

impl<'s, T: TryClone> TryExtendFromSlice<'s, T> for lang_alloc::vec::Vec<T> {
    type Error = TryVecWithCloneError;

    fn try_extend_from_slice(&mut self, other: &'s [T]) -> Result<(), (&'s [T], TryVecWithCloneError)> {
        if other.is_empty() {
            return Ok(());
        }
        self.try_reserve(other.len())
            .map_err(|e| (other, TryVecWithCloneError::Reserve(e)))?;
        for (i, item) in other.iter().enumerate() {
            match item.try_clone() {
                Ok(cloned) => {
                    self.push(cloned);
                }
                Err(e) => {
                    // No rollback: elements pushed before the failure are kept.
                    // Return the unprocessed tail so the caller can retry.
                    return Err((&other[i..], TryVecWithCloneError::Clone(e)));
                }
            }
        }
        Ok(())
    }
}

impl<T> TryExtend<T> for lang_alloc::vec::Vec<T> {
    type Error = TryReserveError;

    fn try_extend<S>(&mut self, source: S) -> Result<(), (Resumable<S::Inner>, TryReserveError)>
    where
        S: crate::recovery::ResumableSource<Item = T>,
    {
        let (head, mut iter) = source.safe_into_iter();

        if let Some(h) = head {
            if self.len() == self.capacity()
                && let Err(e) = self.try_reserve(1)
            {
                return Err((Resumable::new(h, iter), e));
            }
            self.push(h);
        }

        let (lower, _) = iter.size_hint();
        if lower > 0
            && let Err(e) = self.try_reserve(lower)
        {
            return Err((Resumable::from_remainder(iter), e));
        }
        while let Some(item) = iter.next() {
            if self.len() == self.capacity()
                && let Err(e) = self.try_reserve(1)
            {
                return Err((Resumable::new(item, iter), e));
            }
            self.push(item);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang_alloc::vec;
    use lang_alloc::vec::Vec;

    #[test]
    fn vec_try_extend_via_trait() {
        let mut v: Vec<i32> = Vec::new();
        <_ as TryExtend<i32>>::try_extend(&mut v, 0..5).unwrap();
        assert_eq!(v, [0, 1, 2, 3, 4]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn try_extend_empty_source() {
        use lang_std::iter;
        let mut v: Vec<i32> = Vec::new();
        <_ as TryExtend<i32>>::try_extend(&mut v, iter::empty::<i32>()).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn vec_try_extend_from_slice_via_trait() {
        let mut v: Vec<Vec<u8>> = Vec::new();
        let slice: &[Vec<u8>] = &[vec![1], vec![2]];
        <_ as TryExtendFromSlice<'_, Vec<u8>>>::try_extend_from_slice(&mut v, slice).unwrap();
        assert_eq!(v, [vec![1], vec![2]]);
    }

    #[test]
    fn try_extend_from_slice_empty_input() {
        let mut v: Vec<i32> = Vec::new();
        <_ as TryExtendFromSlice<'_, i32>>::try_extend_from_slice(&mut v, &[]).unwrap();
        assert!(v.is_empty());
    }

    // ── Generic bounds check ─────────────────────────────────────────────────

    fn assert_vec_impls_traits(v: &mut Vec<i32>) {
        <_ as TryExtend<i32>>::try_extend(v, 0..2).unwrap();
        <_ as TryExtendFromSlice<'_, i32>>::try_extend_from_slice(v, &[9]).unwrap();
    }

    #[test]
    fn generic_use_of_both_traits() {
        let mut v: Vec<i32> = Vec::new();
        assert_vec_impls_traits(&mut v);
        assert_eq!(v, [0, 1, 9]);
    }
}
