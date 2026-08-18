//! [`TryExtend`] / [`TryExtendFromSlice`] implementations for `HashSet<T, S>`.

use crate::recovery::Resumable;
use crate::std::hashset::TryHashSetError;
use crate::try_clone::TryClone;
use crate::try_extend::{TryExtend, TryExtendFromSlice};
use lang_core::cmp::Eq;
use lang_core::hash::Hash;
use lang_std::hash::BuildHasher;

impl<'s, T, S> TryExtendFromSlice<'s, T> for lang_std::collections::HashSet<T, S>
where
    T: Eq + Hash + TryClone,
    S: BuildHasher,
{
    type Error = TryHashSetError;

    fn try_extend_from_slice(&mut self, other: &'s [T]) -> Result<(), (&'s [T], TryHashSetError)> {
        if other.is_empty() {
            return Ok(());
        }
        self.try_reserve(other.len())
            .map_err(|e| (other, TryHashSetError::Reserve(e)))?;
        for (i, elem) in other.iter().enumerate() {
            match elem.try_clone() {
                Ok(cloned) => {
                    self.insert(cloned);
                }
                Err(e) => {
                    // No rollback: return the remaining subslice starting at
                    // the failing index so the caller can retry with just that
                    // tail. Already-inserted elements are left in place.
                    return Err((&other[i..], TryHashSetError::Clone(e)));
                }
            }
        }
        Ok(())
    }
}

impl<T, S> TryExtend<T> for lang_std::collections::HashSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    type Error = TryHashSetError;

    fn try_extend<Src>(
        &mut self,
        source: Src,
    ) -> Result<(), (TryHashSetError, Resumable<Src::Inner>)>
    where
        Src: crate::recovery::ResumableSource<Item = T>,
    {
        let (head, mut iter) = source.safe_into_iter();

        if let Some(value) = head {
            if self.len() == self.capacity()
                && let Err(e) = self.try_reserve(1)
            {
                return Err((
                    TryHashSetError::from(e),
                    Resumable::new(value, iter),
                ));
            }
            self.insert(value);
        }

        let (lower, _) = iter.size_hint();
        if lower > 0
            && let Err(e) = self.try_reserve(lower)
        {
            return Err((
                TryHashSetError::from(e),
                Resumable::from_remainder(iter),
            ));
        }
        while let Some(value) = iter.next() {
            if self.len() == self.capacity()
                && let Err(e) = self.try_reserve(1)
            {
                return Err((
                    TryHashSetError::from(e),
                    Resumable::new(value, iter),
                ));
            }
            self.insert(value);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang_alloc::vec;
    use lang_alloc::vec::Vec;
    use lang_std::collections::HashSet;

    #[test]
    fn hashset_try_extend_via_trait() {
        let mut s: HashSet<i32> = HashSet::new();
        <_ as TryExtend<i32>>::try_extend(&mut s, [1, 2, 2, 3]).unwrap();
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn hashset_try_extend_from_slice_via_trait() {
        let mut s: HashSet<Vec<u8>> = HashSet::new();
        let slice: &[Vec<u8>] = &[vec![9]];
        <_ as TryExtendFromSlice<'_, Vec<u8>>>::try_extend_from_slice(&mut s, slice).unwrap();
        assert!(s.contains(&vec![9]));
    }
}
