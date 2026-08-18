//! [`TryExtend`] / [`TryExtendFromSlice`] implementations for `VecDeque<T>`.

use crate::alloc::vecdeque::TryVecDequeError;
use crate::recovery::Resumable;
use crate::try_clone::TryClone;
use crate::try_extend::{TryExtend, TryExtendFromSlice};

impl<'s, T: TryClone> TryExtendFromSlice<'s, T> for lang_alloc::collections::VecDeque<T> {
    type Error = TryVecDequeError;

    fn try_extend_from_slice(&mut self, other: &'s [T]) -> Result<(), (&'s [T], TryVecDequeError)> {
        if other.is_empty() {
            return Ok(());
        }
        self.try_reserve(other.len())
            .map_err(|e| (other, TryVecDequeError::Reserve(e)))?;
        for (i, item) in other.iter().enumerate() {
            match item.try_clone() {
                Ok(cloned) => {
                    self.push_back(cloned);
                }
                Err(e) => {
                    // No rollback: elements pushed before the failure are kept.
                    // Return the unprocessed tail so the caller can retry.
                    return Err((&other[i..], TryVecDequeError::Clone(e)));
                }
            }
        }
        Ok(())
    }
}

impl<T> TryExtend<T> for lang_alloc::collections::VecDeque<T> {
    type Error = TryVecDequeError;

    fn try_extend<S>(&mut self, source: S) -> Result<(), (TryVecDequeError, Resumable<S::Inner>)>
    where
        S: crate::recovery::ResumableSource<Item = T>,
    {
        let (head, mut iter) = source.safe_into_iter();

        if let Some(item) = head {
            if self.len() == self.capacity()
                && let Err(e) = self.try_reserve(1)
            {
                return Err((
                    TryVecDequeError::from(e),
                    Resumable::new(item, iter),
                ));
            }
            self.push_back(item);
        }

        let (lower, _) = iter.size_hint();
        if lower > 0
            && let Err(e) = self.try_reserve(lower)
        {
            return Err((
                TryVecDequeError::from(e),
                Resumable::from_remainder(iter),
            ));
        }
        while let Some(item) = iter.next() {
            if self.len() == self.capacity()
                && let Err(e) = self.try_reserve(1)
            {
                return Err((
                    TryVecDequeError::from(e),
                    Resumable::new(item, iter),
                ));
            }
            self.push_back(item);
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
    fn vecdeque_try_extend_via_trait() {
        let mut dq: lang_alloc::collections::VecDeque<i32> = Default::default();
        <_ as TryExtend<i32>>::try_extend(&mut dq, 10..13).unwrap();
        assert_eq!(dq.len(), 3);
        assert_eq!(dq[0], 10);
    }

    #[test]
    fn vecdeque_try_extend_from_slice_via_trait() {
        let mut dq: lang_alloc::collections::VecDeque<Vec<u8>> = Default::default();
        let slice: &[Vec<u8>] = &[vec![7]];
        <_ as TryExtendFromSlice<'_, Vec<u8>>>::try_extend_from_slice(&mut dq, slice).unwrap();
        assert_eq!(dq.len(), 1);
        assert_eq!(dq[0], vec![7]);
    }
}
