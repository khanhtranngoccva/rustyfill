//! [`TryExtend`] / [`TryExtendFromSlice`] implementations for `dashmap::DashSet`.

use crate::dashmap::{TryDashSet, TryDashSetError};
use crate::recovery::Resumable;
use crate::try_clone::TryClone;
use crate::try_extend::{TryExtend, TryExtendFromSlice};
use lang_core::cmp::Eq;
use lang_core::hash::Hash;
use lang_std::hash::BuildHasher;

impl<'s, T, S> TryExtendFromSlice<'s, T> for dashmap::DashSet<T, S>
where
    T: Eq + Hash + TryClone,
    S: BuildHasher + TryClone,
{
    type Error = TryDashSetError;

    fn try_extend_from_slice(&mut self, other: &'s [T]) -> Result<(), (&'s [T], TryDashSetError)> {
        let this: &Self = self;
        for (i, elem) in other.iter().enumerate() {
            if let Err((_, err)) =
                <Self as TryDashSet<T, S>>::try_insert_give_back(this, elem.clone())
            {
                return Err((&other[i..], err));
            }
        }
        Ok(())
    }
}

impl<T, S> TryExtend<T> for dashmap::DashSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher + TryClone,
{
    type Error = TryDashSetError;

    fn try_extend<Src>(
        &mut self,
        source: Src,
    ) -> Result<(), (TryDashSetError, Resumable<Src::Inner>)>
    where
        Src: crate::recovery::ResumableSource<Item = T>,
    {
        let this: &Self = self;
        let (head, mut iter) = source.safe_into_iter();

        if let Some(value) = head
            && let Err((v, e)) = Self::try_insert_give_back(this, value)
        {
            return Err((e, Resumable::new(v, iter)));
        }

        while let Some(value) = iter.next() {
            match Self::try_insert_give_back(this, value) {
                Ok(_) => {}
                Err((v, e)) => {
                    return Err((e, Resumable::new(v, iter)));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang_alloc::vec;
    use lang_alloc::vec::Vec;

    /// Drives the `head` + `while` loop arms of `TryExtend for DashSet`: a
    /// non-empty source inserts its first element via the head path and the
    /// rest through the iterator loop.
    #[test]
    fn dashset_try_extend_nonempty_source() {
        let set: dashmap::DashSet<i32> = dashmap::DashSet::new();
        let src = vec![1, 2, 3];
        set.try_extend(src).unwrap();
        assert_eq!(set.len(), 3);
        assert!(set.contains(&1) && set.contains(&2) && set.contains(&3));
    }

    /// An empty source hits only the `None` head arm and returns immediately.
    #[test]
    fn dashset_try_extend_empty_source() {
        let set: dashmap::DashSet<i32> = dashmap::DashSet::new();
        let src: Vec<i32> = vec![];
        set.try_extend(src).unwrap();
        assert!(set.is_empty());
    }
}
