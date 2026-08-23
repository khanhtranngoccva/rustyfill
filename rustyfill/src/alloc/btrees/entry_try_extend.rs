//! [`TryExtend`] / [`TryExtendFromSlice`] implementations for `BTreeMap<K, V>`
//! and `BTreeSet<T>`.

use crate::alloc::AllocError;
use crate::alloc::btrees::entry::{TryBTreeMap, TryBTreeWithCloneError, TryBTreeSet};
use crate::recovery::Resumable;
use crate::try_clone::TryClone;
use crate::try_extend::{TryExtend, TryExtendFromSlice};
use lang_core::cmp::Ord;

impl<'s, K: Ord + TryClone, V: TryClone> TryExtendFromSlice<'s, (K, V)>
    for lang_alloc::collections::BTreeMap<K, V>
{
    type Error = TryBTreeWithCloneError;

    fn try_extend_from_slice(
        &mut self,
        other: &'s [(K, V)],
    ) -> Result<(), (&'s [(K, V)], TryBTreeWithCloneError)> {
        for (i, (key, value)) in other.iter().enumerate() {
            match (key.try_clone(), value.try_clone()) {
                (Ok(k), Ok(v)) => {
                    // Use give_back so the cloned pair is returned (and dropped
                    // here) rather than silently consumed inside try_insert.
                    if let Err((_, _, e)) =
                        <Self as TryBTreeMap<K, V>>::try_insert_give_back(self, k, v)
                    {
                        return Err((&other[i..], TryBTreeWithCloneError::Alloc(e)));
                    }
                }
                (Err(e), _) | (_, Err(e)) => {
                    return Err((&other[i..], TryBTreeWithCloneError::Clone(e)));
                }
            }
        }
        Ok(())
    }
}

impl<K: Ord, V> TryExtend<(K, V)> for lang_alloc::collections::BTreeMap<K, V> {
    type Error = AllocError;

    fn try_extend<Src>(&mut self, source: Src) -> Result<(), (Resumable<Src::Inner>, AllocError)>
    where
        Src: crate::recovery::ResumableSource<Item = (K, V)>,
    {
        let (head, mut iter) = source.safe_into_iter();

        if let Some((key, value)) = head
            && let Err((k, v, e)) =
                <Self as TryBTreeMap<K, V>>::try_insert_give_back(self, key, value)
        {
            return Err((Resumable::new((k, v), iter), e));
        }

        while let Some((key, value)) = iter.next() {
            if let Err((k, v, e)) =
                <Self as TryBTreeMap<K, V>>::try_insert_give_back(self, key, value)
            {
                return Err((Resumable::new((k, v), iter), e));
            }
        }
        Ok(())
    }
}

impl<'s, T: Ord + TryClone> TryExtendFromSlice<'s, T> for lang_alloc::collections::BTreeSet<T> {
    type Error = TryBTreeWithCloneError;

    fn try_extend_from_slice(
        &mut self,
        other: &'s [T],
    ) -> Result<(), (&'s [T], TryBTreeWithCloneError)> {
        for (i, elem) in other.iter().enumerate() {
            match elem.try_clone() {
                Ok(v) => {
                    // Use give_back so the cloned element is returned (and
                    // dropped here) rather than silently consumed inside
                    // try_insert.
                    if let Err((_, e)) = <Self as TryBTreeSet<T>>::try_insert_give_back(self, v) {
                        return Err((&other[i..], TryBTreeWithCloneError::Alloc(e)));
                    }
                }
                Err(e) => {
                    return Err((&other[i..], TryBTreeWithCloneError::Clone(e)));
                }
            }
        }
        Ok(())
    }
}

impl<T: Ord> TryExtend<T> for lang_alloc::collections::BTreeSet<T> {
    type Error = AllocError;

    fn try_extend<Src>(&mut self, source: Src) -> Result<(), (Resumable<Src::Inner>, AllocError)>
    where
        Src: crate::recovery::ResumableSource<Item = T>,
    {
        let (head, mut iter) = source.safe_into_iter();

        if let Some(value) = head
            && let Err((v, e)) = <Self as TryBTreeSet<T>>::try_insert_give_back(self, value)
        {
            return Err((Resumable::new(v, iter), e));
        }

        while let Some(value) = iter.next() {
            if let Err((v, e)) = <Self as TryBTreeSet<T>>::try_insert_give_back(self, value) {
                return Err((Resumable::new(v, iter), e));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang_alloc::collections::{BTreeMap, BTreeSet};
    use lang_alloc::string::{String, ToString};
    use lang_alloc::vec;
    use lang_alloc::vec::Vec;

    #[test]
    fn btreemap_try_extend_via_trait() {
        let mut m: BTreeMap<i32, &str> = BTreeMap::new();
        <_ as TryExtend<(i32, &str)>>::try_extend(&mut m, [(1, "one"), (2, "two")]).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[&1], "one");
        assert_eq!(m[&2], "two");
    }

    #[test]
    fn btreeset_try_extend_via_trait() {
        let mut s: BTreeSet<i32> = BTreeSet::new();
        <_ as TryExtend<i32>>::try_extend(&mut s, [3, 1, 2, 1]).unwrap();
        assert_eq!(s.len(), 3);
        assert!(s.contains(&1));
        assert!(s.contains(&2));
        assert!(s.contains(&3));
    }

    #[test]
    fn btreemap_try_extend_from_slice_via_trait() {
        let mut m: BTreeMap<String, u8> = BTreeMap::new();
        let slice: &[(String, u8)] = &[("a".to_string(), 1), ("b".to_string(), 2)];
        <_ as TryExtendFromSlice<'_, (String, u8)>>::try_extend_from_slice(&mut m, slice).unwrap();
        assert_eq!(m["a"], 1);
        assert_eq!(m["b"], 2);
    }

    #[test]
    fn btreeset_try_extend_from_slice_via_trait() {
        let mut s: BTreeSet<Vec<u8>> = BTreeSet::new();
        let slice: &[Vec<u8>] = &[vec![5], vec![6]];
        <_ as TryExtendFromSlice<'_, Vec<u8>>>::try_extend_from_slice(&mut s, slice).unwrap();
        assert_eq!(s.len(), 2);
        assert!(s.contains(&vec![5]));
    }

    #[test]
    fn btreemap_try_extend_replaces_existing_keys() {
        let mut m: BTreeMap<i32, i32> = BTreeMap::new();
        m.insert(1, 100);
        <_ as TryExtend<(i32, i32)>>::try_extend(&mut m, [(1, 200), (2, 300)]).unwrap();
        assert_eq!(m[&1], 200);
        assert_eq!(m[&2], 300);
    }

    #[test]
    fn btreeset_try_extend_deduplicates() {
        let mut s: BTreeSet<i32> = BTreeSet::new();
        s.insert(42);
        <_ as TryExtend<i32>>::try_extend(&mut s, [42, 43]).unwrap();
        assert_eq!(s.len(), 2);
    }
}
