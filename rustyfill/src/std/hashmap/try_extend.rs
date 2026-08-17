//! [`TryExtend`] / [`TryExtendFromSlice`] implementations for `HashMap<K, V, S>`.

use crate::alloc::TryReserveError;
use crate::recovery::Resumable;
use crate::std::hashmap::TryHashMapError;
use crate::try_clone::TryClone;
use crate::try_extend::{TryExtend, TryExtendFromSlice};
use lang_core::cmp::Eq;
use lang_core::hash::Hash;
use lang_std::hash::BuildHasher;

impl<'s, K, V, S> TryExtendFromSlice<'s, (K, V)> for lang_std::collections::HashMap<K, V, S>
where
    K: Eq + Hash + TryClone,
    V: TryClone,
    S: BuildHasher,
{
    type Error = TryHashMapError;

    fn try_extend_from_slice(
        &mut self,
        other: &'s [(K, V)],
    ) -> Result<(), (&'s [(K, V)], TryHashMapError)> {
        if other.is_empty() {
            return Ok(());
        }
        self.try_reserve(other.len())
            .map_err(|e| (other, TryHashMapError::Reserve(TryReserveError::from(e))))?;
        for (i, (key, value)) in other.iter().enumerate() {
            match (key.try_clone(), value.try_clone()) {
                (Ok(k), Ok(v)) => {
                    self.insert(k, v);
                }
                (Err(e), _) | (_, Err(e)) => {
                    // No rollback: keys may have been overwritten by later
                    // entries in `other`, so draining would resurrect stale
                    // values. Return the remaining subslice starting at the
                    // failing index so the caller can retry with just that tail.
                    return Err((&other[i..], TryHashMapError::Clone(e)));
                }
            }
        }
        Ok(())
    }
}

impl<K, V, S> TryExtend<(K, V)> for lang_std::collections::HashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    type Error = TryHashMapError;

    fn try_extend<Src>(
        &mut self,
        source: Src,
    ) -> Result<(), (TryHashMapError, Resumable<Src::Inner>)>
    where
        Src: crate::recovery::ResumableSource<Item = (K, V)>,
    {
        let (head, mut iter) = source.safe_into_iter();

        if let Some(pair) = head {
            if self.len() == self.capacity()
                && let Err(e) = self.try_reserve(1)
            {
                return Err((TryHashMapError::from(TryReserveError::from(e)), Resumable::new(pair, iter)));
            }
            self.insert(pair.0, pair.1);
        }

        let (lower, _) = iter.size_hint();
        if lower > 0
            && let Err(e) = self.try_reserve(lower)
        {
            return Err((TryHashMapError::from(TryReserveError::from(e)), Resumable::from_remainder(iter)));
        }
        while let Some(pair) = iter.next() {
            if self.len() == self.capacity()
                && let Err(e) = self.try_reserve(1)
            {
                return Err((TryHashMapError::from(TryReserveError::from(e)), Resumable::new(pair, iter)));
            }
            self.insert(pair.0, pair.1);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang_alloc::string::{String, ToString};
    use lang_alloc::vec;
    use lang_alloc::vec::Vec;
    use lang_std::collections::HashMap;

    #[test]
    fn hashmap_try_extend_via_trait() {
        let mut m: HashMap<i32, &str> = HashMap::new();
        <_ as TryExtend<(i32, &str)>>::try_extend(&mut m, [(1, "one"), (2, "two")]).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[&1], "one");
    }

    #[test]
    fn hashmap_try_extend_from_slice_via_trait() {
        let mut m: HashMap<String, Vec<u8>> = HashMap::new();
        let slice: &[(String, Vec<u8>)] = &[("a".to_string(), vec![1])];
        <_ as TryExtendFromSlice<'_, (String, Vec<u8>)>>::try_extend_from_slice(&mut m, slice)
            .unwrap();
        assert_eq!(m["a"], vec![1]);
    }
}
