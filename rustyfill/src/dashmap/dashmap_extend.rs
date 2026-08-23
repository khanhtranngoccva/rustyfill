//! [`TryExtend`] / [`TryExtendFromSlice`] implementations for `dashmap::DashMap`.

use crate::dashmap::{TryDashMap, TryDashMapError};
use crate::recovery::Resumable;
use crate::try_clone::TryClone;
use crate::try_extend::{TryExtend, TryExtendFromSlice};
use lang_core::cmp::Eq;
use lang_core::hash::Hash;
use lang_std::hash::BuildHasher;

impl<'s, K, V, S> TryExtendFromSlice<'s, (K, V)> for dashmap::DashMap<K, V, S>
where
    K: Eq + Hash + TryClone,
    V: TryClone,
    S: BuildHasher + TryClone,
{
    type Error = TryDashMapError;

    fn try_extend_from_slice(
        &mut self,
        other: &'s [(K, V)],
    ) -> Result<(), (&'s [(K, V)], TryDashMapError)> {
        let this: &Self = self;
        for (i, (key, value)) in other.iter().enumerate() {
            if let Err((_, _, e)) = <Self as TryDashMap<K, V, S>>::try_insert_give_back(
                this,
                key.clone(),
                value.clone(),
            ) {
                return Err((&other[i..], TryDashMapError::Reserve(e)));
            }
        }
        Ok(())
    }
}

impl<K, V, S> TryExtend<(K, V)> for dashmap::DashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + TryClone,
{
    type Error = TryDashMapError;

    fn try_extend<Src>(
        &mut self,
        source: Src,
    ) -> Result<(), (Resumable<Src::Inner>, TryDashMapError)>
    where
        Src: crate::recovery::ResumableSource<Item = (K, V)>,
    {
        let this: &Self = self;
        let (head, mut iter) = source.safe_into_iter();

        if let Some(pair) = head
            && let Err((k, v, e)) = Self::try_insert_give_back(this, pair.0, pair.1)
        {
            return Err((Resumable::new((k, v), iter), TryDashMapError::Reserve(e)));
        }

        while let Some(pair) = iter.next() {
            match Self::try_insert_give_back(this, pair.0, pair.1) {
                Ok(_) => {}
                Err((k, v, e)) => {
                    return Err((Resumable::new((k, v), iter), TryDashMapError::Reserve(e)));
                }
            }
        }
        Ok(())
    }
}
