//! Placeholder for [`TryDashSet`] implementation.
//!
//! To be implemented in a subsequent chunk, following the same pattern as
//! [`TryDashMap`](super::dashmap_::TryDashMap).

use crate::alloc::{TryReserveError, TryReserveErrorExt};
use crate::dashmap::TryDashMap;
use crate::prelude::{TryClone, TryDefault};
use crate::try_clone::TryCloneError;
use crate::try_fmt::{TryDebug, TryDisplay};
use lang_core::alloc::Layout;
use lang_core::fmt;
use lang_core::mem;
use lang_std::cmp::Eq;
use lang_std::hash::{BuildHasher, Hash, RandomState};

type DashMap<K, V, S = RandomState> = dashmap::DashMap<K, V, S>;
type DashSet<T, S = RandomState> = dashmap::DashSet<T, S>;

// ── Error type ────────────────────────────────────────────────────────────────

/// Error returned by [`TryDashSet`] operations.
pub enum TryDashSetError {
    /// A capacity reservation on the DashSet failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An element clone failed during a method that requires [`TryClone`].
    Clone(TryCloneError),
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Debug for TryDashSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(self, f)
    }
}

impl fmt::Display for TryDashSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }
}

impl From<TryReserveError> for TryDashSetError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<TryCloneError> for TryDashSetError {
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

impl From<crate::try_default::TryDefaultError> for TryDashSetError {
    fn from(err: crate::try_default::TryDefaultError) -> Self {
        match err {
            crate::try_default::TryDefaultError::Reserve(e) => Self::Reserve(e),
            crate::try_default::TryDefaultError::Alloc(_) => Self::Other("allocation failed"),
            crate::try_default::TryDefaultError::Other(msg) => Self::Other(msg),
        }
    }
}

impl TryDebug for TryDashSetError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::debug_field(f, "TryDashSetError::Reserve", e),
            Self::Clone(e) => u::debug_field(f, "TryDashSetError::Clone", e),
            Self::Other(msg) => u::debug_field(f, "TryDashSetError::Other", msg),
        }
    }
}

impl TryDisplay for TryDashSetError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::display_delegated(f, "dash set", e),
            Self::Clone(e) => u::display_delegated(f, "dash set", e),
            Self::Other(msg) => u::display_fixed(f, "dash set", msg),
        }
    }
}

impl From<dashmap::TryReserveError> for TryDashSetError {
    fn from(_e: dashmap::TryReserveError) -> Self {
        // `dashmap::TryReserveError` is a zero-sized placeholder that carries
        // no layout information, so we cannot recover the exact failed layout.
        // Record a minimal placeholder layout; the important signal (that an
        // allocation, not a capacity overflow, failed) is preserved.
        Self::Reserve(TryReserveErrorExt::new_alloc(Layout::new::<u8>()))
    }
}

/// Error for fallible DashSet operations whose failure modes are limited to a
/// capacity reservation ([`TryReserveError`]) or an element clone failure
/// ([`TryCloneError`]).
///
/// Covers [`TryExtendFromSlice`](crate::try_extend::TryExtendFromSlice) and
/// other slice-based operations that clone elements before inserting.
pub enum TryDashSetWithCloneError {
    /// A capacity reservation on the DashSet failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An element clone failed during a method that requires [`TryClone`].
    Clone(TryCloneError),
}

impl fmt::Debug for TryDashSetWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(self, f)
    }
}

impl fmt::Display for TryDashSetWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }
}

impl From<TryReserveError> for TryDashSetWithCloneError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<TryCloneError> for TryDashSetWithCloneError {
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

impl TryDebug for TryDashSetWithCloneError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::debug_field(f, "TryDashSetWithCloneError::Reserve", e),
            Self::Clone(e) => u::debug_field(f, "TryDashSetWithCloneError::Clone", e),
        }
    }
}

impl TryDisplay for TryDashSetWithCloneError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::errors::uniform as u;
        match self {
            Self::Reserve(e) => u::display_delegated(f, "dash set", e),
            Self::Clone(e) => u::display_delegated(f, "dash set", e),
        }
    }
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// A trait for fallible DashSet operations.
///
/// Implemented for `dashmap::DashSet<T, S>`. Mirrors the most commonly-used
/// `DashSet` methods that can fail due to allocation pressure, returning
/// [`Result`] values that propagate [`TryDashSetError`] on failure.
pub trait TryDashSet<T, S = RandomState>: Sized {
    // ── Construction ────────────────────────────────────────────────────────

    /// Fallibly construct an empty `DashSet` with a default-constructed hasher.
    ///
    /// Unlike [`Self::try_with_capacity`] which previously hardcoded
    /// [`RandomState`], this method uses [`TryDefault`] to construct the hasher
    /// fallibly. If hasher construction fails (e.g. `RandomState` panics during
    /// seed initialization), the error is returned rather than unwinding.
    ///
    /// Requires `S: [`TryDefault`]` so that hasher creation is safe even when
    /// it involves runtime allocation or thread-local state.
    fn try_new() -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault;

    /// Fallibly construct an empty `DashSet` with at least enough capacity for
    /// `capacity` elements.
    ///
    /// Constructs the hasher via [`TryDefault`] (same as [`Self::try_new`]),
    /// then reserves capacity for `capacity` elements. Returns
    /// [`TryDashSetError::Reserve`] if the capacity reservation fails, or
    /// [`TryDashSetError::Other`] if hasher construction panics.
    ///
    /// Requires `S: [`TryDefault`]` so that hasher creation is safe even when
    /// it involves runtime allocation or thread-local state.
    fn try_with_capacity(capacity: usize) -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault;

    /// Fallibly construct an empty `DashSet` with at least enough capacity for
    /// `capacity` elements, using the provided hash builder.
    fn try_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<DashSet<T, S>, TryDashSetError>;

    // ── Insertion ───────────────────────────────────────────────────────────

    /// Fallibly insert the value into the set.
    ///
    /// Returns `true` if the value was not already present, `false` otherwise.
    fn try_insert(&self, value: T) -> Result<bool, TryReserveError>
    where
        T: Eq + Hash;

    /// Like [`Self::try_insert`] but returns ownership of `value` back on
    /// allocation failure.
    fn try_insert_give_back(&self, value: T) -> Result<bool, (T, TryReserveError)>
    where
        T: Eq + Hash;

    // ── Aliases with `fallible_` prefix ────────────────────────────────────

    /// Alias for [`Self::try_new`].
    fn fallible_new() -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault,
    {
        Self::try_new()
    }

    /// Alias for [`Self::try_with_capacity`].
    fn fallible_with_capacity(capacity: usize) -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault,
    {
        Self::try_with_capacity(capacity)
    }

    /// Alias for [`Self::try_with_capacity_and_hasher`].
    fn fallible_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<DashSet<T, S>, TryDashSetError> {
        Self::try_with_capacity_and_hasher(capacity, hasher)
    }

    /// Alias for [`Self::try_insert`].
    fn fallible_insert(&self, value: T) -> Result<bool, TryReserveError>
    where
        T: Eq + Hash,
    {
        Self::try_insert(self, value)
    }

    /// Alias for [`Self::try_insert_give_back`].
    fn fallible_insert_give_back(&self, value: T) -> Result<bool, (T, TryReserveError)>
    where
        T: Eq + Hash,
    {
        Self::try_insert_give_back(self, value)
    }

    // ── Extension ───────────────────────────────────────────────────────────

    /// Fallibly extend the set with all values from an iterator source.
    ///
    /// Accepts anything that implements [`ResumableSource`](crate::recovery::ResumableSource).
    /// Inserts each value individually via [`Self::try_insert_give_back`] so that
    /// on failure the consumed-but-uncommitted element is returned in a
    /// [`Resumable`](crate::recovery::Resumable).
    ///
    /// Note: elements already inserted before the failure are not rolled back.
    fn try_extend<Src>(
        &self,
        source: Src,
    ) -> Result<(), (crate::recovery::Resumable<Src::Inner>, TryDashSetError)>
    where
        Src: crate::recovery::ResumableSource<Item = T>;

    /// Alias for [`Self::try_extend`].
    fn fallible_extend<Src>(
        &self,
        source: Src,
    ) -> Result<(), (crate::recovery::Resumable<Src::Inner>, TryDashSetError)>
    where
        Src: crate::recovery::ResumableSource<Item = T>,
    {
        Self::try_extend(self, source)
    }

    // ── Capacity / shrink ───────────────────────────────────────────────────

    /// Fallibly shrink the capacity of this DashSet to match its length.
    ///
    /// Rebuilds the internal table so that it holds approximately `len` elements.
    /// Returns [`TryDashSetError::Reserve`] if the allocation for the rebuilt
    /// table fails. Equivalent to [`DashSet::shrink_to_fit`](dashmap::DashSet::shrink_to_fit)
    /// but fallible.
    fn try_shrink_to_fit(&self) -> Result<(), TryDashSetError>;

    /// Fallibly shrink the capacity of this DashSet to match its length.
    ///
    /// Alias for [`Self::try_shrink_to_fit`]
    fn fallible_shrink_to_fit(&self) -> Result<(), TryDashSetError> {
        Self::try_shrink_to_fit(self)
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    /// Fallibly collect an iterator of `T` values into a `DashSet`.
    ///
    /// Constructs the hasher via [`TryDefault`] and uses the iterator's size
    /// hint to pre-allocate when possible.
    fn try_collect<I: IntoIterator<Item = T>>(iter: I) -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault;

    /// Fallibly create a `DashSet` from an iterator using the provided hasher.
    fn try_collect_with_hasher<I: IntoIterator<Item = T>>(
        iter: I,
        hasher: S,
    ) -> Result<DashSet<T, S>, TryDashSetError>;

    /// Alias for [`Self::try_collect`].
    fn fallible_collect<I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault,
    {
        Self::try_collect(iter)
    }

    /// Alias for [`Self::try_collect_with_hasher`].
    fn fallible_collect_with_hasher<I: IntoIterator<Item = T>>(
        iter: I,
        hasher: S,
    ) -> Result<DashSet<T, S>, TryDashSetError> {
        Self::try_collect_with_hasher(iter, hasher)
    }
}

// ── Implementation ────────────────────────────────────────────────────────────

/// Convert to the inner dashmap type. This operation is useful for reserve operations.
fn convert_ref<T, S>(set: &DashSet<T, S>) -> &DashMap<T, (), S> {
    // SAFETY: DashSet<T, S> is supposed to be a thin newtype wrapper around DashMap<T, (), S> with a single `pub(crate) inner` field. The memory layout is identical, so reborrowing as &DashMap<T, (), S> is safe. The assertion below should break if any extra fields are detected.
    const {
        let map_layout: Layout = Layout::new::<DashMap<T, (), S>>();
        let set_layout: Layout = Layout::new::<DashSet<T, S>>();
        assert!(map_layout.size() == set_layout.size());
        assert!(map_layout.align() == set_layout.align());
    };
    unsafe { mem::transmute(set) }
}

/// Mutable variant of [`convert_ref`].
fn convert_mut<T, S>(set: &mut DashSet<T, S>) -> &mut DashMap<T, (), S> {
    // SAFETY: Same layout guarantees as [`convert_ref`], extended to mutable references.
    unsafe { mem::transmute(set) }
}

impl<T: Eq + Hash, S: BuildHasher + TryClone> TryDashSet<T, S> for DashSet<T, S> {
    // ── Construction ────────────────────────────────────────────────────────

    fn try_new() -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault,
    {
        let hasher = S::try_default()?;
        Ok(DashSet::with_hasher(hasher))
    }

    fn try_with_capacity(capacity: usize) -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault,
    {
        let mut set = Self::try_new()?;
        if capacity > 0 {
            convert_mut(&mut set)
                .try_reserve(capacity)
                .map_err(TryDashSetError::from)?;
        }
        Ok(set)
    }

    fn try_with_capacity_and_hasher(
        capacity: usize,
        hasher: S,
    ) -> Result<DashSet<T, S>, TryDashSetError> {
        let mut set = DashSet::with_hasher(hasher);
        if capacity > 0 {
            convert_mut(&mut set)
                .try_reserve(capacity)
                .map_err(TryDashSetError::from)?;
        }
        Ok(set)
    }

    // ── Insertion ───────────────────────────────────────────────────────────

    fn try_insert(&self, value: T) -> Result<bool, TryReserveError>
    where
        T: Eq + Hash,
    {
        let map = convert_ref(self);
        let had_old = map.try_insert(value, ())?;
        Ok(had_old.is_none())
    }

    fn try_insert_give_back(&self, value: T) -> Result<bool, (T, TryReserveError)>
    where
        T: Eq + Hash,
    {
        let map = convert_ref(self);
        match map.try_insert_give_back(value, ()) {
            Ok(had_old) => Ok(had_old.is_none()),
            Err((k, _unit, reserve_err)) => Err((k, reserve_err)),
        }
    }

    // ── Extension ───────────────────────────────────────────────────────────

    fn try_extend<Src>(
        &self,
        source: Src,
    ) -> Result<(), (crate::recovery::Resumable<Src::Inner>, TryDashSetError)>
    where
        Src: crate::recovery::ResumableSource<Item = T>,
        T: Eq + Hash,
    {
        use crate::recovery::Resumable;

        let (head, mut iter) = source.safe_into_iter();

        if let Some(value) = head
            && let Err((v, e)) = Self::try_insert_give_back(self, value)
        {
            return Err((Resumable::new(v, iter), TryDashSetError::Reserve(e)));
        }

        while let Some(value) = iter.next() {
            match Self::try_insert_give_back(self, value) {
                Ok(_) => {}
                Err((v, e)) => {
                    return Err((Resumable::new(v, iter), TryDashSetError::Reserve(e)));
                }
            }
        }
        Ok(())
    }

    // ── Capacity / shrink ───────────────────────────────────────────────────

    fn try_shrink_to_fit(&self) -> Result<(), TryDashSetError> {
        convert_ref(self).try_shrink_to_fit().map_err(|e| match e {
            super::TryDashMapError::Reserve(r) => TryDashSetError::Reserve(r),
            super::TryDashMapError::Clone(c) => TryDashSetError::Clone(c),
            super::TryDashMapError::Other(m) => TryDashSetError::Other(m),
        })
    }

    // ── Bulk construction ───────────────────────────────────────────────────

    fn try_collect<I: IntoIterator<Item = T>>(iter: I) -> Result<DashSet<T, S>, TryDashSetError>
    where
        S: TryDefault,
    {
        let set = Self::try_new()?;
        for value in iter {
            <DashSet<T, S> as TryDashSet<T, S>>::try_insert(&set, value)?;
        }
        Ok(set)
    }

    fn try_collect_with_hasher<I: IntoIterator<Item = T>>(
        iter: I,
        hasher: S,
    ) -> Result<DashSet<T, S>, TryDashSetError> {
        let set = DashSet::with_hasher(hasher);
        for value in iter {
            Self::try_insert(&set, value)?;
        }
        Ok(set)
    }
}

// ── TryClone for DashSet<T, S> ────────────────────────────────────────────────

impl<T, S> crate::try_clone::TryClone for DashSet<T, S>
where
    T: Eq + Hash + crate::try_clone::TryClone,
    S: BuildHasher + TryClone,
{
    fn try_clone(&self) -> Result<Self, crate::try_clone::TryCloneError> {
        let map_ref = convert_ref(self);
        let out = DashSet::with_hasher(map_ref.hasher().clone());
        // We cannot reserve everything immediately because shards may be uneven.
        if !self.is_empty() {
            let map: &DashMap<T, (), S> = convert_ref(&out);
            for elem in self.iter() {
                let entry = TryDashMap::try_entry_ref(map, &elem).map_err(|e| match e {
                    crate::dashmap::TryDashMapError::Reserve(r) => TryCloneError::Reserve(r),
                    crate::dashmap::TryDashMapError::Clone(c) => c,
                    crate::dashmap::TryDashMapError::Other(m) => TryCloneError::Other(m),
                })?;
                entry.insert(());
            }
        }
        Ok(out)
    }
}

// ── TryDefault for DashSet<T, S> ──────────────────────────────────────────────

impl<T: Eq + Hash, S> crate::try_default::TryDefault for DashSet<T, S>
where
    S: BuildHasher + TryClone + TryDefault,
{
    fn try_default() -> Result<Self, crate::try_default::TryDefaultError> {
        Ok(DashSet::with_hasher(S::try_default()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::try_clone::TryClone as _;
    use crate::try_default::TryDefault as _;
    use lang_alloc::format;
    use lang_alloc::string::String;
    use lang_alloc::string::ToString;
    use lang_alloc::vec;
    use lang_core::fmt::Write as _;
    use lang_std::iter;

    /// A `TryReserveError` instance for exercising the `Reserve` arm.
    fn reserve_err() -> TryReserveError {
        TryReserveError::new_capacity_overflow()
    }

    /// Formats a value via its `Display` impl into a fresh String.
    fn render_display(e: &impl fmt::Display) -> String {
        let mut s = String::new();
        // Our error Display impls only call `write!` on literals/wrapped values,
        // so this cannot fail in practice; ignore the infallible-in-practice result.
        let _ = write!(&mut s, "{e}");
        s
    }

    /// Captures the `TryDebug` rendering of a value.
    fn render_trydebug(e: &impl TryDebug) -> String {
        struct Cap<'a>(&'a dyn TryDebug);
        impl fmt::Debug for Cap<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.try_fmt(f)
            }
        }
        format!("{:?}", Cap(e))
    }

    /// Captures the `TryDisplay` rendering of a value (should match `Display`).
    fn render_trydisplay(e: &impl TryDisplay) -> String {
        struct Cap<'a>(&'a dyn TryDisplay);
        impl fmt::Display for Cap<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.try_fmt(f)
            }
        }
        let mut s = String::new();
        let _ = write!(&mut s, "{}", Cap(e));
        s
    }

    /// Exercises every variant of `TryDashSetError` through all three impls
    /// (moved from errors::uniform).
    #[test]
    fn dashset_error_covers_all_variants() {
        let errs = [
            TryDashSetError::Reserve(reserve_err()),
            TryDashSetError::Clone(TryCloneError::Reserve(reserve_err())),
            TryDashSetError::Other("ds"),
        ];
        for err in errs.iter() {
            let disp = render_display(err);
            assert!(
                disp.starts_with("dash set operation failed:"),
                "got {disp:?}"
            );
            let tdisp = render_trydisplay(err);
            assert_eq!(tdisp, disp, "TryDisplay must match Display");
            let dbg = render_trydebug(err);
            assert!(dbg.contains("TryDashSetError::"), "got {dbg:?}");
        }
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn try_with_capacity_zero() {
        let set: DashSet<i32> = DashSet::<i32>::try_with_capacity(0).unwrap();
        assert!(set.is_empty());
    }

    #[test]
    fn try_with_capacity_nonzero() {
        let set: DashSet<String> = DashSet::<String>::try_with_capacity(10).unwrap();
        assert!(set.is_empty());
        assert!(set.capacity() >= 10);
    }

    // ── Insertion ────────────────────────────────────────────────────────────

    #[test]
    fn try_insert_single() {
        let set: DashSet<i32> = DashSet::new();
        let inserted = set.try_insert(42).unwrap();
        assert!(inserted);
        assert!(set.contains(&42));
    }

    #[test]
    fn try_insert_duplicate_returns_false() {
        let set: DashSet<i32> = DashSet::new();
        assert!(set.try_insert(1).unwrap());
        assert!(!set.try_insert(1).unwrap());
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn try_insert_multiple_values() {
        let set: DashSet<&str> = DashSet::new();
        set.try_insert("alpha").unwrap();
        set.try_insert("beta").unwrap();
        set.try_insert("gamma").unwrap();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn fallible_insert_matches_try_insert() {
        let set: DashSet<i32> = DashSet::new();
        set.fallible_insert(99).unwrap();
        assert!(set.contains(&99));
    }

    // ── Give-back variants ───────────────────────────────────────────────────

    #[test]
    fn try_insert_give_back_success() {
        let set: DashSet<String> = DashSet::new();
        set.try_insert_give_back("hello".to_string()).unwrap();
        assert!(set.contains("hello"));
    }

    #[test]
    fn try_insert_give_back_error_type_shape() {
        let set: DashSet<i32> = DashSet::new();
        let result: Result<bool, (i32, TryReserveError)> = set.try_insert_give_back(1);
        assert!(result.is_ok());
    }

    // ── Extension ────────────────────────────────────────────────────────────

    #[test]
    fn try_extend_from_iterator() {
        let set: DashSet<i32> = DashSet::new();
        set.try_extend([1, 2, 3]).unwrap();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn try_extend_empty() {
        let set: DashSet<i32> = DashSet::new();
        set.try_extend(iter::empty::<i32>()).unwrap();
        assert!(set.is_empty());
    }

    // ── Shrink ────────────────────────────────────────────────────────────────

    #[test]
    fn try_shrink_to_fit_preserves_data() {
        let set: DashSet<String> = DashSet::new();
        set.try_insert("hello".to_string()).unwrap();
        set.try_insert("world".to_string()).unwrap();
        set.try_shrink_to_fit().unwrap();
        assert!(set.contains("hello"));
        assert!(set.contains("world"));
        assert_eq!(set.len(), 2);
    }

    // ── Bulk construction ────────────────────────────────────────────────────

    #[test]
    fn try_collect_range() {
        let set: DashSet<i32> =
            <DashSet<i32> as TryDashSet<_, RandomState>>::try_collect(0..5).unwrap();
        assert_eq!(set.len(), 5);
        assert!(set.contains(&3));
    }

    #[test]
    fn try_collect_empty() {
        let set: DashSet<i32> =
            <DashSet<i32> as TryDashSet<_, RandomState>>::try_collect(iter::empty::<i32>())
                .unwrap();
        assert!(set.is_empty());
    }

    #[test]
    fn try_collect_with_deduplication() {
        let set: DashSet<i32> =
            <DashSet<i32> as TryDashSet<_, RandomState>>::try_collect(vec![1, 2, 2, 3, 3]).unwrap();
        assert_eq!(set.len(), 3);
    }

    // ── TryClone ─────────────────────────────────────────────────────────────

    #[test]
    fn try_clone_empty_set() {
        let set: DashSet<i32> = DashSet::new();
        let c = set.try_clone().unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn try_clone_populated_set() {
        let set: DashSet<String> = DashSet::new();
        set.insert("hello".to_string());
        set.insert("world".to_string());
        let c = set.try_clone().unwrap();
        assert!(c.contains("hello"));
        assert!(c.contains("world"));
    }

    // ── TryDefault ───────────────────────────────────────────────────────────

    #[test]
    fn try_default_empty_set() {
        let set: DashSet<i32> = DashSet::try_default().unwrap();
        assert!(set.is_empty());
    }

    // ── Combined workflows ───────────────────────────────────────────────────

    #[test]
    fn build_insert_clone_default() {
        let set: DashSet<String> = DashSet::try_default().unwrap();
        set.fallible_insert("alpha".to_string()).unwrap();
        set.fallible_insert("beta".to_string()).unwrap();
        let c = set.try_clone().unwrap();
        assert!(c.contains("alpha"));
        assert!(c.contains("beta"));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn collect_then_extend() {
        let set: DashSet<i32> =
            <DashSet<i32> as TryDashSet<_, RandomState>>::try_collect([1, 2]).unwrap();
        set.try_extend([3, 4]).unwrap();
        assert_eq!(set.len(), 4);
    }

    // ── OOM tests ─────────────────────────────────────────────────────
    #[cfg(feature = "std")]
    mod oom {
        use super::*;
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        #[test]
        fn dashset_try_insert_fails_on_oom() {
            let set: DashSet<u32> = DashSet::new();
            let r = with_policy(FailPolicy::fail_next_alloc(), || set.try_insert(1));
            assert!(r.is_err());
        }

        #[test]
        fn dashset_oom_restores_allocation_afterwards() {
            let set: DashSet<u32> = DashSet::new();
            let r = with_policy(FailPolicy::fail_next_alloc(), || set.try_insert(1));
            assert!(r.is_err());
            assert!(set.try_insert(1).is_ok());
        }
    }
}
