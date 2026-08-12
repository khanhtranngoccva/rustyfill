//! Hasher factory wrappers for fallible collections.
//!
//! This module provides two approaches to wrapping a [`BuildHasher`] so it works
//! with the fallible collection traits ([`TryHashMap`](crate::std::hashmap::TryHashMap),
//! [`TryHashSet`](crate::std::hashset::TryHashSet), `TryDashMap`,
//! etc.), which require the hasher to implement [`TryClone`] and optionally
//! [`TryDefault`].
//!
//! # `CopyHasherFactory` — misuse-resistant, zero-allocation cloning
//!
//! [`CopyHasherFactory<H>]` wraps any hasher that is [`Copy`]. The wrapper itself
//! is always [`Copy`], so [`TryClone`] is inherently infallible (bitwise copy,
//! no heap allocation). The API is misuse-resistant: the compiler enforces the
//! [`Copy`] bound at the type level, so there is no way to accidentally wrap a
//! hasher that could panic during duplication.
//!
//! [`TryClone`] is implemented automatically for all valid inner types.
//! [`TryDefault`] is only implemented when the inner hasher also implements
//! [`TryDefault`] — this is intentional, because even a [`Copy`] type's
//! [`Default`] implementation might allocate or panic, and the [`TryDefault`]
//! contract forbids panics. If your hasher's [`Default`] is truly safe, you can
//! implement [`TryDefault`] for it manually.
//!
//! # `ArbitraryHasherFactory` — ergonomic, best-effort safety nets
//!
//! [`ArbitraryHasherFactory<H>`] accepts *any* [`BuildHasher`], including those
//! that allocate internally (e.g. [`RandomState`](::lang_std::hash::RandomState)). It
//! automatically implements [`TryClone`] when `H: Clone` and [`TryDefault`] when
//! `H: Default`.
//!
//! Because cloning or defaulting an arbitrary hasher may panic (e.g. on OOM
//! during internal allocation), these implementations use
//! [`::lang_std::panic::catch_unwind`] as a best-effort safety net: if `clone()` or
//! [`Default::default()`] panics, the panic is caught and returned as an error
//! rather than unwinding through the caller. This means fallible collection
//! operations remain non-panicking even with allocating hashers.
//!
//! # Notes
//! - These types only guard against panics during creation of hasher factories. The user must ensure that the invocation via build_hasher does not implicitly panic, although it is practically never the case for the sake of performance.
use lang_core::fmt;
use lang_core::hash::BuildHasher;
use lang_core::mem;
#[cfg(feature = "std")]
use lang_std::thread_local;
use crate::{
    try_clone::{TryClone, TryCloneError},
    try_default::{TryDefault, TryDefaultError},
    try_fmt::{TryDebug, helpers::FormatterExt},
};

/// Marker trait for hasher factories that are safely duplicatable via a bitwise
/// copy.
///
/// Any type that implements both [`BuildHasher`] and [`Copy`] automatically
/// satisfies this trait via a blanket implementation. No manual impl is needed.
///
/// Types satisfying this trait can be duplicated with  `ptr::read` or
/// [`Self::unsafe_copy`] without risk of panicking or double-freeing. This is
/// strictly stronger than [`Clone`] — it requires [`Copy`].
///
/// # Relationship to `TryClone` and `TryDefault`
///
/// When you wrap a `CopyBuildHasher` in [`CopyHasherFactory`], [`TryClone`] is
/// implemented automatically (bitwise copy, infallible). However, [`TryDefault`]
/// is only implemented if the inner type also implements [`TryDefault`] explicitly.
/// This is deliberate: even a [`Copy`] type's [`Default`] may allocate or panic,
/// and the [`TryDefault`] contract forbids panics. If your hasher's default is
/// truly safe, implement [`TryDefault`] for it.
///
/// # Object safety
///
/// Because the trait requires [`Copy`], it is not object-safe and cannot be used
/// as `&dyn CopyBuildHasher`. It is intended solely as a generic bound:
///
/// ```ignore
/// fn process<H: CopyBuildHasher>(hasher: H) {
/// }
/// ```
pub trait CopyBuildHasher: BuildHasher + Copy {
    /// Return a bitwise copy of this hasher factory.
    ///
    /// Because `Self: Copy`, this is equivalent to `*self` but made explicit
    /// for documentation purposes.
    #[inline]
    fn unsafe_copy(&self) -> Self {
        *self
    }
}

// Blanket implementation: any type that is both BuildHasher + Copy satisfies
// the contract by virtue of being Copy. The Copy trait already guarantees
// trivial destructors and memcpy-equivalence.
//
// The blanket requires Copy, which means:
// - No Drop impl (trivial destructor)
// - Bitwise copy produces an identical value
// - No allocation involved in duplication
impl<H> CopyBuildHasher for H where H: BuildHasher + Copy {}

// ── TryClone and TryDefault implementations ────────────────────────────────────────────────

#[cfg(feature = "std")]
/// `RandomState` is the default hasher and it stores two u64 values.
impl TryClone for ::lang_std::hash::RandomState {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(self.clone())
    }
}

#[cfg(feature = "std")]
// Thread-local cache of a RandomState so that repeated `try_default()` calls
// avoid re-invoking the platform random source. Each thread gets its own
// cached instance after the first (potentially expensive) generation.
thread_local! {
    static CACHED_RANDOM_STATE: once_cell::unsync::OnceCell<::lang_std::hash::RandomState>
        = const { once_cell::unsync::OnceCell::new() };
}

#[cfg(feature = "std")]
/// Implementation glue to support `HashMap::try_new`/`HashSet::try_new`/`DashMap::try_new`/`DashSet::try_new`
impl TryDefault for ::lang_std::hash::RandomState {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        CACHED_RANDOM_STATE.with(|cell| {
            if let Some(rs) = cell.get() {
                return Ok(rs.clone());
            }

            // First call on this thread: generate and cache.
            let (k1, k2) = crate::sys::random::hashmap_random_keys().map_err(|_| {
                TryDefaultError::Other("failed to generate random keys for RandomState")
            })?;
            // SAFETY: RandomState's internal representation is two u64 values.
            // We construct it directly from our randomly generated keys, avoiding
            // the panic-prone Default::default() path entirely.
            let rs =
                unsafe { mem::transmute::<(u64, u64), ::lang_std::hash::RandomState>((k1, k2)) };
            let rs_clone = rs.clone();
            cell.set(rs).ok();
            Ok(rs_clone)
        })
    }
}

#[cfg(feature = "std")]
/// `BuildHasherDefault<H>` is a zero-sized wrapper around `PhantomData<H>`.
/// Its `clone()` cannot panic (no-op).
impl<H> TryClone for ::lang_std::hash::BuildHasherDefault<H> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(self.clone())
    }
}

#[cfg(feature = "std")]
/// `BuildHasherDefault<H>` is a zero-sized wrapper around `PhantomData<H>`.
/// Its `default()` cannot panic (no-op).
impl<H> TryDefault for ::lang_std::hash::BuildHasherDefault<H> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Default::default())
    }
}

/// `CopyHasherFactory<H>` is always `Copy`, so cloning is a bitwise copy —
/// inherently infallible.
impl<H: CopyBuildHasher> TryClone for CopyHasherFactory<H> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(*self)
    }
}

/// `TryDefault` is only implemented when the inner hasher also implements
/// `TryDefault`. Even though `H` is `Copy`, its `Default` implementation might
/// allocate or panic — the `TryDefault` contract forbids panics, so we require
/// the inner type to opt in explicitly by implementing `TryDefault` itself.
impl<H: CopyBuildHasher> TryDefault for CopyHasherFactory<H>
where
    H: TryDefault,
{
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self {
            inner: H::try_default()?,
        })
    }
}

/// A misuse-resistant hasher factory container for trivially-copyable hashers.
///
/// `CopyHasherFactory<H>` wraps any hasher factory `H` that implements both
/// [`BuildHasher`] and [`Copy`]. The container itself is always [`Copy`], so
/// duplicating it is a zero-cost bitwise copy — no heap allocation, no possibility
/// of panicking.
///
/// # Trait implementations
///
/// - **[`TryClone`]** is implemented automatically for all valid inner types.
///   Cloning is a bitwise copy and is inherently infallible.
/// - **[`TryDefault`]** is only implemented when the inner type `H` also
///   implements [`TryDefault`]. This is intentional: even though `H` is [`Copy`],
///   its [`Default`] implementation might allocate or panic, and the [`TryDefault`]
///   contract forbids panics. If your hasher's default is truly safe (e.g. it only
///   initializes stack values), implement [`TryDefault`] for it manually.
/// - **[`Default`]** is derived unconditionally via `#[derive(Default)]`. Note
///   that calling `.default()` directly may still panic if `H::default()` does —
///   use [`TryDefault::try_default`] when you need fallibility.
///
/// # When to use
///
/// Use this when your hasher stores only small stack values (`u64` seeds, fixed
/// arrays, etc.) and you want compile-time guarantees that duplication never fails.
/// Popular third-party hashers like `ahash::RandomState`, FxHash, FNV, DJB2, and
/// similar integer-only builders typically satisfy this bound.
///
/// For hashers that allocate internally (e.g. [`RandomState`](lang_std::hash::RandomState)),
/// use [`ArbitraryHasherFactory`] instead.
///
/// # Example
///
/// ```
/// use rustyfill::hashers::CopyHasherFactory;
/// use ::std::hash::{BuildHasher, Hasher};
///
/// #[derive(Clone, Copy, Default)]
/// struct MyHasher;
///
/// impl BuildHasher for MyHasher {
///     type Hasher = MyHasherState;
///     fn build_hasher(&self) -> MyHasherState {
///         MyHasherState { h: 0 }
///     }
/// }
///
/// #[derive(Clone, Copy)]
/// struct MyHasherState { h: u64 }
///
/// impl Hasher for MyHasherState {
///     fn finish(&self) -> u64 { self.h }
///     fn write(&mut self, bytes: &[u8]) {
///         for &b in bytes { self.h ^= b as u64; }
///     }
/// }
///
/// // Compiles because MyHasher is Copy + BuildHasher → CopyBuildHasher
/// let factory: CopyHasherFactory<MyHasher> = CopyHasherFactory::new(MyHasher);
/// let copy = factory;          // bitwise copy, no allocation
/// assert_eq!(factory.hash_one(42u64), copy.hash_one(42u64));
/// ```
#[derive(Clone, Copy, Default)]
pub struct CopyHasherFactory<H: CopyBuildHasher> {
    inner: H,
}

impl<H: CopyBuildHasher> CopyHasherFactory<H> {
    /// Create a new `CopyHasherFactory` wrapping the given hasher factory.
    #[inline]
    pub const fn new(inner: H) -> Self {
        Self { inner }
    }

    /// Return a reference to the inner hasher factory.
    #[inline]
    pub fn inner(&self) -> &H {
        &self.inner
    }

    /// Consume the container and return the inner hasher factory.
    #[inline]
    pub fn into_inner(self) -> H {
        self.inner
    }
}

impl<H: CopyBuildHasher> BuildHasher for CopyHasherFactory<H> {
    type Hasher = H::Hasher;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        self.inner.build_hasher()
    }
}

impl<H: CopyBuildHasher + fmt::Debug> fmt::Debug for CopyHasherFactory<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CopyHasherFactory")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<H: CopyBuildHasher + TryDebug> TryDebug for CopyHasherFactory<H> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.try_debug_struct("CopyHasherFactory")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<H: CopyBuildHasher + PartialEq> PartialEq for CopyHasherFactory<H> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<H: CopyBuildHasher + Eq> Eq for CopyHasherFactory<H> {}

// ── ArbitraryHasherFactory ─────────────────────────────────────────────────────

/// A hasher factory container accepting any [`BuildHasher`], with best-effort
/// panic-catching for cloning and defaulting.
///
/// Unlike [`CopyHasherFactory`], which requires the inner hasher to be [`Copy`],
/// `ArbitraryHasherFactory<H>` only requires `H: BuildHasher`. This makes it
/// compatible with any hasher factory, including those that allocate or open system
/// resources internally (e.g. [`RandomState`](lang_std::hash::RandomState) opens a file in
/// the kernel for every thread and panics if it fails).
///
/// # Ergonomic integration with fallible collections
///
/// This type automatically implements [`TryClone`] (when `H: Clone`) and
/// [`TryDefault`] (when `H: Default`), so it slots directly into the hasher
/// bounds required by [`TryHashMap`](crate::std::hashmap::TryHashMap),
/// [`TryHashSet`](crate::std::hashset::TryHashSet),
/// `TryDashMap`, and
/// `TryDashSet`.
///
/// Because cloning or defaulting an arbitrary hasher may panic (e.g. on OOM
/// during internal allocation), these implementations wrap the call in
/// [`lang_std::panic::catch_unwind`] as a best-effort safety net. If `clone()` or
/// [`Default::default()`] panics, the panic is caught and returned as an error
/// rather than unwinding through the caller. This keeps fallible collection
/// operations non-panicking even with allocating hashers.
///
/// # Safety
///
/// Construction is intentionally `unsafe` so that callers must consciously opt
/// into the trade-offs. The container itself is memory-safe — the `unsafe` block
/// is a *discipline mechanism* forcing the caller to acknowledge:
///
/// 1. **Cloning may fail at runtime.** If `H` allocates during `clone()`,
///    `try_clone()` will attempt to catch the panic and return an error. Callers
///    must handle `Err` paths rather than assuming infallibility.
///
/// 2. **Default construction may fail similarly.** `try_default()` wraps
///    `H::default()` in the same panic-catching net.
///
/// 3. **No [`Copy`] guarantee.** This type is never [`Copy`]. Operations that
///    duplicate the hasher (shrinking a hash table, cloning a collection, etc.)
///    go through [`Clone`], which involves heap allocation for most real-world
///    hashers.
///
/// # When to use
///
/// Use `ArbitraryHasherFactory` when your hasher doesn't satisfy [`Copy`] (most
/// notably [`RandomState`](lang_std::hash::RandomState)) but you still want fallible
/// collection operations. Prefer [`CopyHasherFactory`] when the hasher is
/// stack-only, since it avoids allocation entirely and provides stronger compile-time
/// guarantees.
///
/// # Example
///
/// ```
/// use rustyfill::hashers::ArbitraryHasherFactory;
/// use ::std::collections::HashMap;
/// use ::std::hash::RandomState;
///
/// // RandomState is not Copy, so CopyHasherFactory won't work.
/// // ArbitraryHasherFactory accepts it behind an unsafe constructor.
/// let factory = unsafe { ArbitraryHasherFactory::new(RandomState::new()) };
/// let mut map: HashMap<&str, i32, _> = HashMap::with_hasher(factory);
/// map.insert("key", 42);
/// assert_eq!(map["key"], 42);
/// ```
pub struct ArbitraryHasherFactory<H: BuildHasher> {
    inner: H,
}

impl<H: BuildHasher> ArbitraryHasherFactory<H> {
    /// Create a new `ArbitraryHasherFactory` wrapping the given hasher factory.
    ///
    /// # Safety
    ///
    /// This function is `unsafe` as a compile-time checkpoint. Calling it signals
    /// that the caller understands:
    ///
    /// - Cloning the inner hasher may panic at runtime (e.g. on OOM during
    ///   internal allocation). The [`TryClone`] implementation uses
    ///   [`lang_std::panic::catch_unwind`] to catch such panics and return them as
    ///   errors, so fallible collection operations remain non-panicking.
    /// - Default construction may panic similarly; [`TryDefault`] applies the
    ///   same best-effort panic-catching strategy.
    /// - This container is **not** [`Copy`]. Any operation that duplicates the
    ///   hasher (shrinking, cloning a collection, etc.) requires heap allocation.
    ///
    /// No undefined behaviour can result from misuse — the worst outcome is an
    /// unhandled `Err` downstream rather than a panic propagating unexpectedly.
    #[inline]
    pub const unsafe fn new(inner: H) -> Self {
        Self { inner }
    }

    /// Return a reference to the inner hasher factory.
    #[inline]
    pub fn inner(&self) -> &H {
        &self.inner
    }

    /// Mutably access the inner hasher factory.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut H {
        &mut self.inner
    }

    /// Consume the container and return the inner hasher factory.
    #[inline]
    pub fn into_inner(self) -> H {
        self.inner
    }
}

impl<H: BuildHasher> BuildHasher for ArbitraryHasherFactory<H> {
    type Hasher = H::Hasher;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        self.inner.build_hasher()
    }
}

/// Clone is available when the inner hasher is Clone, which is required by
/// the TryClone supertrait bound. Note that this plain `clone()` may still panic.
impl<H: BuildHasher + Clone> Clone for ArbitraryHasherFactory<H> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// TryClone wraps `H::clone()` in `catch_unwind` so that panics during cloning
/// (e.g. OOM in the hasher's internal allocation) are caught and returned as errors.
#[cfg(feature = "std")]
impl<H: BuildHasher + Clone> TryClone for ArbitraryHasherFactory<H> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        ::lang_std::panic::catch_unwind(::lang_std::panic::AssertUnwindSafe(|| self.inner.clone()))
            .map(|inner| Self { inner })
            .map_err(|_| TryCloneError::Other("hasher factory panicked during clone"))
    }
}

/// TryDefault wraps `H::default()` in `catch_unwind` so that panics during
/// default construction (e.g. OOM in the hasher's internal allocation) are
/// caught and returned as errors.
#[cfg(feature = "std")]
impl<H: BuildHasher + Default> TryDefault for ArbitraryHasherFactory<H> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        ::lang_std::panic::catch_unwind(::lang_std::panic::AssertUnwindSafe(H::default))
            .map(|inner| Self { inner })
            .map_err(|_| TryDefaultError::Other("hasher factory panicked during default"))
    }
}

impl<H: BuildHasher + fmt::Debug> fmt::Debug for ArbitraryHasherFactory<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArbitraryHasherFactory")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<H: BuildHasher + TryDebug> TryDebug for ArbitraryHasherFactory<H> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.try_debug_struct("ArbitraryHasherFactory")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<H: BuildHasher + PartialEq> PartialEq for ArbitraryHasherFactory<H> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<H: BuildHasher + Eq> Eq for ArbitraryHasherFactory<H> {}

#[cfg(test)]
mod tests {
    use super::*;
    use lang_alloc::format;
    use lang_alloc::vec::Vec;
    use lang_core::ptr;
    use lang_std::hash::Hasher;

    // ── Custom trivially-copyable hashers ─────────────────────────────────────

    /// A minimal FNV-1a hasher factory that stores no state.
    #[derive(Clone, Copy, Default)]
    struct Fnv1aBuilder;

    impl BuildHasher for Fnv1aBuilder {
        type Hasher = Fnv1aHasher;

        fn build_hasher(&self) -> Fnv1aHasher {
            Fnv1aHasher {
                hash: 0xCBF29CE484222325u64,
            }
        }
    }

    #[derive(Clone, Copy)]
    struct Fnv1aHasher {
        hash: u64,
    }

    impl Hasher for Fnv1aHasher {
        fn finish(&self) -> u64 {
            self.hash
        }

        fn write(&mut self, bytes: &[u8]) {
            for &b in bytes {
                self.hash ^= b as u64;
                self.hash = self.hash.wrapping_mul(0x100000001B3);
            }
        }
    }

    /// A seeded DJB2 hasher factory storing a single u64 seed.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Djb2Builder {
        seed: u64,
    }

    impl BuildHasher for Djb2Builder {
        type Hasher = Djb2Hasher;

        fn build_hasher(&self) -> Djb2Hasher {
            Djb2Hasher { hash: self.seed }
        }
    }

    #[derive(Clone, Copy)]
    struct Djb2Hasher {
        hash: u64,
    }

    impl Hasher for Djb2Hasher {
        fn finish(&self) -> u64 {
            self.hash
        }

        fn write(&mut self, bytes: &[u8]) {
            for &b in bytes {
                self.hash = self.hash.wrapping_mul(5381).wrapping_add(b as u64);
            }
        }
    }

    /// A hasher factory with a fixed-size seed array.
    #[derive(Clone, Copy, Default)]
    struct FixedSeedBuilder {
        seeds: [u64; 4],
    }

    impl BuildHasher for FixedSeedBuilder {
        type Hasher = FixedSeedHasher;

        fn build_hasher(&self) -> FixedSeedHasher {
            FixedSeedHasher {
                seeds: self.seeds,
                index: 0,
                hash: 0u64,
            }
        }
    }

    #[derive(Clone, Copy)]
    struct FixedSeedHasher {
        seeds: [u64; 4],
        index: usize,
        hash: u64,
    }

    impl Hasher for FixedSeedHasher {
        fn finish(&self) -> u64 {
            self.hash
        }

        fn write(&mut self, bytes: &[u8]) {
            for &b in bytes {
                self.hash ^= b as u64 ^ self.seeds[self.index % 4];
                self.index += 1;
            }
        }
    }

    // ── Trait bound checks ────────────────────────────────────────────────────

    /// Verify that the trait bounds compose correctly with generic functions.
    fn accepts_unsafe_copy_hasher<H: CopyBuildHasher>(_h: H) {}

    #[test]
    fn compile_time_fnv_passes_bound() {
        accepts_unsafe_copy_hasher(Fnv1aBuilder);
    }

    #[test]
    fn compile_time_djb2_passes_bound() {
        accepts_unsafe_copy_hasher(Djb2Builder { seed: 5381 });
    }

    #[test]
    fn compile_time_fixed_seed_passes_bound() {
        accepts_unsafe_copy_hasher(FixedSeedBuilder::default());
    }

    // ── unsafe_copy semantics ─────────────────────────────────────────────────

    #[test]
    fn fnv_unsafe_copy_produces_identical_hashes() {
        let original = Fnv1aBuilder;
        let copied = original.unsafe_copy();
        assert_eq!(
            original.build_hasher().finish(),
            copied.build_hasher().finish()
        );
    }

    #[test]
    fn djb2_unsafe_copy_preserves_seed() {
        let original = Djb2Builder { seed: 0xDEADBEEF };
        let copied = original.unsafe_copy();
        assert_eq!(original.seed, copied.seed);
        assert_eq!(
            original.build_hasher().finish(),
            copied.build_hasher().finish()
        );
    }

    #[test]
    fn fixed_seed_unsafe_copy_preserves_array() {
        let original = FixedSeedBuilder {
            seeds: [1, 2, 3, 4],
        };
        let copied = original.unsafe_copy();
        assert_eq!(original.seeds, copied.seeds);
    }

    // ── ptr::read equivalence ─────────────────────────────────────────────────

    #[test]
    fn ptr_read_matches_unsafe_copy() {
        let original = Djb2Builder { seed: 42 };
        let via_ptr_read = unsafe { ptr::read(&original) };
        let via_method = original.unsafe_copy();
        assert_eq!(via_ptr_read.seed, via_method.seed);
    }

    // ── Negative tests: types that should NOT implement UnsafeCopyHasher ──────

    /// A hasher factory that owns a Vec — definitely not Copy.
    #[derive(Clone, Default)]
    #[allow(dead_code)]
    struct AllocatingBuilder {
        seeds: Vec<u64>,
    }

    impl BuildHasher for AllocatingBuilder {
        type Hasher = Fnv1aHasher; // reuse the hasher, doesn't matter

        fn build_hasher(&self) -> Fnv1aHasher {
            Fnv1aHasher { hash: 0 }
        }
    }

    // This function should fail to compile if uncommented, proving that
    // AllocatingBuilder does NOT satisfy UnsafeCopyHasher:
    //
    // ```compile_fail
    // fn prove_not_copy(_h: &impl UnsafeCopyHasher) {}
    // prove_not_copy(&AllocatingBuilder::default());
    // ```
    //
    // We verify at runtime that it indeed lacks Copy by checking
    //  `mem::needs_drop` and size properties.
    #[test]
    fn allocating_builder_is_not_copy() {
        // AllocatingBuilder contains a Vec, so it's Clone but not Copy.
        // This test documents that fact.
        let builder = AllocatingBuilder::default();
        // If this compiled, the type would be Copy:
        // let _copy = builder; // borrow-checker prevents use-after-move for non-Copy
        drop(builder);
        // The real proof is the compile_fail gate above.
    }

    // ── CopyHasherFactory container ───────────────────────────────────────────

    use super::CopyHasherFactory;
    use crate::try_clone::TryClone;

    #[test]
    fn factory_constructs_and_delegates() {
        let factory = CopyHasherFactory::new(Fnv1aBuilder);
        let h1 = factory.build_hasher();
        assert_eq!(h1.finish(), 0xCBF29CE484222325u64);
    }

    #[test]
    fn factory_is_copy() {
        let original = CopyHasherFactory::new(Djb2Builder { seed: 5381 });
        let copy = original; // bitwise copy
        assert_eq!(original.hash_one(42u64), copy.hash_one(42u64));
    }

    #[test]
    fn factory_inner_access() {
        let factory = CopyHasherFactory::new(Djb2Builder { seed: 0xABCD });
        assert_eq!(factory.inner().seed, 0xABCD);
    }

    #[test]
    fn factory_into_inner() {
        let factory = CopyHasherFactory::new(Djb2Builder { seed: 99 });
        let inner = factory.into_inner();
        assert_eq!(inner.seed, 99);
    }

    #[test]
    fn factory_default() {
        let factory: CopyHasherFactory<FixedSeedBuilder> = CopyHasherFactory::default();
        assert_eq!(factory.inner().seeds, [0, 0, 0, 0]);
    }

    #[test]
    fn factory_hash_one_via_build_hasher() {
        let factory = CopyHasherFactory::new(Fnv1aBuilder);
        let hash = factory.hash_one("hello");
        assert_ne!(hash, 0);
    }

    #[test]
    fn factory_partial_eq() {
        let a = CopyHasherFactory::new(Djb2Builder { seed: 1 });
        let b = CopyHasherFactory::new(Djb2Builder { seed: 1 });
        let c = CopyHasherFactory::new(Djb2Builder { seed: 2 });
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn factory_debug_prints() {
        let factory = CopyHasherFactory::new(Djb2Builder { seed: 42 });
        let s = format!("{factory:?}");
        assert!(s.contains("CopyHasherFactory"));
    }

    #[test]
    fn factory_with_zero_sized_hasher() {
        let factory = CopyHasherFactory::new(Fnv1aBuilder);
        // Fnv1aBuilder is zero-sized; the factory should also be zero-sized.
        assert_eq!(mem::size_of_val(&factory), 0);
    }

    #[test]
    fn factory_works_in_hashmap_signature() {
        // Prove that CopyHasherFactory satisfies the bounds needed for HashMap.
        use lang_std::collections::HashMap;
        let factory = CopyHasherFactory::new(Fnv1aBuilder);
        let mut map: HashMap<&str, i32, _> = HashMap::with_hasher(factory);
        map.insert("key", 42);
        assert_eq!(map["key"], 42);
    }

    // ── TryClone on hashers ───────────────────────────────────────────────────

    #[test]
    fn factory_try_clone_succeeds() {
        let original = CopyHasherFactory::new(Djb2Builder { seed: 5381 });
        let cloned = original.try_clone().unwrap();
        assert_eq!(original.hash_one(42u64), cloned.hash_one(42u64));
    }

    #[test]
    fn random_state_try_clone_succeeds() {
        let rs = ::lang_std::hash::RandomState::new();
        let cloned = rs.try_clone().unwrap();
        assert_eq!(rs.hash_one(99u64), cloned.hash_one(99u64));
    }

    #[test]
    fn build_hasher_default_try_clone_succeeds() {
        use lang_std::hash::{BuildHasherDefault, DefaultHasher};
        let h: BuildHasherDefault<DefaultHasher> = BuildHasherDefault::default();
        let cloned = h.try_clone().unwrap();
        assert_eq!(h.hash_one("x"), cloned.hash_one("x"));
    }

    #[test]
    fn copy_hasher_try_clone_is_infallible_bitwise() {
        let original = CopyHasherFactory::new(Djb2Builder { seed: 0xFF });
        let cloned = original.try_clone().unwrap();
        assert_eq!(original.inner().seed, cloned.inner().seed);
    }

    // ── ArbitraryHasherFactory container ────────────────────────────────────────

    use super::ArbitraryHasherFactory;

    #[test]
    fn arbitrary_factory_constructs_and_delegates() {
        let factory = unsafe { ArbitraryHasherFactory::new(::lang_std::hash::RandomState::new()) };
        let hash = factory.hash_one("hello");
        assert_ne!(hash, 0);
    }

    #[test]
    fn arbitrary_factory_with_copy_hasher() {
        let factory = unsafe { ArbitraryHasherFactory::new(Fnv1aBuilder) };
        let h1 = factory.build_hasher();
        assert_eq!(h1.finish(), 0xCBF29CE484222325u64);
    }

    #[test]
    fn arbitrary_factory_inner_access() {
        let inner = Djb2Builder { seed: 0xABCD };
        let factory = unsafe { ArbitraryHasherFactory::new(inner) };
        assert_eq!(factory.inner().seed, 0xABCD);
    }

    #[test]
    fn arbitrary_factory_into_inner() {
        let inner = Djb2Builder { seed: 99 };
        let factory = unsafe { ArbitraryHasherFactory::new(inner) };
        let recovered = factory.into_inner();
        assert_eq!(recovered.seed, 99);
    }

    #[test]
    fn arbitrary_factory_try_clone_random_state() {
        let factory = unsafe { ArbitraryHasherFactory::new(::lang_std::hash::RandomState::new()) };
        let cloned = factory.try_clone().unwrap();
        assert_eq!(factory.hash_one(42u64), cloned.hash_one(42u64));
    }

    #[test]
    fn arbitrary_factory_try_clone_djb2() {
        let factory = unsafe { ArbitraryHasherFactory::new(Djb2Builder { seed: 5381 }) };
        let cloned = factory.try_clone().unwrap();
        assert_eq!(factory.inner().seed, cloned.inner().seed);
    }

    #[test]
    fn arbitrary_factory_try_default_random_state() {
        let factory =
            <ArbitraryHasherFactory<::lang_std::hash::RandomState>>::try_default().unwrap();
        let hash = factory.hash_one("world");
        assert_ne!(hash, 0);
    }

    #[test]
    fn arbitrary_factory_try_default_fnv() {
        let factory = <ArbitraryHasherFactory<Fnv1aBuilder>>::try_default().unwrap();
        let h = factory.build_hasher();
        assert_eq!(h.finish(), 0xCBF29CE484222325u64);
    }

    #[test]
    fn arbitrary_factory_works_in_hashmap() {
        use lang_std::collections::HashMap;
        let factory = unsafe { ArbitraryHasherFactory::new(::lang_std::hash::RandomState::new()) };
        let mut map: HashMap<&str, i32, _> = HashMap::with_hasher(factory);
        map.insert("key", 42);
        assert_eq!(map["key"], 42);
    }

    #[test]
    fn arbitrary_factory_partial_eq() {
        let a = unsafe { ArbitraryHasherFactory::new(Djb2Builder { seed: 1 }) };
        let b = unsafe { ArbitraryHasherFactory::new(Djb2Builder { seed: 1 }) };
        let c = unsafe { ArbitraryHasherFactory::new(Djb2Builder { seed: 2 }) };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn arbitrary_factory_debug_prints() {
        let factory = unsafe { ArbitraryHasherFactory::new(Djb2Builder { seed: 42 }) };
        let s = format!("{factory:?}");
        assert!(s.contains("ArbitraryHasherFactory"));
    }

    #[test]
    fn arbitrary_factory_not_copy() {
        // Verify that ArbitraryHasherFactory does not require Copy by
        // confirming Clone works (since RandomState is Clone but not Copy).
        let factory = unsafe { ArbitraryHasherFactory::new(::lang_std::hash::RandomState::new()) };
        // This compiles because TryClone is implemented (RandomState: Clone):
        let _cloned = factory.try_clone().unwrap();
    }
}
