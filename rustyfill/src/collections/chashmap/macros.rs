//! Macro for safely declaring a [`ConcurrentHashMap`](super::ConcurrentHashMap) backed by a static array of shards.

/// Declare a `static` [`ConcurrentHashMap`](super::ConcurrentHashMap) backed by a static compile-time array of shards.
///
/// This macro generates a `static mut` array of [`Shard`](super::Shard) instances wrapped in
/// an `once_cell::sync::Lazy` so that the map is constructed exactly once, at first access,
/// with zero heap allocations. The `from_static` constructor is called inside an `unsafe`
/// block within the lazy initializer.
///
/// The declared item is a `static` that derefs directly to the `ConcurrentHashMap` via
/// `Lazy`'s `Deref` implementation.
///
/// All internal state (the shard array) is encapsulated in a private submodule inside the
/// `Lazy` initializer closure, so only the named `static` is exposed.
///
/// # Usage
///
/// ```ignore
/// use rustyfill::collections::chashmap::{declare_concurrent_hash_map, ConcurrentHashMap};
///
/// // Without custom hasher (uses RandomState::try_new_infallible()):
/// declare_concurrent_hash_map!(pub static MY_MAP: ConcurrentHashMap<u32, String> = 16);
///
/// // With custom hasher:
/// declare_concurrent_hash_map!(pub static MY_MAP: ConcurrentHashMap<u32, String> = 16, hasher = MyHasher);
///
/// fn main() {
///     // Use MY_MAP directly — it derefs to the inner map
///     MY_MAP.try_insert(42, "hello".to_string()).unwrap();
/// }
/// ```
///
/// # Arguments
///
/// - `$name` — the identifier for the generated `static` (conventionally SCREAMING_SNAKE_CASE).
/// - `ConcurrentHashMap<$K, $V>` or `ConcurrentHashMap<$K, $V, $S>` — the full type.
/// - `$shard_count` — the number of shards (must be a power of two >= 2).
/// - Optional: `hasher = $S` — a custom hasher type implementing `BuildHasher + Default`.
///
/// # Safety
///
/// The macro is safe to call. Internally it uses `unsafe` to convert the static mutable array
/// into a slice and pass it to [`ConcurrentHashMap::from_static`](super::ConcurrentHashMap::from_static).
/// The `Lazy` wrapper guarantees single-threaded initialization.
#[macro_export]
macro_rules! declare_concurrent_hash_map {
    // With explicit hasher keyword
    ($vis:vis static $name:ident : ConcurrentHashMap<$K:ty, $V:ty> = $shards:expr, hasher = $S:ty) => {
        $vis static $name: once_cell::sync::Lazy<$crate::collections::chashmap::ConcurrentHashMap<$K, $V, $S>> =
            once_cell::sync::Lazy::new({
                || {
                    mod __inner {
                        use $crate::collections::chashmap::Shard;

                        pub(super) const SHARD_COUNT: usize = $shards;
                        const _: () = assert!(
                            SHARD_COUNT >= 2 && (SHARD_COUNT & (SHARD_COUNT - 1)) == 0,
                            "shard count must be a power of two and >= 2"
                        );

                        #[doc(hidden)]
                        pub(super) static mut __SHARDS: [Shard<$K, $V>; SHARD_COUNT] =
                            [const { Shard::<$K, $V>::new() }; SHARD_COUNT];
                    }

                    unsafe {
                        let ptr = &raw mut __inner::__SHARDS as *mut $crate::collections::chashmap::Shard<$K, $V>;
                        let slice = $crate::lang_core::slice::from_raw_parts_mut(ptr, __inner::SHARD_COUNT);
                        $crate::collections::chashmap::ConcurrentHashMap::from_static(
                            slice,
                            <$S>::default(),
                        )
                    }
                }
            });
    };

    // Without custom hasher (uses RandomState)
    ($vis:vis static $name:ident : ConcurrentHashMap<$K:ty, $V:ty> = $shards:expr) => {
        $vis static $name: once_cell::sync::Lazy<$crate::collections::chashmap::ConcurrentHashMap<$K, $V>> =
            once_cell::sync::Lazy::new({
                || {
                    use $crate::TryRandomState;
                    use $crate::lang_std::hash::RandomState;

                    mod __inner {
                        use $crate::collections::chashmap::Shard;

                        pub(super) const SHARD_COUNT: usize = $shards;
                        const _: () = assert!(
                            SHARD_COUNT >= 2 && (SHARD_COUNT & (SHARD_COUNT - 1)) == 0,
                            "shard count must be a power of two and >= 2"
                        );

                        #[doc(hidden)]
                        pub(super) static mut __SHARDS: [Shard<$K, $V>; SHARD_COUNT] =
                            [const { Shard::<$K, $V>::new() }; SHARD_COUNT];
                    }

                    unsafe {
                        let ptr = &raw mut __inner::__SHARDS as *mut $crate::collections::chashmap::Shard<$K, $V>;
                        let slice = $crate::lang_core::slice::from_raw_parts_mut(ptr, __inner::SHARD_COUNT);
                        $crate::collections::chashmap::ConcurrentHashMap::from_static(
                            slice,
                            RandomState::try_new_infallible(),
                        )
                    }
                }
            });
    };

    // With custom hasher type parameter in generic signature
    ($vis:vis static $name:ident : ConcurrentHashMap<$K:ty, $V:ty, $S:ty> = $shards:expr) => {
        $vis static $name: once_cell::sync::Lazy<$crate::collections::chashmap::ConcurrentHashMap<$K, $V, $S>> =
            once_cell::sync::Lazy::new({
                || {
                    mod __inner {
                        use $crate::collections::chashmap::Shard;

                        pub(super) const SHARD_COUNT: usize = $shards;
                        const _: () = assert!(
                            SHARD_COUNT >= 2 && (SHARD_COUNT & (SHARD_COUNT - 1)) == 0,
                            "shard count must be a power of two and >= 2"
                        );

                        #[doc(hidden)]
                        pub(super) static mut __SHARDS: [Shard<$K, $V>; SHARD_COUNT] =
                            [const { Shard::<$K, $V>::new() }; SHARD_COUNT];
                    }

                    unsafe {
                        let ptr = &raw mut __inner::__SHARDS as *mut $crate::collections::chashmap::Shard<$K, $V>;
                        let slice = $crate::lang_core::slice::from_raw_parts_mut(ptr, __inner::SHARD_COUNT);
                        $crate::collections::chashmap::ConcurrentHashMap::from_static(
                            slice,
                            <$S>::default(),
                        )
                    }
                }
            });
    };
}
