//! Infallible construction of [`RandomState`] for use as a hasher factory.
//!
//! [`RandomState::new()`] and [`RandomState::default()`] may panic on first use
//! in a new thread if the OS random source is unavailable (they rely on
//! thread-local seeding). This module provides the [`TryRandomState`] extension
//! trait with methods that **never panic**, falling back to an infallible
//! SplitMix64 PRNG when the platform random source fails.
//!
//! # When to use
//!
//! Use this when you need to construct a [`RandomState`] in a context where
//! panics are unacceptable (e.g. inside `#[no_panic]` functions, during early
//! bootstrapping, or in safety-critical code paths). The generated seeds are
//! sufficient for hashmap diversity — they prevent hash-flooding attacks in
//! typical usage — though they lack cryptographic quality when the PRNG fallback
//! is triggered.
//!
//! For fallible construction that propagates errors instead of falling back, see
//! [`TryDefault`](crate::try_default::TryDefault) which is already implemented
//! for [`RandomState`] in [`crate::hashers`].

use lang_core::mem;
use lang_std::hash::RandomState;
use crate::sys::random::hashmap_random_keys_infallible;

/// Extension trait for infallible [`RandomState`] construction.
///
/// Provides methods that guarantee a valid [`RandomState`] is returned without
/// ever panicking, even when the operating system's random number generator is
/// unavailable or malfunctioning.
///
/// # Example
///
/// ```
/// use rustyfill::TryRandomState;
/// use ::std::hash::RandomState;
/// use ::std::collections::HashMap;
///
/// // Infallible: never panics, even if /dev/urandom is unreadable.
/// let state = RandomState::try_new_infallible();
/// let mut map: HashMap<&str, i32> = HashMap::with_hasher(state);
/// map.insert("hello", 42);
/// assert_eq!(map["hello"], 42);
/// ```
pub trait TryRandomState {
    /// Construct a new [`RandomState`] with randomized seeds, guaranteed to
    /// succeed.
    ///
    /// Attempts to read from the platform's random source first. If that fails,
    /// falls back to a SplitMix64 PRNG seeded from stack addresses. This ensures
    /// the function never panics, making it safe to call in contexts where
    /// unwinding would be catastrophic.
    ///
    /// The resulting [`RandomState`] produces diverse bucket distributions for
    /// hash maps and sets, protecting against algorithmic complexity attacks in
    /// normal circumstances.
    fn try_new_infallible() -> Self;
}

impl TryRandomState for RandomState {
    #[inline]
    fn try_new_infallible() -> Self {
        let (k1, k2) = hashmap_random_keys_infallible();
        // SAFETY: RandomState's internal representation is two u64 values.
        // We construct it directly from our randomly generated keys, avoiding
        // the panic-prone Default::default() path entirely.
        unsafe { mem::transmute::<(u64, u64), RandomState>((k1, k2)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang_alloc::vec::Vec;
    use lang_std::collections::{HashMap, HashSet};
    use lang_std::hash::{BuildHasher, Hasher};

    #[test]
    fn try_new_infallible_produces_valid_state() {
        let state = RandomState::try_new_infallible();
        let mut hasher = state.build_hasher();
        hasher.write(b"hello");
        let h = hasher.finish();
        assert_ne!(h, 0);
    }

    #[test]
    fn try_new_infallible_produces_different_seeds() {
        // Multiple calls should produce different RandomStates (different seeds).
        // This relies on the fact that each invocation gets a fresh set of keys.
        let s1 = RandomState::try_new_infallible();
        let s2 = RandomState::try_new_infallible();

        // They should hash the same input differently (probabilistic, but extremely
        // unlikely to collide given 128-bit seed space).
        assert_ne!(s1.hash_one(42u64), s2.hash_one(42u64));
    }

    #[test]
    fn try_new_infallible_works_with_hashmap() {
        let state = RandomState::try_new_infallible();
        let mut map: HashMap<&str, i32> =
            HashMap::with_hasher(state);
        map.insert("alpha", 1);
        map.insert("beta", 2);
        assert_eq!(map["alpha"], 1);
        assert_eq!(map["beta"], 2);
    }

    #[test]
    fn try_new_infallible_works_with_hashset() {
        let state = RandomState::try_new_infallible();
        let mut set: HashSet<i32> = HashSet::with_hasher(state);
        set.insert(1);
        set.insert(2);
        set.insert(3);
        assert!(set.contains(&2));
        assert!(!set.contains(&4));
    }

    #[test]
    fn try_new_infallible_consistent_hashing() {
        // Same RandomState must produce consistent hashes across multiple calls.
        let state = RandomState::try_new_infallible();
        let h1 = state.hash_one("consistent");
        let h2 = state.hash_one("consistent");
        assert_eq!(h1, h2);
    }

    #[test]
    fn try_new_infallible_many_iterations() {
        // Stress test: create many RandomStates rapidly.
        let mut states = Vec::with_capacity(100);
        for _ in 0..100 {
            states.push(RandomState::try_new_infallible());
        }
        // All should produce non-zero hashes for the same input.
        for state in &states {
            assert_ne!(state.hash_one(999u64), 0);
        }
    }

    #[test]
    fn try_new_infallible_trait_object_safe_usage() {
        // Verify the trait can be used generically.
        fn takes_try_random_state<T: TryRandomState>() -> T {
            T::try_new_infallible()
        }

        let state = takes_try_random_state::<RandomState>();
        assert_ne!(state.hash_one("generic"), 0);
    }

    #[test]
    fn try_new_infallible_stress_diversity() {
        // Check that consecutive calls yield diverse seeds by hashing the same value.
        let hashes: HashSet<u64> = (0..50)
            .map(|_| RandomState::try_new_infallible().hash_one(12345u64))
            .collect();

        // With 128-bit seeds, we should see nearly unique hashes.
        // Allow at most 2 collisions out of 50 trials.
        assert!(
            hashes.len() >= 48,
            "expected at least 48 unique hashes, got {}",
            hashes.len()
        );
    }
}
