// qnx was refactored to nto; kept for backward compatibility.
// https://doc.rust-lang.org/rustc/platform-support/nto-qnx.html
#![allow(unexpected_cfgs, reason = "qnx refactored to nto")]

//! Fallible random data generation.
//!
//! Provides [`fill_bytes`] for filling a buffer with random bytes, and
//! [`hashmap_random_keys_infallible`] for generating two randomized u64 seeds
//! suitable for hashmap seeding that never fails (falls back to MT19937).
//!
//! Both the platform-specific backend functions and the public API return
//! [`Result`] instead of panicking.

use crate::lang_alloc::borrow::Cow;
use core::fmt;

use crate::try_fmt::{TryDebug, helpers::FormatterExt};

// ── Error Type ────────────────────────────────────────────────────────────────────

/// Error returned when random data generation fails.
#[derive(Debug)]
#[allow(unused)]
pub enum RandomError {
    /// The underlying syscall returned an error code.
    Syscall(i32),
    /// A platform-specific failure with a static diagnostic message.
    Platform(Cow<'static, str>),
    /// This target has no supported random data source.
    Unsupported,
}

impl fmt::Display for RandomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syscall(code) => write!(f, "random syscall failed with code {}", code),
            Self::Platform(msg) => write!(f, "platform random source failed: {}", msg),
            Self::Unsupported => write!(f, "this target does not support random data generation"),
        }
    }
}

impl TryDebug for RandomError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syscall(code) => f.try_debug_struct("RandomError::Syscall")
                .field("0", code)
                .finish(),
            Self::Platform(msg) => f.try_debug_struct("RandomError::Platform")
                .field("0", msg)
                .finish(),
            Self::Unsupported => f.write_str("RandomError::Unsupported"),
        }
    }
}

// ── Backend modules ──────────────────────────────────────────────────────────────
// Most backends are cfg-gated and only active on their respective platforms.
// All backends require the `std` feature (they depend on libc or platform crates).
#[cfg(feature = "std")]
cfg_select! {
    // Tier 1
    any(target_os = "linux", target_os = "android") => {
        mod linux;
        pub use linux::{fill_bytes, hashmap_random_keys};
    }
    target_os = "windows" => {
        mod windows;
        pub use windows::fill_bytes;
    }
    target_vendor = "apple" => {
        mod apple;
        pub use apple::fill_bytes;
    // Others, in alphabetical ordering.
    }
    any(
        target_os = "dragonfly",
        target_os = "freebsd",
        target_os = "haiku",
        target_os = "illumos",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "rtems",
        target_os = "solaris",
        target_os = "vita",
        target_os = "nuttx",
    ) => {
        mod arc4random;
        pub use arc4random::fill_bytes;
    }
    target_os = "emscripten" => {
        mod getentropy;
        pub use getentropy::fill_bytes;
    }
    target_os = "espidf" => {
        mod espidf;
        pub use espidf::fill_bytes;
    }
    target_os = "fuchsia" => {
        mod fuchsia;
        pub use fuchsia::fill_bytes;
    }
    target_os = "hermit" => {
        mod hermit;
        pub use hermit::fill_bytes;
    }
    any(target_os = "horizon", target_os = "cygwin") => {
        // FIXME-OLD(horizon): add arc4random_buf to shim-3ds
        mod getrandom;
        pub use getrandom::fill_bytes;
    }
    any(
        target_os = "aix",
        target_os = "hurd",
        target_os = "l4re",
        target_os = "nto",
        target_os = "qnx",
    ) => {
        mod unix_legacy;
        pub use unix_legacy::fill_bytes;
    }
    target_os = "redox" => {
        mod redox;
        pub use redox::fill_bytes;
    }
    target_os = "motor" => {
        mod motor;
        pub use motor::fill_bytes;
    }
    all(target_vendor = "fortanix", target_env = "sgx") => {
        mod sgx;
        pub use sgx::fill_bytes;
    }
    target_os = "solid_asp3" => {
        mod solid;
        pub use solid::fill_bytes;
    }
    target_os = "teeos" => {
        mod teeos;
        pub use teeos::fill_bytes;
    }
    target_os = "trusty" => {
        mod trusty;
        pub use trusty::fill_bytes;
    }
    target_os = "uefi" => {
        mod uefi;
        mod uefi_helpers;
        pub use uefi::fill_bytes;
    }
    target_os = "vxworks" => {
        mod vxworks;
        pub use vxworks::fill_bytes;
    }
    all(target_os = "wasi", target_env = "p1") => {
        mod wasip1;
        pub use wasip1::fill_bytes;
    }
    all(target_os = "wasi", any(target_env = "p2", target_env = "p3")) => {
        mod wasi;
        pub use wasi::{fill_bytes, hashmap_random_keys};
    }
    target_os = "zkvm" => {
        mod zkvm;
        pub use zkvm::fill_bytes;
    }
    any(
        all(target_family = "wasm", target_os = "unknown"),
        target_os = "xous",
        target_os = "vexos",
    ) => {
        // FIXME-OLD: finally remove std support for wasm32-unknown-unknown
        // FIXME-OLD: add random data generation to xous
        mod unsupported;
        pub use unsupported::{fill_bytes, hashmap_random_keys};
    }
    _ => {}
}

// ── SplitMix64 fallback ──────────────────────────────────────────────────────────
/// Inline SplitMix64 PRNG — zero dependencies, zero allocation.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Generate hashmap keys using SplitMix64 as a last resort.
///
/// Used by [`hashmap_random_keys_infallible`] and unsupported-platform backends
/// when no OS random source is available. Seeded from a stack address so it
/// produces varying output across invocations. Not cryptographically secure,
/// but sufficient for hashmap seed diversity.
pub(crate) fn hashmap_random_keys_mt() -> (u64, u64) {
    let stack_addr = (&0u8 as *const u8) as u64;
    let mut state = stack_addr.wrapping_mul(2654435761u64);
    let k1 = splitmix64(&mut state);
    let k2 = splitmix64(&mut state);
    (k1, k2)
}

// ── Public API ───────────────────────────────────────────────────────────────────

/// Fill the provided buffer with cryptographically secure random bytes.
///
/// Returns [`Err`] if the platform random source is unavailable or fails.
/// Does **not** fall back to a PRNG.
#[cfg(feature = "std")]
#[allow(dead_code)]
pub fn _fill_bytes(bytes: &mut [u8]) -> Result<(), RandomError> {
    fill_bytes(bytes)
}

/// Generate two randomized u64 values suitable for seeding a hashmap.
///
/// Returns [`Err`] if the platform random source is unavailable or fails.
/// Does **not** fall back to a PRNG — use [`hashmap_random_keys_infallible`]
/// for guaranteed output. Suitable for [`TryDefault`](crate::try_default::TryDefault)
/// implementations that must propagate failures rather than hide them.
#[cfg(feature = "std")]
pub fn _hashmap_random_keys() -> Result<(u64, u64), RandomError> {
    hashmap_random_keys()
}

/// Generate two randomized u64 values suitable for seeding a hashmap.
///
/// When the `std` feature is enabled, attempts the platform random source first.
/// If it fails (or `std` is disabled), falls back to an infallible SplitMix64 PRNG
/// seeded from stack addresses. This guarantees the function never returns an error,
/// allowing infallible creation of hashmaps even when the OS random source is unavailable.
#[allow(dead_code)]
pub fn hashmap_random_keys_infallible() -> (u64, u64) {
    #[cfg(feature = "std")]
    {
        if let Ok(keys) = hashmap_random_keys() {
            return keys;
        }
    }
    hashmap_random_keys_mt()
}

// Default hashmap_random_keys for platforms without a native insecure path:
// uses the same secure source (good enough for hashmap seeds).
#[cfg(all(feature = "std", not(any(
    target_os = "linux",
    target_os = "android",
    all(target_family = "wasm", target_os = "unknown"),
    all(target_os = "wasi", not(target_env = "p1")),
    target_os = "xous",
    target_os = "vexos",
))))]
pub fn hashmap_random_keys() -> Result<(u64, u64), RandomError> {
    let mut buf = [0; 16];
    fill_bytes(&mut buf)?;
    let k1 = u64::from_ne_bytes(buf[..8].try_into().unwrap());
    let k2 = u64::from_ne_bytes(buf[8..].try_into().unwrap());
    Ok((k1, k2))
}
