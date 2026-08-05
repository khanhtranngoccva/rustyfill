//! Fallible random data generation.
//!
//! Provides [`fill_bytes`] for filling a buffer with random bytes, and
//! [`hashmap_random_keys_infallible`] for generating two randomized u64 seeds
//! suitable for hashmap seeding that never fails (falls back to MT19937).
//!
//! Both the platform-specific backend functions and the public API return
//! [`Result`] instead of panicking.

pub mod uefi;

use core::fmt;

// ── Error Type ────────────────────────────────────────────────────────────────────

/// Error returned when random data generation fails.
#[derive(Debug)]
pub enum RandomError {
    /// The underlying syscall returned an error code.
    Syscall(i32),
    /// A platform-specific failure with a diagnostic message.
    Platform(String),
    #[allow(unused)]
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

// ── Backend modules ──────────────────────────────────────────────────────────────
// Most backends are cfg-gated and only active on their respective platforms.
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
        // FIXME(horizon): add arc4random_buf to shim-3ds
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
        // FIXME: finally remove std support for wasm32-unknown-unknown
        // FIXME: add random data generation to xous
        mod unsupported;
        pub use unsupported::{fill_bytes, hashmap_random_keys};
    }
    _ => {}
}

// ── Stack-based Mersenne Twister (MT19937) ───────────────────────────────────────
#[allow(dead_code)]
/// Infallible, stack-allocated Mersenne Twister PRNG.
///
/// Used as a last-resort fallback for [`hashmap_random_keys_infallible`] when
/// all OS random sources fail. Seeded from stack addresses so it produces
/// varying output across invocations. Not cryptographically secure, but
/// sufficient for hashmap seed diversity.
struct Mt19937 {
    mt: [u32; 624],
    index: usize,
}

impl Mt19937 {
    /// Creates a new MT19937 seeded from a single u32 value.
    fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed & 0xFFFFFFFF;
        for i in 1..624 {
            state[i] = state[i - 1].wrapping_mul(1812433253)
                ^ (state[i - 1] >> 30).wrapping_mul(1812433253)
                ^ i as u32;
        }
        Self {
            mt: state,
            index: 624,
        }
    }

    /// Generates the next u32 from the generator.
    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.regenerate();
        }

        let mut y = self.mt[self.index];
        y ^= y >> 11;
        y ^= y << 7 & 0x9D2C5680;
        y ^= y << 15 & 0xEFC60000;
        y ^= y >> 18;

        self.index += 1;
        y
    }

    fn regenerate(&mut self) {
        for i in 0..624 {
            let _mag_val = self.mt[(i + 397) % 624] & 0x7FFFFFFF;
            let mk = ((self.mt[i] & 0x80000000) | (self.mt[(i + 1) % 624] & 0x7FFFFFFF)) >> 1;
            self.mt[i] =
                self.mt[(i + 397) % 624] ^ (mk >> 1) ^ [0x0, 0x9908B0DF][(mk & 1) as usize];
        }
        self.index = 0;
    }
}

/// Generate hashmap keys using MT19937 as a last resort.
#[allow(dead_code)]
fn hashmap_random_keys_mt() -> (u64, u64) {
    let stack_addr = (&0u8 as *const u8) as u32;
    let seed = stack_addr.wrapping_mul(2654435761);
    let mut mt = Mt19937::new(seed);
    let k1 = (mt.next_u32() as u64) << 32 | mt.next_u32() as u64;
    let k2 = (mt.next_u32() as u64) << 32 | mt.next_u32() as u64;
    (k1, k2)
}

// ── Public API ───────────────────────────────────────────────────────────────────

/// Fill the provided buffer with cryptographically secure random bytes.
///
/// Returns [`Err`] if the platform random source is unavailable or fails.
/// Does **not** fall back to a PRNG.
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
pub fn _hashmap_random_keys() -> Result<(u64, u64), RandomError> {
    hashmap_random_keys()
}

/// Generate two randomized u64 values suitable for seeding a hashmap.
///
/// Attempts the platform random source first. If it fails, falls back to
/// an infallible stack-based Mersenne Twister seeded from stack addresses.
/// This guarantees the function never returns an error, allowing infallible
/// creation of hashmaps even when the OS random source is unavailable.
#[allow(dead_code)]
pub fn hashmap_random_keys_infallible() -> (u64, u64) {
    match hashmap_random_keys() {
        Ok(keys) => keys,
        Err(_) => hashmap_random_keys_mt(),
    }
}

// Default hashmap_random_keys for platforms without a native insecure path:
// uses the same secure source (good enough for hashmap seeds).
#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    all(target_family = "wasm", target_os = "unknown"),
    all(target_os = "wasi", not(target_env = "p1")),
    target_os = "xous",
    target_os = "vexos",
)))]
fn hashmap_random_keys() -> Result<(u64, u64), RandomError> {
    let mut buf = [0; 16];
    backend_fill_bytes(&mut buf)?;
    let k1 = u64::from_ne_bytes(buf[..8].try_into().unwrap());
    let k2 = u64::from_ne_bytes(buf[8..].try_into().unwrap());
    Ok((k1, k2))
}

