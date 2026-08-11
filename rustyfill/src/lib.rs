//! Fallible allocation polyfills.
//!
//! A standard-library-style crate that provides fallible versions of common
//! allocation types, returning allocation errors on failure instead of
//! panicking.
//!
//! # `no_std` support
//!
//! This crate is `no_std` compatible when the `std` feature is disabled.
//! The `std` feature is enabled by default. When disabled, the crate still
//! requires the `alloc` crate (for `Box`, `Vec`, `String`, etc.) but does not
//! depend on `std`. Modules that require `std` (error wrappers, path/FFI types,
//! `RandomState` helpers, etc.) are gated behind `#[cfg(feature = "std")]`.

// Enable unstable UEFI std and const error library features when building for UEFI on nightly.
// Both cfg flags (`nightly_compiler`) are set by build.rs.
#![cfg_attr(all(nightly_compiler = "true", target_os = "uefi"), feature(uefi_std))]
#![cfg_attr(
    all(nightly_compiler = "true", target_os = "uefi"),
    feature(io_const_error)
)]
#![no_std]

// Register `std` and `alloc` under alias names so they're accessible via absolute
// paths (`::lang_std`, `::lang_alloc`) without clashing with our own `pub mod std`
// and `pub mod alloc` modules that re-export fallible wrappers. Made `pub` so that
// `#[macro_export]` macros in downstream crates can reference them via `$crate::lang_std`.
pub extern crate alloc as lang_alloc;
#[cfg(feature = "std")]
pub extern crate std as lang_std;

// Allow the `try_format_args!` proc-macro to reference `::rustyfill::try_fmt::*Wrapper`
// types from within this crate itself. Without this, `::rustyfill` doesn't resolve
// during self-compilation since a crate can't refer to itself by name from inside.
extern crate self as rustyfill;

pub mod alloc;
pub mod collections;
#[cfg(feature = "unstable")]
pub mod dashmap;
#[cfg(feature = "std")]
pub mod errors;
pub mod hashers;
pub mod prelude;
pub mod recovery;
#[cfg(feature = "std")]
pub mod std;
mod sys;
pub mod try_clone;
pub mod try_default;
pub mod try_fmt;
#[cfg(feature = "std")]
pub mod try_random_state;
pub mod try_to_owned;

// Re-export std submodules at crate root for backward compatibility
#[cfg(feature = "std")]
pub use crate::std::arc;
#[cfg(feature = "std")]
pub use crate::std::boxed;
#[cfg(all(feature = "std", feature = "panic"))]
pub use crate::std::btrees;
#[cfg(feature = "std")]
pub use crate::std::cell;
#[cfg(feature = "std")]
pub use crate::std::ffi;
#[cfg(feature = "std")]
pub use crate::std::hashmap;
#[cfg(feature = "std")]
pub use crate::std::hashset;
#[cfg(feature = "std")]
pub use crate::std::path;
#[cfg(feature = "std")]
pub use crate::std::rc;
#[cfg(feature = "std")]
pub use crate::std::string;
#[cfg(feature = "std")]
pub use crate::std::sync;
#[cfg(feature = "std")]
pub use crate::std::vec;
#[cfg(feature = "std")]
pub use crate::std::vecdeque;

pub use rustyfill_macros::{TryClone, TryDebug, TryDefault};
pub use rustyfill_macros::{
    try_format, try_format_args, try_print, try_println, try_write, try_writeln,
};
#[cfg(feature = "std")]
pub use try_random_state::TryRandomState;
