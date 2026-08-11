//! Fallible allocation polyfills.
//!
//! A standard-library-style crate that provides fallible versions of common
//! allocation types, returning allocation errors on failure instead of
//! panicking.

// Enable unstable UEFI std and const error library features when building for UEFI on nightly.
// Both cfg flags (`nightly_compiler`) are set by build.rs.
#![cfg_attr(all(nightly_compiler = "true", target_os = "uefi"), feature(uefi_std))]
#![cfg_attr(
    all(nightly_compiler = "true", target_os = "uefi"),
    feature(io_const_error)
)]

// Allow the `try_format_args!` proc-macro to reference `::rustyfill::try_fmt::*Wrapper`
// types from within this crate itself. Without this, `::rustyfill` doesn't resolve
// during self-compilation since a crate can't refer to itself by name from inside.
extern crate self as rustyfill;

pub mod alloc;
pub mod collections;
#[cfg(feature = "unstable")]
pub mod dashmap;
pub mod hashers;
pub mod prelude;
pub mod recovery;
pub mod std;
mod sys;
pub mod try_clone;
pub mod try_default;
pub mod try_fmt;
pub mod try_random_state;
pub mod try_to_owned;

// Re-export std submodules at crate root for backward compatibility
pub use std::arc;
pub use std::boxed;
#[cfg(feature = "panic")]
pub use std::btrees;
pub use std::cell;
pub use std::ffi;
pub use std::hashmap;
pub use std::hashset;
pub use std::path;
pub use std::rc;
pub use std::string;
pub use std::sync;
pub use std::vec;
pub use std::vecdeque;

pub use rustyfill_macros::{TryClone, TryDebug, TryDefault};
pub use rustyfill_macros::{
    try_format, try_format_args, try_print, try_println, try_write, try_writeln,
};
pub use try_random_state::TryRandomState;
