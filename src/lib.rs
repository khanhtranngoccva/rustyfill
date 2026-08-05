//! Fallible allocation polyfills.
//!
//! A standard-library-style crate that provides fallible versions of common
//! allocation types, returning allocation errors on failure instead of
//! panicking.

// Enable unstable UEFI std and const error library features when building for UEFI on nightly.
// Both cfg flags (`nightly_compiler` and `target_os`) are set by build.rs.
#![cfg_attr(all(nightly_compiler = "true", target_os = "uefi"), feature(uefi_std))]
#![cfg_attr(
    all(nightly_compiler = "true", target_os = "uefi"),
    feature(io_const_error)
)]

pub mod alloc;
pub mod arc;
pub mod boxed;
pub mod btrees;
pub mod dashmap;
pub mod ffi;
pub mod hashers;
pub mod hashmap;
pub mod hashset;
pub mod path;
pub mod prelude;
pub mod recovery;
pub mod string;
mod sys;
pub mod try_clone;
pub mod try_default;
pub mod try_random_state;
pub mod try_to_owned;
pub mod vec;
pub mod vecdeque;

pub use fallibles_macros::{TryClone, TryDefault};
pub use try_random_state::TryRandomState;
