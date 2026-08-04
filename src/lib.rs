//! Fallible allocation polyfills.
//!
//! A standard-library-style crate that provides fallible versions of common
//! allocation types, returning allocation errors on failure instead of
//! panicking.

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
mod random;
pub mod string;
pub mod try_clone;
pub mod try_default;
pub mod try_to_owned;
pub mod vec;
pub mod vecdeque;

pub use fallibles_macros::{TryClone, TryDefault};
