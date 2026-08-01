//! Fallible allocation primitives.
//!
//! A standard-library-style crate that provides fallible versions of common
//! allocation types, returning allocation errors on failure instead of
//! panicking.

pub mod alloc;
pub mod arc;
pub mod boxed;
pub mod string;
pub mod try_clone;
pub mod try_default;
pub mod try_to_owned;
pub mod vec;

pub use fallibles_macros::{TryClone, TryDefault};
