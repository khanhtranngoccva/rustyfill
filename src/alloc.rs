//! Custom allocation error type.
//!
//! The [`alloc`](core::alloc) crate's `AllocError` is not exposed on stable Rust,
//! so we provide our own equivalent for use across this library.

use core::fmt;

/// Returned when a heap allocation fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocError;

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "allocation failed")
    }
}

// Not impling `std::error::Error` to stay no_std compatible by default.
