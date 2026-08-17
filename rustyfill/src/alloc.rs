//! Custom allocation types and allocation errors.
//!
//! The [`alloc`](lang_alloc) (alloc) crate's `AllocError` is not exposed on stable Rust,
//! so we provide our own equivalent for use across this library. We also provide
//! [`TryReserveError`], a unified polyfill for capacity-reservation failures
//! across different collection backends.

use crate::try_fmt::{TryDebug, helpers::FormatterExt};
use lang_alloc::collections;
use lang_core::alloc::Layout;
use lang_core::error;
use lang_core::fmt::{self, Debug};

pub mod arc;
pub mod boxed;
#[cfg(all(feature = "std", feature = "btree-entry"))]
pub mod btrees;
pub mod ffi;
pub mod rc;
pub mod string;
pub mod vec;
pub mod vecdeque;

/// Polyfill allocation error returned when a heap allocation fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocError {
    pub(crate) layout: Layout,
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "allocation failed")
    }
}

impl TryDebug for AllocError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.try_debug_struct("AllocError")
            .field("layout", &self.layout)
            .finish()
    }
}

impl error::Error for AllocError {}

/// Unified error for fallible capacity reservation.
///
/// Different collection types return different reserve-error types:
/// standard collections expose [`lang_alloc::collections::TryReserveError`] which carries
/// diagnostic information, while third-party collections like `dashmap` provide an
/// empty non-exhaustive struct with no usable fields. This enum unifies both cases
/// so that error types across this crate can use a single `Reserve` variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TryReserveError {
    /// The underlying collection provided a [`lang_alloc::collections::TryReserveError`]
    /// with diagnostic details about the failed allocation. Only available when
    /// the `std` feature is enabled.
    Std(collections::TryReserveError),
    /// The underlying collection provided no diagnostic information.
    Other,
}

impl fmt::Display for TryReserveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Std(e) => write!(f, "{}", e),
            Self::Other => write!(f, "capacity reservation failed"),
        }
    }
}

impl TryDebug for lang_alloc::collections::TryReserveError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self, f)
    }
}

impl TryDebug for TryReserveError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Std(e) => f
                .try_debug_struct("TryReserveError::Std")
                .field("0", e)
                .finish(),
            Self::Other => f.write_str("TryReserveError::Other"),
        }
    }
}

impl From<collections::TryReserveError> for TryReserveError {
    fn from(e: collections::TryReserveError) -> Self {
        Self::Std(e)
    }
}

#[cfg(feature = "unstable")]
impl From<dashmap::TryReserveError> for TryReserveError {
    fn from(_e: dashmap::TryReserveError) -> Self {
        Self::Other
    }
}

impl error::Error for TryReserveError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Std(e) => Some(e),
            Self::Other => None,
        }
    }
}
