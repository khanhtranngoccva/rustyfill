//! Custom allocation types, allocation errors, and panic payload wrapper.
//!
//! The [`alloc`](lang_alloc) (alloc) crate's `AllocError` is not exposed on stable Rust,
//! so we provide our own equivalent for use across this library. We also provide
//! [`PayloadBox`], an owning wrapper around the raw panic payload from
//! `catch_unwind`, and [`TryReserveError`], a unified polyfill for
//! capacity-reservation failures across different collection backends.

#[cfg(feature = "std")]
use lang_alloc::borrow::Cow;
#[cfg(feature = "std")]
use lang_alloc::boxed::Box;
use lang_core::alloc::Layout;
use lang_core::any;
use lang_core::fmt;
#[cfg(feature = "std")]
use crate::try_fmt::AssertDebug;
use crate::try_fmt::{TryDebug, helpers::FormatterExt};

pub mod arc;
pub mod boxed;
#[cfg(feature = "panic")]
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

// Not impling `::lang_std::error::Error` to stay no_std compatible by default.

impl TryDebug for AllocError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.try_debug_struct("AllocError")
            .field("layout", &self.layout)
            .finish()
    }
}

/// Unified error for fallible capacity reservation.
///
/// Different collection types return different reserve-error types:
/// standard collections expose [`lang_std::collections::TryReserveError`] which carries
/// diagnostic information, while third-party collections like `dashmap` provide an
/// empty non-exhaustive struct with no usable fields. This enum unifies both cases
/// so that error types across this crate can use a single `Reserve` variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TryReserveError {
    /// The underlying collection provided a [`lang_std::collections::TryReserveError`]
    /// with diagnostic details about the failed allocation. Only available when
    /// the `std` feature is enabled.
    #[cfg(feature = "std")]
    Std(::lang_std::collections::TryReserveError),
    /// The underlying collection provided no diagnostic information.
    Other,
}

impl fmt::Display for TryReserveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "std")]
            Self::Std(e) => write!(f, "{}", e),
            Self::Other => write!(f, "capacity reservation failed"),
        }
    }
}

impl TryDebug for TryReserveError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "std")]
            Self::Std(e) => f
                .try_debug_struct("TryReserveError::Std")
                .field("0", e)
                .finish(),
            Self::Other => f.write_str("TryReserveError::Other"),
        }
    }
}

#[cfg(feature = "std")]
impl From<::lang_std::collections::TryReserveError> for TryReserveError {
    fn from(e: ::lang_std::collections::TryReserveError) -> Self {
        Self::Std(e)
    }
}

#[cfg(feature = "unstable")]
impl From<dashmap::TryReserveError> for TryReserveError {
    fn from(_e: dashmap::TryReserveError) -> Self {
        Self::Other
    }
}

#[cfg(feature = "std")]
impl ::lang_std::error::Error for TryReserveError {
    fn source(&self) -> Option<&(dyn ::lang_std::error::Error + 'static)> {
        match self {
            Self::Std(e) => Some(e),
            Self::Other => None,
        }
    }
}

/// Owning wrapper around the raw panic payload from [`lang_std::panic::catch_unwind`].
///
/// Holds the `Box<dyn Any + Send>` verbatim so that constructing the error
/// after catching a panic performs zero additional allocations.
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct PayloadBox(pub Box<dyn any::Any + Send>);

#[cfg(feature = "std")]
impl PayloadBox {
    /// Extract a human-readable message from the payload.
    ///
    /// May allocate if the payload contains a `&str` or `String`.
    /// Callers should only invoke this outside of tight/OOM-sensitive paths
    /// (e.g. during logging or display formatting).
    pub fn message(&self) -> Cow<'_, str> {
        if let Some(s) = self.0.downcast_ref::<&str>() {
            Cow::Borrowed(*s)
        } else if let Some(s) = self.0.downcast_ref::<::lang_alloc::string::String>() {
            Cow::Borrowed(s.as_str())
        } else {
            Cow::Borrowed("allocation panic (non-string payload)")
        }
    }
}

#[cfg(feature = "std")]
impl TryDebug for PayloadBox {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.try_debug_struct("PayloadBox")
            .field("0", &AssertDebug(&*self.0))
            .finish()
    }
}
