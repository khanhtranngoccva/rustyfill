//! Custom allocation error type and panic payload wrapper.
//!
//! The [`alloc`](core::alloc) crate's `AllocError` is not exposed on stable Rust,
//! so we provide our own equivalent for use across this library. We also provide
//! [`PayloadBox`], an owning wrapper around the raw panic payload from
//! [`std::panic::catch_unwind`].

use core::fmt;
use std::alloc::Layout;
use std::borrow::Cow;

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

// Not impling `std::error::Error` to stay no_std compatible by default.

/// Owning wrapper around the raw panic payload from [`std::panic::catch_unwind`].
///
/// Holds the `Box<dyn Any + Send>` verbatim so that constructing the error
/// after catching a panic performs zero additional allocations.
#[derive(Debug)]
pub struct PayloadBox(pub Box<dyn core::any::Any + Send>);

impl PayloadBox {
    /// Extract a human-readable message from the payload.
    ///
    /// May allocate if the payload contains a `&str` or `String`.
    /// Callers should only invoke this outside of tight/OOM-sensitive paths
    /// (e.g. during logging or display formatting).
    pub fn message(&self) -> Cow<'_, str> {
        if let Some(s) = self.0.downcast_ref::<&str>() {
            Cow::Borrowed(*s)
        } else if let Some(s) = self.0.downcast_ref::<String>() {
            Cow::Borrowed(s.as_str())
        } else {
            Cow::Borrowed("allocation panic (non-string payload)")
        }
    }
}
