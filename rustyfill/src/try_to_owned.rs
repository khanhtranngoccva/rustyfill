//! Fallible owned-value construction for types that implement [`ToOwned`].
//!
//! Provides the [`TryToOwned`] trait, a drop-in analogue of
//! [`ToOwned`] that can fail when producing the owned form requires
//! allocating memory (e.g. turning `&[T]` into `Vec<T>` or `&str` into `String`).
//!
//! # Design
//!
//! [`TryToOwned`] requires [`ToOwned`] as a supertrait so that
//! any type accepting `TryToOwned` can still be used wherever `ToOwned` is expected.
//! Implementors must ensure `try_to_owned` never panics — allocation failures are
//! returned as errors instead.

use crate::alloc::{AllocError, TryReserveError};
use crate::lang_alloc::borrow::ToOwned;
use crate::try_clone::TryCloneError;
use crate::try_fmt::{TryDebug, helpers::FormatterExt};

/// Error returned by [`TryToOwned::try_to_owned`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TryToOwnedError {
    /// A raw heap allocation failed (no collection involved).
    Alloc(AllocError),
    /// A capacity reservation on a collection failed (overflow or OOM).
    Reserve(TryReserveError),
    /// A manually detected arithmetic overflow (e.g., size multiplication).
    Overflow,
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl core::fmt::Display for TryToOwnedError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "to_owned failed: heap allocation error"),
            Self::Reserve(e) => write!(f, "to_owned failed: {}", e),
            Self::Overflow => write!(f, "to_owned failed: capacity calculation overflowed"),
            Self::Other(msg) => write!(f, "to_owned failed: {}", msg),
        }
    }
}

impl TryDebug for TryToOwnedError {
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Alloc(e) => f
                .try_debug_struct("TryToOwnedError::Alloc")
                .field("0", e)
                .finish(),
            Self::Reserve(e) => f
                .try_debug_struct("TryToOwnedError::Reserve")
                .field("0", e)
                .finish(),
            Self::Overflow => f.write_str("TryToOwnedError::Overflow"),
            Self::Other(msg) => f
                .try_debug_struct("TryToOwnedError::Other")
                .field("0", msg)
                .finish(),
        }
    }
}

#[cfg(feature = "std")]
impl core::error::Error for TryToOwnedError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Reserve(e) => Some(e),
            Self::Alloc(_) | Self::Overflow | Self::Other(_) => None,
        }
    }
}

impl From<AllocError> for TryToOwnedError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

impl From<TryReserveError> for TryToOwnedError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

#[cfg(feature = "std")]
impl From<::lang_std::collections::TryReserveError> for TryToOwnedError {
    fn from(err: ::lang_std::collections::TryReserveError) -> Self {
        Self::Reserve(TryReserveError::from(err))
    }
}

impl From<TryCloneError> for TryToOwnedError {
    fn from(err: TryCloneError) -> Self {
        match err {
            TryCloneError::Alloc(e) => Self::Alloc(e),
            TryCloneError::Reserve(e) => Self::Reserve(e),
            TryCloneError::Overflow => Self::Overflow,
            TryCloneError::Other(msg) => Self::Other(msg),
        }
    }
}

/// A fallible analogue of [`ToOwned`].
///
/// Types implementing this trait guarantee that constructing their owned variant
/// will not panic on allocation failure. The [`ToOwned`] supertrait ensures
/// compatibility with existing APIs expecting `ToOwned`.
pub trait TryToOwned: ToOwned {
    /// Construct the owned version of `self`, falling back to an error on
    /// allocation failure rather than panicking.
    fn try_to_owned(&self) -> Result<Self::Owned, TryToOwnedError>;
}
