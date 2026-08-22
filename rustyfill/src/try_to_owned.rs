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
use crate::try_clone::TryCloneError;
use crate::try_fmt::{TryDebug, TryDisplay, helpers::FormatterExt};
use lang_alloc::borrow::ToOwned;
use lang_core::error;
use lang_core::fmt;

/// Error returned by [`TryToOwned::try_to_owned`].
#[derive(Clone, PartialEq, Eq)]
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

impl fmt::Debug for TryToOwnedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(self, f)
    }
}

impl fmt::Display for TryToOwnedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }
}

impl TryDebug for TryToOwnedError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => f
                .try_debug_tuple("TryToOwnedError::Alloc")
                .field(e)
                .finish(),
            Self::Reserve(e) => f
                .try_debug_tuple("TryToOwnedError::Reserve")
                .field(e)
                .finish(),
            Self::Overflow => f.write_str("TryToOwnedError::Overflow"),
            Self::Other(msg) => f
                .try_debug_tuple("TryToOwnedError::Other")
                .field(msg)
                .finish(),
        }
    }
}

impl TryDisplay for TryToOwnedError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "to_owned failed: heap allocation error"),
            Self::Reserve(e) => write!(f, "to_owned failed: {}", e),
            Self::Overflow => write!(f, "to_owned failed: capacity calculation overflowed"),
            Self::Other(msg) => write!(f, "to_owned failed: {}", msg),
        }
    }
}

impl error::Error for TryToOwnedError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
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

    /// Alias for [`Self::try_to_owned`].
    fn fallible_to_owned(&self) -> Result<Self::Owned, TryToOwnedError> {
        Self::try_to_owned(self)
    }
}
