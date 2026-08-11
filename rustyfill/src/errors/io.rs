//! Fallible construction of [`::lang_std::io::Error`].
//!
//! `io::Error::new(kind, source)` boxes the source into a
//! `Box<dyn Error + Send + Sync>` — if that allocation fails, the process
//! aborts. This module provides fallible constructors that use
//! [`Box::fallible_new`] so callers can handle out-of-memory gracefully
//! instead of crashing mid-recovery.
//!
//! # Double-boxing avoidance
//!
//! - [`IoErrorExt::try_new`] takes a concrete error `E` and boxes it once via
//!   [`Box::fallible_new`].
//! - [`IoErrorExt::new_boxed`] takes an already-boxed
//!   `Box<dyn Error + Send + Sync>` and reuses it directly with **no extra
//!   allocation**.

use crate::alloc::AllocError;
use crate::lang_alloc::boxed::Box;
use crate::prelude::TryBox;

/// Extension trait for fallible [`::std::io::Error`] construction.
pub trait IoErrorExt {
    // ── new(kind, source) variants ────────────────────────────────────────────

    /// Fallibly construct an [`::std::io::Error`] from a kind and a concrete error source.
    ///
    /// The source is boxed once via [`Box::fallible_new`], returning
    /// [`AllocError`] on OOM. On success, the boxed error is passed to
    /// [`io::Error::new`] which accepts it without further allocation.
    fn try_new<E>(
        kind: ::lang_std::io::ErrorKind,
        source: E,
    ) -> Result<::lang_std::io::Error, AllocError>
    where
        E: ::lang_std::error::Error + Send + Sync + 'static;

    /// Construct an [`::std::io::Error`] from a kind and an already-boxed error source.
    ///
    /// Reuses the existing box directly — **no extra allocation** occurs.
    /// Prefer this when you already have a `Box<dyn Error + Send + Sync>`
    /// (e.g., from unwrapping another error chain).
    ///
    /// This method never fails because no new allocation is needed.
    fn new_boxed(
        kind: ::lang_std::io::ErrorKind,
        source: Box<dyn ::lang_std::error::Error + Send + Sync>,
    ) -> ::lang_std::io::Error;

    /// Construct an [`::std::io::Error`] from a kind and a concrete error source,
    /// falling back to [`ErrorKind::OutOfMemory`] if boxing fails.
    ///
    /// Unlike [`Self::try_new`], this always returns an [`::std::io::Error`] — never
    /// an [`AllocError`]. If the heap allocation needed to box the source succeeds,
    /// the returned error has the requested kind with the original source attached.
    /// If boxing fails due to OOM, the returned error is constructed from
    /// [`ErrorKind::OutOfMemory`] with no source, so the caller gets a valid
    /// error either way without needing to branch on [`Result`].
    fn new_or_oom<E>(kind: ::lang_std::io::ErrorKind, source: E) -> ::lang_std::io::Error
    where
        E: ::lang_std::error::Error + Send + Sync + 'static;

    // ── other(source) variants ────────────────────────────────────────────────

    /// Like [`Self::try_new`] but defaults to [`ErrorKind::Other`].
    ///
    /// Shorthand for `try_new(ErrorKind::Other, source)`.
    fn try_other<E>(source: E) -> Result<::lang_std::io::Error, AllocError>
    where
        E: ::lang_std::error::Error + Send + Sync + 'static;

    /// Like [`Self::new_boxed`] but defaults to [`ErrorKind::Other`].
    ///
    /// Shorthand for `new_boxed(ErrorKind::Other, source)`. Never fails.
    fn other_boxed(
        source: Box<dyn ::lang_std::error::Error + Send + Sync>,
    ) -> ::lang_std::io::Error;

    /// Like [`Self::new_or_oom`] but defaults to [`ErrorKind::Other`].
    ///
    /// Falls back to [`ErrorKind::OutOfMemory`] if boxing fails.
    fn other_or_oom<E>(source: E) -> ::lang_std::io::Error
    where
        E: ::lang_std::error::Error + Send + Sync + 'static;
}

impl IoErrorExt for ::lang_std::io::Error {
    fn try_new<E>(kind: ::lang_std::io::ErrorKind, source: E) -> Result<Self, AllocError>
    where
        E: ::lang_std::error::Error + Send + Sync + 'static,
    {
        let boxed: Box<E> = TryBox::fallible_new(source)?;
        let dyn_box: Box<dyn ::lang_std::error::Error + Send + Sync> = boxed;
        Ok(Self::new(kind, dyn_box))
    }

    fn new_boxed(
        kind: ::lang_std::io::ErrorKind,
        source: Box<dyn ::lang_std::error::Error + Send + Sync>,
    ) -> Self {
        Self::new(kind, source)
    }

    fn new_or_oom<E>(kind: ::lang_std::io::ErrorKind, source: E) -> Self
    where
        E: ::lang_std::error::Error + Send + Sync + 'static,
    {
        match <Box<E> as TryBox<E>>::fallible_new(source) {
            Ok(boxed) => {
                let dyn_box: Box<dyn ::lang_std::error::Error + Send + Sync> = boxed;
                Self::new(kind, dyn_box)
            }
            Err(_) => Self::from(::lang_std::io::ErrorKind::OutOfMemory),
        }
    }

    fn try_other<E>(source: E) -> Result<Self, AllocError>
    where
        E: ::lang_std::error::Error + Send + Sync + 'static,
    {
        Self::try_new(::lang_std::io::ErrorKind::Other, source)
    }

    fn other_boxed(source: Box<dyn ::lang_std::error::Error + Send + Sync>) -> Self {
        Self::new_boxed(::lang_std::io::ErrorKind::Other, source)
    }

    fn other_or_oom<E>(source: E) -> Self
    where
        E: ::lang_std::error::Error + Send + Sync + 'static,
    {
        Self::new_or_oom(::lang_std::io::ErrorKind::Other, source)
    }
}

// ── TryDebug / TryDisplay for io::Error ────────────────────────────────────────

impl crate::try_fmt::TryDebug for ::lang_std::io::Error {
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

impl crate::try_fmt::TryDisplay for ::lang_std::io::Error {
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

// ── TryDebug / TryDisplay for io::ErrorKind ────────────────────────────────────

impl crate::try_fmt::TryDebug for ::lang_std::io::ErrorKind {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

impl crate::try_fmt::TryDisplay for ::lang_std::io::ErrorKind {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}
