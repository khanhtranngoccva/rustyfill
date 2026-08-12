//! TryDebug / TryDisplay implementations for well-known `core` error types.
//!
//! These types' `Debug` and `Display` implementations are known to never
//! implicitly allocate — they print fixed struct names, enum discriminants,
//! or delegate to primitive/slice formatting.
//!
//! This module is available in `no_std` environments.

use lang_core::array;
use lang_core::char;
use lang_core::fmt;
use lang_core::num;
use lang_core::str;
use crate::try_fmt::{TryDebug, TryDisplay};

// ── num::TryFromIntError ──────────────────────────────────────────────────

impl TryDebug for num::TryFromIntError {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for num::TryFromIntError {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

// ── array::TryFromSliceError ──────────────────────────────────────────────

impl TryDebug for array::TryFromSliceError {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for array::TryFromSliceError {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

// ── str::Utf8Error ────────────────────────────────────────────────────────

impl TryDebug for str::Utf8Error {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for str::Utf8Error {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

// ── num parse errors ──────────────────────────────────────────────────────

impl TryDebug for num::ParseIntError {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for num::ParseIntError {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl TryDebug for num::ParseFloatError {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for num::ParseFloatError {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl TryDebug for str::ParseBoolError {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for str::ParseBoolError {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

// ── fmt::Error ────────────────────────────────────────────────────────────
// fmt::Error is an empty struct whose Debug prints "Error" and Display prints
// "internal or I/O error". Neither allocates.

impl TryDebug for fmt::Error {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for fmt::Error {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

// ── char::CharTryFromError ────────────────────────────────────────────────
// Empty struct (contains only private ()). Debug/Display print fixed strings.

impl TryDebug for char::CharTryFromError {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for char::CharTryFromError {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
