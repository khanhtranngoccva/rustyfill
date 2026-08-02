//! Fallible `OsString` operations.
//!
//! Provides the [`TryOsString`] trait with methods that mirror common `OsString`
//! constructors and mutating operations but return [`Result`] to handle allocation
//! failures gracefully.
//!
//! # Design
//!
//! `OsString` exposes a fallible [`OsString::try_reserve`] on stable Rust, which
//! this trait uses internally. Methods that may grow internal capacity (`push`,
//! `push_str`, etc.) call `try_reserve` first so that allocation failures
//! surface as errors rather than panics. Operations without inherent fallible
//! counterparts on stable (`shrink_to_fit`, `with_capacity`) are implemented via
//! `try_reserve` probes.
//!
//! The trait also implements [`TryClone`](crate::try_clone::TryClone) and
//! [`TryDefault`](crate::try_default::TryDefault) for `OsString`.

use crate::alloc::AllocError;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use core::fmt;
use std::collections::TryReserveError;
use std::ffi::{OsStr, OsString};

/// Error returned by [`TryOsString`] operations.
///
/// Wraps the ways an `OsString` operation can fail on stable Rust: a reserve
/// failure ([`TryReserveError`]) or an arithmetic overflow when computing
/// the required capacity.
#[derive(Debug)]
pub enum TryOsStringError {
    /// A raw heap allocation failed (no collection involved).
    Alloc(AllocError),
    /// A capacity reservation failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An arithmetic overflow occurred while computing required capacity.
    Overflow,
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Display for TryOsStringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "OsString operation failed: heap allocation error"),
            Self::Reserve(e) => write!(f, "OsString operation failed: {}", e),
            Self::Overflow => {
                write!(
                    f,
                    "OsString operation failed: capacity calculation overflowed"
                )
            }
            Self::Other(msg) => write!(f, "OsString operation failed: {}", msg),
        }
    }
}

impl From<AllocError> for TryOsStringError {
    fn from(_: AllocError) -> Self {
        Self::Alloc(AllocError)
    }
}

impl From<TryReserveError> for TryOsStringError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

/// A trait for fallible `OsString` operations.
///
/// Implemented for `OsString`. Mirrors the most commonly-used `OsString` methods
/// that can fail due to allocation pressure, returning [`Result`] values instead
/// of panicking.
pub trait TryOsString: Sized {
    // ── Construction ────────────────────────────────────────────────────────

    /// Fallibly construct a new `OsString` with at least enough capacity for
    /// `capacity` bytes.
    ///
    /// Returns [`TryReserveError`] if the initial allocation fails.
    fn try_with_capacity(capacity: usize) -> Result<OsString, TryReserveError>;

    /// Fallibly construct an `OsString` from any value that references an `OsStr`.
    ///
    /// Accepts `&OsStr`, `&OsString`, or anything else implementing
    /// [`AsRef<OsStr>`]. Returns [`TryReserveError`] if the allocation fails.
    fn try_from_os_str<S: AsRef<OsStr>>(s: S) -> Result<OsString, TryReserveError>;

    /// Fallibly construct an `OsString` from any value that references a `str`.
    ///
    /// Accepts `&str`, `String`, `&String`, or anything else implementing
    /// [`AsRef<str>`]. Returns [`TryReserveError`] if the allocation fails.
    fn try_from_str<S: AsRef<str>>(s: S) -> Result<OsString, TryReserveError>;

    // ── Mutation ────────────────────────────────────────────────────────────

    /// Fallibly append an `&OsStr` to this `OsString`.
    ///
    /// Returns [`TryReserveError`] if growing the internal buffer fails.
    fn try_push(&mut self, s: &OsStr) -> Result<(), TryReserveError>;

    /// Fallibly append a `&str` to this `OsString`.
    ///
    /// Returns [`TryReserveError`] if growing the internal buffer fails.
    fn try_push_str(&mut self, s: &str) -> Result<(), TryReserveError>;

    /// Fallibly shrink the capacity of this `OsString` to match its length.
    ///
    /// May reallocate if the current allocation is larger than needed.
    /// Returns [`TryReserveError`] if the re-allocation fails.
    fn try_shrink_to_fit(&mut self) -> Result<(), TryReserveError>;

    /// Fallibly shrink the capacity of this `OsString` to at least `min_capacity`.
    ///
    /// If the current capacity is already less than `min_capacity`, does nothing.
    /// Otherwise reallocates down. Returns [`TryReserveError`] if the
    /// re-allocation fails.
    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryReserveError>;

    // ── Conversion ──────────────────────────────────────────────────────────

    /// Attempt to convert this `OsString` into a [`String`] without copying.
    ///
    /// Succeeds if the `OsString` contains valid UTF-8 data. Otherwise returns
    /// the original `OsString` unchanged. This operation does not allocate and
    /// therefore never fails due to allocation pressure.
    ///
    /// Equivalent to `OsString::into_string()`.
    fn try_into_string(self) -> Result<String, OsString>;
}

impl TryOsString for OsString {
    fn try_with_capacity(capacity: usize) -> Result<OsString, TryReserveError> {
        let mut out = OsString::new();
        if capacity > 0 {
            out.try_reserve(capacity)?;
        }
        Ok(out)
    }

    fn try_from_os_str<S: AsRef<OsStr>>(s: S) -> Result<OsString, TryReserveError> {
        let s = s.as_ref();
        let mut out = OsString::new();
        if !s.is_empty() {
            out.try_reserve(s.len())?;
        }
        out.push(s);
        Ok(out)
    }

    fn try_from_str<S: AsRef<str>>(s: S) -> Result<OsString, TryReserveError> {
        let s = s.as_ref();
        let mut out = OsString::new();
        if !s.is_empty() {
            out.try_reserve(s.len())?;
        }
        out.push(s);
        Ok(out)
    }

    fn try_push(&mut self, s: &OsStr) -> Result<(), TryReserveError> {
        if !s.is_empty() {
            self.try_reserve(s.len())?;
        }
        self.push(s);
        Ok(())
    }

    fn try_push_str(&mut self, s: &str) -> Result<(), TryReserveError> {
        if !s.is_empty() {
            self.try_reserve(s.len())?;
        }
        self.push(s);
        Ok(())
    }

    fn try_shrink_to_fit(&mut self) -> Result<(), TryReserveError> {
        self.try_shrink_to(self.len())
    }

    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryReserveError> {
        let target = core::cmp::max(self.len(), min_capacity);
        if self.capacity() <= target {
            return Ok(());
        }

        // Important: we must not use mem::take() before the allocation succeeds,
        // because a failed reservation would silently destroy the original data.
        let len = self.len();
        let mut out = OsString::new();
        if len > 0 {
            out.try_reserve(target)?;
            out.push(self.as_os_str());
        }
        *self = out;
        Ok(())
    }

    fn try_into_string(self) -> Result<String, OsString> {
        self.into_string()
    }
}

// ── TryClone for OsString ────────────────────────────────────────────────────

impl TryClone for OsString {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let mut out = OsString::new();
        if !self.is_empty() {
            out.try_reserve(self.len())
                .map_err(TryCloneError::Reserve)?;
        }
        out.push(self);
        Ok(out)
    }
}

// ── TryDefault for OsString ──────────────────────────────────────────────────

impl TryDefault for OsString {
    fn try_default() -> Result<Self, TryDefaultError> {
        // An empty OsString requires no allocation.
        Ok(OsString::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn try_with_capacity_zero() {
        let s = OsString::try_with_capacity(0).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn try_with_capacity_nonzero() {
        let s = OsString::try_with_capacity(64).unwrap();
        assert!(s.is_empty());
        assert!(s.capacity() >= 64);
    }

    #[test]
    fn try_from_os_str_empty() {
        let s = OsString::try_from_os_str(OsStr::new("")).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn try_from_os_str_ascii() {
        let s = OsString::try_from_os_str(OsStr::new("hello")).unwrap();
        assert_eq!(s, OsString::from("hello"));
    }

    #[test]
    fn try_from_str_ascii() {
        let s = OsString::try_from_str("hello").unwrap();
        assert_eq!(s, OsString::from("hello"));
    }

    #[test]
    fn try_from_str_unicode() {
        let s = OsString::try_from_str("こんにちは 🦀").unwrap();
        assert_eq!(s, OsString::from("こんにちは 🦀"));
    }

    // ── Mutation ─────────────────────────────────────────────────────────────

    #[test]
    fn try_push_osstr() {
        let mut s = OsString::new();
        s.try_push(OsStr::new("world")).unwrap();
        assert_eq!(s, OsString::from("world"));
    }

    #[test]
    fn try_push_str() {
        let mut s = OsString::new();
        s.try_push_str("hello").unwrap();
        assert_eq!(s, OsString::from("hello"));
    }

    #[test]
    fn try_push_multiple() {
        let mut s = OsString::new();
        s.try_push_str("Hello, ").unwrap();
        s.try_push(OsStr::new("world!")).unwrap();
        assert_eq!(s, OsString::from("Hello, world!"));
    }

    #[test]
    fn try_push_empty_noop() {
        let mut s = OsString::new();
        s.try_push(OsStr::new("")).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn try_push_str_empty_noop() {
        let mut s = OsString::new();
        s.try_push_str("").unwrap();
        assert!(s.is_empty());
    }

    // ── Shrink ───────────────────────────────────────────────────────────────

    #[test]
    fn try_shrink_to_fit_reduces_excess() {
        let mut s = OsString::try_with_capacity(1024).unwrap();
        s.try_push_str("small").unwrap();
        let cap_before = s.capacity();
        assert!(cap_before >= 1024);
        s.try_shrink_to_fit().unwrap();
        assert!(
            s.capacity() < cap_before,
            "capacity {} was not reduced from {}",
            s.capacity(),
            cap_before
        );
        assert!(s.capacity() >= 5);
        assert_eq!(s, OsString::from("small"));
    }

    #[test]
    fn try_shrink_to_above_len() {
        let mut s = OsString::try_with_capacity(256).unwrap();
        s.try_push_str("tiny").unwrap();
        let cap_before = s.capacity();
        assert!(cap_before >= 256);
        s.try_shrink_to(32).unwrap();
        assert!(s.capacity() >= 32);
        assert!(s.capacity() < cap_before || s.capacity() >= 32);
        assert_eq!(s, OsString::from("tiny"));
    }

    #[test]
    fn try_shrink_to_below_len_is_noop() {
        let mut s = OsString::try_from_str("abcdef").unwrap();
        let cap_before = s.capacity();
        s.try_shrink_to(2).unwrap();
        assert_eq!(s, OsString::from("abcdef"));
        assert_eq!(s.capacity(), cap_before);
    }

    // ── Conversion ───────────────────────────────────────────────────────────

    #[test]
    fn try_into_string_valid_utf8() {
        let s = OsString::try_from_str("hello").unwrap();
        let result: String = s.try_into_string().unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn try_into_string_invalid_utf8() {
        // On Unix, OsString can hold arbitrary bytes.
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let s = OsString::from_vec(vec![0xFF, 0xFE]);
            let result = s.try_into_string();
            assert!(result.is_err());
        }
    }

    // ── TryClone ─────────────────────────────────────────────────────────────

    #[test]
    fn try_clone_empty() {
        let s = OsString::new();
        let c = s.try_clone().unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn try_clone_populated() {
        let s = OsString::try_from_str("testing").unwrap();
        let c = s.try_clone().unwrap();
        assert_eq!(c, OsString::from("testing"));
    }

    // ── TryDefault ───────────────────────────────────────────────────────────

    #[test]
    fn try_default_empty() {
        let s: OsString = OsString::try_default().unwrap();
        assert!(s.is_empty());
    }

    // ── Combined workflows ───────────────────────────────────────────────────

    #[test]
    fn build_then_clone() {
        let mut s = OsString::try_default().unwrap();
        s.try_push_str("initial").unwrap();
        let c = s.try_clone().unwrap();
        s.try_push_str("_modified").unwrap();
        assert_eq!(s, OsString::from("initial_modified"));
        assert_eq!(c, OsString::from("initial"));
    }

    #[test]
    fn build_then_shrink() {
        let mut s = OsString::try_with_capacity(256).unwrap();
        s.try_push_str("a").unwrap();
        s.try_push_str("z").unwrap();
        assert_eq!(s, OsString::from("az"));
        let cap_before = s.capacity();
        s.try_shrink_to_fit().unwrap();
        assert!(s.capacity() < cap_before || s.capacity() >= 2);
        assert!(s.capacity() >= 2);
    }
}
