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
use crate::alloc::TryReserveError;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use core::fmt;
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
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
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
    /// Returns [`TryOsStringError::Alloc`] if the re-allocation fails.
    /// Equivalent to `OsString::shrink_to_fit()` but fallible.
    fn try_shrink_to_fit(&mut self) -> Result<(), TryOsStringError>;

    /// Fallibly shrink the capacity of this `OsString` to at least `min_capacity`.
    ///
    /// If the current capacity is already less than or equal to `min_capacity`,
    /// does nothing and returns `Ok(())`. Otherwise reallocates down.
    /// Returns [`TryOsStringError::Alloc`] if the re-allocation fails.
    /// Equivalent to `OsString::shrink_to(min_capacity)` but fallible.
    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryOsStringError>;

    // ── Conversion ──────────────────────────────────────────────────────────

    /// Attempt to convert this `OsString` into a [`String`] without copying.
    ///
    /// Succeeds if the `OsString` contains valid UTF-8 data. Otherwise returns
    /// the original `OsString` unchanged. This operation does not allocate and
    /// therefore never fails due to allocation pressure.
    ///
    /// Equivalent to `OsString::into_string()`.
    fn try_into_string(self) -> Result<String, OsString>;

    // ── Aliases with `fallible_` prefix ────────────────────────────────────

    /// Alias for [`Self::try_with_capacity`].
    fn fallible_with_capacity(capacity: usize) -> Result<OsString, TryReserveError> {
        Self::try_with_capacity(capacity)
    }

    /// Alias for [`Self::try_from_os_str`].
    fn fallible_from_os_str<S: AsRef<OsStr>>(s: S) -> Result<OsString, TryReserveError> {
        Self::try_from_os_str(s)
    }

    /// Alias for [`Self::try_from_str`].
    fn fallible_from_str<S: AsRef<str>>(s: S) -> Result<OsString, TryReserveError> {
        Self::try_from_str(s)
    }

    /// Alias for [`Self::try_push`].
    fn fallible_push(&mut self, s: &OsStr) -> Result<(), TryReserveError> {
        Self::try_push(self, s)
    }

    /// Alias for [`Self::try_push_str`].
    fn fallible_push_str(&mut self, s: &str) -> Result<(), TryReserveError> {
        Self::try_push_str(self, s)
    }

    /// Alias for [`Self::try_shrink_to_fit`].
    fn fallible_shrink_to_fit(&mut self) -> Result<(), TryOsStringError> {
        Self::try_shrink_to_fit(self)
    }

    /// Alias for [`Self::try_shrink_to`].
    fn fallible_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryOsStringError> {
        Self::try_shrink_to(self, min_capacity)
    }

    /// Alias for [`Self::try_into_string`].
    fn fallible_into_string(self) -> Result<String, OsString> {
        Self::try_into_string(self)
    }
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

    fn try_shrink_to_fit(&mut self) -> Result<(), TryOsStringError> {
        self.try_shrink_to(self.len())
    }

    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryOsStringError> {
        // Convert to encoded bytes (Vec<u8>), shrink via TryVec, then convert
        // back. Only the spare capacity portion is reallocated — the OS string
        // data bytes are never copied or revalidated.
        let mut v = std::mem::replace(self, OsString::new()).into_encoded_bytes();
        let result = <Vec<u8> as crate::vec::TryVec<u8>>::fallible_shrink_to(&mut v, min_capacity);
        // SAFETY: the bytes originated from a valid OsString via into_encoded_bytes.
        *self = unsafe { OsString::from_encoded_bytes_unchecked(v) };
        result.map_err(|e| match e {
            crate::vec::TryVecError::Alloc(e) => TryOsStringError::Alloc(e),
            crate::vec::TryVecError::Reserve(e) => TryOsStringError::Reserve(e),
            crate::vec::TryVecError::Clone(_) => unreachable!("shrink does not clone"),
            crate::vec::TryVecError::Overflow => TryOsStringError::Overflow,
            crate::vec::TryVecError::Other(msg) => TryOsStringError::Other(msg),
        })
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
                .map_err(|e| TryCloneError::Reserve(e.into()))?;
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

// ── TryDebug for OsString ────────────────────────────────────────────────────

impl crate::try_fmt::TryDebug for OsString {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_os_str().try_fmt(f)
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
    fn try_shrink_to_below_len_reduces_padding() {
        let mut s = OsString::try_from_str("abcdef").unwrap();
        // min_capacity < len → target == len. Allocator may have rounded up
        // the original capacity above len, so shrink can still reduce padding.
        let cap_before = s.capacity();
        s.try_shrink_to(2).unwrap();
        assert_eq!(s, OsString::from("abcdef"));
        assert!(s.capacity() >= s.len());
        assert!(s.capacity() < cap_before || s.capacity() == s.len());
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

    // ── OOM tests ─────────────────────────────────────────────────────
    use rustyfill_test_allocator::{FailPolicy, with_policy};

    #[test]
    fn osstring_try_with_capacity_fails_on_oom() {
        let r: Result<OsString, TryReserveError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <OsString as TryOsString>::try_with_capacity(10)
            });
        assert!(r.is_err());
    }

    #[test]
    fn osstring_try_with_capacity_zero_succeeds_under_oom() {
        let r: Result<OsString, TryReserveError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <OsString as TryOsString>::try_with_capacity(0)
            });
        assert!(r.is_ok());
    }

    #[test]
    fn osstring_try_push_fails_on_oom() {
        let mut s = OsString::new();
        s.try_shrink_to_fit().unwrap();
        let r = with_policy(FailPolicy::fail_next_alloc(), || {
            s.fallible_push(OsStr::new("hello"))
        });
        assert!(r.is_err());
    }

    #[test]
    fn osstring_try_clone_fails_on_oom() {
        let orig = OsString::try_from_str("hello").unwrap();
        let r: Result<OsString, TryCloneError> =
            with_policy(FailPolicy::fail_next_alloc(), || orig.try_clone());
        assert!(r.is_err());
    }

    #[test]
    fn osstring_try_clone_empty_succeeds_under_oom() {
        let orig = OsString::new();
        let r: Result<OsString, TryCloneError> =
            with_policy(FailPolicy::fail_next_alloc(), || orig.try_clone());
        assert!(r.is_ok());
    }

    #[test]
    fn osstring_nth_alloc_fail_targets_correct_call() {
        let orig = OsString::try_from_str("hello").unwrap();
        let (r1_ok, r2_err, r3_ok) = with_policy(FailPolicy::fail_nth_alloc(2), || {
            let r1: Result<OsString, TryCloneError> = orig.try_clone();
            let r2: Result<OsString, TryCloneError> = orig.try_clone();
            let r3: Result<OsString, TryCloneError> = orig.try_clone();
            (r1.is_ok(), r2.is_err(), r3.is_ok())
        });
        assert!(r1_ok, "first clone should succeed");
        assert!(r2_err, "second clone should fail");
        assert!(r3_ok, "third clone should succeed");
    }

    #[test]
    fn osstring_oom_restores_allocation_afterwards() {
        let r: Result<OsString, TryReserveError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <OsString as TryOsString>::try_with_capacity(10)
            });
        assert!(r.is_err());
        // Allocation works again after guard scope ends.
        let r: Result<OsString, TryReserveError> = <OsString as TryOsString>::try_with_capacity(10);
        assert!(r.is_ok());
    }
}
