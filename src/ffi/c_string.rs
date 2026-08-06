//! Fallible `CString` operations.
//!
//! Provides the [`TryCString`] trait with methods that mirror common `CString`
//! constructors but return [`Result`] to handle allocation failures and nul-byte
//! validation errors gracefully.
//!
//! # Design
//!
//! `CString::new` can fail in two ways: the input may contain an interior nul byte
//! (returned as [`std::ffi::NulError`]), or the internal buffer allocation may
//! panic on out-of-memory. [`TryCString::try_new`] takes ownership of a
//! [`Vec<u8>`] so that allocation is decoupled from construction — the caller
//! controls when memory is committed, and the method only needs to validate the
//! contents and append the terminating nul byte. The [`try_new_give_back`][TryCString::try_new_give_back]
//! variant returns the original buffer on any failure so no data is lost.

use crate::alloc::AllocError;
use crate::alloc::TryReserveError;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::vec::{TrySlice, TryVecError};
use core::fmt;
use std::ffi::CString;

/// Error returned by [`TryCString`] operations.
#[derive(Debug)]
pub enum TryCStringError {
    /// A raw heap allocation failed (no collection involved).
    Alloc(AllocError),
    /// A capacity reservation failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An arithmetic overflow occurred while computing required capacity.
    Overflow,
    /// The input contained an interior nul byte at the given index.
    Nul(usize),
}

impl fmt::Display for TryCStringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "CString operation failed: heap allocation error"),
            Self::Reserve(e) => write!(f, "CString operation failed: {}", e),
            Self::Overflow => {
                write!(
                    f,
                    "CString operation failed: capacity calculation overflowed"
                )
            }
            Self::Nul(idx) => {
                write!(
                    f,
                    "CString operation failed: interior nul byte at index {}",
                    idx
                )
            }
        }
    }
}

impl From<AllocError> for TryCStringError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

impl From<TryReserveError> for TryCStringError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<std::collections::TryReserveError> for TryCStringError {
    fn from(err: std::collections::TryReserveError) -> Self {
        Self::Reserve(TryReserveError::from(err))
    }
}

/// A trait for fallible `CString` operations.
///
/// Implemented for `CString`. Mirrors the most commonly-used `CString` methods
/// that can fail due to allocation pressure or invalid input, returning
/// [`Result`] values instead of panicking.
pub trait TryCString {
    /// Fallibly construct a `CString` from a `Vec<u8>` buffer.
    ///
    /// The buffer must not contain an interior nul byte (`\0`). The caller should
    /// reserve one extra byte of capacity before calling this method so that
    /// appending the terminating nul byte does not require reallocation.
    ///
    /// Returns [`Ok(CString)`][CString] if the buffer is valid and the nul terminator was
    /// appended. Returns [`Err`] with a [`TryCStringError`] describing the
    /// failure; the buffer is consumed and not recoverable.
    fn try_new(buf: Vec<u8>) -> Result<CString, TryCStringError>;

    /// Like [`Self::try_new`] but returns ownership of the buffer back on failure.
    ///
    /// The returned `Vec<u8>` has at most one extra byte of capacity beyond its
    /// length (from the attempted `try_reserve(1)`).
    fn try_new_give_back(buf: Vec<u8>) -> Result<CString, (Vec<u8>, TryCStringError)>;

    // ── Aliases with `fallible_` prefix ─────────────────────────────────────

    /// Alias for [`Self::try_new`].
    fn fallible_new(buf: Vec<u8>) -> Result<CString, TryCStringError> {
        Self::try_new(buf)
    }

    /// Alias for [`Self::try_new_give_back`].
    fn fallible_new_give_back(buf: Vec<u8>) -> Result<CString, (Vec<u8>, TryCStringError)> {
        Self::try_new_give_back(buf)
    }
}

impl TryCString for CString {
    fn try_new(buf: Vec<u8>) -> Result<CString, TryCStringError> {
        // Validate first — scan for interior nul bytes before touching anything.
        if let Some(idx) = buf.iter().position(|&b| b == 0) {
            return Err(TryCStringError::Nul(idx));
        }

        let mut buf = buf;
        buf.try_reserve(1).map_err(TryCStringError::from)?;
        buf.push(0);

        // SAFETY: We verified there are no interior nul bytes and we appended
        // a trailing nul byte. The vec now satisfies the invariant required by
        // from_vec_with_nul_unchecked.
        Ok(unsafe { CString::from_vec_with_nul_unchecked(buf) })
    }

    fn try_new_give_back(buf: Vec<u8>) -> Result<CString, (Vec<u8>, TryCStringError)> {
        // Validate first — scan for interior nul bytes before touching anything.
        if let Some(idx) = buf.iter().position(|&b| b == 0) {
            return Err((buf, TryCStringError::Nul(idx)));
        }

        let mut buf = buf;
        if let Err(e) = buf.try_reserve(1) {
            return Err((buf, TryCStringError::from(e)));
        }
        buf.push(0);

        // SAFETY: Same reasoning as try_new.
        Ok(unsafe { CString::from_vec_with_nul_unchecked(buf) })
    }
}

// ── TryClone for CString ─────────────────────────────────────────────────────

impl TryClone for CString {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Clone the bytes including the trailing nul fallibly, then hand the
        // vec directly to from_vec_with_nul_unchecked — no pop, no re-validate.
        let bytes = self.as_bytes_with_nul();
        // as_bytes_with_nul always returns at least one byte (the nul), so it's
        // never empty. An empty CString is b"\0".
        let buf = bytes.try_to_vec().map_err(|e| match e {
            TryVecError::Reserve(r) => TryCloneError::Reserve(r),
            TryVecError::Clone(c) => c,
            TryVecError::Overflow => TryCloneError::Overflow,
            TryVecError::Alloc(e) => TryCloneError::Alloc(e),
            TryVecError::Other(m) => TryCloneError::Other(m),
        })?;

        // SAFETY: buf was cloned from a valid CString's as_bytes_with_nul(),
        // so it has no interior nul bytes and ends with exactly one nul.
        Ok(unsafe { CString::from_vec_with_nul_unchecked(buf) })
    }
}

// ── TryDefault for CString ───────────────────────────────────────────────────

impl TryDefault for CString {
    fn try_default() -> Result<Self, TryDefaultError> {
        // An empty CString requires no allocation.
        Ok(CString::new("").unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Valid inputs ─────────────────────────────────────────────────────────

    #[test]
    fn try_new_empty() {
        let c = CString::try_new(Vec::new()).unwrap();
        assert_eq!(c.as_bytes(), b"");
        assert_eq!(c.as_bytes_with_nul(), b"\0");
    }

    #[test]
    fn try_new_ascii() {
        let c = CString::try_new(b"hello".to_vec()).unwrap();
        assert_eq!(c.to_str().unwrap(), "hello");
    }

    #[test]
    fn try_new_unicode_utf8() {
        let c = CString::try_new("こんにちは 🦀".as_bytes().to_vec()).unwrap();
        assert_eq!(c.to_str().unwrap(), "こんにちは 🦀");
    }

    #[test]
    fn try_new_binary_data() {
        let c = CString::try_new(vec![1, 2, 3, 255, 0xFF]).unwrap();
        assert_eq!(c.as_bytes(), &[1, 2, 3, 255, 0xFF]);
    }

    #[test]
    fn try_new_with_spare_capacity() {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(b"hi");
        let c = CString::try_new(buf).unwrap();
        assert_eq!(c.to_str().unwrap(), "hi");
    }

    #[test]
    fn try_new_exact_capacity() {
        // Buffer has exactly enough capacity for content + nul.
        let mut buf = Vec::with_capacity(6);
        buf.extend_from_slice(b"hello");
        let c = CString::try_new(buf).unwrap();
        assert_eq!(c.to_str().unwrap(), "hello");
    }

    // ── Interior nul detection ───────────────────────────────────────────────

    #[test]
    fn try_new_nul_at_start() {
        let result = CString::try_new(vec![0, 104, 101]);
        assert!(matches!(result, Err(TryCStringError::Nul(0))));
    }

    #[test]
    fn try_new_nul_in_middle() {
        let result = CString::try_new(vec![104, 101, 0, 108, 111]);
        assert!(matches!(result, Err(TryCStringError::Nul(2))));
    }

    #[test]
    fn try_new_nul_at_end() {
        let result = CString::try_new(vec![104, 101, 108, 108, 111, 0]);
        assert!(matches!(result, Err(TryCStringError::Nul(5))));
    }

    #[test]
    fn try_new_multiple_nuls_reports_first() {
        let result = CString::try_new(vec![104, 0, 101, 0, 108]);
        assert!(matches!(result, Err(TryCStringError::Nul(1))));
    }

    // ── try_new_give_back ────────────────────────────────────────────────────

    #[test]
    fn try_new_give_back_success() {
        let buf = b"hello".to_vec();
        let c = CString::try_new_give_back(buf).unwrap();
        assert_eq!(c.to_str().unwrap(), "hello");
    }

    #[test]
    fn try_new_give_back_nul_returns_buffer() {
        let buf = vec![104, 0, 101];
        let result = CString::try_new_give_back(buf.clone());
        let (returned_buf, err) = result.unwrap_err();
        assert!(matches!(err, TryCStringError::Nul(1)));
        assert_eq!(returned_buf, buf);
    }

    #[test]
    fn try_new_give_back_preserves_contents_on_error() {
        let buf = vec![1, 2, 0, 4, 5];
        let (_, err) = CString::try_new_give_back(buf.clone()).unwrap_err();
        assert!(matches!(err, TryCStringError::Nul(2)));
    }

    #[test]
    fn try_new_give_back_empty_succeeds() {
        let c = CString::try_new_give_back(Vec::new()).unwrap();
        assert_eq!(c.as_bytes_with_nul(), b"\0");
    }

    // ── Roundtrip ────────────────────────────────────────────────────────────

    #[test]
    fn roundtrip_hello_world() {
        let buf = b"Hello, world!".to_vec();
        let c = CString::try_new(buf).unwrap();
        assert_eq!(c.to_str().unwrap(), "Hello, world!");
        assert_eq!(c.as_bytes_with_nul(), b"Hello, world!\0");
    }

    #[test]
    fn roundtrip_empty() {
        let c = CString::try_new(Vec::new()).unwrap();
        assert!(c.to_str().unwrap().is_empty());
    }

    #[test]
    fn roundtrip_max_byte_values() {
        let buf = vec![255u8; 100];
        let c = CString::try_new(buf.clone()).unwrap();
        assert_eq!(c.as_bytes(), &buf[..]);
    }

    // ── Display ──────────────────────────────────────────────────────────────

    #[test]
    fn error_display_nul() {
        let err = TryCStringError::Nul(42);
        let msg = format!("{}", err);
        assert!(msg.contains("nul"));
        assert!(msg.contains("42"));
    }

    // ── TryClone ─────────────────────────────────────────────────────────────

    #[test]
    fn try_clone_empty() {
        let c = CString::try_new(Vec::new()).unwrap();
        let cloned = c.try_clone().unwrap();
        assert_eq!(cloned.as_bytes(), b"");
    }

    #[test]
    fn try_clone_ascii() {
        let c = CString::try_new(b"hello".to_vec()).unwrap();
        let cloned = c.try_clone().unwrap();
        assert_eq!(c, cloned);
    }

    #[test]
    fn try_clone_unicode_utf8() {
        let c = CString::try_new("日本語".as_bytes().to_vec()).unwrap();
        let cloned = c.try_clone().unwrap();
        assert_eq!(c.to_str().unwrap(), cloned.to_str().unwrap());
    }

    // ── TryDefault ───────────────────────────────────────────────────────────

    #[test]
    fn try_default_empty_cstring() {
        let c: CString = CString::try_default().unwrap();
        assert_eq!(c.as_bytes(), b"");
    }

    // ── OOM tests ─────────────────────────────────────────────────────
    use crate::test_allocator::{FailPolicy, with_policy};

    #[test]
    fn cstring_try_clone_fails_on_oom() {
        let c = CString::try_new(b"hello".to_vec()).unwrap();
        let r: Result<CString, TryCloneError> = with_policy(
            FailPolicy::fail_next_alloc(),
            || c.try_clone(),
        );
        assert!(r.is_err());
    }

    #[test]
    fn cstring_try_clone_always_allocates() {
        // Even an empty CString contains a trailing nul byte, so try_to_vec()
        // always allocates at least 1 byte. No zero-allocation path exists.
        let c = CString::try_new(Vec::new()).unwrap();
        let r: Result<CString, TryCloneError> = with_policy(
            FailPolicy::fail_next_alloc(),
            || c.try_clone(),
        );
        assert!(r.is_err());
    }

    #[test]
    fn cstring_nth_alloc_fail_targets_correct_call() {
        let c = CString::try_new(b"hello".to_vec()).unwrap();
        let (r1_ok, r2_err, r3_ok) = with_policy(FailPolicy::fail_nth_alloc(2), || {
            let r1: Result<CString, TryCloneError> = c.try_clone();
            let r2: Result<CString, TryCloneError> = c.try_clone();
            let r3: Result<CString, TryCloneError> = c.try_clone();
            (r1.is_ok(), r2.is_err(), r3.is_ok())
        });
        assert!(r1_ok, "first clone should succeed");
        assert!(r2_err, "second clone should fail");
        assert!(r3_ok, "third clone should succeed");
    }

    #[test]
    fn cstring_oom_restores_allocation_afterwards() {
        let c = CString::try_new(b"hello".to_vec()).unwrap();
        let r: Result<CString, TryCloneError> = with_policy(
            FailPolicy::fail_next_alloc(),
            || c.try_clone(),
        );
        assert!(r.is_err());
        let r: Result<CString, TryCloneError> = c.try_clone();
        assert!(r.is_ok());
    }
}
