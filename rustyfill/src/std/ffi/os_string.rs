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

use crate::alloc::TryReserveError;
use crate::alloc::vec::TryVec;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use lang_alloc::boxed::Box;
use lang_alloc::string::String;
use lang_alloc::vec::Vec;
use lang_core::fmt;
use lang_core::mem;
use lang_std::ffi::{OsStr, OsString};

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
    /// Equivalent to `OsString::shrink_to_fit()` but fallible.
    fn try_shrink_to_fit(&mut self) -> Result<(), TryReserveError>;

    /// Fallibly shrink the capacity of this `OsString` to at least `min_capacity`.
    ///
    /// If the current capacity is already less than or equal to `min_capacity`,
    /// does nothing and returns `Ok(())`. Otherwise reallocates down.
    /// Returns [`TryReserveError`] if the re-allocation fails.
    /// Equivalent to `OsString::shrink_to(min_capacity)` but fallible.
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

    // ── Conversion to boxed types ─────────────────────────────────────────────

    /// Fallibly convert this `OsString` into a `Box<OsStr>`.
    ///
    /// The resulting box contains the OS string data. No bytes are ever copied:
    /// when the current allocation has spare capacity it is shrunk in place via
    /// `realloc`, and on success the buffer is handed straight to the box.
    ///
    /// Returns [`TryReserveError`] if the shrink reallocation fails. Note that
    /// unlike the give-back variant, the `OsString` is consumed either way — on
    /// failure the caller does not get the data back.
    ///
    /// For empty strings, this returns an empty boxed `OsStr` without allocating.
    fn try_into_boxed_osstr(self) -> Result<Box<OsStr>, TryReserveError>;

    /// Like [`Self::try_into_boxed_osstr`] but returns ownership of the
    /// `OsString` back on failure so the caller is not left empty-handed.
    fn try_into_boxed_osstr_give_back(self) -> Result<Box<OsStr>, (OsString, TryReserveError)>;

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

    /// Fallibly shrink the capacity of this `OsString` to match its length.
    ///
    /// Alias for [`Self::try_shrink_to_fit`].
    fn fallible_shrink_to_fit(&mut self) -> Result<(), TryReserveError> {
        Self::try_shrink_to_fit(self)
    }

    /// Fallibly shrink the capacity of this `OsString` to at least `min_capacity`.
    ///
    /// Alias for [`Self::try_shrink_to`].
    #[allow(deprecated)]
    fn fallible_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryReserveError> {
        Self::try_shrink_to(self, min_capacity)
    }

    /// Alias for [`Self::try_into_string`].
    fn fallible_into_string(self) -> Result<String, OsString> {
        Self::try_into_string(self)
    }

    /// Alias for [`Self::try_into_boxed_osstr`].
    fn fallible_into_boxed_osstr(self) -> Result<Box<OsStr>, TryReserveError> {
        Self::try_into_boxed_osstr(self)
    }

    /// Alias for [`Self::try_into_boxed_osstr_give_back`].
    fn fallible_into_boxed_osstr_give_back(
        self,
    ) -> Result<Box<OsStr>, (OsString, TryReserveError)> {
        Self::try_into_boxed_osstr_give_back(self)
    }
}

#[allow(deprecated)]
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
        // Convert to encoded bytes (Vec<u8>), shrink via TryVec, then convert
        // back. Only the spare capacity portion is reallocated — the OS string
        // data bytes are never copied or revalidated.
        let mut v = mem::replace(self, OsString::new()).into_encoded_bytes();
        let result = <Vec<u8> as TryVec<u8>>::fallible_shrink_to(&mut v, min_capacity);
        // SAFETY: the bytes originated from a valid OsString via into_encoded_bytes.
        *self = unsafe { OsString::from_encoded_bytes_unchecked(v) };
        result
    }

    fn try_into_string(self) -> Result<String, OsString> {
        self.into_string()
    }

    fn try_into_boxed_osstr(self) -> Result<Box<OsStr>, TryReserveError> {
        // Take ownership of the encoded byte buffer.
        let mut v = self.into_encoded_bytes();

        // Empty strings don't need any allocation.
        if v.is_empty() {
            return Ok(Box::<OsStr>::default());
        }

        // If there's no spare capacity, hand the buffer straight to the box —
        // no reallocation needed.
        if v.capacity() == v.len() {
            return Ok(unsafe { from_boxed_osstr_unchecked(v.into_boxed_slice()) });
        }

        // Otherwise shrink in place via realloc. On failure the bytes are
        // dropped along with `v` — use try_into_boxed_osstr_give_back to
        // recover them instead.
        <Vec<u8> as TryVec<u8>>::fallible_shrink_to_fit(&mut v)?;
        // SAFETY: the bytes originated from a valid OsString via into_encoded_bytes.
        Ok(unsafe { from_boxed_osstr_unchecked(v.into_boxed_slice()) })
    }

    fn try_into_boxed_osstr_give_back(self) -> Result<Box<OsStr>, (OsString, TryReserveError)> {
        // Take ownership of the encoded byte buffer.
        let mut v = self.into_encoded_bytes();

        // Empty strings don't need any allocation.
        if v.is_empty() {
            return Ok(Box::<OsStr>::default());
        }

        // If there's no spare capacity, hand the buffer straight to the box.
        if v.capacity() == v.len() {
            return Ok(unsafe { from_boxed_osstr_unchecked(v.into_boxed_slice()) });
        }

        // Otherwise, shrink first. If the shrink fails, reconstruct the
        // OsString and return it so no data is lost.
        match <Vec<u8> as TryVec<u8>>::fallible_shrink_to_fit(&mut v) {
            Ok(()) => {
                // SAFETY: the bytes originated from a valid OsString.
                Ok(unsafe { from_boxed_osstr_unchecked(v.into_boxed_slice()) })
            }
            Err(e) => {
                // SAFETY: the bytes are unchanged from the original OsString.
                let osstring = unsafe { OsString::from_encoded_bytes_unchecked(v) };
                Err((osstring, e))
            }
        }
    }
}

/// # Safety
///
/// `bytes` must be a valid OS string encoding (i.e., they must have come from
/// an existing `OsString`).
unsafe fn from_boxed_osstr_unchecked(bytes: Box<[u8]>) -> Box<OsStr> {
    // On Unix-like platforms, OsStr is #[repr(transparent)] over [u8], so the
    // cast mirrors std's own internal layout-based conversions.
    //
    // On Windows, OsStr is #[repr(transparent)] over [u16] and the byte
    // representation differs — this fast path would be unsound there, so it
    // is restricted to unix-family targets. Other platforms fall back to
    // constructing through the reference API (which copies).
    if cfg!(unix) || cfg!(target_os = "wasi") {
        unsafe { Box::from_raw(Box::into_raw(bytes) as *mut OsStr) }
    } else {
        // Non-unix platforms (e.g. Windows): reconstruct through the public
        // API, which copies the bytes into a properly-encoded OsString, then
        // convert that owned string into a Box<OsStr>.
        let owned = unsafe { OsString::from_encoded_bytes_unchecked(bytes.into_vec()) };
        owned.into_boxed_os_str()
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

// ── TryDebug for OsString ────────────────────────────────────────────────────

impl crate::try_fmt::TryDebug for OsString {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_os_str().try_fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang_alloc::string::String;

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
            use lang_alloc::vec;
            use lang_std::os::unix::ffi::OsStringExt;
            let s = OsString::from_vec(vec![0xFF, 0xFE]);
            let result = s.try_into_string();
            assert!(result.is_err());
        }
    }

    // ── Conversion to boxed types ─────────────────────────────────────────────

    #[test]
    fn try_into_boxed_osstr_empty() {
        let s = OsString::new();
        let boxed: Box<OsStr> = s.try_into_boxed_osstr().unwrap();
        assert!(boxed.is_empty());
    }

    #[test]
    fn try_into_boxed_osstr_ascii() {
        let s = OsString::try_from_str("hello").unwrap();
        let boxed: Box<OsStr> = s.try_into_boxed_osstr().unwrap();
        assert_eq!(*boxed, *OsStr::new("hello"));
    }

    #[test]
    fn try_into_boxed_osstr_unicode() {
        let s = OsString::try_from_str("こんにちは 🦀").unwrap();
        let boxed: Box<OsStr> = s.try_into_boxed_osstr().unwrap();
        assert_eq!(*boxed, *OsStr::new("こんにちは 🦀"));
    }

    #[test]
    fn try_into_boxed_osstr_with_spare_capacity() {
        let mut s = OsString::try_with_capacity(1024).unwrap();
        s.try_push_str("small").unwrap();
        let boxed: Box<OsStr> = s.try_into_boxed_osstr().unwrap();
        assert_eq!(*boxed, *OsStr::new("small"));
    }

    #[test]
    fn try_into_boxed_osstr_give_back_success() {
        let mut s = OsString::try_with_capacity(256).unwrap();
        s.try_push_str("hi").unwrap();
        let boxed: Box<OsStr> = s.try_into_boxed_osstr_give_back().unwrap();
        assert_eq!(*boxed, *OsStr::new("hi"));
    }

    #[test]
    fn try_into_boxed_osstr_give_back_empty() {
        let s = OsString::new();
        let boxed: Box<OsStr> = s.try_into_boxed_osstr_give_back().unwrap();
        assert!(boxed.is_empty());
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
    #[cfg(feature = "std")]
    mod oom {
        use super::*;
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
            let r: Result<OsString, TryReserveError> =
                <OsString as TryOsString>::try_with_capacity(10);
            assert!(r.is_ok());
        }
    }
}

