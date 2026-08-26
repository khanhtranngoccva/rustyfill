//! Fallible `&OsStr` operations.
//!
//! Provides the [`TryOsStr`] trait with methods that mirror allocating `&OsStr`
//! constructors but return [`Result`] to handle allocation failures gracefully.
//! Uses [`TryReserveError`](crate::alloc::TryReserveError) as the error type
//! for consistency with [`TryOsString`](super::os_string::TryOsString).

use crate::alloc::{TryReserveError, TryReserveErrorExt};
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use lang_alloc::alloc;
use lang_alloc::boxed::Box;
use lang_core::alloc::Layout;
use lang_core::fmt;
use lang_core::ptr;
use lang_std::ffi::{OsStr, OsString};

/// A trait for fallibly converting an `OsStr` slice into owned variants.
///
/// Implemented for `OsStr`. Methods reserve capacity upfront so that allocation
/// failures are returned as errors rather than panicking.
pub trait TryOsStr {
    /// Fallibly copy this `OsStr` into a new [`OsString`].
    ///
    /// This is the fallible analogue of [`OsStr::to_os_string`] and
    /// `OsStr::to_owned`. Reserves capacity for the full byte length before
    /// copying, so that allocation failures are caught cleanly.
    ///
    /// Returns [`TryReserveError`] on allocation failure.
    fn try_to_os_string(&self) -> Result<OsString, TryReserveError>;

    /// Fallibly convert ASCII characters in this `OsStr` to uppercase,
    /// returning a new [`OsString`].
    ///
    /// Mirrors [`OsStr::to_ascii_uppercase`] but reserves capacity upfront so
    /// that allocation failures return [`TryReserveError`] instead of
    /// panicking.
    fn try_to_ascii_uppercase(&self) -> Result<OsString, TryReserveError>;

    /// Fallibly convert ASCII characters in this `OsStr` to lowercase,
    /// returning a new [`OsString`].
    ///
    /// Mirrors [`OsStr::to_ascii_lowercase`] but reserves capacity upfront so
    /// that allocation failures return [`TryReserveError`] instead of
    /// panicking.
    fn try_to_ascii_lowercase(&self) -> Result<OsString, TryReserveError>;

    // ── Aliases with `fallible_` prefix ─────────────────────────────────────

    /// Alias for [`Self::try_to_os_string`].
    fn fallible_to_os_string(&self) -> Result<OsString, TryReserveError> {
        Self::try_to_os_string(self)
    }

    /// Alias for [`Self::try_to_ascii_uppercase`].
    fn fallible_to_ascii_uppercase(&self) -> Result<OsString, TryReserveError> {
        Self::try_to_ascii_uppercase(self)
    }

    /// Alias for [`Self::try_to_ascii_lowercase`].
    fn fallible_to_ascii_lowercase(&self) -> Result<OsString, TryReserveError> {
        Self::try_to_ascii_lowercase(self)
    }
}

impl TryOsStr for OsStr {
    fn try_to_os_string(&self) -> Result<OsString, TryReserveError> {
        let mut out = OsString::new();
        if !self.is_empty() {
            out.try_reserve(self.len())?;
        }
        out.push(self);
        Ok(out)
    }

    fn try_to_ascii_uppercase(&self) -> Result<OsString, TryReserveError> {
        let mut out = self.try_to_os_string()?;
        out.make_ascii_uppercase();
        Ok(out)
    }

    fn try_to_ascii_lowercase(&self) -> Result<OsString, TryReserveError> {
        let mut out = self.try_to_os_string()?;
        out.make_ascii_lowercase();
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// TryToOwned impl for OsStr
// ---------------------------------------------------------------------------

use crate::try_to_owned::{TryToOwned, TryToOwnedError};

impl TryToOwned for OsStr {
    fn try_to_owned(&self) -> Result<OsString, TryToOwnedError> {
        let mut out = OsString::new();
        if !self.is_empty() {
            out.try_reserve(self.len())?;
        }
        out.push(self);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// TryDebug for OsStr
// ---------------------------------------------------------------------------

impl crate::try_fmt::TryDebug for OsStr {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Mirrors std's Debug impl for OsStr which shows encoded bytes.
        fmt::Debug::fmt(self, f)
    }
}

// ---------------------------------------------------------------------------
// Boxed OsStr TryClone + TryDefault
// ---------------------------------------------------------------------------
// Box<OsStr> owns a dynamically-sized OsStr on the heap. Cloning goes through
// the platform-agnostic encoded-byte representation exposed by
// `as_encoded_bytes` / `from_encoded_bytes_unchecked`: allocate the exact byte
// length, copy the bytes with a single memcpy-style pass (no per-element
// try_clone), then rebuild the box via the unchecked conversion. This works
// identically on every platform — unix (UTF-8-backed) and Windows
// (WTF-8-backed) alike — because the round-trip preserves whatever encoding
// the source was stored in. It avoids both the overshoot of reserving through
// an intermediate OsString and the cost of routing each element through Clone.

/// # Safety
///
/// `bytes` must be a valid OS string encoding (i.e., they must have come from
/// an existing `OsString`).
unsafe fn from_boxed_osstr_unchecked(bytes: Box<[u8]>) -> Box<OsStr> {
    // Reconstruct the OsString from the raw encoded bytes (no validation —
    // they came from a valid OsStr), then hand its buffer straight to the box.
    // Capacity equals length, so no shrink reallocation occurs.
    let owned = unsafe { OsString::from_encoded_bytes_unchecked(bytes.into_vec()) };
    owned.into_boxed_os_str()
}

impl TryClone for Box<OsStr> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // View the contents as their platform-specific encoded bytes. On unix
        // this is a zero-cost fat-pointer reinterpretation; on Windows it
        // borrows the WTF-8 bytes held inside the OsStr. Either way no data is
        // copied or validated.
        let bytes = self.as_encoded_bytes();
        let len = bytes.len();

        // Empty string — no allocation needed.
        if len == 0 {
            return Ok(Box::<OsStr>::default());
        }

        // Allocate exactly `len` bytes — no excess capacity.
        // Layout::array handles overflow checking internally.
        let layout = Layout::array::<u8>(len)
            .map_err(|_| TryCloneError::Reserve(TryReserveErrorExt::new_capacity_overflow()))?;
        let ptr = unsafe { alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(TryCloneError::Reserve(TryReserveErrorExt::new_alloc(
                layout,
            )));
        }

        // Wrap immediately in a Box<[u8]> so that Drop cleans up on any panic
        // between here and the final conversion to Box<OsStr>.
        // SAFETY: layout matches `len` elements of u8.
        let mut out: Box<[u8]> = unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(ptr, len)) };

        // SAFETY: source is valid for `len` bytes, destination holds exactly
        // `len` initialized bytes.
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr(), len);
        }

        // SAFETY: the bytes were copied verbatim from a valid OsStr, so they
        // form a valid OS string encoding.
        Ok(unsafe { from_boxed_osstr_unchecked(out) })
    }
}

impl TryDefault for Box<OsStr> {
    fn try_default() -> Result<Self, TryDefaultError> {
        // An empty boxed OsStr requires no allocation.
        Ok(Box::<OsStr>::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang_std::borrow::ToOwned;

    // ── try_to_os_string ─────────────────────────────────────────────────────

    #[test]
    fn try_to_os_string_empty() {
        let s = OsStr::new("");
        let owned = s.try_to_os_string().unwrap();
        assert!(owned.is_empty());
    }

    #[test]
    fn try_to_os_string_ascii() {
        let s = OsStr::new("hello");
        let owned = s.try_to_os_string().unwrap();
        assert_eq!(owned, OsString::from("hello"));
    }

    #[test]
    fn try_to_os_string_unicode() {
        let s = OsStr::new("こんにちは 🦀");
        let owned = s.try_to_os_string().unwrap();
        assert_eq!(owned, OsString::from("こんにちは 🦀"));
    }

    #[test]
    fn try_to_os_string_long() {
        let long_str = "x".repeat(100_000);
        let long = OsStr::new(&long_str);
        let owned = long.try_to_os_string().unwrap();
        assert_eq!(owned.len(), 100_000);
    }

    // ── try_to_ascii_uppercase ───────────────────────────────────────────────

    #[test]
    fn try_to_ascii_uppercase_empty() {
        let s = OsStr::new("");
        let r = s.try_to_ascii_uppercase().unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn try_to_ascii_uppercase_lower() {
        let s = OsStr::new("hello world");
        let r = s.try_to_ascii_uppercase().unwrap();
        assert_eq!(r, OsString::from("HELLO WORLD"));
    }

    #[test]
    fn try_to_ascii_uppercase_mixed() {
        let s = OsStr::new("HeLLo WoRLd");
        let r = s.try_to_ascii_uppercase().unwrap();
        assert_eq!(r, OsString::from("HELLO WORLD"));
    }

    #[test]
    fn try_to_ascii_uppercase_already_upper() {
        let s = OsStr::new("ALREADY");
        let r = s.try_to_ascii_uppercase().unwrap();
        assert_eq!(r, OsString::from("ALREADY"));
    }

    #[test]
    fn try_to_ascii_uppercase_preserves_non_ascii() {
        let s = OsStr::new("αβγ hello");
        let r = s.try_to_ascii_uppercase().unwrap();
        assert_eq!(r, OsString::from("αβγ HELLO"));
    }

    // ── try_to_ascii_lowercase ───────────────────────────────────────────────

    #[test]
    fn try_to_ascii_lowercase_empty() {
        let s = OsStr::new("");
        let r = s.try_to_ascii_lowercase().unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn try_to_ascii_lowercase_upper() {
        let s = OsStr::new("HELLO WORLD");
        let r = s.try_to_ascii_lowercase().unwrap();
        assert_eq!(r, OsString::from("hello world"));
    }

    #[test]
    fn try_to_ascii_lowercase_mixed() {
        let s = OsStr::new("HeLLo WoRLd");
        let r = s.try_to_ascii_lowercase().unwrap();
        assert_eq!(r, OsString::from("hello world"));
    }

    #[test]
    fn try_to_ascii_lowercase_already_lower() {
        let s = OsStr::new("already");
        let r = s.try_to_ascii_lowercase().unwrap();
        assert_eq!(r, OsString::from("already"));
    }

    #[test]
    fn try_to_ascii_lowercase_preserves_non_ascii() {
        let s = OsStr::new("ΩΔΓ hello");
        let r = s.try_to_ascii_lowercase().unwrap();
        assert_eq!(r, OsString::from("ΩΔΓ hello"));
    }

    // ── matches std behaviour ────────────────────────────────────────────────

    #[test]
    fn try_to_ascii_uppercase_matches_std() {
        let s = OsStr::new("foo bar baz");
        let expected = s.to_ascii_uppercase();
        let actual = s.try_to_ascii_uppercase().unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn try_to_ascii_lowercase_matches_std() {
        let s = OsStr::new("FOO BAR BAZ");
        let expected = s.to_ascii_lowercase();
        let actual = s.try_to_ascii_lowercase().unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn try_to_os_string_matches_std() {
        let s = OsStr::new("rust test");
        let expected = s.to_os_string();
        let actual = s.try_to_os_string().unwrap();
        assert_eq!(actual, expected);
    }

    // ── TryToOwned tests ─────────────────────────────────────────────────────

    #[test]
    fn try_to_owned_empty() {
        let s = OsStr::new("");
        let owned: OsString = s.try_to_owned().unwrap();
        assert!(owned.is_empty());
    }

    #[test]
    fn try_to_owned_ascii() {
        let s = OsStr::new("rust");
        let owned: OsString = s.try_to_owned().unwrap();
        assert_eq!(owned, OsString::from("rust"));
    }

    #[test]
    fn try_to_owned_unicode() {
        let s = OsStr::new("日本語");
        let owned: OsString = s.try_to_owned().unwrap();
        assert_eq!(owned, OsString::from("日本語"));
    }

    #[test]
    fn try_to_owned_implies_to_owned_bound() {
        let s = OsStr::new("test");
        let owned: OsString = <OsStr as ToOwned>::to_owned(s);
        assert_eq!(owned, OsString::from("test"));
    }

    // ── Boxed OsStr TryClone + TryDefault ─────────────────────────────────────

    #[test]
    fn boxed_osstr_try_clone_ascii() {
        let b: Box<OsStr> = OsString::from("hello").into_boxed_os_str();
        let c = b.try_clone().unwrap();
        assert_eq!(&*c, OsStr::new("hello"));
    }

    #[test]
    fn boxed_osstr_try_clone_unicode() {
        let b: Box<OsStr> = OsString::from("こんにちは 🦀").into_boxed_os_str();
        let c = b.try_clone().unwrap();
        assert_eq!(&*c, OsStr::new("こんにちは 🦀"));
    }

    #[test]
    fn boxed_osstr_try_clone_empty() {
        let b: Box<OsStr> = Box::<OsStr>::default();
        let c = b.try_clone().unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn boxed_osstr_try_default_empty() {
        let b: Box<OsStr> = Box::<OsStr>::try_default().unwrap();
        assert!(b.is_empty());
    }

    // ── OOM tests ─────────────────────────────────────────────────────
    #[cfg(feature = "std")]
    mod oom {
        use super::*;
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        #[test]
        fn boxed_osstr_try_clone_fails_on_oom() {
            let orig: Box<OsStr> = OsString::from("hello").into_boxed_os_str();
            let r: Result<Box<OsStr>, TryCloneError> =
                with_policy(FailPolicy::fail_next_alloc(), || orig.try_clone());
            assert!(r.is_err());
        }

        #[test]
        fn boxed_osstr_try_clone_empty_succeeds_under_oom() {
            let orig: Box<OsStr> = Box::<OsStr>::default();
            let r: Result<Box<OsStr>, TryCloneError> =
                with_policy(FailPolicy::fail_next_alloc(), || orig.try_clone());
            assert!(r.is_ok());
        }
    }
}
