//! Fallible `&OsStr` operations.
//!
//! Provides the [`TryOsStr`] trait with methods that mirror allocating `&OsStr`
//! constructors but return [`Result`] to handle allocation failures gracefully.
//! Uses [`TryReserveError`](crate::alloc::TryReserveError) as the error type
//! for consistency with [`TryOsString`](super::os_string::TryOsString).

use crate::alloc::{AllocError, TryReserveError};
use lang_core::fmt;
use lang_std::ffi::{OsStr, OsString};
use lang_std::ffi::os_str::Display;
use crate::try_fmt::{TryDebug, helpers::FormatterExt};

/// Error returned by [`TryOsStr`] operations.
#[derive(Debug)]
pub enum TryOsStrError {
    /// A raw heap allocation failed (no collection involved).
    Alloc(AllocError),
    /// A capacity reservation failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An arithmetic overflow occurred while computing required capacity.
    Overflow,
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Display for TryOsStrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "OsStr operation failed: heap allocation error"),
            Self::Reserve(e) => write!(f, "OsStr operation failed: {}", e),
            Self::Overflow => {
                write!(f, "OsStr operation failed: capacity calculation overflowed")
            }
            Self::Other(msg) => write!(f, "OsStr operation failed: {}", msg),
        }
    }
}

impl From<AllocError> for TryOsStrError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

impl From<TryReserveError> for TryOsStrError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl TryDebug for TryOsStrError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => f
                .try_debug_struct("TryOsStrError::Alloc")
                .field("0", e)
                .finish(),
            Self::Reserve(e) => f
                .try_debug_struct("TryOsStrError::Reserve")
                .field("0", e)
                .finish(),
            Self::Overflow => f.write_str("TryOsStrError::Overflow"),
            Self::Other(msg) => f
                .try_debug_struct("TryOsStrError::Other")
                .field("0", msg)
                .finish(),
        }
    }
}

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
    /// Returns [`TryOsStrError::Reserve`] on allocation failure.
    fn try_to_os_string(&self) -> Result<OsString, TryOsStrError>;

    /// Fallibly convert ASCII characters in this `OsStr` to uppercase,
    /// returning a new [`OsString`].
    ///
    /// Mirrors [`OsStr::to_ascii_uppercase`] but reserves capacity upfront so
    /// that allocation failures return [`TryOsStrError::Reserve`] instead of
    /// panicking.
    fn try_to_ascii_uppercase(&self) -> Result<OsString, TryOsStrError>;

    /// Fallibly convert ASCII characters in this `OsStr` to lowercase,
    /// returning a new [`OsString`].
    ///
    /// Mirrors [`OsStr::to_ascii_lowercase`] but reserves capacity upfront so
    /// that allocation failures return [`TryOsStrError::Reserve`] instead of
    /// panicking.
    fn try_to_ascii_lowercase(&self) -> Result<OsString, TryOsStrError>;

    // ── Aliases with `fallible_` prefix ─────────────────────────────────────

    /// Alias for [`Self::try_to_os_string`].
    fn fallible_to_os_string(&self) -> Result<OsString, TryOsStrError> {
        Self::try_to_os_string(self)
    }

    /// Alias for [`Self::try_to_ascii_uppercase`].
    fn fallible_to_ascii_uppercase(&self) -> Result<OsString, TryOsStrError> {
        Self::try_to_ascii_uppercase(self)
    }

    /// Alias for [`Self::try_to_ascii_lowercase`].
    fn fallible_to_ascii_lowercase(&self) -> Result<OsString, TryOsStrError> {
        Self::try_to_ascii_lowercase(self)
    }
}

impl TryOsStr for OsStr {
    fn try_to_os_string(&self) -> Result<OsString, TryOsStrError> {
        let mut out = OsString::new();
        if !self.is_empty() {
            out.try_reserve(self.len())
                .map_err(|e| TryOsStrError::Reserve(e.into()))?;
        }
        out.push(self);
        Ok(out)
    }

    fn try_to_ascii_uppercase(&self) -> Result<OsString, TryOsStrError> {
        let mut out = self.try_to_os_string()?;
        out.make_ascii_uppercase();
        Ok(out)
    }

    fn try_to_ascii_lowercase(&self) -> Result<OsString, TryOsStrError> {
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
        write!(f, "b\"")?;
        for &byte in self.as_encoded_bytes() {
            match byte {
                b'"' => write!(f, "\\\"")?,
                b'\\' => write!(f, "\\\\")?,
                b'\n' => write!(f, "\\n")?,
                b'\r' => write!(f, "\\r")?,
                b'\t' => write!(f, "\\t")?,
                b if (0x20..=0x7E).contains(&b) => write!(f, "{}", b as char)?,
                _ => write!(f, "\\x{:02x}", byte)?,
            }
        }
        write!(f, "\"")
    }
}

// ---------------------------------------------------------------------------
// TryDebug + TryDisplay for ::lang_std::ffi::os_str::Display<'_>
// ---------------------------------------------------------------------------
// ::lang_std::ffi::os_str::Display's canonical Debug and Display impls write the OsStr
// to the formatter without allocating. Safe to passthrough.

impl crate::try_fmt::TryDebug for Display<'_> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl crate::try_fmt::TryDisplay for Display<'_> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
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
}
