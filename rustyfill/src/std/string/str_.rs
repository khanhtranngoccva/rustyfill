//! Fallible `&str` operations.
//!
//! Provides the [`TryStr`] trait with methods that mirror allocating `&str`
//! constructors but return [`Result`] to handle allocation failures gracefully.
//! Uses [`TryReserveError`](crate::alloc::TryReserveError) as the error type
//! for consistency with [`TryString`](super::string_::TryString).

use crate::alloc::{AllocError, TryReserveError};
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use core::alloc::Layout;
use core::fmt;

/// Error returned by [`TryStr`] operations.
#[derive(Debug)]
pub enum TryStrError {
    /// A raw heap allocation failed (no collection involved).
    Alloc(AllocError),
    /// A capacity reservation on the string failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An arithmetic overflow occurred while computing required capacity.
    Overflow,
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Display for TryStrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "str operation failed: heap allocation error"),
            Self::Reserve(e) => write!(f, "str operation failed: {}", e),
            Self::Overflow => write!(f, "str operation failed: capacity calculation overflowed"),
            Self::Other(msg) => write!(f, "str operation failed: {}", msg),
        }
    }
}

impl From<AllocError> for TryStrError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

impl From<TryReserveError> for TryStrError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

/// A trait for fallibly converting a string slice into an owned [`String`].
///
/// Implemented for [`str`]. Methods reserve capacity upfront so that allocation
/// failures are returned as errors rather than panicking or aborting.
pub trait TryStr {
    /// Fallibly copy this string slice into a new [`String`].
    ///
    /// This is the fallible analogue of [`ToString::to_string`] and
    /// [`str::to_owned`]. Reserves capacity for the full byte length before
    /// copying, so that allocation failures are caught cleanly.
    ///
    /// Returns [`TryStrError::Reserve`] on allocation failure.
    fn try_to_string(&self) -> Result<String, TryStrError>;

    /// Fallibly repeat this string slice `n` times into a new [`String`].
    ///
    /// Mirrors [`str::repeat`] but reserves capacity upfront so that
    /// allocation failures return [`TryStrError::Reserve`] instead of panicking.
    ///
    /// Returns an empty `String` when `n == 0` or the slice is empty.
    /// Returns [`TryStrError::Overflow`] if `self.len() * n` overflows.
    fn try_repeat(&self, n: usize) -> Result<String, TryStrError>;

    // ── Aliases with `fallible_` prefix ─────────────────────────────────────

    /// Alias for [`Self::try_to_string`].
    fn fallible_to_string(&self) -> Result<String, TryStrError> {
        Self::try_to_string(self)
    }

    /// Alias for [`Self::try_repeat`].
    fn fallible_repeat(&self, n: usize) -> Result<String, TryStrError> {
        Self::try_repeat(self, n)
    }
}

impl TryStr for str {
    fn try_to_string(&self) -> Result<String, TryStrError> {
        let mut out = String::new();
        if !self.is_empty() {
            out.try_reserve(self.len())
                .map_err(|e| TryStrError::Reserve(e.into()))?;
        }
        out.push_str(self);
        Ok(out)
    }

    fn try_repeat(&self, n: usize) -> Result<String, TryStrError> {
        let len = self.len();
        if len == 0 || n == 0 {
            return Ok(String::new());
        }
        let total_len = len.checked_mul(n).ok_or(TryStrError::Overflow)?;
        let mut out = String::new();
        out.try_reserve(total_len)
            .map_err(|e| TryStrError::Reserve(e.into()))?;
        for _ in 0..n {
            out.push_str(self);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// TryToOwned impl for str
// ---------------------------------------------------------------------------

use crate::try_to_owned::{TryToOwned, TryToOwnedError};

impl TryToOwned for str {
    fn try_to_owned(&self) -> Result<String, TryToOwnedError> {
        let mut out = String::new();
        if !self.is_empty() {
            out.try_reserve(self.len())?;
        }
        out.push_str(self);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Boxed str TryClone + TryDefault
// ---------------------------------------------------------------------------

impl TryClone for Box<str> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let bytes = self.as_bytes();
        let len = bytes.len();

        // Empty string — no allocation needed.
        if len == 0 {
            return Ok(<String as Default>::default().into_boxed_str());
        }

        // Allocate exactly `len` bytes — no excess capacity.
        // Layout::array handles overflow checking internally.
        let layout = Layout::array::<u8>(len).map_err(|_| TryCloneError::Overflow)?;
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(TryCloneError::Alloc(AllocError { layout }));
        }

        // Wrap immediately in a Box<[u8]> so that Drop cleans up on any panic
        // between here and the final transmute to Box<str>.
        // SAFETY: layout matches `len` elements of u8.
        let mut out: Box<[u8]> =
            unsafe { Box::from_raw(core::ptr::slice_from_raw_parts_mut(ptr, len)) };

        // SAFETY: `bytes` is valid UTF-8 and lives for at least the duration of
        // this memcpy (borrowed from `self`). Destination has exactly `len` bytes.
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr(), len);
        }

        // SAFETY: we just copied valid UTF-8 bytes. `Box<[u8]>` and `Box<str>`
        // have identical memory layouts (fat pointer: data ptr + length).
        Ok(unsafe { core::mem::transmute::<Box<[u8]>, Box<str>>(out) })
    }
}

impl TryDefault for Box<str> {
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(<String as Default>::default().into_boxed_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── try_to_string ────────────────────────────────────────────────────────

    #[test]
    fn try_to_string_empty() {
        let s: &str = "";
        let owned = s.try_to_string().unwrap();
        assert!(owned.is_empty());
    }

    #[test]
    fn try_to_string_ascii() {
        let s: &str = "hello";
        let owned = s.try_to_string().unwrap();
        assert_eq!(owned, "hello");
    }

    #[test]
    fn try_to_string_unicode() {
        let s: &str = "こんにちは 🦀";
        let owned = s.try_to_string().unwrap();
        assert_eq!(owned, "こんにちは 🦀");
    }

    #[test]
    fn try_to_string_preserves_bytes() {
        let s: &str = "café ☕";
        let owned = s.try_to_string().unwrap();
        assert_eq!(s.as_bytes(), owned.as_bytes());
    }

    #[test]
    fn try_to_string_long() {
        let long = "x".repeat(100_000);
        let owned = long.as_str().try_to_string().unwrap();
        assert_eq!(owned.len(), 100_000);
    }

    // ── try_repeat ───────────────────────────────────────────────────────────

    #[test]
    fn try_repeat_zero_times() {
        let s: &str = "ab";
        let r = s.try_repeat(0).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn try_repeat_one_time() {
        let s: &str = "hi";
        let r = s.try_repeat(1).unwrap();
        assert_eq!(r, "hi");
    }

    #[test]
    fn try_repeat_multiple_times() {
        let s: &str = "ab";
        let r = s.try_repeat(3).unwrap();
        assert_eq!(r, "ababab");
    }

    #[test]
    fn try_repeat_empty_slice() {
        let s: &str = "";
        let r = s.try_repeat(5).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn try_repeat_single_char() {
        let s: &str = "x";
        let r = s.try_repeat(4).unwrap();
        assert_eq!(r, "xxxx");
    }

    #[test]
    fn try_repeat_unicode() {
        let s: &str = "αβ";
        let r = s.try_repeat(2).unwrap();
        assert_eq!(r, "αβαβ");
    }

    #[test]
    fn try_repeat_overflow() {
        let s: &str = "ab";
        let result: Result<String, TryStrError> = s.try_repeat(usize::MAX);
        assert!(matches!(result, Err(TryStrError::Overflow)));
    }

    #[test]
    fn try_repeat_matches_std() {
        let s: &str = "foo";
        let expected = s.repeat(5);
        let actual = s.try_repeat(5).unwrap();
        assert_eq!(actual, expected);
    }

    // ── TryToOwned tests ─────────────────────────────────────────────────────

    #[test]
    fn try_to_owned_empty() {
        let s: &str = "";
        let owned: String = s.try_to_owned().unwrap();
        assert!(owned.is_empty());
    }

    #[test]
    fn try_to_owned_ascii() {
        let s: &str = "rust";
        let owned: String = s.try_to_owned().unwrap();
        assert_eq!(owned, "rust");
    }

    #[test]
    fn try_to_owned_unicode() {
        let s: &str = "日本語";
        let owned: String = s.try_to_owned().unwrap();
        assert_eq!(owned, "日本語");
    }

    #[test]
    fn try_to_owned_implies_to_owned_bound() {
        let s: &str = "test";
        let owned: String = <str as std::borrow::ToOwned>::to_owned(s);
        assert_eq!(owned, "test");
    }
}
