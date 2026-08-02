//! Fallible `&CStr` operations.
//!
//! Provides the [`TryToOwned`] implementation for `CStr`, enabling fallible
//! conversion of a `&CStr` into an owned [`CString`].

use crate::try_to_owned::{TryToOwned, TryToOwnedError};
use crate::vec::{TrySlice, TryVecError};
use std::ffi::{CStr, CString};

impl TryToOwned for CStr {
    fn try_to_owned(&self) -> Result<CString, TryToOwnedError> {
        // Clone the bytes including the trailing nul fallibly, then hand the
        // vec directly to from_vec_with_nul_unchecked — no extra reserve or push.
        let bytes = self.to_bytes_with_nul();
        // to_bytes_with_nul always returns at least one byte (the nul), so it's
        // never empty. An empty CStr is b"\0".
        let buf = bytes.try_to_vec()
            .map_err(|e| match e {
                TryVecError::Reserve(r) => TryToOwnedError::Reserve(r),
                TryVecError::Clone(c) => c.into(),
                TryVecError::Overflow => TryToOwnedError::Overflow,
                TryVecError::Alloc(_) => TryToOwnedError::Alloc(
                    crate::alloc::AllocError,
                ),
                TryVecError::Other(m) => TryToOwnedError::Other(m),
            })?;

        // SAFETY: The bytes came from a valid CStr so there are no interior
        // nul bytes, and the slice already ends with exactly one nul.
        Ok(unsafe { CString::from_vec_with_nul_unchecked(buf) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_to_owned_empty() {
        let c = c"";
        let owned: CString = c.try_to_owned().unwrap();
        assert_eq!(owned.to_str().unwrap(), "");
    }

    #[test]
    fn try_to_owned_ascii() {
        let c = c"hello";
        let owned: CString = c.try_to_owned().unwrap();
        assert_eq!(owned.to_str().unwrap(), "hello");
    }

    #[test]
    fn try_to_owned_unicode_utf8() {
        let c = c"日本語";
        let owned: CString = c.try_to_owned().unwrap();
        assert_eq!(owned.to_str().unwrap(), "日本語");
    }

    #[test]
    fn try_to_owned_matches_std() {
        let c = c"rust test";
        let expected = <CStr as std::borrow::ToOwned>::to_owned(c);
        let actual: CString = c.try_to_owned().unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn try_to_owned_implies_to_owned_bound() {
        let c = c"test";
        let owned: CString = <CStr as std::borrow::ToOwned>::to_owned(c);
        assert_eq!(owned.to_str().unwrap(), "test");
    }
}
