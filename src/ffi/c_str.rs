//! Fallible `&CStr` operations.
//!
//! Provides the [`TryToOwned`] implementation for `CStr`, enabling fallible
//! conversion of a `&CStr` into an owned [`CString`].

use crate::try_to_owned::{TryToOwned, TryToOwnedError};
use std::ffi::{CStr, CString};

impl TryToOwned for CStr {
    fn try_to_owned(&self) -> Result<CString, TryToOwnedError> {
        // Clone the bytes (excluding the trailing nul), then append the nul
        // terminator and construct via CString::from_vec_with_nul_unchecked.
        let mut buf = self.to_bytes().to_vec();
        if buf.try_reserve(1).is_err() {
            return Err(TryToOwnedError::Reserve({
                let mut v = Vec::<u8>::new();
                v.try_reserve(usize::MAX).unwrap_err()
            }));
        }
        buf.push(0);

        // SAFETY: The bytes came from a valid CStr so there are no interior
        // nul bytes, and we just appended the trailing nul.
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
