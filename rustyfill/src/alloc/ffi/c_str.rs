//! Fallible `&CStr` operations.
//!
//! Provides the [`TryToOwned`] implementation for `CStr`, enabling fallible
//! conversion of a `&CStr` into an owned [`CString`], plus [`TryClone`] and
//! [`TryDefault`](crate::try_default::TryDefault) for `Box<CStr>`.

use crate::alloc::vec::{TrySlice, TryVecWithCloneError};
use crate::alloc::{AllocError, TryReserveErrorExt};
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_to_owned::{TryToOwned, TryToOwnedError};
use lang_alloc::alloc;
use lang_alloc::boxed::Box;
use lang_alloc::ffi::CString;
use lang_core::alloc::Layout;
use lang_core::ffi::CStr;
use lang_core::fmt;
use lang_core::mem;
use lang_core::ptr;

impl TryToOwned for CStr {
    fn try_to_owned(&self) -> Result<CString, TryToOwnedError> {
        // Clone the bytes including the trailing nul fallibly, then hand the
        // vec directly to from_vec_with_nul_unchecked — no extra reserve or push.
        let bytes = self.to_bytes_with_nul();
        // to_bytes_with_nul always returns at least one byte (the nul), so it's
        // never empty. An empty CStr is b"\0".
        let buf = bytes.try_to_vec().map_err(|e| match e {
            TryVecWithCloneError::Reserve(r) => TryToOwnedError::Reserve(r),
            TryVecWithCloneError::Clone(c) => c.into(),
        })?;

        // SAFETY: The bytes came from a valid CStr so there are no interior
        // nul bytes, and the slice already ends with exactly one nul.
        Ok(unsafe { CString::from_vec_with_nul_unchecked(buf) })
    }
}

// ---------------------------------------------------------------------------
// TryDebug for CStr
// ---------------------------------------------------------------------------

impl crate::try_fmt::TryDebug for CStr {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Mirrors std's Debug impl for CStr which shows as a quoted UTF-8 string.
        fmt::Debug::fmt(self, f)
    }
}

// ---------------------------------------------------------------------------
// Boxed CStr TryClone + TryDefault
// ---------------------------------------------------------------------------
// Box<CStr> owns a NUL-terminated byte sequence on the heap. Cloning allocates
// exactly `len` bytes (including the trailing NUL), copies them, then rebuilds
// the box via from_bytes_with_nul_unchecked — no validation needed since the
// source is already a valid CStr.

impl TryClone for Box<CStr> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let bytes = self.to_bytes_with_nul();
        let len = bytes.len();

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
        // between here and the final conversion to Box<CStr>.
        // SAFETY: layout matches `len` elements of u8.
        let mut out: Box<[u8]> = unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(ptr, len)) };

        // SAFETY: `bytes` is valid for at least `len` bytes and the destination
        // has exactly `len` bytes of initialized memory.
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr(), len);
        }

        // Rebuild the boxed CStr. The bytes are guaranteed NUL-terminated with
        // no interior NULs because they came from a valid CStr, and
        // Box<[u8]> / Box<CStr> share the same fat-pointer layout.
        // SAFETY: see above.
        Ok(unsafe { mem::transmute::<Box<[u8]>, Box<CStr>>(out) })
    }
}

impl TryDefault for Box<CStr> {
    fn try_default() -> Result<Self, TryDefaultError> {
        // An empty CStr is just the single NUL byte — one tiny allocation.
        let bytes: &[u8] = b"\0";
        let layout = Layout::array::<u8>(1).expect("single-byte layout cannot fail");
        let ptr = unsafe { alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(TryDefaultError::Alloc(AllocError));
        }
        // SAFETY: layout matches 1 element of u8.
        let mut out: Box<[u8]> = unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(ptr, 1)) };
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr(), 1);
        }
        // SAFETY: the single byte is NUL, so this is a valid CStr, and
        // Box<[u8]> / Box<CStr> share the same fat-pointer layout.
        Ok(unsafe { mem::transmute::<Box<[u8]>, Box<CStr>>(out) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang_alloc::borrow::ToOwned;

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
        let expected = <CStr as ToOwned>::to_owned(c);
        let actual: CString = c.try_to_owned().unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn try_to_owned_implies_to_owned_bound() {
        let c = c"test";
        let owned: CString = <CStr as ToOwned>::to_owned(c);
        assert_eq!(owned.to_str().unwrap(), "test");
    }

    // ── Boxed CStr TryClone + TryDefault ───────────────────────────────────────

    #[test]
    fn boxed_cstr_try_clone_ascii() {
        let cs = CString::new("hello").unwrap();
        let b: Box<CStr> = cs.into_boxed_c_str();
        let c = b.try_clone().unwrap();
        assert_eq!(c.to_bytes_with_nul(), b"hello\0");
    }

    #[test]
    fn boxed_cstr_try_clone_unicode() {
        let cs = CString::new("日本語").unwrap();
        let b: Box<CStr> = cs.into_boxed_c_str();
        let c = b.try_clone().unwrap();
        assert_eq!(c.to_bytes_with_nul(), c"日本語".to_bytes_with_nul());
    }

    #[test]
    fn boxed_cstr_try_clone_empty() {
        let cs = CString::new("").unwrap();
        let b: Box<CStr> = cs.into_boxed_c_str();
        let c = b.try_clone().unwrap();
        assert_eq!(c.to_bytes_with_nul(), b"\0");
    }

    #[test]
    fn boxed_cstr_try_default_empty() {
        let b: Box<CStr> = Box::<CStr>::try_default().unwrap();
        assert_eq!(b.to_bytes_with_nul(), b"\0");
    }

    // ── OOM tests ─────────────────────────────────────────────────────
    #[cfg(feature = "std")]
    mod oom {
        use super::*;
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        #[test]
        fn boxed_cstr_try_clone_fails_on_oom() {
            let cs = CString::new("hello").unwrap();
            let orig: Box<CStr> = cs.into_boxed_c_str();
            let r: Result<Box<CStr>, TryCloneError> =
                with_policy(FailPolicy::fail_next_alloc(), || orig.try_clone());
            assert!(r.is_err());
        }

        #[test]
        fn boxed_cstr_try_default_fails_on_oom() {
            // An empty CStr still needs one byte allocated for the NUL.
            let r: Result<Box<CStr>, TryDefaultError> =
                with_policy(FailPolicy::fail_next_alloc(), Box::<CStr>::try_default);
            assert!(r.is_err());
        }
    }
}
