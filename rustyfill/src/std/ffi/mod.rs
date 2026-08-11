//! Fallible FFI string operations.
//!
//! Provides [`os_string::TryOsString`] for fallible construction and mutation of
//! `OsString` values, [`os_str::TryOsStr`] for fallible `&OsStr`-to-`OsString`
//! conversions, [`crate::alloc::ffi::TryCString`] for fallible `CString` construction,
//! and a [`TryToOwned`](crate::try_to_owned::TryToOwned) impl for `CStr`.
//! All return [`Result`] on allocation failure instead of panicking.

mod os_str;
mod os_string;

pub use crate::alloc::ffi::{TryCString, TryCStringError};
pub use os_str::{TryOsStr, TryOsStrError};
pub use os_string::{TryOsString, TryOsStringError};
