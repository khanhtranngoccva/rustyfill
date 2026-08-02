//! Fallible `Path` operations.
//!
//! Provides the [`TryPath`] trait with methods that mirror allocating `Path`
//! constructors but return [`Result`] to handle allocation failures gracefully.
//!
//! # Design
//!
//! `Path` is a borrowed (zero-copy) view into path data, analogous to `str` or
//! `OsStr`. Like those types, most `Path` operations are cheap and infallible.
//! The fallible operations are those that produce owned values ([`PathBuf`]),
//! which may require heap allocation. This trait mirrors the split between
//! [`TryStr`](crate::string::TryStr) / [`TryString`](crate::string::TryString)
//! and [`TryOsStr`](crate::ffi::TryOsStr) / [`TryOsString`](crate::ffi::TryOsString).
//!
//! The trait also implements [`TryClone`](crate::try_clone::TryClone) for
//! `&Path` (mirroring the existing `&str` impl) and
//! [`TryToOwned`](crate::try_to_owned::TryToOwned) for `Path`.

use crate::alloc::AllocError;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_to_owned::{TryToOwned, TryToOwnedError};
use core::fmt;
use std::collections::TryReserveError;
use std::path::{Path, PathBuf};

/// Error returned by [`TryPath`] operations.
#[derive(Debug)]
pub enum TryPathError {
    /// A raw heap allocation failed (no collection involved).
    Alloc(AllocError),
    /// A capacity reservation failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An arithmetic overflow occurred while computing required capacity.
    Overflow,
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Display for TryPathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "Path operation failed: heap allocation error"),
            Self::Reserve(e) => write!(f, "Path operation failed: {}", e),
            Self::Overflow => {
                write!(
                    f,
                    "Path operation failed: capacity calculation overflowed"
                )
            }
            Self::Other(msg) => write!(f, "Path operation failed: {}", msg),
        }
    }
}

impl From<AllocError> for TryPathError {
    fn from(_: AllocError) -> Self {
        Self::Alloc(AllocError)
    }
}

impl From<TryReserveError> for TryPathError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

/// A trait for fallibly converting a [`Path`] slice into an owned [`PathBuf`].
///
/// Implemented for `Path`. Methods reserve capacity upfront so that allocation
/// failures are returned as errors rather than panicking.
pub trait TryPath {
    /// Fallibly copy this `Path` into a new [`PathBuf`].
    ///
    /// This is the fallible analogue of [`Path::to_path_buf`] and
    /// [`Path::to_owned`]. Reserves capacity for the full byte length before
    /// copying, so that allocation failures are caught cleanly.
    ///
    /// Returns [`TryPathError::Reserve`] on allocation failure.
    fn try_to_path_buf(&self) -> Result<PathBuf, TryPathError>;

    /// Fallibly join another path component onto this `Path`, returning a new
    /// [`PathBuf`].
    ///
    /// If `child` is absolute, it becomes the entire result. Otherwise it is
    /// appended with a platform-specific separator.
    ///
    /// Mirrors [`Path::join`] but reserves capacity upfront so that allocation
    /// failures return [`TryPathError::Reserve`] instead of panicking.
    fn try_join<P: AsRef<Path>>(&self, child: P) -> Result<PathBuf, TryPathError>;
}

// ---------------------------------------------------------------------------
// Internal helper: build a PathBuf by manually pushing into the inner OsString
// ---------------------------------------------------------------------------

/// Builds a `PathBuf` from `base` with `child` appended, replicating the
/// logic of `PathBuf::_push` directly on the inner [`OsString`] so every
/// allocation step is guarded by a prior `try_reserve`.
fn os_join(base: &Path, child: &Path) -> Result<PathBuf, TryReserveError> {
    let mut out = PathBuf::new();
    let os = out.as_mut_os_string();

    let base_str = base.as_os_str();
    let child_str = child.as_os_str();

    if child.is_absolute() {
        // Absolute child replaces everything.
        let needed = child_str.len();
        if needed > 0 {
            os.try_reserve(needed)?;
        }
        os.push(child_str);
        return Ok(out);
    }

    // Relative child: copy base, maybe add separator, then append child.
    let base_len = base_str.len();
    // base is empty initially, so we first push base content
    if base_len > 0 {
        os.try_reserve(base_len)?;
        os.push(base_str);
    }

    // Check if we need a separator after base content.
    let encoded = os.as_encoded_bytes();
    let need_sep = encoded
        .last()
        .map(|&b| b != b'/' && b != b'\\')
        .unwrap_or(false);

    let sep_len: usize = if need_sep { 1 } else { 0 };
    let extra = sep_len.saturating_add(child_str.len());
    if extra > 0 {
        os.try_reserve(extra)?;
    }
    if need_sep {
        os.push("/");
    }
    os.push(child_str);
    Ok(out)
}

impl TryPath for Path {
    fn try_to_path_buf(&self) -> Result<PathBuf, TryPathError> {
        let mut out = PathBuf::new();
        let os = out.as_mut_os_string();
        let src = self.as_os_str();
        let len = src.len();
        if len > 0 {
            os.try_reserve(len).map_err(TryPathError::Reserve)?;
        }
        os.push(src);
        Ok(out)
    }

    fn try_join<P: AsRef<Path>>(&self, child: P) -> Result<PathBuf, TryPathError> {
        os_join(self, child.as_ref()).map_err(TryPathError::Reserve)
    }
}

// ---------------------------------------------------------------------------
// TryClone for &Path
// ---------------------------------------------------------------------------
// Path is !Sized and does not implement Clone, so we follow the same pattern
// as &[T] and &str: implement TryClone for the reference type.

impl TryClone for &Path {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // &Path is Copy-like — cloning just duplicates the reference.
        Ok(*self)
    }
}

// ---------------------------------------------------------------------------
// TryToOwned for Path
// ---------------------------------------------------------------------------

impl TryToOwned for Path {
    fn try_to_owned(&self) -> Result<PathBuf, TryToOwnedError> {
        let mut out = PathBuf::new();
        let os = out.as_mut_os_string();
        let src = self.as_os_str();
        let len = src.len();
        if len > 0 {
            os.try_reserve(len)?;
        }
        os.push(src);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::TryPathBuf;

    // ── try_to_path_buf ────────────────────────────────────────────────────────

    #[test]
    fn try_to_path_buf_empty() {
        let p = Path::new("");
        let owned = p.try_to_path_buf().unwrap();
        assert!(owned.as_os_str().is_empty());
    }

    #[test]
    fn try_to_path_buf_ascii() {
        let p = Path::new("/usr/local/bin");
        let owned = p.try_to_path_buf().unwrap();
        assert_eq!(owned, Path::new("/usr/local/bin"));
    }

    #[test]
    fn try_to_path_buf_unicode() {
        let p = Path::new("/home/ユーザー/docs");
        let owned = p.try_to_path_buf().unwrap();
        assert_eq!(owned, Path::new("/home/ユーザー/docs"));
    }

    #[test]
    fn try_to_path_buf_matches_std() {
        let p = Path::new("rust/test/path");
        let expected = p.to_path_buf();
        let actual = p.try_to_path_buf().unwrap();
        assert_eq!(actual, expected);
    }

    // ── try_join ───────────────────────────────────────────────────────────────

    #[test]
    fn try_join_relative() {
        let p = Path::new("/tmp");
        let r = p.try_join("file.txt").unwrap();
        assert_eq!(r, Path::new("/tmp/file.txt"));
    }

    #[test]
    fn try_join_absolute_replaces() {
        let p = Path::new("/tmp/foo");
        let r = p.try_join("/etc/passwd").unwrap();
        assert_eq!(r, Path::new("/etc/passwd"));
    }

    #[test]
    fn try_join_empty_base() {
        let p = Path::new("");
        let r = p.try_join("hello").unwrap();
        assert_eq!(r, Path::new("hello"));
    }

    #[test]
    fn try_join_empty_child() {
        let p = Path::new("/tmp");
        let r = p.try_join("").unwrap();
        assert_eq!(r, Path::new("/tmp"));
    }

    #[test]
    fn try_join_both_empty() {
        let p = Path::new("");
        let r = p.try_join("").unwrap();
        assert!(r.as_os_str().is_empty());
    }

    #[test]
    fn try_join_unicode() {
        let p = Path::new("/docs");
        let r = p.try_join("日本語ファイル.txt").unwrap();
        assert_eq!(r, Path::new("/docs/日本語ファイル.txt"));
    }

    #[test]
    fn try_join_matches_std() {
        let p = Path::new("/var/log");
        let expected = p.join("syslog");
        let actual = p.try_join("syslog").unwrap();
        assert_eq!(actual, expected);
    }

    // ── TryClone for &Path ─────────────────────────────────────────────────────

    #[test]
    fn try_clone_ref_returns_same_path() {
        let p = Path::new("/tmp/test");
        let r: &Path = &p;
        let c: &Path = r.try_clone().unwrap();
        assert_eq!(c, p);
    }

    #[test]
    fn try_clone_ref_empty() {
        let p = Path::new("");
        let r: &Path = &p;
        let c: &Path = r.try_clone().unwrap();
        assert!(c.as_os_str().is_empty());
    }

    // ── TryToOwned ─────────────────────────────────────────────────────────────

    #[test]
    fn try_to_owned_empty() {
        let p = Path::new("");
        let owned: PathBuf = p.try_to_owned().unwrap();
        assert!(owned.as_os_str().is_empty());
    }

    #[test]
    fn try_to_owned_ascii() {
        let p = Path::new("/var/run");
        let owned: PathBuf = p.try_to_owned().unwrap();
        assert_eq!(owned, Path::new("/var/run"));
    }

    #[test]
    fn try_to_owned_unicode() {
        let p = Path::new("/data/日本語");
        let owned: PathBuf = p.try_to_owned().unwrap();
        assert_eq!(owned, Path::new("/data/日本語"));
    }

    #[test]
    fn try_to_owned_implies_to_owned_bound() {
        let p = Path::new("/test");
        let owned: PathBuf = <Path as std::borrow::ToOwned>::to_owned(p);
        assert_eq!(owned, Path::new("/test"));
    }

    // ── Combined workflows ─────────────────────────────────────────────────────

    #[test]
    fn build_via_join_then_clone() {
        let base = Path::new("/home/user");
        let buf = base.try_join("projects/fallibles").unwrap();
        let cloned = buf.try_clone().unwrap();
        assert_eq!(cloned, Path::new("/home/user/projects/fallibles"));
    }

    #[test]
    fn to_path_buf_then_push() {
        let p = Path::new("/tmp");
        let mut buf = p.try_to_path_buf().unwrap();
        buf.try_push("output.log").unwrap();
        assert_eq!(buf, Path::new("/tmp/output.log"));
    }
}
