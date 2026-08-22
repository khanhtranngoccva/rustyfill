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
use crate::alloc::TryReserveError;
use crate::std::path::path_buf::inner_push;
use crate::std::path::{TryPathBuf, TryPathBufError};
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_fmt::{TryDebug, TryDisplay, helpers::FormatterExt};
use crate::try_to_owned::{TryToOwned, TryToOwnedError};
use lang_core::fmt;
use lang_std::ffi::OsStr;
use lang_std::path::Display;
use lang_std::path::{Path, PathBuf};

/// Error returned by [`TryPath`] operations.
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

impl fmt::Debug for TryPathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(self, f)
    }
}

impl fmt::Display for TryPathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }
}

impl From<AllocError> for TryPathError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

impl From<TryReserveError> for TryPathError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl TryDebug for TryPathError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => f
                .try_debug_tuple("TryPathError::Alloc")
                .field(e)
                .finish(),
            Self::Reserve(e) => f
                .try_debug_tuple("TryPathError::Reserve")
                .field(e)
                .finish(),
            Self::Overflow => f.write_str("TryPathError::Overflow"),
            Self::Other(msg) => f
                .try_debug_tuple("TryPathError::Other")
                .field(msg)
                .finish(),
        }
    }
}

impl TryDisplay for TryPathError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "Path operation failed: heap allocation error"),
            Self::Reserve(e) => write!(f, "Path operation failed: {}", e),
            Self::Overflow => {
                write!(f, "Path operation failed: capacity calculation overflowed")
            }
            Self::Other(msg) => write!(f, "Path operation failed: {}", msg),
        }
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
    /// `Path::to_owned`. Reserves capacity for the full byte length before
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

    /// Fallibly produce a new [`PathBuf`] with an extension appended to the
    /// file name.
    ///
    /// Unlike converting via [`try_to_path_buf`](Self::try_to_path_buf) then
    /// calling `set_extension`, this appends `.ext` to whatever filename
    /// currently exists, even if it already has an extension. For example,
    /// `foo.tar.gz.try_with_added_extension("xz")` yields `foo.tar.gz.xz`.
    ///
    /// Returns [`TryPathError::Reserve`] on allocation failure.
    /// Returns [`TryPathError::Other`] if `ext` contains path separators.
    fn try_with_added_extension<E: AsRef<OsStr>>(&self, ext: E) -> Result<PathBuf, TryPathError>;

    // ── Aliases with `fallible_` prefix ────────────────────────────────────

    /// Alias for [`Self::try_to_path_buf`].
    fn fallible_to_path_buf(&self) -> Result<PathBuf, TryPathError> {
        Self::try_to_path_buf(self)
    }

    /// Alias for [`Self::try_join`].
    fn fallible_join<P: AsRef<Path>>(&self, child: P) -> Result<PathBuf, TryPathError> {
        Self::try_join(self, child)
    }

    /// Alias for [`Self::try_with_added_extension`].
    fn fallible_with_added_extension<E: AsRef<OsStr>>(
        &self,
        ext: E,
    ) -> Result<PathBuf, TryPathError> {
        Self::try_with_added_extension(self, ext)
    }
}

// ---------------------------------------------------------------------------
// Internal helper: build a PathBuf by manually pushing into the inner OsString
// ---------------------------------------------------------------------------

/// Builds a `PathBuf` from `base` with `child` appended, delegating to
/// [`inner_push`](crate::path::path_buf::inner_push) so that both `try_join`
/// and `try_push` share the same platform-aware logic.
fn inner_join(base: &Path, child: &Path) -> Result<PathBuf, TryPathError> {
    let mut out = base.try_to_path_buf()?;
    inner_push(&mut out, child)?;
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
        inner_join(self, child.as_ref())
    }

    fn try_with_added_extension<E: AsRef<OsStr>>(&self, ext: E) -> Result<PathBuf, TryPathError> {
        let mut out = self.try_to_path_buf()?;
        out.try_add_extension(ext).map_err(|e| match e {
            TryPathBufError::Alloc(a) => TryPathError::Alloc(a),
            TryPathBufError::Reserve(r) => TryPathError::Reserve(r),
            TryPathBufError::Overflow => TryPathError::Overflow,
            TryPathBufError::Other(m) => TryPathError::Other(m),
        })?;
        Ok(out)
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

// ---------------------------------------------------------------------------
// TryDebug for &Path and PathBuf
// ---------------------------------------------------------------------------

impl crate::try_fmt::TryDebug for Path {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Delegate to std Debug impl for parity. Path's Debug impl writes a
        // best-effort UTF-8 representation without allocating on platforms
        // where OsStr is UTF-8 (Unix) or lossy-converts (Windows).
        fmt::Debug::fmt(self, f)
    }
}

impl crate::try_fmt::TryDebug for PathBuf {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// ---------------------------------------------------------------------------
// TryDebug + TryDisplay for ::lang_std::path::Display<'_>
// ---------------------------------------------------------------------------
// ::lang_std::path::Display's canonical Debug and Display impls write the path to the
// formatter without allocating. Safe to passthrough.

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
    use crate::std::path::TryPathBuf;
    use lang_alloc::vec::Vec;
    use lang_std::borrow::ToOwned;
    use lang_std::format;

    /// Assert that `try_join(base, child)` produces the same result as
    /// `Path::join(base, child)` on this platform.
    fn assert_join_matches_std(base: &str, child: &str) {
        let expected = Path::new(base).join(child);
        let actual = Path::new(base).try_join(child).unwrap();
        assert_eq!(
            expected, actual,
            "join mismatch for base={} child={}",
            base, child
        );
    }

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

    // ── Property tests: try_join matches Path::join ───────────────────────────

    #[test]
    fn join_relative_on_various_bases() {
        assert_join_matches_std("/tmp", "file.txt");
        assert_join_matches_std("/home/user", "projects/rustyfill");
        assert_join_matches_std("", "hello");
        assert_join_matches_std("a/b", "c/d");
    }

    #[test]
    fn join_absolute_replaces() {
        assert_join_matches_std("/tmp/foo", "/etc/passwd");
        assert_join_matches_std("relative", "/absolute");
    }

    #[test]
    fn join_empty_cases() {
        assert_join_matches_std("", "");
        assert_join_matches_std("/tmp", "");
        assert_join_matches_std("", "only-child");
    }

    #[test]
    fn join_trailing_separator() {
        assert_join_matches_std("/tmp/", "file.txt");
        assert_join_matches_std("a/b/", "c");
    }

    #[test]
    fn join_curdir_parentdir() {
        assert_join_matches_std("/tmp", ".");
        assert_join_matches_std("/tmp/a", "..");
        assert_join_matches_std("/tmp/a/b", "../..");
    }

    #[test]
    fn join_unicode() {
        assert_join_matches_std("/docs", "日本語ファイル.txt");
        assert_join_matches_std("/home/用户", "文档/文件");
    }

    #[test]
    fn join_deeply_nested() {
        let deep = (0..50)
            .map(|i| format!("level{i}"))
            .collect::<Vec<_>>()
            .join("/");
        assert_join_matches_std("/root", &deep);
    }

    // ── TryClone for &Path ─────────────────────────────────────────────────────

    #[test]
    fn try_clone_ref_returns_same_path() {
        let p = Path::new("/tmp/test");
        let r: &Path = p;
        let c: &Path = r.try_clone().unwrap();
        assert_eq!(c, p);
    }

    #[test]
    fn try_clone_ref_empty() {
        let r: &Path = Path::new("");
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
        let owned: PathBuf = <Path as ToOwned>::to_owned(p);
        assert_eq!(owned, Path::new("/test"));
    }

    // ── try_with_added_extension ─────────────────────────────────────────────

    fn assert_with_added_ext_matches_std(base: &str, ext: &str) {
        let expected = Path::new(base).with_added_extension(ext);
        let actual = Path::new(base).try_with_added_extension(ext).unwrap();
        assert_eq!(
            expected, actual,
            "with_added_extension mismatch for base={} ext={}",
            base, ext
        );
    }

    #[test]
    fn with_added_ext_simple() {
        assert_with_added_ext_matches_std("notes", "txt");
        assert_with_added_ext_matches_std("/tmp/file", "log");
    }

    #[test]
    fn with_added_ext_appends_to_existing() {
        assert_with_added_ext_matches_std("foo.tar.gz", "xz");
        assert_with_added_ext_matches_std("archive.zip.bak", "enc");
    }

    #[test]
    fn with_added_ext_empty_noop() {
        assert_with_added_ext_matches_std("foo.txt", "");
    }

    #[test]
    fn with_added_ext_no_filename() {
        assert_with_added_ext_matches_std("/", "txt");
        assert_with_added_ext_matches_std("..", "txt");
    }

    #[test]
    fn with_added_ext_unicode() {
        assert_with_added_ext_matches_std("/docs/日本語ファイル", "txt");
    }

    // ── Combined workflows ─────────────────────────────────────────────────────

    #[test]
    fn build_via_join_then_clone() {
        let base = Path::new("/home/user");
        let buf = base.try_join("projects/rustyfill").unwrap();
        let cloned = buf.try_clone().unwrap();
        assert_eq!(cloned, Path::new("/home/user/projects/rustyfill"));
    }

    #[test]
    fn to_path_buf_then_push() {
        let p = Path::new("/tmp");
        let mut buf = p.try_to_path_buf().unwrap();
        buf.try_push("output.log").unwrap();
        assert_eq!(buf, Path::new("/tmp/output.log"));
    }
}
