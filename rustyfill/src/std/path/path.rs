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

use crate::alloc::TryReserveError;
use crate::std::path::TryPathBuf;
use crate::std::path::path_buf::TryPathBufAddExtensionError;
use crate::std::path::path_buf::inner_push;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::{TryDebug, TryDisplay, helpers::FormatterExt};
use crate::try_to_owned::{TryToOwned, TryToOwnedError};
use lang_alloc::boxed::Box;
use lang_core::fmt;
use lang_std::ffi::OsStr;
use lang_std::path::Display;
use lang_std::path::{Path, PathBuf};

/// Error returned by [`TryPath::try_with_added_extension`].
pub enum TryPathWithAddedExtensionError {
    /// A capacity reservation failed (overflow or OOM).
    Reserve(TryReserveError),
    /// The provided extension contains a path separator.
    SeparatorInPath,
}

impl fmt::Debug for TryPathWithAddedExtensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(self, f)
    }
}

impl fmt::Display for TryPathWithAddedExtensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }
}

impl From<TryReserveError> for TryPathWithAddedExtensionError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl TryDebug for TryPathWithAddedExtensionError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => f
                .try_debug_tuple("TryPathWithAddedExtensionError::Reserve")
                .field(e)
                .finish(),
            Self::SeparatorInPath => f
                .try_debug_tuple("TryPathWithAddedExtensionError::SeparatorInPath")
                .finish(),
        }
    }
}

impl TryDisplay for TryPathWithAddedExtensionError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "Path add-extension failed: {}", e),
            Self::SeparatorInPath => write!(f, "extension cannot contain path separators"),
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
    /// Returns [`TryReserveError`] on allocation failure.
    fn try_to_path_buf(&self) -> Result<PathBuf, TryReserveError>;

    /// Fallibly join another path component onto this `Path`, returning a new
    /// [`PathBuf`].
    ///
    /// If `child` is absolute, it becomes the entire result. Otherwise it is
    /// appended with a platform-specific separator.
    ///
    /// Mirrors [`Path::join`] but reserves capacity upfront so that allocation
    /// failures return [`TryReserveError`] instead of panicking.
    fn try_join<P: AsRef<Path>>(&self, child: P) -> Result<PathBuf, TryReserveError>;

    /// Fallibly produce a new [`PathBuf`] with an extension appended to the
    /// file name.
    ///
    /// Unlike converting via [`try_to_path_buf`](Self::try_to_path_buf) then
    /// calling `set_extension`, this appends `.ext` to whatever filename
    /// currently exists, even if it already has an extension. For example,
    /// `foo.tar.gz.try_with_added_extension("xz")` yields `foo.tar.gz.xz`.
    ///
    /// Fails with [`TryPathWithAddedExtensionError::Reserve`] on allocation
    /// failure, or [`TryPathWithAddedExtensionError::SeparatorInPath`] if `ext`
    /// contains path separators.
    fn try_with_added_extension<E: AsRef<OsStr>>(
        &self,
        ext: E,
    ) -> Result<PathBuf, TryPathWithAddedExtensionError>;

    // ── Aliases with `fallible_` prefix ────────────────────────────────────

    /// Alias for [`Self::try_to_path_buf`].
    fn fallible_to_path_buf(&self) -> Result<PathBuf, TryReserveError> {
        Self::try_to_path_buf(self)
    }

    /// Alias for [`Self::try_join`].
    fn fallible_join<P: AsRef<Path>>(&self, child: P) -> Result<PathBuf, TryReserveError> {
        Self::try_join(self, child)
    }

    /// Alias for [`Self::try_with_added_extension`].
    fn fallible_with_added_extension<E: AsRef<OsStr>>(
        &self,
        ext: E,
    ) -> Result<PathBuf, TryPathWithAddedExtensionError> {
        Self::try_with_added_extension(self, ext)
    }
}

// ---------------------------------------------------------------------------
// Internal helper: build a PathBuf by manually pushing into the inner OsString
// ---------------------------------------------------------------------------

/// Builds a `PathBuf` from `base` with `child` appended, delegating to
/// [`inner_push`](crate::path::path_buf::inner_push) so that both `try_join`
/// and `try_push` share the same platform-aware logic.
fn inner_join(base: &Path, child: &Path) -> Result<PathBuf, TryReserveError> {
    let mut out = base.try_to_path_buf()?;
    inner_push(&mut out, child)?;
    Ok(out)
}

impl TryPath for Path {
    fn try_to_path_buf(&self) -> Result<PathBuf, TryReserveError> {
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

    fn try_join<P: AsRef<Path>>(&self, child: P) -> Result<PathBuf, TryReserveError> {
        inner_join(self, child.as_ref())
    }

    fn try_with_added_extension<E: AsRef<OsStr>>(
        &self,
        ext: E,
    ) -> Result<PathBuf, TryPathWithAddedExtensionError> {
        let mut out = self
            .try_to_path_buf()
            .map_err(TryPathWithAddedExtensionError::Reserve)?;
        out.try_add_extension(ext).map_err(|e| match e {
            TryPathBufAddExtensionError::Reserve(r) => TryPathWithAddedExtensionError::Reserve(r),
            TryPathBufAddExtensionError::SeparatorInPath => {
                TryPathWithAddedExtensionError::SeparatorInPath
            }
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
// Boxed Path TryClone + TryDefault
// ---------------------------------------------------------------------------
// Box<Path> owns a dynamically-sized Path on the heap. Since Path is
// repr(transparent) over OsStr, we reinterpret the box as Box<OsStr>, clone it
// via the customized exact-allocation implementation in ffi::os_str, then cast
// the result back — no intermediate PathBuf/OsString construction and no
// overshoot of capacity.

/// # Safety
///
/// `boxed` must be a valid `Box<OsStr>` whose bytes came from an existing
/// `OsString`.
unsafe fn from_boxed_osstr_to_boxed_path(boxed: Box<OsStr>) -> Box<Path> {
    // Path is #[repr(transparent)] over OsStr, so this cast mirrors std's own
    // layout-based conversions between the two types.
    unsafe { lang_core::mem::transmute::<Box<OsStr>, Box<Path>>(boxed) }
}

impl TryClone for Box<Path> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Reinterpret as Box<OsStr> (Path is #[repr(transparent)] over it),
        // clone, then cast the result back to Box<Path>.
        let boxed_os = unsafe { lang_core::mem::transmute::<&Self, &Box<OsStr>>(self) };
        // SAFETY: Path and OsStr are repr(transparent) with identical layout.
        let cloned = boxed_os.try_clone()?;
        Ok(unsafe { from_boxed_osstr_to_boxed_path(cloned) })
    }
}

impl TryDefault for Box<Path> {
    fn try_default() -> Result<Self, TryDefaultError> {
        // An empty path requires no allocation.
        Ok(unsafe { from_boxed_osstr_to_boxed_path(Box::<OsStr>::default()) })
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

    // ── Boxed Path TryClone + TryDefault ──────────────────────────────────────

    #[test]
    fn boxed_path_try_clone_simple() {
        let p = PathBuf::from("/usr/local/bin");
        let boxed: Box<Path> = p.try_into_boxed_path().unwrap();
        let c = boxed.try_clone().unwrap();
        assert_eq!(&*c, Path::new("/usr/local/bin"));
    }

    #[test]
    fn boxed_path_try_clone_unicode() {
        let p = PathBuf::from("/home/ユーザー/docs");
        let boxed: Box<Path> = p.try_into_boxed_path().unwrap();
        let c = boxed.try_clone().unwrap();
        assert_eq!(&*c, Path::new("/home/ユーザー/docs"));
    }

    #[test]
    fn boxed_path_try_clone_empty() {
        let p = PathBuf::new();
        let boxed: Box<Path> = p.try_into_boxed_path().unwrap();
        let c = boxed.try_clone().unwrap();
        assert!(c.as_os_str().is_empty());
    }

    #[test]
    fn boxed_path_try_default_empty() {
        let b: Box<Path> = Box::<Path>::try_default().unwrap();
        assert!(b.as_os_str().is_empty());
    }

    // ── OOM tests ─────────────────────────────────────────────────────
    #[cfg(feature = "std")]
    mod oom {
        use super::*;
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        #[test]
        fn boxed_path_try_clone_fails_on_oom() {
            let orig: Box<Path> = PathBuf::from("/some/path").try_into_boxed_path().unwrap();
            let r: Result<Box<Path>, TryCloneError> =
                with_policy(FailPolicy::fail_next_alloc(), || orig.try_clone());
            assert!(r.is_err());
        }

        #[test]
        fn boxed_path_try_clone_empty_succeeds_under_oom() {
            let orig: Box<Path> = PathBuf::new().try_into_boxed_path().unwrap();
            let r: Result<Box<Path>, TryCloneError> =
                with_policy(FailPolicy::fail_next_alloc(), || orig.try_clone());
            assert!(r.is_ok());
        }
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
    // Component-based assertions: verify the file_name component of the result,
    // which is platform-independent and doesn't duplicate the algorithm.

    #[test]
    fn with_added_ext_simple() {
        let p = Path::new("notes").try_with_added_extension("txt").unwrap();
        assert_eq!(p.file_name().and_then(|f| f.to_str()), Some("notes.txt"));

        let p = Path::new("/tmp/file")
            .try_with_added_extension("log")
            .unwrap();
        assert_eq!(p.file_name().and_then(|f| f.to_str()), Some("file.log"));
    }

    #[test]
    fn with_added_ext_appends_after_filename() {
        // Semantics: truncate at end of file_name, then push ".<ext>".
        // The existing extension is preserved; the new one is appended.
        let p = Path::new("foo.tar.gz")
            .try_with_added_extension("xz")
            .unwrap();
        assert_eq!(
            p.file_name().and_then(|f| f.to_str()),
            Some("foo.tar.gz.xz")
        );

        let p = Path::new("archive.zip.bak")
            .try_with_added_extension("enc")
            .unwrap();
        assert_eq!(
            p.file_name().and_then(|f| f.to_str()),
            Some("archive.zip.bak.enc")
        );
    }

    #[test]
    fn with_added_ext_empty_noop() {
        let p = Path::new("foo.txt").try_with_added_extension("").unwrap();
        assert_eq!(p.file_name().and_then(|f| f.to_str()), Some("foo.txt"));
    }

    #[test]
    fn with_added_ext_no_filename() {
        // No file_name component → path should be unchanged.
        let p = Path::new("/").try_with_added_extension("txt").unwrap();
        assert!(p.file_name().is_none());
        let p = Path::new("..").try_with_added_extension("txt").unwrap();
        assert!(p.file_name().is_none());
    }

    #[test]
    fn with_added_ext_unicode() {
        let p = Path::new("/docs/日本語ファイル")
            .try_with_added_extension("txt")
            .unwrap();
        assert_eq!(
            p.file_name().and_then(|f| f.to_str()),
            Some("日本語ファイル.txt")
        );
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
