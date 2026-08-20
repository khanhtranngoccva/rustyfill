//! Fallible `PathBuf` operations.
//!
//! Provides the [`TryPathBuf`] trait with methods that mirror common `PathBuf`
//! constructors and mutating operations but return [`Result`] to handle allocation
//! failures gracefully.
//!
//! # Design
//!
//! `PathBuf` wraps an [`OsString`](::lang_std::ffi::OsString) internally, so its fallible
//! operations delegate to the same reserve-before-mutate pattern used by
//! [`TryOsString`](crate::ffi::TryOsString). Methods that may grow internal capacity
//! (`push`, etc.) call `try_reserve` first so that allocation failures surface as
//! errors rather than panics.
//!
//! The trait also implements [`TryClone`](crate::try_clone::TryClone) and
//! [`TryDefault`](crate::try_default::TryDefault) for `PathBuf`.

use crate::alloc::AllocError;
use crate::alloc::TryReserveError;
use crate::alloc::vec::TryVec;
use crate::std::ffi::TryOsString;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::{TryDebug, helpers::FormatterExt};
use lang_alloc::vec::Vec;
use lang_core::fmt;
use lang_core::mem;
use lang_std::ffi::{OsStr, OsString};
use lang_std::path::{Component, MAIN_SEPARATOR_STR, Path, PathBuf, Prefix, is_separator};

/// Error returned by [`TryPathBuf`] operations.
///
/// Wraps the ways a `PathBuf` operation can fail on stable Rust: a reserve
/// failure ([`TryReserveError`]) or an arithmetic overflow when computing
/// the required capacity.
#[derive(Debug)]
pub enum TryPathBufError {
    /// A raw heap allocation failed (no collection involved).
    Alloc(AllocError),
    /// A capacity reservation failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An arithmetic overflow occurred while computing required capacity.
    Overflow,
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Display for TryPathBufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "PathBuf operation failed: heap allocation error"),
            Self::Reserve(e) => write!(f, "PathBuf operation failed: {}", e),
            Self::Overflow => {
                write!(
                    f,
                    "PathBuf operation failed: capacity calculation overflowed"
                )
            }
            Self::Other(msg) => write!(f, "PathBuf operation failed: {}", msg),
        }
    }
}

impl From<AllocError> for TryPathBufError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}

impl From<TryReserveError> for TryPathBufError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl TryDebug for TryPathBufError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => f
                .try_debug_struct("TryPathBufError::Alloc")
                .field("0", e)
                .finish(),
            Self::Reserve(e) => f
                .try_debug_struct("TryPathBufError::Reserve")
                .field("0", e)
                .finish(),
            Self::Overflow => f.write_str("TryPathBufError::Overflow"),
            Self::Other(msg) => f
                .try_debug_struct("TryPathBufError::Other")
                .field("0", msg)
                .finish(),
        }
    }
}

/// A trait for fallible `PathBuf` operations.
///
/// Implemented for `PathBuf`. Mirrors the most commonly-used `PathBuf` methods
/// that can fail due to allocation pressure, returning [`Result`] values instead
/// of panicking.
pub trait TryPathBuf: Sized {
    // ── Construction ────────────────────────────────────────────────────────

    /// Fallibly construct a new empty `PathBuf`.
    ///
    /// This never fails since an empty `PathBuf` requires no allocation.
    fn try_new() -> Result<PathBuf, TryPathBufError>;

    /// Fallibly construct a `PathBuf` from any value that references a [`Path`].
    ///
    /// Accepts `&Path`, `&PathBuf`, `&str`, `&OsStr`, or anything else implementing
    /// [`AsRef<Path>`]. Returns [`TryPathBufError::Reserve`] if the allocation fails.
    fn try_from_path<P: AsRef<Path>>(p: P) -> Result<PathBuf, TryPathBufError>;

    // ── Mutation ────────────────────────────────────────────────────────────

    /// Fallibly append a path component to this `PathBuf`.
    ///
    /// If `path` is absolute, it replaces the current contents. Otherwise it is
    /// appended with a platform-specific separator.
    ///
    /// Returns [`TryPathBufError::Reserve`] if growing the internal buffer fails.
    fn try_push<P: AsRef<Path>>(&mut self, path: P) -> Result<(), TryPathBufError>;

    /// Fallibly set the file name extension for this `PathBuf`.
    ///
    /// Replaces any existing extension. Returns [`TryPathBufError::Reserve`] if
    /// growing the internal buffer fails. Returns [`TryPathBufError::Other`] if
    /// there is no file stem to attach the extension to, or if `ext` contains
    /// path separators.
    fn try_set_extension<E: AsRef<OsStr>>(&mut self, ext: E) -> Result<(), TryPathBufError>;

    /// Fallibly append an extension to the file name of this `PathBuf`.
    ///
    /// Unlike [`Self::try_set_extension`], this appends `.ext` to whatever
    /// filename currently exists, even if it already has an extension.
    /// For example, `foo.tar.gz.try_add_extension("xz")` yields `foo.tar.gz.xz`.
    ///
    /// If the path has no file name (e.g., `/` or `..`) returns `Ok(false)`.
    /// On success with a non-empty extension returns `Ok(true)`. An empty
    /// extension also returns `Ok(true)` but makes no modification.
    ///
    /// Returns [`TryPathBufError::Reserve`] if growing the internal buffer fails.
    /// Returns [`TryPathBufError::Other`] if `ext` contains path separators.
    fn try_add_extension<E: AsRef<OsStr>>(&mut self, ext: E) -> Result<bool, TryPathBufError>;

    // ── Aliases with `fallible_` prefix ────────────────────────────────────

    /// Alias for [`Self::try_new`].
    fn fallible_new() -> Result<PathBuf, TryPathBufError> {
        Self::try_new()
    }

    /// Alias for [`Self::try_from_path`].
    fn fallible_from_path<P: AsRef<Path>>(p: P) -> Result<PathBuf, TryPathBufError> {
        Self::try_from_path(p)
    }

    /// Alias for [`Self::try_push`].
    fn fallible_push<P: AsRef<Path>>(&mut self, path: P) -> Result<(), TryPathBufError> {
        Self::try_push(self, path)
    }

    /// Alias for [`Self::try_set_extension`].
    fn fallible_set_extension<E: AsRef<OsStr>>(&mut self, ext: E) -> Result<(), TryPathBufError> {
        Self::try_set_extension(self, ext)
    }

    /// Alias for [`Self::try_add_extension`].
    fn fallible_add_extension<E: AsRef<OsStr>>(&mut self, ext: E) -> Result<bool, TryPathBufError> {
        Self::try_add_extension(self, ext)
    }
}

// ---------------------------------------------------------------------------
// Internal helper: fallible push into the inner OsString
// ---------------------------------------------------------------------------

fn prefix_len(prefix: &Prefix<'_>) -> usize {
    use self::Prefix::*;

    fn os_str_len(s: &OsStr) -> usize {
        s.as_encoded_bytes().len()
    }

    // This cannot overflow - since the prefix length is smaller than string length.
    match *prefix {
        Verbatim(x) => 4usize.saturating_add(os_str_len(x)),
        VerbatimUNC(x, y) => {
            let mut len = 8usize.saturating_add(os_str_len(x));
            if os_str_len(y) > 0 {
                len = len.saturating_add(1).saturating_add(os_str_len(y));
            }
            len
        }
        VerbatimDisk(_) => 6,
        UNC(x, y) => {
            let mut len = 2usize.saturating_add(os_str_len(x));
            if os_str_len(y) > 0 {
                len = len.saturating_add(1).saturating_add(os_str_len(y));
            }
            len
        }
        DeviceNS(x) => 4usize.saturating_add(os_str_len(x)),
        Disk(_) => 2,
    }
}

fn prefix_is_drive(prefix: &Prefix<'_>) -> bool {
    matches!(*prefix, Prefix::Disk(_))
}

fn prefix_is_verbatim(prefix: &Prefix<'_>) -> bool {
    use self::Prefix::*;
    matches!(*prefix, Verbatim(_) | VerbatimDisk(_) | VerbatimUNC(..))
}

/// Replicates the platform-dependent logic of `PathBuf::_push`.
pub(crate) fn inner_push(target: &mut PathBuf, path: &Path) -> Result<(), TryReserveError> {
    // in general, a separator is needed if the rightmost byte is not a separator
    let buf = target.as_os_str().as_encoded_bytes();
    let mut need_sep = buf
        .last()
        .map(|c| !is_separator(*c as char))
        .unwrap_or(false);

    let comps = target.components();

    // Search for prefixes
    let mut prefix = None;
    for component in target.components() {
        match component {
            Component::Prefix(p) => prefix = Some(p.kind()),
            _ => continue,
        }
    }

    // in the special case of `C:` on Windows, do *not* add a separator
    if let Some(prefix) = &prefix {
        let prefix_len = prefix_len(prefix);
        // Prefix-only path
        if prefix_len > 0 && prefix_len == target.as_os_str().len() && prefix_is_drive(prefix) {
            need_sep = false;
        }
    }

    // Check if the child path has a prefix (needed for need_clear decision).
    // Path::prefix() is private on stable, so we scan components instead.
    let child_has_prefix = path
        .components()
        .next()
        .is_some_and(|c| matches!(c, Component::Prefix(_)));

    let need_clear = if cfg!(target_os = "cygwin") {
        // If path is absolute and its prefix is none, it is like `/foo`,
        // and will be handled below.
        child_has_prefix
    } else {
        // On Unix: prefix is always None.
        path.is_absolute() || child_has_prefix
    };

    // absolute `path` replaces `self`
    if need_clear {
        target.as_mut_os_string().clear();

    // verbatim paths need . and .. removed
    } else if let Some(prefix) = &prefix
        && prefix_is_verbatim(prefix)
        && !path.as_os_str().is_empty()
    {
        let mut buf: Vec<_> = Vec::try_collect(comps)?;
        for c in path.components() {
            match c {
                Component::RootDir => {
                    buf.truncate(1);
                    buf.try_push(c)?;
                }
                Component::CurDir => (),
                Component::ParentDir => {
                    if let Some(Component::Normal(_)) = buf.last() {
                        buf.pop();
                    }
                }
                _ => buf.try_push(c)?,
            }
        }

        let mut res = OsString::new();
        let mut need_sep = false;

        for c in buf {
            if need_sep && c != Component::RootDir {
                res.try_push(OsStr::new(MAIN_SEPARATOR_STR))?;
            }
            res.try_push(c.as_os_str())?;

            need_sep = match c {
                Component::RootDir => false,
                Component::Prefix(prefix) => {
                    !prefix_is_drive(&prefix.kind()) && prefix_len(&prefix.kind()) > 0
                }
                _ => true,
            }
        }

        *target.as_mut_os_string() = res;
        return Ok(());

    // `path` has a root but no prefix, e.g., `\windows` (Windows only)
    } else if path.has_root() {
        let prefix_len: usize = prefix.as_ref().map(prefix_len).unwrap_or(0);
        let current = mem::take(target.as_mut_os_string());
        // Swap out the string to enable internal access
        let mut current_bytes = current.into_encoded_bytes();
        // The prefix_bytes is always valid
        current_bytes.truncate(prefix_len);
        *target.as_mut_os_string() =
            unsafe { OsString::from_encoded_bytes_unchecked(current_bytes) };
    // `path` is a pure relative path
    } else if need_sep {
        target
            .as_mut_os_string()
            .try_push(OsStr::new(MAIN_SEPARATOR_STR))?;
    }

    target.as_mut_os_string().try_push(path.as_os_str())?;
    Ok(())
}

impl TryPathBuf for PathBuf {
    fn try_new() -> Result<PathBuf, TryPathBufError> {
        Ok(PathBuf::new())
    }

    fn try_from_path<P: AsRef<Path>>(p: P) -> Result<PathBuf, TryPathBufError> {
        let p = p.as_ref();
        let mut out = PathBuf::new();
        let os = out.as_mut_os_string();
        let needed = p.as_os_str().len();
        if needed > 0 {
            os.try_reserve(needed).map_err(TryPathBufError::Reserve)?;
        }
        os.push(p.as_os_str());
        Ok(out)
    }

    fn try_push<P: AsRef<Path>>(&mut self, path: P) -> Result<(), TryPathBufError> {
        let path = path.as_ref();
        inner_push(self, path).map_err(TryPathBufError::Reserve)?;
        Ok(())
    }

    fn try_set_extension<E: AsRef<OsStr>>(&mut self, ext: E) -> Result<(), TryPathBufError> {
        let ext = ext.as_ref();
        if self.file_stem().is_none() {
            return Err(TryPathBufError::Other(
                "cannot set extension on a path with no file stem",
            ));
        }
        for &b in ext.as_encoded_bytes() {
            if is_separator(b as char) {
                return Err(TryPathBufError::Other(
                    "extension cannot contain path separators",
                ));
            }
        }
        // Reserve room for the dot and extension.
        if !ext.is_empty() {
            let needed = ext.len().checked_add(1).ok_or(TryPathBufError::Overflow)?;
            self.try_reserve(needed).map_err(TryPathBufError::Reserve)?;
        }
        self.set_extension(ext);
        Ok(())
    }

    fn try_add_extension<E: AsRef<OsStr>>(&mut self, ext: E) -> Result<bool, TryPathBufError> {
        let ext = ext.as_ref();

        // Validate: extension must not contain path separators.
        for &b in ext.as_encoded_bytes() {
            if is_separator(b as char) {
                return Err(TryPathBufError::Other(
                    "extension cannot contain path separators",
                ));
            }
        }

        // Must have a file name component to attach an extension to.
        let file_name = match self.file_name() {
            None => return Ok(false),
            Some(f) => f,
        };

        if ext.is_empty() {
            // Empty extension is a no-op but succeeds.
            return Ok(true);
        }

        // Truncate the inner OsString so it ends right after the file name,
        // then append ".<ext>". We calculate the net byte change upfront so
        // that if reservation fails, the original path is untouched.
        //
        // This mirrors std's pointer-arithmetic truncation: we find the byte
        // offset just past the end of the file name within the full path.
        let all = self.as_os_str().as_encoded_bytes();
        let fname = file_name.as_encoded_bytes();
        // Safety: file_name was obtained from self.file_name(), which returns
        // a subslice of self's inner data. Taking the pointer of the empty
        // tail slice gives us the address just past the end of the file name.
        // Safe: `fname` is a subslice of `all`, so its end pointer lies at or
        // after `all`'s start; the subtraction cannot underflow.
        let fname_end_offset =
            (fname[fname.len()..].as_ptr() as usize).wrapping_sub(all.as_ptr() as usize);

        // Reserve enough capacity for the net change: we will remove
        // `(len - fname_end_offset)` bytes (the old extension + separator)
        // and add `ext.len() + 1` bytes ("." + new extension).
        let len = all.len();
        let bytes_to_truncate = len.saturating_sub(fname_end_offset);
        let needed = ext
            .len()
            .checked_add(1)
            .ok_or(TryPathBufError::Overflow)?
            .saturating_sub(bytes_to_truncate);
        if needed > 0 {
            self.as_mut_os_string()
                .try_reserve(needed)
                .map_err(TryPathBufError::Reserve)?;
        }

        // OsString::truncate is unstable, so we swap out the bytes, truncate
        // the owned buffer, and reconstruct. At this point reservation has
        // already succeeded, so the following pushes are infallible.
        let current = mem::take(self.as_mut_os_string());
        let mut current_bytes = current.into_encoded_bytes();
        current_bytes.truncate(fname_end_offset);
        *self.as_mut_os_string() = unsafe { OsString::from_encoded_bytes_unchecked(current_bytes) };

        // Append ".<ext>" — these cannot fail now that capacity is reserved.
        let os = self.as_mut_os_string();
        os.push(OsStr::new("."));
        os.push(ext);
        Ok(true)
    }
}

// ── TryClone for PathBuf ────────────────────────────────────────────────────

impl TryClone for PathBuf {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let mut out = PathBuf::new();
        let os = out.as_mut_os_string();
        let src = self.as_os_str();
        let len = src.len();
        if len > 0 {
            os.try_reserve(len).map_err(TryCloneError::Reserve)?;
        }
        os.push(src);
        Ok(out)
    }
}

// ── TryDefault for PathBuf ──────────────────────────────────────────────────

impl TryDefault for PathBuf {
    fn try_default() -> Result<Self, TryDefaultError> {
        // An empty PathBuf requires no allocation.
        Ok(PathBuf::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::std::path::TryPath;
    use lang_alloc::string::ToString;
    use lang_std::format;

    // ── Helpers ────────────────────────────────────────────────────────────────

    /// Assert that `try_push(base, child)` produces the same result as
    /// `PathBuf::push(base, child)` on this platform.
    fn assert_push_matches_std(base: &str, child: &str) {
        let (mut expected, mut actual) = {
            let mut e = PathBuf::from(base);
            e.push(child);
            let mut a = PathBuf::try_from_path(base).unwrap();
            a.try_push(child).unwrap();
            (e, a)
        };

        assert_eq!(
            expected, actual,
            "push mismatch for base={} child={}",
            base, child
        );

        // Verify idempotency: pushing again should also match.
        expected.push(child);
        actual.try_push(child).unwrap();
        assert_eq!(
            expected, actual,
            "double-push mismatch for base={} child={}",
            base, child
        );
    }

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

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn try_new_empty() {
        let p = PathBuf::try_new().unwrap();
        assert!(p.as_os_str().is_empty());
    }

    #[test]
    fn try_from_path_empty() {
        let p = PathBuf::try_from_path(Path::new("")).unwrap();
        assert!(p.as_os_str().is_empty());
    }

    #[test]
    fn try_from_path_ascii() {
        let p = PathBuf::try_from_path("/usr/local/bin").unwrap();
        assert_eq!(p, Path::new("/usr/local/bin"));
    }

    #[test]
    fn try_from_path_unicode() {
        let p = PathBuf::try_from_path("/home/ユーザー/docs").unwrap();
        assert_eq!(p, Path::new("/home/ユーザー/docs"));
    }

    #[test]
    fn try_from_path_from_str() {
        let p = PathBuf::try_from_path("hello/world.txt").unwrap();
        assert_eq!(p, Path::new("hello/world.txt"));
    }

    #[test]
    fn try_from_path_from_pathbuf() {
        let original = PathBuf::from("/tmp/data");
        let p = PathBuf::try_from_path(&original).unwrap();
        assert_eq!(p, Path::new("/tmp/data"));
    }

    #[test]
    fn try_from_path_long() {
        let long = "/".to_string() + &"a/".repeat(1000);
        let p = PathBuf::try_from_path(&long).unwrap();
        assert_eq!(p.to_string_lossy(), long);
    }

    // ── Property tests: try_push matches PathBuf::push ────────────────────────

    #[test]
    fn push_relative_on_unix_style_base() {
        assert_push_matches_std("/tmp", "file.txt");
        assert_push_matches_std("/home/user", "projects/rustyfill");
        assert_push_matches_std("/var/log", "");
    }

    #[test]
    fn push_absolute_replaces() {
        assert_push_matches_std("/tmp/foo", "/etc/passwd");
        assert_push_matches_std("relative/path", "/absolute");
        assert_push_matches_std("", "/root");
    }

    #[test]
    fn push_empty_child_is_noop() {
        assert_push_matches_std("/tmp", "");
        assert_push_matches_std("", "");
        assert_push_matches_std("a/b/c", "");
    }

    #[test]
    fn push_trailing_separator_base() {
        assert_push_matches_std("/tmp/", "file.txt");
        assert_push_matches_std("a/b/", "c");
    }

    #[test]
    fn push_curdir_parentdir() {
        assert_push_matches_std("/tmp", ".");
        assert_push_matches_std("/tmp/a", "..");
        assert_push_matches_std("/tmp/a/b", "../..");
        assert_push_matches_std(".", "..");
    }

    #[test]
    fn push_unicode_paths() {
        assert_push_matches_std("/docs", "日本語ファイル.txt");
        assert_push_matches_std("/home/用户", "文档/文件");
        assert_push_matches_std("/data/🔥", "emoji/path");
    }

    #[test]
    fn push_deeply_nested() {
        let deep = (0..50)
            .map(|i| format!("level{i}"))
            .collect::<Vec<_>>()
            .join("/");
        assert_push_matches_std("/root", &deep);
    }

    #[test]
    fn push_multiple_sequential() {
        // Simulate sequential pushes: a -> b -> c
        let mut expected = PathBuf::new();
        let mut actual = PathBuf::try_new().unwrap();
        for component in ["a", "b", "c", "d.txt"] {
            expected.push(component);
            actual.try_push(component).unwrap();
            assert_eq!(expected, actual, "diverged at component {:?}", component);
        }
    }

    #[test]
    fn push_alternating_relative_absolute() {
        let mut expected = PathBuf::from("/start");
        let mut actual = PathBuf::try_from_path("/start").unwrap();
        let sequence: &[&str] = if cfg!(target_os = "linux") {
            &["relative", "/abs1", "another", "/abs2", "."]
        } else {
            &["relative", r#"C:\abs1"#, "another", r#"D:\abs2"#, "."]
        };
        for child in sequence {
            expected.push(*child);
            actual.try_push(*child).unwrap();
            assert_eq!(expected, actual, "diverged at {:?}", child);
        }
    }

    // ── Property tests: try_join matches Path::join ──────────────────────────

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

    // ── Set Extension ────────────────────────────────────────────────────────

    #[test]
    fn try_set_extension_simple() {
        let mut p = PathBuf::try_from_path("notes.txt").unwrap();
        p.try_set_extension("md").unwrap();
        assert_eq!(p, Path::new("notes.md"));
    }

    #[test]
    fn try_set_extension_in_path() {
        let mut p = PathBuf::try_from_path("/home/user/report.pdf").unwrap();
        p.try_set_extension("docx").unwrap();
        assert_eq!(p, Path::new("/home/user/report.docx"));
    }

    #[test]
    fn try_set_extension_empty() {
        let mut p = PathBuf::try_from_path("archive.tar.gz").unwrap();
        p.try_set_extension("").unwrap();
        assert_eq!(p, Path::new("archive.tar"));
    }

    #[test]
    fn try_set_extension_no_stem_fails() {
        let mut p = PathBuf::try_from_path(".").unwrap();
        let result = p.try_set_extension("txt");
        assert!(matches!(result, Err(TryPathBufError::Other(_))));
    }

    #[test]
    fn try_set_extension_root_fails() {
        let mut p = PathBuf::try_from_path("/").unwrap();
        let result = p.try_set_extension("txt");
        assert!(matches!(result, Err(TryPathBufError::Other(_))));
    }

    // ── Add Extension ────────────────────────────────────────────────────────

    /// Assert that `try_add_extension(base, ext)` produces the same result as
    /// `PathBuf::add_extension(base, ext)` on this platform.
    fn assert_add_ext_matches_std(base: &str, ext: &str) {
        let (expected, actual) = {
            let mut e = PathBuf::from(base);
            let e_ok = e.add_extension(ext);
            let mut a = PathBuf::try_from_path(base).unwrap();
            let a_ok = a.try_add_extension(ext).unwrap();
            assert_eq!(
                e_ok, a_ok,
                "add_extension bool mismatch for base={} ext={}",
                base, ext
            );
            (e, a)
        };
        assert_eq!(
            expected, actual,
            "add_extension mismatch for base={} ext={}",
            base, ext
        );
    }

    #[test]
    fn add_extension_simple() {
        assert_add_ext_matches_std("notes", "txt");
        assert_add_ext_matches_std("/tmp/file", "log");
    }

    #[test]
    fn add_extension_appends_to_existing() {
        assert_add_ext_matches_std("foo.tar.gz", "xz");
        assert_add_ext_matches_std("archive.zip.bak", "enc");
    }

    #[test]
    fn add_extension_empty_ext_noop() {
        assert_add_ext_matches_std("foo.txt", "");
        assert_add_ext_matches_std("/path/file.tar.gz", "");
    }

    #[test]
    fn add_extension_no_filename_returns_false() {
        let mut p = PathBuf::try_from_path("/").unwrap();
        assert!(!p.try_add_extension("txt").unwrap());
        let mut p = PathBuf::try_from_path("..").unwrap();
        assert!(!p.try_add_extension("txt").unwrap());
        let mut p = PathBuf::try_from_path(".").unwrap();
        assert!(!p.try_add_extension("txt").unwrap());
    }

    #[test]
    fn add_extension_unicode() {
        assert_add_ext_matches_std("/docs/日本語ファイル", "txt");
        assert_add_ext_matches_std("/数据/文件", "json");
    }

    #[test]
    fn add_extension_trailing_separator() {
        assert_add_ext_matches_std("/tmp/file/", "txt");
        assert_add_ext_matches_std("a/b/c/", "dat");
    }

    #[test]
    fn add_extension_multiple_sequential() {
        let mut p = PathBuf::try_from_path("data").unwrap();
        p.try_add_extension("csv").unwrap();
        p.try_add_extension("gz").unwrap();
        p.try_add_extension("bz2").unwrap();
        assert_eq!(p, Path::new("data.csv.gz.bz2"));
    }

    #[test]
    fn add_extension_rejects_separator() {
        let mut p = PathBuf::try_from_path("file").unwrap();
        let result = p.try_add_extension("a/b");
        assert!(matches!(result, Err(TryPathBufError::Other(_))));
    }

    #[test]
    fn set_extension_rejects_separator() {
        let mut p = PathBuf::try_from_path("file.txt").unwrap();
        let result = p.try_set_extension("a/b");
        assert!(matches!(result, Err(TryPathBufError::Other(_))));
    }

    // ── TryClone ─────────────────────────────────────────────────────────────

    #[test]
    fn try_clone_empty() {
        let p = PathBuf::new();
        let c = p.try_clone().unwrap();
        assert!(c.as_os_str().is_empty());
    }

    #[test]
    fn try_clone_populated() {
        let p = PathBuf::try_from_path("/var/log/syslog").unwrap();
        let c = p.try_clone().unwrap();
        assert_eq!(c, Path::new("/var/log/syslog"));
    }

    #[test]
    fn try_clone_unicode() {
        let p = PathBuf::try_from_path("/data/日本語/テスト").unwrap();
        let c = p.try_clone().unwrap();
        assert_eq!(c, Path::new("/data/日本語/テスト"));
    }

    #[test]
    fn try_clone_independent() {
        let mut p = PathBuf::try_from_path("/tmp/original").unwrap();
        let c = p.try_clone().unwrap();
        p.try_push("extra").unwrap();
        assert_eq!(p, Path::new("/tmp/original/extra"));
        assert_eq!(c, Path::new("/tmp/original"));
    }

    // ── TryDefault ───────────────────────────────────────────────────────────

    #[test]
    fn try_default_empty() {
        let p: PathBuf = PathBuf::try_default().unwrap();
        assert!(p.as_os_str().is_empty());
    }

    // ── Combined workflows ───────────────────────────────────────────────────

    #[test]
    fn build_then_clone() {
        let mut p = PathBuf::try_default().unwrap();
        p.try_push("src").unwrap();
        p.try_push("main.rs").unwrap();
        let c = p.try_clone().unwrap();
        p.try_push("backup").unwrap();
        assert_eq!(p, Path::new("src/main.rs/backup"));
        assert_eq!(c, Path::new("src/main.rs"));
    }

    #[test]
    fn build_modify_extension() {
        let mut p = PathBuf::try_from_path("/tmp/draft.txt").unwrap();
        p.try_set_extension("md").unwrap();
        assert_eq!(p, Path::new("/tmp/draft.md"));
        let c = p.try_clone().unwrap();
        assert_eq!(c, Path::new("/tmp/draft.md"));
    }

    // ── OOM tests ─────────────────────────────────────────────────────
    use rustyfill_test_allocator::{FailPolicy, with_policy};

    #[test]
    fn pathbuf_try_from_path_fails_on_oom() {
        let r: Result<PathBuf, TryPathBufError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <PathBuf as TryPathBuf>::try_from_path("/some/path")
            });
        assert!(r.is_err());
    }

    #[test]
    fn pathbuf_try_push_fails_on_oom() {
        // Pushing onto an existing PathBuf triggers realloc (not alloc) when
        // growing the underlying buffer. Use fail_next_realloc to target this.
        let long = format!("/base/{}", "x".repeat(256));
        let mut p = PathBuf::try_from_path(long).unwrap();
        p.as_mut_os_string().try_shrink_to_fit().unwrap();
        let extra = format!("child_{}", "y".repeat(256));
        let r = with_policy(FailPolicy::fail_next_realloc(), || p.fallible_push(extra));
        assert!(r.is_err());
    }

    #[test]
    fn pathbuf_try_clone_fails_on_oom() {
        let orig = PathBuf::try_from_path("/data/files").unwrap();
        let r: Result<PathBuf, TryCloneError> =
            with_policy(FailPolicy::fail_next_alloc(), || orig.try_clone());
        assert!(r.is_err());
    }

    #[test]
    fn pathbuf_try_clone_empty_succeeds_under_oom() {
        let orig: PathBuf = PathBuf::new();
        let r: Result<PathBuf, TryCloneError> =
            with_policy(FailPolicy::fail_next_alloc(), || orig.try_clone());
        assert!(r.is_ok());
    }

    #[test]
    fn pathbuf_nth_alloc_fail_targets_correct_call() {
        let orig = PathBuf::try_from_path("/data/files").unwrap();
        let (r1_ok, r2_err, r3_ok) = with_policy(FailPolicy::fail_nth_alloc(2), || {
            let r1: Result<PathBuf, TryCloneError> = orig.try_clone();
            let r2: Result<PathBuf, TryCloneError> = orig.try_clone();
            let r3: Result<PathBuf, TryCloneError> = orig.try_clone();
            (r1.is_ok(), r2.is_err(), r3.is_ok())
        });
        assert!(r1_ok, "first clone should succeed");
        assert!(r2_err, "second clone should fail");
        assert!(r3_ok, "third clone should succeed");
    }

    #[test]
    fn pathbuf_oom_restores_allocation_afterwards() {
        let r: Result<PathBuf, TryPathBufError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <PathBuf as TryPathBuf>::try_from_path("/x")
            });
        assert!(r.is_err());
        // Allocation works again after guard scope ends.
        let r: Result<PathBuf, TryPathBufError> = <PathBuf as TryPathBuf>::try_from_path("/y");
        assert!(r.is_ok());
    }
}
