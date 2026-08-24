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

use crate::alloc::vec::TryVec;
use crate::alloc::{TryReserveError, TryReserveErrorExt};
use crate::std::ffi::TryOsString;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::{TryDebug, TryDisplay, helpers::FormatterExt};
use lang_alloc::vec::Vec;
use lang_core::fmt;
use lang_core::mem;
use lang_std::ffi::{OsStr, OsString};
use lang_std::path::{Component, MAIN_SEPARATOR_STR, Path, PathBuf, Prefix, is_separator};

/// Error returned by [`TryPathBuf::try_set_extension`].
pub enum TryPathBufSetExtensionError {
    /// A capacity reservation failed (overflow or OOM).
    Reserve(TryReserveError),
    /// The path has no file stem to attach an extension to (e.g., `/`, `.`, `..`).
    NoFileStem,
    /// The provided extension contains a path separator.
    SeparatorInPath,
}

impl fmt::Debug for TryPathBufSetExtensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(self, f)
    }
}

impl fmt::Display for TryPathBufSetExtensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }
}

impl From<TryReserveError> for TryPathBufSetExtensionError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl TryDebug for TryPathBufSetExtensionError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => f
                .try_debug_tuple("TryPathBufSetExtensionError::Reserve")
                .field(e)
                .finish(),
            Self::NoFileStem => f
                .try_debug_tuple("TryPathBufSetExtensionError::NoFileStem")
                .finish(),
            Self::SeparatorInPath => f
                .try_debug_tuple("TryPathBufSetExtensionError::SeparatorInPath")
                .finish(),
        }
    }
}

impl TryDisplay for TryPathBufSetExtensionError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "PathBuf set-extension failed: {}", e),
            Self::NoFileStem => write!(f, "cannot set extension on a path with no file stem"),
            Self::SeparatorInPath => write!(f, "extension cannot contain path separators"),
        }
    }
}

/// Error returned by [`TryPathBuf::try_add_extension`].
pub enum TryPathBufAddExtensionError {
    /// A capacity reservation failed (overflow or OOM).
    Reserve(TryReserveError),
    /// The provided extension contains a path separator.
    SeparatorInPath,
}

impl fmt::Debug for TryPathBufAddExtensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(self, f)
    }
}

impl fmt::Display for TryPathBufAddExtensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }
}

impl From<TryReserveError> for TryPathBufAddExtensionError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl TryDebug for TryPathBufAddExtensionError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => f
                .try_debug_tuple("TryPathBufAddExtensionError::Reserve")
                .field(e)
                .finish(),
            Self::SeparatorInPath => f
                .try_debug_tuple("TryPathBufAddExtensionError::SeparatorInPath")
                .finish(),
        }
    }
}

impl TryDisplay for TryPathBufAddExtensionError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "PathBuf add-extension failed: {}", e),
            Self::SeparatorInPath => write!(f, "extension cannot contain path separators"),
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
    fn try_new() -> Result<PathBuf, TryReserveError>;

    /// Fallibly construct a `PathBuf` from any value that references a [`Path`].
    ///
    /// Accepts `&Path`, `&PathBuf`, `&str`, `&OsStr`, or anything else implementing
    /// [`AsRef<Path>`]. Returns [`TryReserveError`] if the allocation fails.
    fn try_from_path<P: AsRef<Path>>(p: P) -> Result<PathBuf, TryReserveError>;

    // ── Mutation ────────────────────────────────────────────────────────────

    /// Fallibly append a path component to this `PathBuf`.
    ///
    /// If `path` is absolute, it replaces the current contents. Otherwise it is
    /// appended with a platform-specific separator.
    ///
    /// Returns [`TryReserveError`] if growing the internal buffer fails.
    fn try_push<P: AsRef<Path>>(&mut self, path: P) -> Result<(), TryReserveError>;

    /// Fallibly set the file name extension for this `PathBuf`.
    ///
    /// Replaces any existing extension. Fails with [`TryPathBufSetExtensionError::NoFileStem`]
    /// if there is no file stem to attach the extension to, or
    /// [`TryPathBufSetExtensionError::SeparatorInPath`] if `ext` contains path
    /// separators.
    fn try_set_extension<E: AsRef<OsStr>>(
        &mut self,
        ext: E,
    ) -> Result<(), TryPathBufSetExtensionError>;

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
    /// Fails with [`TryPathBufAddExtensionError::SeparatorInPath`] if `ext` contains
    /// path separators.
    fn try_add_extension<E: AsRef<OsStr>>(
        &mut self,
        ext: E,
    ) -> Result<bool, TryPathBufAddExtensionError>;

    // ── Aliases with `fallible_` prefix ────────────────────────────────────

    /// Alias for [`Self::try_new`].
    fn fallible_new() -> Result<PathBuf, TryReserveError> {
        Self::try_new()
    }

    /// Alias for [`Self::try_from_path`].
    fn fallible_from_path<P: AsRef<Path>>(p: P) -> Result<PathBuf, TryReserveError> {
        Self::try_from_path(p)
    }

    /// Alias for [`Self::try_push`].
    fn fallible_push<P: AsRef<Path>>(&mut self, path: P) -> Result<(), TryReserveError> {
        Self::try_push(self, path)
    }

    /// Alias for [`Self::try_set_extension`].
    fn fallible_set_extension<E: AsRef<OsStr>>(
        &mut self,
        ext: E,
    ) -> Result<(), TryPathBufSetExtensionError> {
        Self::try_set_extension(self, ext)
    }

    /// Alias for [`Self::try_add_extension`].
    fn fallible_add_extension<E: AsRef<OsStr>>(
        &mut self,
        ext: E,
    ) -> Result<bool, TryPathBufAddExtensionError> {
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
///
/// Decides which push strategy applies (absolute replacement, verbatim
/// normalization, rooted truncation, or plain relative append) and delegates
/// to the corresponding helper.
pub(crate) fn inner_push(target: &mut PathBuf, path: &Path) -> Result<(), TryReserveError> {
    // in general, a separator is needed if the rightmost byte is not a separator
    let mut need_sep = target
        .as_os_str()
        .as_encoded_bytes()
        .last()
        .map(|c| !is_separator(*c as char))
        .unwrap_or(false);

    // Search for prefixes (a path can only have one prefix)
    let prefix = target.components().find_map(|c| match c {
        Component::Prefix(p) => Some(p.kind()),
        _ => None,
    });

    // in the special case of `C:` on Windows, do *not* add a separator
    if let Some(ref p) = prefix {
        let plen = prefix_len(p);
        // Prefix-only path
        if plen > 0 && plen == target.as_os_str().len() && prefix_is_drive(p) {
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
    } else if let Some(ref p) = prefix
        && prefix_is_verbatim(p)
        && !path.as_os_str().is_empty()
    {
        return push_verbatim_normalized(target, path);

    // `path` has a root but no prefix, e.g., `\windows` (Windows only)
    } else if path.has_root() {
        truncate_to_prefix(target, prefix.as_ref().map(prefix_len).unwrap_or(0));
    // `path` is a pure relative path
    } else if need_sep {
        target
            .as_mut_os_string()
            .try_push(OsStr::new(MAIN_SEPARATOR_STR))?;
    }

    target.as_mut_os_string().try_push(path.as_os_str())?;
    Ok(())
}

/// Verbatim paths (`\\?\...`) require `.` and `..` components to be resolved
/// away before appending, since they are not interpreted by the OS. Collects
/// the merged component list, normalizes it, and rebuilds the target string.
fn push_verbatim_normalized(target: &mut PathBuf, path: &Path) -> Result<(), TryReserveError> {
    let mut buf: Vec<Component<'_>> = Vec::try_collect(target.components())?;
    append_normalized(&mut buf, path)?;

    *target.as_mut_os_string() = render_components(buf)?;
    Ok(())
}

/// Merges the components of `path` into `buf`, resolving `.` and `..` in place.
/// This is the platform-independent normalization core shared by verbatim
/// pushes; kept allocation-light so it can be unit-tested directly.
fn append_normalized<'a>(
    buf: &mut Vec<Component<'a>>,
    path: &'a Path,
) -> Result<(), TryReserveError> {
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
    Ok(())
}

/// Renders a normalized component list back into an `OsString`, inserting
/// separators where needed (a separator follows every component except a root
/// directory, and non-drive prefixes).
fn render_components(components: Vec<Component<'_>>) -> Result<OsString, TryReserveError> {
    let mut res = OsString::new();
    let mut need_sep = false;

    for c in components {
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

    Ok(res)
}

/// Truncates the target's inner string down to its prefix length, discarding
/// everything after it. Used when pushing a rooted-but-prefix-less path such
/// as `\windows` onto a prefixed base.
fn truncate_to_prefix(target: &mut PathBuf, prefix_len: usize) {
    let current = mem::take(target.as_mut_os_string());
    // Swap out the string to enable internal access
    let mut current_bytes = current.into_encoded_bytes();
    // The prefix bytes are always valid
    current_bytes.truncate(prefix_len);
    *target.as_mut_os_string() = unsafe { OsString::from_encoded_bytes_unchecked(current_bytes) };
}

impl TryPathBuf for PathBuf {
    fn try_new() -> Result<PathBuf, TryReserveError> {
        Ok(PathBuf::new())
    }

    fn try_from_path<P: AsRef<Path>>(p: P) -> Result<PathBuf, TryReserveError> {
        let p = p.as_ref();
        let mut out = PathBuf::new();
        let os = out.as_mut_os_string();
        let needed = p.as_os_str().len();
        if needed > 0 {
            os.try_reserve(needed)?;
        }
        os.push(p.as_os_str());
        Ok(out)
    }

    fn try_push<P: AsRef<Path>>(&mut self, path: P) -> Result<(), TryReserveError> {
        let path = path.as_ref();
        inner_push(self, path)?;
        Ok(())
    }

    fn try_set_extension<E: AsRef<OsStr>>(
        &mut self,
        ext: E,
    ) -> Result<(), TryPathBufSetExtensionError> {
        let ext = ext.as_ref();
        if self.file_stem().is_none() {
            return Err(TryPathBufSetExtensionError::NoFileStem);
        }
        for &b in ext.as_encoded_bytes() {
            if is_separator(b as char) {
                return Err(TryPathBufSetExtensionError::SeparatorInPath);
            }
        }
        // Reserve room for the dot and extension.
        if !ext.is_empty() {
            let needed = ext.len().checked_add(1).ok_or_else(|| {
                TryPathBufSetExtensionError::Reserve(TryReserveErrorExt::new_capacity_overflow())
            })?;
            self.try_reserve(needed)
                .map_err(TryPathBufSetExtensionError::Reserve)?;
        }
        self.set_extension(ext);
        Ok(())
    }

    fn try_add_extension<E: AsRef<OsStr>>(
        &mut self,
        ext: E,
    ) -> Result<bool, TryPathBufAddExtensionError> {
        let ext = ext.as_ref();

        // Validate: extension must not contain path separators.
        for &b in ext.as_encoded_bytes() {
            if is_separator(b as char) {
                return Err(TryPathBufAddExtensionError::SeparatorInPath);
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
            .ok_or_else(|| {
                TryPathBufAddExtensionError::Reserve(TryReserveErrorExt::new_capacity_overflow())
            })?
            .saturating_sub(bytes_to_truncate);
        if needed > 0 {
            self.as_mut_os_string()
                .try_reserve(needed)
                .map_err(TryPathBufAddExtensionError::Reserve)?;
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
        assert!(matches!(
            result,
            Err(TryPathBufSetExtensionError::NoFileStem)
        ));
    }

    #[test]
    fn try_set_extension_root_fails() {
        let mut p = PathBuf::try_from_path("/").unwrap();
        let result = p.try_set_extension("txt");
        assert!(matches!(
            result,
            Err(TryPathBufSetExtensionError::NoFileStem)
        ));
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
        assert!(matches!(
            result,
            Err(TryPathBufAddExtensionError::SeparatorInPath)
        ));
    }

    #[test]
    fn set_extension_rejects_separator() {
        let mut p = PathBuf::try_from_path("file.txt").unwrap();
        let result = p.try_set_extension("a/b");
        assert!(matches!(
            result,
            Err(TryPathBufSetExtensionError::SeparatorInPath)
        ));
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
        let r: Result<PathBuf, TryReserveError> =
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
        let r: Result<PathBuf, TryReserveError> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                <PathBuf as TryPathBuf>::try_from_path("/x")
            });
        assert!(r.is_err());
        // Allocation works again after guard scope ends.
        let r: Result<PathBuf, TryReserveError> = <PathBuf as TryPathBuf>::try_from_path("/y");
        assert!(r.is_ok());
    }

    // ── Verbatim-normalization OOM tests (Windows-only paths) ───────────────
    //
    // On Windows these exercise `push_verbatim_normalized`, which allocates a
    // component buffer and rebuilds the target string. On other platforms the
    // same inputs take the plain relative-append branch, so we only assert on
    // allocation behavior there.

    #[cfg(windows)]
    #[test]
    fn push_verbatim_fails_on_oom() {
        let mut p = PathBuf::try_from_path(r"\\?\C:\a\b").unwrap();
        let r = with_policy(FailPolicy::fail_next_alloc(), || p.fallible_push(r"c\d"));
        assert!(r.is_err());
        // Target is left unchanged on failure.
        assert_eq!(p, Path::new(r"\\?\C:\a\b"));
    }

    #[cfg(windows)]
    #[test]
    fn push_verbatim_parent_dir_normalizes() {
        let mut p = PathBuf::try_from_path(r"\\?\C:\a\b").unwrap();
        p.fallible_push(r"..\c").unwrap();
        assert_eq!(p, Path::new(r"\\?\C:\a\c"));
    }

    #[cfg(windows)]
    #[test]
    fn push_verbatim_root_resets() {
        let mut p = PathBuf::try_from_path(r"\\?\C:\a\b").unwrap();
        p.fallible_push(r"\d\e").unwrap();
        assert_eq!(p, Path::new(r"\\?\C:\d\e"));
    }

    #[cfg(not(windows))]
    #[test]
    fn push_relative_long_child_grows_buffer() {
        // Non-Windows stand-in: exercises the relative-append + realloc path.
        let long = format!("/base/{}", "x".repeat(128));
        let mut p = PathBuf::try_from_path(&long).unwrap();
        let extra = format!("child/{}", "y".repeat(128));
        p.fallible_push(extra.clone()).unwrap();
        let expected_len = long.len() + 1 + extra.len();
        assert_eq!(p.as_os_str().len(), expected_len);
    }

    // ── Pure normalization/rendering core (platform-independent) ─────────────
    //
    // `append_normalized` and `render_components` are the verbatim-push logic
    // with allocation stripped out, so they can be exercised on any platform —
    // including Linux, where the full `push_verbatim_normalized` is unreachable.

    /// Helper: parse a base and child into component vectors for direct testing.
    fn comps_of(p: &str) -> Vec<Component<'_>> {
        Path::new(p).components().collect()
    }

    #[test]
    fn append_normalized_drops_curdir() {
        let mut buf = comps_of("a/b");
        append_normalized(&mut buf, Path::new(".")).unwrap();
        assert_eq!(buf, comps_of("a/b"));
    }

    #[test]
    fn append_normalized_resolves_parentdir() {
        let mut buf = comps_of("a/b/c");
        append_normalized(&mut buf, Path::new("../d")).unwrap();
        assert_eq!(buf, comps_of("a/b/d"));
    }

    #[test]
    fn append_normalized_parentdir_at_root_is_noop() {
        // `..` at the root has nothing to pop; std keeps it as a no-op.
        let mut buf = comps_of("");
        append_normalized(&mut buf, Path::new("..")).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    fn append_normalized_root_on_relative_base_appends() {
        // A relative base has no prefix, so an absolute child does NOT reset —
        // the root dir is appended after the existing components. (The
        // truncate-to-1 reset only applies once a prefix is present, i.e. the
        // Windows verbatim case.)
        let mut buf = comps_of("a/b");
        append_normalized(&mut buf, Path::new("/c/d")).unwrap();
        assert_eq!(
            buf,
            lang_alloc::vec![
                Component::Normal(OsStr::new("a")),
                Component::RootDir,
                Component::Normal(OsStr::new("c")),
                Component::Normal(OsStr::new("d")),
            ]
        );
    }

    #[test]
    fn append_normalized_plain_append() {
        let mut buf = comps_of("a");
        append_normalized(&mut buf, Path::new("b/c")).unwrap();
        assert_eq!(buf, comps_of("a/b/c"));
    }

    #[test]
    fn render_components_inserts_separators() {
        let rendered = render_components(comps_of("a/b/c")).unwrap();
        assert_eq!(rendered.as_encoded_bytes(), b"a/b/c");
    }

    #[test]
    fn render_components_leading_root_has_no_double_sep() {
        let rendered = render_components(comps_of("/a/b")).unwrap();
        assert_eq!(rendered.as_encoded_bytes(), b"/a/b");
    }

    #[test]
    fn render_components_roundtrips_normalized() {
        // Normalize then render. Leading `..` at a relative root is dropped,
        // so `a/b + ../../c/../d` reduces to just `d`.
        let mut buf = comps_of("a/b");
        append_normalized(&mut buf, Path::new("../../c/../d")).unwrap();
        let rendered = render_components(buf).unwrap();
        assert_eq!(rendered.as_encoded_bytes(), b"d");
    }

    #[test]
    fn render_components_interior_parentdir_collapses() {
        // Interior `..` pops the preceding normal component: a/b/c + ../d => a/b/d.
        let mut buf = comps_of("a/b/c");
        append_normalized(&mut buf, Path::new("../d")).unwrap();
        let rendered = render_components(buf).unwrap();
        assert_eq!(rendered.as_encoded_bytes(), b"a/b/d");
    }

    #[test]
    fn render_components_empty() {
        let rendered = render_components(Vec::new()).unwrap();
        assert!(rendered.is_empty());
    }

    // ── prefix_len (pure data — all variants constructible on any platform) ──
    //
    // `Prefix` variants carry plain fields, so we can build each one directly
    // and assert the byte-length arithmetic without needing Windows to parse
    // a real path. Expected values mirror the fixed-width header constants:
    // Disk=2, VerbatimDisk=6, Verbatim/DeviceNS=4+len, UNC=2+s(+1+sh),
    // VerbatimUNC=8+s(+1+sh).

    #[test]
    fn prefix_len_disk_and_verbatim_disk_are_fixed() {
        assert_eq!(prefix_len(&Prefix::Disk(b'C')), 2);
        assert_eq!(prefix_len(&Prefix::VerbatimDisk(b'D')), 6);
    }

    #[test]
    fn prefix_len_device_namespace_is_four_plus_name() {
        // "//./con" -> 4-byte header + "con" (3) = 7
        assert_eq!(prefix_len(&Prefix::DeviceNS(OsStr::new("con"))), 7);
        assert_eq!(prefix_len(&Prefix::DeviceNS(OsStr::new(""))), 4);
    }

    #[test]
    fn prefix_len_verbatim_is_four_plus_path() {
        // "\\?\" (4) + "C:/foo" (6) = 10
        assert_eq!(prefix_len(&Prefix::Verbatim(OsStr::new("C:/foo"))), 10);
        assert_eq!(prefix_len(&Prefix::Verbatim(OsStr::new(""))), 4);
    }

    #[test]
    fn prefix_len_unc_counts_server_and_share() {
        // "\\" (2) + "srv" (3) + "\" (1) + "shr" (3) = 9
        assert_eq!(
            prefix_len(&Prefix::UNC(OsStr::new("srv"), OsStr::new("shr"))),
            9
        );
        // Empty share omits the trailing separator: 2 + 3 = 5
        assert_eq!(
            prefix_len(&Prefix::UNC(OsStr::new("srv"), OsStr::new(""))),
            5
        );
    }

    #[test]
    fn prefix_len_verbatim_unc_uses_eight_byte_header() {
        // "\\?\UNC\" (8) + "srv" (3) + "\" (1) + "shr" (3) = 15
        assert_eq!(
            prefix_len(&Prefix::VerbatimUNC(OsStr::new("srv"), OsStr::new("shr"))),
            15
        );
        // Empty share: 8 + 3 = 11
        assert_eq!(
            prefix_len(&Prefix::VerbatimUNC(OsStr::new("srv"), OsStr::new(""))),
            11
        );
    }
}
