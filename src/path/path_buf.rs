//! Fallible `PathBuf` operations.
//!
//! Provides the [`TryPathBuf`] trait with methods that mirror common `PathBuf`
//! constructors and mutating operations but return [`Result`] to handle allocation
//! failures gracefully.
//!
//! # Design
//!
//! `PathBuf` wraps an [`OsString`](std::ffi::OsString) internally, so its fallible
//! operations delegate to the same reserve-before-mutate pattern used by
//! [`TryOsString`](crate::ffi::TryOsString). Methods that may grow internal capacity
//! (`push`, etc.) call `try_reserve` first so that allocation failures surface as
//! errors rather than panics.
//!
//! The trait also implements [`TryClone`](crate::try_clone::TryClone) and
//! [`TryDefault`](crate::try_default::TryDefault) for `PathBuf`.

use crate::alloc::AllocError;
use crate::ffi::TryOsString;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::vec::TryVec;
use core::fmt;
use std::collections::TryReserveError;
use std::ffi::{OsStr, OsString};
use std::path::{Component, MAIN_SEPARATOR_STR, Path, PathBuf, Prefix, is_separator};

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
    fn from(_: AllocError) -> Self {
        Self::Alloc(AllocError)
    }
}

impl From<TryReserveError> for TryPathBufError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
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
    /// Returns [`TryPathBufError::Reserve`] if growing the internal buffer fails.
    /// Returns [`TryPathBufError::Other`] if there is no file stem to attach the
    /// extension to.
    fn try_set_extension(&mut self, ext: &str) -> Result<(), TryPathBufError>;
}

// ---------------------------------------------------------------------------
// Internal helper: fallible push into the inner OsString
// ---------------------------------------------------------------------------

fn prefix_len(prefix: &Prefix<'_>) -> usize {
    use self::Prefix::*;

    fn os_str_len(s: &OsStr) -> usize {
        s.as_encoded_bytes().len()
    }

    match *prefix {
        Verbatim(x) => 4 + os_str_len(x),
        VerbatimUNC(x, y) => {
            8 + os_str_len(x)
                + if os_str_len(y) > 0 {
                    1 + os_str_len(y)
                } else {
                    0
                }
        }
        VerbatimDisk(_) => 6,
        UNC(x, y) => {
            2 + os_str_len(x)
                + if os_str_len(y) > 0 {
                    1 + os_str_len(y)
                } else {
                    0
                }
        }
        DeviceNS(x) => 4 + os_str_len(x),
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
fn inner_push(target: &mut PathBuf, path: &Path) -> Result<(), TryReserveError> {
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

    let need_clear = if cfg!(target_os = "cygwin") {
        // If path is absolute and its prefix is none, it is like `/foo`,
        // and will be handled below.
        prefix.is_some()
    } else {
        // On Unix: prefix is always None.
        path.is_absolute() || prefix.is_some()
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
        let current = std::mem::take(target.as_mut_os_string());
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
        self.push(path);
        inner_push(self, path).map_err(TryPathBufError::Reserve)?;
        Ok(())
    }

    fn try_set_extension(&mut self, ext: &str) -> Result<(), TryPathBufError> {
        if self.file_stem().is_none() {
            return Err(TryPathBufError::Other(
                "cannot set extension on a path with no file stem",
            ));
        }
        // Reserve room for the dot and extension.
        if !ext.is_empty() {
            self.try_reserve(ext.len() + 1)
                .map_err(TryPathBufError::Reserve)?;
        }
        self.set_extension(ext);
        Ok(())
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

    // ── Mutation ─────────────────────────────────────────────────────────────

    #[test]
    fn try_push_relative() {
        let mut p = PathBuf::try_from_path("/tmp").unwrap();
        p.try_push("file.txt").unwrap();
        assert_eq!(p, Path::new("/tmp/file.txt"));
    }

    #[test]
    fn try_push_absolute_replaces() {
        let mut p = PathBuf::try_from_path("/tmp/foo").unwrap();
        p.try_push("/etc/passwd").unwrap();
        assert_eq!(p, Path::new("/etc/passwd"));
    }

    #[test]
    fn try_push_multiple_components() {
        let mut p = PathBuf::try_new().unwrap();
        p.try_push("a").unwrap();
        p.try_push("b").unwrap();
        p.try_push("c.txt").unwrap();
        assert_eq!(p, Path::new("a/b/c.txt"));
    }

    #[test]
    fn try_push_empty_component() {
        let mut p = PathBuf::try_from_path("/tmp").unwrap();
        p.try_push(Path::new("")).unwrap();
        assert_eq!(p, Path::new("/tmp"));
    }

    #[test]
    fn try_push_unicode() {
        let mut p = PathBuf::try_from_path("/docs").unwrap();
        p.try_push(Path::new("日本語ファイル.txt")).unwrap();
        assert_eq!(p, Path::new("/docs/日本語ファイル.txt"));
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
}
