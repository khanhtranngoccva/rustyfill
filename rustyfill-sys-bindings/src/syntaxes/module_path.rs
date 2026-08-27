//! A module path: the position of a module in the library's namespace tree.
//!
//! One `ModulePath` is one canonical spelling — an ordered, non-empty list of
//! identifier segments (`["sys", "pal", "unix", "sync"]`). Every other textual
//! form (slash file-stem, `::`-joined chain, parent/leaf decomposition) is
//! derived from it by method, never reconstructed ad hoc at call sites.

use std::fmt;

/// The position of a module within a single library root.
///
/// Segments are identifiers only (no separators, no `.rs`, no empty entries).
/// The root module is represented by the empty segment list.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModulePath {
    segments: Vec<String>,
}

impl ModulePath {
    /// The library-root module (no segments).
    pub fn root() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    /// Parse a slash-separated file path or module stem
    /// (e.g. `"sys/pal/mod.rs"`, `"collections/btree/map.rs"`, or an already
    /// stem-normalized `"sys/pal/unix/sync"`). Strips a trailing `.rs` first,
    /// then a trailing `/mod`, matching the Rust module-file convention where
    /// `dir/mod.rs` defines the module at `dir`.
    /// Empty separator runs and non-identifier segments are rejected rather
    /// than silently dropped, so malformed input fails loudly instead of
    /// mis-routing.
    pub fn from_file_stem(path: &str) -> Option<Self> {
        let without_rs = path.strip_suffix(".rs").unwrap_or(path);
        let cleaned = without_rs.strip_suffix("/mod").unwrap_or(without_rs);
        if cleaned == "mod" {
            // A root-level `mod.rs` defines the library-root module.
            return Some(Self::root());
        }
        Self::from_slash(cleaned)
    }

    /// Parse a slash-separated module path (`"sys/pal/unix/sync"`).
    /// Returns `None` if any segment is empty or not a bare identifier.
    pub fn from_slash(slash: &str) -> Option<Self> {
        Self::parse(slash.split('/'))
    }

    /// Parse a `::`-separated canonical path (`"std::sys::sync"`).
    /// Returns `None` if any segment is empty or not a bare identifier.
    pub fn from_canonical(canonical: &str) -> Option<Self> {
        Self::parse(canonical.split("::"))
    }

    fn parse<'a>(segments: impl Iterator<Item = &'a str>) -> Option<Self> {
        // An empty input denotes the library root, not an error.
        let collected: Vec<&str> = segments.collect();
        if collected.len() == 1 && collected[0].is_empty() {
            return Some(Self::root());
        }
        for seg in &collected {
            if seg.is_empty() || !is_identifier(seg) {
                return None;
            }
        }
        Some(Self {
            segments: collected.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Build directly from validated segments.
    pub fn from_segments(segments: impl IntoIterator<Item = String>) -> Option<Self> {
        let segments: Vec<String> = segments.into_iter().collect();
        for seg in &segments {
            if !seg.is_empty() && !is_identifier(seg) {
                return None;
            }
        }
        Some(Self { segments })
    }

    /// True for the library-root module.
    pub fn is_root(&self) -> bool {
        self.segments.is_empty()
    }

    /// Number of segments (depth below the library root).
    pub fn depth(&self) -> usize {
        self.segments.len()
    }

    /// The final segment, or `""` for the root.
    pub fn leaf(&self) -> &str {
        self.segments.last().map(String::as_str).unwrap_or("")
    }

    /// Borrowed, allocation-free view of the path minus its final segment.
    /// The root's parent is itself.
    pub fn parent(&self) -> ParentView<'_> {
        ParentView(self)
    }

    /// Owned copy of the path minus its final segment.
    pub fn parent_owned(&self) -> Self {
        let len = self.segments.len().saturating_sub(1);
        Self {
            segments: self.segments[..len].to_vec(),
        }
    }

    /// Append a child segment, returning the extended path.
    pub fn join(&self, name: &str) -> Self {
        let mut segments = self.segments.clone();
        segments.push(name.to_string());
        Self { segments }
    }

    /// The first segment, or `""` for the root. Used when descending into a
    /// tree one level at a time (the root's "name" is empty).
    pub fn head(&self) -> &str {
        self.segments.first().map(String::as_str).unwrap_or("")
    }

    /// Borrowed view of the path minus its first segment. The root and
    /// single-segment paths both yield the root.
    pub fn tail(&self) -> ParentTailView<'_> {
        ParentTailView(self)
    }

    /// True if `other` is this path or lives beneath it (i.e. `self` is an
    /// ancestor of, or identical to, `other`). Reads as "`other` begins with
    /// `self`'s segments".
    pub fn contains(&self, other: &Self) -> bool {
        other.segments.starts_with(&self.segments)
    }

    /// The immediate-child relationship used by sibling/child scans:
    /// true when `other` is exactly one segment deeper and shares `self` as
    /// its full parent.
    pub fn is_direct_parent_of(&self, other: &Self) -> bool {
        other.depth() == self.depth() + 1
            && other.segments[self.depth()..].len() == 1
            && other.segments[..self.depth()] == self.segments
    }

    /// Slash-separated file-relative spelling (`"sys/pal/unix/sync"`).
    /// The root renders as `""`.
    pub fn to_slash(&self) -> String {
        self.segments.join("/")
    }

    /// `::`-separated canonical spelling (`"sys::pal::unix::sync"`).
    /// The root renders as `""`.
    pub fn to_canonical(&self) -> String {
        self.segments.join("::")
    }

    /// Convert to a filesystem [`std::path::PathBuf`] for I/O at the disk
    /// boundary (e.g. `out_dir.join(module.to_file_path())`). The root maps
    /// to an empty path, which joins onto its parent directory unchanged.
    ///
    /// This is deliberately a one-way *rendering* step: `ModulePath` itself
    /// stays a pure segment vector because module namespaces are not
    /// filesystem paths — `Path`'s silent normalization and platform-aware
    /// separators would corrupt the loud-failure guarantees that keep
    /// malformed input from mis-routing downstream.
    pub fn to_file_path(&self) -> std::path::PathBuf {
        let mut pb = std::path::PathBuf::new();
        for seg in &self.segments {
            pb.push(seg.as_str());
        }
        pb
    }

    /// The segment immediately below `prefix`, when `self` is a direct child
    /// of it (`prefix/a`). Replaces the ad hoc
    /// `rsplit_once('/').map(|(_, n)| n)` leaf-extraction idiom.
    pub fn child_name_below(&self, prefix: &Self) -> Option<&str> {
        let rest = self.relative_segments(prefix)?;
        (rest.len() == 1).then(|| rest[0].as_str())
    }

    /// Relative segments of `self` beneath `prefix`; `None` when `self` is
    /// not under `prefix`. Replaces the hand-rolled
    /// `strip_prefix("{pfx}/")` pattern.
    pub fn relative_segments(&self, prefix: &Self) -> Option<&[String]> {
        // We need "prefix contains self", i.e. self begins with prefix's segments.
        if !prefix.contains(self) {
            return None;
        }
        Some(&self.segments[prefix.depth()..])
    }

    /// Borrowed slice of all segments.
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    /// Render the last `n` segments joined by `sep` (used by callers that
    /// compare tails of paths without allocating).
    pub fn tail_joined(&self, n: usize, sep: &str) -> String {
        let start = self.segments.len().saturating_sub(n);
        self.segments[start..].join(sep)
    }
}

/// Borrowed view of a path's parent, avoiding an allocation for containment
/// checks. Implements `Deref`-free accessors only.
pub struct ParentView<'a>(&'a ModulePath);

impl ParentView<'_> {
    pub fn depth(&self) -> usize {
        self.0.depth().saturating_sub(1)
    }

    pub fn to_slash(&self) -> String {
        let len = self.0.depth().saturating_sub(1);
        self.0.segments[..len].join("/")
    }

    pub fn to_canonical(&self) -> String {
        let len = self.0.depth().saturating_sub(1);
        self.0.segments[..len].join("::")
    }
}

/// Borrowed view of a path minus its first segment (the "tail"). Complements
/// [`ParentView`] (which drops the last segment). Used when walking a tree
/// downward: descend by `head()`, recurse on `tail()`.
pub struct ParentTailView<'a>(&'a ModulePath);

impl ParentTailView<'_> {
    pub fn is_root(&self) -> bool {
        self.0.depth() <= 1
    }

    pub fn to_slash(&self) -> String {
        if self.0.depth() <= 1 {
            String::new()
        } else {
            self.0.segments[1..].join("/")
        }
    }

    pub fn to_canonical(&self) -> String {
        if self.0.depth() <= 1 {
            String::new()
        } else {
            self.0.segments[1..].join("::")
        }
    }
}

impl fmt::Debug for ModulePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModulePath({})", self.to_canonical())
    }
}

impl fmt::Display for ModulePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_slash())
    }
}

/// Conservative identifier check matching Rust module-name usage in std:
/// ASCII letters/digits/underscore, not starting with a digit. Rejects
/// anything containing separators, dots, or macro punctuation — the exact
/// characters that previously leaked into path strings and broke splits.
fn is_identifier(seg: &str) -> bool {
    let mut chars = seg.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slash_round_trip() {
        let p = ModulePath::from_slash("sys/pal/unix/sync").unwrap();
        assert_eq!(p.to_slash(), "sys/pal/unix/sync");
        assert_eq!(p.to_canonical(), "sys::pal::unix::sync");
        assert_eq!(p.depth(), 4);
        assert_eq!(p.leaf(), "sync");
    }

    #[test]
    fn canonical_round_trip() {
        let p = ModulePath::from_canonical("std::collections::btree::map").unwrap();
        assert_eq!(p.to_slash(), "std/collections/btree/map");
        assert_eq!(p.leaf(), "map");
    }

    #[test]
    fn file_stem_tolerates_mod_and_rs_suffixes() {
        assert_eq!(
            ModulePath::from_file_stem("sys/pal/mod").unwrap().to_slash(),
            "sys/pal"
        );
        assert_eq!(
            ModulePath::from_file_stem("sys/pal/mod.rs").unwrap().to_slash(),
            "sys/pal"
        );
        assert_eq!(
            ModulePath::from_file_stem("collections/btree/map.rs").unwrap().to_slash(),
            "collections/btree/map"
        );
        assert_eq!(ModulePath::from_file_stem("mod.rs").unwrap().is_root(), true);
    }

    #[test]
    fn rejects_malformed_segments() {
        // Malformed input fails loudly instead of silently mis-routing.
        assert!(ModulePath::from_slash("sys//pal").is_none());
        assert!(ModulePath::from_slash("sys/pal.unix").is_none());
        assert!(ModulePath::from_slash("9lives").is_none());
        assert!(ModulePath::from_canonical("a:::b").is_none());
        assert!(ModulePath::from_canonical("a::b.c").is_none());
        // The empty string is the root, not an error.
        assert!(ModulePath::from_slash("").unwrap().is_root());
        assert!(ModulePath::from_canonical("").unwrap().is_root());
    }

    #[test]
    fn parent_child_relations() {
        let p = ModulePath::from_slash("sys/pal/unix/sync").unwrap();
        assert_eq!(p.parent_owned().to_slash(), "sys/pal/unix");
        assert_eq!(p.parent().to_slash(), "sys/pal/unix");
        // p's parent is a direct parent of p.
        assert!(p.parent_owned().is_direct_parent_of(&p));
        // A direct child: p IS its direct parent.
        let child = p.join("mutex");
        assert!(p.is_direct_parent_of(&child));
        // A grandchild: p is NOT its direct parent (but still contains it).
        let grandchild = child.join("rwlock");
        assert!(!p.is_direct_parent_of(&grandchild));
        assert!(p.contains(&grandchild));
        assert!(!grandchild.contains(&p));
    }

    #[test]
    fn relative_segments_replace_strip_prefix() {
        let canon = ModulePath::from_slash("sys/pal/unix/sync").unwrap();
        let prefix = ModulePath::from_slash("sys/pal/unix").unwrap();
        assert_eq!(canon.relative_segments(&prefix), Some(&["sync".to_string()][..]));
        assert_eq!(canon.relative_segments(&ModulePath::root()), Some(canon.segments()));
        assert_eq!(canon.relative_segments(&canon.join("x")), None);
        // Direct-child name extraction replaces the rsplit leaf idiom.
        assert_eq!(canon.child_name_below(&prefix), Some("sync"));
        assert_eq!(canon.child_name_below(&ModulePath::from_slash("sys").unwrap()), None);
    }

    #[test]
    fn root_behaviour() {
        let r = ModulePath::root();
        assert_eq!(r.to_slash(), "");
        assert_eq!(r.to_canonical(), "");
        assert_eq!(r.leaf(), "");
        assert!(r.contains(&ModulePath::from_slash("sys").unwrap()));
        assert_eq!(r.parent_owned(), r);
    }

    #[test]
    fn file_path_boundary_rendering() {
        use std::path::{Path, PathBuf};
        // Nested path renders to a joinable PathBuf.
        let p = ModulePath::from_slash("sys/pal/unix/sync").unwrap();
        let out = PathBuf::from("/tmp/out");
        let joined = out.join(p.to_file_path());
        assert_eq!(joined, Path::new("/tmp/out/sys/pal/unix/sync"));
        // The root maps to an empty path that joins unchanged.
        let r = ModulePath::root();
        assert!(r.to_file_path().as_os_str().is_empty());
        assert_eq!(out.join(r.to_file_path()), out);
    }
}

