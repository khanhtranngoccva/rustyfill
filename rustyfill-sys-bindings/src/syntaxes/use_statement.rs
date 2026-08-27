//! Parsed `use` statement syntax.
//!
//! These types are the shared vocabulary for everything that deals with
//! imports: the parser produces them, [`crate::resolver::ModuleResolver`]
//! consumes them, and the binding model attaches them to modules as
//! [`super::ImportEdge`]s. They live here (rather than in `resolver`) because
//! they describe *syntax* — what a `use` declaration looks like once parsed —
//! not how it resolves against a module tree.

use super::Visibility;

/// A parsed `use` statement extracted from a source file.
#[derive(Clone, Debug)]
pub struct UseStatement {
    /// Visibility: `pub`, `pub(crate)`, or private.
    pub visibility: Visibility,
    /// The kind of use statement.
    pub kind: UseKind,
}

#[derive(Clone, Debug)]
pub enum UseKind {
    /// `use path::to::Item;` or `use path::to::Item as Alias;`
    Single(PathSegmentList, Option<String>),
    /// `use path::to::module::*;`
    Glob(PathSegmentList),
}

/// A dotted path like `super::super::cvt_nz` or `crate::cell::UnsafeCell`.
#[derive(Clone, Debug)]
pub struct PathSegmentList {
    pub segments: Vec<PathSegment>,
}

#[derive(Clone, Debug)]
pub enum PathSegment {
    Named(String),
    Super,
    Crate,
    Self_,
}

impl UseStatement {
    /// Check if this is a glob re-export (e.g., `pub use self::unix::*`).
    pub fn is_pub_glob(&self) -> bool {
        matches!(self.visibility, Visibility::Public) && matches!(self.kind, UseKind::Glob(_))
    }

    /// For glob re-exports, return the target path being globbed.
    pub fn glob_target(&self) -> Option<&PathSegmentList> {
        match &self.kind {
            UseKind::Glob(p) if matches!(self.visibility, Visibility::Public) => Some(p),
            _ => None,
        }
    }
}
