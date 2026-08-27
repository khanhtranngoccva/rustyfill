//! A `use` statement attached to its declaring module.

use super::QualifiedPath;
use crate::syntaxes::UseStatement;

/// A `use` statement attached to its declaring module, together with where it
/// resolved. Making imports explicit edges (rather than a side table keyed by
/// file path) is what lets the tree answer "what does this module pull in?"
/// without a second lookup.
#[derive(Clone, Debug)]
pub struct ImportEdge {
    /// The raw parsed `use` statement.
    pub stmt: UseStatement,
    /// Where the statement resolved, if resolvable.
    pub target: Option<QualifiedPath>,
}
