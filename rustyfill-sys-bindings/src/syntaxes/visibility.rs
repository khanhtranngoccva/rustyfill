//! Source visibility, as written in the std source.
//!
//! One enum for both places visibility matters: items (`pub struct X`) and
//! `use` statements (`pub use x::Y`). Historically these were two separate
//! enums — `parser::ItemVisibility` (Private/Public/Restricted) and
//! `syntaxes::Visibility` (Public/PubCrate/Private) — with ad-hoc mappings
//! between them at each call site. The distinction that actually mattered was
//! always a single boolean: *is this plain `pub`?* Restricted scopes
//! (`pub(crate)`, `pub(super)`, `pub(in path)`) behave identically in both
//! roles — neither makes an item reachable through re-exports nor qualifies a
//! `use` as a public re-export — so they collapse into one variant.

/// Visibility of a parsed declaration, as written in the source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visibility {
    /// No visibility modifier (private to its defining module).
    Private,
    /// Plain `pub` — visible everywhere; qualifies as a public re-export.
    Public,
    /// `pub(crate)` / `pub(super)` / `pub(in path)` — restricted scope. NOT
    /// public: does not make the item reachable through re-exports outside its
    /// scope, and does not qualify a `use` statement as a public re-export.
    Restricted,
}

impl Visibility {
    /// True for plain `pub`. Restricted visibilities are NOT public: they do
    /// not make the item reachable through re-exports outside their scope.
    pub fn is_public(&self) -> bool {
        matches!(self, Visibility::Public)
    }
}
