//! A named type defined in a module.

use super::super::ModulePath;
use super::QualifiedPath;

/// A named type (struct, enum, union, const, or type alias) defined in a module.
///
/// Its fully-qualified path is *derived*, never stored: it is always
/// `::{lib}::{module_path}::{name}` (serialized qualified path), computed by
/// the owning [`super::ModuleNode`].
#[derive(Clone, Debug)]
pub struct ItemRecord {
    /// The identifier name of the item.
    pub name: String,
    /// Kind of item, determining output structure.
    pub kind: crate::parser::ItemKind,
    /// Source visibility as written in std.
    pub visibility: super::super::Visibility,
    /// Whether the item is re-exported publicly through its module chain.
    pub exported: bool,
    /// Whether the item is explicitly declared in the loader spec (or routed
    /// to a mirror via a known-type stub / re-export shim).
    pub declared: bool,
    /// For type aliases only: the right-hand-side type expression tokens.
    pub alias_rhs: Option<proc_macro2::TokenStream>,
    /// Authoritative definition file (absolute path) used by the field-publicity
    /// checker. Set for declared items; may differ from the module's own file
    /// when the item is reached through a cfg-selected backend.
    pub def_file_abs: Option<String>,
}

impl ItemRecord {
    /// The fully-qualified canonical path of this item within `lib`, rooted at
    /// the given module. Rendered as a serialized qualified path,
    /// `::lib::module::name` (root module omits the empty middle segment).
    pub fn qualified_path(&self, lib: &str, module: &ModulePath) -> QualifiedPath {
        QualifiedPath {
            lib: lib.to_string(),
            module: module.clone(),
            item: Some(self.name.clone()),
        }
    }
}
