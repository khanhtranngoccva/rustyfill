//! Fully-qualified pointers into the binding model.

use super::super::ModulePath;

/// A fully-qualified pointer into the binding model: a library, a module within
/// it, and optionally an item within that module. This is the canonical "address"
/// every part of the tree points at.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct QualifiedPath {
    /// Target library (`core` / `alloc` / `std`).
    pub lib: String,
    /// Module position within that library.
    pub module: ModulePath,
    /// Optional item leaf within the module.
    pub item: Option<String>,
}

impl QualifiedPath {
    /// Build a module-level path (no item leaf).
    pub fn module(lib: &str, module: ModulePath) -> Self {
        Self {
            lib: lib.to_string(),
            module,
            item: None,
        }
    }

    /// Build an item-level path.
    pub fn item(lib: &str, module: ModulePath, item: &str) -> Self {
        Self {
            lib: lib.to_string(),
            module,
            item: Some(item.to_string()),
        }
    }

    /// The item leaf, if any.
    pub fn leaf(&self) -> Option<&str> {
        self.item.as_deref()
    }

    /// The module portion rendered canonically (`sys::sync::mutex`). Empty for
    /// the library root.
    pub fn module_canonical(&self) -> String {
        self.module.to_canonical()
    }

    /// The full canonical spelling, a serialized qualified path that fully
    /// encodes its address: `::lib::module::item` (omitting empty parts).
    ///
    /// The leading `::` is the absolute-path marker — it states that the path
    /// is rooted at the library boundary rather than relative to some context.
    /// Without it the library prefix would be lost from the encoding and the
    /// string would no longer be a self-describing qualified path.
    pub fn to_canonical(&self) -> String {
        let mut out = String::from("::");
        out.push_str(&self.lib);
        let mc = self.module.to_canonical();
        if !mc.is_empty() {
            out.push_str("::");
            out.push_str(&mc);
        }
        if let Some(it) = &self.item {
            out.push_str("::");
            out.push_str(it);
        }
        out
    }

    /// The target library name (`core` / `alloc` / `std`).
    pub fn lib(&self) -> &str {
        &self.lib
    }

    /// The path with the library segment removed: `module::item` (omitting
    /// empty parts). This is the form addressed inside the merged wrapper
    /// module of generated code.
    pub fn rest(&self) -> String {
        let mut out = self.module.to_canonical();
        if let Some(it) = &self.item {
            if !out.is_empty() {
                out.push_str("::");
            }
            out.push_str(it);
        }
        out
    }

    /// The absolute import path used inside generated code:
    /// `crate::{wrapper}::{lib}::{module}::{item}`.
    pub fn to_crate_import(&self, wrapper: &str) -> String {
        let mut out = format!("crate::{wrapper}::{}", self.lib);
        let mc = self.module.to_canonical();
        if !mc.is_empty() {
            out.push_str("::");
            out.push_str(&mc);
        }
        if let Some(it) = &self.item {
            out.push_str("::");
            out.push_str(it);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualified_path_renders() {
        let mp = ModulePath::from_slash("collections/btree/map").unwrap();
        let qp = QualifiedPath::item("core", mp.clone(), "BTreeMap");
        // The canonical spelling carries the absolute-path marker and fully
        // encodes lib, module, and item.
        assert_eq!(qp.to_canonical(), "::core::collections::btree::map::BTreeMap");
        assert_eq!(qp.lib(), "core");
        assert_eq!(qp.rest(), "collections::btree::map::BTreeMap");
        assert_eq!(
            qp.to_crate_import("std"),
            "crate::std::core::collections::btree::map::BTreeMap"
        );
        // Root module omits the empty middle segment.
        let qpr = QualifiedPath::item("std", ModulePath::root(), "Task");
        assert_eq!(qpr.to_canonical(), "::std::Task");
        assert_eq!(qpr.rest(), "Task");
        // A module-level path has no item leaf in its rest.
        let qpm = QualifiedPath::module("alloc", ModulePath::from_slash("sync").unwrap());
        assert_eq!(qpm.to_canonical(), "::alloc::sync");
        assert_eq!(qpm.rest(), "sync");
    }
}
