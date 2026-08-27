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

    /// The full canonical spelling `lib::module::item` (omitting empty parts).
    pub fn to_canonical(&self) -> String {
        let mut out = String::from(&self.lib);
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
        assert_eq!(qp.to_canonical(), "core::collections::btree::map::BTreeMap");
        assert_eq!(
            qp.to_crate_import("std"),
            "crate::std::core::collections::btree::map::BTreeMap"
        );
        // Root module omits the empty middle segment.
        let qpr = QualifiedPath::item("std", ModulePath::root(), "Task");
        assert_eq!(qpr.to_canonical(), "std::Task");
    }
}
