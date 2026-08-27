//! A module node in the binding tree.

use std::collections::BTreeMap;

use super::super::ModulePath;
use super::{FileForm, ImportEdge, ItemRecord, NodeStatus, QualifiedPath};

/// A module node in the binding tree. Holds its own file form, lifecycle status,
/// the items it defines, the imports it pulls in, and its child modules.
#[derive(Clone, Debug)]
pub struct ModuleNode {
    /// The module's own name (empty for a library root).
    pub name: String,
    /// Position of this module within its library root.
    pub module_path: ModulePath,
    /// Which target library this node belongs to (`core` / `alloc` / `std`).
    pub lib: String,
    /// How the module's content lives on disk, if it has any. Structural
    /// parents that exist only to support resolution have no file.
    pub file: Option<FileForm>,
    /// Lifecycle status controlling emission.
    pub status: NodeStatus,
    /// Items defined directly in this module, keyed by name.
    pub items: BTreeMap<String, ItemRecord>,
    /// `use` statements declared in this module.
    pub imports: Vec<ImportEdge>,
    /// Child modules, keyed by name.
    pub children: BTreeMap<String, ModuleNode>,
}

impl ModuleNode {
    /// Create a node for the library root of `lib`.
    pub fn root(lib: &str) -> Self {
        Self {
            name: String::new(),
            module_path: ModulePath::root(),
            lib: lib.to_string(),
            file: None,
            status: NodeStatus::Emittable,
            items: BTreeMap::new(),
            imports: Vec::new(),
            children: BTreeMap::new(),
        }
    }

    /// Create a named child node under `parent_path`.
    pub fn new_child(name: &str, parent_path: &ModulePath, lib: &str) -> Self {
        Self {
            name: name.to_string(),
            module_path: parent_path.join(name),
            lib: lib.to_string(),
            file: None,
            status: NodeStatus::Emittable,
            items: BTreeMap::new(),
            imports: Vec::new(),
            children: BTreeMap::new(),
        }
    }

    /// The fully-qualified path of this module (no item leaf).
    pub fn qualified_module(&self) -> QualifiedPath {
        QualifiedPath {
            lib: self.lib.clone(),
            module: self.module_path.clone(),
            item: None,
        }
    }

    /// The relative file path for this node's file form, if it has one.
    pub fn rel_file_path(&self) -> Option<String> {
        self.file.map(|form| form.rel_path(&self.module_path))
    }

    /// True when this node occupies a real file on disk (either form).
    pub fn has_file(&self) -> bool {
        self.file.is_some()
    }

    /// Insert (or update) an item in this module.
    pub fn insert_item(&mut self, item: ItemRecord) {
        self.items.insert(item.name.clone(), item);
    }

    /// Borrow an item by name.
    pub fn item(&self, name: &str) -> Option<&ItemRecord> {
        self.items.get(name)
    }

    /// Mutable borrow of an item by name.
    pub fn item_mut(&mut self, name: &str) -> Option<&mut ItemRecord> {
        self.items.get_mut(name)
    }

    /// A child module by name, if present.
    pub fn child(&self, name: &str) -> Option<&ModuleNode> {
        self.children.get(name)
    }

    /// A mutable child module by name, if present.
    pub fn child_mut(&mut self, name: &str) -> Option<&mut ModuleNode> {
        self.children.get_mut(name)
    }

    /// Names of all direct children, sorted.
    pub fn child_names(&self) -> Vec<String> {
        self.children.keys().cloned().collect()
    }
}
