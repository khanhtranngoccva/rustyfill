//! The complete binding model: a forest of per-library module trees.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use super::super::ModulePath;
use super::{FileForm, ImportEdge, ItemRecord, ModuleNode, NodeStatus, QualifiedPath};
use crate::parser::ParsedSource;
use crate::syntaxes::{PathSegment, UseKind, Visibility};

/// The complete binding model: a forest of per-library module trees plus a
/// reverse index from fully-qualified paths to their locations, so lookups by
/// canonical string (which the spec and diagnostics still speak in) stay cheap.
#[derive(Default)]
pub struct BindingModel {
    /// One root per target library, keyed by lib name.
    roots: BTreeMap<String, ModuleNode>,
    /// Reverse index: canonical path string → the structured address it names.
    /// Kept in sync eagerly on every mutation that registers an item.
    by_canonical: BTreeMap<String, QualifiedPath>,
    /// Relative file paths that were actually written to disk during the emit
    /// phase. The manifest must only reference files that exist; nodes whose
    /// items all failed the declaration gate are registered but never emitted.
    emitted_files: BTreeSet<String>,
}

impl BindingModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ensure a library root exists and return it.
    pub fn ensure_lib(&mut self, lib: &str) -> &mut ModuleNode {
        self.roots
            .entry(lib.to_string())
            .or_insert_with(|| ModuleNode::root(lib))
    }

    /// Borrow a library root, if present.
    pub fn lib(&self, lib: &str) -> Option<&ModuleNode> {
        self.roots.get(lib)
    }

    /// Mutable borrow of a library root, if present.
    pub fn lib_mut(&mut self, lib: &str) -> Option<&mut ModuleNode> {
        self.roots.get_mut(lib)
    }

    /// All library names present, sorted.
    pub fn libs(&self) -> Vec<String> {
        self.roots.keys().cloned().collect()
    }

    /// Descend (creating nodes as needed) to the module at `module_path` within
    /// `lib`, setting its file form to `file`. Returns the node.
    pub fn ensure_module(
        &mut self,
        lib: &str,
        module_path: &ModulePath,
        file: Option<FileForm>,
    ) -> &mut ModuleNode {
        self.ensure_module_status(lib, module_path, file, NodeStatus::Emittable)
    }

    /// Like [`Self::ensure_module`] but also sets the node's lifecycle status.
    pub fn ensure_module_status(
        &mut self,
        lib: &str,
        module_path: &ModulePath,
        file: Option<FileForm>,
        status: NodeStatus,
    ) -> &mut ModuleNode {
        let segments = module_path.segments();
        let mut cursor = self.ensure_lib(lib);
        for (i, seg) in segments.iter().enumerate() {
            let next = cursor.children.entry(seg.clone()).or_insert_with(|| {
                ModuleNode::new_child(
                    seg,
                    &ModulePath::from_segments(segments[..i].to_vec()).unwrap(),
                    lib,
                )
            });
            if i == segments.len() - 1 {
                if let Some(f) = file {
                    next.file = Some(f);
                }
                next.status = status;
                cursor = next;
            } else {
                cursor = next;
            }
        }
        cursor
    }

    /// Borrow the module at `module_path` within `lib`, if it exists.
    pub fn module(&self, lib: &str, module_path: &ModulePath) -> Option<&ModuleNode> {
        let root = self.roots.get(lib)?;
        let mut cursor = root;
        for seg in module_path.segments() {
            cursor = cursor.children.get(seg)?;
        }
        Some(cursor)
    }

    /// Mutable borrow of the module at `module_path` within `lib`, if present.
    pub fn module_mut(&mut self, lib: &str, module_path: &ModulePath) -> Option<&mut ModuleNode> {
        let root = self.roots.get_mut(lib)?;
        let mut cursor = root;
        for seg in module_path.segments() {
            cursor = cursor.children.get_mut(seg)?;
        }
        Some(cursor)
    }

    /// Register a parsed source file into the tree: create its module node (with
    /// the correct file form), attach every item it defines, and record its
    /// `use` statements as import edges. Inline modules become child nodes.
    /// Returns the relative file path that was registered.
    ///
    /// When `register_inline_files` is true, each inline module declared in the
    /// source additionally occupies its own synthesized file node
    /// (`<dir>/<name>/mod.rs`) with the given `status`. This mirrors the legacy
    /// resolver/cache behaviour where inline modules were registered as
    /// standalone emittable files, which is what makes their content appear in
    /// the manifest. Pass false to keep inline modules purely structural
    /// (child nodes without a file) once the emitter stops emitting them.
    pub fn register_source(
        &mut self,
        lib: &str,
        rel_path: &str,
        status: NodeStatus,
        source: &crate::parser::ParsedSource,
    ) -> String {
        self.register_source_with_inlines(lib, rel_path, status, source, true)
    }

    /// Variant of [`Self::register_source`] with explicit control over whether
    /// inline modules get their own file nodes.
    pub fn register_source_with_inlines(
        &mut self,
        lib: &str,
        rel_path: &str,
        status: NodeStatus,
        source: &crate::parser::ParsedSource,
        register_inline_files: bool,
    ) -> String {
        let mp = ModulePath::from_file_stem(rel_path).unwrap_or_else(ModulePath::root);
        let is_mod_rs = rel_path.ends_with("/mod.rs") || rel_path == "mod.rs";
        let form = if is_mod_rs {
            FileForm::Dir
        } else {
            FileForm::Leaf
        };
        let exported_names = public_reexport_names(source);
        // Stage records first so we don't hold a mutable borrow of the node
        // while mutating `self.by_canonical` in the same scope.
        let mut staged: Vec<(ModulePath, ItemRecord)> = Vec::new();
        for item in &source.items {
            staged.push((
                mp.clone(),
                ItemRecord {
                    name: item.name.clone(),
                    kind: item.kind,
                    visibility: item.visibility,
                    exported: exported_names.contains(&item.name),
                    declared: false,
                    alias_rhs: item.alias_rhs.clone(),
                    def_file_abs: None,
                },
            ));
        }
        for (mod_name, mod_items) in &source.inline_modules {
            let inline_mp = mp.join(mod_name);
            for item in mod_items {
                staged.push((
                    inline_mp.clone(),
                    ItemRecord {
                        name: item.name.clone(),
                        kind: item.kind,
                        visibility: item.visibility,
                        exported: exported_names.contains(&item.name),
                        declared: false,
                        alias_rhs: item.alias_rhs.clone(),
                        def_file_abs: None,
                    },
                ));
            }
        }
        let imports: Vec<ImportEdge> = source
            .use_statements
            .iter()
            .map(|stmt| ImportEdge {
                stmt: stmt.clone(),
                target: None,
            })
            .collect();

        // Set the file form and import edges on the module node. Inline
        // modules are child nodes of this module (their items were staged above
        // under `mp.join(mod_name)`); only when `register_inline_files` is set
        // do they additionally claim their own synthesized `<name>/mod.rs` file
        // node, matching the legacy cache's treatment of inline modules.
        {
            let node = self.ensure_module_status(lib, &mp, Some(form), status);
            node.imports = imports;
        }
        if register_inline_files {
            for (mod_name, _) in &source.inline_modules {
                let inline_mp = mp.join(mod_name);
                // Inline modules are synthesized directory modules: their
                // content lives in the parent file but they occupy their own
                // `<name>/mod.rs` position in the output tree.
                let node = self.ensure_module_status(lib, &inline_mp, Some(FileForm::Dir), status);
                let _ = node;
            }
        }
        // Insert each staged record and update the reverse index. Each iteration
        // takes a fresh mutable borrow so the node borrow and the index borrow
        // never overlap.
        for (node_mp, rec) in staged {
            let addr = QualifiedPath::item(lib, node_mp.clone(), &rec.name);
            let canonical = addr.to_canonical();
            {
                let node = self.ensure_module_status(lib, &node_mp, None, status);
                node.insert_item(rec);
            }
            self.by_canonical.insert(canonical, addr);
        }
        form.rel_path(&mp)
    }

    /// Register a synthesized file (re-export shim or glob alias) that has no
    /// parsed source of its own. Creates the module node with the given status
    /// and file form derived from `rel_path`. If the node already exists with a
    /// different file form (e.g., a leaf file registered by discovery), the
    /// existing form wins — a shim at `<module>/mod.rs` must not overwrite the
    /// `FileForm::Leaf` set by the real `<module>.rs` source.
    pub fn register_synthetic(
        &mut self,
        lib: &str,
        rel_path: &str,
        status: NodeStatus,
    ) -> Option<String> {
        let mp = ModulePath::from_file_stem(rel_path)?;
        let is_mod_rs = rel_path.ends_with("/mod.rs") || rel_path == "mod.rs";
        let form = if is_mod_rs {
            FileForm::Dir
        } else {
            FileForm::Leaf
        };
        // Only set the file form if the node doesn't already have one, so we
        // don't clobber a Leaf form established by register_source.
        let existing = self.module(lib, &mp).map(|n| n.file);
        let file_param = if existing.is_none() { Some(form) } else { None };
        let node = self.ensure_module_status(lib, &mp, file_param, status);
        Some(node.rel_file_path().unwrap_or_else(|| rel_path.to_string()))
    }

    /// Record that a binding file was successfully written to disk. Called by
    /// the emit phase after each successful `emit_binding_file` call. The
    /// manifest uses this set (not node status alone) to decide which files to
    /// include, because a node may have items registered but none of them pass
    /// the declaration gate, resulting in no output file.
    pub fn mark_file_emitted(&mut self, rel_path: &str) {
        self.emitted_files.insert(rel_path.to_string());
    }

    /// Whether a relative file path was actually produced during emission.
    pub fn is_file_emitted(&self, rel_path: &str) -> bool {
        self.emitted_files.contains(rel_path)
    }

    /// Borrow the set of emitted file paths (used by the manifest emitter).
    pub fn emitted_file_set(&self) -> &BTreeSet<String> {
        &self.emitted_files
    }

    /// Upgrade a previously-registered node to [`NodeStatus::Emittable`] once it
    /// turns out that it carries real output content. No-op when the node does not
    /// exist yet.
    pub fn promote_to_emittable(&mut self, lib: &str, rel_path: &str) {
        let Some(mp) = ModulePath::from_file_stem(rel_path) else {
            return;
        };
        if let Some(node) = self.module_mut(lib, &mp) {
            node.status = NodeStatus::Emittable;
        }
    }

    /// Demote every `Emittable` node that carries a file but no items to
    /// [`NodeStatus::Support`]. Called after the emit phase: nodes whose items
    /// were all filtered out by the declaration gate never produced output, so
    /// they must not appear in the manifest. Synthetic statuses (Shim, Alias)
    /// are unaffected — their content is generated independently of item lists.
    pub fn demote_empty_emittable(&mut self) {
        for root in self.roots.values_mut() {
            demote_node(root);
        }
    }

    /// Visit every module node in the forest (pre-order, depth-first), calling
    /// `f(node)` for each. Used by the manifest emitter to walk the tree
    /// directly instead of re-parsing file-path strings into a throwaway tree.
    pub fn visit_all<F>(&self, mut f: F)
    where
        F: FnMut(&ModuleNode),
    {
        for root in self.roots.values() {
            visit_node(root, &mut f);
        }
    }

    /// Iterate over the per-library roots in sorted order. Each root is the
    /// virtual container for one library (`core`, `alloc`, `std`); its direct
    /// children are that library's top-level modules.
    pub fn libraries(&self) -> impl Iterator<Item = (&str, &ModuleNode)> {
        self.roots.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Insert an item into the module at `module_path` within `lib`, creating
    /// the module (with the given file form) if needed. Records the reverse
    /// index entry for the item's canonical path.
    pub fn insert_item(
        &mut self,
        lib: &str,
        module_path: &ModulePath,
        file: Option<FileForm>,
        item: ItemRecord,
    ) {
        let addr = QualifiedPath::item(lib, module_path.clone(), &item.name);
        let canonical = addr.to_canonical();
        let node = self.ensure_module(lib, module_path, file);
        node.insert_item(item);
        self.by_canonical.insert(canonical, addr);
    }

    /// Mark the item at `canonical` as declared, overriding its def file.
    pub fn mark_declared(&mut self, canonical: &str, def_file_abs: Option<String>) {
        // Copy out of the reverse index first so the mutable descent below is
        // not fighting an outstanding borrow of `self`.
        let addr = match self.by_canonical.get(canonical) {
            Some(a) => a.clone(),
            None => return,
        };
        let leaf = addr.leaf().map(str::to_string);
        if let Some(node) = self.module_mut(&addr.lib, &addr.module) {
            if let Some(leaf) = leaf {
                if let Some(rec) = node.item_mut(&leaf) {
                    rec.declared = true;
                    if let Some(df) = def_file_abs {
                        rec.def_file_abs = Some(df);
                    }
                }
            }
        }
    }

    /// Look up an item record by its canonical path string. Returns the lib
    /// name, the module path, and the record.
    pub fn find_item(&self, canonical: &str) -> Option<(String, ModulePath, &ItemRecord)> {
        let addr = self.by_canonical.get(canonical)?.clone();
        let node = self.module(&addr.lib, &addr.module)?;
        let rec = node.item(addr.leaf()?)?;
        Some((addr.lib, addr.module, rec))
    }

    /// Candidates (canonical strings) for a bare leaf name, across all modules.
    pub fn candidates_for_leaf(&self, leaf: &str) -> Vec<String> {
        let mut out = Vec::new();
        self.for_each_module(&mut |lib, node| {
            if node.items.contains_key(leaf) {
                out.push(QualifiedPath::item(lib, node.module_path.clone(), leaf).to_canonical());
            }
        });
        out.sort();
        out
    }

    /// DFS over every module in every library (preorder). The closure receives
    /// the lib name and the node.
    pub fn for_each_module(&self, f: &mut dyn FnMut(&str, &ModuleNode)) {
        for (lib, root) in &self.roots {
            self.walk(root, lib, f);
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    fn walk(&self, node: &ModuleNode, lib: &str, f: &mut dyn FnMut(&str, &ModuleNode)) {
        f(lib, node);
        for child in node.children.values() {
            self.walk(child, lib, f);
        }
    }

    /// The flat list of `(relative_file_path, lib)` for every node that has a
    /// file, in stable (sorted-by-path) order. Replaces the loose `all_files`
    /// accumulator.
    pub fn files(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        self.for_each_module(&mut |lib, node| {
            if let Some(fp) = node.rel_file_path() {
                out.push((fp, lib.to_string()));
            }
        });
        out.sort();
        out
    }

    /// The set of relative file paths currently marked emittable (status other
    /// than `Support`). Replaces the `emitted_canonicals` set.
    pub fn emitted(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        self.for_each_module(&mut |_lib, node| {
            if node.status != NodeStatus::Support {
                if let Some(fp) = node.rel_file_path() {
                    out.insert(fp);
                }
            }
        });
        out
    }

    /// Direct child module names of the module at `module_path` within `lib`.
    pub fn child_names(&self, lib: &str, module_path: &ModulePath) -> Vec<String> {
        self.module(lib, module_path)
            .map(|n| n.child_names())
            .unwrap_or_default()
    }

    /// Sibling module names of the module at `module_path` within `lib`
    /// (peers sharing the same parent, excluding self).
    pub fn siblings(&self, lib: &str, module_path: &ModulePath) -> Vec<String> {
        if module_path.is_root() {
            return self.child_names(lib, &ModulePath::root());
        }
        let parent = module_path.parent_owned();
        let my_name = module_path.leaf().to_string();
        let mut names = self.child_names(lib, &parent);
        names.retain(|n| n != &my_name);
        names
    }
}

fn visit_node<F>(node: &ModuleNode, f: &mut F)
where
    F: FnMut(&ModuleNode),
{
    f(node);
    for child in node.children.values() {
        visit_node(child, f);
    }
}

/// Recursively demote `Emittable` nodes that have a file but zero items.
fn demote_node(node: &mut ModuleNode) {
    if node.has_file() && node.items.is_empty() && node.status == NodeStatus::Emittable {
        node.status = NodeStatus::Support;
    }
    for child in node.children.values_mut() {
        demote_node(child);
    }
}

/// Compute the set of item names that are publicly re-exported from a module:
/// items defined directly with `pub` visibility plus everything pulled in by
/// `pub use` single imports. (Globs are approximated conservatively — left to
/// the per-item visibility check.)
fn public_reexport_names(parsed: &ParsedSource) -> HashSet<String> {
    let mut names: HashSet<String> = HashSet::new();
    for item in &parsed.items {
        if item.visibility.is_public() {
            names.insert(item.name.clone());
        }
    }
    for stmt in &parsed.use_statements {
        if !matches!(stmt.visibility, Visibility::Public) {
            continue;
        }
        match &stmt.kind {
            UseKind::Single(plist, alias) => {
                let name = alias.clone().or_else(|| {
                    plist.segments.iter().rev().find_map(|s| {
                        if let PathSegment::Named(n) = s {
                            Some(n.clone())
                        } else {
                            None
                        }
                    })
                });
                if let Some(n) = name {
                    names.insert(n);
                }
            }
            UseKind::Glob(_) => {}
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ItemKind;

    fn item(name: &str, vis: Visibility) -> ItemRecord {
        ItemRecord {
            name: name.to_string(),
            kind: ItemKind::Struct,
            visibility: vis,
            exported: vis.is_public(),
            declared: false,
            alias_rhs: None,
            def_file_abs: None,
        }
    }

    #[test]
    fn ensure_module_creates_nested_nodes() {
        let mut m = BindingModel::new();
        let mp = ModulePath::from_slash("sys/pal/unix/sync").unwrap();
        let node = m.ensure_module("std", &mp, Some(FileForm::Dir));
        assert_eq!(node.name, "sync");
        assert_eq!(node.module_path.to_slash(), "sys/pal/unix/sync");
        assert_eq!(node.lib, "std");
        assert!(matches!(node.file, Some(FileForm::Dir)));
        // Intermediate nodes were created too.
        let mid = m
            .module("std", &ModulePath::from_slash("sys/pal").unwrap())
            .unwrap();
        assert_eq!(mid.name, "pal");
        assert!(mid.file.is_none());
    }

    #[test]
    fn insert_and_find_item_by_canonical() {
        let mut m = BindingModel::new();
        let mp = ModulePath::from_slash("collections/btree/map").unwrap();
        m.insert_item(
            "core",
            &mp,
            Some(FileForm::Leaf),
            item("BTreeMap", Visibility::Public),
        );
        let found = m
            .find_item("core::collections::btree::map::BTreeMap")
            .unwrap();
        assert_eq!(found.0, "core");
        assert_eq!(found.2.name, "BTreeMap");
        assert!(m.find_item("core::nonexistent::Thing").is_none());
    }

    #[test]
    fn mark_declared_updates_record() {
        let mut m = BindingModel::new();
        let mp = ModulePath::from_slash("sync/atomic").unwrap();
        m.insert_item(
            "core",
            &mp,
            Some(FileForm::Leaf),
            item("AtomicUsize", Visibility::Public),
        );
        m.mark_declared(
            "core::sync::atomic::AtomicUsize",
            Some("/abs/path.rs".into()),
        );
        let (_, _, rec) = m.find_item("core::sync::atomic::AtomicUsize").unwrap();
        assert!(rec.declared);
        assert_eq!(rec.def_file_abs.as_deref(), Some("/abs/path.rs"));
    }

    #[test]
    fn files_view_lists_only_nodes_with_files() {
        let mut m = BindingModel::new();
        m.ensure_module("std", &ModulePath::from_slash("sys").unwrap(), None); // no file
        m.ensure_module(
            "std",
            &ModulePath::from_slash("sys/pal").unwrap(),
            Some(FileForm::Dir),
        );
        m.ensure_module(
            "std",
            &ModulePath::from_slash("sys/pal/unix").unwrap(),
            Some(FileForm::Leaf),
        );
        let files = m.files();
        assert_eq!(
            files,
            vec![
                ("sys/pal/mod.rs".to_string(), "std".to_string()),
                ("sys/pal/unix.rs".to_string(), "std".to_string()),
            ]
        );
    }

    #[test]
    fn siblings_exclude_self_and_share_parent() {
        let mut m = BindingModel::new();
        for name in ["map", "set", "node"] {
            let mp = ModulePath::from_slash(&format!("collections/btree/{name}")).unwrap();
            m.ensure_module("core", &mp, Some(FileForm::Leaf));
        }
        m.ensure_module(
            "core",
            &ModulePath::from_slash("collections/hashbrown/raw").unwrap(),
            Some(FileForm::Leaf),
        );
        let sibs = m.siblings(
            "core",
            &ModulePath::from_slash("collections/btree/node").unwrap(),
        );
        assert_eq!(sibs, vec!["map".to_string(), "set".to_string()]);
    }

    #[test]
    fn child_names_returns_direct_children_sorted() {
        let mut m = BindingModel::new();
        m.ensure_module("std", &ModulePath::from_slash("sys/zeta").unwrap(), None);
        m.ensure_module("std", &ModulePath::from_slash("sys/alpha").unwrap(), None);
        let kids = m.child_names("std", &ModulePath::from_slash("sys").unwrap());
        assert_eq!(kids, vec!["alpha".to_string(), "zeta".to_string()]);
    }

    #[test]
    fn emitted_excludes_support_nodes() {
        let mut m = BindingModel::new();
        m.ensure_module_status(
            "std",
            &ModulePath::from_slash("a").unwrap(),
            Some(FileForm::Leaf),
            NodeStatus::Emittable,
        );
        m.ensure_module_status(
            "std",
            &ModulePath::from_slash("b").unwrap(),
            Some(FileForm::Leaf),
            NodeStatus::Support,
        );
        let e = m.emitted();
        assert!(e.contains("a.rs"));
        assert!(!e.contains("b.rs"));
    }

    #[test]
    fn candidates_for_leaf_spans_modules() {
        let mut m = BindingModel::new();
        m.insert_item(
            "core",
            &ModulePath::from_slash("map/entry").unwrap(),
            None,
            item("VacantEntry", Visibility::Public),
        );
        m.insert_item(
            "core",
            &ModulePath::from_slash("set/entry").unwrap(),
            None,
            item("VacantEntry", Visibility::Public),
        );
        let cands = m.candidates_for_leaf("VacantEntry");
        assert_eq!(cands.len(), 2);
        assert!(cands.iter().any(|c| c.ends_with("map::entry::VacantEntry")));
        assert!(cands.iter().any(|c| c.ends_with("set::entry::VacantEntry")));
    }
}
