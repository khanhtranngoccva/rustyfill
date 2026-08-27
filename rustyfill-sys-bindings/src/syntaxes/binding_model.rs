//! The binding model: a single tree that every pipeline phase reads and writes.
//!
//! Historically the generator carried five parallel accumulators — a parsed-source
//! cache, a resolver with eight string-keyed maps, a type registry with seven
//! fields, an `all_files` list, and an `emitted_canonicals` set — and reconciled
//! them at the end by re-parsing file-path strings into a throwaway manifest tree.
//! That made one domain concept ("where does this module/item live?") span ~40
//! string-manipulation sites and let whole classes of bugs (duplicate-module
//! collisions, wrong parent scoping, dangling re-exports) slip through.
//!
//! [`BindingModel`] collapses those accumulators into one owned forest: one root
//! per target library (`core`, `alloc`, `std`), each a nested map of
//! [`ModuleNode`]s. Structs, enums, unions, consts, and type aliases hang off
//! their defining module as [`ItemRecord`]s; `use` statements become first-class
//! [`ImportEdge`]s. Every node and item carries its fully-qualified path *by
//! construction* (library + module path + name), so nothing downstream has to
//! reconstruct it from a slash-separated filename.
//!
//! The tree is the source of truth for the phases that consume it; the legacy
//! accumulators are being retired incrementally in favour of the derived views
//! exposed here ([`BindingModel::files`], [`BindingModel::emitted`], …).

use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::parser::{ItemKind, ParsedSource};
use crate::resolver::{PathSegment, UseKind, UseStatement, Visibility};

use super::module_path::ModulePath;

/// How a module's content lives on disk. Encodes the `foo/mod.rs` versus
/// `foo.rs` distinction the resolver previously tracked in two separate maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileForm {
    /// A directory module defined by `<path>/mod.rs`.
    Dir,
    /// A single-file module defined by `<path>.rs`.
    Leaf,
}

impl FileForm {
    /// The relative file path for a module at `module_path` in this form.
    pub fn rel_path(&self, module_path: &ModulePath) -> String {
        let slash = module_path.to_slash();
        match self {
            FileForm::Dir => {
                if slash.is_empty() {
                    "mod.rs".to_string()
                } else {
                    format!("{slash}/mod.rs")
                }
            }
            FileForm::Leaf => format!("{slash}.rs"),
        }
    }

    /// True when `rel_path` (a `.rs` file path) denotes this form at
    /// `module_path`.
    pub fn matches_rel_path(&self, module_path: &ModulePath, rel_path: &str) -> bool {
        self.rel_path(module_path) == rel_path
    }
}

/// Lifecycle status of a module within the generated tree. Drives which files
/// get emitted and which are merely present to support import resolution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NodeStatus {
    /// Discovered and eligible for emission (the default once registered).
    #[default]
    Emittable,
    /// Registered solely to resolve imports / walk parents; not itself emitted.
    Support,
    /// A synthesized forwarding shim (`pub use <target>::Leaf;`).
    Shim,
    /// A glob-re-export alias mirroring a canonical module under another name.
    Alias,
}

/// A named type (struct, enum, union, const, or type alias) defined in a module.
///
/// Its fully-qualified path is *derived*, never stored: it is always
/// `{lib}::{module_path}::{name}`, computed by the owning [`ModuleNode`].
#[derive(Clone, Debug)]
pub struct ItemRecord {
    /// The identifier name of the item.
    pub name: String,
    /// Kind of item, determining output structure.
    pub kind: ItemKind,
    /// Source visibility as written in std.
    pub visibility: crate::parser::ItemVisibility,
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
    /// the given module. Rendered `lib::module::name` (root module omits the
    /// empty middle segment).
    pub fn qualified_path(&self, lib: &str, module: &ModulePath) -> QualifiedPath {
        QualifiedPath {
            lib: lib.to_string(),
            module: module.clone(),
            item: Some(self.name.clone()),
        }
    }
}

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

    /// The immediate sibling names (excluding self), sorted.
    pub fn sibling_names(&self) -> Vec<String> {
        // Siblings are computed by the owner (BindingModel) which sees the parent;
        // this helper is a convenience for callers that already hold the parent.
        Vec::new()
    }
}

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

/// The complete binding model: a forest of per-library module trees plus a
/// reverse index from fully-qualified paths to their locations, so lookups by
/// canonical string (which the spec and diagnostics still speak in) stay cheap.
#[derive(Default)]
pub struct BindingModel {
    /// One root per target library, keyed by lib name.
    roots: BTreeMap<String, ModuleNode>,
    /// Reverse index: canonical path string → (lib, module slash path, item?).
    /// Maintained lazily via [`Self::index_lookup`]; kept in sync on mutation.
    by_canonical: BTreeMap<String, (String, String, Option<String>)>,
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
            let canonical = QualifiedPath::item(lib, node_mp.clone(), &rec.name).to_canonical();
            {
                let node = self.ensure_module_status(lib, &node_mp, None, status);
                node.insert_item(rec);
            }
            self.by_canonical
                .insert(canonical, (lib.to_string(), node_mp.to_slash(), None));
        }
        form.rel_path(&mp)
    }

    /// Attach an unresolved import edge to the module at `rel_path` within `lib`.
    #[allow(dead_code)]
    pub fn add_import_edge(&mut self, lib: &str, rel_path: &str, edge: ImportEdge) {
        let mp = ModulePath::from_file_stem(rel_path).unwrap_or_else(ModulePath::root);
        if let Some(node) = self.module_mut(lib, &mp) {
            node.imports.push(edge);
        }
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
        let canonical = QualifiedPath::item(lib, module_path.clone(), &item.name).to_canonical();
        let node = self.ensure_module(lib, module_path, file);
        node.insert_item(item);
        self.by_canonical
            .insert(canonical, (lib.to_string(), module_path.to_slash(), None));
    }

    /// Mark the item at `canonical` as declared, overriding its def file.
    pub fn mark_declared(&mut self, canonical: &str, def_file_abs: Option<String>) {
        // Copy out of the reverse index first so the mutable descent below is
        // not fighting an outstanding borrow of `self`.
        let entry = match self.by_canonical.get(canonical) {
            Some(e) => e.clone(),
            None => return,
        };
        let (lib, mod_slash, _) = entry;
        let mp = ModulePath::from_slash(&mod_slash).unwrap_or_else(ModulePath::root);
        let leaf = canonical.rsplit("::").next().unwrap_or("");
        if let Some(node) = self.module_mut(&lib, &mp) {
            if let Some(rec) = node.item_mut(leaf) {
                rec.declared = true;
                if let Some(df) = def_file_abs {
                    rec.def_file_abs = Some(df);
                }
            }
        }
    }

    /// Look up an item record by its canonical path string. Returns the lib
    /// name, the module path, and the record.
    pub fn find_item(&self, canonical: &str) -> Option<(String, ModulePath, &ItemRecord)> {
        let (lib, mod_slash, _) = self.by_canonical.get(canonical)?.clone();
        let mp = ModulePath::from_slash(&mod_slash).unwrap_or_else(ModulePath::root);
        let leaf = canonical.rsplit("::").next().unwrap_or("");
        let node = self.module(&lib, &mp)?;
        let rec = node.item(leaf)?;
        Some((lib, mp, rec))
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

    /// Total number of modules across all libraries (including roots).
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        let mut n = 0;
        self.for_each_module(&mut |_l, _n| n += 1);
        n
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
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
    use crate::parser::ItemVisibility;

    fn item(name: &str, vis: ItemVisibility) -> ItemRecord {
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
    fn file_form_renders_correct_paths() {
        let mp = ModulePath::from_slash("sys/pal/unix/sync").unwrap();
        assert_eq!(FileForm::Dir.rel_path(&mp), "sys/pal/unix/sync/mod.rs");
        assert_eq!(FileForm::Leaf.rel_path(&mp), "sys/pal/unix/sync.rs");
        let root = ModulePath::root();
        assert_eq!(FileForm::Dir.rel_path(&root), "mod.rs");
        assert_eq!(FileForm::Leaf.rel_path(&root), ".rs");
    }

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
            item("BTreeMap", ItemVisibility::Public),
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
            item("AtomicUsize", ItemVisibility::Public),
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
            item("VacantEntry", ItemVisibility::Public),
        );
        m.insert_item(
            "core",
            &ModulePath::from_slash("set/entry").unwrap(),
            None,
            item("VacantEntry", ItemVisibility::Public),
        );
        let cands = m.candidates_for_leaf("VacantEntry");
        assert_eq!(cands.len(), 2);
        assert!(cands.iter().any(|c| c.ends_with("map::entry::VacantEntry")));
        assert!(cands.iter().any(|c| c.ends_with("set::entry::VacantEntry")));
    }
}
