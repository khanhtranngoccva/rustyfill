//! Module resolution for use/import statements.
//!
//! Parses `use` declarations from Rust source files and resolves paths against
//! a known module tree built from canonical files and discovered dependencies.
//! Handles relative paths (`super::X`), crate-relative paths (`crate::X`),
//! glob re-exports (`pub use X::*`), and cross-library imports (`core::X`).
//! Detects circular import chains.

use std::collections::{HashMap, HashSet};

/// A parsed `use` statement extracted from a source file.
#[derive(Clone, Debug)]
pub struct UseStatement {
    /// Visibility: `pub`, `pub(crate)`, or private.
    pub visibility: Visibility,
    /// The kind of use statement.
    pub kind: UseKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Visibility {
    Public,
    PubCrate,
    Private,
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

/// Result of resolving a use statement against the module tree.
#[derive(Clone, Debug)]
pub struct ResolvedImport {
    /// What was imported.
    pub use_stmt: UseStatement,
    /// Where it resolved to, if resolvable.
    pub resolution: Resolution,
}

#[derive(Clone, Debug)]
pub enum Resolution {
    /// Points to a specific file in our module tree.
    File(String),
    /// Points to a specific item within a file.
    ItemInFile(String, String),
    /// Points to a glob re-export: all public items from the target module.
    GlobModule(String),
    /// Cross-library import (core::..., alloc::...). Not resolved locally.
    ExternalLibrary(String, PathSegmentList),
    /// Could not resolve (path doesn't exist in our tree).
    Unresolved,
}

/// Builds a module tree from a set of canonical file paths and resolves
/// use statements against it.
pub struct ModuleResolver {
    /// Maps module path (e.g., "sys/pal/unix/sync") to file path ("sys/pal/unix/sync/mod.rs").
    modules: HashMap<String, String>,
    /// Maps leaf module path (e.g., "sys/pal/unix/sync/mutex") to file path.
    leaves: HashMap<String, String>,
    /// All use statements indexed by their declaring file path.
    imports_by_file: HashMap<String, Vec<UseStatement>>,
    /// External module declarations (`mod X;`) indexed by declaring file path.
    mods_by_file: HashMap<String, Vec<crate::parser::ModDeclaration>>,
    /// Visited set for cycle detection during recursive resolution.
    visiting: HashSet<String>,
    /// Full parsed sources indexed by file path, for accessing inline modules.
    sources: HashMap<String, crate::parser::ParsedSource>,
    /// Set of files that are eligible for emission (discovered via mod
    /// declarations or structural parents). Files registered solely for import
    /// resolution (Phase 1c) are NOT in this set, so generated use statements
    /// won't reference modules that won't be emitted.
    emittable_files: HashSet<String>,
    /// Canonical paths (`lib::module::Leaf`) of types explicitly declared in the
    /// loader spec. Consulted by the existence checks below so they only count
    /// items the emitter will actually mirror. Without this, checks would count
    /// peripheral public items (iterators, cursors, range views, …) that the
    /// emitter now filters out, producing dangling re-exports.
    declared_paths: HashSet<String>,
    /// Library name (`"std"`, `"core"`, `"alloc"`) for each registered file.
    /// Used to build absolute `crate::{lib}::...` import paths at emit time.
    lib_by_file: HashMap<String, String>,
}

impl ModuleResolver {
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            leaves: HashMap::new(),
            imports_by_file: HashMap::new(),
            mods_by_file: HashMap::new(),
            visiting: HashSet::new(),
            sources: HashMap::new(),
            emittable_files: HashSet::new(),
            declared_paths: HashSet::new(),
            lib_by_file: HashMap::new(),
        }
    }

    /// Remember which library a registered file belongs to, so emitted imports
    /// can be written as absolute `crate::{lib}::...` paths.
    pub fn set_file_lib(&mut self, file_path: &str, lib_name: &str) {
        self.lib_by_file.insert(file_path.to_string(), lib_name.to_string());
    }

    /// Library name for a file (falls back to `"std"` when unknown).
    pub fn lib_of_file(&self, file_path: &str) -> &str {
        self.lib_by_file.get(file_path).map(String::as_str).unwrap_or("std")
    }

    /// Populate the set of spec-declared canonical paths. Called once after the
    /// type registry is built, before any use-statement generation, so the
    /// existence checks agree with what the emitter will actually output.
    pub fn set_declared_paths(&mut self, paths: impl IntoIterator<Item = String>) {
        self.declared_paths = paths.into_iter().collect();
    }

    /// Register a canonical file path. This populates the module tree so that
    /// `sys/pal/unix/sync/mod.rs` registers the module `sys/pal/unix/sync`,
    /// and `sys/pal/unix/sync/mutex.rs` registers the leaf `sys/pal/unix/sync/mutex`.
    pub fn register_file(&mut self, rel_path: &str) {
        let stem = rel_path.strip_suffix(".rs").unwrap_or(rel_path);

        if stem.ends_with("/mod") || stem == "mod" {
            // mod.rs → register as module directory
            let module_path = stem.strip_suffix("/mod").unwrap_or("");
            self.modules
                .insert(module_path.to_string(), rel_path.to_string());
        } else {
            // foo.rs → register as leaf module
            self.leaves.insert(stem.to_string(), rel_path.to_string());
        }
    }

    /// Register parsed use statements for a given file.
    pub fn register_imports(&mut self, file_path: &str, stmts: Vec<UseStatement>) {
        self.imports_by_file.insert(file_path.to_string(), stmts);
    }

    /// Register external module declarations (`mod X;`) for a given file.
    pub fn register_mods(&mut self, file_path: &str, mods: Vec<crate::parser::ModDeclaration>) {
        self.mods_by_file.insert(file_path.to_string(), mods);
    }

    /// Register both the file path and its parsed use statements in one call.
    pub fn register_source(&mut self, file_path: &str, source: crate::parser::ParsedSource) {
        let module_path = Self::file_to_module_path_str(file_path);
        self.modules
            .insert(module_path.clone(), file_path.to_string());
        self.leaves.insert(module_path, file_path.to_string());
        self.imports_by_file
            .insert(file_path.to_string(), source.use_statements.clone());
        self.mods_by_file
            .insert(file_path.to_string(), source.mod_declarations.clone());
        self.sources.insert(file_path.to_string(), source);
    }

    /// Mark a file as eligible for emission. Files registered via Phase 1c
    /// (import-driven discovery) should NOT be marked emittable.
    pub fn mark_emittable(&mut self, file_path: &str) {
        self.emittable_files.insert(file_path.to_string());
    }

    /// Check if a file is eligible for emission.
    pub fn is_emittable(&self, file_path: &str) -> bool {
        self.emittable_files.contains(file_path)
    }

    /// All registered source files with their parsed content, keyed by relative
    /// file path. Includes both emittable canonical files and import-discovered
    /// support files (the latter are registered so their types can be resolved
    /// but are not themselves emitted). Consumers building a type registry use
    /// this so that references into support files resolve correctly.
    pub fn registered_sources(&self) -> &HashMap<String, crate::parser::ParsedSource> {
        &self.sources
    }

    /// Get all inline module names from registered sources.
    pub fn get_inline_module_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for source in self.sources.values() {
            for (mod_name, _) in &source.inline_modules {
                if !names.contains(mod_name) {
                    names.push(mod_name.clone());
                }
            }
        }
        names
    }

    /// Resolve all use statements for a given file, returning resolved imports.
    pub fn resolve_file(&mut self, file_path: &str) -> Vec<ResolvedImport> {
        let stmts = match self.imports_by_file.get(file_path) {
            Some(s) => s,
            None => return Vec::new(),
        };

        let current_module = self.file_to_module_path(file_path);

        stmts
            .iter()
            .map(|stmt| ResolvedImport {
                use_stmt: stmt.clone(),
                resolution: self.resolve_statement(stmt, &current_module, file_path),
            })
            .collect()
    }

    /// Get all glob re-exports from a module, recursively resolving them.
    /// Returns a list of (alias_module_path, canonical_module_path) pairs
    /// representing what gets re-exported where.
    pub fn discover_reexport_aliases(&mut self, file_path: &str) -> Vec<(String, String)> {
        let mut results = Vec::new();
        self.discover_reexports_recursive(file_path, &mut results);
        results
    }

    fn discover_reexports_recursive(
        &mut self,
        file_path: &str,
        results: &mut Vec<(String, String)>,
    ) {
        // Cycle detection.
        if !self.visiting.insert(file_path.to_string()) {
            return;
        }

        let current_module = self.file_to_module_path(file_path);
        let resolved = self.resolve_file(file_path);

        for imp in &resolved {
            if let Resolution::GlobModule(target_module) = &imp.resolution {
                // This module does `pub use <target_module>::*`.
                // Everything under target_module gets re-exported here.
                // Record: alias lives at current_module, canonical lives at target_module.
                results.push((current_module.clone(), target_module.clone()));
            }
        }

        // Recurse into child modules declared by this file.
        let children = self.get_child_modules(&current_module, file_path);
        for child_file in children {
            self.discover_reexports_recursive(&child_file, results);
        }

        self.visiting.remove(file_path);
    }

    /// Get child module files declared by `mod X;` or `pub mod X;` in this file.
    fn get_child_modules(&self, parent_module: &str, _file_path: &str) -> Vec<String> {
        let mut children = Vec::new();

        // Look for registered modules whose parent is this module.
        for (mod_path, file) in &self.modules {
            let parts: Vec<&str> = mod_path.split('/').collect();
            if parts.is_empty() {
                continue;
            }
            let parent = parts[..parts.len() - 1].join("/");
            if parent == parent_module {
                children.push(file.clone());
            }
        }

        // Also check leaves.
        for (leaf_path, file) in &self.leaves {
            let parts: Vec<&str> = leaf_path.split('/').collect();
            if !parts.is_empty() {
                let parent = parts[..parts.len() - 1].join("/");
                if parent == parent_module {
                    children.push(file.clone());
                }
            }
        }

        children
    }

    fn resolve_statement(
        &self,
        stmt: &UseStatement,
        current_module: &str,
        current_file: &str,
    ) -> Resolution {
        match &stmt.kind {
            UseKind::Single(path, _alias) => {
                self.resolve_single_path(path, current_module, current_file)
            }
            UseKind::Glob(path) => self.resolve_glob_path(path, current_module, current_file),
        }
    }

    fn resolve_single_path(
        &self,
        path: &PathSegmentList,
        current_module: &str,
        _current_file: &str,
    ) -> Resolution {
        let resolved = self.resolve_path_segments(&path.segments, current_module);

        match resolved.as_str() {
            "core" | "alloc" | "std" => {
                // Cross-library import.
                let remaining = PathSegmentList {
                    segments: path.segments[path.segments.len().saturating_sub(1)..].to_vec(),
                };
                Resolution::ExternalLibrary(resolved, remaining)
            }
            _ => {
                // Try to find in our module tree.
                if let Some(file) = self.find_module(&resolved) {
                    // Exact module match — prefer this over splitting into item-in-file.
                    // Only treat as ItemInFile if the last segment doesn't correspond
                    // to a known module at all.
                    let parts: Vec<&str> = resolved.split('/').collect();
                    if parts.len() > 1 {
                        let maybe_item = parts.last().unwrap();
                        let container = parts[..parts.len() - 1].join("/");
                        // If the full resolved path is a known module, keep it as File.
                        // Only split if the container is a module but the full path isn't.
                        if self.find_module(&resolved).is_none()
                            && self.find_module(&container).is_some()
                        {
                            return Resolution::ItemInFile(container, maybe_item.to_string());
                        }
                    }
                    Resolution::File(file)
                } else {
                    // The full path isn't a known module. Before decomposing into
                    // item-in-file, try a fallback for `super::X` when X isn't directly
                    // in the parent module but exists as a sibling of the parent
                    // (re-exported via the parent's use statements).
                    if path.segments.len() == 2
                        && matches!(path.segments.first(), Some(PathSegment::Super))
                    {
                        let last_name = path.segments.iter().find_map(|s| {
                            if let PathSegment::Named(n) = s {
                                Some(n.clone())
                            } else {
                                None
                            }
                        });
                        if let Some(last_name) = last_name {
                            // Go up two levels and look for the name there.
                            let parts: Vec<&str> = current_module.split('/').collect();
                            if parts.len() > 2 {
                                let grandparent = parts[..parts.len() - 2].join("/");
                                let candidate = format!("{}/{}", grandparent, last_name);
                                if let Some(file) = self.find_module(&candidate) {
                                    return Resolution::File(file);
                                }
                            }
                        }
                    }

                    // Check if all but the last segment form a valid module — the last
                    // segment is likely an item (struct/enum/trait) within that module.
                    let parts: Vec<&str> = resolved.split('/').collect();
                    if parts.len() > 1 {
                        let maybe_item = parts.last().unwrap();
                        let container = parts[..parts.len() - 1].join("/");
                        if self.find_module(&container).is_some() {
                            return Resolution::ItemInFile(container, maybe_item.to_string());
                        }
                    }

                    // If the path started with a bare Named segment (not super/crate/self),
                    // the original resolution was crate-relative. Try resolving relative
                    // to the current module instead — this handles cases like `entry::Entry`
                    // in `collections/btree/map.rs` where `entry` is a local child module.
                    if matches!(path.segments.first(), Some(PathSegment::Named(_)))
                        && path.segments.len() > 1
                    {
                        let local_resolved =
                            self.resolve_path_segments_local(&path.segments, current_module);
                        let lparts: Vec<&str> = local_resolved.split('/').collect();
                        if let Some(file) = self.find_module(&local_resolved) {
                            if lparts.len() > 1 {
                                let lmaybe_item = lparts.last().unwrap();
                                let lcontainer = lparts[..lparts.len() - 1].join("/");
                                if self.find_module(&local_resolved).is_none()
                                    && self.find_module(&lcontainer).is_some()
                                {
                                    return Resolution::ItemInFile(
                                        lcontainer,
                                        lmaybe_item.to_string(),
                                    );
                                }
                            }
                            return Resolution::File(file);
                        } else if lparts.len() > 1 {
                            let lmaybe_item = lparts.last().unwrap();
                            let lcontainer = lparts[..lparts.len() - 1].join("/");
                            if self.find_module(&lcontainer).is_some() {
                                return Resolution::ItemInFile(lcontainer, lmaybe_item.to_string());
                            }
                        }
                    }

                    Resolution::Unresolved
                }
            }
        }
    }

    /// Resolve path segments treating the first Named segment as relative to
    /// the current module (rather than crate-relative). Used as fallback when
    /// crate-relative resolution fails for multi-segment named paths.
    fn resolve_path_segments_local(
        &self,
        segments: &[PathSegment],
        current_module: &str,
    ) -> String {
        if segments.is_empty() {
            return current_module.to_string();
        }

        let mut cursor = match &segments[0] {
            PathSegment::Named(name) => {
                if current_module.is_empty() {
                    name.clone()
                } else {
                    format!("{}/{}", current_module, name)
                }
            }
            _ => return self.resolve_path_segments(segments, current_module),
        };

        for seg in &segments[1..] {
            cursor = match seg {
                PathSegment::Named(n) => {
                    if cursor.is_empty() {
                        n.clone()
                    } else {
                        format!("{}/{}", cursor, n)
                    }
                }
                PathSegment::Super => {
                    let parts: Vec<&str> = cursor.split('/').collect();
                    if parts.len() > 1 {
                        parts[..parts.len() - 1].join("/")
                    } else {
                        String::new()
                    }
                }
                PathSegment::Crate => String::new(),
                PathSegment::Self_ => cursor.clone(),
            };
        }

        cursor
    }

    fn resolve_glob_path(
        &self,
        path: &PathSegmentList,
        current_module: &str,
        _current_file: &str,
    ) -> Resolution {
        let resolved = self.resolve_path_segments(&path.segments, current_module);

        if let Some(_file) = self.find_module(&resolved) {
            Resolution::GlobModule(resolved)
        } else {
            Resolution::Unresolved
        }
    }

    /// Resolve a path like `super::super::cvt_nz` or `crate::sys::pal` or
    /// `self::unix` into an absolute module path string.
    ///
    /// Iterates linearly through segments, maintaining a mutable cursor that
    /// tracks the current position in the module tree. This avoids the bug where
    /// recursive calls lost context for Named segments following Super hops.
    fn resolve_path_segments(&self, segments: &[PathSegment], current_module: &str) -> String {
        if segments.is_empty() {
            return current_module.to_string();
        }

        // Determine the anchor point from the first segment.
        let mut cursor = match &segments[0] {
            PathSegment::Super => {
                let parts: Vec<&str> = current_module.split('/').collect();
                if parts.len() > 1 {
                    parts[..parts.len() - 1].join("/")
                } else {
                    String::new()
                }
            }
            PathSegment::Crate => String::new(),
            PathSegment::Self_ => current_module.to_string(),
            PathSegment::Named(name) => {
                // A bare name (single segment) is relative to current module.
                // A multi-segment named path (e.g., foo::bar) is crate-relative.
                if segments.len() == 1 {
                    if current_module.is_empty() {
                        return name.clone();
                    }
                    return format!("{}/{}", current_module, name);
                }
                // Multi-segment: start from crate root.
                name.clone()
            }
        };

        // Process remaining segments iteratively.
        for seg in &segments[1..] {
            cursor = match seg {
                PathSegment::Named(n) => {
                    if cursor.is_empty() {
                        n.clone()
                    } else {
                        format!("{}/{}", cursor, n)
                    }
                }
                PathSegment::Super => {
                    let parts: Vec<&str> = cursor.split('/').collect();
                    if parts.len() > 1 {
                        parts[..parts.len() - 1].join("/")
                    } else {
                        String::new()
                    }
                }
                PathSegment::Crate => String::new(),
                PathSegment::Self_ => cursor.clone(),
            };
        }

        cursor
    }

    /// Find the file for a given module path. Checks both modules (directories with mod.rs)
    /// and leaves (single-file modules).
    fn find_module(&self, module_path: &str) -> Option<String> {
        // Try as a module directory (has mod.rs).
        if let Some(file) = self.modules.get(module_path) {
            return Some(file.clone());
        }

        // Try as a leaf file.
        if let Some(file) = self.leaves.get(module_path) {
            return Some(file.clone());
        }

        // Try appending "/mod" (in case caller passed a directory path without /mod suffix).
        let with_mod = format!("{}/mod", module_path);
        if let Some(file) = self.leaves.get(&with_mod) {
            return Some(file.clone());
        }

        None
    }

    /// Convert a file path like "sys/pal/unix/sync/mod.rs" to its module path "sys/pal/unix/sync".
    pub fn file_to_module_path(&self, file_path: &str) -> String {
        let stem = file_path.strip_suffix(".rs").unwrap_or(file_path);

        if let Some(mod_path) = stem.strip_suffix("/mod") {
            return mod_path.to_string();
        }

        stem.to_string()
    }

    /// Find all registered files whose module path starts with the given prefix.
    pub fn find_files_under(&self, prefix: &str) -> Vec<String> {
        let mut results = Vec::new();
        let pfx = if prefix.is_empty() {
            String::new()
        } else {
            format!("{}/", prefix)
        };

        for (mod_path, file) in &self.modules {
            if mod_path == prefix || mod_path.starts_with(&pfx) {
                results.push(file.clone());
            }
        }

        for (leaf_path, file) in &self.leaves {
            if leaf_path == prefix || leaf_path.starts_with(&pfx) {
                results.push(file.clone());
            }
        }

        results.sort();
        results.dedup();
        results
    }

    /// Compute the parent module file paths for a given file, walking upward
    /// from the immediate parent directory to the crate root. Returns them
    /// in order from nearest ancestor to farthest.
    ///
    /// For "sys/pal/unix/sync/mutex.rs":
    ///   ["sys/pal/unix/sync/mod.rs", "sys/pal/unix/mod.rs", "sys/pal/mod.rs", "sys/mod.rs"]
    /// For "sys/pal/mod.rs":
    ///   ["sys/mod.rs"]
    pub fn get_parent_module_paths(&self, file_path: &str) -> Vec<String> {
        // Use the module path (drops trailing /mod) so that mod.rs files don't
        // produce themselves as their own parent.
        let module_path = self.file_to_module_path(file_path);
        let parts: Vec<&str> = module_path.split('/').filter(|s| !s.is_empty()).collect();
        let mut parents = Vec::new();

        // Walk from deepest parent up to crate root.
        for depth in (0..parts.len()).rev() {
            let parent_parts: Vec<&str> = parts[..depth].to_vec();
            if parent_parts.is_empty() {
                break;
            }
            let parent_mod = parent_parts.join("/");
            parents.push(format!("{}/mod.rs", parent_mod));
        }

        parents.reverse();
        parents
    }

    /// Discover all child files transitively under a module by following
    /// `mod X;` declarations. The caller must have already registered the
    /// parent module's parsed source (via `register_source`) and ensured
    /// the corresponding std source files exist on disk.
    ///
    /// Takes the module path (e.g., "sys/pal") and a closure that resolves
    /// a relative child name to an absolute source path string (e.g.,
    /// `"sync"` → `"sys/pal/unix/sync/mod.rs"`). The closure returns None
    /// if the child file doesn't exist on disk.
    ///
    /// Returns deduplicated file paths of all discovered descendants.
    pub fn discover_children<F>(
        &mut self,
        module_path: &str,
        visited: &mut HashSet<String>,
        cfg: &crate::parser::CfgContext,
        resolve_child: &F,
    ) -> Vec<String>
    where
        F: Fn(&str, &str) -> Option<String>,
    {
        // Look up the file for this module first — if it hasn't been registered
        // yet (e.g., because the parent's discover_children recursed ahead of the
        // outer loop that registers children), bail out without marking visited so
        // the caller can retry once registration is complete.
        let file_path = match self.find_module_file(module_path) {
            Some(fp) => fp,
            None => return Vec::new(),
        };

        // Only mark visited after confirming the module exists. This allows a
        // prior speculative recursion (from the parent) to fail silently, then
        // succeed on the real call from discover_and_register's child loop.
        if !visited.insert(module_path.to_string()) {
            return Vec::new();
        }

        let mut results = Vec::new();

        // Get mod declarations for this file.
        let mods = match self.mods_by_file.get(&file_path) {
            Some(m) => m.clone(),
            None => return results,
        };

        for md in &mods {
            // Skip modules gated by cfg predicates that evaluate to false
            // (e.g., #[cfg(test)] modules are invisible outside test builds).
            if crate::parser::is_cfg_inactive(&md.attrs, cfg) {
                continue;
            }

            let mod_name = &md.name;
            // Resolve child name to a file path.
            let child_module = if module_path.is_empty() {
                mod_name.clone()
            } else {
                format!("{}/{}", module_path, mod_name)
            };

            // Try both `X.rs` and `X/mod.rs`.
            let child_file = resolve_child(module_path, mod_name);
            if let Some(cf) = child_file {
                results.push(cf.clone());
                // Recurse into child.
                let grandchildren =
                    self.discover_children(&child_module, visited, cfg, resolve_child);
                results.extend(grandchildren);
            }
        }

        results.sort();
        results.dedup();
        results
    }

    /// Find the registered file for a module path.
    fn find_module_file(&self, module_path: &str) -> Option<String> {
        // Try as a module directory (has mod.rs).
        if let Some(file) = self.modules.get(module_path) {
            return Some(file.clone());
        }
        // Try as a leaf file.
        if let Some(file) = self.leaves.get(module_path) {
            return Some(file.clone());
        }
        None
    }

    /// Whether a parsed item counts as *emitted* content: it must be a data
    /// structure AND explicitly declared in the spec. This mirrors the emitter's
    /// own filter so that use-statement generation doesn't reference modules or
    /// items the emitter stripped out (iterators, cursors, range views, …).
    fn is_emitted_item(&self, item: &crate::parser::ParsedItem, module_path: &str) -> bool {
        let is_data_type = matches!(
            item.kind,
            crate::parser::ItemKind::Struct
                | crate::parser::ItemKind::Enum
                | crate::parser::ItemKind::Union
                | crate::parser::ItemKind::TypeAlias
        );
        if !is_data_type {
            return false;
        }
        // If no declared set was supplied, fall back to the old behaviour
        // (count every data type) so callers that don't care still work.
        if self.declared_paths.is_empty() {
            return true;
        }
        // Build the module+leaf portion with `::` separators (matching the
        // canonical path format used by the registry), then try each known
        // library prefix. The resolver doesn't track which library a file
        // belongs to, so we check against all three (core/alloc/std).
        let module_qualified = module_path.replace('/', "::");
        let suffix = if module_qualified.is_empty() {
            item.name.clone()
        } else {
            format!("{}::{}", module_qualified, item.name)
        };
        for lib in ["core", "alloc", "std"] {
            if self
                .declared_paths
                .contains(&format!("{}::{}", lib, suffix))
            {
                return true;
            }
        }
        false
    }

    /// Check if a module file will emit any data-structure content. Modules whose
    /// only public items are now filtered out (e.g., a set-side entry module whose
    /// types all route to their map-side mirrors) correctly report empty here, so
    /// stale re-exports of them are dropped.
    pub fn module_has_items(&self, file_path: &str) -> bool {
        let module_path = Self::file_to_module_path_str(file_path);
        match self.sources.get(file_path) {
            Some(s) => s
                .items
                .iter()
                .any(|i| self.is_emitted_item(i, &module_path)),
            None => false,
        }
    }

    /// Check whether a specific named item (struct, enum, union, const, type alias)
    /// will actually be emitted from the given file path.
    pub fn item_exists_in_module(&self, file_path: &str, item_name: &str) -> bool {
        let module_path = Self::file_to_module_path_str(file_path);
        match self.sources.get(file_path) {
            Some(s) => s
                .items
                .iter()
                .any(|i| i.name == item_name && self.is_emitted_item(i, &module_path)),
            None => false,
        }
    }

    /// Raw existence probe: does an item with this name appear anywhere in the
    /// parsed source for the file — as a top-level item OR as an inline module?
    /// Unlike [`item_exists_in_module`], this ignores the spec-declaration filter,
    /// because import emission must succeed even when the imported name is a
    /// submodule (e.g., `marker`) or a declared type whose *container* file has
    /// no other declared items. Used only to decide whether a resolved import
    /// points at something real; it never gates what gets emitted.
    pub fn item_present_raw(&self, file_path: &str, item_name: &str) -> bool {
        match self.sources.get(file_path) {
            Some(s) => {
                s.items.iter().any(|i| i.name == item_name)
                    || s.inline_modules.iter().any(|(name, _)| name == item_name)
            }
            None => false,
        }
    }

    /// Follow a one-level re-export chain. When a container module does not
    /// *define* `item_name` but *imports* it under that same name (via a plain
    /// `use path::to::Item;`), return the canonical module path of the module
    /// that actually defines it. This lets callers emit an import pointing at
    /// the defining module rather than dropping the reference.
    ///
    /// Example: `collections/btree/set` does not define `SetValZST`; it carries
    /// `use super::set_val::SetValZST;`. Following that yields the module path
    /// `collections/btree/set_val`, so a dependent can emit
    /// `use ...::set_val::SetValZST;` instead of leaving the bare name dangling.
    /// Returns `None` if the item isn't defined locally nor re-exported by a
    /// single resolvable use statement.
    fn follow_reexport_to_defining_module(
        &self,
        container_file: &str,
        item_name: &str,
    ) -> Option<String> {
        // If the container defines the item itself, nothing to follow.
        if self.item_present_raw(container_file, item_name) {
            return None;
        }
        let src = self.sources.get(container_file)?;
        let container_mod = Self::file_to_module_path_str(container_file);
        for stmt in &src.use_statements {
            if let UseKind::Single(path, alias) = &stmt.kind {
                // The imported binding must be exactly `item_name` (either the
                // last named segment or an explicit `as` alias).
                let bound_name = match alias.as_deref() {
                    Some(a) => a.to_string(),
                    None => path.segments.iter().rev().find_map(|s| match s {
                        PathSegment::Named(n) => Some(n.clone()),
                        _ => None,
                    })?,
                };
                if bound_name != item_name {
                    continue;
                }
                let resolved = self.resolve_path_segments(&path.segments, &container_mod);
                // A use statement whose resolved target IS a known module is a
                // module binding (`use super::btree;`, `use X::{self}`), not an
                // item re-export. Following it would strip the module's own
                // segment and yield its parent — emitting a bogus import like
                // `use crate::alloc::::collections`. Only follow uses whose
                // target extends past a module into an actual item.
                if self.find_module(&resolved).is_some() {
                    continue;
                }
                // Resolve the use target relative to the container module. The
                // resolved string ends in the item name (e.g.,
                // `collections/btree/set_val/SetValZST`), so strip the final
                // segment to get the defining *module*, then look that up.
                let parts: Vec<&str> = resolved.split('/').collect();
                if parts.len() > 1 {
                    let def_mod = parts[..parts.len() - 1].join("/");
                    if let Some(def_file) = self.find_module(&def_mod) {
                        return Some(Self::file_to_module_path_str(&def_file));
                    }
                }
            }
        }
        None
    }

    /// Convert a file path to a module path string (strip .rs, strip trailing /mod).
    fn file_to_module_path_str(file: &str) -> String {
        let stem = file.strip_suffix(".rs").unwrap_or(file);
        stem.strip_suffix("/mod").unwrap_or(stem).to_string()
    }

    /// Generate `pub use` statements for a file's resolved imports that point
    /// to modules within our own tree (not external libraries like core/alloc).
    /// Returns lines of Rust code ready to be inserted into an emitted binding file.
    ///
    /// The `ignored_names` parameter lists leaf identifiers (e.g., `"Allocator"`)
    /// that should be deliberately skipped — these correspond to traits or types
    /// declared in the spec's `ignored_paths` that the emitter strips from output.
    pub fn emit_use_statements_for_file(
        &mut self,
        file_path: &str,
        ignored_names: &[&str],
    ) -> Vec<String> {
        let resolved = self.resolve_file(file_path);
        let current_module = self.file_to_module_path(file_path);
        let mut lines = Vec::new();
        let mut seen_paths: HashSet<String> = HashSet::new();
        // Track modules that already had a glob import emitted, so we can skip
        // redundant individual item imports from those same modules.
        let mut globbed_modules: HashSet<String> = HashSet::new();

        for ri in resolved {
            match &ri.resolution {
                Resolution::File(target_file) => {
                    // Skip if this module wasn't emitted (e.g., discovered via
                    // Phase 1c import resolution but not in the canonical tree).
                    if !self.is_emittable(target_file) {
                        continue;
                    }
                    // For module-level imports (bringing a submodule into scope
                    // by name, e.g. `use node::marker;`), we only need the
                    // module to exist in the emitted tree — not for it to pass
                    // the spec-declaration item filter. The `module_has_items`
                    // check is appropriate for item-level imports but would
                    // incorrectly drop valid submodule references when the
                    // module's items haven't been spec-filtered yet at this
                    // phase of the pipeline.
                    // Convert file path to module path for resolution.
                    let target_mod = Self::file_to_module_path_str(target_file);
                    let lib_name = self.lib_of_file(target_file);
                    let abs_path = self.module_path_to_abs_chain(&target_mod, lib_name);
                    if !abs_path.is_empty() {
                        let last_seg = target_mod.split('/').next_back().unwrap_or("");
                        // Preserve the original alias from the source import when
                        // present (e.g. `use crate::sys::sync as sys;` must keep
                        // the `as sys` suffix so `sys::Mutex` resolves downstream).
                        let bind_name = match &ri.use_stmt.kind {
                            UseKind::Single(_, Some(alias)) => alias.as_str(),
                            _ => last_seg,
                        };
                        let alias_key = format!("module:{abs_path}:{bind_name}");
                        if seen_paths.insert(alias_key) {
                            // An import whose resolved target is the *enclosing* module
                            // (e.g. `use crate::sync::poison;` written in
                            // `sync/poison/mutex.rs`) binds the parent module's own name
                            // (`poison`). The parent does not contain a child by its own
                            // name, so no `use ... as poison;` can be emitted. The
                            // qualifier-route mechanism (recorded during mirror phase)
                            // rewrites all `<parent_leaf>::X` references to absolute
                            // paths at emit time. No import needed.
                            if current_module == target_mod {
                                // Enclosing module: handled by qualifier routes.
                            } else {
                                // Bring the module into scope under its original name so
                                // that `X::item` paths resolve (e.g., `marker::Mut`), then
                                // glob import all items for bare-name access.
                                lines.push(format!(
                                    "#[allow(unused_imports)] use {abs_path} as {bind_name};"
                                ));
                                lines.push(format!(
                                    "#[allow(unused_imports)] use {abs_path}::*;"
                                ));
                                // Mark this module as glob-imported so individual items
                                // from it are skipped below.
                                globbed_modules.insert(abs_path.clone());
                            }
                        }
                    }
                }
                Resolution::ItemInFile(target_path, item_name) => {
                    // Skip if this item matches an ignored path (e.g., Allocator).
                    if ignored_names.contains(&item_name.as_str()) {
                        continue;
                    }
                    // Library of the referring file: all absolute imports are rooted
                    // at `crate::{lib}` so they resolve identically regardless of
                    // where in the include! tree the file lands.
                    let lib_name = self.lib_of_file(file_path);
                    // target_path might be a file path or a module path depending on which
                    // code path created this resolution. Normalize to module path.
                    let mut target_mod = Self::file_to_module_path_str(target_path);
                    // Skip if the target module has no items at all (wasn't emitted).
                    // We don't check for the specific item name here, because items may
                    // be re-exported via use statements rather than defined directly.
                    // However, modules like sys/pal/unix define functions (not structs),
                    // so their parsed items list is empty. For those, also check if the
                    // item is a known non-type (we can't distinguish, so we skip anything
                    // in a module with zero items).
                    let target_file = self.find_module_file(&target_mod);
                    if let Some(tf) = &target_file {
                        if !self.module_has_items(tf) {
                            continue;
                        }
                        // Additionally, check that the specific item actually exists
                        // in the parsed source. If the module was discovered but doesn't
                        // contain this particular name (e.g., io/mod.rs has structs but
                        // not Error), skip the import. Use the raw presence probe here —
                        // NOT the declaration-filtered one — because a valid import can
                        // target a submodule (e.g., `marker`) or a declared type whose
                        // container file carries no other declared items. Filtering by
                        // declaration would silently drop these imports and leave the
                        // generated code referencing unimported names.
                        if !self.item_present_raw(tf, item_name) {
                            // The container doesn't define the item. It may instead
                            // *re-export* it via its own `use` (e.g., set.rs carries
                            // `use super::set_val::SetValZST;`). Follow that single hop
                            // so the emitted import points at the defining module and
                            // the bare name resolves in the dependent file.
                            match self.follow_reexport_to_defining_module(tf, item_name) {
                                Some(def_mod) => {
                                    let def_file = self.find_module(&def_mod);
                                    if def_file.is_none()
                                        || !self.module_has_items(def_file.as_deref().unwrap())
                                    {
                                        continue;
                                    }
                                    target_mod = def_mod;
                                }
                                None => continue,
                            }
                        }
                    } else {
                        continue;
                    }
                    let abs_path = self.module_path_to_abs_chain(&target_mod, lib_name);
                    // Skip if we already glob-imported this entire module.
                    if globbed_modules.contains(&abs_path) {
                        continue;
                    }
                    if !abs_path.is_empty() {
                        // For public re-exports (`pub use`), verify the target name
                        // will actually exist in the emitted tree. Without this check,
                        // a parent module that re-exports many items from a child would
                        // emit `pub use` for types the emitter stripped out (because
                        // they aren't in the spec), producing unresolved imports.
                        // We accept the re-export if the target is either:
                        //   (a) a submodule (inline or file-based) that was extracted,
                        //   (b) an item declared in the spec (will be emitted).
                        // Private `use` statements are safe to keep as-is because they
                        // only bring names into local scope under #[allow(unused_imports)].
                        if ri.use_stmt.visibility == Visibility::Public {
                            let tf = target_file.as_deref().unwrap_or("");
                            let is_submodule = self.find_module(item_name).is_some()
                                || self
                                    .sources
                                    .get(tf)
                                    .map(|s| s.inline_modules.iter().any(|(n, _)| n == item_name))
                                    .unwrap_or(false);
                            let is_declared = self.item_exists_in_module(tf, item_name);
                            if !is_submodule && !is_declared {
                                continue;
                            }
                        }
                        let full = format!("{abs_path}::{item_name}");
                        if seen_paths.insert(full.clone()) {
                            let vis = match ri.use_stmt.visibility {
                                Visibility::Public => "pub use",
                                _ => "use",
                            };
                            lines.push(format!("#[allow(unused_imports)] {vis} {full};"));
                        }
                    }
                }
                Resolution::GlobModule(target_module) => {
                    // Skip if the target module wasn't emitted or has no items.
                    let target_file = self.find_module_file(target_module);
                    if let Some(tf) = &target_file {
                        if !self.is_emittable(tf) || !self.module_has_items(tf) {
                            continue;
                        }
                    } else {
                        continue;
                    }
                    let lib_name = self.lib_of_file(file_path);
                    let abs_path = self.module_path_to_abs_chain(target_module, lib_name);
                    if !abs_path.is_empty() && seen_paths.insert(abs_path.clone()) {
                        let vis = match ri.use_stmt.visibility {
                            Visibility::Public => "pub use",
                            _ => "use",
                        };
                        lines.push(format!("#[allow(unused_imports)] {vis} {abs_path}::*;"));
                        globbed_modules.insert(abs_path);
                    }
                }
                Resolution::ExternalLibrary(_, _) | Resolution::Unresolved => {}
            }
        }

        lines
    }

    /// The leaf name of the module that directly encloses `module_path`, i.e.
    /// the name one would bind with `use super as <name>;`. For
    /// "sync/poison/mutex" this is "poison"; for a top-level module ("sync") it
    /// is None (there is no enclosing named module to alias).
    #[allow(dead_code)]
    fn parent_module_leaf(module_path: &str) -> Option<&str> {
        let parts: Vec<&str> = module_path.split('/').filter(|s| !s.is_empty()).collect();
        if parts.len() >= 2 {
            Some(parts[parts.len() - 2])
        } else {
            None
        }
    }

    /// Convert a target module path to an absolute `crate::std::...` import
    /// path. All emitted imports are absolute so they never depend on the
    /// include!-based module nesting depth or on sibling-name collisions.
    /// The manifest merges all libraries (core, alloc, std) into a single
    /// `crate::std::` wrapper module, so every import is rooted there.
    /// E.g., target "collections/btree/node" yields
    /// "crate::std::collections::btree::node". Returns an empty string when
    /// `to` is empty (nothing to import). Empty segments are dropped as a
    /// defense-in-depth measure so a malformed upstream path can never
    /// produce invalid syntax like `crate::std::::collections`.
    fn module_path_to_abs_chain(&self, to: &str, _lib_name: &str) -> String {
        if to.is_empty() {
            return String::new();
        }
        let clean: Vec<&str> = to.split('/').filter(|s| !s.is_empty()).collect();
        if clean.is_empty() {
            return String::new();
        }
        format!("crate::std::{}", clean.join("::"))
    }
}

impl Default for ModuleResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_source;

    fn reg(r: &mut ModuleResolver, path: &str, src: &str) {
        r.register_source(path, parse_source(src));
        r.mark_emittable(path);
    }

    /// Regression test: a dependent file importing an item that its container
    /// module only *re-exports* (via its own `use`) must still get an import
    /// pointing at the defining module. Mirrors `collections/btree/set/entry.rs`
    /// importing `SetValZST`, which `set.rs` re-exports from `set_val`.
    #[test]
    fn reexported_item_follows_to_defining_module() {
        let mut r = ModuleResolver::new();
        // set_val DEFINES SetValZST.
        reg(
            &mut r,
            "collections/btree/set_val.rs",
            "pub(super) struct SetValZST;\n",
        );
        // set RE-EXPORTS it and defines nothing else relevant.
        reg(
            &mut r,
            "collections/btree/set.rs",
            "use super::set_val::SetValZST;\npub struct BTreeSet<T, A = ()> {}\n",
        );
        // entry imports from super (set).
        reg(
            &mut r,
            "collections/btree/set/entry.rs",
            "use super::{SetValZST, map};\npub struct OccupiedEntry {}\n",
        );
        // sibling map module so `map` resolves.
        reg(&mut r, "collections/btree/map.rs", "pub struct Map {}\n");

        let lines = r.emit_use_statements_for_file("collections/btree/set/entry.rs", &[]);
        let joined = lines.join("\n");
        assert!(
            joined.contains("crate::std::collections::btree::set_val::SetValZST"),
            "expected re-export follow to emit an absolute set_val::SetValZST import, got:\n{joined}"
        );
    }
}
