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
    mods_by_file: HashMap<String, Vec<String>>,
    /// Visited set for cycle detection during recursive resolution.
    visiting: HashSet<String>,
    /// Full parsed sources indexed by file path, for accessing inline modules.
    sources: HashMap<String, crate::parser::ParsedSource>,
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
        }
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
    pub fn register_mods(&mut self, file_path: &str, mods: Vec<String>) {
        self.mods_by_file.insert(file_path.to_string(), mods);
    }

    /// Register both the file path and its parsed use statements in one call.
    pub fn register_source(&mut self, file_path: &str, source: crate::parser::ParsedSource) {
        self.register_file(file_path);
        self.register_imports(file_path, source.use_statements.clone());
        self.register_mods(file_path, source.mod_declarations.clone());
        self.sources.insert(file_path.to_string(), source);
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
                    // The full path isn't a known module. Check if all but the last
                    // segment form a valid module — the last segment is likely an
                    // item (struct/enum/trait) within that module.
                    let parts: Vec<&str> = resolved.split('/').collect();
                    if parts.len() > 1 {
                        let maybe_item = parts.last().unwrap();
                        let container = parts[..parts.len() - 1].join("/");
                        if self.find_module(&container).is_some() {
                            return Resolution::ItemInFile(container, maybe_item.to_string());
                        }
                    }
                    Resolution::Unresolved
                }
            }
        }
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
    fn resolve_path_segments(&self, segments: &[PathSegment], current_module: &str) -> String {
        if segments.is_empty() {
            return current_module.to_string();
        }

        let first = &segments[0];
        let rest = &segments[1..];

        let base = match first {
            PathSegment::Super => {
                // Go up one level.
                let parts: Vec<&str> = current_module.split('/').collect();
                if parts.len() > 1 {
                    parts[..parts.len() - 1].join("/")
                } else {
                    String::new()
                }
            }
            PathSegment::Crate => {
                // Start from crate root.
                String::new()
            }
            PathSegment::Self_ => {
                // Stay in current module.
                current_module.to_string()
            }
            PathSegment::Named(name) => {
                // Absolute or relative lookup. If there's only one segment,
                // it's relative to current module. Otherwise treat as crate-root.
                if segments.len() == 1 {
                    // Bare name — try local module first.
                    let candidate = if current_module.is_empty() {
                        name.clone()
                    } else {
                        format!("{}/{}", current_module, name)
                    };
                    return candidate;
                }
                // Multi-segment named path starting with a name = crate-relative.
                name.clone()
            }
        };

        if rest.is_empty() {
            return base;
        }

        // Continue resolving rest from base.
        let next_segment = &rest[0];
        let remaining = &rest[1..];

        let joined = match next_segment {
            PathSegment::Named(n) => {
                if base.is_empty() {
                    n.clone()
                } else {
                    format!("{}/{}", base, n)
                }
            }
            PathSegment::Super => {
                let parts: Vec<&str> = base.split('/').collect();
                if parts.len() > 1 {
                    parts[..parts.len() - 1].join("/")
                } else {
                    String::new()
                }
            }
            PathSegment::Crate => String::new(),
            PathSegment::Self_ => base,
        };

        if remaining.is_empty() {
            joined
        } else {
            self.resolve_path_segments(remaining, &joined)
        }
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
        resolve_child: &F,
    ) -> Vec<String>
    where
        F: Fn(&str, &str) -> Option<String>,
    {
        if !visited.insert(module_path.to_string()) {
            return Vec::new();
        }

        let mut results = Vec::new();

        // Look up the file for this module.
        let file_path = match self.find_module_file(module_path) {
            Some(fp) => fp,
            None => return results,
        };

        // Get mod declarations for this file.
        let mods = match self.mods_by_file.get(&file_path) {
            Some(m) => m.clone(),
            None => return results,
        };

        for mod_name in mods {
            // Resolve child name to a file path.
            let child_module = if module_path.is_empty() {
                mod_name.clone()
            } else {
                format!("{}/{}", module_path, mod_name)
            };

            // Try both `X.rs` and `X/mod.rs`.
            let child_file = resolve_child(module_path, &mod_name);
            if let Some(cf) = child_file {
                results.push(cf.clone());
                // Recurse into child.
                let grandchildren = self.discover_children(&child_module, visited, resolve_child);
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
                    // Convert file path to module path for resolution.
                    let target_mod = Self::file_to_module_path_str(target_file);
                    let rel_path = self.module_path_to_super_chain(&current_module, &target_mod);
                    if !rel_path.is_empty() {
                        let last_seg = target_mod.split('/').next_back().unwrap_or("");
                        let alias_key = format!("module:{rel_path}");
                        if seen_paths.insert(alias_key) {
                            // Use non-pub imports since source items may be pub(super).
                            lines.push(format!(
                                "#[allow(unused_imports)] use {rel_path} as {last_seg};"
                            ));
                            lines.push(format!("#[allow(unused_imports)] use {rel_path}::*;"));
                            // Mark this module as glob-imported so individual items
                            // from it are skipped below.
                            globbed_modules.insert(rel_path.clone());
                        }
                    }
                }
                Resolution::ItemInFile(target_path, item_name) => {
                    // Skip if this item matches an ignored path (e.g., Allocator).
                    if ignored_names.contains(&item_name.as_str()) {
                        continue;
                    }
                    // target_path might be a file path or a module path depending on which
                    // code path created this resolution. Normalize to module path.
                    let target_mod = Self::file_to_module_path_str(target_path);
                    let rel_path = self.module_path_to_super_chain(&current_module, &target_mod);
                    // Skip if we already glob-imported this entire module.
                    if globbed_modules.contains(&rel_path) {
                        continue;
                    }
                    if !rel_path.is_empty() {
                        let full = format!("{rel_path}::{item_name}");
                        if seen_paths.insert(full.clone()) {
                            lines.push(format!("#[allow(unused_imports)] use {full};"));
                        }
                    }
                }
                Resolution::GlobModule(target_module) => {
                    let rel_path = self.module_path_to_super_chain(&current_module, target_module);
                    if !rel_path.is_empty() && seen_paths.insert(rel_path.clone()) {
                        lines.push(format!("#[allow(unused_imports)] use {rel_path}::*;"));
                        globbed_modules.insert(rel_path);
                    }
                }
                Resolution::ExternalLibrary(_, _) | Resolution::Unresolved => {}
            }
        }

        lines
    }

    /// Convert an absolute module path to a `super::...` relative chain from
    /// the current module. E.g., from "collections/btree/set" to
    /// "collections/btree/node" yields "super::node".
    fn module_path_to_super_chain(&self, from: &str, to: &str) -> String {
        let from_parts: Vec<&str> = from.split('/').filter(|s| !s.is_empty()).collect();
        let to_parts: Vec<&str> = to.split('/').filter(|s| !s.is_empty()).collect();

        let common_len = from_parts
            .iter()
            .zip(to_parts.iter())
            .take_while(|(a, b)| a == b)
            .count();

        let ups = from_parts.len() - common_len;
        let downs: Vec<&str> = to_parts[common_len..].to_vec();

        let mut segments = Vec::new();
        segments.extend(std::iter::repeat_n("super", ups));
        for d in &downs {
            segments.push(*d);
        }

        if segments.is_empty() {
            String::new()
        } else {
            segments.join("::")
        }
    }
}

impl Default for ModuleResolver {
    fn default() -> Self {
        Self::new()
    }
}
