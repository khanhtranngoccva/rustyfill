//! Build script for rustyfill-sys.
//!
//! Orchestrates the binding generation pipeline:
//! 1. Locate the Rust toolchain source tree.
//! 2. Load the loader spec from `rustyfill-sys-bindings`.
//! 3. **Validate** spec paths exist on disk (fail fast).
//! 4. Parse canonical files and discover all inner files transitively via
//!    `mod X;` declarations (evaluating cfg_select! branches for the current target).
//! 5. Validate each parse result and emitted file (fail fast).
//! 6. Discover structural dependencies by walking up parent modules for
//!    re-export resolution.
//! 7. Delegate all emission (preamble modules, binding files, alias files, manifest)
//!    to the bindings crate's emitter module.
//! 8. **Validate** manifest completeness and alias resolution (fail fast).

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use rustyfill_sys_bindings::emitter::{
    check_declared_struct_fields, emit_binding_file, emit_glob_reexport_aliases,
    emit_hierarchical_manifest, emit_preamble_module, TypeRegistry,
};
use rustyfill_sys_bindings::get_loader_spec;
use rustyfill_sys_bindings::parser::{CfgContext, parse_source_with_cfg};
use rustyfill_sys_bindings::resolver::{ModuleResolver, UseKind};
use rustyfill_sys_bindings::validator::ValidationBuilder;

fn main() {
    // Fail immediately if layout randomization is active — it breaks the
    // deterministic layout assumptions that polyfilled mirrors depend on.
    reject_randomize_layout();

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);

    let rust_src = find_rust_source_root();
    let spec = get_loader_spec();
    let cfg = CfgContext::from_env();

    // Collect ignored leaf identifiers and their optional replacements from all
    // targets across the spec. The emitter uses these to strip trait bounds and
    // substitute type positions in generated bindings.
    let mut path_replacement_map: HashMap<String, Option<String>> = HashMap::new();
    for target in &spec.targets {
        for pr in &target.path_replacements {
            let leaf = pr
                .path
                .rsplit_once("::")
                .map(|(_, l)| l.to_string())
                .unwrap_or_else(|| pr.path.clone());
            path_replacement_map.insert(leaf, pr.replacement.clone());
        }
    }
    // Stable ordering for deterministic emission.
    let mut replacement_entries: Vec<(String, Option<String>)> =
        path_replacement_map.into_iter().collect();
    replacement_entries.sort_by_key(|(k, _)| k.clone());
    // Combined slice for emitter: (leaf_name, optional_replacement_text).
    let replacement_entries_slice: Vec<(String, Option<&str>)> = replacement_entries
        .iter()
        .map(|(k, v)| (k.clone(), v.as_deref()))
        .collect();

    // Build per-library ignored struct lists.
    let ignored_structs_by_lib: HashMap<String, Vec<String>> = spec
        .targets
        .iter()
        .map(|t| (t.lib_name.clone(), t.ignored_structs.clone()))
        .collect();

    // Build per-library extra-derive maps (canonical path → list of derive traits).
    let extra_derives_by_lib: HashMap<String, std::collections::HashMap<String, Vec<String>>> =
        spec
            .targets
            .iter()
            .map(|t| {
                let mut map = std::collections::HashMap::new();
                for (path, derives) in &t.extra_derives {
                    map.insert(path.clone(), derives.clone());
                }
                (t.lib_name.clone(), map)
            })
            .collect();

    // Collect all ignored names: both path replacement leaves AND ignored struct
    // leaf names, so the resolver can skip re-exports of items that won't be emitted.
    let mut all_ignored_names: HashSet<String> = replacement_entries_slice
        .iter()
        .map(|(k, _)| k.clone())
        .collect();
    for structs in ignored_structs_by_lib.values() {
        for s in structs {
            if let Some(leaf) = s.rsplit_once("::").map(|(_, l)| l.to_string()) {
                all_ignored_names.insert(leaf);
            } else {
                all_ignored_names.insert(s.clone());
            }
        }
    }
    let mut ignored_name_vec: Vec<String> = all_ignored_names.into_iter().collect();
    ignored_name_vec.sort();
    let ignored_name_refs: Vec<&str> = ignored_name_vec.iter().map(|s| s.as_str()).collect();

    // ── Pre-flight: validate spec paths ────────────────────────────────────
    let mut validator = ValidationBuilder::new();
    validator.check_spec(&spec, &rust_src);

    let mut resolver = ModuleResolver::new();
    let mut processed_parents: HashSet<String> = HashSet::new();
    let mut preamble_emitted: HashSet<String> = HashSet::new();
    // Declarations whose defining file could not be located on disk directly;
    // resolved against the registered module tree after Phase 1.
    let mut pending_declarations: Vec<(String, String)> = Vec::new();

    // ── Phase 0: Emit preamble modules per target library ──────────────────
    for target in &spec.targets {
        if preamble_emitted.insert(target.lib_name.clone()) {
            emit_preamble_module(out_path, &target.lib_name);
        }
    }

    // ── Phase 1: DISCOVER — Parse all files, register with resolver, no emission ──
    let mut parsed_cache: HashMap<String, (rustyfill_sys_bindings::parser::ParsedSource, String)> =
        HashMap::new();

    for target in &spec.targets {
        let lib_src = rust_src.join(&target.lib_name).join("src");

        // Each declared struct drives discovery of its defining file and the
        // module chain above it. The same file may be reached through several
        // declarations; a per-target visited set deduplicates traversal.
        for decl in &target.declared_structs {
            match locate_declared_struct(decl, &lib_src, &cfg) {
                LocatedStruct::Found(def_file) => {
                    let mut parent_visited = HashSet::new();
                    discover_and_register(DiscoverParams {
                        source_rel_path: def_file.as_str(),
                        lib_name: &target.lib_name,
                        lib_src: &lib_src,
                        cfg: &cfg,
                        resolver: &mut resolver,
                        validator: &mut validator,
                        visited: &mut parent_visited,
                        cache: &mut parsed_cache,
                    });
                    // Also ensure the structural parents of the defining file
                    // are registered so re-export aliases resolve.
                    register_parents_of(
                        &def_file,
                        &target.lib_name,
                        &lib_src,
                        &cfg,
                        &mut resolver,
                        &mut parsed_cache,
                        &mut processed_parents,
                    );
                }
                LocatedStruct::NotDefinedOnDisk(path_hint) => {
                    // May live in an inline module or another file; resolved
                    // against the registered module tree after Phase 1.
                    pending_declarations.push((decl.clone(), path_hint));
                }
                LocatedStruct::BadPath(msg) => {
                    eprintln!("cargo:error=[spec] {}", msg);
                    std::process::exit(1);
                }
            }
        }
    }

    // Resolve declarations that could not be located on disk directly (e.g.,
    // types defined in inline modules). Search every registered file's items.
    let unresolved: Vec<(String, String)> = if !pending_declarations.is_empty() {
        let mut still_unresolved = Vec::new();
        for (decl, hint) in pending_declarations {
            let leaf = decl.rsplit("::").next().unwrap_or(&decl);
            let found = parsed_cache
                .iter()
                .find(|(_, (parsed, _))| parsed.items.iter().any(|i| i.name == leaf));
            match found {
                Some((file_path, _)) => {
                    eprintln!(
                        "cargo:warning=[spec] `{}` defined in {} (hint was {})",
                        decl, file_path, hint
                    );
                }
                None => still_unresolved.push((decl, hint)),
            }
        }
        still_unresolved
    } else {
        Vec::new()
    };
    for (decl, hint) in &unresolved {
        eprintln!(
            "cargo:error=[spec] Declared struct `{}` not found in any registered \
             source file (looked near {}). Declare it with a path that matches its \
             actual definition location.",
            decl, hint
        );
    }
    if !unresolved.is_empty() {
        std::process::exit(1);
    }

    // Mark all Phase 1 files as emittable, EXCEPT structural parents that were
    // registered solely to support alias discovery (their module paths end in
    // "/mod"). Emitting them would duplicate the definitions of their child
    // modules under the parent's name (e.g., core::marker duplicating
    // core::variance).
    let mut emitted_files: HashSet<String> = HashSet::new();
    for file_path in parsed_cache.keys() {
        if file_path.ends_with("/mod") || file_path == "mod" {
            continue;
        }
        resolver.mark_emittable(file_path);
        emitted_files.insert(file_path.clone());
    }

    // ── Phase 1b: Register structural parents ──────────────────────────────
    // Structural parents of each declared struct's defining file are already
    // registered during discovery (see register_parents_of). Any remaining
    // ancestors discovered via import-driven expansion are picked up here.
    let phase1b_files: Vec<String> = parsed_cache.keys().cloned().collect();
    for target in &spec.targets {
        let lib_src = rust_src.join(&target.lib_name).join("src");

        for file_path in &phase1b_files {
            register_parents_of(
                file_path,
                &target.lib_name,
                &lib_src,
                &cfg,
                &mut resolver,
                &mut parsed_cache,
                &mut processed_parents,
            );
        }
    }

    // ── Phase 1c: Discover modules referenced by use statements ──────────────
    // After registering canonical files and their parents, walk through all
    // registered sources and discover any modules referenced by their use
    // statements (e.g., sys/pal/unix/futex.rs references crate::sys::fd,
    // which means std/sys/fd/mod.rs needs to be registered too).
    //
    // Constraint: only discover a file if its parent directory already has
    // at least one registered file. This prevents unbounded expansion into
    // the entire stdlib while still catching sibling modules that were missed
    // because they weren't declared via `mod X;` in our canonical tree.
    //
    // IMPORTANT: Files discovered here are registered with the resolver so
    // their types can be resolved during import resolution, but they are
    // NOT marked as emittable. They exist solely to provide type definitions
    // that other files depend on. Emitting them would pull in deep stdlib
    // internals that reference types we don't mirror.
    let mut import_discovered: HashSet<String> = HashSet::new();
    for target in &spec.targets {
        let lib_src = rust_src.join(&target.lib_name).join("src");

        loop {
            let mut newly_found: Vec<String> = Vec::new();
            // Collect set of known parent directories from parsed_cache.
            let known_dirs: HashSet<&str> = parsed_cache
                .keys()
                .filter_map(|p| p.rsplit_once('/'))
                .map(|(dir, _)| dir)
                .collect();

            for (parsed, _) in parsed_cache.values() {
                for stmt in &parsed.use_statements {
                    // Look at all use statements (both glob and single) that
                    // have path segments pointing to modules within our tree.
                    let (segs, is_glob) = match &stmt.kind {
                        rustyfill_sys_bindings::resolver::UseKind::Glob(pl) => {
                            (pl.segments.clone(), true)
                        }
                        rustyfill_sys_bindings::resolver::UseKind::Single(pl, _) => {
                            (pl.segments.clone(), false)
                        }
                    };
                    if segs.is_empty() {
                        continue;
                    }
                    // Build module path from segments, skipping leading
                    // `self`, `super`, `crate` anchors.
                    let mut mod_parts: Vec<String> = Vec::new();
                    for seg in &segs {
                        match seg {
                            rustyfill_sys_bindings::resolver::PathSegment::Super => continue,
                            rustyfill_sys_bindings::resolver::PathSegment::Crate => continue,
                            rustyfill_sys_bindings::resolver::PathSegment::Self_ => continue,
                            rustyfill_sys_bindings::resolver::PathSegment::Named(name) => {
                                mod_parts.push(name.clone());
                            }
                        }
                    }
                    if mod_parts.is_empty() {
                        continue;
                    }
                    // For single imports (not globs), the last segment is likely
                    // an item name (struct/type/function), so strip it to get the
                    // containing module path. E.g., `crate::sys::fd::FileDesc` →
                    // we want `sys/fd`, not `sys/fd/FileDesc`.
                    let module_candidates = if is_glob {
                        vec![mod_parts.clone()]
                    } else if mod_parts.len() > 1 {
                        let mut without_last = mod_parts.clone();
                        without_last.pop();
                        vec![without_last, mod_parts.clone()]
                    } else {
                        vec![mod_parts.clone()]
                    };

                    for parts in module_candidates {
                        let resolved = parts.join("/");
                        // Skip cross-library refs (core/alloc/std at top level).
                        if matches!(
                            resolved.split('/').next().unwrap_or(""),
                            "core" | "alloc" | "std"
                        ) {
                            continue;
                        }
                        // Only discover if the parent directory is already known.
                        let parent_dir = match parts.last() {
                            Some(_) if parts.len() > 1 => {
                                let mut pd = parts.clone();
                                pd.pop();
                                Some(pd.join("/"))
                            }
                            _ => None,
                        };
                        if let Some(ref pd) = parent_dir
                            && !known_dirs.contains(pd.as_str())
                        {
                            continue;
                        }
                        // Try as both .rs and /mod.rs.
                        for candidate in
                            &[format!("{}/mod.rs", resolved), format!("{}.rs", resolved)]
                        {
                            if !import_discovered.insert(candidate.clone()) {
                                continue;
                            }
                            if lib_src.join(candidate).exists() {
                                newly_found.push(candidate.clone());
                            }
                        }
                    }
                }
            }

            if newly_found.is_empty() {
                break;
            }

            for fp in &newly_found {
                let source_path = lib_src.join(fp);
                if !source_path.exists() {
                    continue;
                }
                if let Ok(source_text) = fs::read_to_string(&source_path) {
                    let parsed = parse_source_with_cfg(&source_text, &cfg);
                    resolver.register_source(fp, parsed.clone());
                }
            }
        }
    }

    // ── Phase 1d: Build the type registry ───────────────────────────────────
    // Index every named type in every registered file (visibility + export
    // status) and mark spec-declared structs. The registry drives both the
    // field-type publicity check and reference rewriting at emission time.
    let mut registry = TypeRegistry::empty();
    for target in &spec.targets {
        for (file_path, (parsed, lib_name)) in &parsed_cache {
            if lib_name != &target.lib_name {
                continue;
            }
            let module_path = resolver.file_to_module_path(file_path);
            let exported_names = public_reexport_names(parsed, &module_path);
            // Canonical paths are always lib-prefixed (`lib::module::Leaf`) so that
            // the library name is recoverable as the first segment when routing
            // references to the original builtin crate.
            for item in &parsed.items {
                let canonical = if module_path.is_empty() {
                    format!("{}::{}", target.lib_name, item.name)
                } else {
                    format!(
                        "{}::{}::{}",
                        target.lib_name,
                        module_path.replace('/', "::"),
                        item.name
                    )
                };
                let is_exported = exported_names.contains(&item.name);
                registry.register(
                    &canonical,
                    item.visibility,
                    is_exported,
                    file_path,
                );
                // Record type-alias RHS so declared aliases can be mirrored.
                if let Some(rhs) = &item.alias_rhs {
                    registry.set_alias_rhs(&canonical, rhs.clone());
                }
            }
            for (mod_name, mod_items) in &parsed.inline_modules {
                let inline_module = if module_path.is_empty() {
                    mod_name.clone()
                } else {
                    format!("{}/{}", module_path, mod_name)
                };
                let inline_canonical_base = inline_module.replace('/', "::");
                for item in mod_items {
                    let canonical =
                        format!("{}::{}::{}", target.lib_name, inline_canonical_base, item.name);
                    let is_exported = exported_names.contains(&item.name);
                    registry.register(
                        &canonical,
                        item.visibility,
                        is_exported,
                        file_path,
                    );
                    if let Some(rhs) = &item.alias_rhs {
                        registry.set_alias_rhs(&canonical, rhs.clone());
                    }
                }
            }
        }
        // Mark declared structs (paths are relative to the library root). The
        // def_file is stored as an absolute path so that
        // `check_declared_struct_fields` can read it regardless of the build
        // script's working directory.
        let lib_src = rust_src.join(&target.lib_name).join("src");
        for decl in &target.declared_structs {
            let canonical = format!("{}::{}", target.lib_name, decl);
            let def_file_rel = parsed_cache
                .iter()
                .find(|(_, (parsed, ln))| {
                    ln == &target.lib_name
                        && parsed.items.iter().any(|i| i.name == decl.rsplit("::").next().unwrap_or(""))
                })
                .map(|(fp, _)| fp.clone())
                .unwrap_or_else(|| decl.replace("::", "/") + ".rs");
            let def_file_abs = lib_src.join(&def_file_rel).to_string_lossy().to_string();
            registry.insert_declared(&canonical, &def_file_abs);
        }
    }

    // Hand the resolver the same declared-path set the emitter uses to filter
    // output. This keeps use-statement generation in sync with emission so that
    // modules whose items were all filtered out (e.g., set-side entry types that
    // route to map-side mirrors) don't leave behind dangling re-exports.
    resolver.set_declared_paths(registry.declared_paths().cloned());

    // Field-type publicity check: every field of a declared struct must refer
    // to either a declared type (mirrored) or a public type (original).
    // Private undeclared types are hard errors.
    let field_errors = check_declared_struct_fields(&registry);
    for err in &field_errors {
        eprintln!("cargo:error={}", err);
    }
    if !field_errors.is_empty() {
        eprintln!(
            "cargo:error=Field publicity check failed with {} error(s).",
            field_errors.len()
        );
        std::process::exit(1);
    }

    // ── Phase 2: EMIT — Now that all modules are registered, emit with full resolution ──
    let mut all_files: Vec<(String, String)> = Vec::new();
    let mut emitted_canonicals: HashSet<String> = HashSet::new();
    let mut emitted_paths: Vec<PathBuf> = Vec::new();

    for (file_path, (parsed, lib_name)) in &parsed_cache {
        // Structural parents are registered for alias discovery only; they
        // must not be emitted as standalone files.
        if file_path.ends_with("/mod") || file_path == "mod" {
            continue;
        }
        let depth = compute_module_depth(file_path);
        let extra_uses = resolver.emit_use_statements_for_file(file_path, &ignored_name_refs);
        let siblings = get_sibling_modules(file_path, &all_files);
        let emit_path = out_path.join(file_path);

        let target_ignored_structs = ignored_structs_by_lib
            .get(lib_name)
            .cloned()
            .unwrap_or_default();

        let target_extra_derives = extra_derives_by_lib
            .get(lib_name)
            .cloned()
            .unwrap_or_default();

        let has_content = emit_binding_file(
            &emit_path,
            &parsed.items,
            &rustyfill_sys_bindings::emitter::EmitConfig {
                lib_name,
                file_module_depth: depth,
                extra_uses: &extra_uses,
                sibling_modules: &siblings,
                path_replacements: &replacement_entries_slice,
                ignored_structs: &target_ignored_structs,
                relative_file_path: file_path,
                type_registry: &registry,
                extra_derives: &target_extra_derives,
            },
        );

        if has_content {
            validator.check_emit(&emit_path);
            emitted_paths.push(emit_path);
            emitted_canonicals.insert(file_path.clone());
            all_files.push((file_path.clone(), lib_name.clone()));
        }

        // Also emit inline modules
        for (mod_name, mod_items) in &parsed.inline_modules {
            let inline_dir = if file_path.ends_with("/mod.rs") {
                file_path.strip_suffix("/mod.rs").unwrap_or("")
            } else {
                file_path.strip_suffix(".rs").unwrap_or(file_path.as_str())
            };

            let inline_rel_path = if inline_dir.is_empty() {
                format!("{}/mod.rs", mod_name)
            } else {
                format!("{}/{}/mod.rs", inline_dir, mod_name)
            };

            let inline_emit_path = out_path.join(&inline_rel_path);
            let inline_depth = compute_module_depth(&inline_rel_path);
            let inline_extra_uses =
                resolver.emit_use_statements_for_file(&inline_rel_path, &ignored_name_refs);
            let inline_siblings = get_sibling_modules(&inline_rel_path, &all_files);
            let inline_has_content = emit_binding_file(
                &inline_emit_path,
                mod_items,
                &rustyfill_sys_bindings::emitter::EmitConfig {
                    lib_name,
                    file_module_depth: inline_depth,
                    extra_uses: &inline_extra_uses,
                    sibling_modules: &inline_siblings,
                    path_replacements: &replacement_entries_slice,
                    ignored_structs: &target_ignored_structs,
                    relative_file_path: &inline_rel_path,
                    type_registry: &registry,
                    extra_derives: &target_extra_derives,
                },
            );

            if inline_has_content {
                validator.check_emit(&inline_emit_path);
                emitted_paths.push(inline_emit_path);
                emitted_canonicals.insert(inline_rel_path.clone());
                all_files.push((inline_rel_path.clone(), lib_name.clone()));
            }
        }
    }

    // ── Phase 3: Discover and emit re-export aliases ────────────────────────
    // For every declared struct, walk from its defining file up through the
    // structural parents and discover `pub use` re-exports along the way. The
    // emitted alias files make both the canonical path and any module-level
    // aliases resolve to the same definition.
    let mut discovered_aliases = HashSet::new();
    for target in &spec.targets {
        for decl in &target.declared_structs {
            let leaf = decl.rsplit("::").next().unwrap_or("");
            let def_file = parsed_cache
                .iter()
                .find(|(_, (parsed, ln))| {
                    ln == &target.lib_name
                        && (parsed.items.iter().any(|i| i.name == leaf)
                            || parsed
                                .inline_modules
                                .iter()
                                .any(|(_, items)| items.iter().any(|i| i.name == leaf)))
                })
                .map(|(fp, _)| fp.clone());
            let Some(def_file) = def_file else { continue };

            let parents = resolver.get_parent_module_paths(&def_file);
            let all_related: Vec<String> = std::iter::once(def_file).chain(parents).collect();

            for related_file in all_related {
                let aliases = resolver.discover_reexport_aliases(&related_file);
                for (alias_module, canonical_module) in aliases {
                    let new_files = emit_glob_reexport_aliases(
                        &mut resolver,
                        &alias_module,
                        &canonical_module,
                        &target.lib_name,
                        out_path,
                        &mut discovered_aliases,
                        &emitted_canonicals,
                    );
                    all_files.extend(new_files);
                }
            }
        }
    }

    // ── Phase 4: Emit hierarchical manifest ─────────────────────────────────
    emit_hierarchical_manifest(out_path, &all_files);

    // ── Post-flight: validate everything ────────────────────────────────────
    validator.check_manifest(out_path, &all_files);
    validator.check_aliases(&mut resolver, &discovered_aliases);
    validator.finish().or_fatal();

    println!("cargo:rerun-if-changed=../rustyfill-sys-bindings/src/spec.rs");
    println!("cargo:rerun-if-env-changed=RUSTUP_TOOLCHAIN");
}

// ── Declaration location helpers ────────────────────────────────────────────

enum LocatedStruct {
    /// The struct is defined in this file (relative to the library src root).
    Found(String),
    /// No file on disk matched the declaration's path prefix.
    NotDefinedOnDisk(String),
    /// The declaration itself is malformed.
    BadPath(String),
}

/// Locate the defining file for a declared struct path like
/// `"collections::btree::map::BTreeMap"` under `<lib_src>`.
///
/// Tries progressively shorter prefixes of the path as candidate module
/// directories/files, keeping the longest one whose items actually include
/// the leaf name. This handles both `X.rs` / `X/mod.rs` layouts and inline
/// modules (where the definition sits in an ancestor file).
fn locate_declared_struct(decl: &str, lib_src: &Path, cfg: &CfgContext) -> LocatedStruct {
    let parts: Vec<&str> = decl.split("::").collect();
    if parts.is_empty() || parts.iter().any(|p| p.is_empty()) {
        return LocatedStruct::BadPath(format!(
            "Invalid struct path `{}` — expected `path::to::Struct` syntax",
            decl
        ));
    }
    let leaf = *parts.last().unwrap();

    // Try the full path first, then peel segments off the end.
    for cut in (1..=parts.len()).rev() {
        let prefix: Vec<&str> = parts[..cut].to_vec();
        let rel_prefix = prefix.join("/");
        let candidates = [
            format!("{rel_prefix}.rs"),
            format!("{rel_prefix}/mod.rs"),
        ];
        for cand in &candidates {
            let full = lib_src.join(cand);
            if !full.exists() {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&full) else {
                continue;
            };
            let parsed = parse_source_with_cfg(&text, cfg);
            if parsed.items.iter().any(|i| i.name == leaf) {
                return LocatedStruct::Found(cand.clone());
            }
            // Inline modules count too.
            if parsed.inline_modules.iter().any(|(_, items)| {
                items.iter().any(|i| i.name == leaf)
            }) {
                return LocatedStruct::Found(cand.clone());
            }
        }
    }

    let hint = parts[..parts.len()].join("/");
    LocatedStruct::NotDefinedOnDisk(hint + ".rs")
}

/// Register the structural parent modules of a file with the resolver so that
/// re-export alias discovery can walk up the tree. Mirrors Phase 1b logic for
/// a single file.
fn register_parents_of(
    file_path: &str,
    _lib_name: &str,
    lib_src: &Path,
    cfg: &CfgContext,
    resolver: &mut ModuleResolver,
    cache: &mut HashMap<String, (rustyfill_sys_bindings::parser::ParsedSource, String)>,
    processed_parents: &mut HashSet<String>,
) {
    let parents = resolver.get_parent_module_paths(file_path);
    for parent_mod in parents {
        if !processed_parents.insert(parent_mod.clone()) {
            continue;
        }
        let parent_path = lib_src.join(&parent_mod);
        if !parent_path.exists() {
            continue;
        }
        if let Ok(parent_text) = std::fs::read_to_string(&parent_path) {
            let parsed = parse_source_with_cfg(&parent_text, cfg);
            resolver.register_source(&parent_mod, parsed.clone());
            if !cache.contains_key(&parent_mod) {
                cache.insert(parent_mod.clone(), (parsed, _lib_name.to_string()));
                resolver.mark_emittable(&parent_mod);
            }
        }
    }
}

/// Compute the set of item names that are publicly re-exported from a module:
/// items defined directly with `pub` visibility plus everything pulled in by
/// `pub use` statements (single imports and globs resolved against known
/// sibling names heuristically).
fn public_reexport_names(parsed: &rustyfill_sys_bindings::parser::ParsedSource, _module_path: &str) -> HashSet<String> {
    let mut names: HashSet<String> = HashSet::new();
    for item in &parsed.items {
        if item.visibility.is_public() {
            names.insert(item.name.clone());
        }
    }
    for stmt in &parsed.use_statements {
        if !matches!(stmt.visibility, rustyfill_sys_bindings::resolver::Visibility::Public) {
            continue;
        }
        match &stmt.kind {
            UseKind::Single(plist, alias) => {
                let name = alias.clone().or_else(|| {
                    plist.segments.iter().rev().find_map(|s| {
                        if let rustyfill_sys_bindings::resolver::PathSegment::Named(n) = s {
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
            UseKind::Glob(_) => {
                // Globs pull in every public item of the target module; we
                // approximate conservatively by leaving it to the per-item
                // visibility check above. Nothing to add here.
            }
        }
    }
    names
}

/// Compute how many module levels deep a file is under its library root.
/// e.g. "collections/btree/map.rs" -> 3 (collections / btree / map)
///      "sys/pal/mod.rs" -> 1 (sys/pal defines module sys::pal, depth = 2 segments but mod.rs means it IS that module)
fn compute_module_depth(rel_path: &str) -> usize {
    let stem = rel_path.strip_suffix(".rs").unwrap_or(rel_path);
    let module_path = stem.strip_suffix("/mod").unwrap_or(stem);
    module_path.split('/').filter(|s| !s.is_empty()).count()
}

/// Get all sibling module names in the same parent directory.
/// For "collections/btree/node.rs", returns ["borrow", "map", "marker", ...].
fn get_sibling_modules(rel_path: &str, all_files: &[(String, String)]) -> Vec<String> {
    let my_stem = rel_path.strip_suffix(".rs").unwrap_or(rel_path);
    let my_module = my_stem.strip_suffix("/mod").unwrap_or(my_stem);
    let my_parent = my_module.rsplit_once('/').map(|(p, _)| p).unwrap_or("");

    let mut siblings = HashSet::new();
    for (fp, _) in all_files {
        let stem = fp.strip_suffix(".rs").unwrap_or(fp.as_str());
        let mod_path = stem.strip_suffix("/mod").unwrap_or(stem);
        let parent = mod_path.rsplit_once('/').map(|(p, _)| p).unwrap_or("");
        if parent == my_parent {
            let name = mod_path
                .rsplit_once('/')
                .map(|(_, n)| n)
                .unwrap_or(mod_path);
            if name
                != my_module
                    .rsplit_once('/')
                    .map(|(_, n)| n)
                    .unwrap_or(my_module)
            {
                siblings.insert(name.to_string());
            }
        }
    }
    let mut result: Vec<String> = siblings.into_iter().collect();
    result.sort();
    result
}

/// Parameters for [`discover_and_register`].
struct DiscoverParams<'a> {
    source_rel_path: &'a str,
    lib_name: &'a str,
    lib_src: &'a Path,
    cfg: &'a CfgContext,
    resolver: &'a mut ModuleResolver,
    validator: &'a mut ValidationBuilder,
    visited: &'a mut HashSet<String>,
    cache: &'a mut HashMap<String, (rustyfill_sys_bindings::parser::ParsedSource, String)>,
}

/// Discover phase: parse a file, register it with the resolver, validate,
/// and recursively discover all children. Does NOT emit any files.
fn discover_and_register(params: DiscoverParams) {
    let DiscoverParams {
        source_rel_path,
        lib_name,
        lib_src,
        cfg,
        resolver,
        validator,
        visited,
        cache,
    } = params;
    if !visited.insert(source_rel_path.to_string()) {
        return;
    }

    let source_path = lib_src.join(source_rel_path);
    if !source_path.exists() {
        eprintln!(
            "cargo:warning=Source file not found: {} (skipping)",
            source_path.display()
        );
        return;
    }

    let source_text = match fs::read_to_string(&source_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "cargo:warning=Failed to read {}: {} (skipping)",
                source_path.display(),
                e
            );
            return;
        }
    };

    let parsed = parse_source_with_cfg(&source_text, cfg);

    // Validate parse result
    validator.check_parse(source_rel_path, &parsed, &source_text);
    validator.check_items(source_rel_path, &parsed.items);

    // Register with resolver
    let parsed_clone = parsed.clone();
    resolver.register_source(source_rel_path, parsed_clone);

    // Cache for emission phase
    cache.insert(
        source_rel_path.to_string(),
        (parsed.clone(), lib_name.to_string()),
    );

    // Register inline modules too
    for (mod_name, mod_items) in &parsed.inline_modules {
        let inline_dir = if source_rel_path.ends_with("/mod.rs") {
            source_rel_path.strip_suffix("/mod.rs").unwrap_or("")
        } else {
            source_rel_path
                .strip_suffix(".rs")
                .unwrap_or(source_rel_path)
        };

        let inline_rel_path = if inline_dir.is_empty() {
            format!("{}/mod.rs", mod_name)
        } else {
            format!("{}/{}/mod.rs", inline_dir, mod_name)
        };

        let inline_parsed = rustyfill_sys_bindings::parser::ParsedSource {
            items: mod_items.clone(),
            use_statements: Vec::new(),
            mod_declarations: Vec::new(),
            inline_modules: Vec::new(),
            inline_module_uses: std::collections::HashMap::new(),
        };
        resolver.register_source(&inline_rel_path, inline_parsed);

        cache.insert(
            inline_rel_path.clone(),
            (
                (rustyfill_sys_bindings::parser::ParsedSource {
                    items: mod_items.clone(),
                    use_statements: Vec::new(),
                    mod_declarations: Vec::new(),
                    inline_modules: Vec::new(),
                    inline_module_uses: std::collections::HashMap::new(),
                }),
                lib_name.to_string(),
            ),
        );
    }

    // Discover children via mod declarations.
    let module_path = resolver.file_to_module_path(source_rel_path);
    let dir = if source_rel_path.ends_with("/mod.rs") {
        source_rel_path.strip_suffix("/mod.rs").unwrap_or("")
    } else {
        source_rel_path
            .strip_suffix(".rs")
            .unwrap_or(source_rel_path)
    };

    let children = resolver.discover_children(&module_path, visited, cfg, &|_parent, mod_name| {
        let child_mod_rs = if dir.is_empty() {
            format!("{}/mod.rs", mod_name)
        } else {
            format!("{}/{}/mod.rs", dir, mod_name)
        };

        let child_leaf_rs = if dir.is_empty() {
            format!("{}.rs", mod_name)
        } else {
            format!("{}/{}.rs", dir, mod_name)
        };

        if lib_src.join(&child_mod_rs).exists() {
            Some(child_mod_rs)
        } else if lib_src.join(&child_leaf_rs).exists() {
            Some(child_leaf_rs)
        } else {
            None
        }
    });

    for child in children {
        discover_and_register(DiscoverParams {
            source_rel_path: &child,
            lib_name,
            lib_src,
            cfg,
            resolver,
            validator,
            visited,
            cache,
        });
    }
}

/// Find the root of the Rust standard library source tree.
fn find_rust_source_root() -> PathBuf {
    if let Ok(src) = env::var("RUST_SRC_PATH") {
        let p = PathBuf::from(src);
        if p.exists() {
            return p;
        }
    }

    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    if let Ok(output) = std::process::Command::new(&rustc)
        .arg("--print=sysroot")
        .output()
    {
        let sysroot = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let candidate = PathBuf::from(&sysroot).join("lib/rustlib/src/rust/library");

        if candidate.exists() {
            return candidate;
        }
    }

    if let Ok(home) = env::var("HOME") {
        let candidate2 = PathBuf::from(&home).join(
            ".rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library",
        );

        if candidate2.exists() {
            return candidate2;
        }
    }

    panic!(
        "Could not locate Rust standard library source.\n\
         Install the rust-src component: `rustup component add rust-src`\n\
         Or set RUST_SRC_PATH to the library source root."
    );
}

/// Abort the build if `-Zrandomize-layout` is active in the current
/// compilation environment. Layout randomization shuffles field offsets and
/// type alignments, which completely breaks the deterministic layout
/// assumptions that polyfilled mirror structs rely on (identical field
/// layout with the real stdlib types).
fn reject_randomize_layout() {
    // CARGO_ENCODED_RUSTFLAGS contains all effective flags (from RUSTFLAGS,
    // .cargo/config.toml [target.*.rustflags], etc.) null-separated.
    if let Some(encoded) = env::var_os("CARGO_ENCODED_RUSTFLAGS") {
        for flag in encoded.to_string_lossy().split('\0') {
            if flag == "-Zrandomize-layout" || flag == "-Z randomize-layout" {
                panic!(
                    "rustyfill-sys: -Zrandomize-layout is incompatible with polyfilled \
                     bindings.\nThe mirrored data structures require deterministic field \
                     layout matching the standard library.\n\
                     Remove -Zrandomize-layout from your RUSTFLAGS or cargo config."
                );
            }
        }
    }

    // Also check RUSTFLAGS directly (covers cases where CARGO_ENCODED_RUSTFLAGS
    // might not be set, e.g., manual cargo invocations with unusual profiles).
    if let Some(rustflags) = env::var_os("RUSTFLAGS") {
        for flag in rustflags.to_string_lossy().split_whitespace() {
            if flag == "-Zrandomize-layout" || flag == "-Z" {
                // If bare -Z is present, the next token is the flag name.
                // We can't easily pair them here, so just warn on the explicit form.
                if flag == "-Zrandomize-layout" {
                    panic!(
                        "rustyfill-sys: -Zrandomize-layout is incompatible with polyfilled \
                         bindings.\nThe mirrored data structures require deterministic field \
                         layout matching the standard library.\n\
                         Remove -Zrandomize-layout from your RUSTFLAGS or cargo config."
                    );
                }
            }
        }
    }
}
