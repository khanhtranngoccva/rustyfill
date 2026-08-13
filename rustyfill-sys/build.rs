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
    emit_binding_file, emit_glob_reexport_aliases, emit_hierarchical_manifest, emit_preamble_module,
};
use rustyfill_sys_bindings::get_loader_spec;
use rustyfill_sys_bindings::parser::{CfgContext, parse_source_with_cfg};
use rustyfill_sys_bindings::resolver::ModuleResolver;
use rustyfill_sys_bindings::validator::ValidationBuilder;

fn main() {
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
    let ignored_name_refs: Vec<&str> = replacement_entries_slice
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();

    // Build per-library ignored struct lists.
    let ignored_structs_by_lib: HashMap<String, Vec<String>> = spec
        .targets
        .iter()
        .map(|t| (t.lib_name.clone(), t.ignored_structs.clone()))
        .collect();

    // ── Pre-flight: validate spec paths ────────────────────────────────────
    let mut validator = ValidationBuilder::new();
    validator.check_spec(&spec, &rust_src);

    let mut resolver = ModuleResolver::new();
    let mut processed_parents: HashSet<String> = HashSet::new();
    let mut preamble_emitted: HashSet<String> = HashSet::new();

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

        for source_rel_path in &target.canonical_files {
            let mut child_visited = HashSet::new();
            discover_and_register(DiscoverParams {
                source_rel_path,
                lib_name: &target.lib_name,
                lib_src: &lib_src,
                cfg: &cfg,
                resolver: &mut resolver,
                validator: &mut validator,
                visited: &mut child_visited,
                cache: &mut parsed_cache,
            });
        }
    }

    // ── Phase 1b: Register structural parents ──────────────────────────────
    for target in &spec.targets {
        let lib_src = rust_src.join(&target.lib_name).join("src");

        for file_path in parsed_cache.keys() {
            let parents = resolver.get_parent_module_paths(file_path);
            for parent_mod in parents {
                if !processed_parents.insert(parent_mod.clone()) {
                    continue;
                }

                let parent_path = lib_src.join(&parent_mod);
                if !parent_path.exists() {
                    continue;
                }

                if let Ok(parent_text) = fs::read_to_string(&parent_path) {
                    let parsed = parse_source_with_cfg(&parent_text, &cfg);
                    resolver.register_source(&parent_mod, parsed);
                }
            }
        }
    }

    // ── Phase 2: EMIT — Now that all modules are registered, emit with full resolution ──
    let mut all_files: Vec<(String, String)> = Vec::new();
    let mut emitted_canonicals: HashSet<String> = HashSet::new();
    let mut emitted_paths: Vec<PathBuf> = Vec::new();

    for (file_path, (parsed, lib_name)) in &parsed_cache {
        let depth = compute_module_depth(file_path);
        let extra_uses = resolver.emit_use_statements_for_file(file_path, &ignored_name_refs);
        let siblings = get_sibling_modules(file_path, &all_files);
        let emit_path = out_path.join(file_path);

        let target_ignored_structs = ignored_structs_by_lib
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
    let mut discovered_aliases = HashSet::new();
    for target in &spec.targets {
        for source_rel_path in &target.canonical_files {
            let parents = resolver.get_parent_module_paths(source_rel_path);
            let all_related: Vec<String> = std::iter::once(source_rel_path.clone())
                .chain(parents)
                .collect();

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

    let children = resolver.discover_children(&module_path, visited, &|_parent, mod_name| {
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
