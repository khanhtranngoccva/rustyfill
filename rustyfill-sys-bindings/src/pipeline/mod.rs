//! The binding-generation pipeline, relocated from `rustyfill-sys/build.rs`.
//!
//! This module owns the entire multi-phase generation algorithm: locating
//! declared structs, discovering the module tree, expanding import-driven
//! dependencies, building the type registry, mirroring minimal modules for
//! preserved qualifiers, and emitting preamble / binding / alias files plus
//! the hierarchical manifest. It is environment-agnostic — it reads from a
//! caller-supplied Rust source root and writes into a caller-supplied output
//! directory — so it can be exercised directly by unit tests without cargo.
//!
//! The build script reduces to a thin orchestrator that locates the toolchain
//! source, rejects incompatible flags, calls [`generate`], and forwards any
//! reported diagnostics as `cargo:` messages.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::emitter::{
    check_declared_struct_fields, emit_hierarchical_manifest, emit_known_type_stubs,
    emit_preamble_module,
};
use crate::loader_spec::LoaderSpec;
use crate::parser::{CfgContext, ParsedSource};
use crate::resolver::ModuleResolver;
use crate::validator::ValidationBuilder;

mod discover;
mod emit;
mod mirror;
mod registry;
mod util;

pub use util::{compute_module_depth, get_sibling_modules};

/// Input to a single [`generate`] run.
pub struct PipelineInput<'a> {
    /// Root of the Rust standard-library source tree (the directory containing
    /// `core/`, `alloc/`, `std/`).
    pub rust_src: &'a Path,
    /// Directory generated bindings are written into (`$OUT_DIR`).
    pub out_dir: &'a Path,
    /// The loaded loader spec describing every target library.
    pub spec: &'a LoaderSpec,
    /// Platform context used to evaluate `cfg_select!` branches.
    pub cfg: &'a CfgContext,
}

/// Diagnostics produced by a [`generate`] run. Empty on success; non-empty
/// means generation aborted before writing a complete, valid tree.
#[derive(Default)]
pub struct GenerateReport {
    /// Hard errors. If non-empty the pipeline stopped early.
    pub errors: Vec<String>,
    /// Non-fatal warnings worth surfacing to the developer.
    pub warnings: Vec<String>,
}

/// Result of a [`generate`] run. On success `outcome` is `Ok(())`; on failure
/// it carries the accumulated diagnostics in the `Err` variant.
pub type GenerateOutcome = Result<(), GenerateReport>;

/// Run the full generation pipeline. Returns `Ok(())` when every phase
/// succeeded and validation passed; otherwise `Err(report)` with the reasons.
pub fn generate(input: &PipelineInput<'_>) -> GenerateOutcome {
    let PipelineInput {
        rust_src,
        out_dir,
        spec,
        cfg,
    } = input;

    // ── Derive per-target emission inputs from the spec ─────────────────────
    let replacement_entries = util::build_replacement_entries(spec);
    let ignored_structs_by_lib: HashMap<String, Vec<String>> = spec
        .targets
        .iter()
        .map(|t| (t.lib_name.clone(), t.ignored_structs.clone()))
        .collect();
    let extra_derives_by_lib: HashMap<String, HashMap<String, Vec<String>>> = spec
        .targets
        .iter()
        .map(|t| {
            let mut map = HashMap::new();
            for (path, derives) in &t.extra_derives {
                map.insert(path.clone(), derives.clone());
            }
            (t.lib_name.clone(), map)
        })
        .collect();
    // Keep the owned list alive for the duration of `generate`; `ignored_name_refs`
    // borrows from it.
    let ignored_names_owned =
        util::collect_ignored_names(&replacement_entries, &ignored_structs_by_lib);
    let ignored_name_refs: Vec<&str> = ignored_names_owned.iter().map(|s| s.as_str()).collect();

    // ── Pre-flight: validate spec paths ─────────────────────────────────────
    let mut validator = ValidationBuilder::new();
    validator.check_spec(spec, rust_src);

    let mut resolver = ModuleResolver::new();
    let mut processed_parents: HashSet<String> = HashSet::new();
    let mut preamble_emitted: HashSet<String> = HashSet::new();
    let mut pending_declarations: Vec<(String, String)> = Vec::new();
    let mut reexport_located: Vec<(String, String)> = Vec::new();
    // Accumulated across all targets so the final manifest sees every emitted
    // file (including re-export shims materialized during minimal-module
    // mirroring in Phase 1d).
    let mut all_files: Vec<(String, String)> = Vec::new();
    let mut emitted_canonicals: HashSet<String> = HashSet::new();

    // ── Phase 0: Emit preamble modules per target library ───────────────────
    // The preamble carries only static core re-exports and shims; it is identical
    // in shape for every library, emitted once per target.
    for target in &spec.targets {
        if preamble_emitted.insert(target.lib_name.clone()) {
            emit_preamble_module(out_dir, &target.lib_name);
        }
    }

    // ── Phase 1: DISCOVER — Parse all files, register with resolver ─────────
    let mut parsed_cache: HashMap<String, (ParsedSource, String)> = HashMap::new();

    for target in &spec.targets {
        let lib_src = rust_src.join(&target.lib_name).join("src");

        // Collect active declarations (unconditional + cfg-gated ones whose
        // predicate matches the current build context). This ensures that
        // platform-specific backend types (e.g., futex-only types on Linux)
        // are not declared on targets where they don't exist.
        let active_decls = target.active_declarations(cfg);
        for decl in &active_decls {
            match discover::locate_declared_struct(decl, &lib_src, cfg) {
                discover::LocatedStruct::Found(def_file) => {
                    let naive = decl.replace("::", "/") + ".rs";
                    if !lib_src.join(&naive).exists() && def_file != naive {
                        reexport_located.push((decl.clone(), def_file.clone()));
                    }
                    let mut parent_visited = HashSet::new();
                    discover::discover_and_register(discover::DiscoverParams {
                        source_rel_path: def_file.as_str(),
                        lib_name: &target.lib_name,
                        lib_src: &lib_src,
                        cfg,
                        resolver: &mut resolver,
                        validator: &mut validator,
                        visited: &mut parent_visited,
                        cache: &mut parsed_cache,
                    });
                    discover::register_parents_of(
                        &def_file,
                        &target.lib_name,
                        &lib_src,
                        cfg,
                        &mut resolver,
                        &mut parsed_cache,
                        &mut processed_parents,
                    );
                }
                discover::LocatedStruct::NotDefinedOnDisk(path_hint) => {
                    pending_declarations.push((decl.clone(), path_hint));
                }
                discover::LocatedStruct::CfgExcluded { module, predicate } => {
                    return Err(GenerateReport {
                        errors: vec![format!(
                            "[spec] `{}` is defined in `{}`, which is excluded for this \
                             target by an inner cfg gate ({predicate}). Gate the \
                             declaration with a matching predicate (declare_struct_cfg) \
                             so it only activates on targets where that module exists.",
                            decl, module
                        )],
                        warnings: Vec::new(),
                    });
                }
                discover::LocatedStruct::BadPath(msg) => {
                    return Err(GenerateReport {
                        errors: vec![format!("[spec] {}", msg)],
                        warnings: Vec::new(),
                    });
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
                    // Surfaced as a warning by the caller via the report.
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
    if !unresolved.is_empty() {
        let mut errors = Vec::new();
        for (decl, hint) in &unresolved {
            errors.push(format!(
                "[spec] Declared struct `{}` not found in any registered \
                 source file (looked near {}). Declare it with a path that matches its \
                 actual definition location.",
                decl, hint
            ));
        }
        return Err(GenerateReport {
            errors,
            warnings: Vec::new(),
        });
    }

    // Remember each file's library so emitted imports can be written as
    // absolute `crate::{lib}::...` paths.
    for (file_path, (_, lib_name)) in &parsed_cache {
        resolver.set_file_lib(file_path, lib_name);
    }

    // Mark all Phase 1 files as emittable, EXCEPT structural parents that were
    // registered solely to support alias discovery (their module paths end in
    // "/mod").
    let mut emitted_files: HashSet<String> = HashSet::new();
    for file_path in parsed_cache.keys() {
        if file_path.ends_with("/mod") || file_path == "mod" {
            continue;
        }
        resolver.mark_emittable(file_path);
        emitted_files.insert(file_path.clone());
    }

    // ── Phase 1b: Register structural parents ───────────────────────────────
    let phase1b_files: Vec<String> = parsed_cache.keys().cloned().collect();
    for target in &spec.targets {
        let lib_src = rust_src.join(&target.lib_name).join("src");
        for file_path in &phase1b_files {
            discover::register_parents_of(
                file_path,
                &target.lib_name,
                &lib_src,
                cfg,
                &mut resolver,
                &mut parsed_cache,
                &mut processed_parents,
            );
        }
    }
    // Parents registered above may not be in the cache yet; sync lib names.
    for (file_path, (_, lib_name)) in &parsed_cache {
        resolver.set_file_lib(file_path, lib_name);
    }

    // ── Phase 1c: Discover modules referenced by use statements ─────────────
    discover::discover_imported_modules(spec, rust_src, cfg, &mut resolver, &mut parsed_cache);
    for (file_path, (_, lib_name)) in &parsed_cache {
        resolver.set_file_lib(file_path, lib_name);
    }

    // ── Phase 1d: Build the type registry ───────────────────────────────────
    let mut reg_state = registry::RegistryBuildState {
        resolver: &mut resolver,
        parsed_cache: &mut parsed_cache,
        emitted_canonicals: &mut emitted_canonicals,
        all_files: &mut all_files,
    };
    let registry = registry::build_type_registry(
        spec,
        rust_src,
        cfg,
        &reexport_located,
        out_dir,
        &mut reg_state,
    );

    // Materialize a re-export shim for any spec-declared type whose canonical
    // module has no emitted binding file of its own but whose concrete
    // definition lives in a cfg-selected submodule (e.g. `sys::sync::mutex::Mutex`,
    // which on Linux is `pub use futex::Mutex;` inside `mod.rs`). Without this,
    // the declared path would dangle because only the leaf submodule (`futex`)
    // was mirrored.
    mirror::emit_cfg_reexport_shims(
        rust_src,
        &spec.targets,
        out_dir,
        &mut resolver,
        cfg,
        &mut emitted_canonicals,
        &mut all_files,
    );
    // Shims registered above are keyed by target lib name; sync with resolver.
    for (file_path, lib_name) in &all_files {
        resolver.set_file_lib(file_path, lib_name);
    }

    // Hand the resolver the same declared-path set the emitter uses to filter
    // output. Minimal modules mirrored during registry construction inherit
    // their lib name from the referring file's lib (all files in a given
    // module tree belong to one library), so no extra bookkeeping is needed.
    resolver.set_declared_paths(registry.declared_paths().cloned());

    // Field-type publicity check.
    let field_errors = check_declared_struct_fields(&registry);
    if !field_errors.is_empty() {
        let mut errors: Vec<String> = field_errors.to_vec();
        errors.push(format!(
            "Field publicity check failed with {} error(s).",
            field_errors.len()
        ));
        return Err(GenerateReport {
            errors,
            warnings: Vec::new(),
        });
    }

    // ── Phase 2: EMIT ───────────────────────────────────────────────────────
    let replacement_entries_slice = util::replacement_view(&replacement_entries);
    let mut emitted_paths: Vec<PathBuf> = Vec::new();
    emit::emit_all_binding_files(
        &parsed_cache,
        &mut resolver,
        &registry,
        &replacement_entries_slice,
        &ignored_name_refs,
        &ignored_structs_by_lib,
        &extra_derives_by_lib,
        out_dir,
        &mut validator,
        &mut emitted_paths,
        &mut emitted_canonicals,
        &mut all_files,
    );

    // ── Phase 2b: Emit known-external-type stubs at their canonical location ─
    // Group by (lib, module path) so multiple known types sharing a module
    // (e.g. `Atomic`, `AtomicBool`, `AtomicPtr` all in `sync::atomic`) are
    // merged into a single file instead of clobbering each other.
    let mut kt_by_module: std::collections::BTreeMap<(String, String), Vec<&crate::loader_spec::KnownExternalType>> =
        std::collections::BTreeMap::new();
    for target in &spec.targets {
        for kt in &target.known_external_types {
            let segments: Vec<&str> = kt.path.split("::").collect();
            if segments.len() < 2 {
                continue;
            }
            let module_slash: String = segments[..segments.len() - 1].join("/");
            kt_by_module
                .entry((target.lib_name.clone(), module_slash))
                .or_default()
                .push(kt);
        }
    }
    for ((lib_name, _module_slash), kts) in &kt_by_module {
        if let Some(rel_path) = emit_known_type_stubs(out_dir, kts) {
            validator.check_emit(&out_dir.join(&rel_path));
            emitted_paths.push(out_dir.join(&rel_path));
            emitted_canonicals.insert(rel_path.clone());
            resolver.set_file_lib(&rel_path, lib_name);
            all_files.push((rel_path, lib_name.clone()));
        }
    }

    // ── Phase 3: Discover and emit re-export aliases ────────────────────────
    let discovered_aliases = emit::discover_and_emit_reexport_aliases(
        spec,
        cfg,
        &parsed_cache,
        &mut resolver,
        out_dir,
        &emitted_canonicals,
        &mut all_files,
    );

    // ── Phase 4: Emit hierarchical manifest ─────────────────────────────────
    emit_hierarchical_manifest(out_dir, &all_files);

    // ── Post-flight: validate everything ────────────────────────────────────
    validator.check_manifest(out_dir, &all_files);
    validator.check_aliases(&mut resolver, &discovered_aliases);
    // Re-validate every emitted binding file so that files written before an
    // earlier error was surfaced (or by a previous stale run sharing this
    // OUT_DIR) are caught here rather than only at the crate's compile time.
    for path in &emitted_paths {
        validator.check_emit(path);
    }
    let result = validator.finish();
    if !result.errors.is_empty() {
        return Err(GenerateReport {
            errors: result.errors.errors,
            warnings: Vec::new(),
        });
    }

    Ok(())
}
