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
//!
//! ## Layout
//!
//! [`generate`] in this file is the sole orchestrator: it threads shared
//! mutable state through the phases and aborts on the first hard error. Each
//! phase lives in its own submodule and knows nothing about the others:
//!
//! - [`discover`] — Phase 1: locate declarations, parse and register sources,
//!   follow imports to a fixed point.
//! - [`registry`] — Phase 1d: populate the type registry and mirror minimal
//!   modules for preserved qualifiers.
//! - [`mirror`] — cfg-selected re-export shims (Strategy B materialization).
//! - [`emit`] — Phases 2/3: write binding files, known-type stubs, and
//!   re-export aliases; demote empty nodes.
//! - [`util`] — spec-derived input builders and path/module helpers.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::emitter::emit_preamble_module;
use crate::loader_spec::LoaderSpec;
use crate::parser::{CfgContext, ParsedSource};
use crate::resolver::ModuleResolver;
use crate::syntaxes::BindingModel;
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

    // ── Spec normalization ──────────────────────────────────────────────────
    // Canonical paths may be written with a leading `::` absolute-path marker
    // (e.g. `::core::alloc::Allocator`). Internally, per-library spec entries
    // are stored relative to the library root and the lib is tracked separately
    // on the target; normalize both spellings to the relative form once here so
    // every phase sees one representation.
    let normalized_spec = normalize_spec(spec);

    // ── Spec-derived emission inputs ────────────────────────────────────────
    let replacement_entries = util::build_replacement_entries(&normalized_spec);
    let ignored_structs_by_lib: HashMap<String, Vec<String>> = spec
        .targets
        .iter()
        .map(|t| (t.lib_name.clone(), t.ignored_structs.clone()))
        .collect();
    let extra_derives_by_lib: HashMap<String, HashMap<String, Vec<String>>> = spec
        .targets
        .iter()
        .map(|t| {
            let mut map: HashMap<String, Vec<String>> = HashMap::new();
            for (path, derives) in &t.extra_derives {
                // Accept the absolute-marker spelling as well; keys are matched
                // against lib-relative paths during emission.
                let key = path.strip_prefix("::").unwrap_or(path);
                map.entry(key.to_string()).or_default().extend(derives.iter().cloned());
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
    validator.check_spec(&normalized_spec, rust_src);

    let mut resolver = ModuleResolver::new();
    // The binding tree: the single source of truth for where every module and
    // item lives. Populated in lockstep with the legacy accumulators during the
    // migration; later phases read from it instead of re-parsing file strings.
    let mut model = BindingModel::new();
    // Accumulated across all targets so the final manifest sees every emitted
    // file (including re-export shims materialized during minimal-module
    // mirroring in Phase 1d).
    let mut all_files: Vec<(String, String)> = Vec::new();
    let mut emitted_canonicals: HashSet<String> = HashSet::new();

    // ── Phase 0: Emit preamble modules per target library ───────────────────
    // The preamble carries only static core re-exports and shims; it is identical
    // in shape for every library, emitted once per target.
    let mut preamble_emitted: HashSet<String> = HashSet::new();
    for target in &spec.targets {
        if preamble_emitted.insert(target.lib_name.clone()) {
            emit_preamble_module(out_dir, &target.lib_name);
        }
    }

    // ── Phase 1: DISCOVER — Parse all files, register with resolver ─────────
    let mut parsed_cache: HashMap<String, (ParsedSource, String)> = HashMap::new();
    let mut discovery = match discover::run_discovery(
        &normalized_spec,
        rust_src,
        cfg,
        &mut resolver,
        &mut validator,
        &mut parsed_cache,
        &mut model,
    ) {
        Ok(d) => d,
        Err(errors) => return Err(GenerateReport { errors, warnings: Vec::new() }),
    };

    // Remember each file's library so emitted imports can be written as
    // absolute `crate::{lib}::...` paths.
    sync_file_libs(&parsed_cache, &mut resolver);

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
    for target in &normalized_spec.targets {
        let lib_src = rust_src.join(&target.lib_name).join("src");
        for file_path in &phase1b_files {
            discover::register_parents_of(
                file_path,
                &target.lib_name,
                &lib_src,
                cfg,
                &mut resolver,
                &mut parsed_cache,
                &mut model,
                &mut discovery.processed_parents,
            );
        }
    }
    // Parents registered above may not be in the cache yet; sync lib names.
    sync_file_libs(&parsed_cache, &mut resolver);

    // ── Phase 1c: Discover modules referenced by use statements ─────────────
    discover::discover_imported_modules(
        &normalized_spec,
        rust_src,
        cfg,
        &mut resolver,
        &mut parsed_cache,
        &mut model,
    );
    sync_file_libs(&parsed_cache, &mut resolver);

    // ── Phase 1d: Build the type registry ───────────────────────────────────
    let mut reg_state = registry::RegistryBuildState {
        resolver: &mut resolver,
        parsed_cache: &mut parsed_cache,
        model: &mut model,
        emitted_canonicals: &mut emitted_canonicals,
        all_files: &mut all_files,
    };
    let registry = registry::build_type_registry(
        &normalized_spec,
        rust_src,
        cfg,
        &discovery.reexport_located,
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
        &normalized_spec.targets,
        out_dir,
        &mut resolver,
        &mut model,
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

    // Field-type publicity check. Declared items are enumerated from the tree;
    // reference resolution still consults the registry (retirement pending).
    let field_errors = crate::emitter::check_declared_struct_fields(&model, &registry);
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
        &mut model,
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
    let kt_groups = group_known_types_by_module(&normalized_spec);
    for ((lib_name, _module_slash), kts) in &kt_groups {
        if let Some(rel_path) = crate::emitter::emit_known_type_stubs(out_dir, kts) {
            validator.check_emit(&out_dir.join(&rel_path));
            emitted_paths.push(out_dir.join(&rel_path));
            emitted_canonicals.insert(rel_path.clone());
            resolver.set_file_lib(&rel_path, lib_name);
            all_files.push((rel_path.clone(), lib_name.clone()));
            // Known-type stub files are synthetic emittable leaf modules.
            model.register_synthetic(lib_name, &rel_path, crate::syntaxes::NodeStatus::Emittable);
            model.mark_file_emitted(&rel_path);
        }
    }

    // ── Phase 3: Discover and emit re-export aliases ────────────────────────
    let discovered_aliases = emit::discover_and_emit_reexport_aliases(
        &mut model,
        &normalized_spec,
        cfg,
        &parsed_cache,
        &mut resolver,
        out_dir,
        &emitted_canonicals,
        &mut all_files,
    );

    // ── Phase 3b: Demote empty emittable nodes ──────────────────────────────
    // Nodes whose items were all filtered by the declaration gate never
    // produced output; they must not appear in the manifest.
    model.demote_empty_emittable();

    // ── Phase 4: Emit hierarchical manifest from the binding tree ───────────
    crate::emitter::emit_hierarchical_manifest(out_dir, &model);

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

// ── Orchestrator-local helpers ────────────────────────────────────────────────

/// Strip the leading `::` absolute-path marker from every per-library spec
/// path (declarations, cfg-gated declarations, known external types). Spec
/// entries are stored relative to the library root with the lib tracked
/// separately on the target; accepting both spellings at this single boundary
/// keeps every downstream phase on one representation.
fn normalize_spec(spec: &LoaderSpec) -> LoaderSpec {
    let mut out = (*spec).clone();
    for target in &mut out.targets {
        for decl in &mut target.declared_structs {
            *decl = decl.strip_prefix("::").unwrap_or(decl).to_string();
        }
        for g in &mut target.cfg_gated_decls {
            g.path = g.path.strip_prefix("::").unwrap_or(&g.path).to_string();
        }
        for kt in &mut target.known_external_types {
            kt.path = kt.path.strip_prefix("::").unwrap_or(&kt.path).to_string();
        }
    }
    out
}

/// Record each cached file's owning library on the resolver so emitted imports
/// can be written as absolute `crate::{lib}::...` paths. Called after every
/// phase that registers new files.
fn sync_file_libs(
    parsed_cache: &HashMap<String, (ParsedSource, String)>,
    resolver: &mut ModuleResolver,
) {
    for (file_path, (_, lib_name)) in parsed_cache {
        resolver.set_file_lib(file_path, lib_name);
    }
}

/// Group spec-known external types by `(lib, module path)` so several known
/// types sharing a module merge into one stub file. Types whose path has fewer
/// than two segments have no module to host a stub and are skipped.
fn group_known_types_by_module(
    spec: &LoaderSpec,
) -> std::collections::BTreeMap<(String, String), Vec<&crate::loader_spec::KnownExternalType>> {
    let mut kt_by_module: std::collections::BTreeMap<
        (String, String),
        Vec<&crate::loader_spec::KnownExternalType>,
    > = std::collections::BTreeMap::new();
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
    kt_by_module
}
