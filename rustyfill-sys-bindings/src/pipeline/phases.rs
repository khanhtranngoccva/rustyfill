//! Publicly callable pipeline phases, each with its own observable result.
//!
//! The full [`super::generate`] run is a thin composition of four phases:
//!
//! 1. [`run_discovery_phase`] — parse and register every source file reachable
//!    from the spec declarations; follow imports to a fixed point.
//! 2. [`run_registry_phase`] — populate the binding model (declared paths,
//!    qualifier routes, alias RHS) and mirror minimal modules.
//! 3. [`run_emit_phase`] — write binding files, known-type stubs, re-export
//!    aliases, and demote empty nodes.
//! 4. [`run_manifest_phase`] — emit the hierarchical manifest and validate.
//!
//! Each phase consumes a [`PipelineState`] (the shared mutable accumulators)
//! and returns a small result struct describing what it did. Tests can invoke
//! individual phases against a fixture tree and assert on the intermediate
//! state without running the full pipeline.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::emitter::emit_preamble_module;
use crate::loader_spec::LoaderSpec;
use crate::parser::{CfgContext, ParsedSource};
use crate::resolver::ModuleResolver;
use crate::syntaxes::BindingModel;
use crate::validator::ValidationBuilder;

use super::discover;
use super::emit;
use super::mirror;
use super::registry;
use super::util;
use super::{group_known_types_by_module, normalize_spec, sync_file_libs};

// ── Shared state threaded through all phases ─────────────────────────────────

/// Mutable accumulators shared across pipeline phases. Constructed by
/// [`new_pipeline_state`]; consumed sequentially by the four phase functions.
pub struct PipelineState {
    /// Normalized spec (leading `::` stripped from all path entries).
    pub spec: LoaderSpec,
    /// Root of the Rust standard-library source tree.
    pub rust_src: PathBuf,
    /// Directory generated bindings are written into.
    pub out_dir: PathBuf,
    /// Platform context for cfg evaluation.
    pub cfg: CfgContext,

    // ── Core registries ──────────────────────────────────────────────────────
    /// Module resolver tracking mod declarations, file locations, and libs.
    pub resolver: ModuleResolver,
    /// The binding model: single source of truth for declared status,
    /// visibility, export, alias RHS, and definition file.
    pub model: BindingModel,
    /// Parsed sources keyed by slash-separated relative file path.
    pub parsed_cache: HashMap<String, (ParsedSource, String)>,

    // ── Cross-phase accumulators ─────────────────────────────────────────────
    /// Every file that will appear in the final manifest (file, lib).
    pub all_files: Vec<(String, String)>,
    /// Canonical names already emitted (dedup guard).
    pub emitted_canonicals: HashSet<String>,
    /// Validator collecting diagnostics across all phases.
    pub validator: ValidationBuilder,

    // ── Spec-derived emission inputs (computed once, reused) ─────────────────
    pub replacement_entries: Vec<(String, Option<String>)>,
    pub ignored_structs_by_lib: HashMap<String, Vec<String>>,
    pub extra_derives_by_lib: HashMap<String, HashMap<String, Vec<String>>>,
    pub ignored_names_owned: Vec<String>,
}

impl PipelineState {
    /// Alias for [`new_pipeline_state`] that makes test call sites read more
    /// naturally (`PipelineState::for_test(...)`).
    pub fn for_test(rust_src: &Path, out_dir: &Path, spec: &LoaderSpec, cfg: &CfgContext) -> Self {
        new_pipeline_state(rust_src, out_dir, spec, cfg)
    }

    /// Convenience: iterate over all cached file paths.
    pub fn cached_files(&self) -> impl Iterator<Item = &str> + '_ {
        self.parsed_cache.keys().map(String::as_str)
    }

    /// Number of files registered in the parsed cache.
    pub fn cached_file_count(&self) -> usize {
        self.parsed_cache.len()
    }

    /// Check whether a specific file is in the parsed cache.
    pub fn has_cached_file(&self, path: &str) -> bool {
        self.parsed_cache.contains_key(path)
    }
}

/// Create a fresh [`PipelineState`] from the raw inputs, performing spec
/// normalization and pre-flight validation. This is the entry point for tests
/// that want to drive individual phases.
pub fn new_pipeline_state(
    rust_src: &Path,
    out_dir: &Path,
    spec: &LoaderSpec,
    cfg: &CfgContext,
) -> PipelineState {
    let normalized_spec = normalize_spec(spec);
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
                let key = path.strip_prefix("::").unwrap_or(path);
                map.entry(key.to_string())
                    .or_default()
                    .extend(derives.iter().cloned());
            }
            (t.lib_name.clone(), map)
        })
        .collect();
    let ignored_names_owned =
        util::collect_ignored_names(&replacement_entries, &ignored_structs_by_lib);

    let mut validator = ValidationBuilder::new();
    validator.check_spec(&normalized_spec, rust_src);

    PipelineState {
        spec: normalized_spec,
        rust_src: rust_src.to_path_buf(),
        out_dir: out_dir.to_path_buf(),
        cfg: (*cfg).clone(),
        resolver: ModuleResolver::new(),
        model: BindingModel::new(),
        parsed_cache: HashMap::new(),
        all_files: Vec::new(),
        emitted_canonicals: HashSet::new(),
        validator,
        replacement_entries,
        ignored_structs_by_lib,
        extra_derives_by_lib,
        ignored_names_owned,
    }
}

// ── Phase results ─────────────────────────────────────────────────────────────

/// Result of Phase 1 (Discovery): which files were parsed and registered.
#[derive(Debug, Default)]
pub struct DiscoveryPhaseResult {
    /// All files registered in the parsed cache after discovery completes.
    pub discovered_files: Vec<String>,
    /// Declarations whose definition was found at a non-naive path.
    pub reexport_located: Vec<(String, String)>,
    /// Parent modules processed during structural registration.
    pub processed_parents: HashSet<String>,
}

/// Result of Phase 2 (Registry/Mirror): what was populated in the model.
#[derive(Debug, Default)]
pub struct RegistryPhaseResult {
    /// Number of types registered in the binding model.
    pub types_registered: usize,
    /// Number of declared paths in the binding model.
    pub declared_paths: usize,
    /// Files added to `all_files` by this phase (shims, mirrors).
    pub files_added: Vec<String>,
    /// Field publicity check errors (non-empty = fatal).
    pub field_errors: Vec<String>,
}

/// Result of Phase 3 (Emit): what was written to disk.
#[derive(Debug, Default)]
pub struct EmitPhaseResult {
    /// Absolute paths of all binding files written.
    pub emitted_paths: Vec<PathBuf>,
    /// Relative paths of all files now in `all_files`.
    pub all_files_snapshot: Vec<(String, String)>,
    /// Re-export aliases discovered and emitted.
    pub discovered_aliases: HashSet<String>,
    /// Files marked as emitted in the model.
    pub emitted_file_set: std::collections::BTreeSet<String>,
}

/// Result of Phase 4 (Manifest/Validate): final output and validation.
#[derive(Debug, Default)]
pub struct ManifestPhaseResult {
    /// Whether the manifest was successfully written.
    pub manifest_written: bool,
    /// Validation errors (empty = success).
    pub errors: Vec<String>,
    /// Validation warnings.
    pub warnings: Vec<String>,
}

// ── Phase implementations ─────────────────────────────────────────────────────

/// Phase 1: Parse all files, register with resolver, discover imports.
///
/// After this phase, `state.parsed_cache` contains every source file needed
/// for emission, and `state.resolver` knows the module tree structure.
pub fn run_discovery_phase(state: &mut PipelineState) -> Result<DiscoveryPhaseResult, Vec<String>> {
    let mut discovery = discover::run_discovery(
        &state.spec,
        &state.rust_src,
        &state.cfg,
        &mut state.resolver,
        &mut state.validator,
        &mut state.parsed_cache,
        &mut state.model,
    )?;

    sync_file_libs(&state.parsed_cache, &mut state.resolver);

    // Mark all Phase 1 files as emittable, EXCEPT structural parents.
    for file_path in state.parsed_cache.keys() {
        if file_path.ends_with("/mod") || file_path == "mod" {
            continue;
        }
        state.resolver.mark_emittable(file_path);
    }

    // Phase 1b: Register structural parents.
    let phase1b_files: Vec<String> = state.parsed_cache.keys().cloned().collect();
    for target in &state.spec.targets {
        let lib_src = state.rust_src.join(&target.lib_name).join("src");
        for file_path in &phase1b_files {
            discover::register_parents_of(
                file_path,
                &target.lib_name,
                &lib_src,
                &state.cfg,
                &mut state.resolver,
                &mut state.parsed_cache,
                &mut state.model,
                &mut discovery.processed_parents,
            );
        }
    }
    sync_file_libs(&state.parsed_cache, &mut state.resolver);

    // Phase 1c: Discover modules referenced by use statements.
    discover::discover_imported_modules(
        &state.spec,
        &state.rust_src,
        &state.cfg,
        &mut state.resolver,
        &mut state.parsed_cache,
        &mut state.model,
    );
    sync_file_libs(&state.parsed_cache, &mut state.resolver);

    let discovered_files: Vec<String> = state.parsed_cache.keys().cloned().collect();
    Ok(DiscoveryPhaseResult {
        discovered_files,
        reexport_located: discovery.reexport_located,
        processed_parents: discovery.processed_parents,
    })
}

/// Phase 2: Build the type registry (binding model population) and mirror
/// minimal modules. Also emits cfg re-export shims.
///
/// After this phase, `state.model` is fully populated with declared paths,
/// qualifier routes, and alias RHS values.
pub fn run_registry_phase(
    state: &mut PipelineState,
    discovery: &DiscoveryPhaseResult,
) -> RegistryPhaseResult {
    let files_before = state.all_files.len();

    let mut reg_state = registry::RegistryBuildState {
        resolver: &mut state.resolver,
        parsed_cache: &mut state.parsed_cache,
        model: &mut state.model,
        emitted_canonicals: &mut state.emitted_canonicals,
        all_files: &mut state.all_files,
    };
    registry::build_type_registry(
        &state.spec,
        &state.rust_src,
        &state.cfg,
        &discovery.reexport_located,
        &state.out_dir,
        &mut reg_state,
    );

    // Materialize cfg re-export shims.
    mirror::emit_cfg_reexport_shims(
        &state.rust_src,
        &state.spec.targets,
        &state.out_dir,
        &mut state.resolver,
        &mut state.model,
        &state.cfg,
        &mut state.emitted_canonicals,
        &mut state.all_files,
    );
    for (file_path, lib_name) in &state.all_files {
        state.resolver.set_file_lib(file_path, lib_name);
    }

    // Hand the resolver the declared-path set for filtering.
    state
        .resolver
        .set_declared_paths(state.model.declared_paths().into_iter());

    // Field-type publicity check.
    let field_errors = crate::emitter::check_declared_struct_fields(&state.model);

    let files_added: Vec<String> = state.all_files[files_before..]
        .iter()
        .map(|(f, _)| f.clone())
        .collect();

    let declared = state.model.declared_paths();
    RegistryPhaseResult {
        types_registered: state.model.files().len(),
        declared_paths: declared.len(),
        files_added,
        field_errors,
    }
}

/// Phase 3: Emit binding files, known-type stubs, and re-export aliases.
///
/// After this phase, all `.rs` binding files are written to `out_dir` and
/// `state.model` reflects which files have content.
pub fn run_emit_phase(state: &mut PipelineState) -> EmitPhaseResult {
    // Phase 0 preamble (idempotent; safe to call here or before).
    let mut preamble_emitted: HashSet<String> = HashSet::new();
    for target in &state.spec.targets {
        if preamble_emitted.insert(target.lib_name.clone()) {
            emit_preamble_module(&state.out_dir, &target.lib_name);
        }
    }

    let replacement_entries_vec = util::replacement_view(&state.replacement_entries);
    let ignored_name_refs: Vec<&str> = state
        .ignored_names_owned
        .iter()
        .map(|s| s.as_str())
        .collect();

    let mut emitted_paths: Vec<PathBuf> = Vec::new();
    emit::emit_all_binding_files(
        &mut state.model,
        &state.parsed_cache,
        &mut state.resolver,
        &replacement_entries_vec,
        &ignored_name_refs,
        &state.ignored_structs_by_lib,
        &state.extra_derives_by_lib,
        &state.out_dir,
        &mut state.validator,
        &mut emitted_paths,
        &mut state.emitted_canonicals,
        &mut state.all_files,
    );

    // Phase 2b: Known-external-type stubs.
    let kt_groups = group_known_types_by_module(&state.spec);
    for ((lib_name, _module_slash), kts) in &kt_groups {
        if let Some(rel_path) = crate::emitter::emit_known_type_stubs(&state.out_dir, kts) {
            state.validator.check_emit(&state.out_dir.join(&rel_path));
            emitted_paths.push(state.out_dir.join(&rel_path));
            state.emitted_canonicals.insert(rel_path.clone());
            state.resolver.set_file_lib(&rel_path, lib_name);
            state.all_files.push((rel_path.clone(), lib_name.clone()));
            state.model.register_synthetic(
                lib_name,
                &rel_path,
                crate::syntaxes::NodeStatus::Emittable,
            );
            state.model.mark_file_emitted(&rel_path);
        }
    }

    // Phase 3: Re-export aliases.
    let discovered_aliases = emit::discover_and_emit_reexport_aliases(
        &mut state.model,
        &state.spec,
        &state.cfg,
        &state.parsed_cache,
        &mut state.resolver,
        &state.out_dir,
        &state.emitted_canonicals,
        &mut state.all_files,
    );

    // Phase 3b: Demote empty emittable nodes.
    state.model.demote_empty_emittable();

    EmitPhaseResult {
        emitted_paths,
        all_files_snapshot: state.all_files.clone(),
        discovered_aliases,
        emitted_file_set: state.model.emitted_file_set().clone(),
    }
}

/// Phase 4: Emit the hierarchical manifest and run final validation.
///
/// After this phase, `bindings_generated.rs` exists in `out_dir` and all
/// validation checks have been performed.
pub fn run_manifest_phase(
    state: &mut PipelineState,
    emit_result: &EmitPhaseResult,
) -> ManifestPhaseResult {
    crate::emitter::emit_hierarchical_manifest(&state.out_dir, &state.model);

    state
        .validator
        .check_manifest(&state.out_dir, &state.all_files);
    state
        .validator
        .check_aliases(&mut state.resolver, &emit_result.discovered_aliases);
    for path in &emit_result.emitted_paths {
        state.validator.check_emit(path);
    }

    // `finish` consumes the builder; swap in a fresh one so the state stays usable.
    let validator = std::mem::replace(&mut state.validator, ValidationBuilder::new());
    let result = validator.finish();
    ManifestPhaseResult {
        manifest_written: true,
        errors: result.errors.errors,
        warnings: Vec::new(),
    }
}
