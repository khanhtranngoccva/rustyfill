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
//! [`generate`] in this file is a thin orchestrator over four publicly
//! callable phases defined in [`phases`]:
//!
//! - [`phases::run_discovery_phase`] — locate declarations, parse and register
//!   sources, follow imports to a fixed point.
//! - [`phases::run_registry_phase`] — populate the binding model and mirror
//!   minimal modules for preserved qualifiers.
//! - [`phases::run_emit_phase`] — write binding files, known-type stubs, and
//!   re-export aliases; demote empty nodes.
//! - [`phases::run_manifest_phase`] — emit the hierarchical manifest and
//!   validate.
//!
//! Internal submodules provide the implementation details:
//!
//! - [`discover`] — Phase 1 internals.
//! - [`registry`] — Phase 2 internals.
//! - [`mirror`] — cfg-selected re-export shims (Strategy B materialization).
//! - [`emit`] — Phase 3 internals.
//! - [`util`] — spec-derived input builders and path/module helpers.

use std::collections::HashMap;
use std::path::Path;

use crate::loader_spec::LoaderSpec;
use crate::parser::{CfgContext, ParsedSource};
use crate::resolver::ModuleResolver;

mod discover;
mod emit;
mod mirror;
pub mod phases;
mod registry;
mod util;

pub use phases::{
    DiscoveryPhaseResult, EmitPhaseResult, ManifestPhaseResult, PipelineState, RegistryPhaseResult,
    new_pipeline_state, run_discovery_phase, run_emit_phase, run_manifest_phase, run_registry_phase,
};
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
///
/// This is a thin composition of the four publicly callable phases in
/// [`phases`]. Each phase is independently testable; see
/// [`phases::PipelineState`] for driving them individually.
pub fn generate(input: &PipelineInput<'_>) -> GenerateOutcome {
    let PipelineInput {
        rust_src,
        out_dir,
        spec,
        cfg,
    } = input;

    let mut state = phases::new_pipeline_state(rust_src, out_dir, spec, cfg);

    // Phase 1: Discovery
    let discovery_result = match phases::run_discovery_phase(&mut state) {
        Ok(d) => d,
        Err(errors) => return Err(GenerateReport { errors, warnings: Vec::new() }),
    };

    // Phase 2: Registry / Mirror
    let registry_result = phases::run_registry_phase(&mut state, &discovery_result);
    if !registry_result.field_errors.is_empty() {
        let mut errors: Vec<String> = registry_result.field_errors;
        errors.push(format!(
            "Field publicity check failed with {} error(s).",
            errors.len() - 1
        ));
        return Err(GenerateReport { errors, warnings: Vec::new() });
    }

    // Phase 3: Emit
    let emit_result = phases::run_emit_phase(&mut state);

    // Phase 4: Manifest + Validate
    let manifest_result = phases::run_manifest_phase(&mut state, &emit_result);
    if !manifest_result.errors.is_empty() {
        return Err(GenerateReport {
            errors: manifest_result.errors,
            warnings: manifest_result.warnings,
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
