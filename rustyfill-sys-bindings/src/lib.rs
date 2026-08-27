//! Parser, code generator, and binding specification for mirroring Rust
//! standard library data structures.
//!
//! Reads `.rs` source files from the Rust toolchain's library source tree,
//! extracts `struct`, `enum`, and `union` definitions via `syn`, and emits
//! regenerated Rust source that preserves field layout, alignment attributes,
//! visibility, and generic parameters — producing bit-identical mirrors of
//! the original types.

pub mod emitter;
pub mod formatter;
pub mod loader_spec;
pub mod parser;
pub mod pipeline;
pub mod resolver;
pub mod syntaxes;
pub mod validator;

pub use syntaxes::ModulePath;

pub use emitter::{
    EmitConfig, FieldRefResolution, check_declared_struct_fields, emit_binding_file,
    emit_glob_reexport_aliases, emit_hierarchical_manifest, emit_parsed_items,
};
pub use loader_spec::{BindingTarget, LoaderSpec};
pub use parser::{
    CfgContext, ModDeclaration, ParsedItem, ParsedSource, cfg_select_reexport_targets, parse_file,
    parse_item, parse_mod_declarations, parse_source, parse_source_with_cfg, parse_use_statements,
};
pub use pipeline::{
    DiscoveryPhaseResult, EmitPhaseResult, GenerateOutcome, GenerateReport, ManifestPhaseResult,
    PipelineInput, PipelineState, RegistryPhaseResult, generate, new_pipeline_state,
    run_discovery_phase, run_emit_phase, run_manifest_phase, run_registry_phase,
};
pub use resolver::{ModuleResolver, Resolution, UseStatement};
// Unified visibility type (formerly `parser::ItemVisibility` and
// `syntaxes::Visibility`). The old name is kept as a deprecated alias so
// external import paths keep working while callers migrate to `Visibility`.
#[deprecated(note = "renamed to `crate::Visibility`")]
pub use Visibility as ItemVisibility;
pub use syntaxes::Visibility;
pub use validator::{ValidationBuilder, ValidationErrors, ValidationResult};
