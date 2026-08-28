//! Binding generator for mirroring Rust standard library data structures.
//!
//! The active generation model is [`docjson`]: it invokes `cargo doc
//! --output-format=json` inside the rust-src tree to obtain authoritative
//! type definitions from the compiler, then emits regenerated Rust source
//! that preserves field layout, alignment attributes, visibility, and
//! generic parameters.
//!
//! The legacy source-parsing model (`parser`, `pipeline`, `resolver`,
//! `validator`) is deprecated and scheduled for removal. It remains
//! available for backward compatibility but should not be used for new work.

pub mod docjson;
pub mod formatter;
pub mod loader_spec;
pub mod syntaxes;

// ── Deprecated: legacy source-parsing model ───────────────────────────────────

#[deprecated(note = "replaced by `docjson::emitter`; scheduled for removal")]
pub mod emitter;
#[deprecated(note = "replaced by `docjson`; scheduled for removal")]
pub mod parser;
#[deprecated(note = "replaced by `docjson`; scheduled for removal")]
pub mod pipeline;
#[deprecated(note = "replaced by `docjson`; scheduled for removal")]
pub mod resolver;
#[deprecated(note = "replaced by `docjson`; scheduled for removal")]
pub mod validator;

pub use syntaxes::ModulePath;

pub use loader_spec::{BindingTarget, LoaderSpec};
pub use syntaxes::Visibility;

// Deprecated re-exports for backward compatibility.
#[allow(deprecated)]
pub use parser::CfgContext;
#[allow(deprecated)]
pub use resolver::{ModuleResolver, Resolution, UseStatement};
#[allow(deprecated)]
pub use validator::{ValidationBuilder, ValidationErrors, ValidationResult};
