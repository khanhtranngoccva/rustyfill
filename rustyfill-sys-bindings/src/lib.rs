//! Binding generator for mirroring Rust standard library data structures.
//!
//! The generation model is [`docjson`]: it invokes `cargo doc
//! --output-format=json` inside the rust-src tree to obtain authoritative
//! type definitions from the compiler, then emits regenerated Rust source
//! that preserves field layout, alignment attributes, visibility, and
//! generic parameters.

pub mod docjson;
pub mod formatter;
pub mod loader_spec;
pub mod syntaxes;

pub use syntaxes::ModulePath;

pub use loader_spec::{BindingTarget, LoaderSpec};
