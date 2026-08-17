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
pub mod resolver;
pub mod spec;
pub mod validator;

pub use emitter::{
    check_declared_struct_fields, emit_binding_file, emit_glob_reexport_aliases,
    emit_hierarchical_manifest, emit_parsed_items, EmitConfig, FieldRefResolution, TypeRegistry,
    TypeInfo,
};
pub use loader_spec::{BindingTarget, LoaderSpec};
pub use parser::{
    CfgContext, ItemVisibility, ModDeclaration, ParsedItem, ParsedSource, parse_file, parse_item,
    parse_mod_declarations, parse_source, parse_source_with_cfg, parse_use_statements,
};
pub use resolver::{ModuleResolver, Resolution, UseStatement};
pub use spec::get_loader_spec;
pub use validator::{ValidationBuilder, ValidationErrors, ValidationResult};
