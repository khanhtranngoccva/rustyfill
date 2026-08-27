//! Path-syntax newtypes for the binding pipeline.
//!
//! The pipeline moves between three textual spellings of "where a module or
//! type lives": slash-separated file-relative paths (`sys/pal/unix/sync`),
//! `::`-separated canonical paths (`std::sys::sync::mutex::Mutex`), and
//! structured segment lists. Historically each conversion was inlined at the
//! call site (`replace("::", "/")`, `strip_suffix("/mod")`,
//! `rsplit("::").next()`, …), which scattered one domain concept across ~40
//! string-manipulation sites and made whole classes of bugs (duplicate-module
//! collisions, wrong parent scoping) easy to introduce.
//!
//! This module centralizes those syntactic concepts as one owned type per
//! file. Each type owns its invariants and exposes the conversions as methods,
//! so the rest of the crate can stop doing path string surgery.

pub mod binding_model;
pub mod module_path;

pub use binding_model::{BindingModel, FileForm, ImportEdge, ItemRecord, ModuleNode, NodeStatus, QualifiedPath};
pub use module_path::ModulePath;
