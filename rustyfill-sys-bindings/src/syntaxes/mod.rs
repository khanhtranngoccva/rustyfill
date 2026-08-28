//! Path-syntax newtypes for the binding pipeline.
//!
//! The pipeline moves between two textual spellings of "where a module or
//! type lives": slash-separated file-relative paths (`sys/pal/unix/sync`)
//! and `::`-separated canonical paths (`std::sys::sync::mutex::Mutex`).
//! Conversions are centralized in [`ModulePath`] so the rest of the crate
//! doesn't do ad-hoc path string surgery.

pub mod module_path;

pub use module_path::ModulePath;
