//! The binding model: a single tree that every pipeline phase reads and writes.
//!
//! Historically the generator carried five parallel accumulators — a parsed-source
//! cache, a resolver with eight string-keyed maps, a type registry with seven
//! fields, an `all_files` list, and an `emitted_canonicals` set — and reconciled
//! them at the end by re-parsing file-path strings into a throwaway manifest tree.
//! That made one domain concept ("where does this module/item live?") span ~40
//! string-manipulation sites and let whole classes of bugs (duplicate-module
//! collisions, wrong parent scoping, dangling re-exports) slip through.
//!
//! [`BindingModel`] collapses those accumulators into one owned forest: one root
//! per target library (`core`, `alloc`, `std`), each a nested map of
//! [`ModuleNode`]s. Structs, enums, unions, consts, and type aliases hang off
//! their defining module as [`ItemRecord`]s; `use` statements become first-class
//! [`ImportEdge`]s. Every node and item carries its fully-qualified path *by
//! construction* (library + module path + name), so nothing downstream has to
//! reconstruct it from a slash-separated filename.
//!
//! The tree is the source of truth for the phases that consume it; the legacy
//! accumulators are being retired incrementally in favour of the derived views
//! exposed here ([`BindingModel::files`], [`BindingModel::emitted`], …).

mod file_form;
mod import_edge;
mod item_record;
mod model;
mod module_node;
mod node_status;
mod qualified_path;

pub use file_form::FileForm;
pub use import_edge::ImportEdge;
pub use item_record::ItemRecord;
pub use model::BindingModel;
pub use module_node::ModuleNode;
pub use node_status::NodeStatus;
pub use qualified_path::QualifiedPath;
