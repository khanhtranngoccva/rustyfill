//! Typed deserialization layer for rustdoc JSON.
//!
//! One file per domain type. The wire format is **externally tagged**: every
//! sum value is `{"<tag>": <payload>}`, so plain derived `Deserialize` with
//! `#[serde(rename_all = "snake_case")]` handles it directly — including the
//! self-recursive types (`TypeRepr`, `GenericArg`, `GenericArgs`). External
//! tagging reads one key and dispatches to exactly one variant, so there is no
//! fan-out over variants and no recursion hazard on malformed input; unknown
//! tags fail loudly with "unknown variant …".
//!
//! Format-version drift that stays within a single field (e.g. `name` ↔ `path`)
//! is absorbed with `#[serde(alias)]`. If a future format forks an entire
//! sub-shape, model it as versioned variants in that file — one domain type per
//! file keeps such changes local.

pub mod array_ty;
mod attributes;
pub mod borrowed_ref;
pub mod crate_;
pub mod dyn_trait;
pub mod function_pointer;
pub mod generic_arg;
pub mod generic_args;
pub mod id;
pub mod item;
pub mod pat_ty;
pub mod primitive;
pub mod qualified_path;
pub mod raw_pointer;
pub mod resolved_path;
pub mod span;
pub mod type_repr;
pub mod visibility;
pub mod original;

// Re-export all public types at module level for ergonomic imports.
pub use array_ty::ArrayTy;
pub use borrowed_ref::BorrowedRef;
pub use dyn_trait::{BoundTrait, DynTrait, ImplBound, ImplicitBound, TraitBound};
pub use function_pointer::{Abi, FnHeader, FnInput, FnSig, FunctionPointer};
pub use generic_arg::{AssocArgs, AssocBinding, AssocConstraint, ConstArg, GenericArg};
pub use generic_args::GenericArgs;
pub use pat_ty::PatTy;
pub use qualified_path::{QpTrait, QualifiedPath};
pub use raw_pointer::RawPointer;
pub use resolved_path::ResolvedPath;
pub use type_repr::TypeRepr;

/// Normalize a rustdoc path for emission.
/// Converts `$crate::foo::Bar` or `crate::foo::Bar` → `foo::Bar`.
pub(crate) fn normalize_path(path: &str) -> String {
    let stripped = path
        .strip_prefix("$crate::")
        .or_else(|| path.strip_prefix("crate::"))
        .unwrap_or(path);
    stripped.to_string()
}
