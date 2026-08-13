//! Loader specification for rustyfill-sys binding generation.
//!
//! Declares which data structures from the Rust standard library (core, alloc, std)
//! need mirrored bindings. Called by `build.rs` at compile time — not part of the
//! runtime library.
//!
//! Canonical files get real type definitions emitted. Re-exports are discovered
//! automatically by parsing each canonical file's `use` statements and resolving
//! paths against the module tree built from all registered files. No manual
//! alias declarations are needed.
//!
//! Change this file when the main polyfill's needs change.

use crate::LoaderSpec;
use crate::loader_spec::BindingTarget;

/// Returns the complete loader specification for all three built-in libraries.
pub fn get_loader_spec() -> LoaderSpec {
    let mut spec = LoaderSpec::new();

    // ── std ────────────────────────────────────────────────────────────────
    spec.add_target(std_target());

    // ── core ───────────────────────────────────────────────────────────────
    spec.add_target(core_target());

    // ── alloc ──────────────────────────────────────────────────────────────
    spec.add_target(alloc_target());

    spec
}

fn std_target() -> BindingTarget {
    let mut target = BindingTarget::new("std");

    // Canonical entry point: the platform abstraction layer root.
    // The build script discovers all inner files transitively via `mod X;`
    // declarations, emitting bindings for every file that contains types.
    target.add_canonical("sys/pal/mod.rs");

    target
}

fn core_target() -> BindingTarget {
    let mut target = BindingTarget::new("core");

    // Replace Unique with NonNull: both are thin pointer wrappers with identical
    // layout. The generated Box<T,A>(Unique<T>, A) becomes Box<T,A>(NonNull<T>, A),
    // avoiding the need to mirror core::ptr::Unique and its PointeeSized bound.
    target.replace_path("core::ptr::Unique", "NonNull");

    target
}

fn alloc_target() -> BindingTarget {
    let mut target = BindingTarget::new("alloc");

    // Add binding types for btreemap
    target.add_canonical("collections/btree/mod.rs");

    // Add binding for Box<T, A> so that generated bindings reference a mirrored
    // Box struct owned by this crate rather than the real alloc::boxed::Box.
    // This avoids the unstable allocator_api feature entirely.
    target.add_canonical("boxed.rs");

    // Ignore Allocator trait: the polyfill only operates over the global
    // allocator (a ZST), so callers cast references to unit structs and
    // invoke fallible box methods manually. Stripped from trait bounds.
    target.ignore_path("core::alloc::Allocator");

    // Replace Global with () since it requires unstable allocator_api.
    target.replace_path("alloc::alloc::Global", "()");

    // Skip structs whose generated definitions fail to compile due to missing
    // trait impls on their inner types (Iterator, Debug, Clone) that we don't
    // emit bindings for. These can be re-enabled once the missing impls are added.
    target.ignore_struct("collections::btree::set::Iter");
    target.ignore_struct("collections::btree::set::IntoIter");
    target.ignore_struct("collections::btree::set::Range");
    target.ignore_struct("collections::btree::set::Cursor");
    target.ignore_struct("collections::btree::set::SymmetricDifference");
    target.ignore_struct("collections::btree::set::Union");
    target.ignore_struct("collections::btree::set::Difference");
    target.ignore_struct("collections::btree::set::DifferenceInner");
    target.ignore_struct("collections::btree::set::Intersection");
    target.ignore_struct("collections::btree::set::IntersectionInner");
    target.ignore_struct("collections::btree::map::IntoIter");
    target.ignore_struct("collections::btree::map::IntoKeys");
    target.ignore_struct("collections::btree::map::IntoValues");
    target.ignore_struct("collections::btree::map::Range");
    target.ignore_struct("collections::btree::map::Cursor");

    target
}
