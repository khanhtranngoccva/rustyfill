//! Loader specification for rustyfill-sys binding generation.
//!
//! Declares which data structures from the Rust standard library (core, alloc, std)
//! need mirrored bindings. Called by `build.rs` at compile time — not part of the
//! runtime library.
//!
//! Structs are declared explicitly in path syntax, relative to the library root
//! (e.g., `"collections::btree::map::BTreeMap"`). For each declaration the build
//! script locates the defining source file, emits the definition, and makes every
//! re-export alias of the struct resolve to that same emitted definition. Field
//! types of declared structs are checked for publicity: public undeclared types
//! keep referring to the original type, private undeclared types are an error,
//! and declared types point at their mirrored bindings.
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

    // The platform abstraction layer's synchronization primitives. Declared by
    // path so that both the canonical location and any module-level aliases
    // (e.g., through `sys/pal/sync`) resolve to the same emitted definitions.
    target.declare_struct("sys::pal::unix::sync::mutex::Mutex");
    target.declare_struct("sys::pal::unix::sync::condvar::Condvar");

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

    // ── B-tree containers ───────────────────────────────────────────────────
    // Public container shells plus every type reachable through their fields.
    // Private internals (node markers, handles, ranges, iterators) are
    // declared explicitly so that the field-publicity check passes and every
    // reference routes to a mirrored definition in our tree.
    target.declare_struct("collections::btree::map::BTreeMap");
    target.declare_struct("collections::btree::set::BTreeSet");

    // map.rs public API surface (containers, iterators, cursors, entry).
    target.declare_struct("collections::btree::map::Iter");
    target.declare_struct("collections::btree::map::IterMut");
    target.declare_struct("collections::btree::map::IntoIter");
    target.declare_struct("collections::btree::map::Keys");
    target.declare_struct("collections::btree::map::Values");
    target.declare_struct("collections::btree::map::ValuesMut");
    target.declare_struct("collections::btree::map::IntoKeys");
    target.declare_struct("collections::btree::map::IntoValues");
    target.declare_struct("collections::btree::map::Range");
    target.declare_struct("collections::btree::map::RangeMut");
    target.declare_struct("collections::btree::map::ExtractIf");
    target.declare_struct("collections::btree::map::ExtractIfInner");
    target.declare_struct("collections::btree::map::Cursor");
    target.declare_struct("collections::btree::map::CursorMut");
    target.declare_struct("collections::btree::map::CursorMutKey");
    target.declare_struct("collections::btree::map::UnorderedKeyError");

    // set.rs public API surface.
    target.declare_struct("collections::btree::set::Iter");
    target.declare_struct("collections::btree::set::IntoIter");
    target.declare_struct("collections::btree::set::Range");
    target.declare_struct("collections::btree::set::Difference");
    target.declare_struct("collections::btree::set::DifferenceInner");
    target.declare_struct("collections::btree::set::SymmetricDifference");
    target.declare_struct("collections::btree::set::Intersection");
    target.declare_struct("collections::btree::set::IntersectionInner");
    target.declare_struct("collections::btree::set::Union");
    target.declare_struct("collections::btree::set::ExtractIf");
    target.declare_struct("collections::btree::set::Cursor");
    target.declare_struct("collections::btree::set::CursorMut");
    target.declare_struct("collections::btree::set::CursorMutKey");

    // Entry API (map/entry.rs and set/entry.rs).
    target.declare_struct("collections::btree::map::entry::Entry");
    target.declare_struct("collections::btree::map::entry::VacantEntry");
    target.declare_struct("collections::btree::map::entry::OccupiedEntry");
    target.declare_struct("collections::btree::map::entry::OccupiedError");
    target.declare_struct("collections::btree::set::entry::Entry");
    target.declare_struct("collections::btree::set::entry::OccupiedEntry");
    target.declare_struct("collections::btree::set::entry::VacantEntry");

    // ── B-tree internals (private, declared for mirroring) ──────────────────
    // node.rs: the physical node layout and borrow-type machinery.
    target.declare_struct("collections::btree::node::LeafNode");
    // Type alias: `BoxedNode<K, V> = NonNull<LeafNode<K, V>>` — the element
    // type of InternalNode's edges array. Declared so it is mirrored and its
    // RHS (NonNull<LeafNode>) routes through the registry.
    target.declare_struct("collections::btree::node::BoxedNode");
    target.declare_struct("collections::btree::node::InternalNode");
    target.declare_struct("collections::btree::node::NodeRef");
    target.declare_struct("collections::btree::node::Root");
    target.declare_struct("collections::btree::node::Handle");
    target.declare_struct("collections::btree::node::LeftOrRight");
    target.declare_struct("collections::btree::node::BalancingContext");
    target.declare_struct("collections::btree::node::ForceResult");
    target.declare_struct("collections::btree::node::SplitResult");
    // node/marker.rs: zero-sized marker enums/structs parameterizing NodeRef.
    target.declare_struct("collections::btree::node::marker::Leaf");
    target.declare_struct("collections::btree::node::marker::Internal");
    target.declare_struct("collections::btree::node::marker::LeafOrInternal");
    target.declare_struct("collections::btree::node::marker::Owned");
    target.declare_struct("collections::btree::node::marker::Dying");
    target.declare_struct("collections::btree::node::marker::DormantMut");
    target.declare_struct("collections::btree::node::marker::Immut");
    target.declare_struct("collections::btree::node::marker::Mut");
    target.declare_struct("collections::btree::node::marker::ValMut");
    target.declare_struct("collections::btree::node::marker::KV");
    target.declare_struct("collections::btree::node::marker::Edge");

    // navigate.rs: leaf-range navigation state used by iterators.
    target.declare_struct("collections::btree::navigate::LazyLeafRange");
    target.declare_struct("collections::btree::navigate::LeafRange");
    target.declare_struct("collections::btree::navigate::LazyLeafHandle");

    // search.rs: bound classification for lookups.
    target.declare_struct("collections::btree::search::SearchBound");

    // borrow.rs: stacked-borrow helper for mutable iteration.
    target.declare_struct("collections::btree::borrow::DormantMutRef");

    // merge_iter.rs: dual-iterator engine behind set Difference/SymmetricDiff/Union.
    target.declare_struct("collections::btree::merge_iter::MergeIterInner");
    target.declare_struct("collections::btree::merge_iter::Peeked");

    // dedup_sorted_iter.rs: wrapper used by from_sorted-style construction.
    target.declare_struct("collections::btree::dedup_sorted_iter::DedupSortedIter");

    // set_val.rs: zero-sized value type making BTreeSet a BTreeMap<T, SetValZST>.
    target.declare_struct("collections::btree::set_val::SetValZST");

    // Add binding for Box<T, A> so that generated bindings reference a mirrored
    // Box struct owned by this crate rather than the real alloc::boxed::Box.
    // This avoids the unstable allocator_api feature entirely.
    target.declare_struct("boxed::Box");

    // Mirror collections::TryReserveError and its TryReserveErrorKind enum so the
    // polyfill can construct capacity-reservation errors with diagnostic detail
    // (layout of the failed allocation, or overflow). The private `kind` field is
    // widened to public during emission.
    target.declare_struct("collections::TryReserveError");
    target.declare_struct("collections::TryReserveErrorKind");

    // Ignore Allocator trait: the polyfill only operates over the global
    // allocator (a ZST), so callers cast references to unit structs and
    // invoke fallible box methods manually. Stripped from trait bounds.
    target.ignore_path("core::alloc::Allocator");

    // Replace Global with () since it requires unstable allocator_api.
    target.replace_path("alloc::alloc::Global", "()");

    // BoxedArrayIntoIter requires unstable allocator_api feature and references
    // vec::IntoIter which needs Allocator bounds we can't satisfy in no_std.
    target.ignore_struct("boxed::iter::BoxedArrayIntoIter");

    target
}
