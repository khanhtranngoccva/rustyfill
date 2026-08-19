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

    // The canonical, cfg-selected sys mutex (`std::sys::sync::mutex::Mutex`).
    // On Linux this is the futex-backed implementation; on other unix targets
    // it is the pthread-backed one. The fallible `Mutex` polyfill reserves its
    // backing storage ahead of time via this type, so we mirror it and its
    // lazy-allocation helper `OnceBox`.
    target.declare_struct("sys::sync::mutex::Mutex");
    // The futex Mutex (the active backend on Linux) stores its state through two
    // file-local private type aliases. Declaring them mirrors the aliases and
    // routes their RHS (`futex::SmallFutex` / `futex::SmallPrimitive`, both
    // public) through the registry, satisfying the field-publicity check — the
    // same treatment as btree's private `BoxedNode` alias.
    target.declare_struct("sys::sync::mutex::futex::Futex");
    target.declare_struct("sys::sync::mutex::futex::State");
    target.declare_struct("sys::sync::once_box::OnceBox");

    // ── Known external types (emitted into the shared preamble) ──────────────
    // The futex Mutex and OnceBox reference a bare `Atomic<T>` (via
    // `use crate::sync::atomic::{... Atomic ...}`). The real type is
    // `#[unstable(feature = "generic_atomic")]` and holds `UnsafeCell<T::Storage>`
    // behind an `AtomicPrimitive` bound, which won't compile in our no_std
    // downstream tree. So we polyfill just the *shape* — a transparent
    // `UnsafeCell<T>` wrapper — as a spec-declared known type instead of
    // mirroring the generic machinery. Only the type shape matters for the
    // bindings; atomic operations are provided by the main crate.
    target.add_known_type(
        "Atomic",
        "#[repr(transparent)] pub struct Atomic<T>(::__rustyfill_builtin_core::cell::UnsafeCell<T>);",
    );

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

    // Mirror collections::TryReserveError and its TryReserveErrorKind enum. The
    // main crate re-exports the standard library's own `TryReserveError` and
    // constructs it by transmuting from this generated mirror (see
    // `rustyfill/src/alloc.rs`). Because these bindings are emitted directly
    // from the std source, any change to the real type's fields or variants in
    // the standard library breaks compilation here — surfacing layout drift at
    // build time rather than letting a hand-written mirror silently diverge.
    target.declare_struct("collections::TryReserveError");
    target.declare_struct("collections::TryReserveErrorKind");
    // The real `TryReserveError` derives Clone/PartialEq/Eq/Debug and embeds a
    // `kind: TryReserveErrorKind`, so the mirrored enum must carry the same set
    // for those container derives to expand. Inject them explicitly rather than
    // relying on the generator to pick up std's per-type derives.
    for derive in ["Clone", "PartialEq", "Eq", "Debug"] {
        target.add_derive("collections::TryReserveErrorKind", derive);
    }

    // ── B-tree containers ───────────────────────────────────────────────────
    // Only the two public container shells are mirrored. Their iterators,
    // cursors, range views, and set-algebra engines are peripheral to the
    // fallible-insertion polyfill and are deliberately left undeclared: any
    // references to them route straight back to the original builtin types.
    target.declare_struct("collections::btree::map::BTreeMap");
    target.declare_struct("collections::btree::set::BTreeSet");

    // Entry API (map/entry.rs and set/entry.rs). The fallible entry methods
    // manipulate these directly, so they must be mirrored.
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

    // borrow.rs: stacked-borrow helper used by the fallible entry methods.
    target.declare_struct("collections::btree::borrow::DormantMutRef");

    // set_val.rs: zero-sized value type making BTreeSet a BTreeMap<T, SetValZST>.
    target.declare_struct("collections::btree::set_val::SetValZST");

    // Add binding for Box<T, A> so that generated bindings reference a mirrored
    // Box struct owned by this crate rather than the real alloc::boxed::Box.
    // This avoids the unstable allocator_api feature entirely.
    target.declare_struct("boxed::Box");

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
