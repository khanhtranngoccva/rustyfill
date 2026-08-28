//! Loader specification for rustyfill-sys binding generation.
//!
//! This is the single source of truth for what gets mirrored; it lives here
//! (rather than in `rustyfill-sys-bindings`) so that changes to the polyfill's
//! needs only touch the crate whose bindings they affect. Called by
//! `build.rs` at compile time — not part of the runtime library.
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

use rustyfill_sys_bindings::{BindingTarget, LoaderSpec};

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

    // The platform abstraction layer's synchronization primitives (pthread
    // backend). Declared by path so that both the canonical location and any
    // module-level aliases (e.g., through `sys/pal/sync`) resolve to the same
    // emitted definitions. Gated to non-futex unix targets only: the
    // `pal::unix` subtree is absent on Windows (the outer `sys/pal/mod.rs`
    // selects `pal::windows` instead), and the inner `#![cfg(not(any(...)))]`
    // on `sys/pal/unix/sync/mod.rs` further excludes futex-based unix targets
    // (Linux, FreeBSD, etc.). Combined: unix family AND not-in-the-futex-list.
    const PAL_UNIX_SYNC_ACTIVE: &str = concat!(
        "all(unix, not(any(",
        "target_os = \"linux\", ",
        "target_os = \"android\", ",
        "all(target_os = \"emscripten\", target_feature = \"atomics\"), ",
        "target_os = \"freebsd\", ",
        "target_os = \"openbsd\", ",
        "target_os = \"dragonfly\", ",
        "target_os = \"fuchsia\"",
        ")))",
    );
    target.declare_struct_cfg("sys::pal::unix::sync::mutex::Mutex", PAL_UNIX_SYNC_ACTIVE);
    target.declare_struct_cfg(
        "sys::pal::unix::sync::condvar::Condvar",
        PAL_UNIX_SYNC_ACTIVE,
    );

    // The canonical, cfg-selected sys mutex (`std::sys::sync::mutex::Mutex`).
    // Resolves via cfg_select! to the futex backend on Windows/Linux/FreeBSD/
    // etc., to pthread on macOS/iOS, and to a dedicated win7 backend on Win7.
    target.declare_struct("sys::sync::mutex::Mutex");
    // The futex Mutex (the active backend on Windows non-win7, Linux, Android,
    // FreeBSD, etc.) stores its state through two file-local private type
    // aliases. Declaring them mirrors the aliases and routes their RHS
    // (`futex::SmallFutex` / `futex::SmallPrimitive`, both public) through the
    // registry, satisfying the field-publicity check. Gated to exactly the
    // platforms where std's cfg_select picks the futex module.
    const FUTEX_ACTIVE: &str = concat!(
        "any(",
        "all(target_os = \"windows\", not(target_vendor = \"win7\")), ",
        "target_os = \"linux\", ",
        "target_os = \"android\", ",
        "target_os = \"freebsd\", ",
        "target_os = \"openbsd\", ",
        "target_os = \"motor\", ",
        "target_os = \"dragonfly\", ",
        "target_os = \"hermit\", ",
        "all(target_family = \"wasm\", target_feature = \"atomics\")",
        ")",
    );
    target.declare_struct_cfg("sys::sync::mutex::futex::Mutex", FUTEX_ACTIVE);
    target.declare_struct_cfg("sys::sync::mutex::futex::Futex", FUTEX_ACTIVE);
    target.declare_struct_cfg("sys::sync::mutex::futex::State", FUTEX_ACTIVE);
    // Note: SmallFutex/SmallPrimitive type aliases are no longer declared.
    // With the doc-JSON approach, field types are already fully resolved by
    // the compiler (e.g., the futex Mutex's field is directly AtomicU32, not
    // SmallFutex), so these aliases don't need separate binding files.
    // The lazy-allocation helper used by the pthread backend (active on
    // macOS/iOS). Mirrored so the polyfill can interact with its pointer slot.
    target.declare_struct("sys::sync::once_box::OnceBox");

    // The public `std::sync::Mutex<T>` (poison variant) and its poison flag.
    // Mirrored as a generic layout template so the fallible `TryMutex` polyfill
    // can construct the struct field-by-field with an uninitialised data slot.
    target.declare_struct("sync::poison::mutex::Mutex");
    target.declare_struct("sync::poison::Flag");

    target
}

fn core_target() -> BindingTarget {
    let mut target = BindingTarget::new("core");

    // Replace Unique with NonNull: both are thin pointer wrappers with identical
    // layout. core::ptr::unique::Unique is private/unstable; the public re-export
    // is core::ptr::NonNull. Generic args are preserved by the emitter.
    target.replace_path(
        "core::ptr::Unique",
        "::__rustyfill_builtin_core::ptr::NonNull",
    );

    // ── Known external types (recognized at their canonical location) ───────
    // The futex Mutex and OnceBox reference `Atomic<T>` via
    // `use crate::sync::atomic::{... Atomic ...}`. In std that resolves through
    // `pub use core::sync::atomic`, so the canonical home is
    // `core::sync::atomic::Atomic`. The real definition is
    // `#[unstable(feature = "generic_atomic")]` and holds `UnsafeCell<T::Storage>`
    // behind an `AtomicPrimitive` bound, which won't compile in our no_std
    // downstream tree. Rather than float it as a bare prelude name, we recognize
    // it at its original location: register it under `core::sync::atomic::Atomic`
    // so references route there, and emit a stub body (a transparent
    // `UnsafeCell<T>` wrapper) in place of the parsed source. Only the type
    // shape matters for the bindings; atomic ops come from the main crate.
    // The inner field is public so downstream polyfills (e.g. `TryMutex`) can
    // construct and inspect the atomic word directly through its `UnsafeCell`,
    // matching the byte layout std's real `Atomic` would have. A `new` helper
    // mirrors std's constructor for ergonomic in-place initialisation.
    target.add_known_type(
        "sync::atomic::Atomic",
        concat!(
            "#[repr(transparent)]\n",
            "pub struct Atomic<T> {\n",
            "    pub inner: ::__rustyfill_builtin_core::cell::UnsafeCell<T>,\n",
            "}\n",
            "impl<T> Atomic<T> {\n",
            "    #[inline]\n",
            "    pub const fn new(v: T) -> Self {\n",
            "        Self { inner: ::__rustyfill_builtin_core::cell::UnsafeCell::new(v) }\n",
            "    }\n",
            "    #[inline]\n",
            "    pub const unsafe fn assume_init(&self) -> &T {\n",
            "        unsafe { &*self.inner.get() }\n",
            "    }\n",
            "    #[inline]\n",
            "    pub const unsafe fn assume_init_mut(&mut self) -> &mut T {\n",
            "        unsafe { &mut *self.inner.get() }\n",
            "    }\n",
            "}",
        ),
    );

    // AtomicBool: transparent wrapper over a single bool word. Referenced bare
    // in sync/poison.rs (`pub failed: AtomicBool`). Registered at its canonical
    // core location so the emitter routes the bare name to this stub.
    target.add_known_type(
        "sync::atomic::AtomicBool",
        concat!(
            "#[repr(transparent)]\n",
            "pub struct AtomicBool(::__rustyfill_builtin_core::cell::UnsafeCell<bool>);\n",
            "impl AtomicBool {\n",
            "    #[inline]\n",
            "    pub const fn new(v: bool) -> Self {\n",
            "        Self(::__rustyfill_builtin_core::cell::UnsafeCell::new(v))\n",
            "    }\n",
            "}",
        ),
    );

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

    // ── Linked list ─────────────────────────────────────────────────────────
    // The container shell and its private Node type. The fallible push methods
    // allocate Node<T> directly via TryBox and splice it into the list through
    // raw pointer surgery on the mirrored fields (head/tail/len), so both must
    // be mirrored for compile-time layout enforcement.
    target.declare_struct("collections::linked_list::LinkedList");
    target.declare_struct("collections::linked_list::Node");

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
    // Constants used by the main crate's btree operations.
    target.declare_const("collections::btree::node::CAPACITY");
    target.declare_const("collections::btree::node::KV_IDX_CENTER");
    target.declare_const("collections::btree::node::EDGE_IDX_LEFT_OF_CENTER");
    target.declare_const("collections::btree::node::EDGE_IDX_RIGHT_OF_CENTER");
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

    // Layout is defined in core::alloc but re-exported through alloc.
    // Route references to the core builtin.
    target.replace_path(
        "alloc::alloc::Layout",
        "::__rustyfill_builtin_core::alloc::Layout",
    );

    // BoxedArrayIntoIter requires unstable allocator_api feature and references
    // vec::IntoIter which needs Allocator bounds we can't satisfy in no_std.
    target.ignore_struct("boxed::iter::BoxedArrayIntoIter");

    target
}
