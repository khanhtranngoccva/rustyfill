//! Integration tests driving individual pipeline phases against the real
//! rust-src tree and asserting on intermediate results.
//!
//! These tests catch regressions at phase boundaries: if discovery fails to
//! register a file, or emission drops a module from the manifest, the failing
//! assertion names the exact phase and the missing artifact — no need to
//! reverse-engineer from generated output.
//!
//! Requires `rust-src` (the Rust standard-library source). Skips gracefully
//! when unavailable.

use std::path::PathBuf;

use rustyfill_sys_bindings::{
    BindingTarget, CfgContext, LoaderSpec, PipelineState, run_discovery_phase, run_emit_phase,
    run_manifest_phase, run_registry_phase,
};

/// Locate the rust-src library directory (contains `core/`, `alloc/`, `std/`).
fn find_rust_src() -> Option<PathBuf> {
    let out = std::process::Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sysroot = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let candidate = PathBuf::from(sysroot).join("lib").join("rustlib").join("src").join("rust").join("library");
    candidate.exists().then_some(candidate)
}

/// Build the same spec the production build script uses (mirrors
/// `rustyfill-sys/build/spec.rs`). Kept in sync manually; if the two drift,
/// these tests may assert on a different surface than the real build.
fn test_spec() -> LoaderSpec {
    let mut spec = LoaderSpec::new();

    // ── std ────────────────────────────────────────────────────────────────
    let mut std_t = BindingTarget::new("std");
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
        ")"
    );
    std_t.declare_struct("sys::sync::mutex::Mutex");
    std_t.declare_struct_cfg("sys::sync::mutex::futex::Mutex", FUTEX_ACTIVE);
    std_t.declare_struct_cfg("sys::sync::mutex::futex::Futex", FUTEX_ACTIVE);
    std_t.declare_struct_cfg("sys::sync::mutex::futex::State", FUTEX_ACTIVE);
    std_t.declare_struct_cfg("sys::sync::mutex::futex::futex::SmallFutex", FUTEX_ACTIVE);
    std_t.declare_struct_cfg(
        "sys::sync::mutex::futex::futex::SmallPrimitive",
        FUTEX_ACTIVE,
    );
    std_t.declare_struct("sys::sync::once_box::OnceBox");
    std_t.declare_struct("sync::poison::mutex::Mutex");
    std_t.declare_struct("sync::poison::Flag");
    spec.add_target(std_t);

    // ── core ───────────────────────────────────────────────────────────────
    let mut core_t = BindingTarget::new("core");
    core_t.replace_path("core::ptr::Unique", "NonNull");
    core_t.add_known_type(
        "sync::atomic::Atomic",
        "#[repr(transparent)]\npub struct Atomic<T> {\n    pub inner: ::__rustyfill_builtin_core::cell::UnsafeCell<T>,\n}\nimpl<T> Atomic<T> {\n    #[inline]\n    pub const fn new(v: T) -> Self {\n        Self { inner: ::__rustyfill_builtin_core::cell::UnsafeCell::new(v) }\n    }\n}",
    );
    core_t.add_known_type(
        "sync::atomic::AtomicBool",
        "#[repr(transparent)]\npub struct AtomicBool(::__rustyfill_builtin_core::cell::UnsafeCell<bool>);\nimpl AtomicBool {\n    #[inline]\n    pub const fn new(v: bool) -> Self {\n        Self(::__rustyfill_builtin_core::cell::UnsafeCell::new(v))\n    }\n}",
    );
    spec.add_target(core_t);

    // ── alloc ──────────────────────────────────────────────────────────────
    let mut alloc_t = BindingTarget::new("alloc");
    alloc_t.declare_struct("collections::TryReserveError");
    alloc_t.declare_struct("collections::TryReserveErrorKind");
    for derive in ["Clone", "PartialEq", "Eq", "Debug"] {
        alloc_t.add_derive("collections::TryReserveErrorKind", derive);
    }
    alloc_t.declare_struct("collections::linked_list::LinkedList");
    alloc_t.declare_struct("collections::linked_list::Node");
    alloc_t.declare_struct("collections::btree::map::BTreeMap");
    alloc_t.declare_struct("collections::btree::set::BTreeSet");
    alloc_t.declare_struct("collections::btree::map::entry::Entry");
    alloc_t.declare_struct("collections::btree::map::entry::VacantEntry");
    alloc_t.declare_struct("collections::btree::map::entry::OccupiedEntry");
    alloc_t.declare_struct("collections::btree::map::entry::OccupiedError");
    alloc_t.declare_struct("collections::btree::set::entry::Entry");
    alloc_t.declare_struct("collections::btree::set::entry::OccupiedEntry");
    alloc_t.declare_struct("collections::btree::set::entry::VacantEntry");
    alloc_t.declare_struct("collections::btree::node::LeafNode");
    alloc_t.declare_struct("collections::btree::node::BoxedNode");
    alloc_t.declare_struct("collections::btree::node::InternalNode");
    alloc_t.declare_struct("collections::btree::node::NodeRef");
    alloc_t.declare_struct("collections::btree::node::Root");
    alloc_t.declare_struct("collections::btree::node::Handle");
    alloc_t.declare_struct("collections::btree::node::LeftOrRight");
    alloc_t.declare_struct("collections::btree::node::BalancingContext");
    alloc_t.declare_struct("collections::btree::node::ForceResult");
    alloc_t.declare_struct("collections::btree::node::SplitResult");
    alloc_t.declare_struct("collections::btree::node::marker::Leaf");
    alloc_t.declare_struct("collections::btree::node::marker::Internal");
    alloc_t.declare_struct("collections::btree::node::marker::LeafOrInternal");
    alloc_t.declare_struct("collections::btree::node::marker::Owned");
    alloc_t.declare_struct("collections::btree::node::marker::Dying");
    alloc_t.declare_struct("collections::btree::node::marker::DormantMut");
    alloc_t.declare_struct("collections::btree::node::marker::Immut");
    alloc_t.declare_struct("collections::btree::node::marker::Mut");
    alloc_t.declare_struct("collections::btree::node::marker::ValMut");
    alloc_t.declare_struct("collections::btree::node::marker::KV");
    alloc_t.declare_struct("collections::btree::node::marker::Edge");
    alloc_t.declare_struct("collections::btree::borrow::DormantMutRef");
    alloc_t.declare_struct("collections::btree::set_val::SetValZST");
    alloc_t.declare_struct("boxed::Box");
    alloc_t.ignore_path("core::alloc::Allocator");
    alloc_t.replace_path("alloc::alloc::Global", "()");
    alloc_t.ignore_struct("boxed::iter::BoxedArrayIntoIter");
    spec.add_target(alloc_t);

    spec
}

/// Linux x86_64 cfg context (matches the CI/dev environment).
fn linux_cfg() -> CfgContext {
    CfgContext::from_target_triple("x86_64-unknown-linux-gnu")
}

#[test]
fn phase1_discovery_registers_all_declared_module_trees() {
    let Some(rust_src) = find_rust_src() else {
        eprintln!("SKIP: rust-src not available");
        return;
    };
    let out_dir = tempfile::tempdir().expect("tempdir");
    let spec = test_spec();
    let cfg = linux_cfg();

    let mut state = PipelineState::for_test(&rust_src, out_dir.path(), &spec, &cfg);

    let result = run_discovery_phase(&mut state)
        .expect("discovery should succeed against a valid spec");

    // Every declared leaf module must be present in the parsed cache.
    let expected_files = [
        "collections/btree/map.rs",
        "collections/btree/set.rs",
        "collections/btree/map/entry.rs",
        "collections/btree/set/entry.rs",
        "collections/btree/node.rs",
        "collections/btree/borrow.rs",
        "collections/btree/set_val.rs",
        "collections/linked_list.rs",
        "collections/mod.rs",
        "boxed.rs",
        "sys/sync/mutex/futex.rs",
        "sys/sync/once_box.rs",
        "sync/poison/mutex.rs",
    ];
    for f in &expected_files {
        assert!(
            result.discovered_files.iter().any(|d| d == f),
            "Phase 1 missed `{f}`. Discovered {} files total; first 20: {:?}",
            result.discovered_files.len(),
            &result.discovered_files[..result.discovered_files.len().min(20)]
        );
    }

    // Sanity: discovery should find a meaningful number of files.
    assert!(
        result.discovered_files.len() >= 30,
        "Only {} files discovered; expected at least 30",
        result.discovered_files.len()
    );
}

#[test]
fn phase2_registry_populates_model_and_declares_paths() {
    let Some(rust_src) = find_rust_src() else {
        eprintln!("SKIP: rust-src not available");
        return;
    };
    let out_dir = tempfile::tempdir().expect("tempdir");
    let spec = test_spec();
    let cfg = linux_cfg();

    let mut state = PipelineState::for_test(&rust_src, out_dir.path(), &spec, &cfg);
    let disc = run_discovery_phase(&mut state).expect("discovery failed");
    let reg = run_registry_phase(&mut state, &disc);

    // Field publicity check must pass (it gates the whole build).
    assert!(
        reg.field_errors.is_empty(),
        "Field publicity errors after registry phase:\n{}",
        reg.field_errors.join("\n")
    );

    // The model must have registered a substantial set of types.
    assert!(
        reg.types_registered >= 50,
        "Only {} types registered; expected at least 50",
        reg.types_registered
    );

    // Declared paths must include every spec declaration.
    let declared = state.model.declared_paths();
    for needle in [
        "::alloc::collections::btree::map::BTreeMap",
        "::alloc::collections::btree::set::BTreeSet",
        "::alloc::boxed::Box",
        "::std::sys::sync::mutex::Mutex",
    ] {
        assert!(
            declared.iter().any(|p| p == needle || p.ends_with(&needle[2..])),
            "Declared paths missing `{needle}`. Sample: {:?}",
            declared.iter().take(10).collect::<Vec<_>>()
        );
    }
}

#[test]
fn phase3_emit_writes_binding_files_for_every_declared_module() {
    let Some(rust_src) = find_rust_src() else {
        eprintln!("SKIP: rust-src not available");
        return;
    };
    let out_dir = tempfile::tempdir().expect("tempdir");
    let spec = test_spec();
    let cfg = linux_cfg();

    let mut state = PipelineState::for_test(&rust_src, out_dir.path(), &spec, &cfg);
    let disc = run_discovery_phase(&mut state).expect("discovery failed");
    let reg = run_registry_phase(&mut state, &disc);
    assert!(reg.field_errors.is_empty(), "field errors: {:?}", reg.field_errors);
    let emit = run_emit_phase(&mut state);

    // Every file in all_files that is marked emitted must exist on disk.
    let mut checked = 0;
    for (rel_path, _lib) in &emit.all_files_snapshot {
        if emit.emitted_file_set.contains(rel_path) {
            let abs = out_dir.path().join(rel_path);
            assert!(
                abs.exists(),
                "Model marks `{rel_path}` as emitted but the file is missing from out_dir"
            );
            checked += 1;
        }
    }
    assert!(checked >= 10, "Only {checked} emitted files verified; expected at least 10");

    // Specific critical files must be in the emitted set.
    for required in [
        "collections/btree/map.rs",
        "collections/btree/set.rs",
        "collections/btree/map/entry.rs",
        "collections/btree/set/entry.rs",
        "collections/btree/borrow.rs",
        "collections/btree/set_val.rs",
        "boxed.rs",
    ] {
        assert!(
            emit.emitted_file_set.contains(required),
            "`{required}` not in emitted_file_set. Set has {} entries: {:?}",
            emit.emitted_file_set.len(),
            emit.emitted_file_set.iter().take(20).collect::<Vec<_>>()
        );
    }
}

#[test]
fn phase4_manifest_lists_every_emitted_module() {
    let Some(rust_src) = find_rust_src() else {
        eprintln!("SKIP: rust-src not available");
        return;
    };
    let out_dir = tempfile::tempdir().expect("tempdir");
    let spec = test_spec();
    let cfg = linux_cfg();

    let mut state = PipelineState::for_test(&rust_src, out_dir.path(), &spec, &cfg);
    let disc = run_discovery_phase(&mut state).expect("discovery failed");
    let reg = run_registry_phase(&mut state, &disc);
    assert!(reg.field_errors.is_empty(), "field errors: {:?}", reg.field_errors);
    let emit = run_emit_phase(&mut state);
    let manifest = run_manifest_phase(&mut state, &emit);

    // No validation errors.
    assert!(
        manifest.errors.is_empty(),
        "Manifest phase validation errors:\n{}",
        manifest.errors.join("\n")
    );

    // Manifest file exists and declares the key modules.
    let manifest_path = out_dir.path().join("bindings_generated.rs");
    assert!(manifest_path.exists(), "bindings_generated.rs was not written");
    let content = std::fs::read_to_string(&manifest_path).expect("read manifest");
    for module in ["pub mod btree", "pub mod boxed", "pub mod linked_list", "pub mod mutex"] {
        assert!(
            content.contains(module),
            "Manifest missing `{module}`. First 500 chars: {}",
            &content[..content.len().min(500)]
        );
    }
}

#[test]
fn full_pipeline_produces_consistent_intermediate_states() {
    let Some(rust_src) = find_rust_src() else {
        eprintln!("SKIP: rust-src not available");
        return;
    };
    let out_dir = tempfile::tempdir().expect("tempdir");
    let spec = test_spec();
    let cfg = linux_cfg();

    let mut state = PipelineState::for_test(&rust_src, out_dir.path(), &spec, &cfg);

    let disc = run_discovery_phase(&mut state).expect("phase 1 failed");
    let files_after_disc = state.cached_file_count();
    assert_eq!(
        disc.discovered_files.len(),
        files_after_disc,
        "Discovery result count ({}) != state cache size ({})",
        disc.discovered_files.len(),
        files_after_disc
    );

    let reg = run_registry_phase(&mut state, &disc);
    assert!(reg.field_errors.is_empty());

    let emit = run_emit_phase(&mut state);
    // Emitted set must be a subset of all_files.
    let all_rel: std::collections::HashSet<&str> =
        emit.all_files_snapshot.iter().map(|(f, _)| f.as_str()).collect();
    for fp in &emit.emitted_file_set {
        assert!(
            all_rel.contains(fp.as_str()),
            "Emitted file `{fp}` not present in all_files snapshot"
        );
    }

    let manifest = run_manifest_phase(&mut state, &emit);
    assert!(manifest.errors.is_empty(), "errors: {:?}", manifest.errors);
}
