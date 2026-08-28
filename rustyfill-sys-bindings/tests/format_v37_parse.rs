//! Regression test: parse real Rust 1.85 rustdoc JSON (format_version 37)
//! against the typed `Crate` wire types in `wire::original`.
//!
//! The fixtures are cached doc-JSON blobs produced by the pinned toolchain
//! (rustc 1.85.1, see `rust-toolchain.toml`). If no cache is present, the
//! test generates them via the same driver the build uses — which requires
//! the `rust-src` component on the active toolchain.

use rustyfill_sys_bindings::docjson::driver::{self, DocGenConfig};
use rustyfill_sys_bindings::docjson::wire::original::Crate;
use std::path::{Path, PathBuf};

/// Locate (or generate) the cached doc-JSON for `lib_name` and return its path.
fn fixture_path(lib_name: &str) -> PathBuf {
    // Cache layout: ~/.cache/rustyfill/docjson-<hash>/<lib>.json
    let home = std::env::var("HOME").unwrap_or_default();
    let cache_root = Path::new(&home).join(".cache").join("rustyfill");
    if let Ok(entries) = std::fs::read_dir(&cache_root) {
        let mut candidates: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_dir()
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("docjson-"))
            })
            .collect();
        candidates.sort();
        for dir in candidates.into_iter().rev() {
            let candidate = dir.join(format!("{lib_name}.json"));
            if candidate.exists() {
                return candidate;
            }
        }
    }

    // No cache hit: generate with the active toolchain.
    let config = DocGenConfig::current();
    let output = driver::generate(&config).unwrap_or_else(|errs| {
        panic!(
            "no cached doc-JSON found under {} and generation failed:\n  {}",
            cache_root.display(),
            errs.join("\n  ")
        )
    });
    // Persist so subsequent runs are fast.
    let target = cache_root.join("docjson-generated");
    std::fs::create_dir_all(&target).expect("create cache dir");
    for (name, value) in &output.data {
        let file = target.join(format!("{name}.json"));
        std::fs::write(&file, serde_json::to_vec(value).unwrap()).expect("write fixture");
    }
    let file = target.join(format!("{lib_name}.json"));
    assert!(file.exists(), "generated output missing {lib_name}");
    file
}

/// Parse the full crate blob into the typed wire model and sanity-check it.
fn parse_fixture(lib_name: &str) -> Crate {
    let path = fixture_path(lib_name);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let value: serde_json::Value = serde_json::from_str(&raw).expect("fixture is valid JSON");

    let format_version = value["format_version"]
        .as_u64()
        .expect("missing format_version");
    assert_eq!(
        format_version, 37,
        "{lib_name}: expected format_version 37 from rustc 1.85.x, got {format_version}"
    );

    let crate_: Crate = serde_json::from_value(value).unwrap_or_else(|e| {
        panic!(
            "failed to parse {lib_name} ({}) as wire::original::Crate: {e}",
            path.display()
        )
    });

    // Basic structural sanity.
    assert!(!crate_.index.is_empty(), "{lib_name}: empty item index");
    // The root must exist in the index and be the crate-root module.
    assert!(
        crate_.index.contains_key(&crate_.root),
        "{lib_name}: root {:?} missing from index",
        crate_.root
    );
    assert!(
        matches!(
            crate_.index[&crate_.root].inner,
            rustyfill_sys_bindings::docjson::wire::original::ItemEnum::Module(ref m) if m.is_crate
        ),
        "{lib_name}: root item is not the crate-root module"
    );
    // Local path entries must resolve to indexed items; entries with a foreign
    // `crate_id` are re-exported items from other crates and legitimately have
    // no local index entry.
    for (id, summary) in &crate_.paths {
        if summary.crate_id == 0 {
            assert!(
                crate_.index.contains_key(id),
                "{lib_name}: local path entry {:?} has no matching index item",
                id
            );
        }
    }
    // The index also embeds a handful of external items (traits/methods from other
    // crates that are referenced locally); every such entry must resolve to a known
    // external crate.
    for (id, item) in &crate_.index {
        if item.crate_id != 0 {
            assert!(
                crate_.external_crates.contains_key(&item.crate_id),
                "{lib_name}: item {:?} references unknown crate_id {}",
                id,
                item.crate_id
            );
        }
    }
    crate_
}

#[test]
fn parses_alloc_format_v37() {
    let crate_ = parse_fixture("alloc");
    // Spot-check a well-known type survived the round trip.
    let vec_item = crate_
        .index
        .values()
        .find(|item| item.name.as_deref() == Some("Vec"))
        .expect("Vec not found in alloc index");
    assert!(matches!(
        vec_item.inner,
        rustyfill_sys_bindings::docjson::wire::original::ItemEnum::Struct(_)
    ));
}

#[test]
fn parses_core_format_v37() {
    let crate_ = parse_fixture("core");
    let option = crate_
        .index
        .values()
        .find(|item| item.name.as_deref() == Some("Option"))
        .expect("Option not found in core index");
    assert!(matches!(
        option.inner,
        rustyfill_sys_bindings::docjson::wire::original::ItemEnum::Enum(_)
    ));
}

#[test]
fn parses_std_format_v37() {
    let crate_ = parse_fixture("std");
    assert!(
        !crate_.external_crates.is_empty(),
        "std must reference external crates (core, alloc)"
    );
    // `String` lives in `alloc`, so in the std blob it only appears as a re-export:
    // a path entry pointing at an external crate_id, not a local index item.
    let string_summary = crate_
        .paths
        .values()
        .find(|s| s.path.last().is_some_and(|seg| seg == "String"))
        .expect("String path entry not found in std paths");
    assert_ne!(
        string_summary.crate_id, 0,
        "String should be re-exported from alloc, not defined in std"
    );
    assert!(
        crate_
            .external_crates
            .contains_key(&string_summary.crate_id),
        "String's defining crate must be a known external crate"
    );
    // Spot-check a type actually defined in std instead.
    let vec_map = crate_
        .index
        .values()
        .find(|item| item.name.as_deref() == Some("HashMap"))
        .expect("HashMap not found in std index");
    assert!(matches!(
        vec_map.inner,
        rustyfill_sys_bindings::docjson::wire::original::ItemEnum::Struct(_)
    ));
}
