//! Cargo-doc JSON based type extraction.
//!
//! Replaces the brittle source-parsing pipeline with a toolchain-driven
//! approach: invoke `cargo doc` with `RUSTDOCFLAGS="-Zunstable-options
//! --output-format=json --document-private-items"` inside the rust-src tree,
//! then extract authoritative type definitions from the structured JSON output.
//!
//! This eliminates all the problems with source parsing:
//! - `cfg_if!` / `cfg_select!` macros are already resolved by the compiler
//! - Import following is unnecessary (resolved paths are in the JSON)
//! - Private fields are included via `--document-private-items`
//! - Type representations are fully resolved (no ambiguous names)
//!
//! # Design
//!
//! We use `serde_json::Value` for parsing rather than the `rustdoc-types` crate
//! because the latter requires exact format_version matching. Since the user's
//! toolchain determines both the compiler and the JSON schema version, pinning
//! to one `rustdoc-types` release would break on any toolchain update. The
//! Value-based approach is flexible across versions at negligible cost for a
//! build-time tool that runs once and caches.

pub mod driver;
pub mod emit;
pub mod model;
pub mod type_repr;

use std::collections::HashMap;

use crate::loader_spec::LoaderSpec;

/// Extract the types declared in the spec from pre-loaded doc-JSON data.
///
/// `json_data` maps library name → parsed top-level JSON object.
/// Returns one [`model::DocType`] per successfully located declaration.
///
/// Missing declarations are diagnosed, not swallowed:
/// - An **unconditional** declaration that isn't found is a hard error — it
///   means the spec is out of date with this toolchain's std (renamed or
///   moved type), and failing loudly here beats a confusing compile error
///   deep in the downstream polyfill.
/// - A **cfg-gated** declaration that isn't found is expected when its
///   predicate didn't activate for the current target (the compiler excluded
///   it). Those are skipped with a note on stderr, never an error.
pub fn extract_types(
    json_data: &HashMap<String, serde_json::Value>,
    spec: &LoaderSpec,
) -> Result<Vec<model::DocType>, Vec<String>> {
    let mut results = Vec::new();
    let mut errors = Vec::new();

    for target in &spec.targets {
        let Some(data) = json_data.get(&target.lib_name) else {
            errors.push(format!(
                "No doc-JSON data available for library '{}'",
                target.lib_name
            ));
            continue;
        };

        // Split declarations by gating so missing ones can be diagnosed
        // appropriately. Unconditional set first, then cfg-gated paths.
        let unconditional: Vec<&String> = target.declared_structs.iter().collect();
        let gated_paths: Vec<&String> = target.cfg_gated_decls.iter().map(|g| &g.path).collect();

        for decl in &unconditional {
            match locate_and_extract(data, decl, &target.lib_name) {
                Ok(mut doc_type) => {
                    push_with_module_path(&mut results, &mut doc_type, decl);
                }
                Err(e) => {
                    errors.push(format!(
                        "[{}] declared type '{}' not found in this toolchain's std \
                         ({}) — the standard library may have renamed or moved it",
                        target.lib_name, decl, e
                    ));
                }
            }
        }

        for decl in &gated_paths {
            match locate_and_extract(data, decl, &target.lib_name) {
                Ok(mut doc_type) => {
                    push_with_module_path(&mut results, &mut doc_type, decl);
                }
                Err(_) => {
                    // Expected: the cfg predicate didn't activate for this
                    // target, so the compiler excluded the item. Note it so
                    // a genuine rename still leaves a trace in build logs.
                    eprintln!(
                        "cargo:warning=rustyfill-sys: cfg-gated declaration \
                          [{}::{}] not present in doc-JSON (predicate inactive \
                          for this target); skipping",
                        target.lib_name, decl
                    );
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(results)
    } else {
        Err(errors)
    }
}

/// Attach the module path derived from the spec declaration (everything
/// except the leaf type name) and append to the results.
fn push_with_module_path(
    results: &mut Vec<model::DocType>,
    doc_type: &mut model::DocType,
    decl: &str,
) {
    let segments: Vec<&str> = decl.split("::").collect();
    let module_path = if segments.len() > 1 {
        segments[..segments.len() - 1].join("::")
    } else {
        String::new()
    };
    doc_type.set_module_path(module_path);
    results.push(doc_type.clone());
}

/// Locate a type by its spec path (e.g., `"collections::btree::map::BTreeMap"`)
/// within a single crate's doc-JSON and convert it to our model.
///
/// Handles three lookup strategies:
/// 1. Direct path match in the paths table (works for most types).
/// 2. Re-export resolution: if the exact path isn't found, look up the parent
///    module in the paths table, scan its items for a `use` entry with the
///    matching name, and follow the re-exported id to the actual definition.
///    This handles `cfg_select!`-resolved paths like `sys::sync::mutex::Mutex`.
/// 3. Module alias traversal: if an intermediate path segment refers to a
///    module imported via `use ... {self}`, follow that import to the actual
///    module and continue the search there. This handles paths like
///    `sys::sync::mutex::futex::futex::SmallFutex` where the second `futex`
///    is a `use crate::sys::pal::unix::futex as futex` alias.
fn locate_and_extract(
    data: &serde_json::Value,
    spec_path: &str,
    lib_name: &str,
) -> Result<model::DocType, String> {
    let index = data
        .get("index")
        .and_then(|v| v.as_object())
        .ok_or("missing 'index' in doc JSON")?;

    let paths_table = data
        .get("paths")
        .and_then(|v| v.as_object())
        .ok_or("missing 'paths' in doc JSON")?;

    // One-time O(N) scan of the paths table into hash indexes; every lookup
    // below is then O(segments) instead of O(table size).
    let idx = PathIndex::build(paths_table);

    // Build the expected full path: [lib_name, seg1, seg2, ..., TypeName]
    let target_full_path: Vec<&str> = std::iter::once(lib_name)
        .chain(spec_path.split("::"))
        .collect();

    let item_id = resolve_type_id(index, &idx, &target_full_path)?;

    let item = index
        .get(&item_id.to_string())
        .ok_or_else(|| format!("item id {} not in index", item_id))?;

    model::DocType::from_json(item, index, lib_name)
}

/// Hash indexes over a single library's `paths` table, built once per lookup
/// session. Keys are the joined path segments (`core::cell::UnsafeCell`).
struct PathIndex {
    /// Exact path → item id, restricted to type-ish kinds.
    types: HashMap<String, u64>,
    /// Exact path → item id, restricted to modules.
    modules: HashMap<String, u64>,
}

impl PathIndex {
    fn build(paths_table: &serde_json::Map<String, serde_json::Value>) -> Self {
        let mut types = HashMap::new();
        let mut modules = HashMap::new();

        for (id_str, path_entry) in paths_table {
            let Ok(id) = id_str.parse::<u64>() else {
                continue;
            };
            let path_arr = match path_entry.get("path").and_then(|v| v.as_array()) {
                Some(a) => a,
                None => continue,
            };
            let key: String = path_arr
                .iter()
                .filter_map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("::");
            if key.is_empty() {
                continue;
            }
            let kind = path_entry
                .get("kind")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            match kind {
                "struct" | "enum" | "union" | "type_alias" | "constant" => {
                    types.entry(key).or_insert(id);
                }
                "module" => {
                    modules.entry(key).or_insert(id);
                }
                _ => {}
            }
        }

        Self { types, modules }
    }

    fn find_type(&self, path: &[&str]) -> Option<u64> {
        self.types.get(path.join("::").as_str()).copied()
    }

    fn find_module(&self, path: &[&str]) -> Option<u64> {
        self.modules.get(path.join("::").as_str()).copied()
    }
}

/// Resolve a full path to an item ID, handling re-exports and module aliases.
fn resolve_type_id(
    index: &serde_json::Map<String, serde_json::Value>,
    idx: &PathIndex,
    full_path: &[&str],
) -> Result<u64, String> {
    // Strategy 1: direct path match.
    if let Some(id) = idx.find_type(full_path) {
        return Ok(id);
    }

    // Strategy 2+3: walk the path segment by segment, following modules and
    // resolving re-exports / module aliases along the way.
    resolve_by_walking(index, idx, full_path)
}

/// Walk path segments through the module tree, following re-exports and
/// module aliases. Navigates from the deepest known module prefix.
fn resolve_by_walking(
    index: &serde_json::Map<String, serde_json::Value>,
    idx: &PathIndex,
    full_path: &[&str],
) -> Result<u64, String> {
    // Find the deepest prefix that exists as a module in the paths table,
    // then navigate the remaining segments from there.
    let mut start_idx = 0;
    let mut current_mod_id: Option<u64> = None;

    for prefix_len in (2..full_path.len()).rev() {
        if let Some(mod_id) = idx.find_module(&full_path[..prefix_len]) {
            start_idx = prefix_len;
            current_mod_id = Some(mod_id);
            break;
        }
    }

    let Some(mut mod_id) = current_mod_id else {
        return Err(format!(
            "cannot resolve path {:?}: no known module prefix found",
            full_path
        ));
    };

    // Navigate remaining segments.
    let remaining = &full_path[start_idx..];
    for (i, segment) in remaining.iter().enumerate() {
        let is_last = i == remaining.len() - 1;

        if is_last {
            // Last segment: look for the type (struct/enum/union/type_alias)
            // or a re-export of it in the current module.
            let type_id = find_type_in_module(index, mod_id, segment).ok_or_else(|| {
                format!(
                    "type '{}' not found in module (resolving {:?})",
                    segment, full_path
                )
            })?;
            return Ok(type_id);
        } else {
            // Intermediate segment: must be a submodule or module alias.
            mod_id = find_submodule_in_module(index, mod_id, segment).ok_or_else(|| {
                format!(
                    "submodule '{}' not found (resolving {:?})",
                    segment, full_path
                )
            })?;
        }
    }

    // Shouldn't reach here (last segment always returns).
    Err(format!(
        "unreachable: path {:?} exhausted without finding type",
        full_path
    ))
}

/// Find a type (struct/enum/union/type_alias) in a module's items, either
/// directly or through a `use` re-export.
fn find_type_in_module(
    index: &serde_json::Map<String, serde_json::Value>,
    mod_id: u64,
    type_name: &str,
) -> Option<u64> {
    let mod_item = index.get(&mod_id.to_string())?;
    let items = mod_item.pointer("/inner/module/items")?.as_array()?;

    for item_val in items {
        let iid = item_val.as_u64()?;
        let item = index.get(&iid.to_string())?;

        // Direct match: item has the right name and is a type.
        if item.get("name").and_then(|v| v.as_str()) == Some(type_name) {
            let kind = item.get("kind").and_then(|v| v.as_str()).unwrap_or("");
            if matches!(kind, "struct" | "enum" | "union" | "type_alias") {
                return Some(iid);
            }
            // Also check inner kind (some items have null top-level kind).
            let inner = item.get("inner");
            if let Some(inner_obj) = inner.and_then(|v| v.as_object()) {
                if inner_obj
                    .keys()
                    .any(|k| matches!(k.as_str(), "struct" | "enum" | "union" | "type_alias"))
                {
                    return Some(iid);
                }
            }
        }

        // Re-export: `use` entry with matching name pointing to a type.
        if let Some(use_inner) = item.pointer("/inner/use") {
            let use_name = use_inner.get("name").and_then(|v| v.as_str()).unwrap_or("");
            if use_name == type_name {
                let target_id = use_inner.get("id").and_then(|v| v.as_u64())?;
                return Some(target_id);
            }
        }
    }
    None
}

/// Find a submodule (or module alias via `use ... {self}`) in a module's items.
fn find_submodule_in_module(
    index: &serde_json::Map<String, serde_json::Value>,
    mod_id: u64,
    sub_name: &str,
) -> Option<u64> {
    let mod_item = index.get(&mod_id.to_string())?;
    let items = mod_item.pointer("/inner/module/items")?.as_array()?;

    for item_val in items {
        let iid = item_val.as_u64()?;
        let item = index.get(&iid.to_string())?;

        // Direct submodule: item is a module with matching name.
        if item.get("name").and_then(|v| v.as_str()) == Some(sub_name) {
            let inner = item.get("inner");
            if let Some(inner_obj) = inner.and_then(|v| v.as_object()) {
                if inner_obj.contains_key("module") {
                    return Some(iid);
                }
            }
        }

        // Module alias: `use` entry importing a module (`{self}`).
        if let Some(use_inner) = item.pointer("/inner/use") {
            let use_name = use_inner.get("name").and_then(|v| v.as_str()).unwrap_or("");
            if use_name == sub_name {
                let target_id = use_inner.get("id").and_then(|v| v.as_u64())?;
                // Verify the target is actually a module.
                let target_item = index.get(&target_id.to_string())?;
                let inner = target_item.get("inner");
                if let Some(inner_obj) = inner.and_then(|v| v.as_object()) {
                    if inner_obj.contains_key("module") {
                        return Some(target_id);
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate doc-JSON using the active toolchain (the same one compiling
    /// this test binary) and verify extraction works for known types.
    ///
    /// Requires the `rust-src` component to be installed on the active
    /// toolchain. Skips gracefully if it is not available.
    #[test]
    fn test_extract_active_toolchain() {
        // Outside a cargo build there are no $CARGO/$RUSTC vars; current()
        // degrades to PATH resolution, which still picks up the active toolchain.
        let config = driver::DocGenConfig::current();

        let output = match driver::generate(&config) {
            Ok(o) => o,
            Err(errs) => {
                eprintln!("SKIP: doc-JSON generation failed:\n  {}", errs.join("\n  "));
                return;
            }
        };

        assert!(
            output.data.contains_key("core"),
            "core JSON missing from output"
        );
        let core_data = &output.data["core"];

        // Verify format_version is sane (>= 37 for rustc 1.85+, our MSRV)
        let fv = core_data["format_version"].as_u64().unwrap_or(0);
        assert!(fv >= 37, "unexpected format_version {} (need >= 37)", fv);

        // Verify canonical path resolution via id lookup.
        // Find AtomicUsize's id and confirm its canonical path.
        let paths_table = core_data["paths"].as_object().unwrap();
        let mut found_atomic = false;
        for (id_str, entry) in paths_table {
            let Some(path_arr) = entry.get("path").and_then(|v| v.as_array()) else {
                continue;
            };
            let segs: Vec<&str> = path_arr.iter().filter_map(|s| s.as_str()).collect();
            if segs.last() == Some(&"AtomicUsize") && segs.first() == Some(&"core") {
                // The paths table is keyed by numeric ID; make sure ours parses.
                let _id: u64 = id_str.parse().unwrap();
                let canon = segs.join("::");
                assert_eq!(canon, "core::sync::atomic::AtomicUsize");
                found_atomic = true;
                break;
            }
        }
        assert!(found_atomic, "AtomicUsize not found in core paths table");

        // --- UnsafeCell: plain struct, one private field, repr(transparent) ---
        let cell = locate_and_extract(core_data, "cell::UnsafeCell", "core")
            .expect("failed to extract UnsafeCell");
        assert_eq!(cell.name, "UnsafeCell");
        assert_eq!(cell.lib, "core");
        match &cell.kind {
            model::DocTypeKind::Struct { fields, tuple } => {
                assert!(!tuple, "UnsafeCell should be a plain struct");
                assert_eq!(fields.len(), 1, "UnsafeCell has exactly one field");
                assert_eq!(fields[0].name, "value");
                assert_eq!(fields[0].ty.to_source(), "T");
                assert!(
                    matches!(fields[0].visibility, model::DocVisibility::Restricted(_)),
                    "UnsafeCell::value should be restricted (private)"
                );
            }
            _ => panic!("UnsafeCell expected struct, got {:?}", cell.kind),
        }
        assert_eq!(cell.generics.len(), 1);
        assert_eq!(cell.generics[0].name, "T");
        assert!(
            cell.repr_attrs.iter().any(|r| r == "transparent"),
            "UnsafeCell should have repr(transparent), got {:?}",
            cell.repr_attrs
        );

        // --- Option: enum with unit + tuple variants ---
        let opt = locate_and_extract(core_data, "option::Option", "core")
            .expect("failed to extract Option");
        assert_eq!(opt.name, "Option");
        match &opt.kind {
            model::DocTypeKind::Enum { variants } => {
                assert_eq!(variants.len(), 2);
                assert_eq!(variants[0].name, "None");
                assert!(matches!(variants[0].kind, model::DocVariantKind::Unit));
                assert_eq!(variants[1].name, "Some");
                match &variants[1].kind {
                    model::DocVariantKind::Tuple(fields) => {
                        assert_eq!(fields.len(), 1);
                        assert_eq!(fields[0].ty.to_source(), "T");
                    }
                    other => panic!("Option::Some expected tuple, got {:?}", other),
                }
            }
            _ => panic!("Option expected enum, got {:?}", opt.kind),
        }

        // --- Range: plain struct with two named public fields ---
        let range = locate_and_extract(core_data, "ops::range::Range", "core")
            .expect("failed to extract Range");
        assert_eq!(range.name, "Range");
        match &range.kind {
            model::DocTypeKind::Struct { fields, tuple } => {
                assert!(!tuple);
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "start");
                assert_eq!(fields[1].name, "end");
                assert!(matches!(fields[0].visibility, model::DocVisibility::Public));
            }
            _ => panic!("Range expected struct, got {:?}", range.kind),
        }

        // --- Cell: plain struct with repr(transparent) ---
        let cell2 =
            locate_and_extract(core_data, "cell::Cell", "core").expect("failed to extract Cell");
        assert_eq!(cell2.name, "Cell");
        assert!(
            cell2.repr_attrs.iter().any(|r| r == "transparent"),
            "Cell should have repr(transparent)"
        );

        println!(
            "Extraction verified (format_version={}, sysroot={})",
            fv,
            output.sysroot.display()
        );
    }

    /// Test type rendering for various type representations.
    #[test]
    fn test_type_rendering() {
        use type_repr::TypeRepr;

        // Generic
        let val = serde_json::json!({"generic": "T"});
        let ty = TypeRepr::from_json(&val).unwrap();
        assert_eq!(ty.to_source(), "T");

        // Primitive
        let val = serde_json::json!({"primitive": "u64"});
        let ty = TypeRepr::from_json(&val).unwrap();
        assert_eq!(ty.to_source(), "u64");

        // Resolved path with generics
        let val = serde_json::json!({
            "resolved_path": {
                "path": "crate::vec::Vec",
                "id": 123,
                "args": {
                    "angle_bracketed": {
                        "args": [{"type": {"primitive": "u8"}}],
                        "constraints": []
                    }
                }
            }
        });
        let ty = TypeRepr::from_json(&val).unwrap();
        assert_eq!(ty.to_source(), "vec::Vec<u8>");

        // Borrowed reference
        let val = serde_json::json!({
            "borrowed_ref": {
                "lifetime": null,
                "mutable": false,
                "type": {"primitive": "str"}
            }
        });
        let ty = TypeRepr::from_json(&val).unwrap();
        assert_eq!(ty.to_source(), "&str");

        // Raw pointer
        let val = serde_json::json!({
            "raw_pointer": {
                "mutable": true,
                "type": {"generic": "T"}
            }
        });
        let ty = TypeRepr::from_json(&val).unwrap();
        assert_eq!(ty.to_source(), "*mut T");

        // Tuple
        let val = serde_json::json!({
            "tuple": [
                {"primitive": "i32"},
                {"generic": "T"}
            ]
        });
        let ty = TypeRepr::from_json(&val).unwrap();
        assert_eq!(ty.to_source(), "(i32, T)");

        // Array
        let val = serde_json::json!({
            "array": {
                "type": {"primitive": "u8"},
                "len": "32"
            }
        });
        let ty = TypeRepr::from_json(&val).unwrap();
        assert_eq!(ty.to_source(), "[u8; 32]");

        // Lifetime reference
        let val = serde_json::json!({
            "borrowed_ref": {
                "lifetime": "'a",
                "mutable": true,
                "type": {"generic": "T"}
            }
        });
        let ty = TypeRepr::from_json(&val).unwrap();
        assert_eq!(ty.to_source(), "&'a mut T");
    }
}

pub mod wire;
