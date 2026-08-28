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
//! The wire layer ([`wire`]) deserializes each library's blob into a typed
//! [`wire::Crate`] (rustdoc format v37+). Extraction reduces those blobs to
//! two compact tables consumed by the emitter:
//!
//! - the **type table** ([`model::TypeTable`]), one entry per declared type,
//!   and
//! - the **export table** ([`model::ExportTable`]), a flat item-id → routed
//!   absolute-path map used for every type reference at render time.
//!
//! Item ids are per-library namespaces, so export-table rows are keyed by
//! `(lib, item_id)`; within a single library's JSON, `item_id` alone uniquely
//! identifies a paths-table entry.

pub mod driver;
pub mod emit;
pub mod model;
pub mod type_repr;
pub mod wire;

use std::collections::HashMap;

use crate::loader_spec::LoaderSpec;

/// Extract the types declared in the spec from pre-loaded doc-JSON data.
///
/// `data` maps library name → typed crate (`core`, `alloc`, `std`). Returns
/// the pair of tables the emitter consumes: the type table (one entry per
/// successfully located declaration) and the export table (routing for every
/// resolvable item id across all libraries).
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
    data: &HashMap<String, wire::Crate>,
    spec: &LoaderSpec,
) -> Result<(model::TypeTable, model::ExportTable), Vec<String>> {
    let mut type_table = model::TypeTable::new();
    let mut export_table = model::ExportTable::new();
    let mut errors = Vec::new();

    // First pass: build the export table for every loaded library so routes
    // exist before any declared type is looked up.
    for (lib_name, crate_) in data {
        build_export_table(crate_, lib_name, &mut export_table);
    }

    for target in &spec.targets {
        let Some(crate_) = data.get(&target.lib_name) else {
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
            match locate_and_convert(crate_, decl, &target.lib_name) {
                Ok(doc_type) => {
                    finalize_entry(doc_type, decl, &mut export_table, &mut type_table);
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
            match locate_and_convert(crate_, decl, &target.lib_name) {
                Ok(doc_type) => {
                    finalize_entry(doc_type, decl, &mut export_table, &mut type_table);
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
        Ok((type_table, export_table))
    } else {
        Err(errors)
    }
}

/// Attach the module path derived from the spec declaration (everything
/// except the leaf type name), assign the type-table slot, register the
/// mirror-tree route in the export table, and insert into the type table.
fn finalize_entry(
    mut doc_type: model::DocType,
    decl: &str,
    export_table: &mut model::ExportTable,
    type_table: &mut model::TypeTable,
) {
    let segments: Vec<&str> = decl.split("::").collect();
    let module_path = if segments.len() > 1 {
        segments[..segments.len() - 1].join("::")
    } else {
        String::new()
    };
    doc_type.module_path = module_path;
    doc_type.id = type_table.entries.len();

    // Mirror-tree route for the declared type itself.
    let rest = if doc_type.module_path.is_empty() {
        doc_type.name.clone()
    } else {
        format!("{}::{}", doc_type.module_path, doc_type.name)
    };
    export_table.insert(model::ExportEntry {
        lib: doc_type.lib.clone(),
        item_id: doc_type.item_id,
        route: format!("crate::std::{rest}"),
    });

    type_table.insert(doc_type);
}

// ── Export table construction ─────────────────────────────────────────────────

/// Correction map from canonical definition-site paths to their public
/// re-export paths. Rustdoc records where a type is *defined* (often in a
/// private submodule), but downstream code accesses it via the public
/// re-export one level up. This map covers the fundamental core/alloc types
/// that follow this pattern and are stable across toolchain versions.
const PUBLIC_PATH_CORRECTIONS: &[(&str, &str)] = &[
    // core::ptr
    ("core::ptr::non_null::NonNull", "core::ptr::NonNull"),
    ("core::ptr::unique::Unique", "core::ptr::Unique"),
    ("core::ptr::mut_ptr::MutPtr", "core::ptr::MutPtr"),
    ("core::ptr::const_ptr::ConstPtr", "core::ptr::ConstPtr"),
    // core::mem
    (
        "core::mem::maybe_uninit::MaybeUninit",
        "core::mem::MaybeUninit",
    ),
    (
        "core::mem::manually_drop::ManuallyDrop",
        "core::mem::ManuallyDrop",
    ),
    // core::alloc
    ("core::alloc::layout::Layout", "core::alloc::Layout"),
    // core::cell
    ("core::cell::cell::Cell", "core::cell::Cell"),
    ("core::cell::once_cell::OnceCell", "core::cell::OnceCell"),
    ("core::cell::lazy::LazyCell", "core::cell::LazyCell"),
    (
        "core::cell::unsafe_cell::UnsafeCell",
        "core::cell::UnsafeCell",
    ),
];

/// Build export-table rows for every entry in one library's paths table.
///
/// Routing rules:
/// - `crate_id == 0` (self): items defined in this library. Declared types
///   get mirror-tree routes (inserted later by `finalize_entry`); everything
///   else routes to the mangled builtin extern.
/// - `crate_id > 0` (foreign): look up `external_crates[crate_id]`; core /
///   alloc / std route to their mangled builtin externs, other dependencies
///   pass through as their canonical path.
fn build_export_table(
    crate_: &wire::Crate,
    lib_name: &str,
    export_table: &mut model::ExportTable,
) {
    use std::collections::BTreeSet;
    let mut corrections_used: BTreeSet<usize> = BTreeSet::new();

    for (item_id, summary) in &crate_.paths {
        let item_id = item_id.0;
        if summary.path.is_empty() {
            continue;
        }
        let canon_raw = summary.path.join("::");

        // Apply public path correction for well-known re-exports.
        let correction_idx = PUBLIC_PATH_CORRECTIONS
            .iter()
            .position(|(from, _)| *from == canon_raw);
        if let Some(ci) = correction_idx {
            corrections_used.insert(ci);
        }
        let segments: Vec<&str> = match correction_idx {
            Some(ci) => PUBLIC_PATH_CORRECTIONS[ci].1.split("::").collect(),
            None => summary.path.iter().map(String::as_str).collect(),
        };

        let route = if summary.crate_id == 0 {
            // Self-reference: item is defined in this library and was not a
            // declared type (those get mirror routes in finalize_entry).
            builtin_route(lib_name, &segments)
        } else {
            // Foreign crate reference.
            let foreign_name = crate_
                .external_crates
                .get(&summary.crate_id)
                .map(|ec| ec.name.as_str())
                .unwrap_or("");

            if matches!(foreign_name, "core" | "alloc" | "std") {
                let rest = &segments[1..].join("::");
                format!("::__rustyfill_builtin_{foreign_name}::{rest}")
            } else {
                // External dependency (libc, hashbrown, etc.): pass through.
                segments.join("::")
            }
        };

        export_table.insert(model::ExportEntry {
            lib: lib_name.to_string(),
            item_id,
            route,
        });
    }

    // A correction entry that never matched means std renamed or moved the
    // private definition site on this toolchain — references to that type now
    // route to the (private) canonical path instead of the public re-export.
    for (i, (from, _to)) in PUBLIC_PATH_CORRECTIONS.iter().enumerate() {
        if !corrections_used.contains(&i) {
            eprintln!(
                "cargo:warning=rustyfill-sys: public-path correction \
                 '{from}' did not match any item in this toolchain's doc-JSON \
                 — the standard library may have moved this type; check that \
                 references still route to the public re-export"
            );
        }
    }
}

/// Build the builtin extern route for a locally-defined type that is NOT a
/// declared (mirrored) type.
fn builtin_route(lib_name: &str, segments: &[&str]) -> String {
    let rest = &segments[1..].join("::");
    match lib_name {
        "core" => format!("::__rustyfill_builtin_core::{rest}"),
        "alloc" => format!("::__rustyfill_builtin_alloc::{rest}"),
        "std" => format!("::__rustyfill_builtin_std::{rest}"),
        _ => segments.join("::"),
    }
}

// ── Location ──────────────────────────────────────────────────────────────────

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
fn locate_and_convert(
    crate_: &wire::Crate,
    spec_path: &str,
    lib_name: &str,
) -> Result<model::DocType, String> {
    let idx = PathIndex::build(crate_);

    // Build the expected full path: [lib_name, seg1, seg2, ..., TypeName]
    let target_full_path: Vec<&str> = std::iter::once(lib_name)
        .chain(spec_path.split("::"))
        .collect();

    let item_id = resolve_type_id(crate_, &idx, &target_full_path)?;

    let item = crate_
        .index
        .get(&wire::Id(item_id))
        .ok_or_else(|| format!("item id {item_id} not in index"))?;

    let mut doc_type = model::DocType::from_item(item, &crate_.index)?;
    doc_type.lib = lib_name.to_string();
    doc_type.item_id = item_id;
    Ok(doc_type)
}

/// Hash indexes over a single library's `paths` table, built once per lookup
/// session. Keys are the joined path segments (`core::cell::UnsafeCell`).
struct PathIndex {
    /// Exact path → item id, restricted to type-ish kinds.
    types: HashMap<String, u32>,
    /// Exact path → item id, restricted to modules.
    modules: HashMap<String, u32>,
}

impl PathIndex {
    fn build(crate_: &wire::Crate) -> Self {
        let mut types = HashMap::new();
        let mut modules = HashMap::new();

        for (id, summary) in &crate_.paths {
            let id = id.0;
            let key: String = summary.path.join("::");
            if key.is_empty() {
                continue;
            }
            match summary.kind {
                wire::ItemKind::Struct
                | wire::ItemKind::Enum
                | wire::ItemKind::Union
                | wire::ItemKind::TypeAlias
                | wire::ItemKind::Constant => {
                    types.entry(key).or_insert(id);
                }
                wire::ItemKind::Module => {
                    modules.entry(key).or_insert(id);
                }
                _ => {}
            }
        }

        Self { types, modules }
    }

    fn find_type(&self, path: &[&str]) -> Option<u32> {
        self.types.get(path.join("::").as_str()).copied()
    }

    fn find_module(&self, path: &[&str]) -> Option<u32> {
        self.modules.get(path.join("::").as_str()).copied()
    }
}

/// Resolve a full path to an item ID, handling re-exports and module aliases.
fn resolve_type_id(
    crate_: &wire::Crate,
    idx: &PathIndex,
    full_path: &[&str],
) -> Result<u32, String> {
    // Strategy 1: direct path match.
    if let Some(id) = idx.find_type(full_path) {
        return Ok(id);
    }

    // Strategy 2+3: walk the path segment by segment, following modules and
    // resolving re-exports / module aliases along the way.
    resolve_by_walking(crate_, idx, full_path)
}

/// Walk path segments through the module tree, following re-exports and
/// module aliases. Navigates from the deepest known module prefix.
fn resolve_by_walking(
    crate_: &wire::Crate,
    idx: &PathIndex,
    full_path: &[&str],
) -> Result<u32, String> {
    // Find the deepest prefix that exists as a module in the paths table,
    // then navigate the remaining segments from there.
    let mut start_idx = 0;
    let mut current_mod_id: Option<u32> = None;

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
            let type_id = find_type_in_module(crate_, mod_id, segment).ok_or_else(|| {
                format!(
                    "type '{}' not found in module (resolving {:?})",
                    segment, full_path
                )
            })?;
            return Ok(type_id);
        } else {
            // Intermediate segment: must be a submodule or module alias.
            mod_id = find_submodule_in_module(crate_, mod_id, segment).ok_or_else(|| {
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
    crate_: &wire::Crate,
    mod_id: u32,
    type_name: &str,
) -> Option<u32> {
    let mod_item = crate_.index.get(&wire::Id(mod_id))?;
    let wire::ItemEnum::Module(m) = &mod_item.inner else {
        return None;
    };

    for iid in &m.items {
        let item = crate_.index.get(iid)?;

        // Direct match: item has the right name and is a type.
        if item.name.as_deref() == Some(type_name) {
            let kind = item.inner.item_kind();
            if matches!(
                kind,
                wire::ItemKind::Struct
                    | wire::ItemKind::Enum
                    | wire::ItemKind::Union
                    | wire::ItemKind::TypeAlias
            ) {
                return Some(iid.0);
            }
        }

        // Re-export: `use` entry with matching name pointing to a type.
        if let wire::ItemEnum::Use(u) = &item.inner {
            if u.name == type_name {
                return u.id.map(|id| id.0);
            }
        }
    }
    None
}

/// Find a submodule (or module alias via `use ... {self}`) in a module's items.
fn find_submodule_in_module(
    crate_: &wire::Crate,
    mod_id: u32,
    sub_name: &str,
) -> Option<u32> {
    let mod_item = crate_.index.get(&wire::Id(mod_id))?;
    let wire::ItemEnum::Module(m) = &mod_item.inner else {
        return None;
    };

    for iid in &m.items {
        let item = crate_.index.get(iid)?;

        // Direct submodule: item is a module with matching name.
        if item.name.as_deref() == Some(sub_name)
            && matches!(item.inner, wire::ItemEnum::Module(_))
        {
            return Some(iid.0);
        }

        // Module alias: `use` entry importing a module (`{self}`).
        if let wire::ItemEnum::Use(u) = &item.inner {
            if u.name == sub_name {
                let target_id = u.id?;
                // Verify the target is actually a module.
                let target_item = crate_.index.get(&target_id)?;
                if matches!(target_item.inner, wire::ItemEnum::Module(_)) {
                    return Some(target_id.0);
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
        assert!(
            core_data.format_version >= 37,
            "unexpected format_version {} (need >= 37)",
            core_data.format_version
        );

        // Verify canonical path resolution via id lookup.
        // Find AtomicUsize's id and confirm its canonical path.
        let mut found_atomic = false;
        for (id, summary) in &core_data.paths {
            if summary.path.first() == Some(&"core".to_string())
                && summary.path.last() == Some(&"AtomicUsize".to_string())
            {
                let canon = summary.path.join("::");
                assert_eq!(canon, "core::sync::atomic::AtomicUsize");
                let _ = id;
                found_atomic = true;
                break;
            }
        }
        assert!(found_atomic, "AtomicUsize not found in core paths table");

        // --- UnsafeCell: plain struct, one private field, repr(transparent) ---
        let cell = locate_and_convert(core_data, "cell::UnsafeCell", "core")
            .expect("failed to extract UnsafeCell");
        assert_eq!(cell.name, "UnsafeCell");
        assert_eq!(cell.lib, "core");
        match &cell.kind {
            model::DocTypeKind::Struct { fields, tuple } => {
                assert!(!tuple, "UnsafeCell should be a plain struct");
                assert_eq!(fields.len(), 1, "UnsafeCell has exactly one field");
                assert_eq!(fields[0].name, "value");
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
        let opt = locate_and_convert(core_data, "option::Option", "core")
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
                    }
                    other => panic!("Option::Some expected tuple, got {:?}", other),
                }
            }
            _ => panic!("Option expected enum, got {:?}", opt.kind),
        }

        // --- Range: plain struct with two named public fields ---
        let range = locate_and_convert(core_data, "ops::range::Range", "core")
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
            locate_and_convert(core_data, "cell::Cell", "core").expect("failed to extract Cell");
        assert_eq!(cell2.name, "Cell");
        assert!(
            cell2.repr_attrs.iter().any(|r| r == "transparent"),
            "Cell should have repr(transparent)"
        );

        println!(
            "Extraction verified (format_version={}, sysroot={})",
            core_data.format_version,
            output.sysroot.display()
        );
    }
}
