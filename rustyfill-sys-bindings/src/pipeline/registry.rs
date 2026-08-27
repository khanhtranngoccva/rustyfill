//! Phase 1d — Type registry construction: populate a [`TypeRegistry`] from all
//! registered sources, spec declarations, known external types, and re-export
//! shims. Also runs minimal-module mirroring for each target so preserved
//! qualifiers resolve correctly.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::emitter::TypeRegistry;
use crate::loader_spec::LoaderSpec;
use crate::parser::{CfgContext, ItemKind, ParsedSource};
use crate::resolver::{ModuleResolver, UseKind, Visibility};
use crate::syntaxes::{BindingModel, ModulePath, NodeStatus};

use super::mirror::{EmitSink, mirror_minimal_modules};

/// Mutable accumulators threaded through [`build_type_registry`] and its
/// call to [`mirror_minimal_modules`].
pub(super) struct RegistryBuildState<'a> {
    pub(super) resolver: &'a mut ModuleResolver,
    pub(super) parsed_cache: &'a mut HashMap<String, (ParsedSource, String)>,
    pub(super) model: &'a mut BindingModel,
    pub(super) emitted_canonicals: &'a mut HashSet<String>,
    pub(super) all_files: &'a mut Vec<(String, String)>,
}

pub(super) fn build_type_registry(
    spec: &LoaderSpec,
    rust_src: &Path,
    cfg: &CfgContext,
    reexport_located: &[(String, String)],
    out_dir: &Path,
    state: &mut RegistryBuildState<'_>,
) -> TypeRegistry {
    let RegistryBuildState {
        resolver,
        parsed_cache,
        model,
        emitted_canonicals,
        all_files,
    } = state;
    let mut registry = TypeRegistry::empty();
    for target in &spec.targets {
        let lib_prefix = format!("{}/", target.lib_name);
        for (file_path, parsed) in resolver.registered_sources().iter() {
            if !file_path.starts_with(&lib_prefix) {
                continue;
            }
            let module_path = resolver.file_to_module_path(file_path);
            let exported_names = public_reexport_names(parsed, &module_path);
            for item in &parsed.items {
                // `module_path` is a pure module path; render it canonically.
                let module_canonical = ModulePath::from_slash(&module_path)
                    .map(|mp| mp.to_canonical())
                    .unwrap_or_else(|| module_path.replace('/', "::"));
                // Canonical keys are serialized qualified paths: the leading
                // `::` marks the address as absolute (rooted at the library).
                let canonical = if module_canonical.is_empty() {
                    format!("::{}::{}", target.lib_name, item.name)
                } else {
                    format!("::{}::{}::{}", target.lib_name, module_canonical, item.name)
                };
                let is_exported = exported_names.contains(&item.name);
                registry.register(&canonical, item.visibility, is_exported, file_path);
                if let Some(rhs) = &item.alias_rhs {
                    registry.set_alias_rhs(&canonical, rhs.clone());
                }
            }
            for (mod_name, mod_items) in &parsed.inline_modules {
                let inline_module = if module_path.is_empty() {
                    mod_name.clone()
                } else {
                    format!("{}/{}", module_path, mod_name)
                };
                // `inline_module` is a pure module path (parent + child).
                let inline_canonical_base = ModulePath::from_slash(&inline_module)
                    .map(|mp| mp.to_canonical())
                    .unwrap_or_else(|| inline_module.replace('/', "::"));
                for item in mod_items {
                    let canonical = format!(
                        "::{}::{}::{}",
                        target.lib_name, inline_canonical_base, item.name
                    );
                    let is_exported = exported_names.contains(&item.name);
                    registry.register(&canonical, item.visibility, is_exported, file_path);
                    if let Some(rhs) = &item.alias_rhs {
                        registry.set_alias_rhs(&canonical, rhs.clone());
                    }
                }
            }
        }

        let lib_src = rust_src.join(&target.lib_name).join("src");
        let active_decls = target.active_declarations(cfg);
        for decl in &active_decls {
            let canonical = format!("::{}::{}", target.lib_name, decl);
            let leaf = decl.rsplit("::").next().unwrap_or("");
            let mut found_item: Option<&crate::parser::ParsedItem> = None;
            let def_file_rel = parsed_cache
                .iter()
                .find(|(_, (parsed, ln))| {
                    ln == &target.lib_name && parsed.items.iter().any(|i| i.name == leaf)
                })
                .map(|(fp, (parsed, _))| {
                    found_item = parsed.items.iter().find(|i| i.name == leaf);
                    fp.clone()
                })
                .unwrap_or_else(|| decl.replace("::", "/") + ".rs");
            let def_file_abs = lib_src.join(&def_file_rel).to_string_lossy().to_string();
            registry.insert_declared(&canonical, &def_file_abs);
            // Mirror the declaration into the binding tree so the item's node
            // carries `declared = true` and its authoritative def file.
            model.mark_declared(&canonical, Some(def_file_abs));
            if let Some(item) = found_item {
                if item.kind == ItemKind::TypeAlias {
                    if let Some(rhs) = &item.alias_rhs {
                        registry.set_alias_rhs(&canonical, rhs.clone());
                    }
                }
            }
        }

        // Register spec-declared known external types at their canonical path so
        // references route to them like any other mirrored type. Their definition
        // is emitted as a standalone stub file (Phase 2), not parsed from source,
        // so the def_file points at the generated stub's relative path.
        for kt in &target.known_external_types {
            let canonical = format!("::{}::{}", target.lib_name, kt.path);
            let segments: Vec<&str> = kt.path.split("::").collect();
            let stub_rel = if segments.len() >= 2 {
                format!("{}.rs", segments[..segments.len() - 1].join("/"))
            } else {
                continue;
            };
            registry.insert_declared(&canonical, &stub_rel);
            // Known-type stubs are synthetic leaf modules that will be emitted
            // in Phase 2b; register them in the tree now so they participate in
            // sibling/child scans and the manifest.
            if let Some(stub_mp) = ModulePath::from_slash(&segments[..segments.len() - 1].join("/"))
            {
                model.register_synthetic(&target.lib_name, &stub_rel, NodeStatus::Emittable);
                let _ = stub_mp;
            }
        }

        // Register re-export-shim declarations.
        for (decl, def_file) in reexport_located {
            if !def_file.ends_with(".rs") {
                continue;
            }
            let leaf = decl.rsplit("::").next().unwrap_or("");
            let Some((parsed, _ln)) = parsed_cache.get(def_file) else {
                continue;
            };
            if !parsed.items.iter().any(|i| i.name == leaf) {
                continue;
            }
            // `def_file` stem is a pure module path; render it canonically.
            let mod_path = ModulePath::from_file_stem(def_file)
                .map(|mp| mp.to_canonical())
                .unwrap_or_else(|| {
                    def_file
                        .strip_suffix(".rs")
                        .unwrap_or(def_file)
                        .replace('/', "::")
                });
            let alias_canonical = format!("::{}::{}::{}", target.lib_name, mod_path, leaf);
            let def_file_abs = lib_src.join(def_file).to_string_lossy().to_string();
            registry.insert_declared_alias(&alias_canonical, &def_file_abs);
            // Re-export shims are treated as declared for emission; reflect that
            // in the tree so the item's node is marked accordingly.
            model.mark_declared(&alias_canonical, Some(def_file_abs));
        }

        // ── Minimal-module mirroring for preserved qualifiers ───────────────
        // Runs after the type registry is fully populated (including this
        // target's declared types) so that field references route correctly.
        // It also materializes re-export shims (Strategy B) for non-sibling
        // preserved qualifiers, registering them with the resolver and the
        // emitted-file sets used by later phases.
        let mut sink = EmitSink {
            resolver,
            parsed_cache,
            model,
            registry: &mut registry,
            emitted_canonicals,
            all_files,
        };
        mirror_minimal_modules(target, &lib_src, out_dir, cfg, &mut sink);
    }
    registry
}

/// Compute the set of item names that are publicly re-exported from a module:
/// items defined directly with `pub` visibility plus everything pulled in by
/// `pub use` single imports.
fn public_reexport_names(parsed: &ParsedSource, _module_path: &str) -> HashSet<String> {
    let mut names: HashSet<String> = HashSet::new();
    for item in &parsed.items {
        if item.visibility.is_public() {
            names.insert(item.name.clone());
        }
    }
    for stmt in &parsed.use_statements {
        if !matches!(stmt.visibility, Visibility::Public) {
            continue;
        }
        match &stmt.kind {
            UseKind::Single(plist, alias) => {
                let name = alias.clone().or_else(|| plist.last_named().map(str::to_string));
                if let Some(n) = name {
                    names.insert(n);
                }
            }
            UseKind::Glob(_) => {
                // Globs pull in every public item of the target module; we
                // approximate conservatively by leaving it to the per-item
                // visibility check above.
            }
        }
    }
    names
}
