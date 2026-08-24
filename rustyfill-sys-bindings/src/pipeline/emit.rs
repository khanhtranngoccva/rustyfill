//! Phase 2 — Emission: write binding files for every parsed source file (and
//! their inline modules), plus known-external-type stubs. Also handles Phase 3
//! re-export alias discovery and emission.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::emitter::{
    EmitConfig, TypeRegistry, emit_binding_file, emit_glob_reexport_aliases,
};
use crate::loader_spec::LoaderSpec;
use crate::parser::{CfgContext, ParsedSource};
use crate::resolver::ModuleResolver;
use crate::validator::ValidationBuilder;

use super::util::{compute_module_depth, get_sibling_modules};

/// Emit binding files for every parsed source file (and their inline modules),
/// plus known-external-type stubs. Accumulates emitted paths and canonicals.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_all_binding_files(
    parsed_cache: &HashMap<String, (ParsedSource, String)>,
    resolver: &mut ModuleResolver,
    registry: &TypeRegistry,
    replacement_entries_slice: &[(String, Option<&str>)],
    ignored_name_refs: &[&str],
    ignored_structs_by_lib: &HashMap<String, Vec<String>>,
    extra_derives_by_lib: &HashMap<String, HashMap<String, Vec<String>>>,
    out_dir: &Path,
    validator: &mut ValidationBuilder,
    emitted_paths: &mut Vec<PathBuf>,
    emitted_canonicals: &mut HashSet<String>,
    all_files: &mut Vec<(String, String)>,
) {
    for (file_path, (parsed, lib_name)) in parsed_cache {
        if file_path.ends_with("/mod") || file_path == "mod" {
            continue;
        }
        let depth = compute_module_depth(file_path);
        let mut extra_uses = resolver.emit_use_statements_for_file(file_path, ignored_name_refs);
        let stem = file_path.strip_suffix(".rs").unwrap_or(file_path.as_str());
        let module_key = stem.strip_suffix("/mod").unwrap_or(stem);
        let already_bound = collect_bound_names(&extra_uses);
        for (alias, crate_path) in registry.module_alias_routes(module_key) {
            if already_bound.contains(alias) {
                continue;
            }
            extra_uses.push(format!(
                "#[allow(unused_imports)] use {crate_path} as {alias};"
            ));
        }
        let siblings = get_sibling_modules(file_path, all_files);
        let emit_path = out_dir.join(file_path);

        let target_ignored_structs = ignored_structs_by_lib
            .get(lib_name)
            .cloned()
            .unwrap_or_default();
        let target_extra_derives = extra_derives_by_lib
            .get(lib_name)
            .cloned()
            .unwrap_or_default();

        let has_content = emit_binding_file(
            &emit_path,
            &parsed.items,
            &EmitConfig {
                lib_name,
                file_module_depth: depth,
                extra_uses: &extra_uses,
                sibling_modules: &siblings,
                path_replacements: replacement_entries_slice,
                ignored_structs: &target_ignored_structs,
                relative_file_path: file_path,
                type_registry: registry,
                extra_derives: &target_extra_derives,
            },
        );

        if has_content {
            validator.check_emit(&emit_path);
            emitted_paths.push(emit_path);
            emitted_canonicals.insert(file_path.clone());
            all_files.push((file_path.clone(), lib_name.clone()));
        }

        // Also emit inline modules.
        for (mod_name, mod_items) in &parsed.inline_modules {
            let inline_dir = if file_path.ends_with("/mod.rs") {
                file_path.strip_suffix("/mod.rs").unwrap_or("")
            } else {
                file_path.strip_suffix(".rs").unwrap_or(file_path.as_str())
            };

            let inline_rel_path = if inline_dir.is_empty() {
                format!("{}/mod.rs", mod_name)
            } else {
                format!("{}/{}/mod.rs", inline_dir, mod_name)
            };

            let inline_emit_path = out_dir.join(&inline_rel_path);
            let inline_depth = compute_module_depth(&inline_rel_path);
            let mut inline_extra_uses =
                resolver.emit_use_statements_for_file(&inline_rel_path, ignored_name_refs);
            for (alias, crate_path) in registry.module_alias_routes(&inline_rel_path) {
                inline_extra_uses.push(format!(
                    "#[allow(unused_imports)] use {crate_path} as {alias};"
                ));
            }
            let inline_siblings = get_sibling_modules(&inline_rel_path, all_files);
            let inline_has_content = emit_binding_file(
                &inline_emit_path,
                mod_items,
                &EmitConfig {
                    lib_name,
                    file_module_depth: inline_depth,
                    extra_uses: &inline_extra_uses,
                    sibling_modules: &inline_siblings,
                    path_replacements: replacement_entries_slice,
                    ignored_structs: &target_ignored_structs,
                    relative_file_path: &inline_rel_path,
                    type_registry: registry,
                    extra_derives: &target_extra_derives,
                },
            );

            if inline_has_content {
                validator.check_emit(&inline_emit_path);
                emitted_paths.push(inline_emit_path);
                emitted_canonicals.insert(inline_rel_path.clone());
                all_files.push((inline_rel_path.clone(), lib_name.clone()));
            }
        }
    }
}

/// Extract the set of names already bound by a list of `use` statement lines,
/// so that module-alias imports can skip conflicting identifiers.
fn collect_bound_names(use_lines: &[String]) -> HashSet<String> {
    let mut already_bound = HashSet::new();
    for line in use_lines {
        let trimmed = line
            .trim()
            .strip_prefix("#[allow(unused_imports)]")
            .unwrap_or(line.trim())
            .trim();
        if let Some(body) = trimmed.strip_prefix("use ") {
            let body = body.strip_suffix(';').unwrap_or(body);
            if let Some((_, alias_name)) = body.rsplit_once(" as ") {
                already_bound.insert(alias_name.trim().to_string());
            } else if !body.ends_with("::*")
                && let Some(last_seg) = body.rsplit_once(':').map(|(_, n)| n.trim())
            {
                already_bound.insert(last_seg.to_string());
            }
        }
    }
    already_bound
}

// ── Re-export alias discovery (Phase 3) ─────────────────────────────────────

/// For each active declaration, locate its defining file and all parent module
/// files, discover re-export aliases from each, and emit glob re-export alias
/// binding files. Returns the set of discovered alias modules.
pub(super) fn discover_and_emit_reexport_aliases(
    spec: &LoaderSpec,
    cfg: &CfgContext,
    parsed_cache: &HashMap<String, (ParsedSource, String)>,
    resolver: &mut ModuleResolver,
    out_dir: &Path,
    emitted_canonicals: &HashSet<String>,
    all_files: &mut Vec<(String, String)>,
) -> HashSet<String> {
    let mut discovered_aliases = HashSet::new();
    for target in &spec.targets {
        let active_decls = target.active_declarations(cfg);
        for decl in &active_decls {
            let leaf = decl.rsplit("::").next().unwrap_or("");
            let def_file = parsed_cache
                .iter()
                .find(|(_, (parsed, ln))| {
                    ln == &target.lib_name
                        && (parsed.items.iter().any(|i| i.name == leaf)
                            || parsed
                                .inline_modules
                                .iter()
                                .any(|(_, items)| items.iter().any(|i| i.name == leaf)))
                })
                .map(|(fp, _)| fp.clone());
            let Some(def_file) = def_file else { continue };

            let parents = resolver.get_parent_module_paths(&def_file);
            let all_related: Vec<String> = std::iter::once(def_file).chain(parents).collect();

            for related_file in all_related {
                let aliases = resolver.discover_reexport_aliases(&related_file);
                for (alias_module, canonical_module) in aliases {
                    let new_files = emit_glob_reexport_aliases(
                        resolver,
                        &alias_module,
                        &canonical_module,
                        &target.lib_name,
                        out_dir,
                        &mut discovered_aliases,
                        emitted_canonicals,
                    );
                    all_files.extend(new_files);
                }
            }
        }
    }
    discovered_aliases
}
