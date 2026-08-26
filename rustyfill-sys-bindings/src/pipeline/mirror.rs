//! Minimal-module mirroring: detect preserved module qualifiers in declared
//! alias RHSes and struct fields, resolve each to its defining module, record
//! a qualifier route for the emitter, and mirror the minimal defining module
//! (declaring only the referenced leaves) so the preserved qualifier resolves.
//! Also materializes re-export shims (Strategy B) for non-sibling preserved
//! qualifiers and cfg-selected re-export shims.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use crate::emitter::{QualifierResolver, TypeRegistry, collect_qualified_refs, emit_reexport_shim};
use crate::loader_spec::BindingTarget;
use crate::parser::{CfgContext, ItemKind, ParsedSource, parse_source_with_cfg};
use crate::resolver::{ModuleResolver, PathSegment, UseKind};
use crate::syntaxes::ModulePath;

use super::discover::locate_declared_struct;

/// Mutable state that [`mirror_minimal_modules`] reads from and writes into as
/// it discovers preserved qualifiers and materializes their mirrors/shims.
/// Bundling these accumulators keeps the mirroring entry point's argument list
/// small; they are threaded through here rather than passed individually.
pub(super) struct EmitSink<'a> {
    pub(super) resolver: &'a mut ModuleResolver,
    pub(super) parsed_cache: &'a mut HashMap<String, (ParsedSource, String)>,
    pub(super) registry: &'a mut TypeRegistry,
    pub(super) emitted_canonicals: &'a mut HashSet<String>,
    pub(super) all_files: &'a mut Vec<(String, String)>,
}

/// Detect preserved module qualifiers in declared alias RHSes and struct
/// fields, resolve each to its defining module, record a qualifier route for
/// the emitter, and mirror the minimal defining module (declaring only the
/// referenced leaves) so the preserved qualifier resolves.
pub(super) fn mirror_minimal_modules(
    target: &BindingTarget,
    lib_src: &Path,
    out_dir: &Path,
    cfg: &CfgContext,
    sink: &mut EmitSink<'_>,
) {
    let seed: Vec<_> = sink
        .parsed_cache
        .iter()
        .filter(|(_, (_, ln))| ln == &target.lib_name)
        .map(|(fp, (parsed, _))| (fp.clone(), parsed.clone()))
        .collect();
    let mut qres = QualifierResolver::new(lib_src, cfg, seed);

    // Collect (module_ctx, lead, leaf) triples from every active declaration's
    // alias RHS and struct fields.
    let qual_refs = collect_qualifier_refs(target, cfg, sink.parsed_cache);

    // Map each resolving defining module to the set of leaf aliases actually
    // referenced from it, and record a qualifier route for the emitter. For
    // non-sibling qualifiers we additionally record a *module-alias import*:
    // rather than rewriting every `lead::leaf` reference to an absolute mirror
    // path, emit `use <mirror-of-def_mod> as lead;` at the top of the referring
    // file so the preserved qualifier resolves through source-parity paths.
    // This is what makes e.g. `pal::Mutex` in `sys/sync/mutex/pthread` resolve
    // to `crate::std::sys::pal::sync::Mutex`.
    // Resolve a preserved-qualifier reference `lead::lf` seen from
    // `module_ctx` to the slash-separated module that defines `lf`. Tries the
    // standard import-following resolution first, then falls back to probing
    // the named segments of the binding's own path — needed for
    // `use ...::{self, ..}` bindings whose target module is not a direct
    // child of the current module (e.g. nightly moved the futex types to
    // `sys/sync/futex`, reached from `sys/sync/mutex/futex` via
    // `use crate::sys::sync::futex::{..}`).
    let resolve_def_mod = |qres: &mut QualifierResolver<'_>,
                           module_ctx: &str,
                           lead: &str,
                           lf: &str|
     -> Option<String> {
        if let Some(m) = qres.resolve_qualified_ref(module_ctx, Some(lead), lf) {
            return Some(m);
        }
        let cur = qres.source_module(module_ctx)?;
        for stmt in &cur.use_statements {
            let (target_segs, alias_name) = match &stmt.kind {
                UseKind::Single(pl, alias) => (pl.segments.clone(), alias.clone()),
                _ => continue,
            };
            let bound_name: Option<String> = match &alias_name {
                Some(a) => Some(a.clone()),
                None => {
                    let last_named = target_segs.iter().rev().find_map(|s| match s {
                        PathSegment::Named(n) => Some(n.clone()),
                        _ => None,
                    });
                    match last_named.as_deref() {
                        Some("self") => target_segs.iter().rev().find_map(|s| match s {
                            PathSegment::Named(n) if n != "self" => Some(n.clone()),
                            _ => None,
                        }),
                        other => other.map(str::to_string),
                    }
                }
            };
            if bound_name.as_deref() != Some(lead) {
                continue;
            }
            let base: Vec<String> = module_ctx
                .split('/')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            let mut resolved_base = base;
            for seg in &target_segs {
                match seg {
                    PathSegment::Crate => resolved_base.clear(),
                    PathSegment::Super => {
                        resolved_base.pop();
                    }
                    PathSegment::Self_ => {}
                    PathSegment::Named(n) => resolved_base.push(n.clone()),
                }
            }
            // Probe progressively longer prefixes of the binding's concrete
            // path; descend through re-export layers when a segment isn't a
            // direct child module.
            let mut trail: Vec<String> = Vec::new();
            for name in &resolved_base {
                let candidate = if trail.is_empty() {
                    name.clone()
                } else {
                    format!("{}/{}", trail.join("/"), name)
                };
                if qres.source_module(&candidate).is_some() {
                    trail.push(name.clone());
                    continue;
                }
                if trail.is_empty() {
                    break;
                }
                let cur_mod = trail.join("/");
                match qres.descend_through_reexport(&cur_mod, name) {
                    Some(t) => {
                        for part in t.split('/') {
                            trail.push(part.to_string());
                        }
                    }
                    None => break,
                }
            }
            if !trail.is_empty() {
                if let Some(def) = qres.find_defining_module(&trail.join("/"), lf) {
                    return Some(def);
                }
            }
        }
        None
    };

    let mut needed_leaves: HashMap<String, HashSet<String>> = HashMap::new();
    for (module_ctx, lead_opt, lf) in &qual_refs {
        // For bare names (lead_opt = None), synthesize a lead from the import
        // binding that brings the name into scope. This lets us reuse the
        // existing qualified-ref resolution machinery (import-following,
        // re-export descent) to find the defining module.
        let lead: String = match lead_opt {
            Some(l) => l.clone(),
            None => {
                // Find the import binding whose bound name equals the bare leaf.
                let cur = match qres.source_module(module_ctx) {
                    Some(s) => s,
                    None => continue,
                };
                let mut found_lead: Option<String> = None;
                for stmt in &cur.use_statements {
                    let (target_segs, alias_name) = match &stmt.kind {
                        UseKind::Single(pl, alias) => (pl.segments.clone(), alias.clone()),
                        _ => continue,
                    };
                    let bound_name: Option<String> = match &alias_name {
                        Some(a) => Some(a.clone()),
                        None => {
                            let last_named = target_segs.iter().rev().find_map(|s| match s {
                                PathSegment::Named(n) => Some(n.clone()),
                                _ => None,
                            });
                            match last_named.as_deref() {
                                Some("self") => target_segs.iter().rev().find_map(|s| match s {
                                    PathSegment::Named(n) if n != "self" => Some(n.clone()),
                                    _ => None,
                                }),
                                other => other.map(str::to_string),
                            }
                        }
                    };
                    if bound_name.as_deref() == Some(lf.as_str()) {
                        found_lead = Some(bound_name.unwrap_or_else(|| lf.clone()));
                        break;
                    }
                }
                match found_lead {
                    Some(l) => l,
                    None => continue,
                }
            }
        };
        let Some(def_mod) = resolve_def_mod(&mut qres, module_ctx, &lead, lf) else {
            continue;
        };
        needed_leaves
            .entry(def_mod.clone())
            .or_default()
            .insert(lf.clone());
        sink.registry
            .set_qualifier_route(module_ctx, &lead, &def_mod);

        // Record an alias import for any non-sibling qualifier so that
        // references like `pal::Mutex` resolve without path rewriting.
        // At emit time, the pipeline checks whether the resolver already
        // bound the same name and skips the alias to avoid E0252.
        //
        // IMPORTANT: only record the alias when the leaf is DIRECTLY defined
        // in the alias-target module. If the leaf lives in a sub-module of the
        // alias target (e.g., `sys` → `sys/sync` but `Mutex` is in
        // `sys/sync/mutex`), the alias would dangle. In that case, skip the
        // alias and let the qualifier-route rewrite absolutize the reference.
        let sibling = format!("{module_ctx}/{lead}");
        if sibling != def_mod {
            let import_target = match qres.resolve_import_target(module_ctx, &lead) {
                Some(t) if t == def_mod => t,
                _ => def_mod.clone(),
            };
            // Verify the leaf is directly accessible from the alias target.
            // If not, skip the alias — the qualifier route will absolutize.
            let leaf_accessible = if import_target == def_mod {
                true // Leaf is in the defining module itself.
            } else {
                // Check if the alias target module directly defines or
                // re-exports the leaf.
                qres.source_module(&import_target)
                    .map(|src| {
                        src.items
                            .iter()
                            .any(|i| i.name.as_str() == lf.as_str() && i.kind.is_type_def())
                    })
                    .unwrap_or(false)
            };
            if leaf_accessible {
                // `import_target` is a pure module path; render it canonically.
                let import_canonical = ModulePath::from_slash(&import_target)
                    .map(|mp| mp.to_canonical())
                    .unwrap_or_else(|| import_target.replace('/', "::"));
                let crate_path = format!(
                    "crate::{}::{import_canonical}",
                    sink.registry.wrapper_mod()
                );
                sink.registry
                    .set_module_alias_route(module_ctx, &lead, &crate_path);
            }
        }
    }

    // Mirror each unmirrored defining module, declaring only the referenced
    // leaves so the emitter writes a slim mirror.
    for (def_mod, leaves) in &needed_leaves {
        let def_file = format!("{def_mod}.rs");
        if sink.parsed_cache.contains_key(&def_file)
            || sink.parsed_cache.contains_key(&format!("{def_mod}/mod.rs"))
        {
            continue;
        }
        let abs = lib_src.join(&def_file);
        let Ok(text) = fs::read_to_string(&abs) else {
            continue;
        };
        let parsed = parse_source_with_cfg(&text, cfg);
        // `def_mod` is a pure module path; render it canonically.
        let mod_path = ModulePath::from_slash(def_mod)
            .map(|mp| mp.to_canonical())
            .unwrap_or_else(|| def_mod.replace('/', "::"));
        let def_file_abs = lib_src.join(&def_file).to_string_lossy().to_string();
        for item in &parsed.items {
            if !leaves.contains(&item.name) {
                continue;
            }
            let canonical = format!("{}::{}::{}", target.lib_name, mod_path, item.name);
            match item.kind {
                ItemKind::TypeAlias => {
                    sink.registry
                        .insert_declared_alias(&canonical, &def_file_abs);
                    if let Some(rhs) = &item.alias_rhs {
                        sink.registry.set_alias_rhs(&canonical, rhs.clone());
                    }
                }
                ItemKind::Struct | ItemKind::Enum | ItemKind::Union => {
                    sink.registry.insert_declared(&canonical, &def_file_abs);
                }
                _ => {}
            }
        }
        sink.parsed_cache
            .insert(def_file, (parsed, target.lib_name.clone()));
    }

    // Strategy B: materialize a re-export shim for every preserved qualifier
    // whose defining module is NOT a sibling of the referring file.
    materialize_reexport_shims(target, out_dir, cfg, &mut qres, &qual_refs, sink);
}

/// Collect `(module_ctx, lead, leaf)` triples from every active declaration's
/// alias RHS and struct fields by locating the defining file for each
/// declaration and extracting qualified type references.
fn collect_qualifier_refs(
    target: &BindingTarget,
    cfg: &CfgContext,
    parsed_cache: &HashMap<String, (ParsedSource, String)>,
) -> Vec<(String, Option<String>, String)> {
    let active_decls = target.active_declarations(cfg);
    let mut qual_refs: Vec<(String, Option<String>, String)> = Vec::new();
    for decl in &active_decls {
        let leaf = decl.rsplit("::").next().unwrap_or("");
        let decl_mod: Vec<&str> = decl.split("::").collect();
        let Some((def_file_rel, found_item)) = parsed_cache
            .iter()
            .find(|(fp, (parsed, ln))| {
                if ln != &target.lib_name {
                    return false;
                }
                if !parsed.items.iter().any(|i| i.name == leaf) {
                    return false;
                }
                let stem = fp.strip_suffix(".rs").unwrap_or(fp.as_str());
                let fp_mod: Vec<&str> = stem
                    .strip_suffix("/mod")
                    .unwrap_or(stem)
                    .split('/')
                    .filter(|s| !s.is_empty())
                    .collect();
                let file_module: &[&str] = if stem.ends_with("/mod") {
                    &fp_mod
                } else {
                    &fp_mod[..fp_mod.len().saturating_sub(1)]
                };
                let is_prefix = file_module.len() <= decl_mod.len()
                    && decl_mod[..file_module.len()] == file_module[..];
                let is_suffix = file_module.len() >= decl_mod.len()
                    && file_module[file_module.len() - decl_mod.len()..] == decl_mod[..];
                is_prefix || is_suffix
            })
            .map(|(fp, (parsed, _))| (fp.clone(), parsed.items.iter().find(|i| i.name == leaf)))
        else {
            continue;
        };
        let module_ctx = def_file_rel
            .strip_suffix(".rs")
            .unwrap_or(&def_file_rel)
            .to_string();
        let Some(item) = found_item else { continue };
        match item.kind {
            ItemKind::TypeAlias => {
                if let Some(rhs) = &item.alias_rhs {
                    if let Ok(ty) = syn::parse2::<syn::Type>(rhs.clone()) {
                        for (lead, lf) in collect_qualified_refs(&ty) {
                            qual_refs.push((module_ctx.clone(), lead, lf));
                        }
                    }
                }
            }
            ItemKind::Struct => {
                if let Ok(s) = syn::parse2::<syn::ItemStruct>(item.full_tokens.clone()) {
                    for f in &s.fields {
                        for (lead, lf) in collect_qualified_refs(&f.ty) {
                            qual_refs.push((module_ctx.clone(), lead, lf));
                        }
                    }
                }
            }
            ItemKind::Const => {
                // Const items carry a type annotation (e.g., `pub const ONCE_INIT: Once`).
                // Bare names in the type position need routing just like struct fields.
                if let Ok(c) = syn::parse2::<syn::ItemConst>(item.full_tokens.clone()) {
                    for (lead, lf) in collect_qualified_refs(&c.ty) {
                        qual_refs.push((module_ctx.clone(), lead, lf));
                    }
                }
            }
            _ => {}
        }
    }
    qual_refs
}

/// Materialize re-export shims (Strategy B) for every preserved qualifier whose
/// defining module is NOT a sibling of the referring file. The shim lives at the
/// canonical alias location and forwards the leaf to the actual definition.
#[allow(clippy::too_many_arguments)]
fn materialize_reexport_shims(
    target: &BindingTarget,
    out_dir: &Path,
    cfg: &CfgContext,
    qres: &mut QualifierResolver<'_>,
    qual_refs: &[(String, Option<String>, String)],
    sink: &mut EmitSink<'_>,
) {
    for (module_ctx, lead, lf) in qual_refs {
        let Some(lead) = lead else { continue };
        let Some(def_mod) = qres.resolve_qualified_ref(module_ctx, Some(lead), lf) else {
            continue;
        };
        let sibling = format!("{module_ctx}/{lead}");
        if sibling == def_mod {
            continue;
        }
        let cur = match qres.source_module(module_ctx) {
            Some(s) => s,
            None => continue,
        };
        let bound_name_of = |segs: &[PathSegment], alias: &Option<String>| -> Option<String> {
            match alias {
                Some(a) => Some(a.clone()),
                None => {
                    let last_named = segs.iter().rev().find_map(|s| match s {
                        PathSegment::Named(n) => Some(n.clone()),
                        _ => None,
                    });
                    match last_named.as_deref() {
                        Some("self") => segs.iter().rev().find_map(|s| match s {
                            PathSegment::Named(n) if n != "self" => Some(n.clone()),
                            _ => None,
                        }),
                        other => other.map(str::to_string),
                    }
                }
            }
        };
        let mut alias_mod: Option<String> = None;
        for stmt in &cur.use_statements {
            let (target_segs, alias_name) = match &stmt.kind {
                UseKind::Single(pl, alias) => (pl.segments.clone(), alias.clone()),
                _ => continue,
            };
            let Some(bound_name) = bound_name_of(&target_segs, &alias_name) else {
                continue;
            };
            if bound_name != *lead {
                continue;
            }
            let base: Vec<String> = module_ctx
                .split('/')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            let mut resolved_base = base;
            for seg in &target_segs {
                match seg {
                    PathSegment::Crate => resolved_base.clear(),
                    PathSegment::Super => {
                        resolved_base.pop();
                    }
                    PathSegment::Self_ => {}
                    PathSegment::Named(n) => resolved_base.push(n.clone()),
                }
            }
            let mut trail: Vec<String> = Vec::new();
            let mut ok = true;
            for name in &resolved_base {
                let candidate = if trail.is_empty() {
                    name.clone()
                } else {
                    format!("{}/{}", trail.join("/"), name)
                };
                if qres.source_module(&candidate).is_some() {
                    trail.push(name.clone());
                    continue;
                }
                if trail.is_empty() {
                    ok = false;
                    break;
                }
                let cur_mod = trail.join("/");
                match qres.descend_through_reexport(&cur_mod, name) {
                    Some(t) => {
                        for part in t.split('/') {
                            trail.push(part.to_string());
                        }
                    }
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && !trail.is_empty() {
                alias_mod = Some(trail.join("/"));
                break;
            }
        }
        let Some(alias_mod) = alias_mod else { continue };
        let existing_alias_file = format!("{alias_mod}.rs");
        let existing_alias_mod = format!("{alias_mod}/mod.rs");
        if sink.emitted_canonicals.contains(&existing_alias_file)
            || sink.emitted_canonicals.contains(&existing_alias_mod)
        {
            continue;
        }
        let def_submodule = if def_mod == alias_mod {
            String::new()
        } else if let Some(rest) = def_mod.strip_prefix(&format!("{alias_mod}/")) {
            rest.to_string()
        } else {
            def_mod.rsplit('/').next().unwrap_or("").to_string()
        };
        let shim_rel = emit_reexport_shim(
            out_dir,
            &target.lib_name,
            &alias_mod,
            lf,
            &alias_mod,
            &def_submodule,
        );
        let Some(shim_rel) = shim_rel else { continue };
        let shim_content = fs::read_to_string(out_dir.join(&shim_rel)).unwrap_or_default();
        let parsed = parse_source_with_cfg(&shim_content, cfg);
        sink.resolver.register_source(&shim_rel, parsed);
        sink.resolver.mark_emittable(&shim_rel);
        sink.emitted_canonicals.insert(shim_rel.clone());
        sink.all_files.push((shim_rel, target.lib_name.clone()));
    }
}

/// For each spec-declared type whose canonical module has no emitted binding
/// file of its own, emit a re-export shim forwarding the leaf to the concrete
/// definition (located via `locate_declared_struct`). This covers cfg-selected
/// re-exports like `sys::sync::mutex::Mutex` (= `pub use futex::Mutex;` on
/// Linux) where only the active backend submodule was mirrored.
pub(super) fn emit_cfg_reexport_shims(
    rust_src: &Path,
    targets: &[BindingTarget],
    out_dir: &Path,
    resolver: &mut ModuleResolver,
    cfg: &CfgContext,
    emitted_canonicals: &mut HashSet<String>,
    all_files: &mut Vec<(String, String)>,
) {
    for target in targets {
        let lib_src = rust_src.join(&target.lib_name).join("src");
        let active_decls = target.active_declarations(cfg);
        for decl in &active_decls {
            let parts: Vec<&str> = decl.split("::").collect();
            if parts.len() < 2 {
                continue;
            }
            let leaf = *parts.last().unwrap();
            // The canonical module is everything but the leaf.
            let canon_module = parts[..parts.len() - 1].join("/");
            // Skip if this module already produced an emitted binding file.
            let existing_file = format!("{canon_module}.rs");
            let existing_mod = format!("{canon_module}/mod.rs");
            if emitted_canonicals.contains(&existing_file)
                || emitted_canonicals.contains(&existing_mod)
            {
                continue;
            }
            // Locate the concrete defining file. If it's the same as the naive
            // canonical path there's nothing to forward.
            if let LocatedStruct::Found(def_file) = locate_declared_struct(decl, &lib_src, cfg) {
                let def_stem = def_file.strip_suffix(".rs").unwrap_or(&def_file);
                let def_module = def_stem.to_string();
                // Same module — the struct is defined directly here; the
                // normal emitter handles it. No shim needed.
                if def_module == canon_module {
                    continue;
                }
                // Compute the relative submodule from canon_module to def_module.
                let def_submodule =
                    if let Some(rest) = def_module.strip_prefix(&format!("{canon_module}/")) {
                        rest.to_string()
                    } else {
                        // Definition is not nested under the canonical module —
                        // can't express as a simple re-export shim.
                        continue;
                    };
                let shim_rel = emit_reexport_shim(
                    out_dir,
                    &target.lib_name,
                    &canon_module,
                    leaf,
                    &canon_module,
                    &def_submodule,
                );
                let Some(shim_rel) = shim_rel else { continue };
                let shim_content =
                    std::fs::read_to_string(out_dir.join(&shim_rel)).unwrap_or_default();
                let parsed = parse_source_with_cfg(&shim_content, cfg);
                resolver.register_source(&shim_rel, parsed);
                resolver.mark_emittable(&shim_rel);
                emitted_canonicals.insert(shim_rel.clone());
                all_files.push((shim_rel, target.lib_name.clone()));
            }
        }
    }
}

// Re-import LocatedStruct from discover for use in emit_cfg_reexport_shims
use super::discover::LocatedStruct;
