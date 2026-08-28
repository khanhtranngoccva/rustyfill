//! Emission of binding files from the extracted type/export tables.
//!
//! Consumes [`model::TypeTable`] + [`model::ExportTable`] (produced by
//! [`super::extract_types`]) and writes Rust source files into the output
//! directory, mirroring the module hierarchy of the original library. Also
//! emits known-type stubs and the final hierarchical manifest.
//!
//! # Path resolution strategy
//!
//! Every type reference in the doc-JSON carries an authoritative item id.
//! The export table maps each `(lib, item_id)` pair to its final routed
//! absolute path — built once during extraction, consulted on every
//! resolved-path reference at render time via a [`type_repr::PathResolver`]
//! callback. Spec path-replacements are applied first (they are complete
//! substitutions); everything else goes through the export table, with a
//! string-based fallback for ids that have no route entry.
//!
//! No prelude module is emitted. All type references use absolute paths.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::docjson::model::{
    DocField, DocGenericParam, DocType, DocTypeKind, DocVariant, DocVisibility, ExportTable,
    TypeTable,
};
use crate::docjson::type_repr;
use crate::formatter::format_source;
use crate::loader_spec::LoaderSpec;
use crate::syntaxes::ModulePath;

/// Safe derives that can be kept on mirrored types without requiring extra
/// trait impls from the downstream environment.
const SAFE_DERIVES: &[&str] = &["PartialEq", "Eq", "Debug", "Clone"];

/// Internal marker traits that must be stripped from bounds (they are private
/// to core and cannot be named from a downstream crate).
const INTERNAL_TRAIT_STRIPS: &[&str] =
    &["PointeeSized", "StructuralPartialEq", "MetaSized", "Unsize"];

/// Name of the wrapper module around all generated bindings.
const WRAPPER_MOD: &str = "std";

// ── Public API ────────────────────────────────────────────────────────────────

/// Configuration for a single doc-JSON emission run.
pub struct EmitInput<'a> {
    /// Directory where generated files are written (`$OUT_DIR`).
    pub out_dir: &'a Path,
    /// The loaded loader spec (drives replacements, ignored types, extra derives).
    pub spec: &'a LoaderSpec,
    /// The type table: one entry per successfully located declaration.
    pub type_table: &'a TypeTable,
    /// The export table: (lib, item_id) → routed absolute path.
    pub export_table: &'a ExportTable,
}

/// Run the full emission: binding files, known-type stubs, manifest.
/// Returns a list of errors (empty on success).
pub fn emit_all(input: &EmitInput) -> Vec<String> {
    let mut errors = Vec::new();

    // Group types by their file location (module path → file).
    let mut files_by_path: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, ty) in input.type_table.entries.iter().enumerate() {
        let rel_path = type_rel_path(ty);
        files_by_path.entry(rel_path).or_default().push(i);
    }

    // Build the flat list of all emitted file paths for child computation.
    let all_files: Vec<String> = files_by_path.keys().cloned().collect();

    // Emit each binding file into its proper subdirectory.
    for (rel_path, indices) in &files_by_path {
        let file_types: Vec<&DocType> =
            indices.iter().map(|&i| &input.type_table.entries[i]).collect();
        let child_modules = get_children(rel_path, &all_files);

        let content = render_binding_file(&file_types, input.spec, input.export_table);

        if !content.is_empty() {
            // Determine the output file path:
            // - If this module has children, it's a directory → write mod.rs
            // - Otherwise, it's a leaf file → write <name>.rs
            let out_path = if child_modules.is_empty() {
                input.out_dir.join(format!("{}.rs", rel_path))
            } else {
                input.out_dir.join(format!("{}/mod.rs", rel_path))
            };
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            let formatted = format_source(&content);
            if let Err(e) = std::fs::write(&out_path, formatted) {
                errors.push(format!("failed to write {}: {}", out_path.display(), e));
            }
        }
    }

    // Emit known-type stubs.
    for target in &input.spec.targets {
        let kts: Vec<&crate::loader_spec::KnownExternalType> =
            target.known_external_types.iter().collect();
        if kts.is_empty() {
            continue;
        }
        let _ = emit_known_type_stubs(input.out_dir, &kts);
    }

    // Emit the hierarchical manifest.
    emit_manifest(input.out_dir, &all_files);

    errors
}

// ── Binding file rendering ────────────────────────────────────────────────────

/// Render a complete binding file's source content.
fn render_binding_file(
    types: &[&DocType],
    spec: &LoaderSpec,
    exports: &ExportTable,
) -> String {
    let mut out = String::from("// Auto-generated by rustyfill-sys.\n\n");

    // Build the replacement lookup: full path → optional replacement.
    let replacements = build_replacement_lookup(spec);

    // Collect ignored trait leaf names for bound filtering (from path_replacements
    // where replacement is None, i.e., ignore_path).
    let ignored_traits: Vec<String> = spec
        .targets
        .iter()
        .flat_map(|t| t.path_replacements.iter())
        .filter(|pr| pr.replacement.is_none())
        .map(|pr| {
            pr.path
                .rsplit("::")
                .next()
                .unwrap_or(pr.path.as_str())
                .to_string()
        })
        .collect();

    // Local names guard: bare references to types defined in THIS file should
    // not be rewritten to cross-module mirror paths.
    let local_names: BTreeSet<&str> = types.iter().map(|t| t.name.as_str()).collect();

    // Extra derives lookup per type in this file.
    let extra_derives: Vec<Vec<String>> = types
        .iter()
        .map(|t| {
            let canon = if t.module_path.is_empty() {
                t.name.clone()
            } else {
                format!("{}::{}", t.module_path, t.name)
            };
            spec.targets
                .iter()
                .find(|tgt| tgt.lib_name == t.lib)
                .and_then(|tgt| tgt.extra_derives.get(&canon))
                .cloned()
                .unwrap_or_default()
        })
        .collect();

    // Determine the source library and module path for this batch of types
    // (all types in a single file share the same lib and module).
    let first_ty = types.first();
    let lib = first_ty.map(|t| t.lib.as_str()).unwrap_or("");
    let current_module = first_ty.map(|t| t.module_path.as_str()).unwrap_or("");

    // Construct the render context.
    let ctx = RenderCtx {
        replacements: &replacements,
        local_names: &local_names,
        exports,
        lib,
        current_module,
    };

    for (i, ty) in types.iter().enumerate() {
        // Check if this type is in the ignored list.
        let fq = if ty.module_path.is_empty() {
            ty.name.clone()
        } else {
            format!("{}::{}", ty.module_path, ty.name)
        };
        let is_ignored = spec
            .targets
            .iter()
            .any(|t| t.lib_name == ty.lib && t.ignored_structs.contains(&fq));
        if is_ignored {
            continue;
        }

        render_doc_type(
            &mut out,
            ty,
            &ctx,
            extra_derives.get(i).map(|v| v.as_slice()).unwrap_or(&[]),
            &ignored_traits,
        );
    }

    out
}

/// Render a single `DocType` as Rust source.
fn render_doc_type(
    out: &mut String,
    ty: &DocType,
    ctx: &RenderCtx,
    extra_derives: &[String],
    ignored_traits: &[String],
) {
    // Attributes: repr
    for repr in &ty.repr_attrs {
        out.push_str(&format!("#[repr({})]\n", repr));
    }

    // Attributes: derive (filtered to safe set + extra derives)
    let mut all_derives: Vec<String> = Vec::new();
    for group in &ty.derive_attrs {
        for d in group {
            if SAFE_DERIVES.contains(&d.as_str()) {
                all_derives.push(d.clone());
            }
        }
    }
    for d in extra_derives {
        if !all_derives.contains(d) {
            all_derives.push(d.clone());
        }
    }
    if !all_derives.is_empty() {
        out.push_str(&format!("#[derive({})]\n", all_derives.join(", ")));
    }

    // Other attributes (must_use, etc.) — strip compiler-internal attrs.
    for attr in &ty.other_attrs {
        if !attr.starts_with("#[lang")
            && !attr.starts_with("#[stable")
            && !attr.starts_with("#[unstable")
            && !attr.contains("rustc_")
            && !attr.contains("fundamental")
            && !attr.contains("doc(search_unbox)")
        {
            out.push_str(attr);
            out.push('\n');
        }
    }

    // Generics header
    let generics_str = render_generics(&ty.generics, ignored_traits, ctx);
    let where_clause = if ty.where_predicates.is_empty() {
        String::new()
    } else {
        format!(" where {}", ty.where_predicates.join(", "))
    };

    match &ty.kind {
        DocTypeKind::Struct { fields, tuple } => {
            if *tuple {
                // Tuple struct: pub struct Name<T>(field1, field2);
                let field_strs: Vec<String> = fields
                    .iter()
                    .map(|f| {
                        let vis = field_vis(&f.visibility);
                        format!("{}{}", vis, render_type(&f.ty, ctx))
                    })
                    .collect();
                out.push_str(&format!(
                    "pub struct {}{}({});\n\n",
                    ty.name,
                    generics_str,
                    field_strs.join(", ")
                ));
            } else {
                out.push_str(&format!(
                    "pub struct {}{}{}\n{{\n",
                    ty.name, generics_str, where_clause
                ));
                for f in fields {
                    let vis = field_vis(&f.visibility);
                    out.push_str(&format!("    {}{},\n", vis, render_field(f, ctx)));
                }
                out.push_str("}\n\n");
            }
        }

        DocTypeKind::Enum { variants } => {
            out.push_str(&format!(
                "pub enum {}{}{}\n{{\n",
                ty.name, generics_str, where_clause
            ));
            for v in variants {
                render_variant(out, v, ctx);
            }
            out.push_str("}\n\n");
        }

        DocTypeKind::Union { fields } => {
            out.push_str(&format!(
                "pub union {}{}{}\n{{\n",
                ty.name, generics_str, where_clause
            ));
            for f in fields {
                let vis = field_vis(&f.visibility);
                out.push_str(&format!("    {}{},\n", vis, render_field(f, ctx)));
            }
            out.push_str("}\n\n");
        }

        DocTypeKind::TypeAlias { rhs } => {
            let rhs_src = render_type(rhs, ctx);
            out.push_str(&format!(
                "pub type {}{} = {};\n\n",
                ty.name, generics_str, rhs_src
            ));
        }

        DocTypeKind::Constant {
            ty: const_ty,
            value,
        } => {
            // Strip the trailing type suffix rustdoc appends to literal
            // values (e.g. "11usize" → "11"), using the known const type so
            // the stripping is exact rather than guesswork.
            let clean_value = strip_type_suffix(value, const_ty);
            out.push_str(&format!(
                "pub const {}: {} = {};\n\n",
                ty.name, const_ty, clean_value
            ));
        }
    }
}

/// Strip the trailing type suffix rustdoc appends to literal constant values
/// (e.g. `"11usize"` → `"11"`, `"truebool"` → `"true"`).
///
/// The expected type is passed in because it is authoritative: rustdoc gives
/// us the const's type separately, so we strip exactly that suffix when the
/// value ends with it. For non-literal values (expressions) no suffix exists
/// and the value is returned unchanged.
fn strip_type_suffix<'a>(value: &'a str, expected_ty: &str) -> &'a str {
    if !expected_ty.is_empty() && value.ends_with(expected_ty) {
        return &value[..value.len() - expected_ty.len()];
    }
    value
}

/// Render an enum variant line.
fn render_variant(out: &mut String, v: &DocVariant, ctx: &RenderCtx) {
    let disc = v
        .discriminant_expr
        .as_deref()
        .map(|d| format!(" = {}", d))
        .unwrap_or_default();

    match &v.kind {
        crate::docjson::model::DocVariantKind::Unit => {
            out.push_str(&format!("    {}{},\n", v.name, disc));
        }
        crate::docjson::model::DocVariantKind::Tuple(fields) => {
            let rendered: Vec<String> = fields.iter().map(|f| render_type(&f.ty, ctx)).collect();
            out.push_str(&format!(
                "    {}{}({},),\n",
                v.name,
                disc,
                rendered.join(", ")
            ));
        }
        crate::docjson::model::DocVariantKind::Struct(fields) => {
            out.push_str(&format!("    {}{} {{\n", v.name, disc));
            for f in fields {
                // Enum variant fields cannot have visibility qualifiers.
                out.push_str(&format!("        {},\n", render_field(f, ctx)));
            }
            out.push_str("    },\n");
        }
    }
}

// ── Rendering context ─────────────────────────────────────────────────────────

/// Context passed through all type-rendering calls. Carries the replacement
/// table, local name set, export table, and the source library for self-routing.
struct RenderCtx<'a> {
    replacements: &'a [(String, Option<String>)],
    local_names: &'a BTreeSet<&'a str>,
    exports: &'a ExportTable,
    /// The source library this file's types belong to ("core", "alloc", "std").
    lib: &'a str,
    /// The module path of the file currently being rendered (e.g., "sync::poison::mutex").
    /// Used to determine whether a resolved route points to the same module.
    current_module: &'a str,
}

// ── Type rendering with routing ───────────────────────────────────────────────

/// Render a field: `name: Type`.
fn render_field(f: &DocField, ctx: &RenderCtx) -> String {
    format!("{}: {}", f.name, render_type(&f.ty, ctx))
}

/// Visibility keyword for a field.
/// All fields in generated bindings are made public. The mirror crate exists
/// so that the main crate can access internal layout details directly; without
/// `pub`, cross-crate field access would be impossible.
fn field_vis(_vis: &DocVisibility) -> String {
    "pub ".to_string()
}

/// Render a wire type as source, applying spec replacements and export-table
/// routing for path references. Local names (types defined in the same file)
/// are left bare.
fn render_type(ty: &crate::docjson::wire::Type, ctx: &RenderCtx) -> String {
    // A spec replacement applies to resolved paths before any routing.
    if let crate::docjson::wire::Type::ResolvedPath(p) = ty {
        if let Some(replaced_base) = try_spec_replacement(&p.path, ctx) {
            if replaced_base.is_empty() {
                return String::new();
            }
            if replaced_base == "()" {
                return "()".to_string();
            }
            let resolver = TypeResolver { ctx };
            let args_str = p
                .args
                .as_ref()
                .map(|a| type_repr::render_args(a, &resolver))
                .unwrap_or_default();
            return format!("{}{}", replaced_base, args_str);
        }
    }

    let resolver = TypeResolver { ctx };
    type_repr::render(ty, &resolver)
}

/// Build the path-resolution callback used by [`type_repr::render`].
///
/// Order of precedence for a resolved path:
/// 1. Spec path replacement (complete substitution, handled upstream).
/// 2. Export-table lookup by (lib, item_id) — authoritative.
/// 3. String-based fallback heuristics for ids with no route entry.
/// The path resolver handed to [`type_repr::render`]: carries the render
/// context and answers every resolved-path reference.
struct TypeResolver<'a> {
    ctx: &'a RenderCtx<'a>,
}

impl type_repr::PathResolver for TypeResolver<'_> {
    fn resolve(&self, path: &str, id: u32) -> String {
        let ctx = self.ctx;
        // Step 1: authoritative export-table lookup by (lib, item_id).
        if let Some(route) = ctx.exports.resolve(ctx.lib, id) {
            // If the route points to a type in the SAME module as the current
            // file, render bare. Compare the resolved route's leaf against
            // local names AND verify the route's module matches ours.
            if route.starts_with("crate::") {
                let route_leaf = route.rsplit("::").next().unwrap_or(route);
                if ctx.local_names.contains(route_leaf) && same_module_as_current(route, ctx) {
                    return route_leaf.to_string();
                }
            }
            return route.to_string();
        }

        // Step 2: string-based fallback heuristics for ids with no route.
        route_fallback(path, ctx)
    }
}

/// Attempt to match a spec path replacement against the raw path.
/// Returns the replacement base path if matched, or None.
fn try_spec_replacement(path: &str, ctx: &RenderCtx) -> Option<String> {
    let path = path.strip_suffix("<>").unwrap_or(path);
    let candidates = build_candidates(path);
    for cand in &candidates {
        for (full_path, replacement) in ctx.replacements {
            if cand == full_path || cand.starts_with(&format!("{}::", full_path)) {
                return Some(match replacement {
                    Some(r) => r.clone(),
                    None => String::new(),
                });
            }
        }
    }
    None
}

/// Build candidate path strings for replacement matching.
fn build_candidates(path: &str) -> Vec<String> {
    let mut cands = vec![path.to_string()];

    // Strip crate-relative prefixes to get the inner path.
    let stripped = path
        .strip_prefix("$crate::")
        .or_else(|| path.strip_prefix("crate::"))
        .unwrap_or(path);

    if stripped != path {
        // Try with each known crate prefix prepended.
        cands.push(format!("core::{}", stripped));
        cands.push(format!("alloc::{}", stripped));
        cands.push(format!("std::{}", stripped));
        cands.push(stripped.to_string());
    } else {
        // Already looks like a crate-qualified path; also try doubling the
        // first segment (handles "alloc::Global" → "alloc::alloc::Global").
        if let Some(first_seg) = stripped.split("::").next() {
            if matches!(first_seg, "core" | "alloc" | "std") {
                cands.push(format!("{}::{}", first_seg, stripped));
            }
        }
    }

    cands
}

/// Fallback routing when the export table has no entry for an id.
fn route_fallback(path: &str, ctx: &RenderCtx) -> String {
    let stripped = path
        .strip_prefix("$crate::")
        .or_else(|| path.strip_prefix("crate::"))
        .unwrap_or(path);

    // Local names stay bare (no qualification).
    let leaf = stripped.rsplit("::").next().unwrap_or(stripped);
    if ctx.local_names.contains(leaf) {
        return leaf.to_string();
    }

    // Explicit crate-qualified paths.
    if let Some(rest) = stripped.strip_prefix("core::") {
        return format!("::__rustyfill_builtin_core::{}", rest);
    }
    if let Some(rest) = stripped.strip_prefix("alloc::") {
        return format!("::__rustyfill_builtin_alloc::{}", rest);
    }
    if let Some(rest) = stripped.strip_prefix("std::") {
        return format!("::__rustyfill_builtin_std::{}", rest);
    }

    // Crate-relative paths (had crate::/$crate:: prefix): route through
    // our mirror tree.
    if path.starts_with("crate::") || path.starts_with("$crate::") {
        return format!("crate::{WRAPPER_MOD}::{}", stripped);
    }

    // Bare module-qualified paths: these are intra-crate references that
    // rustdoc didn't fully qualify. Route through the mirror tree.
    if stripped.contains("::") {
        return format!("crate::{WRAPPER_MOD}::{}", stripped);
    }

    // Single-segment unknown: pass through (likely a primitive alias or
    // external type we can't resolve).
    stripped.to_string()
}

/// Check whether a resolved route points to the same module as the current file.
/// The route looks like `crate::std::sync::poison::mutex::Mutex` and our
/// current_module is `sync::poison::mutex`. We strip the `crate::std::` prefix
/// from the route, remove the leaf type name, and compare.
fn same_module_as_current(route: &str, ctx: &RenderCtx) -> bool {
    // Route format: crate::std::<module>::<TypeName>
    let stripped = route.strip_prefix("crate::").unwrap_or(route);
    let stripped = stripped.strip_prefix(WRAPPER_MOD).unwrap_or(stripped);
    let stripped = stripped.strip_prefix("::").unwrap_or(stripped);
    // Remove the last segment (type name) to get the module portion.
    let route_module = match stripped.rfind("::") {
        Some(pos) => &stripped[..pos],
        None => "",
    };

    route_module == ctx.current_module
}

// ── Generics rendering ────────────────────────────────────────────────────────

/// Render the `<...>` generics clause for a type definition.
fn render_generics(params: &[DocGenericParam], ignored_traits: &[String], ctx: &RenderCtx) -> String {
    if params.is_empty() {
        return String::new();
    }

    let parts: Vec<String> = params
        .iter()
        .map(|p| render_generic_param(p, ignored_traits, ctx))
        .collect();
    format!("<{}>", parts.join(", "))
}

fn render_generic_param(p: &DocGenericParam, ignored_traits: &[String], ctx: &RenderCtx) -> String {
    if p.is_lifetime {
        let outlives = if p.outlives.is_empty() {
            String::new()
        } else {
            format!(": {}", p.outlives.join(" + "))
        };
        format!("'{}{}", p.name.trim_start_matches('\''), outlives)
    } else if p.is_const {
        let ty = p.const_ty.as_deref().unwrap_or("usize");
        format!("const {}: {}", p.name, ty)
    } else {
        let mut s = p.name.clone();
        if !p.bounds.is_empty() {
            let filtered: Vec<String> = p
                .bounds
                .iter()
                .filter(|b| !INTERNAL_TRAIT_STRIPS.contains(&b.as_str()))
                .filter(|b| !ignored_traits.iter().any(|ig| ig.as_str() == b.as_str()))
                .cloned()
                .collect();
            if !filtered.is_empty() {
                s.push_str(": ");
                s.push_str(&filtered.join(" + "));
            }
        }
        if let Some(default_ty) = &p.default_type {
            let rendered = render_type(default_ty, ctx);
            s.push_str(&format!(" = {}", rendered));
        } else if let Some(default_val) = &p.default_value {
            s.push_str(&format!(" = {}", default_val));
        }
        s
    }
}

// ── Replacement lookup ────────────────────────────────────────────────────────

/// Build the flat replacement list from the spec: (full_path, replacement).
fn build_replacement_lookup(spec: &LoaderSpec) -> Vec<(String, Option<String>)> {
    spec.targets
        .iter()
        .flat_map(|t| t.path_replacements.iter())
        .map(|pr| (pr.path.clone(), pr.replacement.clone()))
        .collect()
}

// ── File layout helpers ───────────────────────────────────────────────────────

/// Compute the relative file path (slash-separated, no `.rs`) for a type.
fn type_rel_path(ty: &DocType) -> String {
    if ty.module_path.is_empty() {
        // Root-level type: emit at the library root as `<name>.rs`
        ty.name.to_lowercase()
    } else {
        // The type lives in its module's file: `collections/btree/map.rs`
        ty.module_path.replace("::", "/")
    }
}

/// Get child module names (direct children of this module).
fn get_children(rel_path: &str, all_files: &[String]) -> Vec<String> {
    let Some(my_mp) = ModulePath::from_slash(rel_path) else {
        return Vec::new();
    };
    let mut children = BTreeSet::new();
    for fp in all_files {
        let Some(other) = ModulePath::from_slash(fp) else {
            continue;
        };
        if my_mp.is_direct_parent_of(&other) {
            children.insert(other.leaf().to_string());
        }
    }
    children.into_iter().collect()
}

// ── Known-type stubs ──────────────────────────────────────────────────────────

/// Emit a standalone file carrying hand-written stub definitions for known
/// external types that share a module. Returns the relative path written.
fn emit_known_type_stubs(
    out_dir: &Path,
    kts: &[&crate::loader_spec::KnownExternalType],
) -> Option<String> {
    let first = kts.first()?;
    let segments: Vec<&str> = first.path.split("::").collect();
    if segments.len() < 2 {
        return None;
    }
    let module_slash: String = segments[..segments.len() - 1].join("/");
    let rel_path = format!("{module_slash}.rs");

    let mut content = String::from("// Auto-generated by rustyfill-sys.\n\n");
    for kt in kts {
        content.push_str(&kt.definition);
        content.push('\n');
    }

    let path = out_dir.join(&rel_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let content = format_source(&content);
    std::fs::write(&path, &content)
        .unwrap_or_else(|e| panic!("Failed to write known-type stub {}: {}", path.display(), e));
    Some(rel_path)
}

// ── Manifest ──────────────────────────────────────────────────────────────────

/// Emit the hierarchical manifest (`bindings_generated.rs`) that declares the
/// module tree and includes every generated file.
fn emit_manifest(out_dir: &Path, all_files: &[String]) {
    let manifest_path = out_dir.join("bindings_generated.rs");
    let mut content = String::new();

    content.push_str(
        "// Auto-generated manifest by rustyfill-sys.\n\
         // Hierarchical module tree mirroring std/core/alloc structure.\n\
         // All types are intentionally public.\n\n",
    );

    content.push_str(&format!("pub mod {WRAPPER_MOD} {{\n"));

    // Build the set of modules that have children (i.e., need mod.rs).
    let modules_with_children: BTreeSet<String> = {
        let mut set = BTreeSet::new();
        for fp in all_files {
            let segments: Vec<&str> = fp.split('/').filter(|s| !s.is_empty()).collect();
            for i in 0..segments.len().saturating_sub(1) {
                let prefix: Vec<&str> = segments[..=i].to_vec();
                set.insert(prefix.join("/"));
            }
        }
        set
    };

    // Build the tree and emit recursively.
    let mut tree: BTreeMap<String, ManifestNode> = BTreeMap::new();
    for fp in all_files {
        insert_into_tree(&mut tree, fp);
    }

    // Set of all actual file paths for checking existence.
    let all_files_set: BTreeSet<&str> = all_files.iter().map(|s| s.as_str()).collect();

    for (name, node) in &tree {
        // Top-level segments are already full paths (e.g., "boxed", "collections").
        emit_manifest_node(
            &mut content,
            name,
            node,
            1,
            &modules_with_children,
            &all_files_set,
        );
    }

    content.push_str("}\n\n");

    let content = format_source(&content);
    std::fs::write(&manifest_path, content).unwrap_or_else(|e| {
        panic!(
            "Failed to write manifest {}: {}",
            manifest_path.display(),
            e
        )
    });
}

/// A node in the manifest tree.
#[derive(Default)]
struct ManifestNode {
    children: BTreeMap<String, ManifestNode>,
}

/// Insert a file path into the manifest tree.
fn insert_into_tree(tree: &mut BTreeMap<String, ManifestNode>, rel_path: &str) {
    let segments: Vec<&str> = rel_path.split('/').filter(|s| !s.is_empty()).collect();
    if segments.is_empty() {
        return;
    }

    let mut current: &mut BTreeMap<String, ManifestNode> = tree;
    for seg in &segments {
        let node = current.entry((*seg).to_string()).or_default();
        current = &mut node.children;
    }
}

/// Recursively emit a manifest node with its full path context.
fn emit_manifest_node(
    content: &mut String,
    full_path: &str,
    node: &ManifestNode,
    indent_level: usize,
    modules_with_children: &BTreeSet<String>,
    all_files_set: &BTreeSet<&str>,
) {
    let indent = "    ".repeat(indent_level);
    let inner_indent = "    ".repeat(indent_level + 1);
    let module_name = full_path.rsplit('/').next().unwrap_or(full_path);

    content.push_str(&format!("{}pub mod {} {{\n", indent, module_name));
    content.push_str(&format!("{}    #![allow(warnings)]\n", indent));

    // Include directive for this module's own content — only if a file exists.
    let has_own_file = all_files_set.contains(full_path);
    if has_own_file {
        let include_path = if modules_with_children.contains(full_path) {
            format!("{}/mod.rs", full_path)
        } else {
            format!("{}.rs", full_path)
        };
        content.push_str(&format!(
            "{}    include!(concat!(env!(\"OUT_DIR\"), \"/{}\"));\n",
            inner_indent, include_path
        ));
    }

    // Recurse into children.
    for (child_name, child_node) in &node.children {
        let child_full_path = format!("{}/{}", full_path, child_name);
        emit_manifest_node(
            content,
            &child_full_path,
            child_node,
            indent_level + 1,
            modules_with_children,
            all_files_set,
        );
    }

    content.push_str(&format!("{}}}\n", indent));
}
