//! Emits parsed items back into valid Rust source code, generates alias files,
//! and builds the hierarchical module manifest.
//!
//! Takes `ParsedItem`s and regenerates them as compilable Rust, preserving
//! all attributes, repr attributes, visibility, generics, and field layout.
//!
//! **Preamble strategy**: each target library (std, core, alloc) gets its own
//! preamble module emitted at the library root (e.g., `std::__prelude`). Each
//! generated binding file imports from it via a known path like
//! `use crate::<target>::__prelude::*;`. This avoids namespace clashes with
//! local modules (e.g., btree's internal `mod marker` shadowing `core::marker`).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::path::Path;

use proc_macro2::{Span, TokenStream, TokenTree};
use quote::ToTokens;
use syn::{punctuated::Punctuated, Ident, Generics, ItemStruct, Type};

use crate::formatter::format_source;
use crate::parser::{ItemKind, ItemVisibility, ParsedItem};
use crate::resolver::ModuleResolver;

// ── Type registry and field-type publicity checking ────────────────────────

/// Outcome of resolving one type reference inside a declared struct's fields.
#[derive(Clone, Debug)]
pub enum FieldRefResolution {
    /// The referenced type is public and undeclared — keep referring to the
    /// original (real core/alloc/std) type through the builtin extern crate.
    Original(String),
    /// The referenced type is itself declared — the binding must point at the
    /// mirrored definition instead of the original.
    Mirrored(String),
    /// The referenced type is private and undeclared — this is an error; the
    /// build script collects these and fails fast.
    UndeclaredPrivate(String),
    /// The reference does not name a known std type (primitive, generic
    /// parameter, or external crate item) — leave it untouched.
    Unknown(String),
}

/// Information about a single named type reachable from the registered
/// source files.
#[derive(Clone, Debug)]
pub struct TypeInfo {
    /// Canonical module path (`::`-separated, relative to the library root).
    pub canonical_path: String,
    /// Source visibility as written in std.
    pub visibility: ItemVisibility,
    /// Whether the item is re-exported publicly through its module chain
    /// (i.e., usable by name outside its defining module).
    pub is_exported: bool,
    /// File path (relative to the library src root) where the item is defined.
    pub def_file: String,
    /// Whether this type is explicitly declared in the loader spec.
    pub declared: bool,
    /// For type aliases: the right-hand-side type expression of the alias
    /// (`type Root<K, V> = NodeRef<...>` → RHS tokens). Present only for
    /// [`ItemKind::TypeAlias`] entries.
    pub alias_rhs: Option<TokenStream>,
}

impl TypeInfo {
    /// True when the type may legally be referred to from bindings: it is
    /// either explicitly declared in the spec or public in the source.
    pub fn is_usable(&self) -> bool {
        self.visibility.is_public() || self.declared
    }

    /// True when the binding should point at the mirrored definition rather
    /// than the original type.
    pub fn is_declared(&self) -> bool {
        self.declared
    }

    /// The leaf identifier of the type.
    pub fn leaf(&self) -> &str {
        self.canonical_path.rsplit("::").next().unwrap_or(&self.canonical_path)
    }
}

/// Registry of all named types discovered in the std source tree plus the set
/// of types explicitly declared in the loader spec. Built once by the build
/// script after discovery, then handed to the emitter for both field-type
/// publicity checking and reference rewriting.
pub struct TypeRegistry {
    /// All known types indexed by canonical path (`lib::module::Leaf`).
    by_path: HashMap<String, TypeInfo>,
    /// Known types indexed by leaf name.
    by_leaf: HashMap<String, Vec<String>>,
    /// Types explicitly declared in the spec, with their canonical paths.
    declared: HashSet<String>,
    /// Name of the manifest wrapper module that all mirrored bindings live
    /// under (e.g., `std`). Mirror references are emitted as
    /// `crate::{wrapper_mod}::<path-without-lib-prefix>`. Defaults to
    /// [`WRAPPER_MOD`].
    wrapper_mod: String,
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self {
            by_path: HashMap::new(),
            by_leaf: HashMap::new(),
            declared: HashSet::new(),
            wrapper_mod: WRAPPER_MOD.to_string(),
        }
    }
}

impl TypeRegistry {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Set the manifest wrapper module name used when emitting mirror paths.
    pub fn set_wrapper_mod(&mut self, name: &str) {
        self.wrapper_mod = name.to_string();
    }

    /// The manifest wrapper module name for mirror-path emission.
    pub fn wrapper_mod(&self) -> &str {
        &self.wrapper_mod
    }

    /// Register a type discovered in a source file.
    pub fn register(
        &mut self,
        canonical_path: &str,
        visibility: ItemVisibility,
        is_exported: bool,
        def_file: &str,
    ) {
        let leaf = canonical_path.rsplit("::").next().unwrap_or(canonical_path);
        self.by_leaf.entry(leaf.to_string()).or_default().push(canonical_path.to_string());
        let info = self
            .by_path
            .entry(canonical_path.to_string())
            .or_insert_with(|| TypeInfo {
                canonical_path: canonical_path.to_string(),
                visibility,
                is_exported,
                def_file: def_file.to_string(),
                declared: false,
                alias_rhs: None,
            });
        // If seen multiple times (e.g. via inline module + file), prefer the
        // more informative record.
        if !info.is_exported && is_exported {
            info.is_exported = true;
        }
        if !matches!(info.visibility, ItemVisibility::Public)
            && matches!(visibility, ItemVisibility::Public)
        {
            info.visibility = visibility;
        }
    }

    /// Record the right-hand side of a type alias so that emission can mirror
    /// the alias definition itself (rewriting its RHS through the registry).
    pub fn set_alias_rhs(&mut self, canonical_path: &str, rhs: TokenStream) {
        if let Some(info) = self.by_path.get_mut(canonical_path) {
            info.alias_rhs = Some(rhs);
        }
    }

    /// Mark a canonical path as explicitly declared in the spec.
    ///
    /// The `def_file` here is authoritative (typically an absolute path) and
    /// overrides whatever def_file was recorded during discovery, because the
    /// field-publicity check reads the source file from this path.
    pub fn insert_declared(&mut self, canonical_path: &str, def_file: &str) {
        self.register(canonical_path, ItemVisibility::Public, true, def_file);
        if let Some(info) = self.by_path.get_mut(canonical_path) {
            info.declared = true;
            // Force-update def_file: `register` uses or_insert_with, so a
            // previously-discovered record (with a relative def_file) would
            // otherwise win. The declared def_file is the one the checker needs.
            info.def_file = def_file.to_string();
        }
        self.declared.insert(canonical_path.to_string());
    }

    /// Look up a type by canonical path.
    pub fn get(&self, canonical_path: &str) -> Option<&TypeInfo> {
        self.by_path.get(canonical_path)
    }

    /// Candidate canonical paths for a bare leaf name (may be ambiguous).
    pub fn candidates_for_leaf(&self, leaf: &str) -> &[String] {
        self.by_leaf.get(leaf).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Resolve a type reference appearing in a declared struct's fields.
    ///
    /// Rules:
    /// - Declared (in the spec) → `Mirrored`: bindings point at the mirror.
    /// - Public in source and not declared → `Original`: refer to the real type.
    /// - Private/restricted and not declared → `UndeclaredPrivate`: error.
    /// - Unknown (primitive, generic param, foreign) → `Unknown`: leave alone.
    pub fn resolve_field_ref(&self, leaf: &str) -> FieldRefResolution {
        self.resolve_field_ref_in(leaf, "")
    }

    /// Resolve a type reference with module context. When multiple modules
    /// declare the same leaf name (e.g., `map::entry::VacantEntry` and
    /// `set::entry::VacantEntry`), the candidate whose canonical path shares
    /// the longest suffix with `module_context` is preferred. This prevents
    /// cross-module misrouting where a bare `VacantEntry` in map/entry.rs
    /// would otherwise resolve to the set-side definition.
    pub fn resolve_field_ref_in(&self, leaf: &str, module_context: &str) -> FieldRefResolution {
        let guard = LocalNameGuard::new(None);
        self.resolve_with_guard(leaf, module_context, &guard)
    }

    /// Resolve a type reference with a [`LocalNameGuard`] that marks certain
    /// names as "local" (should not be rewritten). Used by the emission
    /// pipeline to prevent bare references to file-local types or generic
    /// type parameters from being misrouted to cross-module mirrors.
    fn resolve_with_guard(
        &self,
        leaf: &str,
        module_context: &str,
        guard: &LocalNameGuard<'_>,
    ) -> FieldRefResolution {
        let candidates = self.candidates_for_leaf(leaf);
        if candidates.is_empty() {
            // Not a known std type — primitive, generic parameter, or external
            // crate item. Leave the reference untouched.
            return FieldRefResolution::Unknown(leaf.to_string());
        }

        // Local-first: if this name is marked as local (defined in this file
        // or is a generic type parameter), leave it bare.
        if guard.contains(leaf) {
            return FieldRefResolution::Unknown(leaf.to_string());
        }

        // Score each candidate by how much its canonical path overlaps with
        // the current module context. Longer shared suffix = closer module.
        let scored: Vec<(usize, &String)> = candidates
            .iter()
            .map(|p| (suffix_overlap(p, module_context), p))
            .collect();
        let mut scored_sorted = scored;
        scored_sorted.sort_by_key(|b| std::cmp::Reverse(b.0));

        // Among the highest-scoring candidates, prefer declared, then exported.
        let max_score = scored_sorted.first().map(|(s, _)| *s).unwrap_or(0);
        let top: Vec<&String> = scored_sorted
            .iter()
            .filter(|(s, _)| *s == max_score)
            .map(|(_, p)| *p)
            .collect();

        let chosen = top
            .iter()
            .find(|p| self.declared.contains(**p))
            .or_else(|| top.iter().find(|p| {
                self.by_path
                    .get(p.as_str())
                    .is_some_and(|t| t.is_exported)
            }))
            .copied()
            .unwrap_or(top[0]);

        match self.by_path.get(chosen) {
            Some(info) if info.is_declared() => FieldRefResolution::Mirrored(chosen.clone()),
            Some(info) if info.is_usable() => FieldRefResolution::Original(chosen.clone()),
            Some(_) => FieldRefResolution::UndeclaredPrivate(chosen.clone()),
            None => FieldRefResolution::Unknown(leaf.to_string()),
        }
    }

    /// Iterate over all declared types.
    pub fn declared_paths(&self) -> impl Iterator<Item = &String> {
        self.declared.iter()
    }

    /// True when a type at `lib_name::{module_path}::{leaf}` is explicitly
    /// declared in the loader spec. Used by the emitter to restrict output to
    /// declared data structures only, so peripheral public items that merely sit
    /// alongside them (iterators, cursors, range views, …) are not mirrored
    /// unless they are part of the polyfill's core surface.
    pub fn is_declared_in_module(&self, lib_name: &str, module_path: &str, leaf: &str) -> bool {
        let mut canonical = String::from(lib_name);
        if !module_path.is_empty() {
            canonical.push_str("::");
            canonical.push_str(module_path);
        }
        canonical.push_str("::");
        canonical.push_str(leaf);
        self.declared.contains(&canonical)
    }
}

/// Number of trailing `::`-separated segments that two paths share. Used to
/// score candidate types by proximity to the current module context.
fn suffix_overlap(a: &str, b: &str) -> usize {
    if b.is_empty() {
        return 0;
    }
    let a_segs: Vec<&str> = a.split("::").collect();
    let b_segs: Vec<&str> = b.split("::").collect();
    let mut count = 0;
    for (x, y) in a_segs.iter().rev().zip(b_segs.iter().rev()) {
        if x == y {
            count += 1;
        } else {
            break;
        }
    }
    count
}

impl TypeRegistry {
    /// Total number of registered types.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.by_path.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.by_path.is_empty()
    }
}

/// Check every field type of every declared struct against the registry.
/// Returns a list of human-readable errors (empty when all checks pass).
pub fn check_declared_struct_fields(registry: &TypeRegistry) -> Vec<String> {
    let mut errors = Vec::new();
    for path in registry.declared_paths() {
        let Some(info) = registry.get(path) else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(&info.def_file) else {
            continue;
        };
        let leaf = info.leaf();
        let Ok(file) = syn::parse_file(&text) else {
            continue;
        };
        for item in &file.items {
            match item {
                syn::Item::Struct(s) if s.ident == leaf => {
                    collect_fields(&s.fields, registry, path, &mut errors);
                }
                syn::Item::Enum(e) if e.ident == leaf => {
                    // Check variant payloads too.
                    for v in &e.variants {
                        collect_fields(&v.fields, registry, path, &mut errors);
                    }
                }
                // A declared type alias is checked through its right-hand side:
                // every type the alias expands to must itself be usable.
                syn::Item::Type(t) if t.ident == leaf => {
                    check_alias_rhs(&t.ty, registry, path, &mut errors);
                }
                _ => {}
            }
        }
    }
    errors
}

/// Check that a declared type alias's RHS only references usable types.
fn check_alias_rhs(
    ty: &syn::Type,
    registry: &TypeRegistry,
    owner: &str,
    errors: &mut Vec<String>,
) {
    for ty_name in type_leaves(ty) {
        if let FieldRefResolution::UndeclaredPrivate(resolved) = registry.resolve_field_ref(&ty_name)
        {
            errors.push(format!(
                "[fields] type alias `{owner}` expands to `{ty_name}`, which resolves \
                 to `{resolved}` — a non-public, undeclared type. Declare it in the spec \
                 (declare_struct) or make sure it is publicly exported."
            ));
        }
    }
}

fn field_iter(fields: &syn::Fields) -> Box<dyn Iterator<Item = &syn::Field> + '_> {
    match fields {
        syn::Fields::Named(named) => Box::new(named.named.iter()),
        syn::Fields::Unnamed(unnamed) => Box::new(unnamed.unnamed.iter()),
        syn::Fields::Unit => Box::new(std::iter::empty()),
    }
}

fn collect_fields(
    fields: &syn::Fields,
    registry: &TypeRegistry,
    owner: &str,
    errors: &mut Vec<String>,
) {
    for f in field_iter(fields) {
        for ty in type_leaves(&f.ty) {
            if let FieldRefResolution::UndeclaredPrivate(resolved) = registry.resolve_field_ref(&ty)
            {
                errors.push(format!(
                    "[fields] `{owner}` has a field of type `{ty}`, which resolves to \
                     `{resolved}` — a non-public, undeclared type. Declare it in the spec \
                     (declare_struct) or make sure it is publicly exported."
                ));
            }
        }
    }
}

/// Extract the base type names referenced by a type expression: the first
/// segment of each path type, skipping lifetimes, references, and pointers.
fn type_leaves(ty: &Type) -> Vec<String> {
    let mut out = Vec::new();
    visit_type(ty, &mut out);
    out
}

fn visit_type(ty: &Type, out: &mut Vec<String>) {
    match ty {
        Type::Path(tp) => {
            if let Some(seg) = tp.path.segments.first() {
                out.push(seg.ident.to_string());
            }
            // Recurse into generic arguments so that nested type references
            // (e.g., `Root` inside `Option<Root<K, V>>`) are also checked.
            for seg in &tp.path.segments {
                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                    for arg in &args.args {
                        if let syn::GenericArgument::Type(inner) = arg {
                            visit_type(inner, out);
                        }
                    }
                }
            }
        }
        Type::Reference(tr) => visit_type(&tr.elem, out),
        Type::Ptr(tp) => visit_type(&tp.elem, out),
        Type::Tuple(tt) => {
            for elem in &tt.elems {
                visit_type(elem, out);
            }
        }
        Type::Slice(ts) => visit_type(&ts.elem, out),
        Type::Array(ta) => visit_type(&ta.elem, out),
        Type::Paren(tp) => visit_type(&tp.elem, out),
        Type::Group(tg) => visit_type(&tg.elem, out),
        _ => {}
    }
}

// ── Drop-safety field annotation ────────────────────────────────────────────

/// True when the consuming crate opted into drop annotations (the
/// `drop-annotations` feature of `rustyfill-sys`). The build script runs
/// with `CARGO_FEATURE_DROP_ANNOTATIONS` set in its environment when the
/// feature is active.
fn drop_annotations_enabled() -> bool {
    std::env::var_os("CARGO_FEATURE_DROP_ANNOTATIONS").is_some()
}

/// Value of the emitted `#[rustyfill_drop]` field attribute when the field
/// WILL be dropped: its type is fully realized (references no declared/
/// polyfilled type and no generic parameter), so it carries real drop glue
/// that must run on destruction.
const DROP_WILL_DROP: &str = "yes";
/// Value of the emitted `#[rustyfill_drop]` field attribute when the field
/// MAY NOT be dropped: its type contains at least one declared (polyfilled)
/// type whose mirror is layout-only scaffolding without real drop glue, or
/// it involves a generic parameter whose concrete instantiation is unknown
/// to the emitter. Downstream crates must not assume these fields are ever
/// dropped.
const DROP_MAYBE_NO: &str = "maybe-no";

/// Classify a field's type for drop safety:
/// - `DROP_WILL_DROP` (`"yes"`) when the type is fully realized — it names
///   no declared (polyfilled) type and no generic parameter, so real drop
///   glue will run;
/// - `DROP_MAYBE_NO` (`"maybe-no"`) otherwise — it references at least one
///   polyfilled type (whose mirror has no drop glue) or mentions a generic
///   parameter (always wins, since the concrete instantiation is unknown).
fn classify_field_drop(ty: &syn::Type, registry: &TypeRegistry, guard: &LocalNameGuard<'_>) -> &'static str {
    let mut has_polyfill = false;
    let mut has_generic = false;
    visit_drop_classification(ty, registry, guard, &mut has_polyfill, &mut has_generic);
    if has_polyfill || has_generic {
        DROP_MAYBE_NO
    } else {
        DROP_WILL_DROP
    }
}

/// Walk a type expression tracking whether it mentions any declared (mirrored)
/// type and whether it mentions any generic parameter. Generic parameters are
/// recognized from the enclosing item's own generics (via the local-name
/// guard); bare identifiers that resolve to neither stay neutral.
fn visit_drop_classification(
    ty: &syn::Type,
    registry: &TypeRegistry,
    guard: &LocalNameGuard<'_>,
    has_polyfill: &mut bool,
    has_generic: &mut bool,
) {
    match ty {
        Type::Path(tp) => {
            if let Some(seg) = tp.path.segments.first() {
                let name = seg.ident.to_string();
                if guard.contains(&name) {
                    *has_generic = true;
                } else if matches!(
                    registry.resolve_field_ref(&name),
                    FieldRefResolution::Mirrored(_)
                ) {
                    *has_polyfill = true;
                }
            }
            // Recurse into generic arguments (`Option<Root<K, V>>`).
            for seg in &tp.path.segments {
                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                    for arg in &args.args {
                        if let syn::GenericArgument::Type(inner) = arg {
                            visit_drop_classification(inner, registry, guard, has_polyfill, has_generic);
                        }
                    }
                }
            }
        }
        Type::Reference(tr) => visit_drop_classification(&tr.elem, registry, guard, has_polyfill, has_generic),
        Type::Ptr(tp) => visit_drop_classification(&tp.elem, registry, guard, has_polyfill, has_generic),
        Type::Tuple(tt) => {
            for elem in &tt.elems {
                visit_drop_classification(elem, registry, guard, has_polyfill, has_generic);
            }
        }
        Type::Slice(ts) => visit_drop_classification(&ts.elem, registry, guard, has_polyfill, has_generic),
        Type::Array(ta) => visit_drop_classification(&ta.elem, registry, guard, has_polyfill, has_generic),
        Type::Paren(tp) => visit_drop_classification(&tp.elem, registry, guard, has_polyfill, has_generic),
        Type::Group(tg) => visit_drop_classification(&tg.elem, registry, guard, has_polyfill, has_generic),
        _ => {}
    }
}

/// Build a doc-comment annotation carrying the drop-safety classification.
/// Emits `/// rustyfill-drop: <value>` which is always valid on any field
/// position and requires no proc-macro or compiler feature.
fn drop_doc_comment(value: &str) -> syn::Attribute {
    let text = format!("rustyfill-drop: {}", value);
    syn::parse_quote!(#[doc = #text])
}

/// Marker prefix used to identify our drop-safety doc comments among a
/// field's existing attributes.
const DROP_DOC_PREFIX: &str = "rustyfill-drop:";

/// Check whether a field already carries a drop-safety doc comment.
fn has_drop_annotation(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        a.path().is_ident("doc")
            && a.meta.require_name_value().ok().is_some_and(|nv| {
                nv.value
                    .to_token_stream()
                    .to_string()
                    .trim_matches('"')
                    .starts_with(DROP_DOC_PREFIX)
            })
    })
}

/// Annotate named fields of a struct/enum/union node with its drop-safety
/// classification before serialization. Tuple (unnamed) fields are skipped
/// since doc comments on tuple positions are less discoverable.
fn annotate_fields_drop(fields: &mut syn::Fields, registry: &TypeRegistry, guard: &LocalNameGuard<'_>) {
    if let syn::Fields::Named(named) = fields {
        for f in named.named.iter_mut() {
            annotate_one_field(f, registry, guard);
        }
    }
}

fn annotate_one_field(f: &mut syn::Field, registry: &TypeRegistry, guard: &LocalNameGuard<'_>) {
    if !drop_annotations_enabled() {
        return;
    }
    let value = classify_field_drop(&f.ty, registry, guard);
    if !has_drop_annotation(&f.attrs) {
        f.attrs.push(drop_doc_comment(value));
    }
}

/// Rewrite the tokens of a single struct/enum/union item so that every
fn rewrite_fields_named(fields: &mut syn::FieldsNamed, registry: &TypeRegistry, module_ctx: &str, guard: &LocalNameGuard<'_>) {
    for f in fields.named.iter_mut() {
        f.ty = rewrite_type(f.ty.clone(), registry, module_ctx, guard);
    }
}

fn rewrite_fields(fields: &mut syn::Fields, registry: &TypeRegistry, module_ctx: &str, guard: &LocalNameGuard<'_>) {
    match fields {
        syn::Fields::Named(named) => rewrite_fields_named(named, registry, module_ctx, guard),
        syn::Fields::Unnamed(unnamed) => {
            for f in unnamed.unnamed.iter_mut() {
                f.ty = rewrite_type(f.ty.clone(), registry, module_ctx, guard);
            }
        }
        syn::Fields::Unit => {}
    }
}

fn rewrite_struct_node(mut node: ItemStruct, registry: &TypeRegistry, module_ctx: &str, base_guard: &LocalNameGuard<'_>) -> TokenStream {
    // Extend the guard with this struct's own generic type parameter names so
    // that field types referencing them (e.g., `ManuallyDrop<A>` where A is a
    // generic param) are classified as `maybe-no` rather than `yes`.
    let gen_names: Vec<String> = node.generics.type_params().map(|p| p.ident.to_string()).collect();
    let guard = LocalNameGuard::new(base_guard.file_local).with_generics(&gen_names);
    node.generics = rewrite_generics(node.generics, registry, module_ctx, &guard);
    if let syn::Fields::Named(named) = &mut node.fields {
        rewrite_fields_named(named, registry, module_ctx, &guard);
        for f in named.named.iter_mut() {
            annotate_one_field(f, registry, &guard);
        }
    } else {
        // Tuple/unit structs: no drop annotations possible on unnamed fields.
        rewrite_fields(&mut node.fields, registry, module_ctx, &guard);
    }
    let mut ts = TokenStream::new();
    node.to_tokens(&mut ts);
    ts
}

fn rewrite_enum_node(mut node: syn::ItemEnum, registry: &TypeRegistry, module_ctx: &str, base_guard: &LocalNameGuard<'_>) -> TokenStream {
    // Extend the guard with this enum's own generic type parameter names so
    // that variant fields referencing them (e.g., ForceResult<Leaf, Internal>'s
    // `Leaf(Leaf)`) are left bare instead of being routed to a same-named type
    // in another module.
    let gen_names: Vec<String> = node.generics.type_params().map(|p| p.ident.to_string()).collect();
    let guard = LocalNameGuard::new(base_guard.file_local).with_generics(&gen_names);
    node.generics = rewrite_generics(node.generics, registry, module_ctx, &guard);
    for v in &mut node.variants {
        rewrite_fields(&mut v.fields, registry, module_ctx, &guard);
        annotate_fields_drop(&mut v.fields, registry, &guard);
    }
    let mut ts = TokenStream::new();
    node.to_tokens(&mut ts);
    ts
}

/// A view over "names that should not be rewritten" — the union of file-local
/// type definitions and an item's own generic type parameters. Used to prevent
/// bare references to type parameters (e.g., `ForceResult<Leaf, Internal>`'s
/// variant field `Leaf`) from being misrouted to same-named types in other
/// modules.
struct LocalNameGuard<'a> {
    file_local: Option<&'a [&'a str]>,
    generics: Vec<String>,
}

impl<'a> LocalNameGuard<'a> {
    fn new(file_local: Option<&'a [&'a str]>) -> Self {
        Self { file_local, generics: Vec::new() }
    }
    fn with_generics(mut self, gen_names: &[String]) -> Self {
        for g in gen_names {
            if !self.generics.contains(g) {
                self.generics.push(g.clone());
            }
        }
        self
    }
    fn contains(&self, name: &str) -> bool {
        self.file_local.is_some_and(|n| n.contains(&name))
            || self.generics.iter().any(|g| g == name)
    }
}

fn rewrite_union_node(mut node: syn::ItemUnion, registry: &TypeRegistry, module_ctx: &str, base_guard: &LocalNameGuard<'_>) -> TokenStream {
    let gen_names: Vec<String> = node.generics.type_params().map(|p| p.ident.to_string()).collect();
    let guard = LocalNameGuard::new(base_guard.file_local).with_generics(&gen_names);
    node.generics = rewrite_generics(node.generics, registry, module_ctx, &guard);
    rewrite_fields_named(&mut node.fields, registry, module_ctx, &guard);
    for f in node.fields.named.iter_mut() {
        annotate_one_field(f, registry, &guard);
    }
    let mut ts = TokenStream::new();
    node.to_tokens(&mut ts);
    ts
}

fn rewrite_generics(generics: Generics, registry: &TypeRegistry, module_ctx: &str, guard: &LocalNameGuard<'_>) -> Generics {
    let mut out = Generics::default();
    out.lt_token = generics.lt_token;
    out.gt_token = generics.gt_token;
    for p in generics.params {
        let rewritten = match p {
            syn::GenericParam::Lifetime(l) => syn::GenericParam::Lifetime(l),
            syn::GenericParam::Type(mut t) => {
                t.bounds = t
                    .bounds
                    .into_iter()
                    .map(|b| rewrite_generic_bound(b, registry, module_ctx, guard))
                    .collect();
                syn::GenericParam::Type(t)
            }
            syn::GenericParam::Const(c) => syn::GenericParam::Const(c),
        };
        out.params.push(rewritten);
    }
    if let Some(where_clause) = generics.where_clause {
        let mut predicates = Punctuated::new();
        for pred in where_clause.predicates {
            let rewritten = match pred {
                syn::WherePredicate::Type(mut pt) => {
                    pt.bounded_ty = rewrite_type(pt.bounded_ty, registry, module_ctx, guard);
                    pt.bounds = pt
                        .bounds
                        .into_iter()
                        .map(|b| rewrite_generic_bound(b, registry, module_ctx, guard))
                        .collect();
                    syn::WherePredicate::Type(pt)
                }
                other => other,
            };
            predicates.push(rewritten);
        }
        out.where_clause = Some(syn::WhereClause {
            where_token: where_clause.where_token,
            predicates,
        });
    }
    out
}

fn rewrite_generic_bound(
    bound: syn::TypeParamBound,
    registry: &TypeRegistry,
    module_ctx: &str,
    guard: &LocalNameGuard<'_>,
) -> syn::TypeParamBound {
    match bound {
        syn::TypeParamBound::Trait(mut t) => {
            t.path = rewrite_path(t.path, registry, module_ctx, guard);
            syn::TypeParamBound::Trait(t)
        }
        other => other,
    }
}

/// Rewrite a type expression: declared types are routed to their mirrored
/// bindings (absolute `crate::` paths into the synthetic tree), public but
/// undeclared types are routed to the original builtin crate, everything
/// else passes through unchanged.
fn rewrite_type(ty: Type, registry: &TypeRegistry, module_ctx: &str, guard: &LocalNameGuard<'_>) -> Type {
    match ty {
        Type::Path(mut tp) => {
            if let Some(q) = &mut tp.qself {
                q.ty = Box::new(rewrite_type((*q.ty).clone(), registry, module_ctx, guard));
            }
            tp.path = rewrite_path(tp.path, registry, module_ctx, guard);
            Type::Path(tp)
        }
        Type::Reference(mut tr) => {
            tr.elem = Box::new(rewrite_type(*tr.elem, registry, module_ctx, guard));
            Type::Reference(tr)
        }
        Type::Ptr(mut tp) => {
            tp.elem = Box::new(rewrite_type(*tp.elem, registry, module_ctx, guard));
            Type::Ptr(tp)
        }
        Type::Tuple(mut tt) => {
            tt.elems = tt
                .elems
                .into_iter()
                .map(|e| rewrite_type(e, registry, module_ctx, guard))
                .collect();
            Type::Tuple(tt)
        }
        Type::Slice(mut ts) => {
            ts.elem = Box::new(rewrite_type(*ts.elem, registry, module_ctx, guard));
            Type::Slice(ts)
        }
        Type::Array(mut ta) => {
            ta.elem = Box::new(rewrite_type(*ta.elem, registry, module_ctx, guard));
            Type::Array(ta)
        }
        Type::Paren(mut tp) => {
            tp.elem = Box::new(rewrite_type(*tp.elem, registry, module_ctx, guard));
            Type::Paren(tp)
        }
        Type::Group(mut tg) => {
            tg.elem = Box::new(rewrite_type(*tg.elem, registry, module_ctx, guard));
            Type::Group(tg)
        }
        other => other,
    }
}

/// Rewrite a path, substituting the leading segment when it names a declared
/// or known type. Qualified paths (`<T>::Assoc`) only have their inner type
/// rewritten. `module_ctx` is the current file's module path (e.g.,
/// `"collections/btree/map/entry"`) used to disambiguate same-named types.
fn rewrite_path(path: syn::Path, registry: &TypeRegistry, module_ctx: &str, guard: &LocalNameGuard<'_>) -> syn::Path {
    let leading_colon = path.leading_colon;
    let segs: Vec<syn::PathSegment> = path.segments.into_iter().collect();
    if segs.is_empty() {
        return syn::Path {
            leading_colon,
            segments: segs.into_iter().collect(),
        };
    }
    let head = segs[0].ident.to_string();
    // Resolve the leading segment against the registry. `Self` is never
    // rewritten (it refers to the enclosing type); everything else goes
    // through the registry lookup with module context and the local-name guard.
    let resolved: Option<FieldRefResolution> = if head == "Self" {
        None
    } else {
        Some(registry.resolve_with_guard(&head, module_ctx, guard))
    };

    let abs_path = match resolved {
        Some(FieldRefResolution::Mirrored(canonical)) => {
            // Declared types are mirrored into our synthetic tree. The mirror
            // always lives under the manifest's single wrapper module (named by
            // the registry) — every library's files merge into that one
            // hierarchy, and cross-library references (e.g., a `std`-pathed
            // re-export of a core entity) resolve through it too — so the
            // leading library segment is dropped here regardless of which
            // crate the declaration came from.
            let rest = canonical
                .split_once("::")
                .map(|(_, r)| r)
                .unwrap_or(canonical.as_str());
            format!("crate::{}::{rest}", registry.wrapper_mod())
        }
        Some(FieldRefResolution::Original(canonical)) => {
            // Public but undeclared: point straight at the original type in its
            // builtin crate (`__rustyfill_builtin_core` / `_alloc` / `_std`).
            // Never route these through the preamble — the preamble is only a
            // convenience for bare names inside nested files, and going through
            // it adds indirection (and breaks when the preamble is omitted).
            let lib = canonical.split("::").next().unwrap_or("");
            let rest = canonical.strip_prefix(lib).unwrap_or("").trim_start_matches("::");
            format!("::__rustyfill_builtin_{lib}::{rest}")
        }
        Some(FieldRefResolution::UndeclaredPrivate(_))
        | Some(FieldRefResolution::Unknown(_))
        | None => {
            // Not a routable reference — restore the original path unchanged.
            // Unknown references include primitives, generic parameters, and
            // types from crates we don't mirror. They are resolved through the
            // preamble glob import (which re-exports from the builtin crates)
            // or through the language prelude.
            return syn::Path {
                leading_colon,
                segments: segs.into_iter().collect(),
            };
        }
    };

    // Build the substituted path from its `::`-separated parts plus any
    // remaining original segments (associated items). Each part becomes a
    // segment; the last one carries the generic arguments of the original
    // head (the only place type arguments can legally appear here).
    let all_parts: Vec<String> = abs_path
        .split("::")
        .filter(|p| !p.is_empty())
        .map(|p| p.to_string())
        .chain(segs[1..].iter().map(|s| s.ident.to_string()))
        .collect();
    let last_idx = all_parts.len().saturating_sub(1);
    let mut result_segs: Punctuated<syn::PathSegment, syn::Token![::]> = Punctuated::new();
    for (idx, part) in all_parts.iter().enumerate() {
        let ident = Ident::new(part, Span::call_site());
        let mut seg = syn::PathSegment::from(ident);
        if idx == last_idx {
            seg.arguments = segs[0].arguments.clone();
        }
        result_segs.push_value(seg);
        result_segs.push_punct(syn::Token![::](Span::call_site()));
    }
    // Drop the trailing separator we added after the last segment.
    result_segs.pop_punct();

    syn::Path {
        leading_colon: None,
        segments: result_segs,
    }
}

/// Parameters passed to [`emit_binding_file`] to control how a single output
/// file is generated from parsed items.
pub struct EmitConfig<'a> {
    /// Library name: "core", "alloc", or "std".
    pub lib_name: &'a str,
    /// Depth of the file in the module tree, used to compute `super::` hops
    /// back to the preamble.
    pub file_module_depth: usize,
    /// Resolved intra-module imports (e.g., `use super::node::*`).
    pub extra_uses: &'a [String],
    /// Sibling module names that need `use super::<name>;` aliases.
    pub sibling_modules: &'a [String],
    /// Leaf-name → optional replacement token pairs from the spec's
    /// `path_replacements`.
    pub path_replacements: &'a [(String, Option<&'a str>)],
    /// Fully qualified struct paths to skip during emission.
    pub ignored_structs: &'a [String],
    /// Relative file path within the library tree, e.g.
    /// `"collections/btree/set.rs"`, used to build module-qualified names.
    pub relative_file_path: &'a str,
    /// Registry of known/declared types, used to route field references to
    /// mirrored bindings (declared) or original types (public, undeclared).
    pub type_registry: &'a TypeRegistry,
    /// Additional derive traits to inject into emitted definitions, keyed by
    /// canonical path relative to the library root.
    pub extra_derives: &'a std::collections::HashMap<String, Vec<String>>,
}

/// Mangled name for the per-target preamble module. Unlikely to collide with
/// any real std/core/alloc module name.
const PREAMBLE_MOD: &str = "__rustyfill_prelude";

/// Name of the single wrapper module that the manifest emits around all
/// generated bindings. Every library's files merge into this one hierarchy,
/// so a mirrored type from *any* library is addressed as
/// `crate::{WRAPPER_MOD::<path-without-lib-prefix>}`. Kept in sync with the
/// literal used by [`emit_hierarchical_manifest`].
const WRAPPER_MOD: &str = "std";

/// Std-internal marker traits that are private to core and therefore cannot
/// be named from a downstream crate. When they appear as trait bounds on a
/// mirrored item (e.g., `PhantomData<T: PointeeSized>`), the bound is stripped
/// during emission rather than left dangling. This list is intentionally
/// small and explicit — it is not a general "strip any unknown trait" rule,
/// which would silently drop legitimate user-facing bounds.
const INTERNAL_TRAIT_STRIPS: &[&str] = &[
    "PointeeSized",
    "StructuralPartialEq",
    "MetaSized",
    "Unsize",
];

/// Content of the preamble module. Lives in its own namespace so that
/// `pub use core::marker::*` doesn't clash with a local `mod marker`.
pub fn preamble_content() -> String {
    // We deliberately avoid re-exporting traits like Clone, Debug, PartialEq,
    // Eq, etc. because those are already in the Rust prelude and will be
    // available without import. We only re-export things that are NOT in the
    // language prelude but commonly referenced by std internals.
    //
    // Known core types are imported through __rustyfill_builtin_core. Types from
    // alloc (Box, Vec) are re-exported from the generated bindings via relative
    // paths so that all references converge on the single mirrored definitions.
    r#"// Auto-generated prelude by rustyfill-sys.
 // Provides well-known types as bare names for generated bindings.
 // This module is isolated from local module namespaces (e.g. btree::marker).

 #[allow(unused_imports)]
 pub use ::__rustyfill_builtin_core::alloc::Layout;
 #[allow(unused_imports)]
 pub use ::__rustyfill_builtin_core::borrow::Borrow;
 #[allow(unused_imports)]
 pub use ::__rustyfill_builtin_core::cell::UnsafeCell;
 #[allow(unused_imports)]
 pub use ::__rustyfill_builtin_core::cmp::Ordering;
 #[allow(unused_imports)]
 pub use ::__rustyfill_builtin_core::fmt::Debug;
 #[allow(unused_imports)]
 pub use ::__rustyfill_builtin_core::hash::{Hash, Hasher};
 #[allow(unused_imports)]
 pub use ::__rustyfill_builtin_core::iter::Peekable;
 #[allow(unused_imports)]
 pub use ::__rustyfill_builtin_core::marker::{PhantomData, PhantomPinned};
 #[allow(unused_imports)]
 pub use ::__rustyfill_builtin_core::mem::{ManuallyDrop, MaybeUninit};
 #[allow(unused_imports)]
 pub use ::__rustyfill_builtin_core::num::NonZero;
 #[allow(unused_imports)]
 pub use ::__rustyfill_builtin_core::ops::{Bound, RangeBounds};
 #[allow(unused_imports)]
 pub use ::__rustyfill_builtin_core::pin::Pin;
 #[allow(unused_imports)]
 pub use ::__rustyfill_builtin_core::ptr::NonNull;
 #[allow(unused_imports)]
 pub use ::__rustyfill_builtin_core::slice::SliceIndex;
 #[allow(unused_imports)]
 pub use crate::std::boxed::Box;
 #[allow(unused_imports)]
 pub use ::__rustyfill_builtin_alloc::vec::Vec;
 #[allow(unused_imports)]
 pub mod boxed { pub use crate::std::boxed::Box; }
 #[allow(unused_imports)]
 pub mod alloc { pub use ::__rustyfill_builtin_core::alloc::Layout; }
  // Module shims so that bare module-qualified references (e.g., `marker::PhantomData`,
  // `mem::ManuallyDrop`, `vec::IntoIter`) resolve to the original builtin types
  // through this prelude. Defined once here — do not duplicate above.
  #[allow(unused_imports)]
  pub mod marker { pub use ::__rustyfill_builtin_core::marker::*; }
  #[allow(unused_imports)]
  pub mod mem { pub use ::__rustyfill_builtin_core::mem::*; }
  #[allow(unused_imports)]
  pub mod ptr { pub use ::__rustyfill_builtin_core::ptr::*; }
  #[allow(unused_imports)]
  pub mod cell { pub use ::__rustyfill_builtin_core::cell::*; }
  #[allow(unused_imports)]
  pub mod num { pub use ::__rustyfill_builtin_core::num::*; }
  #[allow(unused_imports)]
  pub mod ops { pub use ::__rustyfill_builtin_core::ops::*; }
  #[allow(unused_imports)]
  pub mod pin { pub use ::__rustyfill_builtin_core::pin::*; }
  #[allow(unused_imports)]
  pub mod borrow { pub use ::__rustyfill_builtin_core::borrow::*; }
  #[allow(unused_imports)]
  pub mod hash { pub use ::__rustyfill_builtin_core::hash::*; }
  #[allow(unused_imports)]
  pub mod iter { pub use ::__rustyfill_builtin_core::iter::*; }
  #[allow(unused_imports)]
  pub mod cmp { pub use ::__rustyfill_builtin_core::cmp::*; }
  #[allow(unused_imports)]
  pub mod fmt { pub use ::__rustyfill_builtin_core::fmt::*; }
  #[allow(unused_imports)]
  pub mod slice { pub use ::__rustyfill_builtin_core::slice::*; }
  #[allow(unused_imports)]
  pub mod sync { pub use ::__rustyfill_builtin_core::sync::*; }
  #[allow(unused_imports)]
  pub mod vec { pub use ::__rustyfill_builtin_alloc::vec::*; }
  // Minimal polyfills for platform-specific types referenced by PAL modules
  // but not mirrored in our bindings. These come from external sources:
  // - Atomic<T>: from core::sync::atomic, a generic type with complex impls
  //   (only the type shape matters for bindings, not atomic operations)
  // - FileDesc: wraps std::os::fd::OwnedFd, platform-specific
  // - lwpid_t: from libc, used by netbsd thread parking
  // - Nanoseconds: from core::num::niche_types, internal type alias
  // - SetValZST: zero-sized marker from alloc::collections::btree::set_val,
  //   used by BTreeSet's internal representation as BTreeMap<T, SetValZST>
  # [repr(transparent)]
  pub struct Atomic < T > (::__rustyfill_builtin_core::cell::UnsafeCell < T >);
  # [derive(Debug)] # [allow(dead_code)] pub struct FileDesc(pub i32);
  # [allow(non_camel_case_types)] pub type lwpid_t = i32;
  pub type Nanoseconds = u32;
  # [derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Default)]
  pub struct SetValZST;
"#
    .to_string()
}

// ── Attribute filtering (AST + token-stream level) ──────────────────────────

/// True when an attribute may be emitted into generated bindings. Doc and all
/// std/compiler-internal attributes ([`is_blocked_attr_name`]) are dropped;
/// everything else (repr, derive, cfg, allow, …) is kept. Used by the AST
/// re-parse path so that re-serializing an item node does not reintroduce
/// attributes that [`strip_blocked_attributes`] already removed from the token
/// stream.
/// Derives that are safe to keep on mirrored types — the derived trait only
/// requires the type's own fields to implement it, and our mirrored fields
/// use the same (or simpler) types as the original.
const SAFE_DERIVES: &[&str] = &["PartialEq", "Eq", "Debug", "Clone"];

fn is_emittable_attr(attr: &syn::Attribute) -> bool {
    let Some(ident) = attr.path().get_ident() else {
        // Pathed attributes like `#[diagnostic::on_unimplemented]` have no
        // single leading ident; block them conservatively unless they are a
        // known-safe form.
        return !attr.path().is_ident("diagnostic");
    };
    let name = ident.to_string();
    if name == "derive" {
        // Check each trait in the derive list against the safe set.
        // Strip the entire derive if any trait is unsafe.
        if let syn::Meta::List(list) = &attr.meta {
            let tokens_str = list.tokens.to_string();
            let traits: Vec<&str> = tokens_str
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();
            let all_safe = !traits.is_empty()
                && traits.iter().all(|t| SAFE_DERIVES.contains(t));
            return all_safe;
        }
        return false;
    }
    !is_blocked_attr_name(&name)
}

/// Check if a bracket group (after a `#`) corresponds to an attribute we should strip.
fn is_blocked_attr_group(group: &proc_macro2::Group) -> bool {
    if group.delimiter() != proc_macro2::Delimiter::Bracket {
        return false;
    }
    let first = group.stream().into_iter().next();
    let Some(TokenTree::Ident(id)) = first else {
        return false;
    };
    let name = id.to_string();
    is_blocked_attr_name(&name)
}

/// Names of attributes that are reserved for std/the compiler and must never
/// appear in generated bindings. A downstream crate cannot use any of these:
/// they either gate on unstable features (`#[lang]`, `#[fundamental]`,
/// `#[allow_internal_unstable]`), carry compiler-internal metadata
/// (`#[diagnostic::on_unimplemented]`, `#[rustc_*]`), or duplicate lang items
/// already owned by the real core (`#[lang = "owned_box"]`). Stability attrs
/// (`#[stable]/#[unstable]/#[deprecated]`) and `#[cfg_attr]` are stripped too
/// since they are meaningless (or actively harmful) outside std.
fn is_blocked_attr_name(name: &str) -> bool {
    matches!(
        name,
        "doc"
            | "stable"
            | "unstable"
            | "deprecated"
            | "cfg_attr"
            | "lang"
            | "fundamental"
            | "allow_internal_unstable"
            | "deny_internal_unstable"
            | "diagnostic"
    ) || name.starts_with("rustc_")
}

/// Recursively walk a token stream, stripping blocked attributes at every nesting level
/// and removing identifiers listed in `ignored_names` from trait bounds and use lists.
fn strip_tokens_recursive(tokens: TokenStream, ignored_names: &[&str]) -> TokenStream {
    let mut result = TokenStream::new();
    let trees: Vec<TokenTree> = tokens.into_iter().collect();
    let mut iter = trees.into_iter().peekable();

    while let Some(tt) = iter.next() {
        match &tt {
            TokenTree::Punct(p) if p.as_char() == '#' => {
                if let Some(TokenTree::Group(next_group)) = iter.peek()
                    && is_blocked_attr_group(next_group)
                {
                    // Consume and discard the group too.
                    iter.next();
                    continue;
                }
                result.extend(Some(tt));
            }
            TokenTree::Group(g) => {
                // Recurse first to strip attributes inside nested groups.
                let cleaned_inner = strip_tokens_recursive(g.stream(), ignored_names);
                // Then strip ignored identifiers from brace-delimited groups
                // (use lists, struct literals, etc.) and bracket groups.
                // Note: angle brackets (< >) are NOT a Delimiter variant in proc_macro2;
                // they appear as flat Punct tokens, so ignored identifiers in generic
                // bounds are caught by the flat-strip pass below on the outer stream.
                let cleaned_inner = match g.delimiter() {
                    proc_macro2::Delimiter::Brace | proc_macro2::Delimiter::Bracket => {
                        strip_ignored_identifiers(cleaned_inner, ignored_names)
                    }
                    _ => cleaned_inner,
                };
                let mut new_group = proc_macro2::Group::new(g.delimiter(), cleaned_inner);
                new_group.set_span(g.span());
                result.extend(Some(TokenTree::Group(new_group)));
            }
            _ => {
                result.extend(Some(tt));
            }
        }
    }

    // Strip ignored identifiers from the flat token stream (catches generics
    // where < > are just Punct tokens, not a Group).
    strip_ignored_identifiers(result, ignored_names)
}

/// Strip identifiers matching any of the given ignored names from trait bounds
/// and use-list imports inside delimiter groups.
///
/// For each ignored name (e.g., `"Allocator"` derived from the spec path
/// `core::alloc::Allocator`), transforms patterns like:
///   `<A: Allocator + Clone>` -> `<A: Clone>`
///   `<A: Allocator>`         -> `<A>`
///   `{Allocator, Layout}`    -> `{Layout}`
///
/// Cleans up adjacent separators (`+`, `,`) and leading colons when the bound
/// list becomes empty.
fn strip_ignored_identifiers(tokens: TokenStream, ignored_names: &[&str]) -> TokenStream {
    if ignored_names.is_empty() {
        return tokens;
    }

    let trees: Vec<TokenTree> = tokens.into_iter().collect();
    let mut result = Vec::new();
    let mut i = 0;

    while i < trees.len() {
        #[allow(clippy::cmp_owned)]
        let should_strip = matches!(
            &trees[i],
            TokenTree::Ident(id) if ignored_names.iter().any(|n| *n == id.to_string())
        );

        if should_strip {
            // Clean up preceding separator (`+` for trait bounds, `,` for use lists).
            if let Some(TokenTree::Punct(p)) = result.last()
                && (p.as_char() == '+' || p.as_char() == ',')
            {
                result.pop();
            }
            // Advance past the identifier.
            i += 1;
            // Skip trailing separator if present.
            if i < trees.len()
                && let TokenTree::Punct(p) = &trees[i]
                && (p.as_char() == '+' || p.as_char() == ',')
            {
                i += 1;
            }
            // If previous token is now `:`, check if there are any remaining bounds after us.
            // If not, remove the colon too (was `A: IgnoredTrait`, becoming just `A`).
            if let Some(TokenTree::Punct(p)) = result.last()
                && p.as_char() == ':'
            {
                let has_more_bounds = if i < trees.len() {
                    !matches!(
                        &trees[i],
                        TokenTree::Punct(pp)
                            if pp.as_char() == '>' || pp.as_char() == ',' || pp.as_char() == '+'
                    )
                } else {
                    false
                };
                if !has_more_bounds {
                    result.pop();
                }
            }
        } else {
            result.push(trees[i].clone());
            i += 1;
        }
    }

    result.into_iter().collect()
}

/// Strip all blocked attributes from a token stream and remove ignored identifiers.
/// Removes:
/// - `#[doc = "..."]` — bloats output, can embed spurious derives
/// - `#[stable]/#[unstable]/#[deprecated]` — reserved for std
/// - `#[rustc_*]` — compiler internals
/// - `#[cfg_attr(...)]` — may duplicate rustc_diagnostic_item
///
/// Also strips identifiers matching `ignored_names` from trait bounds and use lists
/// inside angle-bracket and brace-delimited groups.
///
/// Processes recursively to catch doc attributes on struct fields and other
/// nested items, not just top-level attributes.
fn strip_blocked_attributes(tokens: TokenStream, ignored_names: &[&str]) -> TokenStream {
    strip_tokens_recursive(tokens, ignored_names)
}

// ── Canonical binding emission ──────────────────────────────────────────────

/// Emit a complete `.rs` file's worth of Rust source from the given parsed items.
pub fn emit_parsed_items(
    items: &[ParsedItem],
    config: &EmitConfig,
    preamble_use_path: &str,
    extra_uses: &[String],
    module_path: &str,
) -> String {
    let mut out = String::new();

    out.push_str("// Auto-generated by rustyfill-sys.\n");
    // The preamble glob import is omitted when the path is empty (depth-0
    // files whose computed path would point back at their own crate root).
    if !preamble_use_path.is_empty() {
        out.push_str(&format!(
            "#[allow(unused_imports)]\npub use {preamble_use_path}::*;\n"
        ));
    }

    for line in extra_uses {
        out.push_str(line);
        out.push('\n');
    }
    if !extra_uses.is_empty() {
        out.push('\n');
    }

    // Build a guard marking file-local type names so that bare references to
    // them are NOT routed to cross-module mirrors.
    let local_names: Vec<&str> = items
        .iter()
        .filter(|i| matches!(i.kind, ItemKind::Struct | ItemKind::Enum | ItemKind::Union | ItemKind::TypeAlias))
        .map(|i| i.name.as_str())
        .collect();
    let guard = LocalNameGuard::new(Some(&local_names));

    for item in items {
        // Constants are always emitted: file-level consts (e.g., `const B` in
        // btree/node.rs) are referenced by bare name from the data structures
        // that follow them, so dropping one would leave its dependents dangling.
        // They carry no cross-module surface, so they're safe to keep wholesale.
        let is_const = item.kind == ItemKind::Const;
        // For data structures, restrict output to types explicitly declared in
        // the loader spec. This keeps peripheral public items that live
        // alongside the core data structures (iterators, cursors, range views,
        // set-algebra engines, …) out of the mirrored tree unless they are
        // needed by the polyfill. Any reference to such an undeclared type
        // routes back to the original builtin crate instead.
        if !is_const
            && !config.type_registry.is_declared_in_module(config.lib_name, module_path, &item.name)
        {
            continue;
        }
        // Skip items whose fully qualified path matches an ignored struct.
        let fq_path = if module_path.is_empty() {
            item.name.clone()
        } else {
            format!("{}::{}", module_path, item.name)
        };
        if config.ignored_structs.contains(&fq_path) {
            continue;
        }
        emit_item(
            &mut out,
            item,
            preamble_use_path,
            config.path_replacements,
            config.type_registry,
            module_path,
            &guard,
            config.extra_derives,
        );
    }

    out
}

fn emit_item(
    out: &mut String,
    item: &ParsedItem,
    preamble_use_path: &str,
    path_replacements: &[(String, Option<&str>)],
    type_registry: &TypeRegistry,
    module_ctx: &str,
    guard: &LocalNameGuard<'_>,
    extra_derives: &std::collections::HashMap<String, Vec<String>>,
) {
    // The full_tokens already include all attributes + the item definition.
    // Pipeline:
    // 1. Strip blocked attributes (doc/stable/rustc_*) and remove ignored
    //    identifiers from trait bounds and use lists.
    // 2. Remove the `const` modifier from trait definitions (syn 2 incompatible).
    // 3. Rewrite field/generic references: declared types point at their
    //    mirrored bindings, public-but-undeclared types point at the original
    //    builtin crate. Done via AST when possible, token fallback otherwise.
    // 4. Widen pub(super)/pub(crate) to pub so that types are accessible from
    //    sibling modules in our generated tree.
    // 5. Substitute configured path replacements (e.g., Global -> ()).
    // Declared type aliases are emitted as mirrored definitions: the alias is
    // re-emitted with its RHS routed through the registry (e.g. `Root` →
    // `crate::alloc::collections::btree::node::NodeRef<...>`), so references
    // from other modules converge on our tree instead of dangling.
    if item.kind == ItemKind::TypeAlias
        && let Some(info) = find_declared_alias_info(item, type_registry)
        && let Some(rhs) = &info.alias_rhs
    {
        let mut ts = TokenStream::new();
        for attr in &item.attrs {
            if !is_emittable_attr(attr) {
                continue;
            }
            attr.to_tokens(&mut ts);
        }
        // Classify the alias RHS as a pseudo-field: it names exactly what the
        // alias expands to, so its drop-safety classification is inherited by
        // every field whose type is this alias.
        if drop_annotations_enabled()
            && let Ok(rhs_ty) = syn::parse2::<syn::Type>(rhs.clone())
        {
            let value = classify_field_drop(&rhs_ty, type_registry, guard);
            drop_doc_comment(value).to_tokens(&mut ts);
        }
        emit_declared_type_alias(&item.name, rhs, &item.full_tokens, type_registry, module_ctx, guard, &mut ts);
        let widened = widen_visibility(ts);
        let rewritten = rewrite_crate_paths(widened, preamble_use_path, path_replacements);
        write!(out, "{}", rewritten).ok();
        out.push('\n');
        return;
    }

    // Names to strip from trait bounds and use lists: spec-configured ignored
    // paths (e.g., `Allocator`) plus a small set of std-internal marker traits
    // that are private to core and cannot be referenced by name from a
    // downstream crate (see [`INTERNAL_TRAIT_STRIPS`]).
    let mut strip_names: Vec<&str> = path_replacements
        .iter()
        .filter(|(_, v)| v.is_none())
        .map(|(k, _)| k.as_str())
        .collect();
    for name in INTERNAL_TRAIT_STRIPS {
        if !strip_names.contains(name) {
            strip_names.push(name);
        }
    }
    let clean_tokens = strip_blocked_attributes(item.full_tokens.clone(), &strip_names);
    let const_stripped = strip_const_trait_modifier(clean_tokens);
    // The tokens already carry the filtered attributes (doc/internal attrs were
    // stripped by [`strip_blocked_attributes`]), so the AST-based reference
    // rewrite parses them directly without re-prepending `item.attrs`.
    let rerouted = rewrite_item_references_rerouted(
        &const_stripped,
        item,
        type_registry,
        module_ctx,
        guard,
        extra_derives,
    );
    let widened = widen_visibility(rerouted);
    let rewritten = rewrite_crate_paths(widened, preamble_use_path, path_replacements);
    write!(out, "{}", rewritten).ok();
    out.push('\n');
}

/// Lookup helper for the alias-emission path in [`emit_item`]: finds the
/// registry entry for a parsed item by leaf name (the emitter does not know
/// the library prefix), returning it only when the entry is declared AND has
/// a recorded alias RHS.
fn find_declared_alias_info<'a>(
    item: &ParsedItem,
    registry: &'a TypeRegistry,
) -> Option<&'a TypeInfo> {
    registry
        .candidates_for_leaf(&item.name)
        .iter()
        .find_map(|p| {
            let info = registry.get(p)?;
            (info.declared && info.alias_rhs.is_some()).then_some(info)
        })
}

/// Emit a declared type alias whose RHS has been routed through the registry.
/// The alias's own generic parameters (e.g., `<K, V>` on `BoxedNode<K, V>`)
/// are recovered from the original item's full tokens, since the stored RHS
/// alone does not carry them.
fn emit_declared_type_alias(
    name: &str,
    rhs: &TokenStream,
    item_full_tokens: &TokenStream,
    registry: &TypeRegistry,
    module_ctx: &str,
    guard: &LocalNameGuard<'_>,
    out: &mut TokenStream,
) {
    // Parse the original alias to recover its generic parameters and bounds.
    let original_node: syn::ItemType = match syn::parse2(item_full_tokens.clone()) {
        Ok(n) => n,
        Err(_) => {
            // Can't recover generics — fall back to a non-generic alias.
            let ty: syn::Type = match syn::parse2(rhs.clone()) {
                Ok(t) => t,
                Err(_) => {
                    out.extend(rhs.clone());
                    return;
                }
            };
            let routed = rewrite_type(ty, registry, module_ctx, guard);
            let mut ts = TokenStream::new();
            ts.extend(format!("type {} = ", name).parse::<TokenStream>().unwrap_or_default());
            routed.to_tokens(&mut ts);
            ts.extend(";".parse::<TokenStream>().unwrap_or_default());
            out.extend(ts);
            return;
        }
    };

    // Route the RHS through the registry and substitute it into the original
    // node (preserving the alias's own generics, vis, and attrs).
    let ty: syn::Type = match syn::parse2(rhs.clone()) {
        Ok(t) => t,
        Err(_) => {
            let mut ts = TokenStream::new();
            original_node.to_tokens(&mut ts);
            out.extend(ts);
            return;
        }
    };
    let routed = rewrite_type(ty, registry, module_ctx, guard);

    let mut cloned = original_node;
    cloned.ty = Box::new(routed);
    let mut ts = TokenStream::new();
    cloned.to_tokens(&mut ts);
    out.extend(ts);
}

/// Remove the `const` modifier from trait definitions (`pub const trait X`).
///
/// `const trait` is a newer Rust feature that syn 2's parser does not accept:
/// it reads `const` as the start of a const item and then fails on the
/// following `trait` keyword. These are unstable marker traits pulled in from
/// std (e.g., `Destruct`, `ConstParamTy_`); dropping the `const` modifier keeps
/// them parseable without changing their meaning for our binding purposes.
fn strip_const_trait_modifier(tokens: TokenStream) -> TokenStream {
    let trees: Vec<TokenTree> = tokens.into_iter().collect();
    let mut result = Vec::with_capacity(trees.len());
    let n = trees.len();
    let mut i = 0;
    while i < n {
        // Look for the sequence: [attrs] ... 'const' 'trait'.
        if let TokenTree::Ident(id) = &trees[i]
            && id == "const"
            && i + 1 < n
            && let TokenTree::Ident(next) = &trees[i + 1]
            && next == "trait"
        {
            // Drop the `const` token; keep `trait` and everything after.
            i += 1;
            continue;
        }
        result.push(trees[i].clone());
        i += 1;
    }
    result.into_iter().collect()
}

/// Reroute type references inside an item using the type registry. Tries the
/// AST-based rewrite first (exact), falling back to the legacy token-level
/// rewrite when the item cannot be re-parsed standalone.
///
/// The `tokens` argument is expected to already carry the item's (filtered)
/// attributes — the caller runs [`strip_blocked_attributes`] first and passes
/// the result through. We therefore parse `tokens` directly rather than
/// re-prepending `item.attrs`, which would duplicate every attribute (the
/// struct/enum/union nodes serialize their own attrs).
fn rewrite_item_references_rerouted(
    tokens: &TokenStream,
    item: &ParsedItem,
    registry: &TypeRegistry,
    module_ctx: &str,
    guard: &LocalNameGuard<'_>,
    extra_derives: &std::collections::HashMap<String, Vec<String>>,
) -> TokenStream {
    let name = item.name.as_str();
    let kind = item.kind;
    // Compute the path relative to the library root (matching the spec's key
    // format) so we can look up any spec-requested extra derives.
    let rel_path = if module_ctx.is_empty() {
        name.to_string()
    } else {
        format!("{}::{}", module_ctx.replace('/', "::"), name)
    };
    let injected_derives = extra_derives.get(&rel_path).cloned().unwrap_or_default();

    fn inject_derives(attrs: &mut Vec<syn::Attribute>, new_traits: &[String]) {
        if new_traits.is_empty() {
            return;
        }
        // Find an existing derive attribute and merge into it, or create one.
        let existing_idx = attrs.iter().position(|a| a.path().is_ident("derive"));
        match existing_idx {
            Some(idx) => {
                if let syn::Meta::List(list) = &mut attrs[idx].meta {
                    // Parse existing trait names from the token stream.
                    let existing_tokens: Vec<String> = list
                        .tokens
                        .clone()
                        .into_iter()
                        .filter_map(|tt| match tt {
                            proc_macro2::TokenTree::Ident(id) => Some(id.to_string()),
                            _ => None,
                        })
                        .collect();
                    let mut all = existing_tokens;
                    for t in new_traits {
                        if !all.contains(t) {
                            all.push(t.clone());
                        }
                    }
                    // Rebuild the token stream as a comma-separated ident list.
                    let mut new_tokens = proc_macro2::TokenStream::new();
                    for (i, name) in all.iter().enumerate() {
                        if i > 0 {
                            new_tokens.extend(
                                ",".to_string()
                                    .parse::<proc_macro2::TokenStream>()
                                    .unwrap(),
                            );
                        }
                        new_tokens.extend(
                            name.parse::<proc_macro2::TokenStream>().unwrap(),
                        );
                    }
                    list.tokens = new_tokens;
                }
            }
            None => {
                let mut tokens = proc_macro2::TokenStream::new();
                for (i, t) in new_traits.iter().enumerate() {
                    if i > 0 {
                        tokens.extend(",".parse::<proc_macro2::TokenStream>().unwrap());
                    }
                    tokens.extend(t.parse::<proc_macro2::TokenStream>().unwrap());
                }
                let meta = syn::Meta::List(syn::MetaList {
                    path: syn::Path::from(syn::Ident::new(
                        "derive",
                        proc_macro2::Span::call_site(),
                    )),
                    delimiter: syn::MacroDelimiter::Paren(syn::token::Paren::default()),
                    tokens,
                });
                attrs.push(syn::Attribute {
                    pound_token: syn::token::Pound::default(),
                    style: syn::AttrStyle::Outer,
                    bracket_token: syn::token::Bracket::default(),
                    meta,
                });
            }
        }
    }

    match item.kind {
        ItemKind::Struct => match syn::parse2::<ItemStruct>(tokens.clone()) {
            Ok(mut node) => {
                node.attrs.retain(is_emittable_attr);
                inject_derives(&mut node.attrs, &injected_derives);
                rewrite_struct_node(node, registry, module_ctx, guard)
            }
            Err(_) => rewrite_crate_paths_legacy(tokens.clone(), name, registry, kind, guard),
        },
        ItemKind::Enum => match syn::parse2::<syn::ItemEnum>(tokens.clone()) {
            Ok(mut node) => {
                node.attrs.retain(is_emittable_attr);
                inject_derives(&mut node.attrs, &injected_derives);
                rewrite_enum_node(node, registry, module_ctx, guard)
            }
            Err(_) => rewrite_crate_paths_legacy(tokens.clone(), name, registry, kind, guard),
        },
        ItemKind::Union => match syn::parse2::<syn::ItemUnion>(tokens.clone()) {
            Ok(mut node) => {
                node.attrs.retain(is_emittable_attr);
                inject_derives(&mut node.attrs, &injected_derives);
                rewrite_union_node(node, registry, module_ctx, guard)
            }
            Err(_) => rewrite_crate_paths_legacy(tokens.clone(), name, registry, kind, guard),
        },
        // Type aliases: route every reference in the RHS through the registry.
        ItemKind::TypeAlias => match syn::parse2::<syn::ItemType>(tokens.clone()) {
            Ok(mut node) => {
                node.attrs.retain(is_emittable_attr);
                node.ty = Box::new(rewrite_type(*node.ty, registry, module_ctx, guard));
                let mut ts = TokenStream::new();
                node.to_tokens(&mut ts);
                ts
            }
            Err(_) => rewrite_crate_paths_legacy(tokens.clone(), name, registry, kind, guard),
        },
        _ => rewrite_crate_paths_legacy(tokens.clone(), name, registry, kind, guard),
    }
}

/// Token-level fallback for reference rerouting: substitutes bare identifiers
/// that name declared types with absolute paths into the synthetic tree.
///
/// The item's own name is never rewritten — only references *inside* the body
/// are. This guards against a malformed `full_tokens` blob (e.g., a struct
/// definition concatenated with its impl block) where the item's own identifier
/// would otherwise be clobbered by the substitution.
///
/// Also injects `/// rustyfill-drop: ...` doc comments on struct/union fields
/// when the item is one of those kinds (best-effort: enum variant payloads are
/// not annotated on this path since field boundaries cannot be distinguished
/// from nested braces at token level).
fn rewrite_crate_paths_legacy(
    tokens: TokenStream,
    item_name: &str,
    registry: &TypeRegistry,
    _item_kind: ItemKind,
    guard: &LocalNameGuard<'_>,
) -> TokenStream {
    let trees: Vec<TokenTree> = tokens.into_iter().collect();

    // Detect the item keyword to decide whether field annotation applies.
    let item_kw = trees.iter().find_map(|tt| match tt {
        TokenTree::Ident(id) => {
            let name = id.to_string();
            matches!(name.as_str(), "struct" | "enum" | "union" | "const" | "type").then_some(name)
        }
        _ => None,
    });
    let annotate_fields = drop_annotations_enabled()
        && matches!(item_kw.as_deref(), Some("struct") | Some("union"));

    let mut result = Vec::with_capacity(trees.len());
    // Track whether we've passed the item-definition keyword (struct/enum/union/
    // const/type). Before it, any ident matching the item name is the item's own
    // name and must not be touched. After it, all occurrences are references.
    let def_keywords = ["struct", "enum", "union", "const", "type"];
    let mut past_def_kw = false;
    // Field-boundary state machine for struct/union bodies: inside the first
    // brace group after the item name, each top-level comma separates fields.
    let mut in_body = false;
    let mut body_depth = 0usize;
    // Tokens accumulated for the current field (to classify before emitting).
    let mut field_buf: Vec<TokenTree> = Vec::new();

    macro_rules! flush_field {
        () => {
            if annotate_fields && in_body && !field_buf.is_empty() {
                let buf_ts: TokenStream = field_buf.iter().cloned().collect();
                if let Ok(ty) = syn::parse2(buf_ts) {
                    let value = classify_field_drop(&ty, registry, guard);
                    let attr: TokenStream = drop_doc_comment(value).to_token_stream();
                    result.extend(attr.into_iter());
                }
            }
            field_buf.clear();
        };
    }

    for tt in &trees {
        if let TokenTree::Ident(id) = tt {
            let name = id.to_string();
            if !past_def_kw && def_keywords.contains(&name.as_str()) {
                past_def_kw = true;
                result.push(tt.clone());
                continue;
            }
            if past_def_kw && !name.starts_with('_') && !is_keywordish(&name) && name != item_name
            {
                if let FieldRefResolution::Mirrored(canonical) = registry.resolve_field_ref(&name)
                {
                    // Mirrors always live under the manifest's single wrapper
                    // module (named by the registry), so drop the leading
                    // library segment.
                    let rest = canonical
                        .split_once("::")
                        .map(|(_, r)| r)
                        .unwrap_or(canonical.as_str());
                    let abs = format!("crate::{}::{rest}", registry.wrapper_mod());
                    if let Ok(subst) = abs.parse::<TokenStream>() {
                        if annotate_fields && in_body && body_depth == 1 {
                            field_buf.extend(subst.into_iter());
                        } else {
                            result.extend(subst.into_iter());
                        }
                        continue;
                    }
                }
            }
        }

        // Body tracking for field annotation.
        if let TokenTree::Group(g) = tt {
            if g.delimiter() == proc_macro2::Delimiter::Brace {
                if in_body {
                    body_depth += 1;
                } else if annotate_fields && past_def_kw {
                    in_body = true;
                    body_depth = 1;
                }
            }
        }

        if annotate_fields && in_body {
            match tt {
                TokenTree::Punct(p) if p.as_char() == ',' && body_depth == 1 => {
                    flush_field!();
                    result.push(tt.clone());
                    continue;
                }
                _ => {}
            }
            // Buffer non-attribute tokens belonging to the current field.
            if !(matches!(tt, TokenTree::Punct(p) if p.as_char() == '#')) {
                field_buf.push(tt.clone());
                continue;
            }
        }

        result.push(tt.clone());
        if let TokenTree::Group(g) = tt {
            if g.delimiter() == proc_macro2::Delimiter::Brace && in_body && body_depth > 0 {
                body_depth -= 1;
                if body_depth == 0 {
                    flush_field!();
                    in_body = false;
                }
            }
        }
    }
    if in_body {
        flush_field!();
    }
    result.into_iter().collect()
}

fn is_keywordish(name: &str) -> bool {
    matches!(
        name,
        "struct" | "enum" | "union" | "type" | "const" | "fn" | "let" | "mut" | "ref" | "self"
            | "Self" | "where" | "for" | "impl" | "trait" | "async" | "await" | "dyn" | "unsafe"
            | "extern" | "use" | "mod" | "crate" | "super" | "true" | "false" | "pub"
    )
}


/// Widen all visibility modifiers to plain `pub` so that generated bindings
/// are fully accessible from any crate. This handles three cases:
/// 1. `pub(super)` / `pub(crate)` → strip parens, keep `pub`
/// 2. Plain `pub` → pass through unchanged
/// 3. No visibility (private struct/enum/union/type/const) → inject `pub`
///    before the item keyword so the type becomes public.
///
/// Additionally, for structs and unions with braced bodies, the FIRST braced
/// group after the item name (and optional generics) is treated as the struct/union
/// body, and private fields inside are widened to `pub`.
fn widen_visibility(tokens: TokenStream) -> TokenStream {
    let tts: Vec<TokenTree> = tokens.into_iter().collect();
    let mut result = TokenStream::new();
    let mut i = 0;

    // Skip leading attributes (`# [...]` pairs).
    while i < tts.len() {
        if let TokenTree::Punct(p) = &tts[i]
            && p.as_char() == '#'
            && i + 1 < tts.len()
            && matches!(&tts[i + 1], TokenTree::Group(g) if g.delimiter() == proc_macro2::Delimiter::Bracket)
        {
            result.extend(Some(tts[i].clone()));
            i += 1;
            result.extend(Some(tts[i].clone()));
            i += 1;
            continue;
        }
        break;
    }

    // Determine if we have an existing visibility modifier and whether this
    // is a struct/union needing field widening. Do NOT consume tokens yet.
    let (has_vis, is_struct_or_union) = if i < tts.len() {
        match &tts[i] {
            TokenTree::Ident(id) if id == "pub" => {
                let mut peek = i + 1;
                if peek < tts.len()
                    && matches!(&tts[peek], TokenTree::Group(g) if g.delimiter() == proc_macro2::Delimiter::Parenthesis)
                {
                    peek += 1;
                }
                let is_su = peek < tts.len()
                    && matches!(&tts[peek], TokenTree::Ident(kw) if {
                        let s = kw.to_string();
                        s == "struct" || s == "union"
                    });
                (true, is_su)
            }
            _ => {
                // No `pub` — check if this is an item keyword needing visibility injection.
                let is_su = i < tts.len()
                    && matches!(&tts[i], TokenTree::Ident(kw) if {
                        let s = kw.to_string();
                        s == "struct" || s == "union"
                    });
                (false, is_su)
            }
        }
    } else {
        (false, false)
    };

    // If no visibility on an item keyword, inject `pub`.
    if !has_vis
        && i < tts.len()
        && let TokenTree::Ident(kw) = &tts[i]
    {
        let kw_str = kw.to_string();
        if matches!(
            kw_str.as_str(),
            "struct" | "enum" | "union" | "type" | "const"
        ) {
            result.extend(Some(TokenTree::Ident(proc_macro2::Ident::new(
                "pub",
                tts[i].span(),
            ))));
        }
    }

    // Emit remaining tokens, stripping scope parens from `pub` and widening
    // struct/union bodies.
    let mut struct_body_widened = false;
    while i < tts.len() {
        // Strip scope parens from `pub(X)` → `pub` everywhere in the item.
        if let TokenTree::Ident(id) = &tts[i]
            && id == "pub"
            && i + 1 < tts.len()
            && matches!(&tts[i + 1], TokenTree::Group(g) if g.delimiter() == proc_macro2::Delimiter::Parenthesis)
        {
            // Emit just `pub`, skip the scope parens.
            result.extend(Some(tts[i].clone()));
            i += 1;
            i += 1; // skip parenthesized scope
            continue;
        }

        // For struct/union, widen the first braced body.
        if is_struct_or_union
            && !struct_body_widened
            && let TokenTree::Group(group) = &tts[i]
            && group.delimiter() == proc_macro2::Delimiter::Brace
        {
            let widened_body = widen_struct_field_visibility(group.stream());
            let new_group = TokenTree::Group(proc_macro2::Group::new(
                proc_macro2::Delimiter::Brace,
                widened_body,
            ));
            result.extend(Some(new_group));
            struct_body_widened = true;
            i += 1;
            continue;
        }

        result.extend(Some(tts[i].clone()));
        i += 1;
    }

    result
}

/// Widen private field visibility inside a struct/union body.
///
/// Strategy: walk through the body tracking "field boundary" state. At each
/// field boundary (start of body or after `,`), the next meaningful token is
/// either `pub` (already public) or a field name (needs widening). Once we've
/// handled the visibility token, we consume the rest of the field until the
/// next `,` by passing tokens through unchanged.
fn widen_struct_field_visibility(tokens: TokenStream) -> TokenStream {
    let tts: Vec<TokenTree> = tokens.into_iter().collect();
    let mut result = TokenStream::new();
    let mut i = 0;
    let mut at_field_boundary = true;

    while i < tts.len() {
        // Skip attributes: `#` followed by `[...]`.
        if let TokenTree::Punct(p) = &tts[i]
            && p.as_char() == '#'
            && i + 1 < tts.len()
            && matches!(&tts[i + 1], TokenTree::Group(g) if g.delimiter() == proc_macro2::Delimiter::Bracket)
        {
            result.extend(Some(tts[i].clone()));
            i += 1;
            result.extend(Some(tts[i].clone()));
            i += 1;
            continue;
        }

        // Pass through nested groups (they're inside field types).
        if matches!(&tts[i], TokenTree::Group(_)) {
            result.extend(Some(tts[i].clone()));
            i += 1;
            continue;
        }

        // At a field boundary, check for `pub` or inject it.
        if at_field_boundary {
            if let TokenTree::Ident(id) = &tts[i]
                && id == "pub"
            {
                // Already public — emit `pub` and optional scope, then field name.
                result.extend(Some(tts[i].clone()));
                i += 1;
                // Skip scope parens if present.
                if i < tts.len()
                    && matches!(&tts[i], TokenTree::Group(g) if g.delimiter() == proc_macro2::Delimiter::Parenthesis)
                {
                    i += 1;
                }
                // Next ident is the field name — emit it.
                if i < tts.len() {
                    result.extend(Some(tts[i].clone()));
                    i += 1;
                }
                at_field_boundary = false;
                continue;
            }

            // Not `pub` — this should be the field name. Inject `pub` before it.
            // But only if it looks like a field (identifier followed by `:` somewhere).
            if let TokenTree::Ident(field_id) = &tts[i] {
                let field_name = field_id.to_string();
                // Verify this isn't a keyword or enum variant.
                if !matches!(
                    field_name.as_str(),
                    "struct"
                        | "enum"
                        | "union"
                        | "type"
                        | "const"
                        | "fn"
                        | "mut"
                        | "ref"
                        | "self"
                        | "Self"
                        | "where"
                        | "for"
                        | "impl"
                        | "trait"
                        | "async"
                        | "await"
                        | "dyn"
                        | "unsafe"
                        | "extern"
                        | "use"
                        | "mod"
                        | "crate"
                        | "super"
                        | "true"
                        | "false"
                ) && peek_for_colon_simple(&tts, i + 1)
                {
                    result.extend(Some(TokenTree::Ident(proc_macro2::Ident::new(
                        "pub",
                        tts[i].span(),
                    ))));
                    result.extend(Some(tts[i].clone()));
                    i += 1;
                    at_field_boundary = false;
                    continue;
                }
            }
        }

        // Comma means end of current field, next token starts a new field.
        if let TokenTree::Punct(p) = &tts[i]
            && p.as_char() == ','
        {
            result.extend(Some(tts[i].clone()));
            i += 1;
            at_field_boundary = true;
            continue;
        }

        // Everything else passes through unchanged.
        result.extend(Some(tts[i].clone()));
        i += 1;
    }

    result
}

/// Simple peek: does a SINGLE `:` (not `::`) appear before any `,` or end-of-tokens?
/// A field type annotation uses single `:`, while path separators use `::`.
/// We check that the token AFTER `:` is NOT another `:`.
fn peek_for_colon_simple(tts: &[TokenTree], start: usize) -> bool {
    let mut i = start;
    while i < tts.len() {
        match &tts[i] {
            TokenTree::Punct(p) if p.as_char() == ':' => {
                // Check if next token is also `:` (making it `::`).
                if i + 1 < tts.len() {
                    match &tts[i + 1] {
                        TokenTree::Punct(p2) if p2.as_char() == ':' => {
                            // This is `::`, skip both colons.
                            i += 2;
                            continue;
                        }
                        _ => {
                            // Single `:` — this is a field type annotation.
                            return true;
                        }
                    }
                } else {
                    // Lone `:` at end — treat as field annotation.
                    return true;
                }
            }
            TokenTree::Punct(p) if p.as_char() == ',' => return false,
            TokenTree::Group(_) => {}
            _ => {}
        }
        i += 1;
    }
    false
}

/// Recursively rewrite `crate::core`, `crate::alloc`, and other `crate::` paths
/// in emitted token streams, since our synthetic crate root doesn't match the
/// original library layout. Enters all Group delimiters to catch paths nested
/// inside struct bodies, impl blocks, etc.
///
/// Rewrites `crate::X::Y` paths in generated bindings:
/// - `crate::core::...` and `crate::alloc::...` → substituted with the preamble
///   path so types resolve through our reserved extern crate aliases
///   (`__rustyfill_builtin_core` / `__rustyfill_builtin_alloc`).
/// - Any leaf identifier listed in `path_replacements` is replaced with its
///   configured replacement tokens (e.g., `Global` → `()`). If no replacement
///   is configured for that leaf, it is left alone.
fn rewrite_crate_paths(
    tokens: TokenStream,
    preamble_path: &str,
    path_replacements: &[(String, Option<&str>)],
) -> TokenStream {
    // Build a lookup from leaf name → replacement string.
    let repl_map: HashMap<String, Option<&str>> = path_replacements
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    rewrite_crate_paths_recursive(tokens, preamble_path, &repl_map)
}

fn rewrite_crate_paths_recursive(
    tokens: TokenStream,
    preamble_path: &str,
    repl_map: &HashMap<String, Option<&str>>,
) -> TokenStream {
    let mut result = TokenStream::new();
    let trees: Vec<TokenTree> = tokens.into_iter().collect();
    let mut i = 0;

    while i < trees.len() {
        // Recurse into Groups so we catch `crate::...` paths nested inside
        // struct bodies, impl blocks, function signatures, etc.
        if let TokenTree::Group(g) = &trees[i] {
            let cleaned_inner = rewrite_crate_paths_recursive(g.stream(), preamble_path, repl_map);
            let mut new_group = proc_macro2::Group::new(g.delimiter(), cleaned_inner);
            new_group.set_span(g.span());
            result.extend(Some(TokenTree::Group(new_group)));
            i += 1;
            continue;
        }

        // Check for `crate :: <ident>` pattern.
        let is_crate = matches!(&trees[i], TokenTree::Ident(id) if id == "crate");

        if is_crate && i + 3 < trees.len() {
            let looks_like_path = matches!(&trees[i + 1], TokenTree::Punct(p) if p.as_char() == ':')
                && matches!(&trees[i + 2], TokenTree::Punct(p) if p.as_char() == ':')
                && matches!(&trees[i + 3], TokenTree::Ident(_));

            if looks_like_path && let TokenTree::Ident(mod_name) = &trees[i + 3] {
                let name = mod_name.to_string();
                match name.as_str() {
                    "alloc" | "core" | "boxed" => {
                        // Route crate::core::..., crate::alloc::..., and
                        // crate::boxed::... through the preamble, which
                        // re-exports from __rustyfill_builtin_core,
                        // __rustyfill_builtin_alloc, and the mirrored
                        // crate::std::boxed module respectively.
                        result.extend(token_stream_from_str(preamble_path));
                        i += 4;
                        continue;
                    }
                    _ => {
                        // Check if the next segment (after `crate::<mod>::`) matches
                        // a configured replacement leaf. For example,
                        // `crate::boxed::Box<T,A>` where "Box" has a replacement.
                        if i + 6 < trees.len()
                            && matches!(&trees[i + 4], TokenTree::Punct(p) if p.as_char() == ':')
                            && matches!(&trees[i + 5], TokenTree::Punct(p) if p.as_char() == ':')
                            && matches!(&trees[i + 6], TokenTree::Ident(_))
                            && let TokenTree::Ident(leaf_ident) = &trees[i + 6]
                        {
                            let leaf = leaf_ident.to_string();
                            if let Some(replacement) = repl_map.get(&leaf) {
                                i += 7; // skip `crate :: <mod> :: <leaf>`
                                // Skip `< ... >` generic arguments if present.
                                if i < trees.len()
                                    && let TokenTree::Punct(p) = &trees[i]
                                    && p.as_char() == '<'
                                {
                                    i = skip_angle_brackets(&trees, i);
                                }
                                if let Some(repl_text) = replacement {
                                    result.extend(token_stream_from_str(repl_text));
                                }
                                // If replacement is None, just drop it entirely.
                                continue;
                            }
                        }
                    }
                }
            }
        }

        // Handle bare identifiers that have replacements (e.g., `Global` → `()`).
        // Only applied when the identifier is NOT immediately followed by `::`,
        // so that routed absolute paths (`crate::...`, `::__rustyfill_builtin_...`)
        // and module-qualified references are never clobbered.
        if let TokenTree::Ident(id) = &trees[i]
            && !(i + 2 < trees.len()
                && matches!(&trees[i + 1], TokenTree::Punct(p) if p.as_char() == ':')
                && matches!(&trees[i + 2], TokenTree::Punct(p) if p.as_char() == ':'))
        {
            let name = id.to_string();
            if let Some(Some(repl_text)) = repl_map.get(&name) {
                result.extend(token_stream_from_str(repl_text));
                i += 1;
                continue;
            }
        }

        result.extend(Some(trees[i].clone()));
        i += 1;
    }

    result
}

/// Advance past a `< ... >` pair, handling nested angle brackets and groups.
/// Returns the index of the first token after the closing `>`.
fn skip_angle_brackets(trees: &[TokenTree], start: usize) -> usize {
    let mut depth = 0;
    let mut i = start;
    while i < trees.len() {
        match &trees[i] {
            TokenTree::Punct(p) if p.as_char() == '<' => {
                depth += 1;
            }
            TokenTree::Punct(p) if p.as_char() == '>' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            TokenTree::Group(_) => {
                // Groups are self-contained; don't enter them.
            }
            _ => {}
        }
        i += 1;
    }
    i
}

/// Create a TokenStream from a string of whitespace-separated tokens.
fn token_stream_from_str(s: &str) -> TokenStream {
    s.parse().unwrap_or_default()
}

/// Write parsed items to an output file, creating parent directories as needed.
/// Returns `true` if the file was written (items were non-empty).
pub fn emit_binding_file(output_path: &Path, items: &[ParsedItem], config: &EmitConfig) -> bool {
    if items.is_empty() {
        return false;
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // Compute the path from this module back to the preamble.
    // The manifest wraps all files under `pub mod std { ... }`, and the preamble
    // lives as a sibling module: `std::__rustyfill_prelude`. A file at depth N
    // in the manifest (e.g. std::collections::btree::borrow = depth 3) needs
    // exactly N `super::` hops to reach `std`, then `::__rustyfill_prelude`.
    // Since file_module_depth counts only the file's own path segments
    // (collections/btree/borrow = 3), and the manifest adds one more level
    // (the `std` wrapper), the number of supers equals file_module_depth.
    //
    // When the computed path points at the crate root itself (depth-0 files,
    // where `supers` is empty), the preamble use would be
    // `crate::__rustyfill_prelude::*` — i.e., it points into the same crate
    // that already contains the preamble as its own child module. Emitting it
    // unconditionally would glob-import every prelude name into the crate root
    // for no benefit, so we omit it; references are routed directly instead.
    let supers: Vec<&str> = std::iter::repeat_n("super", config.file_module_depth).collect();
    let preamble_use_path = if supers.is_empty() {
        String::new()
    } else {
        format!("{}::{PREAMBLE_MOD}", supers.join("::"))
    };

    // Build combined use statements: resolved imports + sibling module aliases.
    // Deduplicate by tracking which module names we've already imported.
    let mut all_uses = config.extra_uses.to_vec();
    let mut imported_names: HashSet<String> = HashSet::new();
    for line in config.extra_uses.iter() {
        // Extract the bound name from each use statement for dedup tracking.
        // Handles: "use foo as bar;", "use foo::bar;", "use foo::*;"
        let trimmed = line
            .trim()
            .strip_prefix("#[allow(unused_imports)]")
            .unwrap_or(line)
            .trim();
        if let Some(use_body) = trimmed.strip_prefix("use ") {
            let body = use_body.strip_suffix(';').unwrap_or(use_body);
            // Check for "as NAME" suffix
            if let Some((_, alias_name)) = body.rsplit_once(" as ") {
                imported_names.insert(alias_name.trim().to_string());
            } else if body.ends_with("::*") {
                // Glob import — extract the module name
                if let Some(mod_path) = body.strip_suffix("::*")
                    && let Some(last_seg) = mod_path.rsplit_once(':').map(|(_, n)| n.trim())
                {
                    imported_names.insert(last_seg.to_string());
                }
            } else if let Some(last_seg) = body.rsplit_once(':').map(|(_, n)| n.trim()) {
                imported_names.insert(last_seg.to_string());
            }
        }
    }
    for sib in config.sibling_modules {
        if !imported_names.contains(sib) {
            all_uses.push(format!("#[allow(unused_imports)] use super::{sib};"));
        }
    }

    // Compute the module path from the relative file path
    // (e.g., "collections/btree/set.rs" -> "collections::btree::set").
    let module_path = config
        .relative_file_path
        .strip_suffix(".rs")
        .unwrap_or(config.relative_file_path)
        .strip_suffix("/mod")
        .unwrap_or(config.relative_file_path.strip_suffix(".rs").unwrap_or(""))
        .replace('/', "::");

    let mut content = emit_parsed_items(items, config, &preamble_use_path, &all_uses, &module_path);

    // Append manual trait impls for types whose derives were stripped or
    // whose inner types lack the required impls in our mirrored tree.
    append_manual_impls(&mut content, config.relative_file_path);

    // Run the assembled file through cargo-fmt (via rustfmt) so that every
    // generated binding is formatted consistently. Falls back to a light
    // internal normalizer when rustfmt is unavailable or chokes.
    let content = format_source(&content);

    std::fs::write(output_path, content)
        .unwrap_or_else(|e| panic!("Failed to write {}: {}", output_path.display(), e));
    true
}

/// Append hand-written trait impls that the mirrored types need but can't
/// derive (because their inner types lack the corresponding impls in our
/// synthetic tree). These are stubs sufficient for type-checking; the polyfill
/// provides real implementations where needed.
fn append_manual_impls(_content: &mut String, _relative_file_path: &str) {
    // No hand-written trait impls are currently required. The mirrored B-tree
    // node types that once needed stub `Iterator`/`Debug`/`Clone` impls have
    // been reduced to their core data structures; any ergonomic methods belong
    // in the main crate rather than the bindings generator.
}

/// Emit the preamble module file for a given target library.
/// Writes to `$OUT_DIR/__rustyfill_prelude_<lib>.rs`.
pub fn emit_preamble_module(out_dir: &Path, _lib_name: &str) -> String {
    let filename = format!("{}_{}.rs", PREAMBLE_MOD, _lib_name);
    let path = out_dir.join(&filename);
    let content = format_source(&preamble_content());
    std::fs::write(&path, &content)
        .unwrap_or_else(|e| panic!("Failed to write preamble {}: {}", path.display(), e));
    filename
}

// ── Alias file emission ────────────────────────────────────────────────────

/// Generate alias files for a glob re-export relationship.
///
/// When a module at `alias_module` does `pub use <canonical_module>::*`, this
/// function creates alias files for every emitted canonical file under the
/// canonical module, placing them at the corresponding path under the alias
/// module.
///
/// Returns the list of `(alias_file_path, lib_name)` tuples that were created.
pub fn emit_glob_reexport_aliases(
    resolver: &mut ModuleResolver,
    alias_module: &str,
    canonical_module: &str,
    lib_name: &str,
    out_dir: &Path,
    discovered: &mut HashSet<String>,
    emitted_canonicals: &HashSet<String>,
) -> Vec<(String, String)> {
    let mut results = Vec::new();

    let canon_leaves = resolver.find_files_under(canonical_module);

    for canon_file in canon_leaves {
        // Skip structural parents that had no actual type definitions emitted.
        if !emitted_canonicals.contains(&canon_file) {
            continue;
        }

        let canon_module_path = resolver.file_to_module_path(&canon_file);
        let relative = strip_prefix(&canon_module_path, canonical_module);

        if relative.is_empty() {
            continue;
        }

        let alias_path = if alias_module.is_empty() {
            relative.clone()
        } else {
            format!("{}/{}", alias_module, relative)
        };

        if !discovered.insert(alias_path.clone()) {
            continue;
        }

        let alias_parent = alias_path.rsplit_once('/').map(|(p, _)| p).unwrap_or("");
        let reexport_path = compute_reexport_path(alias_parent, &canon_module_path);
        // When the alias and canonical are at the same level, `reexport_path` is
        // empty. We emit `super::*` to glob-re-export all items from the parent.
        // Bare `pub use super;` is not valid Rust (imports must be explicitly named).
        let reexport_use = if reexport_path.is_empty() {
            "super::*".to_string()
        } else {
            format!("super::{reexport_path}::*")
        };

        // Mirror the canonical file's naming convention: if the canonical is
        // a `mod.rs`, the alias is also a `mod.rs` (directory module).
        let is_mod_rs = canon_file.ends_with("/mod.rs") || canon_file == "mod.rs";
        let alias_file = if is_mod_rs {
            format!("{}/mod.rs", alias_path)
        } else {
            format!("{}.rs", alias_path)
        };
        let alias_output = out_dir.join(&alias_file);
        if let Some(parent) = alias_output.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        let raw_content = format!(
            "// Auto-generated alias by rustyfill-sys.\n\
             // Re-exports from canonical module: {}\n\n\
             pub use {reexport_use};\n",
            canon_file,
        );
        let content = format_source(&raw_content);

        std::fs::write(&alias_output, &content)
            .unwrap_or_else(|e| panic!("Failed to write alias {}: {}", alias_file, e));

        // Register the alias file with the resolver so that validation can
        // resolve its single pub use statement back to the canonical target.
        let parsed = crate::parser::parse_source_with_cfg(
            &content,
            &crate::parser::CfgContext {
                target_os: None,
                target_family: None,
                target_arch: None,
                target_env: None,
                target_vendor: None,
                is_unix: false,
                is_windows: false,
            },
        );
        resolver.register_source(&alias_file, parsed);

        results.push((alias_file, lib_name.to_string()));
    }

    results
}

// ── Manifest generation ────────────────────────────────────────────────────

/// Build a hierarchical module tree from all file paths and emit it as
/// `bindings_generated.rs`. Also declares the preamble module at each
/// target library root.
pub fn emit_hierarchical_manifest(out_dir: &Path, all_files: &[(String, String)]) {
    // Collect which libraries contributed files so we know which preambles to emit.
    let mut libs_seen: BTreeSet<String> = BTreeSet::new();

    // Build per-library trees as before.
    let mut by_lib: BTreeMap<String, TreeNode> = BTreeMap::new();

    for (rel_path, lib_name) in all_files {
        libs_seen.insert(lib_name.clone());
        let node = by_lib.entry(lib_name.clone()).or_default();
        let stem = rel_path.strip_suffix(".rs").unwrap_or(rel_path);
        let module_path = stem.strip_suffix("/mod").unwrap_or(stem);
        let parts: Vec<&str> = module_path.split('/').filter(|s| !s.is_empty()).collect();
        node.insert(parts, rel_path.clone());
    }

    let manifest_path = out_dir.join("bindings_generated.rs");
    let mut content = String::new();

    content.push_str(
        "// Auto-generated manifest by rustyfill-sys.\n\
         // Hierarchical module tree mirroring std/core/alloc structure.\n\
         // All types are intentionally public.\n\n",
    );

    // Emit a single wrapper module around all generated bindings. Inside it,
    // emit preamble modules for every contributing library, then merge all
    // library subtrees.
    content.push_str(&format!("pub mod {WRAPPER_MOD} {{\n"));

    // Emit a single preamble module. All library preambles have identical
    // content (same set of re-exports), so including one is sufficient.
    content.push_str(&format!("    pub mod {} {{\n", PREAMBLE_MOD));
    if let Some(first_lib) = libs_seen.iter().next() {
        let preamble_filename = format!("{}_{}.rs", PREAMBLE_MOD, first_lib);
        content.push_str(&format!(
            "        include!(concat!(env!(\"OUT_DIR\"), \"/{}\"));\n",
            preamble_filename
        ));
    }
    content.push_str("    }\n\n");

    // Merge all library trees into one and emit children at depth 1.
    // The first library's tree provides the base; subsequent libraries' trees
    // are merged into it so their nodes share the same hierarchy.
    if let Some((first_lib, first_tree)) = by_lib.iter().next() {
        // Merge remaining trees into the first one.
        let mut merged = TreeNode::default();
        merge_trees(&mut merged, first_tree);
        for (_lib_name, other_tree) in by_lib.iter().skip(1) {
            merge_trees(&mut merged, other_tree);
        }

        // Emit merged children at depth 1 (inside `pub mod std`).
        for (child_name, child_node) in &merged.children {
            child_node.emit(&mut content, child_name, 1, first_lib);
        }
    }

    content.push_str("}\n\n");

    // Format the manifest like every other emitted file. The internal
    // `include!` macro invocations are opaque to rustfmt (it never expands
    // macros), so formatting is safe here.
    let content = format_source(&content);

    std::fs::write(&manifest_path, content).unwrap_or_else(|e| {
        panic!(
            "Failed to write manifest {}: {}",
            manifest_path.display(),
            e
        )
    });
}

/// Recursively merge `other` into `target`. If a child exists in both, its
/// subtree is merged recursively. File paths from `other` overwrite `target`
/// when both define the same leaf.
fn merge_trees(target: &mut TreeNode, other: &TreeNode) {
    if let Some(fp) = &other.file_path {
        target.file_path = Some(fp.clone());
    }
    for (name, other_child) in &other.children {
        let target_child = target.children.entry(name.clone()).or_default();
        merge_trees(target_child, other_child);
    }
}

/// A node in the module tree. Either a directory (with children) or a leaf file.
#[derive(Default)]
struct TreeNode {
    children: BTreeMap<String, TreeNode>,
    file_path: Option<String>,
}

impl TreeNode {
    #[allow(unused)]
    fn new() -> Self {
        Self::default()
    }

    fn insert(&mut self, parts: Vec<&str>, full_path: String) {
        if parts.is_empty() {
            return;
        }

        let (first, rest) = parts.split_first().unwrap();
        let child = self.children.entry(first.to_string()).or_default();

        if rest.is_empty() {
            child.file_path = Some(full_path);
        } else {
            child.insert(rest.to_vec(), full_path);
        }
    }

    fn emit(&self, out: &mut String, name: &str, depth: usize, lib_name: &str) {
        let indent = "    ".repeat(depth);
        let sname = sanitize(name);

        // At depth 0 (the library root, e.g. `std`), inject the preamble module.
        if depth == 0 {
            out.push_str(&format!("{indent}pub mod {sname} {{\n"));
            // Declare the preamble module
            let preamble_filename = format!("{}_{}.rs", PREAMBLE_MOD, lib_name);
            out.push_str(&format!("{indent}    pub mod {} {{\n", PREAMBLE_MOD));
            out.push_str(&format!(
                "{indent}        include!(concat!(env!(\"OUT_DIR\"), \"/{preamble_filename}\"));\n",
            ));
            out.push_str(&format!("{indent}    }}\n\n",));
        } else if self.file_path.is_some() && self.children.is_empty() {
            // Pure leaf node with file, no children
            out.push_str(&format!("{indent}pub mod {sname} {{\n"));
            out.push_str(&format!("{indent}    #![allow(warnings)]\n",));
            let fp = self.file_path.as_deref().unwrap();
            out.push_str(&format!(
                "{indent}    include!(concat!(env!(\"OUT_DIR\"), \"/{fp}\"));\n",
            ));
            out.push_str(&format!("{indent}}}\n\n",));
            return;
        } else {
            out.push_str(&format!("{indent}pub mod {sname} {{\n"));
            // If this node also has a file (e.g., node.rs alongside node/marker/),
            // include the file content too. This handles the Rust convention where
            // foo.rs and foo/ can coexist.
            if let Some(fp) = &self.file_path {
                out.push_str(&format!("{indent}    #![allow(warnings)]\n",));
                out.push_str(&format!(
                    "{indent}    include!(concat!(env!(\"OUT_DIR\"), \"/{fp}\"));\n",
                ));
                out.push_str(&format!("{indent}\n",));
            }
        }

        for (_child_name, child_node) in &self.children {
            child_node.emit(out, _child_name, depth + 1, lib_name);
        }

        out.push_str(&format!("{indent}}}\n\n",));
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Strip a module-path prefix from a string, handling the `/` separator correctly.
fn strip_prefix(s: &str, prefix: &str) -> String {
    if s == prefix {
        return String::new();
    }
    let pfx = format!("{}/", prefix);
    s.strip_prefix(&pfx)
        .map(|r| r.to_string())
        .unwrap_or_else(|| s.to_string())
}

/// Given two module paths, compute the `super::...` chain that navigates
/// from one to the other.
fn compute_reexport_path(from_module: &str, to_module: &str) -> String {
    let from_parts: Vec<&str> = from_module.split('/').filter(|s| !s.is_empty()).collect();
    let to_parts: Vec<&str> = to_module.split('/').filter(|s| !s.is_empty()).collect();

    let common_len = from_parts
        .iter()
        .zip(to_parts.iter())
        .take_while(|(a, c)| a == c)
        .count();

    let ups = from_parts.len() - common_len;
    let downs: Vec<&str> = to_parts[common_len..].to_vec();

    let mut segments = Vec::new();
    segments.extend(std::iter::repeat_n("super", ups));
    for d in &downs {
        segments.push(*d);
    }

    segments.join("::")
}

/// Ensure a module name is a valid Rust identifier.
fn sanitize(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();

    if s.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        format!("_{}", s)
    } else {
        s
    }
}

