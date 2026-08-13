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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::path::Path;

use proc_macro2::{TokenStream, TokenTree};

use crate::parser::ParsedItem;
use crate::resolver::ModuleResolver;

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
}

/// Mangled name for the per-target preamble module. Unlikely to collide with
/// any real std/core/alloc module name.
const PREAMBLE_MOD: &str = "__rustyfill_prelude";

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
 "#
    .to_string()
}

// ── Attribute filtering (token-stream level) ───────────────────────────────

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
    matches!(
        name.as_str(),
        "doc"
            | "stable"
            | "unstable"
            | "deprecated"
            | "cfg_attr"
            | "lang"
            | "fundamental"
            | "rustc_pass_by_value"
            | "rustc_skip_array_during_method_dispatch"
            | "rustc_allow_incoherent_impl"
            | "rustc_coherence_is_core"
            | "rustc_do_not_log_fails"
            | "rustc_allow_const_fn_unstable"
            | "rustc_reservation_encountered"
            | "rustc_diagnostic_item"
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
    out.push_str(&format!(
        "#[allow(unused_imports)]\npub use {preamble_use_path}::*;\n"
    ));

    for line in extra_uses {
        out.push_str(line);
        out.push('\n');
    }
    if !extra_uses.is_empty() {
        out.push('\n');
    }

    for item in items {
        // Skip items whose fully qualified path matches an ignored struct.
        let fq_path = if module_path.is_empty() {
            item.name.clone()
        } else {
            format!("{}::{}", module_path, item.name)
        };
        if config.ignored_structs.contains(&fq_path) {
            continue;
        }
        emit_item(&mut out, item, preamble_use_path, config.path_replacements);
    }

    out
}

fn emit_item(
    out: &mut String,
    item: &ParsedItem,
    preamble_use_path: &str,
    path_replacements: &[(String, Option<&str>)],
) {
    // The full_tokens already include all attributes + the item definition.
    // We strip blocked attributes, remove ignored identifiers from trait bounds,
    // substitute replaced types, and widen pub(super)/pub(crate) to pub so that
    // types are accessible from sibling modules in our generated tree.
    // Only names WITHOUT a replacement are stripped; names WITH a replacement
    // are preserved for rewrite_crate_paths to substitute.
    let strip_names: Vec<&str> = path_replacements
        .iter()
        .filter(|(_, v)| v.is_none())
        .map(|(k, _)| k.as_str())
        .collect();
    let clean_tokens = strip_blocked_attributes(item.full_tokens.clone(), &strip_names);
    let widened = widen_visibility(clean_tokens);
    let rewritten = rewrite_crate_paths(widened, preamble_use_path, path_replacements);
    write!(out, "{}", rewritten).ok();
    out.push('\n');
}

/// Widen `pub(super)` and `pub(crate)` visibility to plain `pub` so that
/// generated bindings are accessible across the synthetic module tree.
fn widen_visibility(tokens: TokenStream) -> TokenStream {
    let mut result = TokenStream::new();
    let mut iter = tokens.into_iter().peekable();

    while let Some(tt) = iter.next() {
        match &tt {
            TokenTree::Ident(id) if id == "pub" => {
                // Look ahead for `(` indicating pub(super) or pub(crate)
                if let Some(TokenTree::Group(next)) = iter.peek()
                    && next.delimiter() == proc_macro2::Delimiter::Parenthesis
                {
                    // Consume the parenthesized group and replace with plain pub
                    iter.next();
                    result.extend(Some(tt));
                    continue;
                }
                result.extend(Some(tt));
            }
            _ => {
                result.extend(Some(tt));
            }
        }
    }

    result
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
                        // Route both crate::core::... and crate::alloc::... through
                        // the preamble, which re-exports from __rustyfill_builtin_core
                        // and __rustyfill_builtin_alloc respectively.
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
        if let TokenTree::Ident(id) = &trees[i] {
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
    let supers: Vec<&str> = std::iter::repeat_n("super", config.file_module_depth).collect();
    let preamble_use_path = format!("{}::{PREAMBLE_MOD}", supers.join("::"));

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
    let module_path = config.relative_file_path
        .strip_suffix(".rs")
        .unwrap_or(config.relative_file_path)
        .strip_suffix("/mod")
        .unwrap_or(config.relative_file_path.strip_suffix(".rs").unwrap_or(""))
        .replace('/', "::");

    let content = emit_parsed_items(items, config, &preamble_use_path, &all_uses, &module_path);
    std::fs::write(output_path, content)
        .unwrap_or_else(|e| panic!("Failed to write {}: {}", output_path.display(), e));
    true
}

/// Emit the preamble module file for a given target library.
/// Writes to `$OUT_DIR/__rustyfill_prelude_<lib>.rs`.
pub fn emit_preamble_module(out_dir: &Path, _lib_name: &str) -> String {
    let filename = format!("{}_{}.rs", PREAMBLE_MOD, _lib_name);
    let path = out_dir.join(&filename);
    let content = preamble_content();
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
        let reexport_path = if reexport_path.is_empty() {
            "super".to_string()
        } else {
            format!("super::{reexport_path}")
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

        let content = format!(
            "// Auto-generated alias by rustyfill-sys.\n\
             // Re-exports from canonical module: {}\n\n\
             pub use {reexport_path};\n",
            canon_file,
        );

        std::fs::write(&alias_output, content)
            .unwrap_or_else(|e| panic!("Failed to write alias {}: {}", alias_output.display(), e));

        results.push((alias_file, lib_name.to_string()));
    }

    results
}

// ── Manifest generation ────────────────────────────────────────────────────

/// Build a hierarchical module tree from all file paths and emit it as
/// `bindings_generated.rs`. Also declares the preamble module at each
/// target library root.
pub fn emit_hierarchical_manifest(out_dir: &Path, all_files: &[(String, String)]) {
    let mut by_lib: BTreeMap<String, TreeNode> = BTreeMap::new();

    for (rel_path, lib_name) in all_files {
        let node = by_lib.entry(lib_name.clone()).or_default();
        // Convert file path to module path segments: strip `.rs`, then drop
        // a trailing `mod` segment since `foo/mod.rs` defines the module `foo`,
        // not `foo::mod`.
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

    for (lib_name, tree) in &by_lib {
        tree.emit(&mut content, "std", 0, lib_name);
    }

    std::fs::write(&manifest_path, content).unwrap_or_else(|e| {
        panic!(
            "Failed to write manifest {}: {}",
            manifest_path.display(),
            e
        )
    });
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
                "{indent}        include!(concat!(env!(\"OUT_DIR\"), \"/{}\"));\n",
                preamble_filename
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
