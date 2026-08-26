//! Parses Rust source files and extracts item definitions (structs, enums, unions)
//! as well as `use` statements for module resolution.
//!
//! Also provides [`ParsedSource`] which bundles all outputs from a single parse
//! pass, and the `register_source` method which feeds results directly into a
//! [`ModuleResolver`](crate::resolver::ModuleResolver).
//!
//! When the AST yields no module declarations (e.g., because they're wrapped in
//! macros like `cfg_select!`), a text-based scanner evaluates cfg predicates
//! against the current build target and only follows the active branch.

use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::{Attribute, Item, UseTree};

use crate::resolver::{PathSegment, PathSegmentList, UseKind, UseStatement, Visibility};

/// Visibility of a parsed item, as written in the source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemVisibility {
    /// No visibility modifier (private to its defining module).
    Private,
    /// `pub` — visible everywhere.
    Public,
    /// `pub(crate)` / `pub(super)` / `pub(in path)` — restricted scope.
    Restricted,
}

impl ItemVisibility {
    /// True for plain `pub`. Restricted visibilities are NOT public: they do
    /// not make the item reachable through re-exports outside their scope.
    pub fn is_public(&self) -> bool {
        matches!(self, ItemVisibility::Public)
    }
}

/// A parsed top-level item extracted from a Rust source file.
#[derive(Clone)]
pub struct ParsedItem {
    /// All attributes on the item (`#[repr(C)]`, `#[derive(...)]`, cfg gates, etc.)
    pub attrs: Vec<Attribute>,
    /// The complete token stream of the item (vis + keyword + name + generics + body).
    pub full_tokens: TokenStream,
    /// Kind of item, determines output structure.
    pub kind: ItemKind,
    /// The identifier name of the item (e.g., `"BTreeMap"`, `"Iter"`).
    /// Used by the spec to match against fully qualified ignore paths.
    pub name: String,
    /// Source visibility of the item, used when checking field-type publicity.
    pub visibility: ItemVisibility,
    /// For type aliases only: the right-hand-side type expression tokens
    /// (`type Root<K, V> = NodeRef<...>` → `NodeRef<...>`). `None` for all
    /// other item kinds and for text-scanner-extracted items.
    pub alias_rhs: Option<TokenStream>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKind {
    Struct,
    Enum,
    Union,
    Const,
    TypeAlias,
}

impl ItemKind {
    /// True for item kinds that define a type (struct, enum, union, alias).
    /// Constants are excluded — they are not resolvable as type references.
    pub fn is_type_def(self) -> bool {
        !matches!(self, ItemKind::Const)
    }
}

/// An external module declaration (`mod X;` / `pub mod X;`) carrying its
/// cfg attributes so consumers can decide whether the module is active.
#[derive(Clone)]
pub struct ModDeclaration {
    /// Module name (e.g., `"entry"`, `"tests"`).
    pub name: String,
    /// Attributes on the mod declaration (`#[cfg(test)]`, `#[cfg(unix)]`, etc.).
    pub attrs: Vec<Attribute>,
}

/// Results of parsing a single `.rs` source file in one pass.
#[derive(Clone)]
pub struct ParsedSource {
    /// Top-level type definitions (structs, enums, unions).
    pub items: Vec<ParsedItem>,
    /// `use` statements for module resolution.
    pub use_statements: Vec<UseStatement>,
    /// External module declarations (`mod X;` / `pub mod X;`) that reference
    /// separate files. Inline modules (`mod X { ... }`) are excluded.
    pub mod_declarations: Vec<ModDeclaration>,
    /// Inline modules (`mod X { ... }`) that contain type-defining items.
    /// Each entry is `(module_name, items)` so the emitter can write them
    /// to a separate `<parent>/<name>/mod.rs` file.
    pub inline_modules: Vec<(String, Vec<ParsedItem>)>,
    /// `use` statements declared inside inline modules, keyed by module name.
    /// Needed for alias discovery through re-exports (e.g., `pub use map::*;`).
    pub inline_module_uses: HashMap<String, Vec<UseStatement>>,
}

/// Build-time cfg context for evaluating conditional compilation predicates.
/// Populated from cargo's `CARGO_CFG_*` environment variables in the build script.
#[derive(Clone, Default)]
pub struct CfgContext {
    pub target_os: Option<String>,
    pub target_family: Option<String>,
    pub target_arch: Option<String>,
    pub target_env: Option<String>,
    pub target_vendor: Option<String>,
    pub is_unix: bool,
    pub is_windows: bool,
}

impl CfgContext {
    /// Populate from cargo build-script environment variables.
    pub fn from_env() -> Self {
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").ok();
        let target_family = std::env::var("CARGO_CFG_TARGET_FAMILY").ok();
        let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").ok();
        let target_env = std::env::var("CARGO_CFG_TARGET_ENV").ok();
        let target_vendor = std::env::var("CARGO_CFG_TARGET_VENDOR").ok();
        // Cargo exports CARGO_CFG_TARGET_FAMILY when present but not always;
        // fall back to deriving it from the OS so the two constructors agree.
        let target_family =
            target_family.or_else(|| Self::family_for_os(target_os.as_deref()).map(String::from));
        // Cargo does not export CARGO_CFG_UNIX, so derive unix-ness from the
        // resolved family / OS instead. Without this, cfg_select! branches
        // keyed on the bare `unix` predicate never activate and the resolver
        // falls through to the `_` fallback (e.g. picking `unsupported` over
        // `unix`).
        let is_unix = target_family.as_deref() == Some("unix")
            || Self::family_for_os(target_os.as_deref()) == Some("unix");
        let is_windows = target_os.as_deref() == Some("windows");

        Self {
            target_os,
            target_family,
            target_arch,
            target_env,
            target_vendor,
            is_unix,
            is_windows,
        }
    }

    /// Build a context from a Rust target triple (e.g.
    /// `x86_64-unknown-linux-gnu`). Cargo always sets `TARGET`, but does NOT
    /// export the individual `CARGO_CFG_*` variables to build scripts, so this
    /// is the reliable way to learn the platform for cfg_select! evaluation.
    pub fn from_target_triple(triple: &str) -> Self {
        let parts: Vec<&str> = triple.split('-').collect();
        // Known vendors in the traditional 3-part triple layout
        // (`arch-vendor-os[-env]`). Anything else in the vendor slot is an
        // OS/architecture fragment we don't model.
        let known_vendors = [
            "unknown", "pc", "none", "fortanix", "apple", "nintendo", "sony", "uwp", "hurd",
            "contiki", "newlib", "hermit", "kmc", "wrs", "gnu", "musl", "win7",
        ];
        let is_known_vendor = |p: &str| known_vendors.contains(&p);
        // Environment suffixes that can appear at the end of a triple.
        let is_env_suffix = |p: &str| {
            matches!(
                p,
                "gnu"
                    | "musl"
                    | "msvc"
                    | "win7"
                    | "haiku"
                    | "none"
                    | "kernel"
                    | "softfloat"
                    | "double"
                    | "eabi"
                    | "eabihf"
                    | "armv6"
                    | "armv7"
                    | "thumbv6"
                    | "thumbv7"
                    | "thumbv8"
                    | "qemu"
                    | "simulator"
            )
        };

        // Layout disambiguation. The traditional triple is `arch-vendor-os[-env]`
        // with a closed set of known vendors; modern triples moved the OS into
        // the second slot (`arch-os-vendor[-env]`). We detect the traditional
        // layout when the second segment is a known vendor AND the third
        // segment is not itself an environment suffix (which would make it the
        // OS in a modern layout, e.g. `x86_64-unknown-linux-musl`).
        let (vendor_slot, os_seg, env_seg): (Option<&str>, Option<&str>, Option<&str>) =
            match parts.len() {
                1 => (None, None, None),
                2 => (None, parts.get(1).copied(), None),
                _ => {
                    let seg1 = parts.get(1).copied();
                    let seg2 = parts.get(2).copied();
                    let last = parts.last().copied();
                    let traditional =
                        seg1.is_some_and(is_known_vendor) && !seg2.is_some_and(is_env_suffix);
                    if traditional {
                        // arch-vendor-os[-env]: vendor in slot 2, OS in slot 3,
                        // trailing env when present.
                        (
                            seg1,
                            seg2,
                            if parts.len() >= 4 && last.is_some_and(is_env_suffix) {
                                last
                            } else {
                                None
                            },
                        )
                    } else {
                        // arch-os-vendor[-env]: OS in slot 2, vendor in slot 3,
                        // trailing env when present.
                        (
                            seg2,
                            seg1,
                            if parts.len() >= 4 && last.is_some_and(is_env_suffix) {
                                last
                            } else {
                                None
                            },
                        )
                    }
                }
            };

        // Normalize triple OS names to their `cfg(target_os = "...")` values,
        // then derive the family from the (normalized) OS through the single
        // shared lookup so this constructor agrees with `from_env`.
        let target_os = os_seg.map(normalize_target_os);
        let target_family = Self::family_for_os(target_os.as_deref()).map(String::from);
        // The vendor slot feeds `cfg(target_vendor = "...")` predicates such as
        // `not(target_vendor = "win7")` in cfg_select!. Without this, Windows
        // targets fall through to the wrong backend branch (e.g. no_threads).
        let target_vendor = vendor_slot.filter(|p| is_known_vendor(p)).map(String::from);
        let is_unix = target_family.as_deref() == Some("unix");
        let is_windows = target_os.as_deref() == Some("windows");
        Self {
            target_os,
            target_family,
            target_arch: parts.first().map(|s| s.to_string()),
            target_env: env_seg.map(String::from),
            target_vendor,
            is_unix,
            is_windows,
        }
    }

    /// Derive `target_family` from a normalized `cfg(target_os)` value. This is
    /// the single source of truth for family classification, shared by both
    /// `from_env` and `from_target_triple`, so adding an OS here updates every
    /// code path at once instead of scattering per-OS lists.
    fn family_for_os(os: Option<&str>) -> Option<&'static str> {
        match os? {
            "windows" => Some("windows"),
            "linux" | "android" | "macos" | "ios" | "freebsd" | "netbsd" | "openbsd"
            | "dragonfly" | "solaris" | "illumos" | "aix" | "haiku" | "l4re" | "horizon"
            | "emscripten" | "wasi" => Some("unix"),
            _ => None,
        }
    }
}

/// Map a target-triple OS segment to its `cfg(target_os = "...")` value. Most
/// segments are already the cfg name; a few (notably Apple's) differ between
/// the triple spelling and the cfg spelling.
fn normalize_target_os(os: &str) -> String {
    match os {
        "darwin" => "macos".to_string(),
        // iOS triples spell the OS as `ios` already in modern toolchains, but
        // older ones used `ios` too — keep as-is.
        other => other.to_string(),
    }
}

impl CfgContext {
    /// Evaluate a simple cfg predicate string against this context.
    /// Supports: bare names (`unix`), key-value pairs (`target_os = "linux"`),
    /// `all(...)`, `any(...)`, `not(...)`.
    pub fn eval_predicate(&self, pred: &str) -> bool {
        let pred = pred.trim();
        // Peel one balanced-paren group at a time. This handles nested
        // predicates like `not(any(a, b))` where the outermost token is `not`
        // but the string ends with two closing parens.
        if let Some((func, inner)) = Self::peel_outer_parens(pred) {
            return match func {
                "all" => split_commas(inner)
                    .into_iter()
                    .all(|p| self.eval_predicate(&p)),
                "any" => split_commas(inner)
                    .into_iter()
                    .any(|p| self.eval_predicate(&p)),
                "not" => !self.eval_predicate(inner),
                _ => self.eval_atom(pred),
            };
        }
        self.eval_atom(pred)
    }

    /// Split `name(args...)` into `(name, args)` when the entire input is a
    /// single balanced parenthesized call. Returns `None` for atoms or
    /// malformed input.
    fn peel_outer_parens(s: &str) -> Option<(&str, &str)> {
        let open = s.find('(')?;
        let name = s[..open].trim();
        // Walk from the first `(` to its matching `)`. That close must be the
        // last non-whitespace character in the string.
        let bytes = s.as_bytes();
        let mut depth = 0i32;
        for i in open..bytes.len() {
            match bytes[i] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        if s[i + 1..].trim().is_empty() {
                            return Some((name, &s[open + 1..i]));
                        }
                        return None;
                    }
                }
                _ => {}
            }
        }
        None
    }

    fn eval_atom(&self, atom: &str) -> bool {
        let atom = atom.trim();

        // Key-value pair: `target_os = "linux"`
        if let Some(eq_pos) = atom.find('=') {
            let key = atom[..eq_pos].trim();
            let value = atom[eq_pos + 1..].trim().trim_matches('"');

            return match key {
                "target_os" => self.target_os.as_deref() == Some(value),
                "target_family" => self.target_family.as_deref() == Some(value),
                "target_arch" => self.target_arch.as_deref() == Some(value),
                "target_env" => self.target_env.as_deref() == Some(value),
                "target_vendor" => self.target_vendor.as_deref() == Some(value),
                _ => false,
            };
        }

        // Bare name
        match atom {
            "unix" => self.is_unix,
            "windows" => self.is_windows,
            "target_thread_local" => true, // commonly used, always true on supported platforms
            _ => false,
        }
    }
}

/// Check whether any `#[cfg(...)]` attribute on a module declaration evaluates
/// to false under the given build context. Returns `true` if the module should
/// be skipped (e.g., `#[cfg(test)]` on a non-test build). Modules with no cfg
/// attributes or whose all cfg predicates evaluate to true are considered active.
pub fn is_cfg_inactive(attrs: &[Attribute], cfg: &CfgContext) -> bool {
    for attr in attrs {
        // We only care about `#[cfg(...)]`, not `#[cfg_attr(...)]`.
        let meta = match &attr.meta {
            syn::Meta::List(ml) if ml.path.is_ident("cfg") => ml,
            _ => continue,
        };

        // Token-stream the inner predicate and evaluate it as text.
        let pred_text = meta.tokens.to_string();
        if !cfg.eval_predicate(&pred_text) {
            return true;
        }
    }
    false
}

/// True when the source file is excluded from compilation for the given build
/// context via an inner `#![cfg(...)]` attribute (e.g.
/// `sys/pal/unix/sync/mod.rs`, which carries
/// `#![cfg(not(any(target_os = "linux", ...)))]`). Inner attributes are not
/// surfaced by [`parse_source_with_cfg`]'s item walk, so this scans the raw
/// text directly. Returns `None` when the file has no cfg inner attribute
/// (i.e., it is unconditionally active).
pub fn module_file_cfg_excluded(source: &str, cfg: &CfgContext) -> Option<bool> {
    // Strip comments but keep string literals intact: the cfg predicate values
    // (`target_os = "linux"`) live inside strings and must survive.
    let no_comments = strip_line_comments(source);
    let mut lines = no_comments.lines().peekable();
    while let Some(raw) = lines.next() {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !trimmed.starts_with("#![") {
            // Inner attributes must precede all items; the first non-attribute
            // line ends the attribute block.
            break;
        }
        // Accumulate continuation lines until the attribute's parens balance,
        // since `#![cfg(not(any(...)))]` frequently spans multiple lines.
        let mut buf = String::from(trimmed);
        loop {
            if balanced_parens(&buf) {
                break;
            }
            match lines.next() {
                Some(cont) => {
                    buf.push(' ');
                    buf.push_str(cont.trim());
                }
                None => break,
            }
        }
        // Look for a `cfg(...)` list inside this attribute.
        if let Some(pred) = extract_inner_list_arg(&buf, "cfg") {
            return Some(cfg.eval_predicate(&pred));
        }
    }
    None
}

/// Remove `//` line comments from source text, preserving newlines and string
/// literals. Unlike [`strip_comments_and_strings`], string contents are kept —
/// needed when inspecting cfg predicates whose values live in quoted strings.
fn strip_line_comments(source: &str) -> String {
    let mut result = String::with_capacity(source.len());
    let chars: Vec<char> = source.chars().collect();
    let len = chars.len();
    let mut i = 0;
    while i < len {
        let ch = chars[i];
        if ch == '/' && i + 1 < len && chars[i + 1] == '/' {
            while i < len && chars[i] != '\n' {
                i += 1;
            }
        } else {
            result.push(ch);
            i += 1;
        }
    }
    result
}

/// True when every `(` in `s` has a matching `)` (parentheses are balanced).
fn balanced_parens(s: &str) -> bool {
    let mut depth = 0i32;
    for &b in s.as_bytes() {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

/// Given an attribute text like `#![cfg(not(any(a, b)))]`, return the
/// parenthesized argument text of the named list (`not(any(a, b))` here).
fn extract_inner_list_arg(attr: &str, name: &str) -> Option<String> {
    let needle = format!("{name}(");
    let start = attr.find(&needle)? + name.len(); // index of '('
    let bytes = attr.as_bytes();
    let mut depth = 0i32;
    for i in start..bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(attr[start + 1..i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Public wrapper around the internal comment/string stripper, used for
/// raw-text attribute inspection where full stripping is appropriate.
pub fn strip_comments_and_strings_pub(source: &str) -> String {
    strip_comments_and_strings(source)
}

/// Split a string by commas, respecting parentheses nesting. Parentheses are
/// preserved in the output tokens so that nested predicates like
/// `not(target_vendor = "win7")` survive intact.
fn split_commas(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();

    for ch in s.chars() {
        match ch {
            '(' => {
                depth += 1;
                current.push(ch);
            }
            ')' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                let token = current.trim().to_string();
                if !token.is_empty() {
                    parts.push(token);
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    let token = current.trim().to_string();
    if !token.is_empty() {
        parts.push(token);
    }

    parts
}

/// Strips glob-from-type use statements (`use TypeName::*;`) that are valid
/// Rust but unsupported by syn 2.x's parser. These lines import associated
/// items from a type (e.g., `use Entry::*;` in btree/map.rs) and cause
/// `syn::parse_file` to fail with "expected identifier or `_`". Removing
/// them lets the rest of the file parse successfully. Module-path globs
/// (`use crate::foo::*;`) are preserved since syn handles those fine.
fn strip_glob_from_type_uses(source: &str) -> String {
    let mut result = String::with_capacity(source.len());
    for line in source.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("use ") {
            let rest_trimmed = rest.trim_end();
            if let Some(star_pos) = rest_trimmed.find("::*;") {
                let path_part = &rest_trimmed[..star_pos];
                // Glob-from-type: a single PascalCase identifier followed by
                // `::*`. Distinguished from module-path globs (`use super::*`,
                // `use crate::foo::*`) by having no `::` separators AND the
                // identifier starting with an uppercase letter (type naming
                // convention). This avoids stripping legitimate module globs.
                let is_single_ident = !path_part.contains("::") && !path_part.contains(' ');
                let is_pascal_case = path_part.chars().next().is_some_and(char::is_uppercase);
                if is_single_ident && is_pascal_case {
                    continue;
                }
            }
        }
        result.push_str(line);
        result.push('\n');
    }
    result
}

/// Parse a single `.rs` file in one pass, extracting type definitions,
/// `use` statements, and external module declarations.
///
/// Uses `cfg_context` to evaluate `cfg_select!` branches when module
/// declarations are wrapped in macro invocations.
pub fn parse_source_with_cfg(source: &str, cfg: &CfgContext) -> ParsedSource {
    let mut items = Vec::new();
    let mut use_statements = Vec::new();
    let mut mod_declarations_from_ast = Vec::new();
    let mut inline_modules: Vec<(String, Vec<ParsedItem>)> = Vec::new();
    let mut inline_module_uses: HashMap<String, Vec<UseStatement>> = HashMap::new();

    // Preprocess: remove glob-from-type use statements that break syn parsing.
    let preprocessed = strip_glob_from_type_uses(source);

    // Try to parse via syn AST. This may fail for files that contain
    // macro-heavy content like cfg_select!, or for nightly toolchains where
    // the stdlib source uses syntax ahead of what syn supports.
    match syn::parse_file(&preprocessed) {
        Err(_e) => {
            // Fall back to text-based extraction when syn can't parse the file
            // (e.g., nightly stdlib source uses syntax ahead of what syn supports)
            let text_result = text_scan_source(source, cfg);
            items = text_result.items;
            use_statements = text_result.use_statements;
            mod_declarations_from_ast = text_result.mod_declarations;
            inline_modules = text_result.inline_modules;
        }
        Ok(file) => {
            for item in &file.items {
                match item {
                    Item::Struct(s) => items.push(parse_struct(s.clone())),
                    Item::Enum(e) => items.push(parse_enum(e.clone())),
                    Item::Union(u) => items.push(parse_union(u.clone())),
                    Item::Const(ic) => items.push(parse_const(ic.clone())),
                    Item::Type(it) => items.push(parse_type_alias(it.clone())),
                    Item::Mod(im) if im.content.is_none() => {
                        mod_declarations_from_ast.push(ModDeclaration {
                            name: im.ident.to_string(),
                            attrs: im.attrs.clone(),
                        });
                    }
                    Item::Mod(im) if im.content.is_some() => {
                        // Inline module — extract type-defining items and its
                        // own use statements (needed for alias discovery).
                        let (_brace, mod_items) = im.content.as_ref().unwrap();
                        let mut inner_items = Vec::new();
                        let mut inner_uses = Vec::new();
                        for inner in mod_items {
                            match inner {
                                Item::Struct(s) => inner_items.push(parse_struct(s.clone())),
                                Item::Enum(e) => inner_items.push(parse_enum(e.clone())),
                                Item::Union(u) => inner_items.push(parse_union(u.clone())),
                                Item::Use(iu) => inner_uses.extend(parse_use_item(iu.clone())),
                                _ => {}
                            }
                        }
                        if !inner_items.is_empty() || !inner_uses.is_empty() {
                            inline_modules.push((im.ident.to_string(), inner_items));
                            if !inner_uses.is_empty() {
                                inline_module_uses.insert(im.ident.to_string(), inner_uses);
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Recursively extract use statements from all scopes in the file,
            // not just top-level ones. This catches imports inside impl blocks,
            // functions, inline modules, etc.
            extract_all_uses_from_file(&file, &mut use_statements);
        }
    }

    // For mod declarations: prefer AST results, but fall back to text-based
    // scanning when the AST yielded nothing (e.g., mods are inside cfg_select!).
    let mod_declarations = if !mod_declarations_from_ast.is_empty() {
        mod_declarations_from_ast
    } else {
        scan_mod_declarations_with_cfg(source, cfg)
    };

    ParsedSource {
        items,
        use_statements,
        mod_declarations,
        inline_modules,
        inline_module_uses,
    }
}

/// Map a syn visibility to the item's source visibility.
fn vis_of(vis: &syn::Visibility) -> ItemVisibility {
    match vis {
        syn::Visibility::Public(_) => ItemVisibility::Public,
        syn::Visibility::Restricted(_) => ItemVisibility::Restricted,
        syn::Visibility::Inherited => ItemVisibility::Private,
    }
}

/// Recursively walk the AST and collect all `use` statements from every scope:
/// top-level, inline modules, impl blocks, trait definitions, functions, etc.
fn extract_all_uses_from_file(file: &syn::File, out: &mut Vec<UseStatement>) {
    for item in &file.items {
        extract_all_uses_from_item(item, out);
    }
}

fn extract_all_uses_from_item(item: &Item, out: &mut Vec<UseStatement>) {
    match item {
        Item::Use(iu) => out.extend(parse_use_item(iu.clone())),
        Item::Mod(im) => {
            if let Some((_brace, mod_items)) = &im.content {
                for inner in mod_items {
                    extract_all_uses_from_item(inner, out);
                }
            }
        }
        Item::Impl(ii) => {
            for inner in &ii.items {
                if let syn::ImplItem::Fn(ifn) = inner {
                    extract_all_uses_from_block(&ifn.block, out);
                }
            }
        }
        Item::Trait(it) => {
            for inner in &it.items {
                if let syn::TraitItem::Fn(tf) = inner {
                    if let Some(block) = &tf.default {
                        extract_all_uses_from_block(block, out);
                    }
                }
            }
        }
        Item::Fn(iff) => {
            extract_all_uses_from_block(&iff.block, out);
        }
        // Macro *invocations* (e.g., `cfg_if! { ... }`) can contain `use`
        // statements in their brace-delimited body. The body tokens are not a
        // plain block (they start with `if #[cfg(...)] {...}`), so walk the
        // token tree manually and parse each brace group as a block.
        Item::Macro(im) => {
            extract_uses_from_macro_tokens(&im.mac.tokens, out);
        }
        Item::ExternCrate(_)
        | Item::Struct(_)
        | Item::Enum(_)
        | Item::Union(_)
        | Item::Type(_)
        | Item::Static(_)
        | Item::Const(_)
        | Item::Verbatim(_)
        | Item::ForeignMod(_) => {}
        _ => {}
    }
}

/// Extract use statements from within a block body. `use` statements can
/// appear as nested items inside function bodies.
fn extract_all_uses_from_block(block: &syn::Block, out: &mut Vec<UseStatement>) {
    for stmt in &block.stmts {
        if let syn::Stmt::Item(item) = stmt {
            extract_all_uses_from_item(item, out);
        }
    }
}

/// Extract `use` statements from a macro invocation's token body. The body of
/// a `cfg_if!` / `cfg_select!` invocation is not a plain block — it starts with
/// `if #[cfg(...)] { ... } else { ... }`, which syn cannot parse as an
/// expression because of the `#[cfg]` attribute. Instead, walk the top-level
/// token tree and wrap each brace-delimited group's content in an outer brace
/// pair so syn can parse it as a [`syn::Block`], extracting any `use` items
/// found inside. Groups whose content fails to parse as a block (e.g., nested
/// macro invocations) are recursed into so their inner brace groups are still
/// visited.
fn extract_uses_from_macro_tokens(tokens: &proc_macro2::TokenStream, out: &mut Vec<UseStatement>) {
    for tt in tokens.clone().into_iter() {
        if let proc_macro2::TokenTree::Group(g) = tt {
            if g.delimiter() != proc_macro2::Delimiter::Brace {
                continue;
            }
            // Wrap the group's content in braces so syn sees a complete block.
            let wrapped: proc_macro2::TokenStream =
                format!("{{ {} }}", g.stream()).parse().unwrap_or_default();
            match syn::parse2::<syn::Block>(wrapped) {
                Ok(block) => {
                    for stmt in &block.stmts {
                        if let syn::Stmt::Item(item) = stmt {
                            extract_all_uses_from_item(item, out);
                        }
                    }
                }
                Err(_) => {
                    // Not a plain block (nested macro body or similar).
                    // Recurse to find inner brace groups.
                    extract_uses_from_macro_tokens(&g.stream(), out);
                }
            }
        }
    }
}

/// Parse a single `.rs` file using auto-detected cfg context. Prefer
/// [`parse_source_with_cfg`] for explicit control.
pub fn parse_source(source: &str) -> ParsedSource {
    parse_source_with_cfg(source, &CfgContext::from_env())
}

/// Legacy wrapper: extract only type definitions. Prefer [`parse_source`].
pub fn parse_file(source: &str) -> Vec<ParsedItem> {
    parse_source(source).items
}

/// Legacy wrapper: extract only `use` statements. Prefer [`parse_source`].
pub fn parse_use_statements(source: &str) -> Vec<UseStatement> {
    parse_source(source).use_statements
}

/// Extract external module declarations (`mod X;`). Prefer [`parse_source`].
pub fn parse_mod_declarations(source: &str) -> Vec<ModDeclaration> {
    parse_source(source).mod_declarations
}

/// Attempt to parse a single `syn::Item` into a `ParsedItem`.
pub fn parse_item(item: Item) -> Option<ParsedItem> {
    match item {
        Item::Struct(s) => Some(parse_struct(s)),
        Item::Enum(e) => Some(parse_enum(e)),
        Item::Union(u) => Some(parse_union(u)),
        _ => None,
    }
}

fn parse_struct(s: syn::ItemStruct) -> ParsedItem {
    let mut tokens = TokenStream::new();
    s.to_tokens(&mut tokens);
    ParsedItem {
        attrs: s.attrs,
        full_tokens: tokens,
        kind: ItemKind::Struct,
        name: s.ident.to_string(),
        visibility: vis_of(&s.vis),
        alias_rhs: None,
    }
}

fn parse_enum(e: syn::ItemEnum) -> ParsedItem {
    let mut tokens = TokenStream::new();
    e.to_tokens(&mut tokens);
    ParsedItem {
        attrs: e.attrs,
        full_tokens: tokens,
        kind: ItemKind::Enum,
        name: e.ident.to_string(),
        visibility: vis_of(&e.vis),
        alias_rhs: None,
    }
}

fn parse_union(u: syn::ItemUnion) -> ParsedItem {
    let mut tokens = TokenStream::new();
    u.to_tokens(&mut tokens);
    ParsedItem {
        attrs: u.attrs,
        full_tokens: tokens,
        kind: ItemKind::Union,
        name: u.ident.to_string(),
        visibility: vis_of(&u.vis),
        alias_rhs: None,
    }
}

fn parse_const(c: syn::ItemConst) -> ParsedItem {
    let mut tokens = TokenStream::new();
    c.to_tokens(&mut tokens);
    ParsedItem {
        attrs: c.attrs,
        full_tokens: tokens,
        kind: ItemKind::Const,
        name: c.ident.to_string(),
        visibility: vis_of(&c.vis),
        alias_rhs: None,
    }
}

fn parse_type_alias(t: syn::ItemType) -> ParsedItem {
    let mut tokens = TokenStream::new();
    t.to_tokens(&mut tokens);
    // Capture the RHS type expression separately so the emitter can mirror
    // declared aliases (routing the RHS through the type registry).
    let mut rhs = TokenStream::new();
    t.ty.to_tokens(&mut rhs);
    ParsedItem {
        attrs: t.attrs,
        full_tokens: tokens,
        kind: ItemKind::TypeAlias,
        name: t.ident.to_string(),
        visibility: vis_of(&t.vis),
        alias_rhs: Some(rhs),
    }
}

/// Extract the item name from a token stream produced by the text scanner.
/// Skips past attributes and visibility keywords to find the struct/enum/union
/// identifier, or the const/type name.
fn extract_item_name_from_tokens(tokens: &TokenStream, kind: ItemKind) -> String {
    let keywords = match kind {
        ItemKind::Struct => "struct",
        ItemKind::Enum => "enum",
        ItemKind::Union => "union",
        ItemKind::Const => "const",
        ItemKind::TypeAlias => "type",
    };

    // Walk tokens: skip `# [...]` attribute groups, skip `pub`, then find the
    // keyword, then grab the next ident as the name.
    let tts: Vec<proc_macro2::TokenTree> = tokens.clone().into_iter().collect();
    let mut i = 0;
    while i < tts.len() {
        match &tts[i] {
            proc_macro2::TokenTree::Group(g)
                if g.delimiter() == proc_macro2::Delimiter::Bracket =>
            {
                // Skip attribute brackets
                i += 1;
                continue;
            }
            proc_macro2::TokenTree::Ident(id) => {
                let s = id.to_string();
                if s == keywords {
                    // Next ident should be the name
                    i += 1;
                    while i < tts.len() {
                        if let proc_macro2::TokenTree::Ident(next) = &tts[i] {
                            return next.to_string();
                        }
                        i += 1;
                    }
                    return String::new();
                }
            }
            _ => {}
        }
        i += 1;
    }
    String::new()
}

fn parse_use_item(iu: syn::ItemUse) -> Vec<UseStatement> {
    let visibility = match &iu.vis {
        syn::Visibility::Public(_) => Visibility::Public,
        syn::Visibility::Restricted(vr) if vr.path.segments.is_empty() => Visibility::PubCrate,
        _ => Visibility::Private,
    };
    parse_use_trees(&iu.tree, visibility)
}

fn parse_use_trees(tree: &UseTree, vis: Visibility) -> Vec<UseStatement> {
    match tree {
        UseTree::Path(use_path) => {
            let kinds = collect_use_kinds(use_path);
            kinds
                .into_iter()
                .map(|kind| UseStatement {
                    visibility: vis.clone(),
                    kind,
                })
                .collect()
        }
        UseTree::Name(_n) => vec![UseStatement {
            visibility: vis,
            kind: UseKind::Single(
                PathSegmentList {
                    segments: Vec::new(),
                },
                None,
            ),
        }],
        UseTree::Rename(r) => vec![UseStatement {
            visibility: vis,
            kind: UseKind::Single(
                PathSegmentList {
                    segments: Vec::new(),
                },
                Some(r.rename.to_string()),
            ),
        }],
        UseTree::Glob(_) => vec![UseStatement {
            visibility: vis,
            kind: UseKind::Glob(PathSegmentList {
                segments: Vec::new(),
            }),
        }],
        UseTree::Group(g) => {
            let mut all = Vec::new();
            for inner in &g.items {
                all.extend(parse_use_trees(inner, vis.clone()));
            }
            all
        }
    }
}

fn collect_use_kinds(use_path: &syn::UsePath) -> Vec<UseKind> {
    let mut segments = vec![ident_to_segment(&use_path.ident)];
    let tail = collect_path_segments(&use_path.tree, &mut segments);
    match tail {
        TreeTerminal::Glob => vec![UseKind::Glob(PathSegmentList { segments })],
        TreeTerminal::Name(name) => {
            segments.push(PathSegment::Named(name));
            vec![UseKind::Single(PathSegmentList { segments }, None)]
        }
        TreeTerminal::Rename(name, alias) => {
            segments.push(PathSegment::Named(name));
            vec![UseKind::Single(PathSegmentList { segments }, Some(alias))]
        }
        TreeTerminal::Group(items) => {
            let mut kinds = Vec::new();
            for inner in &items {
                match inner {
                    // A `use ...::{self, ...}` entry binds the module itself under
                    // its own name. Emit it as a Single whose target path is the
                    // prefix plus the module's own identifier; the resolver maps
                    // the trailing `Self_` back to that identifier.
                    UseTree::Path(p) => {
                        // A `use ...::{self, ...}` entry binds the module itself
                        // under its own name. Emit it as a Single whose target
                        // path is the prefix plus the module's own identifier;
                        // the resolver maps the trailing `Self_` back to that
                        // identifier. (Nested groups are still dropped.)
                        let mut segs = segments.clone();
                        segs.push(ident_to_segment(&p.ident));
                        segs.push(PathSegment::Self_);
                        kinds.push(UseKind::Single(PathSegmentList { segments: segs }, None));
                    }
                    UseTree::Glob(_) => {
                        kinds.push(UseKind::Glob(PathSegmentList {
                            segments: segments.clone(),
                        }));
                    }
                    UseTree::Name(n) => {
                        let mut segs = segments.clone();
                        segs.push(ident_to_segment(&n.ident));
                        kinds.push(UseKind::Single(PathSegmentList { segments: segs }, None));
                    }
                    UseTree::Rename(r) => {
                        let mut segs = segments.clone();
                        segs.push(PathSegment::Named(r.ident.to_string()));
                        kinds.push(UseKind::Single(
                            PathSegmentList { segments: segs },
                            Some(r.rename.to_string()),
                        ));
                    }
                    _ => {}
                }
            }
            kinds
        }
    }
}

enum TreeTerminal {
    Glob,
    Name(String),
    Rename(String, String),
    Group(Vec<UseTree>),
}

fn collect_path_segments(tree: &UseTree, segments: &mut Vec<PathSegment>) -> TreeTerminal {
    match tree {
        UseTree::Path(p) => {
            segments.push(ident_to_segment(&p.ident));
            collect_path_segments(&p.tree, segments)
        }
        UseTree::Glob(_) => TreeTerminal::Glob,
        UseTree::Name(n) => TreeTerminal::Name(n.ident.to_string()),
        UseTree::Rename(r) => TreeTerminal::Rename(r.ident.to_string(), r.rename.to_string()),
        UseTree::Group(g) => TreeTerminal::Group(g.items.iter().cloned().collect()),
    }
}

fn ident_to_segment(ident: &syn::Ident) -> PathSegment {
    match ident.to_string().as_str() {
        "super" => PathSegment::Super,
        "crate" => PathSegment::Crate,
        "self" => PathSegment::Self_,
        other => PathSegment::Named(other.to_string()),
    }
}

// ── Text-based cfg_select! scanning ────────────────────────────────────────

/// Scan raw source text for `mod X;` declarations, handling `cfg_select!`
/// blocks by evaluating cfg predicates against the build target.
fn scan_mod_declarations_with_cfg(source: &str, cfg: &CfgContext) -> Vec<ModDeclaration> {
    // Check if this file uses cfg_select!.
    if source.contains("cfg_select!") {
        if let Some(body) = extract_cfg_select_body(source) {
            return scan_cfg_select_branches(&body, cfg);
        }
    }

    // No cfg_select — fall back to simple line-by-line scan.
    scan_mod_declarations_simple(source)
}

/// Extract the body of the first `cfg_select! { ... }` invocation.
fn extract_cfg_select_body(source: &str) -> Option<String> {
    let idx = source.find("cfg_select!")?;
    let after = &source[idx + 11..]; // skip "cfg_select!"

    // Find the opening brace.
    let trimmed = after.trim_start();
    let brace_idx = trimmed.find('{')?;
    let after_brace = &trimmed[brace_idx + 1..];

    // Match braces to find the end.
    let mut depth = 1;
    let mut end = 0;
    for (i, ch) in after_brace.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }

    Some(after_brace[..end].to_string())
}

/// Parse cfg_select! branches and extract mod declarations from the matching one.
/// Each branch has the form: `predicate => { ... }` or `_ => { ... }` (fallback).
fn scan_cfg_select_branches(body: &str, cfg: &CfgContext) -> Vec<ModDeclaration> {
    let mut best_match: Option<Vec<ModDeclaration>> = None;
    let mut fallback: Option<Vec<ModDeclaration>> = None;

    // Split branches by `=>` at the top level (depth 0).
    let branches = split_cfg_select_branches(body);

    for (predicate, branch_body) in branches {
        let predicate = predicate.trim();
        let mods = scan_mod_declarations_simple(branch_body.trim());

        if predicate == "_" || predicate == ".." {
            // Fallback branch.
            fallback.get_or_insert(mods);
        } else if cfg.eval_predicate(predicate) {
            best_match = Some(mods);
            break; // First match wins, same as cfg_select!.
        }
    }

    best_match.or(fallback).unwrap_or_default()
}

/// For a file whose top-level structure is a `cfg_select!`, return the set of
/// re-export source modules named in the *active* branch's `pub use <mod>::…;`
/// statements. This lets callers follow a canonical type through a
/// cfg-gated re-export shim down to its defining submodule (e.g.
/// `sys::sync::mutex::Mutex` → `futex::Mutex` on Linux).
///
/// Returns an empty vec when the file has no `cfg_select!` or the active
/// branch carries no single-module re-exports.
pub fn cfg_select_reexport_targets(source: &str, cfg: &CfgContext) -> Vec<String> {
    if !source.contains("cfg_select!") {
        return Vec::new();
    }
    let Some(body) = extract_cfg_select_body(source) else {
        return Vec::new();
    };
    let branches = split_cfg_select_branches(&body);
    for (predicate, branch_body) in branches {
        let predicate = predicate.trim();
        let active = if predicate == "_" || predicate == ".." {
            true // fallback handled below
        } else {
            cfg.eval_predicate(predicate)
        };
        if !active {
            continue;
        }
        let targets = scan_reexport_sources(branch_body.trim());
        if !targets.is_empty() {
            return targets;
        }
        // Remember fallback so we can return it if no concrete match fired.
        if predicate == "_" || predicate == ".." {
            return targets;
        }
    }
    Vec::new()
}

/// For a file whose platform selection uses `cfg_if!` (the older macro with
/// `if #[cfg(...)] { ... } else if ...` syntax), return the set of re-export
/// source modules named in the *active* branch's `pub use <mod>::…;`
/// statements. This is the counterpart to [`cfg_select_reexport_targets`] for
/// files like `sys/pal/mod.rs` that select the platform backend via `cfg_if!`.
///
/// Returns an empty vec when the file has no `cfg_if!` or the active branch
/// carries no single-module re-exports.
pub fn cfg_if_reexport_targets(source: &str, cfg: &CfgContext) -> Vec<String> {
    if !source.contains("cfg_if") {
        return Vec::new();
    }
    // Find each `#[cfg(predicate)]` followed by `{ ... }` inside a cfg_if block.
    // The structure is: `if #[cfg(pred)] { body } else if #[cfg(pred2)] { body2 } ...`
    // We scan for `#[cfg(` occurrences within the cfg_if region and pair each with
    // its following brace-delimited body. Predicates may span multiple lines
    // (e.g., `any(\n all(...),\n target_os = "linux",\n)`), so the inner text is
    // collected verbatim and evaluated as-is — [`CfgContext::eval_predicate`]
    // splits on commas at any parenthesis depth, which tolerates embedded
    // newlines.
    let mut branches: Vec<(String, String)> = Vec::new();
    let start = source.find("cfg_if").unwrap_or(0);
    let region = &source[start..];
    let chars: Vec<char> = region.chars().collect();
    let len = chars.len();
    let mut i = 0;
    while i < len {
        // Look for `#[cfg(` pattern. The literal `#[cfg(` is six characters
        // (`#`, `[`, `c`, `f`, `g`, `(`), so the opening paren sits at
        // `i + 5` and the predicate text begins at `i + 6`.
        if region[i..].starts_with("#[cfg(") {
            let pred_begin = i + 6;
            // Find the closing ')' balanced against the one at `i + 5`.
            let mut depth = 1;
            let mut j = pred_begin;
            while j < len && depth > 0 {
                match chars[j] {
                    '(' => depth += 1,
                    ')' => depth -= 1,
                    _ => {}
                }
                j += 1;
            }
            if j > len || depth != 0 {
                break; // Unbalanced; bail out rather than panic on slicing.
            }
            // Predicate spans `pred_begin..j-1`; `j-1` is the closing ')'.
            let predicate: String = chars[pred_begin..j - 1].iter().collect();
            // After ')', expect `]` then optionally whitespace (including
            // newlines) then `{`.
            let mut k = j; // position after ')'
            while k < len && (chars[k] == ']' || chars[k].is_whitespace()) {
                k += 1;
            }
            if k < len && chars[k] == '{' {
                k += 1; // skip '{'
                let body_start = k;
                let mut bdepth = 1;
                while k < len && bdepth > 0 {
                    match chars[k] {
                        '{' => bdepth += 1,
                        '}' => bdepth -= 1,
                        _ => {}
                    }
                    k += 1;
                }
                let body: String = chars[body_start..k.saturating_sub(1)].iter().collect();
                branches.push((predicate, body));
                i = k;
                continue;
            }
        }
        i += 1;
    }

    // Evaluate predicates in order; first active branch wins (same semantics
    // as cfg_if!). An empty predicate list means "else" fallback.
    let mut fallback: Option<Vec<String>> = None;
    for (predicate, body) in &branches {
        let active = if predicate.is_empty() {
            true
        } else {
            cfg.eval_predicate(predicate)
        };
        if !active {
            continue;
        }
        let targets = scan_reexport_sources(body.trim());
        if !targets.is_empty() {
            return targets;
        }
        if predicate.is_empty() {
            fallback = Some(targets);
        }
    }
    fallback.unwrap_or_default()
}

/// Scan a cfg_select!/cfg_if! branch body for `pub use <path>…;` statements
/// and collect the leading module name each re-export comes from. Handles
/// `self::`, `super::`, and `crate::` prefixes by skipping them to find the
/// actual module name. Works for both glob re-exports
/// (`pub use self::unix::*;` → `"unix"`, `pub use pal::*;` → `"pal"`) and
/// specific-item re-exports used by `cfg_if!` platform shims
/// (`pub use futex::Mutex;` → `"futex"`). Callers must verify the returned
/// candidate modules actually exist before following them.
fn scan_reexport_sources(branch_body: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in branch_body.lines() {
        let trimmed = line.trim();
        // Match `pub use <path>;`
        let rest = match trimmed.strip_prefix("pub") {
            Some(r) => r.trim_start(),
            None => continue,
        };
        let rest = match rest.strip_prefix("use") {
            Some(r) => r.trim_start(),
            None => continue,
        };
        let rest = rest.trim_end_matches(';');
        if rest.is_empty() {
            continue;
        }
        // Split into path segments on `::`.
        let segs: Vec<&str> = rest
            .split("::")
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        // Skip leading keywords (self, super, crate) and glob wildcards to
        // find the actual module name. For `pub use self::unix::*;` the
        // segments are ["self", "unix", "*"] — we must skip "self" to get
        // "unix".
        let is_keyword = |s: &str| matches!(s, "self" | "super" | "crate");
        let is_ident = |s: &str| s.chars().all(|c| c.is_alphanumeric() || c == '_');
        let named: Vec<&str> = segs
            .iter()
            .copied()
            .filter(|s| *s != "*" && !is_keyword(s) && is_ident(s))
            .collect();
        // The source module is the first non-keyword identifier segment.
        let name = named.first().copied();
        if let Some(name) = name {
            if !out.iter().any(|s| s.as_str() == name) {
                out.push(name.to_string());
            }
        }
    }
    out
}

/// Split cfg_select! body into (predicate, body) pairs.
/// Branches are separated by `=>` at the top level.
fn split_cfg_select_branches(body: &str) -> Vec<(String, String)> {
    let mut branches = Vec::new();
    let mut i = 0;
    let chars: Vec<char> = body.chars().collect();
    let len = chars.len();

    while i < len {
        // Skip whitespace and comments.
        while i < len && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= len {
            break;
        }

        // Collect predicate until `=>`.
        let pred_start = i;
        let mut arrow_pos: Option<usize> = None;
        while i + 1 < len {
            if chars[i] == '=' && chars[i + 1] == '>' {
                arrow_pos = Some(i);
                break;
            }
            i += 1;
        }
        let Some(arrow_pos) = arrow_pos else {
            break;
        };

        let predicate: String = chars[pred_start..arrow_pos].iter().collect();
        i = arrow_pos + 2; // skip past `=>`

        // Skip whitespace, then expect `{`.
        while i < len && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= len || chars[i] != '{' {
            break;
        }
        i += 1; // skip `{`

        // Collect body until matching `}`.
        let body_start = i;
        let mut depth = 1;
        while i < len && depth > 0 {
            match chars[i] {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
            i += 1;
        }

        let branch_body: String = chars[body_start..i].iter().collect();
        branches.push((predicate, branch_body));
    }

    branches
}

/// Simple line-by-line scan for `mod X;` / `pub mod X;` declarations.
/// Captures `#[cfg(...)]` attributes from lines immediately preceding each
/// mod declaration so that inactive modules (e.g., `#[cfg(test)]`) can be
/// filtered by consumers like [`is_cfg_inactive`].
fn scan_mod_declarations_simple(source: &str) -> Vec<ModDeclaration> {
    let lines: Vec<&str> = source.lines().collect();
    let mut results = Vec::new();

    for i in 0..lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.ends_with('{') || trimmed.contains('=') {
            continue;
        }

        let words: Vec<&str> = trimmed.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }

        for (idx, &word) in words.iter().enumerate() {
            if word == "mod" && idx + 1 < words.len() {
                let name = words[idx + 1];
                let cleaned = name.trim_end_matches(';').trim_end_matches('{');
                if !cleaned.is_empty() && cleaned.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    // Collect #[cfg(...)] attributes from preceding lines.
                    let attrs = collect_preceding_cfg_attrs(&lines, i);
                    results.push(ModDeclaration {
                        name: cleaned.to_string(),
                        attrs,
                    });
                }
                break;
            }
        }
    }
    results
}

/// Walk backwards from line `index` collecting `#[cfg(...)]` attribute lines.
/// Stops at the first line that is neither a cfg attribute nor blank/comment-only.
fn collect_preceding_cfg_attrs(lines: &[&str], index: usize) -> Vec<Attribute> {
    let mut attrs = Vec::new();
    let mut j = index;

    while j > 0 {
        j -= 1;
        let prev = lines[j].trim();

        // Accept #[cfg(...)] or #[cfg_attr(...)] on the preceding line.
        if prev.starts_with("#[cfg") {
            // Parse the attribute from text by wrapping in a dummy item so that
            // syn can tokenize it as an attribute.
            let dummy = format!("{} fn _f() {}", prev, "{}");
            if let Ok(item) = syn::parse_str::<syn::ItemFn>(&dummy) {
                attrs.extend(item.attrs);
            }
        } else if prev.is_empty() || prev.starts_with("//") {
            // Blank line or comment — stop scanning backwards.
            break;
        } else {
            // Some other code — stop.
            break;
        }
    }

    attrs.reverse();
    attrs
}

/// Result of text-based source scanning (fallback when syn can't parse).
struct TextScanResult {
    items: Vec<ParsedItem>,
    use_statements: Vec<UseStatement>,
    mod_declarations: Vec<ModDeclaration>,
    inline_modules: Vec<(String, Vec<ParsedItem>)>,
}

/// True when a trimmed line begins a macro definition (`macro name` or
/// `pub macro name`, optionally followed by `(` or `{`). Macro invocations
/// (`name!`) are not matched because they lack the leading `macro` keyword.
fn is_macro_definition(trimmed: &str) -> bool {
    let rest = trimmed.strip_prefix("pub ").unwrap_or(trimmed);
    match rest.strip_prefix("macro ") {
        Some(after) => {
            // Expect an identifier (the macro name) next.
            after
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        }
        None => false,
    }
}

/// Given the index of a line that starts a macro definition, return the index
/// of the first line *after* the macro's closing brace (or EOF). Uses
/// brace-depth counting over the raw lines; string/comment stripping has
/// already been applied upstream so braces inside strings are not counted.
fn skip_macro_body(lines: &[&str], start: usize) -> usize {
    let mut depth = 0usize;
    let mut j = start;
    while j < lines.len() {
        for ch in lines[j].chars() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth = depth.saturating_sub(1);
                }
                _ => {}
            }
        }
        j += 1;
        if depth == 0 && j > start + 1 {
            break;
        }
    }
    j
}

/// Strip a visibility prefix (`pub`, `pub(super)`, `pub(crate)`, `pub(in path)`)
/// from a trimmed line, returning the remainder starting at the item keyword.
fn strip_visibility_prefix(trimmed: &str) -> &str {
    if let Some(rest) = trimmed.strip_prefix("pub") {
        // Check for restricted visibility: pub(...)
        if rest.starts_with('(') {
            // Find matching closing paren
            if let Some(close) = rest.find(')') {
                return rest[close + 1..].trim_start();
            }
        } else if let Some(rest) = rest.strip_prefix(' ') {
            return rest;
        }
    }
    trimmed
}

/// Result of collecting an item's full text from the line buffer.
enum CollectedItem {
    /// Successfully collected; `0` is the joined text, `1` is the next line index.
    Done(String, usize),
    /// Item was unterminated (ran off end of file); skip to the given index.
    Skipped(usize),
}

/// Collect the full text of a top-level item starting at line `start`,
/// including preceding attribute lines. Returns the joined text and the
/// index of the next line to process, or `Skipped` if the item ran off
/// the end of the file without a terminator.
fn collect_item_text(lines: &[&str], start: usize, kind: ItemKind) -> CollectedItem {
    // Include preceding attributes
    let mut attr_start = start;
    while attr_start > 0 {
        let prev = lines[attr_start - 1].trim();
        if prev.starts_with('#') {
            attr_start -= 1;
        } else {
            break;
        }
    }

    if kind == ItemKind::Const || kind == ItemKind::TypeAlias {
        let mut item_lines: Vec<&str> = lines[attr_start..=start.min(lines.len() - 1)].to_vec();
        let mut j = start + 1;
        let mut found_semi = lines[start].contains(';');
        while j < lines.len() && !found_semi {
            item_lines.push(lines[j]);
            if lines[j].contains(';') {
                found_semi = true;
            }
            j += 1;
        }
        CollectedItem::Done(item_lines.join("\n"), j)
    } else {
        let mut item_lines: Vec<&str> = Vec::new();
        let mut terminated = false;
        let mut brace_depth = 0usize;
        let mut saw_brace = false;
        let mut next_i = lines.len();

        for (j, line) in lines.iter().enumerate().skip(attr_start) {
            item_lines.push(*line);
            for ch in line.chars() {
                match ch {
                    '{' => {
                        brace_depth += 1;
                        saw_brace = true;
                    }
                    '}' => {
                        if saw_brace {
                            brace_depth = brace_depth.saturating_sub(1);
                        }
                    }
                    ';' if !saw_brace => {
                        next_i = j + 1;
                        terminated = true;
                        break;
                    }
                    _ => {}
                }
            }
            if terminated {
                break;
            }
            if saw_brace && brace_depth == 0 {
                next_i = j + 1;
                terminated = true;
                break;
            }
        }
        if !terminated {
            return CollectedItem::Skipped(lines.len());
        }
        CollectedItem::Done(item_lines.join("\n"), next_i)
    }
}

/// Text-based fallback scanner for extracting struct/enum/union/type/const
/// declarations from a Rust source file when syn::parse_file fails.
///
/// This handles cases where the nightly stdlib source uses syntax ahead of
/// what syn supports. Uses brace-counting to extract complete item definitions.
fn text_scan_source(source: &str, cfg: &CfgContext) -> TextScanResult {
    let mut items = Vec::new();
    // The AST path extracts uses via `extract_all_uses_from_file`; the text
    // fallback scans them in the item loop below (indented lines included for
    // this purpose), so that files syn cannot parse (e.g., cfg_if!/cfg_select!
    // platform shims) still contribute their import bindings to qualifier
    // resolution.
    let mut use_statements = Vec::new();
    let mod_declarations = scan_mod_declarations_with_cfg(source, cfg);

    // Remove comments and string literals to avoid false matches
    let cleaned = strip_comments_and_strings(source);

    // Find top-level pub struct/enum/union/const/type declarations
    let lines: Vec<&str> = cleaned.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        // Skip attribute-only lines
        if trimmed.starts_with('#') || trimmed.starts_with('}') || trimmed.is_empty() {
            i += 1;
            continue;
        }

        // Skip macro definitions entirely. A `macro name { ... }` or
        // `pub macro name(...) { ... }` block is not a type definition and
        // cannot be compiled in a downstream crate (the `macro` keyword is
        // unstable). Consume through its closing brace so its body — which may
        // contain stray identifiers — is never mistaken for an item.
        if is_macro_definition(trimmed) {
            i = skip_macro_body(&lines, i);
            continue;
        }

        // Check indentation - we only want truly top-level items (no leading whitespace).
        // Items inside impl blocks, functions, or modules will be indented.
        let leading_spaces = line.len() - line.trim_start_matches(' ').len();
        let leading_tabs = line.len() - line.trim_start_matches('\t').len();
        let indent = if leading_tabs > 0 {
            leading_tabs
        } else {
            leading_spaces / 4
        };

        // Indented lines are never top-level *definitions*, but they can be
        // `use` statements inside macro-invocation bodies (e.g., cfg_if!
        // branches). Extract those before skipping the line so import bindings
        // from files syn cannot parse still reach qualifier resolution.
        if indent > 0 {
            if trimmed.starts_with("use ") || trimmed.starts_with("pub use ") {
                let use_line = trimmed.strip_suffix(';').unwrap_or(trimmed);
                use_statements.extend(text_parse_use_statement(use_line));
            }
            i += 1;
            continue;
        }

        // Determine item kind. Only plain `const NAME : TYPE = EXPR;` items are
        // collected — `const fn`, `const trait`, `const impl`, etc. are skipped
        // because their bodies use braces (not a terminating `;`) which would
        // send the collector running off the end of the file and swallowing
        // every subsequent top-level item into one giant blob.
        // Strip visibility prefix to get the item keyword. Handles:
        //   pub struct, pub(super) struct, pub(crate) struct, pub(in path) struct, struct
        let after_vis = strip_visibility_prefix(trimmed);
        let kind = if after_vis.starts_with("struct ") {
            Some(ItemKind::Struct)
        } else if after_vis.starts_with("enum ") {
            Some(ItemKind::Enum)
        } else if after_vis.starts_with("union ") {
            Some(ItemKind::Union)
        } else if after_vis.starts_with("const ")
            && !after_vis.starts_with("const fn")
            && !after_vis.starts_with("const trait")
            && !after_vis.starts_with("const impl")
        {
            Some(ItemKind::Const)
        } else if after_vis.starts_with("type ") {
            Some(ItemKind::TypeAlias)
        } else {
            None
        };

        if let Some(kind) = kind {
            let collected = collect_item_text(&lines, i, kind);
            let (item_text, next_i) = match collected {
                CollectedItem::Done(text, next) => (text, next),
                CollectedItem::Skipped(next) => {
                    i = next;
                    continue;
                }
            };
            i = next_i;

            // Try to parse as tokens
            if let Ok(tokens) = item_text.parse::<TokenStream>() {
                let name = extract_item_name_from_tokens(&tokens, kind);
                let visibility = if trimmed.starts_with("pub ") {
                    ItemVisibility::Public
                } else {
                    ItemVisibility::Private
                };
                let alias_rhs = if kind == ItemKind::TypeAlias {
                    let text = item_text.to_string();
                    text.split_once('=')
                        .map(|(_, rhs)| rhs.trim().trim_end_matches(';').to_string())
                        .and_then(|rhs_text| rhs_text.parse::<TokenStream>().ok())
                } else {
                    None
                };
                items.push(ParsedItem {
                    attrs: Vec::new(),
                    full_tokens: tokens,
                    kind,
                    name,
                    visibility,
                    alias_rhs,
                });
            }
        } else if trimmed.starts_with("use ") || trimmed.starts_with("pub use ") {
            // Extract use statements via text scan. A grouped import with a
            // `self` alias yields two statements (module import + glob).
            let use_line = trimmed.strip_suffix(';').unwrap_or(trimmed);
            use_statements.extend(text_parse_use_statement(use_line));
            i += 1;
        } else {
            i += 1;
        }
    }

    TextScanResult {
        items,
        use_statements,
        mod_declarations,
        inline_modules: Vec::new(),
    }
}

/// Strip block comments and string/char literals from source text to avoid
/// false pattern matches in the text scanner.
fn strip_comments_and_strings(source: &str) -> String {
    let mut result = String::with_capacity(source.len());
    let chars_vec: Vec<char> = source.chars().collect();
    let len = chars_vec.len();
    let mut i = 0;

    while i < len {
        let ch = chars_vec[i];
        match ch {
            '/' => {
                if i + 1 < len && chars_vec[i + 1] == '/' {
                    // Line comment - consume rest of line
                    i += 2;
                    while i < len && chars_vec[i] != '\n' {
                        i += 1;
                    }
                } else if i + 1 < len && chars_vec[i + 1] == '*' {
                    // Block comment
                    i += 2;
                    while i < len {
                        if chars_vec[i] == '*' && i + 1 < len && chars_vec[i + 1] == '/' {
                            i += 2;
                            break;
                        }
                        if chars_vec[i] == '\n' {
                            result.push('\n');
                        }
                        i += 1;
                    }
                } else {
                    result.push(ch);
                    i += 1;
                }
            }
            '"' => {
                result.push('"');
                i += 1;
                while i < len {
                    if chars_vec[i] == '\\' {
                        i += 2; // skip escaped char
                    } else if chars_vec[i] == '"' {
                        result.push('"');
                        i += 1;
                        break;
                    } else {
                        result.push('x');
                        i += 1;
                    }
                }
            }
            '\'' => {
                // Distinguish char literals ('a', '\n') from lifetimes ('a, 'static).
                // A lifetime is `'` followed by an identifier character (alpha or _).
                // A char literal is `'` followed by either a backslash (escape) or
                // a single non-identifier character.
                let next = chars_vec.get(i + 1).copied();
                let is_lifetime = matches!(next, Some(c) if c.is_ascii_alphabetic() || c == '_');
                if is_lifetime {
                    // Lifetime: just emit the quote and move on.
                    result.push('\'');
                    i += 1;
                } else {
                    // Char literal: consume through the closing quote.
                    result.push('\'');
                    i += 1;
                    while i < len {
                        if chars_vec[i] == '\\' {
                            i += 2;
                        } else if chars_vec[i] == '\'' {
                            result.push('\'');
                            i += 1;
                            break;
                        } else {
                            i += 1;
                        }
                    }
                }
            }
            _ => {
                result.push(ch);
                i += 1;
            }
        }
    }

    result
}

/// Attempt to parse a use statement from text into one or more UseStatements.
/// Returns a vector because a grouped import containing `self`
/// (`use foo::bar::{self, a, b}`) expands to two logical statements: a
/// module import of `foo::bar` (from the `self` alias, which brings the module
/// name itself into scope so `bar::item` paths resolve) plus a glob of the
/// base path (approximating the named items). Returning nothing leaves the
/// caller unchanged.
fn text_parse_use_statement(text: &str) -> Vec<UseStatement> {
    let visibility = if text.starts_with("pub use") {
        Visibility::Public
    } else {
        Visibility::Private
    };

    let path_str = if visibility == Visibility::Public {
        text.strip_prefix("pub use ").unwrap_or("")
    } else {
        text.strip_prefix("use ").unwrap_or("")
    };

    // Handle glob imports
    if let Some(path) = path_str.strip_suffix("::*") {
        let segments = parse_path_segments_text(path);
        if segments.is_empty() {
            return Vec::new();
        }
        return vec![UseStatement {
            visibility,
            kind: UseKind::Glob(PathSegmentList { segments }),
        }];
    }

    // Handle grouped imports: `use foo::bar::{a, b, c}` → treat as glob of
    // `foo::bar` for resolution purposes. If the group contains `self`, also
    // emit a module import of the base path so the module name itself is
    // brought into scope (matching the AST path's handling of the `self`
    // alias, which the text scanner would otherwise drop).
    if let Some(brace_pos) = path_str.find('{') {
        let base_path = path_str[..brace_pos].trim();
        if !base_path.is_empty() {
            let segments = parse_path_segments_text(base_path);
            if !segments.is_empty() {
                let has_self = path_str
                    [brace_pos + 1..path_str.find('}').unwrap_or(path_str.len())]
                    .split(',')
                    .any(|m| m.trim() == "self");
                let mut stmts = vec![UseStatement {
                    visibility: visibility.clone(),
                    kind: UseKind::Glob(PathSegmentList {
                        segments: segments.clone(),
                    }),
                }];
                if has_self {
                    stmts.push(UseStatement {
                        visibility,
                        kind: UseKind::Single(PathSegmentList { segments }, None),
                    });
                }
                return stmts;
            }
        }
    }

    // Handle simple path imports
    let segments = parse_path_segments_text(path_str);
    if segments.is_empty() {
        Vec::new()
    } else {
        vec![UseStatement {
            visibility,
            kind: UseKind::Single(PathSegmentList { segments }, None),
        }]
    }
}

/// Parse `foo::bar::baz` into PathSegment list (text version).
fn parse_path_segments_text(path: &str) -> Vec<PathSegment> {
    path.split("::")
        .map(|seg| seg.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(|name| match name.as_str() {
            "super" => PathSegment::Super,
            "crate" => PathSegment::Crate,
            "self" => PathSegment::Self_,
            _ => PathSegment::Named(name),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_mod_names(mods: &[ModDeclaration], expected: &[&str]) {
        let names: Vec<&str> = mods.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, expected);
    }

    fn linux_cfg() -> CfgContext {
        CfgContext {
            target_os: Some("linux".to_string()),
            target_family: Some("unix".to_string()),
            target_arch: Some("x86_64".to_string()),
            target_env: Some("gnu".to_string()),
            target_vendor: Some("unknown".to_string()),
            is_unix: true,
            is_windows: false,
        }
    }

    fn windows_cfg() -> CfgContext {
        CfgContext {
            target_os: Some("windows".to_string()),
            target_family: None,
            target_arch: Some("x86_64".to_string()),
            target_env: Some("msvc".to_string()),
            target_vendor: Some("microsoft".to_string()),
            is_unix: false,
            is_windows: true,
        }
    }

    // ── cfg_select! body extraction ──────────────────────────────────────

    #[test]
    fn test_extract_cfg_select_body_finds_content() {
        let source = r#"cfg_select! {
            unix => {
                mod unix;
            }
            _ => {
                mod fallback;
            }
        }"#;
        let body = extract_cfg_select_body(source).expect("should find body");
        assert!(body.contains("mod unix"));
        assert!(body.contains("mod fallback"));
    }

    #[test]
    fn test_extract_cfg_select_body_nested_braces() {
        let source = r#"cfg_select! {
            unix => {
                mod unix;
                pub use self::unix::{foo, bar};
            }
        }"#;
        let body = extract_cfg_select_body(source).expect("should handle nested braces");
        assert!(body.contains("mod unix"));
    }

    // ── cfg_select! branch splitting ─────────────────────────────────────

    #[test]
    fn test_split_cfg_select_branches_basic() {
        let body = "
            unix => { mod unix; }
            windows => { mod windows; }
        ";
        let branches = split_cfg_select_branches(body);
        assert_eq!(branches.len(), 2);
        assert!(branches[0].0.trim().ends_with("unix"));
        assert!(branches[1].0.trim().ends_with("windows"));
    }

    #[test]
    fn test_split_cfg_select_branches_predicate_no_trailing_arrow() {
        let body = "
            unix => { mod unix; }
            target_os = \"linux\" => { mod linux; }
        ";
        let branches = split_cfg_select_branches(body);
        assert_eq!(branches.len(), 2);
        // Predicates should NOT include `=>`
        assert!(!branches[0].0.contains("=>"));
        assert!(!branches[1].0.contains("=>"));
        assert_eq!(branches[0].0.trim(), "unix");
        assert_eq!(branches[1].0.trim(), r#"target_os = "linux""#);
    }

    #[test]
    fn test_split_cfg_select_branches_with_fallback() {
        let body = "
            unix => { mod unix; }
            _ => { mod unsupported; }
        ";
        let branches = split_cfg_select_branches(body);
        assert_eq!(branches.len(), 2);
        assert_eq!(branches[1].0.trim(), "_");
    }

    // ── cfg_select! scanning end-to-end ──────────────────────────────────

    #[test]
    fn test_scan_cfg_select_picks_unix_on_linux() {
        let source = r#"cfg_select! {
            unix => {
                mod unix;
                pub use self::unix::*;
            }
            windows => {
                mod windows;
                pub use self::windows::*;
            }
            _ => {
                mod unsupported;
            }
        }"#;
        let mods = scan_mod_declarations_with_cfg(source, &linux_cfg());
        assert_mod_names(&mods, &["unix"]);
    }

    #[test]
    fn test_scan_cfg_select_picks_windows_on_win() {
        let source = r#"cfg_select! {
            unix => {
                mod unix;
            }
            windows => {
                mod windows;
            }
        }"#;
        let mods = scan_mod_declarations_with_cfg(source, &windows_cfg());
        assert_mod_names(&mods, &["windows"]);
    }

    #[test]
    fn test_scan_cfg_select_target_os_predicate() {
        let source = r#"cfg_select! {
            target_os = "linux" => {
                mod linux;
            }
            target_os = "macos" => {
                mod macos;
            }
            _ => {
                mod other;
            }
        }"#;
        let mods = scan_mod_declarations_with_cfg(
            source,
            &CfgContext {
                target_os: Some("linux".to_string()),
                ..Default::default()
            },
        );
        assert_mod_names(&mods, &["linux"]);
    }

    #[test]
    fn test_scan_cfg_select_all_predicate() {
        let source = r#"cfg_select! {
            all(target_os = "linux", target_env = "gnu") => {
                mod gnu_linux;
            }
            _ => {
                mod other;
            }
        }"#;
        let mods = scan_mod_declarations_with_cfg(source, &linux_cfg());
        assert_mod_names(&mods, &["gnu_linux"]);
    }

    #[test]
    fn test_scan_cfg_select_any_predicate() {
        let source = r#"cfg_select! {
            any(target_os = "linux", target_os = "freebsd") => {
                mod bsd_like;
            }
            _ => {
                mod other;
            }
        }"#;
        let mods = scan_mod_declarations_with_cfg(source, &linux_cfg());
        assert_mod_names(&mods, &["bsd_like"]);
    }

    #[test]
    fn test_scan_cfg_select_not_predicate() {
        let source = r#"cfg_select! {
            not(windows) => {
                mod not_win;
            }
            _ => {
                mod fallback;
            }
        }"#;
        let mods = scan_mod_declarations_with_cfg(source, &linux_cfg());
        assert_mod_names(&mods, &["not_win"]);
    }

    #[test]
    fn test_scan_cfg_select_fallback_used() {
        let source = r#"cfg_select! {
            target_os = "solid_asp3" => {
                mod solid;
            }
            _ => {
                mod hermit;
            }
        }"#;
        let mods = scan_mod_declarations_with_cfg(source, &linux_cfg());
        assert_mod_names(&mods, &["hermit"]);
    }

    // ── Recursive use statement extraction ───────────────────────────────

    #[test]
    fn test_extract_uses_top_level_only() {
        let source = r#"
use std::fmt::Debug;
use crate::sys::pal::mutex::Mutex;
        "#;
        let parsed = parse_source_with_cfg(source, &linux_cfg());
        assert_eq!(parsed.use_statements.len(), 2);
    }

    #[test]
    fn test_extract_uses_inside_impl_block() {
        let source = r#"
struct Foo;
impl Foo {
    fn bar() {
        use std::cell::Cell;
        let x = Cell::new(0);
    }
}
        "#;
        let parsed = parse_source_with_cfg(source, &linux_cfg());
        assert_eq!(parsed.use_statements.len(), 1);
        assert_eq!(parsed.items.len(), 1); // struct Foo
    }

    #[test]
    fn test_extract_uses_inside_inline_module() {
        let source = r#"
mod inner {
    use std::option::Option;
    pub fn foo() {}
}
        "#;
        let parsed = parse_source_with_cfg(source, &linux_cfg());
        assert_eq!(parsed.use_statements.len(), 1);
    }

    #[test]
    fn test_extract_uses_nested_blocks() {
        let source = r#"
fn outer() {
    use std::vec::Vec;
    fn inner() {
        use std::string::String;
    }
}
        "#;
        let parsed = parse_source_with_cfg(source, &linux_cfg());
        assert_eq!(parsed.use_statements.len(), 2);
    }

    // ── parse_source_with_cfg integration ────────────────────────────────

    #[test]
    fn test_parse_source_with_cfg_syn_failure_falls_back_to_text_scan() {
        // cfg_select! causes syn to fail to parse, but text scanner should work
        let source = r#"cfg_select! {
            unix => {
                mod unix;
            }
            _ => {
                mod fallback;
            }
        }"#;
        let parsed = parse_source_with_cfg(source, &linux_cfg());
        assert_mod_names(&parsed.mod_declarations, &["unix"]);
        assert!(parsed.items.is_empty());
    }

    #[test]
    fn test_parse_source_with_cfg_normal_file() {
        let source = r#"
use std::sync::atomic::AtomicUsize;

pub struct MyStruct {
    pub val: AtomicUsize,
}

mod child;
        "#;
        let parsed = parse_source_with_cfg(source, &linux_cfg());
        assert_eq!(parsed.items.len(), 1);
        assert_eq!(parsed.use_statements.len(), 1);
        assert_mod_names(&parsed.mod_declarations, &["child"]);
    }

    // ── CfgContext predicate evaluation ──────────────────────────────────

    #[test]
    fn test_eval_bare_unix_true() {
        assert!(linux_cfg().eval_predicate("unix"));
    }

    #[test]
    fn test_eval_bare_unix_false_on_windows() {
        assert!(!windows_cfg().eval_predicate("unix"));
    }

    #[test]
    fn test_eval_bare_windows_true() {
        assert!(windows_cfg().eval_predicate("windows"));
    }

    #[test]
    fn test_eval_target_os_key_value() {
        assert!(linux_cfg().eval_predicate(r#"target_os = "linux""#));
        assert!(!linux_cfg().eval_predicate(r#"target_os = "macos""#));
    }

    #[test]
    fn test_eval_unknown_predicate_returns_false() {
        assert!(!linux_cfg().eval_predicate("foobar"));
    }

    // ── Emitter preamble round-trip tests ──────────────────────────────────

    /// Verify that the preamble content itself is valid, compilable Rust by
    /// checking it parses without errors. This ensures we haven't accidentally
    /// imported something that doesn't exist or conflicts.
    #[test]
    fn test_preamble_content_is_valid_rust() {
        use crate::emitter::preamble_content;
        let parsed = syn::parse_file(&preamble_content());
        assert!(
            parsed.is_ok(),
            "Preamble content must be valid Rust: {:?}",
            parsed.err()
        );
    }

    /// Verify that emitting a struct that references preamble types compiles
    /// through the preamble import (i.e., the preamble provides the right names).
    #[test]
    fn test_emitted_struct_with_preamble_types_parses() {
        use super::super::parser::ItemKind;
        use crate::emitter::{EmitConfig, TypeRegistry, emit_parsed_items};
        use quote::quote;

        let item = ParsedItem {
            attrs: vec![syn::parse_quote!(#[repr(C)])],
            full_tokens: quote! {
                pub struct TestStruct<T> {
                    pub data: UnsafeCell<u64>,
                    pub phantom: PhantomData<T>,
                    pub pinned: PhantomPinned,
                }
            },
            kind: ItemKind::Struct,
            name: "TestStruct".to_string(),
            visibility: ItemVisibility::Public,
            alias_rhs: None,
        };

        // Registry routing: both marker types are registered as public,
        // undeclared core types, so references to them route straight at the
        // builtin crate (`__rustyfill_builtin_core`) instead of through the
        // synthetic tree or the preamble.
        let mut registry = TypeRegistry::empty();
        // The struct under test must be declared for the emitter to keep it;
        // otherwise the declaration filter drops undeclared data structures.
        registry.insert_declared("alloc::TestStruct", "");
        registry.register(
            "core::marker::PhantomData",
            ItemVisibility::Public,
            true,
            "core/marker.rs",
        );
        registry.register(
            "core::marker::PhantomPinned",
            ItemVisibility::Public,
            true,
            "core/marker.rs",
        );

        let output = emit_parsed_items(
            std::slice::from_ref(&item),
            &EmitConfig {
                lib_name: "alloc",
                file_module_depth: 0,
                extra_uses: &[],
                sibling_modules: &[],
                child_module_names: &[],
                path_replacements: &[],
                ignored_structs: &[],
                relative_file_path: "",
                type_registry: &registry,
                extra_derives: &std::collections::HashMap::new(),
            },
            "crate::__prelude",
            &[],
            "",
        );
        // Must parse as valid Rust.
        let parsed = syn::parse_file(&output);
        assert!(
            parsed.is_ok(),
            "Emitted output must be valid Rust:\n{}",
            output
        );
        // Declared types are routed to their mirrored bindings via an absolute
        // `crate::std::` path into the merged synthetic tree (all libraries
        // merge under one `std` wrapper module in the manifest).
        let mut registry_declared = TypeRegistry::empty();
        registry_declared.insert_declared("alloc::TestStruct", "");
        registry_declared.insert_declared("core::marker::PhantomData", "core/marker.rs");
        let declared_out = emit_parsed_items(
            std::slice::from_ref(&item),
            &EmitConfig {
                lib_name: "alloc",
                file_module_depth: 0,
                extra_uses: &[],
                sibling_modules: &[],
                child_module_names: &[],
                path_replacements: &[],
                ignored_structs: &[],
                relative_file_path: "",
                type_registry: &registry_declared,
                extra_derives: &std::collections::HashMap::new(),
            },
            "crate::__prelude",
            &[],
            "",
        );
        let declared_norm: String = declared_out.split_whitespace().collect::<Vec<_>>().join("");
        assert!(
            declared_norm.contains("crate::std::marker::PhantomData"),
            "Declared type should be rewritten to its mirror:\n{}",
            declared_out
        );

        // Public undeclared types route straight at the builtin crate, never
        // through the synthetic tree or the preamble (token spacing may vary,
        // so normalize whitespace before asserting).
        let normalized: String = output.split_whitespace().collect::<Vec<_>>().join("");
        assert!(
            normalized.contains("__rustyfill_builtin_core::marker::PhantomData")
                && normalized.contains("__rustyfill_builtin_core::marker::PhantomPinned")
                && !normalized.contains("crate::core::"),
            "Public undeclared types should point at the builtin crate:\n{}",
            output
        );
    }

    /// Verify that emitting a struct with generic bounds referencing marker
    /// traits works (e.g., `T: Send + Sync`). These are in the language prelude
    /// so they don't need the preamble.
    #[test]
    fn test_emitted_struct_with_trait_bounds_parses() {
        use super::super::parser::ItemKind;
        use crate::emitter::{EmitConfig, TypeRegistry, emit_parsed_items};
        use quote::quote;

        let item = ParsedItem {
            attrs: vec![],
            full_tokens: quote! {
                pub struct Wrapper<T: Send + Sync + Unpin> {
                    pub inner: *mut T,
                }
            },
            kind: ItemKind::Struct,
            name: "Wrapper".to_string(),
            visibility: ItemVisibility::Public,
            alias_rhs: None,
        };

        let output = emit_parsed_items(
            &[item],
            &EmitConfig {
                lib_name: "alloc",
                file_module_depth: 0,
                extra_uses: &[],
                sibling_modules: &[],
                child_module_names: &[],
                path_replacements: &[],
                ignored_structs: &[],
                relative_file_path: "",
                type_registry: &TypeRegistry::empty(),
                extra_derives: &std::collections::HashMap::new(),
            },
            "crate::__prelude",
            &[],
            "",
        );
        let parsed = syn::parse_file(&output);
        assert!(
            parsed.is_ok(),
            "Emitted output with trait bounds must be valid Rust:\n{}",
            output
        );
    }

    /// Verify that the preamble module name is mangled and unlikely to collide.
    #[test]
    fn test_preamble_module_name_is_mangled() {
        // The constant is used internally; verify via emitted content.
        let content = crate::emitter::preamble_content();
        assert!(
            content.contains("rustyfill"),
            "Preamble should identify itself"
        );
    }

    // ── Glob-from-type use stripping tests ───────────────────────────────

    #[test]
    fn test_strip_glob_from_type_single_ident() {
        let src = "use Entry::*;\nstruct Foo {}\n";
        let out = strip_glob_from_type_uses(src);
        assert!(!out.contains("use Entry::*;"));
        assert!(out.contains("struct Foo {}"));
    }

    #[test]
    fn test_strip_preserves_module_path_globs() {
        let src = "use crate::collections::btree::*;\nstruct Bar {}\n";
        let out = strip_glob_from_type_uses(src);
        assert!(out.contains("use crate::collections::btree::*;"));
    }

    #[test]
    fn test_strip_preserves_regular_uses() {
        let src = "use std::cell::Cell;\nuse super::NodeRef;\nstruct Baz {}\n";
        let out = strip_glob_from_type_uses(src);
        assert!(out.contains("use std::cell::Cell;"));
        assert!(out.contains("use super::NodeRef;"));
    }

    #[test]
    fn test_strip_multiple_glob_from_type() {
        let src = "use Entry::*;\nuse NodeRef::*;\nfn foo() {}\n";
        let out = strip_glob_from_type_uses(src);
        assert!(!out.contains("use Entry::*;"));
        assert!(!out.contains("use NodeRef::*;"));
        assert!(out.contains("fn foo() {}"));
    }

    #[test]
    fn test_strip_indented_glob_from_type() {
        let src = "mod inner {\n    use Entry::*;\n    struct X {}\n}\n";
        let out = strip_glob_from_type_uses(src);
        assert!(!out.contains("use Entry::*;"));
        assert!(out.contains("struct X {}"));
    }

    #[test]
    fn test_strip_preserves_lowercase_module_globs() {
        let src = "use super::*;\nuse self::inner::*;\nstruct Y {}\n";
        let out = strip_glob_from_type_uses(src);
        assert!(out.contains("use super::*;"));
        assert!(out.contains("use self::inner::*;"));
    }

    // ── Target triple parsing ───────────────────────────────────────────────

    #[test]
    fn test_from_triple_windows_msvc_vendor() {
        // Modern triple: arch-os-vendor-env. Vendor slot is `pc`, so
        // `target_vendor = "win7"` is FALSE — the generic Windows branch fires.
        let cfg = CfgContext::from_target_triple("x86_64-pc-windows-msvc");
        assert_eq!(cfg.target_os.as_deref(), Some("windows"));
        assert_eq!(cfg.target_vendor.as_deref(), Some("pc"));
        assert_eq!(cfg.target_env.as_deref(), Some("msvc"));
        assert!(cfg.is_windows);
        assert!(!cfg.is_unix);
        // The futex/SRWLock branch in std's sys/sync/mutex/mod.rs activates:
        //   all(target_os = "windows", not(target_vendor = "win7"))
        assert!(cfg.eval_predicate("all(target_os = \"windows\", not(target_vendor = \"win7\"))"));
        // The win7-specific branch does NOT.
        assert!(!cfg.eval_predicate("all(target_os = \"windows\", target_vendor = \"win7\")"));
    }

    #[test]
    fn test_from_triple_windows_win7() {
        // The real rustc triple for Win7 is `i686-win7-windows-msvc`
        // (arch-vendor-os-env with vendor=`win7`). This is what makes
        // `target_vendor = "win7"` evaluate to true.
        let cfg = CfgContext::from_target_triple("i686-win7-windows-msvc");
        assert_eq!(cfg.target_os.as_deref(), Some("windows"));
        assert_eq!(cfg.target_vendor.as_deref(), Some("win7"));
        assert_eq!(cfg.target_env.as_deref(), Some("msvc"));
        assert!(cfg.is_windows);
        // Generic windows branch suppressed.
        assert!(!cfg.eval_predicate("all(target_os = \"windows\", not(target_vendor = \"win7\"))"));
        // Dedicated win7 branch active.
        assert!(cfg.eval_predicate("all(target_os = \"windows\", target_vendor = \"win7\")"));
    }

    #[test]
    fn test_from_triple_linux_gnu() {
        let cfg = CfgContext::from_target_triple("x86_64-unknown-linux-gnu");
        assert_eq!(cfg.target_os.as_deref(), Some("linux"));
        assert_eq!(cfg.target_family.as_deref(), Some("unix"));
        assert_eq!(cfg.target_vendor.as_deref(), Some("unknown"));
        assert_eq!(cfg.target_env.as_deref(), Some("gnu"));
        assert!(cfg.is_unix);
        assert!(!cfg.is_windows);
    }

    #[test]
    fn test_from_triple_macos_apple() {
        let cfg = CfgContext::from_target_triple("aarch64-apple-darwin");
        assert_eq!(cfg.target_os.as_deref(), Some("macos"));
        assert_eq!(cfg.target_family.as_deref(), Some("unix"));
        assert_eq!(cfg.target_vendor.as_deref(), Some("apple"));
        assert!(cfg.is_unix);
    }

    #[test]
    fn test_from_triple_pal_unix_sync_gate() {
        // The inner gate on sys/pal/unix/sync/mod.rs excludes futex-based
        // unix targets (linux, android, freebsd, etc.) but stays active on
        // pthread unix platforms (macOS) AND on Windows (where the module is
        // simply unused because the outer `pal/unix` path isn't selected).
        let pred = concat!(
            "not(any(",
            "target_os = \"linux\", ",
            "target_os = \"android\", ",
            "all(target_os = \"emscripten\", target_feature = \"atomics\"), ",
            "target_os = \"freebsd\", ",
            "target_os = \"openbsd\", ",
            "target_os = \"dragonfly\", ",
            "target_os = \"fuchsia\"",
            "))"
        );
        // Active on macOS (pthread unix).
        assert!(CfgContext::from_target_triple("aarch64-apple-darwin").eval_predicate(pred));
        // Inactive on Linux (futex).
        assert!(!CfgContext::from_target_triple("x86_64-unknown-linux-gnu").eval_predicate(pred));
        // Technically active on Windows too (target_os="windows" is not in the
        // exclusion list), but irrelevant since the outer module tree doesn't
        // include pal/unix on Windows.
        assert!(CfgContext::from_target_triple("x86_64-pc-windows-msvc").eval_predicate(pred));
    }

    // ── Inner #![cfg(...)] module exclusion detection ─────────────────────

    #[test]
    fn test_module_file_cfg_excluded_inactive() {
        let source = "#![cfg(not(any(\n    target_os = \"linux\",\n    target_os = \"android\",\n)))]\nmod mutex;\n";
        let linux = CfgContext::from_target_triple("x86_64-unknown-linux-gnu");
        assert_eq!(module_file_cfg_excluded(source, &linux), Some(false));
        let macos = CfgContext::from_target_triple("aarch64-apple-darwin");
        assert_eq!(module_file_cfg_excluded(source, &macos), Some(true));
    }

    #[test]
    fn test_module_file_cfg_excluded_no_attr() {
        let source = "mod mutex;\npub use mutex::Mutex;\n";
        let linux = CfgContext::from_target_triple("x86_64-unknown-linux-gnu");
        assert_eq!(module_file_cfg_excluded(source, &linux), None);
    }

    // ── cfg_if! multi-line predicate parsing ───────────────────────────────

    #[test]
    fn test_cfg_if_multiline_any_predicate_activates_first_branch() {
        // Mirrors the shape of std's sys/sync/mutex/mod.rs: a cfg_if! whose
        // first branch predicate spans multiple lines inside any(...). The
        // active branch must be detected and its re-export target returned.
        let source = r#"cfg_if::cfg_if! {
    if #[cfg(any(
        all(target_os = "windows", not(target_vendor = "win7")),
        target_os = "linux",
        target_os = "android",
    ))] {
        mod futex;
        pub use futex::Mutex;
    } else if #[cfg(target_os = "fuchsia")] {
        mod fuchsia;
        pub use fuchsia::Mutex;
    } else {
        mod no_threads;
        pub use no_threads::Mutex;
    }
}"#;
        let linux = CfgContext::from_target_triple("x86_64-unknown-linux-gnu");
        assert_eq!(cfg_if_reexport_targets(source, &linux), vec!["futex"]);
    }

    #[test]
    fn test_cfg_if_multiline_predicate_inactive_falls_through() {
        let source = r#"cfg_if::cfg_if! {
    if #[cfg(any(
        target_os = "freebsd",
        target_os = "openbsd",
    ))] {
        mod bsd;
        pub use bsd::Thing;
    } else if #[cfg(target_os = "linux")] {
        mod futex;
        pub use futex::Thing;
    } else {
        mod none;
        pub use none::Thing;
    }
}"#;
        let linux = CfgContext::from_target_triple("x86_64-unknown-linux-gnu");
        assert_eq!(cfg_if_reexport_targets(source, &linux), vec!["futex"]);
    }

    // ── Text-fallback use-statement extraction ─────────────────────────────

    #[test]
    fn test_parse_source_text_fallback_extracts_indented_uses() {
        // A cfg_if! shim defeats syn's file parser (the macro call is not a
        // valid item), forcing the text-scan fallback. Indented `pub use`
        // lines inside the macro body must still be extracted so qualifier
        // resolution can follow the import binding to its defining module.
        let source = r#"cfg_if::cfg_if! {
    if #[cfg(target_os = "linux")] {
        mod futex;
        pub use futex::Mutex;
    } else {
        mod pthread;
        pub use pthread::Mutex;
    }
}"#;
        let linux = CfgContext::from_target_triple("x86_64-unknown-linux-gnu");
        let parsed = parse_source_with_cfg(source, &linux);
        let bases: Vec<Vec<String>> = parsed
            .use_statements
            .iter()
            .map(|s| match &s.kind {
                UseKind::Single(pl, _) | UseKind::Glob(pl) => pl
                    .segments
                    .iter()
                    .map(|seg| match seg {
                        PathSegment::Named(n) => n.clone(),
                        PathSegment::Crate => "crate".to_string(),
                        PathSegment::Super => "super".to_string(),
                        PathSegment::Self_ => "self".to_string(),
                    })
                    .collect(),
            })
            .collect();
        // Both branches' indented uses are extracted (branch selection for
        // *modules* happens via cfg_if_reexport_targets; use bindings from all
        // branches feed the resolver, which discards unresolvable ones).
        assert!(bases.iter().any(|b| b == &vec!["futex", "Mutex"]));
        assert!(bases.iter().any(|b| b == &vec!["pthread", "Mutex"]));
    }
}
