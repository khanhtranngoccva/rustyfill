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

use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::{Attribute, Item, UseTree};

use crate::resolver::{PathSegment, PathSegmentList, UseKind, UseStatement, Visibility};

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKind {
    Struct,
    Enum,
    Union,
    Const,
    TypeAlias,
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
        let is_unix = std::env::var("CARGO_CFG_UNIX").is_ok();
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

    /// Evaluate a simple cfg predicate string against this context.
    /// Supports: bare names (`unix`), key-value pairs (`target_os = "linux"`),
    /// `all(...)`, `any(...)`, `not(...)`.
    pub fn eval_predicate(&self, pred: &str) -> bool {
        let pred = pred.trim();
        match () {
            _ if pred.starts_with("all(") && pred.ends_with(')') => {
                let inner = &pred[4..pred.len() - 1];
                split_commas(inner).iter().all(|p| self.eval_predicate(p))
            }
            _ if pred.starts_with("any(") && pred.ends_with(')') => {
                let inner = &pred[4..pred.len() - 1];
                split_commas(inner).iter().any(|p| self.eval_predicate(p))
            }
            _ if pred.starts_with("not(") && pred.ends_with(')') => {
                let inner = &pred[4..pred.len() - 1];
                !self.eval_predicate(inner.trim())
            }
            _ => self.eval_atom(pred),
        }
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

/// Split a string by commas, respecting parentheses nesting.
fn split_commas(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0;
    let mut current = String::new();

    for ch in s.chars() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
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

    // Try to parse via syn AST. This may fail for files that contain
    // macro-heavy content like cfg_select!, or for nightly toolchains where
    // the stdlib source uses syntax ahead of what syn supports.
    match syn::parse_file(source) {
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
                        // Inline module — extract type-defining items from it
                        let (_brace, mod_items) = im.content.as_ref().unwrap();
                        let mut inner_items = Vec::new();
                        for inner in mod_items {
                            match inner {
                                Item::Struct(s) => inner_items.push(parse_struct(s.clone())),
                                Item::Enum(e) => inner_items.push(parse_enum(e.clone())),
                                Item::Union(u) => inner_items.push(parse_union(u.clone())),
                                _ => {}
                            }
                        }
                        if !inner_items.is_empty() {
                            inline_modules.push((im.ident.to_string(), inner_items));
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
                if let syn::TraitItem::Fn(tf) = inner
                    && let Some(block) = &tf.default
                {
                    extract_all_uses_from_block(block, out);
                }
            }
        }
        Item::Fn(iff) => {
            extract_all_uses_from_block(&iff.block, out);
        }
        Item::ExternCrate(_)
        | Item::Struct(_)
        | Item::Enum(_)
        | Item::Union(_)
        | Item::Type(_)
        | Item::Static(_)
        | Item::Const(_)
        | Item::Macro(_)
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
    }
}

fn parse_type_alias(t: syn::ItemType) -> ParsedItem {
    let mut tokens = TokenStream::new();
    t.to_tokens(&mut tokens);
    ParsedItem {
        attrs: t.attrs,
        full_tokens: tokens,
        kind: ItemKind::TypeAlias,
        name: t.ident.to_string(),
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
                    UseTree::Path(p) => {
                        let mut segs = segments.clone();
                        segs.push(ident_to_segment(&p.ident));
                        let tail = collect_path_segments(&p.tree, &mut segs);
                        match tail {
                            TreeTerminal::Glob => {
                                kinds.push(UseKind::Glob(PathSegmentList { segments: segs }))
                            }
                            TreeTerminal::Name(n) => {
                                segs.push(PathSegment::Named(n));
                                kinds.push(UseKind::Single(
                                    PathSegmentList { segments: segs },
                                    None,
                                ));
                            }
                            TreeTerminal::Rename(n, a) => {
                                segs.push(PathSegment::Named(n));
                                kinds.push(UseKind::Single(
                                    PathSegmentList { segments: segs },
                                    Some(a),
                                ));
                            }
                            TreeTerminal::Group(_) => {}
                        }
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
    if source.contains("cfg_select!")
        && let Some(body) = extract_cfg_select_body(source)
    {
        return scan_cfg_select_branches(&body, cfg);
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
        } else {
            if cfg.eval_predicate(predicate) {
                best_match = Some(mods);
                break; // First match wins, same as cfg_select!.
            }
        }
    }

    best_match.or(fallback).unwrap_or_default()
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

/// Text-based fallback scanner for extracting struct/enum/union/type/const
/// declarations from a Rust source file when syn::parse_file fails.
///
/// This handles cases where the nightly stdlib source uses syntax ahead of
/// what syn supports. Uses brace-counting to extract complete item definitions.
fn text_scan_source(source: &str, cfg: &CfgContext) -> TextScanResult {
    let mut items = Vec::new();
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

        // Check indentation - we only want truly top-level items (no leading whitespace).
        // Items inside impl blocks, functions, or modules will be indented.
        let leading_spaces = line.len() - line.trim_start_matches(' ').len();
        let leading_tabs = line.len() - line.trim_start_matches('\t').len();
        let indent = if leading_tabs > 0 {
            leading_tabs
        } else {
            leading_spaces / 4
        };

        // Only consider top-level items (indent == 0)
        if indent > 0 {
            i += 1;
            continue;
        }

        // Determine item kind
        let kind = if trimmed.starts_with("pub struct ") || trimmed.starts_with("struct ") {
            Some(ItemKind::Struct)
        } else if trimmed.starts_with("pub enum ") || trimmed.starts_with("enum ") {
            Some(ItemKind::Enum)
        } else if trimmed.starts_with("pub union ") || trimmed.starts_with("union ") {
            Some(ItemKind::Union)
        } else if trimmed.starts_with("pub const ") || trimmed.starts_with("const ") {
            Some(ItemKind::Const)
        } else if trimmed.starts_with("pub type ") || trimmed.starts_with("type ") {
            Some(ItemKind::TypeAlias)
        } else {
            None
        };

        if let Some(kind) = kind {
            // Collect the full item text including attributes above it
            let item_text;

            // Include preceding attributes
            let mut attr_start = i;
            while attr_start > 0 {
                let prev = lines[attr_start - 1].trim();
                if prev.starts_with('#') {
                    attr_start -= 1;
                } else {
                    break;
                }
            }

            // For const/type, the item is typically one line (ends with ;)
            if kind == ItemKind::Const || kind == ItemKind::TypeAlias {
                // Collect until we find a semicolon
                let mut item_lines = Vec::new();
                for line in lines.iter().take(i + 1).skip(attr_start) {
                    item_lines.push(*line);
                }
                // Continue collecting multi-line const/type defs
                let mut j = i + 1;
                let mut found_semi = lines[i].contains(';');
                while j < lines.len() && !found_semi {
                    item_lines.push(lines[j]);
                    if lines[j].contains(';') {
                        found_semi = true;
                    }
                    j += 1;
                }
                item_text = item_lines.join("\n");
                i = j;
            } else {
                // For struct/enum/union, collect until matching closing brace
                let mut brace_depth = 0;
                let mut found_open = false;
                let mut item_lines = Vec::new();

                for (j, line) in lines.iter().enumerate().skip(attr_start) {
                    item_lines.push(*line);
                    for ch in line.chars() {
                        if ch == '{' {
                            brace_depth += 1;
                            found_open = true;
                        } else if ch == '}' {
                            brace_depth -= 1;
                        }
                    }
                    if found_open && brace_depth == 0 {
                        i = j + 1;
                        break;
                    }
                }
                if !found_open {
                    i += 1;
                    continue;
                }
                item_text = item_lines.join("\n");
            }

            // Try to parse as tokens
            if let Ok(tokens) = item_text.parse::<TokenStream>() {
                let name = extract_item_name_from_tokens(&tokens, kind);
                items.push(ParsedItem {
                    attrs: Vec::new(),
                    full_tokens: tokens,
                    kind,
                    name,
                });
            }
        } else if trimmed.starts_with("use ") || trimmed.starts_with("pub use ") {
            // Extract use statements via text scan
            let use_line = trimmed.strip_suffix(';').unwrap_or(trimmed);
            if let Some(stmt) = text_parse_use_statement(use_line) {
                use_statements.push(stmt);
            }
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
            _ => {
                result.push(ch);
                i += 1;
            }
        }
    }

    result
}

/// Attempt to parse a use statement from text into a UseStatement.
/// Handles simple paths, glob imports, and grouped imports (treating grouped
/// imports as glob imports of the base path for resolution purposes).
fn text_parse_use_statement(text: &str) -> Option<UseStatement> {
    let visibility = if text.starts_with("pub use") {
        Visibility::Public
    } else {
        Visibility::Private
    };

    let path_str = if visibility == Visibility::Public {
        text.strip_prefix("pub use ")?
    } else {
        text.strip_prefix("use ")?
    };

    // Handle glob imports
    if let Some(path) = path_str.strip_suffix("::*") {
        let segments = parse_path_segments_text(path);
        let plist = PathSegmentList { segments };
        return Some(UseStatement {
            visibility,
            kind: UseKind::Glob(plist),
        });
    }

    // Handle grouped imports: `use foo::bar::{a, b, c}` → treat as glob of `foo::bar`
    // for resolution purposes. Extract the base path before `{`.
    if let Some(brace_pos) = path_str.find('{') {
        let base_path = path_str[..brace_pos].trim();
        if !base_path.is_empty() {
            let segments = parse_path_segments_text(base_path);
            if !segments.is_empty() {
                let plist = PathSegmentList { segments };
                return Some(UseStatement {
                    visibility,
                    kind: UseKind::Glob(plist),
                });
            }
        }
    }

    // Handle simple path imports
    let segments = parse_path_segments_text(path_str);
    if !segments.is_empty() {
        let plist = PathSegmentList { segments };
        Some(UseStatement {
            visibility,
            kind: UseKind::Single(plist, None),
        })
    } else {
        None
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
        use crate::emitter::{EmitConfig, emit_parsed_items};
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
        };

        let output = emit_parsed_items(
            &[item],
            &EmitConfig {
                lib_name: "alloc",
                file_module_depth: 0,
                extra_uses: &[],
                sibling_modules: &[],
                path_replacements: &[],
                ignored_structs: &[],
                relative_file_path: "",
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
    }

    /// Verify that emitting a struct with generic bounds referencing marker
    /// traits works (e.g., `T: Send + Sync`). These are in the language prelude
    /// so they don't need the preamble.
    #[test]
    fn test_emitted_struct_with_trait_bounds_parses() {
        use super::super::parser::ItemKind;
        use crate::emitter::{EmitConfig, emit_parsed_items};
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
        };

        let output = emit_parsed_items(
            &[item],
            &EmitConfig {
                lib_name: "alloc",
                file_module_depth: 0,
                extra_uses: &[],
                sibling_modules: &[],
                path_replacements: &[],
                ignored_structs: &[],
                relative_file_path: "",
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
}
