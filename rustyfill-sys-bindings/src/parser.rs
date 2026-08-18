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
                    if depth > 0 {
                        depth -= 1;
                    }
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
                return &rest[close + 1..].trim_start();
            }
        } else if rest.starts_with(' ') {
            return &rest[1..];
        }
    }
    trimmed
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

        // Only consider top-level items (indent == 0)
        if indent > 0 {
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
                // For struct/enum/union there are two shapes:
                //   - Unit items ending in `;` (e.g. `pub struct Foo<T>;`)
                //   - Braced items ending at the matching `}` (field/tuple structs, enums, unions)
                // Detect which by scanning forward for the first terminating `;` or `{`.
                // A naive "collect until balanced brace" approach would swallow every
                // subsequent top-level item when the declaration is a unit struct,
                // because no opening brace ever appears on the declaration itself.
                let mut item_lines = Vec::new();
                let mut terminated = false;
                let mut brace_depth = 0usize;
                let mut saw_brace = false;

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
                                // Unit struct/enum/union: declaration ends here.
                                i = j + 1;
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
                        i = j + 1;
                        terminated = true;
                        break;
                    }
                }
                if !terminated {
                    // Ran off the end of the file without finding a terminator.
                    i = lines.len();
                    continue;
                }
                item_text = item_lines.join("\n");
            }

            // Try to parse as tokens
            if let Ok(tokens) = item_text.parse::<TokenStream>() {
                let name = extract_item_name_from_tokens(&tokens, kind);
                let visibility = if trimmed.starts_with("pub ") {
                    ItemVisibility::Public
                } else {
                    ItemVisibility::Private
                };
                // For text-scanned type aliases, extract the RHS (everything
                // after `=` up to the terminating `;`) so declared-alias
                // mirroring works even when syn can't parse the whole file.
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
                let has_self = path_str[brace_pos + 1..path_str
                    .find('}')
                    .unwrap_or(path_str.len())]
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
            &[item.clone()],
            &EmitConfig {
                lib_name: "alloc",
                file_module_depth: 0,
                extra_uses: &[],
                sibling_modules: &[],
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
            &[item.clone()],
            &EmitConfig {
                lib_name: "alloc",
                file_module_depth: 0,
                extra_uses: &[],
                sibling_modules: &[],
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
        let declared_norm: String = declared_out
            .split_whitespace()
            .collect::<Vec<_>>()
            .join("");
        assert!(
            declared_norm.contains("crate::std::marker::PhantomData"),
            "Declared type should be rewritten to its mirror:\n{}",
            declared_out
        );

        // Public undeclared types route straight at the builtin crate, never
        // through the synthetic tree or the preamble (token spacing may vary,
        // so normalize whitespace before asserting).
        let normalized: String = output
            .split_whitespace()
            .collect::<Vec<_>>()
            .join("");
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
}
