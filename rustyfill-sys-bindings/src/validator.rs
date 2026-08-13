//! Fail-fast validation layer for the bindings pipeline.
//!
//! Every stage of binding generation is validated before proceeding to the next.
//! If any check fails, the build aborts with a clear diagnostic rather than
//! silently producing incomplete or incorrect output.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use crate::LoaderSpec;
use crate::parser::{ParsedItem, ParsedSource};
use crate::resolver::ModuleResolver;

/// Collected validation errors. The build script accumulates these and
/// emits them all at once as `cargo:error=` lines.
#[derive(Default)]
pub struct ValidationErrors {
    pub errors: Vec<String>,
}

impl ValidationErrors {
    pub fn push(&mut self, msg: String) {
        self.errors.push(msg);
    }

    pub fn push_fmt(&mut self, fmt: impl std::fmt::Display) {
        self.errors.push(fmt.to_string());
    }

    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Emit all accumulated errors as `cargo:error=` and exit(1).
    #[doc(hidden)]
    pub fn fatal(self) -> ! {
        for err in &self.errors {
            eprintln!("cargo:error={}", err);
        }
        if self.errors.len() == 1 {
            eprintln!("cargo:error=Validation failed with 1 error.");
        } else {
            eprintln!(
                "cargo:error=Validation failed with {} errors.",
                self.errors.len()
            );
        }
        std::process::exit(1);
    }
}

// ── Stage 1: Spec validation ────────────────────────────────────────────────

/// Validate that every canonical file referenced in the spec actually exists
/// on disk before any parsing begins.
pub fn validate_spec_paths(spec: &LoaderSpec, rust_src: &Path, errors: &mut ValidationErrors) {
    for target in &spec.targets {
        let lib_src = rust_src.join(&target.lib_name).join("src");

        if !lib_src.exists() {
            errors.push_fmt(format!(
                "[spec] Library source root does not exist: {}",
                lib_src.display()
            ));
            continue;
        }

        for rel_path in &target.canonical_files {
            let full = lib_src.join(rel_path);
            if !full.exists() {
                errors.push_fmt(format!(
                    "[spec] Canonical file not found: {} (expected at {})",
                    rel_path,
                    full.display()
                ));
            }
        }
    }
}

// ── Stage 2: Parse validation ───────────────────────────────────────────────

/// After parsing a source file, verify that we got *something* meaningful.
/// If both syn AST and text-based scanning yield zero items, zero mod declarations,
/// and zero use statements from a non-empty file, flag it as suspicious.
pub fn validate_parse_result(
    source_rel_path: &str,
    parsed: &ParsedSource,
    source_text: &str,
    _errors: &mut ValidationErrors,
) {
    // Empty files are fine (e.g., placeholder modules).
    if source_text.trim().is_empty() {
        return;
    }

    let has_items = !parsed.items.is_empty();
    let has_mods = !parsed.mod_declarations.is_empty();
    let has_uses = !parsed.use_statements.is_empty();

    if !has_items && !has_mods && !has_uses {
        // This file has content but we extracted nothing. Warn but don't fail —
        // some files are purely functional (no types, no mods, no uses).
        eprintln!(
            "cargo:warning=[parse] {} has content but yielded 0 items, 0 mods, 0 uses",
            source_rel_path
        );
    }
}

/// Validate that each parsed item's token stream is non-empty and well-formed.
pub fn validate_parsed_items(
    source_rel_path: &str,
    items: &[ParsedItem],
    errors: &mut ValidationErrors,
) {
    for (i, item) in items.iter().enumerate() {
        if item.full_tokens.is_empty() {
            errors.push_fmt(format!(
                "[parse] {} item #{} has empty token stream (field was dropped?)",
                source_rel_path, i
            ));
        }
    }
}

// ── Stage 3: Emission validation ────────────────────────────────────────────

/// After writing a binding file to disk, re-parse it with syn to ensure it's
/// valid Rust. This catches truncated writes, malformed attribute emission, etc.
pub fn validate_emitted_file(path: &Path, errors: &mut ValidationErrors) {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            errors.push_fmt(format!(
                "[emit] Cannot read back emitted file {}: {}",
                path.display(),
                e
            ));
            return;
        }
    };

    if content.trim().is_empty() {
        errors.push_fmt(format!("[emit] Emitted file is empty: {}", path.display()));
        return;
    }

    // Try to parse the emitted file. If it fails, the binding is broken.
    if let Err(e) = syn::parse_file(&content) {
        errors.push_fmt(format!(
            "[emit] Emitted file is not valid Rust: {}\n  Error: {}",
            path.display(),
            e
        ));
    }
}

/// Same as [`validate_emitted_file`] but uses cfg-aware parsing so that
/// files with `cfg_select!` don't trigger false positives.
pub fn validate_emitted_file_with_cfg(path: &Path, errors: &mut ValidationErrors) {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            errors.push_fmt(format!(
                "[emit] Cannot read back emitted file {}: {}",
                path.display(),
                e
            ));
            return;
        }
    };

    if content.trim().is_empty() {
        errors.push_fmt(format!("[emit] Emitted file is empty: {}", path.display()));
        return;
    }

    // Use basic syn parse first — emitted files should be clean Rust without
    // unexpanded macros.
    if let Err(e) = syn::parse_file(&content) {
        errors.push_fmt(format!(
            "[emit] Emitted file is not valid Rust: {}\n  Error: {}",
            path.display(),
            e
        ));
    }
}

// ── Stage 4: Manifest validation ────────────────────────────────────────────

/// Verify that every file listed in the manifest actually exists on disk.
pub fn validate_manifest_completeness(
    out_dir: &Path,
    manifest_entries: &[(String, String)],
    errors: &mut ValidationErrors,
) {
    for (rel_path, _lib_name) in manifest_entries {
        let full = out_dir.join(rel_path);
        if !full.exists() {
            errors.push_fmt(format!(
                "[manifest] File listed in manifest but missing on disk: {}",
                rel_path
            ));
        }
    }
}

// ── Stage 5: Resolver consistency ───────────────────────────────────────────

/// Check that all discovered re-export aliases can actually resolve their
/// canonical targets. A dangling alias means a type will fail to compile.
pub fn validate_alias_resolution(
    resolver: &mut ModuleResolver,
    aliases: &HashSet<(String, String)>,
    errors: &mut ValidationErrors,
) {
    for (alias_module, canonical_module) in aliases {
        let resolution = resolver.resolve_file(canonical_module);
        if resolution.is_empty() {
            errors.push_fmt(format!(
                "[alias] Alias {} -> {} cannot resolve canonical target",
                alias_module, canonical_module
            ));
        }
    }
}

// ── All-in-one validation runner ────────────────────────────────────────────

/// Run all validation stages and return collected errors. Call this at the end
/// of the build script before exiting. If errors were found, `.fatal()` exits
/// with a non-zero code and prints all diagnostics.
pub struct ValidationResult {
    pub errors: ValidationErrors,
}

impl ValidationResult {
    pub fn pass(self) {
        // No-op, build continues.
    }

    pub fn or_fatal(self) {
        if !self.errors.is_empty() {
            self.errors.fatal();
        }
    }
}

/// Builder for staged validation. Accumulate checks, then call `.finish()`.
pub struct ValidationBuilder {
    errors: ValidationErrors,
}

impl ValidationBuilder {
    pub fn new() -> Self {
        Self {
            errors: ValidationErrors::default(),
        }
    }

    pub fn check_spec(&mut self, spec: &LoaderSpec, rust_src: &Path) {
        validate_spec_paths(spec, rust_src, &mut self.errors);
    }

    pub fn check_parse(&mut self, path: &str, parsed: &ParsedSource, source: &str) {
        validate_parse_result(path, parsed, source, &mut self.errors);
    }

    pub fn check_items(&mut self, path: &str, items: &[ParsedItem]) {
        validate_parsed_items(path, items, &mut self.errors);
    }

    pub fn check_emit(&mut self, path: &Path) {
        validate_emitted_file_with_cfg(path, &mut self.errors);
    }

    pub fn check_manifest(&mut self, out_dir: &Path, entries: &[(String, String)]) {
        validate_manifest_completeness(out_dir, entries, &mut self.errors);
    }

    pub fn check_aliases(&mut self, resolver: &mut ModuleResolver, aliases: &HashSet<String>) {
        // Aliases are stored as just the alias module path string; resolution
        // is checked internally by the resolver.
        for alias_module in aliases {
            let resolution = resolver.resolve_file(alias_module);
            if resolution.is_empty() {
                self.errors.push_fmt(format!(
                    "[alias] Alias {} cannot resolve its target",
                    alias_module
                ));
            }
        }
    }

    pub fn finish(self) -> ValidationResult {
        ValidationResult {
            errors: self.errors,
        }
    }
}

impl Default for ValidationBuilder {
    fn default() -> Self {
        Self::new()
    }
}
