//! Build script for rustyfill-sys.
//!
//! Orchestrates the doc-JSON-based binding generation pipeline:
//!
//! 1. Rejects `-Zrandomize-layout` (breaks deterministic field layout).
//! 2. Locates the Rust standard library sources. No rustup dependency:
//!    `$RUST_SRC_PATH` wins, then the sysroot's `rust-src` component, then
//!    well-known distro locations (Debian/Ubuntu, Homebrew-style splits).
//!    Only as a last-resort convenience does it try `rustup component add`.
//! 3. Invokes `cargo doc --output-format=json` inside each std library's source
//!    directory to produce authoritative type definitions.
//! 4. Extracts declared types from the JSON and emits binding files.
//!
//! The loader spec (`build/spec.rs`) declares what gets mirrored; the
//! extraction and emission algorithms live in `rustyfill-sys-bindings::docjson`.

#[path = "build/spec.rs"]
mod spec;

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use rustyfill_sys_bindings::docjson::{driver, emit, extract_types};

fn main() {
    reject_randomize_layout();

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);

    // Build the doc-gen config from cargo's own environment ($CARGO, $RUSTC,
    // $TARGET, $CARGO_CFG_FEATURES) so doc-JSON generation is driven by the
    // *exact* toolchain compiling this crate — not whatever resolves on PATH.
    // This means `rustup run nightly cargo build` documents against nightly's
    // std automatically.
    let mut gen_config = driver::DocGenConfig::current();

    // Locate the std library sources for that same toolchain. rustup is NOT
    // required: RUST_SRC_PATH, the sysroot's rust-src component, and distro
    // locations are all honored.
    let src_root = find_library_root(gen_config.rustc.as_deref());
    gen_config.src_root = Some(src_root);

    let target_spec = spec::get_loader_spec();

    // Generate doc-JSON with the procured toolchain. The compiler evaluates
    // all cfgs for the target, so no manual cfg interpretation is needed.
    let doc_output = driver::generate(&gen_config).unwrap_or_else(|errs| {
        for e in &errs {
            eprintln!("cargo:error={}", e);
        }
        std::process::exit(1);
    });

    // Extract declared types from the JSON.
    let types = extract_types(&doc_output.data, &target_spec).unwrap_or_else(|errs| {
        for e in &errs {
            eprintln!("cargo:error={}", e);
        }
        std::process::exit(1);
    });

    // Emit binding files. Pass the full JSON data for canonical path
    // resolution via resolved_path.id + crate_id + external_crates.
    let input = emit::EmitInput {
        out_dir: out_path,
        spec: &target_spec,
        types: &types,
        json_data: &doc_output.data,
    };
    let errors = emit::emit_all(&input);
    for e in &errors {
        eprintln!("cargo:error={}", e);
    }
    if !errors.is_empty() {
        std::process::exit(1);
    }

    println!("cargo:rerun-if-changed=build/spec.rs");
    // Re-run when the driving toolchain or target changes.
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=CARGO");
    println!("cargo:rerun-if-env-changed=TARGET");
}

// ── Std source discovery ──────────────────────────────────────────────────────

/// Locate the root of the Rust standard library sources
/// (the directory containing `core/`, `alloc/`, `std/`).
///
/// Discovery order — rustup is never required:
/// 1. `$RUST_SRC_PATH` — the conventional escape hatch for distro-packaged
///    compilers and Nix-style installs that ship rust-src outside the sysroot.
/// 2. The active toolchain's own `rust-src` component location.
/// 3. Well-known non-rustup locations (Debian/Ubuntu `/usr/share/rustc`,
///    Homebrew-style splits next to the sysroot).
/// 4. Best-effort convenience: if the active toolchain is rustup-managed,
///    try `rustup component add rust-src` once.
///
/// Panics with an actionable message if nothing works.
fn find_library_root(rustc: Option<&Path>) -> PathBuf {
    let mut tried: Vec<String> = Vec::new();

    // 1. RUST_SRC_PATH (explicit user/distro override always wins).
    if let Some(src_path) = env::var_os("RUST_SRC_PATH") {
        let p = PathBuf::from(src_path);
        tried.push(format!("RUST_SRC_PATH={}", p.display()));
        if looks_like_library_root(&p) {
            return p;
        }
    }

    // 2+3. Sysroot-derived candidates and distro locations.
    for candidate in library_root_candidates(rustc) {
        tried.push(candidate.display().to_string());
        if looks_like_library_root(&candidate) {
            return candidate;
        }
    }

    // 4. Last-resort convenience: auto-install via rustup when available.
    if try_install_via_rustup(rustc) {
        for candidate in library_root_candidates(rustc) {
            if looks_like_library_root(&candidate) {
                return candidate;
            }
        }
    }

    panic!(
        "Could not locate Rust standard library sources.\n\
          Tried:\n\
          {}\n\n\
          Fix by pointing RUST_SRC_PATH at the `library` directory of a \
          rust-src checkout that matches your active rustc version \
          (containing core/, alloc/, and std/), or install the rust-src \
          component however your distribution provides it \
          (e.g. `rustup component add rust-src`, `apt install \
          librust-std-dev` on Debian/Ubuntu).",
        tried
            .iter()
            .map(|t| format!("  - {}", t))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Candidate locations for the std `library/` root, in preference order.
fn library_root_candidates(rustc: Option<&Path>) -> Vec<PathBuf> {
    let mut out = Vec::new();

    if let Some(sysroot) = active_sysroot(rustc) {
        // Standard rust-src component layout inside the sysroot.
        out.push(sysroot.join("lib/rustlib/src/rust/library"));

        // Distributions that split the compiler from its sources keep them
        // side by side: /usr/lib/rustlib → /usr/share/rustc/lib/rustlib.
        if let Some(top) = sysroot.parent() {
            out.push(top.join("share/rustc/lib/rustlib/src/rust/library"));
        }
    }

    // Debian/Ubuntu canonical location (also covers most derivatives).
    out.push(PathBuf::from(
        "/usr/share/rustc/lib/rustlib/src/rust/library",
    ));

    out
}

/// The sysroot of the procured rustc, if it can be determined.
fn active_sysroot(rustc: Option<&Path>) -> Option<PathBuf> {
    let bin = rustc
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("rustc"));
    let output = Command::new(&bin).arg("--print=sysroot").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let sysroot = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if sysroot.is_empty() {
        return None;
    }
    Some(PathBuf::from(sysroot))
}

/// Whether `dir` looks like the root of a rust-src `library/` tree.
fn looks_like_library_root(dir: &Path) -> bool {
    dir.join("std/src/lib.rs").is_file()
        && dir.join("core/src/lib.rs").is_file()
        && dir.join("alloc/src/lib.rs").is_file()
}

// ── Optional rustup convenience ───────────────────────────────────────────────

/// As a best-effort convenience (never a hard dependency), try to install the
/// rust-src component for the procured toolchain. Returns true on success.
fn try_install_via_rustup(rustc: Option<&Path>) -> bool {
    let Some(bin) = rustc.map(Path::to_path_buf).filter(|p| p.is_file()) else {
        return false;
    };
    if !is_rustup_managed(&bin) {
        return false;
    }
    let Some(toolchain) = resolve_toolchain_name(&bin) else {
        return false;
    };

    eprintln!(
        "cargo:warning=rustyfill-sys: rust-src not found; attempting \
          `rustup component add rust-src --toolchain {toolchain}`"
    );
    let Ok(output) = Command::new("rustup")
        .args(["component", "add", "rust-src", "--toolchain", &toolchain])
        .output()
    else {
        return false;
    };
    output.status.success()
}

/// Whether the given binary is backed by a rustup-managed toolchain.
/// Used only to decide whether the optional auto-install above is worth trying.
fn is_rustup_managed(binary: &Path) -> bool {
    // A working `rustup toolchain list` implies rustup manages this machine.
    if let Ok(output) = Command::new("rustup").arg("toolchain").arg("list").output() {
        if output.status.success() {
            return true;
        }
    }

    // Heuristic fallback: rustup shims live under <rustup-home>/bin and their
    // sysroot sits under <rustup-home>/toolchains/.
    let Ok(bin) = std::fs::canonicalize(binary) else {
        return false;
    };
    let Some(parent) = bin.parent() else {
        return false;
    };
    parent.file_name().and_then(|n| n.to_str()) == Some("bin")
        && parent.join("..").join("toolchains").is_dir()
}

/// Determine the rustup toolchain name (e.g. `stable-x86_64-unknown-linux-gnu`)
/// backing the given binary, by matching its sysroot against
/// `rustup toolchain list -v`.
fn resolve_toolchain_name(binary: &Path) -> Option<String> {
    let sysroot_output = Command::new(binary).arg("--print=sysroot").output().ok()?;
    let sysroot = String::from_utf8_lossy(&sysroot_output.stdout)
        .trim()
        .to_string();

    let list_output = Command::new("rustup")
        .args(["toolchain", "list", "-v"])
        .output()
        .ok()?;
    if !list_output.status.success() {
        return None;
    }

    for line in String::from_utf8_lossy(&list_output.stdout).lines() {
        // Lines look like: `<name> (<qualifier>) <path>` where the qualifier
        // (e.g. "active, default") itself contains spaces and parentheses.
        let (head, path) = line.rsplit_once(' ')?;
        if path.trim() != sysroot {
            continue;
        }
        let name = head.split('(').next()?.trim();
        if !name.is_empty() {
            return Some(name.to_string());
        }
    }
    None
}

/// Abort the build if `-Zrandomize-layout` is active in the current
/// compilation environment. Layout randomization shuffles field offsets and
/// type alignments, which completely breaks the deterministic layout
/// assumptions that polyfilled mirror structs rely on (identical field
/// layout with the real stdlib types).
fn reject_randomize_layout() {
    // CARGO_ENCODED_RUSTFLAGS contains all effective flags (from RUSTFLAGS,
    // .cargo/config.toml [target.*.rustflags], etc.) null-separated.
    if let Some(encoded) = env::var_os("CARGO_ENCODED_RUSTFLAGS") {
        for flag in encoded.to_string_lossy().split('\0') {
            if flag == "-Zrandomize-layout" || flag == "-Z randomize-layout" {
                panic!(
                    "rustyfill-sys: -Zrandomize-layout is incompatible with polyfilled \
                     bindings.\nThe mirrored data structures require deterministic field \
                     layout matching the standard library.\n\
                     Remove -Zrandomize-layout from your RUSTFLAGS or cargo config."
                );
            }
        }
    }

    // Also check RUSTFLAGS directly (covers cases where CARGO_ENCODED_RUSTFLAGS
    // might not be set, e.g., manual cargo invocations with unusual profiles).
    if let Some(rustflags) = env::var_os("RUSTFLAGS") {
        for flag in rustflags.to_string_lossy().split_whitespace() {
            if flag == "-Zrandomize-layout" {
                panic!(
                    "rustyfill-sys: -Zrandomize-layout is incompatible with polyfilled \
                     bindings.\nThe mirrored data structures require deterministic field \
                     layout matching the standard library.\n\
                     Remove -Zrandomize-layout from your RUSTFLAGS or cargo config."
                );
            }
        }
    }
}
