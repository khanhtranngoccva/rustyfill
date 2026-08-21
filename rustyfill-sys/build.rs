//! Build script for rustyfill-sys.
//!
//! Thin orchestrator over the binding-generation pipeline that lives in
//! `rustyfill-sys-bindings::pipeline`. This file is responsible only for the
//! cargo/toolchain-specific concerns:
//!
//! 1. Rejecting `-Zrandomize-layout` (it breaks deterministic field layout).
//! 2. Locating the Rust standard-library source tree, installing the
//!    `rust-src` component via rustup when it is missing from the active
//!    toolchain.
//! 3. Deriving the platform [`CfgContext`] from the `TARGET` triple.
//! 4. Calling [`pipeline::generate`] and forwarding its diagnostics as
//!    `cargo:error=` / `cargo:warning=` messages.
//!
//! The loader spec (`build/spec.rs`) is declared locally so that changes to
//! what gets mirrored only touch this crate; the multi-phase generation
//! algorithm itself (discovery, import expansion, type-registry construction,
//! minimal-module mirroring, emission, and validation) remains implemented
//! and unit-tested in the bindings crate.

#[path = "build/spec.rs"]
mod spec;

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use rustyfill_sys_bindings::parser::CfgContext;
use rustyfill_sys_bindings::pipeline::{self, PipelineInput};

fn main() {
    reject_randomize_layout();

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);

    let rust_src = find_rust_source_root();
    let spec = spec::get_loader_spec();
    // Cargo sets TARGET but not the individual CARGO_CFG_* vars in build
    // scripts, so derive the platform from the target triple. Fall back to env
    // (which works when run outside cargo) if TARGET is absent.
    let cfg = match env::var("TARGET") {
        Ok(t) => CfgContext::from_target_triple(&t),
        Err(_) => CfgContext::from_env(),
    };

    let input = PipelineInput {
        rust_src: &rust_src,
        out_dir: out_path,
        spec: &spec,
        cfg: &cfg,
    };

    if let Err(report) = pipeline::generate(&input) {
        for w in &report.warnings {
            eprintln!("cargo:warning={}", w);
        }
        for e in &report.errors {
            eprintln!("cargo:error={}", e);
        }
        std::process::exit(1);
    }

    println!("cargo:rerun-if-changed=build/spec.rs");
    println!("cargo:rerun-if-env-changed=RUSTUP_TOOLCHAIN");
}

/// Find the root of the Rust standard library source tree.
///
/// Resolution order:
/// 1. `$RUST_SRC_PATH`, if set and present.
/// 2. The `rust-src` component installed alongside the active toolchain
///    (`<sysroot>/lib/rustlib/src/rust/library`).
/// 3. If the component is missing and the toolchain was provisioned by
///    rustup, install it with `rustup component add rust-src` (idempotent;
///    downloads only the ~10 MB rust-src artifact, reusing rustup's own
///    cache and integrity checking) and retry step 2.
fn find_rust_source_root() -> PathBuf {
    if let Ok(src) = env::var("RUST_SRC_PATH") {
        let p = PathBuf::from(src);
        if p.exists() {
            return p;
        }
    }

    let sysroot_library = sysroot_library_candidate();
    if looks_like_library_root(&sysroot_library) {
        return sysroot_library;
    }

    let detail = match ensure_rust_src_component(&sysroot_library) {
        ComponentInstallOutcome::Installed => return sysroot_library,
        ComponentInstallOutcome::NotApplicable => String::new(),
        ComponentInstallOutcome::Failed(err) => format!("\nrustup fallback failed: {err}"),
    };

    panic!(
        "Could not locate Rust standard library source.\n\
         Attempted: $RUST_SRC_PATH and the rust-src component of the \
         active toolchain ({}){detail}\n\n\
         Fix by installing the rust-src component (`rustup component \
         add rust-src`) or setting RUST_SRC_PATH to the library \
         source root.",
        sysroot_library.display()
    );
}

enum ComponentInstallOutcome {
    /// The component was (re)installed successfully.
    Installed,
    /// The toolchain is not managed by rustup, so there is nothing to do.
    NotApplicable,
    /// rustup ran but reported a failure.
    Failed(String),
}

/// Try to install the `rust-src` component into the active toolchain using
/// rustup itself.
fn ensure_rust_src_component(sysroot_library: &Path) -> ComponentInstallOutcome {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());

    // A bare rustc (no rustup shim in front of it) cannot be augmented.
    if !is_rustup_provisioned(&rustc) {
        return ComponentInstallOutcome::NotApplicable;
    }

    let Some(toolchain) = resolve_toolchain_name(&rustc) else {
        return ComponentInstallOutcome::Failed(
            "could not determine the active rustup toolchain name".into(),
        );
    };

    eprintln!(
        "cargo:warning=rustyfill-sys: rust-src component missing from toolchain \
         `{toolchain}`; running `rustup component add rust-src --toolchain {toolchain}`"
    );

    let output = Command::new("rustup")
        .args(["component", "add", "rust-src", "--toolchain", &toolchain])
        .output();

    let output = match output {
        Ok(o) => o,
        Err(e) => {
            return ComponentInstallOutcome::Failed(format!("failed to run rustup: {e}"));
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return ComponentInstallOutcome::Failed(stderr.trim().to_string());
    }

    if looks_like_library_root(sysroot_library) {
        ComponentInstallOutcome::Installed
    } else {
        ComponentInstallOutcome::Failed(format!(
            "rustup reported success but {} is still missing",
            sysroot_library.display()
        ))
    }
}

/// Whether the given rustc binary is a rustup proxy/shim rather than a bare
/// compiler installation.
fn is_rustup_provisioned(rustc: &std::ffi::OsStr) -> bool {
    // Explicit override wins.
    if env::var_os("RUSTUP_HOME").is_some() {
        return true;
    }

    // Ask rustup directly: it exits non-zero with a clear message when no
    // toolchains are visible to it.
    if let Ok(output) = Command::new("rustup").arg("toolchain").arg("list").output() {
        return output.status.success();
    }

    // Heuristic fallback: rustup shims live under <rustup-home>/bin and their
    // sysroot sits under <rustup-home>/toolchains/.
    let Ok(bin) = std::fs::canonicalize(rustc) else {
        return false;
    };
    let Some(parent) = bin.parent() else {
        return false;
    };
    parent.file_name().and_then(|n| n.to_str()) == Some("bin")
        && parent.join("..").join("toolchains").is_dir()
}

/// Determine the rustup toolchain name (e.g. `stable-x86_64-unknown-linux-gnu`)
/// backing the active rustc, by matching its sysroot against
/// `rustup toolchain list -v`.
fn resolve_toolchain_name(rustc: &std::ffi::OsStr) -> Option<String> {
    let sysroot_output = Command::new(rustc).arg("--print=sysroot").output().ok()?;
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

/// `<sysroot>/lib/rustlib/src/rust/library` for the active rustc.
fn sysroot_library_candidate() -> PathBuf {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let default = PathBuf::from("__no_sysroot__").join("lib/rustlib/src/rust/library");
    let Ok(output) = Command::new(&rustc).arg("--print=sysroot").output() else {
        return default;
    };
    let sysroot = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if sysroot.is_empty() {
        return default;
    }
    PathBuf::from(sysroot).join("lib/rustlib/src/rust/library")
}

/// Whether `dir` looks like the root of a rust-src `library/` tree.
fn looks_like_library_root(dir: &Path) -> bool {
    dir.join("std/src/lib.rs").is_file()
        && dir.join("core/src/lib.rs").is_file()
        && dir.join("alloc/src/lib.rs").is_file()
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
