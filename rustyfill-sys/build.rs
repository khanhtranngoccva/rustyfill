//! Build script for rustyfill-sys.
//!
//! Thin orchestrator over the binding-generation pipeline that lives in
//! `rustyfill-sys-bindings::pipeline`. This file is responsible only for the
//! cargo/toolchain-specific concerns:
//!
//! 1. Rejecting `-Zrandomize-layout` (it breaks deterministic field layout).
//! 2. Locating the Rust standard-library source tree.
//! 3. Deriving the platform [`CfgContext`] from the `TARGET` triple.
//! 4. Calling [`pipeline::generate`] and forwarding its diagnostics as
//!    `cargo:error=` / `cargo:warning=` messages.
//!
//! All of the multi-phase generation algorithm (discovery, import expansion,
//! type-registry construction, minimal-module mirroring, emission, and
//! validation) is implemented and unit-tested in the bindings crate.

use std::env;
use std::path::{Path, PathBuf};

use rustyfill_sys_bindings::get_loader_spec;
use rustyfill_sys_bindings::parser::CfgContext;
use rustyfill_sys_bindings::pipeline::{self, PipelineInput};

fn main() {
    reject_randomize_layout();

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);

    let rust_src = find_rust_source_root();
    let spec = get_loader_spec();
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

    println!("cargo:rerun-if-changed=../rustyfill-sys-bindings/src/spec.rs");
    println!("cargo:rerun-if-env-changed=RUSTUP_TOOLCHAIN");
}

/// Find the root of the Rust standard library source tree.
fn find_rust_source_root() -> PathBuf {
    if let Ok(src) = env::var("RUST_SRC_PATH") {
        let p = PathBuf::from(src);
        if p.exists() {
            return p;
        }
    }

    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    if let Ok(output) = std::process::Command::new(&rustc)
        .arg("--print=sysroot")
        .output()
    {
        let sysroot = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let candidate = PathBuf::from(&sysroot).join("lib/rustlib/src/rust/library");

        if candidate.exists() {
            return candidate;
        }
    }

    if let Ok(home) = env::var("HOME") {
        let candidate2 = PathBuf::from(&home).join(
            ".rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library",
        );

        if candidate2.exists() {
            return candidate2;
        }
    }

    panic!(
        "Could not locate Rust standard library source.\n\
         Install the rust-src component: `rustup component add rust-src`\n\
         Or set RUST_SRC_PATH to the library source root."
    );
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
