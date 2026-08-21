//! Build script that detects the compiler channel, target OS, and panic strategy,
//! emitting cfg flags for conditional compilation of unstable features.

use std::env;
use std::io::Write;
use std::path::Path;

fn main() {
    // Fail immediately if layout randomization is active — it breaks the
    // deterministic layout assumptions that polyfilled mirrors depend on.
    reject_randomize_layout();

    let out_dir = env::var("OUT_DIR").unwrap_or_default();

    // Detect if we're compiling with a nightly compiler by probing an unstable feature.
    let is_nightly = probe_nightly(&out_dir);
    cargo_build::rustc_check_cfg("nightly_compiler", Vec::<&str>::new());
    if is_nightly {
        println!("cargo:rustc-cfg=nightly_compiler");
    }

    // Emit `allocator_api_enabled` only when BOTH the `allocator-api` Cargo
    // feature is active AND we're on nightly. This single cfg gates the
    // `#![feature(allocator_api)]` / `#![feature(try_reserve_kind)]` attrs in
    // lib.rs and the real-vs-ponyfill type exports in alloc.rs.
    let feature_allocator_api = env::var_os("CARGO_FEATURE_ALLOCATOR_API").is_some();
    cargo_build::rustc_check_cfg("allocator_api_enabled", Vec::<&str>::new());
    if is_nightly && feature_allocator_api {
        println!("cargo:rustc-cfg=allocator_api_enabled");
    }

    // Re-run if the target changes (e.g., switching between host and UEFI targets).
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_OS");
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
                    "rustyfill: -Zrandomize-layout is incompatible with polyfilled \
                     bindings.\nThe mirrored data structures require deterministic field \
                     layout matching the standard library.\n\
                     Remove -Zrandomize-layout from your RUSTFLAGS or cargo config."
                );
            }
        }
    }

    // Also check RUSTFLAGS directly as a fallback.
    if let Some(rustflags) = env::var_os("RUSTFLAGS") {
        for flag in rustflags.to_string_lossy().split_whitespace() {
            if flag == "-Zrandomize-layout" {
                panic!(
                    "rustyfill: -Zrandomize-layout is incompatible with polyfilled \
                     bindings.\nThe mirrored data structures require deterministic field \
                     layout matching the standard library.\n\
                     Remove -Zrandomize-layout from your RUSTFLAGS or cargo config."
                );
            }
        }
    }
}

/// Probe whether we're on a nightly compiler by feeding a snippet that uses
/// `#![feature(...)]` to rustc via stdin. Returns `true` only on nightly compilers.
/// The compiled output is directed into `out_dir` so it doesn't pollute the source tree.
fn probe_nightly(out_dir: &str) -> bool {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let target = env::var("TARGET").ok();

    // Direct the probe's output rlib into OUT_DIR instead of the source directory.
    let out_rlib = Path::new(out_dir).join("probe_nightly.rlib");
    let out_path = out_rlib.to_string_lossy().to_string();

    let mut cmd = std::process::Command::new(&rustc);
    cmd.arg("--crate-type=rlib")
        .arg("-")
        .args(["-o", &out_path])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());

    if let Some(ref t) = target {
        cmd.args(["--target", t]);
    }

    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(_) => return false,
    };

    // Feed a minimal crate that enables an unstable feature.
    // Stable rustc rejects this with error[E0554]; nightly accepts it.
    if child
        .stdin
        .as_ref()
        .unwrap()
        .write_all(b"#![feature(core_intrinsics)]\npub fn f() {}\n")
        .is_err()
    {
        return false;
    }

    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(_) => return false,
    };

    let stderr = String::from_utf8_lossy(&output.stderr);
    !stderr.contains("error[E0554]")
}
