//! Build script that detects the compiler channel and target OS,
//! emitting cfg flags for conditional compilation of unstable features.

use std::env;
use std::io::Write;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap_or_default();

    // Detect if we're compiling with a nightly compiler by probing an unstable feature.
    let is_nightly = probe_nightly(&out_dir);
    cargo_build::rustc_check_cfg("nightly_compiler", ["true", "false"]);
    if is_nightly {
        println!("cargo:rustc-cfg=nightly_compiler=\"true\"");
    } else {
        println!("cargo:rustc-cfg=nightly_compiler=\"false\"");
    }

    // Re-run if the target changes (e.g., switching between host and UEFI targets).
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_OS");
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
