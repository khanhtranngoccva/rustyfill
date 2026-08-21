//! Build script that detects the compiler channel and emits cfg flags for
//! conditional compilation of unstable features. Mirrors `rustyfill`'s build
//! script so that this crate can legally name types re-exported from
//! `core::alloc` (e.g. `AllocError`) when they are gated behind unstable
//! feature flags on nightly.

use std::env;
use std::io::Write;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap_or_default();
    let is_nightly = probe_nightly(&out_dir);
    cargo_build::rustc_check_cfg("nightly_compiler", Vec::<&str>::new());
    if is_nightly {
        println!("cargo:rustc-cfg=nightly_compiler");
    }
}

/// Probe whether we're on a nightly compiler by feeding a snippet that uses
/// `#![feature(...)]` to rustc via stdin. Returns `true` only on nightly.
fn probe_nightly(out_dir: &str) -> bool {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let target = env::var("TARGET").ok();

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
