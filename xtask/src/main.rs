//! Developer task runner.
//!
//! Usage:
//! ```text
//! cargo xtask sanitize   # run tests on nightly with leak sanitizer
//! ```

use std::process::Command;

fn main() {
    let subcommand = std::env::args().nth(1);

    match subcommand.as_deref() {
        Some("sanitize") => cmd_sanitize(),
        _ => {
            eprintln!("Usage:");
            eprintln!("  cargo xtask sanitize   — run tests on nightly with -Zsanitizer=leak");
            std::process::exit(1);
        }
    }
}

/// Run `cargo +nightly test` with the leak sanitizer enabled.
fn cmd_sanitize() {
    // Sanitizers require debug builds (no optimizations) and a cross-compilation
    // target so that LLVM instrumentation is applied. We build libstd from source
    // via `-Zbuild-std` so that panic paths are also instrumented.
    let mut cmd = Command::new("cargo");
    cmd.env(
        "RUSTFLAGS",
        "-Zunstable-options -Cpanic=immediate-abort -Zsanitizer=leak -Zpanic_abort_tests",
    );
    cmd.args([
        "+nightly",
        "test",
        "-Z",
        "build-std",
        "--target",
        "x86_64-unknown-linux-gnu",
        "--",
        "--nocapture",
    ]);

    let status = cmd.status();
    match status {
        Ok(s) => {
            if !s.success() {
                std::process::exit(s.code().unwrap_or(1));
            }
        }
        Err(e) => {
            eprintln!("Failed to spawn cargo: {}", e);
            std::process::exit(1);
        }
    }
}
