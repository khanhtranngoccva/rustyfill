//! Developer task runner.
//!
//! Usage:
//! ```text
//! cargo xtask sanitize   # run tests on nightly with leak sanitizer
//! cargo xtask miri       # run tests under Miri for UB detection
//! ```

use std::process::Command;

fn main() {
    let subcommand = std::env::args().nth(1);

    match subcommand.as_deref() {
        Some("sanitize") => cmd_sanitize(),
        Some("miri") => cmd_miri(),
        _ => {
            eprintln!("Usage:");
            eprintln!("  cargo xtask sanitize   — run tests on nightly with -Zsanitizer=leak");
            eprintln!("  cargo xtask miri       — run tests under Miri for undefined behavior");
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

/// Run `cargo +nightly miri test` to detect undefined behavior.
fn cmd_miri() {
    // Ensure Miri is installed on the nightly toolchain.
    let mut setup = Command::new("rustup");
    setup.args(["component", "add", "miri", "--toolchain", "nightly"]);
    let setup_status = setup.status();
    match setup_status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            eprintln!("Failed to install Miri component: {:?}", s.code());
            std::process::exit(s.code().unwrap_or(1));
        }
        Err(e) => {
            eprintln!("Failed to spawn rustup: {}", e);
            std::process::exit(1);
        }
    }

    let mut cmd = Command::new("cargo");
    cmd.args(["+nightly", "miri", "test"]);

    let status = cmd.status();
    match status {
        Ok(s) => {
            if !s.success() {
                std::process::exit(s.code().unwrap_or(1));
            }
        }
        Err(e) => {
            eprintln!("Failed to spawn cargo miri: {}", e);
            std::process::exit(1);
        }
    }
}
