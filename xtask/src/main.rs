//! Developer task runner.
//!
//! Usage:
//! ```text
//! cargo xtask sanitize   # run tests on nightly with leak sanitizer
//! cargo xtask miri       # run tests under Miri for UB detection
//! cargo xtask crap       # CRAP complexity report (coverage-aware)
//! ```

use std::path::Path;
use std::process::Command;

const LCOV_PATH: &str = "target/lcov.info";

fn main() {
    let subcommand = std::env::args().nth(1);

    match subcommand.as_deref() {
        Some("sanitize") => cmd_sanitize(),
        Some("miri") => cmd_miri(),
        Some("crap") => cmd_crap(&std::env::args().skip(2).collect::<Vec<_>>()),
        _ => {
            eprintln!("Usage:");
            eprintln!("  cargo xtask sanitize   — run tests on nightly with -Zsanitizer=leak");
            eprintln!("  cargo xtask miri       — run tests under Miri for undefined behavior");
            eprintln!("  cargo xtask crap [ARGS] — CRAP metric; forwards ARGS to cargo-crap");
            eprintln!("                              (-p <crate> limits to one or more workspace members)");
            std::process::exit(1);
        }
    }
}

/// Run `cargo llvm-cov --workspace --lcov`, then `cargo crap --workspace`.
///
/// Extra arguments are forwarded verbatim to `cargo-crap` (e.g.
/// `-p rustyfill` to limit the report to one crate, `--threshold 30`,
/// `--top 50`, `--format json`). The coverage run is
/// skipped when a fresh-enough lcov file already exists and `--no-cache`
/// is not given.
fn cmd_crap(extra_args: &[String]) {
    let no_cache = extra_args.iter().any(|a| a == "--no-cache");
    let skip_cov_arg = extra_args.iter().position(|a| a == "--skip-coverage").is_some();

    if !skip_cov_arg && (!Path::new(LCOV_PATH).exists() || no_cache) {
        println!("== Generating LCOV coverage (first time or --no-cache) ==");
        let status = Command::new("cargo")
            .args(["llvm-cov", "--workspace", "--lcov", "--output-path", LCOV_PATH])
            .status();
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!(
                    "cargo llvm-cov exited with {:?}; continuing without coverage data",
                    s.code()
                );
            }
            Err(e) => {
                eprintln!("Failed to spawn cargo llvm-cov: {}", e);
                std::process::exit(1);
            }
        }
    } else if Path::new(LCOV_PATH).exists() {
        println!("== Reusing cached coverage at {} (pass --no-cache to regenerate) ==", LCOV_PATH);
    }

    // `-p` selects specific workspace members and conflicts with `--workspace`,
    // so drop the latter when a package filter is present.
    let has_package_filter = extra_args.iter().any(|a| a == "-p" || a == "--package");
    let mut args = vec![String::from("crap")];
    if !has_package_filter {
        args.push(String::from("--workspace"));
    }
    args.push(String::from("--fail-above"));
    if Path::new(LCOV_PATH).exists() {
        args.push(String::from("--lcov"));
        args.push(String::from(LCOV_PATH));
    }
    args.extend(extra_args.iter().filter(|a| *a != "--no-cache").cloned());

    let status = Command::new("cargo").args(&args).status();
    match status {
        Ok(s) => {
            if !s.success() {
                std::process::exit(s.code().unwrap_or(1));
            }
        }
        Err(e) => {
            eprintln!("Failed to spawn cargo crap: {}", e);
            std::process::exit(1);
        }
    }
}

/// Run `cargo +nightly test` with the leak sanitizer enabled.
fn cmd_sanitize() {
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
        "--lib",
        "--tests",
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
