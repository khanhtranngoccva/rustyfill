//! Developer task runner.
//!
//! Usage:
//! ```text
//! cargo xtask clippy     # lint every supported cross target (-D warnings)
//! cargo xtask sanitize   # run tests on nightly with leak sanitizer
//! cargo xtask miri       # run tests under Miri for UB detection
//! cargo xtask crap       # CRAP complexity report (coverage-aware)
//! ```

use clap::{Args, Parser, Subcommand};
use std::path::Path;
use std::process::{Command, ExitCode};

const LCOV_PATH: &str = "target/lcov.info";

/// The no_std-capable core crates. Verification tasks scope to these by
/// default instead of `--workspace`: the other workspace members are either
/// compile-time tooling (`rustyfill-macros`, `rustyfill-sys-bindings`),
/// test-time std-only code (`rustyfill-test-allocator`, whose thread-local
/// failure hooks require std), or std-only consumers (`antipatterns`,
/// `experiments`) — any of them would force `std` back on via feature
/// unification when a non-default feature set is requested.
const CORE_CRATES: &[&str] = &["rustyfill", "rustyfill-errors", "rustyfill-sys"];

#[derive(Parser)]
#[command(name = "xtask", about = "Developer task runner")]
struct Cli {
    #[command(subcommand)]
    command: CommandKind,
}

/// Feature selection shared by every task that builds the workspace.
///
/// Tasks pass `--all-features` by default so feature-gated code is exercised;
/// these flags let callers override that choice. `--no-default-features` and
/// `--features` compose (matching cargo's own semantics): together they mean
/// "build without defaults, but additionally enable exactly these".
#[derive(Debug, Clone, Default, Args)]
pub struct FeatureArgs {
    /// Disable default features (overrides the default --all-features)
    #[arg(long)]
    pub no_default_features: bool,
    /// Comma-separated list of features to enable (overrides the default --all-features; composes with --no-default-features)
    #[arg(short, long, value_delimiter = ',')]
    pub features: Vec<String>,
}

impl FeatureArgs {
    /// The effective feature-selection flags for a cargo invocation.
    ///
    /// Without any override this is `["--all-features"]`. Otherwise the flags
    /// are emitted exactly as given: `--no-default-features` when set, and
    /// `--features <list>` when a list was provided — the two can coexist.
    pub fn cargo_flags(&self) -> Vec<String> {
        if !self.no_default_features && self.features.is_empty() {
            return vec!["--all-features".into()];
        }
        let mut flags = Vec::new();
        if self.no_default_features {
            flags.push("--no-default-features".into());
        }
        if !self.features.is_empty() {
            flags.push("--features".into());
            flags.push(self.features.join(","));
        }
        flags
    }
}

/// Target selection shared by every task that builds crates.
///
/// By default tasks scope to the core (no_std-capable) crates; `--workspace`
/// widens that to every workspace member.
#[derive(Debug, Clone, Default, Args)]
pub struct ScopeArgs {
    /// Include every workspace member instead of just the core crates
    #[arg(long)]
    pub workspace: bool,
}

impl ScopeArgs {
    /// The `-p` flags selecting the target crates for a cargo invocation.
    pub fn cargo_flags(&self) -> Vec<String> {
        if self.workspace {
            // `--workspace` is handled by the caller; nothing to emit here.
            return Vec::new();
        };
        CORE_CRATES
            .iter()
            .flat_map(|c| vec!["-p".to_string(), c.to_string()])
            .collect()
    }

    /// Whether the caller should pass cargo's own `--workspace` flag.
    pub fn use_workspace_flag(&self) -> bool {
        self.workspace
    }
}

/// Cross targets linted by `cargo xtask clippy`.
///
/// Clippy is a checker, not a linker, so it can validate code for targets we
/// cannot actually build/run on the host. The set mirrors the OS families the
/// project supports (Linux gnu + musl, Windows MSVC, macOS, Android, and the
/// BSDs) across both common architectures. Each triple must have its `rust-std`
/// component installed (`rustup target add <triple>`); missing ones are
/// reported and cause a non-zero exit rather than being silently skipped.
const CLIPPY_TARGETS: &[&str] = &[
    // Linux
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-unknown-linux-musl",
    "aarch64-unknown-linux-musl",
    // Windows
    "x86_64-pc-windows-msvc",
    "aarch64-pc-windows-msvc",
    // macOS
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    // Android
    "x86_64-linux-android",
    "aarch64-linux-android",
    // BSDs
    "x86_64-unknown-freebsd",
    "aarch64-unknown-freebsd",
    "x86_64-unknown-openbsd",
    "aarch64-unknown-openbsd",
    "x86_64-unknown-netbsd",
    "aarch64-unknown-netbsd",
];

#[derive(Subcommand)]
enum CommandKind {
    /// Lint every supported cross target with `-D warnings`
    Clippy {
        /// Only lint these targets (comma-separated; default: all of them)
        #[arg(short, long, value_delimiter = ',')]
        targets: Vec<String>,
        #[command(flatten)]
        features: FeatureArgs,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    /// Run tests on nightly with -Zsanitizer=leak
    Sanitize {
        #[command(flatten)]
        features: FeatureArgs,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    /// Run tests under Miri for undefined behavior
    Miri {
        #[command(flatten)]
        features: FeatureArgs,
        #[command(flatten)]
        scope: ScopeArgs,
    },
    /// CRAP metric; forwards extra arguments to cargo-crap
    Crap {
        #[command(flatten)]
        features: FeatureArgs,
        #[command(flatten)]
        scope: ScopeArgs,
        /// Regenerate the coverage report even if a cached one exists
        #[arg(long)]
        no_cache: bool,
        /// Skip generating/reusing coverage data entirely
        #[arg(long)]
        skip_coverage: bool,
        /// Arguments forwarded verbatim to `cargo-crap`
        /// (e.g. `-p <crate>` limits to one or more workspace members)
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        extra_args: Vec<String>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        CommandKind::Clippy {
            targets,
            features,
            scope,
        } => cmd_clippy(targets, features, scope),
        CommandKind::Sanitize { features, scope } => cmd_sanitize(features, scope),
        CommandKind::Miri { features, scope } => cmd_miri(features, scope),
        CommandKind::Crap {
            features,
            scope,
            no_cache,
            skip_coverage,
            extra_args,
        } => cmd_crap(features, scope, no_cache, skip_coverage, &extra_args),
    }
}

/// Run `cargo llvm-cov --workspace --lcov`, then `cargo crap --workspace`.
///
/// Extra arguments are forwarded verbatim to `cargo-crap` (e.g.
/// `-p rustyfill` to limit the report to one crate, `--threshold 30`,
/// `--top 50`, `--format json`). The coverage run is
/// skipped when a fresh-enough lcov file already exists and `--no-cache`
/// is not given.
fn cmd_crap(
    features: FeatureArgs,
    scope: ScopeArgs,
    no_cache: bool,
    skip_coverage: bool,
    extra_args: &[String],
) -> ExitCode {
    if !skip_coverage && (!Path::new(LCOV_PATH).exists() || no_cache) {
        println!("== Generating LCOV coverage (first time or --no-cache) ==");
        // Feature flags come from the parsed overrides (defaulting to
        // --all-features) so feature-gated modules (e.g. `dashmap` behind
        // `unstable`) get instrumented too.
        let mut cov_args: Vec<String> = vec![
            "llvm-cov".into(),
            "--lcov".into(),
            "--output-path".into(),
            LCOV_PATH.into(),
        ];
        if scope.use_workspace_flag() {
            cov_args.push("--workspace".into());
        } else {
            cov_args.extend(scope.cargo_flags());
        }
        cov_args.extend(features.cargo_flags());
        match run("cargo", &cov_args) {
            Ok(()) => {}
            Err(code) => {
                eprintln!(
                    "cargo llvm-cov exited with {}; continuing without coverage data",
                    code
                );
            }
        }
    } else if Path::new(LCOV_PATH).exists() {
        println!(
            "== Reusing cached coverage at {} (pass --no-cache to regenerate) ==",
            LCOV_PATH
        );
    }

    // A forwarded `-p` takes precedence over the default core-crate scope;
    // it also conflicts with `--workspace`, so drop the latter then.
    let has_package_filter = extra_args.iter().any(|a| a == "-p" || a == "--package");
    let mut args = vec![String::from("crap")];
    if has_package_filter {
        // Nothing: the forwarded `-p` selects the crates.
    } else if scope.use_workspace_flag() {
        args.push(String::from("--workspace"));
    } else {
        args.extend(scope.cargo_flags());
    }
    args.push(String::from("--fail-above"));
    if Path::new(LCOV_PATH).exists() {
        args.push(String::from("--lcov"));
        args.push(String::from(LCOV_PATH));
    }
    // Our own control flags were parsed by Clap and never end up in
    // `extra_args`; everything left goes to cargo-crap verbatim
    // (e.g. `-p <crate>`, `--top 50`, `--format json`).
    args.extend(extra_args.iter().cloned());

    match run("cargo", &args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(code) => ExitCode::from(code),
    }
}

/// Lint every supported cross target with `-D warnings`.
///
/// Clippy only type-checks (it never links), so a single Linux host can
/// validate the codebase against every OS/architecture family at once — this
/// is the local equivalent of the CI clippy matrix and catches target-specific
/// regressions (wrong `cfg` gates, platform-conditional dead code, etc.) before
/// they reach CI. Each target is checked independently; all are run even if an
/// earlier one fails, and the process exits non-zero if any target produced a
/// warning or error.
fn cmd_clippy(targets: Vec<String>, features: FeatureArgs, scope: ScopeArgs) -> ExitCode {
    let targets: Vec<&str> = if targets.is_empty() {
        CLIPPY_TARGETS.to_vec()
    } else {
        targets.iter().map(String::as_str).collect()
    };

    // Verify each target's rust-std component is installed up front so a
    // missing component is reported clearly instead of surfacing as a cryptic
    // "target not found" mid-run. Query rustup once for all installed targets.
    let output = match Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return ExitCode::FAILURE,
    };
    let installed = String::from_utf8_lossy(&output.stdout);
    let installed_set: std::collections::HashSet<&str> =
        installed.lines().map(str::trim).collect();
    let missing: Vec<&str> = targets
        .iter()
        .copied()
        .filter(|t| !installed_set.contains(t))
        .collect();
    if !missing.is_empty() {
        eprintln!("Missing rust-std components for: {}", missing.join(", "));
        eprintln!("Install them with: rustup target add {}", missing.join(" "));
        return ExitCode::FAILURE;
    }

    let mut failures = 0usize;
    for t in &targets {
        println!("\n== clippy --target {} ==", t);
        let mut args: Vec<String> = vec![
            "clippy".into(),
            "--all-targets".into(),
            "--target".into(),
            (*t).into(),
        ];
        if scope.use_workspace_flag() {
            args.push("--workspace".into());
        } else {
            args.extend(scope.cargo_flags());
        }
        args.extend(features.cargo_flags());
        args.extend(["--".into(), "-D".into(), "warnings".into()]);

        match run("cargo", &args) {
            Ok(()) => {}
            Err(code) => {
                failures += 1;
                eprintln!("clippy FAILED for {} (exit {})", t, code);
            }
        }
    }

    if failures > 0 {
        eprintln!("\n{} of {} targets failed clippy.", failures, targets.len());
        ExitCode::FAILURE
    } else {
        println!("\nAll {} targets passed clippy cleanly.", targets.len());
        ExitCode::SUCCESS
    }
}

/// Run `cargo +nightly test` with the leak sanitizer enabled.
fn cmd_sanitize(features: FeatureArgs, scope: ScopeArgs) -> ExitCode {
    let mut cmd = Command::new("cargo");
    cmd.env("RUSTFLAGS", "-Zunstable-options -Zsanitizer=leak");
    let mut args: Vec<String> = vec![
        "+nightly".into(),
        "test".into(),
        "-Z".into(),
        "build-std".into(),
        "--target".into(),
        "x86_64-unknown-linux-gnu".into(),
        "--lib".into(),
        "--tests".into(),
    ];
    if scope.use_workspace_flag() {
        args.push("--workspace".into());
    } else {
        args.extend(scope.cargo_flags());
    }
    args.extend(features.cargo_flags());
    args.extend(["--".into(), "--nocapture".into()]);
    cmd.args(args);

    match run_cmd(cmd) {
        Ok(()) => ExitCode::SUCCESS,
        Err(code) => ExitCode::from(code),
    }
}

/// Run `cargo +nightly miri test` to detect undefined behavior.
fn cmd_miri(features: FeatureArgs, scope: ScopeArgs) -> ExitCode {
    // Ensure Miri is installed on the nightly toolchain.
    let mut setup = Command::new("rustup");
    setup.args(["component", "add", "miri", "--toolchain", "nightly"]);
    match run_cmd(setup) {
        Ok(()) => {}
        Err(code) => {
            eprintln!("Failed to install Miri component: {}", code);
            return ExitCode::from(code);
        }
    }

    let mut cmd = Command::new("cargo");
    // Some tests invoke real filesystem syscalls, which Miri's isolated
    // syscall shim rejects — disable isolation so they can run.
    cmd.env("MIRIFLAGS", "-Zdisable-isolation");
    let mut args: Vec<String> = vec!["+nightly".into(), "miri".into(), "test".into()];
    if scope.use_workspace_flag() {
        args.push("--workspace".into());
    } else {
        args.extend(scope.cargo_flags());
    }
    args.extend(features.cargo_flags());
    cmd.args(args);

    match run_cmd(cmd) {
        Ok(()) => ExitCode::SUCCESS,
        Err(code) => ExitCode::from(code),
    }
}

/// Spawn `program` with `args`, returning the child's exit code as an error
/// on failure (or spawn failure).
fn run<S: AsRef<std::ffi::OsStr>>(program: &str, args: &[S]) -> Result<(), u8> {
    let mut cmd = Command::new(program);
    cmd.args(args);
    run_cmd(cmd)
}

/// Translate a [`Command`] status into `Ok(())` / `Err(exit_code)`.
fn run_cmd(mut cmd: Command) -> Result<(), u8> {
    match cmd.status() {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(s.code().unwrap_or(1) as u8),
        Err(e) => {
            eprintln!("Failed to spawn command: {}", e);
            Err(1)
        }
    }
}
