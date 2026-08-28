//! Driver for invoking `cargo doc` with JSON output inside the rust-src tree.
//!
//! This module orchestrates the toolchain invocation that produces the
//! authoritative JSON data our extraction pipeline consumes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Configuration for the doc-JSON generation run.
#[derive(Debug, Clone, Default)]
pub struct DocGenConfig {
    /// Target triple to compile for (e.g., "x86_64-unknown-linux-gnu").
    /// Defaults to the host triple detected from `rustc`.
    pub target_triple: String,
    /// Additional cfg flags to pass through via RUSTFLAGS (e.g., ['--cfg', 'foo']).
    pub extra_rustflags: Vec<String>,
    /// Feature flags to enable when documenting the library crates.
    pub features: Vec<String>,
}

impl DocGenConfig {
    /// Create a config targeting the current host.
    pub fn host() -> Result<Self, String> {
        let output = Command::new("rustc")
            .args(["-vV"])
            .output()
            .map_err(|e| format!("failed to run 'rustc -vV': {}", e))?;

        if !output.status.success() {
            return Err("rustc -vV failed".into());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let host = stdout
            .lines()
            .find_map(|l| l.strip_prefix("host: "))
            .map(|s| s.trim().to_string())
            .ok_or("could not determine host triple from rustc -vV")?;

        Ok(Self {
            target_triple: host,
            ..Default::default()
        })
    }
}

/// The result of a successful doc-JSON generation run.
pub struct DocJsonOutput {
    /// Parsed JSON data keyed by library name ("core", "alloc", "std").
    pub data: HashMap<String, serde_json::Value>,
    /// The format_version reported by the toolchain.
    pub format_version: u32,
    /// The sysroot path used.
    pub sysroot: PathBuf,
}

/// Generate doc-JSON for the given set of libraries by running `cargo doc`
/// inside each library's source directory in the rust-src tree.
///
/// # Requirements
/// - The active Rust toolchain must have the `rust-src` component installed.
/// - `RUSTC_BOOTSTRAP=1` is set automatically to unlock `-Zunstable-options`.
///
/// # Returns
/// Parsed JSON data for each successfully documented library.
pub fn generate(config: &DocGenConfig) -> Result<DocJsonOutput, Vec<String>> {
    let sysroot = find_sysroot()?;
    let src_root = sysroot
        .join("lib")
        .join("rustlib")
        .join("src")
        .join("rust")
        .join("library");

    if !src_root.is_dir() {
        return Err(vec![format!(
            "rust-src not found at {}. Install it with: rustup component add rust-src",
            src_root.display()
        )]);
    }

    // Use a temp directory for cargo's target to avoid polluting the toolchain
    let target_dir = std::env::temp_dir().join(format!(
        "rustyfill-docgen-{}-{}",
        std::process::id(),
        &config.target_triple
    ));
    let _ = std::fs::create_dir_all(&target_dir);

    let libs = ["core", "alloc", "std"];
    let mut data = HashMap::new();
    let mut errors = Vec::new();
    let mut format_version: Option<u32> = None;

    for lib in &libs {
        let lib_dir = src_root.join(lib);
        if !lib_dir.is_dir() {
            errors.push(format!("Library directory not found: {}", lib_dir.display()));
            continue;
        }

        match run_cargo_doc(lib, &lib_dir, &target_dir, config) {
            Ok(json_path) => {
                match load_json(&json_path) {
                    Ok(value) => {
                        let fv = value
                            .get("format_version")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as u32);
                        if let Some(v) = fv {
                            format_version = Some(v);
                        }
                        data.insert(lib.to_string(), value);
                    }
                    Err(e) => {
                        errors.push(format!("[{}] Failed to parse JSON: {}", lib, e));
                    }
                }
            }
            Err(e) => {
                errors.push(format!("[{}] cargo doc failed: {}", lib, e));
            }
        }
    }

    // Clean up temp dir
    let _ = std::fs::remove_dir_all(&target_dir);

    if errors.is_empty() && data.contains_key("core") {
        Ok(DocJsonOutput {
            data,
            format_version: format_version.unwrap_or(0),
            sysroot,
        })
    } else {
        Err(errors)
    }
}

/// Find the sysroot of the active Rust toolchain.
fn find_sysroot() -> Result<PathBuf, Vec<String>> {
    let output = Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .map_err(|e| vec![format!("failed to run 'rustc --print sysroot': {}", e)])?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(vec![format!(
            "rustc --print sysroot failed: {}",
            stderr.trim()
        )]);
    }

    let sysroot = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(PathBuf::from(sysroot))
}

/// Run `cargo doc --no-deps` inside a single library's source directory.
fn run_cargo_doc(
    lib_name: &str,
    lib_dir: &Path,
    target_dir: &Path,
    config: &DocGenConfig,
) -> Result<PathBuf, String> {
    let mut cmd = Command::new("cargo");
    cmd.current_dir(lib_dir)
        .arg("doc")
        .arg("--no-deps")
        .arg("--target")
        .arg(&config.target_triple);

    // Note: extra cfgs are passed via RUSTDOCFLAGS below, not as cargo args

    // Set environment variables
    cmd.env("RUSTC_BOOTSTRAP", "1");

    let mut rustdoc_flags = String::from(
        "-Zunstable-options --output-format=json --document-private-items --document-hidden-items",
    );
    for flag in &config.extra_rustflags {
        rustdoc_flags.push(' ');
        rustdoc_flags.push_str(flag);
    }
    cmd.env("RUSTDOCFLAGS", &rustdoc_flags);
    cmd.env("CARGO_TARGET_DIR", target_dir);

    // Suppress cargo's stdout noise
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let output = cmd
        .output()
        .map_err(|e| format!("failed to spawn cargo: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "exit {}\nstdout: {}\nstderr: {}",
            output.status.code().unwrap_or(-1),
            stdout.trim(),
            stderr.trim()
        ));
    }

    // The JSON output lands at: $CARGO_TARGET_DIR/$TRIPLE/doc/$LIB_NAME.json
    let json_path = target_dir
        .join(&config.target_triple)
        .join("doc")
        .join(format!("{}.json", lib_name));

    if !json_path.is_file() {
        // Try without the target subdirectory (some cargo versions put it directly in doc/)
        let alt_path = target_dir.join("doc").join(format!("{}.json", lib_name));
        if alt_path.is_file() {
            return Ok(alt_path);
        }
        return Err(format!(
            "Expected JSON output not found at {} or {}",
            json_path.display(),
            alt_path.display()
        ));
    }

    Ok(json_path)
}

/// Load and parse a JSON file.
fn load_json(path: &Path) -> Result<serde_json::Value, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("invalid JSON in {}: {}", path.display(), e))
}
