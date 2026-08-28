//! Driver for invoking `cargo doc` with JSON output inside the rust-src tree.
//!
//! This module orchestrates the toolchain invocation that produces the
//! authoritative JSON data our extraction pipeline consumes. Each library's
//! blob is deserialized into the typed [`super::wire::Crate`] immediately, so
//! downstream stages never touch raw JSON.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::wire::Crate;

/// Configuration for the doc-JSON generation run.
#[derive(Debug, Clone, Default)]
pub struct DocGenConfig {
    /// Target triple to compile for (e.g., "x86_64-unknown-linux-gnu").
    /// Sourced from `$TARGET` by [`Self::current`].
    pub target_triple: String,
    /// Additional cfg flags to pass through via RUSTDOCFLAGS (e.g., ['--cfg', 'foo']).
    pub extra_rustflags: Vec<String>,
    /// Explicit std `library/` root (containing `core/`, `alloc/`, `std/`).
    /// When set, this overrides sysroot-based discovery — use it for
    /// non-rustup toolchain layouts (distro packages, Nix, RUST_SRC_PATH).
    pub src_root: Option<PathBuf>,
    /// Absolute path to the `cargo` binary to invoke. When set, all
    /// toolchain invocations (`cargo doc`) use this binary instead of
    /// whatever `cargo` resolves on PATH. Sourced from `$CARGO`.
    pub cargo_bin: Option<PathBuf>,
    /// Absolute path to the `rustc` binary paired with the driving cargo.
    /// Used for version/sysroot probing so the doc-JSON always matches the
    /// exact toolchain compiling the crate. Sourced from `$RUSTC`.
    pub rustc: Option<PathBuf>,
    /// Identity string for the toolchain producing the doc-JSON (its full
    /// `rustc --version`). Part of the cache key so different toolchains
    /// never share cached data.
    pub toolchain_id: String,
}

impl DocGenConfig {
    /// Build a config from the cargo-provided environment variables, so the
    /// doc-JSON generation is driven by the *exact* toolchain compiling the
    /// crate rather than whatever happens to resolve on PATH.
    ///
    /// Reads:
    /// - `$CARGO`   → [`Self::cargo_bin`] (the cargo binary driving the build).
    /// - `$RUSTC`   → [`Self::rustc`] (the rustc driving the build); its
    ///   `--version` becomes [`Self::toolchain_id`] and its `--print sysroot`
    ///   seeds the default source location.
    /// - `$TARGET`  → [`Self::target_triple`] (falls back to the rustc host).
    /// - `$CARGO_CFG_FEATURES` → [`Self::extra_rustflags`] as `--cfg feature=…`
    ///   passthroughs, so feature-gated std items are documented consistently.
    ///
    /// Every variable degrades gracefully: a missing or unusable value falls
    /// back to PATH resolution / rustc defaults, so this works both inside a
    /// real cargo build and in standalone test contexts.
    pub fn current() -> Self {
        let rustc = env_path("RUSTC");
        let cargo_bin = env_path("CARGO");

        // Prefer $TARGET; fall back to the rustc's own host triple.
        let target_triple = env_var("TARGET")
            .or_else(|| rustc_host_triple(rustc.as_deref()))
            .unwrap_or_default();

        // Version identity straight from the resolved rustc.
        let toolchain_id = rustc_version(rustc.as_deref());

        // Surface enabled crate features as cfg flags so the documented std
        // surface matches what the consuming crate will actually see.
        let mut extra_rustflags = Vec::new();
        if let Some(features) = env_var("CARGO_CFG_FEATURES") {
            for f in features.split(',') {
                let f = f.trim();
                if !f.is_empty() {
                    extra_rustflags.push(format!("--cfg feature={}", f));
                }
            }
        }

        Self {
            target_triple,
            extra_rustflags,
            cargo_bin,
            rustc,
            toolchain_id,
            ..Default::default()
        }
    }
}

/// Read an environment variable, returning `None` when unset or empty.
fn env_var(name: &str) -> Option<String> {
    std::env::var_os(name)
        .map(|v| v.to_string_lossy().into_owned())
        .filter(|v| !v.is_empty())
}

/// Read an environment variable that holds a filesystem path, keeping only
/// values that point at an existing file (guards against stale/garbage).
fn env_path(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(env_var(name)?);
    p.is_file().then_some(p)
}

/// The host triple reported by the given rustc (or PATH `rustc`).
fn rustc_host_triple(rustc: Option<&Path>) -> Option<String> {
    let bin = rustc
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("rustc"));
    let output = Command::new(&bin).args(["-vV"]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|l| l.strip_prefix("host: "))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Full `rustc --version` for the given binary (or PATH `rustc`).
fn rustc_version(rustc: Option<&Path>) -> String {
    let bin = rustc
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("rustc"));
    Command::new(&bin)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown-rustc".into())
}

/// The result of a successful doc-JSON generation run.
pub struct DocJsonOutput {
    /// Typed crate data keyed by library name ("core", "alloc", "std").
    pub data: HashMap<String, Crate>,
    /// The format_version reported by the toolchain.
    pub format_version: u32,
    /// The sysroot path used.
    pub sysroot: PathBuf,
}

/// Generate doc-JSON for the given set of libraries by running `cargo doc`
/// inside each library's source directory in the rust-src tree.
///
/// Results are cached on disk keyed by `(rustc version, target triple,
/// src-root path)` so repeat builds within the same toolchain skip the
/// expensive re-documentation entirely. Set `RUSTYFILL_NO_DOC_CACHE=1` to
/// bypass the cache.
///
/// # Requirements
/// - The active Rust toolchain must have the `rust-src` component installed.
/// - `RUSTC_BOOTSTRAP=1` is set automatically to unlock `-Zunstable-options`.
///
/// # Returns
/// Parsed JSON data for each successfully documented library.
pub fn generate(config: &DocGenConfig) -> Result<DocJsonOutput, Vec<String>> {
    let (sysroot, src_root) = match &config.src_root {
        Some(root) => (find_sysroot(config).unwrap_or_default(), root.clone()),
        None => {
            let sysroot = find_sysroot(config)?;
            let src_root = sysroot
                .join("lib")
                .join("rustlib")
                .join("src")
                .join("rust")
                .join("library");
            (sysroot, src_root)
        }
    };

    if !src_root.is_dir() {
        return Err(vec![format!(
            "std library sources not found at {}. Point RUST_SRC_PATH at a \
             rust-src `library` directory matching your rustc version, or \
             install the rust-src component however your distribution provides it.",
            src_root.display()
        )]);
    }

    let use_cache = std::env::var_os("RUSTYFILL_NO_DOC_CACHE").is_none();

    if use_cache {
        if let Some(cached) = try_load_cache(
            &config.toolchain_id,
            &config.target_triple,
            &src_root,
            sysroot.clone(),
        ) {
            return Ok(cached);
        }
    }

    let result = run_generation(&sysroot, &src_root, config)?;

    if use_cache {
        if let Err(e) = save_cache(
            &result.data,
            result.format_version,
            &config.toolchain_id,
            &config.target_triple,
            &src_root,
        ) {
            eprintln!("warning: failed to write doc-JSON cache: {}", e);
        }
    }

    Ok(result)
}

/// Run `cargo doc` for each library and assemble the parsed output.
fn run_generation(
    sysroot: &Path,
    src_root: &Path,
    config: &DocGenConfig,
) -> Result<DocJsonOutput, Vec<String>> {
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
            errors.push(format!(
                "Library directory not found: {}",
                lib_dir.display()
            ));
            continue;
        }

        match run_cargo_doc(lib, &lib_dir, &target_dir, config) {
            Ok(json_path) => match load_crate(&json_path) {
                Ok(crate_) => {
                    format_version = Some(crate_.format_version);
                    data.insert(lib.to_string(), crate_);
                }
                Err(e) => {
                    errors.push(format!("[{}] Failed to parse JSON: {}", lib, e));
                }
            },
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
            sysroot: sysroot.to_owned(),
        })
    } else {
        Err(errors)
    }
}

// ── Cache layer ───────────────────────────────────────────────────────────────

/// Directory holding per-toolchain doc-JSON caches.
fn cache_base_dir() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| dirs_fallback_home().join(".cache"))
        .join("rustyfill")
}

fn dirs_fallback_home() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
}

// Short hash of arbitrary strings, used to keep cache paths filesystem-safe.
fn short_hash(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", h)
}

/// Cache directory for a specific (version, triple, src-root) combination.
fn cache_dir_for(version: &str, triple: &str, src_root: &Path) -> PathBuf {
    let canonical = src_root
        .canonicalize()
        .unwrap_or_else(|_| src_root.to_path_buf());
    let key = format!("{}|{}|{}", version, triple, canonical.display());
    cache_base_dir().join(format!("docjson-{}", short_hash(&key)))
}

fn try_load_cache(
    version: &str,
    triple: &str,
    src_root: &Path,
    sysroot: PathBuf,
) -> Option<DocJsonOutput> {
    let dir = cache_dir_for(version, triple, src_root);
    let meta_path = dir.join("meta.json");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(meta_path).ok()?).ok()?;
    let format_version = meta.get("format_version")?.as_u64()? as u32;

    let mut data = HashMap::new();
    for lib in ["core", "alloc", "std"] {
        let file = dir.join(format!("{}.json", lib));
        let crate_ = load_crate(&file).ok()?;
        data.insert(lib.to_string(), crate_);
    }

    Some(DocJsonOutput {
        data,
        format_version,
        sysroot,
    })
}

fn save_cache(
    data: &HashMap<String, Crate>,
    format_version: u32,
    version: &str,
    triple: &str,
    src_root: &Path,
) -> Result<(), String> {
    let dir = cache_dir_for(version, triple, src_root);
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {}", dir.display(), e))?;

    // Serialize everything first, then write files, so a mid-write failure
    // leaves the previous cache intact rather than half-updated.
    let mut serialized: Vec<(String, String)> = Vec::with_capacity(data.len());
    for (k, v) in data {
        let json = serde_json::to_string(v).map_err(|e| format!("serialize {}: {}", k, e))?;
        serialized.push((k.clone(), json));
    }

    for (lib, json) in serialized {
        let path = dir.join(format!("{}.json", lib));
        std::fs::write(&path, json)
            .map_err(|e| format!("cannot write {}: {}", path.display(), e))?;
    }

    let meta = serde_json::json!({
        "format_version": format_version,
        "rustc_version": version,
        "triple": triple,
    });
    std::fs::write(dir.join("meta.json"), meta.to_string())
        .map_err(|e| format!("cannot write meta.json: {}", e))?;

    Ok(())
}

/// The rustc binary paired with the configured cargo (same toolchain dir).
fn rustc_for(config: &DocGenConfig) -> PathBuf {
    config
        .rustc
        .clone()
        .unwrap_or_else(|| PathBuf::from("rustc"))
}

/// Find the sysroot of the toolchain in use (honoring the procured rustc).
fn find_sysroot(config: &DocGenConfig) -> Result<PathBuf, Vec<String>> {
    let rustc = rustc_for(config);
    let output = Command::new(&rustc)
        .args(["--print", "sysroot"])
        .output()
        .map_err(|e| {
            vec![format!(
                "failed to run '{} --print sysroot': {}",
                rustc.display(),
                e
            )]
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(vec![format!(
            "'{} --print sysroot' failed: {}",
            rustc.display(),
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
    let cargo = config.cargo_bin.as_deref().unwrap_or(Path::new("cargo"));
    let mut cmd = Command::new(cargo);
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

/// Load and deserialize a rustdoc JSON blob into the typed wire model.
fn load_crate(path: &Path) -> Result<Crate, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    serde_json::from_str(&content).map_err(|e| format!("invalid JSON in {}: {}", path.display(), e))
}
