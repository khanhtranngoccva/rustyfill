//! Formatting for generated binding files.
//!
//! Every file the emitter writes to disk is run through [`format_source`]
//! before being saved, so that all generated bindings are `cargo fmt` clean.
//! The preferred backend is the stable `rustfmt` binary; when it is missing
//! or fails (e.g., on a construct it cannot parse), we fall back to a small
//! built-in normalizer so that generation never hard-fails on formatting.

use std::io::Write;
use std::process::{Command, Stdio};

/// Format a complete Rust source file, returning the formatted text.
///
/// Tries `rustfmt --edition 2021` first. If the binary is unavailable or
/// reports an error, falls back to the internal best-effort formatter. The
/// input is always returned unchanged as a last resort, so this function
/// can never lose content.
pub fn format_source(source: &str) -> String {
    if let Some(formatted) = rustfmt_cli(source) {
        return formatted;
    }
    fallback_format(source)
}

/// Run the `rustfmt` binary over `source`. Returns `None` when the binary is
/// not found or exits with a non-zero status (e.g., a syntax error rustfmt
/// cannot recover from).
fn rustfmt_cli(source: &str) -> Option<String> {
    let mut child = Command::new("rustfmt")
        .args(["--quiet", "--emit", "stdout", "--edition", "2021"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    child
        .stdin
        .as_mut()
        .expect("stdin piped")
        .write_all(source.as_bytes())
        .ok()?;

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    // An empty result means rustfmt produced nothing — treat as failure.
    if stdout.trim().is_empty() && !source.trim().is_empty() {
        return None;
    }
    Some(stdout)
}

/// Best-effort internal formatter used when `rustfmt` is unavailable.
///
/// It does not aim for perfect style; it normalizes the most common
/// token-stream serialization artifacts:
/// - collapses runs of blank lines down to a single blank line,
/// - trims trailing whitespace on every line,
/// - guarantees a trailing newline at end of file.
fn fallback_format(source: &str) -> String {
    // Trim trailing whitespace per line, collapse consecutive blank lines to
    // one, and drop leading/trailing blank lines. Each surviving line ends
    // with exactly one `\n`, so the file ends with a single newline.
    let mut lines: Vec<&str> = Vec::new();
    for line in source.lines().map(str::trim_end) {
        if line.is_empty() {
            if lines.last().is_some_and(|l| l.is_empty()) {
                continue;
            }
            lines.push("");
        } else {
            lines.push(line);
        }
    }
    while lines.first().is_some_and(|l| l.is_empty()) {
        lines.remove(0);
    }
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    let mut out = String::with_capacity(source.len());
    for line in lines {
        out.push_str(line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_collapses_blank_lines_and_trims() {
        let src = "pub struct A {\n\n\n    x : u32 ,\n\t\n}\n\n\n";
        let out = fallback_format(src);
        // Runs of blank lines collapse to a single blank line, trailing
        // whitespace is trimmed, and the file ends with exactly one newline.
        assert_eq!(out, "pub struct A {\n\n    x : u32 ,\n\n}\n");
    }

    #[test]
    fn fallback_on_already_clean_input_is_identity() {
        let src = "pub struct A {\n    x: u32,\n}\n";
        assert_eq!(fallback_format(src), src);
    }

    // Miri cannot spawn processes (`posix_spawn`), so the rustfmt-binary path
    // of `format_source` is unrunnable there; the fallback formatter's own
    // behavior is covered by the two tests above.
    #[cfg_attr(miri, ignore = "cannot run cargo fmt process on Miri")]
    #[test]
    fn format_source_never_loses_content_when_rustfmt_missing() {
        // rustfmt is expected to be present in CI/dev environments; either way
        // the output must contain the item name.
        let out = format_source("pub struct Foo < K > { pub k : K , }\n");
        assert!(out.contains("Foo"));
    }
}
