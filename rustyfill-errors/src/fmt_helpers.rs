//! Internal helpers used by the [`Display`] / [`Debug`] implementations.
use alloc::boxed::Box;
use alloc::vec::Vec;

use core::fmt;
use core::panic::Location;

use rustyfill::alloc::TryReserveError;
use rustyfill::prelude::TryVec;
use rustyfill::{try_write, try_writeln};

use crate::ItemImpl;

// ── Connector constants ────────────────────────────────────────────────────────

/// Thin branch connector for sub-items: `├╴`
pub(super) const THIN_BRANCH: &str = "\u{251c}\u{2574}";

/// Thin last-item connector: `╰╴`
pub(super) const THIN_LAST: &str = "\u{2570}\u{2574}";

/// Source header first branch (curves from separator, forks down): `╰┬▶ `
pub(super) const SOURCE_FIRST: &str = "\u{2570}\u{252c}\u{25b6} ";

/// Source header exclusive / sole child (en-dash in middle): `╰─▶ `
pub(super) const SOURCE_EXCLUSIVE: &str = "\u{2570}\u{2500}\u{25b6} ";

/// Source header middle branch (vertical continues): `├▶ `
pub(super) const SOURCE_MID: &str = "\u{251c}\u{25b6} ";

/// Source header last branch (terminates): `╰▶ `
pub(super) const SOURCE_LAST: &str = "\u{2570}\u{25b6} ";

/// Vertical separator character: `│`
pub(super) const SEP_CHAR: &str = "\u{2502}";

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Extend the `continuing_below` tracker so it covers the ancestor slot for
/// `depth` (index = depth - 1). Newly added slots default to `true`. Depth 0
/// is a no-op.
///
/// Returns [`Err`] if allocating capacity for new slots fails. The vector is
/// left in a valid (shorter) state; callers treat missing slots as `true` via
/// [`write_indent`] and friends, so rendering degrades gracefully.
pub(super) fn extend_continuing(
    continuing: &mut Vec<bool>,
    depth: usize,
) -> Result<(), TryReserveError> {
    let needed = depth.saturating_sub(1);
    while continuing.len() <= needed {
        continuing.try_push(true)?;
    }
    Ok(())
}

/// Mark that the ancestor at `ancestor_depth` (1-based) was the last or
/// exclusive sibling, so deeper levels use spaces for that column.
pub(super) fn mark_last_at(continuing: &mut [bool], ancestor_depth: usize) {
    let idx = ancestor_depth.saturating_sub(1);
    if idx < continuing.len() {
        continuing[idx] = false;
    }
}

/// Write the indentation prefix for content at `depth`.
///
/// Depth 0 → no indent.
/// Depth 1 → no indent (source connectors carry their own offset).
/// Depth ≥ 2 → one fixed 4-char unit per ancestor from `continuing_below`.
pub(super) fn write_indent(
    f: &mut fmt::Formatter<'_>,
    continuing: &[bool],
    depth: usize,
) -> fmt::Result {
    if depth <= 1 {
        return Ok(());
    }

    for i in 0..depth.saturating_sub(1) {
        let is_cont = continuing.get(i).copied().unwrap_or(true);
        if is_cont {
            f.write_str(" \u{2502}  ")?; // " │  " = 4 chars
        } else {
            f.write_str("    ")?; // "    " = 4 spaces
        }
    }
    Ok(())
}

/// Write a separator bar between sibling source frames at `depth`.
///
/// Produces a line of ancestor-column prefixes ending with a vertical bar
/// at the current depth's position. For depth 1 this yields ` │\n`.
pub(super) fn write_sibling_separator(
    f: &mut fmt::Formatter<'_>,
    continuing: &[bool],
    depth: usize,
) -> fmt::Result {
    let ancestors = depth.saturating_sub(1);
    for i in 0..ancestors {
        let is_cont = continuing.get(i).copied().unwrap_or(true);
        if is_cont {
            f.write_str(" \u{2502}")?; // " │" = 2 chars
        } else {
            f.write_str("  ")?; // "  " = 2 spaces
        }
    }
    f.write_str(" \u{2502}")?; // " │" = 2 chars
    try_writeln!(f)?;
    Ok(())
}

/// Counts printable vs opaque attachments in a slice.
pub(super) fn count_attachments(attachments: &[Box<dyn ItemImpl>]) -> (usize, usize) {
    let printable = attachments.iter().filter(|a| a.is_printable()).count();
    let opaque = attachments.len().saturating_sub(printable);
    (printable, opaque)
}

/// Write `at <file>:<line>:<col>` to the formatter, splitting backslash
/// path components into forward slashes when the test guard is active.
/// No allocation occurs even under normalization.
pub(super) fn write_location(f: &mut fmt::Formatter<'_>, loc: &Location<'_>) -> fmt::Result {
    f.write_str("at ")?;
    let file = loc.file();

    #[cfg(feature = "std")]
    {
        if crate::FORCE_FORWARD_SLASHES.with(|fl| fl.get()) && file.contains('\\') {
            let mut parts = file.split('\\');
            if let Some(first) = parts.next() {
                f.write_str(first)?;
            }
            for part in parts {
                f.write_str("/")?;
                f.write_str(part)?;
            }
        } else {
            f.write_str(file)?;
        }
    }
    #[cfg(not(feature = "std"))]
    {
        f.write_str(file)?;
    }

    f.write_str(":")?;
    try_write!(f, "{}", loc.line())?;
    f.write_str(":")?;
    try_write!(f, "{}", loc.column())?;
    Ok(())
}

/// Renders attachments for a frame at the given sub-depth.
pub(super) fn render_attachments(
    f: &mut fmt::Formatter<'_>,
    continuing: &[bool],
    sub_depth: usize,
    attachments: &[Box<dyn ItemImpl>],
    total_lines: usize,
    first_line_idx: usize,
) -> fmt::Result {
    let (_, opaque_count) = count_attachments(attachments);

    for (line_idx_off, att) in attachments.iter().filter(|a| a.is_printable()).enumerate() {
        let line_idx = first_line_idx + line_idx_off;
        write_indent(f, continuing, sub_depth)?;
        let is_last_line = line_idx == total_lines.saturating_sub(1);
        let connector = if is_last_line && opaque_count == 0 {
            THIN_LAST
        } else {
            THIN_BRANCH
        };
        f.write_str(connector)?;
        att.try_display_fmt(f)?;
        try_writeln!(f)?;
    }

    if opaque_count > 0 {
        write_indent(f, continuing, sub_depth)?;
        f.write_str(THIN_LAST)?;
        let suffix = if opaque_count == 1 { "" } else { "s" };
        try_write!(f, "{}", opaque_count)?;
        f.write_str(" additional opaque attachment")?;
        try_write!(f, "{}", suffix)?;
        try_writeln!(f)?;
    }

    Ok(())
}
