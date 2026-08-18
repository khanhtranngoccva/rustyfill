//! Formatting implementations for [`Report`](super::Report).
//!
//! Provides a human-readable [`Display`] implementation that walks all frames
//! in the report tree and renders them with tree-style connectors. The [`Debug`]
//! impl delegates to [`Display`] so that `{:#?}` and `{}` produce the same output.
//!
//! Rendering symbols are drawn from box-drawing Unicode characters, mapped to
//! match the visual style of [`error-stack`].
//!
//! [`error-stack`]: https://crates.io/crates/error-stack
use alloc::vec::Vec;
use rustyfill::{try_write, try_writeln};

use core::error::Error;
use core::fmt;

use rustyfill::prelude::{Stallable, TryVec};
use rustyfill::try_fmt::{TryDebug, TryDisplay};

use super::{FrameRef, Report};

use super::fmt_helpers::*;

// ── Display ────────────────────────────────────────────────────────────────────

impl<C> fmt::Display for Report<C>
where
    C: Error + TryDebug + TryDisplay + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut continuing_below: Vec<bool> = Vec::new();

        // Collect frames into a vec so we can look ahead/behind for sibling
        // counts and lost-frame markers. Use fallible collection so that OOM
        // during formatting doesn't panic — we degrade gracefully instead.
        // Auto-unstall so that each allocation error is emitted once and the
        // iterator continues rather than stalling indefinitely.
        let mut frames: Vec<_> = Vec::new();
        let walker = self.frames().with_auto_unstall(true);
        for item in walker {
            match <Vec<_> as TryVec<_>>::try_push(&mut frames, item) {
                Ok(()) => {}
                Err(_) => return try_write!(f, "<failed to render report, out of memory>"),
            }
        }
        let total = frames.len();

        let mut i = 0;
        while i < total {
            let (frame_result, depth) = &frames[i];

            match frame_result {
                Ok(FrameRef::Static(sf)) => {
                    let has_own_children = !sf.children().is_empty() || sf.lost_children() > 0;
                    let is_head = *depth == 0 && i == 0;

                    if is_head {
                        render_static_frame(
                            f,
                            sf,
                            &continuing_below,
                            has_own_children,
                            /* is_last — unused by render_static_frame */ false,
                        )?;
                        if has_own_children {
                            try_writeln!(f, "{}", SEP_CHAR)?;
                        }
                        extend_continuing(&mut continuing_below, 1).ok();
                    } else {
                        // Blank line before every peer, separating it from the
                        // preceding frame group.
                        try_writeln!(f)?;

                        // Check if a LostFrames marker immediately follows at
                        // depth 0 — that requires a branch connector on the
                        // location line.
                        // Safe: `i` is a valid frame index below `total`.
                        let next_i = i.checked_add(1).expect("frame index below total");
                        let lost_frames_follow = next_i < total
                            && matches!(
                                (&frames[next_i].0, &frames[next_i].1),
                                (Ok(FrameRef::LostFrames(_)), d) if *d == 0
                            );
                        let effective_has_children = has_own_children || lost_frames_follow;

                        render_static_frame(
                            f,
                            sf,
                            &continuing_below,
                            effective_has_children,
                            /* is_last — unused by render_static_frame */ true,
                        )?;
                        if has_own_children {
                            try_writeln!(f, "{}", SEP_CHAR)?;
                        }
                        extend_continuing(&mut continuing_below, 1).ok();
                    }
                }
                Ok(FrameRef::Dynamic(df)) => {
                    let siblings_before = frames[..i].iter().filter(|&(_, d)| *d == *depth).count();
                    let siblings_after = frames[i..]
                        .iter()
                        .skip(1)
                        .filter(|&(_, d)| *d == *depth)
                        .count();
                    let is_first = siblings_before == 0;
                    let is_last = siblings_after == 0;

                    extend_continuing(&mut continuing_below, *depth).ok();

                    // Mark terminated before rendering sub-items so deeper
                    // levels use space-only indentation.
                    if is_last || (is_first && is_last) {
                        mark_last_at(&mut continuing_below, *depth);
                    }

                    // --- Source header at depth d ---
                    write_indent(f, &continuing_below, *depth)?;

                    if is_first && !is_last {
                        f.write_str(SOURCE_FIRST)?;
                    } else if is_first && is_last {
                        f.write_str(SOURCE_EXCLUSIVE)?;
                    } else if !is_last {
                        f.write_str(" ")?;
                        f.write_str(SOURCE_MID)?;
                    } else {
                        f.write_str(" ")?;
                        f.write_str(SOURCE_LAST)?;
                    }

                    if TryDisplay::try_fmt(df.context_item(), f).is_err() {
                        // Formatting failed (OOM during display) — fall back to debug.
                        let _ = try_write!(f, "<failed to format context>");
                    }
                    try_writeln!(f)?;

                    // --- Sub-items at depth d+1 ---
                    // Safe: stack-trace depth is bounded well below usize::MAX.
                    let sub_depth = depth.checked_add(1).expect("trace depth fits in usize");

                    // Determine whether location is the last sub-item: it is
                    // last when there are no attachments and no children / lost
                    // children markers.
                    let atts = df.attachments();
                    let has_atts = !atts.is_empty();
                    let has_lost = df.lost_children() > 0 && df.children().is_empty();
                    let has_child_sources = !df.children().is_empty();
                    let loc_is_last = !has_atts && !has_lost && !has_child_sources;

                    write_indent(f, &continuing_below, sub_depth)?;
                    if loc_is_last {
                        f.write_str(THIN_LAST)?;
                    } else {
                        f.write_str(THIN_BRANCH)?;
                    }
                    df.context_item().write_location(f);
                    try_writeln!(f)?;

                    // Attachments.
                    if has_atts {
                        let (_, opaque_count) = count_attachments(atts);
                        let printable_count = atts.iter().filter(|a| a.is_printable()).count();
                        // Safe: small line-count sum, bounded by attachment count.
                        let remaining_after_loc: usize = printable_count
                            .saturating_add(if opaque_count > 0 { 1 } else { 0 })
                            .saturating_add(if has_lost { 1 } else { 0 });
                        let total_from_loc = remaining_after_loc.saturating_add(1); // 1 for location itself
                        render_attachments(
                            f,
                            &continuing_below,
                            sub_depth,
                            atts,
                            total_from_loc,
                            1, // location was line index 0
                        )?;
                    }

                    // Lost children marker.
                    if has_lost {
                        write_indent(f, &continuing_below, sub_depth)?;
                        f.write_str(THIN_LAST)?;
                        f.write_str("<")?;
                        try_write!(f, "{}", df.lost_children())?;
                        f.write_str(" frame")?;
                        let suffix = if df.lost_children() == 1 { "" } else { "s" };
                        try_write!(f, "{}", suffix)?;
                        f.write_str(" lost>")?;
                        try_writeln!(f)?;
                    }

                    // Separator between sibling source frames at the same depth.
                    // Only write it when there are no child sources interleaved;
                    // when child sources exist, the separator is deferred to the
                    // next sibling's source-header rendering below.
                    if !is_last && !has_child_sources {
                        write_sibling_separator(f, &continuing_below, *depth)?;
                    }
                    // Separator before nested child source frames — written at
                    // sub_depth so the bar aligns with child sub-items.
                    if has_child_sources {
                        write_indent(f, &continuing_below, sub_depth)?;
                        try_writeln!(f, "{}", SEP_CHAR)?;
                    }

                    extend_continuing(&mut continuing_below, sub_depth).ok();
                }
                Ok(FrameRef::LostFrames(n)) => {
                    extend_continuing(&mut continuing_below, *depth).ok();
                    write_indent(f, &continuing_below, *depth)?;
                    f.write_str(THIN_LAST)?;
                    f.write_str("<")?;
                    try_write!(f, "{}", n)?;
                    f.write_str(" frame")?;
                    let suffix = if *n == 1 { "" } else { "s" };
                    try_write!(f, "{}", suffix)?;
                    f.write_str(" lost>")?;
                    try_writeln!(f)?;
                }
                Err(_) => {
                    extend_continuing(&mut continuing_below, *depth).ok();
                    write_indent(f, &continuing_below, *depth)?;
                    f.write_str(THIN_LAST)?;
                    try_writeln!(f, "<failed to display, out of memory>")?;
                }
            }

            // Safe: `i` indexes into the frames slice, so it stays below its length.
            i = i.checked_add(1).expect("loop index below frame count");
        }

        Ok(())
    }
}

/// Render a static frame (head or peer) at depth 0 — no indent.
///
/// `has_children` indicates whether a separator + child subtree follows.
/// `is_last` indicates whether this is the final item overall (affects
/// connector choice for the location line).
fn render_static_frame<C>(
    f: &mut fmt::Formatter<'_>,
    sf: &crate::StaticFrame<C>,
    _continuing: &[bool],
    has_children: bool,
    _is_last: bool,
) -> fmt::Result
where
    C: core::fmt::Display + TryDisplay,
{
    // Use fallible Display so that OOM during formatting degrades gracefully
    // instead of panicking (e.g. on macOS where some Display impls implicitly
    // allocate and abort rather than return an error).
    if TryDisplay::try_fmt(sf.context(), f).is_err() {
        let _ = try_write!(f, "<failed to format context>");
    }
    if let Some(seg) = sf.context().segment() {
        try_write!(f, " [{}]", seg)?;
    }
    try_writeln!(f)?;

    let atts = sf.attachments();
    let has_atts = !atts.is_empty();

    // Location: ├╴ if attachments or children follow, ╰╴ otherwise.
    let loc_connector = if has_atts || has_children {
        THIN_BRANCH
    } else {
        THIN_LAST
    };

    let loc = sf.context().location();
    try_write!(f, "{}", loc_connector)?;
    super::fmt_helpers::write_location(f, loc)?;
    try_writeln!(f)?;

    // Attachments.
    if has_atts {
        let (_, opaque_count) = count_attachments(atts);
        let printable_count = atts.iter().filter(|a| a.is_printable()).count();
        // Total lines from location onward: 1 (loc) + attachments + possibly children.
        // Safe: small line-count sum, bounded by attachment count.
        let att_lines = printable_count.saturating_add(if opaque_count > 0 { 1 } else { 0 });
        let total_from_loc = att_lines.saturating_add(1);
        render_attachments(f, _continuing, 0, atts, total_from_loc, 1)?;
    }

    Ok(())
}

// ── Debug ──────────────────────────────────────────────────────────────────────

impl<C> fmt::Debug for Report<C>
where
    C: Error + TryDebug + TryDisplay + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

// ── TryDisplay / TryDebug ────────────────────────────────────────────────────

impl<C> TryDisplay for Report<C>
where
    C: Error + TryDebug + TryDisplay + Send + Sync + 'static,
{
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Delegate to Display — the rendering logic already handles OOM gracefully.
        fmt::Display::fmt(self, f)
    }
}

impl<C> TryDebug for Report<C>
where
    C: Error + TryDebug + TryDisplay + Send + Sync + 'static,
{
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Delegate to Debug which delegates to Display.
        fmt::Debug::fmt(self, f)
    }
}
