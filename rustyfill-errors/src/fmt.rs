//! Formatting implementations for [`Report`](super::Report).
//!
//! Provides a human-readable [`Display`] implementation that walks all frames
//! in the report tree and renders them with indentation. The [`Debug`] impl
//! delegates to [`Display`] so that `{:#?}` and `{}` produce the same output.
//!
//! [`Display`]: core::fmt::Display
use core::error::Error;
use core::fmt;

use rustyfill::prelude::{TryString, TryVec};

use super::{FrameRef, Report};

/// Vertical connector repeated per depth level.
const VERT: &str = "│   ";
/// Branch connector for attachments.
const BRANCH: &str = "├─ ";
/// Arrow marking the start of a child frame subtree.
const CHILD_ARROW: &str = "──▶ ";

// ── Display ────────────────────────────────────────────────────────────────────

impl<C> fmt::Display for Report<C>
where
    C: Error + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Memoized indentation strings: index i holds the prefix for depth i.
        // Grown fallibly; on OOM we fall back to inline character writes.
        let mut indents: alloc::vec::Vec<Option<alloc::string::String>> = alloc::vec::Vec::new();

        /// Fallibly ensure an indent entry exists at `depth`. On allocation
        /// failure the slot stays `None` and the caller falls back to inline
        /// writes — which is fine since the formatter itself never panics.
        fn try_ensure_indent(
            indents: &mut alloc::vec::Vec<Option<alloc::string::String>>,
            depth: usize,
        ) {
            // Grow the vec fallibly.
            while indents.len() <= depth {
                if <alloc::vec::Vec<_> as TryVec<_>>::try_push(indents, None).is_err() {
                    return;
                }
            }
            // Build the string only if this slot hasn't been filled yet.
            if let Some(entry) = indents.get_mut(depth)
                && entry.is_none()
            {
                let needed = depth.saturating_mul(VERT.len());
                let mut buf =
                    match <alloc::string::String as TryString>::fallible_with_capacity(needed) {
                        Ok(b) => b,
                        Err(_) => return,
                    };
                for _ in 0..depth {
                    if buf.try_push_str(VERT).is_err() {
                        return;
                    }
                }
                *entry = Some(buf);
            }
        }

        /// Write the indentation prefix for the given depth. Uses the cached
        /// string when available, otherwise falls back to repeating `VERT`
        /// directly into the formatter (zero allocation).
        fn write_vert_indent(
            f: &mut fmt::Formatter<'_>,
            indents: &[Option<alloc::string::String>],
            depth: usize,
        ) -> fmt::Result {
            if let Some(Some(s)) = indents.get(depth) {
                f.write_str(s)
            } else {
                for _ in 0..depth {
                    f.write_str(VERT)?;
                }
                Ok(())
            }
        }

        let mut walker = self.frames();
        for (frame_result, depth) in walker.by_ref() {
            match frame_result {
                Ok(FrameRef::Static(sf)) => {
                    write_vert_indent(f, &indents, depth)?;
                    write!(f, "{}", sf.context())?;
                    if let Some(seg) = sf.context().segment() {
                        write!(f, " [{seg}]")?;
                    }
                    writeln!(f)?;
                    // Print attachments.
                    for att in sf.attachments() {
                        write_vert_indent(f, &indents, depth)?;
                        write!(f, "{BRANCH}{att:?}")?;
                        writeln!(f)?;
                    }
                    // Signal child subtree.
                    if !sf.children().is_empty() || sf.lost_children() > 0 {
                        write_vert_indent(f, &indents, depth)?;
                        writeln!(f, "{CHILD_ARROW}")?;
                    }
                    try_ensure_indent(&mut indents, depth.saturating_add(1));
                }
                Ok(FrameRef::Dynamic(df)) => {
                    write_vert_indent(f, &indents, depth)?;
                    write!(f, "{:?}", df.context_item())?;
                    writeln!(f)?;
                    for att in df.attachments() {
                        write_vert_indent(f, &indents, depth)?;
                        write!(f, "{BRANCH}{att:?}")?;
                        writeln!(f)?;
                    }
                    if !df.children().is_empty() || df.lost_children() > 0 {
                        write_vert_indent(f, &indents, depth)?;
                        writeln!(f, "{CHILD_ARROW}")?;
                    }
                    try_ensure_indent(&mut indents, depth.saturating_add(1));
                }
                Ok(FrameRef::LostFrames(n)) => {
                    write_vert_indent(f, &indents, depth)?;
                    write!(f, "<{} frame{} lost>", n, if n == 1 { "" } else { "s" })?;
                    writeln!(f)?;
                }
                Err(_) => {
                    write_vert_indent(f, &indents, depth)?;
                    writeln!(f, "<failed to display, out of memory>")?;
                }
            }
        }

        Ok(())
    }
}

// ── Debug ──────────────────────────────────────────────────────────────────────

impl<C> fmt::Debug for Report<C>
where
    C: Error + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
