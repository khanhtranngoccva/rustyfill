//! Shared helpers for antipattern demonstration binaries.
//!
//! Each binary in this crate demonstrates one specific antipattern. 
//! Under OOM conditions (all allocations blocked), the process aborts instead of 
//! returning an error — proving the risk.

use std::fmt::Write;

/// A `fmt::Write` target backed by a fixed-size stack array.
/// Writes that exceed capacity are silently truncated, avoiding any reallocation
/// and keeping the focus on whether the *formatter impl* allocates.
pub struct StackBuffer {
    buf: [u8; 4096],
    len: usize,
}

impl StackBuffer {
    #[inline]
    pub const fn new() -> Self {
        Self {
            buf: [0u8; 4096],
            len: 0,
        }
    }
}

impl Default for StackBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl Write for StackBuffer {
    #[inline]
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let end = self.len.saturating_add(s.len());
        let to_copy = end.min(self.buf.len()) - self.len;
        if to_copy > 0 {
            self.buf[self.len..self.len + to_copy].copy_from_slice(&s.as_bytes()[..to_copy]);
        }
        self.len = end.min(self.buf.len());
        Ok(())
    }
}

/// Format `args` into a stack buffer, exercising the Display/Debug impl without
/// allocating a `String` (which `format!()` would do). The allocation guard is
/// already active when this is called, so only the impl itself is tested.
#[inline]
pub fn format_on_stack(args: std::fmt::Arguments<'_>) {
    let mut buf = StackBuffer::new();
    let _ = write!(buf, "{args}");
}
