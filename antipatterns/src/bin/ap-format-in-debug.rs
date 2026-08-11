//! Antipattern: calling `format!()` inside a `Debug::fmt` implementation.
//!
//! # What happens
//!
//! `format!()` allocates a `String` on the heap before writing to the formatter.
//! Under OOM (all allocations blocked), the allocator returns null, Rust's
//! `handle_alloc_error` calls `abort()`, and the process dies. The caller never
//! sees an error — they get a SIGABRT.
//!
//! This is the most common antipattern because `format!()` feels like "just
//! building a string" without considering that it goes through the global
//! allocator.
//!
//! # How clippy responds
//!
//! Run `cargo clippy --bin ap-format-in-debug` to see what lints fire.

use std::fmt;

use rustyfill_test_allocator::FailAllocGuard;

// ── Antipattern type ───────────────────────────────────────────────────────────

/// A struct whose `Debug` impl uses `format!()` to compose its output.
struct BadDebug {
    name: &'static str,
    value: i32,
}

impl fmt::Debug for BadDebug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // ANTI-PATTERN: format!() allocates a String on the heap.
        // Under OOM this panics → abort(), killing the entire process.
        let msg = format!("BadDebug(name={}, value={})", self.name, self.value);
        write!(f, "{}", msg)
    }
}

// ── Correct version for comparison ─────────────────────────────────────────────

/// Same shape, writes directly to the formatter — no allocation.
struct GoodDebug {
    name: &'static str,
    value: i32,
}

impl fmt::Debug for GoodDebug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // CORRECT: write!() delegates formatting to the Formatter's stack buffer.
        write!(f, "GoodDebug(name={}, value={})", self.name, self.value)
    }
}

// ── Main ───────────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Antipattern: format!() in Debug::fmt ===\n");

    // Demonstrate the good path first (should survive OOM).
    println!("Testing GoodDebug (correct impl)...");
    let good = GoodDebug { name: "test", value: 42 };
    let _guard = FailAllocGuard::fail_all();
    antipatterns::format_on_stack(format_args!("{good:?}"));
    drop(_guard);
    println!("  -> survived OOM, no implicit allocation.\n");

    // Now demonstrate the bad path (will abort under OOM).
    println!("Testing BadDebug (antipattern impl)...");
    println!("  This will abort the process if format!() implicitly allocates.");
    let bad = BadDebug { name: "test", value: 42 };
    let _guard = FailAllocGuard::fail_all();
    antipatterns::format_on_stack(format_args!("{bad:?}"));
    // If we reach here, the antipattern didn't allocate (unexpected).
    drop(_guard);
    println!("  -> survived OOM (UNEXPECTED — antipattern did not allocate)");
}
