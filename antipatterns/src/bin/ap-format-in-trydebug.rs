//! Antipattern: calling `format!()` inside a `TryDebug::try_fmt` implementation.
//!
//! # What happens
//!
//! Even though `TryDebug` is explicitly designed to be fallible and return an
//! error on allocation failure, using `format!()` inside the impl bypasses that
//! contract entirely. `format!()` allocates via the global allocator, which
//! returns null under OOM, triggering `abort()` before any `Err(fmt::Error)`
//! can be returned.
//!
//! This is insidious because the trait name *suggests* safety — "Try" implies
//! resilience — but the implementation undermines it with a hidden allocation.
//!
//! # How clippy responds
//!
//! Run `cargo clippy --bin ap-format-in-trydebug` to see what lints fire.

use std::fmt;

use rustyfill::try_fmt::{TryDebug, TryDebugWrapper};
use rustyfill_test_allocator::{FailAllocGuard, TestAllocator};

// Install the OOM-simulating test allocator so `FailAllocGuard` can intercept
// allocation calls made by this binary.
#[global_allocator]
static GLOBAL: TestAllocator = TestAllocator;

// ── Antipattern type ───────────────────────────────────────────────────────────

/// A struct whose `TryDebug` impl uses `format!()` internally.
struct BadTryDebug {
    host: &'static str,
    port: u16,
}

impl fmt::Debug for BadTryDebug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Required supertrait impl — also bad, mirrors the same pattern.
        let msg = format!("BadTryDebug(host={}, port={})", self.host, self.port);
        write!(f, "{}", msg)
    }
}

impl TryDebug for BadTryDebug {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // ANTI-PATTERN: Despite being a "fallible" Debug impl, format!() still
        // allocates and panics on OOM. The caller gets abort(), not Err(_).
        let msg = format!("BadTryDebug(host={}, port={})", self.host, self.port);
        write!(f, "{}", msg)
    }
}

// ── Correct version for comparison ─────────────────────────────────────────────

/// Same shape, writes directly to the formatter — no allocation.
struct GoodTryDebug {
    host: &'static str,
    port: u16,
}

impl fmt::Debug for GoodTryDebug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GoodTryDebug(host={}, port={})", self.host, self.port)
    }
}

impl TryDebug for GoodTryDebug {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // CORRECT: no intermediate String, Formatter handles everything on stack.
        write!(f, "GoodTryDebug(host={}, port={})", self.host, self.port)
    }
}

// ── Main ───────────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Antipattern: format!() in TryDebug::try_fmt ===\n");

    // Demonstrate the good path first (should survive OOM).
    println!("Testing GoodTryDebug (correct impl)...");
    let good = GoodTryDebug {
        host: "localhost",
        port: 8080,
    };
    let _guard = FailAllocGuard::fail_all();
    antipatterns::format_on_stack(format_args!("{good:?}"));
    antipatterns::format_on_stack(format_args!("{:?}", TryDebugWrapper(good)));
    drop(_guard);
    println!("  -> survived OOM, no implicit allocation.\n");

    // Now demonstrate the bad path (will abort under OOM).
    println!("Testing BadTryDebug (antipattern impl)...");
    println!("  This will abort the process if format!() implicitly allocates.");
    let bad = BadTryDebug {
        host: "localhost",
        port: 8080,
    };
    let _guard = FailAllocGuard::fail_all();
    antipatterns::format_on_stack(format_args!("{:?}", TryDebugWrapper(bad)));
    // If we reach here, the antipattern didn't allocate (unexpected).
    drop(_guard);
    println!("  -> survived OOM (UNEXPECTED — antipattern did not allocate)");
}
