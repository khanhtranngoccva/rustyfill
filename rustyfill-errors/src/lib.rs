//! Context-aware error handling with frame stacks.
//!
//! `rustyfill-errors` is inspired by [`error-stack`] but redesigned for OOM
//! resilience — peers and children can be discarded if allocation fails while
//! building a report, so the head context is always available even under memory
//! pressure. Lost frames are tracked via counters on each frame and the report.
//!
//! ## Architecture
//!
//! A [`Report<C>`] stores error nodes in two regions:
//!
//! 1. **Head** (inline, no allocation): a [`StaticFrame`] carrying the current
//!    error type `C`, optional segment label, source `core::panic::Location`, attachments,
//!    and child frames from previous demotions.
//! 2. **Peers** (allocated, discardable): a `alloc::collections::VecDeque` of additional
//!    [`StaticFrame`]s all sharing the same context type `C`. The deque can be
//!    optionally capped; oldest peers are evicted first when full.
//!
//! Each frame may carry arbitrary `Box<dyn ItemImpl>` attachments and a list
//! of [`DynamicFrame`] children created when a report is demoted during
//! [`change_context`](Report::change_context).
//!
//! [`error-stack`]: https://crates.io/crates/error-stack

// On nightly, `rustyfill` may re-export the real `core::alloc::AllocError`
// (gated behind `feature(allocator_api)`). We enable the same feature to
// legally name that type in our public signatures. The cfg is emitted by
// build.rs as a simple boolean flag.
#![cfg_attr(nightly_compiler, feature(allocator_api))]
#![no_std]
#![warn(missing_docs)]
// Arithmetic in library code must never silently overflow; use checked/wrapping
// variants explicitly where wrap-around or saturation is intended.
#![deny(clippy::arithmetic_side_effects)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod fmt;
mod fmt_helpers;
pub mod frame;
mod report;
pub mod result_ext;

pub use frame::{ContextFrame, DynamicFrame, ItemImpl, ItemKind, StaticFrame};
pub use report::{
    ChangeContextError, ChronoFrames, FrameRef, FrameRefMut, Frames, PeerIter, PeerIterMut, Report,
};
pub use result_ext::ResultExt;

// ── Cross-platform path separator control (test support) ─────────────────────

#[cfg(feature = "std")]
pub(crate) use self::__force_forward_slashes::FORCE_FORWARD_SLASHES;

#[cfg(feature = "std")]
mod __force_forward_slashes {
    std::thread_local! {
        pub static FORCE_FORWARD_SLASHES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
}

/// Scoped guard that forces forward-slash path separators in location output.
///
/// On Windows, `Location::file()` returns paths with backslashes. Wrapping
/// `format!("{}", report)` inside this guard normalizes them to `/`, making
/// snapshot tests portable across platforms.
#[cfg(feature = "std")]
#[must_use]
pub struct ForceForwardSlashes {
    active: bool,
}

#[cfg(feature = "std")]
impl ForceForwardSlashes {
    /// Create a new guard, enabling forward-slash normalization.
    pub fn new() -> Self {
        FORCE_FORWARD_SLASHES.with(|flag| flag.set(true));
        Self { active: false }
    }
}

#[cfg(feature = "std")]
impl Drop for ForceForwardSlashes {
    fn drop(&mut self) {
        if !self.active {
            FORCE_FORWARD_SLASHES.with(|flag| flag.set(false));
            self.active = true;
        }
    }
}

#[cfg(feature = "std")]
impl Default for ForceForwardSlashes {
    fn default() -> Self {
        Self::new()
    }
}

// Install the OOM-simulating test allocator for this crate's own unit tests.
// Requires `std` because the failure-policy hooks are thread-local (std-only).
#[cfg(all(test, feature = "std"))]
#[global_allocator]
static GLOBAL: rustyfill_test_allocator::TestAllocator = rustyfill_test_allocator::TestAllocator;
