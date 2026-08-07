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
//!    error type `C`, optional segment label, source [`Location`], attachments,
//!    and child frames from previous demotions.
//! 2. **Peers** (allocated, discardable): a [`VecDeque`] of additional
//!    [`StaticFrame`]s all sharing the same context type `C`. The deque can be
//!    optionally capped; oldest peers are evicted first when full.
//!
//! Each frame may carry arbitrary [`Box<dyn ItemImpl>`] attachments and a list
//! of [`DynamicFrame`] children created when a report is demoted during
//! [`change_context`](Report::change_context).
//!
//! [`error-stack`]: https://crates.io/crates/error-stack

#![no_std]
#![warn(missing_docs)]

extern crate alloc;

pub mod frame;
mod report;

pub use frame::{ContextFrame, DynamicFrame, ItemImpl, ItemKind, StaticFrame};
pub use report::{
    ChangeContextError, ChronoFrames, FrameRef, FrameRefMut, Frames, PeerIter, PeerIterMut, Report,
};
