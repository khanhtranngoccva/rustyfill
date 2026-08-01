//! Fallible atomically reference-counted smart pointer.
//!
//! Provides [`TryArc`], a fallible analogue of [`std::sync::Arc`] that returns
//! [`Result`] on allocation failure instead of panicking. Implemented directly
//! for `std::sync::Arc<T>` so all standard methods (`clone`, `downgrade`,
//! `strong_count`, etc.) are available without reimplementation.
//!
//! # Construction strategy
//!
//! Allocation is delegated to [`TryBox`](crate::boxed::TryBox) via a boxed
//! [`MaybeUninit<ArcInner<T>>`]. After initialising the strong/weak counters
//! and the data in place, ownership transfers to std's `Arc` through
//! [`Arc::from_raw`] — no second allocation is performed.

mod arc_;

pub use arc_::TryArc;
