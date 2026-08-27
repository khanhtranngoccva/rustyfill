//! Fallible single-threaded reference-counted smart pointer.
//!
//! Provides [`TryRc`], a fallible analogue of `std::rc::Rc` that returns
//! [`Result`] on allocation failure instead of panicking. Implemented directly
//! for `std::rc::Rc<T>` so all standard methods (`clone`, `downgrade`,
//! `strong_count`, etc.) are available without reimplementation.
//!
//! # Construction strategy
//!
//! Allocation is delegated to `crate::alloc::boxed::TryBox` via a boxed
//! `MaybeUninit<RcInner<T>>`. After initialising the strong/weak counters
//! and the data in place, ownership transfers to std's `Rc` through
//! `std::rc::Rc::from_raw` — no second allocation is performed.

mod rc_;

pub use rc_::{TryDowngradeError, TryRc, TryUpgradeError, TryWeak};
