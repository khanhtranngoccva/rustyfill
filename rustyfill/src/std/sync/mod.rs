//! Fallible synchronization primitives for `std::sync::Mutex` and
//! `std::sync::RwLock`, plus fallible debug formatting for both.
//!
//! [`TryMutex`] provides a fallible constructor for [`lang_std::sync::Mutex`]:
//! on platforms whose sys mutex lazily allocates its backing storage on first
//! lock (pthread), that allocation is hoisted into an explicit fallible
//! constructor so OOM can be recovered instead of aborting. On futex
//! platforms the construction is allocation-free and always succeeds.
//!
//! Both types also implement `Debug` in std (showing `<locked>` when
//! contention prevents inspection); the `TryDebug` impls delegate to the
//! inner value when readable, falling back to a non-allocating placeholder
//! otherwise.

mod mutex_;
mod rwlock_;

pub use mutex_::TryMutex;
