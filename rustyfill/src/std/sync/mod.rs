//! Fallible debug formatting for [`RwLock`] and [`Mutex`].
//!
//! Both types already implement `Debug` in std (showing `<locked>` when
//! contention prevents inspection). These `TryDebug` impls delegate to the
//! inner value when readable, falling back to a non-allocating placeholder
//! otherwise.

mod mutex_;
mod rwlock_;
