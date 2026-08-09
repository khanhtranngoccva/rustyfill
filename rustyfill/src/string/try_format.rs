//! Fallible formatting into a new [`String`].
//!
//! The [`try_format`](crate::try_format) proc-macro mirrors the standard
//! [`format!`] macro but returns [`Result<String, TryReserveError>`] so that
//! allocation failures are handled gracefully instead of panicking.
//!
//! # Warning: Display implementations that allocate
//!
//! Some `Display` or `Debug` implementations in the standard library perform
//! hidden allocations (e.g. `Duration`, `SystemTime`, `PathBuf`, floating
//! point with precision specifiers). If such a value is formatted through
//! `try_format!` while using a constrained allocator, the allocation will
//! happen *inside* the formatter callback and will panic rather than return
//! an error. Run the `display-allocation-tests` binary crate (`cargo run -p
//! display-allocation-tests`) for a matrix of which types are safe under
//! zero-allocation conditions.
