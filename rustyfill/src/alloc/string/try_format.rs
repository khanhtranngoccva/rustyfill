//! Fallible formatting into a new [`String`].
//!
//! The [`try_format`](crate::try_format) proc-macro mirrors the standard
//! [`format!`] macro but returns [`Result<String, TryReserveError>`] so that
//! allocation failures are handled gracefully instead of panicking.
//!
//! # Warning: Display implementations that allocate
//!
//! Some `Display` or `Debug` implementations perform hidden allocations. If such a
//! value is formatted, the allocation will happen *inside* the formatter callback and will
//! panic or abort rather than return an error. Run the `display-allocation-tests` binary crate
//! (`cargo run -p display-allocation-tests`)
//! for a full matrix of which types are safe under zero-allocation conditions.
