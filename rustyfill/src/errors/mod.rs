//! Fallible error construction for well-known crates.
//!
//! Constructing errors from the standard library (e.g. `io::Error::new`) can
//! panic on allocation failure because it internally boxes the source error
//! into a `Box<dyn Error + Send + Sync>`. This module provides fallible
//! constructors that return [`Result`] so callers can handle out-of-memory
//! gracefully instead of crashing mid-recovery.
//!
//! The [`core`] submodule provides `TryDebug` / `TryDisplay` implementations
//! for well-known `core` error types and is available in `no_std` environments.
//! The [`std`] submodule (gated behind the `std` feature) adds `IoErrorExt`
//! for fallible `std::io::Error` construction and impls for `std`-only error types.

mod core;
#[cfg(feature = "std")]
mod io;
#[cfg(feature = "std")]
mod std;

#[cfg(feature = "std")]
pub use io::IoErrorExt;
