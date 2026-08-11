//! Fallible error construction for well-known crates.
//!
//! Constructing errors from the standard library (e.g. `io::Error::new`) can
//! panic on allocation failure because it internally boxes the source error
//! into a `Box<dyn Error + Send + Sync>`. This module provides fallible
//! constructors that return [`Result`] so callers can handle out-of-memory
//! gracefully instead of crashing mid-recovery.

mod io;
mod std;

pub use io::IoErrorExt;
