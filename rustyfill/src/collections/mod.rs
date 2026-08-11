//! Fallible concurrent collections.
//!
//! **Note:** The `collections` module requires the `std` feature, as concurrent
//! data structures depend on threading primitives (`parking_lot`, `crossbeam_utils`,
//! `once_cell::sync`).

#[cfg(feature = "std")]
pub mod chashmap;
#[cfg(feature = "std")]
pub mod interner;

#[cfg(feature = "std")]
pub use chashmap::{ConcurrentHashMap, ConcurrentHashMapError, ConcurrentHashMapNonblockError};
