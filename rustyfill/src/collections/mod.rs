//! Fallible collections.
//!
//! Includes both concurrent data structures (which require the `std` feature)
//! and standalone fallible containers like [`slotmap::TrySlotMap`].

#[cfg(feature = "std")]
pub mod chashmap;
#[cfg(feature = "std")]
pub mod interner;
pub mod slotmap;

#[cfg(feature = "std")]
pub use chashmap::{ConcurrentHashMap, ConcurrentHashMapError, ConcurrentHashMapNonblockError};
