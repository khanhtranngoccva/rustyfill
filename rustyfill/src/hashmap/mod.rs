//! Fallible hash map operations.
//!
//! Provides [`TryHashMap`] for fallible `HashMap` construction, insertion,
//! extension, and capacity management — returning [`Result`] values instead of
//! panicking on allocation failure.

mod hashmap_;

pub use hashmap_::{TryHashMap, TryHashMapError};
