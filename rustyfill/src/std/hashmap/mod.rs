//! Fallible hash map operations.
//!
//! Provides [`TryHashMap`] for fallible `HashMap` construction, insertion,
//! extension, and capacity management — returning [`Result`] values instead of
//! panicking on allocation failure.

mod hashmap_;
mod try_extend;

pub use hashmap_::{
    TryHashMap, TryHashMapConstructionError, TryHashMapInsertUniqueError,
    TryHashMapWithCloneError,
};
