//! Fallible hash set operations.
//!
//! Provides [`TryHashSet`] for fallible `HashSet` construction, insertion,
//! extension, and capacity management — returning [`Result`] values instead of
//! panicking on allocation failure.

mod hashset_;
mod try_extend;

pub use hashset_::{TryHashSet, TryHashSetError, TryHashSetWithCloneError};
