//! Fallible B-tree map and set operations.
//!
//! Provides [`TryBTreeMap`] and [`TryBTreeSet`] for fallible B-tree collection
//! construction, insertion, extension, and capacity management — returning
//! [`Result`] values instead of panicking on allocation failure.

mod btreemap_;
mod btreeset_;

pub use btreemap_::{TryBTreeMap, TryBTreeMapError};
pub use btreeset_::{TryBTreeSet, TryBTreeSetError};
