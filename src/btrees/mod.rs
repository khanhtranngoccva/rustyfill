//! Fallible B-tree map and set operations.
//!
//! Provides [`TryBTreeMap`] and [`TryBTreeSet`] for fallible B-tree collection
//! construction, insertion, extension, and capacity management — returning
//! [`Result`] values instead of panicking on allocation failure.
//!
//! **Deprecated:** The `catch_unwind`-based approach cannot safely recover
//! elements on allocation failure. Prefer `HashMap`/`HashSet` with their
//! `Try*` counterparts which use proper `try_reserve`-based semantics.

mod btreemap_;
mod btreeset_;

#[allow(deprecated)]
pub use btreemap_::{TryBTreeMap, TryBTreeMapError};
#[allow(deprecated)]
pub use btreeset_::{TryBTreeSet, TryBTreeSetError};
