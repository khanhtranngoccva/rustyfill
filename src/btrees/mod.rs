//! Fallible B-tree map and set operations.
//!
//! Provides [`TryBTreeMap`] and [`TryBTreeSet`] for fallible B-tree collection
//! construction, insertion, extension, and capacity management — returning
//! [`Result`] values instead of panicking on allocation failure.
//!
//! These wrappers rely on [`std::panic::catch_unwind`] to intercept allocation
//! panics from B-tree internal node growth (since `BTreeMap::try_reserve` does
//! not exist). This feature is guarded behind the `"panic"` cargo feature and
//! requires that the crate be compiled with `panic = "unwind"`. The build script
//! will error if this feature is enabled but the panic strategy is `"abort"`.

mod btreemap_;
mod btreeset_;

pub use btreemap_::{TryBTreeMap, TryBTreeMapError};
pub use btreeset_::{TryBTreeSet, TryBTreeSetError};
