//! Fallible operations for `dashmap::DashMap` and `dashmap::DashSet`.
//!
//! Provides [`TryDashMap`] and [`TryDashSet`] traits that mirror common
//! construction and mutating operations but return [`Result`] values instead of
//! panicking on allocation failure. Uses the `raw-api` feature to lock and
//! operate on a single shard at a time.

mod dashmap_;
mod dashmap_extend;
mod dashset_;
mod dashset_extend;
pub mod mapref;

pub use dashmap_::{
    TryDashMap, TryDashMapConstructionError, TryDashMapEntryByRefError, TryDashMapError,
    TryDashMapInsertUniqueError, TryDashMapInsertUniqueNonblockError, TryDashMapNonblockError,
    TryDashMapWithCloneError,
};
pub use dashset_::{TryDashSet, TryDashSetConstructionError, TryDashSetWithCloneError};
