//! A concurrent hash map backed by sharded `RwLock<RawTable>` instances.
//!
//! The backing store is either a `Box<[Shard]>` (heap-allocated) or a
//! `&'static mut [Shard]`, allowing static initialization with zero allocations.
//!
//! All mutating operations are fallible: they return [`Result`] instead of panicking
//! on out-of-memory.
//!
//! # Construction
//!
//! Use [`declare_concurrent_hash_map!`] to declare a compile-time static map without
//! heap allocation. This macro generates a `once_cell::sync::Lazy` static that derefs
//! directly to the `ConcurrentHashMap`, so no intermediate accessor type is needed.

mod entry;
mod iter;
mod macros;
mod map;
mod refs;
mod shard;

pub use crate::declare_concurrent_hash_map;
pub use entry::{Entry, OccupiedEntry, VacantEntry};
pub use iter::{Iter, IterError, IterMut};
pub use map::{
    ConcurrentHashMap, ConcurrentHashMapError, ConcurrentHashMapNonblockError,
    TryConcurrentHashMapInsertUniqueError,
};
pub use refs::{Ref, RefMulti, RefMut, RefMutMulti};
pub use shard::Shard;

#[cfg(test)]
mod __test_static_map {
    crate::declare_concurrent_hash_map!(pub static TEST_CHASHMAP_STATIC: ConcurrentHashMap<i32, lang_alloc::string::String> = 4);
}
