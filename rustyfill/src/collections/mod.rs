//! Fallible concurrent collections built on `hashbrown` shards.
//!
//! Provides [`ConcurrentHashMap`], a concurrent hash map whose backing store is
//! either a [`Box<[Shard]>`](ConcurrentHashMap) or a `&'static mut [Shard]`, letting
//! users declare maps in static variables and skip the heap allocation entirely.
//!
//! All mutating operations are fallible: they return [`Result`] instead of panicking
//! on out-of-memory.

pub mod chashmap;
pub mod interner;

pub use chashmap::{
    ConcurrentHashMap, ConcurrentHashMapError, ConcurrentHashMapNonblockError,
};
