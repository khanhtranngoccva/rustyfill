//! Our own entry and reference types for fallible DashMap operations.
//!
//! Mirrors the API of `dashmap::mapref` but is constructed by
//! [`TryDashMap`](super::TryDashMap) so that capacity is reserved *before*
//! insertion, guaranteeing that methods like [`VacantEntry::insert`] cannot
//! panic on out-of-memory.

pub mod entry;
mod ref_;
mod ref_mut;

pub use entry::{Entry, OccupiedEntry, VacantEntry};
pub use ref_::Ref;
pub use ref_mut::RefMut;
