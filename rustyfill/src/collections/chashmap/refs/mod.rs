//! Guard types that keep shard locks alive while providing references into the map.
//!
//! Two flavours exist:
//! - [`one`] — single-owner guards used by `get` / `get_mut` / Entry API
//! - [`multi`] — shared-ownership (Arc-backed) guards used by iterators

pub(crate) mod multi;
mod one;

pub use multi::{RefMulti, RefMutMulti};
pub use one::{Ref, RefMut};
