//! Prelude module that re-exports every `Try*` trait.
//!
//! Import this module with a glob to bring all fallible-operation traits into
//! scope as if their methods were inherent on the standard types:
//!
//! ```
//! use rustyfill::prelude::*;
//!
//! // Now these all work without qualifying the trait name:
//! let s = <str as TryStr>::try_to_string("hello").unwrap();
//! let boxed = Box::<i32>::fallible_new(42).unwrap();
//!
//! assert_eq!(s, "hello");
//! assert_eq!(*boxed, 42);
//! ```
//!
//! With the `std` feature enabled, additional traits like `TrySlice`, `TryVec`,
//! `TryHashMap`, and `TryHashSet` are also available:
//!
//! ```rust,ignore
//! use rustyfill::prelude::*;
//!
//! let v = <[i32] as TrySlice<i32>>::try_to_vec(&[1, 2, 3]).unwrap();
//! assert_eq!(v.as_slice(), &[1, 2, 3]);
//! ```

// ── Foundational traits (always available, no_std-compatible) ─────────────────

/// Extension trait providing constructors and accessors for
/// [`crate::alloc::TryReserveError`]. Imported so that `new_alloc`,
/// `new_capacity_overflow`, `is_alloc`, and `is_capacity_overflow` are in scope
/// wherever the prelude is glob-imported.
pub use crate::alloc::TryReserveErrorExt;
pub use crate::recovery::{Resumable, ResumableSource, Stallable};
pub use crate::try_clone::TryClone;
pub use crate::try_default::TryDefault;
pub use crate::try_extend::{TryExtend, TryExtendFromSlice};
pub use crate::try_fmt::{TryDebug, TryDisplay, TryLowerHex, TryUpperHex};
pub use crate::try_to_owned::TryToOwned;

#[cfg(feature = "std")]
pub use crate::try_random_state::TryRandomState;

// ── Box ──────────────────────

pub use crate::alloc::boxed::TryBox;

// ── Vec & slice ──────────────────────────────────────

#[cfg(feature = "std")]
pub use crate::alloc::vec::{TrySlice, TryVec};

// ── HashMap & HashSet (requires `std` feature) ────────────────────────────────

#[cfg(feature = "std")]
pub use crate::std::hashmap::TryHashMap;
#[cfg(feature = "std")]
pub use crate::std::hashset::TryHashSet;

// ── VecDeque ─────────────────────────────────────────

pub use crate::alloc::vecdeque::TryVecDeque;

// ── String & str ─────────────────────────────────────

pub use crate::alloc::string::{TryStr, TryString};

// ── Arc & Weak ───────────────────────

pub use crate::alloc::arc::{TryArc, TryWeak as TryArcWeak};

// ── Rc & Weak ──────────────────────

pub use crate::alloc::rc::{TryRc, TryWeak as TryRcWeak};

// ── FFI strings ──────────────────────────────────────

pub use crate::alloc::ffi::{TryCString, TryCStringError};

#[cfg(feature = "std")]
pub use crate::std::ffi::{TryOsStr, TryOsString};

// ── Paths (requires `std` feature) ────────────────────────────────────────────

#[cfg(feature = "std")]
pub use crate::std::path::{TryPath, TryPathBuf};

// ── DashMap & DashSet (unstable — see `unstable` feature) ────────────────────

#[cfg(feature = "unstable")]
pub use crate::dashmap::{TryDashMap, TryDashSet};

// ── BTreeMap entry API (requires `std` feature) ───────────────────────────────

#[cfg(feature = "std")]
pub use crate::alloc::btrees::entry::{
    TryBTreeMap, TryBTreeMapEntry, TryBTreeMapEntryWithError, TryBTreeMapEntryWithGiveBackError,
    TryBTreeMapExtendFromSliceError, TryBTreeMapVacantEntry,
};

// ── Mutex (requires `std` feature) ─────────────────────

#[cfg(feature = "std")]
pub use crate::std::sync::TryMutex;

// ── RefCell ──────────────────────────────────────────

pub use crate::core::cell::TryRefCell;
