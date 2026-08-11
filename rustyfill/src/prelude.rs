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
//! let v = <[i32] as TrySlice<i32>>::try_to_vec(&[1, 2, 3]).unwrap();
//! let boxed = Box::<i32>::fallible_new(42).unwrap();
//!
//! assert_eq!(s, "hello");
//! assert_eq!(v.as_slice(), &[1, 2, 3]);
//! assert_eq!(*boxed, 42);
//! ```

// ── Foundational traits (always available, no_std-compatible) ─────────────────

pub use crate::recovery::{Resumable, ResumableSource};
pub use crate::try_clone::TryClone;
pub use crate::try_default::TryDefault;
pub use crate::try_fmt::{TryDebug, TryDisplay, TryLowerHex, TryUpperHex};
pub use crate::try_to_owned::TryToOwned;

#[cfg(feature = "std")]
pub use crate::try_random_state::TryRandomState;

// ── Box (requires `std` feature for std::boxed wrappers) ──────────────────────

#[cfg(feature = "std")]
pub use crate::boxed::TryBox;

// ── Vec & slice (requires `std` feature) ──────────────────────────────────────

#[cfg(feature = "std")]
pub use crate::vec::{TrySlice, TryVec};

// ── HashMap & HashSet (requires `std` feature) ────────────────────────────────

#[cfg(feature = "std")]
pub use crate::hashmap::TryHashMap;
#[cfg(feature = "std")]
pub use crate::hashset::TryHashSet;

// ── VecDeque (requires `std` feature) ─────────────────────────────────────────

#[cfg(feature = "std")]
pub use crate::vecdeque::TryVecDeque;

// ── String & str (requires `std` feature) ─────────────────────────────────────

#[cfg(feature = "std")]
pub use crate::string::{TryStr, TryString};

// ── Arc & Weak (multi-threaded, requires `std` feature) ───────────────────────

#[cfg(feature = "std")]
pub use crate::arc::{TryArc, TryWeak as TryArcWeak};

// ── Rc & Weak (single-threaded, requires `std` feature) ──────────────────────

#[cfg(feature = "std")]
pub use crate::rc::{TryRc, TryWeak as TryRcWeak};

// ── FFI strings (requires `std` feature) ──────────────────────────────────────

#[cfg(feature = "std")]
pub use crate::ffi::{TryCString, TryOsStr, TryOsString};

// ── Paths (requires `std` feature) ────────────────────────────────────────────

#[cfg(feature = "std")]
pub use crate::path::{TryPath, TryPathBuf};

// ── DashMap & DashSet (unstable — see `unstable` feature) ────────────────────

#[cfg(feature = "unstable")]
pub use crate::dashmap::{TryDashMap, TryDashSet};

// ── BTreeMap & BTreeSet (deprecated — use the `scapegoat` crate) ─────────────

#[cfg(feature = "panic")]
#[allow(deprecated)]
pub use crate::btrees::{TryBTreeMap, TryBTreeSet};

// ── RefCell (requires `std` feature) ──────────────────────────────────────────

#[cfg(feature = "std")]
pub use crate::cell::TryRefCell;

// ── RwLock & Mutex (requires `std` feature) ───────────────────────────────────
// TryDebug impls delegate to std Debug (allocation-free, verified by OOM tests).
// See rustyfill::sync for the implementations.
