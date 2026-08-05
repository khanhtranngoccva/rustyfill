//! Prelude module that re-exports every `Try*` trait.
//!
//! Import this module with a glob to bring all fallible-operation traits into
//! scope as if their methods were inherent on the standard types:
//!
//! ```
//! use fallibles::prelude::*;
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

// ── Foundational traits ──────────────────────────────────────────────────────

pub use crate::recovery::{Resumable, ResumableSource};
pub use crate::try_clone::TryClone;
pub use crate::try_default::TryDefault;
pub use crate::try_random_state::TryRandomState;
pub use crate::try_to_owned::TryToOwned;

// ── Box ──────────────────────────────────────────────────────────────────────

pub use crate::boxed::TryBox;

// ── Vec & slice ──────────────────────────────────────────────────────────────

pub use crate::vec::{TrySlice, TryVec};

// ── HashMap & HashSet ────────────────────────────────────────────────────────

pub use crate::hashmap::TryHashMap;
pub use crate::hashset::TryHashSet;

// ── VecDeque ─────────────────────────────────────────────────────────────────

pub use crate::vecdeque::TryVecDeque;

// ── String & str ─────────────────────────────────────────────────────────────

pub use crate::string::{TryStr, TryString};

// ── Arc & Weak ───────────────────────────────────────────────────────────────

pub use crate::arc::{TryArc, TryWeak};

// ── FFI strings ──────────────────────────────────────────────────────────────

pub use crate::ffi::{TryCString, TryOsStr, TryOsString};

// ── Paths ────────────────────────────────────────────────────────────────────

pub use crate::path::{TryPath, TryPathBuf};

// ── DashMap & DashSet ────────────────────────────────────────────────────────

pub use crate::dashmap::{TryDashMap, TryDashSet};
