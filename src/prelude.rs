//! Prelude module that re-exports every `Try*` trait and associated error type.
//!
//! Import this module with a glob to bring all fallible-operation traits into
//! scope as if their methods were inherent on the standard types:
//!
//! ```
//! use fallibles::prelude::*;
//!
//! // Now these all work without qualifying the trait name:
//! let s: Result<String, TryStrError> = "hello".try_to_string();
//! let v: Result<Vec<i32>, TryVecError> = [1i32, 2, 3].try_to_vec();
//! let boxed: Result<Box<i32>, AllocError> = Box::<i32>::fallible_new(42);
//!
//! assert_eq!(s.unwrap(), "hello");
//! assert_eq!(v.unwrap(), [1, 2, 3]);
//! assert_eq!(*boxed.unwrap(), 42);
//! ```

// ── Core allocation error ────────────────────────────────────────────────────

pub use crate::alloc::AllocError;

// ── Foundational traits ──────────────────────────────────────────────────────

pub use crate::try_clone::{TryClone, TryCloneError};
pub use crate::try_default::{TryDefault, TryDefaultError};
pub use crate::try_to_owned::{TryToOwned, TryToOwnedError};

// ── Box ──────────────────────────────────────────────────────────────────────

pub use crate::boxed::TryBox;

// ── Vec & slice ──────────────────────────────────────────────────────────────

pub use crate::vec::{TrySlice, TryVec, TryVecError};

// ── String & str ─────────────────────────────────────────────────────────────

pub use crate::string::{TryStr, TryStrError, TryString, TryStringError};

// ── Arc & Weak ───────────────────────────────────────────────────────────────

pub use crate::arc::{TryArc, TryUpgradeError, TryWeak};

// ── FFI strings ──────────────────────────────────────────────────────────────

pub use crate::ffi::{TryCString, TryCStringError, TryOsStr, TryOsStrError, TryOsString, TryOsStringError};

// ── Paths ────────────────────────────────────────────────────────────────────

pub use crate::path::{TryPath, TryPathBuf, TryPathBufError, TryPathError};
