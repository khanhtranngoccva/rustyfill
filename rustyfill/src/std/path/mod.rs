//! Fallible path operations.
//!
//! Provides [`path_buf::TryPathBuf`] for fallible construction and mutation of
//! `PathBuf` values, [`path::TryPath`] for fallible `&Path`-to-`PathBuf`
//! conversions, plus [`TryClone`](crate::try_clone::TryClone) for `PathBuf`
//! and `&Path`, [`TryToOwned`](crate::try_to_owned::TryToOwned) for `Path`,
//! and [`TryDefault`](crate::try_default::TryDefault) for `PathBuf`.
//! All return [`Result`] on allocation failure instead of panicking.

#[allow(clippy::module_inception)]
mod path;
mod path_buf;

pub use path::{TryPath, TryPathError};
pub use path_buf::{TryPathBuf, TryPathBufError};
