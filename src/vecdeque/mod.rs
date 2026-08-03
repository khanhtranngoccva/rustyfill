//! Fallible double-ended queue operations.
//!
//! Provides [`TryVecDeque`] for fallible `VecDeque` construction, insertion,
//! extension, and capacity management — returning [`Result`] values instead of
//! panicking on allocation failure.

mod vecdeque_;

pub use vecdeque_::{TryVecDeque, TryVecDequeError};
