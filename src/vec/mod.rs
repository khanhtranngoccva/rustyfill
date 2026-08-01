//! Fallible vector and slice operations.
//!
//! Provides [`TryVec`] for fallible `Vec` mutations and [`TrySlice`] for
//! fallible slice-to-`Vec` conversions.

mod slice_;
mod vec_;

pub use slice_::TrySlice;
pub use vec_::TryVec;
