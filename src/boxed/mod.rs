//! Fallible heap allocation for boxed values.
//!
//! Provides [`TryBox`], a trait implemented for `Box<T>` that mirrors standard
//! `Box` constructors but returns [`Result`] on allocation failure instead of
//! panicking. All non-allocating `Box` behaviour (dereferencing, dropping, DST
//! support) delegates to the standard library.

mod box_;

pub use box_::TryBox;
