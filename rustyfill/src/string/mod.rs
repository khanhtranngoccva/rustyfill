//! Fallible `String` and `&str` operations.
//!
//! Provides [`string_::TryString`] for fallible construction and mutation of
//! `String` values, and [`str_::TryStr`] for fallible `&str`-to-`String`
//! conversions. Both return [`Result`] on allocation failure instead of
//! panicking.

mod str_;
mod string_;
mod try_format;

pub use str_::{TryStr, TryStrError};
pub use string_::{TryString, TryStringError};
