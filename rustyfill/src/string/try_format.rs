//! Fallible formatting into a new [`String`].
//!
//! The [`try_format`] macro mirrors the standard [`format!`] macro but returns
//! [`Result<String, TryReserveError>`] so that allocation failures are handled
//! gracefully instead of panicking.
//!
//! # Example
//!
//! ```
//! use rustyfill::prelude::TryString;
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     let result = rustyfill::try_format!("hello {}", 42)?;
//!     assert_eq!(result, "hello 42");
//!     Ok(())
//! }
//! ```
//!
//! # Warning: Display implementations that allocate
//!
//! Some `Display` or `Debug` implementations in the standard library perform
//! hidden allocations (e.g. `Duration`, `SystemTime`, `PathBuf`, floating
//! point with precision specifiers). If such a value is formatted through
//! `try_format!` while using a constrained allocator, the allocation will
//! happen *inside* the formatter callback and will panic rather than return
//! an error. Run the `display-allocation-tests` binary crate (`cargo run -p
//! display-allocation-tests`) for a matrix of which types are safe under
//! zero-allocation conditions.

/// Fallibly format arguments into a newly allocated [`String`].
///
/// Mirrors [`format!`] but returns [`Result<String, TryReserveError>`].
#[macro_export]
macro_rules! try_format {
    ($($arg:tt)+) => {{
        let mut buf = String::new();
        $crate::string::TryString::try_write_fmt(&mut buf, core::format_args!($($arg)+)).map(|()| buf)
    }};
}
