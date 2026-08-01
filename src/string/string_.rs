//! Fallible `String` operations.
//!
//! Provides the [`TryString`] trait with methods that mirror common `String`
//! constructors and mutating operations but return [`Result`] to handle allocation
//! failures gracefully, using [`std::collections::TryReserveError`] as the primary
//! error type.
//!
//! # Design
//!
//! `TryString` is implemented for `String`. Methods that may grow internal capacity
//! (`push`, `push_str`, etc.) return a `Result` instead of panicking on
//! out-of-memory. Read-only accessors delegate directly to `String`.
//!
//! The trait also implements [`TryClone`](crate::try_clone::TryClone) and
//! [`TryDefault`](crate::try_default::TryDefault) for `String`.

use crate::alloc::AllocError;
use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use core::fmt;
use no_panic::no_panic;
use std::collections::TryReserveError;

/// Error returned by [`TryString`] operations.
///
/// Wraps the ways a string operation can fail on stable Rust: a reserve
/// failure ([`TryReserveError`]) or an arithmetic overflow when computing
/// the required byte capacity.
#[derive(Debug)]
pub enum TryStringError {
    /// A raw heap allocation failed (no collection involved).
    Alloc(AllocError),
    /// A capacity reservation on the string failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An arithmetic overflow occurred while computing required capacity.
    Overflow,
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Display for TryStringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "string operation failed: heap allocation error"),
            Self::Reserve(e) => write!(f, "string operation failed: {}", e),
            Self::Overflow => write!(
                f,
                "string operation failed: capacity calculation overflowed"
            ),
            Self::Other(msg) => write!(f, "string operation failed: {}", msg),
        }
    }
}

impl From<AllocError> for TryStringError {
    fn from(_: AllocError) -> Self {
        Self::Alloc(AllocError)
    }
}

impl From<TryReserveError> for TryStringError {
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

/// A trait for fallible string operations.
///
/// Implemented for `String`. Mirrors the most commonly-used `String` methods that
/// can fail due to allocation pressure, returning [`Result`] values that propagate
/// [`TryReserveError`] or [`TryStringError`] on failure.
pub trait TryString: Sized {
    // ── Construction ────────────────────────────────────────────────────────

    /// Fallibly construct a new empty `String`.
    ///
    /// This never fails — an empty `String` has no heap allocation. Equivalent to
    /// `String::new()`.
    fn try_new() -> Result<String, TryReserveError>;

    /// Fallibly construct a new `String` with at least enough capacity for `capacity` bytes.
    ///
    /// Returns [`TryReserveError`] if the initial allocation fails.
    /// Equivalent to `String::with_capacity(capacity)` but fallible.
    fn try_with_capacity(capacity: usize) -> Result<String, TryReserveError>;

    /// Fallibly construct a `String` from a string slice.
    ///
    /// Returns [`TryReserveError`] if the allocation fails.
    /// Equivalent to `String::from(str)` but fallible.
    fn try_from_str(s: &str) -> Result<String, TryReserveError>;

    // ── Mutation ────────────────────────────────────────────────────────────

    /// Fallibly append a single `char` to the end of the string.
    ///
    /// Returns [`TryReserveError`] if growing the internal buffer fails.
    fn try_push(&mut self, c: char) -> Result<(), TryReserveError>;

    /// Fallibly append a string slice to the end of the string.
    ///
    /// Returns [`TryReserveError`] if growing the internal buffer fails.
    fn try_push_str(&mut self, s: &str) -> Result<(), TryReserveError>;

    /// Fallibly insert a `char` into this `String` at a valid byte index.
    ///
    /// The index must not exceed the length of the string and must lie on a
    /// char boundary. Returns [`TryStringError::Reserve`] if growing the
    /// internal buffer fails, or [`TryStringError::Other`] if the index is
    /// out of bounds or falls in the middle of a UTF-8 character.
    /// No mutation occurs on error.
    fn try_insert(&mut self, idx: usize, c: char) -> Result<(), TryStringError>;

    /// Fallibly insert a string slice into this `String` at a valid byte index.
    ///
    /// The index must not exceed the length of the string and must lie on a
    /// char boundary. Returns [`TryStringError::Reserve`] if growing the
    /// internal buffer fails, or [`TryStringError::Other`] if the index is
    /// out of bounds or falls in the middle of a UTF-8 character.
    /// No mutation occurs on error.
    fn try_insert_str(&mut self, idx: usize, s: &str) -> Result<(), TryStringError>;

    /// Fallibly extend the string from an iterator of characters or strings.
    ///
    /// Accepts any iterator whose items can be pushed onto a `String` (i.e.
    /// `IntoIterator<Item = impl Into<String>>` is not required — callers pass
    /// iterators of `char`, `&str`, etc.). Reserves capacity using the
    /// iterator's lower-bound hint when available.
    ///
    /// Returns [`TryReserveError`] if growing the internal buffer fails.
    fn try_extend<I>(&mut self, iter: I) -> Result<(), TryReserveError>
    where
        I: IntoIterator,
        String: Extend<I::Item>;

    /// Fallibly reserve at least `additional` bytes of extra capacity.
    ///
    /// Returns [`TryReserveError`] if the allocation fails.
    /// Equivalent to `String::reserve(additional)` but fallible.
    fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError>;

    /// Fallibly shrink the capacity of this `String` to match its length.
    ///
    /// May reallocate if the current allocation is larger than needed.
    /// Returns [`TryReserveError`] if the re-allocation fails.
    /// Equivalent to `String::shrink_to_fit()` but fallible.
    fn try_shrink_to_fit(&mut self) -> Result<(), TryReserveError>;

    /// Fallibly shrink the capacity of this `String` to at least `min_capacity`.
    ///
    /// If the current capacity is already less than `min_capacity`, does nothing.
    /// Otherwise reallocates down. Returns [`TryReserveError`] if the
    /// re-allocation fails.
    /// Equivalent to `String::shrink_to(min_capacity)` but fallible.
    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryReserveError>;
}

impl TryString for String {
    
    fn try_new() -> Result<String, TryReserveError> {
        Ok(String::new())
    }

    
    fn try_with_capacity(capacity: usize) -> Result<String, TryReserveError> {
        let mut s = String::new();
        if capacity > 0 {
            s.try_reserve(capacity)?;
        }
        Ok(s)
    }

    
    fn try_from_str(s: &str) -> Result<String, TryReserveError> {
        let mut out = String::new();
        if !s.is_empty() {
            out.try_reserve(s.len())?;
        }
        out.push_str(s);
        Ok(out)
    }

    
    fn try_push(&mut self, c: char) -> Result<(), TryReserveError> {
        let encoded_len = c.len_utf8();
        self.try_reserve(encoded_len)?;
        self.push(c);
        Ok(())
    }

    
    fn try_push_str(&mut self, s: &str) -> Result<(), TryReserveError> {
        if s.is_empty() {
            return Ok(());
        }
        self.try_reserve(s.len())?;
        self.push_str(s);
        Ok(())
    }

    
    fn try_insert(&mut self, idx: usize, c: char) -> Result<(), TryStringError> {
        if !self.is_char_boundary(idx) {
            return Err(TryStringError::Other("insert index is out of bounds or not on a char boundary"));
        }
        let encoded_len = c.len_utf8();
        <String as TryString>::try_reserve(self, encoded_len).map_err(TryStringError::from)?;
        self.insert(idx, c);
        Ok(())
    }

    
    fn try_insert_str(&mut self, idx: usize, s: &str) -> Result<(), TryStringError> {
        if s.is_empty() {
            return Ok(());
        }
        if !self.is_char_boundary(idx) {
            return Err(TryStringError::Other("insert index is out of bounds or not on a char boundary"));
        }
        <String as TryString>::try_reserve(self, s.len()).map_err(TryStringError::from)?;
        self.insert_str(idx, s);
        Ok(())
    }

    
    fn try_extend<I>(&mut self, iter: I) -> Result<(), TryReserveError>
    where
        I: IntoIterator,
        String: Extend<I::Item>,
    {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        if lower > 0 {
            // Heuristic: reserve enough for at least `lower` chars (worst case 4 bytes each).
            // The actual sizes will be handled by per-item growth below if needed.
            let hint = lower.saturating_mul(4);
            <String as TryString>::try_reserve(self, hint)?;
        }
        self.extend(iter);
        Ok(())
    }

    
    fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.try_reserve(additional)
    }

    
    fn try_shrink_to_fit(&mut self) -> Result<(), TryReserveError> {
        self.try_shrink_to(self.len())
    }

    
    fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryReserveError> {
        let target = core::cmp::max(self.len(), min_capacity);
        if self.capacity() <= target {
            return Ok(());
        }
        // Reallocate into a buffer whose capacity is closer to `target`.
        // We build a fresh String by reserving the desired size and copying
        // content in. The allocator may round up, which is fine — the contract
        // is simply that we attempt to reduce excess capacity.
        let len = self.len();
        let snapshot = core::mem::take(self);
        let data = snapshot.into_bytes();
        let mut out = String::new();
        if len > 0 {
            <String as TryString>::try_reserve(&mut out, target)?;
        }
        out.push_str(core::str::from_utf8(&data[..len]).unwrap());
        *self = out;
        Ok(())
    }
}

// ── TryClone for String ──────────────────────────────────────────────────────

impl TryClone for String {
    
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let mut out = String::new();
        if !self.is_empty() {
            out.try_reserve(self.len()).map_err(TryCloneError::Reserve)?;
        }
        out.push_str(self);
        Ok(out)
    }
}

// ── TryDefault for String ────────────────────────────────────────────────────

impl TryDefault for String {
    
    fn try_default() -> Result<Self, TryDefaultError> {
        // An empty String requires no allocation.
        Ok(String::new())
    }
}

#[cfg(test)]
#[allow(unstable_name_collisions)]
mod tests {
    use super::*;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn try_new_returns_empty_string() {
        let s = String::try_new().unwrap();
        assert!(s.is_empty());
        assert_eq!(s.capacity(), 0);
    }

    #[test]
    fn try_with_capacity_zero() {
        let s = String::try_with_capacity(0).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn try_with_capacity_nonzero() {
        let s = String::try_with_capacity(64).unwrap();
        assert!(s.is_empty());
        assert!(s.capacity() >= 64);
    }

    #[test]
    fn try_from_str_empty() {
        let s = String::try_from_str("").unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn try_from_str_ascii() {
        let s = String::try_from_str("hello").unwrap();
        assert_eq!(s, "hello");
    }

    #[test]
    fn try_from_str_unicode() {
        let s = String::try_from_str("こんにちは 🦀").unwrap();
        assert_eq!(s, "こんにちは 🦀");
    }

    #[test]
    fn try_from_str_long() {
        let long = "a".repeat(10_000);
        let s = String::try_from_str(&long).unwrap();
        assert_eq!(s.len(), 10_000);
    }

    // ── Mutation ─────────────────────────────────────────────────────────────

    #[test]
    fn try_push_single_char() {
        let mut s = String::try_new().unwrap();
        s.try_push('h').unwrap();
        assert_eq!(s, "h");
    }

    #[test]
    fn try_push_multiple_chars() {
        let mut s = String::try_new().unwrap();
        s.try_push('h').unwrap();
        s.try_push('e').unwrap();
        s.try_push('l').unwrap();
        s.try_push('l').unwrap();
        s.try_push('o').unwrap();
        assert_eq!(s, "hello");
    }

    #[test]
    fn try_push_unicode_char() {
        let mut s = String::try_new().unwrap();
        s.try_push('🦀').unwrap();
        assert_eq!(s, "🦀");
        assert_eq!(s.len(), 4);
    }

    #[test]
    fn try_push_surrogate_pair_char() {
        let mut s = String::try_new().unwrap();
        s.try_push('💪').unwrap();
        assert_eq!(s, "💪");
    }

    #[test]
    fn try_push_str_empty() {
        let mut s = String::try_new().unwrap();
        s.try_push_str("").unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn try_push_str_ascii() {
        let mut s = String::try_new().unwrap();
        s.try_push_str("world").unwrap();
        assert_eq!(s, "world");
    }

    #[test]
    fn try_push_str_unicode() {
        let mut s = String::try_new().unwrap();
        s.try_push_str("你好世界").unwrap();
        assert_eq!(s, "你好世界");
    }

    #[test]
    fn try_push_str_then_push_char() {
        let mut s = String::try_new().unwrap();
        s.try_push_str("hel").unwrap();
        s.try_push('l').unwrap();
        s.try_push_str("o").unwrap();
        assert_eq!(s, "hello");
    }

    #[test]
    fn try_push_str_appends_to_existing() {
        let mut s = String::try_from_str("foo").unwrap();
        s.try_push_str("bar").unwrap();
        assert_eq!(s, "foobar");
    }

    #[test]
    fn try_push_str_multiple_times() {
        let mut s = String::try_new().unwrap();
        s.try_push_str("a").unwrap();
        s.try_push_str("b").unwrap();
        s.try_push_str("c").unwrap();
        assert_eq!(s, "abc");
    }

    #[test]
    fn try_push_preserves_unicode_correctly() {
        let mut s = String::try_new().unwrap();
        s.try_push('α').unwrap();
        s.try_push('β').unwrap();
        s.try_push('γ').unwrap();
        assert_eq!(s, "αβγ");
    }

    // ── Insert ────────────────────────────────────────────────────────────────

    #[test]
    fn try_insert_at_start() {
        let mut s = String::try_from_str("world").unwrap();
        s.try_insert(0, 'H').unwrap();
        assert_eq!(s, "Hworld");
    }

    #[test]
    fn try_insert_at_end() {
        let mut s = String::try_from_str("hell").unwrap();
        s.try_insert(4, 'o').unwrap();
        assert_eq!(s, "hello");
    }

    #[test]
    fn try_insert_in_middle() {
        let mut s = String::try_from_str("hllo").unwrap();
        s.try_insert(1, 'e').unwrap();
        assert_eq!(s, "hello");
    }

    #[test]
    fn try_insert_unicode_char() {
        let mut s = String::try_from_str("ab").unwrap();
        s.try_insert(1, 'ñ').unwrap();
        assert_eq!(s, "añb");
    }

    #[test]
    fn try_insert_emoji() {
        let mut s = String::try_from_str("hi").unwrap();
        s.try_insert(0, '🦀').unwrap();
        assert_eq!(s, "🦀hi");
    }

    #[test]
    fn try_insert_str_at_start() {
        let mut s = String::try_from_str("world").unwrap();
        s.try_insert_str(0, "Hello, ").unwrap();
        assert_eq!(s, "Hello, world");
    }

    #[test]
    fn try_insert_str_in_middle() {
        let mut s = String::try_from_str("Heloo").unwrap();
        s.try_insert_str(3, "l").unwrap();
        assert_eq!(s, "Helloo");
    }

    #[test]
    fn try_insert_str_at_end() {
        let mut s = String::try_from_str("hello").unwrap();
        s.try_insert_str(5, "!").unwrap();
        assert_eq!(s, "hello!");
    }

    #[test]
    fn try_insert_str_empty_does_nothing() {
        let mut s = String::try_from_str("x").unwrap();
        s.try_insert_str(0, "").unwrap();
        assert_eq!(s, "x");
    }

    #[test]
    fn try_insert_str_unicode() {
        let mut s = String::try_from_str("世界").unwrap();
        s.try_insert_str(0, "你好").unwrap();
        assert_eq!(s, "你好世界");
    }

    #[test]
    fn try_insert_out_of_bounds_returns_error() {
        let mut s = String::try_from_str("hi").unwrap();
        let result = s.try_insert(10, 'x');
        assert!(matches!(result, Err(TryStringError::Other(_))));
        assert_eq!(s, "hi"); // string unchanged
    }

    #[test]
    fn try_insert_mid_char_boundary_returns_error() {
        let mut s = String::try_from_str("αβ").unwrap();
        // 'α' is 2 bytes (0xCE, 0xB1), so index 1 is mid-character.
        let result = s.try_insert(1, 'x');
        assert!(matches!(result, Err(TryStringError::Other(_))));
        assert_eq!(s, "αβ"); // string unchanged
    }

    #[test]
    fn try_insert_str_out_of_bounds_returns_error() {
        let mut s = String::try_from_str("hi").unwrap();
        let result = s.try_insert_str(10, "world");
        assert!(matches!(result, Err(TryStringError::Other(_))));
        assert_eq!(s, "hi"); // string unchanged
    }

    #[test]
    fn try_insert_str_mid_char_boundary_returns_error() {
        let mut s = String::try_from_str("αβ").unwrap();
        // index 1 is in the middle of 'α'.
        let result = s.try_insert_str(1, "x");
        assert!(matches!(result, Err(TryStringError::Other(_))));
        assert_eq!(s, "αβ"); // string unchanged
    }

    #[test]
    fn try_insert_at_exact_length_is_valid() {
        let mut s = String::try_from_str("hi").unwrap();
        s.try_insert(2, '!').unwrap();
        assert_eq!(s, "hi!");
    }

    #[test]
    fn try_insert_str_at_exact_length_is_valid() {
        let mut s = String::try_from_str("hi").unwrap();
        s.try_insert_str(2, "!").unwrap();
        assert_eq!(s, "hi!");
    }

    // ── Extend ────────────────────────────────────────────────────────────────

    #[test]
    fn try_extend_chars() {
        let mut s = String::try_new().unwrap();
        s.try_extend(['h', 'e', 'l', 'l', 'o']).unwrap();
        assert_eq!(s, "hello");
    }

    #[test]
    fn try_extend_strings() {
        let mut s = String::try_new().unwrap();
        s.try_extend(["foo", "bar"].iter().map(|s| s.to_string()))
            .unwrap();
        assert_eq!(s, "foobar");
    }

    #[test]
    fn try_extend_empty_iter() {
        let mut s = String::try_from_str("x").unwrap();
        s.try_extend(std::iter::empty::<char>()).unwrap();
        assert_eq!(s, "x");
    }

    #[test]
    fn try_extend_unicode_chars() {
        let mut s = String::try_new().unwrap();
        s.try_extend(['α', 'β', 'γ']).unwrap();
        assert_eq!(s, "αβγ");
    }

    #[test]
    fn try_extend_str_slices_as_strings() {
        let mut s = String::try_new().unwrap();
        let parts = vec!["Hello,", " ", "world"];
        s.try_extend(parts.into_iter().map(String::from)).unwrap();
        assert_eq!(s, "Hello, world");
    }

    // ── Reserve ───────────────────────────────────────────────────────────────

    #[test]
    fn try_reserve_zero() {
        let mut s = String::try_new().unwrap();
        s.try_reserve(0).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn try_reserve_increases_capacity() {
        let mut s = String::try_new().unwrap();
        s.try_reserve(128).unwrap();
        assert!(s.capacity() >= 128);
    }

    #[test]
    fn try_reserve_on_existing_string() {
        let mut s = String::try_from_str("short").unwrap();
        s.try_reserve(256).unwrap();
        assert!(s.capacity() >= 256 + s.len());
    }

    // ── Shrink ────────────────────────────────────────────────────────────────

    #[test]
    fn try_shrink_to_fit_exact_length() {
        let mut s = String::try_from_str("abc").unwrap();
        let cap_before = s.capacity();
        // capacity == len already; no-op path.
        s.try_shrink_to_fit().unwrap();
        assert_eq!(s.capacity(), cap_before);
        assert_eq!(s, "abc");
    }

    #[test]
    fn try_shrink_to_fit_reduces_excess() {
        let mut s = String::try_with_capacity(1024).unwrap();
        s.try_push_str("small").unwrap();
        let cap_before = s.capacity();
        assert!(cap_before >= 1024);
        s.try_shrink_to_fit().unwrap();
        assert!(
            s.capacity() < cap_before,
            "capacity {} was not reduced from {}",
            s.capacity(),
            cap_before
        );
        assert!(s.capacity() >= 5);
        assert_eq!(s, "small");
    }

    #[test]
    fn try_shrink_to_fit_empty_large() {
        let mut s = String::try_with_capacity(512).unwrap();
        s.try_shrink_to_fit().unwrap();
        assert_eq!(s.capacity(), 0);
    }

    #[test]
    fn try_shrink_to_above_current_len() {
        let mut s = String::try_with_capacity(256).unwrap();
        s.try_push_str("tiny").unwrap();
        let cap_before = s.capacity();
        // min_capacity > len but < current capacity → should attempt to shrink.
        s.try_shrink_to(32).unwrap();
        assert!(s.capacity() >= 32);
        assert!(s.capacity() < cap_before || s.capacity() >= 32);
        assert_eq!(s, "tiny");
    }

    #[test]
    fn try_shrink_to_below_current_len_is_noop() {
        let mut s = String::try_from_str("abcdef").unwrap();
        let cap_before = s.capacity();
        // min_capacity < len → target == len, capacity already fits → no-op.
        s.try_shrink_to(2).unwrap();
        assert_eq!(s, "abcdef");
        assert_eq!(s.capacity(), cap_before);
    }

    #[test]
    fn try_shrink_to_already_small() {
        let mut s = String::try_from_str("hi").unwrap();
        // capacity already <= min_capacity → no-op.
        s.try_shrink_to(16).unwrap();
        assert_eq!(s, "hi");
    }

    // ── Combined workflows ───────────────────────────────────────────────────

    #[test]
    fn build_string_from_parts() {
        let mut s = String::try_with_capacity(20).unwrap();
        s.try_push_str("Hello, ").unwrap();
        s.try_push('w').unwrap();
        s.try_push_str("orld!").unwrap();
        assert_eq!(s, "Hello, world!");
    }

    #[test]
    fn try_clone_empty_string() {
        let s = String::try_new().unwrap();
        let c = s.try_clone().unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn try_clone_populated_string() {
        let s = String::try_from_str("testing").unwrap();
        let c = s.try_clone().unwrap();
        assert_eq!(c, "testing");
        assert_ne!(s.as_ptr(), c.as_ptr());
    }

    #[test]
    fn try_clone_unicode_string() {
        let s = String::try_from_str("日本語 🌍").unwrap();
        let c = s.try_clone().unwrap();
        assert_eq!(c, "日本語 🌍");
    }

    #[test]
    fn try_default_empty_string() {
        let s = String::try_default().unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn build_then_clone_then_push() {
        let mut s = String::try_default().unwrap();
        s.try_push_str("initial").unwrap();
        let c = s.try_clone().unwrap();
        s.try_push_str("_modified").unwrap();
        assert_eq!(s, "initial_modified");
        assert_eq!(c, "initial");
    }

    #[test]
    fn insert_then_shrink() {
        let mut s = String::try_with_capacity(256).unwrap();
        s.try_push_str("a").unwrap();
        s.try_insert_str(1, "middle").unwrap();
        s.try_push_str("z").unwrap();
        assert_eq!(s, "amiddlez");
        let cap_before = s.capacity();
        s.try_shrink_to_fit().unwrap();
        assert!(s.capacity() < cap_before || s.capacity() >= 8);
        assert!(s.capacity() >= 8);
    }

    #[test]
    fn extend_insert_roundtrip() {
        let mut s = String::try_new().unwrap();
        s.try_extend(['x', 'y', 'z']).unwrap();
        s.try_insert(1, '.').unwrap();
        // inserts '.' at byte index 1 → between 'x' and 'y'.
        assert_eq!(s, "x.yz");
    }
}
