//! Unified assertion wrapper for foreign types whose standard formatting
//! implementations are known to never implicitly allocate and panic.
//!
//! [`AssertFmt<T>`] replaces the family of individual assert wrappers
//! (`AssertDebug`, `AssertDisplay`, `AssertLowerHex`, …) with a single type
//! that delegates to whichever traits `T` happens to implement. Blanket impls
//! are gated on the presence of each trait, so the compiler rejects use of
//! `AssertFmt<T>` for a formatting mode that `T` doesn't support.
//!
//! # Why one type instead of many?
//!
//! The previous design required six distinct wrapper structs, each carrying its
//! own bound and constructor. Callers had to pick the right one per formatting
//! mode. A single `AssertFmt<T>` lets you wrap once and use everywhere — the
//! blanket impls activate automatically based on what `T` provides.
//!
//! # Example
//!
//! ```ignore
//! // Wrap a foreign type once…
//! let wrapped = AssertFmt(std::path::PathBuf::from("/tmp"));
//!
//! // …then use it in any supported formatting context:
//! write!(f, "{}", wrapped)?;   // Display — works because PathBuf: Display (via .display())
//! write!(f, "{:?}", wrapped)?; // Debug — works because PathBuf: Debug
//! ```

use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::{TryDebug, TryDisplay, TryLowerExp, TryLowerHex, TryUpperExp, TryUpperHex};
use lang_core::any::Any;
use lang_core::cmp;
use lang_core::error;
use lang_core::fmt;
use lang_core::hash;
use lang_core::ops::{Deref, DerefMut};

/// Assertion wrapper around a value whose standard formatting implementations
/// are guaranteed to never implicitly allocate and panic.
///
/// `AssertFmt<T>` is a transparent newtype: it dereferences to `T` in both
/// shared and mutable modes, forwards all standard formatting traits via
/// blanket impls gated on `T`'s capabilities, and additionally implements
/// the fallible `Try*` counterparts so it can be used in OOM-safe contexts.
///
/// Construction is infallible — the "assertion" is social, enforced by code
/// review. If `T`'s `Debug`/`Display` impl secretly allocates and panics under
/// memory pressure, callers of `TryDebug::try_fmt` / `TryDisplay::try_fmt` will
/// observe an abort rather than a graceful error.
pub struct AssertFmt<T>(pub T);

// ── Construction ────────────────────────────────────────────────────────────────

impl<T> AssertFmt<T> {
    /// Wrap a value, asserting that its standard formatting implementations
    /// never implicitly allocate and panic.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Returns a reference to the inner value.
    #[inline]
    #[must_use]
    pub fn get(&self) -> &T {
        &self.0
    }

    /// Returns a mutable reference to the inner value.
    #[inline]
    #[must_use]
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.0
    }

    /// Consumes the wrapper, returning the inner value.
    #[inline]
    #[must_use]
    pub fn into_inner(self) -> T {
        self.0
    }
}

// ── From / Into ─────────────────────────────────────────────────────────────────

impl<T> From<T> for AssertFmt<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self(value)
    }
}

// ── Deref / DerefMut ───────────────────────────────────────────────────────────

impl<T> Deref for AssertFmt<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for AssertFmt<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// ── Standard formatting traits (delegate to T) ─────────────────────────────────

impl<T: fmt::Debug> fmt::Debug for AssertFmt<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl<T: fmt::Display> fmt::Display for AssertFmt<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl<T: fmt::LowerHex> fmt::LowerHex for AssertFmt<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

impl<T: fmt::UpperHex> fmt::UpperHex for AssertFmt<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperHex::fmt(&self.0, f)
    }
}

impl<T: fmt::LowerExp> fmt::LowerExp for AssertFmt<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerExp::fmt(&self.0, f)
    }
}

impl<T: fmt::UpperExp> fmt::UpperExp for AssertFmt<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperExp::fmt(&self.0, f)
    }
}

impl<T: fmt::Octal> fmt::Octal for AssertFmt<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Octal::fmt(&self.0, f)
    }
}

impl<T: fmt::Binary> fmt::Binary for AssertFmt<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Binary::fmt(&self.0, f)
    }
}

impl<T: fmt::Pointer> fmt::Pointer for AssertFmt<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.0, f)
    }
}

// ── Fallible formatting traits (delegate to T's std impl) ──────────────────────

impl<T: fmt::Debug> TryDebug for AssertFmt<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl<T: fmt::Display> TryDisplay for AssertFmt<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl<T: fmt::LowerHex> TryLowerHex for AssertFmt<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

impl<T: fmt::UpperHex> TryUpperHex for AssertFmt<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperHex::fmt(&self.0, f)
    }
}

impl<T: fmt::LowerExp> TryLowerExp for AssertFmt<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerExp::fmt(&self.0, f)
    }
}

impl<T: fmt::UpperExp> TryUpperExp for AssertFmt<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperExp::fmt(&self.0, f)
    }
}

// ── Error trait ─────────────────────────────────────────────────────────────────

impl<T: error::Error + 'static> error::Error for AssertFmt<T> {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        self.0.source()
    }
}

// ── Clone / Copy / Default ──────────────────────────────────────────────────────

impl<T: Clone> Clone for AssertFmt<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Copy> Copy for AssertFmt<T> {}

impl<T: Default> Default for AssertFmt<T> {
    #[inline]
    fn default() -> Self {
        Self(Default::default())
    }
}

// ── TryClone / TryDefault ───────────────────────────────────────────────────────

impl<T: TryClone> TryClone for AssertFmt<T> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(Self(TryClone::try_clone(&self.0)?))
    }
}

impl<T: TryDefault> TryDefault for AssertFmt<T> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self(TryDefault::try_default()?))
    }
}

// ── PartialEq / Eq / PartialOrd / Ord / Hash ────────────────────────────────────

impl<T: PartialEq> PartialEq for AssertFmt<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T: Eq> Eq for AssertFmt<T> {}

impl<T: PartialOrd> PartialOrd for AssertFmt<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T: Ord> Ord for AssertFmt<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T: hash::Hash> hash::Hash for AssertFmt<T> {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

// ── Type introspection ──────────────────────────────────────────────────────────

impl<T: Any> AssertFmt<T> {
    /// Downcast to [`Any`] for runtime type inspection.
    #[inline]
    #[must_use]
    pub fn as_any(&self) -> &dyn Any {
        &self.0
    }

    /// Mutable downcast to [`Any`] for runtime type mutation.
    #[inline]
    #[must_use]
    pub fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.0
    }
}

// ── Individual assert wrappers ────────────────────────────────────────────────────
// Per-trait wrappers kept for ergonomics: each carries a hard bound on its type
// parameter so the compiler rejects misuse at the construction site.

/// Wraps any value that implements [`fmt::Debug`] so it can be used as a [`TryDebug`]
/// value. Use this for foreign types whose `Debug` implementation you have verified
/// never implicitly allocates (may allocate and return an error, but will not panic).
pub struct AssertDebug<T: fmt::Debug>(pub T);

impl<T: fmt::Debug> AssertDebug<T> {
    /// Create an assertion wrapper around a [`fmt::Debug`] value.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: fmt::Debug> Deref for AssertDebug<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: fmt::Debug> DerefMut for AssertDebug<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: fmt::Debug> fmt::Debug for AssertDebug<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl<T: fmt::Debug> TryDebug for AssertDebug<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

/// Wraps any value that implements [`fmt::Display`] so it can be used as a
/// [`TryDisplay`] value. Use this for foreign types whose `Display` implementation
/// you have verified never implicitly allocates (may allocate and return an error,
/// but will not panic).
pub struct AssertDisplay<T: fmt::Display>(pub T);

impl<T: fmt::Display> AssertDisplay<T> {
    /// Create an assertion wrapper around a [`fmt::Display`] value.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: fmt::Display> Deref for AssertDisplay<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: fmt::Display> DerefMut for AssertDisplay<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: fmt::Display> fmt::Display for AssertDisplay<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl<T: fmt::Display> TryDisplay for AssertDisplay<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// Wraps any value that implements [`fmt::LowerHex`] so it can be used as a
/// [`TryLowerHex`] value. Use this for foreign types whose `LowerHex` implementation
/// you have verified never implicitly allocates (may allocate and return an error,
/// but will not panic).
pub struct AssertLowerHex<T: fmt::LowerHex>(pub T);

impl<T: fmt::LowerHex> AssertLowerHex<T> {
    /// Create an assertion wrapper around a [`fmt::LowerHex`] value.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: fmt::LowerHex> Deref for AssertLowerHex<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: fmt::LowerHex> DerefMut for AssertLowerHex<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: fmt::LowerHex> fmt::LowerHex for AssertLowerHex<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

impl<T: fmt::LowerHex> TryLowerHex for AssertLowerHex<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

/// Wraps any value that implements [`fmt::UpperHex`] so it can be used as a
/// [`TryUpperHex`] value. Use this for foreign types whose `UpperHex` implementation
/// you have verified never implicitly allocates (may allocate and return an error,
/// but will not panic).
pub struct AssertUpperHex<T: fmt::UpperHex>(pub T);

impl<T: fmt::UpperHex> AssertUpperHex<T> {
    /// Create an assertion wrapper around a [`fmt::UpperHex`] value.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: fmt::UpperHex> Deref for AssertUpperHex<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: fmt::UpperHex> DerefMut for AssertUpperHex<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: fmt::UpperHex> fmt::UpperHex for AssertUpperHex<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperHex::fmt(&self.0, f)
    }
}

impl<T: fmt::UpperHex> TryUpperHex for AssertUpperHex<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperHex::fmt(&self.0, f)
    }
}

/// Wraps any value that implements [`fmt::LowerExp`] so it can be used as a
/// [`TryLowerExp`] value. Use this for foreign types whose `LowerExp` implementation
/// you have verified never implicitly allocates (may allocate and return an error,
/// but will not panic).
pub struct AssertLowerExp<T: fmt::LowerExp>(pub T);

impl<T: fmt::LowerExp> AssertLowerExp<T> {
    /// Create an assertion wrapper around a [`fmt::LowerExp`] value.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: fmt::LowerExp> Deref for AssertLowerExp<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: fmt::LowerExp> DerefMut for AssertLowerExp<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: fmt::LowerExp> fmt::LowerExp for AssertLowerExp<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerExp::fmt(&self.0, f)
    }
}

impl<T: fmt::LowerExp> TryLowerExp for AssertLowerExp<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerExp::fmt(&self.0, f)
    }
}

/// Wraps any value that implements [`fmt::UpperExp`] so it can be used as a
/// [`TryUpperExp`] value. Use this for foreign types whose `UpperExp` implementation
/// you have verified never implicitly allocates (may allocate and return an error,
/// but will not panic).
pub struct AssertUpperExp<T: fmt::UpperExp>(pub T);

impl<T: fmt::UpperExp> AssertUpperExp<T> {
    /// Create an assertion wrapper around a [`fmt::UpperExp`] value.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: fmt::UpperExp> Deref for AssertUpperExp<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: fmt::UpperExp> DerefMut for AssertUpperExp<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: fmt::UpperExp> fmt::UpperExp for AssertUpperExp<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperExp::fmt(&self.0, f)
    }
}

impl<T: fmt::UpperExp> TryUpperExp for AssertUpperExp<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperExp::fmt(&self.0, f)
    }
}
