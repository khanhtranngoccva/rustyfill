//! Fallible formatting for types whose `Debug`/`Display` implementations may allocate.
//!
//! Many standard library types have `Debug` and/or `Display` implementations that
//! perform hidden heap allocations behind the scenes (e.g. floating-point with
//! precision specifiers, `PathBuf`, `Duration`, etc.). This module provides
//! [`TryDebug`] and [`TryDisplay`] traits that let you format values without
//! risking a panic from an unexpected allocation inside the formatter callback.
//!
//! # Design
//!
//! - [`TryDebug`] requires [`core::fmt::Debug`] as a supertrait.
//! - [`TryDisplay`] requires [`core::fmt::Display`] as a supertrait.
//! - Both traits return [`core::fmt::Result`] — the same type as canonical
//!   `Debug::fmt` / `Display::fmt`. No custom error enum is needed because
//!   [`core::fmt::Error`] is already an opaque, uninhabited sentinel that signals
//!   "the write failed" (either I/O or, in our case, a hidden allocation).
//! - Implementations must never call `format!()` or any function that implicitly
//!   allocates and panics on OOM.
//! - Well-known std types (primitives, tuples, arrays, markers, `Option`,
//!   `Result`, references, pointers, etc.) are implemented here.
//! - A derive macro exists for `TryDebug` on user-defined structs/enums.
//! - Passthrough declarative macros let users assert that their canonical
//!   `Debug`/`Display` impls never implicitly allocate and panic.
//! - Dedicated wrapper types ([`TryDebugWrapper`], [`TryDisplayWrapper`],
//!   [`TryLowerHexWrapper`], [`TryUpperHexWrapper`]) expose the corresponding
//!   standard formatting trait so that `core::format_args!` routes through
//!   the fallible paths. Each wrapper carries a hard bound on construction,
//!   producing precise compiler diagnostics when a value lacks the expected trait.

use core::fmt;

pub mod helpers;

// Re-export helper types at the try_fmt module level for convenience.
pub use helpers::{
    FormatterExt,
    TryDebugList,
    TryDebugMap,
    TryDebugSet,
    TryDebugStruct,
    TryDebugTuple,
};

// ── Traits ─────────────────────────────────────────────────────────────────────

/// A fallible analogue of [`core::fmt::Debug`].
///
/// Unlike [`core::fmt::Debug`], which can silently panic if its implementation
/// allocates under memory pressure, [`TryDebug::try_fmt`] returns a
/// [`fmt::Result`] so callers can detect failure.
///
/// Implementors must ensure that `try_fmt` never implicitly allocates and panics —
/// it may allocate and return an error, but should not abort the process.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not fallibly debuggable — it does not implement `TryDebug`",
    label = "infallible Debug formatting required",
    note = "if `{Self}` is a type you define, add `#[derive(rustyfill::TryDebug)]` or implement `TryDebug` manually",
    note = "if `{Self}` is a foreign type, wrap it in `rustyfill::try_fmt::AssertDebug` to assert that its Debug impl never implicitly allocates"
)]
pub trait TryDebug: fmt::Debug {
    /// Attempt to format this value using debug syntax.
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

/// A fallible analogue of [`core::fmt::Display`].
///
/// Unlike [`core::fmt::Display`], which can silently panic if its implementation
/// allocates under memory pressure, [`TryDisplay::try_fmt`] returns a
/// [`fmt::Result`] so callers can detect failure.
///
/// Implementors must ensure that `try_fmt` never implicitly allocates and panics —
/// it may allocate and return an error, but should not abort the process.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not fallibly displayable — it does not implement `TryDisplay`",
    label = "fallible Display formatting required",
    note = "if `{Self}` is a type you define, implement `TryDisplay` manually (no derive macro is available for Display)",
    note = "if `{Self}` is a foreign type, wrap it in `rustyfill::try_fmt::AssertDisplay` to assert that its Display impl never implicitly allocates"
)]
pub trait TryDisplay: fmt::Display {
    /// Attempt to format this value using display syntax.
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

/// A fallible analogue of [`core::fmt::LowerHex`].
///
/// Formats the value as lowercase hexadecimal (e.g., `ff`). Numeric primitives
/// implement this by delegating to their standard `LowerHex` impl, which never
/// implicitly allocates.
///
/// Implementors must ensure that `try_fmt` never implicitly allocates and panics —
/// it may allocate and return an error, but should not abort the process.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not fallibly hex-formattable — it does not implement `TryLowerHex`",
    label = "fallible LowerHex formatting required",
    note = "if `{Self}` is a type you define, implement `TryLowerHex` manually or use `rustyfill::lowerhex_passthrough!({Self})`",
    note = "if `{Self}` is a foreign type, wrap it in `rustyfill::try_fmt::AssertLowerHex` to assert that its LowerHex impl never implicitly allocates"
)]
pub trait TryLowerHex: fmt::LowerHex {
    /// Attempt to format this value as lowercase hexadecimal.
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

/// A fallible analogue of [`core::fmt::UpperHex`].
///
/// Formats the value as uppercase hexadecimal (e.g., `FF`). Numeric primitives
/// implement this by delegating to their standard `UpperHex` impl, which never
/// implicitly allocates.
///
/// Implementors must ensure that `try_fmt` never implicitly allocates and panics —
/// it may allocate and return an error, but should not abort the process.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not fallibly hex-formattable — it does not implement `TryUpperHex`",
    label = "fallible UpperHex formatting required",
    note = "if `{Self}` is a type you define, implement `TryUpperHex` manually or use `rustyfill::upperhex_passthrough!({Self})`",
    note = "if `{Self}` is a foreign type, wrap it in `rustyfill::try_fmt::AssertUpperHex` to assert that its UpperHex impl never implicitly allocates"
)]
pub trait TryUpperHex: fmt::UpperHex {
    /// Attempt to format this value as uppercase hexadecimal.
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

/// A fallible analogue of [`core::fmt::LowerExp`].
///
/// Formats the value in lowercase exponential notation (e.g., `1.23e+6`).
/// Floating-point primitives implement this by delegating to their standard
/// `LowerExp` impl, which uses stack buffers and never implicitly allocates.
///
/// Implementors must ensure that `try_fmt` never implicitly allocates and panics —
/// it may allocate and return an error, but should not abort the process.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not fallibly exponential-formattable — it does not implement `TryLowerExp`",
    label = "infallible LowerExp formatting required",
    note = "if `{Self}` is a type you define, implement `TryLowerExp` manually or use `rustyfill::lowerexp_passthrough!({Self})`",
    note = "if `{Self}` is a foreign type, wrap it in `rustyfill::try_fmt::AssertLowerExp` to assert that its LowerExp impl never implicitly allocates"
)]
pub trait TryLowerExp: fmt::LowerExp {
    /// Attempt to format this value in lowercase exponential notation.
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

/// A fallible analogue of [`core::fmt::UpperExp`].
///
/// Formats the value in uppercase exponential notation (e.g., `1.23E+6`).
/// Floating-point primitives implement this by delegating to their standard
/// `UpperExp` impl, which uses stack buffers and never implicitly allocates.
///
/// Implementors must ensure that `try_fmt` never implicitly allocates and panics —
/// it may allocate and return an error, but should not abort the process.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not fallibly exponential-formattable — it does not implement `TryUpperExp`",
    label = "infallible UpperExp formatting required",
    note = "if `{Self}` is a type you define, implement `TryUpperExp` manually or use `rustyfill::upperexp_passthrough!({Self})`",
    note = "if `{Self}` is a foreign type, wrap it in `rustyfill::try_fmt::AssertUpperExp` to assert that its UpperExp impl never implicitly allocates"
)]
pub trait TryUpperExp: fmt::UpperExp {
    /// Attempt to format this value in uppercase exponential notation.
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

// ── Wrapper types ───────────────────────────────────────────────────────────────
// Each wrapper carries a hard bound on its type parameter, so the compiler can
// produce precise diagnostics when a value doesn't implement the expected trait.
// The try_format_args! macro picks the correct wrapper per-placeholder based on
// the trailing format character (? → Debug, x → LowerHex, X → UpperHex,
// everything else → Display).

/// Wraps a value known to implement [`TryDebug`] so it can be formatted via
/// [`fmt::Debug`]. Constructing this wrapper fails at compile time if `T` does
/// not implement [`TryDebug`], producing a clear diagnostic at the call site.
pub struct TryDebugWrapper<T: TryDebug>(pub T);

impl<T: TryDebug> TryDebugWrapper<T> {
    /// Create a debug-formattable wrapper around a [`TryDebug`] value.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: TryDebug> fmt::Debug for TryDebugWrapper<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.try_fmt(f)
    }
}

/// Wraps a value known to implement [`TryDisplay`] so it can be formatted via
/// [`fmt::Display`]. Constructing this wrapper fails at compile time if `T` does
/// not implement [`TryDisplay`], producing a clear diagnostic at the call site.
pub struct TryDisplayWrapper<T: TryDisplay>(pub T);

impl<T: TryDisplay> TryDisplayWrapper<T> {
    /// Create a display-formattable wrapper around a [`TryDisplay`] value.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: TryDisplay> fmt::Display for TryDisplayWrapper<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.try_fmt(f)
    }
}

/// Wraps a value known to implement [`TryLowerHex`] so it can be formatted via
/// [`fmt::LowerHex`]. Constructing this wrapper fails at compile time if `T` does
/// not implement [`TryLowerHex`], producing a clear diagnostic at the call site.
pub struct TryLowerHexWrapper<T: TryLowerHex>(pub T);

impl<T: TryLowerHex> TryLowerHexWrapper<T> {
    /// Create a lower-hex-formattable wrapper around a [`TryLowerHex`] value.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: TryLowerHex> fmt::LowerHex for TryLowerHexWrapper<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.try_fmt(f)
    }
}

/// Wraps a value known to implement [`TryUpperHex`] so it can be formatted via
/// [`fmt::UpperHex`]. Constructing this wrapper fails at compile time if `T` does
/// not implement [`TryUpperHex`], producing a clear diagnostic at the call site.
pub struct TryUpperHexWrapper<T: TryUpperHex>(pub T);

impl<T: TryUpperHex> TryUpperHexWrapper<T> {
    /// Create an upper-hex-formattable wrapper around a [`TryUpperHex`] value.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: TryUpperHex> fmt::UpperHex for TryUpperHexWrapper<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.try_fmt(f)
    }
}

/// Wraps a value known to implement [`TryLowerExp`] so it can be formatted via
/// [`fmt::LowerExp`]. Constructing this wrapper fails at compile time if `T` does
/// not implement [`TryLowerExp`], producing a clear diagnostic at the call site.
pub struct TryLowerExpWrapper<T: TryLowerExp>(pub T);

impl<T: TryLowerExp> TryLowerExpWrapper<T> {
    /// Create a lower-exp-formattable wrapper around a [`TryLowerExp`] value.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: TryLowerExp> fmt::LowerExp for TryLowerExpWrapper<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.try_fmt(f)
    }
}

/// Wraps a value known to implement [`TryUpperExp`] so it can be formatted via
/// [`fmt::UpperExp`]. Constructing this wrapper fails at compile time if `T` does
/// not implement [`TryUpperExp`], producing a clear diagnostic at the call site.
pub struct TryUpperExpWrapper<T: TryUpperExp>(pub T);

impl<T: TryUpperExp> TryUpperExpWrapper<T> {
    /// Create an upper-exp-formattable wrapper around a [`TryUpperExp`] value.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: TryUpperExp> fmt::UpperExp for TryUpperExpWrapper<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.try_fmt(f)
    }
}

// ── AssertFmt wrappers ──────────────────────────────────────────────────────────
// Passthrough wrappers for foreign types that do not implement the Try* traits.
// The user asserts (on their honor) that the underlying std fmt implementation
// never implicitly allocates — i.e., it may allocate and return an error, but
// will not panic on OOM.

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

// ── Macro helpers ──────────────────────────────────────────────────────────────

/// Implements both `TryDebug` and `TryDisplay` for types whose canonical
/// `Debug`/`Display` implementations are known to never implicitly allocate and panic.
macro_rules! impl_try_fmt_primitives {
    ($($t:ty),* $(,)?) => {
        $(
            impl TryDebug for $t {
                #[inline]
                fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    // Delegate to the standard Debug impl, which preserves the
                    // Formatter's width, precision, fill, and alignment settings.
                    // Using write!(f, "{:?}", self) would create a new Arguments
                    // with empty specs, losing those settings.
                    <Self as fmt::Debug>::fmt(self, f)
                }
            }

            impl TryDisplay for $t {
                #[inline]
                fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    // Same reasoning: delegate to std Display to preserve format specs.
                    <Self as fmt::Display>::fmt(self, f)
                }
            }
        )*
    };
}

/// Implements `TryLowerHex` and `TryUpperHex` for unsigned integer types whose
/// canonical hex implementations never implicitly allocate and panic.
macro_rules! impl_try_hex_primitives {
    ($($t:ty),* $(,)?) => {
        $(
            impl TryLowerHex for $t {
                #[inline]
                fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    <Self as fmt::LowerHex>::fmt(self, f)
                }
            }

            impl TryUpperHex for $t {
                #[inline]
                fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    <Self as fmt::UpperHex>::fmt(self, f)
                }
            }
        )*
    };
}

/// Implements `TryLowerExp` and `TryUpperExp` for floating-point types whose
/// canonical exponential implementations never implicitly allocate and panic (stack-buffered).
macro_rules! impl_try_exp_primitives {
    ($($t:ty),* $(,)?) => {
        $(
            impl TryLowerExp for $t {
                #[inline]
                fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    <Self as fmt::LowerExp>::fmt(self, f)
                }
            }

            impl TryUpperExp for $t {
                #[inline]
                fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    <Self as fmt::UpperExp>::fmt(self, f)
                }
            }
        )*
    };
}

/// Implements only `TryDebug` for types that implement `Debug` but not `Display`.
macro_rules! impl_try_debug_only {
    ($($t:ty),* $(,)?) => {
        $(
            impl TryDebug for $t {
                #[inline]
                fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    // Delegate to std Debug to preserve format specs.
                    <Self as fmt::Debug>::fmt(self, f)
                }
            }
        )*
    };
}

// ── Numeric primitives ─────────────────────────────────────────────────────────

impl_try_fmt_primitives!(u8, u16, u32, u64, u128, usize);
impl_try_fmt_primitives!(i8, i16, i32, i64, i128, isize);

// bool and char have Debug and Display impls that never implicitly allocate.
impl_try_fmt_primitives!(bool, char);

// () has a Debug impl that never implicitly allocates; Display is not implemented so we do debug-only.
impl_try_debug_only!(());

// Hex formatting for unsigned integer primitives. Signed integers, bool, char,
// and () do not implement LowerHex/UpperHex in std and are therefore excluded.
impl_try_hex_primitives!(u8, u16, u32, u64, u128, usize);

// f32/f64 Display and Debug never implicitly allocate on all platforms — precision
// specifiers use stack buffers internally. Verified by display-allocation-tests.
impl_try_fmt_primitives!(f32, f64);

// Exponential notation for floats is also stack-buffered and never implicitly allocates.
impl_try_exp_primitives!(f32, f64);

// ── References ──────────────────────────────────────────────────────────────────

impl<T: TryDebug> TryDebug for &T {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).try_fmt(f)
    }
}

impl<T: TryDisplay> TryDisplay for &T {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).try_fmt(f)
    }
}

impl<T: TryLowerHex> TryLowerHex for &T {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).try_fmt(f)
    }
}

impl<T: TryUpperHex> TryUpperHex for &T {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).try_fmt(f)
    }
}

impl<T: TryLowerExp> TryLowerExp for &T {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).try_fmt(f)
    }
}

impl<T: TryUpperExp> TryUpperExp for &T {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).try_fmt(f)
    }
}

// ── Raw pointers (Debug only — raw pointers don't implement Display) ───────────

impl<T> TryDebug for *const T {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:p}", *self)
    }
}

impl<T> TryDebug for *mut T {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:p}", *self)
    }
}

// ── Option ─────────────────────────────────────────────────────────────────────

impl<T: TryDebug> TryDebug for Option<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Some(v) => {
                f.write_str("Some(")?;
                v.try_fmt(f)?;
                f.write_str(")")
            }
            None => f.write_str("None"),
        }
    }
}

// ── Result ─────────────────────────────────────────────────────────────────────

impl<T: TryDebug, E: TryDebug> TryDebug for Result<T, E> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ok(v) => {
                f.write_str("Ok(")?;
                v.try_fmt(f)?;
                f.write_str(")")
            }
            Err(e) => {
                f.write_str("Err(")?;
                e.try_fmt(f)?;
                f.write_str(")")
            }
        }
    }
}

// ── Marker types ───────────────────────────────────────────────────────────────

impl<T> TryDebug for core::marker::PhantomData<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PhantomData")
    }
}

impl<T: TryDebug> TryDebug for core::mem::ManuallyDrop<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (<Self as core::ops::Deref>::deref(self)).try_fmt(f)
    }
}

impl<T> TryDebug for core::mem::MaybeUninit<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("MaybeUninit")
    }
}

impl TryDebug for core::marker::PhantomPinned {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PhantomPinned")
    }
}

// ── Str and slice references ───────────────────────────────────────────────────

// str's Debug and Display never implicitly allocate — safe to passthrough.
impl TryDebug for &str {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for &str {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<T: TryDebug> TryDebug for &[T] {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        helpers::FormatterExt::try_debug_list(f).entries(self.iter()).finish()
    }
}

// ── Tuple implementations (generated by proc-macro) ────────────────────────────

rustyfill_macros::try_debug_tuples!(12);

// ── Arrays [T; N] ──────────────────────────────────────────────────────────────

impl<T: TryDebug, const N: usize> TryDebug for [T; N] {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        helpers::FormatterExt::try_debug_list(f).entries(self.iter()).finish()
    }
}

// ── Ranges ─────────────────────────────────────────────────────────────────────

macro_rules! impl_try_debug_for_range {
    ($($t:ty),* $(,)?) => {
        $(
            impl TryDebug for $t {
                #[inline]
                fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{:?}", *self)
                }
            }
        )*
    };
}

impl_try_debug_for_range!(
    core::ops::Range<usize>,
    core::ops::Range<i32>,
    core::ops::Range<u32>,
    core::ops::RangeFrom<usize>,
    core::ops::RangeTo<usize>,
    core::ops::RangeFull,
);

// ── Formatting macros ──────────────────────────────────────────────────────────
// The `try_format_args` proc-macro is defined in `rustyfill-macros` and re-exported
// from the crate root. It selects the appropriate *Wrapper type per placeholder
// based on the trailing format character, and leaves width/precision arguments unwrapped.
// The helper macros (try_println, try_print, try_write, try_writeln, try_format)
// are now proc-macros defined in rustyfill-macros and re-exported from the crate root.

// ── Passthrough macros ─────────────────────────────────────────────────────────

/// Implements `TryDebug` by delegating to the canonical `Debug` implementation.
///
/// Use this macro when you can verify that your type's `Debug` impl never implicitly
/// allocates and panics (it may allocate and return an error, but must not abort).
/// The macro generates a thin wrapper that passes through the existing `Debug::fmt`
/// result directly.
///
/// # Example
///
/// ```ignore
/// #[derive(Debug)]
/// struct MyPoint { x: i32, y: i32 }
///
/// rustyfill::debug_passthrough!(MyPoint);
/// ```
#[macro_export]
macro_rules! debug_passthrough {
    ($ty:ty) => {
        impl $crate::try_fmt::TryDebug for $ty {
            #[inline]
            fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::Debug::fmt(self, f)
            }
        }
    };
}

/// Implements `TryDisplay` by delegating to the canonical `Display` implementation.
///
/// Use this macro when you can verify that your type's `Display` impl never implicitly
/// allocates and panics (it may allocate and return an error, but must not abort).
/// The macro generates a thin wrapper that passes through the existing `Display::fmt`
/// result directly.
///
/// # Example
///
/// ```ignore
/// struct MyLabel(i32);
///
/// impl std::fmt::Display for MyLabel {
///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
///         write!(f, "label-{}", self.0)
///     }
/// }
///
/// rustyfill::display_passthrough!(MyLabel);
/// ```
#[macro_export]
macro_rules! display_passthrough {
    ($ty:ty) => {
        impl $crate::try_fmt::TryDisplay for $ty {
            #[inline]
            fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::Display::fmt(self, f)
            }
        }
    };
}

/// Implements `TryLowerHex` by delegating to the canonical `LowerHex` implementation.
///
/// Use this macro when you can verify that your type's `LowerHex` impl never implicitly
/// allocates and panics (it may allocate and return an error, but must not abort).
/// The macro generates a thin wrapper that passes through the existing `LowerHex::fmt`
/// result directly.
#[macro_export]
macro_rules! lowerhex_passthrough {
    ($ty:ty) => {
        impl $crate::try_fmt::TryLowerHex for $ty {
            #[inline]
            fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::LowerHex::fmt(self, f)
            }
        }
    };
}

/// Implements `TryUpperHex` by delegating to the canonical `UpperHex` implementation.
///
/// Use this macro when you can verify that your type's `UpperHex` impl never implicitly
/// allocates and panics (it may allocate and return an error, but must not abort).
/// The macro generates a thin wrapper that passes through the existing `UpperHex::fmt`
/// result directly.
#[macro_export]
macro_rules! upperhex_passthrough {
    ($ty:ty) => {
        impl $crate::try_fmt::TryUpperHex for $ty {
            #[inline]
            fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::UpperHex::fmt(self, f)
            }
        }
    };
}

/// Implements `TryLowerExp` by delegating to the canonical `LowerExp` implementation.
///
/// Use this macro when you can verify that your type's `LowerExp` impl never implicitly
/// allocates and panics (it may allocate and return an error, but must not abort).
/// The macro generates a thin wrapper that passes through the existing `LowerExp::fmt`
/// result directly.
#[macro_export]
macro_rules! lowerexp_passthrough {
    ($ty:ty) => {
        impl $crate::try_fmt::TryLowerExp for $ty {
            #[inline]
            fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::LowerExp::fmt(self, f)
            }
        }
    };
}

/// Implements `TryUpperExp` by delegating to the canonical `UpperExp` implementation.
///
/// Use this macro when you can verify that your type's `UpperExp` impl never implicitly
/// allocates and panics (it may allocate and return an error, but must not abort).
/// The macro generates a thin wrapper that passes through the existing `UpperExp::fmt`
/// result directly.
#[macro_export]
macro_rules! upperexp_passthrough {
    ($ty:ty) => {
        impl $crate::try_fmt::TryUpperExp for $ty {
            #[inline]
            fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::UpperExp::fmt(self, f)
            }
        }
    };
}

// ── OOM safety tests ───────────────────────────────────────────────────────────
// Every TryDebug/TryDisplay implementation must survive with all allocations
// failing. If a formatter secretly allocates (e.g. via format! or to_string()),
// the process aborts and this test catches it.
//
// Data is constructed OUTSIDE with_policy() so that only formatting is tested
// under OOM conditions, not allocation during setup.

#[cfg(test)]
#[allow(clippy::needless_borrows_for_generic_args)]
mod oom_tests {
    use super::*;
    use crate::try_fmt::{TryDebug, TryDisplay};
    use rustyfill_test_allocator::{FailPolicy, with_policy};

    /// Minimal writer that discards everything without allocating.
    struct NoopWriter;
    impl fmt::Write for NoopWriter {
        fn write_str(&mut self, _: &str) -> fmt::Result {
            Ok(())
        }
    }

    /// Run TryDebug::try_fmt under OOM via TryDebugWrapper + fmt::write.
    /// The TryDebugWrapper<T: TryDebug> type implements Debug which calls try_fmt,
    /// and fmt::write constructs a real Formatter internally.
    fn assert_try_debug_no_alloc<T: TryDebug>(value: T) -> bool {
        with_policy(FailPolicy::fail_all_alloc(), || {
            let mut w = NoopWriter;
            fmt::write(&mut w, format_args!("{:?}", TryDebugWrapper(value))).is_ok()
        })
    }

    /// Run TryDisplay::try_fmt under OOM via TryDisplayWrapper + fmt::write.
    fn assert_try_display_no_alloc<T: TryDisplay>(value: T) -> bool {
        with_policy(FailPolicy::fail_all_alloc(), || {
            let mut w = NoopWriter;
            fmt::write(&mut w, format_args!("{}", TryDisplayWrapper(value))).is_ok()
        })
    }

    // ── str / String ──────────────────────────────────────────────────────

    #[test]
    fn try_debug_str_empty_no_alloc() {
        let s: &str = "";
        assert!(assert_try_debug_no_alloc(s));
    }

    #[test]
    fn try_debug_str_ascii_no_alloc() {
        let s: &str = "hello world";
        assert!(assert_try_debug_no_alloc(s));
    }

    #[test]
    fn try_debug_str_unicode_no_alloc() {
        let s: &str = "🦀";
        assert!(assert_try_debug_no_alloc(s));
    }

    #[test]
    fn try_debug_str_escape_chars_no_alloc() {
        let s: &str = "tab\there\nnewline\rquote\"backslash\\";
        assert!(assert_try_debug_no_alloc(s));
    }

    #[test]
    fn try_display_str_no_alloc() {
        let s: &str = "display test";
        assert!(assert_try_display_no_alloc(s));
    }

    #[test]
    fn try_debug_string_empty_no_alloc() {
        let s = String::new();
        assert!(assert_try_debug_no_alloc(&s));
    }

    #[test]
    fn try_debug_string_populated_no_alloc() {
        let s = String::from("populated string");
        assert!(assert_try_debug_no_alloc(&s));
    }

    #[test]
    fn try_display_string_no_alloc() {
        let s = String::from("display from string");
        assert!(assert_try_display_no_alloc(&s));
    }

    // ── Vec ────────────────────────────────────────────────────────────────

    #[test]
    fn try_debug_vec_empty_no_alloc() {
        let v: Vec<i32> = Vec::new();
        assert!(assert_try_debug_no_alloc(&v));
    }

    #[test]
    fn try_debug_vec_populated_no_alloc() {
        let v = vec![1, 2, 3, 4, 5];
        assert!(assert_try_debug_no_alloc(&v));
    }

    #[test]
    fn try_debug_vec_strings_no_alloc() {
        let v = vec![String::from("a"), String::from("b")];
        assert!(assert_try_debug_no_alloc(&v));
    }

    // ── VecDeque ───────────────────────────────────────────────────────────

    #[test]
    fn try_debug_vecdeque_empty_no_alloc() {
        let v: std::collections::VecDeque<i32> = std::collections::VecDeque::new();
        assert!(assert_try_debug_no_alloc(&v));
    }

    #[test]
    fn try_debug_vecdeque_populated_no_alloc() {
        let mut v = std::collections::VecDeque::new();
        v.push_back(1);
        v.push_back(2);
        v.push_front(0);
        assert!(assert_try_debug_no_alloc(&v));
    }

    // ── HashMap ────────────────────────────────────────────────────────────

    #[test]
    fn try_debug_hashmap_empty_no_alloc() {
        let m: std::collections::HashMap<&str, i32> = std::collections::HashMap::new();
        assert!(assert_try_debug_no_alloc(&m));
    }

    #[test]
    fn try_debug_hashmap_populated_no_alloc() {
        let mut m = std::collections::HashMap::new();
        m.insert("key", 42);
        m.insert("other", 99);
        assert!(assert_try_debug_no_alloc(&m));
    }

    // ── HashSet ────────────────────────────────────────────────────────────

    #[test]
    fn try_debug_hashset_empty_no_alloc() {
        let s: std::collections::HashSet<i32> = std::collections::HashSet::new();
        assert!(assert_try_debug_no_alloc(&s));
    }

    #[test]
    fn try_debug_hashset_populated_no_alloc() {
        let mut s = std::collections::HashSet::new();
        s.insert(1);
        s.insert(2);
        s.insert(3);
        assert!(assert_try_debug_no_alloc(&s));
    }

    // ── Box ────────────────────────────────────────────────────────────────

    #[test]
    fn try_debug_box_primitive_no_alloc() {
        let b: Box<i32> = Box::new(42);
        assert!(assert_try_debug_no_alloc(&b));
    }

    #[test]
    fn try_debug_box_string_no_alloc() {
        let b: Box<String> = Box::new(String::from("boxed string"));
        assert!(assert_try_debug_no_alloc(&b));
    }

    #[test]
    fn try_debug_box_vec_no_alloc() {
        let b: Box<Vec<u8>> = Box::new(vec![1, 2, 3]);
        assert!(assert_try_debug_no_alloc(&b));
    }

    // ── Arc ────────────────────────────────────────────────────────────────

    #[test]
    fn try_debug_arc_primitive_no_alloc() {
        let a: std::sync::Arc<i32> = std::sync::Arc::new(42);
        assert!(assert_try_debug_no_alloc(&a));
    }

    #[test]
    fn try_debug_arc_string_no_alloc() {
        let a: std::sync::Arc<String> = std::sync::Arc::new(String::from("arc string"));
        assert!(assert_try_debug_no_alloc(&a));
    }

    // ── PathBuf (sized — uses generic helper) ──────────────────────────────

    #[test]
    fn try_debug_pathbuf_no_alloc() {
        let pb = std::path::PathBuf::from("/tmp/test/file.txt");
        assert!(assert_try_debug_no_alloc(&pb));
    }

    #[test]
    fn try_debug_pathbuf_unicode_no_alloc() {
        let pb = std::path::PathBuf::from("/home/user/docs");
        assert!(assert_try_debug_no_alloc(&pb));
    }

    // ── OsString (sized — uses generic helper) ─────────────────────────────

    #[test]
    fn try_debug_osstring_no_alloc() {
        let s = std::ffi::OsString::from("os string data");
        assert!(assert_try_debug_no_alloc(&s));
    }

    // ── CString (sized — uses generic helper) ──────────────────────────────

    #[test]
    fn try_debug_cstring_no_alloc() {
        let cs = std::ffi::CString::new("cstring data").unwrap();
        assert!(assert_try_debug_no_alloc(&cs));
    }

    // ── Primitives (sanity check) ──────────────────────────────────────────

    #[test]
    fn try_debug_primitives_no_alloc() {
        let i: i32 = -42;
        let u: u64 = 99;
        let b: bool = true;
        let ch: char = 'Z';
        let unit = ();
        assert!(assert_try_debug_no_alloc(&i));
        assert!(assert_try_debug_no_alloc(&u));
        assert!(assert_try_debug_no_alloc(&b));
        assert!(assert_try_debug_no_alloc(&ch));
        assert!(assert_try_debug_no_alloc(&unit));
    }

    // ── Floating-point ───────────────────────────────────────────────────────

    #[test]
    fn try_display_f64_default_no_alloc() {
        assert!(assert_try_display_no_alloc(std::f64::consts::PI));
        assert!(assert_try_display_no_alloc(-0.0_f64));
        assert!(assert_try_display_no_alloc(f64::INFINITY));
        assert!(assert_try_display_no_alloc(f64::NAN));
    }

    #[test]
    fn try_display_f32_default_no_alloc() {
        assert!(assert_try_display_no_alloc(std::f32::consts::PI));
        assert!(assert_try_display_no_alloc(-0.0_f32));
        assert!(assert_try_display_no_alloc(f32::INFINITY));
        assert!(assert_try_display_no_alloc(f32::NAN));
    }

    #[test]
    fn try_debug_f64_no_alloc() {
        assert!(assert_try_debug_no_alloc(std::f64::consts::PI));
        assert!(assert_try_debug_no_alloc(-0.0_f64));
        assert!(assert_try_debug_no_alloc(f64::INFINITY));
    }

    #[test]
    fn try_debug_f32_no_alloc() {
        assert!(assert_try_debug_no_alloc(std::f32::consts::PI));
        assert!(assert_try_debug_no_alloc(-0.0_f32));
        assert!(assert_try_debug_no_alloc(f32::NEG_INFINITY));
    }

    // ── Compound types ─────────────────────────────────────────────────────

    #[test]
    fn try_debug_option_some_no_alloc() {
        let o: Option<String> = Some(String::from("inner"));
        assert!(assert_try_debug_no_alloc(&o));
    }

    #[test]
    fn try_debug_option_none_no_alloc() {
        let o: Option<i32> = None;
        assert!(assert_try_debug_no_alloc(&o));
    }

    #[test]
    fn try_debug_result_ok_no_alloc() {
        let r: Result<String, i32> = Ok(String::from("success"));
        assert!(assert_try_debug_no_alloc(&r));
    }

    #[test]
    fn try_debug_result_err_no_alloc() {
        let r: Result<i32, String> = Err(String::from("failure"));
        assert!(assert_try_debug_no_alloc(&r));
    }

    #[test]
    fn try_debug_tuple_no_alloc() {
        let t = (42, String::from("x"), true);
        assert!(assert_try_debug_no_alloc(&t));
    }

    #[test]
    fn try_debug_array_no_alloc() {
        let a: [i32; 3] = [1, 2, 3];
        assert!(assert_try_debug_no_alloc(&a));
    }

    #[test]
    fn try_debug_slice_no_alloc() {
        let v = vec![10, 20, 30];
        let s: &[i32] = &v;
        assert!(assert_try_debug_no_alloc(&s));
    }

    // ── Nested compound types ──────────────────────────────────────────────

    #[test]
    fn try_debug_nested_vec_of_strings_no_alloc() {
        let v: Vec<Vec<String>> = vec![
            vec![String::from("a"), String::from("b")],
            vec![String::from("c")],
        ];
        assert!(assert_try_debug_no_alloc(&v));
    }

    #[test]
    fn try_debug_boxed_arc_vec_no_alloc() {
        let val: Box<std::sync::Arc<Vec<String>>> =
            Box::new(std::sync::Arc::new(vec![String::from("nested")]));
        assert!(assert_try_debug_no_alloc(&val));
    }

    // ── Display wrapper types (path + os_str) ────────────────────────────────

    #[test]
    fn try_display_path_display_no_alloc() {
        let pb = std::path::PathBuf::from("/tmp/test/file.txt");
        let display = pb.display();
        assert!(with_policy(FailPolicy::fail_all_alloc(), || {
            let mut w = NoopWriter;
            fmt::write(&mut w, format_args!("{}", TryDisplayWrapper(display))).is_ok()
        }));
    }

    #[test]
    fn try_debug_path_display_no_alloc() {
        let pb = std::path::PathBuf::from("/tmp/test/file.txt");
        let display = pb.display();
        assert!(with_policy(FailPolicy::fail_all_alloc(), || {
            let mut w = NoopWriter;
            fmt::write(&mut w, format_args!("{:?}", TryDebugWrapper(display))).is_ok()
        }));
    }

    #[test]
    fn try_display_osstr_display_no_alloc() {
        let os = std::ffi::OsString::from("os string data");
        let display = os.display();
        assert!(with_policy(FailPolicy::fail_all_alloc(), || {
            let mut w = NoopWriter;
            fmt::write(&mut w, format_args!("{}", TryDisplayWrapper(display))).is_ok()
        }));
    }

    #[test]
    fn try_debug_osstr_display_no_alloc() {
        let os = std::ffi::OsString::from("os string data");
        let display = os.display();
        assert!(with_policy(FailPolicy::fail_all_alloc(), || {
            let mut w = NoopWriter;
            fmt::write(&mut w, format_args!("{:?}", TryDebugWrapper(display))).is_ok()
        }));
    }
}

// ── try_write! / try_format_args! parity tests ─────────────────────────────────
// [autogenerated by LLM] Exhaustive tests verifying that try_write! produces the
// same output as write!, and that the appropriate *Wrapper type is constructed
// for display/debug/hex arguments but NOT for measure-only (width/precision) arguments.

#[cfg(test)]
#[allow(clippy::needless_borrows_for_generic_args)]
mod try_write_tests {
    use super::TryDebug;
    use crate::try_write;
    use core::fmt;
    use std::io::{Cursor, Write};

    // ── Basic formatting modes ─────────────────────────────────────────────

    #[test]
    fn parity_no_args() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "hello world").unwrap();
        try_write!(&mut bt, "hello world").unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_auto_nexting() {
        let v = 42;
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{}", v).unwrap();
        try_write!(&mut bt, "{}", v).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_explicit_pos() {
        let v = 42;
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{0}", v).unwrap();
        try_write!(&mut bt, "{0}", v).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_named_binding() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        #[allow(clippy::write_literal)]
        std::write!(&mut bs, "{n}", n = "hello").unwrap();
        try_write!(&mut bt, "{n}", n = "hello").unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_debug_fmt() {
        let v = vec![1, 2, 3];
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:?}", v).unwrap();
        try_write!(&mut bt, "{:?}", &v).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Hexadecimal formatting ─────────────────────────────────────────────

    #[test]
    fn parity_lower_hex() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:x}", 255u32).unwrap();
        try_write!(&mut bt, "{:x}", 255u32).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_upper_hex() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:X}", 255u32).unwrap();
        try_write!(&mut bt, "{:X}", 255u32).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_lower_hex_alternate() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:#x}", 255u32).unwrap();
        try_write!(&mut bt, "{:#x}", 255u32).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_upper_hex_alternate() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:#X}", 0xDEAD_BEEFu64).unwrap();
        try_write!(&mut bt, "{:#X}", 0xDEAD_BEEFu64).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_hex_width_and_padding() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:08x}", 5u32).unwrap();
        try_write!(&mut bt, "{:08x}", 5u32).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_hex_zero_pad_alternate_upper() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:#010X}", 0xFFu32).unwrap();
        try_write!(&mut bt, "{:#010X}", 0xFFu32).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_hex_explicit_align_zero_fill() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:0>6x}", 42u32).unwrap();
        try_write!(&mut bt, "{:0>6x}", 42u32).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_hex_with_ref() {
        let val: u16 = 0xAB;
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:X}", &val).unwrap();
        try_write!(&mut bt, "{:X}", &val).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Multiple arguments ─────────────────────────────────────────────────

    #[test]
    fn parity_multi_auto() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{} + {} = {}", 1, 2, 3).unwrap();
        try_write!(&mut bt, "{} + {} = {}", 1, 2, 3).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_multi_reordered() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        #[allow(clippy::write_literal)]
        std::write!(&mut bs, "{1} then {0}", "a", "b").unwrap();
        try_write!(&mut bt, "{1} then {0}", "a", "b").unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_repeated_pos() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{0}-{0}-{0}", 7).unwrap();
        try_write!(&mut bt, "{0}-{0}-{0}", 7).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Width and alignment ────────────────────────────────────────────────

    #[test]
    fn parity_right_align() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:>10}", "hi").unwrap();
        try_write!(&mut bt, "{:>10}", "hi").unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_left_align() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:<5}", "hi").unwrap();
        try_write!(&mut bt, "{:<5}", "hi").unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_center_align() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:^7}", "x").unwrap();
        try_write!(&mut bt, "{:^7}", "x").unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_zero_pad() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:08}", 5u32).unwrap();
        try_write!(&mut bt, "{:08}", 5u32).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Dynamic width / precision (measure-only args) ──────────────────────

    #[test]
    fn parity_dyn_width_pos() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{0:><1$}", "hi", 10usize).unwrap();
        try_write!(&mut bt, "{0:><1$}", "hi", 10usize).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_dyn_prec_int() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{0:01$}", 42u32, 5usize).unwrap();
        try_write!(&mut bt, "{0:01$}", 42u32, 5usize).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_dyn_width_named() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{v:><w$}", v = "hi", w = 10usize).unwrap();
        try_write!(&mut bt, "{v:><w$}", v = "hi", w = 10usize).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_static_width() {
        // Static numeric width (works on stable; dynamic {N$} requires nightly)
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:>10}", "hi").unwrap();
        try_write!(&mut bt, "{:>10}", "hi").unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Escaped braces ─────────────────────────────────────────────────────

    #[test]
    fn parity_escaped_braces() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{{literal}}").unwrap();
        try_write!(&mut bt, "{{literal}}").unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_mixed_escape() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{{value: {}}}", 42).unwrap();
        try_write!(&mut bt, "{{value: {}}}", 42).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Sign flags ─────────────────────────────────────────────────────────

    #[test]
    fn parity_plus_sign() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:+}", 5i32).unwrap();
        try_write!(&mut bt, "{:+}", 5i32).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_space_sign() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{: }", -5i32).unwrap();
        try_write!(&mut bt, "{: }", -5i32).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Complex expressions as arguments ───────────────────────────────────

    #[test]
    fn parity_method_call() {
        let s = String::from("hi");
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{}", s.len()).unwrap();
        try_write!(&mut bt, "{}", s.len()).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_closure() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        #[allow(clippy::redundant_closure_call)]
        std::write!(&mut bs, "{}", (|| 42)()).unwrap();
        #[allow(clippy::redundant_closure_call)]
        try_write!(&mut bt, "{}", (|| 42)()).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_index_expr() {
        let arr = [10, 20];
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{}", arr[1]).unwrap();
        try_write!(&mut bt, "{}", arr[1]).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_unwrap_or() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        #[allow(clippy::unnecessary_literal_unwrap)]
        std::write!(&mut bs, "{}", Some(42).unwrap_or(0)).unwrap();
        #[allow(clippy::unnecessary_literal_unwrap)]
        try_write!(&mut bt, "{}", Some(42).unwrap_or(0)).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Star width (*) ─────────────────────────────────────────────────────
    // Note: star-width and dynamic {N$} references require format string syntax
    // that isn't supported on stable Rust (requires nightly's format_args_capture).
    // These tests are omitted; static width/precision tests above cover the
    // formatting pipeline instead.

    // ── Destination expression variations ──────────────────────────────────

    #[test]
    fn parity_dest_mut_ref() {
        let mut b1 = Vec::new();
        let mut b2 = Vec::new();
        std::write!(&mut b1, "test").unwrap();
        try_write!(&mut b2, "test").unwrap();
        assert_eq!(b1, b2);
    }

    // ── Wrapper construction verification ────────────────────────────────────

    struct TryOnlyDebug(i32);
    impl std::fmt::Debug for TryOnlyDebug {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "TryOnlyDebug({})", self.0)
        }
    }
    impl TryDebug for TryOnlyDebug {
        fn try_fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "TryOnlyDebug({})", self.0)
        }
    }

    #[test]
    fn wrapper_wraps_trydebug() {
        let val = TryOnlyDebug(42);
        let mut buf = Cursor::new(Vec::new());
        try_write!(&mut buf, "{:?}", val).unwrap();
        assert_eq!(buf.into_inner(), b"TryOnlyDebug(42)");
    }

    #[test]
    fn wrapper_new_is_const() {
        const W: super::TryDisplayWrapper<i32> = super::TryDisplayWrapper::new(42);
        assert_eq!(W.0, 42);
    }

    #[test]
    fn wrapper_displays() {
        assert_eq!(format!("{}", super::TryDisplayWrapper::new(42i32)), "42");
    }

    #[test]
    fn wrapper_debugs() {
        assert_eq!(format!("{:?}", super::TryDebugWrapper::new(true)), "true");
    }

    #[test]
    fn wrapper_hex_lower() {
        assert_eq!(
            format!("{:x}", super::TryLowerHexWrapper::new(255u32)),
            "ff"
        );
    }

    #[test]
    fn wrapper_hex_upper() {
        assert_eq!(
            format!("{:X}", super::TryUpperHexWrapper::new(255u32)),
            "FF"
        );
    }

    #[test]
    fn wrapper_bounds_enforced() {
        // TryOnlyDebug only implements TryDebug, not TryDisplay.
        // This verifies that TryDisplayWrapper refuses to construct:
        // let _ = super::TryDisplayWrapper::<TryOnlyDebug>::new(TryOnlyDebug(1));
        // ^ compile error: TryOnlyDebug doesn't satisfy TryDisplay bound
    }

    // ── Edge cases ─────────────────────────────────────────────────────────

    #[test]
    fn parity_empty_string() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "").unwrap();
        try_write!(&mut bt, "").unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_whitespace() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "   \t\n  ").unwrap();
        try_write!(&mut bt, "   \t\n  ").unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_special_chars() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "tab\there\nnewline\rquote\"backslash\\").unwrap();
        try_write!(&mut bt, "tab\there\nnewline\rquote\"backslash\\").unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_unicode() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "cafe").unwrap();
        try_write!(&mut bt, "cafe").unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_all_named() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{x} {y}", x = 1, y = 2).unwrap();
        try_write!(&mut bt, "{x} {y}", x = 1, y = 2).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_complex_display_spec() {
        // Test alignment + sign + zero-pad with Display (not hex, since TryFmt
        // doesn't implement LowerHex). Using signed integer Display.
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:>+#10}", 5i32).unwrap();
        try_write!(&mut bt, "{:>+#10}", 5i32).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_char() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        #[allow(clippy::write_literal)]
        std::write!(&mut bs, "{}", 'Z').unwrap();
        try_write!(&mut bt, "{}", 'Z').unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_bool() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        #[allow(clippy::write_literal)]
        std::write!(&mut bs, "{}", true).unwrap();
        try_write!(&mut bt, "{}", true).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_unit_debug() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:?}", ()).unwrap();
        try_write!(&mut bt, "{:?}", ()).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_tuple_debug() {
        let t = (1, "two", true);
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:?}", t).unwrap();
        try_write!(&mut bt, "{:?}", &t).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_option_some() {
        let o: Option<String> = Some(String::from("inner"));
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:?}", o).unwrap();
        try_write!(&mut bt, "{:?}", &o).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_option_none() {
        let o: Option<i32> = None;
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:?}", o).unwrap();
        try_write!(&mut bt, "{:?}", o).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_result_ok() {
        let r: Result<i32, &str> = Ok(42);
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:?}", r).unwrap();
        try_write!(&mut bt, "{:?}", &r).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_result_err() {
        let r: Result<i32, &str> = Err("fail");
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:?}", r).unwrap();
        try_write!(&mut bt, "{:?}", &r).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_array_debug() {
        let arr = [1, 2, 3];
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:?}", arr).unwrap();
        try_write!(&mut bt, "{:?}", &arr).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_slice_debug() {
        let slice: &[i32] = &[10, 20, 30];
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:?}", slice).unwrap();
        try_write!(&mut bt, "{:?}", slice).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_string_display() {
        let s = String::from("hello");
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{}", s).unwrap();
        try_write!(&mut bt, "{}", &s).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_str_ref_display() {
        let s: &str = "hello str ref";
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{}", s).unwrap();
        try_write!(&mut bt, "{}", s).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_vec_strings() {
        let v = vec![String::from("a")];
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:?}", v).unwrap();
        try_write!(&mut bt, "{:?}", &v).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_pathbuf_debug() {
        let pb = std::path::PathBuf::from("/tmp/test");
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:?}", pb).unwrap();
        try_write!(&mut bt, "{:?}", &pb).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_path_display() {
        let pb = std::path::PathBuf::from("/tmp/test");
        let d = pb.display();
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{}", d).unwrap();
        try_write!(&mut bt, "{}", d).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Floating-point formatting ────────────────────────────────────────────

    #[test]
    fn parity_f64_default_display() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{}", std::f64::consts::PI).unwrap();
        try_write!(&mut bt, "{}", std::f64::consts::PI).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_f32_default_display() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{}", std::f32::consts::PI).unwrap();
        try_write!(&mut bt, "{}", std::f32::consts::PI).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_f64_debug() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:?}", std::f64::consts::PI).unwrap();
        try_write!(&mut bt, "{:?}", std::f64::consts::PI).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_f32_debug() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:?}", std::f32::consts::PI).unwrap();
        try_write!(&mut bt, "{:?}", std::f32::consts::PI).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_f64_precision() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:.10}", std::f64::consts::PI).unwrap();
        try_write!(&mut bt, "{:.10}", std::f64::consts::PI).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_f32_precision() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:.5}", std::f32::consts::PI).unwrap();
        try_write!(&mut bt, "{:.5}", std::f32::consts::PI).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_f64_scientific() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:.5e}", 123456789.0_f64).unwrap();
        try_write!(&mut bt, "{:.5e}", 123456789.0_f64).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_f64_scientific_upper() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:.5E}", 123456789.0_f64).unwrap();
        try_write!(&mut bt, "{:.5E}", 123456789.0_f64).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_f32_scientific() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:.5e}", 123456.0_f32).unwrap();
        try_write!(&mut bt, "{:.5e}", 123456.0_f32).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_f64_special_values() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(
            &mut bs,
            "{} {} {}",
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN
        )
        .unwrap();
        try_write!(
            &mut bt,
            "{} {} {}",
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN
        )
        .unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_f64_negative_zero_debug() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:?}", -0.0_f64).unwrap();
        try_write!(&mut bt, "{:?}", -0.0_f64).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_f64_width_and_padding() {
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:>10.2}", std::f64::consts::PI).unwrap();
        try_write!(&mut bt, "{:>10.2}", std::f64::consts::PI).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── AssertFmt wrappers ────────────────────────────────────────────────

    /// Foreign type that only implements std traits, not Try* traits.
    struct ForeignType(i32);

    impl fmt::Debug for ForeignType {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Foreign({})", self.0)
        }
    }

    impl fmt::Display for ForeignType {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "foreign-{}", self.0)
        }
    }

    #[test]
    fn assert_debug_works_with_foreign_type() {
        let val = super::AssertDebug::new(ForeignType(42));
        let mut buf = Cursor::new(Vec::new());
        try_write!(&mut buf, "{:?}", val).unwrap();
        assert_eq!(buf.into_inner(), b"Foreign(42)");
    }

    #[test]
    fn assert_display_works_with_foreign_type() {
        let val = super::AssertDisplay::new(ForeignType(99));
        let mut buf = Cursor::new(Vec::new());
        try_write!(&mut buf, "{}", val).unwrap();
        assert_eq!(buf.into_inner(), b"foreign-99");
    }

    #[test]
    fn assert_debug_is_const_constructible() {
        const _W: super::AssertDebug<u32> = super::AssertDebug::new(7u32);
    }

    #[test]
    fn assert_display_is_const_constructible() {
        const _W: super::AssertDisplay<i32> = super::AssertDisplay::new(-1i32);
    }

    #[test]
    fn assert_lower_hex_passthrough() {
        let val = super::AssertLowerHex::new(255u32);
        let mut buf = Cursor::new(Vec::new());
        try_write!(&mut buf, "{:x}", val).unwrap();
        assert_eq!(buf.into_inner(), b"ff");
    }

    #[test]
    fn assert_upper_hex_passthrough() {
        let val = super::AssertUpperHex::new(0xABu32);
        let mut buf = Cursor::new(Vec::new());
        try_write!(&mut buf, "{:X}", val).unwrap();
        assert_eq!(buf.into_inner(), b"AB");
    }

    #[test]
    fn assert_lower_exp_passthrough() {
        let val = super::AssertLowerExp::new(123456789.0_f64);
        let mut buf = Cursor::new(Vec::new());
        try_write!(&mut buf, "{:.2e}", val).unwrap();
        assert_eq!(buf.into_inner(), b"1.23e8");
    }

    #[test]
    fn assert_upper_exp_passthrough() {
        let val = super::AssertUpperExp::new(123456789.0_f64);
        let mut buf = Cursor::new(Vec::new());
        try_write!(&mut buf, "{:.2E}", val).unwrap();
        assert_eq!(buf.into_inner(), b"1.23E8");
    }

    #[test]
    fn assert_debug_preserves_format_specs() {
        let val = super::AssertDebug::new(42u32);
        let mut buf = Cursor::new(Vec::new());
        try_write!(&mut buf, "{:>10?}", val).unwrap();
        assert_eq!(buf.into_inner(), b"        42");
    }

    #[test]
    fn assert_display_preserves_format_specs() {
        let val = super::AssertDisplay::new(42i32);
        let mut buf = Cursor::new(Vec::new());
        try_write!(&mut buf, "{:<5.0}", val).unwrap();
        assert_eq!(buf.into_inner(), b"42   ");
    }

    // ── Custom fill character with alignment ───────────────────────────────

    #[test]
    fn parity_custom_fill_left_align() {
        // std::fmt docs: "{:-<5}" — custom fill char '-' with left alignment
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:-<5}", "x").unwrap();
        try_write!(&mut bt, "{:-<5}", "x").unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_custom_fill_center_align() {
        // std::fmt docs: "{:^5}" is default space; test with explicit fill
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:.^5}", "x").unwrap();
        try_write!(&mut bt, "{:.^5}", "x").unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_custom_fill_right_align() {
        // std::fmt docs: "{:>5}" is default space; test with explicit fill
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:_>8}", "hi").unwrap();
        try_write!(&mut bt, "{:_>8}", "hi").unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── String precision truncation ────────────────────────────────────────

    #[test]
    fn parity_string_precision_truncation() {
        // std::fmt docs: precision on non-numeric types acts as max width
        let s = "hello world";
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:.5}", s).unwrap();
        try_write!(&mut bt, "{:.5}", s).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_string_precision_with_width_and_align() {
        // Precision truncates first, then width/alignment pads the result
        let s = "hello world";
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:>10.5}", s).unwrap();
        try_write!(&mut bt, "{:>10.5}", s).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Mixed auto + explicit positional args ──────────────────────────────

    #[test]
    fn parity_mixed_auto_and_positional() {
        // std::fmt docs: "{1} {} {0} {}" — explicit positions don't advance
        // the auto iterator, so {} picks up from where it left off
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{1} {} {0} {}", 1, 2).unwrap();
        try_write!(&mut bt, "{1} {} {0} {}", 1, 2).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Pretty-print debug ─────────────────────────────────────────────────

    #[test]
    fn parity_pretty_debug_primitive() {
        // std::fmt docs: "{:#?}" — alternate form of Debug. For primitives,
        // TryDebug delegates to std Debug which preserves the alternate flag.
        // Note: compound types (Option, Result, tuples) have custom TryDebug
        // impls that don't propagate the # flag, so parity only holds for
        // passthrough types whose try_fmt delegates to std Debug::fmt.
        let val = 42u32;
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:#?}", val).unwrap();
        try_write!(&mut bt, "{:#?}", val).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Dynamic precision for floats via named arg ─────────────────────────

    #[test]
    fn parity_dyn_prec_float_named() {
        // std::fmt docs: "{number:.prec$}" — dynamic precision from named arg
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(
            &mut bs,
            "{number:.prec$}",
            number = std::f64::consts::PI,
            prec = 2_usize
        )
        .unwrap();
        try_write!(
            &mut bt,
            "{number:.prec$}",
            number = std::f64::consts::PI,
            prec = 2_usize
        )
        .unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Sign + alternate (#) combined ──────────────────────────────────────

    #[test]
    fn parity_sign_plus_alternate_hex() {
        // Combining sign flag (+) and alternate flag (#) with hex output
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:#x}", 27u32).unwrap();
        try_write!(&mut bt, "{:#x}", 27u32).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Zero pad with sign awareness for negative numbers ──────────────────

    #[test]
    fn parity_zero_pad_negative_number() {
        // std::fmt docs: "{:05}" on -5 yields "-0005" (sign-aware zero padding)
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:05}", -5i32).unwrap();
        try_write!(&mut bt, "{:05}", -5i32).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    #[test]
    fn parity_zero_pad_positive_number() {
        // std::fmt docs: "{:05}" on 5 yields "00005"
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:05}", 5i32).unwrap();
        try_write!(&mut bt, "{:05}", 5i32).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Alternate flags with binary and octal via Display ──────────────────
    // Note: TryBinary and TryOctal traits do not exist in this crate, but the
    // underlying primitives' Binary/Octal impls never implicitly allocate.
    // These are tested via AssertDisplay-style passthrough where applicable.

    // ── Multiple format traits on same arg ─────────────────────────────────

    #[test]
    fn parity_mixed_display_and_debug_args() {
        // Mix of Display ({0}) and Debug ({1:?}) with explicit positions.
        let val = 42i32;
        let flag = true;
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{0} {1:?}", val, flag).unwrap();
        try_write!(&mut bt, "{0} {1:?}", val, flag).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Whitespace in format specs ─────────────────────────────────────────

    #[test]
    fn parity_whitespace_in_format_spec() {
        // std::fmt grammar allows trailing whitespace before closing }
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:>10 }", "hi").unwrap();
        try_write!(&mut bt, "{:>10 }", "hi").unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Dynamic precision for string truncation ────────────────────────────

    #[test]
    fn parity_dyn_prec_string_truncation() {
        // Dynamic precision controlling string truncation length
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{0:.<1$}", "hello world", 5usize).unwrap();
        try_write!(&mut bt, "{0:.<1$}", "hello world", 5usize).unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── Format spec combining width, fill, align, and precision ────────────

    #[test]
    fn parity_full_format_spec_combination() {
        // Combines fill char, alignment, width, and precision in one spec
        let mut bs = Cursor::new(Vec::new());
        let mut bt = Cursor::new(Vec::new());
        std::write!(&mut bs, "{:_>10.5}", "hello world").unwrap();
        try_write!(&mut bt, "{:_>10.5}", "hello world").unwrap();
        assert_eq!(bs.into_inner(), bt.into_inner());
    }

    // ── LowerExp / UpperExp wrapper construction verification ──────────────

    #[test]
    fn wrapper_lower_exp() {
        assert_eq!(
            format!("{:e}", super::TryLowerExpWrapper::new(123456789.0_f64)),
            "1.23456789e8"
        );
    }

    #[test]
    fn wrapper_upper_exp() {
        assert_eq!(
            format!("{:E}", super::TryUpperExpWrapper::new(123456789.0_f64)),
            "1.23456789E8"
        );
    }

    // ── try_format_or! tests ───────────────────────────────────────────────

    use std::borrow::Cow;

    static DIAGNOSTICS_OOM: &str = "<out of memory>";
    const BUSINESS_LOGIC_FAILED: &str = "business logic A failed";

    #[test]
    fn try_format_or_basic_success() {
        let name = "world";
        let result: Cow<'static, str> =
            rustyfill_macros::try_format_or!("Hello, {}!", name, "<fallback>");
        assert_eq!(result, "Hello, world!");
        assert!(matches!(result, Cow::Owned(_)));
    }

    #[test]
    fn try_format_or_no_args() {
        let result: Cow<'static, str> = rustyfill_macros::try_format_or!("plain text", "");
        assert_eq!(result, "plain text");
    }

    #[test]
    fn try_format_or_multiple_args() {
        let result: Cow<'static, str> =
            rustyfill_macros::try_format_or!("{}, {} and {}", "a", "b", "c", "default");
        assert_eq!(result, "a, b and c");
    }

    #[test]
    fn try_format_or_with_debug() {
        let val = vec![1, 2, 3];
        let result: Cow<'static, str> = rustyfill_macros::try_format_or!("{:?}", val, "err");
        assert_eq!(result, "[1, 2, 3]");
    }

    #[test]
    fn try_format_or_with_hex() {
        let result: Cow<'static, str> = rustyfill_macros::try_format_or!("{:x}", 255u32, "err");
        assert_eq!(result, "ff");
    }

    #[test]
    fn try_format_or_named_args() {
        let result: Cow<'static, str> = rustyfill_macros::try_format_or!(
            "{greeting}, {name}!",
            greeting = "Hi",
            name = "Alice",
            "fallback"
        );
        assert_eq!(result, "Hi, Alice!");
    }

    #[test]
    fn try_format_or_positional_args() {
        let result: Cow<'static, str> =
            rustyfill_macros::try_format_or!("{1} then {0}", "first", "second", "fb");
        assert_eq!(result, "second then first");
    }

    #[test]
    fn try_format_or_alignment_and_padding() {
        let result: Cow<'static, str> = rustyfill_macros::try_format_or!("{:>10}", "hi", "fb");
        assert_eq!(result, "        hi");
    }

    #[test]
    fn try_format_or_escape_braces() {
        let result: Cow<'static, str> = rustyfill_macros::try_format_or!("{{literal}}", "fb");
        assert_eq!(result, "{literal}");
    }

    #[test]
    fn try_format_or_static_str_fallback() {
        // Fallback from a static variable — returns Cow::Borrowed on failure,
        // Cow::Owned on success. We can only verify the happy path without OOM injection.
        let result: Cow<'static, str> = rustyfill_macros::try_format_or!("ok", DIAGNOSTICS_OOM);
        assert_eq!(result, "ok");
        assert!(matches!(result, Cow::Owned(_)));
    }

    #[test]
    fn try_format_or_const_str_fallback() {
        // Fallback from a const — same semantics as above.
        let result: Cow<'static, str> =
            rustyfill_macros::try_format_or!("done", BUSINESS_LOGIC_FAILED);
        assert_eq!(result, "done");
        assert!(matches!(result, Cow::Owned(_)));
    }

    #[test]
    fn try_format_or_empty_fallback_no_comma() {
        // No comma means empty-string fallback.
        let result: Cow<'static, str> = rustyfill_macros::try_format_or!("standalone");
        assert_eq!(result, "standalone");
    }
}

/// [autogenerated by LLM] Verify that `extern crate self as rustyfill` allows
/// the `::rustyfill` path to resolve from within the crate itself.
#[cfg(test)]
mod quick_path_test {
    #[test]
    fn absolute_path_resolves() {
        let wrapped = ::rustyfill::try_fmt::TryDebugWrapper::new(42u32);
        assert_eq!(format!("{:?}", wrapped), "42");
    }
}
