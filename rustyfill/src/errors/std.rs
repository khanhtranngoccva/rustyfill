//! TryDebug / TryDisplay implementations for well-known std error types.
//!
//! These types' `Debug` and `Display` implementations are known to never
//! implicitly allocate — they print fixed struct names, enum discriminants,
//! or delegate to primitive/slice formatting.
//!
//! Generic wrappers ([`PoisonError`], [`IntoInnerError`], [`LockError`]) implement
//! `TryDebug` conditionally when their inner type also implements `TryDebug`.
//! Some wrappers use reduced-functionality debug output when the inner type
//! cannot guarantee allocation-free formatting.
//!
//! `Display` impls are unconditional across all types in this module because
//! they write fixed strings or delegate to primitive formatting.

use crate::try_fmt::helpers::FormatterExt;
use crate::try_fmt::{TryDebug, TryDisplay};

// ── ::lang_std::num::TryFromIntError ──────────────────────────────────────────────────

impl TryDebug for ::lang_std::num::TryFromIntError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for ::lang_std::num::TryFromIntError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

// ── ::lang_std::array::TryFromSliceError ──────────────────────────────────────────────

impl TryDebug for ::lang_std::array::TryFromSliceError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for ::lang_std::array::TryFromSliceError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

// ── ::lang_std::str::Utf8Error ────────────────────────────────────────────────────────

impl TryDebug for ::lang_std::str::Utf8Error {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for ::lang_std::str::Utf8Error {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

// ── ::lang_std::ffi::NulError ─────────────────────────────────────────────────────────

impl TryDebug for ::lang_std::ffi::NulError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // NulError's Debug prints the inner Vec<u8> and nul position — both
        // delegate to slice/primitive Debug which never allocates.
        core::fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for ::lang_std::ffi::NulError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

// ── ::lang_std::num parse errors ──────────────────────────────────────────────────────

impl TryDebug for ::lang_std::num::ParseIntError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for ::lang_std::num::ParseIntError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

impl TryDebug for ::lang_std::num::ParseFloatError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for ::lang_std::num::ParseFloatError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

impl TryDebug for ::lang_std::str::ParseBoolError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for ::lang_std::str::ParseBoolError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

// ── ::lang_std::time::SystemTimeError ─────────────────────────────────────────────────

impl TryDebug for ::lang_std::time::SystemTimeError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for ::lang_std::time::SystemTimeError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

// ── Generic error wrappers ─────────────────────────────────────────────────────
// These delegate to the inner type's TryDebug when available. Display impls are
// unconditional because they write fixed strings.

// ── ::lang_std::sync::PoisonError<G> ──────────────────────────────────────────────────

impl<G: TryDebug> TryDebug for ::lang_std::sync::PoisonError<G> {
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.try_debug_struct("PoisonError")
            .field("inner", self.get_ref())
            .finish()
    }
}

impl<G> TryDisplay for ::lang_std::sync::PoisonError<G> {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // PoisonError's Display writes "poisoned" — no hidden allocations.
        core::fmt::Display::fmt(self, f)
    }
}

// ── ::lang_std::io::IntoInnerError<W> ─────────────────────────────────────────────────

impl<W: TryDebug> TryDebug for ::lang_std::io::IntoInnerError<W> {
    /// Reduced functionality: IntoInnerError has no `get_ref()` accessor on stable,
    /// only consuming `into_inner()`. Prints struct name and type info without inner data.
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.try_debug_struct("IntoInnerError")
            .field_owned(
                "inner_type",
                ::lang_std::borrow::Cow::Borrowed::<str>(core::any::type_name::<W>()),
            )
            .finish()
    }
}

impl<W> TryDisplay for ::lang_std::io::IntoInnerError<W> {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // IntoInnerError's Display writes a static message about pending I/O —
        // no hidden allocations.
        core::fmt::Display::fmt(self, f)
    }
}

// ── ::lang_std::thread::JoinError ─────────────────────────────────────────────────────
// Note: JoinError was removed from std in recent Rust editions. The join result
// is now expressed via `::lang_std::thread::Thread::join()` returning `Result<T, Box<dyn Any + Send>>`.
// No impl needed.

// ── ::lang_std::fmt::Error ────────────────────────────────────────────────────────────
// fmt::Error is an empty struct whose Debug prints "Error" and Display prints
// "internal or I/O error". Neither allocates.

impl TryDebug for ::lang_std::fmt::Error {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for ::lang_std::fmt::Error {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

// ── ::lang_std::char::CharTryFromError ────────────────────────────────────────────────
// Empty struct (contains only private ()). Debug/Display print fixed strings.

impl TryDebug for ::lang_std::char::CharTryFromError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for ::lang_std::char::CharTryFromError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

// ── ::lang_std::ffi::FromBytesWithNulError ────────────────────────────────────────────
// Enum with variants containing byte slices and positions. Debug delegates to
// slice/primitive formatting. Safe passthrough.

impl TryDebug for ::lang_std::ffi::FromBytesWithNulError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for ::lang_std::ffi::FromBytesWithNulError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

// ── ::lang_std::ffi::IntoStringError ──────────────────────────────────────────────────
// Holds a CString (no non-consuming accessor) and a Copy Utf8Error. Reduced
// functionality: includes the utf8_error (which is Copy and implements TryDebug),
// but suppresses the inner CString since only into_cstring() is available.

impl TryDebug for ::lang_std::ffi::IntoStringError {
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.try_debug_struct("IntoStringError")
            .field_owned("cstring", "<suppressed>")
            .field("utf8_error", &self.utf8_error())
            .finish()
    }
}

impl TryDisplay for ::lang_std::ffi::IntoStringError {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

// ── ::lang_std::sync::TryLockError<G> ─────────────────────────────────────────────────
// TryLockError has variants WouldBlock (no inner data) and Poisoned(PoisonError<G>).
// The guard G may not implement TryDebug. Reduced functionality: prints variant
// info without the guard contents.

impl<G> TryDebug for ::lang_std::sync::TryLockError<G> {
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ::lang_std::sync::TryLockError::WouldBlock => f.write_str("TryLockError::WouldBlock"),
            ::lang_std::sync::TryLockError::Poisoned(_) => f
                .try_debug_struct("TryLockError::Poisoned")
                .field_owned("inner", "<PoisonError suppressed>")
                .finish(),
        }
    }
}

impl<G> TryDisplay for ::lang_std::sync::TryLockError<G> {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

// ── ::lang_std::sync::MutexGuard / RwLockReadGuard / RwLockWriteGuard ────────────────
// These are guard types, not error types per se, but are commonly encountered
// inside error wrappers (PoisonError, TryLockError). Their Debug impls delegate
// to the inner type's Debug, which may allocate. Reduced functionality: print
// struct name via try_debug_struct. Requires T: Debug because TryDebug supertrait
// requires Debug. When T: TryDebug, callers should route through PoisonError<G:
// TryDebug> instead of holding the guard directly.

impl<T: core::fmt::Debug> TryDebug for ::lang_std::sync::MutexGuard<'_, T> {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.try_debug_struct("MutexGuard")
            .field_owned("inner", "<suppressed>")
            .finish()
    }
}

impl<T: core::fmt::Debug> TryDebug for ::lang_std::sync::RwLockReadGuard<'_, T> {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.try_debug_struct("RwLockReadGuard")
            .field_owned("inner", "<suppressed>")
            .finish()
    }
}

impl<T: core::fmt::Debug> TryDebug for ::lang_std::sync::RwLockWriteGuard<'_, T> {
    #[inline]
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.try_debug_struct("RwLockWriteGuard")
            .field_owned("inner", "<suppressed>")
            .finish()
    }
}
