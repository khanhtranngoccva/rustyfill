//! Helper types that mirror `std::fmt`'s debug builders (`DebugStruct`, `DebugTuple`,
//! `DebugList`, `DebugMap`, `DebugSet`) but accept values that implement
//! [`TryDebug`](super::TryDebug).
//!
//! Each builder wraps members in [`TryDebugWrapper`](super::TryDebugWrapper) before
//! delegating to the corresponding `std::fmt` debug type, ensuring that formatting
//! flows through the fallible path even under OOM conditions.
//!
//! # Usage
//!
//! Extension methods are provided on `&mut fmt::Formatter<'_>`:
//!
//! ```ignore
//! fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//!     f.try_debug_list().entries(self.iter()).finish()
//! }
//! ```

use core::fmt;

use super::TryDebug;

// ── Wrapper ─────────────────────────────────────────────────────────────────────
// Thin shim that implements fmt::Debug by delegating to TryDebug::try_fmt,
// so TryDebug references can be passed into std's Debug* builders.

struct D<'a>(&'a dyn TryDebug);

impl fmt::Debug for D<'_> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.try_fmt(f)
    }
}

// Owned wrapper for field_owned
struct WO<T: TryDebug>(T);

impl<T: TryDebug> fmt::Debug for WO<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.try_fmt(f)
    }
}

// ── Formatter extension trait ───────────────────────────────────────────────────

/// Extension methods on [`fmt::Formatter`] that return fallible debug builders.
pub trait FormatterExt<'f> {
    /// Returns a [`TryDebugList`] builder for formatting a list or sequence.
    fn try_debug_list(&mut self) -> TryDebugList<'_, 'f>;

    /// Returns a [`TryDebugSet`] builder for formatting a set-like collection.
    fn try_debug_set(&mut self) -> TryDebugSet<'_, 'f>;

    /// Returns a [`TryDebugMap`] builder for formatting a map-like collection.
    fn try_debug_map(&mut self) -> TryDebugMap<'_, 'f>;

    /// Returns a [`TryDebugStruct`] builder for formatting a struct with the given name.
    fn try_debug_struct(&mut self, name: &'f str) -> TryDebugStruct<'_, 'f>;

    /// Returns a [`TryDebugTuple`] builder for formatting a tuple.
    fn try_debug_tuple(&mut self) -> TryDebugTuple<'_, 'f>;
}

impl<'f> FormatterExt<'f> for fmt::Formatter<'f> {
    #[inline]
    fn try_debug_list(&mut self) -> TryDebugList<'_, 'f> {
        TryDebugList { inner: self.debug_list() }
    }

    #[inline]
    fn try_debug_set(&mut self) -> TryDebugSet<'_, 'f> {
        TryDebugSet { inner: self.debug_set() }
    }

    #[inline]
    fn try_debug_map(&mut self) -> TryDebugMap<'_, 'f> {
        TryDebugMap { inner: self.debug_map() }
    }

    #[inline]
    fn try_debug_struct(&mut self, name: &'f str) -> TryDebugStruct<'_, 'f> {
        TryDebugStruct { inner: self.debug_struct(name) }
    }

    #[inline]
    fn try_debug_tuple(&mut self) -> TryDebugTuple<'_, 'f> {
        TryDebugTuple { inner: self.debug_tuple("") }
    }
}

// ── TryDebugList ────────────────────────────────────────────────────────────────

/// Builder for formatting a list whose elements implement [`TryDebug`].
///
/// Wraps each element before passing it to the inner [`core::fmt::DebugList`],
/// so formatting routes through [`TryDebug::try_fmt`].
pub struct TryDebugList<'b, 'f> {
    inner: fmt::DebugList<'b, 'f>,
}

impl<'d, 'b, 'f> TryDebugList<'b, 'f> {
    /// Add a single entry to the list.
    #[inline]
    pub fn entry<T: TryDebug>(&mut self, value: &'d T) -> &mut Self {
        self.inner.entry(&D(value));
        self
    }

    /// Extend the list from an iterator of references to [`TryDebug`] values.
    pub fn entries<I, T>(&mut self, iter: I) -> &mut Self
    where
        I: IntoIterator<Item = &'d T>,
        T: TryDebug + 'd,
    {
        for item in iter {
            self.inner.entry(&D(item));
        }
        self
    }

    /// Finish building and write the list to the formatter.
    pub fn finish(&mut self) -> fmt::Result {
        self.inner.finish()
    }
}

// ── TryDebugSet ─────────────────────────────────────────────────────────────────

/// Builder for formatting a set whose elements implement [`TryDebug`].
pub struct TryDebugSet<'b, 'f> {
    inner: fmt::DebugSet<'b, 'f>,
}

impl<'d, 'b, 'f> TryDebugSet<'b, 'f> {
    /// Add a single entry to the set.
    #[inline]
    pub fn entry<T: TryDebug>(&mut self, value: &'d T) -> &mut Self {
        self.inner.entry(&D(value));
        self
    }

    /// Extend the set from an iterator of references to [`TryDebug`] values.
    pub fn entries<I, T>(&mut self, iter: I) -> &mut Self
    where
        I: IntoIterator<Item = &'d T>,
        T: TryDebug + 'd,
    {
        for item in iter {
            self.inner.entry(&D(item));
        }
        self
    }

    /// Finish building and write the set to the formatter.
    pub fn finish(&mut self) -> fmt::Result {
        self.inner.finish()
    }
}

// ── TryDebugMap ─────────────────────────────────────────────────────────────────

/// Builder for formatting a map whose keys and values implement [`TryDebug`].
pub struct TryDebugMap<'b, 'f> {
    inner: fmt::DebugMap<'b, 'f>,
}

impl<'d, 'b, 'f> TryDebugMap<'b, 'f> {
    /// Add a single key-value entry to the map.
    #[inline]
    pub fn entry<K: TryDebug, V: TryDebug>(
        &mut self,
        key: &'d K,
        value: &'d V,
    ) -> &mut Self {
        self.inner.entry(&D(key), &D(value));
        self
    }

    /// Extend the map from an iterator of `(key, value)` reference pairs.
    pub fn entries<I, K, V>(&mut self, iter: I) -> &mut Self
    where
        I: IntoIterator<Item = (&'d K, &'d V)>,
        K: TryDebug + 'd,
        V: TryDebug + 'd,
    {
        for (k, v) in iter {
            self.inner.entry(&D(k), &D(v));
        }
        self
    }

    /// Finish building and write the map to the formatter.
    pub fn finish(&mut self) -> fmt::Result {
        self.inner.finish()
    }
}

// ── TryDebugStruct ──────────────────────────────────────────────────────────────

/// Builder for formatting a struct whose fields implement [`TryDebug`].
pub struct TryDebugStruct<'b, 'f> {
    inner: fmt::DebugStruct<'b, 'f>,
}

impl<'d, 'b, 'f> TryDebugStruct<'b, 'f> {
    /// Add a field with the given name and a [`TryDebug`] value.
    #[inline]
    pub fn field<T: TryDebug>(&mut self, name: &str, value: &'d T) -> &mut Self {
        self.inner.field(name, &D(value));
        self
    }

    /// Add a field with the given name and an owned [`TryDebug`] value.
    #[inline]
    pub fn field_owned<T: TryDebug>(&mut self, name: &str, value: T) -> &mut Self {
        self.inner.field(name, &WO(value));
        self
    }

    /// Finish building and write the struct to the formatter.
    pub fn finish(&mut self) -> fmt::Result {
        self.inner.finish()
    }
}

// ── TryDebugTuple ───────────────────────────────────────────────────────────────

/// Builder for formatting a tuple whose elements implement [`TryDebug`].
pub struct TryDebugTuple<'b, 'f> {
    inner: fmt::DebugTuple<'b, 'f>,
}

impl<'d, 'b, 'f> TryDebugTuple<'b, 'f> {
    /// Add a field to the tuple.
    #[inline]
    pub fn field<T: TryDebug>(&mut self, value: &'d T) -> &mut Self {
        self.inner.field(&D(value));
        self
    }

    /// Finish building and write the tuple to the formatter.
    pub fn finish(&mut self) -> fmt::Result {
        self.inner.finish()
    }
}
