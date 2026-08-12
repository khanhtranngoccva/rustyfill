//! Attachment wrappers — printable and opaque variants for arbitrary data stored
//! on a [`StaticFrame`](crate::frame::StaticFrame).
//!
//! Attachments are boxed as [`Box<dyn ItemImpl>`](crate::ItemImpl) inside frames.
//! Two wrapper types exist depending on whether the inner value is human-readable:
//!
//! - [`PrintableAttachment<T>`] — for values that implement `TryDebug` + `TryDisplay`.
//! - [`OpaqueAttachment<T>`] — for any `Send + Sync + 'static` value.

use core::any::Any;
use core::fmt;

use rustyfill::try_fmt::{TryDebug, TryDisplay};

use crate::frame::{ItemImpl, ItemKind};

// ── PrintableAttachment ──────────────────────────────────────────────────────

/// An attachment wrapping a value that implements [`TryDebug`] and [`TryDisplay`].
///
/// The trait objects produced from this type can be printed via the inherited
/// `dyn fmt::Debug` and `dyn fmt::Display` vtables.
pub struct PrintableAttachment<T>(pub(crate) T);

impl<T> PrintableAttachment<T> {
    pub(crate) fn new(value: T) -> Self {
        Self(value)
    }

    /// Returns a reference to the inner value.
    #[must_use]
    pub fn get(&self) -> &T {
        &self.0
    }

    /// Returns a mutable reference to the inner value.
    #[must_use]
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: TryDebug + TryDisplay + Send + Sync + 'static> ItemImpl for PrintableAttachment<T> {
    fn kind(&self) -> ItemKind {
        ItemKind::PrintableAttachment
    }

    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.0
    }

    fn is_printable(&self) -> bool {
        true
    }

    fn try_display_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(&self.0, f)
    }
}

impl<T: TryDebug> TryDebug for PrintableAttachment<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(&self.0, f)
    }
}

impl<T: TryDisplay> TryDisplay for PrintableAttachment<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(&self.0, f)
    }
}

impl<T: TryDebug> fmt::Debug for PrintableAttachment<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TryDebug has Debug as a supertrait, so this always works.
        fmt::Debug::fmt(&self.0, f)
    }
}

impl<T: TryDisplay> fmt::Display for PrintableAttachment<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TryDisplay has Display as a supertrait, so this always works.
        fmt::Display::fmt(&self.0, f)
    }
}

// ── OpaqueAttachment ─────────────────────────────────────────────────────────

/// An attachment wrapping an arbitrary value with no formatting requirements.
///
/// Cannot be printed through the `ItemImpl` trait object; callers must
/// downcast via [`as_any`](ItemImpl::as_any) to recover the concrete type.
pub struct OpaqueAttachment<T>(pub(crate) T);

impl<T> OpaqueAttachment<T> {
    pub(crate) fn new(value: T) -> Self {
        Self(value)
    }

    /// Returns a reference to the inner value.
    #[must_use]
    pub fn get(&self) -> &T {
        &self.0
    }

    /// Returns a mutable reference to the inner value.
    #[must_use]
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: Send + Sync + 'static> TryDebug for OpaqueAttachment<T> {
    #[inline]
    fn try_fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl<T: Send + Sync + 'static> TryDisplay for OpaqueAttachment<T> {
    #[inline]
    fn try_fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl<T: Send + Sync + 'static> fmt::Debug for OpaqueAttachment<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<opaque>")
    }
}

impl<T: Send + Sync + 'static> fmt::Display for OpaqueAttachment<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<opaque>")
    }
}

impl<T: Send + Sync + 'static> ItemImpl for OpaqueAttachment<T> {
    fn kind(&self) -> ItemKind {
        ItemKind::OpaqueAttachment
    }

    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.0
    }
}
