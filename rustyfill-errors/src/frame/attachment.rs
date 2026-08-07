//! Attachment wrappers — printable and opaque variants for arbitrary data stored
//! on a [`StaticFrame`](crate::frame::StaticFrame).
//!
//! Attachments are boxed as [`Box<dyn ItemImpl>`](crate::ItemImpl) inside frames.
//! Two wrapper types exist depending on whether the inner value is human-readable:
//!
//! - [`PrintableAttachment<T>`] — for values that implement `Debug` + `Display`.
//! - [`OpaqueAttachment<T>`] — for any `Send + Sync + 'static` value.

use core::any::Any;
use core::fmt;

use crate::frame::{ItemImpl, ItemKind};

// ── PrintableAttachment ──────────────────────────────────────────────────────

/// An attachment wrapping a value that implements [`Debug`] and [`Display`].
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

impl<T: Send + Sync + 'static> ItemImpl for PrintableAttachment<T> {
    fn kind(&self) -> ItemKind {
        ItemKind::Attachment
    }

    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.0
    }
}

impl<T: fmt::Debug> fmt::Debug for PrintableAttachment<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl<T: fmt::Display> fmt::Display for PrintableAttachment<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

impl<T: Send + Sync + 'static> ItemImpl for OpaqueAttachment<T> {
    fn kind(&self) -> ItemKind {
        ItemKind::Attachment
    }

    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.0
    }
}
