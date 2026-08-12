//! Item trait and context frame — the leaf-level building blocks.
//!
//! Every piece of data stored inside a frame is an *item*: either a typed
//! [`ContextFrame`] carrying an error plus segment and location, or an
//! arbitrary attachment boxed as `Box<dyn ItemImpl>`.

use core::any::Any;
use core::fmt;
use core::panic::Location;

use rustyfill::try_fmt::{TryDebug, TryDisplay};

// ── ItemKind ────────────────────────────────────────────────────────────────

/// Classification of an item's contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKind {
    /// A context item holding an error type plus its segment label and location.
    Context,
    /// A printable attachment wrapping a value that implements [`TryDisplay`](rustyfill::try_fmt::TryDisplay).
    PrintableAttachment,
    /// An opaque attachment with no formatting guarantees.
    OpaqueAttachment,
}

/// `ItemKind` is a simple enum with no inner data — Debug delegates to enum
/// discriminant printing, which never allocates. Safe full passthrough.
impl TryDebug for ItemKind {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// ── ItemImpl ────────────────────────────────────────────────────────────────

/// Common trait implemented by every concrete item type.
///
/// This is the type-erased interface behind `Box<dyn ItemImpl>` inside both
/// [`StaticFrame`](super::StaticFrame) attachments and
/// [`DynamicFrame`](super::DynamicFrame) heads.
///
/// [`TryDisplay`] and [`TryDebug`] are supertraits so that `dyn ItemImpl` can
/// be formatted fallibly without needing a cached string.
pub trait ItemImpl: TryDisplay + TryDebug + Send + Sync + 'static {
    /// Returns what kind of item this is.
    fn kind(&self) -> ItemKind;

    /// Downcast to [`Any`] for runtime type inspection.
    fn as_any(&self) -> &dyn Any;

    /// Mutable downcast to [`Any`] for runtime type mutation.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Returns `true` if this attachment can be displayed via [`fmt::Display`].
    ///
    /// Printable attachments return `true`; opaque attachments return `false`.
    /// Context items always return `false` — they are rendered separately.
    fn is_printable(&self) -> bool {
        false
    }

    /// Formats this item for human-readable display output (fallible path).
    ///
    /// For printable attachments, delegates to [`TryDisplay::try_fmt`].
    /// For opaque attachments, returns `Ok(())`.
    fn try_display_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(self, f)
    }

    /// Writes the "at file:line:col" location line for context items.
    ///
    /// Returns `Ok(())` if a location was written, does nothing for non-context items.
    fn write_location(&self, _f: &mut fmt::Formatter<'_>) {}
}

// ── ContextFrame ─────────────────────────────────────────────────────────────

use alloc::borrow::Cow;

/// A context item holding an error, its business logic segment label, and the
/// source location where it was created.
///
/// Stored inline in [`StaticFrame`](super::StaticFrame), or boxed as
/// `Box<dyn ItemImpl>` inside a [`DynamicFrame`](super::DynamicFrame) when
/// demoted during [`change_context`](super::super::Report::change_context).
pub struct ContextFrame<C> {
    pub(crate) context: C,
    pub(crate) segment: Option<Cow<'static, str>>,
    pub(crate) location: Location<'static>,
}

impl<C> ContextFrame<C> {
    pub(crate) fn new(context: C, location: Location<'static>) -> Self {
        Self {
            context,
            segment: None,
            location,
        }
    }

    /// Sets the business logic segment label.
    #[must_use]
    pub fn attach_segment(mut self, segment: impl Into<Cow<'static, str>>) -> Self {
        self.segment = Some(segment.into());
        self
    }

    /// Returns a reference to the stored context.
    #[must_use]
    pub const fn context(&self) -> &C {
        &self.context
    }

    /// Returns a mutable reference to the stored context.
    #[must_use]
    pub fn context_mut(&mut self) -> &mut C {
        &mut self.context
    }

    /// Returns the segment label if one was set.
    #[must_use]
    pub fn segment(&self) -> Option<&str> {
        self.segment.as_deref()
    }

    /// Returns the captured source location.
    #[must_use]
    pub const fn location(&self) -> &Location<'static> {
        &self.location
    }
}

impl<C: core::error::Error + TryDebug + TryDisplay + Send + Sync + 'static> ItemImpl
    for ContextFrame<C>
{
    fn kind(&self) -> ItemKind {
        ItemKind::Context
    }

    fn as_any(&self) -> &dyn Any {
        &self.context
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.context
    }

    fn is_printable(&self) -> bool {
        false
    }

    fn write_location(&self, f: &mut fmt::Formatter<'_>) {
        let loc = &self.location;
        let _ = f
            .write_str("at ")
            .and_then(|_| rustyfill::try_write!(f, "{}", loc.file()))
            .and_then(|_| f.write_str(":"))
            .and_then(|_| rustyfill::try_write!(f, "{}", loc.line()))
            .and_then(|_| f.write_str(":"))
            .and_then(|_| rustyfill::try_write!(f, "{}", loc.column()));
    }
}

impl<C: TryDebug> TryDebug for ContextFrame<C> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDebug::try_fmt(&self.context, f)
    }
}

impl<C: TryDisplay> TryDisplay for ContextFrame<C> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TryDisplay::try_fmt(&self.context, f)
    }
}

/// Graceful-degradation [`Debug`] implementation: avoids delegating to `C`'s
/// [`Debug`] which may implicitly allocate (e.g. on macOS with types like
/// `PathBuf`, `Duration`, or floats with precision specifiers). Instead prints
/// only the context type name and structural metadata.
impl<C: fmt::Debug> fmt::Debug for ContextFrame<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("ContextFrame");
        debug.field("context_type", &core::any::type_name::<C>());
        if let Some(ref seg) = self.segment {
            debug.field("segment", seg);
        }
        debug.field("location", &self.location);
        debug.finish()
    }
}

impl<C: fmt::Display> fmt::Display for ContextFrame<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.context, f)
    }
}
