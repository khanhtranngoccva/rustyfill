//! Item trait and context frame — the leaf-level building blocks.
//!
//! Every piece of data stored inside a frame is an *item*: either a typed
//! [`ContextFrame`] carrying an error plus segment and location, or an
//! arbitrary attachment boxed as [`Box<dyn ItemImpl>`].

use alloc::borrow::Cow;
use core::any::Any;
use core::fmt;
use core::panic::Location;

// ── ItemKind ────────────────────────────────────────────────────────────────

/// Classification of an item's contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKind {
    /// A context item holding an error type plus its segment label and location.
    Context,
    /// An arbitrary attachment (opaque or printable).
    Attachment,
}

// ── ItemImpl ────────────────────────────────────────────────────────────────

/// Common trait implemented by every concrete item type.
///
/// This is the type-erased interface behind [`Box<dyn ItemImpl>`] inside both
/// [`StaticFrame`](super::StaticFrame) attachments and
/// [`DynamicFrame`](super::DynamicFrame) heads.
pub trait ItemImpl: Send + Sync + 'static {
    /// Returns what kind of item this is.
    fn kind(&self) -> ItemKind;

    /// Downcast to [`Any`] for runtime type inspection.
    fn as_any(&self) -> &dyn Any;

    /// Mutable downcast to [`Any`] for runtime type mutation.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// Prevent accidental Debug/Display of the raw trait object.
impl fmt::Debug for dyn ItemImpl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ItemImpl({:?})", self.kind())
    }
}

// ── ContextFrame ─────────────────────────────────────────────────────────────

/// A context item holding an error, its business logic segment label, and the
/// source location where it was created.
///
/// Stored inline in [`StaticFrame`](super::StaticFrame), or boxed as
/// [`Box<dyn ItemImpl>`] inside a [`DynamicFrame`](super::DynamicFrame) when
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

impl<C: core::error::Error + Send + Sync + 'static> ItemImpl for ContextFrame<C> {
    fn kind(&self) -> ItemKind {
        ItemKind::Context
    }

    fn as_any(&self) -> &dyn Any {
        &self.context
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.context
    }
}

impl<C: fmt::Debug> fmt::Debug for ContextFrame<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("ContextFrame");
        debug.field("context", &self.context);
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
