//! Static and dynamic frames — the container types that bundle a context item,
//! attachments, and child frames together.

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;

use core::error::Error;
use core::panic::Location;

use rustyfill::try_fmt::{TryDebug, TryDisplay, helpers::FormatterExt};

use crate::frame::item::{ContextFrame, ItemImpl};
use rustyfill::prelude::TryBox;

// ── StaticFrame ──────────────────────────────────────────────────────────────

/// A typed frame holding an error context, arbitrary attachments, and child
/// frames (sources).
///
/// Stored inline as the head of [`Report`](super::super::Report) or in the
/// peers [`VecDeque`](alloc::collections::VecDeque). All peers in a single
/// report share the same context type `C`.
pub struct StaticFrame<C> {
    pub(crate) context: ContextFrame<C>,
    pub(crate) attachments: Vec<Box<dyn ItemImpl>>,
    pub(crate) children: VecDeque<DynamicFrame>,
    /// Number of attachments silently dropped due to allocation failure.
    pub(crate) lost_attachments: usize,
    /// Number of child frames silently dropped due to allocation failure.
    pub(crate) lost_children: usize,
}

impl<C> StaticFrame<C> {
    /// Creates a new static frame from a context item.
    #[must_use]
    pub(crate) fn new(context: ContextFrame<C>) -> Self {
        Self {
            context,
            attachments: Vec::new(),
            children: VecDeque::new(),
            lost_attachments: 0,
            lost_children: 0,
        }
    }

    // ── Context accessors ────────────────────────────────────────────────

    /// Returns a reference to the stored context.
    #[must_use]
    pub const fn context(&self) -> &ContextFrame<C> {
        &self.context
    }

    /// Returns a mutable reference to the stored context.
    #[must_use]
    pub fn context_mut(&mut self) -> &mut ContextFrame<C> {
        &mut self.context
    }

    // ── Segment label ────────────────────────────────────────────────────

    /// Sets the business logic segment label on this frame's context.
    #[must_use]
    pub fn attach_segment(mut self, segment: impl Into<alloc::borrow::Cow<'static, str>>) -> Self {
        self.context = self.context.attach_segment(segment);
        self
    }

    // ── Attachment accessors ─────────────────────────────────────────────

    /// Returns references to all attachments on this frame.
    #[must_use]
    pub fn attachments(&self) -> &[Box<dyn ItemImpl>] {
        &self.attachments
    }

    /// Returns the number of attachments silently dropped due to OOM.
    #[must_use]
    pub const fn lost_attachments(&self) -> usize {
        self.lost_attachments
    }

    // ── Children accessors ───────────────────────────────────────────────

    /// Returns a reference to the children deque.
    #[must_use]
    pub fn children(&self) -> &VecDeque<DynamicFrame> {
        &self.children
    }

    /// Returns mutable access to the children deque.
    #[must_use]
    pub(crate) fn children_mut(&mut self) -> &mut VecDeque<DynamicFrame> {
        &mut self.children
    }

    /// Returns the number of child frames silently dropped due to OOM.
    #[must_use]
    pub const fn lost_children(&self) -> usize {
        self.lost_children
    }

    // ── Search helpers ───────────────────────────────────────────────────

    /// Checks whether any attachment or child frame contains a value of type `T`.
    #[must_use]
    pub(crate) fn contains<T: Send + Sync + 'static>(&self) -> bool {
        for att in &self.attachments {
            if att.as_any().is::<T>() {
                return true;
            }
        }
        for child in &self.children {
            if child.contains::<T>() {
                return true;
            }
        }
        false
    }

    /// Searches attachments and children for a value of type `T`, returning
    /// the most recent match.
    #[must_use]
    pub(crate) fn downcast_ref<T: Send + Sync + 'static>(&self) -> Option<&T> {
        for att in &self.attachments {
            if let Some(r) = att.as_any().downcast_ref() {
                return Some(r);
            }
        }
        for child in self.children.iter().rev() {
            if let Some(r) = child.downcast_ref::<T>() {
                return Some(r);
            }
        }
        None
    }

    /// Searches context, attachments and children for a value of type `T`,
    /// returning the most recent match as a mutable reference.
    #[must_use]
    pub(crate) fn downcast_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T>
    where
        C: Error + TryDebug + TryDisplay + Send + Sync + 'static,
    {
        // Check context first.
        if let Some(r) = self.context.as_any_mut().downcast_mut::<T>() {
            return Some(r);
        }
        for att in &mut self.attachments {
            if let Some(r) = att.as_any_mut().downcast_mut() {
                return Some(r);
            }
        }
        // Iterate children in reverse using as_slices to handle wrap-around.
        let (first, second) = self.children.as_slices();
        let ptr2 = second.as_ptr() as *mut DynamicFrame;
        for i in (0..second.len()).rev() {
            let child = unsafe { &mut *ptr2.add(i) };
            if let Some(r) = child.downcast_mut::<T>() {
                return Some(r);
            }
        }
        let ptr1 = first.as_ptr() as *mut DynamicFrame;
        for i in (0..first.len()).rev() {
            let child = unsafe { &mut *ptr1.add(i) };
            if let Some(r) = child.downcast_mut::<T>() {
                return Some(r);
            }
        }
        None
    }
}

impl<C> From<C> for StaticFrame<C> {
    #[track_caller]
    fn from(context: C) -> Self {
        Self::new(ContextFrame::new(context, *Location::caller()))
    }
}

/// Graceful-degradation [`Debug`] implementation: avoids delegating to `C`'s
/// [`Debug`] which may implicitly allocate (e.g. on macOS with types like
/// `PathBuf`, `Duration`, or floats with precision specifiers). Instead prints
/// only the context type name and structural metadata.
impl<C: core::fmt::Debug> core::fmt::Debug for StaticFrame<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StaticFrame")
            .field("context_type", &core::any::type_name::<C>())
            .field("attachments_len", &self.attachments.len())
            .field("children_len", &self.children.len())
            .field("lost_attachments", &self.lost_attachments)
            .field("lost_children", &self.lost_children)
            .finish()
    }
}

/// TryDebug with reduced functionality: the context `C` may not implement
/// `TryDebug`, and attachments are boxed trait objects. Prints struct metadata
/// (counts, lossage) without recursing into inner data. Requires `C: Debug`
/// because `TryDebug` has `Debug` as a supertrait.
impl<C: core::fmt::Debug> TryDebug for StaticFrame<C> {
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.try_debug_struct("StaticFrame")
            .field_owned(
                "context_type",
                alloc::borrow::Cow::Borrowed::<str>(core::any::type_name::<C>()),
            )
            .field_owned("attachments_len", self.attachments.len())
            .field_owned("children_len", self.children.len())
            .field_owned("lost_attachments", self.lost_attachments)
            .field_owned("lost_children", self.lost_children)
            .finish()
    }
}

// ── DynamicFrame ─────────────────────────────────────────────────────────────

/// A type-erased frame node created when a [`StaticFrame`] is demoted during
/// [`change_context`](super::super::Report::change_context).
///
/// Mirrors [`StaticFrame`] but with a boxed, type-erased context item instead
/// of a known-type [`ContextFrame`].
pub struct DynamicFrame {
    pub(crate) context: Box<dyn ItemImpl>,
    pub(crate) attachments: Vec<Box<dyn ItemImpl>>,
    pub(crate) children: VecDeque<DynamicFrame>,
    /// Number of attachments silently dropped due to allocation failure.
    pub(crate) lost_attachments: usize,
    /// Number of child frames silently dropped due to allocation failure.
    pub(crate) lost_children: usize,
}

impl DynamicFrame {
    /// Converts a [`StaticFrame`] into a [`DynamicFrame`] by boxing its context.
    ///
    /// Returns `Err((sf, err))` if the heap allocation fails, giving back the
    /// original static frame so it can be recovered.
    #[allow(clippy::result_large_err, reason = "cannot allocate on the heap")]
    pub(crate) fn from_static<C>(
        sf: StaticFrame<C>,
    ) -> Result<Self, (StaticFrame<C>, rustyfill::alloc::AllocError)>
    where
        C: Error + TryDebug + TryDisplay + Send + Sync + 'static,
    {
        let StaticFrame {
            context,
            attachments,
            children,
            lost_attachments,
            lost_children,
        } = sf;

        match Box::<ContextFrame<C>>::fallible_new_give_back(context) {
            Ok(boxed) => Ok(Self {
                context: boxed,
                attachments,
                children,
                lost_attachments,
                lost_children,
            }),
            Err((ctx, err)) => {
                let sf = StaticFrame {
                    context: ctx,
                    attachments,
                    children,
                    lost_attachments,
                    lost_children,
                };
                Err((sf, err))
            }
        }
    }

    /// Returns what kind of item the context is.
    #[must_use]
    pub fn kind(&self) -> crate::frame::item::ItemKind {
        self.context.kind()
    }

    // ── Accessors ────────────────────────────────────────────────────────

    /// Returns a reference to the boxed context item.
    #[must_use]
    pub fn context_item(&self) -> &dyn ItemImpl {
        &*self.context
    }

    /// Returns a mutable reference to the boxed context item.
    #[must_use]
    pub fn context_item_mut(&mut self) -> &mut dyn ItemImpl {
        &mut *self.context
    }

    /// Returns references to all attachments on this frame.
    #[must_use]
    pub fn attachments(&self) -> &[Box<dyn ItemImpl>] {
        &self.attachments
    }

    /// Returns the number of attachments silently dropped due to OOM.
    #[must_use]
    pub const fn lost_attachments(&self) -> usize {
        self.lost_attachments
    }

    /// Returns a reference to the children deque.
    #[must_use]
    pub fn children(&self) -> &VecDeque<DynamicFrame> {
        &self.children
    }

    /// Returns mutable access to the children deque.
    #[must_use]
    pub fn children_mut(&mut self) -> &mut VecDeque<DynamicFrame> {
        &mut self.children
    }

    /// Returns the number of child frames silently dropped due to OOM.
    #[must_use]
    pub const fn lost_children(&self) -> usize {
        self.lost_children
    }

    // ── Downcast helpers ─────────────────────────────────────────────────

    /// Returns whether the held context has type `T`.
    #[must_use]
    pub fn is<T: Send + Sync + 'static>(&self) -> bool {
        self.context.as_any().is::<T>()
    }

    /// Downcasts the held context to `&T` if the type matches.
    #[must_use]
    pub fn downcast_ref<T: Send + Sync + 'static>(&self) -> Option<&T> {
        if let Some(r) = self.context.as_any().downcast_ref() {
            return Some(r);
        }
        for att in &self.attachments {
            if let Some(r) = att.as_any().downcast_ref() {
                return Some(r);
            }
        }
        for child in self.children.iter().rev() {
            if let Some(r) = child.downcast_ref::<T>() {
                return Some(r);
            }
        }
        None
    }

    /// Downcasts the held context to `&mut T` if the type matches.
    #[must_use]
    pub fn downcast_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        if let Some(r) = self.context.as_any_mut().downcast_mut() {
            return Some(r);
        }
        for att in &mut self.attachments {
            if let Some(r) = att.as_any_mut().downcast_mut() {
                return Some(r);
            }
        }
        for child in self.children.iter_mut().rev() {
            if let Some(r) = child.downcast_mut::<T>() {
                return Some(r);
            }
        }
        None
    }

    /// Checks whether any attachment or child frame contains a value of type `T`.
    #[must_use]
    pub(crate) fn contains<T: Send + Sync + 'static>(&self) -> bool {
        if self.context.as_any().is::<T>() {
            return true;
        }
        for att in &self.attachments {
            if att.as_any().is::<T>() {
                return true;
            }
        }
        for child in &self.children {
            if child.contains::<T>() {
                return true;
            }
        }
        false
    }
}

impl core::fmt::Debug for DynamicFrame {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DynamicFrame")
            .field("kind", &self.kind())
            .field("attachments_len", &self.attachments.len())
            .field("children_len", &self.children.len())
            .field("lost_attachments", &self.lost_attachments)
            .field("lost_children", &self.lost_children)
            .finish()
    }
}

/// TryDebug with reduced functionality: the context is a type-erased trait object
/// and children are recursive `DynamicFrame`s. Prints struct metadata without
/// recursing into inner data to avoid unbounded allocation.
impl TryDebug for DynamicFrame {
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.try_debug_struct("DynamicFrame")
            .field("kind", &self.kind())
            .field_owned("attachments_len", self.attachments.len())
            .field_owned("children_len", self.children.len())
            .field_owned("lost_attachments", self.lost_attachments)
            .field_owned("lost_children", self.lost_children)
            .finish()
    }
}
