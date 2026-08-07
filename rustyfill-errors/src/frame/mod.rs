//! Frame types — the building blocks of error reports.
//!
//! A [`StaticFrame`] holds a typed context, attachments, and child frames.
//! When demoted during [`change_context`](super::Report::change_context), it
//! becomes a [`DynamicFrame`] with a type-erased context.
//!
//! Items inside frames implement [`ItemImpl`]; the primary item type is
//! [`ContextFrame`], which carries an error value plus segment label and
//! source location.

mod attachment;
mod item;
mod static_frame;

pub use attachment::{OpaqueAttachment, PrintableAttachment};
pub use item::{ContextFrame, ItemImpl, ItemKind};
pub use static_frame::{DynamicFrame, StaticFrame};
