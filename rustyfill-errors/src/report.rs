//! The [`Report`] type — a context-aware error container with OOM resilience.

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;

use core::error::Error;
use core::marker::PhantomData;
use core::ops::ControlFlow;
use core::panic::Location;

use crate::frame::{ContextFrame, DynamicFrame, ItemImpl, StaticFrame};
use crate::frame::{OpaqueAttachment, PrintableAttachment};
use rustyfill::alloc::TryReserveError;
use rustyfill::prelude::{TryBox, TryVec, TryVecDeque};

// ── Report ───────────────────────────────────────────────────────────────────

/// Contains a frame stack consisting of typed peer errors and dynamic children.
///
/// The generic parameter `C` is the *primary context* — every peer in this
/// report carries the same error type `C`. The head is stored inline (no
/// allocation), while additional peers live in a [`VecDeque`] that can be
/// optionally capped. Child frames created during demotion are type-erased
/// [`DynamicFrame`]s nested inside each peer.
///
/// ## Layout
///
/// ```text
/// Report<C>
/// ├── head: StaticFrame<C>          (inline)
/// │   ├── context: ContextFrame<C>
/// │   ├── attachments: Vec<Box<dyn ItemImpl>>
/// │   ├── children: VecDeque<DynamicFrame>
/// │   ├── lost_attachments: usize
/// │   └── lost_children: usize
/// ├── peers: VecDeque<StaticFrame<C>>
/// ├── lost_peers: usize
/// └── capacity: Option<usize>
/// ```
#[must_use]
pub struct Report<C> {
    head: StaticFrame<C>,
    peers: VecDeque<StaticFrame<C>>,
    /// Number of peers silently evicted due to capacity or allocation pressure.
    lost_peers: usize,
    /// Optional upper bound on total peer count (head + peers).
    capacity: Option<usize>,
}

// ── Construction ─────────────────────────────────────────────────────────────

impl<C> Report<C> {
    /// Creates a new `Report` from an error context.
    ///
    /// Captures the call site via `#[track_caller]`. No segment label is set.
    #[track_caller]
    pub fn new(context: C) -> Self {
        Self {
            head: StaticFrame::new(ContextFrame::new(context, *Location::caller())),
            peers: VecDeque::new(),
            lost_peers: 0,
            capacity: None,
        }
    }

    /// Creates a new `Report` with a business logic segment label.
    #[track_caller]
    pub fn with_segment(context: C, segment: impl Into<Cow<'static, str>>) -> Self {
        Self {
            head: StaticFrame::new(
                ContextFrame::new(context, *Location::caller()).attach_segment(segment),
            ),
            peers: VecDeque::new(),
            lost_peers: 0,
            capacity: None,
        }
    }

    /// Sets an optional capacity cap on the number of peers.
    ///
    /// When the total peer count (head + peers deque) exceeds `cap`, the
    /// oldest peer is evicted on the next [`push`](Self::push) and
    /// [`lost_peers`](Self::lost_peers) is incremented.
    pub fn with_capacity(mut self, cap: usize) -> Self {
        self.capacity = Some(cap);
        self
    }

    // ── Current context access ───────────────────────────────────────────

    /// Returns a reference to the current (head) context.
    #[must_use]
    pub fn current_context(&self) -> &C {
        self.head.context().context()
    }

    /// Returns a mutable reference to the current (head) context.
    #[must_use]
    pub fn current_context_mut(&mut self) -> &mut C {
        self.head.context_mut().context_mut()
    }

    /// Returns the segment label of the current context, if set.
    #[must_use]
    pub fn segment(&self) -> Option<&str> {
        self.head.context().segment()
    }

    /// Returns the source location of the current context.
    #[must_use]
    pub fn location(&self) -> &Location<'static> {
        self.head.context().location()
    }

    /// Sets the segment label on the current context.
    pub fn attach_segment(mut self, segment: impl Into<Cow<'static, str>>) -> Self {
        self.head = self.head.attach_segment(segment);
        self
    }

    /// Returns references to all peers (most recent first).
    pub fn current_contexts(&self) -> PeerIter<'_, C> {
        PeerIter {
            head: Some(&self.head),
            peers: self.peers.iter(),
            _phantom: PhantomData,
        }
    }

    /// Returns mutable references to all peers (most recent first).
    pub fn current_contexts_mut(&mut self) -> PeerIterMut<'_, C> {
        // Split self into &mut head and an IterMut over peers by transmuting
        // through raw pointers — avoiding the double-borrow problem of taking
        // &mut self.head and &mut self.peers at the same time.
        let head_ptr = core::ptr::addr_of_mut!(self.head);
        let peers_iter = unsafe {
            // SAFETY: peers_iter borrows only self.peers, head_ref borrows only
            // self.head — they are disjoint fields of Report. The lifetime 'a
            // is tied to &mut self, so both references remain valid for the
            // same duration.
            let peers_raw = core::ptr::addr_of_mut!(self.peers);
            (*peers_raw).iter_mut()
        };
        let head_ref = unsafe { &mut *head_ptr };
        PeerIterMut {
            head: Some(head_ref),
            peers: peers_iter,
        }
    }

    /// Returns the number of peers silently evicted due to capacity or OOM.
    #[must_use]
    pub const fn lost_peers(&self) -> usize {
        self.lost_peers
    }

    /// Returns the total number of peers (head counts as 1).
    #[must_use]
    pub fn peer_count(&self) -> usize {
        // Saturate so that when peers.len() == usize::MAX we don't overflow.
        1_usize.saturating_add(self.peers.len())
    }

    /// Returns the length of the report.
    #[must_use]
    pub fn len(&self) -> usize {
        self.peer_count()
    }

    /// Checks if the report is empty. Always return `false` because the head exists.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.peer_count() != 0
    }

    // ── Attachments (printable, lossy) ───────────────────────────────────

    /// Attaches printable data to the head frame.
    ///
    /// The value must implement [`Debug`] and [`Display`] so that the
    /// attachment can be rendered when the report is formatted.
    ///
    /// If boxing or vec growth fails, the attachment is silently dropped
    /// and [`head.lost_attachments`](StaticFrame::lost_attachments) is incremented.
    pub fn attach<A>(mut self, attachment: A) -> Self
    where
        A: core::fmt::Debug + core::fmt::Display + Send + Sync + 'static,
    {
        let pa = PrintableAttachment::new(attachment);
        match Box::<PrintableAttachment<A>>::fallible_new_give_back(pa) {
            Ok(b) => {
                if TryVec::try_push_give_back(&mut self.head.attachments, b).is_err() {
                    self.head.lost_attachments += 1;
                }
            }
            Err(_) => {
                self.head.lost_attachments += 1;
            }
        }
        self
    }

    // ── Attachments (opaque, lossy) ──────────────────────────────────────

    /// Attaches opaque data to the head frame without formatting requirements.
    ///
    /// Unlike [`attach`](Self::attach), the value need not implement [`Debug`]
    /// or [`Display`]. It can only be recovered by downcasting via
    /// [`contains`](Self::contains) / [`downcast_ref`](Self::downcast_ref).
    ///
    /// If boxing or vec growth fails, the attachment is silently dropped
    /// and [`head.lost_attachments`](StaticFrame::lost_attachments) is incremented.
    pub fn attach_opaque<A>(mut self, attachment: A) -> Self
    where
        A: Send + Sync + 'static,
    {
        let oa = OpaqueAttachment::new(attachment);
        match Box::<OpaqueAttachment<A>>::fallible_new_give_back(oa) {
            Ok(b) => {
                if TryVec::try_push_give_back(&mut self.head.attachments, b).is_err() {
                    self.head.lost_attachments += 1;
                }
            }
            Err(_) => {
                self.head.lost_attachments += 1;
            }
        }
        self
    }

    // ── Attachments (printable, lossless) ────────────────────────────────

    /// Attaches printable data to the head frame, returning the attachment
    /// on allocation failure.
    ///
    /// The value must implement [`Debug`] and [`Display`].
    ///
    /// Returns `Ok(report)` on success, or `Err((report, attachment))` giving
    /// back ownership so the caller isn't surprised by silent data loss.
    #[allow(clippy::result_large_err, reason = "cannot allocate error on the heap")]
    pub fn try_attach<A>(mut self, attachment: A) -> Result<Self, (Self, A)>
    where
        A: core::fmt::Debug + core::fmt::Display + Send + Sync + 'static,
    {
        // Reserve space in the vec first so that if it fails we haven't yet
        // boxed the attachment and can return it cleanly.
        if self.head.attachments.try_reserve(1).is_err() {
            return Err((self, attachment));
        }

        let pa = PrintableAttachment::new(attachment);
        let boxed = match Box::<PrintableAttachment<A>>::fallible_new_give_back(pa) {
            Ok(b) => b,
            Err((PrintableAttachment(a), _)) => return Err((self, a)),
        };

        // Capacity was reserved above, so this push cannot fail.
        self.head.attachments.push(boxed);
        Ok(self)
    }

    // ── Attachments (opaque, lossless) ───────────────────────────────────

    /// Attaches opaque data to the head frame, returning the attachment
    /// on allocation failure.
    ///
    /// Unlike [`try_attach`](Self::try_attach), the value need not implement
    /// [`Debug`] or [`Display`].
    ///
    /// Returns `Ok(report)` on success, or `Err((report, attachment))` giving
    /// back ownership so the caller isn't surprised by silent data loss.
    #[allow(clippy::result_large_err, reason = "cannot allocate error on the heap")]
    pub fn try_attach_opaque<A>(mut self, attachment: A) -> Result<Self, (Self, A)>
    where
        A: Send + Sync + 'static,
    {
        // Reserve space in the vec first so that if it fails we haven't yet
        // boxed the attachment and can return it cleanly.
        if self.head.attachments.try_reserve(1).is_err() {
            return Err((self, attachment));
        }

        let oa = OpaqueAttachment::new(attachment);
        let boxed = match Box::<OpaqueAttachment<A>>::fallible_new_give_back(oa) {
            Ok(b) => b,
            Err((OpaqueAttachment(a), _)) => return Err((self, a)),
        };

        // Capacity was reserved above, so this push cannot fail.
        self.head.attachments.push(boxed);
        Ok(self)
    }

    // ── Push peers ───────────────────────────────────────────────────────

    /// Pushes a new peer onto the front of the peers deque.
    ///
    /// Accepts anything that converts into a [`StaticFrame<C>`] — most commonly
    /// a bare `C` thanks to the blanket `From<C>` impl.
    ///
    /// If a capacity cap is set and exceeded, the oldest peer (back of deque)
    /// is evicted and [`lost_peers`](Self::lost_peers) is incremented.
    /// If allocation for the push fails, the same eviction logic applies.
    /// If `peers.len()` is already `usize::MAX`, the oldest peer is evicted
    /// to make room and [`lost_peers`](Self::lost_peers) is incremented.
    ///
    /// Captures the call site via `#[track_caller]`.
    #[track_caller]
    pub fn push(mut self, frame: impl Into<StaticFrame<C>>) -> Self {
        // Enforce capacity before pushing (drop context on evict failure). When pop_back is done, try_push_front_give_back always succeeds.
        if let Some(cap) = self.capacity {
            while self.peer_count() >= cap {
                self.peers.pop_back();
                self.lost_peers += 1;
            }
        }

        // Guard against usize::MAX overflow: evict oldest to make room. When pop_back is done, try_push_front_give_back always succeeds.
        if self.peers.len() == usize::MAX {
            self.peers.pop_back();
            self.lost_peers += 1;
        }

        let sf = frame.into();

        if let Err((sf, _)) = TryVecDeque::try_push_front_give_back(&mut self.peers, sf) {
            self.peers.pop_back();
            self.lost_peers += 1;
            self.peers
                .try_push_front_give_back(sf)
                .map_err(|(_, e)| e)
                .expect("pop_back followed by push_front should succeed");
        }

        self
    }

    /// Pushes a new peer, returning the frame on failure.
    ///
    /// Accepts anything that converts into a [`StaticFrame<C>`] — most commonly
    /// a bare `C` thanks to the blanket `From<C>` impl.
    ///
    /// Does NOT evict existing peers. Returns `Err((report, frame))` if at
    /// capacity, at `usize::MAX` peers, or if allocation fails.
    #[track_caller]
    #[allow(clippy::result_large_err, reason = "may not allocate on heap")]
    pub fn try_push(
        mut self,
        frame: impl Into<StaticFrame<C>>,
    ) -> Result<Self, (Self, StaticFrame<C>)> {
        let sf: StaticFrame<C> = frame.into();

        // Guard against usize::MAX overflow.
        if self.peers.len() == usize::MAX {
            return Err((self, sf));
        }

        if let Some(cap) = self.capacity
            && self.peer_count() >= cap
        {
            return Err((self, sf));
        }

        match TryVecDeque::try_push_front_give_back(&mut self.peers, sf) {
            Ok(()) => Ok(self),
            Err((recovered_sf, _)) => Err((self, recovered_sf)),
        }
    }
}

// ── Context changes ──────────────────────────────────────────────────────────

/// Error returned by [`Report::try_change_context`] on allocation failure.
///
/// Because the implementation uses recoverable fallible pushes, the original
/// report is always reconstructed and given back alongside the new context
/// that was supplied to [`try_change_context`](Report::try_change_context).
pub struct ChangeContextError<C, T> {
    /// The original report, recovered from whatever frames could be rebuilt.
    pub report: Report<C>,
    /// The new context that was passed to [`try_change_context`].
    pub context: T,
}

impl<C: core::fmt::Debug + Error + Send + Sync + 'static, T: core::fmt::Debug> core::fmt::Debug
    for ChangeContextError<C, T>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ChangeContextError")
            .field("report", &self.report)
            .field("context", &self.context)
            .finish()
    }
}

/// Intermediate frame used during [`try_change_context`] to defer boxing until
/// all allocations succeed. Holds a `Box<ContextFrame<C>>` so that on success
/// we can erase the type, and we a costly downcast on the failure path.
struct IntermediateFrame<C> {
    context: Box<ContextFrame<C>>,
    attachments: Vec<Box<dyn ItemImpl>>,
    children: VecDeque<DynamicFrame>,
    lost_attachments: usize,
    lost_children: usize,
}

impl<C> IntermediateFrame<C>
where
    C: Error + Send + Sync + 'static,
{
    /// Converts an intermediate frame into a [`DynamicFrame`] by casting the
    /// boxed context to a trait object (no additional allocation).
    fn into_dynamic(self) -> DynamicFrame {
        DynamicFrame {
            context: self.context as Box<dyn ItemImpl>,
            attachments: self.attachments,
            children: self.children,
            lost_attachments: self.lost_attachments,
            lost_children: self.lost_children,
        }
    }

    /// Recovers a [`StaticFrame<C>`] from this intermediate by unboxing the
    /// context (cheap — just dereferencing the box).
    fn recover_static(self) -> StaticFrame<C> {
        StaticFrame {
            context: *self.context,
            attachments: self.attachments,
            children: self.children,
            lost_attachments: self.lost_attachments,
            lost_children: self.lost_children,
        }
    }
}

impl<C> Report<C>
where
    C: Error + Send + Sync + 'static,
{
    /// Changes the current context to a new type `T`.
    ///
    /// All existing peers (including the head) are demoted into [`DynamicFrame`]s
    /// and pushed into the new head's children deque. Oldest peers are dropped
    /// first if allocation fails, incrementing [`lost_children`](StaticFrame::lost_children).
    ///
    /// If the allocation for the new head fails entirely, only the new context
    /// survives inline.
    #[track_caller]
    pub fn change_context<T>(self, context: T) -> Report<T>
    where
        T: Error + Send + Sync + 'static,
    {
        let Self {
            head: old_head,
            mut peers,
            lost_peers,
            capacity,
        } = self;

        let mut new_head_sf = StaticFrame::new(ContextFrame::new(context, *Location::caller()));
        new_head_sf.lost_children = lost_peers;

        // Reserve space in children for all peers + the old head.
        let total = peers.len().saturating_add(1);
        let _ = new_head_sf.children.try_reserve(total);

        // Helper: convert a StaticFrame to a DynamicFrame and push it onto
        // children. On from_static failure, evicts oldest children repeatedly
        // and retries until success or no children remain to evict. On push
        // failure, evicts one oldest child and retries the push.
        let mut demote_and_push = |sf: StaticFrame<C>| {
            let mut current_sf = sf;
            loop {
                match DynamicFrame::from_static(current_sf) {
                    Ok(df) => {
                        if let Err((dropped_df, _)) =
                            TryVecDeque::try_push_front_give_back(&mut new_head_sf.children, df)
                        {
                            // Evict oldest child to make room, then retry push.
                            new_head_sf.children.pop_back();
                            new_head_sf.lost_children += 1;
                            new_head_sf
                                .children
                                .try_push_front_give_back(dropped_df)
                                .map_err(|(_, e)| e)
                                .expect("eviction guarantees space");
                        }
                        return;
                    }
                    Err((recovered_sf, _)) => {
                        // Conversion failed — try evicting an old child to free
                        // memory, then retry with the recovered static frame.
                        if new_head_sf.children.pop_back().is_some() {
                            new_head_sf.lost_children += 1;
                            current_sf = recovered_sf;
                            continue;
                        }
                        // Nothing left to evict; give up on this frame.
                        new_head_sf.lost_children += 1;
                        return;
                    }
                }
            }
        };

        // Demote peers oldest-first (pop_back), pushing newest to front of children.
        while let Some(sf) = peers.pop_back() {
            demote_and_push(sf);
        }

        // Demote the old head last — it becomes the deepest child.
        demote_and_push(old_head);

        Report {
            head: new_head_sf,
            peers: VecDeque::new(),
            lost_peers: 0,
            capacity,
        }
    }

    /// Changes the current context to a new type `T`, giving back the original
    /// report on any allocation failure.
    ///
    /// Reserves capacity for both the intermediates vector and the new head's
    /// children deque before any boxing begins. Once reservations succeed,
    /// pushing into either container cannot fail — only the individual box
    /// allocations can trigger recovery.
    #[track_caller]
    #[allow(clippy::result_large_err, reason = "cannot allocate err on the heap")]
    pub fn try_change_context<T>(self, context: T) -> Result<Report<T>, ChangeContextError<C, T>>
    where
        T: Error + Send + Sync + 'static,
    {
        let Self {
            head: old_head,
            mut peers,
            lost_peers,
            capacity,
        } = self;

        // Total frames to demote: all peers + old_head.
        let total = peers.len().saturating_add(1);

        // ── Upfront reservations ────────────────────────────────────────────

        // Create the new head and reserve children capacity before any boxing.
        let mut new_head_sf = StaticFrame::new(ContextFrame::new(context, *Location::caller()));
        new_head_sf.lost_children = lost_peers;
        if new_head_sf.children.try_reserve(total).is_err() {
            let new_ctx = new_head_sf.context.context;
            return Err(ChangeContextError {
                report: Report {
                    head: old_head,
                    peers,
                    lost_peers,
                    capacity,
                },
                context: new_ctx,
            });
        }

        // Allocate intermediates vec with exact capacity. The oldest item is inserted first.
        let mut intermediates: Vec<IntermediateFrame<C>> =
            match <Vec<IntermediateFrame<C>> as TryVec<IntermediateFrame<C>>>::try_with_capacity(
                total,
            ) {
                Ok(v) => v,
                Err(_) => {
                    let new_ctx = new_head_sf.context.context;
                    return Err(ChangeContextError {
                        report: Report {
                            head: old_head,
                            peers,
                            lost_peers,
                            capacity,
                        },
                        context: new_ctx,
                    });
                }
            };

        // ── Boxing phase ────────────────────────────────────────────────────

        // Box peers oldest-first (pop_back). On failure, recover all already-boxed
        // intermediates plus remaining unboxed peers into the original report.
        while let Some(sf) = peers.pop_back() {
            let StaticFrame {
                context: ctx,
                attachments,
                children,
                lost_attachments,
                lost_children,
            } = sf;

            let boxed_ctx = match Box::<ContextFrame<C>>::fallible_new_give_back(ctx) {
                Ok(b) => b,
                Err((ctx_recovered, _)) => {
                    let sf = StaticFrame {
                        context: ctx_recovered,
                        attachments,
                        children,
                        lost_attachments,
                        lost_children,
                    };
                    let report = Self::recover_report(
                        intermediates,
                        Some(sf),
                        peers,
                        old_head,
                        lost_peers,
                        capacity,
                    );
                    let new_ctx = new_head_sf.context.context;
                    return Err(ChangeContextError {
                        report,
                        context: new_ctx,
                    });
                }
            };

            let inter = IntermediateFrame {
                context: boxed_ctx,
                attachments,
                children,
                lost_attachments,
                lost_children,
            };
            // Cannot fail — capacity was reserved above.
            TryVec::try_push_give_back(&mut intermediates, inter)
                .map_err(|(_, e)| e)
                .expect("capacity was reserved");
        }

        // Box old_head last.
        let StaticFrame {
            context: ctx,
            attachments,
            children,
            lost_attachments,
            lost_children,
        } = old_head;

        let boxed_ctx = match Box::<ContextFrame<C>>::fallible_new_give_back(ctx) {
            Ok(b) => b,
            Err((ctx_recovered, _)) => {
                let sf = StaticFrame {
                    context: ctx_recovered,
                    attachments,
                    children,
                    lost_attachments,
                    lost_children,
                };
                let report =
                    Self::recover_report(intermediates, None, peers, sf, lost_peers, capacity);
                let new_ctx = new_head_sf.context.context;
                return Err(ChangeContextError {
                    report,
                    context: new_ctx,
                });
            }
        };

        let inter = IntermediateFrame {
            context: boxed_ctx,
            attachments,
            children,
            lost_attachments,
            lost_children,
        };
        // Cannot fail — capacity was reserved above.
        TryVec::try_push_give_back(&mut intermediates, inter)
            .map_err(|(_, e)| e)
            .expect("capacity was reserved");

        // ── Demotion ────────────────────────────────────────────────────────

        // Convert intermediates to dynamic frames and push into children.
        // Cannot fail — children capacity was reserved above and into_dynamic
        // is a zero-allocation cast.
        for inter in intermediates.into_iter().rev() {
            let df = inter.into_dynamic();
            TryVecDeque::try_push_front_give_back(&mut new_head_sf.children, df)
                .map_err(|(_, e)| e)
                .expect("children capacity was reserved");
        }

        Ok(Report {
            head: new_head_sf,
            peers: VecDeque::new(),
            lost_peers: 0,
            capacity,
        })
    }

    // ── Search ───────────────────────────────────────────────────────────

    /// Returns whether any frame in the report holds a value of type `T`.
    #[must_use]
    pub fn contains<T: Send + Sync + 'static>(&self) -> bool {
        self.head.context.as_any().is::<T>()
            || self.head.contains::<T>()
            || self
                .peers
                .iter()
                .any(|p| p.context.as_any().is::<T>() || p.contains::<T>())
    }

    /// Searches all frames for a value of type `T`, returning the most recent.
    ///
    /// The head (current context) is checked first, then peers front-to-back
    /// (most recent first), then children depth-first.
    #[must_use]
    pub fn downcast_ref<T: Send + Sync + 'static>(&self) -> Option<&T> {
        if let Some(r) = self.head.context.as_any().downcast_ref() {
            return Some(r);
        }
        if let Some(r) = self.head.downcast_ref::<T>() {
            return Some(r);
        }
        for peer in self.peers.iter() {
            if let Some(r) = peer.context.as_any().downcast_ref() {
                return Some(r);
            }
            if let Some(r) = peer.downcast_ref::<T>() {
                return Some(r);
            }
        }
        None
    }

    /// Searches all frames for a value of type `T`, returning the most recent
    /// as a mutable reference.
    #[must_use]
    pub fn downcast_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        // Check head first.
        if let Some(r) = self.head.downcast_mut::<T>() {
            return Some(r);
        }
        // Search peers by iterating through the VecDeque directly.
        // Uses addr_of_mut! to avoid the double-borrow problem of holding
        // &mut self while also borrowing peers mutably.
        for peer in self.peers.iter_mut() {
            if let Some(r) = peer.downcast_mut::<T>() {
                return Some(r);
            }
        }
        None
    }

    // ── Recovery helper ────────────────────────────────────────────────────

    /// Rebuilds the original [`Report<C>`] after a boxing failure during
    /// [`try_change_context`](Self::try_change_context).
    ///
    /// - `intermediates`: already-boxed frames in oldest-peer-first order.
    /// - `trailing`: `Some(sf)` if a peer failed to box (that peer);
    ///   `None` if old_head failed to box.
    /// - `remaining_peers`: peers still in the deque (front=newest, back=next-to-process).
    /// - `head_frame`: the old_head static frame.
    #[allow(clippy::too_many_arguments)]
    fn recover_report(
        intermediates: Vec<IntermediateFrame<C>>,
        trailing: Option<StaticFrame<C>>,
        remaining_peers: VecDeque<StaticFrame<C>>,
        head_frame: StaticFrame<C>,
        lost_peers: usize,
        capacity: Option<usize>,
    ) -> Report<C> {
        // Build peers deque in correct order: [newest_peer, …, oldest_peer].
        // Sources from newest to oldest:
        //   1. remaining_peers (still in deque, front=newest)
        //   2. trailing (the frame whose boxing just failed)
        //   3. intermediates reversed (were boxed oldest-first, so
        //      pop_back yields newest first — exactly what we need)
        let mut peers = remaining_peers;

        if let Some(sf) = trailing {
            let _ = TryVecDeque::try_push_back_give_back(&mut peers, sf);
        }

        for inter in intermediates.into_iter().rev() {
            let sf = inter.recover_static();
            let _ = TryVecDeque::try_push_back_give_back(&mut peers, sf);
        }

        Report {
            head: head_frame,
            peers,
            lost_peers,
            capacity,
        }
    }
}

// ── Iteration ────────────────────────────────────────────────────────────────

/// A view into an error node during tree traversal.
#[derive(Debug)]
pub enum FrameRef<'a, C> {
    /// A typed static frame (head or peer).
    Static(&'a StaticFrame<C>),
    /// A type-erased dynamic frame (child from demotion).
    Dynamic(&'a DynamicFrame),
    /// Synthetic marker for frames lost due to OOM or capacity eviction.
    LostFrames(usize),
}

impl<'a, C> FrameRef<'a, C>
where
    C: Error + Send + Sync + 'static,
{
    /// Returns what kind of item this frame's context is.
    ///
    /// Returns `None` for pseudo-frames ([`FrameRef::LostFrames`]).
    #[must_use]
    pub fn kind(&self) -> Option<crate::frame::ItemKind> {
        match self {
            Self::Static(sf) => Some(sf.context().kind()),
            Self::Dynamic(df) => Some(df.kind()),
            Self::LostFrames(_) => None,
        }
    }

    /// Downcasts the held context to `&T` if the type matches.
    #[must_use]
    pub fn downcast_ref<T: Send + Sync + 'static>(&self) -> Option<&T> {
        match self {
            Self::Static(sf) => sf.context().as_any().downcast_ref(),
            Self::Dynamic(df) => df.context_item().as_any().downcast_ref(),
            Self::LostFrames(_) => None,
        }
    }
}

/// Mutable view into an error node during tree traversal.
#[derive(Debug)]
pub enum FrameRefMut<'a, C> {
    /// A typed static frame (head or peer).
    Static(&'a mut StaticFrame<C>),
    /// A type-erased dynamic frame (child from demotion).
    Dynamic(&'a mut DynamicFrame),
    /// Synthetic marker for frames lost due to OOM or capacity eviction.
    LostFrames(usize),
}

impl<'a, C> FrameRefMut<'a, C>
where
    C: Error + Send + Sync + 'static,
{
    /// Returns what kind of item this frame's context is.
    ///
    /// Returns `None` for pseudo-frames ([`FrameRefMut::LostFrames`]).
    #[must_use]
    pub fn kind(&self) -> Option<crate::frame::ItemKind> {
        match self {
            Self::Static(sf) => Some(sf.context().kind()),
            Self::Dynamic(df) => Some(df.kind()),
            Self::LostFrames(_) => None,
        }
    }

    /// Downcasts the held context to `&mut T` if the type matches.
    #[must_use]
    pub fn downcast_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        match self {
            Self::Static(sf) => sf.context_mut().as_any_mut().downcast_mut(),
            Self::Dynamic(df) => df.context_item_mut().as_any_mut().downcast_mut(),
            Self::LostFrames(_) => None,
        }
    }
}

impl<C> Report<C>
where
    C: Error + Send + Sync + 'static,
{
    /// Returns a depth-first walker over every error node in the report.
    ///
    /// Order: head → head's children (recursively) → next peer → that peer's
    /// children → … → oldest peer → its children. Synthetic
    /// [`FrameRef::LostFrames`] markers are
    /// yielded where frames were lost to OOM or capacity eviction.
    ///
    /// Each item is `(Result<FrameRef<'a>, TryReserveError>, usize)` where the
    /// `usize` is the tree depth (head and peers at 0, children at 1, etc.).
    /// An `Err` variant is emitted when pushing a child iterator onto the internal
    /// stack fails due to allocation pressure; the frame that triggered the failure
    /// is still yielded as `Ok` beforehand, so no frame is silently skipped.
    #[must_use]
    pub fn frames(&'_ self) -> Frames<'_, C> {
        Frames {
            peer_iter: self.peers.iter(),
            head_remaining: true,
            head_ref: &self.head,
            stack: Vec::new(),
            root_lost_emitted: false,
            root_lost_peers: self.lost_peers,
            pending_err: None,
        }
    }

    /// Returns a reverse (chronological) depth-first walker: oldest peer first,
    /// head last. Useful for displaying errors in the order they occurred.
    ///
    /// Each item is `(Result<FrameRef<'a>, TryReserveError>, usize)` where the
    /// `usize` is the tree depth. An `Err` variant is emitted when pushing a child
    /// iterator onto the internal stack fails; the triggering frame is still yielded
    /// as `Ok` beforehand.
    #[must_use]
    pub fn frames_chronological(&'_ self) -> ChronoFrames<'_, C> {
        ChronoFrames {
            peer_iter: self.peers.iter().rev(),
            root_peer_remaining: true,
            head_ref: &self.head,
            stack: Vec::new(),
            root_lost_emitted: false,
            root_lost_peers: self.lost_peers,
            pending_err: None,
        }
    }

    /// Visits every error node depth-first with mutable access.
    ///
    /// Uses internal iteration so that [`&mut FrameRefMut`] cannot escape the
    /// callback — preventing aliasing vulnerabilities.
    ///
    /// Returns [`Err(TryReserveError)`] if stack allocation fails during descent.
    pub fn frames_mut<F, B>(&mut self, visitor: F) -> Result<ControlFlow<B>, TryReserveError>
    where
        F: FnMut(&mut FrameRefMut<'_, C>) -> ControlFlow<B>,
    {
        dfs_mut_report(self, visitor)
    }
}

/// One level of the DFS stack during frame traversal.
struct StackEntry<'a> {
    /// Iterator over the parent's dynamic children.
    iter: alloc::collections::vec_deque::Iter<'a, DynamicFrame>,
    /// Lost children count for the parent whose iterator this entry holds.
    /// Set once when the entry is created; never modified.
    lost_children: usize,
}

/// Depth-first walker over [`FrameRef`]s produced by [`Report::frames`].
///
/// Visits: head → head's children recursively → next peer → that peer's children
/// → … → oldest peer → its children. Yields synthetic [`FrameRef::LostFrames`]
/// markers after visiting real frames at each level, and finally any lost peers
/// from report-level eviction as the last item.
pub struct Frames<'a, C> {
    /// Remaining peers to visit (front = most recent).
    peer_iter: alloc::collections::vec_deque::Iter<'a, StaticFrame<C>>,
    /// Current head ref, yielded on first call.
    head_remaining: bool,
    head_ref: &'a StaticFrame<C>,
    /// Stack of child iterators paired with their parent's lost-children count.
    stack: Vec<StackEntry<'a>>,
    /// Lost peers from report-level capacity eviction. Emitted once at the end.
    root_lost_peers: usize,
    /// Whether we've emitted the final lost-peers marker.
    root_lost_emitted: bool,
    /// Pending error from a failed stack reservation, queued for emission on the
    /// next call so that the frame which triggered the failure is not silently skipped.
    pending_err: Option<(TryReserveError, usize)>,
}

impl<'a, C> Iterator for Frames<'a, C>
where
    C: Error + Send + Sync + 'static,
{
    type Item = (Result<FrameRef<'a, C>, TryReserveError>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        // Emit a pending allocation error before continuing iteration.
        if let Some((e, depth)) = self.pending_err.take() {
            return Some((Err(e), depth));
        }

        // Yield head first.
        if self.head_remaining {
            self.head_remaining = false;
            if !self.head_ref.children().is_empty() {
                let child_iter = self.head_ref.children().iter();
                if let Err((_, e)) = TryVec::try_push_give_back(
                    &mut self.stack,
                    StackEntry {
                        iter: child_iter,
                        lost_children: self.head_ref.lost_children(),
                    },
                ) {
                    self.pending_err = Some((e, 1));
                }
            }
            return Some((Ok(FrameRef::Static(self.head_ref)), 0));
        }

        loop {
            if let Some(top) = self.stack.last_mut() {
                if let Some(df) = top.iter.next() {
                    // Push new stack entry to descend into this child's subtree.
                    if !df.children().is_empty() {
                        let child_iter = df.children().iter();
                        if let Err((_, e)) = TryVec::try_push_give_back(
                            &mut self.stack,
                            StackEntry {
                                iter: child_iter,
                                lost_children: df.lost_children(),
                            },
                        ) {
                            self.pending_err = Some((e, self.stack.len() + 1));
                        }
                    }
                    let depth = self.stack.len();
                    return Some((Ok(FrameRef::Dynamic(df)), depth));
                } else {
                    // Children exhausted — emit lost marker if any.
                    let n = top.lost_children;
                    let depth = self.stack.len();
                    self.stack.pop();
                    if n > 0 {
                        return Some((Ok(FrameRef::LostFrames(n)), depth));
                    }
                    continue;
                }
            }

            // Move to next peer.
            if let Some(peer) = self.peer_iter.next() {
                if !peer.children().is_empty() {
                    let child_iter = peer.children().iter();
                    if let Err((_, e)) = TryVec::try_push_give_back(
                        &mut self.stack,
                        StackEntry {
                            iter: child_iter,
                            lost_children: peer.lost_children(),
                        },
                    ) {
                        self.pending_err = Some((e, self.stack.len() + 1));
                    }
                }
                return Some((Ok(FrameRef::Static(peer)), 0));
            }

            // All frames visited — emit lost peers last, then finish.
            if !self.root_lost_emitted && self.root_lost_peers > 0 {
                self.root_lost_emitted = true;
                let n = self.root_lost_peers;
                return Some((Ok(FrameRef::LostFrames(n)), 0));
            }

            return None;
        }
    }
}

// Errors are emitted directly as `Err` items in the iterator stream;
// no separate take_err method is needed.

/// Chronological walker — iterates peers from oldest to newest first, then
/// yields head and its children last (matching the order errors were pushed).
pub struct ChronoFrames<'a, C> {
    peer_iter: core::iter::Rev<alloc::collections::vec_deque::Iter<'a, StaticFrame<C>>>,
    /// Whether we're still in the peer phase. Once false, we move to head.
    root_peer_remaining: bool,
    head_ref: &'a StaticFrame<C>,
    /// Stack entries paired with per-parent lost-children metadata.
    stack: Vec<StackEntry<'a>>,
    /// Lost peers from report-level capacity eviction. Emitted once at the end.
    root_lost_peers: usize,
    /// Whether we've emitted the final lost-peers marker.
    root_lost_emitted: bool,
    /// Pending error from a failed stack reservation, queued for emission on the
    /// next call so that the frame which triggered the failure is not silently skipped.
    pending_err: Option<(TryReserveError, usize)>,
}

impl<'a, C> Iterator for ChronoFrames<'a, C>
where
    C: Error + Send + Sync + 'static,
{
    type Item = (Result<FrameRef<'a, C>, TryReserveError>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        // Emit a pending allocation error before continuing iteration.
        if let Some((e, depth)) = self.pending_err.take() {
            return Some((Err(e), depth));
        }

        loop {
            if let Some(top) = self.stack.last_mut() {
                if let Some(df) = top.iter.next() {
                    if !df.children().is_empty() {
                        let child_iter = df.children().iter();
                        if let Err((_, e)) = TryVec::try_push_give_back(
                            &mut self.stack,
                            StackEntry {
                                iter: child_iter,
                                lost_children: df.lost_children(),
                            },
                        ) {
                            self.pending_err = Some((e, self.stack.len() + 1));
                        }
                    }
                    let depth = self.stack.len();
                    return Some((Ok(FrameRef::Dynamic(df)), depth));
                } else {
                    let n = top.lost_children;
                    let depth = self.stack.len();
                    self.stack.pop();
                    if n > 0 {
                        return Some((Ok(FrameRef::LostFrames(n)), depth));
                    }
                    continue;
                }
            }

            // While peers remain, pull from the peer iterator.
            if self.root_peer_remaining {
                if let Some(peer) = self.peer_iter.next() {
                    if !peer.children().is_empty() {
                        let child_iter = peer.children().iter();
                        if let Err((_, e)) = TryVec::try_push_give_back(
                            &mut self.stack,
                            StackEntry {
                                iter: child_iter,
                                lost_children: peer.lost_children(),
                            },
                        ) {
                            self.pending_err = Some((e, self.stack.len() + 1));
                        }
                    }
                    return Some((Ok(FrameRef::Static(peer)), 0));
                } else {
                    // Peers exhausted — transition to head.
                    self.root_peer_remaining = false;
                    if !self.head_ref.children().is_empty() {
                        let child_iter = self.head_ref.children().iter();
                        if let Err((_, e)) = TryVec::try_push_give_back(
                            &mut self.stack,
                            StackEntry {
                                iter: child_iter,
                                lost_children: self.head_ref.lost_children(),
                            },
                        ) {
                            self.pending_err = Some((e, 1));
                        }
                    }
                    return Some((Ok(FrameRef::Static(self.head_ref)), 0));
                }
            }

            // All frames visited — emit lost peers last, then finish.
            if !self.root_lost_emitted && self.root_lost_peers > 0 {
                self.root_lost_emitted = true;
                let n = self.root_lost_peers;
                return Some((Ok(FrameRef::LostFrames(n)), 0));
            }

            // Head is done, nothing left.
            return None;
        }
    }
}

// ── Peer iterators ───────────────────────────────────────────────────────────

/// Iterator over references to all peers in a report (head first, then most
/// recent peer to oldest).
pub struct PeerIter<'a, C> {
    head: Option<&'a StaticFrame<C>>,
    peers: alloc::collections::vec_deque::Iter<'a, StaticFrame<C>>,
    _phantom: PhantomData<&'a C>,
}

impl<'a, C> Iterator for PeerIter<'a, C> {
    type Item = &'a C;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(head) = self.head.take() {
            return Some(head.context().context());
        }
        self.peers.next().map(|sf| sf.context().context())
    }
}

/// Iterator over mutable references to all peers in a report.
pub struct PeerIterMut<'a, C> {
    head: Option<&'a mut StaticFrame<C>>,
    peers: alloc::collections::vec_deque::IterMut<'a, StaticFrame<C>>,
}

impl<'a, C> Iterator for PeerIterMut<'a, C> {
    type Item = &'a mut C;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(head) = self.head.take() {
            return Some(head.context_mut().context_mut());
        }
        self.peers.next().map(|sf| sf.context_mut().context_mut())
    }
}

// ── Mutable DFS helper ───────────────────────────────────────────────────────

fn dfs_mut_report<C, F, B>(
    report: &mut Report<C>,
    mut visitor: F,
) -> Result<ControlFlow<B>, TryReserveError>
where
    C: Error + Send + Sync + 'static,
    F: FnMut(&mut FrameRefMut<'_, C>) -> ControlFlow<B>,
{
    // Visit head.
    match visitor(&mut FrameRefMut::Static(&mut report.head)) {
        ControlFlow::Break(b) => return Ok(ControlFlow::Break(b)),
        ControlFlow::Continue(()) => {}
    }

    // Stack holds entries for dynamic frame children at various depths.
    let mut stack: Vec<DfsMutEntry> = Vec::new();

    // Seed with head's children.
    seed_stack_from_deque(&mut stack, &mut report.head.children)?;

    // Process peers by iterating through the VecDeque's internal buffers.
    // We use addr_of_mut! to avoid the double-borrow problem of holding
    // &mut report (for the stack seeding) while also iterating peers.
    let peers_ptr = core::ptr::addr_of_mut!(report.peers);
    for peer in unsafe { (*peers_ptr).iter_mut() } {
        match visitor(&mut FrameRefMut::Static(peer)) {
            ControlFlow::Break(b) => return Ok(ControlFlow::Break(b)),
            ControlFlow::Continue(()) => {}
        }
        seed_stack_from_deque(&mut stack, peer.children_mut())?;
    }

    // Now drain the dynamic frame stack depth-first.
    loop {
        let df = match stack.last_mut() {
            Some(entry) => match entry.iter.next() {
                Some(frame) => frame,
                None => {
                    stack.pop();
                    continue;
                }
            },
            None => return Ok(ControlFlow::Continue(())),
        };

        // Push this frame's children onto stack.
        if !df.children().is_empty() {
            seed_stack_from_deque(&mut stack, df.children_mut())?;
        }

        match visitor(&mut FrameRefMut::Dynamic(df)) {
            ControlFlow::Break(b) => return Ok(ControlFlow::Break(b)),
            ControlFlow::Continue(()) => {}
        }
    }
}

struct DfsMutEntry {
    iter: alloc::collections::vec_deque::IterMut<'static, DynamicFrame>,
}

fn seed_stack_from_deque(
    stack: &mut Vec<DfsMutEntry>,
    deque: &mut VecDeque<DynamicFrame>,
) -> Result<(), TryReserveError> {
    if !deque.is_empty() {
        // SAFETY: We transmute the iterator's lifetime from the local borrow
        // of `deque` to 'static. This is sound because:
        // 1. Each DfsMutEntry lives only on `stack`, which is local to dfs_mut_report.
        // 2. The iterator borrows into `deque`, which is owned by a DynamicFrame or
        //    StaticFrame that itself lives inside `report` — outliving this function.
        // 3. The stack is drained before dfs_mut_report returns, so no dangling refs escape.
        let iter = unsafe {
            core::mem::transmute::<
                alloc::collections::vec_deque::IterMut<'_, DynamicFrame>,
                alloc::collections::vec_deque::IterMut<'static, DynamicFrame>,
            >(deque.iter_mut())
        };
        let entry = DfsMutEntry { iter };
        if let Err((_, e)) = TryVec::try_push_give_back(stack, entry) {
            return Err(e);
        }
    }
    Ok(())
}

// ── From impl ────────────────────────────────────────────────────────────────

impl<C> From<C> for Report<C>
where
    C: Error + Send + Sync + 'static,
{
    #[track_caller]
    fn from(context: C) -> Self {
        Self::new(context)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::ItemKind;

    #[derive(Debug)]
    struct TestError(&'static str);
    impl core::fmt::Display for TestError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl Error for TestError {}

    #[derive(Debug)]
    struct OtherError(&'static str);
    impl core::fmt::Display for OtherError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl Error for OtherError {}

    #[test]
    fn report_new_stores_context_and_location() {
        let report = Report::new(TestError("oops"));
        assert_eq!(report.current_context().0, "oops");
        assert!(report.segment().is_none());
        let loc = report.location();
        assert!(!loc.file().is_empty());
        assert!(loc.line() > 0);
    }

    #[test]
    fn report_with_segment_stores_label() {
        let report = Report::with_segment(TestError("fail"), "parsing config");
        assert_eq!(report.segment(), Some("parsing config"));
    }

    #[test]
    fn attach_segment_sets_label() {
        let report = Report::new(TestError("x")).attach_segment("loading data");
        assert_eq!(report.segment(), Some("loading data"));
    }

    #[test]
    fn attach_adds_to_head_attachments() {
        let report = Report::new(TestError("root")).attach("extra info");
        assert_eq!(report.head.attachments.len(), 1);
    }

    #[test]
    fn try_attach_success() {
        let result = Report::new(TestError("root")).try_attach(42i32);
        assert!(result.is_ok());
        let report = result.unwrap();
        assert_eq!(report.head.attachments.len(), 1);
    }

    #[test]
    fn push_adds_peer() {
        let report = Report::new(TestError("first")).push(TestError("second"));
        assert_eq!(report.peer_count(), 2);
        // Most recent peer is at front of deque.
        assert_eq!(
            report.peers.front().unwrap().context().context().0,
            "second"
        );
    }

    #[test]
    fn try_push_at_capacity_fails() {
        let report_result = Report::new(TestError("first"))
            .with_capacity(1)
            .try_push(TestError("second"));
        assert!(report_result.is_err());
        let (report, sf) = report_result.unwrap_err();
        assert_eq!(sf.context.context.0, "second");
        assert_eq!(report.peer_count(), 1);
    }

    #[test]
    fn push_evicts_oldest_when_capped() {
        let report = Report::new(TestError("first"))
            .with_capacity(2)
            .push(TestError("second"))
            .push(TestError("third"));
        // "first" was evicted because cap=2 means head + 1 peer max.
        assert!(report.lost_peers() >= 1);
        assert!(report.peer_count() <= 2);
    }

    #[test]
    fn change_context_demotes_to_children() {
        let report: Report<OtherError> =
            Report::new(TestError("inner")).change_context(OtherError("outer"));
        assert_eq!(report.current_context().0, "outer");
        assert_eq!(report.head.children().len(), 1);
    }

    #[test]
    fn change_context_transfers_lost_peers() {
        let report: Report<OtherError> = Report::new(TestError("inner"))
            .push(TestError("peer"))
            .change_context(OtherError("outer"));
        assert_eq!(report.current_context().0, "outer");
        // Old peers + head become children.
        assert_eq!(report.head.children().len(), 2);
    }

    #[test]
    fn contains_finds_head_context() {
        let report = Report::new(TestError("found"));
        assert!(report.contains::<TestError>());
        assert!(!report.contains::<OtherError>());
    }

    #[test]
    fn contains_finds_attachment() {
        let report = Report::new(TestError("root")).attach("hello");
        assert!(report.contains::<&'static str>());
    }

    #[test]
    fn downcast_ref_returns_head_context() {
        let report = Report::new(TestError("target"));
        let found = report.downcast_ref::<TestError>();
        assert!(found.is_some());
        assert_eq!(found.unwrap().0, "target");
    }

    #[test]
    fn downcast_ref_searches_demoted_frames() {
        let report: Report<OtherError> =
            Report::new(TestError("deep")).change_context(OtherError("shallow"));
        let inner = report.downcast_ref::<TestError>();
        assert!(inner.is_some());
        assert_eq!(inner.unwrap().0, "deep");
    }

    #[test]
    fn downcast_mut_modifies_head_context() {
        #[derive(Debug)]
        struct MutableError(i32);
        impl core::fmt::Display for MutableError {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "count={}", self.0)
            }
        }
        impl Error for MutableError {}

        let mut report = Report::new(MutableError(42));
        if let Some(e) = report.downcast_mut::<MutableError>() {
            e.0 = 99;
        }
        assert_eq!(report.current_context().0, 99);
    }

    #[test]
    fn frames_walks_depth_first() {
        let inner = Report::new(TestError("inner")).attach("inner-attach");
        let report: Report<OtherError> = inner.change_context(OtherError("outer"));

        let mut walker = report.frames();

        // 1. Head: OtherError("outer"), depth 0
        let (frame_result, depth) = walker.next().unwrap();
        assert_eq!(depth, 0);
        let frame = frame_result.unwrap();
        assert!(matches!(frame, FrameRef::Static(_)));

        // 2. Child: demoted frame with TestError("inner"), depth 1
        let (frame_result, depth) = walker.next().unwrap();
        assert_eq!(depth, 1);
        let frame = frame_result.unwrap();
        assert!(matches!(frame, FrameRef::Dynamic(_)));

        // Exhausted (attachments are NOT yielded by frames())
        assert!(walker.next().is_none());
    }

    #[test]
    fn frames_mut_visits_nodes() {
        let inner = Report::new(TestError("inner")).attach("inner-attach");
        let mut report: Report<OtherError> = inner.change_context(OtherError("outer"));

        let mut kinds = alloc::vec::Vec::<ItemKind>::new();
        let _ = report.frames_mut::<_, ()>(|fr| {
            if let Some(kind) = fr.kind() {
                kinds.push(kind);
            }
            ControlFlow::Continue(())
        });

        // At minimum: head (Context), child dynamic frame (Context)
        assert!(kinds.len() >= 2);
        assert_eq!(kinds[0], ItemKind::Context);
    }

    #[test]
    fn frames_mut_early_exit() {
        let inner = Report::new(TestError("inner")).attach("inner-attach");
        let mut report: Report<OtherError> = inner.change_context(OtherError("outer"));

        let mut count = 0usize;
        let result = report
            .frames_mut(|_| {
                count += 1;
                if count >= 2 {
                    ControlFlow::Break(count)
                } else {
                    ControlFlow::Continue(())
                }
            })
            .unwrap();
        assert!(matches!(result, ControlFlow::Break(2)));
    }

    #[test]
    fn current_contexts_iterates_peers() {
        let report = Report::new(TestError("first"))
            .push(TestError("second"))
            .push(TestError("third"));

        let contexts: alloc::vec::Vec<_> = report.current_contexts().collect();
        // Head + 2 peers
        assert_eq!(contexts.len(), 3);
        assert_eq!(contexts[0].0, "first");
        assert_eq!(contexts[1].0, "third"); // most recent peer first
        assert_eq!(contexts[2].0, "second");
    }

    #[test]
    fn from_impl_creates_report() {
        let report: Report<TestError> = TestError("from").into();
        assert_eq!(report.current_context().0, "from");
    }

    #[test]
    fn multiple_change_context_builds_tree() {
        let r1 = Report::new(TestError("level1"));
        let r2: Report<OtherError> = r1.change_context(OtherError("level2"));
        let r3: Report<TestError> = r2.change_context(TestError("level3"));

        assert_eq!(r3.current_context().0, "level3");
        assert!(!r3.head.children().is_empty());
        assert!(r3.contains::<OtherError>());
    }

    #[test]
    fn downcast_ref_returns_most_recent() {
        let report: Report<TestError> = Report::new(TestError("inner"))
            .change_context(OtherError("middle"))
            .change_context(TestError("outer"));

        let found = report.downcast_ref::<TestError>();
        assert!(found.is_some());
        assert_eq!(found.unwrap().0, "outer");
    }

    // ── Snapshot display tests ────────────────────────────────────────────
    // Snapshots are written to the build target directory and overwritten on
    // each run to avoid excessive disk consumption. Tests compare the current
    // Report Display output against the saved snapshot file.
    // Skipped under Miri which does not support filesystem access in isolation mode.

    #[cfg(not(miri))]
    fn get_snapshot_dir() -> std::path::PathBuf {
        std::env::var("CARGO_TARGET_DIR")
            .ok()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap()
                    .to_path_buf()
                    .join("target")
            })
            .join("snapshots")
    }

    #[cfg(not(miri))]
    fn write_snapshot(name: &str, content: &str) {
        let dir = get_snapshot_dir();
        std::fs::create_dir_all(&dir).expect("failed to create snapshot directory");
        let path = dir.join(alloc::format!("{name}.snap"));
        std::fs::write(&path, content).expect("failed to write snapshot");
    }

    #[cfg(not(miri))]
    fn read_snapshot(name: &str) -> Option<alloc::string::String> {
        let path = get_snapshot_dir().join(alloc::format!("{name}.snap"));
        std::fs::read_to_string(path).ok()
    }

    #[cfg(not(miri))]
    fn assert_snapshot(name: &str, actual: &str) {
        match read_snapshot(name) {
            Some(expected) if expected == actual => {} // matches
            Some(expected) => {
                write_snapshot(name, actual);
                panic!(
                    "Snapshot mismatch for '{}'.\nExpected:\n---\n{}---\nActual:\n---\n{}---\nSnapshot file updated.",
                    name, expected, actual
                );
            }
            None => {
                write_snapshot(name, actual);
                panic!(
                    "No snapshot found for '{}'. Written initial snapshot.\nOutput:\n---\n{}---",
                    name, actual
                );
            }
        }
    }

    #[cfg(not(miri))]
    #[test]
    fn display_single_frame() {
        let report = Report::new(TestError("something went wrong"));
        let output = alloc::format!("{}", report);
        assert_snapshot("single_frame", &output);
    }

    #[cfg(not(miri))]
    #[test]
    fn display_single_frame_with_segment() {
        let report = Report::with_segment(TestError("parse failed"), "parsing config");
        let output = alloc::format!("{}", report);
        assert_snapshot("single_frame_with_segment", &output);
    }

    #[cfg(not(miri))]
    #[test]
    fn display_with_attachment() {
        let report = Report::new(TestError("root error")).attach("extra context");
        let output = alloc::format!("{}", report);
        assert_snapshot("with_attachment", &output);
    }

    #[cfg(not(miri))]
    #[test]
    fn display_with_multiple_attachments() {
        let report = Report::new(TestError("root error"))
            .attach("detail one")
            .attach(42i32);
        let output = alloc::format!("{}", report);
        assert_snapshot("with_multiple_attachments", &output);
    }

    #[cfg(not(miri))]
    #[test]
    fn display_with_peers() {
        let report = Report::new(TestError("first"))
            .push(TestError("second"))
            .push(TestError("third"));
        let output = alloc::format!("{}", report);
        assert_snapshot("with_peers", &output);
    }

    #[cfg(not(miri))]
    #[test]
    fn display_after_change_context() {
        let report: Report<OtherError> =
            Report::new(TestError("inner error")).change_context(OtherError("outer error"));
        let output = alloc::format!("{}", report);
        assert_snapshot("after_change_context", &output);
    }

    #[cfg(not(miri))]
    #[test]
    fn display_deeply_nested_change_context() {
        let r1 = Report::new(TestError("level 1"));
        let r2: Report<OtherError> = r1.change_context(OtherError("level 2"));
        let r3: Report<TestError> = r2.change_context(TestError("level 3"));
        let output = alloc::format!("{}", r3);
        assert_snapshot("deeply_nested_change_context", &output);
    }

    #[cfg(not(miri))]
    #[test]
    fn display_change_context_with_attachments() {
        let inner = Report::new(TestError("inner")).attach("inner-attach");
        let report: Report<OtherError> = inner.change_context(OtherError("outer"));
        let output = alloc::format!("{}", report);
        assert_snapshot("change_context_with_attachments", &output);
    }

    #[cfg(not(miri))]
    #[test]
    fn display_peers_then_change_context() {
        let report: Report<OtherError> = Report::new(TestError("base"))
            .push(TestError("peer"))
            .change_context(OtherError("top"));
        let output = alloc::format!("{}", report);
        assert_snapshot("peers_then_change_context", &output);
    }

    #[cfg(not(miri))]
    #[test]
    fn display_with_capacity_eviction() {
        let report = Report::new(TestError("first"))
            .with_capacity(2)
            .push(TestError("second"))
            .push(TestError("third"));
        let output = alloc::format!("{}", report);
        assert_snapshot("with_capacity_eviction", &output);
    }

    #[cfg(not(miri))]
    #[test]
    fn display_minimal_report() {
        let report = Report::new(TestError("minimal"));
        let output = alloc::format!("{}", report);
        assert!(!output.is_empty());
        assert_snapshot("minimal_report", &output);
    }

    #[cfg(not(miri))]
    #[test]
    fn display_debug_delegates_to_display() {
        let report = Report::with_segment(TestError("debug test"), "checking debug");
        let display_output = alloc::format!("{}", report);
        let debug_output = alloc::format!("{:?}", report);
        assert_eq!(display_output, debug_output);
    }

    #[cfg(not(miri))]
    #[test]
    fn display_multilevel_tree_with_segments() {
        let r1 = Report::with_segment(TestError("database connection failed"), "db.connect");
        let r2: Report<OtherError> = r1.change_context(OtherError("query execution failed"));
        let r3: Report<TestError> =
            Report::with_segment(TestError("transaction aborted"), "tx.commit");
        let _ = r2;
        let output = alloc::format!("{}", r3);
        assert_snapshot("multilevel_tree_with_segments", &output);
    }

    #[cfg(not(miri))]
    #[test]
    fn display_mixed_error_types_in_tree() {
        let r1 = Report::new(TestError("io error"));
        let r2: Report<OtherError> = r1.change_context(OtherError("network timeout"));
        let r3: Report<TestError> = r2
            .push(OtherError("retry exhausted"))
            .change_context(TestError("service unavailable"));
        let output = alloc::format!("{}", r3);
        assert_snapshot("mixed_error_types_in_tree", &output);
    }
}
