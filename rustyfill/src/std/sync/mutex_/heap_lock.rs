//! Shared machinery for std's lazily-allocated, heap-backed mutex backends.
//!
//! Several of std's sys-mutex backends do not hold their real locking object
//! inline; instead they wrap a [`OnceBox`](lang_std::sync::Mutex) that lazily
//! allocates and initialises the backing object on the *first* lock:
//!
//! - **pthread** — `OnceBox<pal::Mutex>` (a `pthread_mutex_t`), used by macOS,
//!   iOS, Solaris, NetBSD, AIX, Hurd, Cygwin, Haiku, Redox, emscripten, teeos,
//!   and other Unix-family targets.
//! - **sgx** — `OnceBox<SpinMutex<WaitVariable<bool>>>`, used by Fortanix SGX.
//!
//! Because that allocation is deferred, a mutex built with plain
//! [`Mutex::new`] can still abort under memory pressure on its first lock. This
//! module provides the one generic routine both allocating backends need to
//! hoist that allocation into their fallible constructors / [`try_arm`]:
//! [`arm_once_box`], driven by the [`OnceBoxPayload`] lifecycle contract. The
//! pthread and sgx backends each supply their own payload type implementing the
//! contract, so the acquire-load / allocate / release-CAS dance lives here once.
//!
//! The name "heap lock" reflects what this operates on: a mutex whose actual
//! locking state lives on the heap behind a lazily-initialised pointer, as
//! opposed to the allocation-free backends whose entire state is at most a small
//! fixed-size value carried inline.

use crate::alloc::AllocError;
use lang_core::pin::Pin;
use lang_core::ptr;

/// The lifecycle contract shared by every lazily-backed `OnceBox` payload that
/// [`arm_once_box`] knows how to allocate, activate, and reclaim. Both the
/// pthread [`super::pthread::PalMutex`] and the SGX spin-mutex owner
/// ([`super::sgx::SgxPayload`]) implement this, which lets the two backends share
/// one generic implementation of the OnceBox acquire-load / allocate /
/// release-CAS dance.
///
/// # Safety
///
/// Implementors guarantee:
/// - [`OnceBoxPayload::new`] produces a valid *placeholder* that is safe to store
///   at any address before being activated (for pthread this is
///   `PTHREAD_MUTEX_INITIALIZER`; for SGX it is a zero-filled spin mutex).
/// - [`OnceBoxPayload::activate`] turns the placeholder into a fully usable,
///   unlocked backend object in place, and may be called exactly once per
///   instance while it is still at the address it was created at.
/// - [`OnceBoxPayload::deactivate`] tears down an activated object in place; it
///   requires the object to be unlocked and at its original address.
pub(super) unsafe trait OnceBoxPayload: Sized + Send + Sync {
    /// Construct a valid placeholder (safe at any address pre-activation).
    fn new() -> Self;
    /// Activate the placeholder into a live, unlocked backend object in place.
    unsafe fn activate(self: Pin<&mut Self>);
    /// Tear down an activated, unlocked backend object in place.
    unsafe fn deactivate(self: Pin<&Self>);
}

/// Allocate, initialise, and publish a lazily-backed payload into the `OnceBox`
/// pointed to by `slot`.
///
/// Generic over the payload type `P` because the SGX backend shares this exact
/// mechanism with a different payload than the pthread backend. For pthread
/// `P` is [`super::pthread::PalMutex`]; for SGX it is
/// [`super::sgx::SgxPayload`]. The payload contract is captured by
/// [`OnceBoxPayload`].
///
/// The generated mirror's atomic primitive (`rustyfill_sys::...::atomic::Atomic<T>`)
/// is a `#[repr(transparent)]` wrapper around `UnsafeCell<T>` and exposes no
/// atomic operations — it stands in for std's generic atomic purely so the
/// layout mirrors compile downstream. Here we view its single word as a real
/// [`lang_core::sync::atomic::AtomicPtr<P>`] (identical size and alignment, both
/// just a machine-width pointer) so we can perform the load/CAS that std's own
/// `OnceBox::initialize` performs.
///
/// This mirrors std's `OnceBox::get_or_init`: a cheap acquire-load fast path
/// returns immediately if the slot is already populated, and only the slow path
/// allocates. Returns `Ok(())` when the slot ends up populated (either by this
/// call or already by a prior arming / first-lock), and `Err(AllocError)` only
/// when the allocation fails and the slot is left untouched. Idempotent.
///
/// # Safety
///
/// `slot` must reference the `OnceBox` pointer cell of a valid `Mutex<_>`, and
/// the caller must guarantee no other alias to that cell exists except through
/// this atomic. Publishing uses the same Release/Acquire CAS as std's
/// `OnceBox::initialize`.
pub(super) unsafe fn arm_once_box<P>(
    slot: &lang_core::sync::atomic::AtomicPtr<P>,
) -> Result<(), AllocError>
where
    P: OnceBoxPayload,
{
    use crate::alloc::boxed::TryBox;
    use lang_alloc::boxed::Box;
    use lang_core::sync::atomic::Ordering;

    // Fast path: already armed? Nothing to do. Mirrors `get_or_init`'s leading
    // acquire-load.
    if !slot.load(Ordering::Acquire).is_null() {
        return Ok(());
    }

    // Slow path: fallibly allocate the payload behind a `Pin<Box>`. Pinning
    // from the moment of allocation guarantees the backend object is never
    // moved after activation, which is what makes activating it safe. On
    // allocation failure the slot stays null (so the native lazy path still
    // works on first lock) and the error propagates.
    let mut pinned: Pin<Box<P>> = <Box<P> as TryBox<_>>::try_pin(P::new())?;

    // SAFETY: `pinned` is a `Pin<Box<P>>`, so the value is guaranteed to stay at
    // this heap address. `as_mut()` yields a `Pin<&mut P>` for a freshly
    // allocated, not-yet-activated instance — exactly the precondition
    // `P::activate` requires.
    unsafe { pinned.as_mut().activate() };

    // Unwrap the `Pin` back into the plain `Box`, then hand the box to the global
    // allocator's ownership and publish the pointer. Matches std's `initialize`:
    // `Box::into_raw` before the CAS, and on a lost race reclaim + deactivate our
    // box in favour of the winner's.
    //
    // SAFETY: `into_inner_unchecked` requires that the value will not be moved
    // out of its current address after this point. We immediately turn the box
    // into a raw pointer and only ever touch the contained `P` through that
    // stable heap address (via `Pin::new_unchecked` on reclamation), so the
    // pinning invariant holds for the remainder of its life.
    let boxed = unsafe { Pin::into_inner_unchecked(pinned) };
    let new_ptr = Box::into_raw(boxed).cast::<P>();
    match slot.compare_exchange(
        ptr::null_mut(),
        new_ptr,
        Ordering::Release,
        Ordering::Acquire,
    ) {
        Ok(_) => Ok(()),
        Err(winner) => {
            // Lost the race to another thread. Reclaim our freshly-activated
            // backend object, tear it down in place, and free the box.
            let reclaimed = unsafe { Box::from_raw(new_ptr) };
            // SAFETY: `reclaimed` was just reconstituted from the very pointer we
            // published, so the `P` is at its original heap address and is
            // unlocked. `Pin::new_unchecked` is therefore sound, and deactivating
            // the live backend object in place is required before freeing.
            unsafe { Pin::new_unchecked(&*reclaimed).deactivate() };
            drop(reclaimed);
            debug_assert!(!winner.is_null());
            Ok(())
        }
    }
}
