//! Pthread backend for `TryMutex` (macOS, iOS, most other Unix targets).
//!
//! The sys mutex wraps an `OnceBox<pal::Mutex>` that lazily allocates and
//! initialises the backing `pthread_mutex_t` on the first lock. This backend
//! hoists that allocation into the `try_new*` constructors (which therefore
//! return armed mutexes) and exposes it via `try_arm` for repairing mutexes
//! built with plain `Mutex::new`.

use super::heap_lock::{OnceBoxPayload, arm_once_box};
use super::{SysMutexMirror, TryMutex, assert_layout, assume_init_impl};
use crate::alloc::AllocError;
use lang_core::mem::{self, MaybeUninit};
use lang_core::pin::Pin;
use lang_std::sync::Mutex;

/// Mirror of std's private `sys::pal::unix::sync::mutex::Mutex`, likewise
/// generated from the standard library source. Our [`PalMutex`] below is
/// layout-identical to this, so a pointer to one may be read as the other.
type SysPalMutex = rustyfill_sys::std::sys::pal::unix::sync::mutex::Mutex;

/// Rust-side owner of the pthread backend's raw `pthread_mutex_t`.
///
/// A transparent wrapper around the generated [`SysPalMutex`] mirror (itself a
/// faithful copy of std's private `sys::pal::unix::sync::mutex::Mutex`). Because
/// `#[repr(transparent)]` collapses us onto our single `SysPalMutex` field, we are
/// byte-for-byte identical to the mirror — so a heap-allocated `PalMutex` can be
/// published into the `OnceBox` slot and later read back through the mirror
/// without any translation. Wrapping the mirror (rather than redeclaring the raw
/// `pthread_mutex_t`) keeps the layout contract anchored in the generated type.
///
/// The lifetime methods take [`Pin`] to encode the invariant that a live pthread
/// mutex must not be moved. Moving an already-initialised `pthread_mutex_t` is
/// undefined behaviour; std enforces this with `impl !Unpin`, but that requires
/// nightly. We instead carry the pinning discipline explicitly: we always hold the
/// value behind a `Pin<Box<_>>` from allocation onward and never move it, which is
/// what makes calling [`Self::activate`] / [`Self::deactivate`] sound.
#[repr(transparent)]
struct PalMutex {
    inner: SysPalMutex,
}

unsafe impl Send for PalMutex {}
unsafe impl Sync for PalMutex {}

impl PalMutex {
    /// Construct the wrapper holding a statically-initialised placeholder.
    ///
    /// `PTHREAD_MUTEX_INITIALIZER` is valid at every address, so this is safe to
    /// build anywhere before [`Self::activate`] upgrades it to a real mutex.
    const fn new() -> Self {
        PalMutex {
            inner: SysPalMutex {
                inner: lang_core::cell::UnsafeCell::new(libc::PTHREAD_MUTEX_INITIALIZER),
            },
        }
    }

    /// Address of the underlying `pthread_mutex_t`.
    #[inline]
    fn raw(&self) -> *mut libc::pthread_mutex_t {
        self.inner.inner.get()
    }

    /// Activate the placeholder into a real `pthread_mutex_t` in place with
    /// `PTHREAD_MUTEX_NORMAL`.
    ///
    /// Mirrors std's `pal::Mutex::init`: creating the mutex with
    /// `PTHREAD_MUTEX_NORMAL` makes same-thread re-locking deadlock (detectable)
    /// rather than exhibit the UB of default-type mutexes (see
    /// rust-lang/rust#33770). Requires `Pin<&mut Self>` because moving the
    /// struct after this call would relocate a live mutex.
    ///
    /// # Safety
    ///
    /// May only be called once per instance, and only while the instance is
    /// still at the address it was created at (guaranteed by `Pin`).
    pub(super) unsafe fn activate(self: Pin<&mut Self>) {
        // SAFETY: `self` is a valid, uniquely-located `PalMutex`; the attribute
        // object is local and freshly allocated on the stack.
        unsafe {
            let mut attr = MaybeUninit::<libc::pthread_mutexattr_t>::uninit();
            let rc = libc::pthread_mutexattr_init(attr.as_mut_ptr());
            debug_assert_eq!(rc, 0);
            let rc = libc::pthread_mutexattr_settype(attr.as_mut_ptr(), libc::PTHREAD_MUTEX_NORMAL);
            debug_assert_eq!(rc, 0);
            let rc = libc::pthread_mutex_init(self.raw(), attr.as_ptr());
            debug_assert_eq!(rc, 0);
            let rc = libc::pthread_mutexattr_destroy(attr.as_mut_ptr());
            debug_assert_eq!(rc, 0);
        }
    }

    /// Deactivate (destroy) the `pthread_mutex_t` in place.
    ///
    /// Mirrors std's `Drop for pal::Mutex`. Requires `Pin<&Self>` for the same
    /// reason as [`Self::activate`]: the mutex must still be where it was created.
    ///
    /// # Safety
    ///
    /// The mutex must be unlocked and must have been created at this address.
    pub(super) unsafe fn deactivate(self: Pin<&Self>) {
        // SAFETY: `self` is an unlocked `PalMutex` at its original address.
        let rc = unsafe { libc::pthread_mutex_destroy(self.raw()) };
        if cfg!(any(target_os = "aix", target_os = "dragonfly")) {
            // On AIX and DragonFly, destroying a mutex that was only ever built
            // with PTHREAD_MUTEX_INITIALIZER (never locked or re-init'd) returns
            // EINVAL. See std's Drop impl for the same caveat.
            debug_assert!(rc == 0 || rc == libc::EINVAL);
        } else {
            debug_assert_eq!(rc, 0);
        }
    }
}

// SAFETY: `PalMutex` satisfies the [`OnceBoxPayload`] contract. `new` yields a
// statically-initialised placeholder (`PTHREAD_MUTEX_INITIALIZER`) that is valid
// at any address before activation; `activate` runs `pthread_mutex_init` exactly
// once at a stable (pinned) address; `deactivate` runs `pthread_mutex_destroy` on
// an unlocked mutex still at its original address. The pinning discipline in
// [`arm_once_box`] enforces the "never moved after creation" invariant these
// require.
unsafe impl OnceBoxPayload for PalMutex {
    #[inline]
    fn new() -> Self {
        PalMutex::new()
    }
    #[inline]
    unsafe fn activate(self: Pin<&mut Self>) {
        unsafe { PalMutex::activate(self) }
    }
    #[inline]
    unsafe fn deactivate(self: Pin<&Self>) {
        unsafe { PalMutex::deactivate(self) }
    }
}

impl<T: ?Sized> TryMutex<T> for Mutex<T> {
    fn try_new(value: T) -> Result<Mutex<T>, AllocError>
    where
        T: Sized,
    {
        // Allocate the backend eagerly, then install the value and activate.
        let uninit = Self::try_new_uninit()?;
        let mut filled = uninit;
        filled.get_mut().unwrap().write(value);
        // SAFETY: we just wrote the data slot above.
        Ok(unsafe { Self::assume_init(filled) })
    }

    fn try_new_give_back(value: T) -> Result<Mutex<T>, (T, AllocError)>
    where
        T: Sized,
    {
        // On allocation failure hand the original value back to the caller.
        let uninit = match Self::try_new_uninit() {
            Ok(u) => u,
            Err(e) => return Err((value, e)),
        };
        let mut filled = uninit;
        filled.get_mut().unwrap().write(value);
        // SAFETY: we just wrote the data slot above.
        Ok(unsafe { Self::assume_init(filled) })
    }

    fn try_new_uninit() -> Result<Mutex<MaybeUninit<T>>, AllocError>
    where
        T: Sized,
    {
        // Start from a real, freshly-constructed mutex (its `OnceBox` is null)
        // so the payload bytes are whatever `Mutex::new` writes; then arm the
        // backend eagerly. On allocation failure the slot stays null, so the
        // native lazy path still works on first lock and dropping leaks nothing.
        let this = Mutex::new(MaybeUninit::uninit());
        // SAFETY: `this` is a fresh, uniquely-owned mutex whose OnceBox slot is
        // null and cannot be observed by any other thread yet. Viewing its
        // leading pointer word as an `AtomicPtr` is sound (layout-identical).
        let slot = unsafe { oncebox_slot(&this) };
        match unsafe { arm_once_box::<PalMutex>(slot) } {
            Ok(()) => Ok(this),
            Err(e) => {
                drop(this);
                Err(e)
            }
        }
    }

    fn try_new_zeroed() -> Result<Mutex<MaybeUninit<T>>, AllocError>
    where
        T: Sized,
    {
        let this = Mutex::new(MaybeUninit::zeroed());
        // SAFETY: as in `try_new_uninit`.
        let slot = unsafe { oncebox_slot(&this) };
        match unsafe { arm_once_box::<PalMutex>(slot) } {
            Ok(()) => Ok(this),
            Err(e) => {
                drop(this);
                Err(e)
            }
        }
    }

    unsafe fn assume_init(this: Mutex<MaybeUninit<T>>) -> Mutex<T>
    where
        T: Sized,
    {
        unsafe { assume_init_impl(this) }
    }

    fn try_arm(&self) -> Result<(), AllocError> {
        // Arming mutates only the OnceBox pointer cell, which is interior-
        // mutable (an atomic), so a shared reference suffices — exactly how
        // std's `OnceBox::get_or_init(&self)` works. Works for unsized payloads
        // too: the surgery never touches the data slot.
        let slot = unsafe { oncebox_slot(self) };
        // SAFETY: `self` is a valid `Mutex<T>`; arming only touches the leading
        // OnceBox pointer cell through its atomic. If it is already armed this
        // is a no-op.
        unsafe { arm_once_box(slot) }
    }
}

/// Locate the `OnceBox` pointer cell backing a `Mutex<_>`'s pthread backend and
/// view it as an [`lang_core::sync::atomic::AtomicPtr`] so we can perform the
/// load/CAS that std's own `OnceBox::initialize` performs.
///
/// We route through the generated layout mirror first rather than casting the
/// public type straight to a bare pointer: `&Mutex<T>` → `&SysMutexMirror<T>`
/// (layout-identical) → the real mirror field path `.data.pal.ptr`. Only the
/// final step — viewing the mirror's opaque `Atomic<*mut SysPalMutex>` word as a
/// usable `AtomicPtr<PalMutex>` — is a raw reinterpretation, because the mirror's
/// atomic stub exposes no operations. That keeps every intermediate hop anchored
/// in the actual generated struct layout instead of an unstated offset guess.
///
/// # Safety
///
/// The caller must hold a reference to a valid `Mutex<_>` whose `OnceBox`
/// pointer cell is not aliased by any non-atomic access. Both `try_new*`
/// (unique ownership of a fresh mutex) and `try_arm` (shared reference, with all
/// other accesses going through the mutex's own atomics) satisfy this.
pub(super) unsafe fn oncebox_slot<T: ?Sized>(
    this: &Mutex<T>,
) -> &lang_core::sync::atomic::AtomicPtr<PalMutex> {
    // Step 1: public `Mutex<T>` → its layout mirror. Identical field layout
    // (both generated from the same std source), so reinterpreting the shared
    // reference is sound. Proven by the size/alignment assertion below.
    assert_layout::<&Mutex<T>, &SysMutexMirror<T>>();
    let mirror: &SysMutexMirror<T> = unsafe { mem::transmute(this) };

    // Step 2: walk the real mirror fields down to the OnceBox pointer cell. The
    // mirror types this as `*mut SysPalMutex`; our [`PalMutex`] is
    // layout-identical, so we read the same word as `*mut PalMutex`.
    let slot_cell: &rustyfill_sys::std::sync::atomic::Atomic<*mut SysPalMutex> =
        &mirror.inner.pal.ptr;

    // Step 3: the mirror's atomic is a repr(transparent) wrapper over
    // `UnsafeCell<*mut _>` — layout-identical to `AtomicPtr<_>` (a single
    // machine-width pointer). Reinterpret the shared reference as a real
    // `AtomicPtr<PalMutex>` so we can call its atomic methods.
    assert_layout::<
        &rustyfill_sys::std::sync::atomic::Atomic<*mut SysPalMutex>,
        &lang_core::sync::atomic::AtomicPtr<PalMutex>,
    >();
    unsafe { mem::transmute(slot_cell) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Layout parity between our Rust-side [`PalMutex`] and the generated
    /// [`SysPalMutex`] mirror is what lets us publish one where the other is
    /// expected. Guard it explicitly.
    #[test]
    fn pal_mutex_layout_matches_mirror() {
        use lang_core::mem::{align_of, size_of};
        assert_eq!(size_of::<PalMutex>(), size_of::<SysPalMutex>());
        assert_eq!(align_of::<PalMutex>(), align_of::<SysPalMutex>());
    }

    /// A pinned `PalMutex` must initialise, be usable for a lock/unlock cycle,
    /// and destroy cleanly — exercising the full pinning discipline on a real
    /// pthread target.
    #[test]
    fn pal_mutex_activate_use_deactivate_roundtrip() {
        use lang_alloc::boxed::Box;
        let mut pinned = Box::pin(PalMutex::new());
        // SAFETY: freshly allocated, not yet initialised, at a stable address.
        unsafe { pinned.as_mut().activate() };

        // Lock and unlock through the raw handle to prove the mutex is live.
        // SAFETY: the mutex was just initialised and is currently unlocked.
        unsafe {
            let rc = libc::pthread_mutex_lock(pinned.raw());
            assert_eq!(rc, 0);
            let rc = libc::pthread_mutex_unlock(pinned.raw());
            assert_eq!(rc, 0);
        }

        // SAFETY: unlocked and at its original address.
        unsafe { pinned.as_ref().deactivate() };
    }

    /// The pthread backend allocates on first lock, so `try_new_give_back` must
    /// report an OOM and hand the original value back to the caller. This test
    /// lives in this module precisely because it is only compiled on the
    /// pthread (allocating) target — no cross-module gating is required.
    #[test]
    fn reports_oom_and_gives_back_value() {
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        // Force the very next heap allocation to fail. The allocating backend
        // reports the error via try_new_give_back, handing the original value
        // back to the caller.
        let r: Result<Mutex<i32>, (i32, AllocError)> =
            with_policy(FailPolicy::fail_next_alloc(), || {
                Mutex::<i32>::try_new_give_back(42)
            });
        let (value_back, _err) = r.unwrap_err();
        assert_eq!(value_back, 42);
    }

    /// std's `Mutex` Debug impl locks internally to inspect the value, which on
    /// this backend triggers the deferred `OnceBox` allocation. Formatting a
    /// *lazy* mutex under OOM must therefore bail via `try_arm()` and print a
    /// placeholder — not abort the process. This test lives in this module
    /// precisely because it is only compiled on the pthread (allocating) target.
    #[test]
    fn try_debug_lazy_mutex_oom_bails_without_aborting() {
        use crate::try_format;
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        // A plain `Mutex::new` is lazy on this backend: nothing allocated yet.
        let lazy = Mutex::new(42i32);
        let dbg = with_policy(FailPolicy::fail_next_alloc(), || {
            // If the impl skipped arming, std's Debug would take the lock here,
            // hit the failed allocation inside OnceBox, and abort the test
            // binary. Surviving this call is the assertion.
            try_format!("{:?}", lazy).unwrap()
        });
        assert!(
            dbg.starts_with("Mutex"),
            "expected non-exhaustive Mutex placeholder, got: {dbg}"
        );
    }
}
}
