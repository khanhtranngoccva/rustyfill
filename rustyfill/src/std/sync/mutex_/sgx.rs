//! Fortanix SGX backend for `TryMutex`.
//!
//! The SGX sys mutex wraps an `OnceBox<SpinMutex<WaitVariable<bool>>>` that
//! lazily allocates and initialises its backing wait-queue on the *first* lock.
//! Like the pthread backend, this backend hoists that allocation into the
//! `try_new*` constructors (which therefore return armed mutexes) and exposes it
//! via `try_arm` for repairing mutexes built with plain `Mutex::new`.
//!
//! The mechanism is identical to [`super::pthread`]: locate the `OnceBox`
//! pointer cell of the public `Mutex`, then perform the acquire-load /
//! allocate / release-CAS dance that std's own `OnceBox::get_or_init` performs.
//! Only the payload type differs — here it is the SGX `SpinMutex` mirror rather
//! than a `pthread_mutex_t` — so we reuse the shared [`arm_once_box`] helper by
//! pointing it at a layout-compatible Rust-side owner.

use super::heap_lock::{arm_once_box, OnceBoxPayload};
use super::{assume_init_impl, assert_layout, SysMutexMirror, TryMutex};
use crate::alloc::AllocError;
use lang_core::mem::{self, MaybeUninit};
use lang_core::pin::Pin;
use lang_core::ptr;
use lang_std::sync::Mutex;

/// Mirror of std's private `sys::pal::waitqueue::SpinMutex<T>` (the SGX mutex
/// payload), likewise generated from the standard library source. Our
/// [`SgxPayload`] below is layout-identical to this, so a pointer to one may be
/// read as the other.
type SysSgxSpinMutex = rustyfill_sys::std::sys::pal::waitqueue::SpinMutex<
    rustyfill_sys::std::sys::pal::waitqueue::WaitVariable<bool>,
>;

/// Rust-side owner of the SGX backend's raw spin-mutex payload.
///
/// A transparent wrapper around the generated [`SysSgxSpinMutex`] mirror, so it
/// is byte-for-byte identical to the mirror — a heap-allocated `SgxPayload` can
/// be published into the `OnceBox` slot and later read back through the mirror
/// without translation. Wrapping the mirror keeps the layout contract anchored
/// in the generated type.
#[repr(transparent)]
pub(super) struct SgxPayload {
    inner: SysSgxSpinMutex,
}

unsafe impl Send for SgxPayload {}
unsafe impl Sync for SgxPayload {}

impl SgxPayload {
    /// Construct a zero-initialised placeholder.
    ///
    /// The SGX `SpinMutex` is laid out over a `Cell<usize>`-style word plus a
    /// wait queue; a fresh, never-locked instance is all-zeros, which is exactly
    /// what `SpinMutex::new` produces before any contention. Writing zeros is
    /// therefore a valid "uninitialised but safe" state for our purposes: we
    /// only ever hand the pointer to std's own locking code, which treats the
    /// first lock as acquiring an empty queue.
    const fn new() -> Self {
        SgxPayload {
            inner: unsafe { mem::zeroed() },
        }
    }
}

// SAFETY: `SgxPayload` satisfies the [`OnceBoxPayload`] contract. `new` yields a
// zero-filled spin mutex — a valid placeholder safe at any address before first
// use, since std's own locking code treats the first lock as acquiring an empty
// queue. The SGX `SpinMutex` has no separate init/destroy steps (it is not a
// heap- or kernel-backed object that must be explicitly constructed), so both
// `activate` and `deactivate` are no-ops; the only lifecycle work is the
// allocation/reclamation handled by the shared [`arm_once_box`] machinery.
unsafe impl OnceBoxPayload for SgxPayload {
    #[inline]
    fn new() -> Self {
        SgxPayload::new()
    }
    #[inline]
    unsafe fn activate(self: Pin<&mut Self>) {}
    #[inline]
    unsafe fn deactivate(self: Pin<&Self>) {}
}

impl<T> TryMutex<T> for Mutex<T> {
    fn try_new(value: T) -> Result<Mutex<T>, AllocError> {
        // Allocate the backend eagerly, then install the value and activate.
        let uninit = Self::try_new_uninit()?;
        let mut filled = uninit;
        filled.get_mut().unwrap().write(value);
        // SAFETY: we just wrote the data slot above.
        Ok(unsafe { Self::assume_init(filled) })
    }

    fn try_new_give_back(value: T) -> Result<Mutex<T>, (T, AllocError)> {
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

    fn try_new_uninit() -> Result<Mutex<MaybeUninit<T>>, AllocError> {
        // Start from a real, freshly-constructed mutex (its `OnceBox` is null)
        // so the payload bytes are whatever `Mutex::new` writes; then arm the
        // backend eagerly. On allocation failure the slot stays null, so the
        // native lazy path still works on first lock and dropping leaks nothing.
        let this = Mutex::new(MaybeUninit::uninit());
        // SAFETY: `this` is a fresh, uniquely-owned mutex whose OnceBox slot is
        // null and cannot be observed by any other thread yet. Viewing its
        // leading pointer word as an `AtomicPtr` is sound (layout-identical).
        let slot = unsafe { oncebox_slot(&this) };
        match unsafe { arm_once_box::<SgxPayload>(slot) } {
            Ok(()) => Ok(this),
            Err(e) => {
                drop(this);
                Err(e)
            }
        }
    }

    fn try_new_zeroed() -> Result<Mutex<MaybeUninit<T>>, AllocError> {
        let this = Mutex::new(MaybeUninit::zeroed());
        // SAFETY: as in `try_new_uninit`.
        let slot = unsafe { oncebox_slot(&this) };
        match unsafe { arm_once_box::<SgxPayload>(slot) } {
            Ok(()) => Ok(this),
            Err(e) => {
                drop(this);
                Err(e)
            }
        }
    }

    unsafe fn assume_init(this: Mutex<MaybeUninit<T>>) -> Mutex<T> {
        unsafe { assume_init_impl(this) }
    }

    fn try_arm(&self) -> Result<(), AllocError> {
        // Arming mutates only the OnceBox pointer cell, which is interior-
        // mutable (an atomic), so a shared reference suffices — exactly how
        // std's `OnceBox::get_or_init(&self)` works.
        let slot = unsafe { oncebox_slot(self) };
        // SAFETY: `self` is a valid `Mutex<T>`; arming only touches the leading
        // OnceBox pointer cell through its atomic. If it is already armed this
        // is a no-op.
        unsafe { arm_once_box::<SgxPayload>(slot) }
    }
}

/// Locate the `OnceBox` pointer cell backing a `Mutex<_>`'s SGX backend and view
/// it as an [`lang_core::sync::atomic::AtomicPtr`] so we can perform the
/// load/CAS that std's own `OnceBox::initialize` performs. Mirrors
/// [`super::pthread::oncebox_slot`], routed through the shared layout mirror.
///
/// # Safety
///
/// The caller must hold a reference to a valid `Mutex<_>` whose `OnceBox`
/// pointer cell is not aliased by any non-atomic access. Both `try_new*`
/// (unique ownership of a fresh mutex) and `try_arm` (shared reference, with all
/// other accesses going through the mutex's own atomics) satisfy this.
unsafe fn oncebox_slot<T>(this: &Mutex<T>) -> &lang_core::sync::atomic::AtomicPtr<SgxPayload> {
    // Step 1: public `Mutex<T>` → its layout mirror. Identical field layout
    // (both generated from the same std source), so reinterpreting the shared
    // reference is sound. Proven by the size/alignment assertion below.
    assert_layout::<&Mutex<T>, &SysMutexMirror<T>>();
    let mirror: &SysMutexMirror<T> = unsafe { mem::transmute(this) };

    // Step 2: walk the real mirror fields down to the OnceBox pointer cell. The
    // mirror types this as `*mut <SGX SpinMutex>`; our [`SgxPayload`] is
    // layout-identical, so we read the same word as `*mut SgxPayload`.
    let slot_cell: &rustyfill_sys::std::sync::atomic::Atomic<*mut SysSgxSpinMutex> =
        &mirror.inner.inner.ptr;

    // Step 3: the mirror's atomic is a repr(transparent) wrapper over
    // `UnsafeCell<*mut _>` — layout-identical to `AtomicPtr<_>` (a single
    // machine-width pointer). Reinterpret the shared reference as a real
    // `AtomicPtr<SgxPayload>` so we can call its atomic methods.
    assert_layout::<
        &rustyfill_sys::std::sync::atomic::Atomic<*mut SysSgxSpinMutex>,
        &lang_core::sync::atomic::AtomicPtr<SgxPayload>,
    >();
    unsafe { mem::transmute(slot_cell) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Layout parity between our Rust-side [`SgxPayload`] and the generated
    /// [`SysSgxSpinMutex`] mirror is what lets us publish one where the other is
    /// expected. Guard it explicitly.
    #[test]
    fn sgx_payload_layout_matches_mirror() {
        use lang_core::mem::{align_of, size_of};
        assert_eq!(size_of::<SgxPayload>(), size_of::<SysSgxSpinMutex>());
        assert_eq!(align_of::<SgxPayload>(), align_of::<SysSgxSpinMutex>());
    }

    /// The SGX backend allocates on first lock, so `try_new_give_back` must
    /// report an OOM and hand the original value back to the caller. This test
    /// lives in this module precisely because it is only compiled on the SGX
    /// (allocating) target — no cross-module gating is required.
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
}
