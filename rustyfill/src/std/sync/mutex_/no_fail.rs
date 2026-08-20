//! Allocation-free backend for `TryMutex`.
//!
//! Active on every platform whose std sys mutex never touches the heap during
//! construction or locking: the futex family (Linux, Android, FreeBSD, OpenBSD,
//! WASM atomics, Hermit, DragonFly, motor, and Windows except Win7), Fuchsia,
//! μITRON, Xous, and the single-threaded `no_threads` fallback. Their sys mutex
//! is a small fixed-size value (one or two atomic words, an SRWLock, a lazily-
//! created kernel ID, or a plain `Cell`) that carries no `OnceBox` / `Box`, so
//! `Mutex::new` performs zero allocations and `.lock()` never allocates either.
//!
//! Because there is nothing to arm, every fallible entry point simply delegates
//! to [`Mutex::new`] and always succeeds — no raw-pointer surgery, no unsafety
//! beyond the shared [`assume_init`](super::assume_init_impl) retag.

use super::{assume_init_impl, TryMutex};
use crate::alloc::AllocError;
use lang_core::mem::MaybeUninit;
use lang_std::sync::Mutex;

impl<T> TryMutex<T> for Mutex<T> {
    fn try_new(value: T) -> Result<Mutex<T>, AllocError> {
        // No allocation on this backend; the returned mutex is armed by
        // construction.
        Ok(Mutex::new(value))
    }

    fn try_new_give_back(value: T) -> Result<Mutex<T>, (T, AllocError)> {
        // Infallible here, so the give-back arm is unreachable.
        Ok(Mutex::new(value))
    }

    fn try_new_uninit() -> Result<Mutex<MaybeUninit<T>>, AllocError> {
        // The data slot is `MaybeUninit<T>` (uninit) by construction.
        Ok(Mutex::new(MaybeUninit::uninit()))
    }

    fn try_new_zeroed() -> Result<Mutex<MaybeUninit<T>>, AllocError> {
        // Zero-filled slot; still typed as `MaybeUninit<T>` per the convention.
        Ok(Mutex::new(MaybeUninit::zeroed()))
    }

    unsafe fn assume_init(this: Mutex<MaybeUninit<T>>) -> Mutex<T> {
        unsafe { assume_init_impl(this) }
    }

    fn try_arm(&self) -> Result<(), AllocError> {
        // This backend never defers allocation, so there is nothing to arm.
        // Succeed unconditionally.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The no_fail backend never touches the heap, so construction must succeed
    /// even when the very next allocation is forced to fail. This test lives in
    /// this module precisely because it is only compiled on allocation-free
    /// targets — no cross-module gating is required.
    #[test]
    fn never_allocates_even_under_oom() {
        use rustyfill_test_allocator::{FailPolicy, with_policy};

        // Force the very next heap allocation to fail. The no_fail backend must
        // not allocate at all, so it still succeeds.
        let r: Result<Mutex<MaybeUninit<i32>>, AllocError> = with_policy(
            FailPolicy::fail_next_alloc(),
            <Mutex<i32> as TryMutex<i32>>::try_new_uninit,
        );
        assert!(r.is_ok());

        // And the mutex remains fully functional afterwards.
        let mut uninit = r.unwrap();
        uninit.get_mut().unwrap().write(7);
        let m = unsafe { Mutex::assume_init(uninit) };
        assert_eq!(*m.lock().unwrap(), 7);
    }
}
