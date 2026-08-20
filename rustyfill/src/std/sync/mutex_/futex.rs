//! Futex backend for `TryMutex` (Linux, Android, FreeBSD, OpenBSD, WASM
//! atomics, Hermit, DragonFly, motor).
//!
//! The sys mutex is a single atomic word (`sys::sync::mutex::futex::Mutex {
//! futex: Atomic<u32> }`). Construction never touches the heap, so every
//! fallible entry point is infallible in practice: it always succeeds and the
//! returned mutex is armed by construction.

use super::{assume_init_impl, mirror_atomic, shell, SysInnerMutex, TryMutex};
use crate::alloc::AllocError;
use lang_core::mem::MaybeUninit;
use lang_std::sync::Mutex;

/// Build the futex sys-mutex mirror in its freshly-constructed state,
/// replicating `sys::Mutex::new()` — a single atomic word initialised to
/// UNLOCKED (0), exactly as `futex::Mutex::new` does (`Futex::new(UNLOCKED)`).
pub(super) fn fresh_inner_mutex() -> SysInnerMutex {
    // The mirror's `Futex` is `Atomic<u32>` (the transparent stub), built
    // through its field.
    SysInnerMutex {
        futex: mirror_atomic(0u32),
    }
}

impl<T> TryMutex<T> for Mutex<T> {
    fn try_new(value: T) -> Result<Mutex<T>, AllocError> {
        // No allocation on futex platforms; the shell is armed by construction.
        Ok(shell(value))
    }

    fn try_new_give_back(value: T) -> Result<Mutex<T>, (T, AllocError)> {
        // Infallible here, so the give-back arm is unreachable.
        Ok(shell(value))
    }

    fn try_new_uninit() -> Result<Mutex<MaybeUninit<T>>, AllocError> {
        // The shell's data slot is `MaybeUninit<T>` (uninit) by construction.
        Ok(shell(MaybeUninit::uninit()))
    }

    fn try_new_zeroed() -> Result<Mutex<MaybeUninit<T>>, AllocError> {
        // Zero-filled slot; still typed as `MaybeUninit<T>` per the convention.
        Ok(shell(MaybeUninit::zeroed()))
    }

    unsafe fn assume_init(this: Mutex<MaybeUninit<T>>) -> Mutex<T> {
        unsafe { assume_init_impl(this) }
    }

    fn try_arm(&self) -> Result<(), AllocError> {
        // The futex backend never defers allocation, so there is nothing to
        // arm. Succeed unconditionally.
        Ok(())
    }
}
