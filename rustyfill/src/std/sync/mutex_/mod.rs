//! Fallible construction and arming for [`lang_std::sync::Mutex`].
//!
//! The platform backends of std's `Mutex` differ in when they allocate:
//!
//! - **allocation-free** (the futex family — Linux, Android, FreeBSD, OpenBSD,
//!   WASM atomics, Hermit, DragonFly, motor; plus Fuchsia, μITRON, Xous, Windows
//!   except Win7, and the single-threaded `no_threads` fallback) — the sys mutex
//!   is a small fixed-size value that carries no heap allocation, so both
//!   construction and `.lock()` never touch the allocator.
//! - **pthread** (macOS, iOS, most other Unix targets) — the sys mutex wraps an
//!   `OnceBox<pal::Mutex>` that lazily allocates and initialises the backing
//!   `pthread_mutex_t` on the *first* lock, so construction looks infallible but
//!   the allocation cost is deferred and can still abort under memory pressure.
//! - **sgx** (Fortanix SGX) — like pthread, the sys mutex wraps an
//!   `OnceBox<SpinMutex<WaitVariable<bool>>>` whose wait-queue is allocated on the
//!   first lock.
//!
//! Only the pthread and sgx backends allocate; every other backend is
//! allocation-free. [`TryMutex`] gives two families of fallible operations:
//!
//! **Construction.** Every `try_new*` constructor performs all the backend
//! allocation up front and returns an *armed* mutex — one whose `.lock()` is
//! guaranteed not to allocate (and therefore not to abort) again. They follow
//! the same conventions as [`TryBox`](crate::alloc::boxed::TryBox):
//!
//! - [`try_new`](TryMutex::try_new) / [`try_new_give_back`](TryMutex::try_new_give_back)
//!   build a fully-initialised `Mutex<T>` holding a value.
//! - [`try_new_uninit`](TryMutex::try_new_uninit) and
//!   [`try_new_zeroed`](TryMutex::try_new_zeroed) return a low-level
//!   `Mutex<MaybeUninit<T>>` whose data slot is left uninitialised or
//!   zero-filled; the caller fills it and calls
//!   [`assume_init`](TryMutex::assume_init) to obtain a typed `Mutex<T>`.
//!
//! **Arming.** A mutex created with plain [`Mutex::new`] on a pthread or sgx
//! target is *not* armed: its backend will still perform the deferred allocation
//! on first lock, which can abort under OOM. [`try_arm`](TryMutex::try_arm)
//! performs that allocation eagerly in place, converting a lazy mutex into an
//! armed one so no later call can fail. On allocation-free backends (and for
//! already-armed mutexes) it is a no-op that always succeeds.
//!
//! **Formatting.** [`TryDebug for Mutex<T>`](crate::try_fmt::TryDebug) arms the
//! mutex before delegating to std's `Debug` impl, because that impl takes the
//! lock internally to inspect the value — and taking the lock is exactly where
//! the pthread/sgx backends perform their deferred allocation. If arming fails
//! (OOM), the impl bails out with a placeholder instead of letting the process
//! abort mid-format.
//!
//! All fallible entry points return [`AllocError`] on failure. See the safety
//! notes inside each backend branch for the invariants that make the
//! raw-pointer surgery sound.

#![allow(unexpected_cfgs, reason = "niche targets absent from the cfg table")]

// Dispatch to exactly one backend module based on which std sys-mutex backend is
// active. We use `cfg_if!` (first-match-wins, lowest-common-denominator MSRV)
// and mirror std's own `sys::sync::mutex` branch order verbatim, because several
// allocation-free targets (Linux, FreeBSD, …) are themselves
// `target_family = "unix"`: their specific `target_os` arms must precede the
// generic unix/pthread arm, or they would be misrouted to the allocating
// backend.
//
// Every branch routes to one of three modules:
//
// - `no_fail` — the futex family (Linux, Android, FreeBSD, OpenBSD, WASM
//   atomics, Hermit, DragonFly, motor), Fuchsia, μITRON, Xous, Windows (with or without
//   Win7), and the single-threaded `no_threads` fallback. Their sys mutex is a
//   small fixed-size value with no heap allocation, so every entry point simply
//   delegates to `Mutex::new`.
// - `pthread` — macOS, iOS, Solaris, NetBSD, AIX, Hurd, Cygwin, Haiku, Redox,
//   emscripten, teeos, and other Unix-family targets. They use
//   `OnceBox<pal::Mutex>`, which allocates on the first lock.
// - `sgx` — Fortanix SGX, which uses `OnceBox<SpinMutex<…>>`, also allocating.
cfg_if! {
    if #[cfg(any(
        all(target_os = "windows", not(target_vendor = "win7")),
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "motor",
        target_os = "dragonfly",
        all(target_family = "wasm", target_feature = "atomics"),
        target_os = "hermit",
        target_os = "fuchsia",
    ))] {
        mod no_fail;
    } else if #[cfg(any(
        target_family = "unix",
        target_os = "teeos",
    ))] {
        mod heap_lock;
        mod pthread;
    } else if #[cfg(all(target_os = "windows", target_vendor = "win7"))] {
        mod no_fail;
    } else if #[cfg(all(target_vendor = "fortanix", target_env = "sgx"))] {
        mod heap_lock;
        mod sgx;
    } else if #[cfg(any(target_os = "solid_asp3", target_os = "xous"))] {
        mod no_fail;
    } else {
        mod no_fail;
    }
}

use crate::alloc::AllocError;
use crate::try_fmt::{TryDebug, TryDisplay};
use cfg_if::cfg_if;
use lang_core::fmt;
use lang_core::mem::{self, MaybeUninit};
use lang_std::sync::{Mutex, MutexGuard};

/// Layout mirror of std's public `sync::Mutex<T>` (the poisoned variant),
/// generated by `rustyfill-sys` from the standard library source. Its fields
/// are public here, which lets us construct and inspect the exact byte layout
/// of the real type without going through unstable APIs. Used by the allocating
/// backends (pthread / sgx) to route pointer surgery through the real struct
/// layout, and by the layout-parity tests.
#[allow(unused, reason = "used by child modules")]
pub(crate) type SysMutexMirror<T> = rustyfill_sys::std::sync::poison::mutex::Mutex<T>;

/// Fallible construction and arming for [`Mutex`].
///
/// Implemented for [`Mutex<T>`]. Mirrors the [`TryBox`](crate::alloc::boxed::TryBox)
/// pattern: the allocating entry points are fallible; all other mutex behaviour
/// (locking, unlocking, poisoning) delegates to the standard library.
///
/// Every `try_new*` constructor returns an **armed** mutex — one whose `.lock()`
/// will not allocate again. A mutex made with plain [`Mutex::new`] can be
/// upgraded in place via [`try_arm`](Self::try_arm).
///
/// # Examples
///
/// ```rust
/// use rustyfill::prelude::*;
/// use std::sync::Mutex;
///
/// // One-shot: armed mutex holding a value.
/// let m: Mutex<i32> = Mutex::try_new(42).expect("alloc");
/// assert_eq!(*m.lock().unwrap(), 42);
///
/// // Low-level two-phase: allocate an armed shell, fill it, activate it.
/// let mut uninit: Mutex<core::mem::MaybeUninit<i32>> =
///     Mutex::<i32>::try_new_uninit().expect("alloc");
/// *uninit.get_mut().unwrap() = core::mem::MaybeUninit::new(7); // write the slot
/// let m = unsafe { Mutex::assume_init(uninit) };
/// assert_eq!(*m.lock().unwrap(), 7);
///
/// // Repair a lazily-constructed mutex so its first lock cannot abort.
/// let lazy = Mutex::new(0u8);
/// Mutex::try_arm(&lazy).expect("arm");
/// *lazy.lock().unwrap() += 1;
/// ```
///
/// The payload bound is relaxed to `?Sized` so that [`try_arm`](Self::try_arm)
/// is available for fat-payload mutexes (`Mutex<dyn Trait>`): arming only ever
/// touches the backend's interior-mutable pointer cell and never the data
/// slot, so it needs no knowledge of the payload's size. Every other method
/// carries its own `T: Sized` bound — either because it moves a whole value
/// into the data slot (`try_new`, `try_new_give_back`, `assume_init`) or
/// because it constructs a `MaybeUninit<T>` shell, which is itself sized-only
/// (`try_new_uninit`, `try_new_zeroed` and their aliases).
///
/// This trait deliberately has no supertraits. A `: Sized` bound on the
/// implementor looks redundant (every `Mutex<T>` is `Sized` anyway), but rustc
/// currently refuses to prove `Mutex<T>: Sized` under a `?Sized` payload in
/// both the impl header and method resolution, rejecting otherwise-valid code.
/// Keeping the bound off the trait sidesteps that limitation; the guarantee
/// still holds in practice because the only implementors are concrete
/// `Mutex<T>` values.
pub trait TryMutex<T: ?Sized> {
    /// Fallibly construct a fully-initialised, unlocked **armed** `Mutex<T>`
    /// holding `value`.
    ///
    /// Requires `T: Sized` because it moves a whole value into the data slot.
    ///
    /// Performs every allocation the platform backend needs up front, so the
    /// returned mutex's `.lock()` will never allocate again. Returns
    /// [`AllocError`] if the backend allocation fails.
    fn try_new(value: T) -> Result<Mutex<T>, AllocError>
    where
        T: Sized;

    /// Like [`try_new`](Self::try_new) but returns ownership of `value` back on
    /// failure, so no data is lost to an OOM.
    ///
    /// Requires `T: Sized` because it moves a whole value into the data slot.
    fn try_new_give_back(value: T) -> Result<Mutex<T>, (T, AllocError)>
    where
        T: Sized;

    /// Fallibly construct an **armed** `Mutex<MaybeUninit<T>>` whose data slot
    /// is left uninitialised.
    ///
    /// Requires `T: Sized` because the shell payload `MaybeUninit<T>` is
    /// sized-only.
    ///
    /// This is the low-level primitive: all backend allocation happens here, so
    /// the returned mutex is armed. Fill the slot through
    /// [`Mutex::get_mut`](lang_std::sync::Mutex::get_mut) and convert with
    /// [`assume_init`](Self::assume_init).
    fn try_new_uninit() -> Result<Mutex<MaybeUninit<T>>, AllocError>
    where
        T: Sized;

    /// Fallibly construct an **armed** `Mutex<MaybeUninit<T>>` whose data slot
    /// is zero-filled.
    ///
    /// Requires `T: Sized` because the shell payload `MaybeUninit<T>` is
    /// sized-only.
    ///
    /// The slot holds all-zero bytes rather than being left uninitialised. This
    /// is only meaningful for types where an all-zero bit pattern is a valid
    /// value (ints, arrays, pointers, ZSTs); the caller is responsible for that
    /// precondition before calling [`assume_init`](Self::assume_init).
    fn try_new_zeroed() -> Result<Mutex<MaybeUninit<T>>, AllocError>
    where
        T: Sized;

    /// Convert a `Mutex<MaybeUninit<T>>` whose data slot has been written into
    /// a real `Mutex<T>`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the data slot has been fully initialised
    /// (e.g. via `MaybeUninit::write` or `MaybeUninit::zeroed`) before calling
    /// this. Reading an unwritten slot is undefined behaviour. Because the input
    /// type is `Mutex<MaybeUninit<T>>`, the compiler enforces that you obtained
    /// it from [`try_new_uninit`](Self::try_new_uninit) or
    /// [`try_new_zeroed`](Self::try_new_zeroed) — there is no other source.
    ///
    /// Requires `T: Sized` because the retag is a move of the whole value.
    unsafe fn assume_init(this: Mutex<MaybeUninit<T>>) -> Mutex<T>
    where
        T: Sized;

    /// Arm an existing mutex, performing any deferred backend allocation in
    /// place so that no later call can fail.
    ///
    /// A mutex created with plain [`Mutex::new`] on a pthread or sgx target is
    /// *lazy*: its backing object is not allocated until the first lock, at
    /// which point an OOM would abort the process. Calling this method forces
    /// that allocation now and reports failure via [`AllocError`] instead. After
    /// a successful return, the mutex is armed and `.lock()` is guaranteed
    /// infallible.
    ///
    /// This is idempotent: arming an already-armed mutex (one built by any
    /// `try_new*` constructor, or already locked once) is a no-op that succeeds.
    /// On allocation-free backends construction never allocates, so this always
    /// succeeds.
    ///
    /// Takes a shared reference because the only mutation is to the backend's
    /// interior-mutable atomic pointer cell — the same reason std's
    /// `OnceBox::get_or_init(&self)` needs no exclusive access.
    ///
    /// Unlike every other method here, this one deliberately has **no**
    /// `T: Sized` bound: it never touches the data slot, so fat-payload mutexes
    /// (`Mutex<dyn Trait>`) can be armed too. (The trait object bound on the
    /// implementor still applies, but `Mutex<T>` is always `Sized`, so that is
    /// vacuous in practice.)
    fn try_arm(&self) -> Result<(), AllocError>;

    // ── Aliases with `fallible_` prefix ─────────────────────────────────────

    /// Alias for [`Self::try_new_uninit`].
    fn fallible_new_uninit() -> Result<Mutex<MaybeUninit<T>>, AllocError>
    where
        T: Sized,
    {
        Self::try_new_uninit()
    }

    /// Alias for [`Self::try_new_zeroed`].
    fn fallible_new_zeroed() -> Result<Mutex<MaybeUninit<T>>, AllocError>
    where
        T: Sized,
    {
        Self::try_new_zeroed()
    }
}

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Compile-time proof that two types share size and alignment — the precondition
/// for treating one as a byte-for-byte re-tagging of the other. Mirrors the
/// `const _: () = assert!(…)` idiom used elsewhere in the crate (see
/// [`crate::alloc`]). If a mirror ever drifts from its real counterpart this
/// fails at compile time rather than silently corrupting values at runtime.
pub(crate) const fn assert_layout<A, B>() {
    const {
        assert!(
            mem::size_of::<A>() == mem::size_of::<B>(),
            "assert_layout: A and B differ in size"
        );
        assert!(
            mem::align_of::<A>() == mem::align_of::<B>(),
            "assert_layout: A and B differ in alignment"
        );
    }
}

/// Move a whole value from one layout-identical type to another.
///
/// Copies the bytes out of `src` into a new destination-typed value, then drops
/// `src`'s ownership without running its destructor — i.e. a true move, not a
/// clone followed by a drop. This is sound only when `From` and `To` share size
/// and alignment (checked by the caller via [`assert_layout`]) and the bytes are
/// a valid `To`.
unsafe fn move_retag<From, To>(src: From) -> To {
    // SAFETY: caller guarantees `From` and `To` are size/alignment-identical and
    // that `src`'s bytes form a valid `To`. `transmute_copy` reads those bytes
    // into a fresh `To`; forgetting `src` transfers ownership so its destructor
    // does not run a second time on the now-reinterpreted memory.
    let dest = unsafe { mem::transmute_copy(&src) };
    mem::forget(src);
    dest
}

/// Shared `assume_init` body: reinterpret a `Mutex<MaybeUninit<T>>` whose data
/// slot has been written as a `Mutex<T>`.
///
/// # Safety
///
/// Same contract as [`TryMutex::assume_init`]: the data slot must be fully
/// initialised. `MaybeUninit<T>` and `T` have identical size and alignment, so
/// the retag is a pure tag change over already-valid bytes.
pub(crate) unsafe fn assume_init_impl<T>(this: Mutex<MaybeUninit<T>>) -> Mutex<T>
where
    T: Sized,
{
    // Prove the two payloads are size/alignment-identical before retagging.
    assert_layout::<Mutex<MaybeUninit<T>>, Mutex<T>>();
    unsafe { move_retag(this) }
}

// Arm before delegating. std's `Debug` impl takes the lock internally to
// inspect the value (printing `<locked>` only if it cannot acquire it in
// time), and taking the lock is exactly where the pthread/sgx backends perform
// their deferred `OnceBox` allocation — so formatting a lazy mutex would abort
// on OOM instead of failing gracefully. Arming first moves that allocation
// ahead of time and reports failure via `AllocError`; on allocation-free
// backends it is a no-op that always succeeds. Because `try_arm` works through
// `&self` and never touches the data slot, this covers unsized payloads too.
impl<T: ?Sized + TryDebug> TryDebug for Mutex<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.try_arm().is_err() {
            // Bail without touching the unallocated backend: printing the
            // payload would require locking, which could not have been made
            // safe. A non-exhaustive placeholder keeps the output honest about
            // what we know (a Mutex exists; its contents are uninspected).
            return f.debug_struct("Mutex").finish_non_exhaustive();
        }
        // Armed: the inspection lock below can no longer allocate. Std's Debug
        // is otherwise allocation-free (verified by OOM tests) and already
        // shows "<locked>" when contention prevents inspection.
        fmt::Debug::fmt(self, f)
    }
}

// ── TryDebug / TryDisplay for MutexGuard<'_, T> ─────────────────────────────
// The guard derefs to the locked value, so both impls route through the inner
// type's fallible formatter — full fidelity, no suppressed fields. This mirrors
// std's own Debug/Display impls for the guard, which also forward to `T`.

impl<T: ?Sized + TryDebug> TryDebug for MutexGuard<'_, T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).try_fmt(f)
    }
}

impl<T: ?Sized + TryDisplay> TryDisplay for MutexGuard<'_, T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).try_fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::try_format;
    use lang_alloc::string::String;

    // ── Guard formatting: full fidelity, forwarding to the locked value ─────

    #[test]
    fn mutex_guard_try_debug_forwards() {
        let m = Mutex::new(42i32);
        let g = m.lock().unwrap();
        let dbg = try_format!("{:?}", g).unwrap();
        assert_eq!(dbg, "42");
    }

    #[test]
    fn mutex_guard_try_display_forwards() {
        let m = Mutex::new(String::from("guarded"));
        let g = m.lock().unwrap();
        let disp = try_format!("{}", g).unwrap();
        assert_eq!(disp, "guarded");
    }

    // ── Layout parity: the transmute between the mirror and the real public
    //    Mutex type is only sound if their layouts match exactly. Assert size
    //    and alignment for both sized and unsized payloads.

    #[test]
    fn layout_parity_with_mirror() {
        use lang_core::mem::{align_of, size_of};

        macro_rules! check {
            ($t:ty) => {{
                assert_eq!(
                    size_of::<Mutex<$t>>(),
                    size_of::<SysMutexMirror<$t>>(),
                    stringify!($t)
                );
                assert_eq!(
                    align_of::<Mutex<$t>>(),
                    align_of::<SysMutexMirror<$t>>(),
                    stringify!($t)
                );
            }};
        }
        check!(i32);
        check!(u64);
        check!([u8; 16]);
        check!(());
    }

    #[test]
    fn try_new_one_shot() {
        let m: Mutex<i32> = Mutex::try_new(42).unwrap();
        assert_eq!(*m.lock().unwrap(), 42);
    }

    #[test]
    fn try_new_uninit_then_assume_init() {
        let mut uninit: Mutex<MaybeUninit<i32>> = Mutex::<i32>::try_new_uninit().unwrap();
        uninit.get_mut().unwrap().write(7);
        let m = unsafe { Mutex::assume_init(uninit) };
        assert_eq!(*m.lock().unwrap(), 7);
    }

    #[test]
    fn try_new_uninit_zst() {
        let mut uninit: Mutex<MaybeUninit<()>> = Mutex::<()>::try_new_uninit().unwrap();
        uninit.get_mut().unwrap().write(());
        let m = unsafe { Mutex::assume_init(uninit) };
        drop(m.lock().unwrap());
    }

    #[test]
    fn try_new_zeroed_gives_zero_payload() {
        // Zero-filled slot: an int payload reads back as 0.
        let uninit: Mutex<MaybeUninit<i32>> = Mutex::<i32>::try_new_zeroed().unwrap();
        let m = unsafe { Mutex::assume_init(uninit) };
        assert_eq!(*m.lock().unwrap(), 0);
    }

    #[test]
    fn try_new_zeroed_array_is_all_zeros() {
        let uninit: Mutex<MaybeUninit<[u8; 4]>> = Mutex::<[u8; 4]>::try_new_zeroed().unwrap();
        let m = unsafe { Mutex::assume_init(uninit) };
        assert_eq!(*m.lock().unwrap(), [0u8; 4]);
    }

    #[test]
    fn fallible_alias_matches_try_new_uninit() {
        let mut uninit: Mutex<MaybeUninit<u64>> = Mutex::<u64>::fallible_new_uninit().unwrap();
        uninit.get_mut().unwrap().write(1);
        let m = unsafe { Mutex::assume_init(uninit) };
        assert_eq!(*m.lock().unwrap(), 1);
    }

    #[test]
    fn fallible_alias_matches_try_new_zeroed() {
        let uninit: Mutex<MaybeUninit<i32>> = Mutex::<i32>::fallible_new_zeroed().unwrap();
        let m = unsafe { Mutex::assume_init(uninit) };
        assert_eq!(*m.lock().unwrap(), 0);
    }

    #[test]
    fn try_arm_on_lazy_mutex_succeeds_and_locks() {
        // A plain Mutex::new is lazy on pthread/sgx targets; arming it must
        // succeed and leave it fully lockable. On allocation-free targets it is
        // a trivial no-op. Note: arming takes &self — the mutation is confined
        // to the backend's interior-mutable atomic pointer cell.
        let lazy = Mutex::new(5u32);
        Mutex::try_arm(&lazy).expect("arm");
        *lazy.lock().unwrap() += 1;
        assert_eq!(*lazy.lock().unwrap(), 6);
    }

    #[test]
    fn try_arm_is_idempotent() {
        // Arming twice (and arming an already-armed mutex) must keep succeeding.
        let m = Mutex::try_new(0u8).unwrap();
        Mutex::try_arm(&m).expect("first arm");
        Mutex::try_arm(&m).expect("second arm");
        *m.lock().unwrap() += 1;
        assert_eq!(*m.lock().unwrap(), 1);
    }

    #[test]
    fn try_arm_works_on_unsized_payload() {
        use lang_alloc::boxed::Box;

        // The relaxed ?Sized bound means arming is available for fat-payload
        // mutexes too — the backend surgery never touches the data slot.
        let boxed = Box::new(42i32);
        let m: Box<Mutex<dyn fmt::Debug + Send>> = Box::new(Mutex::new(42i32));
        Mutex::try_arm(&m).expect("arm");
        let g = m.lock().unwrap();
        assert_eq!(
            lang_alloc::format!("{:?}", boxed),
            lang_alloc::format!("{:?}", g)
        )
    }

    #[test]
    fn initialized_mutex_survives_thread_contention() {
        use lang_std::sync::Arc;
        use lang_std::thread;
        use lang_std::vec::Vec;

        let m = Arc::new(Mutex::try_new(0u64).unwrap());
        let mut handles = Vec::new();
        for i in 0..8 {
            let m = Arc::clone(&m);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    *m.lock().unwrap() += i as u64;
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        // Each of the 8 threads adds its index 1000 times.
        let expected: u64 = (0..8).map(|i| i * 1000).sum();
        assert_eq!(*m.lock().unwrap(), expected);
    }
}
