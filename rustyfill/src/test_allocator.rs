//! Test-time global allocator that intercepts allocation calls and can be
//! instructed via thread-local state to return null, simulating OOM conditions.
//!
//! Gated behind `#[cfg(test)]`. Only compiled during `cargo test`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::marker::PhantomData;

// ── Failure policy ────────────────────────────────────────────────────────────

/// Describes which kinds of allocation should fail on the current thread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FailPolicy {
    /// Fail the Nth consecutive `alloc` / `alloc_zeroed` call (1-indexed).
    /// `None` means never fail on fresh allocations.
    fail_alloc_at: Option<u64>,
    /// Fail the Nth consecutive `realloc` call (1-indexed).
    /// `None` means never fail on reallocations.
    fail_realloc_at: Option<u64>,
}

impl FailPolicy {
    /// Never fail anything (the default).
    pub const fn nothing() -> Self {
        Self {
            fail_alloc_at: None,
            fail_realloc_at: None,
        }
    }

    /// Fail the very next alloc or alloc_zeroed.
    pub const fn fail_next_alloc() -> Self {
        Self {
            fail_alloc_at: Some(1),
            fail_realloc_at: None,
        }
    }

    /// Fail the very next realloc.
    #[allow(unused)]
    pub const fn fail_next_realloc() -> Self {
        Self {
            fail_alloc_at: None,
            fail_realloc_at: Some(1),
        }
    }

    /// Fail the Nth alloc/alloc_zeroed (1-indexed).
    pub const fn fail_nth_alloc(n: u64) -> Self {
        Self {
            fail_alloc_at: Some(n),
            fail_realloc_at: None,
        }
    }

    /// Fail the Nth realloc (1-indexed).
    #[allow(unused)]
    pub const fn fail_nth_realloc(n: u64) -> Self {
        Self {
            fail_alloc_at: None,
            fail_realloc_at: Some(n),
        }
    }
}

// ── Thread-local state ────────────────────────────────────────────────────────

thread_local! {
    static FAIL_POLICY: Cell<FailPolicy> = Cell::new(FailPolicy::default());
    static ALLOC_INVOCATIONS: Cell<u64> = const { Cell::new(0) };
    static REALLOC_INVOCATIONS: Cell<u64> = const { Cell::new(0) };
}

/// Snapshot of the current thread-local policy and invocation counters.
type PolicySnapshot = (FailPolicy, u64, u64);

fn get_snapshot() -> PolicySnapshot {
    let policy = FAIL_POLICY.with(|p| p.get());
    let alloc_n = ALLOC_INVOCATIONS.with(|c| c.get());
    let realloc_n = REALLOC_INVOCATIONS.with(|c| c.get());
    (policy, alloc_n, realloc_n)
}

fn restore_snapshot(snap: PolicySnapshot) {
    let (policy, alloc_n, realloc_n) = snap;
    FAIL_POLICY.set(policy);
    ALLOC_INVOCATIONS.set(alloc_n);
    REALLOC_INVOCATIONS.set(realloc_n);
}

/// Run `f` with `policy` active on the current thread, restoring the previous
/// policy and counters when `f` returns (whether normally or via panic).
///
/// The invocation counters are reset to zero for the duration of `f` so that
/// "fail the Nth call" semantics start counting from inside the closure.
pub fn with_policy<R>(policy: FailPolicy, f: impl FnOnce() -> R) -> R {
    let prev = get_snapshot();
    FAIL_POLICY.set(policy);
    ALLOC_INVOCATIONS.set(0);
    REALLOC_INVOCATIONS.set(0);
    // Use catch_unwind so we restore even on panic.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    restore_snapshot(prev);
    match result {
        Ok(r) => r,
        Err(e) => std::panic::resume_unwind(e),
    }
}

// ── Custom global allocator ───────────────────────────────────────────────────

/// Wraps the system allocator and returns null when the current thread-local
/// policy dictates, delegating everything else to [`System`].
struct TestAllocator;

unsafe impl GlobalAlloc for TestAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if check_should_fail_alloc() {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if check_should_fail_alloc() {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if check_should_fail_realloc() {
            return std::ptr::null_mut();
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn check_should_fail_alloc() -> bool {
    let at = match FAIL_POLICY.with(|p| p.get().fail_alloc_at) {
        Some(at) => at,
        None => return false,
    };
    ALLOC_INVOCATIONS.with(|c| {
        let n = c.get();
        c.set(n + 1);
        n + 1 == at
    })
}

fn check_should_fail_realloc() -> bool {
    let at = match FAIL_POLICY.with(|p| p.get().fail_realloc_at) {
        Some(at) => at,
        None => return false,
    };
    REALLOC_INVOCATIONS.with(|c| {
        let n = c.get();
        c.set(n + 1);
        n + 1 == at
    })
}

#[global_allocator]
static GLOBAL: TestAllocator = TestAllocator;

// ── Guard ─────────────────────────────────────────────────────────────────────

/// RAII guard that installs an allocation-failure policy for the current
/// thread and restores the previous policy when dropped.
///
/// Snapshots the active policy and invocation counters at construction time,
/// so nested guards correctly restore the outer guard's state on drop.
///
/// # Thread affinity
///
/// This type is `!Send` and `!Sync` because it manipulates thread-local state.
/// It must not be moved to another thread or shared across threads.
///
/// # Async safety
///
/// Must **not** be held across `.await` points. An async task may resume on a
/// different OS thread, silently observing a different (cleared) policy. If you
/// need failure injection in async code, create a fresh guard after each await.
pub struct FailAllocGuard {
    snapshot: PolicySnapshot,
    _marker: PhantomData<*const ()>,
}

// PhantomData<*const ()> guarantees !Send + !Sync.

impl FailAllocGuard {
    /// Install `policy` for the current thread, resetting invocation counters
    /// to zero. The previous policy and counters are saved and restored when
    /// this guard is dropped.
    fn install(policy: FailPolicy) -> Self {
        let snapshot = get_snapshot();
        FAIL_POLICY.set(policy);
        ALLOC_INVOCATIONS.set(0);
        REALLOC_INVOCATIONS.set(0);
        Self {
            snapshot,
            _marker: PhantomData,
        }
    }

    /// Convenience: fail the next allocation on this thread.
    #[allow(unused)]
    fn fail_next_alloc() -> Self {
        Self::install(FailPolicy::fail_next_alloc())
    }

    /// Convenience: fail the next reallocation on this thread.
    #[allow(unused)]
    fn fail_next_realloc() -> Self {
        Self::install(FailPolicy::fail_next_realloc())
    }

    /// Convenience: fail the Nth allocation on this thread.
    #[allow(unused)]
    fn fail_nth_alloc(n: u64) -> Self {
        Self::install(FailPolicy::fail_nth_alloc(n))
    }

    /// Convenience: fail the Nth reallocation on this thread.
    #[allow(unused)]
    fn fail_nth_realloc(n: u64) -> Self {
        Self::install(FailPolicy::fail_nth_realloc(n))
    }
}

impl Drop for FailAllocGuard {
    fn drop(&mut self) {
        restore_snapshot(self.snapshot);
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::AllocError;
    use crate::boxed::TryBox;

    // ── Guard snapshot/restore ─────────────────────────────────────────────

    #[test]
    fn guard_restores_previous_policy() {
        let outer = FailAllocGuard::install(FailPolicy::fail_nth_alloc(2));
        let _inner = FailAllocGuard::install(FailPolicy::fail_next_alloc());
        let r: Result<Box<i32>, AllocError> = <Box<i32> as TryBox<i32>>::fallible_new(1);
        assert!(r.is_err(), "inner guard should have caused OOM");
        drop(_inner);
        // Outer restored with 0 invocations counted.
        let r1: Result<Box<i32>, AllocError> = <Box<i32> as TryBox<i32>>::fallible_new(2);
        assert!(
            r1.is_ok(),
            "first alloc under restored outer should succeed"
        );
        let r2: Result<Box<i32>, AllocError> = <Box<i32> as TryBox<i32>>::fallible_new(3);
        assert!(r2.is_err(), "second alloc under restored outer should fail");
        drop(outer);
    }

    #[test]
    fn guard_restore_allows_allocation_afterwards() {
        let _guard = FailAllocGuard::fail_next_alloc();
        let r: Result<Box<i32>, AllocError> = <Box<i32> as TryBox<i32>>::fallible_new(42);
        assert!(r.is_err());
        drop(_guard);
        let r: Result<Box<i32>, AllocError> = <Box<i32> as TryBox<i32>>::fallible_new(99);
        assert!(r.is_ok());
        assert_eq!(*r.unwrap(), 99);
    }

    #[test]
    fn deeply_nested_guards_restore_correctly() {
        let g1 = FailAllocGuard::install(FailPolicy::fail_nth_alloc(2));
        let g2 = FailAllocGuard::install(FailPolicy::fail_next_alloc());
        let g3 = FailAllocGuard::install(FailPolicy::nothing());
        // g3 active (nothing fails).
        let r: Result<Box<u8>, AllocError> = <Box<u8> as TryBox<u8>>::fallible_new(0);
        assert!(r.is_ok());
        drop(g3); // restore g2 policy (fail next).
        let r: Result<Box<u8>, AllocError> = <Box<u8> as TryBox<u8>>::fallible_new(1);
        assert!(r.is_err());
        drop(g2); // restore g1 policy (fail on 2nd, counter was 0).
        let r: Result<Box<u8>, AllocError> = <Box<u8> as TryBox<u8>>::fallible_new(2);
        assert!(r.is_ok());
        let r: Result<Box<u8>, AllocError> = <Box<u8> as TryBox<u8>>::fallible_new(3);
        assert!(r.is_err());
        drop(g1);
    }

    // ── with_policy() ─────────────────────────────────────────────────────

    #[test]
    fn with_policy_fails_alloc() {
        let r: Result<Box<i32>, AllocError> = with_policy(FailPolicy::fail_next_alloc(), || {
            <Box<i32> as TryBox<i32>>::fallible_new(1)
        });
        assert!(r.is_err());
        let r: Result<Box<i32>, AllocError> = <Box<i32> as TryBox<i32>>::fallible_new(2);
        assert!(r.is_ok());
    }

    #[test]
    fn with_policy_returns_value() {
        let val = with_policy(FailPolicy::nothing(), || 42);
        assert_eq!(val, 42);
    }

    #[test]
    fn with_policy_nested() {
        let inner_r: Result<Box<u8>, AllocError> = with_policy(FailPolicy::nothing(), || {
            with_policy(FailPolicy::fail_next_alloc(), || {
                <Box<u8> as TryBox<u8>>::fallible_new(0)
            })
        });
        assert!(inner_r.is_err());
        let outer_r: Result<Box<u8>, AllocError> = <Box<u8> as TryBox<u8>>::fallible_new(1);
        assert!(outer_r.is_ok());
    }

    #[test]
    fn with_policy_restores_after_closure_returns() {
        // Verify that after with_policy returns (even with Err), allocation works.
        let result = with_policy(
            FailPolicy::fail_next_alloc(),
            || -> Result<(), &'static str> {
                let r: Result<Box<i32>, AllocError> = <Box<i32> as TryBox<i32>>::fallible_new(1);
                if r.is_err() {
                    Ok(())
                } else {
                    Err("should have failed")
                }
            },
        );
        assert!(result.is_ok());
        // Policy restored — allocation works again.
        let r: Result<Box<i32>, AllocError> = <Box<i32> as TryBox<i32>>::fallible_new(2);
        assert!(r.is_ok());
    }

    // ── Box allocation failure ────────────────────────────────────────────

    #[test]
    fn box_fallible_new_fails_on_oom() {
        let _g = FailAllocGuard::fail_next_alloc();
        let r: Result<Box<i32>, AllocError> = <Box<i32> as TryBox<i32>>::fallible_new(42);
        assert!(matches!(r, Err(AllocError { .. })));
    }

    #[test]
    fn box_fallible_new_uninit_fails_on_oom() {
        let _g = FailAllocGuard::fail_next_alloc();
        let r: Result<Box<std::mem::MaybeUninit<i32>>, AllocError> =
            <Box<i32> as TryBox<i32>>::fallible_new_uninit();
        assert!(r.is_err());
    }

    #[test]
    fn box_fallible_new_zeroed_fails_on_oom() {
        let _g = FailAllocGuard::fail_next_alloc();
        let r: Result<Box<std::mem::MaybeUninit<[u8; 16]>>, AllocError> =
            <Box<[u8; 16]> as TryBox<[u8; 16]>>::fallible_new_zeroed();
        assert!(r.is_err());
    }

    #[test]
    fn box_fallible_new_give_back_returns_value_on_oom() {
        // Use i32 to avoid any clone-time allocations after guard is active.
        let _g = FailAllocGuard::fail_next_alloc();
        let r: Result<Box<i32>, (i32, AllocError)> =
            <Box<i32> as TryBox<i32>>::fallible_new_give_back(42);
        match r {
            Err((returned, _err)) => assert_eq!(returned, 42),
            Ok(_) => panic!("expected allocation failure"),
        }
    }

    #[test]
    fn box_fallible_pin_fails_on_oom() {
        let _g = FailAllocGuard::fail_next_alloc();
        let r: Result<std::pin::Pin<Box<i32>>, AllocError> =
            <Box<i32> as TryBox<i32>>::fallible_pin(42);
        assert!(r.is_err());
    }

    #[test]
    fn box_fallible_pin_give_back_returns_value_on_oom() {
        let _g = FailAllocGuard::fail_next_alloc();
        let r: Result<std::pin::Pin<Box<i64>>, (i64, AllocError)> =
            <Box<i64> as TryBox<i64>>::fallible_pin_give_back(99);
        match r {
            Err((returned, _err)) => assert_eq!(returned, 99),
            Ok(_) => panic!("expected allocation failure"),
        }
    }

    #[test]
    fn box_zst_never_fails() {
        let _g = FailAllocGuard::fail_next_alloc();
        let r: Result<Box<()>, AllocError> = <Box<()> as TryBox<()>>::fallible_new(());
        assert!(r.is_ok());
    }

    // ── Nth-call targeting ────────────────────────────────────────────────

    #[test]
    fn fail_nth_alloc_skips_then_fails() {
        let _g = FailAllocGuard::install(FailPolicy::fail_nth_alloc(3));
        let r1: Result<Box<u8>, AllocError> = <Box<u8> as TryBox<u8>>::fallible_new(1);
        assert!(r1.is_ok());
        let r2: Result<Box<u8>, AllocError> = <Box<u8> as TryBox<u8>>::fallible_new(2);
        assert!(r2.is_ok());
        let r3: Result<Box<u8>, AllocError> = <Box<u8> as TryBox<u8>>::fallible_new(3);
        assert!(r3.is_err());
        let r4: Result<Box<u8>, AllocError> = <Box<u8> as TryBox<u8>>::fallible_new(4);
        assert!(r4.is_ok());
    }

    // ── Guard !Send / !Sync ───────────────────────────────────────────────

    static_assertions::assert_not_impl_all!(FailAllocGuard: Send, Sync);

    // ── Panic handler allocation investigation ────────────────────────────
    //
    // Documents why the `btrees` module (and the `panic` feature) were deprecated.
    // See `src/btrees/mod.rs` for the full rationale and the recommendation to use
    // the `scapegoat` crate instead.
    //
    // Summary of findings (confirmed empirically on Linux x86_64, Rust stable 1.97):
    //
    // 1. The panic infrastructure allocates a `Box<dyn Any + Send>` for every panic
    //    payload. Under sustained OOM this allocation fails and the process aborts
    //    before `catch_unwind` can intervene.
    //
    // 2. `handle_alloc_error` calls `abort()` directly on OOM, bypassing the unwind
    //    mechanism entirely. No amount of `catch_unwind` nesting can intercept an
    //    abort.
    //
    // 3. The `-Z oom=panic` flag that would have made `handle_alloc_error` panic
    //    instead of abort was removed from Rust in 1.94.0 (PR #147725) due to
    //    reentrancy soundness concerns. The replacement (`set_alloc_error_hook`)
    //    suffers from the same payload-box allocation problem as (1).
    //
    // The tests below are intentionally marked `ignore` because they demonstrate
    // aborting behavior. Run them individually with `--ignored` to observe.

    /// Demonstrates that raising a panic while the allocator is disabled causes
    /// the process to abort. The panic infrastructure needs to allocate a
    /// `Box<dyn Any + Send>` for the payload, and when that fails, `abort()` is
    /// called before `catch_unwind` can intervene.
    ///
    /// Expected outcome: SIGABRT ("memory allocation of N bytes failed").
    #[test]
    #[ignore = "intentionally aborts - demonstrates that panic infrastructure allocates"]
    fn panic_infrastructure_allocates_payload_box() {
        let _g = FailAllocGuard::install(FailPolicy::fail_nth_alloc(1));

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            panic!("synthetic");
        }));

        // Unreachable: the process aborts before reaching here.
        assert!(result.is_err());
    }

    /// Demonstrates that `handle_alloc_error` aborts the process rather than
    /// panicking. This is the root cause of why `catch_unwind` cannot intercept
    /// OOM from standard library collections: the allocation failure path never
    /// produces a catchable panic to begin with.
    ///
    /// Expected outcome: SIGABRT ("memory allocation of 8 bytes failed").
    #[test]
    #[ignore = "intentionally aborts - demonstrates that handle_alloc_error calls abort()"]
    fn handle_alloc_error_aborts_instead_of_panicking() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Force an allocation failure and feed it into handle_alloc_error.
            // Layout for a single u64 — small, innocuous, just needs to fail.
            let layout = std::alloc::Layout::new::<u64>();
            std::alloc::handle_alloc_error(layout);
        }));

        // Unreachable: handle_alloc_error aborts the process.
        assert!(result.is_err());
    }

    /// Basically, if output capturing is enabled, panicking locks the backtrace and attempts to write to an output
    /// vector buffer, and fail because it tries to allocate while the FailAllocGuard is active. The second allocation
    /// error tries to lock the backtrace output lock again, which causes a deadlock.
    ///
    /// In short, the allocator must not fail while encountering a panic if output capturing is set. If all else fails,
    /// run the suite with --nocapture.
    ///
    /// https://github.com/rust-lang/rust/issues/130187
    #[test]
    #[ignore = "deadlocks - demonstrates issue #130187"]
    fn backtrace_deadlock() {
        let _g = FailAllocGuard::fail_next_alloc();
        panic!("this may trigger re-entrant deadlock");
    }
}
