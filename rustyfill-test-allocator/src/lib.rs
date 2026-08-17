//! Test-time global allocator that intercepts allocation calls and can be
//! instructed via thread-local state to return null, simulating OOM conditions.
//!
//! This crate installs a custom `#[global_allocator]`
//! at compile time — it is intended as a **dev-dependency** only. Do not depend on
//! it in production code.

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
    pub const fn fail_nth_realloc(n: u64) -> Self {
        Self {
            fail_alloc_at: None,
            fail_realloc_at: Some(n),
        }
    }

    /// Fail every alloc and alloc_zeroed call unconditionally.
    /// Useful for verifying that a code path performs zero heap allocations.
    pub const fn fail_all_alloc() -> Self {
        Self {
            fail_alloc_at: Some(0),
            fail_realloc_at: None,
        }
    }

    /// Fail every realloc call unconditionally.
    pub const fn fail_all_realloc() -> Self {
        Self {
            fail_alloc_at: None,
            fail_realloc_at: Some(0),
        }
    }

    /// Fail every allocation call (both alloc and realloc) unconditionally.
    pub const fn fail_all() -> Self {
        Self {
            fail_alloc_at: Some(0),
            fail_realloc_at: Some(0),
        }
    }
}

// ── Thread-local state ────────────────────────────────────────────────────────

thread_local! {
    static FAIL_POLICY: Cell<FailPolicy> = const { Cell::new(FailPolicy::nothing()) };
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
    // Zero means "fail every call" — no counter increment needed.
    if at == 0 {
        return true;
    }
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
    // Zero means "fail every call" — no counter increment needed.
    if at == 0 {
        return true;
    }
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
/// Callers should use with_policy to avoid unpredictable behaviors.
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
    pub fn install(policy: FailPolicy) -> Self {
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
    pub fn fail_next_alloc() -> Self {
        Self::install(FailPolicy::fail_next_alloc())
    }

    /// Convenience: fail the next reallocation on this thread.
    pub fn fail_next_realloc() -> Self {
        Self::install(FailPolicy::fail_next_realloc())
    }

    /// Convenience: fail the Nth allocation on this thread.
    pub fn fail_nth_alloc(n: u64) -> Self {
        Self::install(FailPolicy::fail_nth_alloc(n))
    }

    /// Convenience: fail the Nth reallocation on this thread.
    pub fn fail_nth_realloc(n: u64) -> Self {
        Self::install(FailPolicy::fail_nth_realloc(n))
    }

    /// Convenience: fail every allocation on this thread.
    pub fn fail_all() -> Self {
        Self::install(FailPolicy::fail_all())
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
    use rustyfill::alloc::AllocError;
    use rustyfill::alloc::boxed::TryBox;

    #[test]
    fn guard_restores_previous_policy() {
        let outer = FailAllocGuard::install(FailPolicy::fail_nth_alloc(2));
        let _inner = FailAllocGuard::install(FailPolicy::fail_next_alloc());
        let r: Result<Box<i32>, AllocError> = <Box<i32> as TryBox<i32>>::fallible_new(1);
        assert!(r.is_err(), "inner guard should have caused OOM");
        drop(_inner);
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
        let r: Result<Box<u8>, AllocError> = <Box<u8> as TryBox<u8>>::fallible_new(0);
        assert!(r.is_ok());
        drop(g3);
        let r: Result<Box<u8>, AllocError> = <Box<u8> as TryBox<u8>>::fallible_new(1);
        assert!(r.is_err());
        drop(g2);
        let r: Result<Box<u8>, AllocError> = <Box<u8> as TryBox<u8>>::fallible_new(2);
        assert!(r.is_ok());
        let r: Result<Box<u8>, AllocError> = <Box<u8> as TryBox<u8>>::fallible_new(3);
        assert!(r.is_err());
        drop(g1);
    }

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
    fn box_fallible_new_fails_on_oom() {
        let _g = FailAllocGuard::fail_next_alloc();
        let r: Result<Box<i32>, AllocError> = <Box<i32> as TryBox<i32>>::fallible_new(42);
        assert!(matches!(r, Err(AllocError { .. })));
    }

    #[test]
    fn box_zst_never_fails() {
        let _g = FailAllocGuard::fail_next_alloc();
        let r: Result<Box<()>, AllocError> = <Box<()> as TryBox<()>>::fallible_new(());
        assert!(r.is_ok());
    }

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

    /// Documented pitfall: println! with captured stdout while allocation is disabled
    ///
    /// When the test harness captures stdout (the default), `println!` buffers
    /// output into an internal `Vec<u8>`. If that buffer needs to grow while
    /// `fail_all_alloc` is active, the allocation returns null and the OOM
    /// handler itself tries to allocate (to format its error message), causing
    /// a cascade that aborts the process via SIGABRT.
    ///
    /// With `--nocapture`, `println!` writes directly to fd 1 without buffering
    /// through a heap-allocated Vec, so no allocation occurs and the test
    /// completes normally.
    ///
    /// In practice, please avoid any heap-allocating operations (including debug logging
    /// via `println!`/`eprintln!` when output is captured) inside a
    /// `with_policy(fail_all_alloc(), ...)` span. If an abort suddenly happens, one may
    /// attempt to use --nocapture to detect this pitfall.
    #[test]
    #[ignore = "aborts the process when run with captured output (default); use --nocapture to verify it prints safely"]
    fn fail_all_alloc_panics_if_println_buffers() {
        with_policy(FailPolicy::fail_all_alloc(), || {
            // This println! allocates internally when stdout is captured by
            // the test harness. Under fail_all_alloc, that allocation fails,
            // triggering the OOM handler which also needs to allocate → abort.
            println!("hello from inside fail_all_alloc");
        });
    }

    static_assertions::assert_not_impl_all!(FailAllocGuard: Send, Sync);
}
