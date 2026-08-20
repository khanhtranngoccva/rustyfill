use lang_core::fmt;
use lang_std::sync::RwLock;

impl<T: crate::try_fmt::TryDebug> crate::try_fmt::TryDebug for RwLock<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // std's Debug for RwLock is allocation-free (verified by OOM tests)
        // and already shows "<locked>" when contention prevents inspection.
        fmt::Debug::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::try_fmt::{AssertDebug, TryDebug, TryDebugWrapper};
    use lang_alloc::string::String;
    use lang_alloc::vec;
    use lang_alloc::vec::Vec;
    use rustyfill_test_allocator::{FailPolicy, with_policy};

    /// Minimal writer that discards everything without allocating.
    struct NoopWriter;
    impl fmt::Write for NoopWriter {
        fn write_str(&mut self, _: &str) -> fmt::Result {
            Ok(())
        }
    }

    /// Run `TryDebug::try_fmt` under a fully-failing allocator via
    /// [`TryDebugWrapper`] + `fmt::write`. The wrapper's `Debug` impl routes
    /// through `try_fmt`, and `fmt::write` constructs a real `Formatter`.
    fn assert_try_debug_no_alloc<T: TryDebug>(value: T) -> bool {
        with_policy(FailPolicy::fail_all_alloc(), || {
            let mut w = NoopWriter;
            fmt::write(&mut w, format_args!("{:?}", TryDebugWrapper(value))).is_ok()
        })
    }

    // ── OOM safety of TryDebug::try_fmt ───────────────────────────────────────
    // These were previously in `try_fmt::oom_tests`; they live here alongside the
    // `TryDebug` impl they verify. The mutex/RwLock backends allocate at most on
    // first lock, so constructing the value *outside* the failing-allocator scope
    // means only formatting is exercised under OOM.

    #[test]
    fn try_debug_rwlock_primitive_no_alloc() {
        let rw: RwLock<i32> = RwLock::new(42);
        assert!(assert_try_debug_no_alloc(&rw));
    }

    #[test]
    fn try_debug_rwlock_string_no_alloc() {
        let rw: RwLock<String> = RwLock::new(String::from("rwlock string"));
        assert!(assert_try_debug_no_alloc(&rw));
    }

    #[test]
    fn try_debug_rwlock_vec_no_alloc() {
        let rw: RwLock<Vec<u8>> = RwLock::new(vec![1, 2, 3]);
        assert!(assert_try_debug_no_alloc(&rw));
    }

    // Baseline: verify std's Debug itself is allocation-free. If this fails,
    // std's Debug impl changed and our TryDebug passthrough will too.
    #[test]
    fn std_debug_rwlock_primitive_no_alloc() {
        let rw: RwLock<i32> = RwLock::new(42);
        assert!(assert_try_debug_no_alloc(AssertDebug(&rw)));
    }
}
