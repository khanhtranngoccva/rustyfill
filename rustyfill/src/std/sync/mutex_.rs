use crate::lang_std::sync::Mutex;

impl<T: crate::try_fmt::TryDebug> crate::try_fmt::TryDebug for Mutex<T> {
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // std's Debug for Mutex is allocation-free (verified by OOM tests)
        // and already shows "<locked>" when the lock is held.
        core::fmt::Debug::fmt(self, f)
    }
}
