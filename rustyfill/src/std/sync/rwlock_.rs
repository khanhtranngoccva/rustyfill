use crate::lang_std::sync::RwLock;

impl<T: crate::try_fmt::TryDebug> crate::try_fmt::TryDebug for RwLock<T> {
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // std's Debug for RwLock is allocation-free (verified by OOM tests)
        // and already shows "<locked>" when contention prevents inspection.
        core::fmt::Debug::fmt(self, f)
    }
}
