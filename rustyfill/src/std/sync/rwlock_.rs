use crate::lang_core::fmt;
use crate::lang_std::sync::RwLock;

impl<T: crate::try_fmt::TryDebug> crate::try_fmt::TryDebug for RwLock<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // std's Debug for RwLock is allocation-free (verified by OOM tests)
        // and already shows "<locked>" when contention prevents inspection.
        fmt::Debug::fmt(self, f)
    }
}
