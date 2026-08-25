use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::TryDebug;
use lang_core::cell::LazyCell;
use lang_core::fmt;

// ── TryDebug for LazyCell<T> ───────────────────────────────────────────────────
// Dereferencing forces initialization if needed (the initializer is expected to
// be infallible — it panics on failure, matching std). Mirrors std's Debug impl.

impl<T: crate::try_fmt::TryDebug> TryDebug for LazyCell<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val: &T = self;
        f.debug_struct("LazyCell").field("value", val).finish()
    }
}

// NOTE: no `TryClone` impl for `LazyCell<T>` — `LazyCell` does not implement
// `std::clone::Clone` (its initializer is a closure that cannot be copied), and
// `TryClone` requires `Clone` as a supertrait. To snapshot a lazy cell's value,
// dereference it first and clone the inner `T`.

// ── TryDefault for LazyCell<T> ─────────────────────────────────────────────────
// A fresh LazyCell is uninitialized — no allocation needed. The `T: TryDefault`
// bound mirrors the pattern used by RefCell so that callers can uniformly
// construct default cells across interior-mutability types.

impl<T: TryDefault + Default> TryDefault for LazyCell<T> {
    fn try_default() -> Result<Self, TryDefaultError> {
        // Uninitialized — no allocation. The initializer runs lazily on first
        // access; if `T::try_default` fails at that point the cell panics,
        // matching how a plain `LazyCell::new` with a failing closure behaves.
        Ok(LazyCell::new(|| T::try_default().expect("LazyCell default initializer failed")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::try_format;
    use lang_alloc::string::String;

    #[test]
    fn lazy_cell_try_debug_forces_init() {
        let cell = LazyCell::<i32>::new(|| 42);
        let dbg = try_format!("{:?}", cell).unwrap();
        assert!(dbg.contains("42"));
    }

    #[test]
    fn lazy_cell_try_debug_after_deref() {
        let cell = LazyCell::<i32>::new(|| 7);
        let _ = *cell; // force initialization
        let dbg = try_format!("{:?}", cell).unwrap();
        assert!(dbg.contains("7"));
    }

    #[test]
    fn lazy_cell_try_default_uninitialized() {
        let cell: LazyCell<i32> = LazyCell::try_default().unwrap();
        // Not yet accessed — deref would initialize it.
        assert_eq!(core::cell::LazyCell::<i32>::get(&cell), None);
    }

    #[test]
    fn lazy_cell_try_default_evaluates_lazily() {
        let cell: LazyCell<String> = LazyCell::try_default().unwrap();
        assert_eq!(core::cell::LazyCell::<String>::get(&cell), None);
        let val: &String = &cell;
        assert!(val.is_empty());
    }
}
