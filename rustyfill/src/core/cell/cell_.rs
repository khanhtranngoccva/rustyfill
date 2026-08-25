use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::TryDebug;
use lang_core::cell::Cell;
use lang_core::fmt;

// ── TryClone for Cell<T> where T: Copy ─────────────────────────────────────────
// Reads the current value via `get()` (infallible for Copy types) and wraps a
// fresh copy in a new Cell. No allocation involved.

impl<T: Copy + Clone> TryClone for Cell<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(Cell::new(self.get()))
    }
}

// ── TryDefault for Cell<T> ─────────────────────────────────────────────────────

impl<T: TryDefault> TryDefault for Cell<T> {
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Cell::new(T::try_default()?))
    }
}

// ── TryDebug for Cell<T> ───────────────────────────────────────────────────────
// Mirrors std's Debug impl: "Cell { value: ... }" reading through get().

impl<T: Copy + crate::try_fmt::TryDebug> TryDebug for Cell<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cell").field("value", &self.get()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::try_format;

    #[test]
    fn cell_try_clone_copy() {
        let cell = Cell::new(42i32);
        let cloned = cell.try_clone().unwrap();
        assert_eq!(cloned.get(), 42);
        assert_ne!(cell.as_ptr(), cloned.as_ptr());
    }

    #[test]
    fn cell_try_clone_bool() {
        let cell = Cell::new(true);
        let cloned = cell.try_clone().unwrap();
        assert!(cloned.get());
    }

    #[test]
    fn cell_try_default() {
        let cell: Cell<i32> = Cell::try_default().unwrap();
        assert_eq!(cell.get(), 0);
    }

    #[test]
    fn cell_try_debug() {
        let cell = Cell::new(7u8);
        let dbg = try_format!("{:?}", cell).unwrap();
        assert!(dbg.contains("7"));
    }
}
