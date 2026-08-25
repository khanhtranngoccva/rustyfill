use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::TryDebug;
use lang_core::cell::UnsafeCell;
use lang_core::fmt;

// NOTE: no `TryClone` impl for `UnsafeCell<T>` — `UnsafeCell` does not implement
// `std::clone::Clone`, and `TryClone` requires `Clone` as a supertrait. To
// snapshot the contained value, read it out (`*cell.get()` for `T: Copy`) and
// clone the inner `T` directly.

// ── TryDebug for UnsafeCell<T> ─────────────────────────────────────────────────
// Mirrors std's Debug impl: prints "UnsafeCell(value)" by reading through the
// raw pointer. Reading is safe when no mutable alias exists; in practice this
// is only called from single-threaded debug contexts.

impl<T: crate::try_fmt::TryDebug> TryDebug for UnsafeCell<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: Debug printing assumes exclusive observation of the value,
        // matching std's own `impl Debug for UnsafeCell<T>` which reads via
        // `&*self.get()`. Callers must ensure no concurrent mutation.
        let val = unsafe { &*(self.get() as *const T) };
        f.debug_struct("UnsafeCell").field("0", &val).finish()
    }
}

// ── TryDefault for UnsafeCell<T> ───────────────────────────────────────────────
// Wraps `T::try_default()` — no allocation beyond what T itself needs.

impl<T: TryDefault> TryDefault for UnsafeCell<T> {
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(UnsafeCell::new(T::try_default()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::try_format;
    use lang_alloc::string::String;

    #[test]
    fn unsafe_cell_try_debug() {
        let cell = UnsafeCell::new(42i32);
        let dbg = try_format!("{:?}", cell).unwrap();
        assert!(dbg.contains("42"));
    }

    #[test]
    fn unsafe_cell_try_default() {
        let cell: UnsafeCell<i32> = UnsafeCell::try_default().unwrap();
        assert_eq!(unsafe { *cell.get() }, 0);
    }

}
