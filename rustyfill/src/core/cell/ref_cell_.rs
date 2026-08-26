use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::{TryDebug, TryDisplay};
use lang_core::cell::{BorrowError, BorrowMutError, RefCell};
use lang_core::fmt;

// ── TryDebug / TryDisplay for BorrowError ───────────────────────────────────
// Both are zero-sized marker types whose std Debug/Display write fixed strings
// with no allocation risk.

impl TryDebug for BorrowError {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for BorrowError {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

// ── TryDebug / TryDisplay for BorrowMutError ────────────────────────────────

impl TryDebug for BorrowMutError {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for BorrowMutError {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

// ── TryClone for RefCell<T> ───────────────────────────────────────────────────

impl<T: TryClone> TryClone for RefCell<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let inner = RefCell::try_borrow(self).map_err(|_| {
            TryCloneError::Other("RefCell clone failed: immutable borrow unavailable")
        })?;
        let cloned = (*inner).try_clone()?;
        Ok(RefCell::new(cloned))
    }
}

// ── TryDefault for RefCell<T> ─────────────────────────────────────────────────

impl<T: TryDefault> TryDefault for RefCell<T> {
    fn try_default() -> Result<Self, TryDefaultError> {
        let val = T::try_default()?;
        Ok(RefCell::new(val))
    }
}

// ── TryDebug for RefCell<T> ───────────────────────────────────────────────────

impl<T: ?Sized + crate::try_fmt::TryDebug> crate::try_fmt::TryDebug for RefCell<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match RefCell::try_borrow(self) {
            Ok(inner) => {
                f.write_str("RefCell { value: ")?;
                inner.try_fmt(f)?;
                f.write_str(" }")
            }
            Err(_) => f.write_str("RefCell { <borrowed> }"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::try_format;
    use lang_alloc::string::String;
    use lang_alloc::vec;

    // ── TryClone tests ────────────────────────────────────────────────────────

    #[test]
    fn refcell_try_clone_success() {
        let cell = RefCell::new(vec![1, 2, 3]);
        let cloned = cell.try_clone().unwrap();
        assert_eq!(*cloned.borrow(), [1, 2, 3]);
        assert_ne!(cell.as_ptr(), cloned.as_ptr());
    }

    #[test]
    fn refcell_try_clone_fails_when_borrowed() {
        let cell = RefCell::new(42);
        let _borrow = cell.borrow_mut();
        let result = cell.try_clone();
        assert!(result.is_err());
    }

    #[test]
    fn refcell_try_clone_zst() {
        let cell = RefCell::new(());
        // ZST — nothing to compare; successful clone is the assertion.
        cell.try_clone().unwrap();
    }

    // ── TryDefault tests ──────────────────────────────────────────────────────

    #[test]
    fn refcell_try_default() {
        let cell: RefCell<i32> = RefCell::try_default().unwrap();
        assert_eq!(*cell.borrow(), 0);
    }

    #[test]
    fn refcell_try_default_string() {
        let cell: RefCell<String> = RefCell::try_default().unwrap();
        assert!(cell.borrow().is_empty());
    }

    // ── TryDebug tests ────────────────────────────────────────────────────────

    #[test]
    fn refcell_try_debug_unborrowed() {
        let cell = RefCell::new(42i32);
        let dbg = try_format!("{:?}", cell).unwrap();
        assert!(dbg.contains("42"));
    }

    #[test]
    fn refcell_try_debug_while_mutably_borrowed() {
        let cell = RefCell::new(42i32);
        let _r = cell.borrow_mut();
        let dbg = try_format!("{:?}", cell).unwrap();
        assert_eq!(dbg, "RefCell { <borrowed> }");
    }
}
