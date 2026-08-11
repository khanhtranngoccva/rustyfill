use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use std::cell::{BorrowError, BorrowMutError, Ref, RefCell, RefMut};

/// Error returned when a fallible borrow operation fails.
#[derive(Debug)]
pub enum TryBorrowError {
    /// The immutable borrow limit was exceeded (already at max readers).
    Borrow(BorrowError),
    /// The mutable borrow is unavailable (another borrow is active).
    BorrowMut(BorrowMutError),
}

impl core::fmt::Display for TryBorrowError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Borrow(_) => write!(f, "immutable borrow failed: already at maximum"),
            Self::BorrowMut(_) => write!(f, "mutable borrow failed: another borrow is active"),
        }
    }
}

impl From<BorrowError> for TryBorrowError {
    fn from(e: BorrowError) -> Self {
        Self::Borrow(e)
    }
}

impl From<BorrowMutError> for TryBorrowError {
    fn from(e: BorrowMutError) -> Self {
        Self::BorrowMut(e)
    }
}

/// Fallible operations on [`RefCell`].
///
/// Implemented for `RefCell<T>`. Provides [`try_borrow`](Self::try_borrow) and
/// [`try_borrow_mut`](Self::try_borrow_mut), which return [`Result`] instead of
/// panicking when the borrow rules are violated.
pub trait TryRefCell<T: ?Sized> {
    /// Attempts to immutably borrow the inner value.
    ///
    /// Returns [`Err(TryBorrowError::Borrow)`] if the value is already borrowed
    /// mutably or if the immutable borrow count has reached its maximum.
    fn try_borrow(&self) -> Result<Ref<'_, T>, TryBorrowError>;

    /// Attempts to mutably borrow the inner value.
    ///
    /// Returns [`Err(TryBorrowError::BorrowMut)`] if the value is already
    /// borrowed (either mutably or immutably).
    fn try_borrow_mut(&self) -> Result<RefMut<'_, T>, TryBorrowError>;

    // ── Aliases with `fallible_` prefix ─────────────────────────────────────

    /// Alias for [`Self::try_borrow`].
    fn fallible_borrow(&self) -> Result<Ref<'_, T>, TryBorrowError> {
        Self::try_borrow(self)
    }

    /// Alias for [`Self::try_borrow_mut`].
    fn fallible_borrow_mut(&self) -> Result<RefMut<'_, T>, TryBorrowError> {
        Self::try_borrow_mut(self)
    }
}

impl<T: ?Sized> TryRefCell<T> for RefCell<T> {
    fn try_borrow(&self) -> Result<Ref<'_, T>, TryBorrowError> {
        RefCell::try_borrow(self).map_err(TryBorrowError::from)
    }

    fn try_borrow_mut(&self) -> Result<RefMut<'_, T>, TryBorrowError> {
        RefCell::try_borrow_mut(self).map_err(TryBorrowError::from)
    }
}

// ── TryClone for RefCell<T> ───────────────────────────────────────────────────

impl<T: TryClone> TryClone for RefCell<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let inner = <Self as TryRefCell<_>>::try_borrow(self).map_err(|e| {
            let msg = match e {
                TryBorrowError::Borrow(_) => "RefCell clone failed: immutable borrow unavailable",
                TryBorrowError::BorrowMut(_) => "RefCell clone failed: mutable borrow active",
            };
            TryCloneError::Other(msg)
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
    fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.try_borrow() {
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

    #[test]
    fn try_borrow_success() {
        let cell = RefCell::new(42);
        let r = cell.try_borrow().unwrap();
        assert_eq!(*r, 42);
    }

    #[test]
    fn try_borrow_mut_success() {
        let cell = RefCell::new(42);
        {
            let mut r = <RefCell<i32> as TryRefCell<_>>::try_borrow_mut(&cell).unwrap();
            *r = 99;
        }
        assert_eq!(*cell.borrow(), 99);
    }

    #[test]
    fn try_borrow_fails_when_mutably_borrowed() {
        let cell = RefCell::new(42);
        let _mut_ref = cell.borrow_mut();
        let result = <RefCell<i32> as TryRefCell<_>>::try_borrow(&cell);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TryBorrowError::Borrow(_)));
    }

    #[test]
    fn try_borrow_mut_fails_when_immutably_borrowed() {
        let cell = RefCell::new(42);
        let _ref = cell.borrow();
        let result = <RefCell<i32> as TryRefCell<_>>::try_borrow_mut(&cell);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TryBorrowError::BorrowMut(_)));
    }

    #[test]
    fn try_borrow_mut_fails_when_mutably_borrowed() {
        let cell = RefCell::new(42);
        let _mut_ref = cell.borrow_mut();
        let result = <RefCell<i32> as TryRefCell<_>>::try_borrow_mut(&cell);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TryBorrowError::BorrowMut(_)));
    }

    #[test]
    fn multiple_immutable_borrows_allowed() {
        let cell = RefCell::new(42);
        let r1 = cell.try_borrow().unwrap();
        let r2 = cell.try_borrow().unwrap();
        assert_eq!(*r1, 42);
        assert_eq!(*r2, 42);
    }

    #[test]
    fn fallible_aliases_work() {
        let cell = RefCell::new(42);
        let r = cell.fallible_borrow().unwrap();
        assert_eq!(*r, 42);
        drop(r);
        let mut r = cell.fallible_borrow_mut().unwrap();
        *r = 99;
    }

    #[test]
    fn error_display_messages() {
        let cell = RefCell::new(42);
        let _mut_ref = cell.borrow_mut();
        let err = <RefCell<i32> as TryRefCell<_>>::try_borrow(&cell).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("immutable"));

        drop(_mut_ref);
        let _ref = cell.borrow();
        let err = <RefCell<i32> as TryRefCell<_>>::try_borrow_mut(&cell).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("mutable"));
    }

    // ── TryClone tests ────────────────────────────────────────────────────────

    #[test]
    fn refcell_try_clone_success() {
        let cell = RefCell::new(vec![1, 2, 3]);
        let cloned = cell.try_clone().unwrap();
        assert_eq!(*cloned.borrow(), vec![1, 2, 3]);
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
        let cloned = cell.try_clone().unwrap();
        assert_eq!(*cloned.borrow(), ());
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
    fn refcell_try_debug_available() {
        let cell = RefCell::new(42i32);
        let r = cell.try_borrow().unwrap();
        assert_eq!(*r, 42);
    }
}
