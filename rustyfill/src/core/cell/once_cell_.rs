use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::TryDebug;
use lang_core::cell::OnceCell;
use lang_core::fmt;

// ── TryClone for OnceCell<T> ───────────────────────────────────────────────────
// A clone of a populated cell contains the cloned value; a clone of an empty
// cell is also empty.

impl<T: TryClone + Clone> TryClone for OnceCell<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let out = OnceCell::new();
        if let Some(val) = self.get() {
            out.set(val.try_clone()?)
                .map_err(|_| TryCloneError::Other("OnceCell clone failed: set returned Err"))?;
        }
        Ok(out)
    }
}

// ── TryDefault for OnceCell<T> ─────────────────────────────────────────────────
// An empty OnceCell requires no allocation. The `T: TryDefault` bound mirrors
// the pattern used by RefCell so callers can uniformly construct default cells.

impl<T: TryDefault> TryDefault for OnceCell<T> {
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(OnceCell::new())
    }
}

// ── TryDebug for OnceCell<T> ───────────────────────────────────────────────────
// Mirrors std's Debug impl: "Some(value)" when initialized, "None" otherwise.

impl<T: crate::try_fmt::TryDebug> TryDebug for OnceCell<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(val) => val.try_fmt(f),
            None => f.write_str("None"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::try_format;
    use lang_alloc::string::String;
    use lang_alloc::vec;

    #[test]
    fn once_cell_try_clone_populated() {
        let cell = OnceCell::new();
        cell.set(vec![1, 2, 3]).unwrap();
        let cloned = cell.try_clone().unwrap();
        assert_eq!(
            cloned.get().map(|v| v.as_slice()),
            Some([1, 2, 3].as_slice())
        );
    }

    #[test]
    fn once_cell_try_clone_empty() {
        let cell: OnceCell<i32> = OnceCell::new();
        let cloned = cell.try_clone().unwrap();
        assert!(cloned.get().is_none());
    }

    #[test]
    fn once_cell_try_clone_string() {
        let cell = OnceCell::new();
        cell.set(String::from("hello")).unwrap();
        let cloned = cell.try_clone().unwrap();
        assert_eq!(cloned.get(), Some(&String::from("hello")));
    }

    #[test]
    fn once_cell_try_default_empty() {
        let cell: OnceCell<i32> = OnceCell::try_default().unwrap();
        assert!(cell.get().is_none());
    }

    #[test]
    fn once_cell_try_debug_initialized() {
        let cell = OnceCell::new();
        cell.set(42i32).unwrap();
        let dbg = try_format!("{:?}", cell).unwrap();
        assert!(dbg.contains("42"));
    }

    #[test]
    fn once_cell_try_debug_uninitialized() {
        let cell: OnceCell<i32> = OnceCell::new();
        let dbg = try_format!("{:?}", cell).unwrap();
        assert_eq!(dbg, "None");
    }
}
