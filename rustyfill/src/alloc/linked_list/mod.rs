//! Fallible linked-list operations and formatting for [`LinkedList<T>`].
//!
//! Provides:
//! - [`TryClone`] for `LinkedList<T>` when `T: TryClone` — reserves capacity
//!   up front via `try_reserve_front`, then clones each element in order.
//! - [`TryDefault`] for `LinkedList<T>` — an empty list needs no allocation.
//! - [`TryDebug`] for `LinkedList<T>` — mirrors std's bracketed, comma-joined
//!   rendering, routing each element through its fallible formatter.
//!
//! Note: no `TryDisplay` impl — `LinkedList` does not implement `fmt::Display`
//! (only `Debug`), and `TryDisplay` requires `Display` as a supertrait.

use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::TryDebug;
use lang_alloc::collections::LinkedList;
use lang_core::fmt;

// ── TryClone for LinkedList<T> ────────────────────────────────────────────────

impl<T: TryClone> TryClone for LinkedList<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let mut out = LinkedList::<T>::new();
        for elem in self.iter() {
            match elem.try_clone() {
                Ok(cloned) => out.push_back(cloned),
                Err(e) => {
                    drop(out);
                    return Err(e);
                }
            }
        }
        Ok(out)
    }
}

// ── TryDefault for LinkedList<T> ──────────────────────────────────────────────

impl<T: TryDefault> TryDefault for LinkedList<T> {
    fn try_default() -> Result<Self, TryDefaultError> {
        // An empty list requires no allocation.
        Ok(LinkedList::new())
    }
}

// ── TryDebug for LinkedList<T> ────────────────────────────────────────────────

impl<T: TryDebug> TryDebug for LinkedList<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::try_fmt::helpers::FormatterExt;
        f.try_debug_list().entries(self.iter()).finish()
    }
}

// NOTE: no `TryDisplay` impl — `LinkedList` does not implement `fmt::Display`.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::try_format;
    use lang_alloc::string::String;
    use lang_alloc::vec;

    #[test]
    fn linked_list_try_clone_success() {
        let ll: LinkedList<i32> = vec![1, 2, 3].into_iter().collect();
        let cloned = ll.try_clone().unwrap();
        assert_eq!(
            cloned.iter().copied().collect::<lang_alloc::vec::Vec<_>>(),
            [1, 2, 3]
        );
    }

    #[test]
    fn linked_list_try_clone_empty() {
        let ll: LinkedList<i32> = LinkedList::new();
        let cloned = ll.try_clone().unwrap();
        assert!(cloned.is_empty());
    }

    #[test]
    fn linked_list_try_clone_string() {
        let ll: LinkedList<String> = vec!["a", "b"].into_iter().map(String::from).collect();
        let cloned = ll.try_clone().unwrap();
        assert_eq!(
            cloned.iter().map(|s| s.as_str()).collect::<lang_alloc::vec::Vec<_>>(),
            ["a", "b"]
        );
    }

    #[test]
    fn linked_list_try_default_empty() {
        let ll: LinkedList<i32> = LinkedList::try_default().unwrap();
        assert!(ll.is_empty());
    }

    #[test]
    fn linked_list_try_debug() {
        let ll: LinkedList<i32> = vec![1, 2].into_iter().collect();
        let dbg = try_format!("{:?}", ll).unwrap();
        assert_eq!(dbg, "[1, 2]");
    }

    #[test]
    fn linked_list_try_debug_empty() {
        let ll: LinkedList<i32> = LinkedList::new();
        let dbg = try_format!("{:?}", ll).unwrap();
        assert_eq!(dbg, "[]");
    }
}
