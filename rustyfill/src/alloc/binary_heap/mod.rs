//! Fallible binary-heap operations and formatting for [`BinaryHeap<T>`].
//!
//! Provides:
//! - [`TryClone`] for `BinaryHeap<T>` when `T: TryClone` — reserves capacity
//!   up front, then clones each element (heap order is restored by the
//!   standard `push` on the clone).
//! - [`TryDefault`] for `BinaryHeap<T>` — an empty heap needs no allocation.
//! - [`TryDebug`] for `BinaryHeap<T>` — mirrors std's bracketed, comma-joined
//!   rendering, routing each element through its fallible formatter.
//!
//! Note: no `TryDisplay` impl — `BinaryHeap` does not implement `fmt::Display`
//! (only `Debug`), and `TryDisplay` requires `Display` as a supertrait.

use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::TryDebug;
use lang_alloc::collections::BinaryHeap;
use lang_core::fmt;

// ── TryClone for BinaryHeap<T> ────────────────────────────────────────────────

impl<T: TryClone + Ord> TryClone for BinaryHeap<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let mut out = BinaryHeap::<T>::new();
        if !self.is_empty() {
            // One reservation covers every slot we will push.
            out.try_reserve(self.len())
                .map_err(TryCloneError::Reserve)?;
        }
        for elem in self.iter() {
            match elem.try_clone() {
                Ok(cloned) => out.push(cloned),
                Err(e) => {
                    drop(out);
                    return Err(e);
                }
            }
        }
        Ok(out)
    }
}

// ── TryDefault for BinaryHeap<T> ──────────────────────────────────────────────

impl<T: TryDefault> TryDefault for BinaryHeap<T> {
    fn try_default() -> Result<Self, TryDefaultError> {
        // An empty heap requires no allocation.
        Ok(BinaryHeap::new())
    }
}

// ── TryDebug for BinaryHeap<T> ────────────────────────────────────────────────

impl<T: TryDebug> TryDebug for BinaryHeap<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::try_fmt::helpers::FormatterExt;
        f.try_debug_list().entries(self.iter()).finish()
    }
}

// NOTE: no `TryDisplay` impl — `BinaryHeap` does not implement `fmt::Display`.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::try_format;
    use lang_alloc::string::String;
    use lang_alloc::vec;

    #[test]
    fn binary_heap_try_clone_success() {
        let bh: BinaryHeap<i32> = vec![3, 1, 2].into_iter().collect();
        let cloned = bh.try_clone().unwrap();
        // Same multiset of elements; the internal buffer pointers differ.
        let mut a: lang_alloc::vec::Vec<i32> = bh.into_iter().collect();
        let mut b: lang_alloc::vec::Vec<i32> = cloned.into_iter().collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        assert_eq!(b, vec![1, 2, 3]);
    }

    #[test]
    fn binary_heap_try_clone_empty() {
        let bh: BinaryHeap<i32> = BinaryHeap::new();
        let cloned = bh.try_clone().unwrap();
        assert!(cloned.is_empty());
    }

    #[test]
    fn binary_heap_try_default_empty() {
        let bh: BinaryHeap<i32> = BinaryHeap::try_default().unwrap();
        assert!(bh.is_empty());
    }

    #[test]
    fn binary_heap_try_debug_sorted() {
        let bh: BinaryHeap<i32> = vec![3, 1, 2].into_iter().collect();
        let dbg = try_format!("{:?}", bh).unwrap();
        // std Debug prints in descending order for a max-heap.
        assert_eq!(dbg, "[3, 1, 2]");
    }

    #[test]
    fn binary_heap_try_debug_empty() {
        let bh: BinaryHeap<String> = BinaryHeap::new();
        let dbg = try_format!("{:?}", bh).unwrap();
        assert_eq!(dbg, "[]");
    }
}
