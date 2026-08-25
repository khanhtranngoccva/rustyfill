//! Fallible formatting for [`BTreeMap`] and [`BTreeSet`].
//!
//! Both types' std `Debug` impls iterate without allocating, so these are thin
//! passthroughs that route each element through its fallible formatter.
//!
//! Note: no `TryDisplay` impls — neither `BTreeMap` nor `BTreeSet` implements
//! `fmt::Display` (only `Debug`), and `TryDisplay` requires `Display` as a
//! supertrait.

use crate::try_fmt::TryDebug;
use lang_alloc::collections::{BTreeMap, BTreeSet};
use lang_core::fmt;

// ── TryDebug for BTreeMap<K, V> ───────────────────────────────────────────────

impl<K: Ord + TryDebug, V: TryDebug> TryDebug for BTreeMap<K, V> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::try_fmt::helpers::FormatterExt;
        f.try_debug_map().entries(self.iter()).finish()
    }
}

// ── TryDebug for BTreeSet<T> ──────────────────────────────────────────────────

impl<T: Ord + TryDebug> TryDebug for BTreeSet<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::try_fmt::helpers::FormatterExt;
        f.try_debug_set().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::try_format;
    use lang_alloc::string::String;
    use lang_alloc::vec;

    #[test]
    fn btree_map_try_debug() {
        let mut m: BTreeMap<&str, i32> = BTreeMap::new();
        m.insert("a", 1);
        m.insert("b", 2);
        let dbg = try_format!("{:?}", m).unwrap();
        assert_eq!(dbg, "{\"a\": 1, \"b\": 2}");
    }

    #[test]
    fn btree_map_try_debug_empty() {
        let m: BTreeMap<String, i32> = BTreeMap::new();
        let dbg = try_format!("{:?}", m).unwrap();
        assert_eq!(dbg, "{}");
    }

    #[test]
    fn btree_set_try_debug() {
        let s: BTreeSet<i32> = vec![2, 1].into_iter().collect();
        let dbg = try_format!("{:?}", s).unwrap();
        assert_eq!(dbg, "{1, 2}");
    }

    #[test]
    fn btree_set_try_debug_empty() {
        let s: BTreeSet<String> = BTreeSet::new();
        let dbg = try_format!("{:?}", s).unwrap();
        assert_eq!(dbg, "{}");
    }
}
