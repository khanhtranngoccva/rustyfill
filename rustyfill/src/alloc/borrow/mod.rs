//! Fallible trait implementations for [`Cow<'_, B>`](lang_alloc::borrow::Cow).
//!
//! The blanket impls below cover every `B` whose owned form satisfies the
//! relevant fallible trait. Because they are generic over `B`, they also cover
//! the common concrete cases (`Cow<'_, str>`, `Cow<'_, [T]>`, ...) — no separate
//! per-type impls are needed.
//!
//! Bounds:
//! - `TryDebug` / `TryDisplay`: require `B: TryDebug` / `B: TryDisplay` AND
//!   `B::Owned: TryDebug` / `B::Owned: TryDisplay`. Both arms must be
//!   fallibly-formattable; the impl routes each arm through its own type's
//!   fallible formatter so the rendering matches what the user would get by
//!   formatting the inner value directly.
//! - `TryDefault`: requires `B::Owned: TryDefault`; always produces an owned
//!   default so the result is usable without a borrowed lifetime.
//! - `TryClone`: requires `B::Owned: TryClone`; the borrowed arm copies the
//!   reference (allocation-free), the owned arm clones the inner value so OOM
//!   stays visible.
//!
//! Note: these impls intentionally do NOT add a `B: Clone` bound. `Cow`'s own
//! std `Clone` impl requires `B: ToOwned + Clone`, but our fallible supertraits
//! do not — so e.g. `Cow<'_, str>` gets `TryDebug` even though `str: !Clone`.

use crate::try_clone::{TryClone, TryCloneError};
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::{TryDebug, TryDisplay};
use lang_alloc::borrow::{Cow, ToOwned};
use lang_core::fmt;

impl<B: TryDebug + ToOwned + ?Sized> TryDebug for Cow<'_, B>
where
    B::Owned: TryDebug,
{
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Route each arm through its own type's fallible Debug so the output
        // matches what the user would get by formatting the inner value.
        match self {
            Cow::Borrowed(b) => b.try_fmt(f),
            Cow::Owned(o) => o.try_fmt(f),
        }
    }
}

impl<B: TryDisplay + ToOwned + ?Sized> TryDisplay for Cow<'_, B>
where
    B::Owned: TryDisplay,
{
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cow::Borrowed(b) => b.try_fmt(f),
            Cow::Owned(o) => o.try_fmt(f),
        }
    }
}

impl<B: ToOwned + ?Sized> TryDefault for Cow<'_, B>
where
    B::Owned: TryDefault,
{
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Cow::Owned(<B::Owned as TryDefault>::try_default()?))
    }
}

impl<B: ToOwned + ?Sized> TryClone for Cow<'_, B>
where
    B::Owned: TryClone,
{
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        match self {
            Cow::Borrowed(b) => Ok(Cow::Borrowed(*b)),
            Cow::Owned(o) => o.try_clone().map(Cow::Owned),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::try_format;
    use lang_alloc::string::String;
    use lang_alloc::vec;

    // ── Cow<'_, str> ────────────────────────────────────────────────────────

    #[test]
    fn cow_str_try_debug_borrowed() {
        let c: Cow<'_, str> = Cow::Borrowed("hello");
        let dbg = try_format!("{:?}", c).unwrap();
        assert_eq!(dbg, "\"hello\"");
    }

    #[test]
    fn cow_str_try_debug_owned() {
        let c: Cow<'_, str> = Cow::Owned(String::from("world"));
        let dbg = try_format!("{:?}", c).unwrap();
        assert_eq!(dbg, "\"world\"");
    }

    #[test]
    fn cow_str_try_display_borrowed_and_owned() {
        let b: Cow<'_, str> = Cow::Borrowed("abc");
        let o: Cow<'_, str> = Cow::Owned(String::from("def"));
        assert_eq!(try_format!("{}", b).unwrap(), "abc");
        assert_eq!(try_format!("{}", o).unwrap(), "def");
    }

    #[test]
    fn cow_str_try_clone_borrowed_is_free() {
        let c: Cow<'_, str> = Cow::Borrowed("borrowed");
        let cloned = c.try_clone().unwrap();
        assert!(matches!(cloned, Cow::Borrowed(_)));
        assert_eq!(cloned, "borrowed");
    }

    #[test]
    fn cow_str_try_clone_owned_rebuilds() {
        let c: Cow<'_, str> = Cow::Owned(String::from("owned-value"));
        let cloned = c.try_clone().unwrap();
        assert!(matches!(cloned, Cow::Owned(_)));
        assert_eq!(cloned, "owned-value");
    }

    #[test]
    fn cow_str_try_default_produces_owned() {
        let c: Cow<'_, str> = Cow::try_default().unwrap();
        assert!(matches!(c, Cow::Owned(_)));
        assert_eq!(c, "");
    }

    // ── Cow<'_, [T]> ────────────────────────────────────────────────────────

    #[test]
    fn cow_slice_try_debug() {
        let v = vec![1i32, 2, 3];
        let c: Cow<'_, [i32]> = Cow::Borrowed(v.as_slice());
        let dbg = try_format!("{:?}", c).unwrap();
        assert_eq!(dbg, "[1, 2, 3]");
    }

    #[test]
    fn cow_slice_try_clone_owned() {
        let v = vec![4i32, 5, 6];
        let c: Cow<'_, [i32]> = Cow::Owned(v);
        let cloned = c.try_clone().unwrap();
        assert!(matches!(cloned, Cow::Owned(_)));
        assert_eq!(&*cloned, &[4, 5, 6][..]);
    }
}
