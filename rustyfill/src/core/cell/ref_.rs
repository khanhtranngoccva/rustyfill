use crate::try_fmt::{TryDebug, TryDisplay};
use lang_core::cell::{Ref, RefMut};
use lang_core::fmt;

// ── TryDebug / TryDisplay for Ref<'_, T> and RefMut<'_, T> ─────────────────────
// Both deref to T, so formatting delegates to the inner value. Mirrors std's
// Debug impls which also forward to the contained value.

impl<T: ?Sized + TryDebug> TryDebug for Ref<'_, T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).try_fmt(f)
    }
}

impl<T: ?Sized + TryDisplay> TryDisplay for Ref<'_, T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).try_fmt(f)
    }
}

impl<T: ?Sized + TryDebug> TryDebug for RefMut<'_, T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).try_fmt(f)
    }
}

impl<T: ?Sized + TryDisplay> TryDisplay for RefMut<'_, T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).try_fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use crate::try_format;
    use lang_alloc::string::String;
    use lang_core::cell::RefCell;

    #[test]
    fn ref_try_debug_forwards() {
        let cell = RefCell::new(42i32);
        let r = cell.borrow();
        let dbg = try_format!("{:?}", r).unwrap();
        assert_eq!(dbg, "42");
    }

    #[test]
    fn ref_try_display_forwards() {
        let cell = RefCell::new(String::from("hello"));
        let r = cell.borrow();
        let disp = try_format!("{}", r).unwrap();
        assert_eq!(disp, "hello");
    }

    #[test]
    fn ref_mut_try_debug_forwards() {
        let cell = RefCell::new(7u8);
        let mut r = cell.borrow_mut();
        *r = 9;
        let dbg = try_format!("{:?}", r).unwrap();
        assert_eq!(dbg, "9");
    }

    #[test]
    fn ref_mut_try_display_forwards() {
        let cell = RefCell::new(String::from("world"));
        let mut r = cell.borrow_mut();
        r.push('!');
        let disp = try_format!("{}", r).unwrap();
        assert_eq!(disp, "world!");
    }
}
