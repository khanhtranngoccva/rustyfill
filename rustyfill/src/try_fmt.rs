//! Fallible formatting for types whose `Debug`/`Display` implementations may allocate.
//!
//! Many standard library types have `Debug` and/or `Display` implementations that
//! perform hidden heap allocations behind the scenes (e.g. floating-point with
//! precision specifiers, `PathBuf`, `Duration`, etc.). This module provides
//! [`TryDebug`] and [`TryDisplay`] traits that let you format values without
//! risking a panic from an unexpected allocation inside the formatter callback.
//!
//! # Design
//!
//! - [`TryDebug`] requires [`core::fmt::Debug`] as a supertrait.
//! - [`TryDisplay`] requires [`core::fmt::Display`] as a supertrait.
//! - Both traits return [`core::fmt::Result`] — the same type as canonical
//!   `Debug::fmt` / `Display::fmt`. No custom error enum is needed because
//!   [`core::fmt::Error`] is already an opaque, uninhabited sentinel that signals
//!   "the write failed" (either I/O or, in our case, a hidden allocation).
//! - Implementations must never call `format!()` or any function that implicitly
//!   allocates and panics on OOM.
//! - Well-known std types (primitives, tuples, arrays, markers, `Option`,
//!   `Result`, references, pointers, etc.) are implemented here.
//! - A derive macro exists for `TryDebug` on user-defined structs/enums.
//! - Passthrough declarative macros let users assert that their canonical
//!   `Debug`/`Display` impls are allocation-free.
//! - [`TryFmt`] is a generic wrapper that exposes `Debug` when the inner type
//!   is [`TryDebug`] and `Display` when the inner type is [`TryDisplay`].

use core::fmt;

// ── Traits ─────────────────────────────────────────────────────────────────────

/// A fallible analogue of [`core::fmt::Debug`].
///
/// Unlike [`core::fmt::Debug`], which can silently panic if its implementation
/// allocates under memory pressure, [`TryDebug::try_fmt`] returns a
/// [`fmt::Result`] so callers can detect failure.
///
/// Implementors must ensure that `try_fmt` never panics — it should only use
/// fallible write operations on the formatter.
pub trait TryDebug: fmt::Debug {
    /// Attempt to format this value using debug syntax.
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

/// A fallible analogue of [`core::fmt::Display`].
///
/// Unlike [`core::fmt::Display`], which can silently panic if its implementation
/// allocates under memory pressure, [`TryDisplay::try_fmt`] returns a
/// [`fmt::Result`] so callers can detect failure.
///
/// Implementors must ensure that `try_fmt` never panics — it should only use
/// fallible write operations on the formatter.
pub trait TryDisplay: fmt::Display {
    /// Attempt to format this value using display syntax.
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

// ── TryFmt wrapper ─────────────────────────────────────────────────────────────

/// A generic wrapper that exposes safe formatting traits based on the inner type.
///
/// If `T: TryDebug`, then [`TryFmt<T>`] implements [`fmt::Debug`].
/// If `T: TryDisplay`, then [`TryFmt<T>`] implements [`fmt::Display`].
///
/// This is useful in macro-generated code where format arguments need to be
/// wrapped so that the standard formatting machinery routes through the
/// fallible paths.
pub struct TryFmt<T>(pub T);

impl<T> TryFmt<T> {
    /// Wrap a value so that standard formatting routes through fallible paths.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: TryDebug> fmt::Debug for TryFmt<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.try_fmt(f)
    }
}

impl<T: TryDisplay> fmt::Display for TryFmt<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.try_fmt(f)
    }
}

// ── Macro helpers ──────────────────────────────────────────────────────────────

/// Implements both `TryDebug` and `TryDisplay` for types whose canonical
/// `Debug`/`Display` implementations are known to never allocate (primitives).
macro_rules! impl_try_fmt_primitives {
    ($($t:ty),* $(,)?) => {
        $(
            impl TryDebug for $t {
                #[inline]
                fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{:?}", self)
                }
            }

            impl TryDisplay for $t {
                #[inline]
                fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{}", self)
                }
            }
        )*
    };
}

/// Implements only `TryDebug` for types that implement `Debug` but not `Display`.
macro_rules! impl_try_debug_only {
    ($($t:ty),* $(,)?) => {
        $(
            impl TryDebug for $t {
                #[inline]
                fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{:?}", self)
                }
            }
        )*
    };
}

// ── Numeric primitives ─────────────────────────────────────────────────────────

impl_try_fmt_primitives!(u8, u16, u32, u64, u128, usize);
impl_try_fmt_primitives!(i8, i16, i32, i64, i128, isize);

// bool and char have allocation-free Debug and Display.
impl_try_fmt_primitives!(bool, char);

// () has allocation-free Debug; Display is not implemented so we do debug-only.
impl_try_debug_only!(());

// Note: f32/f64 are NOT included here because their Display/Debug implementations
// CAN allocate when precision specifiers are used (e.g. {:.5}). They are omitted
// from the blanket safe-list. Users who know they won't use precision specifiers
// can use the passthrough macros.

// ── References ──────────────────────────────────────────────────────────────────

impl<T: TryDebug> TryDebug for &T {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).try_fmt(f)
    }
}

impl<T: TryDisplay> TryDisplay for &T {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).try_fmt(f)
    }
}

// ── Raw pointers (Debug only — raw pointers don't implement Display) ───────────

impl<T> TryDebug for *const T {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:p}", *self)
    }
}

impl<T> TryDebug for *mut T {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:p}", *self)
    }
}

// ── Option ─────────────────────────────────────────────────────────────────────

impl<T: TryDebug> TryDebug for Option<T> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Some(v) => {
                f.write_str("Some(")?;
                v.try_fmt(f)?;
                f.write_str(")")
            }
            None => f.write_str("None"),
        }
    }
}

// ── Result ─────────────────────────────────────────────────────────────────────

impl<T: TryDebug, E: TryDebug> TryDebug for Result<T, E> {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ok(v) => {
                f.write_str("Ok(")?;
                v.try_fmt(f)?;
                f.write_str(")")
            }
            Err(e) => {
                f.write_str("Err(")?;
                e.try_fmt(f)?;
                f.write_str(")")
            }
        }
    }
}

// ── Marker types ───────────────────────────────────────────────────────────────

impl<T> TryDebug for core::marker::PhantomData<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PhantomData")
    }
}

impl<T: TryDebug> TryDebug for core::mem::ManuallyDrop<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (<Self as core::ops::Deref>::deref(self)).try_fmt(f)
    }
}

impl<T> TryDebug for core::mem::MaybeUninit<T> {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("MaybeUninit")
    }
}

impl TryDebug for core::marker::PhantomPinned {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PhantomPinned")
    }
}

// ── Str and slice references ───────────────────────────────────────────────────

// str's Debug and Display are allocation-free — safe to passthrough.
impl TryDebug for &str {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl TryDisplay for &str {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<T: TryDebug> TryDebug for &[T] {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", *self)
    }
}

// ── Tuple implementations (generated by proc-macro) ────────────────────────────

rustyfill_macros::try_debug_tuples!(12);

// ── Arrays [T; N] ──────────────────────────────────────────────────────────────

impl<T: TryDebug, const N: usize> TryDebug for [T; N] {
    #[inline]
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", *self)
    }
}

// ── Ranges ─────────────────────────────────────────────────────────────────────

macro_rules! impl_try_debug_for_range {
    ($($t:ty),* $(,)?) => {
        $(
            impl TryDebug for $t {
                #[inline]
                fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{:?}", *self)
                }
            }
        )*
    };
}

impl_try_debug_for_range!(
    core::ops::Range<usize>,
    core::ops::Range<i32>,
    core::ops::Range<u32>,
    core::ops::RangeFrom<usize>,
    core::ops::RangeTo<usize>,
    core::ops::RangeFull,
);

// ── Formatting macros ──────────────────────────────────────────────────────────
// The `try_format_args` proc-macro is defined in `rustyfill-macros` and re-exported
// from the crate root. It selectively wraps display/debug arguments in TryFmt while
// leaving width/precision specifier arguments unwrapped.
// The helper macros (try_println, try_print, try_write, try_writeln, try_format)
// are now proc-macros defined in rustyfill-macros and re-exported from the crate root.

// ── Passthrough macros ─────────────────────────────────────────────────────────

/// Implements `TryDebug` by delegating to the canonical `Debug` implementation.
///
/// Use this macro when you can verify that your type's `Debug` impl never performs
/// hidden allocations (e.g. it only uses `f.write_str()` and delegates to other
/// allocation-free types). The macro generates a thin wrapper that passes through
/// the existing `Debug::fmt` result directly.
///
/// # Example
///
/// ```ignore
/// #[derive(Debug)]
/// struct MyPoint { x: i32, y: i32 }
///
/// rustyfill::debug_passthrough!(MyPoint);
/// ```
#[macro_export]
macro_rules! debug_passthrough {
    ($ty:ty) => {
        impl $crate::try_fmt::TryDebug for $ty {
            #[inline]
            fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::Debug::fmt(self, f)
            }
        }
    };
}

/// Implements `TryDisplay` by delegating to the canonical `Display` implementation.
///
/// Use this macro when you can verify that your type's `Display` impl never performs
/// hidden allocations. The macro generates a thin wrapper that passes through the
/// existing `Display::fmt` result directly.
///
/// # Example
///
/// ```ignore
/// struct MyLabel(i32);
///
/// impl std::fmt::Display for MyLabel {
///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
///         write!(f, "label-{}", self.0)
///     }
/// }
///
/// rustyfill::display_passthrough!(MyLabel);
/// ```
#[macro_export]
macro_rules! display_passthrough {
    ($ty:ty) => {
        impl $crate::try_fmt::TryDisplay for $ty {
            #[inline]
            fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::Display::fmt(self, f)
            }
        }
    };
}

// ── OOM safety tests ───────────────────────────────────────────────────────────
// Every TryDebug/TryDisplay implementation must survive with all allocations
// failing. If a formatter secretly allocates (e.g. via format! or to_string()),
// the process aborts and this test catches it.
//
// Data is constructed OUTSIDE with_policy() so that only formatting is tested
// under OOM conditions, not allocation during setup.

#[cfg(test)]
mod oom_tests {
    use super::*;
    use crate::try_fmt::{TryDebug, TryDisplay, TryFmt};
    use rustyfill_test_allocator::{FailPolicy, with_policy};

    /// Minimal writer that discards everything without allocating.
    struct NoopWriter;
    impl fmt::Write for NoopWriter {
        fn write_str(&mut self, _: &str) -> fmt::Result {
            Ok(())
        }
    }

    /// Run TryDebug::try_fmt under OOM via TryFmt wrapper + fmt::write.
    /// The TryFmt<T: TryDebug> type implements Debug which calls try_fmt,
    /// and fmt::write constructs a real Formatter internally.
    fn assert_try_debug_no_alloc<T: TryDebug>(value: &T) -> bool {
        with_policy(FailPolicy::fail_all_alloc(), || {
            let mut w = NoopWriter;
            fmt::write(&mut w, format_args!("{:?}", TryFmt(value))).is_ok()
        })
    }

    /// Run TryDisplay::try_fmt under OOM via TryFmt wrapper + fmt::write.
    fn assert_try_display_no_alloc<T: TryDisplay>(value: &T) -> bool {
        with_policy(FailPolicy::fail_all_alloc(), || {
            let mut w = NoopWriter;
            fmt::write(&mut w, format_args!("{}", TryFmt(value))).is_ok()
        })
    }

    // ── str / String ──────────────────────────────────────────────────────

    #[test]
    fn try_debug_str_empty_no_alloc() {
        let s: &str = "";
        assert!(assert_try_debug_no_alloc(&s));
    }

    #[test]
    fn try_debug_str_ascii_no_alloc() {
        let s: &str = "hello world";
        assert!(assert_try_debug_no_alloc(&s));
    }

    #[test]
    fn try_debug_str_unicode_no_alloc() {
        let s: &str = "konnichiwa cafe";
        assert!(assert_try_debug_no_alloc(&s));
    }

    #[test]
    fn try_debug_str_escape_chars_no_alloc() {
        let s: &str = "tab\there\nnewline\rquote\"backslash\\";
        assert!(assert_try_debug_no_alloc(&s));
    }

    #[test]
    fn try_display_str_no_alloc() {
        let s: &str = "display test";
        assert!(assert_try_display_no_alloc(&s));
    }

    #[test]
    fn try_debug_string_empty_no_alloc() {
        let s = String::new();
        assert!(assert_try_debug_no_alloc(&s));
    }

    #[test]
    fn try_debug_string_populated_no_alloc() {
        let s = String::from("populated string");
        assert!(assert_try_debug_no_alloc(&s));
    }

    #[test]
    fn try_display_string_no_alloc() {
        let s = String::from("display from string");
        assert!(assert_try_display_no_alloc(&s));
    }

    // ── Vec ────────────────────────────────────────────────────────────────

    #[test]
    fn try_debug_vec_empty_no_alloc() {
        let v: Vec<i32> = Vec::new();
        assert!(assert_try_debug_no_alloc(&v));
    }

    #[test]
    fn try_debug_vec_populated_no_alloc() {
        let v = vec![1, 2, 3, 4, 5];
        assert!(assert_try_debug_no_alloc(&v));
    }

    #[test]
    fn try_debug_vec_strings_no_alloc() {
        let v = vec![String::from("a"), String::from("b")];
        assert!(assert_try_debug_no_alloc(&v));
    }

    // ── VecDeque ───────────────────────────────────────────────────────────

    #[test]
    fn try_debug_vecdeque_empty_no_alloc() {
        let v: std::collections::VecDeque<i32> = std::collections::VecDeque::new();
        assert!(assert_try_debug_no_alloc(&v));
    }

    #[test]
    fn try_debug_vecdeque_populated_no_alloc() {
        let mut v = std::collections::VecDeque::new();
        v.push_back(1);
        v.push_back(2);
        v.push_front(0);
        assert!(assert_try_debug_no_alloc(&v));
    }

    // ── HashMap ────────────────────────────────────────────────────────────

    #[test]
    fn try_debug_hashmap_empty_no_alloc() {
        let m: std::collections::HashMap<&str, i32> = std::collections::HashMap::new();
        assert!(assert_try_debug_no_alloc(&m));
    }

    #[test]
    fn try_debug_hashmap_populated_no_alloc() {
        let mut m = std::collections::HashMap::new();
        m.insert("key", 42);
        m.insert("other", 99);
        assert!(assert_try_debug_no_alloc(&m));
    }

    // ── HashSet ────────────────────────────────────────────────────────────

    #[test]
    fn try_debug_hashset_empty_no_alloc() {
        let s: std::collections::HashSet<i32> = std::collections::HashSet::new();
        assert!(assert_try_debug_no_alloc(&s));
    }

    #[test]
    fn try_debug_hashset_populated_no_alloc() {
        let mut s = std::collections::HashSet::new();
        s.insert(1);
        s.insert(2);
        s.insert(3);
        assert!(assert_try_debug_no_alloc(&s));
    }

    // ── Box ────────────────────────────────────────────────────────────────

    #[test]
    fn try_debug_box_primitive_no_alloc() {
        let b: Box<i32> = Box::new(42);
        assert!(assert_try_debug_no_alloc(&b));
    }

    #[test]
    fn try_debug_box_string_no_alloc() {
        let b: Box<String> = Box::new(String::from("boxed string"));
        assert!(assert_try_debug_no_alloc(&b));
    }

    #[test]
    fn try_debug_box_vec_no_alloc() {
        let b: Box<Vec<u8>> = Box::new(vec![1, 2, 3]);
        assert!(assert_try_debug_no_alloc(&b));
    }

    // ── Arc ────────────────────────────────────────────────────────────────

    #[test]
    fn try_debug_arc_primitive_no_alloc() {
        let a: std::sync::Arc<i32> = std::sync::Arc::new(42);
        assert!(assert_try_debug_no_alloc(&a));
    }

    #[test]
    fn try_debug_arc_string_no_alloc() {
        let a: std::sync::Arc<String> = std::sync::Arc::new(String::from("arc string"));
        assert!(assert_try_debug_no_alloc(&a));
    }

    // ── PathBuf (sized — uses generic helper) ──────────────────────────────

    #[test]
    fn try_debug_pathbuf_no_alloc() {
        let pb = std::path::PathBuf::from("/tmp/test/file.txt");
        assert!(assert_try_debug_no_alloc(&pb));
    }

    #[test]
    fn try_debug_pathbuf_unicode_no_alloc() {
        let pb = std::path::PathBuf::from("/home/user/docs");
        assert!(assert_try_debug_no_alloc(&pb));
    }

    // ── OsString (sized — uses generic helper) ─────────────────────────────

    #[test]
    fn try_debug_osstring_no_alloc() {
        let s = std::ffi::OsString::from("os string data");
        assert!(assert_try_debug_no_alloc(&s));
    }

    // ── CString (sized — uses generic helper) ──────────────────────────────

    #[test]
    fn try_debug_cstring_no_alloc() {
        let cs = std::ffi::CString::new("cstring data").unwrap();
        assert!(assert_try_debug_no_alloc(&cs));
    }

    // ── Primitives (sanity check) ──────────────────────────────────────────

    #[test]
    fn try_debug_primitives_no_alloc() {
        let i: i32 = -42;
        let u: u64 = 99;
        let b: bool = true;
        let ch: char = 'Z';
        let unit = ();
        assert!(assert_try_debug_no_alloc(&i));
        assert!(assert_try_debug_no_alloc(&u));
        assert!(assert_try_debug_no_alloc(&b));
        assert!(assert_try_debug_no_alloc(&ch));
        assert!(assert_try_debug_no_alloc(&unit));
    }

    // ── Compound types ─────────────────────────────────────────────────────

    #[test]
    fn try_debug_option_some_no_alloc() {
        let o: Option<String> = Some(String::from("inner"));
        assert!(assert_try_debug_no_alloc(&o));
    }

    #[test]
    fn try_debug_option_none_no_alloc() {
        let o: Option<i32> = None;
        assert!(assert_try_debug_no_alloc(&o));
    }

    #[test]
    fn try_debug_result_ok_no_alloc() {
        let r: Result<String, i32> = Ok(String::from("success"));
        assert!(assert_try_debug_no_alloc(&r));
    }

    #[test]
    fn try_debug_result_err_no_alloc() {
        let r: Result<i32, String> = Err(String::from("failure"));
        assert!(assert_try_debug_no_alloc(&r));
    }

    #[test]
    fn try_debug_tuple_no_alloc() {
        let t = (42, String::from("x"), true);
        assert!(assert_try_debug_no_alloc(&t));
    }

    #[test]
    fn try_debug_array_no_alloc() {
        let a: [i32; 3] = [1, 2, 3];
        assert!(assert_try_debug_no_alloc(&a));
    }

    #[test]
    fn try_debug_slice_no_alloc() {
        let v = vec![10, 20, 30];
        let s: &[i32] = &v;
        assert!(assert_try_debug_no_alloc(&s));
    }

    // ── Nested compound types ──────────────────────────────────────────────

    #[test]
    fn try_debug_nested_vec_of_strings_no_alloc() {
        let v: Vec<Vec<String>> = vec![
            vec![String::from("a"), String::from("b")],
            vec![String::from("c")],
        ];
        assert!(assert_try_debug_no_alloc(&v));
    }

    #[test]
    fn try_debug_boxed_arc_vec_no_alloc() {
        let val: Box<std::sync::Arc<Vec<String>>> =
            Box::new(std::sync::Arc::new(vec![String::from("nested")]));
        assert!(assert_try_debug_no_alloc(&val));
    }

    // ── Display wrapper types (path + os_str) ────────────────────────────────

    #[test]
    fn try_display_path_display_no_alloc() {
        let pb = std::path::PathBuf::from("/tmp/test/file.txt");
        let display = pb.display();
        assert!(with_policy(FailPolicy::fail_all_alloc(), || {
            let mut w = NoopWriter;
            fmt::write(&mut w, format_args!("{}", TryFmt(display))).is_ok()
        }));
    }

    #[test]
    fn try_debug_path_display_no_alloc() {
        let pb = std::path::PathBuf::from("/tmp/test/file.txt");
        let display = pb.display();
        assert!(with_policy(FailPolicy::fail_all_alloc(), || {
            let mut w = NoopWriter;
            fmt::write(&mut w, format_args!("{:?}", TryFmt(display))).is_ok()
        }));
    }

    #[test]
    fn try_display_osstr_display_no_alloc() {
        let os = std::ffi::OsString::from("os string data");
        let display = os.display();
        assert!(with_policy(FailPolicy::fail_all_alloc(), || {
            let mut w = NoopWriter;
            fmt::write(&mut w, format_args!("{}", TryFmt(display))).is_ok()
        }));
    }

    #[test]
    fn try_debug_osstr_display_no_alloc() {
        let os = std::ffi::OsString::from("os string data");
        let display = os.display();
        assert!(with_policy(FailPolicy::fail_all_alloc(), || {
            let mut w = NoopWriter;
            fmt::write(&mut w, format_args!("{:?}", TryFmt(display))).is_ok()
        }));
    }
}
