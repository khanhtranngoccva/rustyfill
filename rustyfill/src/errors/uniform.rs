//! Shared [`TryDisplay`] / [`TryDebug`] arm bodies for the uniform fallible-collection
//! error enums.
//!
//! Most of these error types wrap the same canonical variants —
//! `{Alloc(AllocError), Reserve(TryReserveError), Clone(TryCloneError), Overflow,
//! Other(&'static str)}` plus an occasional extra like `Locked` or `Nul` — under a
//! single human-readable prefix such as `"vector"` or `"hash map"`. Hand-writing the
//! per-variant match arms for every type was pure duplication: each arm is a
//! cyclomatic branch with no logic, which inflates CRAP complexity while contributing
//! nothing testable. These helpers centralize the arm bodies so each error type's
//! impl reduces to a thin delegation, and the helpers themselves are covered by the
//! tests below (which construct real error variants directly).

// FIXME: where is TryDisplay and Display? Should not we just make this deal with TryDisplay and TryDebug only?
use crate::try_fmt::{TryDebug, TryDisplay, helpers::FormatterExt};
use lang_core::fmt;

/// Renders a fixed-message `Display` arm: `"{prefix} operation failed: {msg}"`.
///
/// Used for the `Alloc`, `Overflow`, and `Locked` variants, whose detail is a
/// constant string rather than a wrapped value.
#[inline]
pub(crate) fn display_fixed(f: &mut fmt::Formatter<'_>, prefix: &str, msg: &str) -> fmt::Result {
    write!(f, "{prefix} operation failed: {msg}")
}

/// Renders a delegated `Display` arm: `"{prefix} operation failed: {e}"`, where the
/// detail comes from the wrapped value's own `Display`.
///
/// Used for the `Reserve`, `Clone`, and `Other` variants.
#[inline]
pub(crate) fn display_delegated<T: fmt::Display>(
    f: &mut fmt::Formatter<'_>,
    prefix: &str,
    e: &T,
) -> fmt::Result {
    write!(f, "{prefix} operation failed: {e}")
}

/// Renders a `TryDebug` arm for a tuple variant carrying a single field, producing
/// `<name> { 0: <field> }`. The `name` should be the fully-qualified
/// `Type::Variant` form to match the previous hand-written output.
#[inline]
pub(crate) fn debug_field<'f, T: TryDebug>(
    f: &mut fmt::Formatter<'f>,
    name: &'f str,
    value: &T,
) -> fmt::Result {
    f.try_debug_struct(name).field("0", value).finish()
}

/// Renders a `TryDebug` arm for a unit variant, producing just `<name>`.
#[inline]
pub(crate) fn debug_unit(f: &mut fmt::Formatter<'_>, name: &str) -> fmt::Result {
    f.write_str(name)
}

// ── Tests ────────────────────────────────────────────────────────────────────────
//
// These exercise every helper across representative inputs. The per-data-structure
// error-enum coverage (driving each collection's error type through all its variants)
// lives in the respective modules' own test suites — see e.g.
// `alloc::vec::tests::vec_error_*`, `std::hashmap::tests::hashmap_error_*`, etc.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::{TryReserveError, TryReserveErrorExt};
    use lang_alloc::format;
    use lang_alloc::string::String;
    use lang_core::fmt::Write as _;

    /// A `TryReserveError` instance for exercising the `Reserve` arm.
    fn reserve_err() -> TryReserveError {
        TryReserveError::new_capacity_overflow()
    }

    /// Formats a value via its `Display` impl into a fresh String.
    fn render_display(e: &impl fmt::Display) -> String {
        let mut s = String::new();
        // Our error Display impls only call `write!` on literals/wrapped values,
        // so this cannot fail in practice; ignore the infallible-in-practice result.
        let _ = write!(&mut s, "{e}");
        s
    }

    /// Captures the `TryDebug` rendering of a value.
    fn render_trydebug(e: &impl TryDebug) -> String {
        struct Cap<'a>(&'a dyn TryDebug);
        impl fmt::Debug for Cap<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.try_fmt(f)
            }
        }
        format!("{:?}", Cap(e))
    }

    // ── Direct helper coverage ─────────────────────────────────────────────────

    #[test]
    fn display_fixed_renders_prefix_and_message() {
        let err = crate::alloc::vec::TryVecError::Overflow;
        assert_eq!(
            render_display(&err),
            "vector operation failed: capacity calculation overflowed"
        );
    }

    #[test]
    fn display_delegated_renders_wrapped_value() {
        let err = crate::alloc::vec::TryVecError::Other("boom");
        assert_eq!(render_display(&err), "vector operation failed: boom");
    }

    #[test]
    fn debug_field_renders_single_field_struct() {
        let err = crate::alloc::vec::TryVecError::Reserve(reserve_err());
        let got = render_trydebug(&err);
        assert!(
            got.contains("TryVecError::Reserve"),
            "missing tag in {got:?}"
        );
    }

    #[test]
    fn debug_unit_renders_bare_name() {
        let err = crate::alloc::vec::TryVecError::Overflow;
        assert_eq!(render_trydebug(&err), "TryVecError::Overflow");
    }
}
