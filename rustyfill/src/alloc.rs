//! Custom allocation types and allocation errors.
//!
//! By default (and always on stable), this crate provides its own
//! [`AllocError`] ponyfill and a layout-matched [`TryReserveErrorKind`]
//! ponyfill so that downstream crates can name these types without enabling
//! unstable feature gates. When the `allocator-api` Cargo feature is enabled
//! on nightly, the real `core::alloc::AllocError` and
//! `alloc::collections::TryReserveErrorKind` replace the ponyfills, giving
//! identity with the standard library types.
//!
//! [`TryReserveError`] is a re-export of the standard library's
//! [`lang_alloc::collections::TryReserveError`]. There is exactly one such type,
//! and every call site sees it directly — no wrapper. Its constructor and
//! `.kind()` accessor are gated behind the unstable `try_reserve_kind` feature,
//! so the ergonomic constructors and accessors live on the
//! [`TryReserveErrorExt`] extension trait defined below, which is re-exported
//! through the prelude. Those methods construct and return the real
//! standard-library type itself, bridging from a layout-matched mirror generated
//! by `rustyfill-sys`.

use crate::try_fmt::{TryDebug, TryDisplay, helpers::FormatterExt};
use lang_core::alloc::Layout;
#[cfg(not(allocator_api_enabled))]
use lang_core::error;
#[cfg(allocator_api_enabled)]
use lang_core::fmt;
#[cfg(not(allocator_api_enabled))]
use lang_core::fmt::{self, Debug};

// The layout-matched mirror of the private `TryReserveErrorKind`, generated
// directly from the standard library source by `rustyfill-sys`. Referencing it
// here means any change to the real type in std breaks compilation at the
// source code layer before it can silently corrupt a transmute.
use rustyfill_sys::std::collections::TryReserveError as SysTryReserveError;
use rustyfill_sys::std::collections::TryReserveErrorKind as SysTryReserveErrorKind;

pub mod arc;
pub mod boxed;
#[cfg(feature = "std")]
pub mod btrees;
pub mod ffi;
pub mod rc;
pub mod string;
pub mod vec;
pub mod vecdeque;

// ── AllocError ────────────────────────────────────────────────────────────────
//
// When `allocator-api` is enabled on nightly, we re-export the real
// `core::alloc::AllocError` (a zero-sized unit struct from core). Otherwise
// (stable, or nightly without the feature), we provide a ponyfill with the
// same shape so downstream crates can name the type without feature gates.

#[cfg(not(allocator_api_enabled))]
mod alloc_error_ponyfill {
    use super::*;

    /// Allocation error returned when a heap allocation fails.
    /// Unit struct, matching the shape of the standard library's `AllocError`.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct AllocError;

    impl fmt::Debug for AllocError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            TryDebug::try_fmt(self, f)
        }
    }

    impl fmt::Display for AllocError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            TryDisplay::try_fmt(self, f)
        }
    }

    impl TryDebug for AllocError {
        fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("AllocError")
        }
    }

    impl TryDisplay for AllocError {
        fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "allocation failed")
        }
    }

    impl error::Error for AllocError {}
}

#[cfg(not(allocator_api_enabled))]
pub use alloc_error_ponyfill::AllocError;

// With `allocator-api` on nightly, `AllocError` is the real (foreign) unit
// struct from core, which already provides `Debug`, `Display`, and
// `error::Error`. We only add the fallible formatting traits, delegating to
// those std impls (a unit struct prints fixed text and never allocates).
#[cfg(allocator_api_enabled)]
pub use lang_core::alloc::AllocError;

#[cfg(allocator_api_enabled)]
impl TryDebug for AllocError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

#[cfg(allocator_api_enabled)]
impl TryDisplay for AllocError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

// ── TryReserveError ───────────────────────────────────────────────────────────
//
// We re-export the standard library's own `TryReserveError` so that there is
// exactly one such type and every call site sees it directly — no wrapper. On
// stable the type's only constructor (`From<TryReserveErrorKind>`) and its
// `.kind()` accessor are gated behind the unstable `try_reserve_kind` feature,
// and the `kind` field is private, so we cannot name the variants or construct
// the value through the public API. Instead the [`TryReserveErrorExt`] methods
// below build the layout-matched sys mirror (`SysTryReserveError`, generated
// from the std source by `rustyfill-sys`) and `transmute` it into the real
// type.
//
// Why the mirror comes from `rustyfill-sys`: those bindings are emitted
// directly from the standard library source, so if the real `TryReserveError`
// or `TryReserveErrorKind` ever changes shape, the bindings fail to compile —
// surfacing the drift at build time rather than letting a hand-written mirror
// silently diverge and corrupt transmuted values.
//
// Soundness: the transmute relies on the deterministic field layout of
// `TryReserveError` matching the sys mirror. That guarantee is enforced at
// build time — `build.rs` aborts compilation when `-Zrandomize-layout` is
// active, because layout randomization would shuffle the very field offsets and
// discriminant encodings this construction depends on. With randomization off,
// the standard library's default-repr layout is stable across builds.

/// The standard library's capacity-reservation error, re-exported verbatim.
///
/// This is `lang_alloc::collections::TryReserveError` itself, not a wrapper.
/// Use the [`TryReserveErrorExt`] trait (re-exported through the prelude) to
/// construct instances and inspect which kind of failure occurred.
pub use lang_alloc::collections::TryReserveError;

// With `allocator-api` on nightly, the real `TryReserveErrorKind` is available
// (behind `try_reserve_kind`, enabled in `lib.rs`), so we alias it directly.
// Otherwise we provide a layout-matched ponyfill. Either way, downstream code
// names the type `crate::alloc::TryReserveErrorKind`.
#[cfg(allocator_api_enabled)]
pub use lang_alloc::collections::TryReserveErrorKind;

/// A ponyfill of the private `alloc::collections::TryReserveErrorKind`, providing
/// **error-kind enumeration** when the real enum is unreachable (stable, or
/// nightly without the `allocator-api` feature).
///
/// This is a *ponyfill*, not a full replica: it exposes the same two variants so
/// callers can enumerate which kind of reservation failure occurred, but it does
/// not carry the real type's hidden `non_exhaustive` field or any future fields
/// the standard library may add. It is deliberately independent of the
/// `rustyfill-sys` mirror so that stable-side matching does not couple to the
/// generated bindings.
///
/// It is layout-matched to the real enum under the default representation — a
/// niche-free two-variant enum whose second variant carries a [`Layout`] — so
/// decoding a `TryReserveError` into it via `transmute_copy` is sound (same
/// guarantee as the construction path: `-Zrandomize-layout` is rejected at build
/// time).
#[cfg(not(allocator_api_enabled))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TryReserveErrorKind {
    /// Error due to the computed capacity exceeding the collection's maximum.
    CapacityOverflow,
    /// The memory allocator returned an error.
    AllocError { layout: Layout },
}

/// Compile-time proof that the sys mirror's size and alignment match the real
/// type. If the standard library ever changes the layout of `TryReserveError`,
/// these assertions fail at compile time rather than silently corrupting values
/// at runtime. (The sys crate already breaks first if the *shape* changes; this
/// catches any residual size/alignment skew.)
const _: () = assert!(
    core::mem::size_of::<TryReserveError>() == core::mem::size_of::<SysTryReserveError>(),
    "TryReserveError layout changed: size mismatch with sys mirror"
);
const _: () = assert!(
    core::mem::align_of::<TryReserveError>() == core::mem::align_of::<SysTryReserveError>(),
    "TryReserveError layout changed: alignment mismatch with sys mirror"
);

/// Extension trait providing constructors and accessors for
/// [`TryReserveError`]. Re-exported through the prelude so callers can build
/// and inspect reservation errors without reaching into the private fields.
pub trait TryReserveErrorExt {
    /// Construct an error representing a failed heap allocation, carrying the
    /// [`Layout`] of the allocation that failed.
    fn new_alloc(layout: Layout) -> Self
    where
        Self: Sized;

    /// Construct an error representing a capacity computation that overflowed
    /// before any allocation was attempted.
    fn new_capacity_overflow() -> Self
    where
        Self: Sized;

    /// Enumerate which kind of reservation failure occurred, returning the
    /// [`TryReserveErrorKind`] variant.
    ///
    /// Named `error_kind` rather than `kind` so that it does not collide with the
    /// standard library's own (unstable, `try_reserve_kind`) inherent `.kind()`
    /// accessor on nightly. On stable the returned enum is a ponyfill: it
    /// carries the same two variants but not the real type's hidden
    /// `non_exhaustive` field. On nightly it aliases the real std enum.
    ///
    /// On stable the value is decoded by reading it through the layout-matched
    /// sys mirror and rebuilding the result from the discriminant and `Layout`;
    /// on nightly it delegates to the inherent `.kind()`. This is the inverse of
    /// the construction path in [`new_alloc`](Self::new_alloc) /
    /// [`new_capacity_overflow`](Self::new_capacity_overflow).
    fn error_kind(&self) -> TryReserveErrorKind;

    /// Returns true if the failure was a failed heap allocation.
    fn is_alloc(&self) -> bool;

    /// Returns true if the failure was a capacity arithmetic overflow.
    fn is_capacity_overflow(&self) -> bool;
}

impl TryReserveErrorExt for TryReserveError {
    fn new_alloc(layout: Layout) -> Self {
        let mirror = SysTryReserveError {
            kind: SysTryReserveErrorKind::AllocError {
                layout,
                non_exhaustive: (),
            },
        };
        // SAFETY: `SysTryReserveError` is generated from the std source with the
        // same size, alignment, and default-repr field layout as
        // `TryReserveError` (asserted above), and `-Zrandomize-layout` is
        // rejected at build time. The `AllocError` discriminant selects the
        // `Layout` payload, which holds a well-formed `Layout` value.
        unsafe { core::mem::transmute(mirror) }
    }

    fn new_capacity_overflow() -> Self {
        let mirror = SysTryReserveError {
            kind: SysTryReserveErrorKind::CapacityOverflow,
        };
        // SAFETY: As in `new_alloc`; the `CapacityOverflow` variant carries no
        // payload, so the resulting value is trivially valid.
        unsafe { core::mem::transmute(mirror) }
    }

    #[cfg(allocator_api_enabled)]
    fn error_kind(&self) -> TryReserveErrorKind {
        // With `allocator-api` on nightly the real `.kind()` accessor is
        // available (behind `try_reserve_kind`), so we use it directly —
        // no transmute needed.
        self.kind()
    }

    #[cfg(not(allocator_api_enabled))]
    fn error_kind(&self) -> TryReserveErrorKind {
        // Decode through the sys mirror, whose fields are always public and whose
        // layout provably matches the real type. We only read the discriminant
        // and the `Layout`, then rebuild the ponyfill `TryReserveErrorKind` with
        // plain literal patterns — no need to name or hold the hidden
        // `non_exhaustive` field.
        //
        // SAFETY: `SysTryReserveError` is generated from the std source with the
        // same size, alignment, and default-repr field layout as
        // `TryReserveError` (asserted above), and `-Zrandomize-layout` is
        // rejected at build time. Reading the decoded `kind` therefore yields a
        // valid `SysTryReserveErrorKind`.
        let decoded: SysTryReserveError = unsafe { core::mem::transmute_copy(self) };
        match decoded.kind {
            SysTryReserveErrorKind::CapacityOverflow => TryReserveErrorKind::CapacityOverflow,
            SysTryReserveErrorKind::AllocError { layout, .. } => {
                TryReserveErrorKind::AllocError { layout }
            }
        }
    }

    fn is_alloc(&self) -> bool {
        matches!(self.error_kind(), TryReserveErrorKind::AllocError { .. })
    }

    fn is_capacity_overflow(&self) -> bool {
        matches!(self.error_kind(), TryReserveErrorKind::CapacityOverflow)
    }
}

impl TryDebug for TryReserveError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.error_kind() {
            TryReserveErrorKind::CapacityOverflow => {
                f.write_str("TryReserveError::CapacityOverflow")
            }
            TryReserveErrorKind::AllocError { .. } => {
                f.try_debug_struct("TryReserveError::AllocError").finish()
            }
        }
    }
}

impl TryDisplay for TryReserveError {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The std Display impl prints fixed text ("memory allocation failed
        // because …") and never allocates, so delegating is safe.
        fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_overflow_round_trip() {
        let err = TryReserveError::new_capacity_overflow();
        assert!(err.is_capacity_overflow());
        assert!(!err.is_alloc());
        assert!(matches!(
            err.error_kind(),
            TryReserveErrorKind::CapacityOverflow
        ));
    }

    /// Read the `(size, align)` of the `Layout` carried by an
    /// [`AllocError`](TryReserveErrorKind::AllocError) kind, without naming the
    /// variant's fields (the real type's field is private, so we cannot bind it
    /// by name). Both toolchains expose the same shape here — on stable the
    /// ponyfill is layout-matched to the real enum, on nightly it *is* the real
    /// enum — so a single `transmute_copy` into the sys mirror works uniformly.
    fn alloc_layout_of(kind: &TryReserveErrorKind) -> Option<(usize, usize)> {
        if !matches!(kind, TryReserveErrorKind::AllocError { .. }) {
            return None;
        }
        // SAFETY: `SysTryReserveErrorKind` is generated from the std source with
        // the same size, alignment, and default-repr layout as
        // `TryReserveErrorKind`, and `-Zrandomize-layout` is rejected at build
        // time. We have just confirmed the discriminant selects the
        // `AllocError` arm, which carries exactly one `Layout` payload, so the
        // copy yields a valid value.
        let decoded: SysTryReserveErrorKind = unsafe { core::mem::transmute_copy(kind) };
        match decoded {
            SysTryReserveErrorKind::AllocError { layout, .. } => {
                Some((layout.size(), layout.align()))
            }
            SysTryReserveErrorKind::CapacityOverflow => unreachable!(),
        }
    }

    #[test]
    fn alloc_error_preserves_layout() {
        let layout = Layout::new::<u64>();
        let err = TryReserveError::new_alloc(layout);
        assert!(err.is_alloc());
        assert!(!err.is_capacity_overflow());
        let kind = err.error_kind();
        assert_eq!(
            alloc_layout_of(&kind),
            Some((layout.size(), layout.align())),
            "decoded layout must round-trip"
        );
    }

    #[test]
    fn distinct_layouts_are_distinguished() {
        let small = TryReserveError::new_alloc(Layout::new::<u8>());
        let big = TryReserveError::new_alloc(Layout::new::<[u8; 256]>());
        assert_eq!(alloc_layout_of(&small.error_kind()), Some((1, 1)));
        assert_eq!(alloc_layout_of(&big.error_kind()), Some((256, 1)));
    }

    #[test]
    fn display_delegates_to_std() {
        use lang_alloc::format;
        let co = TryReserveError::new_capacity_overflow();
        let al = TryReserveError::new_alloc(Layout::new::<u8>());
        // Both variants print the std "memory allocation failed" prefix.
        assert!(format!("{co}").contains("memory allocation failed"));
        assert!(format!("{al}").contains("memory allocation failed"));
    }

    #[test]
    fn clone_and_equality_round_trip() {
        let original = TryReserveError::new_alloc(Layout::new::<u32>());
        let cloned = original.clone();
        assert_eq!(original, cloned);
        assert!(matches!(
            cloned.error_kind(),
            TryReserveErrorKind::AllocError { .. }
        ));
    }
}
