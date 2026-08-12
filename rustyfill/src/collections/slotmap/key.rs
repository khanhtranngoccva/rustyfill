//! Key types and the [`Key`] trait.
//!
//! Keys are 64-bit values composed of a 32-bit slot index and a 32-bit version
//! counter. The version ensures that after a slot is freed and reused, old keys
//! pointing at that slot remain invalid (ABA protection).

use lang_core::fmt::{self, Debug};
use lang_core::hash::{Hash, Hasher};
use lang_core::num::NonZeroU32;
use crate::try_fmt::TryDebug;

/// The raw data stored inside any slot-map key.
///
/// A key consists of a slot index (`idx`) and a monotonically increasing
/// version (`version`). Two keys compare equal only when both fields match.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct KeyData {
    idx: u32,
    version: NonZeroU32,
}

impl KeyData {
    /// Construct a new `KeyData`. Panics in debug mode if `version == 0`.
    pub(crate) const fn new(idx: u32, version: u32) -> Self {
        debug_assert!(version > 0);
        Self {
            idx,
            version: unsafe { NonZeroU32::new_unchecked(version | 1) },
        }
    }

    /// Return the null key — always invalid and distinct from any real key.
    pub(crate) fn null() -> Self {
        Self::new(u32::MAX, 1)
    }

    /// Returns the slot index component of this key.
    pub(crate) fn idx(self) -> u32 {
        self.idx
    }

    /// Returns the version as a raw `u32`.
    pub(crate) fn version_raw(self) -> u32 {
        self.version.get()
    }

    /// Whether this key is the null sentinel.
    pub fn is_null(self) -> bool {
        self.idx == u32::MAX
    }

    /// Encode as a 64-bit integer for FFI interop.
    ///
    /// Passing the result to [`Self::from_ffi`] yields an equal key.
    pub fn as_ffi(self) -> u64 {
        (u64::from(self.version.get()) << 32) | u64::from(self.idx)
    }

    /// Decode from an FFI handle produced by [`Self::as_ffi`].
    ///
    /// Behavior is safe but unspecified if `value` did not originate from
    /// `as_ffi()`.
    pub const fn from_ffi(value: u64) -> Self {
        let idx = value & 0xFFFF_FFFF;
        let version = (value >> 32) | 1;
        Self::new(idx as u32, version as u32)
    }
}

impl Debug for KeyData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            f.write_str("null")
        } else {
            write!(f, "{}v{}", self.idx, self.version.get())
        }
    }
}

impl TryDebug for KeyData {
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            f.write_str("null")
        } else {
            write!(f, "{}v{}", self.idx, self.version.get())
        }
    }
}

impl Default for KeyData {
    fn default() -> Self {
        Self::null()
    }
}

impl Hash for KeyData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.as_ffi());
    }
}

/// Trait implemented by all slot-map key types.
///
/// Implementations must be thin wrappers around [`KeyData`] and delegate every
/// method identically to operating on `KeyData` directly. Internal unsafe code
/// relies on this contract, which is why the trait is `unsafe`. Prefer using
/// [`new_key_type!`](crate::new_key_type!) instead of
/// implementing manually.
///
/// # Safety
///
/// All methods must behave exactly as if operating on a [`KeyData`] directly.
pub unsafe trait Key:
    lang_core::convert::From<KeyData>
    + Copy
    + Clone
    + Default
    + Eq
    + PartialEq
    + Ord
    + PartialOrd
    + Hash
    + Debug
{
    /// Create a null key — always invalid for any slot map.
    fn null() -> Self {
        KeyData::null().into()
    }

    /// Check whether this key is null.
    fn is_null(&self) -> bool {
        self.data().is_null()
    }

    /// Access the underlying [`KeyData`].
    fn data(&self) -> KeyData;
}

/// The default slot-map key type.
///
/// Equivalent to any key created by [`new_key_type!`](crate::new_key_type!),
/// just with a predefined name.
#[derive(Copy, Clone, Default, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[repr(transparent)]
pub struct DefaultKey(KeyData);

impl From<KeyData> for DefaultKey {
    fn from(kd: KeyData) -> Self {
        DefaultKey(kd)
    }
}

unsafe impl Key for DefaultKey {
    fn data(&self) -> KeyData {
        self.0
    }
}
