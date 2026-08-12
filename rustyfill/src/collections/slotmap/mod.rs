//! Slot map — a container with persistent unique keys.
//!
//! A slot map stores values indexed by stable, versioned keys that are
//! generated on insertion. Unlike `Vec` indices or `HashMap` keys, a slot-map
//! key remains valid for the lifetime of the inserted value and never points
//! at a different element after deletion-and-reinsertion (thanks to per-slot
//! version counters).
//!
//! Every operation that may allocate returns a [`Result`] instead of panicking.
//!
//! # Types
//!
//! - [`SlotMap`] — the primary slot map
//! - [`SecondaryMap`] — associates extra data with keys from a [`SlotMap`]
//! - [`KeyData`] / [`Key`] / [`DefaultKey`] — key representation and trait
//! - [`new_key_type!`] — macro for creating distinct key types
//!
//! [`slotmap`]: https://crates.io/crates/slotmap

mod basic;
mod key;
mod secondary;

pub use basic::{Drain, IntoIter, Iter, IterMut, Keys, SlotMap, SlotMapError, Values, ValuesMut};
pub use key::{DefaultKey, Key, KeyData};
pub use secondary::{
    Drain as SecondaryDrain, Entry, IntoIter as SecondaryIntoIter, Iter as SecondaryIter,
    IterMut as SecondaryIterMut, Keys as SecondaryKeys, OccupiedEntry, SecondaryMap,
    SecondaryMapError, VacantEntry, Values as SecondaryValues, ValuesMut as SecondaryValuesMut,
};

/// Creates one or more new key types, preventing accidental cross-slot-map usage.
///
/// Each declared type is identical in layout and behavior to [`DefaultKey`],
/// but is a distinct type so the compiler rejects mixing keys between
/// different slot maps.
///
/// # Examples
///
/// ```
/// use rustyfill::new_key_type;
/// use rustyfill::collections::slotmap::SlotMap;
///
/// new_key_type! {
///     struct PlayerKey;
///     (pub struct EntityKey;)
/// }
///
/// let mut players: SlotMap<PlayerKey, String> = SlotMap::with_key();
/// let mut entities: SlotMap<EntityKey, u32> = SlotMap::with_key();
///
/// // Compiles fine — correct key type.
/// let p = players.try_insert("Alice".to_string()).unwrap();
/// // Type error — wrong key type for this slot map.
/// // let _ = players.get(entities.try_insert(42).unwrap());
/// ```
#[macro_export]
macro_rules! new_key_type {
    ($(#[$outer:meta])* $vis:vis struct $name:ident; $($rest:tt)*) => {
        $(#[$outer])*
        #[derive(Copy, Clone, Default, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
        #[repr(transparent)]
        $vis struct $name($crate::collections::slotmap::KeyData);

        impl $crate::lang_core::convert::From<$crate::collections::slotmap::KeyData> for $name {
            fn from(kd: $crate::collections::slotmap::KeyData) -> Self {
                $name(kd)
            }
        }

        unsafe impl $crate::collections::slotmap::Key for $name {
            fn data(&self) -> $crate::collections::slotmap::KeyData {
                self.0
            }
        }

        $crate::new_key_type!(@cont $($rest)*);
    };

    (@cont ($(#[$attrs:meta])* $vis:vis struct $name:ident;) $($rest:tt)*) => {
        $(#[$attrs])*
        #[derive(Copy, Clone, Default, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
        #[repr(transparent)]
        $vis struct $name($crate::collections::slotmap::KeyData);

        impl $crate::lang_core::convert::From<$crate::collections::slotmap::KeyData> for $name {
            fn from(kd: $crate::collections::slotmap::KeyData) -> Self {
                $name(kd)
            }
        }

        unsafe impl $crate::collections::slotmap::Key for $name {
            fn data(&self) -> $crate::collections::slotmap::KeyData {
                self.0
            }
        }

        $crate::new_key_type!(@cont $($rest)*);
    };

    (@cont) => {};
}
