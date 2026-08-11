//! Fallible interior-mutability helpers for [`crate::lang_core::cell::RefCell`].
//!
//! Provides [`TryRefCell`] with methods that mirror `RefCell` borrowing but
//! return [`Result`] on borrow failure instead of panicking. Also provides
//! `TryClone` and `TryDefault` implementations for `RefCell<T>` when `T`
//! supports the corresponding traits.

mod cell_;

pub use cell_::TryRefCell;
