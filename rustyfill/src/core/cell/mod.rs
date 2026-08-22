//! Fallible interior-mutability helpers for [`lang_core::cell::RefCell`].
//!
//! Provides `TryDebug` and `TryDisplay` implementations for `BorrowError` and
//! `BorrowMutError`, plus `TryClone` and `TryDefault` implementations for
//! `RefCell<T>` when `T` supports the corresponding traits.

mod cell_;
