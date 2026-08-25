//! Fallible interior-mutability helpers for the `core::cell` family.
//!
//! Provides:
//! - `TryDebug` / `TryDisplay` for `BorrowError` and `BorrowMutError`
//! - `TryClone`, `TryDefault`, `TryDebug` for `RefCell<T>`
//! - `TryClone`, `TryDefault`, `TryDebug` for `Cell<T>` (Copy types)
//! - `TryDebug`, `TryDefault` for `LazyCell<T>`
//! - `TryClone`, `TryDefault`, `TryDebug` for `OnceCell<T>`
//! - `TryDebug`, `TryDisplay` for `Ref<'_, T>` and `RefMut<'_, T>`
//! - `TryDebug`, `TryDefault` for `UnsafeCell<T>`

mod ref_cell_;
mod cell_;
mod lazy_cell_;
mod once_cell_;
mod ref_;
mod unsafe_cell_;
