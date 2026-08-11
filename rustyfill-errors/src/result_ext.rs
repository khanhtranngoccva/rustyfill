//! Convenience extension for [`Result<T, Report<E>`] that chains report-building
//! operations without requiring explicit `map_err`.
//!
//! Every method operates only when the result is an `Err`, leaving `Ok` values
//! untouched. All variants use the *lossy* set of [`Report`] APIs — attachments
//! may be silently dropped under memory pressure rather than propagating a secondary error.

use core::error::Error;

use rustyfill::try_fmt::{TryDebug, TryDisplay};

use crate::Report;

/// Extension trait for [`Result`] values whose error type is a [`Report`].
///
/// Provides chaining helpers so you can attach context, add attachments, or
/// change the report's context type in a single fluent call:
///
/// ```ignore
/// let result: Result<MyData, Report<ParseError>> = parse_input(data);
///
/// result
///     .attach("user_id")
///     .change_context_lazy(|_| RuntimeError("failed to process request"))?;
/// ```
pub trait ResultExt<T, E> {
    /// Attaches printable data to the head frame if this result is an error.
    ///
    /// The value must implement [`TryDebug`](rustyfill::try_fmt::TryDebug) and
    /// [`TryDisplay`](rustyfill::try_fmt::TryDisplay). Silently drops the
    /// attachment on OOM.
    fn attach<A>(self, attachment: A) -> Self
    where
        A: rustyfill::try_fmt::TryDebug + rustyfill::try_fmt::TryDisplay + Send + Sync + 'static;

    /// Lazily evaluates and attaches printable data to the head frame if this
    /// result is an error. The closure is only called when the result is `Err`.
    fn attach_lazy<A, F>(self, f: F) -> Self
    where
        A: rustyfill::try_fmt::TryDebug + rustyfill::try_fmt::TryDisplay + Send + Sync + 'static,
        F: FnOnce() -> A;

    /// Attaches opaque data to the head frame if this result is an error.
    ///
    /// Unlike [`attach`](Self::attach), the value need not implement [`Debug`]
    /// or [`Display`]. Silently drops the attachment on OOM.
    fn attach_opaque<A>(self, attachment: A) -> Self
    where
        A: Send + Sync + 'static;

    /// Lazily evaluates and attaches opaque data to the head frame if this
    /// result is an error. The closure is only called when the result is `Err`.
    fn attach_opaque_lazy<A, F>(self, f: F) -> Self
    where
        A: Send + Sync + 'static,
        F: FnOnce() -> A;

    /// Changes the current context to a new type `U` if this result is an error.
    ///
    /// All existing peers (including the head) are demoted into children. Oldest
    /// peers are dropped first if allocation fails.
    #[track_caller]
    #[allow(clippy::result_large_err)]
    fn change_context<U>(self, context: U) -> Result<T, Report<U>>
    where
        U: Error + TryDebug + TryDisplay + Send + Sync + 'static;

    /// Lazily evaluates and changes the current context if this result is an
    /// error. The closure receives no arguments and returns the new context.
    #[track_caller]
    #[allow(clippy::result_large_err)]
    fn change_context_lazy<U, F>(self, f: F) -> Result<T, Report<U>>
    where
        U: Error + TryDebug + TryDisplay + Send + Sync + 'static,
        F: FnOnce() -> U;

    /// Changes the current context using the map_err like syntax if this result is an error.
    ///
    /// This function acts similarly to map_err() but retains control of the report building -
    /// this allows surfacing domain-level concerns from the old topmost error(s) in the new error type.
    #[track_caller]
    #[allow(clippy::result_large_err)]
    fn change_context_adaptive<U, F>(self, f: F) -> Result<T, Report<U>>
    where
        U: Error + TryDebug + TryDisplay + Send + Sync + 'static,
        F: FnOnce(&mut Report<E>) -> U;
}

impl<T, E> ResultExt<T, E> for Result<T, Report<E>>
where
    E: Error + TryDebug + TryDisplay + Send + Sync + 'static,
{
    fn attach<A>(self, attachment: A) -> Self
    where
        A: rustyfill::try_fmt::TryDebug + rustyfill::try_fmt::TryDisplay + Send + Sync + 'static,
    {
        match self {
            Ok(val) => Ok(val),
            Err(report) => Err(report.attach(attachment)),
        }
    }

    fn attach_lazy<A, F>(self, f: F) -> Self
    where
        A: rustyfill::try_fmt::TryDebug + rustyfill::try_fmt::TryDisplay + Send + Sync + 'static,
        F: FnOnce() -> A,
    {
        match self {
            Ok(val) => Ok(val),
            Err(report) => Err(report.attach(f())),
        }
    }

    fn attach_opaque<A>(self, attachment: A) -> Self
    where
        A: Send + Sync + 'static,
    {
        match self {
            Ok(val) => Ok(val),
            Err(report) => Err(report.attach_opaque(attachment)),
        }
    }

    fn attach_opaque_lazy<A, F>(self, f: F) -> Self
    where
        A: Send + Sync + 'static,
        F: FnOnce() -> A,
    {
        match self {
            Ok(val) => Ok(val),
            Err(report) => Err(report.attach_opaque(f())),
        }
    }

    #[track_caller]
    fn change_context<U>(self, context: U) -> Result<T, Report<U>>
    where
        U: Error + TryDebug + TryDisplay + Send + Sync + 'static,
    {
        match self {
            Ok(val) => Ok(val),
            Err(report) => Err(report.change_context(context)),
        }
    }

    #[track_caller]
    fn change_context_lazy<U, F>(self, f: F) -> Result<T, Report<U>>
    where
        U: Error + TryDebug + TryDisplay + Send + Sync + 'static,
        F: FnOnce() -> U,
    {
        match self {
            Ok(val) => Ok(val),
            Err(report) => Err(report.change_context(f())),
        }
    }

    #[track_caller]
    fn change_context_adaptive<U, F>(self, f: F) -> Result<T, Report<U>>
    where
        U: Error + TryDebug + TryDisplay + Send + Sync + 'static,
        F: FnOnce(&mut Report<E>) -> U,
    {
        match self {
            Ok(val) => Ok(val),
            Err(mut original) => {
                let new_err = f(&mut original);
                let report = original.change_context(new_err);
                Err(report)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::borrow::Cow;
    use rustyfill::string::TryStr;

    use super::*;
    use crate::Report;

    #[derive(Debug)]
    struct IoError(&'static str);
    impl core::fmt::Display for IoError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "IO: {}", self.0)
        }
    }
    impl Error for IoError {}
    rustyfill::debug_passthrough!(IoError);
    rustyfill::display_passthrough!(IoError);

    #[derive(Debug)]
    struct AppError(Cow<'static, str>);
    impl core::fmt::Display for AppError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "App: {}", self.0)
        }
    }
    impl Error for AppError {}
    rustyfill::debug_passthrough!(AppError);
    rustyfill::display_passthrough!(AppError);

    #[test]
    fn attach_on_ok_is_noop() {
        let result: Result<i32, Report<IoError>> = Ok(42);
        let result = result.attach("some context");
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn attach_on_err_adds_attachment() {
        let result: Result<i32, Report<IoError>> = Err(Report::new(IoError("disk full")));
        let result = result.attach("request_id: abc123");
        let report = result.unwrap_err();
        assert!(report.contains::<&'static str>());
    }

    #[test]
    fn attach_lazy_only_evaluates_on_err() {
        let result: Result<i32, Report<IoError>> = Ok(42);
        let evaluated = std::cell::Cell::new(false);
        let result = result.attach_lazy(|| {
            evaluated.set(true);
            "should not evaluate"
        });
        assert!(!evaluated.get());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn attach_lazy_evaluates_on_err() {
        let result: Result<i32, Report<IoError>> = Err(Report::new(IoError("timeout")));
        let evaluated = std::cell::Cell::new(false);
        let result = result.attach_lazy(|| {
            evaluated.set(true);
            "lazy context"
        });
        assert!(evaluated.get());
        let _report = result.unwrap_err();
        // If we got here without panicking, the attachment was accepted.
    }

    #[test]
    fn attach_opaque_on_err() {
        let result: Result<i32, Report<IoError>> = Err(Report::new(IoError("connection refused")));
        let result = result.attach_opaque([1u8, 2, 3]);
        let report = result.unwrap_err();
        assert!(report.contains::<[u8; 3]>());
    }

    #[test]
    fn attach_opaque_lazy_on_err() {
        let result: Result<i32, Report<IoError>> = Err(Report::new(IoError("rate limited")));
        let result = result.attach_opaque_lazy(|| alloc::string::String::from("trace data"));
        let _report = result.unwrap_err();
        // If we got here without panicking, the attachment was accepted.
    }

    #[test]
    fn change_context_on_ok_preserves_value() {
        let result: Result<alloc::string::String, Report<IoError>> =
            Ok(alloc::string::String::from("hello"));
        let result: Result<alloc::string::String, Report<AppError>> =
            result.change_context(AppError(Cow::Borrowed("wrapper")));
        assert_eq!(result.unwrap(), "hello");
    }

    #[test]
    fn change_context_on_err_demotes_original() {
        let result: Result<i32, Report<IoError>> =
            Err(Report::with_segment(IoError("not found"), "file.read"));
        let result: Result<i32, Report<AppError>> =
            result.change_context(AppError(Cow::Borrowed("resource unavailable")));
        let report = result.unwrap_err();
        assert_eq!(report.current_context().0, "resource unavailable");
        // Check that frames walker finds a child (demoted original).
        let mut count = 0;
        for (_, _) in report.frames() {
            count += 1;
        }
        assert!(
            count >= 2,
            "expected at least head + 1 child, got {}",
            count
        );
    }

    #[test]
    fn change_context_lazy_on_err() {
        let result: Result<i32, Report<IoError>> = Err(Report::new(IoError("parse failed")));
        let result: Result<i32, Report<AppError>> =
            result.change_context_lazy(|| AppError(Cow::Borrowed("input invalid")));
        let report = result.unwrap_err();
        assert_eq!(report.current_context().0, "input invalid");
    }

    #[test]
    fn change_context_adaptive_on_ok_discards_new_report() {
        let result: Result<i32, Report<IoError>> = Ok(99);
        let result: Result<i32, Report<AppError>> =
            result.change_context_adaptive(|_| AppError(Cow::Borrowed("would be used")));
        assert_eq!(result.unwrap(), 99);
    }

    #[test]
    fn change_context_adaptive_on_err_calls_adapter() {
        let result: Result<i32, Report<IoError>> =
            Err(Report::with_segment(IoError("original"), "segment.label"));
        let mut was_called = false;
        let result: Result<i32, Report<AppError>> = result.change_context_adaptive(|r| {
            // Adapter can inspect, modify the old report, and return the new one
            let duplicated = r
                .current_context_mut()
                .0
                .try_repeat(2)
                .expect("no OOM in this test");
            r.current_context_mut().0 = "replacement";
            was_called = true;
            AppError(Cow::Owned(duplicated))
        });

        assert!(was_called);
        let report = result.unwrap_err();
        assert_eq!(report.current_context().0, "originaloriginal");
        // Original should be demoted to children — verify via frames walker.
        let mut count = 0;
        for (_, _) in report.frames() {
            count += 1;
        }
        assert!(
            count >= 2,
            "expected at least head + 1 child, got {}",
            count
        );
    }

    #[test]
    fn chained_operations() {
        let result: Result<i32, Report<IoError>> = Err(Report::new(IoError("initial error")));

        let result = result.attach("step one").attach_opaque(42i64);

        let report = result.unwrap_err();
        assert!(report.contains::<&'static str>());
        assert!(report.contains::<i64>());
    }

    #[test]
    fn chain_attach_then_change_context() {
        let result: Result<i32, Report<IoError>> = Err(Report::new(IoError("db down")));

        let result: Result<i32, Report<AppError>> = result
            .attach("query: SELECT * FROM users")
            .change_context(AppError(Cow::Borrowed("database unreachable")));

        let report = result.unwrap_err();
        assert_eq!(report.current_context().0, "database unreachable");
        // Frames walker should find head + at least one child.
        let mut count = 0;
        for (_, _) in report.frames() {
            count += 1;
        }
        assert!(
            count >= 2,
            "expected at least head + 1 child, got {}",
            count
        );
    }
}
