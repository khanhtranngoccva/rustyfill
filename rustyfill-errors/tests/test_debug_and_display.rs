use core::error::Error;
use rustyfill_errors::Report;

extern crate alloc;
extern crate std;

// ── Snapshot display tests ────────────────────────────────────────────
// Snapshots are written to the build target directory and overwritten on
// each run. Tests compare the current Report Display output against the saved snapshot file.
// Skipped under Miri which does not support filesystem access in isolation mode.

#[derive(Debug)]
struct TestError(&'static str);
impl core::fmt::Display for TestError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Error for TestError {}
rustyfill::debug_passthrough!(TestError);
rustyfill::display_passthrough!(TestError);

#[derive(Debug)]
struct OtherError(&'static str);
impl core::fmt::Display for OtherError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Error for OtherError {}
rustyfill::debug_passthrough!(OtherError);
rustyfill::display_passthrough!(OtherError);

#[cfg_attr(miri, ignore)]
fn get_snapshot_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("snapshots")
}

#[cfg_attr(miri, ignore)]
fn write_snapshot(name: &str, content: &str) {
    let dir = get_snapshot_dir();
    std::fs::create_dir_all(&dir).expect("failed to create snapshot directory");
    let path = dir.join(alloc::format!("{name}.snap"));
    std::fs::write(&path, content).expect("failed to write snapshot");
}

#[cfg_attr(miri, ignore)]
fn read_snapshot(name: &str) -> Option<alloc::string::String> {
    let path = get_snapshot_dir().join(alloc::format!("{name}.snap"));
    std::fs::read_to_string(path).ok()
}

/// Normalize CRLF to LF so that snapshots pass on Windows where
/// outputting may emit `\r\n` instead of `\n`.
fn normalize_newlines(s: &str) -> alloc::borrow::Cow<'_, str> {
    if s.contains('\r') {
        alloc::borrow::Cow::Owned(s.replace("\r\n", "\n"))
    } else {
        alloc::borrow::Cow::Borrowed(s)
    }
}

#[cfg_attr(miri, ignore)]
fn assert_snapshot(name: &str, actual: &str) {
    let allow_update = std::env::var("UPDATE_SNAPSHOTS").is_ok();
    let actual_norm = normalize_newlines(actual);
    let snapshot_str = read_snapshot(name);
    let expected_opt = snapshot_str.as_deref().map(normalize_newlines);
    match expected_opt {
        Some(ref expected) if **expected == *actual_norm => {} // matches
        Some(expected) => {
            if allow_update {
                write_snapshot(name, &actual_norm);
            }
            panic!(
                "Snapshot mismatch for '{}'.\nExpected:\n---\n{}---\nActual:\n---\n{}---{}",
                name,
                expected,
                actual_norm,
                if allow_update {
                    "\nSnapshot file updated."
                } else {
                    "\nSet UPDATE_SNAPSHOTS=1 to update."
                },
            );
        }
        None => {
            if allow_update {
                write_snapshot(name, &actual_norm);
            }
            panic!(
                "No snapshot found for '{}'.{}\nOutput:\n---\n{}---",
                name,
                if allow_update {
                    "Written initial snapshot."
                } else {
                    "Set UPDATE_SNAPSHOTS=1 to create one."
                },
                actual_norm,
            );
        }
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn display_single_frame() {
    let report = Report::new(TestError("something went wrong"));
    let output = alloc::format!("{}", report);
    assert_snapshot("single_frame", &output);
}

#[cfg_attr(miri, ignore)]
#[test]
fn display_single_frame_with_segment() {
    let report = Report::with_segment(TestError("parse failed"), "parsing config");
    let output = alloc::format!("{}", report);
    assert_snapshot("single_frame_with_segment", &output);
}

#[cfg_attr(miri, ignore)]
#[test]
fn display_with_attachment() {
    let report = Report::new(TestError("root error")).attach("extra context");
    let output = alloc::format!("{}", report);
    assert_snapshot("with_attachment", &output);
}

#[cfg_attr(miri, ignore)]
#[test]
fn display_with_multiple_attachments() {
    let report = Report::new(TestError("root error"))
        .attach("detail one")
        .attach(42i32);
    let output = alloc::format!("{}", report);
    assert_snapshot("with_multiple_attachments", &output);
}

#[cfg_attr(miri, ignore)]
#[test]
fn display_with_peers() {
    let report = Report::new(TestError("first"))
        .push(TestError("second"))
        .push(TestError("third"));
    let output = alloc::format!("{}", report);
    assert_snapshot("with_peers", &output);
}

#[cfg_attr(miri, ignore)]
#[test]
fn display_after_change_context() {
    let report: Report<OtherError> =
        Report::new(TestError("inner error")).change_context(OtherError("outer error"));
    let output = alloc::format!("{}", report);
    assert_snapshot("after_change_context", &output);
}

#[cfg_attr(miri, ignore)]
#[test]
fn display_deeply_nested_change_context() {
    let r1 = Report::new(TestError("level 1"));
    let r2: Report<OtherError> = r1.change_context(OtherError("level 2"));
    let r3: Report<TestError> = r2.change_context(TestError("level 3"));
    let output = alloc::format!("{}", r3);
    assert_snapshot("deeply_nested_change_context", &output);
}

#[cfg_attr(miri, ignore)]
#[test]
fn display_change_context_with_attachments() {
    let inner = Report::new(TestError("inner")).attach("inner-attach");
    let report: Report<OtherError> = inner.change_context(OtherError("outer"));
    let output = alloc::format!("{}", report);
    assert_snapshot("change_context_with_attachments", &output);
}

#[cfg_attr(miri, ignore)]
#[test]
fn display_peers_then_change_context() {
    let report: Report<OtherError> = Report::new(TestError("base"))
        .push(TestError("peer"))
        .change_context(OtherError("top"));
    let output = alloc::format!("{}", report);
    assert_snapshot("peers_then_change_context", &output);
}

#[cfg_attr(miri, ignore)]
#[test]
fn display_with_capacity_eviction() {
    let report = Report::new(TestError("first"))
        .with_capacity(2)
        .push(TestError("second"))
        .push(TestError("third"));
    let output = alloc::format!("{}", report);
    assert_snapshot("with_capacity_eviction", &output);
}

#[cfg_attr(miri, ignore)]
#[test]
fn display_minimal_report() {
    let report = Report::new(TestError("minimal"));
    let output = alloc::format!("{}", report);
    assert!(!output.is_empty());
    assert_snapshot("minimal_report", &output);
}

#[cfg_attr(miri, ignore)]
#[test]
fn display_debug_delegates_to_display() {
    let report = Report::with_segment(TestError("debug test"), "checking debug");
    let display_output = alloc::format!("{}", report);
    let debug_output = alloc::format!("{:?}", report);
    assert_eq!(display_output, debug_output);
}

#[cfg_attr(miri, ignore)]
#[test]
fn display_multilevel_tree_with_segments() {
    let r1 = Report::with_segment(TestError("database connection failed"), "db.connect");
    let r2: Report<OtherError> = r1.change_context(OtherError("query execution failed"));
    let r3: Report<TestError> = Report::with_segment(TestError("transaction aborted"), "tx.commit");
    let _ = r2;
    let output = alloc::format!("{}", r3);
    assert_snapshot("multilevel_tree_with_segments", &output);
}

#[cfg(not(miri))]
#[test]
fn display_mixed_error_types_in_tree() {
    let r1 = Report::new(TestError("io error"));
    let r2: Report<OtherError> = r1.change_context(OtherError("network timeout"));
    let r3: Report<TestError> = r2
        .push(OtherError("retry exhausted"))
        .change_context(TestError("service unavailable"));
    let output = alloc::format!("{}", r3);
    assert_snapshot("mixed_error_types_in_tree", &output);
}
