//! Multi-process test runner that verifies whether various `Display`/`Debug`
//! implementations trigger heap allocations during formatting.
//!
//! Each test case runs in its own child process with heap allocations
//! disabled via [`rustyfill_test_allocator`] at critical sections.
//! Formatting is done into a fixed-size stack buffer (via [`StackBuffer`])
//! so that only the `Display`/`Debug` impl itself is exercised.
//!
//! If the formatting code attempts to implicitly allocate during the Display/Debug invocation,
//! the process aborts due to allocation failure and the test is marked as `IMPLICITLY ALLOCATES`.
//! If the process exits cleanly, the test is marked as `SAFE` — no implicit allocations occurred.
//!
//! Because tests may abort the process, they must run in separate child
//! processes rather than in the same process (a-la cargo test).

use std::fmt::Write;
use std::process::{Command, Stdio};

use rustyfill_test_allocator::FailAllocGuard;

// ── Stack-backed formatter ────────────────────────────────────────────────────

/// A `fmt::Write` target backed by a fixed-size stack array.
/// Writes that exceed capacity are silently truncated — this avoids any
/// reallocation and keeps the focus on whether the *formatter* allocates.
struct StackBuffer {
    buf: [u8; 4096],
    len: usize,
}

impl StackBuffer {
    const fn new() -> Self {
        Self {
            buf: [0u8; 4096],
            len: 0,
        }
    }
}

impl Write for StackBuffer {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let end = self.len.saturating_add(s.len());
        let to_copy = end.min(self.buf.len()) - self.len;
        if to_copy > 0 {
            self.buf[self.len..self.len + to_copy].copy_from_slice(&s.as_bytes()[..to_copy]);
        }
        self.len = end.min(self.buf.len());
        Ok(())
    }
}

/// Format `args` into a stack buffer, discarding the result.
/// This exercises the Display/Debug impl without touching stdio.
fn format_on_stack(args: std::fmt::Arguments<'_>) {
    let mut buf = StackBuffer::new();
    let _ = write!(buf, "{args}");
}

// ── Test case definition ──────────────────────────────────────────────────────

pub struct TestCase {
    /// Stable string identifier used to select this test on the command line.
    pub id: &'static str,
    /// Human-readable label shown in the summary table.
    pub name: &'static str,
    /// The code to execute.
    pub run: fn(),
}

// ── Individual test bodies ────────────────────────────────────────────────────

/// Static string formatting — no allocation.
fn test_static_string() {
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("hello from a static string"));
}

/// Integer formatting — no allocation expected.
fn test_integers() {
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{}", 42));
    format_on_stack(format_args!("{:x}", 255));
    format_on_stack(format_args!("{:o}", 64));
    format_on_stack(format_args!("{:b}", 13));
    format_on_stack(format_args!("{val:>width$}", val = 7, width = 10));
}

/// Char formatting — no allocation expected.
fn test_chars() {
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{}", 'A'));
    format_on_stack(format_args!("{:?}", '\u{1F600}'));
}

/// Bool formatting — no allocation expected.
fn test_bool() {
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{}", true));
    format_on_stack(format_args!("{:?}", false));
}

/// Reference formatting — no allocation expected.
fn test_references() {
    let _guard = FailAllocGuard::fail_all();
    let x = 42;
    let s = "hello";
    format_on_stack(format_args!("{:p}", &x));
    format_on_stack(format_args!("{:?}", s));
}

/// Tuple formatting — no allocation expected.
fn test_tuples() {
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:?}", (1, 2, 3)));
    format_on_stack(format_args!("{:?}", ('a', 42, true)));
}

/// Array formatting — no allocation expected.
fn test_arrays() {
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:?}", [1, 2, 3]));
    format_on_stack(format_args!("{:?}", ['a', 'b', 'c']));
}

/// Option formatting — no allocation expected.
fn test_option() {
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:?}", Some(42)));
    format_on_stack(format_args!("{:?}", None::<i32>));
}

/// Result formatting — no allocation expected.
fn test_result() {
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:?}", Ok::<i32, &str>(42)));
    format_on_stack(format_args!("{:?}", Err::<i32, &str>("bad")));
}

/// Duration Debug — no allocation.
fn test_duration() {
    let d = std::time::Duration::from_secs(3600) + std::time::Duration::from_nanos(123_456_789);
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:?}", d));
}

/// SystemTime Debug — no allocation.
fn test_system_time() {
    let t = std::time::SystemTime::now();
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:?}", t));
}

/// Instant Debug — no allocation.
fn test_instant() {
    let i = std::time::Instant::now();
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:?}", i));
}

/// PathBuf Display and Debug — no allocation.
fn test_pathbuf() {
    let p = std::path::PathBuf::from("/tmp/test");
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{}", p.display()));
    format_on_stack(format_args!("{:?}", p));
}

/// OsString Debug — no allocation.
fn test_osstring() {
    let o = std::ffi::OsString::from("test-value");
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:?}", o));
}

/// CString Debug — no allocation.
fn test_cstring() {
    let c = std::ffi::CString::new("hello").unwrap();
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:?}", c));
}

/// IP address Display and Debug — no allocation.
fn test_ip_addr() {
    let ipv4 = std::net::Ipv4Addr::LOCALHOST;
    let ipv6 = std::net::Ipv6Addr::LOCALHOST;
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{}", ipv4));
    format_on_stack(format_args!("{}", ipv6));
    format_on_stack(format_args!("{:?}", ipv4));
}

/// SocketAddr Display and Debug — no allocation.
fn test_socket_addr() {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 8080));
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{}", addr));
    format_on_stack(format_args!("{:?}", addr));
}

/// f64 default Display — no allocation.
fn test_f64_default() {
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{}", std::f64::consts::PI));
    format_on_stack(format_args!("{:?}", -0.0_f64));
    format_on_stack(format_args!("{}", f64::INFINITY));
    format_on_stack(format_args!("{}", f64::NAN));
}

/// f64 with precision specifier — no allocation.
fn test_f64_precision() {
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:.10}", std::f64::consts::PI));
    format_on_stack(format_args!("{:.5e}", 123456789.0));
    format_on_stack(format_args!("{:.5E}", 123456789.0));
}

/// f32 with precision specifier — no allocation.
fn test_f32_precision() {
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:.10}", std::f32::consts::PI));
    format_on_stack(format_args!("{:.5e}", 123456.0_f32));
}

/// Large integer formatting — no allocation.
fn test_large_ints() {
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{}", u128::MAX));
    format_on_stack(format_args!("{:x}", u128::MAX));
    format_on_stack(format_args!("{}", i128::MIN));
}

/// Alignment and padding — no allocation.
fn test_alignment() {
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:<10}", "left"));
    format_on_stack(format_args!("{:^10}", "center"));
    format_on_stack(format_args!("{:>10}", "right"));
    format_on_stack(format_args!("{:010}", 42));
}

/// Nested compound types — vec constructed before guard, only Debug is tested.
fn test_nested_types() {
    let v = vec![Some((1, 2)), None, Some((3, 4))];
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:?}", v));
}

/// eprint! — tests actual stderr output path, no allocation.
fn test_eprintln() {
    let _guard = FailAllocGuard::fail_all();
    eprint!("error message");
}

/// to_string() — control test, DOES allocate (builds a String).
fn test_format_macro() {
    let _guard = FailAllocGuard::fail_all();
    let _s = "this will definitely allocate".to_string();
}

/// Debug for an empty struct — no allocation.
fn test_debug_empty_struct() {
    let _guard = FailAllocGuard::fail_all();
    #[derive(Debug)]
    struct Empty;
    format_on_stack(format_args!("{:?}", Empty));
}

/// Debug for a struct with primitive fields — no allocation.
fn test_debug_struct_with_primitives() {
    let point = Point {
        x: 1,
        y: 2,
        label: "origin",
    };
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:?}", point));
}

#[allow(dead_code)]
#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
    label: &'static str,
}

/// Network-related types — no allocation.
fn test_network_types() {
    let addr_v6 = std::net::SocketAddrV6::new(std::net::Ipv6Addr::LOCALHOST, 443, 0, 0);
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:?}", addr_v6));
}

/// Thread ID — no allocation.
fn test_thread_id() {
    let id = std::thread::current().id();
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:?}", id));
}

/// Panic location — no allocation.
fn test_panic_location() {
    let loc = std::panic::Location::caller();
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:?}", loc));
}

/// OnceLock interior — no allocation.
fn test_once_lock() {
    let lock = std::sync::OnceLock::<i32>::new();
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:?}", lock));
    lock.set(42).ok();
    format_on_stack(format_args!("{:?}", lock.get()));
}

/// Cell / RefCell with primitives — no allocation.
fn test_cell_refcell() {
    let cell = std::cell::Cell::new(99);
    let rc = std::cell::RefCell::new([1, 2, 3]);
    let _guard = FailAllocGuard::fail_all();
    format_on_stack(format_args!("{:?}", cell.get()));
    format_on_stack(format_args!("{:?}", rc.borrow()));
}

/// Vec::new().push() — control test, Vec growth allocates.
fn test_vec_push() {
    let mut v: Vec<i32> = Vec::new();
    let _guard = FailAllocGuard::fail_all();
    v.push(1);
}

// ── Registry ──────────────────────────────────────────────────────────────────

pub fn all_tests() -> &'static [TestCase] {
    &[
        TestCase {
            id: "static-string",
            name: "static string",
            run: test_static_string,
        },
        TestCase {
            id: "integers",
            name: "integers",
            run: test_integers,
        },
        TestCase {
            id: "chars",
            name: "chars",
            run: test_chars,
        },
        TestCase {
            id: "bool",
            name: "bool",
            run: test_bool,
        },
        TestCase {
            id: "references",
            name: "references",
            run: test_references,
        },
        TestCase {
            id: "tuples",
            name: "tuples",
            run: test_tuples,
        },
        TestCase {
            id: "arrays",
            name: "arrays",
            run: test_arrays,
        },
        TestCase {
            id: "option",
            name: "Option",
            run: test_option,
        },
        TestCase {
            id: "result",
            name: "Result",
            run: test_result,
        },
        TestCase {
            id: "duration",
            name: "Duration",
            run: test_duration,
        },
        TestCase {
            id: "system-time",
            name: "SystemTime",
            run: test_system_time,
        },
        TestCase {
            id: "instant",
            name: "Instant",
            run: test_instant,
        },
        TestCase {
            id: "pathbuf",
            name: "PathBuf",
            run: test_pathbuf,
        },
        TestCase {
            id: "osstring",
            name: "OsString",
            run: test_osstring,
        },
        TestCase {
            id: "cstring",
            name: "CString",
            run: test_cstring,
        },
        TestCase {
            id: "ip-addr",
            name: "IpAddr",
            run: test_ip_addr,
        },
        TestCase {
            id: "socket-addr",
            name: "SocketAddr",
            run: test_socket_addr,
        },
        TestCase {
            id: "f64-default",
            name: "f64 default",
            run: test_f64_default,
        },
        TestCase {
            id: "f64-precision",
            name: "f64 precision",
            run: test_f64_precision,
        },
        TestCase {
            id: "f32-precision",
            name: "f32 precision",
            run: test_f32_precision,
        },
        TestCase {
            id: "large-ints",
            name: "large ints",
            run: test_large_ints,
        },
        TestCase {
            id: "alignment-padding",
            name: "alignment/padding",
            run: test_alignment,
        },
        TestCase {
            id: "nested-compound-types",
            name: "nested compound types",
            run: test_nested_types,
        },
        TestCase {
            id: "eprint-stderr",
            name: "eprint! (stderr)",
            run: test_eprintln,
        },
        TestCase {
            id: "to-string-control",
            name: "to_string() (control — expects ALLOCATES)",
            run: test_format_macro,
        },
        TestCase {
            id: "debug-empty-struct",
            name: "debug(empty struct)",
            run: test_debug_empty_struct,
        },
        TestCase {
            id: "debug-struct-primitives",
            name: "debug(struct with primitives)",
            run: test_debug_struct_with_primitives,
        },
        TestCase {
            id: "network-types",
            name: "network types",
            run: test_network_types,
        },
        TestCase {
            id: "thread-id",
            name: "ThreadId",
            run: test_thread_id,
        },
        TestCase {
            id: "panic-location",
            name: "PanicInfo::Location",
            run: test_panic_location,
        },
        TestCase {
            id: "once-lock",
            name: "OnceLock",
            run: test_once_lock,
        },
        TestCase {
            id: "cell-refcell",
            name: "Cell/RefCell",
            run: test_cell_refcell,
        },
        TestCase {
            id: "vec-push-control",
            name: "Vec::push (control — expects ALLOCATES)",
            run: test_vec_push,
        },
    ]
}

// ── Child mode ────────────────────────────────────────────────────────────────

/// When invoked with a string argument, run only that test (by ID) and exit.
pub fn run_single_test(id: &str) {
    let tests = all_tests();
    let test = tests.iter().find(|t| t.id == id).unwrap_or_else(|| {
        eprintln!("unknown test id \"{id}\"");
        eprintln!(
            "available ids: {}",
            tests.iter().map(|t| t.id).collect::<Vec<_>>().join(", ")
        );
        std::process::exit(2);
    });
    // initialization of structs may allocate, we only want to test display,
    // so we have to let test cases call with_policy() manually
    (test.run)();
}

// ── Runner mode ───────────────────────────────────────────────────────────────

#[derive(Clone, PartialEq, Eq)]
pub enum TestResult {
    /// Process exited cleanly — no allocations detected.
    Safe,
    /// Process aborted — an allocation was attempted.
    ImplicitlyAllocates,
    /// Something unexpected happened (e.g., binary not found).
    Error(String),
}

pub fn run_in_child(test_id: &str) -> TestResult {
    let me = std::env::current_exe().unwrap_or_else(|_| {
        eprintln!("cannot determine executable path");
        std::process::exit(2);
    });

    let output = Command::new(&me)
        .arg(test_id)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output();

    match output {
        Ok(o) if o.status.success() => TestResult::Safe,
        Ok(_) => TestResult::ImplicitlyAllocates,
        Err(e) => TestResult::Error(e.to_string()),
    }
}
