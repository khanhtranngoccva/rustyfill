//! Benchmarks comparing variants of integer arithmetic operations.
//!
//! For each operation (add, sub, mul, div, rem) this module compares the
//! available variants:
//!
//! - `normal`: plain operators (`+`, `-`, ...). Panics on overflow in debug
//!   builds, wraps silently in release builds.
//! - `checked`: `a.checked_add(b)` and friends. Returns `Option<T>`; no panic.
//! - `wrapping`: `a.wrapping_add(b)` and friends. Wraps modulo 2^n; no panic.
//! - `saturating`: `a.saturating_add(b)` and friends. Clamps to min/max; no
//!   panic. Not available for `div`/`rem` — std defines no saturating
//!   variants for them (saturation semantics for division/remainder are not
//!   well-defined), so those operations only benchmark the other three.
//!
//! The harness is intentionally dependency-free so the experiment stays fast
//! to build. It uses `Instant` with repeated trials and reports median time
//! per iteration for each variant. See `run_all` / the binary entry point.

use std::hint::black_box;
use std::time::{Duration, Instant};

/// Number of iterations executed inside one trial.
const TRIAL_ITERS: u64 = 1_000_000;

/// Number of trials taken per benchmark; the median is reported.
const NUM_TRIALS: usize = 5;

/// Result of benchmarking one operation variant.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Human-readable name of the variant, e.g. `"mul/wrapping"`.
    pub name: String,
    /// Median wall-clock time per single operation across all trials.
    pub median_per_op: Duration,
    /// Minimum observed per-op time across trials.
    pub min_per_op: Duration,
    /// Maximum observed per-op time across trials.
    pub max_per_op: Duration,
}

impl BenchmarkResult {
    fn new(name: impl Into<String>, durations: Vec<Duration>) -> Self {
        let mut sorted = durations;
        sorted.sort();
        // Convert total-trial durations into per-operation durations.
        let per_op: Vec<Duration> = sorted
            .iter()
            .map(|d| Duration::from_nanos(d.as_nanos() as u64 / TRIAL_ITERS))
            .collect();
        let median = per_op[per_op.len() / 2];
        Self {
            name: name.into(),
            median_per_op: median,
            min_per_op: *per_op.first().unwrap(),
            max_per_op: *per_op.last().unwrap(),
        }
    }
}

fn bench<F>(name: &str, f: F) -> BenchmarkResult
where
    F: FnMut(u32, u32),
{
    let mut samples: Vec<Duration> = Vec::with_capacity(NUM_TRIALS);
    let mut f = f;
    for _ in 0..NUM_TRIALS {
        let start = Instant::now();
        for i in 0..TRIAL_ITERS {
            let a = black_box(i as u32 % 97 + 1);
            let b = black_box((i as u32 >> 3) % 89 + 1);
            f(a, b);
        }
        samples.push(start.elapsed());
    }
    BenchmarkResult::new(name, samples)
}

// ---------------------------------------------------------------------------
// Operation variants
// ---------------------------------------------------------------------------

/// Declare one operation module with its available variants.
///
/// The saturating form is optional because std has no `saturating_div` or
/// `saturating_rem`; operations without one simply omit it.
macro_rules! op_variants {
    ($module_name:ident, $normal_fn:path, $checked:ident, $wrapping:ident, $saturating:ident) => {
        #[allow(dead_code)]
        pub mod $module_name {
            use super::*;

            pub fn normal(a: u32, b: u32) -> u32 {
                $normal_fn(a, b)
            }

            pub fn checked(a: u32, b: u32) -> Option<u32> {
                a.$checked(b)
            }

            pub fn wrapping(a: u32, b: u32) -> u32 {
                a.$wrapping(b)
            }

            pub fn saturating(a: u32, b: u32) -> u32 {
                a.$saturating(b)
            }

            pub const NAME: &'static str = stringify!($module_name);

            pub fn run_benchmarks() -> Vec<BenchmarkResult> {
                vec![
                    bench(&format!("{NAME}/normal"), |a, b| {
                        let _ = normal(a, b);
                    }),
                    bench(&format!("{NAME}/checked"), |a, b| {
                        let _ = checked(a, b);
                    }),
                    bench(&format!("{NAME}/wrapping"), |a, b| {
                        let _ = wrapping(a, b);
                    }),
                    bench(&format!("{NAME}/saturating"), |a, b| {
                        let _ = saturating(a, b);
                    }),
                ]
            }
        }
    };
    // No saturating variant (std defines none for div/rem).
    ($module_name:ident, $normal_fn:path, $checked:ident, $wrapping:ident) => {
        #[allow(dead_code)]
        pub mod $module_name {
            use super::*;

            pub fn normal(a: u32, b: u32) -> u32 {
                $normal_fn(a, b)
            }

            pub fn checked(a: u32, b: u32) -> Option<u32> {
                a.$checked(b)
            }

            pub fn wrapping(a: u32, b: u32) -> u32 {
                a.$wrapping(b)
            }

            pub const NAME: &'static str = stringify!($module_name);

            pub fn run_benchmarks() -> Vec<BenchmarkResult> {
                vec![
                    bench(&format!("{NAME}/normal"), |a, b| {
                        let _ = normal(a, b);
                    }),
                    bench(&format!("{NAME}/checked"), |a, b| {
                        let _ = checked(a, b);
                    }),
                    bench(&format!("{NAME}/wrapping"), |a, b| {
                        let _ = wrapping(a, b);
                    }),
                ]
            }
        }
    };
}

/// Plain operators for the "normal" variant. Debug builds trap on overflow;
/// release builds wrap silently.
macro_rules! normal_impl {
    ($name:ident, $expr:tt) => {
        fn $name(a: u32, b: u32) -> u32 {
            let a = black_box(a);
            let b = black_box(b);
            a $expr b
        }
    };
}

normal_impl!(add_normal, +);
normal_impl!(sub_normal, -);
normal_impl!(mul_normal, *);
normal_impl!(div_normal, /);
normal_impl!(rem_normal, %);

op_variants!(add, add_normal, checked_add, wrapping_add, saturating_add);
op_variants!(sub, sub_normal, checked_sub, wrapping_sub, saturating_sub);
op_variants!(mul, mul_normal, checked_mul, wrapping_mul, saturating_mul);
op_variants!(div, div_normal, checked_div, wrapping_div);
op_variants!(rem, rem_normal, checked_rem, wrapping_rem);

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Run every benchmark and return results grouped by operation.
pub fn run_all() -> Vec<(String, Vec<BenchmarkResult>)> {
    vec![
        (add::NAME.to_string(), add::run_benchmarks()),
        (sub::NAME.to_string(), sub::run_benchmarks()),
        (mul::NAME.to_string(), mul::run_benchmarks()),
        (div::NAME.to_string(), div::run_benchmarks()),
        (rem::NAME.to_string(), rem::run_benchmarks()),
    ]
}

/// Pretty-print all results to stdout.
pub fn print_report(results: &[(String, Vec<BenchmarkResult>)]) {
    println!("Arithmetic operation benchmarks");
    println!("({TRIAL_ITERS} iters/trial, {NUM_TRIALS} trials, median per op shown)\n");
    for (op, rows) in results {
        println!("{op}:");
        for r in rows {
            println!(
                "  {:<12} median {:>10.3e} ns   min {:>10.3e} ns   max {:>10.3e} ns",
                r.name.split('/').next_back().unwrap_or(""),
                r.median_per_op.as_nanos() as f64,
                r.min_per_op.as_nanos() as f64,
                r.max_per_op.as_nanos() as f64,
            );
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanity_checked_overflow() {
        assert_eq!(u32::MAX.checked_add(1), None);
        assert_eq!(0u32.checked_sub(1), None);
        assert_eq!(u32::MAX.checked_mul(2), None);
        assert_eq!(1u32.checked_div(0), None);
        assert_eq!(1u32.checked_rem(0), None);
    }

    // Note: `wrapping_div`/`wrapping_rem` and `saturating_div` document their
    // divide-by-zero results (0 / dividend / MAX respectively), but rustc
    // emits an unconditional div-by-zero check for these intrinsics on all
    // targets, so they panic even in release builds. We therefore never call
    // them with a zero divisor here — the benchmark loop feeds non-zero b.
    #[test]
    fn sanity_wrapping() {
        assert_eq!(u32::MAX.wrapping_add(1), 0);
        assert_eq!(0u32.wrapping_sub(1), u32::MAX);
        assert_eq!(u32::MAX.wrapping_mul(2), u32::MAX - 1);
        assert_eq!(div::wrapping(12, 3), 4);
        assert_eq!(rem::wrapping(12, 5), 2);
    }

    #[test]
    fn sanity_saturating() {
        assert_eq!(u32::MAX.saturating_add(1), u32::MAX);
        assert_eq!(0u32.saturating_sub(1), 0);
        assert_eq!(u32::MAX.saturating_mul(2), u32::MAX);
    }

    #[test]
    fn sanity_normal_in_range() {
        assert_eq!(add::normal(2, 3), 5);
        assert_eq!(sub::normal(5, 3), 2);
        assert_eq!(mul::normal(4, 3), 12);
        assert_eq!(div::normal(12, 3), 4);
        assert_eq!(rem::normal(12, 5), 2);
    }
}
