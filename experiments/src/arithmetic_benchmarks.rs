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
//! to build. Each operation runs against a fixed pair of constants (see
//! `*_OPERANDS`) chosen so no variant can overflow or divide by zero — this
//! isolates the intrinsic cost of each variant without input-dependent
//! effects and prevent panics. Work is split into many small interleaved batches,
//! each timed individually; because a single op is too fast to measure on its own,
//! all reported figures are wall-clock durations of a whole batch (mean, median, min, max).
//! See `run_all` / the binary entry point.

use std::collections::{BTreeMap, HashSet};
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Iterations executed in a single batch. Each batch is timed individually and
/// becomes one measurement sample; keeping batch count small also bounds memory
/// strain since only a running accumulator and a timestamp are held at a time.
const BATCH_ITERS: u64 = 100_000;

/// Number of batches run per benchmark. Every batch yields one sample, so this
/// is also the number of samples feeding the statistics.
const NUM_BATCHES: u64 = 100;

/// Length of one scrambled operation sequence used by the mixed-workflow
/// scenario. Must be a multiple of the number of operations (5) so that every
/// op can appear an equal whole number of times per sequence — e.g. 10 means
/// each of the five ops occurs exactly twice per sequence. Larger values give
/// longer, more varied instruction streams before the pattern could repeat.
const WORKFLOW_PERM_SIZE: usize = 10;

/// How many *distinct* scrambled orderings to pre-generate for the mixed
/// workflow. Each batch cycles through all of them, so a batch's wall-clock
/// time reflects an average over this many different instruction orders rather
/// than one fixed sequence. Kept modest on purpose: it only needs to be large
/// enough to sample a representative spread of orderings, not to enumerate the
/// (astronomically large) full permutation space. Must be <= the total number
/// of distinct permutations ([`workflow_permutation_count`]).
const WORKFLOW_NUM_PERMS: u64 = 64;

/// Number of distinct operations in the mixed workflow.
const NUM_OPS: usize = 5;

/// Frequency counter over batch durations, backed by a `BTreeMap`.
///
/// The map stores each distinct duration as a key with its occurrence count as
/// the value, which is what makes a `BTreeMap` suitable here despite it
/// collapsing duplicate keys — the counts preserve every sample. Keys iterate
/// in ascending order, so min/max/median/mean can all be derived without a
/// separate sort pass.
#[derive(Debug, Clone, Default)]
struct BatchCounter {
    /// Distinct batch durations mapped to how many batches produced them.
    counts: BTreeMap<Duration, u64>,
}

impl BatchCounter {
    fn new() -> Self {
        Self::default()
    }

    /// Record one batch's duration, incrementing its count.
    fn add(&mut self, d: Duration) {
        *self.counts.entry(d).or_insert(0) += 1;
    }

    /// Total number of recorded batches (sum of all counts).
    fn total(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Smallest recorded duration.
    fn min(&self) -> Option<Duration> {
        self.counts.keys().next().copied()
    }

    /// Largest recorded duration.
    fn max(&self) -> Option<Duration> {
        self.counts.keys().last().copied()
    }

    /// Mean recorded duration (total time / number of batches).
    fn mean(&self) -> Option<Duration> {
        let total = self.total();
        if total == 0 {
            return None;
        }
        let sum_ns: u128 = self
            .counts
            .iter()
            .map(|(d, c)| d.as_nanos() * (*c as u128))
            .sum();
        Some(Duration::from_nanos((sum_ns / total as u128) as u64))
    }

    /// Duration occupying the given 0-indexed rank in the count-expanded
    /// ordering (i.e. as if every sample were laid out individually).
    fn at_rank(&self, rank: u64) -> Option<Duration> {
        let mut seen = 0u64;
        for (d, c) in &self.counts {
            if seen <= rank && rank < seen + c {
                return Some(*d);
            }
            seen += c;
        }
        None
    }

    /// Median recorded duration: the middle sample for an odd total, or the
    /// mean of the two central samples for an even total.
    fn median(&self) -> Option<Duration> {
        let total = self.total();
        if total == 0 {
            return None;
        }
        if total.is_multiple_of(2) {
            let lo = self.at_rank(total / 2 - 1)?.as_nanos();
            let hi = self.at_rank(total / 2)?.as_nanos();
            Some(Duration::from_nanos(((lo + hi) / 2) as u64))
        } else {
            self.at_rank(total / 2)
        }
    }
}

/// Statistics computed over the per-batch samples of one operation variant.
///
/// All figures are wall-clock durations of a whole batch (i.e. `BATCH_ITERS`
/// operations), not normalized per operation — a single op is far too fast to
/// measure reliably on its own, so the batch as a unit is the meaningful sample.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Human-readable name of the variant, e.g. `"mul/wrapping"`.
    pub name: String,
    /// Number of batches sampled.
    pub num_batches: u64,
    /// Median wall-clock time for one full batch.
    pub median_batch: Duration,
    /// Mean wall-clock time for one full batch.
    pub mean_batch: Duration,
    /// Fastest observed batch.
    pub min_batch: Duration,
    /// Slowest observed batch.
    pub max_batch: Duration,
}

impl BenchmarkResult {
    /// Build a result from a counter of raw per-batch durations.
    ///
    /// The statistics are read straight off the counter's ordered frequency
    /// table; each value is the duration of an entire batch.
    fn new(name: impl Into<String>, counter: &BatchCounter) -> Self {
        Self {
            name: name.into(),
            num_batches: counter.total(),
            median_batch: counter.median().expect("at least one batch"),
            mean_batch: counter.mean().expect("at least one batch"),
            min_batch: counter.min().expect("at least one batch"),
            max_batch: counter.max().expect("at least one batch"),
        }
    }
}

/// In-range operand pairs, one per operation. Using constants (rather than
/// varying inputs) keeps the benchmark simple and removes any chance of hitting
/// an undefined or trapping input — notably a zero divisor for `div`/`rem`. The
/// values are chosen so that even the plain operators stay in range in debug
/// builds (no overflow traps, no divide-by-zero).
pub(crate) const ADD_OPERANDS: (u32, u32) = (0x1234_5678, 0x0ABC_DEF01);
pub(crate) const SUB_OPERANDS: (u32, u32) = (0x7654_3210, 0x0FED_CBA9);
pub(crate) const MUL_OPERANDS: (u32, u32) = (0x0001_2345, 0x0000_0AB9);
pub(crate) const DIV_OPERANDS: (u32, u32) = (0x00FF_FFFF, 0x0000_0007);
pub(crate) const REM_OPERANDS: (u32, u32) = (0x00FF_FFFE, 0x0000_0005);

/// Overflow operand pairs for the operations that have both `wrapping` and
/// `saturating` variants (add/sub/mul). These are used exclusively to compare
/// how `wrapping` and `saturating` behave when the result actually leaves the
/// representable range. They must NOT be fed to the `normal` or `checked`
/// variants in a way that would trap — see the overflow-scenario benchmarks,
/// which only exercise `wrapping` and `saturating`.
pub(crate) const ADD_OVERFLOW: (u32, u32) = (0xFFFF_FFFF, 0x0000_0001);
pub(crate) const SUB_OVERFLOW: (u32, u32) = (0x0000_0000, 0x0000_0001);
// 0x7FFF_FFFF * 4 = 0x1_FFFF_FFFC > u32::MAX, so this genuinely overflows.
pub(crate) const MUL_OVERFLOW: (u32, u32) = (0x7FFF_FFFF, 0x0000_0004);

// Compile-time proof that the in-range operands keep the plain operators in
// range, so the "normal" variant can never trap in debug builds. If any of
// these start overflowing, the build fails here instead of panicking
// mid-benchmark.
const _: () = {
    let (a, b) = ADD_OPERANDS;
    assert!(a.checked_add(b).is_some());
    let (a, b) = SUB_OPERANDS;
    assert!(a >= b); // checked_sub requires a >= b
    let (a, b) = MUL_OPERANDS;
    assert!(a.checked_mul(b).is_some());
    let (_, b) = DIV_OPERANDS;
    assert!(b != 0);
    let (_, b) = REM_OPERANDS;
    assert!(b != 0);
};

// Compile-time proof that the overflow operands REALLY overflow. This is what
// guarantees the overflow scenario actually exercises the wrapping/saturating
// paths rather than silently staying in range. Uses checked_* (which never
// traps) to detect the overflow condition.
const _: () = {
    let (a, b) = ADD_OVERFLOW;
    assert!(a.checked_add(b).is_none(), "ADD_OVERFLOW must overflow");
    let (a, b) = SUB_OVERFLOW;
    assert!(a.checked_sub(b).is_none(), "SUB_OVERFLOW must underflow");
    let (a, b) = MUL_OVERFLOW;
    assert!(a.checked_mul(b).is_none(), "MUL_OVERFLOW must overflow");
};

/// An operation variant to be sampled: its display name plus the closure that
/// performs one op. Grouped into [`Variant`]s so several can be benchmarked
/// together under interleaved sampling.
struct Variant<'f, R> {
    name: String,
    f: Box<dyn FnMut(u32, u32) -> R + 'f>,
}

/// Benchmark several variants under *interleaved* sampling.
///
/// Benchmarks run back-to-back (as `bench` does) are vulnerable to slow drift:
/// turbo-boost frequency decays over time, thermals creep up, and the OS may
/// reschedule us. If variant A runs first while the CPU is fast and variant B
/// runs later while it has slowed down, the difference we measure is the drift,
/// not the code. Interleaving — cycling through every variant one batch at a
/// time, rotating the lead each round — makes all variants experience the same
/// drifting conditions within each cycle, so the drift cancels out of their
/// pairwise ratios. This is what makes comparisons like wrapping-vs-saturating
/// trustworthy enough to settle whether a gap is real.
fn bench_set<'f, R>(mut variants: Vec<Variant<'f, R>>, a: u32, b: u32) -> Vec<BenchmarkResult>
where
    R: Copy,
{
    let n = variants.len();
    let mut counters: Vec<BatchCounter> = vec![BatchCounter::new(); n];
    // One sink per variant; feeding each result into `black_box` is what stops
    // the compiler from constant-folding the loops away.
    let mut sinks: Vec<R> = Vec::with_capacity(n);
    for v in &mut variants {
        sinks.push((v.f)(a, b));
    }

    for round in 0..NUM_BATCHES {
        // Rotate the starting index each round so no single variant always goes
        // first (avoiding a systematic head-start like a warmer cache or a
        // momentarily higher boost clock).
        let start = (round as usize) % n;
        for k in 0..n {
            let idx = (start + k) % n;
            let t0 = Instant::now();
            for _ in 0..BATCH_ITERS {
                sinks[idx] = black_box((variants[idx].f)(black_box(a), black_box(b)));
            }
            counters[idx].add(t0.elapsed());
        }
    }

    let _ = black_box(sinks);
    variants
        .into_iter()
        .zip(counters)
        .map(|(v, c)| BenchmarkResult::new(v.name, &c))
        .collect()
}

/// A single mixed-workflow variant: a display name plus, for each of the five
/// operations, the closure implementing that op under this variant. All five
/// closures share the variant's semantics (e.g. all `checked_*`).
struct WorkflowVariant<'f> {
    name: String,
    add: Box<dyn FnMut(u32, u32) -> u32 + 'f>,
    sub: Box<dyn FnMut(u32, u32) -> u32 + 'f>,
    mul: Box<dyn FnMut(u32, u32) -> u32 + 'f>,
    div: Box<dyn FnMut(u32, u32) -> u32 + 'f>,
    rem: Box<dyn FnMut(u32, u32) -> u32 + 'f>,
}

/// Total number of *distinct* operation sequences for the current configuration:
/// the multinomial count `(copies * NUM_OPS)! / (copies!)^NUM_OPS`, where
/// `copies = WORKFLOW_PERM_SIZE / NUM_OPS`. This is the size of the space the
/// deduplicating sampler draws from, so it must be >= `NUM_BATCHES` or the
/// sampler would eventually exhaust every ordering and spin forever. Computed
/// with exact integer arithmetic; saturates to `u64::MAX` if the true value
/// overflows (which only ever makes the feasibility check easier to pass).
fn workflow_permutation_count() -> u64 {
    debug_assert_eq!(
        WORKFLOW_PERM_SIZE % NUM_OPS,
        0,
        "perm size must be a multiple of NUM_OPS"
    );
    let copies = (WORKFLOW_PERM_SIZE / NUM_OPS) as u64;
    let n = WORKFLOW_PERM_SIZE as u64; // == copies * NUM_OPS

    // numerator = n! ; denominator = (copies!)^NUM_OPS
    let mut numerator = 1u64;
    for k in 2..=n {
        numerator = numerator.saturating_mul(k);
    }
    let mut fact_copies = 1u64;
    for k in 2..=copies {
        fact_copies = fact_copies.saturating_mul(k);
    }
    let denominator = fact_copies.pow(NUM_OPS as u32);

    numerator / denominator.max(1)
}

/// Generate a scrambled operation sequence of length `WORKFLOW_PERM_SIZE` in
/// which each of the five operations appears exactly `WORKFLOW_PERM_SIZE / 5`
/// times. The order is shuffled using a small deterministic LCG seeded by
/// `seed`, so a given seed always reproduces the same sequence (keeping runs
/// comparable) while different seeds yield different scrambles.
fn workflow_permutation(seed: u64) -> Vec<usize> {
    debug_assert_eq!(
        WORKFLOW_PERM_SIZE % NUM_OPS,
        0,
        "perm size must be a multiple of NUM_OPS"
    );
    let copies = WORKFLOW_PERM_SIZE / NUM_OPS;

    // Start from the canonical multiset: each op repeated `copies` times.
    let mut seq: Vec<usize> = (0..NUM_OPS)
        .flat_map(|op| std::iter::repeat_n(op, copies))
        .collect();

    // Deterministic xorshift64 PRNG so the scramble is reproducible per seed.
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    if state == 0 {
        state = 0x2545_F491_4F6C_DD1D;
    }
    let mut next_u64 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    // Fisher-Yates shuffle driven by the PRNG.
    for i in (1..seq.len()).rev() {
        let j = (next_u64() % (i as u64 + 1)) as usize;
        seq.swap(i, j);
    }
    seq
}

/// Pre-generate `count` distinct scrambled orderings, deduplicated so no two
/// are identical. Seeds advance monotonically and collisions redraw, so the
/// result is deterministic for a given `count`. Panics if the permutation space
/// is smaller than `count` — which would make it impossible to satisfy without
/// reusing an ordering — rather than looping forever.
fn generate_workflow_perms(count: u64) -> Vec<Vec<usize>> {
    let total = workflow_permutation_count();
    assert!(
        total >= count,
        "requested {count} distinct workflow permutations but only {total} exist; \
         increase WORKFLOW_PERM_SIZE or decrease WORKFLOW_NUM_PERMS"
    );

    let mut perms: Vec<Vec<usize>> = Vec::with_capacity(count as usize);
    let mut used: HashSet<Vec<usize>> = HashSet::new();
    let mut seed: u64 = 1;
    while perms.len() < count as usize {
        let p = workflow_permutation(seed);
        if used.insert(p.clone()) {
            perms.push(p);
        }
        seed = seed.wrapping_add(1);
    }
    perms
}

/// Benchmark several *mixed-workflow* variants under interleaved sampling.
///
/// Unlike [`bench_set`] — which hammers a single operation in a tight loop and
/// therefore lets the CPU over-pipeline a homogeneous instruction stream — this
/// simulates a realistic workload by running a *scrambled* sequence of all five
/// operations within every batch. Two things keep the measurement honest:
///
/// 1. **Scrambling.** Ops run in a permuted order (each op appearing an equal
///    number of times per sequence, see [`workflow_permutation`]). A changing
///    instruction mix is a natural pipeline breaker — a `div` after a `mul`
///    alters the instruction mix, register pressure, and dependency chains — so
///    the out-of-order engine can't amortize one op's cost across a run of
///    identical successors, which is what inflates single-op results.
/// 2. **Pre-generated distinct pool.** Up front we build
///    [`WORKFLOW_NUM_PERMS`] unique orderings (deduplicated via a `HashSet`, see
///    [`generate_workflow_perms`]); every batch cycles through all of them. So a
///    single batch's wall-clock time already averages over many different
///    instruction orders, and the `NUM_BATCHES` samples then average over that —
///    no need to inflate `NUM_BATCHES` just to cover the permutation space.
///
/// An explicit `black_box` barrier precedes each op so the previous result is
/// consumed and the chain stays data-dependent. Variants are still sampled
/// interleaved (one batch of each per round, rotating the lead) so
/// clock/thermal drift cancels out of their ratios.
fn bench_workflow(mut variants: Vec<WorkflowVariant<'_>>) -> Vec<BenchmarkResult> {
    let n_variants = variants.len();
    // Operand pairs indexed by op slot (0=add, 1=sub, 2=mul, 3=div, 4=rem).
    const OPS: [(u32, u32); NUM_OPS] = [
        ADD_OPERANDS,
        SUB_OPERANDS,
        MUL_OPERANDS,
        DIV_OPERANDS,
        REM_OPERANDS,
    ];

    let mut counters: Vec<BatchCounter> = vec![BatchCounter::new(); n_variants];
    // One sink per variant; feeding each result into `black_box` stops the
    // compiler from constant-folding or reordering the mixed loop away.
    let mut sinks: Vec<u32> = vec![0; n_variants];

    // Pre-generate a fixed pool of distinct scrambled orderings. Every batch
    // cycles through all of them, so a single batch's wall-clock time averages
    // over many different instruction orders — no need for NUM_BATCHES to be
    // huge just to cover the permutation space.
    let perms = generate_workflow_perms(WORKFLOW_NUM_PERMS);
    println!("Mixed workflow permutation generation completed");

    for round in 0..NUM_BATCHES {
        // Rotate which variant leads each round so none gets a systematic
        // head-start (warmer cache, momentarily higher boost clock, ...).
        let start = (round as usize) % n_variants;
        for k in 0..n_variants {
            let vi = (start + k) % n_variants;
            let t0 = Instant::now();
            for _ in 0..BATCH_ITERS {
                // Cycle through every distinct ordering; within each, execute
                // one scrambled pass (every op exactly once per block).
                let mut acc = sinks[vi];
                for perm in &perms {
                    for &op_idx in perm {
                        acc = black_box(acc);
                        let (a, b) = OPS[op_idx];
                        let r = match op_idx {
                            0 => (variants[vi].add)(black_box(a), black_box(b)),
                            1 => (variants[vi].sub)(black_box(a), black_box(b)),
                            2 => (variants[vi].mul)(black_box(a), black_box(b)),
                            3 => (variants[vi].div)(black_box(a), black_box(b)),
                            _ => (variants[vi].rem)(black_box(a), black_box(b)),
                        };
                        acc ^= r;
                    }
                }
                sinks[vi] = black_box(acc);
            }
            println!("Mixed workflow batch {:?}, completed", (round, k));
            counters[vi].add(t0.elapsed());
        }
    }

    let _ = black_box(sinks);
    variants
        .into_iter()
        .zip(counters)
        .map(|(v, c)| BenchmarkResult::new(v.name, &c))
        .collect()
}

// ---------------------------------------------------------------------------
// Operation variants
// ---------------------------------------------------------------------------

/// Declare one operation module with its available variants.
///
/// Each module exposes the individual variant functions plus two drivers:
/// `run_benchmarks` (the in-range scenario, all applicable variants) and, for
/// operations that have both, `run_overflow_benchmarks` (the overflow scenario,
/// comparing `wrapping` against `saturating`).
///
/// The saturating form is optional because std has no `saturating_div` or
/// `saturating_rem`; operations without one simply omit it and have no overflow
/// scenario.
macro_rules! op_variants {
    ($module_name:ident, $operands:path, $overflow_operands:expr, $normal_fn:path, $checked:ident, $wrapping:ident, $saturating:ident) => {
        #[allow(dead_code)]
        pub mod $module_name {
            use super::*;

            /// The in-range operand pair this operation is benchmarked with.
            pub const OPERANDS: (u32, u32) = $operands;

            /// The overflow operand pair used by the overflow scenario.
            pub const OVERFLOW_OPERANDS: (u32, u32) = $overflow_operands;

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

            /// In-range scenario: every applicable variant, sampled interleaved
            /// so pairwise ratios aren't skewed by clock/thermal drift. `checked`
            /// is safe to unwrap here because `OPERANDS` is compile-time proven
            /// in range.
            pub fn run_benchmarks() -> Vec<BenchmarkResult> {
                let (a, b) = OPERANDS;
                bench_set(
                    vec![
                        Variant {
                            name: format!("{NAME}/normal"),
                            f: Box::new(|a, b| normal(a, b)),
                        },
                        Variant {
                            name: format!("{NAME}/checked"),
                            f: Box::new(|a, b| checked(a, b).unwrap()),
                        },
                        Variant {
                            name: format!("{NAME}/wrapping"),
                            f: Box::new(|a, b| wrapping(a, b)),
                        },
                        Variant {
                            name: format!("{NAME}/saturating"),
                            f: Box::new(|a, b| saturating(a, b)),
                        },
                    ],
                    a,
                    b,
                )
            }

            /// Overflow scenario: compare `wrapping` against `saturating` on
            /// operands whose result leaves the representable range. Only these
            /// two are meaningful here — `normal` would trap in debug builds and
            /// `checked` would just return `None`. Sampled interleaved (see
            /// `bench_set`) so slow clock/thermal drift cancels out of their
            /// ratio.
            pub fn run_overflow_benchmarks() -> Vec<BenchmarkResult> {
                let (a, b) = OVERFLOW_OPERANDS;
                bench_set(
                    vec![
                        Variant {
                            name: format!("{NAME}/wrapping[overflow]"),
                            f: Box::new(|a, b| wrapping(a, b)),
                        },
                        Variant {
                            name: format!("{NAME}/saturating[overflow]"),
                            f: Box::new(|a, b| saturating(a, b)),
                        },
                    ],
                    a,
                    b,
                )
            }
        }
    };
    // No saturating variant (std defines none for div/rem), hence no overflow
    // scenario either.
    ($module_name:ident, $operands:path, $normal_fn:path, $checked:ident, $wrapping:ident) => {
        #[allow(dead_code)]
        pub mod $module_name {
            use super::*;

            /// The in-range operand pair this operation is benchmarked with.
            pub const OPERANDS: (u32, u32) = $operands;

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

            /// In-range scenario: every applicable variant, sampled interleaved
            /// so pairwise ratios aren't skewed by clock/thermal drift. `checked`
            /// is safe to unwrap here because `OPERANDS` is compile-time proven
            /// in range.
            pub fn run_benchmarks() -> Vec<BenchmarkResult> {
                let (a, b) = OPERANDS;
                bench_set(
                    vec![
                        Variant {
                            name: format!("{NAME}/normal"),
                            f: Box::new(|a, b| normal(a, b)),
                        },
                        Variant {
                            name: format!("{NAME}/checked"),
                            f: Box::new(|a, b| checked(a, b).unwrap()),
                        },
                        Variant {
                            name: format!("{NAME}/wrapping"),
                            f: Box::new(|a, b| wrapping(a, b)),
                        },
                    ],
                    a,
                    b,
                )
            }
        }
    };
}

/// Plain operators for the "normal" variant. Debug builds trap on overflow;
/// release builds wrap silently.
macro_rules! normal_impl {
    ($name:ident, $expr:tt) => {
        fn $name(a: u32, b: u32) -> u32 {
            a $expr b
        }
    };
}

normal_impl!(add_normal, +);
normal_impl!(sub_normal, -);
normal_impl!(mul_normal, *);
normal_impl!(div_normal, /);
normal_impl!(rem_normal, %);

op_variants!(
    add,
    ADD_OPERANDS,
    ADD_OVERFLOW,
    add_normal,
    checked_add,
    wrapping_add,
    saturating_add
);
op_variants!(
    sub,
    SUB_OPERANDS,
    SUB_OVERFLOW,
    sub_normal,
    checked_sub,
    wrapping_sub,
    saturating_sub
);
op_variants!(
    mul,
    MUL_OPERANDS,
    MUL_OVERFLOW,
    mul_normal,
    checked_mul,
    wrapping_mul,
    saturating_mul
);
op_variants!(div, DIV_OPERANDS, div_normal, checked_div, wrapping_div);
op_variants!(rem, REM_OPERANDS, rem_normal, checked_rem, wrapping_rem);

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Run the in-range benchmarks for every operation, grouped by operation.
pub fn run_all() -> Vec<(String, Vec<BenchmarkResult>)> {
    vec![
        (add::NAME.to_string(), add::run_benchmarks()),
        (sub::NAME.to_string(), sub::run_benchmarks()),
        (mul::NAME.to_string(), mul::run_benchmarks()),
        (div::NAME.to_string(), div::run_benchmarks()),
        (rem::NAME.to_string(), rem::run_benchmarks()),
    ]
}

/// Run the overflow benchmarks (`wrapping` vs `saturating`) for the operations
/// that have both variants, grouped by operation.
pub fn run_overflow_all() -> Vec<(String, Vec<BenchmarkResult>)> {
    vec![
        (add::NAME.to_string(), add::run_overflow_benchmarks()),
        (sub::NAME.to_string(), sub::run_overflow_benchmarks()),
        (mul::NAME.to_string(), mul::run_overflow_benchmarks()),
    ]
}

/// Build the four mixed-workflow variants. Each supplies all five operations so
/// the round-robin is uniform. `checked` unwraps (safe: operands are in range);
/// `saturating` has no std `div`/`rem`, so those two steps fall back to the
/// plain operators — noted here because it means the saturating workflow's
/// div/rem cost reflects the base op, not a saturating path.
fn build_workflow_variants() -> Vec<WorkflowVariant<'static>> {
    vec![
        WorkflowVariant {
            name: "workflow/normal".into(),
            add: Box::new(add_normal),
            sub: Box::new(sub_normal),
            mul: Box::new(mul_normal),
            div: Box::new(div_normal),
            rem: Box::new(rem_normal),
        },
        WorkflowVariant {
            name: "workflow/checked".into(),
            add: Box::new(|a, b| a.checked_add(b).unwrap()),
            sub: Box::new(|a, b| a.checked_sub(b).unwrap()),
            mul: Box::new(|a, b| a.checked_mul(b).unwrap()),
            div: Box::new(|a, b| a.checked_div(b).unwrap()),
            rem: Box::new(|a, b| a.checked_rem(b).unwrap()),
        },
        WorkflowVariant {
            name: "workflow/wrapping".into(),
            add: Box::new(|a, b| a.wrapping_add(b)),
            sub: Box::new(|a, b| a.wrapping_sub(b)),
            mul: Box::new(|a, b| a.wrapping_mul(b)),
            div: Box::new(|a, b| a.wrapping_div(b)),
            rem: Box::new(|a, b| a.wrapping_rem(b)),
        },
        WorkflowVariant {
            name: "workflow/saturating".into(),
            add: Box::new(|a, b| a.saturating_add(b)),
            sub: Box::new(|a, b| a.saturating_sub(b)),
            mul: Box::new(|a, b| a.saturating_mul(b)),
            // No saturating div/rem in std; use the plain operators.
            div: Box::new(div_normal),
            rem: Box::new(rem_normal),
        },
    ]
}

/// Run the mixed-workflow benchmark: each variant round-robines all five
/// operations per batch, with pipeline-breaking barriers between them. Returns
/// one result per variant.
pub fn run_workflow_all() -> Vec<BenchmarkResult> {
    bench_workflow(build_workflow_variants())
}

fn print_row(r: &BenchmarkResult) {
    println!(
        "  {:<22} mean {:>12.3e} ns   median {:>12.3e} ns   min {:>12.3e} ns   max {:>12.3e} ns",
        r.name,
        r.mean_batch.as_nanos() as f64,
        r.median_batch.as_nanos() as f64,
        r.min_batch.as_nanos() as f64,
        r.max_batch.as_nanos() as f64,
    );
}

/// Print every row for one operation, then a summary line giving each variant's
/// mean time as a multiple of the first (baseline) variant. Because the rows
/// were sampled interleaved, these ratios are the quantities most likely to
/// reflect a real code difference rather than clock drift; a value near 1.0
/// means no measurable difference from the baseline.
fn print_rows(op: &str, rows: &[BenchmarkResult]) {
    println!("{op}:");
    let base = rows.first().map(|r| r.mean_batch.as_nanos() as f64);
    for r in rows {
        print_row(r);
    }
    if let Some(base) = base.filter(|b| *b > 0.0) {
        let baseline_label = rows[0].name.rsplit('/').next().unwrap_or("");
        let parts: Vec<String> = rows
            .iter()
            .map(|r| {
                let label = r.name.rsplit('/').next().unwrap_or("");
                let ratio = r.mean_batch.as_nanos() as f64 / base;
                format!("{label}={ratio:.3}x")
            })
            .collect();
        // The first entry is the baseline itself (always 1.000x); keeping it makes
        // explicit which variant the others are measured against.
        println!("  -> vs {baseline_label} (mean): {}", parts.join("  "));
    }
    println!();
}

fn print_section(title: &str, subtitle: &str, results: &[(String, Vec<BenchmarkResult>)]) {
    println!("{title}");
    println!("{subtitle}\n");
    for (op, rows) in results {
        print_rows(op, rows);
    }
}

/// Pretty-print all results to stdout: the in-range scenario, the overflow
/// scenario, and the mixed-workflow scenario. All three are sampled interleaved,
/// so the per-variant ratios printed under each are directly comparable.
pub fn print_report(
    in_range: &[(String, Vec<BenchmarkResult>)],
    overflow: &[(String, Vec<BenchmarkResult>)],
    workflow: &[BenchmarkResult],
) {
    print_section(
        "=== In-range operands (all variants, interleaved) ===",
        &format!(
            "({NUM_BATCHES} batches of {BATCH_ITERS} ops each; figures are wall-clock time per batch)"
        ),
        in_range,
    );
    print_section(
        "=== Overflowing operands (wrapping vs saturating, interleaved) ===",
        &format!(
            "({NUM_BATCHES} batches of {BATCH_ITERS} ops each; figures are wall-clock time per batch)"
        ),
        overflow,
    );
    // The workflow scenario cycles through WORKFLOW_NUM_PERMS distinct scrambled
    // orderings per iteration, each of length WORKFLOW_PERM_SIZE ops, with
    // pipeline barriers between them. Each batch therefore covers
    // BATCH_ITERS * WORKFLOW_NUM_PERMS * WORKFLOW_PERM_SIZE ops.
    print_section(
        "=== Mixed workflow (scrambled op order, deduplicated, interleaved) ===",
        &format!(
            "({NUM_BATCHES} batches of {BATCH_ITERS} iters x {WORKFLOW_NUM_PERMS} perms x {WORKFLOW_PERM_SIZE} ops each; figures are wall-clock time per batch)"
        ),
        &[("workflow".to_string(), workflow.to_vec())],
    );
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

    /// Prove the overflow operand pairs genuinely leave the representable range.
    /// This is the guarantee that the overflow scenario actually exercises the
    /// wrapping/saturating paths rather than silently staying in range. Uses
    /// `checked_*`, which never traps, to detect the condition.
    #[test]
    fn sanity_overflow_operands_really_overflow() {
        let (a, b) = ADD_OVERFLOW;
        assert!(a.checked_add(b).is_none(), "ADD_OVERFLOW must overflow");
        assert_eq!(a.wrapping_add(b), 0);
        assert_eq!(a.saturating_add(b), u32::MAX);

        let (a, b) = SUB_OVERFLOW;
        assert!(a.checked_sub(b).is_none(), "SUB_OVERFLOW must underflow");
        assert_eq!(a.wrapping_sub(b), u32::MAX);
        assert_eq!(a.saturating_sub(b), 0);

        let (a, b) = MUL_OVERFLOW;
        assert!(a.checked_mul(b).is_none(), "MUL_OVERFLOW must overflow");
        // 0x7FFF_FFFF * 4 = 0x1_FFFF_FFFC; low 32 bits are 0xFFFF_FFFC.
        assert_eq!(a.wrapping_mul(b), 0xFFFF_FFFC);
        assert_eq!(a.saturating_mul(b), u32::MAX);
    }

    /// Confirm the in-range `checked` unwrap used by the benchmarks cannot
    /// panic: every in-range operand pair yields `Some`.
    #[test]
    fn sanity_in_range_checked_is_some() {
        let (a, b) = ADD_OPERANDS;
        assert!(a.checked_add(b).is_some());
        let (a, b) = SUB_OPERANDS;
        assert!(a.checked_sub(b).is_some());
        let (a, b) = MUL_OPERANDS;
        assert!(a.checked_mul(b).is_some());
        let (a, b) = DIV_OPERANDS;
        assert!(a.checked_div(b).is_some());
        let (a, b) = REM_OPERANDS;
        assert!(a.checked_rem(b).is_some());
    }

    /// Lock in the permutation generator's contract: correct length, each op
    /// appearing exactly `WORKFLOW_PERM_SIZE / NUM_OPS` times, determinism for a
    /// fixed seed, and actual variation across different seeds.
    #[test]
    fn workflow_permutation_invariants() {
        let copies = WORKFLOW_PERM_SIZE / NUM_OPS;

        // Length is exact.
        let p0 = workflow_permutation(1);
        assert_eq!(p0.len(), WORKFLOW_PERM_SIZE);

        // Every op appears exactly `copies` times.
        let mut counts = [0usize; NUM_OPS];
        for &op in &p0 {
            assert!(op < NUM_OPS, "op index out of range");
            counts[op] += 1;
        }
        for c in counts {
            assert_eq!(c, copies, "each op must appear exactly `copies` times");
        }

        // Deterministic for a fixed seed.
        assert_eq!(workflow_permutation(42), workflow_permutation(42));

        // Different seeds yield at least one different ordering (draw a handful
        // and confirm we don't get all-identical sequences).
        let mut seen: HashSet<Vec<usize>> = HashSet::new();
        for s in 1..=8u64 {
            seen.insert(workflow_permutation(s));
        }
        assert!(
            seen.len() > 1,
            "different seeds should scramble differently"
        );
    }

    /// Verify the multinomial permutation count against a hand-computed value
    /// and confirm it comfortably exceeds `WORKFLOW_NUM_PERMS` (the pool size we
    /// pre-generate; if the space were smaller, [`generate_workflow_perms`] would
    /// be unable to fill the pool without reusing an ordering). For the default
    /// config: copies = 2, so 10! / (2!)^5 = 3628800 / 32 = 113400.
    #[test]
    fn workflow_permutation_count_feasible() {
        let total = workflow_permutation_count();
        // Exact expected value for WORKFLOW_PERM_SIZE = 10, NUM_OPS = 5.
        assert_eq!(total, 113_400);
        // The pre-generated pool must fit inside the permutation space.
        assert!(
            total >= WORKFLOW_NUM_PERMS,
            "permutation space must cover the requested pool"
        );
    }

    /// Confirm the pre-generator returns exactly `count` mutually-distinct
    /// sequences, each satisfying the same invariants as a single permutation.
    #[test]
    fn generate_workflow_perms_unique_and_valid() {
        let perms = generate_workflow_perms(WORKFLOW_NUM_PERMS);
        assert_eq!(perms.len(), WORKFLOW_NUM_PERMS as usize);

        // All distinct.
        let mut set: HashSet<Vec<usize>> = HashSet::new();
        for p in &perms {
            assert!(set.insert(p.clone()), "duplicate ordering generated");
        }

        // Each is a valid multiset permutation (right length, balanced ops).
        let copies = WORKFLOW_PERM_SIZE / NUM_OPS;
        for p in &perms {
            assert_eq!(p.len(), WORKFLOW_PERM_SIZE);
            let mut counts = [0usize; NUM_OPS];
            for &op in p {
                assert!(op < NUM_OPS);
                counts[op] += 1;
            }
            for c in counts {
                assert_eq!(c, copies);
            }
        }
    }
}
