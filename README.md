# rustyfill

[![CI](https://github.com/khanhtranngoccva/rustyfill/actions/workflows/ci.yml/badge.svg)](https://github.com/khanhtranngoccva/rustyfill/actions/workflows/ci.yml)
[![CRAP](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkhanhtranngoccva%2Frustyfill%2Fbadges%2Fcrap-badge.json)](https://github.com/khanhtranngoccva/rustyfill/actions/workflows/ci.yml)

Fallible allocation abstractions for Rust's standard library types and constructs that prevent and mitigate out of memory conditions, packaged as a monorepo.

Many trivial operations like `Clone::clone()`, `Default::default()`, and `Vec::push()` panic on out-of-memory. In safety-critical code, embedded systems, or any context where panics are unacceptable, that behavior is untenable. **rustyfill** provides drop-in, `Result`-returning alternatives so callers can handle allocation failures gracefully — including recovery from mid-operation failure without losing data.

## Why this exists

Allocation failure in Rust is treated as unrecoverable: the allocator usually aborts, taking the whole process with it. But "allocation failed" is a perfectly normal, recoverable condition in constrained environments — the program can retry later or fail one operation while keeping everything else alive. The problem is structural: stdlib APIs return values directly rather than `Result`s, so there is no way to express failure at the call site.

rustyfill solves this by wrapping every allocating operation in a fallible counterpart (`try_push`, `fallible_new`, `try_clone`, …) built on top of the same underlying data structures. 

Where an operation can fail partway through, it guarantees atomicity or resumability: either the operation fully succeeds, or the data structure is restored to its prior state (or a resumable handle is handed back so the caller can continue from where it stopped).

## Repository layout

| Crate | Published | Role |
|---|---|---|
| [`rustyfill`](rustyfill/) | yes | The user-facing crate: `Try*` extension traits, ponyfilled error types, iterator recovery, hasher factories. |
| [`rustyfill-errors`](rustyfill-errors/) | yes | Context-aware error reporting with frame stacks (inspired by `error-stack`), redesigned so reports degrade gracefully under memory pressure instead of failing to build. |
| [`rustyfill-macros`](rustyfill-macros/) | internal | Procedural macros: `#[derive(TryClone)]`, `#[derive(TryDefault)]`, `#[derive(TryDebug)]`, tuple impl generators, and format macros. |
| [`rustyfill-sys`](rustyfill-sys/) | internal | Build-time mirror of stdlib internal data structures with identical field layout. This is what makes "wrap the real type" possible without depending on unstable internals. |
| [`rustyfill-sys-bindings`](rustyfill-sys-bindings/) | internal | The binding-generation pipeline (parsing, discovery, emission, validation) used by `rustyfill-sys`'s build script. |
| [`rustyfill-test-allocator`](rustyfill-test-allocator/) | dev-only | A global test allocator that simulates OOM on demand — how the entire suite proves its non-panicking guarantees. Dev-dependency only; never ship it. |
| [`antipatterns`](antipatterns/) | unpublished | Executable demonstrations of things that look safe but aren't (e.g. hidden allocations inside `Format`/`Debug`). |
| [`experiments`](experiments/) | unpublished | Scratch benchmarks and investigations that inform design decisions. |
| [`xtask`](xtask/) | unpublished | Developer task runner (`cargo xtask sanitize`, `cargo xtask miri`). |

## API and Architecture

### rustyfill: core data structures and abstractions

The central challenge is that the interesting interior of `Vec`, `BTreeMap`, `Mutex`, and friends lives in `core`/`alloc`/`std` internals that are not public API. The repo addresses this in layers:

- **Mirrored layouts.** `rustyfill-sys` parses the actual stdlib source tree (via the `rust-src` component) at build time and emits synthetic copies of the relevant internal types — same fields, same order, same alignment — organized under a mirrored module hierarchy. Because the layout is byte-identical, `rustyfill` can operate on the innards of a real `Vec<T>` or `BTreeMap<K, V>` through these mirrors without any unsound transmutes. 

*Important: The build refuses to run if `-Zrandomize-layout` is active, since layout stability is a hard invariant. This requirement propagates down to the end user application.*

- **Ponyfilled errors.** Stable Rust has no stable representation of "the allocator said no." `rustyfill` ships its own `AllocError` and `TryReserveErrorKind` equivalents that behave like their future stdlib counterparts. On nightly with the `allocator-api` feature, it swaps in the real types transparently, so code written against the ponyfills keeps working as the ecosystem catches up.

- **Atomic-or-resume semantics.** Many mutating operations reserve capacity before doing logical work, so a failure short-circuits with nothing half-done. Iterator-driven operations (`try_extend`) can't be rolled back because iterators consume their source, and some map/set operations cannot be rolled back because values may have been overwritten; instead they hand back a `Resumable` handle carrying the stranded element plus the unconsumed remainder, letting the caller retry from exactly where it stopped.

- **Provable non-panic.** The guarantee is enforced, not assumed. Tests run against `rustyfill-test-allocator`, which flips a thread-local policy to make the Nth allocation fail, then asserts the operation returned `Err` and left the structure intact. CI additionally runs Miri (UB detection) and leak/address sanitizers on nightly.

### rustyfill-errors: context-aware error reports

Building an error report is itself an allocating operation — and in exactly the situations where you want rich diagnostics (memory pressure), that's precisely when it can fail. `rustyfill-errors` is a frame-stack error container (in the spirit of `error-stack`) redesigned so that reporting degrades gracefully instead of becoming a second failure mode:

- **Inline head, discardable tail.** A `Report<C>` stores its current error inline (zero allocation), with additional peer frames in a deque that can be optionally capped. If allocation fails while attaching data or pushing peers, the affected frame is dropped and a counter (`lost_attachments`, `lost_children`, `lost_peers`) records how much context was sacrificed — the surviving report always renders, and the render output makes the loss visible rather than silently pretending nothing happened.
- **Demotion on context change.** When you wrap an error in a new context type (`change_context`), the old frames aren't thrown away: they're demoted into type-erased child frames under the new head, preserving the causal chain. Under memory pressure, oldest peers are evicted first, again tracked by counters.
- **Lossy-by-design API.** The fluent helpers (`attach`, `attach_lazy`, `change_context`, …) never return secondary errors for dropped context — a diagnostic that fails halfway is worse than a thinner one. Attachments come in two flavors: printable (must implement `TryDebug` + `TryDisplay`, rendered into the report) and opaque (any `'static` payload, carried along without rendering).
- **Built on rustyfill.** Every internal allocation goes through the fallible traits (`TryVecDeque`, `TryBox`, …), so the crate inherits the same provable non-panic guarantees as the rest of the workspace, including `no_std` support.

## Feature flags

On the published `rustyfill` crate:

- **`std`** *(default)* — path/FFI string wrappers, `RandomState` helpers, hash collections, the fallible B-tree entry API. When disabled, the crate is `no_std` + `alloc`.
- **`unstable`** — `TryDashMap` / `TryDashSet` wrappers. Deliberately caveated: `DashMap::new()` allocates shards with no fallible constructor, so *construction* can still panic; only mutations are covered.
- **`allocator-api`** — nightly-only; re-exports the real allocation error types instead of the allocation error ponyfills. Silently ignored on stable.

## Development

```bash
cargo test --workspace --all-features           # full test suite
cargo xtask miri                                # UB detection under Miri (nightly)
cargo xtask sanitize                            # leak sanitizer (nightly)
```

CI (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs formatting, tests on Linux/macOS/Windows/FreeBSD, all-features builds, Miri, `no_std` checks, sanitizers, and a CRAP complexity gate that posts per-function coverage-weighted risk scores to pull requests and publishes the badge shown above.

## Status

The crate is pre-1.0. The API surface is stabilizing but may still change; breaking changes will be noted in release notes. Please avoid using it during production for now.

The crate is being prototyped using LLMs, specifically [Thaura](https://thaura.ai) and uses an experimental approach to resemble as much of std as possible, so expect some rough edges during compilation or runtime. Once the codebase stabilizes, it is expected in the near future that all code will be fully validated by a human.

## Roadmap

This crate can serve as a future foundation for building OOM-resilient libraries and apps.

In the future, it may port over important libraries like `compio` to enable a performant async runtime with OOM resilience.

## License

This crate's license is being considered, but it is likely not going to be fully open-source (even though in practice it's *almost* open-source anyway). It is likely that the Hippocratic License version 3.0 or a custom variant will be used. Protective measures may be implemented at build time to ensure that the spirit of such license is followed.

This crate borrows the implementation from a number of MIT-licensed crates like `std`/`alloc`/`core`, `dashmap`, and `error-stack`. To ensure compliance, copies of these licenses are left at the respective `licenses/originals` directory. These licenses are non-binding, however.

