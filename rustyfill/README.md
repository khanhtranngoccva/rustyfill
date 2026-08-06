# rustyfill

Fallible allocation polyfills for Rust's standard library types.

Standard-library operations like `Clone::clone()`, `Default::default()`, and `Vec::push()` panic on out-of-memory. In safety-critical code, embedded systems, or any context where panics are unacceptable, that behavior is untenable. rustyfill provides drop-in, `Result`-returning alternatives so callers can handle allocation failures gracefully.

## Quick start

```toml
[dependencies]
rustyfill = "0.1"
```

```rust
use rustyfill::prelude::*;

// Fallible clone instead of .clone() which may panic
let s = String::from("hello");
let cloned: String = s.try_clone()?;

// Fallible default construction
let v: Vec<i32> = Vec::try_default()?;

// Fallible push, insert, extend, etc.
let mut vec = Vec::<i32>::new();
vec.try_push(1)?;
vec.try_extend(2..=5)?;

// Fallible Box, Arc, HashMap, HashSet, String, CString, PathBuf...
let boxed = Box::<i32>::fallible_new(42)?;
let arc = Arc::<String>::fallible_new("shared".into())?;
let mut map = HashMap::<&str, i32>::try_with_capacity(10)?;
map.try_insert("key", 42)?;
```

Import `rustyfill::prelude::*` to bring all `Try*` extension traits into scope as inherent-style methods on the standard types.

## Design philosophy

Outside of deprecated structures, every operation in this crate guarantees it will not panic on allocation failure. The foundational traits (`TryClone`, `TryDefault`, `TryToOwned`) require their standard counterparts as supertraits, ensuring compatibility with existing APIs while providing a safe escape hatch.

Certain operations guarantee atomicity in the event of failure - when they fail, these operations restore the old state of the data structure. They do so by reserving capacity before performing logical work so that allocation failures short-circuit early, avoiding wasted computation or partially constructed intermediate values. If short circuiting is impossible because the operation is midway, failures automatically trigger a rollback.

Operations involving extending from iterators cannot be atomic because iterators cannot be restored to a previous state, but as long as the library can hold a stranded element, the caller can resume the operation from that point after a delay. The crate offers facilities for this pattern.

## What it covers

### Foundational traits

| Trait | Analogue | Purpose |
|---|---|---|
| `TryClone` | `Clone` | Clone without panicking on OOM |
| `TryDefault` | `Default` | Construct defaults without panicking |
| `TryToOwned` | `ToOwned` | Produce owned values fallibly |
| `TryRandomState` | -- | Build `RandomState` without panicking (falls back to SplitMix64 PRNG if the OS random source fails) |

The library implements `TryClone` for most built-in types that are `Clone`, and `TryDefault` for most built-in types that are `Default`. Derive macros `#[derive(TryClone)]` and `#[derive(TryDefault)]` are provided for custom structs and enums.

### Extension traits for standard and popular types

| Trait | Type | Key operations |
|---|---|---|
| `TryVec` | `Vec<T>` | `try_push`, `try_insert`, `try_extend`, `try_reserve`, `try_append` |
| `TrySlice` | `[T]` | `try_to_vec` |
| `TryString` | `String` | `try_push_str`, `try_push`, `try_insert_str`, `try_insert` |
| `TryStr` | `str` | `try_to_string` |
| `TryBox` | `Box<T>` | `fallible_new`, `fallible_new_uninit`, `fallible_pin` |
| `TryArc` | `Arc<T>` | `fallible_new` |
| `TryWeak` | `Weak<T>` | `fallible_new` |
| `TryHashMap` | `HashMap<K, V, S>` | `try_insert`, `try_extend`, `try_reserve`, entry API |
| `TryHashSet` | `HashSet<T, S>` | `try_insert`, `try_extend`, `try_reserve` |
| `TryVecDeque` | `VecDeque<T>` | `try_push_back`, `try_push_front`, `try_extend`, `try_reserve` |
| `TryCString` | `CString` | `try_new`, `try_new_give_back` |
| `TryOsString` | `OsString` | `try_new`, `try_push` |
| `TryPath` | `&Path` | `try_to_path_buf`, `try_join`, `try_with_added_extension` |
| `TryPathBuf` | `PathBuf` | `try_new`, `try_push`, `try_set_extension`, `try_add_extension` |

### Iterator recovery

When `try_extend` fails mid-stream, elements from the iterator may have been consumed but not committed. The `recovery` module provides `Resumable<I>` to re-package a stranded element alongside the remainder so the caller can retry without losing data or introducing new generic parameters across retries.

```rust
use rustyfill::prelude::*;

let mut vec = Vec::<i32>::new();
let items = 0..10_000;

let remaining = match vec.try_extend(items) {
    Ok(()) => return,
    Err((_err, resumable)) => resumable.into_remainder(),
};

// Retry with the unconsumed tail
match vec.try_extend(Resumable::from_remainder(remaining)) {
    Ok(()) => {},
    Err(_) => /* handle */ {},
};
```

## Features

- **`panic`** — enables `TryBTreeMap` and `TryBTreeSet` wrappers that rely on `std::panic::catch_unwind`. Requires compiling with `panic = "unwind"`. The build script errors if this feature is enabled under `panic = "abort"`.
    - Note that these structures does not work by default and is not resilient against true out-of-memory conditions, making them unsuitable for production scenarios: 
        - The default OOM handler when linked against std aborts the process.
        - Since panic allocates memory, these structures abort the process when absolutely no memory can be allocated.
    - If you wish to use it anyway, you also need to call `std::alloc::set_alloc_error_hook` to prevent the default abort behavior and trigger a panic, but you risk non-cooperating code interfering with the process.
- **`unstable`** - enables `TryDashMap` and `TryDashSet`. Currently, they do not work, because the fallible construction API is not there yet, although [these APIs are pending review](https://github.com/xacrimon/dashmap/pull/372). They might be removed from this crate and moved into a downstream crate to allow usage in libraries.

## Hasher factories

Fallible hash collections require the hasher to implement `TryClone` (and optionally `TryDefault`). The `hashers` module provides two wrapper types:

- **`CopyHasherFactory<H>`** — for hashers that are `Copy`. Duplication is a zero-cost bitwise copy, inherently infallible and misuse-resistant.
- **`ArbitraryHasherFactory<H>`** — for any `BuildHasher`, including allocating ones like `RandomState`. Wraps `clone()` and `default()` in `catch_unwind` as a best-effort safety net. Constructor is `unsafe` as a discipline checkpoint.


