# rustyfill — Project Notes

## Core design: reference-only vs mirrored types

The bindings generator mirrors *selected* std/core/alloc data structures into a
synthetic `crate::std` / `crate::core` / `crate::alloc` tree (under a wrapper
module), but **most** public types are *referenced*, not mirrored. The distinction
matters and is easy to get wrong:

- **Mirrored (declared)**: a type explicitly named in the loader spec
  (`target.declare_struct(...)`) or registered as a known external type
  (`target.add_known_type(...)`). The emitter writes out its struct body into a
  generated binding file at its canonical location. In the `TypeRegistry`,
  `TypeInfo.declared == true`, so bare references resolve to
  `FieldRefResolution::Mirrored` → rewritten to `crate::{wrapper}::<path>`.

- **Reference-only (original)**: a public type that is *not* declared. It stays
  routed to the **real builtin crate** through the preamble prelude
  (`emitter.rs` `PREAMBLE_CORE_CONTENT`), e.g.
  `pub use ::__rustyfill_builtin_core::sync::atomic::{AtomicBool, AtomicPtr, ...}`.
  Bare references resolve to `FieldRefResolution::Original`. No body is emitted.

### Rule of thumb for choosing between them

If a type is a **stable, public, self-contained** item in real core/std/alloc
(e.g. `core::sync::atomic::AtomicPtr<T>` = `{ p: UnsafeCell<*mut T> }`), do **not**
stub it — let it route to the original via the prelude. Hand-writing a stub body
diverges from the real type's shape and causes downstream mismatches (the
darwin/netbsd `Atomic<*mut Mutex>` vs `AtomicPtr<Mutex>` build break was exactly
this: a fake `UnsafeCell<*mut T>` stub shadowed the real `AtomicPtr`).

Only use `add_known_type` for types that genuinely **cannot compile** when
mirrored — e.g. generic `Atomic<T>` (unstable `generic_atomic`, absent from stable
core) or types whose real definition pulls in unavailable machinery. Those need a
polyfill body because there is no usable original to point at.

### Emission gate

`emit_parsed_items` (`emitter.rs`) only emits an item if
`type_registry.is_declared_in_module(lib, module, leaf)` is true — i.e. it is
spec-declared, a declared alias, or already registered at that exact path. So a
public-but-undeclared item sitting in a parsed source file is silently dropped
from output; declaring it makes the pipeline mirror the real definition instead.

## Verification protocol

- Unit/integration: `cargo test` (workspace; ~1400+ tests across 23 binaries).
- Multi-target lint matrix: `cargo run -p xtask -- clippy [--targets <csv>]`.
  Default lints all supported cross targets with `-D warnings`. Some BSD targets
  (openbsd/freebsd-aarch64) may lack installed rust-std components; pass an
  explicit `--targets` list of installed triples to run the rest.
- Regenerating bindings after a spec change: delete
  `target/<triple>/debug/build/rustyfill-sys-*` then rebuild, since the build
  script only re-runs on `build/spec.rs` mtime changes.

## Path handling

Pure-module path manipulation lives in `rustyfill-sys-bindings/src/syntaxes/`
(one OO type per file). `ModulePath` owns slash ↔ `::`-canonical ↔ segment-list
conversions and renders filesystem paths via `to_file_path()`. Prefer it over ad-hoc
`replace('/', "::")` string surgery. Qualified-item strings (module + trailing item
name) and cross-boundary HashMap keys (`qualifier_routes`, `module_alias_routes`)
are intentionally still handled as raw strings.
