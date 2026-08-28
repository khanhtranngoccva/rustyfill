//! Typed deserialization layer for rustdoc JSON.
//!
//! One file per domain type. The wire format is **externally tagged**: every
//! sum value is `{"<tag>": <payload>}`, so plain derived `Deserialize` with
//! `#[serde(rename_all = "snake_case")]` handles it directly — including the
//! self-recursive types (`TypeRepr`, `GenericArg`, `GenericArgs`). External
//! tagging reads one key and dispatches to exactly one variant, so there is no
//! fan-out over variants and no recursion hazard on malformed input; unknown
//! tags fail loudly with "unknown variant …".
//!
//! Format-version drift that stays within a single field (e.g. `name` ↔ `path`)
//! is absorbed with `#[serde(alias)]`. If a future format forks an entire
//! sub-shape, model it as versioned variants in that file — one domain type per
//! file keeps such changes local.

pub mod original;

pub use original::*;
