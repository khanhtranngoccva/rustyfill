//! Loader specification: describes which types to bind.
//!
//! Bindings are driven by explicit struct declarations in path syntax
//! (e.g., `"collections::btree::map::BTreeMap"`). For every declared struct,
//! its definition is emitted from the std source file that defines it, and
//! any re-export aliases of that struct (discovered through `pub use`
//! statements) resolve to the same emitted definition.
//!
//! Field types of declared structs are checked for publicity: a field whose
//! type is public and undeclared refers to the original (real core/alloc/std)
//! type; a field whose type is private and undeclared is an error; a field
//! whose type is itself declared refers to the mirrored binding.

/// Describes a type or trait that the emitter should not generate bindings for.
/// Instead, every reference to this path is either stripped entirely (for trait
/// bounds) or replaced with an arbitrary token sequence (for type positions).
#[derive(Clone, Debug)]
pub struct PathReplacement {
    /// Fully qualified path to ignore, e.g., `core::alloc::Allocator`.
    pub path: String,
    /// Optional replacement token stream emitted in place of this path.
    ///
    /// - `None` → strip the reference entirely (used for trait bounds like
    ///   `A: Allocator + Clone` becoming `A: Clone`).
    /// - `Some(replacement)` → substitute the given tokens at every occurrence
    ///   of this path in type position (e.g., replacing `Global` with `()` or
    ///   `Box<T, A>` with `MaybeUninit<u8>`).
    ///
    /// The replacement is a raw token string that will be parsed by
    /// `proc_macro2::TokenStream::from_str`.
    pub replacement: Option<String>,
}

/// A type that generated bindings reference at a canonical location in the
/// standard-library tree, but whose real definition cannot be mirrored
/// verbatim (it would not compile downstream — unstable feature gates,
/// private marker bounds, etc.).
///
/// Instead of floating as a bare name in the shared preamble, such a type is
/// *recognized at its original location*: it is registered in the type
/// registry under its canonical path so every reference routes to that exact
/// path (like any other mirrored type), and a standalone binding file is
/// emitted at that module containing a hand-written **stub body** rather than
/// the parsed source. The stub preserves just enough shape for the polyfill to
/// type-check (e.g. `Atomic<T>` as a transparent `UnsafeCell<T>` wrapper) while
/// dropping the machinery that won't compile (`AtomicPrimitive`, `T::Storage`).
///
/// The stub body must be self-contained — it may only refer to names already
/// available through the prelude's core re-exports or the language prelude.
#[derive(Clone, Debug)]
pub struct KnownExternalType {
    /// Leaf identifier of the type (used for diagnostics / ordering).
    pub name: String,
    /// Canonical path of the type relative to its library root, e.g.
    /// `"sync::atomic::Atomic"`. The pipeline registers the type at
    /// `<lib>::<path>` and emits its stub at the corresponding module.
    pub path: String,
    /// The full item definition to emit verbatim in place of the parsed source,
    /// e.g. `"#[repr(transparent)] pub struct Atomic<T>(...)".`
    pub definition: String,
}

/// Top-level loader specification, built by the consuming crate's build script.
#[derive(Clone)]
pub struct LoaderSpec {
    /// Targets (core, alloc, std) with their struct declarations.
    pub targets: Vec<BindingTarget>,
}

/// A cfg-gated struct declaration: the struct is only declared when the
/// predicate evaluates to true under the current build context.
#[derive(Debug, Clone)]
pub struct CfgGatedDecl {
    /// The struct path, e.g. `"sys::sync::mutex::futex::Futex"`.
    pub path: String,
    /// The cfg predicate string, e.g. `"any(target_os = \"linux\", target_os = \"android\")"`.
    pub predicate: String,
}

/// A single library target (e.g., "std", "core", "alloc").
#[derive(Clone)]
pub struct BindingTarget {
    /// Library name: "core", "alloc", or "std".
    pub lib_name: String,
    /// Explicitly declared structs in path syntax, relative to the library
    /// root, e.g. `"collections::btree::map::BTreeMap"`. These drive binding
    /// generation: each declaration causes the defining source file to be
    /// parsed and the struct's definition emitted. Re-export aliases of the
    /// declared struct (through its module tree) resolve to the same emitted
    /// definition.
    pub declared_structs: Vec<String>,
    /// Cfg-gated struct declarations: only active when the predicate matches
    /// the current build context. Used for platform-specific backend types
    /// (e.g., futex-only types that don't exist on pthread targets).
    pub cfg_gated_decls: Vec<CfgGatedDecl>,
    /// Paths to traits or types that the emitter should deliberately skip or
    /// replace during binding generation. Each entry specifies a fully
    /// qualified path and an optional replacement.
    ///
    /// For example, `core::alloc::Allocator` can be ignored (no replacement)
    /// so that `A: Allocator + Clone` becomes `A: Clone`. Meanwhile
    /// `alloc::alloc::Global` might be replaced with `()` since it requires
    /// the unstable `allocator_api` feature.
    pub path_replacements: Vec<PathReplacement>,
    /// Fully qualified paths of structs/enums/unions that the emitter should
    /// not emit at all. When encountered, the item is silently skipped during
    /// binding generation. Useful for types whose generated definition would
    /// fail to compile due to missing trait impls or other dependencies.
    ///
    /// Paths are relative to the library root, e.g.
    /// `"collections::btree::set::Iter"` means the `Iter` struct inside
    /// `alloc::collections::btree::set`.
    pub ignored_structs: Vec<String>,
    /// Additional derive traits to inject into the emitted definition of a
    /// declared type when the original source lacks them. Keyed by the
    /// canonical path (relative to the library root) of the type. For example,
    /// `"collections::TryReserveErrorKind": vec!["Clone"]` adds `#[derive(Clone)]`
    /// to the mirrored enum even though the std source only derives
    /// `PartialEq, Eq, Debug`.
    pub extra_derives: std::collections::HashMap<String, Vec<String>>,
    /// Types recognized at their canonical location with a hand-written stub
    /// body, because the real definition won't compile downstream. Each is
    /// registered under its canonical path so references route there, and a
    /// standalone binding file carrying the stub is emitted at that module.
    /// This replaces what was previously hardcoded as bare names in the
    /// prelude (e.g. the `Atomic<T>` polyfill now lives at
    /// `core::sync::atomic::Atomic`). See [`KnownExternalType`].
    pub known_external_types: Vec<KnownExternalType>,
}

impl LoaderSpec {
    pub fn new() -> Self {
        Self {
            targets: Vec::new(),
        }
    }

    pub fn add_target(&mut self, target: BindingTarget) {
        self.targets.push(target);
    }
}

impl Default for LoaderSpec {
    fn default() -> Self {
        Self::new()
    }
}

impl BindingTarget {
    pub fn new(lib_name: &str) -> Self {
        Self {
            lib_name: lib_name.to_string(),
            declared_structs: Vec::new(),
            cfg_gated_decls: Vec::new(),
            path_replacements: Vec::new(),
            ignored_structs: Vec::new(),
            extra_derives: std::collections::HashMap::new(),
            known_external_types: Vec::new(),
        }
    }

    /// Recognize a type at its canonical location with a hand-written stub
    /// body. The type is registered in the registry under `<lib>::<path>` so
    /// references route to that exact path, and a standalone binding file is
    /// emitted at that module carrying `definition` instead of the parsed
    /// source. For example, `core::sync::atomic::Atomic` (unstable, holds
    /// `UnsafeCell<T::Storage>` behind an `AtomicPrimitive` bound) is stubbed
    /// as a transparent `UnsafeCell<T>` wrapper rather than mirroring the real
    /// generic machinery.
    pub fn add_known_type(&mut self, path: &str, definition: &str) {
        // Leaf is the last non-empty segment; a leading `::` absolute-path
        // marker (if present) yields an empty first segment that we skip.
        let name = path
            .split("::")
            .filter(|s| !s.is_empty())
            .last()
            .unwrap_or(path)
            .to_string();
        self.known_external_types.push(KnownExternalType {
            name,
            path: path.to_string(),
            definition: definition.to_string(),
        });
    }

    /// Register an additional derive trait to inject into a declared type's
    /// emitted definition. The path is relative to the library root.
    pub fn add_derive(&mut self, path: &str, trait_name: &str) {
        self.extra_derives
            .entry(path.to_string())
            .or_default()
            .push(trait_name.to_string());
    }

    /// Declare a struct to bind by its path within the library, e.g.
    /// `target.declare_struct("collections::btree::map::BTreeMap")`. The build
    /// script locates the defining source file, emits the definition, and makes
    /// every re-export alias of the struct resolve to that definition.
    pub fn declare_struct(&mut self, path: &str) {
        self.declared_structs.push(path.to_string());
    }

    /// Declare a constant to bind by its path within the library, e.g.
    /// `target.declare_const("collections::btree::node::CAPACITY")`.
    /// Constants are located in the doc-JSON index and emitted as
    /// `pub const NAME: Type = value;`.
    pub fn declare_const(&mut self, path: &str) {
        self.declared_structs.push(path.to_string());
    }

    /// Declare a struct conditionally, gated on a cfg predicate. The
    /// declaration is only active when the predicate evaluates to true under
    /// the current build context. Used for platform-specific backend types
    /// (e.g., futex-only types that don't exist on pthread targets).
    pub fn declare_struct_cfg(&mut self, path: &str, predicate: &str) {
        self.cfg_gated_decls.push(CfgGatedDecl {
            path: path.to_string(),
            predicate: predicate.to_string(),
        });
    }

    /// Return all declared structs (both unconditional and cfg-gated).
    /// With the doc-JSON approach, the compiler has already evaluated cfgs,
    /// so all declarations in the spec are potentially active. The extraction
    /// step will naturally skip any that don't exist in the JSON output.
    pub fn declarations(&self) -> Vec<String> {
        let mut out: Vec<String> = self.declared_structs.clone();
        for g in &self.cfg_gated_decls {
            out.push(g.path.clone());
        }
        out
    }

    /// Force-ignore a struct/enum/union by its fully qualified path within the
    /// library. The emitter will skip this item entirely during binding generation.
    /// For example: `target.ignore_struct("collections::btree::set::Iter")`.
    pub fn ignore_struct(&mut self, path: &str) {
        self.ignored_structs.push(path.to_string());
    }

    /// Mark a fully qualified path as ignored with no replacement.
    /// References in trait bounds are stripped; references in type positions
    /// are also removed. Convenience wrapper for `add_path_replacement(path, None)`.
    pub fn ignore_path(&mut self, path: &str) {
        self.path_replacements.push(PathReplacement {
            path: path.to_string(),
            replacement: None,
        });
    }

    /// Mark a fully qualified path as replaced with the given token string.
    /// For example: `target.replace_path("alloc::alloc::Global", "()")` means
    /// every occurrence of `Global` in type position becomes `()`.
    pub fn replace_path(&mut self, path: &str, replacement: &str) {
        self.path_replacements.push(PathReplacement {
            path: path.to_string(),
            replacement: Some(replacement.to_string()),
        });
    }

    /// Extract the leaf identifier from each replacement path. For example,
    /// `core::alloc::Allocator` yields `"Allocator"`. These are what appear
    /// as bare identifiers in token streams during emission.
    pub fn ignored_leaf_names(&self) -> Vec<&str> {
        // Leaf is the last non-empty segment; a leading `::` absolute-path
        // marker (if present) yields an empty first segment that we skip.
        self.path_replacements
            .iter()
            .filter_map(|pr| pr.path.split("::").filter(|s| !s.is_empty()).last())
            .collect()
    }
}
