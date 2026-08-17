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

/// Top-level spec returned by [`crate::spec::get_loader_spec`].
#[derive(Clone)]
pub struct LoaderSpec {
    /// Targets (core, alloc, std) with their struct declarations.
    pub targets: Vec<BindingTarget>,
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
            path_replacements: Vec::new(),
            ignored_structs: Vec::new(),
        }
    }

    /// Declare a struct to bind by its path within the library, e.g.
    /// `target.declare_struct("collections::btree::map::BTreeMap")`. The build
    /// script locates the defining source file, emits the definition, and makes
    /// every re-export alias of the struct resolve to that definition.
    pub fn declare_struct(&mut self, path: &str) {
        self.declared_structs.push(path.to_string());
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
        self.path_replacements
            .iter()
            .map(|pr| {
                pr.path
                    .rsplit_once("::")
                    .map(|(_, leaf)| leaf)
                    .unwrap_or(pr.path.as_str())
            })
            .collect()
    }
}
