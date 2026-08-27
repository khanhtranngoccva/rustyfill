//! Lifecycle status of a module within the generated tree.

/// Lifecycle status of a module within the generated tree. Drives which files
/// get emitted and which are merely present to support import resolution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NodeStatus {
    /// Discovered and eligible for emission (the default once registered).
    #[default]
    Emittable,
    /// Registered solely to resolve imports / walk parents; not itself emitted.
    Support,
    /// A synthesized forwarding shim (`pub use <target>::Leaf;`).
    Shim,
    /// A glob-re-export alias mirroring a canonical module under another name.
    Alias,
}
