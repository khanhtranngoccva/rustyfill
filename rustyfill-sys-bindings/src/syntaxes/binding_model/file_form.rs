//! How a module's content lives on disk.

use super::super::ModulePath;

/// How a module's content lives on disk. Encodes the `foo/mod.rs` versus
/// `foo.rs` distinction the resolver previously tracked in two separate maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileForm {
    /// A directory module defined by `<path>/mod.rs`.
    Dir,
    /// A single-file module defined by `<path>.rs`.
    Leaf,
}

impl FileForm {
    /// The relative file path for a module at `module_path` in this form.
    pub fn rel_path(&self, module_path: &ModulePath) -> String {
        let slash = module_path.to_slash();
        match self {
            FileForm::Dir => {
                if slash.is_empty() {
                    "mod.rs".to_string()
                } else {
                    format!("{slash}/mod.rs")
                }
            }
            FileForm::Leaf => format!("{slash}.rs"),
        }
    }

    /// True when `rel_path` (a `.rs` file path) denotes this form at
    /// `module_path`.
    pub fn matches_rel_path(&self, module_path: &ModulePath, rel_path: &str) -> bool {
        self.rel_path(module_path) == rel_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_form_renders_correct_paths() {
        let mp = ModulePath::from_slash("sys/pal/unix/sync").unwrap();
        assert_eq!(FileForm::Dir.rel_path(&mp), "sys/pal/unix/sync/mod.rs");
        assert_eq!(FileForm::Leaf.rel_path(&mp), "sys/pal/unix/sync.rs");
        let root = ModulePath::root();
        assert_eq!(FileForm::Dir.rel_path(&root), "mod.rs");
        assert_eq!(FileForm::Leaf.rel_path(&root), ".rs");
    }
}
