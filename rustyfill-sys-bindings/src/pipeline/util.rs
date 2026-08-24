//! Shared path/module utilities and spec-derived input builders used across
//! pipeline phases.

use std::collections::HashMap;

use crate::loader_spec::LoaderSpec;

/// Build the stable-ordered `(leaf, optional_replacement)` list consumed by
/// the emitter, from every target's `path_replacements`. Owned `String`s are
/// kept; a borrowed `&[(String, Option<&str>)]` view is derived per emission
/// call via [`replacement_view`].
pub(crate) fn build_replacement_entries(spec: &LoaderSpec) -> Vec<(String, Option<String>)> {
    let mut path_replacement_map: HashMap<String, Option<String>> = HashMap::new();
    for target in &spec.targets {
        for pr in &target.path_replacements {
            let leaf = pr
                .path
                .rsplit_once("::")
                .map(|(_, l)| l.to_string())
                .unwrap_or_else(|| pr.path.clone());
            path_replacement_map.insert(leaf, pr.replacement.clone());
        }
    }
    let mut replacement_entries: Vec<(String, Option<String>)> =
        path_replacement_map.into_iter().collect();
    replacement_entries.sort_by_key(|(k, _)| k.clone());
    replacement_entries
}

/// Borrow an owned replacement list as the `&[(String, Option<&str>)]` shape
/// expected by [`crate::emitter::EmitConfig::path_replacements`].
pub(crate) fn replacement_view(
    entries: &[(String, Option<String>)],
) -> Vec<(String, Option<&str>)> {
    entries
        .iter()
        .map(|(k, v)| (k.clone(), v.as_deref()))
        .collect()
}

/// Collect the union of path-replacement leaves and ignored-struct leaves into
/// a sorted owned list. The caller keeps this vec alive and derives a
/// `Vec<&str>` view over it (see [`super::generate`]).
pub(crate) fn collect_ignored_names(
    replacement_entries: &[(String, Option<String>)],
    ignored_structs_by_lib: &HashMap<String, Vec<String>>,
) -> Vec<String> {
    let mut all_ignored_names: std::collections::HashSet<String> =
        replacement_entries.iter().map(|(k, _)| k.clone()).collect();
    for structs in ignored_structs_by_lib.values() {
        for s in structs {
            if let Some(leaf) = s.rsplit_once("::").map(|(_, l)| l.to_string()) {
                all_ignored_names.insert(leaf);
            } else {
                all_ignored_names.insert(s.clone());
            }
        }
    }
    let mut ignored_name_vec: Vec<String> = all_ignored_names.into_iter().collect();
    ignored_name_vec.sort();
    ignored_name_vec
}

/// Compute how many module levels deep a file is under its library root.
/// e.g. "collections/btree/map.rs" -> 3 (collections / btree / map)
pub fn compute_module_depth(rel_path: &str) -> usize {
    let stem = rel_path.strip_suffix(".rs").unwrap_or(rel_path);
    let module_path = stem.strip_suffix("/mod").unwrap_or(stem);
    module_path.split('/').filter(|s| !s.is_empty()).count()
}

/// Get all sibling module names in the same parent directory.
/// For "collections/btree/node.rs", returns ["borrow", "map", "marker", ...].
pub fn get_sibling_modules(rel_path: &str, all_files: &[(String, String)]) -> Vec<String> {
    let my_stem = rel_path.strip_suffix(".rs").unwrap_or(rel_path);
    let my_module = my_stem.strip_suffix("/mod").unwrap_or(my_stem);
    let my_parent = my_module.rsplit_once('/').map(|(p, _)| p).unwrap_or("");

    let mut siblings = std::collections::HashSet::new();
    for (fp, _) in all_files {
        let stem = fp.strip_suffix(".rs").unwrap_or(fp.as_str());
        let mod_path = stem.strip_suffix("/mod").unwrap_or(stem);
        let parent = mod_path.rsplit_once('/').map(|(p, _)| p).unwrap_or("");
        if parent == my_parent {
            let name = mod_path
                .rsplit_once('/')
                .map(|(_, n)| n)
                .unwrap_or(mod_path);
            if name
                != my_module
                    .rsplit_once('/')
                    .map(|(_, n)| n)
                    .unwrap_or(my_module)
            {
                siblings.insert(name.to_string());
            }
        }
    }
    let mut result: Vec<String> = siblings.into_iter().collect();
    result.sort();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_depth_counts_segments() {
        assert_eq!(compute_module_depth("collections/btree/map.rs"), 3);
        assert_eq!(compute_module_depth("sys/pal/mod.rs"), 2);
        assert_eq!(compute_module_depth("top.rs"), 1);
        assert_eq!(compute_module_depth("a/b/c/d.rs"), 4);
    }

    #[test]
    fn module_depth_handles_mod_rs() {
        // A `mod.rs` defines the module at its own directory depth: the `/mod`
        // suffix is stripped, leaving the directory's segment count.
        assert_eq!(compute_module_depth("sys/pal/unix/mod.rs"), 3);
        // Root-level `mod.rs` strips to "" but still counts as one module level.
        assert_eq!(compute_module_depth("mod.rs"), 1);
    }

    #[test]
    fn siblings_are_same_directory_peers_only() {
        let all_files: Vec<(String, String)> = vec![
            ("collections/btree/map.rs".into(), "core".into()),
            ("collections/btree/set.rs".into(), "core".into()),
            ("collections/btree/node.rs".into(), "core".into()),
            ("collections/hashbrown/raw.rs".into(), "core".into()), // different dir
            ("other/top.rs".into(), "core".into()),                 // different dir
        ];
        let got = get_sibling_modules("collections/btree/node.rs", &all_files);
        assert_eq!(got, vec!["map", "set"]);
    }

    #[test]
    fn siblings_excludes_self_and_top_level_isolated() {
        let all_files: Vec<(String, String)> = vec![
            ("alpha.rs".into(), "core".into()),
            ("beta.rs".into(), "core".into()),
        ];
        // Top-level: parent is "" for both, so they ARE siblings.
        assert_eq!(get_sibling_modules("alpha.rs", &all_files), vec!["beta"]);
    }
}
