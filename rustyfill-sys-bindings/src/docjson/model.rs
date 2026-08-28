//! The intermediate, optimized representation of extracted types.
//!
//! This is the conversion target for the typed wire model ([`super::wire`]):
//! every [`wire::Crate`] blob is reduced to two compact tables —
//!
//! - the **type table** ([`TypeTable`]), one entry per declared type, holding
//!   everything needed to emit its definition (kind, fields, variants,
//!   generics, attributes), and
//! - the **export table** ([`ExportTable`]), a flat item-id → routed-path map
//!   that answers "where does this reference point in the emitted tree?".
//!
//! Type *references* are not flattened into strings here: they stay as the
//! wire [`wire::Type`] values, which carry the authoritative `Path.id` of the
//! referenced item. Resolution against the export table happens at render
//! time (see [`super::emit`]), so routing decisions (mirror tree vs builtin
//! extern) are applied exactly once, at emission.

use std::collections::HashMap;

use super::wire::{self, Item, ItemEnum};

// ── Tables ────────────────────────────────────────────────────────────────────

/// One entry per successfully located declaration.
#[derive(Clone, Debug)]
pub struct DocType {
    /// Position in the [`TypeTable`]. Assigned by the builder.
    pub id: usize,
    /// The item id within the owning library's doc-JSON namespace; the key
    /// used to look this type up in the [`ExportTable`].
    pub item_id: u32,
    /// The library this type belongs to ("core", "alloc", "std").
    pub lib: String,
    /// The leaf name of the type (e.g., "BTreeMap").
    pub name: String,
    /// Full module path relative to the library root (e.g., "collections::btree::map").
    pub module_path: String,
    /// The structural kind of the definition.
    pub kind: DocTypeKind,
    /// Generic parameters.
    pub generics: Vec<DocGenericParam>,
    /// Where-clause predicates rendered as source strings.
    pub where_predicates: Vec<String>,
    /// Repr attributes (e.g., ["C"], ["transparent"], ["packed(2)"]).
    pub repr_attrs: Vec<String>,
    /// Derive macros (e.g., [["Clone", "Debug"]]).
    pub derive_attrs: Vec<Vec<String>>,
    /// Other attributes preserved for emission.
    pub other_attrs: Vec<String>,
}

impl DocType {
    /// Canonical path within the emitting crate's mirror tree, e.g.
    /// `crate::std::sync::poison::mutex::Mutex`.
    pub fn mirror_path(&self) -> String {
        let rel = if self.module_path.is_empty() {
            self.name.clone()
        } else {
            format!("{}::{}", self.module_path, self.name)
        };
        format!("crate::std::{rel}")
    }

    /// Canonical path in the original library, e.g. `core::cell::UnsafeCell`.
    pub fn canonical_lib_path(&self) -> String {
        if self.module_path.is_empty() {
            format!("{}::{}", self.lib, self.name)
        } else {
            format!("{}::{}::{}", self.lib, self.module_path, self.name)
        }
    }
}

/// The type table: all declared types, deduplicated by (lib, canonical path).
#[derive(Default)]
pub struct TypeTable {
    pub entries: Vec<DocType>,
}

impl TypeTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, entry: DocType) {
        self.entries.push(entry);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// A single row of the export table: how one item id routes to an absolute
/// path in the emitted world.
#[derive(Clone, Debug)]
pub struct ExportEntry {
    /// The originating library's name ("core", "alloc", "std").
    pub lib: String,
    /// The item id within that library's doc-JSON namespace.
    pub item_id: u32,
    /// The fully qualified destination path, ready to be spliced into source:
    /// `crate::std::<module>::<Name>` for mirrored items, or
    /// `::__rustyfill_builtin_<lib>::<rest>` for unbuiltins.
    pub route: String,
}

/// The export table: flat, per-library, item-id → routed absolute path.
///
/// Built once from the wire data; consulted on every resolved-path reference
/// during rendering. Within a single library's JSON, `item_id` alone uniquely
/// identifies a paths-table entry, so the lookup key is `(lib, item_id)`.
#[derive(Default)]
pub struct ExportTable {
    pub entries: Vec<ExportEntry>,
    /// Fast index: (lib, item_id) → position in `entries`.
    index: HashMap<(String, u32), usize>,
}

impl ExportTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, entry: ExportEntry) {
        let pos = self.entries.len();
        self.index.insert((entry.lib.clone(), entry.item_id), pos);
        self.entries.push(entry);
    }

    /// Resolve an item id (within `lib`) to its exported route.
    pub fn resolve(&self, lib: &str, item_id: u32) -> Option<&str> {
        let pos = *self.index.get(&(lib.to_string(), item_id))?;
        Some(self.entries[pos].route.as_str())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ── Structural kinds ──────────────────────────────────────────────────────────

/// The structural kind of a type definition.
#[derive(Clone, Debug)]
pub enum DocTypeKind {
    Struct {
        fields: Vec<DocField>,
        tuple: bool,
    },
    Enum {
        variants: Vec<DocVariant>,
    },
    Union {
        fields: Vec<DocField>,
    },
    TypeAlias {
        rhs: wire::Type,
    },
    /// A constant item: `const NAME: Type = value;`.
    Constant {
        /// The type annotation (e.g., "usize").
        ty: String,
        /// The evaluated constant expression/value from rustdoc.
        value: String,
    },
}

/// A single field in a struct, union, or variant.
#[derive(Clone, Debug)]
pub struct DocField {
    pub name: String,
    pub visibility: DocVisibility,
    pub ty: wire::Type,
}

/// A variant in an enum.
#[derive(Clone, Debug)]
pub struct DocVariant {
    pub name: String,
    pub kind: DocVariantKind,
    /// The discriminant expression (e.g., "1", "(1 << 4)").
    pub discriminant_expr: Option<String>,
}

/// The shape of an enum variant.
#[derive(Clone, Debug)]
pub enum DocVariantKind {
    /// Unit variant: `None`
    Unit,
    /// Tuple variant: `Some(T)` — fields are positional.
    Tuple(Vec<DocField>),
    /// Struct variant: `Point { x: i32, y: i32 }` — fields are named.
    Struct(Vec<DocField>),
}

/// A generic parameter declaration.
#[derive(Clone, Debug)]
pub struct DocGenericParam {
    pub name: String,
    /// Whether this is a lifetime parameter.
    pub is_lifetime: bool,
    /// Whether this is a const parameter.
    pub is_const: bool,
    /// Trait bounds (rendered as source strings).
    pub bounds: Vec<String>,
    /// Default type for a type parameter.
    pub default_type: Option<wire::Type>,
    /// Default value expression for a const parameter (e.g., "42").
    pub default_value: Option<String>,
    /// The type of a const parameter (e.g., "usize").
    pub const_ty: Option<String>,
    /// Lifetime outlives constraints.
    pub outlives: Vec<String>,
}

/// Field/item visibility as extracted from the JSON.
#[derive(Clone, Debug, PartialEq)]
pub enum DocVisibility {
    Public,
    Crate,
    Restricted(String),
    Private,
}

// ── Conversion: wire → IR ─────────────────────────────────────────────────────

impl DocType {
    /// Convert a wire [`Item`] into our model.
    ///
    /// `index` is the owning crate's item index, used to chase field/variant
    /// ids. Returns `Err` when the item is not a bindable type.
    pub fn from_item(item: &Item, index: &HashMap<wire::Id, Item>) -> Result<Self, String> {
        let name = item
            .name
            .clone()
            .ok_or_else(|| format!("item {:?} has no name", item.id))?;

        let (repr_attrs, derive_attrs, other_attrs) = parse_attributes(&item.attrs);

        match &item.inner {
            ItemEnum::Struct(s) => {
                let (fields, is_tuple) = parse_struct_kind(&s.kind, index, &name)?;
                let (generics, where_predicates) = parse_generics(&s.generics)?;
                Ok(Self {
                    id: 0, // filled in by the builder
                    item_id: 0, // filled in by the builder
                    lib: String::new(),
                    name,
                    module_path: String::new(),
                    kind: DocTypeKind::Struct {
                        fields,
                        tuple: is_tuple,
                    },
                    generics,
                    where_predicates,
                    repr_attrs,
                    derive_attrs,
                    other_attrs,
                })
            }

            ItemEnum::Enum(e) => {
                let variants = e
                    .variants
                    .iter()
                    .map(|vid| {
                        let var_item = index
                            .get(vid)
                            .ok_or_else(|| format!("variant id {:?} not in index", vid))?;
                        DocVariant::from_item(var_item, index)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let (generics, where_predicates) = parse_generics(&e.generics)?;
                Ok(Self {
                    id: 0,
                    item_id: 0, // filled in by the builder
                    lib: String::new(),
                    name,
                    module_path: String::new(),
                    kind: DocTypeKind::Enum { variants },
                    generics,
                    where_predicates,
                    repr_attrs,
                    derive_attrs,
                    other_attrs,
                })
            }

            ItemEnum::Union(u) => {
                let fields = u
                    .fields
                    .iter()
                    .map(|fid| {
                        let field_item = index
                            .get(fid)
                            .ok_or_else(|| format!("field id {:?} not in index", fid))?;
                        DocField::from_item(field_item)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let (generics, where_predicates) = parse_generics(&u.generics)?;
                Ok(Self {
                    id: 0,
                    item_id: 0, // filled in by the builder
                    lib: String::new(),
                    name,
                    module_path: String::new(),
                    kind: DocTypeKind::Union { fields },
                    generics,
                    where_predicates,
                    repr_attrs,
                    derive_attrs,
                    other_attrs,
                })
            }

            ItemEnum::TypeAlias(a) => {
                let (generics, where_predicates) = parse_generics(&a.generics)?;
                Ok(Self {
                    id: 0,
                    item_id: 0, // filled in by the builder
                    lib: String::new(),
                    name,
                    module_path: String::new(),
                    kind: DocTypeKind::TypeAlias { rhs: a.type_.clone() },
                    generics,
                    where_predicates,
                    repr_attrs,
                    derive_attrs,
                    other_attrs,
                })
            }

            ItemEnum::Constant { type_, const_ } => {
                let ty_val = render_type_plain(type_);
                let value = const_.value.clone().unwrap_or_else(|| const_.expr.clone());
                Ok(Self {
                    id: 0,
                    item_id: 0, // filled in by the builder
                    lib: String::new(),
                    name,
                    module_path: String::new(),
                    kind: DocTypeKind::Constant { ty: ty_val, value },
                    generics: Vec::new(),
                    where_predicates: Vec::new(),
                    repr_attrs,
                    derive_attrs,
                    other_attrs,
                })
            }

            _ => Err(format!(
                "unsupported item kind '{:?}' for type {}",
                item.inner.item_kind(),
                name
            )),
        }
    }
}

impl DocVariant {
    fn from_item(item: &Item, index: &HashMap<wire::Id, Item>) -> Result<Self, String> {
        let name = item
            .name
            .clone()
            .ok_or_else(|| format!("variant {:?} has no name", item.id))?;

        let wire::ItemEnum::Variant(v) = &item.inner else {
            return Err("inner is not variant".to_string());
        };

        let discriminant_expr = v.discriminant.as_ref().map(|d| d.expr.clone());
        let kind = parse_variant_kind(&v.kind, index, &name)?;

        Ok(Self {
            name,
            kind,
            discriminant_expr,
        })
    }
}

impl DocField {
    fn from_item(item: &Item) -> Result<Self, String> {
        let name = item
            .name
            .clone()
            .ok_or_else(|| format!("field {:?} has no name", item.id))?;

        let visibility = match &item.visibility {
            wire::Visibility::Public => DocVisibility::Public,
            wire::Visibility::Crate => DocVisibility::Crate,
            wire::Visibility::Restricted { path, .. } => DocVisibility::Restricted(path.clone()),
            wire::Visibility::Default => DocVisibility::Private,
        };

        let wire::ItemEnum::StructField(ty) = &item.inner else {
            return Err("inner is not struct_field".to_string());
        };

        Ok(Self {
            name,
            visibility,
            ty: ty.clone(),
        })
    }
}

// ── Parsing helpers ───────────────────────────────────────────────────────────

fn parse_struct_kind(
    kind: &wire::StructKind,
    index: &HashMap<wire::Id, Item>,
    type_name: &str,
) -> Result<(Vec<DocField>, bool), String> {
    match kind {
        wire::StructKind::Unit => Ok((Vec::new(), false)),

        wire::StructKind::Plain { fields, .. } => {
            let fields = fields
                .iter()
                .map(|fid| {
                    let field_item = index.get(fid).ok_or_else(|| {
                        format!("field id {:?} not in index (struct {})", fid, type_name)
                    })?;
                    DocField::from_item(field_item)
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok((fields, false))
        }

        wire::StructKind::Tuple(entries) => {
            let mut fields = Vec::new();
            for (i, entry) in entries.iter().enumerate() {
                let Some(fid) = entry else {
                    return Err(format!(
                        "stripped tuple field at position {} in {}",
                        i, type_name
                    ));
                };
                let field_item = index.get(fid).ok_or_else(|| {
                    format!("tuple field id {:?} not in index (struct {})", fid, type_name)
                })?;
                fields.push(DocField::from_item(field_item)?);
            }
            Ok((fields, true))
        }
    }
}

fn parse_variant_kind(
    kind: &wire::VariantKind,
    index: &HashMap<wire::Id, Item>,
    variant_name: &str,
) -> Result<DocVariantKind, String> {
    match kind {
        wire::VariantKind::Plain => Ok(DocVariantKind::Unit),

        wire::VariantKind::Tuple(entries) => {
            let mut fields = Vec::new();
            for (i, entry) in entries.iter().enumerate() {
                let Some(fid) = entry else {
                    return Err(format!(
                        "stripped tuple field at position {} in variant {}",
                        i, variant_name
                    ));
                };
                let field_item = index
                    .get(fid)
                    .ok_or_else(|| format!("variant field id {:?} not in index", fid))?;
                fields.push(DocField::from_item(field_item)?);
            }
            Ok(DocVariantKind::Tuple(fields))
        }

        wire::VariantKind::Struct { fields, .. } => {
            let fields = fields
                .iter()
                .map(|fid| {
                    let field_item = index
                        .get(fid)
                        .ok_or_else(|| format!("variant field id {:?} not in index", fid))?;
                    DocField::from_item(field_item)
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(DocVariantKind::Struct(fields))
        }
    }
}

fn parse_generics(generics: &wire::Generics) -> Result<(Vec<DocGenericParam>, Vec<String>), String> {
    let mut params = Vec::new();
    for p in &generics.params {
        let param = match &p.kind {
            wire::GenericParamDefKind::Lifetime { outlives } => DocGenericParam {
                name: p.name.clone(),
                is_lifetime: true,
                is_const: false,
                bounds: Vec::new(),
                default_type: None,
                default_value: None,
                const_ty: None,
                outlives: outlives.clone(),
            },
            wire::GenericParamDefKind::Type { bounds, default, .. } => DocGenericParam {
                name: p.name.clone(),
                is_lifetime: false,
                is_const: false,
                bounds: bounds
                    .iter()
                    .map(render_generic_bound)
                    .collect::<Result<Vec<_>, _>>()?,
                default_type: default.clone(),
                default_value: None,
                const_ty: None,
                outlives: Vec::new(),
            },
            wire::GenericParamDefKind::Const { type_, default } => DocGenericParam {
                name: p.name.clone(),
                is_lifetime: false,
                is_const: true,
                bounds: Vec::new(),
                default_type: None,
                default_value: default.clone(),
                const_ty: Some(render_type_plain(type_)),
                outlives: Vec::new(),
            },
        };
        params.push(param);
    }

    let where_predicates = generics
        .where_predicates
        .iter()
        .map(render_where_predicate)
        .collect::<Result<Vec<_>, _>>()?;

    Ok((params, where_predicates))
}

fn render_generic_bound(bound: &wire::GenericBound) -> Result<String, String> {
    match bound {
        wire::GenericBound::TraitBound { trait_, .. } => {
            Ok(trait_.path.clone())
        }
        wire::GenericBound::Outlives(lt) => Ok(lt.clone()),
        wire::GenericBound::Use(args) => {
            let rendered: Vec<String> = args
                .iter()
                .map(|a| match a {
                    wire::PreciseCapturingArg::Lifetime(lt) => lt.clone(),
                    wire::PreciseCapturingArg::Param(p) => p.clone(),
                })
                .collect();
            Ok(format!("use<{}>", rendered.join(", ")))
        }
    }
}

fn render_where_predicate(pred: &wire::WherePredicate) -> Result<String, String> {
    match pred {
        wire::WherePredicate::BoundPredicate { type_, bounds, .. } => {
            let ty_src = render_type_plain(type_);
            let bounds: Vec<String> =
                bounds.iter().map(render_generic_bound).collect::<Result<Vec<_>, _>>()?;
            Ok(format!("{}: {}", ty_src, bounds.join(" + ")))
        }
        wire::WherePredicate::LifetimePredicate { lifetime, outlives } => {
            Ok(format!("{}: {}", lifetime, outlives.join(" + ")))
        }
        wire::WherePredicate::EqPredicate { lhs, rhs } => {
            let lhs_src = render_type_plain(lhs);
            let rhs_src = match rhs {
                wire::Term::Type(t) => render_type_plain(t),
                wire::Term::Constant(c) => c.expr.clone(),
            };
            Ok(format!("{} = {}", lhs_src, rhs_src))
        }
    }
}

/// Render a wire type as source WITHOUT routing (no id resolution).
///
/// Used only for contexts where references cannot occur (const annotations,
/// const-param types, where-predicate subjects) or where the raw spelling is
/// acceptable. Routing-aware rendering lives in [`super::emit`].
fn render_type_plain(ty: &wire::Type) -> String {
    super::type_repr::render(ty, &|path: &str, _: u32| path.to_string())
}

/// Parse the `attrs` array from a wire item.
///
/// In older format versions attrs are raw strings (`"#[repr(C)]"`). Newer
/// formats emit structured attribute objects; those are skipped here (the
/// pipeline targets the string form) rather than misparsed.
fn parse_attributes(attrs: &[String]) -> (Vec<String>, Vec<Vec<String>>, Vec<String>) {
    let mut repr_attrs = Vec::new();
    let mut derive_attrs = Vec::new();
    let mut other_attrs = Vec::new();

    for s in attrs {
        classify_other_attr(s, &mut repr_attrs, &mut derive_attrs, &mut other_attrs);
    }

    (repr_attrs, derive_attrs, other_attrs)
}

fn classify_other_attr(
    s: &str,
    repr_attrs: &mut Vec<String>,
    derive_attrs: &mut Vec<Vec<String>>,
    other_attrs: &mut Vec<String>,
) {
    // Strip leading '#' and surrounding brackets
    let cleaned = s.trim_start_matches('#').trim();
    let inner = cleaned
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(cleaned);

    if let Some(repr_part) = inner.strip_prefix("repr(") {
        let repr_inner = repr_part.strip_suffix(')').unwrap_or(repr_part);
        repr_attrs.push(repr_inner.to_string());
    } else if let Some(derive_part) = inner.strip_prefix("derive(") {
        let derive_inner = derive_part.strip_suffix(')').unwrap_or(derive_part);
        let derives: Vec<String> = derive_inner
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        derive_attrs.push(derives);
    } else if inner.starts_with("lang") || inner.starts_with("attr") {
        // Skip lang items and internal stability attrs — not relevant for binding emission
    } else {
        other_attrs.push(format!("#[{}]", inner));
    }
}
