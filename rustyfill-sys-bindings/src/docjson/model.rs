//! Data model for types extracted from cargo-doc JSON output.
//!
//! This is the authoritative representation of a type's structure, derived
//! entirely from the compiler's own analysis. No source parsing involved.
//!
//! Parsing uses `serde_json::Value` directly to remain compatible across
//! rustdoc format versions (the schema evolves with each toolchain release).

use serde_json::Value;

use super::type_repr::TypeRepr;

/// A type definition extracted from doc-JSON. Represents a struct, enum, union,
/// or type alias with full field/variant information and resolved type references.
#[derive(Clone, Debug)]
pub struct DocType {
    /// The library this type belongs to ("core", "alloc", "std").
    pub lib: String,
    /// The leaf name of the type (e.g., "BTreeMap").
    pub name: String,
    /// Full module path relative to the library root (e.g., "collections::btree::map").
    pub module_path: String,
    /// The kind of type definition.
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
    /// Convert a doc-JSON index entry into our model.
    pub fn from_json(
        item: &Value,
        index: &serde_json::Map<String, Value>,
        lib_name: &str,
    ) -> Result<Self, String> {
        let name = item
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or("item has no name")?
            .to_string();

        let inner = item.get("inner").ok_or("item has no 'inner'")?;
        let attrs_raw = item
            .get("attrs")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let (repr_attrs, derive_attrs, other_attrs) = parse_attributes(&attrs_raw);

        // Determine the item kind from the first key in "inner"
        let kind_tag = inner
            .as_object()
            .and_then(|o| o.keys().next())
            .ok_or("empty inner object")?
            .clone();

        match kind_tag.as_str() {
            "struct" => {
                let struct_data = inner.get("struct").ok_or("missing struct data")?;
                let kind_val = struct_data.get("kind").ok_or("struct missing kind")?;

                let (fields, is_tuple) = parse_struct_kind(kind_val, index, &name)?;

                let (generics, where_predicates) = parse_generics(struct_data.get("generics"))?;

                Ok(Self {
                    lib: lib_name.to_string(),
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

            "enum" => {
                let enum_data = inner.get("enum").ok_or("missing enum data")?;
                let variant_ids: Vec<u64> = enum_data
                    .get("variants")
                    .and_then(|v| v.as_array())
                    .ok_or("enum missing variants")?
                    .iter()
                    .filter_map(|id| id.as_u64())
                    .collect();

                let variants = variant_ids
                    .iter()
                    .map(|&vid| {
                        let var_item = index
                            .get(&vid.to_string())
                            .ok_or_else(|| format!("variant id {} not in index", vid))?;
                        DocVariant::from_json(var_item, index)
                    })
                    .collect::<Result<Vec<_>, String>>()?;

                let (generics, where_predicates) = parse_generics(enum_data.get("generics"))?;

                Ok(Self {
                    lib: lib_name.to_string(),
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

            "union" => {
                let union_data = inner.get("union").ok_or("missing union data")?;
                let field_ids: Vec<u64> = union_data
                    .get("fields")
                    .and_then(|v| v.as_array())
                    .ok_or("union missing fields")?
                    .iter()
                    .filter_map(|id| id.as_u64())
                    .collect();

                let fields = field_ids
                    .iter()
                    .map(|&fid| {
                        let field_item = index
                            .get(&fid.to_string())
                            .ok_or_else(|| format!("field id {} not in index", fid))?;
                        DocField::from_json(field_item)
                    })
                    .collect::<Result<Vec<_>, String>>()?;

                let (generics, where_predicates) = parse_generics(union_data.get("generics"))?;

                Ok(Self {
                    lib: lib_name.to_string(),
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

            "type_alias" => {
                let alias_data = inner.get("type_alias").ok_or("missing alias data")?;
                let ty_val = alias_data.get("type").ok_or("alias missing type")?;
                let rhs = TypeRepr::from_json(ty_val)?;

                let (generics, where_predicates) = parse_generics(alias_data.get("generics"))?;

                Ok(Self {
                    lib: lib_name.to_string(),
                    name,
                    module_path: String::new(),
                    kind: DocTypeKind::TypeAlias { rhs },
                    generics,
                    where_predicates,
                    repr_attrs,
                    derive_attrs,
                    other_attrs,
                })
            }

            "constant" => {
                let const_data = inner.get("constant").ok_or("missing constant data")?;
                let ty_val = if let Some(primitive) = const_data
                    .get("type")
                    .and_then(|t| t.get("primitive"))
                    .and_then(|p| p.as_str())
                {
                    primitive.to_string()
                } else {
                    // Fallback: try to render the type from its JSON form.
                    const_data
                        .get("type")
                        .and_then(|tv| TypeRepr::from_json(tv).ok())
                        .map(|t| t.to_source())
                        .unwrap_or_else(|| "?".to_string())
                };
                let value = const_data
                    .get("const")
                    .and_then(|c| c.get("value"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| {
                        const_data
                            .get("const")
                            .and_then(|c| c.get("expr"))
                            .and_then(|e| e.as_str())
                            .map(|s| s.to_string())
                    })
                    .unwrap_or_default();

                Ok(Self {
                    lib: lib_name.to_string(),
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
                "unsupported item kind '{}' for type {}",
                kind_tag, name
            )),
        }
    }

    /// Set the module path after location is determined.
    pub fn set_module_path(&mut self, path: String) {
        self.module_path = path;
    }
}

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
        rhs: TypeRepr,
    },
    /// A constant item: `const NAME: Type = value;`.
    Constant {
        /// The type annotation (e.g., "usize").
        ty: String,
        /// The evaluated constant expression/value from rustdoc.
        value: String,
    },
}

/// A single field in a struct or union.
#[derive(Clone, Debug)]
pub struct DocField {
    pub name: String,
    pub visibility: DocVisibility,
    pub ty: TypeRepr,
}

impl DocField {
    fn from_json(item: &Value) -> Result<Self, String> {
        let name = item
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or("field has no name")?
            .to_string();

        let vis_raw = item.get("visibility").ok_or("field has no visibility")?;
        let visibility = parse_visibility(vis_raw);

        // In format v57+: inner.struct_field IS the type directly
        // In older formats: inner.struct_field might be {"type": <Type>}
        let inner = item.get("inner").ok_or("field has no inner")?;
        let sf_val = inner
            .get("struct_field")
            .ok_or("inner is not struct_field")?;

        let ty = TypeRepr::from_json(sf_val)?;

        Ok(Self {
            name,
            visibility,
            ty,
        })
    }
}

/// A variant in an enum.
#[derive(Clone, Debug)]
pub struct DocVariant {
    pub name: String,
    pub kind: DocVariantKind,
    /// The discriminant expression (e.g., "1", "(1 << 4)").
    pub discriminant_expr: Option<String>,
}

impl DocVariant {
    fn from_json(item: &Value, index: &serde_json::Map<String, Value>) -> Result<Self, String> {
        let name = item
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or("variant has no name")?
            .to_string();

        let inner = item.get("inner").ok_or("variant has no inner")?;
        let var = inner.get("variant").ok_or("inner is not variant")?;

        let kind_val = var.get("kind").ok_or("variant has no kind")?;

        // Discriminant: null | {"expr": "...", "value": "..."}
        let discriminant_expr = var
            .get("discriminant")
            .filter(|d| !d.is_null())
            .and_then(|d| d.get("expr"))
            .and_then(|e| e.as_str())
            .map(String::from);

        let kind = parse_variant_kind(kind_val, index, &name)?;

        Ok(Self {
            name,
            kind,
            discriminant_expr,
        })
    }
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
    /// Default type for a type parameter (preserves ID for routing).
    pub default_type: Option<TypeRepr>,
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

// ── Parsing helpers ───────────────────────────────────────────────────────────

fn parse_struct_kind(
    kind_val: &Value,
    index: &serde_json::Map<String, Value>,
    type_name: &str,
) -> Result<(Vec<DocField>, bool), String> {
    match kind_val {
        // Unit struct: just the string "unit"
        Value::String(s) if s == "unit" => Ok((Vec::new(), false)),

        // Plain (named fields): {"plain": {"fields": [ids], "has_stripped_fields": bool}}
        Value::Object(obj) if obj.contains_key("plain") => {
            let plain = &obj["plain"];
            let field_ids: Vec<u64> = plain
                .get("fields")
                .and_then(|v| v.as_array())
                .ok_or("plain struct missing fields")?
                .iter()
                .filter_map(|id| id.as_u64())
                .collect();

            let fields = field_ids
                .iter()
                .map(|&fid| {
                    let field_item = index.get(&fid.to_string()).ok_or_else(|| {
                        format!("field id {} not in index (struct {})", fid, type_name)
                    })?;
                    DocField::from_json(field_item)
                })
                .collect::<Result<Vec<_>, String>>()?;

            Ok((fields, false))
        }

        // Tuple: {"tuple": [id_or_null, ...]}
        Value::Object(obj) if obj.contains_key("tuple") => {
            let tuple_arr = obj
                .get("tuple")
                .and_then(|v| v.as_array())
                .ok_or("tuple struct missing fields array")?;

            let mut fields = Vec::new();
            for (i, entry) in tuple_arr.iter().enumerate() {
                if entry.is_null() {
                    // Stripped field (shouldn't happen with --document-private-items)
                    return Err(format!(
                        "stripped tuple field at position {} in {}",
                        i, type_name
                    ));
                }
                let fid = entry
                    .as_u64()
                    .ok_or_else(|| format!("invalid tuple field id at position {}", i))?;
                let field_item = index.get(&fid.to_string()).ok_or_else(|| {
                    format!("tuple field id {} not in index (struct {})", fid, type_name)
                })?;
                fields.push(DocField::from_json(field_item)?);
            }

            Ok((fields, true))
        }

        other => Err(format!(
            "unrecognized struct kind for {}: {}",
            type_name,
            short_val(other)
        )),
    }
}

fn parse_variant_kind(
    kind_val: &Value,
    index: &serde_json::Map<String, Value>,
    variant_name: &str,
) -> Result<DocVariantKind, String> {
    match kind_val {
        // Unit variant: the string "plain" (counterintuitively, "plain" means no fields)
        Value::String(s) if s == "plain" || s == "unit" => Ok(DocVariantKind::Unit),

        // Tuple variant: {"tuple": [id_or_null, ...]}
        Value::Object(obj) if obj.contains_key("tuple") => {
            let tuple_arr = obj
                .get("tuple")
                .and_then(|v| v.as_array())
                .ok_or("tuple variant missing fields array")?;

            let mut fields = Vec::new();
            for (i, entry) in tuple_arr.iter().enumerate() {
                if entry.is_null() {
                    return Err(format!(
                        "stripped tuple field at position {} in variant {}",
                        i, variant_name
                    ));
                }
                let fid = entry
                    .as_u64()
                    .ok_or_else(|| format!("invalid variant field id at position {}", i))?;
                let field_item = index
                    .get(&fid.to_string())
                    .ok_or_else(|| format!("variant field id {} not in index", fid))?;
                fields.push(DocField::from_json(field_item)?);
            }

            Ok(DocVariantKind::Tuple(fields))
        }

        // Struct variant: {"struct": {"fields": [ids], "has_stripped_fields": bool}}
        Value::Object(obj) if obj.contains_key("struct") => {
            let struct_data = &obj["struct"];
            let field_ids: Vec<u64> = struct_data
                .get("fields")
                .and_then(|v| v.as_array())
                .ok_or("struct variant missing fields")?
                .iter()
                .filter_map(|id| id.as_u64())
                .collect();

            let fields = field_ids
                .iter()
                .map(|&fid| {
                    let field_item = index
                        .get(&fid.to_string())
                        .ok_or_else(|| format!("variant field id {} not in index", fid))?;
                    DocField::from_json(field_item)
                })
                .collect::<Result<Vec<_>, String>>()?;

            Ok(DocVariantKind::Struct(fields))
        }

        // Fallback: "plain" as an object (some versions encode it differently)
        Value::Object(obj) if obj.contains_key("plain") => {
            // Could be a struct-like variant encoded as "plain" in some versions
            let plain = &obj["plain"];
            if let Some(field_ids) = plain.get("fields").and_then(|v| v.as_array()) {
                let ids: Vec<u64> = field_ids.iter().filter_map(|id| id.as_u64()).collect();
                if ids.is_empty() {
                    Ok(DocVariantKind::Unit)
                } else {
                    let fields = ids
                        .iter()
                        .map(|&fid| {
                            let field_item = index
                                .get(&fid.to_string())
                                .ok_or_else(|| format!("variant field id {} not in index", fid))?;
                            DocField::from_json(field_item)
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    Ok(DocVariantKind::Struct(fields))
                }
            } else {
                Ok(DocVariantKind::Unit)
            }
        }

        other => Err(format!(
            "unrecognized variant kind for '{}': {}",
            variant_name,
            short_val(other)
        )),
    }
}

fn parse_generics(val: Option<&Value>) -> Result<(Vec<DocGenericParam>, Vec<String>), String> {
    let Some(generics) = val else {
        return Ok((Vec::new(), Vec::new()));
    };

    let params_arr = generics
        .get("params")
        .and_then(|v| v.as_array())
        .ok_or("generics missing params")?;

    let mut params = Vec::new();
    for p in params_arr {
        let name = p
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or("generic param missing name")?
            .to_string();

        let kind_obj = p.get("kind").ok_or("generic param missing kind")?;

        let param = if let Some(lt) = kind_obj.get("lifetime") {
            let outlives: Vec<String> = lt
                .get("outlives")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|s| s.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            DocGenericParam {
                name,
                is_lifetime: true,
                is_const: false,
                bounds: Vec::new(),
                default_type: None,
                default_value: None,
                const_ty: None,
                outlives,
            }
        } else if let Some(tp) = kind_obj.get("type") {
            let bounds: Vec<String> = tp
                .get("bounds")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(render_generic_bound)
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default();

            let default_type = tp
                .get("default")
                .filter(|d| !d.is_null())
                .map(TypeRepr::from_json)
                .transpose()?;

            DocGenericParam {
                name,
                is_lifetime: false,
                is_const: false,
                bounds,
                default_type,
                default_value: None,
                const_ty: None,
                outlives: Vec::new(),
            }
        } else if let Some(cp) = kind_obj.get("const") {
            let const_ty = cp
                .get("type")
                .map(TypeRepr::from_json)
                .transpose()?
                .map(|t| t.to_source());
            let default_value = cp.get("default").and_then(|v| v.as_str()).map(String::from);

            DocGenericParam {
                name,
                is_lifetime: false,
                is_const: true,
                bounds: Vec::new(),
                default_type: None,
                default_value,
                const_ty,
                outlives: Vec::new(),
            }
        } else {
            return Err(format!(
                "unrecognized generic param kind for '{}': {}",
                name,
                short_val(kind_obj)
            ));
        };

        params.push(param);
    }

    // Where predicates
    let where_predicates: Vec<String> = generics
        .get("where_predicates")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(render_where_predicate)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    Ok((params, where_predicates))
}

fn render_generic_bound(val: &Value) -> Result<String, String> {
    let obj = val
        .as_object()
        .ok_or_else(|| format!("expected bound object, got: {}", short_val(val)))?;

    let (tag, payload) = obj.iter().next().ok_or("empty bound")?;

    match tag.as_str() {
        "trait_bound" => {
            let trait_path = payload
                .get("trait")
                .and_then(|t| t.get("path"))
                .or_else(|| payload.get("trait").and_then(|t| t.get("name")))
                .and_then(|v| v.as_str())
                .ok_or("trait_bound missing path")?;
            Ok(trait_path.to_string())
        }
        "outlives" => {
            let lt = payload.as_str().ok_or("outlives expected string")?;
            Ok(format!("'{}", lt.trim_start_matches("'")))
        }
        other => Err(format!("unknown bound tag '{}'", other)),
    }
}

fn render_where_predicate(val: &Value) -> Result<String, String> {
    let obj = val
        .as_object()
        .ok_or_else(|| format!("expected predicate object, got: {}", short_val(val)))?;

    let (tag, payload) = obj.iter().next().ok_or("empty predicate")?;

    match tag.as_str() {
        "bound_predicate" => {
            let ty = payload.get("type").ok_or("bound_predicate missing type")?;
            let ty_src = TypeRepr::parse_type_public(ty)?.to_source();
            let bounds: Vec<String> = payload
                .get("bounds")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(render_generic_bound)
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default();
            Ok(format!("{}: {}", ty_src, bounds.join(" + ")))
        }
        "lifetime_predicate" => {
            let lt = payload
                .get("lifetime")
                .and_then(|v| v.as_str())
                .ok_or("lifetime_predicate missing lifetime")?;
            let outlives: Vec<String> = payload
                .get("outlives")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|s| s.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            Ok(format!("{}: {}", lt, outlives.join(" + ")))
        }
        "eq_predicate" => {
            let lhs = payload.get("lhs").ok_or("eq_predicate missing lhs")?;
            let rhs = payload.get("rhs").ok_or("eq_predicate missing rhs")?;
            let lhs_src = TypeRepr::parse_type_public(lhs)?.to_source();
            // RHS is a Term (Type or Constant)
            let rhs_src = if rhs.get("type").is_some() {
                TypeRepr::parse_type_public(&rhs["type"])?.to_source()
            } else if rhs.get("constant").is_some() {
                rhs["constant"]["expr"].as_str().unwrap_or("?").to_string()
            } else {
                "?".into()
            };
            Ok(format!("{} = {}", lhs_src, rhs_src))
        }
        other => Err(format!("unknown predicate tag '{}'", other)),
    }
}

fn parse_visibility(val: &Value) -> DocVisibility {
    match val {
        Value::String(s) => match s.as_str() {
            "public" => DocVisibility::Public,
            "crate" => DocVisibility::Crate,
            _ => DocVisibility::Private, // "default" means private
        },
        Value::Object(obj) => {
            if let Some(restricted) = obj.get("restricted") {
                let path = restricted
                    .get("path")
                    .and_then(|p| p.as_str())
                    .unwrap_or("")
                    .to_string();
                DocVisibility::Restricted(path)
            } else {
                DocVisibility::Private
            }
        }
        _ => DocVisibility::Private,
    }
}

/// Parse the `attrs` array from a doc-JSON item.
///
/// In format v57+ (and likely v37 too based on testing), attrs are structured
/// objects: `{"other": "#[...]"}`, `{"repr": {...}}`, `{"must_use": {...}}`, etc.
/// In very old formats they might be raw strings. We handle both.
fn parse_attributes(attrs: &[Value]) -> (Vec<String>, Vec<Vec<String>>, Vec<String>) {
    let mut repr_attrs = Vec::new();
    let mut derive_attrs = Vec::new();
    let mut other_attrs = Vec::new();

    for attr in attrs {
        match attr {
            // Structured object format (v57+)
            Value::Object(obj) => {
                if let Some(repr) = obj.get("repr") {
                    let parts = parse_repr_object(repr);
                    if !parts.is_empty() {
                        repr_attrs.push(parts);
                    }
                }
                if let Some(other_str) = obj.get("other").and_then(|v| v.as_str()) {
                    classify_other_attr(
                        other_str,
                        &mut repr_attrs,
                        &mut derive_attrs,
                        &mut other_attrs,
                    );
                }
                if let Some(must_use) = obj.get("must_use") {
                    let reason = must_use
                        .get("reason")
                        .and_then(|v| v.as_str())
                        .map(|r| format!("#[must_use = \"{}\"]", r))
                        .unwrap_or_else(|| "#[must_use]".to_string());
                    other_attrs.push(reason);
                }
            }
            // Raw string format (older versions)
            Value::String(s) => {
                classify_other_attr(s, &mut repr_attrs, &mut derive_attrs, &mut other_attrs);
            }
            _ => {}
        }
    }

    (repr_attrs, derive_attrs, other_attrs)
}

fn parse_repr_object(repr: &Value) -> String {
    let mut parts: Vec<String> = Vec::new();

    if let Some(kind) = repr.get("kind").and_then(|v| v.as_str()) {
        parts.push(match kind {
            "c" => "C".into(),
            "rust" => "Rust".into(),
            "transparent" => "transparent".into(),
            "simd" => "simd".into(),
            other => other.to_string(),
        });
    }
    if let Some(packed) = repr.get("packed").and_then(|v| v.as_u64()) {
        if packed > 0 {
            parts.push(format!("packed({})", packed));
        } else {
            parts.push("packed".into());
        }
    }
    if let Some(align) = repr.get("align").and_then(|v| v.as_u64()) {
        parts.push(format!("align({})", align));
    }
    if let Some(int) = repr.get("int").and_then(|v| v.as_str()) {
        parts.push(int.to_string());
    }

    parts.join(", ")
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

fn short_val(val: &Value) -> String {
    let s = val.to_string();
    if s.len() > 80 {
        format!("{}…", &s[..80])
    } else {
        s
    }
}
