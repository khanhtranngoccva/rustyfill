//! Recursive type representation extracted from rustdoc JSON.
//!
//! Mirrors the `Type` enum in `rustdoc_json_types` but as our own lightweight
//! representation optimized for source emission. Uses `serde_json::Value` for
//! parsing to remain compatible across format versions.

use serde_json::Value;

/// A fully-resolved type reference, ready for rendering as Rust source.
#[derive(Clone, Debug)]
pub enum TypeRepr {
    /// A path to a named type: `core::cell::UnsafeCell<T>` or just `T`.
    ResolvedPath {
        /// The path string as emitted by rustdoc (e.g., "crate::cell::UnsafeCell").
        path: String,
        /// The item ID in the doc-JSON index, used for canonical path lookup.
        id: Option<u64>,
        /// Generic arguments, if any.
        args: Option<GenericArgsRepr>,
    },
    /// A generic parameter in scope: `T`, `K`, etc.
    Generic(String),
    /// A primitive type: `u32`, `bool`, `char`, etc.
    Primitive(String),
    /// A raw pointer: `*mut T`, `*const T`.
    RawPointer {
        mutable: bool,
        pointee: Box<TypeRepr>,
    },
    /// A reference: `&'a mut T`, `&str`, etc.
    BorrowedRef {
        lifetime: Option<String>,
        mutable: bool,
        referent: Box<TypeRepr>,
    },
    /// A tuple type: `(A, B, C)`.
    Tuple(Vec<TypeRepr>),
    /// An array type: `[T; N]`.
    Array { element: Box<TypeRepr>, len: String },
    /// An unsized slice: `[T]`.
    Slice(Box<TypeRepr>),
    /// A function pointer: `fn(A, B) -> C`.
    FunctionPointer {
        abi: Option<String>,
        inputs: Vec<TypeRepr>,
        output: Option<Box<TypeRepr>>,
        variadic: bool,
    },
    /// A dynamic trait object: `dyn Trait + 'lifetime`.
    DynTrait {
        bounds: Vec<String>,
        lifetime: Option<String>,
    },
    /// An opaque type: `impl Trait + Bound`.
    ImplTrait(Vec<String>),
    /// Inferred type: `_`.
    Infer,
    /// A qualified path: `<T as Trait>::Assoc` or `Struct::Assoc`.
    QualifiedPath {
        name: String,
        self_type: Box<TypeRepr>,
        trait_path: Option<String>,
        args: Option<GenericArgsRepr>,
    },
}

/// Generic arguments attached to a path: `<A, B, 'a, const N: usize>`.
#[derive(Clone, Debug)]
pub struct GenericArgsRepr {
    pub args: Vec<GenericArgRepr>,
    /// Associated type bindings: `Item = u32`.
    pub constraints: Vec<String>,
}

/// A single generic argument.
#[derive(Clone, Debug)]
pub enum GenericArgRepr {
    Lifetime(String),
    Type(TypeRepr),
    Const(String),
    Infer,
}

impl TypeRepr {
    /// Parse a type from its JSON representation.
    ///
    /// Handles both format v37-style (where `struct_field` wraps the type in
    /// `{"type": ...}`) and v57+ style (where the type IS the value directly).
    pub fn from_json(val: &Value) -> Result<Self, String> {
        // In some contexts (older formats), the type is wrapped: {"type": <actual>}
        // In newer formats, it's the type directly.
        let actual = if val.get("type").is_some() && !val.has_type_tag() {
            val.get("type").unwrap()
        } else {
            val
        };

        Self::parse_type(actual)
    }

    /// Parse a type from its JSON representation (public access for model.rs).
    pub fn parse_type_public(val: &Value) -> Result<Self, String> {
        Self::parse_type(val)
    }

    fn parse_type(val: &Value) -> Result<Self, String> {
        let obj = val
            .as_object()
            .ok_or_else(|| format!("expected type object, got: {}", short(val)))?;

        // Exactly one key identifies the type variant
        let (tag, payload) = obj
            .iter()
            .next()
            .ok_or_else(|| "empty type object".to_string())?;

        match tag.as_str() {
            "resolved_path" => {
                let rp = payload;
                // Handle both "path" (v57+) and "name" (v37) field names
                let path = rp
                    .get("path")
                    .or_else(|| rp.get("name"))
                    .and_then(|v| v.as_str())
                    .ok_or("resolved_path missing path/name")?
                    .to_string();

                let id = rp.get("id").and_then(|v| v.as_u64());

                let args = rp
                    .get("args")
                    .filter(|a| !a.is_null())
                    .map(GenericArgsRepr::from_json)
                    .transpose()?;

                Ok(Self::ResolvedPath { path, id, args })
            }

            "generic" => {
                let name = payload
                    .as_str()
                    .ok_or("generic expected string")?
                    .to_string();
                Ok(Self::Generic(name))
            }

            "primitive" => {
                let name = payload
                    .as_str()
                    .ok_or("primitive expected string")?
                    .to_string();
                Ok(Self::Primitive(name))
            }

            "raw_pointer" => {
                let mutable = payload
                    .get("mutable")
                    .and_then(|v| v.as_bool())
                    .ok_or("raw_pointer missing mutable")?;
                let pointee_ty = payload.get("type").ok_or("raw_pointer missing type")?;
                let pointee = Box::new(Self::parse_type(pointee_ty)?);
                Ok(Self::RawPointer { mutable, pointee })
            }

            "borrowed_ref" => {
                let lifetime = payload
                    .get("lifetime")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                // In older format versions, `mutable` may be omitted for
                // immutable references. Default to false when absent.
                let mutable = payload
                    .get("mutable")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let referent_ty = payload.get("type").ok_or("borrowed_ref missing type")?;
                let referent = Box::new(Self::parse_type(referent_ty)?);
                Ok(Self::BorrowedRef {
                    lifetime,
                    mutable,
                    referent,
                })
            }

            "tuple" => {
                let arr = payload.as_array().ok_or("tuple expected array")?;
                let elems = arr
                    .iter()
                    .map(Self::parse_type)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::Tuple(elems))
            }

            "array" => {
                let elem_ty = payload.get("type").ok_or("array missing type")?;
                let len = payload
                    .get("len")
                    .and_then(|v| v.as_str())
                    .ok_or("array missing len")?
                    .to_string();
                let element = Box::new(Self::parse_type(elem_ty)?);
                Ok(Self::Array { element, len })
            }

            "slice" => {
                // In the JSON, slice is just {"slice": <Type>} — the payload IS the element type
                let element = Box::new(Self::parse_type(payload)?);
                Ok(Self::Slice(element))
            }

            "function_ptr" => {
                let fp = payload;
                let sig = fp.get("sig").ok_or("function_ptr missing sig")?;
                let inputs_arr = sig
                    .get("inputs")
                    .and_then(|v| v.as_array())
                    .ok_or("function sig missing inputs")?;
                let inputs = inputs_arr
                    .iter()
                    .map(|pair| {
                        // Each input is [name, type]
                        let ty_val = pair.get(1).ok_or("function input missing type")?;
                        Self::parse_type(ty_val)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let output = sig
                    .get("output")
                    .filter(|o| !o.is_null())
                    .map(|o| Self::parse_type(o).map(Box::new))
                    .transpose()?;
                let variadic = sig
                    .get("c_variadic")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let abi = fp
                    .get("header")
                    .and_then(|h| h.get("abi"))
                    .and_then(|a| a.as_str())
                    .map(String::from);

                Ok(Self::FunctionPointer {
                    abi,
                    inputs,
                    output,
                    variadic,
                })
            }

            "dyn_trait" => {
                let dt = payload;
                let traits_arr = dt
                    .get("traits")
                    .and_then(|v| v.as_array())
                    .ok_or("dyn_trait missing traits")?;
                let bounds = traits_arr
                    .iter()
                    .map(|t| {
                        t.get("trait")
                            .and_then(|tp| tp.get("path"))
                            .or_else(|| t.get("trait").and_then(|tp| tp.get("name")))
                            .and_then(|v| v.as_str())
                            .map(String::from)
                            .ok_or("dyn_trait bound missing path")
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let lifetime = dt
                    .get("lifetime")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                Ok(Self::DynTrait { bounds, lifetime })
            }

            "impl_trait" => {
                let bounds_arr = payload.as_array().ok_or("impl_trait expected array")?;
                let bounds = bounds_arr
                    .iter()
                    .map(|b| {
                        b.get("trait_bound")
                            .and_then(|tb| tb.get("trait"))
                            .and_then(|tp| tp.get("path"))
                            .or_else(|| {
                                b.get("trait_bound")
                                    .and_then(|tb| tb.get("trait"))
                                    .and_then(|tp| tp.get("name"))
                            })
                            .and_then(|v| v.as_str())
                            .map(String::from)
                            .ok_or("impl_trait bound missing path")
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::ImplTrait(bounds))
            }

            "infer" => Ok(Self::Infer),

            "qualified_path" => {
                let qp = payload;
                let name = qp
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or("qualified_path missing name")?
                    .to_string();
                let self_ty = qp
                    .get("self_type")
                    .ok_or("qualified_path missing self_type")?;
                let self_type = Box::new(Self::parse_type(self_ty)?);
                let trait_path = qp
                    .get("trait")
                    .filter(|t| !t.is_null())
                    .and_then(|t| t.get("path"))
                    .or_else(|| {
                        qp.get("trait")
                            .filter(|t| !t.is_null())
                            .and_then(|t| t.get("name"))
                    })
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let args = qp
                    .get("args")
                    .filter(|a| !a.is_null())
                    .map(GenericArgsRepr::from_json)
                    .transpose()?;

                Ok(Self::QualifiedPath {
                    name,
                    self_type,
                    trait_path,
                    args,
                })
            }

            other => Err(format!("unknown type tag '{}'", other)),
        }
    }

    /// Render this type as a Rust source string.
    pub fn to_source(&self) -> String {
        match self {
            Self::ResolvedPath { path, id: _, args } => {
                let base = normalize_path(path);
                match args {
                    Some(args) => format!("{}{}", base, args.to_source()),
                    None => base,
                }
            }
            Self::Generic(name) => name.clone(),
            Self::Primitive(name) => name.clone(),
            Self::RawPointer { mutable, pointee } => {
                let mutability = if *mutable { "mut " } else { "const " };
                format!("*{}{}", mutability, pointee.to_source())
            }
            Self::BorrowedRef {
                lifetime,
                mutable,
                referent,
            } => {
                let lt = lifetime
                    .as_deref()
                    .map(|l| format!("{} ", l))
                    .unwrap_or_default();
                let m = if *mutable { "mut " } else { "" };
                format!("&{}{}{}", lt, m, referent.to_source())
            }
            Self::Tuple(elems) => {
                if elems.len() == 1 {
                    format!("({},{})", elems[0].to_source(), "")
                } else {
                    let inner: Vec<String> = elems.iter().map(|e| e.to_source()).collect();
                    format!("({})", inner.join(", "))
                }
            }
            Self::Array { element, len } => {
                format!("[{}; {}]", element.to_source(), len)
            }
            Self::Slice(element) => format!("[{}]", element.to_source()),
            Self::FunctionPointer {
                abi,
                inputs,
                output,
                variadic,
            } => {
                let abi_str = abi
                    .as_deref()
                    .map(|a| {
                        if a == "Rust" {
                            String::new()
                        } else {
                            format!("extern \"{}\" ", a)
                        }
                    })
                    .unwrap_or_default();
                let params: Vec<String> = inputs.iter().map(|i| i.to_source()).collect();
                let mut param_str = params.join(", ");
                if *variadic {
                    if param_str.is_empty() {
                        param_str = "...".into();
                    } else {
                        param_str.push_str(", ...");
                    }
                }
                let out = output
                    .as_ref()
                    .map(|o| format!(" -> {}", o.to_source()))
                    .unwrap_or_default();
                format!("{}fn({}){}", abi_str, param_str, out)
            }
            Self::DynTrait { bounds, lifetime } => {
                let lt = lifetime
                    .as_deref()
                    .map(|l| format!(" + {}", l))
                    .unwrap_or_default();
                format!("dyn {}{}", bounds.join(" + "), lt)
            }
            Self::ImplTrait(bounds) => format!("impl {}", bounds.join(" + ")),
            Self::Infer => "_".into(),
            Self::QualifiedPath {
                name,
                self_type,
                trait_path,
                args,
            } => {
                let args_str = args.as_ref().map(|a| a.to_source()).unwrap_or_default();
                match trait_path {
                    Some(trait_) => format!(
                        "<{} as {}>::{}{}",
                        self_type.to_source(),
                        trait_,
                        name,
                        args_str
                    ),
                    None => format!("{}::{}{}", self_type.to_source(), name, args_str),
                }
            }
        }
    }
}

impl GenericArgsRepr {
    fn from_json(val: &Value) -> Result<Self, String> {
        // GenericArgs is either "angle_bracketed" or "parenthesized"
        let obj = val
            .as_object()
            .ok_or_else(|| format!("expected generic args object, got: {}", short(val)))?;

        let (tag, payload) = obj.iter().next().ok_or("empty generic args")?;

        match tag.as_str() {
            "angle_bracketed" => {
                let args_arr = payload
                    .get("args")
                    .and_then(|v| v.as_array())
                    .ok_or("angle_bracketed missing args")?;
                let args = args_arr
                    .iter()
                    .map(GenericArgRepr::from_json)
                    .collect::<Result<Vec<_>, _>>()?;

                // Constraints are complex; serialize them as simplified strings
                let constraints: Vec<String> = payload
                    .get("constraints")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .map(|c| {
                                let name = c.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                                let binding = c.get("binding");
                                match binding {
                                    Some(b) if b.get("equality").is_some() => {
                                        // Try to extract the type from the equality term
                                        let eq_val = &b["equality"];
                                        // Equality can be a direct Type or wrapped in an array
                                        let ty_src = if eq_val.is_object() {
                                            TypeRepr::parse_type(eq_val)
                                                .map(|t| t.to_source())
                                                .unwrap_or_else(|_| "???".into())
                                        } else {
                                            "???".to_string()
                                        };
                                        format!("{} = {}", name, ty_src)
                                    }
                                    _ => name.to_string(),
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                Ok(Self { args, constraints })
            }
            "parenthesized" => {
                // Fn(A, B) -> C — treat as a special case
                let inputs = payload
                    .get("inputs")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .map(TypeRepr::parse_type)
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .transpose()?
                    .unwrap_or_default();
                let output = payload
                    .get("output")
                    .filter(|o| !o.is_null())
                    .map(TypeRepr::parse_type)
                    .transpose()?;

                // Represent as a pseudo-type in the args list
                let fn_ty = TypeRepr::FunctionPointer {
                    abi: None,
                    inputs,
                    output: output.map(Box::new),
                    variadic: false,
                };
                Ok(Self {
                    args: vec![GenericArgRepr::Type(fn_ty)],
                    constraints: vec![],
                })
            }
            other => Err(format!("unknown generic args tag '{}'", other)),
        }
    }

    pub fn to_source(&self) -> String {
        let args: Vec<String> = self.args.iter().map(|a| a.to_source()).collect();
        let mut inner = args.join(", ");
        if !self.constraints.is_empty() {
            if !inner.is_empty() {
                inner.push(',');
            }
            inner.push(' ');
            inner.push_str(&self.constraints.join(", "));
        }
        format!("<{}>", inner)
    }
}

impl GenericArgRepr {
    fn from_json(val: &Value) -> Result<Self, String> {
        let obj = val
            .as_object()
            .ok_or_else(|| format!("expected generic arg object, got: {}", short(val)))?;

        let (tag, payload) = obj.iter().next().ok_or("empty generic arg")?;

        match tag.as_str() {
            "lifetime" => {
                let name = payload
                    .as_str()
                    .ok_or("lifetime expected string")?
                    .to_string();
                Ok(Self::Lifetime(name))
            }
            "type" => {
                let ty = TypeRepr::parse_type(payload)?;
                Ok(Self::Type(ty))
            }
            "const" => {
                let expr = payload
                    .get("expr")
                    .and_then(|v| v.as_str())
                    .ok_or("const missing expr")?
                    .to_string();
                Ok(Self::Const(expr))
            }
            "infer" => Ok(Self::Infer),
            other => Err(format!("unknown generic arg tag '{}'", other)),
        }
    }

    pub fn to_source(&self) -> String {
        match self {
            Self::Lifetime(lt) => lt.clone(),
            Self::Type(ty) => ty.to_source(),
            Self::Const(expr) => expr.clone(),
            Self::Infer => "_".into(),
        }
    }
}

/// Normalize a rustdoc path for emission.
/// Converts `crate::foo::Bar` → `foo::Bar` (strip crate prefix).
fn normalize_path(path: &str) -> String {
    let stripped = path
        .strip_prefix("$crate::")
        .or_else(|| path.strip_prefix("crate::"))
        .unwrap_or(path);
    stripped.to_string()
}

/// Short debug representation of a JSON value for error messages.
fn short(val: &Value) -> String {
    let s = val.to_string();
    if s.len() > 80 {
        format!("{}…", &s[..80])
    } else {
        s
    }
}

// Helper trait to detect if a Value already IS a type (vs wrapping one)
trait HasTypeTag {
    fn has_type_tag(&self) -> bool;
}

impl HasTypeTag for Value {
    fn has_type_tag(&self) -> bool {
        // A "wrapped" type looks like {"type": {...}} where the outer object
        // has ONLY the "type" key (plus maybe "generics").
        // An actual type has tags like "resolved_path", "generic", "primitive", etc.
        let known_tags = [
            "resolved_path",
            "generic",
            "primitive",
            "raw_pointer",
            "borrowed_ref",
            "tuple",
            "array",
            "slice",
            "function_ptr",
            "dyn_trait",
            "impl_trait",
            "infer",
            "qualified_path",
        ];
        if let Some(obj) = self.as_object() {
            if obj.len() == 1 {
                let key = obj.keys().next().unwrap().as_str();
                return !known_tags.contains(&key);
            }
        }
        false
    }
}
