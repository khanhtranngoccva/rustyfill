//! Rendering of wire [`Type`] values as Rust source.
//!
//! The wire format carries fully-resolved types; this module turns them into
//! source strings for emission, with path routing applied at the caller via
//! a callback so that spec replacements and mirror-tree routing stay in
//! `emit`.

use super::wire::{Abi, GenericArg, GenericArgs, Path, Term, Type};

/// A resolver for [`Path`] references: turns a raw rustdoc path string plus
/// the referenced item's id into its final emitted form.
///
/// Implemented generically over closures so callers can pass either a plain
/// closure or a function pointer without boxing.
pub trait PathResolver {
    fn resolve(&self, path: &str, id: u32) -> String;
}

impl<F> PathResolver for F
where
    F: Fn(&str, u32) -> String,
{
    fn resolve(&self, path: &str, id: u32) -> String {
        self(path, id)
    }
}

/// Render a wire type as Rust source, resolving path references through `resolve`.
pub fn render(ty: &Type, resolve: &impl PathResolver) -> String {
    let mut out = String::new();
    write_type(&mut out, ty, resolve);
    out
}

/// Render generic arguments in angle-bracket form (`<A, B>`), used when the
/// base path has already been substituted (e.g., by a spec replacement).
pub fn render_args(args: &GenericArgs, resolve: &impl PathResolver) -> String {
    let mut out = String::new();
    write_generic_args(&mut out, args, resolve);
    out
}

fn write_type(out: &mut String, ty: &Type, resolve: &impl PathResolver) {
    match ty {
        Type::ResolvedPath(p) => {
            out.push_str(&resolve.resolve(&p.path, p.id.0));
            if let Some(args) = &p.args {
                write_generic_args(out, args, resolve);
            }
        }
        Type::DynTrait(dt) => {
            out.push_str("dyn ");
            let bounds: Vec<String> = dt
                .traits
                .iter()
                .map(|t| render_poly_trait_path(&t.trait_, resolve))
                .collect();
            out.push_str(&bounds.join(" + "));
            if let Some(lt) = &dt.lifetime {
                out.push_str(" + ");
                out.push_str(lt);
            }
        }
        Type::Generic(name) => out.push_str(name),
        Type::Primitive(name) => out.push_str(name),
        Type::FunctionPointer(fp) => write_fn_pointer(out, fp, resolve),
        Type::Tuple(elems) => {
            out.push('(');
            let mut first = true;
            for e in elems {
                if !first {
                    out.push_str(", ");
                }
                first = false;
                write_type(out, e, resolve);
            }
            if elems.len() == 1 {
                out.push(',');
            }
            out.push(')');
        }
        Type::Slice(elem) => {
            out.push('[');
            write_type(out, elem, resolve);
            out.push(']');
        }
        Type::Array { type_, len } => {
            out.push('[');
            write_type(out, type_, resolve);
            out.push_str("; ");
            out.push_str(len);
            out.push(']');
        }
        Type::ImplTrait(bounds) => {
            out.push_str("impl ");
            let rendered: Vec<String> =
                bounds.iter().map(|b| render_bound(b, resolve)).collect();
            out.push_str(&rendered.join(" + "));
        }
        Type::RawPointer { is_mutable, type_ } => {
            out.push('*');
            out.push_str(if *is_mutable { "mut " } else { "const " });
            write_type(out, type_, resolve);
        }
        Type::BorrowedRef {
            lifetime,
            is_mutable,
            type_,
        } => {
            out.push('&');
            if let Some(lt) = lifetime {
                out.push_str(lt);
                out.push(' ');
            }
            if *is_mutable {
                out.push_str("mut ");
            }
            write_type(out, type_, resolve);
        }
        Type::QualifiedPath {
            name,
            args,
            self_type,
            trait_,
        } => {
            if let Some(trait_path) = trait_ {
                out.push('<');
                write_type(out, self_type, resolve);
                out.push_str(" as ");
                out.push_str(&render_poly_trait_path(trait_path, resolve));
                out.push_str(">::");
            } else {
                write_type(out, self_type, resolve);
                out.push_str("::");
            }
            out.push_str(name);
            if let Some(args) = args {
                write_generic_args(out, args, resolve);
            }
        }
    }
}

fn render_poly_trait_path(path: &Path, resolve: &impl PathResolver) -> String {
    let base = resolve.resolve(&path.path, path.id.0);
    let mut out = base;
    if let Some(args) = &path.args {
        write_generic_args(&mut out, args, resolve);
    }
    out
}

fn write_fn_pointer(out: &mut String, fp: &super::wire::FunctionPointer, resolve: &impl PathResolver) {
    // ABI: only emit when it differs from the default Rust ABI.
    match &fp.header.abi {
        Abi::Rust => {}
        other => {
            out.push_str("extern \"");
            out.push_str(&abi_name(other));
            out.push_str("\" ");
        }
    }
    out.push_str("fn(");
    let mut first = true;
    for (_name, ty) in &fp.sig.inputs {
        if !first {
            out.push_str(", ");
        }
        first = false;
        write_type(out, ty, resolve);
    }
    if fp.sig.is_c_variadic {
        if !first {
            out.push_str(", ");
        }
        out.push_str("...");
    }
    out.push(')');
    if let Some(ret) = &fp.sig.output {
        out.push_str(" -> ");
        write_type(out, ret, resolve);
    }
}

/// Render an ABI tag the way it appears in source (`C`, `System`, ...).
fn abi_name(abi: &Abi) -> String {
    match abi {
        Abi::Rust => "Rust".into(),
        Abi::C { .. } => "C".into(),
        Abi::Cdecl { .. } => "cdecl".into(),
        Abi::Stdcall { .. } => "stdcall".into(),
        Abi::Fastcall { .. } => "fastcall".into(),
        Abi::Aapcs { .. } => "aapcs".into(),
        Abi::Win64 { .. } => "win64".into(),
        Abi::SysV64 { .. } => "sysv64".into(),
        Abi::System { .. } => "system".into(),
        Abi::Other(name) => name.clone(),
    }
}

fn write_generic_args(out: &mut String, args: &GenericArgs, resolve: &impl PathResolver) {
    match args {
        GenericArgs::AngleBracketed { args, constraints } => {
            out.push('<');
            let mut parts: Vec<String> = Vec::new();
            for arg in args {
                parts.push(render_arg(arg, resolve));
            }
            for c in constraints {
                let binding = match &c.binding {
                    super::wire::AssocItemConstraintKind::Equality(term) => {
                        format!(" = {}", render_term(term, resolve))
                    }
                    super::wire::AssocItemConstraintKind::Constraint(bounds) => {
                        let rendered: Vec<String> =
                            bounds.iter().map(|b| render_bound(b, resolve)).collect();
                        format!(": {}", rendered.join(" + "))
                    }
                };
                parts.push(format!("{}{}", c.name, binding));
            }
            out.push_str(&parts.join(", "));
            out.push('>');
        }
        // Parenthesized args are function-pointer sugar: `<Fn(A) -> B>`.
        GenericArgs::Parenthesized { inputs, output } => {
            out.push_str("<fn(");
            let rendered: Vec<String> = inputs.iter().map(|t| render(t, resolve)).collect();
            out.push_str(&rendered.join(", "));
            out.push(')');
            if let Some(o) = output {
                out.push_str(" -> ");
                write_type(out, o, resolve);
            }
            out.push('>');
        }
        GenericArgs::ReturnTypeNotation => {
            out.push_str("(..)");
        }
    }
}

fn render_arg(arg: &GenericArg, resolve: &impl PathResolver) -> String {
    match arg {
        GenericArg::Lifetime(lt) => lt.clone(),
        GenericArg::Type(ty) => render(ty, resolve),
        GenericArg::Const(c) => c.expr.clone(),
        GenericArg::Infer => "_".into(),
    }
}

fn render_bound(bound: &super::wire::GenericBound, resolve: &impl PathResolver) -> String {
    match bound {
        super::wire::GenericBound::TraitBound {
            trait_: path,
            modifier,
            ..
        } => {
            let base = render_poly_trait_path(path, resolve);
            match modifier {
                super::wire::TraitBoundModifier::None => base,
                super::wire::TraitBoundModifier::Maybe => format!("{base}?"),
                super::wire::TraitBoundModifier::MaybeConst => format!("{base}?"),
            }
        }
        super::wire::GenericBound::Outlives(lt) => lt.clone(),
        super::wire::GenericBound::Use(args) => {
            let rendered: Vec<String> = args
                .iter()
                .map(|a| match a {
                    super::wire::PreciseCapturingArg::Lifetime(lt) => lt.clone(),
                    super::wire::PreciseCapturingArg::Param(p) => p.clone(),
                })
                .collect();
            format!("use<{}>", rendered.join(", "))
        }
    }
}

fn render_term(term: &Term, resolve: &impl PathResolver) -> String {
    match term {
        Term::Type(ty) => render(ty, resolve),
        Term::Constant(c) => c.expr.clone(),
    }
}
