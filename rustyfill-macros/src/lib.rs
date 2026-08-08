//! Procedural macros for [`rustyfill`](https://docs.rs/rustyfill).

extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

/// Derives `TryClone` for a struct or enum.
///
/// All fields in every variant must themselves implement `TryClone`. The generated
/// implementation clones each field fallibly and propagates the first error encountered.
#[proc_macro_derive(TryClone)]
pub fn derive_try_clone(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let clone_body = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => {
                let field_clones = fields.named.iter().map(|field| {
                    let ident = &field.ident;
                    quote! {
                        #ident: self.#ident.try_clone()?,
                    }
                });
                quote! {
                    Ok(Self { #(#field_clones)* })
                }
            }
            Fields::Unnamed(fields) => {
                let indices = 0..fields.unnamed.len();
                let field_clones = indices.map(|i| {
                    let idx = syn::Index::from(i);
                    quote! {
                        self.#idx.try_clone()?,
                    }
                });
                quote! {
                    Ok(Self (#(#field_clones)*))
                }
            }
            Fields::Unit => {
                quote! {
                    Ok(Self)
                }
            }
        },
        Data::Enum(data) => {
            let arms = data.variants.iter().map(|variant| {
                let variant_ident = &variant.ident;
                match &variant.fields {
                    Fields::Named(fields) if !fields.named.is_empty() => {
                        let field_idents: Vec<_> = fields
                            .named
                            .iter()
                            .map(|f| f.ident.as_ref().unwrap())
                            .collect();
                        let field_patterns: Vec<_> =
                            field_idents.iter().map(|id| quote!(#id)).collect();
                        let field_clones = field_idents.iter().map(|id| {
                            quote! {
                                #id: #id.try_clone()?,
                            }
                        });
                        quote! {
                            Self::#variant_ident { #(#field_patterns),* } => {
                                Ok(Self::#variant_ident { #(#field_clones)* })
                            },
                        }
                    }
                    Fields::Named(_) | Fields::Unit => {
                        quote! {
                            Self::#variant_ident => Ok(Self::#variant_ident),
                        }
                    }
                    Fields::Unnamed(fields) => {
                        let field_names: Vec<_> = (0..fields.unnamed.len())
                            .map(|i| quote::format_ident!("f{i}"))
                            .collect();
                        let field_clones = field_names.iter().map(|fn_| {
                            quote! {
                                #fn_.try_clone()?,
                            }
                        });
                        quote! {
                            Self::#variant_ident (#(#field_names),*) => {
                                Ok(Self::#variant_ident (#(#field_clones)*))
                            },
                        }
                    }
                }
            });
            quote! {
                match self {
                    #(#arms)*
                }
            }
        }
        Data::Union(_) => {
            return syn::Error::new_spanned(name, "#[derive(TryClone)] is not supported on unions")
                .to_compile_error()
                .into();
        }
    };

    let expanded = quote! {
        impl #impl_generics ::rustyfill::try_clone::TryClone for #name #ty_generics #where_clause {
            fn try_clone(&self) -> Result<Self, ::rustyfill::try_clone::TryCloneError> {
                #clone_body
            }
        }
    };

    TokenStream::from(expanded)
}

/// Generates `TryClone` implementations for tuples of arities 1 through `max`.
#[proc_macro]
pub fn try_clone_tuples(input: TokenStream) -> TokenStream {
    let max: usize = syn::parse::<syn::LitInt>(input)
        .ok()
        .and_then(|lit| lit.base10_parse().ok())
        .unwrap_or(12);
    let max = max.clamp(1, 16);

    let mut output = Vec::new();

    for arity in 1..=max {
        let type_params: Vec<_> = (0..arity).map(|i| quote::format_ident!("T{i}")).collect();
        let bounds: Vec<_> = type_params.iter().map(|t| quote!(#t: TryClone)).collect();

        let types_joined: proc_macro2::TokenStream = {
            let mut ts = proc_macro2::TokenStream::new();
            for (i, tp) in type_params.iter().enumerate() {
                if i > 0 { ts.extend(quote!(,)); }
                ts.extend(quote!(#tp));
            }
            ts
        };

        let tuple_pat = if arity == 1 {
            quote!((#types_joined ,))
        } else {
            quote!((#types_joined))
        };

        let fields_joined: proc_macro2::TokenStream = {
            let mut ts = proc_macro2::TokenStream::new();
            for i in 0..arity {
                if i > 0 { ts.extend(quote!(,)); }
                let idx = syn::Index::from(i);
                ts.extend(quote!(self.#idx.try_clone()?));
            }
            ts
        };

        let body = if arity == 1 {
            quote!(((#fields_joined ),))
        } else {
            quote!((#fields_joined))
        };

        output.push(quote! {
            impl<#(#bounds),*> TryClone for #tuple_pat {
                #[inline]
                fn try_clone(&self) -> Result<Self, TryCloneError> {
                    Ok(#body)
                }
            }
        });
    }

    TokenStream::from(quote!(#(#output)*))
}

/// Derives `TryDebug` for a struct or enum.
///
/// All fields in every variant must themselves implement `TryDebug`. The generated
/// implementation formats each field fallibly via `try_fmt` and propagates the first
/// error encountered. Unlike `TryDisplay`, this is safe to derive because the macro
/// generates the formatting code directly and can guarantee no hidden allocations.
#[proc_macro_derive(TryDebug)]
pub fn derive_try_debug(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let debug_body = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => {
                if fields.named.is_empty() {
                    quote! {
                        f.write_str(concat!(stringify!(#name), "()"))?;
                        Ok(())
                    }
                } else {
                    let field_fmts = fields.named.iter().map(|field| {
                        let ident = &field.ident;
                        let field_name = ident.as_ref().map(|i| i.to_string()).unwrap_or_default();
                        quote! {
                            f.write_str(concat!(" ", #field_name, ": "))?;
                            ::rustyfill::try_fmt::TryDebug::try_fmt(&self.#ident, f)?;
                        }
                    });
                    quote! {
                        f.write_str(concat!(stringify!(#name), "{"))?;
                        #(#field_fmts)*
                        f.write_str("}")?;
                        Ok(())
                    }
                }
            }
            Fields::Unnamed(fields) => {
                if fields.unnamed.is_empty() {
                    quote! {
                        f.write_str(concat!(stringify!(#name), "()"))?;
                        Ok(())
                    }
                } else {
                    let field_fmts = (0..fields.unnamed.len()).map(|i| {
                        let idx = syn::Index::from(i);
                        quote! {
                            if #i > 0 { f.write_str(", ")?; }
                            ::rustyfill::try_fmt::TryDebug::try_fmt(&self.#idx, f)?;
                        }
                    });
                    quote! {
                        f.write_str(concat!(stringify!(#name), "("))?;
                        #(#field_fmts)*
                        f.write_str(")")?;
                        Ok(())
                    }
                }
            }
            Fields::Unit => {
                quote! {
                    f.write_str(stringify!(#name))?;
                    Ok(())
                }
            }
        },
        Data::Enum(data) => {
            if data.variants.is_empty() {
                return syn::Error::new_spanned(
                    name,
                    "#[derive(TryDebug)] on empty enums is not supported",
                )
                .to_compile_error()
                .into();
            }
            let arms = data.variants.iter().map(|variant| {
                let variant_ident = &variant.ident;
                match &variant.fields {
                    Fields::Named(fields) if !fields.named.is_empty() => {
                        let field_idents: Vec<_> = fields
                            .named
                            .iter()
                            .map(|f| f.ident.as_ref().unwrap())
                            .collect();
                        let field_patterns: Vec<_> =
                            field_idents.iter().map(|id| quote!(#id)).collect();
                        let field_fmts = field_idents.iter().map(|id| {
                            let fname = id.to_string();
                            quote! {
                                f.write_str(concat!(" ", #fname, ": "))?;
                                ::rustyfill::try_fmt::TryDebug::try_fmt(#id, f)?;
                            }
                        });
                        quote! {
                            Self::#variant_ident { #(#field_patterns),* } => {
                                f.write_str(concat!(stringify!(#name), "::", stringify!(#variant_ident), "{"))?;
                                #(#field_fmts)*
                                f.write_str("}")?;
                                Ok(())
                            },
                        }
                    }
                    Fields::Named(_) | Fields::Unit => {
                        quote! {
                            Self::#variant_ident => {
                                f.write_str(concat!(stringify!(#name), "::", stringify!(#variant_ident)))?;
                                Ok(())
                            },
                        }
                    }
                    Fields::Unnamed(fields) => {
                        let field_names: Vec<_> = (0..fields.unnamed.len())
                            .map(|i| quote::format_ident!("f{i}"))
                            .collect();
                        let field_fmts = field_names.iter().enumerate().map(|(i, fn_)| {
                            quote! {
                                if #i > 0 { f.write_str(", ")?; }
                                ::rustyfill::try_fmt::TryDebug::try_fmt(#fn_, f)?;
                            }
                        });
                        quote! {
                            Self::#variant_ident (#(#field_names),*) => {
                                f.write_str(concat!(stringify!(#name), "::", stringify!(#variant_ident), "("))?;
                                #(#field_fmts)*
                                f.write_str(")")?;
                                Ok(())
                            },
                        }
                    }
                }
            });
            quote! {
                match self {
                    #(#arms)*
                }
            }
        }
        Data::Union(_) => {
            return syn::Error::new_spanned(name, "#[derive(TryDebug)] is not supported on unions")
                .to_compile_error()
                .into();
        }
    };

    let expanded = quote! {
        impl #impl_generics ::rustyfill::try_fmt::TryDebug for #name #ty_generics #where_clause {
            fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                #debug_body
            }
        }
    };

    TokenStream::from(expanded)
}

/// Generates `TryDebug` implementations for tuples of arities 1 through `max`.
#[proc_macro]
pub fn try_debug_tuples(input: TokenStream) -> TokenStream {
    let max: usize = syn::parse::<syn::LitInt>(input)
        .ok()
        .and_then(|lit| lit.base10_parse().ok())
        .unwrap_or(12);
    let max = max.clamp(1, 16);

    let mut output = Vec::new();

    for arity in 1..=max {
        let type_params: Vec<_> = (0..arity).map(|i| quote::format_ident!("T{i}")).collect();
        let bounds: Vec<_> = type_params.iter()
            .map(|t| quote!(#t: TryDebug))
            .collect();

        let types_joined: proc_macro2::TokenStream = {
            let mut ts = proc_macro2::TokenStream::new();
            for (i, tp) in type_params.iter().enumerate() {
                if i > 0 { ts.extend(quote!(,)); }
                ts.extend(quote!(#tp));
            }
            ts
        };

        let tuple_pat = if arity == 1 {
            quote!((#types_joined ,))
        } else {
            quote!((#types_joined))
        };

        let body_lines: Vec<_> = (0..arity).map(|i| {
            let idx = syn::Index::from(i);
            quote! {
                if #i > 0 { f.write_str(", ")?; }
                TryDebug::try_fmt(&self.#idx, f)?;
            }
        }).collect();

        let close_paren = if arity == 1 { ",)" } else { ")" };

        output.push(quote! {
            impl<#(#bounds),*> TryDebug for #tuple_pat {
                #[inline]
                fn try_fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    f.write_str("(")?;
                    #(#body_lines)*
                    f.write_str(#close_paren)?;
                    Ok(())
                }
            }
        });
    }

    TokenStream::from(quote!(#(#output)*))
}

/// Derives `TryDefault` for a struct or enum.
///
/// All fields in every variant must themselves implement `TryDefault`. The generated
/// implementation constructs each field fallibly and propagates the first error encountered.
#[proc_macro_derive(TryDefault)]
pub fn derive_try_default(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let default_body = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => {
                let field_defaults = fields.named.iter().map(|field| {
                    let ident = &field.ident;
                    quote! {
                        #ident: ::rustyfill::try_default::TryDefault::try_default()?,
                    }
                });
                quote! {
                    Ok(Self { #(#field_defaults)* })
                }
            }
            Fields::Unnamed(fields) => {
                let field_defaults = fields.unnamed.iter().map(|field| {
                    let ty = &field.ty;
                    quote! {
                        <#ty as ::rustyfill::try_default::TryDefault>::try_default()?,
                    }
                });
                quote! {
                    Ok(Self (#(#field_defaults)*))
                }
            }
            Fields::Unit => {
                quote! {
                    Ok(Self)
                }
            }
        },
        Data::Enum(data) => {
            if data.variants.is_empty() {
                return syn::Error::new_spanned(
                    name,
                    "#[derive(TryDefault)] on empty enums is not supported",
                )
                .to_compile_error()
                .into();
            }
            let first_variant = &data.variants[0];
            let variant_ident = &first_variant.ident;

            match &first_variant.fields {
                Fields::Named(fields) => {
                    let field_defaults = fields.named.iter().map(|field| {
                        let ident = &field.ident;
                        quote! {
                            #ident: ::rustyfill::try_default::TryDefault::try_default()?,
                        }
                    });
                    quote! {
                        Ok(Self::#variant_ident { #(#field_defaults)* })
                    }
                }
                Fields::Unnamed(fields) => {
                    let field_defaults = fields.unnamed.iter().map(|field| {
                        let ty = &field.ty;
                        quote! {
                            <#ty as ::rustyfill::try_default::TryDefault>::try_default()?,
                        }
                    });
                    quote! {
                        Ok(Self::#variant_ident (#(#field_defaults)*))
                    }
                }
                Fields::Unit => {
                    quote! {
                        Ok(Self::#variant_ident)
                    }
                }
            }
        }
        Data::Union(_) => {
            return syn::Error::new_spanned(
                name,
                "#[derive(TryDefault)] is not supported on unions",
            )
            .to_compile_error()
            .into();
        }
    };

    let expanded = quote! {
        impl #impl_generics ::rustyfill::try_default::TryDefault for #name #ty_generics #where_clause {
            fn try_default() -> Result<Self, ::rustyfill::try_default::TryDefaultError> {
                #default_body
            }
        }
    };

    TokenStream::from(expanded)
}

/// Generates `TryDefault` implementations for tuples of arities 1 through `max`.
#[proc_macro]
pub fn try_default_tuples(input: TokenStream) -> TokenStream {
    let max: usize = syn::parse::<syn::LitInt>(input)
        .ok()
        .and_then(|lit| lit.base10_parse().ok())
        .unwrap_or(12);
    let max = max.clamp(1, 16);

    let mut output = Vec::new();

    for arity in 1..=max {
        let type_params: Vec<_> = (0..arity).map(|i| quote::format_ident!("T{i}")).collect();
        let bounds: Vec<_> = type_params.iter().map(|t| quote!(#t: TryDefault)).collect();

        let types_joined: proc_macro2::TokenStream = {
            let mut ts = proc_macro2::TokenStream::new();
            for (i, tp) in type_params.iter().enumerate() {
                if i > 0 { ts.extend(quote!(,)); }
                ts.extend(quote!(#tp));
            }
            ts
        };

        let tuple_pat = if arity == 1 {
            quote!((#types_joined ,))
        } else {
            quote!((#types_joined))
        };

        let fields_joined: proc_macro2::TokenStream = {
            let mut ts = proc_macro2::TokenStream::new();
            (0..arity).for_each(|type_params_index| {
                if type_params_index > 0 { ts.extend(quote!(,)); }
                let tp = &type_params[type_params_index];
                ts.extend(quote!(<#tp as TryDefault>::try_default()?));
            });
            ts
        };

        let body = if arity == 1 {
            quote!(((#fields_joined ),))
        } else {
            quote!((#fields_joined))
        };

        output.push(quote! {
            impl<#(#bounds),*> TryDefault for #tuple_pat {
                #[inline]
                fn try_default() -> Result<Self, TryDefaultError> {
                    Ok(#body)
                }
            }
        });
    }

    TokenStream::from(quote!(#(#output)*))
}
