use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

pub(crate) fn derive_try_debug(input: TokenStream) -> TokenStream {
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

pub(crate) fn try_debug_tuples(input: TokenStream) -> TokenStream {
    let max: usize = syn::parse::<syn::LitInt>(input)
        .ok()
        .and_then(|lit| lit.base10_parse().ok())
        .unwrap_or(12);
    let max = max.clamp(1, 16);

    let mut output = Vec::new();

    for arity in 1..=max {
        let type_params: Vec<_> = (0..arity).map(|i| quote::format_ident!("T{i}")).collect();
        let bounds: Vec<_> = type_params.iter().map(|t| quote!(#t: TryDebug)).collect();

        let types_joined: proc_macro2::TokenStream = {
            let mut ts = proc_macro2::TokenStream::new();
            for (i, tp) in type_params.iter().enumerate() {
                if i > 0 {
                    ts.extend(quote!(,));
                }
                ts.extend(quote!(#tp));
            }
            ts
        };

        let tuple_pat = if arity == 1 {
            quote!((#types_joined ,))
        } else {
            quote!((#types_joined))
        };

        let body_lines: Vec<_> = (0..arity)
            .map(|i| {
                let idx = syn::Index::from(i);
                quote! {
                    if #i > 0 { f.write_str(", ")?; }
                    TryDebug::try_fmt(&self.#idx, f)?;
                }
            })
            .collect();

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
