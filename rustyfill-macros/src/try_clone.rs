use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

pub(crate) fn derive_try_clone(input: TokenStream) -> TokenStream {
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

pub(crate) fn try_clone_tuples(input: TokenStream) -> TokenStream {
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

        let fields_joined: proc_macro2::TokenStream = {
            let mut ts = proc_macro2::TokenStream::new();
            for i in 0..arity {
                if i > 0 {
                    ts.extend(quote!(,));
                }
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
