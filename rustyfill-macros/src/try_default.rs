use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

pub(crate) fn derive_try_default(input: TokenStream) -> TokenStream {
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

pub(crate) fn try_default_tuples(input: TokenStream) -> TokenStream {
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
            (0..arity).for_each(|type_params_index| {
                if type_params_index > 0 {
                    ts.extend(quote!(,));
                }
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
