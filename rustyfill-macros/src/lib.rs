//! Procedural macros for [`rustyfill`](https://docs.rs/rustyfill).

extern crate proc_macro;

mod try_clone;
mod try_debug;
mod try_default;
mod try_format_args;

use proc_macro::TokenStream;
use quote::quote;

/// Derives `TryClone` for a struct or enum.
///
/// All fields in every variant must themselves implement `TryClone`. The generated
/// implementation clones each field fallibly and propagates the first error encountered.
#[proc_macro_derive(TryClone)]
pub fn derive_try_clone(input: TokenStream) -> TokenStream {
    try_clone::derive_try_clone(input)
}

/// Generates `TryClone` implementations for tuples of arities 1 through `max`.
#[proc_macro]
pub fn try_clone_tuples(input: TokenStream) -> TokenStream {
    try_clone::try_clone_tuples(input)
}

/// Derives `TryDebug` for a struct or enum.
///
/// All fields in every variant must themselves implement `TryDebug`. The generated
/// implementation formats each field fallibly via `try_fmt` and propagates the first
/// error encountered. Unlike `TryDisplay`, this is safe to derive because the macro
/// generates the formatting code directly and can guarantee no hidden allocations.
#[proc_macro_derive(TryDebug)]
pub fn derive_try_debug(input: TokenStream) -> TokenStream {
    try_debug::derive_try_debug(input)
}

/// Generates `TryDebug` implementations for tuples of arities 1 through `max`.
#[proc_macro]
pub fn try_debug_tuples(input: TokenStream) -> TokenStream {
    try_debug::try_debug_tuples(input)
}

/// Derives `TryDefault` for a struct or enum.
///
/// All fields in every variant must themselves implement `TryDefault`. The generated
/// implementation constructs each field fallibly and propagates the first error encountered.
#[proc_macro_derive(TryDefault)]
pub fn derive_try_default(input: TokenStream) -> TokenStream {
    try_default::derive_try_default(input)
}

/// Generates `TryDefault` implementations for tuples of arities 1 through `max`.
#[proc_macro]
pub fn try_default_tuples(input: TokenStream) -> TokenStream {
    try_default::try_default_tuples(input)
}

/// Produces a [`core::fmt::Arguments`] value identical to what
/// [`core::format_args!`] would produce, except that every argument used as
/// a formatted value is wrapped in the appropriate wrapper type
/// ([`TryDebugWrapper`], [`TryDisplayWrapper`], [`TryLowerHexWrapper`], or
/// [`TryUpperHexWrapper`]) so that formatting routes through the fallible
/// [`TryDebug`] / [`TryDisplay`] / [`TryLowerHex`] / [`TryUpperHex`] paths.
///
/// The wrapper is selected based on the trailing format character of each
/// placeholder: `?` → Debug, `x` → LowerHex, `X` → UpperHex, everything else
/// (including bare `{}`) → Display.
///
/// Arguments that are *only* referenced as width or precision specifiers
/// (e.g., `"{val:<width$}"`) are passed through unwrapped, since those
/// positions expect raw numeric values rather than formatted output.
#[proc_macro]
pub fn try_format_args(input: TokenStream) -> TokenStream {
    try_format_args::try_format_args(input)
}

/// Fallibly print with a newline, writing to `std::io::stdout()`.
#[proc_macro]
pub fn try_println(input: TokenStream) -> TokenStream {
    let tokens: proc_macro2::TokenStream = input.clone().into();
    if tokens.is_empty() {
        return quote! { std::write!(std::io::stdout(), "\n") }.into();
    }
    quote! {{
        let mut out = std::io::stdout().lock();
        std::io::Write::write_fmt(
            &mut out,
            ::rustyfill::try_format_args!(#tokens),
        )
    }}
    .into()
}

/// Fallibly print without a newline, writing to `std::io::stdout()`.
#[proc_macro]
pub fn try_print(input: TokenStream) -> TokenStream {
    let tokens: proc_macro2::TokenStream = input.clone().into();
    if tokens.is_empty() {
        return quote! { Ok::<_, std::io::Error>(()) }.into();
    }
    quote! {{
        let mut out = std::io::stdout().lock();
        std::io::Write::write_fmt(
            &mut out,
            ::rustyfill::try_format_args!(#tokens),
        )
    }}
    .into()
}

/// Fallibly write formatted output to a writer.
#[proc_macro]
pub fn try_write(input: TokenStream) -> TokenStream {
    let tokens: proc_macro2::TokenStream = input.clone().into();
    let tts: Vec<proc_macro2::TokenTree> = tokens.clone().into_iter().collect();

    let mut depth = 0i32;
    let mut comma_pos = None;
    for (i, tt) in tts.iter().enumerate() {
        match tt {
            proc_macro2::TokenTree::Group(_) => {
                depth += 1;
            }
            proc_macro2::TokenTree::Punct(p) if p.as_char() == ',' && depth == 0 => {
                comma_pos = Some(i);
                break;
            }
            _ => {}
        }
    }

    match comma_pos {
        None => {
            let dst_ts: proc_macro2::TokenStream = tts.into_iter().collect();
            quote! {{
                let _ = #dst_ts;
                Ok::<_, std::io::Error>(())
            }}
            .into()
        }
        Some(idx) => {
            let dst_ts: proc_macro2::TokenStream = tts[..idx].iter().cloned().collect();
            let fmt_ts: proc_macro2::TokenStream = tts[idx + 1..].iter().cloned().collect();
            quote! {{
                let mut dst = #dst_ts;
                std::io::Write::write_fmt(
                    &mut dst,
                    ::rustyfill::try_format_args!(#fmt_ts),
                )
            }}
            .into()
        }
    }
}

/// Fallibly write formatted output with a newline to a writer.
#[proc_macro]
pub fn try_writeln(input: TokenStream) -> TokenStream {
    let tokens: proc_macro2::TokenStream = input.clone().into();
    let tts: Vec<proc_macro2::TokenTree> = tokens.clone().into_iter().collect();

    let mut depth = 0i32;
    let mut comma_pos = None;
    for (i, tt) in tts.iter().enumerate() {
        match tt {
            proc_macro2::TokenTree::Group(_) => {
                depth += 1;
            }
            proc_macro2::TokenTree::Punct(p) if p.as_char() == ',' && depth == 0 => {
                comma_pos = Some(i);
                break;
            }
            _ => {}
        }
    }

    match comma_pos {
        None => {
            let dst_ts: proc_macro2::TokenStream = tts.into_iter().collect();
            quote! { std::write!(#dst_ts, "\n") }.into()
        }
        Some(idx) => {
            let dst_ts: proc_macro2::TokenStream = tts[..idx].iter().cloned().collect();
            let fmt_ts: proc_macro2::TokenStream = tts[idx + 1..].iter().cloned().collect();
            quote! {{
                let mut dst = #dst_ts;
                std::io::Write::write_fmt(&mut dst, ::rustyfill::try_format_args!(#fmt_ts))
                    .and_then(|()| std::io::Write::write_all(&mut dst, b"\n"))
            }}
            .into()
        }
    }
}

/// Fallibly format arguments into a newly allocated `String`.
#[proc_macro]
pub fn try_format(input: TokenStream) -> TokenStream {
    let tokens: proc_macro2::TokenStream = input.clone().into();
    quote! {{
        let mut buf = String::new();
        ::rustyfill::string::TryString::try_write_fmt(
            &mut buf,
            ::rustyfill::try_format_args!(#tokens),
        ).map(|()| buf)
    }}
    .into()
}
