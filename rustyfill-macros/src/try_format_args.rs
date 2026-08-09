use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use std::collections::BTreeMap;

pub(crate) fn try_format_args(input: TokenStream) -> TokenStream {
    match TryFormatArgsInput::parse(input.clone()) {
        Ok(result) => result.emit().into(),
        Err(err) => err.to_compile_error().into(),
    }
}

struct TryFormatArgsInput {
    format_string: String,
    args: Vec<(Option<String>, proc_macro2::TokenStream)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotUse {
    Display,
    Measure,
    Both,
}

impl TryFormatArgsInput {
    fn parse(input: TokenStream) -> syn::Result<Self> {
        let tokens: proc_macro2::TokenStream = input.into();
        let tts: Vec<proc_macro2::TokenTree> = tokens.clone().into_iter().collect();
        let mut pos = 0;

        // First token must be a string literal.
        if pos >= tts.len() {
            return Err(syn::Error::new_spanned(
                tokens,
                "expected format string literal",
            ));
        }

        let format_string = if let proc_macro2::TokenTree::Literal(ref lit) = tts[pos] {
            let lit_str: syn::LitStr = syn::parse2(lit.to_token_stream()).map_err(|_| {
                syn::Error::new_spanned(lit.to_token_stream(), "expected a string literal")
            })?;
            pos += 1;
            lit_str.value()
        } else {
            return Err(syn::Error::new_spanned(
                tts[pos].to_token_stream(),
                "expected a string literal",
            ));
        };

        let mut args = Vec::new();

        while pos < tts.len() {
            // Skip commas
            if let proc_macro2::TokenTree::Punct(ref p) = tts[pos] {
                if p.as_char() == ',' {
                    pos += 1;
                    continue;
                }
            }

            if pos >= tts.len() {
                break;
            }

            // Collect arg tokens until top-level comma or EOF
            let (arg_trees, end_pos) = collect_arg_trees(&tts, pos);
            if arg_trees.is_empty() {
                break;
            }
            let arg_tokens: proc_macro2::TokenStream = arg_trees.iter().cloned().collect();
            let (name, expr_ts) = parse_arg(&arg_tokens).unwrap_or((None, arg_tokens));
            args.push((name, expr_ts));
            pos = end_pos;
        }

        Ok(Self {
            format_string,
            args,
        })
    }

    fn emit(&self) -> proc_macro2::TokenStream {
        let fmt_str = &self.format_string;
        let analysis = analyze_format_string(fmt_str);
        let has_auto = analysis.has_auto;

        let mut formatted_args = Vec::new();

        for (i, (maybe_name, expr_ts)) in self.args.iter().enumerate() {
            let usage = match maybe_name.as_ref() {
                Some(name) => analysis.named.get(name.as_str()).copied(),
                None => {
                    if has_auto {
                        Some(SlotUse::Display)
                    } else {
                        analysis.positional.get(&i).copied()
                    }
                }
            };
            let needs_wrap = matches!(usage, Some(SlotUse::Display) | Some(SlotUse::Both));

            if let Some(name) = maybe_name {
                let name_ident = syn::Ident::new(name, proc_macro2::Span::call_site());
                if needs_wrap {
                    formatted_args.push(quote! {
                        #name_ident = ::rustyfill::try_fmt::TryFmt::new(#expr_ts)
                    });
                } else {
                    formatted_args.push(quote! {
                        #name_ident = #expr_ts
                    });
                }
            } else {
                if needs_wrap {
                    formatted_args.push(quote! {
                        ::rustyfill::try_fmt::TryFmt::new(#expr_ts)
                    });
                } else {
                    formatted_args.push(expr_ts.clone());
                }
            }
        }

        quote! {
            core::format_args!(#fmt_str #(#formatted_args),*)
        }
    }
}

/// Collect token trees for one argument from position `start` until a top-level comma or EOF.
/// Returns the collected trees and the position after the consumed range.
fn collect_arg_trees(
    tts: &[proc_macro2::TokenTree],
    start: usize,
) -> (Vec<proc_macro2::TokenTree>, usize) {
    let mut collected = Vec::new();
    let mut depth: i32 = 0;
    let mut pos = start;

    while pos < tts.len() {
        match &tts[pos] {
            proc_macro2::TokenTree::Group(_) => {
                depth += 1;
                collected.push(tts[pos].clone());
                pos += 1;
            }
            proc_macro2::TokenTree::Punct(p) if p.as_char() == ',' && depth == 0 => {
                // Skip the comma too
                pos += 1;
                break;
            }
            _ => {
                collected.push(tts[pos].clone());
                pos += 1;
            }
        }
    }

    (collected, pos)
}

fn parse_arg(
    tokens: &proc_macro2::TokenStream,
) -> syn::Result<(Option<String>, proc_macro2::TokenStream)> {
    let mut depth = 0u32;
    let mut eq_pos_found = false;
    let mut first_ident = None;

    for tt in tokens.clone().into_iter() {
        match &tt {
            proc_macro2::TokenTree::Group(_) => {
                depth += 1;
            }
            proc_macro2::TokenTree::Punct(p) if p.as_char() == '=' && depth == 0 => {
                eq_pos_found = true;
                break;
            }
            proc_macro2::TokenTree::Ident(id) if first_ident.is_none() && depth == 0 => {
                first_ident = Some(id.clone());
            }
            _ => {}
        }
    }

    if eq_pos_found {
        if let Some(ident) = first_ident {
            let name = ident.to_string();
            let expr_tokens = split_after_eq(tokens);
            return Ok((Some(name), expr_tokens));
        }
    }

    Ok((None, tokens.clone()))
}

fn split_after_eq(tokens: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let mut out = proc_macro2::TokenStream::new();
    let mut past_eq = false;

    for tt in tokens.clone() {
        match &tt {
            proc_macro2::TokenTree::Group(_) => {
                if past_eq {
                    out.extend(proc_macro2::TokenStream::from(tt));
                }
            }
            proc_macro2::TokenTree::Punct(p) if p.as_char() == '=' => {
                past_eq = true;
            }
            _ => {
                if past_eq {
                    out.extend(proc_macro2::TokenStream::from(tt));
                }
            }
        }
    }

    out
}

/* ── Format string analysis ─────────────────────────────────────────────────── */

struct FormatAnalysis {
    positional: BTreeMap<usize, SlotUse>,
    named: BTreeMap<String, SlotUse>,
    has_auto: bool,
}

fn analyze_format_string(s: &str) -> FormatAnalysis {
    let mut positional = BTreeMap::new();
    let mut named = BTreeMap::new();
    let mut has_auto = false;
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c != '{' {
            continue;
        }
        if chars.peek() == Some(&'{') {
            chars.next();
            continue;
        }

        let kind = parse_arg_spec(&mut chars);
        if matches!(kind, ArgKind::Auto) {
            has_auto = true;
        }

        if chars.peek() == Some(&':') {
            chars.next();
            parse_format_spec(&mut chars, &mut positional, &mut named);
        }

        let _ = chars.next();

        match kind {
            ArgKind::Positional(idx) => {
                merge_usage_p(&mut positional, idx, SlotUse::Display);
            }
            ArgKind::Named(name) => {
                merge_usage_n(&mut named, name, SlotUse::Display);
            }
            ArgKind::Auto => {}
        }
    }

    FormatAnalysis {
        positional,
        named,
        has_auto,
    }
}

#[derive(Debug)]
enum ArgKind {
    Positional(usize),
    Named(String),
    Auto,
}

fn parse_arg_spec<I>(chars: &mut std::iter::Peekable<I>) -> ArgKind
where
    I: Iterator<Item = char>,
{
    let Some(&c) = chars.peek() else {
        return ArgKind::Auto;
    };

    if c == ':' || c == '}' {
        return ArgKind::Auto;
    }

    if c.is_ascii_digit() {
        let mut num = String::new();
        while let Some(&ch) = chars.peek() {
            if ch.is_ascii_digit() {
                num.push(ch);
                chars.next();
            } else {
                break;
            }
        }
        let idx = num.parse::<usize>().unwrap_or(0);
        ArgKind::Positional(idx)
    } else {
        let mut name = String::new();
        while let Some(&ch) = chars.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                name.push(ch);
                chars.next();
            } else {
                break;
            }
        }
        if name.is_empty() {
            ArgKind::Auto
        } else {
            ArgKind::Named(name)
        }
    }
}

fn parse_format_spec<I>(
    chars: &mut std::iter::Peekable<I>,
    positional: &mut BTreeMap<usize, SlotUse>,
    named: &mut BTreeMap<String, SlotUse>,
) where
    I: Iterator<Item = char>,
{
    // Alignment
    if let Some(&c) = chars.peek() {
        if c == '<' || c == '>' || c == '^' || c == '=' {
            chars.next();
        }
    }

    // Sign
    if let Some(&c) = chars.peek() {
        if c == '+' || c == '-' || c == ' ' {
            chars.next();
        }
    }

    // Alternate form '#'
    if let Some(&'#') = chars.peek() {
        chars.next();
    }

    // Zero padding '0'
    if let Some(&'0') = chars.peek() {
        chars.next();
    }

    // Width
    parse_width_or_precision(chars, positional, named);

    // Precision
    if let Some('.') = chars.peek() {
        chars.next();
        parse_width_or_precision(chars, positional, named);
    }

    // Additional format character
    if let Some(&c) = chars.peek() {
        if c != '}' {
            chars.next();
        }
    }
}

fn parse_width_or_precision<I>(
    chars: &mut std::iter::Peekable<I>,
    positional: &mut BTreeMap<usize, SlotUse>,
    named: &mut BTreeMap<String, SlotUse>,
) where
    I: Iterator<Item = char>,
{
    let Some(&c) = chars.peek() else {
        return;
    };

    if c == '*' {
        chars.next();
        return;
    }

    if c.is_ascii_digit() {
        let mut num = String::new();
        while let Some(&ch) = chars.peek() {
            if ch.is_ascii_digit() {
                num.push(ch);
                chars.next();
            } else {
                break;
            }
        }
        if let Some('$') = chars.peek() {
            chars.next();
            let idx = num.parse::<usize>().unwrap_or(0);
            merge_usage_p(positional, idx, SlotUse::Measure);
        }
    } else if c.is_alphabetic() || c == '_' {
        let mut name = String::new();
        while let Some(&ch) = chars.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                name.push(ch);
                chars.next();
            } else {
                break;
            }
        }
        if let Some('$') = chars.peek() {
            chars.next();
            merge_usage_n(named, name, SlotUse::Measure);
        }
    }
}

fn merge_usage_p(map: &mut BTreeMap<usize, SlotUse>, key: usize, new: SlotUse) {
    let merged = match map.remove(&key) {
        None => new,
        Some(SlotUse::Display) if new == SlotUse::Measure => SlotUse::Both,
        Some(SlotUse::Measure) if new == SlotUse::Display => SlotUse::Both,
        Some(e) => e,
    };
    map.insert(key, merged);
}

fn merge_usage_n(map: &mut BTreeMap<String, SlotUse>, key: String, new: SlotUse) {
    let merged = match map.remove(&key) {
        None => new,
        Some(SlotUse::Display) if new == SlotUse::Measure => SlotUse::Both,
        Some(SlotUse::Measure) if new == SlotUse::Display => SlotUse::Both,
        Some(e) => e,
    };
    map.insert(key, merged);
}
