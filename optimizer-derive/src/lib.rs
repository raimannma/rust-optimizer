use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

/// Derive macro for the `Categorical` trait on fieldless enums.
///
/// Generates an implementation of `optimizer::Categorical` that maps
/// enum variants to/from sequential indices.
///
/// # Example
///
/// ```ignore
/// use optimizer::Categorical;
///
/// #[derive(Clone, Categorical)]
/// enum Color {
///     Red,
///     Green,
///     Blue,
/// }
/// ```
#[proc_macro_derive(Categorical)]
pub fn derive_categorical(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let Data::Enum(data_enum) = &input.data else {
        return syn::Error::new_spanned(&input, "Categorical can only be derived for enums")
            .to_compile_error()
            .into();
    };

    // Validate all variants are fieldless
    for variant in &data_enum.variants {
        if !matches!(variant.fields, Fields::Unit) {
            return syn::Error::new_spanned(
                variant,
                "Categorical can only be derived for enums with unit variants (no fields)",
            )
            .to_compile_error()
            .into();
        }
    }

    let n_choices = data_enum.variants.len();
    let variant_names: Vec<_> = data_enum.variants.iter().map(|v| &v.ident).collect();
    let indices: Vec<usize> = (0..n_choices).collect();

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics optimizer::Categorical for #name #ty_generics #where_clause {
            const N_CHOICES: usize = #n_choices;

            fn from_index(index: usize) -> Self {
                match index {
                    #(#indices => #name::#variant_names,)*
                    _ => panic!("invalid index {} for {} with {} variants", index, stringify!(#name), #n_choices),
                }
            }

            fn to_index(&self) -> usize {
                match self {
                    #(#name::#variant_names => #indices,)*
                }
            }
        }
    };

    expanded.into()
}
