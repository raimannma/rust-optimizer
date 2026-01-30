//! Parameter value storage types.

/// Represents a sampled parameter value.
///
/// This enum stores different parameter value types uniformly.
/// For categorical parameters, the `Categorical` variant stores
/// the index into the choices array.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamValue {
    /// A floating-point parameter value.
    Float(f64),
    /// An integer parameter value.
    Int(i64),
    /// A categorical parameter value, stored as an index into the choices array.
    Categorical(usize),
}
