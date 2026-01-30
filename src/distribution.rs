//! Parameter distribution types.

/// Distribution for floating-point parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct FloatDistribution {
    /// Lower bound (inclusive).
    pub low: f64,
    /// Upper bound (inclusive).
    pub high: f64,
    /// Whether to sample in log space.
    pub log_scale: bool,
    /// Optional step size for discretization.
    pub step: Option<f64>,
}

/// Distribution for integer parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct IntDistribution {
    /// Lower bound (inclusive).
    pub low: i64,
    /// Upper bound (inclusive).
    pub high: i64,
    /// Whether to sample in log space.
    pub log_scale: bool,
    /// Optional step size for discretization.
    pub step: Option<i64>,
}

/// Distribution for categorical parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct CategoricalDistribution {
    /// Number of choices available.
    pub n_choices: usize,
}

/// Enum wrapping all parameter distribution types.
#[derive(Clone, Debug, PartialEq)]
pub enum Distribution {
    /// A floating-point distribution.
    Float(FloatDistribution),
    /// An integer distribution.
    Int(IntDistribution),
    /// A categorical distribution.
    Categorical(CategoricalDistribution),
}
