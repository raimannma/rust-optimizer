#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Returned when the lower bound is greater than the upper bound.
    #[error("invalid bounds: low ({low}) must be less than or equal to high ({high})")]
    InvalidBounds {
        /// The lower bound value.
        low: f64,
        /// The upper bound value.
        high: f64,
    },

    /// Returned when log scale is used with non-positive bounds.
    #[error("invalid log bounds: low must be positive for log scale")]
    InvalidLogBounds,

    /// Returned when step size is not positive.
    #[error("invalid step: step must be positive")]
    InvalidStep,

    /// Returned when categorical choices are empty.
    #[error("categorical choices cannot be empty")]
    EmptyChoices,

    /// Returned when a parameter is suggested with a different configuration.
    #[error("parameter conflict for '{name}': {reason}")]
    ParameterConflict {
        /// The name of the conflicting parameter.
        name: String,
        /// The reason for the conflict.
        reason: String,
    },

    /// Returned when requesting the best trial but no trials have completed.
    #[error("no completed trials available")]
    NoCompletedTrials,

    /// Returned when gamma is not in the valid range (0.0, 1.0).
    #[error("invalid gamma: {0} must be in (0.0, 1.0)")]
    InvalidGamma(f64),

    /// Returned when bandwidth is not positive.
    #[error("invalid bandwidth: {0} must be positive")]
    InvalidBandwidth(f64),

    /// Returned when KDE is created with empty samples.
    #[error("KDE requires at least one sample")]
    EmptySamples,

    /// Returned when multivariate KDE samples have zero dimensions.
    #[error("multivariate KDE samples must have at least one dimension")]
    ZeroDimensions,

    /// Returned when multivariate KDE samples have inconsistent dimensions.
    #[error(
        "dimension mismatch: expected {expected} dimensions but sample {sample_index} has {got}"
    )]
    DimensionMismatch {
        /// The expected number of dimensions.
        expected: usize,
        /// The actual number of dimensions in the sample.
        got: usize,
        /// The index of the sample with mismatched dimensions.
        sample_index: usize,
    },

    /// Returned when bandwidth vector length doesn't match the number of dimensions.
    #[error("bandwidth dimension mismatch: expected {expected} bandwidths but got {got}")]
    BandwidthDimensionMismatch {
        /// The expected number of bandwidths.
        expected: usize,
        /// The actual number of bandwidths provided.
        got: usize,
    },

    /// Returned when a trial is pruned (stopped early by the objective function).
    #[error("trial was pruned")]
    TrialPruned,

    /// Returned when the objective returns the wrong number of values.
    #[error("objective dimension mismatch: expected {expected} values, got {got}")]
    ObjectiveDimensionMismatch {
        /// The expected number of objective values.
        expected: usize,
        /// The actual number of objective values returned.
        got: usize,
    },

    /// Returned when an internal invariant is violated.
    #[error("internal error: {0}")]
    Internal(&'static str),

    /// Returned when an async task fails.
    #[cfg(feature = "async")]
    #[error("async task error: {0}")]
    TaskError(String),

    /// Returned when a storage operation fails.
    #[cfg(feature = "journal")]
    #[error("storage error: {0}")]
    Storage(String),
}

pub type Result<T> = core::result::Result<T, Error>;

/// Convenience type for signalling a pruned trial from an objective function.
///
/// Implements `Into<Error>` so it can be used with `?` in objectives that
/// return `Result<V, Error>`.
///
/// # Examples
///
/// ```
/// use optimizer::{Error, TrialPruned};
///
/// fn objective_that_prunes() -> Result<f64, Error> {
///     // ... some computation ...
///     Err(TrialPruned)?
/// }
/// ```
#[derive(Debug)]
pub struct TrialPruned;

impl core::fmt::Display for TrialPruned {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "trial was pruned")
    }
}

impl From<TrialPruned> for Error {
    fn from(_: TrialPruned) -> Self {
        Error::TrialPruned
    }
}
