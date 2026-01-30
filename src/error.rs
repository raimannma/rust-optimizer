//! Error types for the optimize library.

use thiserror::Error;

/// The error type for TPE operations.
#[derive(Debug, Error)]
pub enum TpeError {
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
}

/// A specialized Result type for TPE operations.
pub type Result<T> = std::result::Result<T, TpeError>;
