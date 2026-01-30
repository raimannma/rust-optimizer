//! Core types for the optimizer library.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The direction of optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Direction {
    /// Minimize the objective value.
    Minimize,
    /// Maximize the objective value.
    Maximize,
}

/// The state of a trial in its lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TrialState {
    /// The trial is currently running.
    Running,
    /// The trial completed successfully.
    Complete,
    /// The trial failed with an error.
    Failed,
}
