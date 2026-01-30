//! Core types for the optimizer library.

/// The direction of optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Minimize the objective value.
    Minimize,
    /// Maximize the objective value.
    Maximize,
}

/// The state of a trial in its lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrialState {
    /// The trial is currently running.
    Running,
    /// The trial completed successfully.
    Complete,
    /// The trial failed with an error.
    Failed,
}
