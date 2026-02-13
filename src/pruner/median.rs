//! Median pruner — the recommended default pruner for most use cases.
//!
//! At each step, the current trial's intermediate value is compared against
//! the median of all completed trials' values at the same step. Trials
//! performing worse than the median are pruned.
//!
//! This is a convenience wrapper around [`PercentilePruner`](super::PercentilePruner)
//! with a fixed percentile of 50%.
//!
//! # When to use
//!
//! - **Default choice** for any iterative objective (e.g., neural network training)
//! - Works well when intermediate values are a reasonable proxy for final performance
//! - Prunes roughly half of unpromising trials, giving a good speed/accuracy balance
//!
//! If your intermediate values are noisy, consider [`WilcoxonPruner`](super::WilcoxonPruner)
//! or wrapping this pruner in a [`PatientPruner`](super::PatientPruner).
//!
//! # Configuration
//!
//! | Option | Default | Description |
//! |--------|---------|-------------|
//! | `n_warmup_steps` | 0 | Skip pruning in the first N steps |
//! | `n_min_trials` | 1 | Require at least N completed trials before pruning |
//!
//! # Example
//!
//! ```
//! use optimizer::Direction;
//! use optimizer::pruner::MedianPruner;
//!
//! let pruner = MedianPruner::new(Direction::Minimize)
//!     .n_warmup_steps(5)
//!     .n_min_trials(3);
//! ```

use super::Pruner;
use super::percentile::compute_percentile;
use crate::sampler::CompletedTrial;
use crate::types::{Direction, TrialState};

/// Prune trials that are performing worse than the median of completed trials
/// at the same step.
///
/// This is the most commonly used pruner. It compares the current trial's
/// intermediate value at each step with the median of all completed trials'
/// values at that same step.
///
/// Equivalent to `PercentilePruner::new(50.0, direction)`.
///
/// # Examples
///
/// ```
/// use optimizer::Direction;
/// use optimizer::pruner::MedianPruner;
///
/// // Prune trials worse than median when minimizing, after 5 warmup steps
/// let pruner = MedianPruner::new(Direction::Minimize)
///     .n_warmup_steps(5)
///     .n_min_trials(3);
/// ```
pub struct MedianPruner {
    /// The optimization direction.
    direction: Direction,
    /// Don't prune in the first N steps (let the trial warm up).
    n_warmup_steps: u64,
    /// Require at least N completed trials before pruning.
    n_min_trials: usize,
}

impl MedianPruner {
    /// Create a new `MedianPruner` for the given optimization direction.
    ///
    /// By default, `n_warmup_steps` is 0 and `n_min_trials` is 1.
    #[must_use]
    pub fn new(direction: Direction) -> Self {
        Self {
            direction,
            n_warmup_steps: 0,
            n_min_trials: 1,
        }
    }

    /// Set the number of warmup steps. No pruning occurs before this step.
    #[must_use]
    pub fn n_warmup_steps(mut self, n: u64) -> Self {
        self.n_warmup_steps = n;
        self
    }

    /// Set the minimum number of completed trials required before pruning.
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
    #[must_use]
    pub fn n_min_trials(mut self, n: usize) -> Self {
        assert!(n >= 1, "n_min_trials must be >= 1, got {n}");
        self.n_min_trials = n;
        self
    }
}

impl Pruner for MedianPruner {
    fn should_prune(
        &self,
        _trial_id: u64,
        step: u64,
        intermediate_values: &[(u64, f64)],
        completed_trials: &[CompletedTrial],
    ) -> bool {
        // 1. Don't prune during warmup
        if step < self.n_warmup_steps {
            return false;
        }

        // Get the current trial's latest value
        let Some(&(_, current_value)) = intermediate_values.last() else {
            return false;
        };

        // 2. Collect values at this step from completed (non-pruned) trials
        let mut values_at_step: Vec<f64> = completed_trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .filter_map(|t| {
                t.intermediate_values
                    .iter()
                    .find(|(s, _)| *s == step)
                    .map(|(_, v)| *v)
            })
            .collect();

        // 3. Not enough trials
        if values_at_step.len() < self.n_min_trials {
            return false;
        }

        // 4. Compute median (50th percentile)
        let median = compute_percentile(&mut values_at_step, 50.0);

        // 5. Compare against median based on direction
        match self.direction {
            Direction::Minimize => current_value > median,
            Direction::Maximize => current_value < median,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_median_odd() {
        assert!((compute_percentile(&mut [3.0, 1.0, 2.0], 50.0) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_median_even() {
        assert!((compute_percentile(&mut [4.0, 1.0, 3.0, 2.0], 50.0) - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_median_single() {
        assert!((compute_percentile(&mut [5.0], 50.0) - 5.0).abs() < f64::EPSILON);
    }
}
