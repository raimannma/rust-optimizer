//! Percentile pruner — prune trials outside the top N% at each step.
//!
//! A generalization of [`MedianPruner`](super::MedianPruner) that lets you
//! control how aggressively to prune. At each step, the current trial's
//! intermediate value is compared against the given percentile of all
//! completed trials' values at the same step.
//!
//! # When to use
//!
//! - When you want finer control over pruning aggressiveness than median pruning
//! - Lower percentiles (e.g., 25%) are more aggressive — only keep the best quarter
//! - Higher percentiles (e.g., 75%) are more lenient — keep the top three quarters
//! - Percentile 50% is equivalent to [`MedianPruner`](super::MedianPruner)
//!
//! # Configuration
//!
//! | Option | Default | Description |
//! |--------|---------|-------------|
//! | `percentile` | *(required)* | Keep trials in the top N% — range `(0, 100)` |
//! | `n_warmup_steps` | 0 | Skip pruning in the first N steps |
//! | `n_min_trials` | 1 | Require at least N completed trials before pruning |
//!
//! # Example
//!
//! ```
//! use optimizer::Direction;
//! use optimizer::pruner::PercentilePruner;
//!
//! // Keep only the top 25% of trials (aggressive pruning)
//! let pruner = PercentilePruner::new(25.0, Direction::Minimize)
//!     .n_warmup_steps(5)
//!     .n_min_trials(3);
//! ```

use super::Pruner;
use crate::sampler::CompletedTrial;
use crate::types::{Direction, TrialState};

/// Prune trials that are not in the top `percentile`% of completed trials
/// at the same training step.
///
/// `PercentilePruner::new(50.0, direction)` is equivalent to `MedianPruner`.
/// `PercentilePruner::new(25.0, direction)` keeps only the top 25% of trials.
///
/// # Examples
///
/// ```
/// use optimizer::Direction;
/// use optimizer::pruner::PercentilePruner;
///
/// // Keep only the top 25% of trials (aggressive pruning)
/// let pruner = PercentilePruner::new(25.0, Direction::Minimize)
///     .n_warmup_steps(5)
///     .n_min_trials(3);
/// ```
pub struct PercentilePruner {
    /// Keep trials in the top `percentile`%. Range: (0.0, 100.0).
    percentile: f64,
    /// Don't prune in the first N steps (let the trial warm up).
    n_warmup_steps: u64,
    /// Require at least N completed trials before pruning.
    n_min_trials: usize,
    /// The optimization direction.
    direction: Direction,
}

impl PercentilePruner {
    /// Create a new `PercentilePruner` for the given percentile and direction.
    ///
    /// The `percentile` value must be in `(0.0, 100.0)`.
    /// A percentile of 50.0 is equivalent to median pruning.
    ///
    /// # Panics
    ///
    /// Panics if `percentile` is not in `(0.0, 100.0)`.
    #[must_use]
    pub fn new(percentile: f64, direction: Direction) -> Self {
        assert!(
            percentile > 0.0 && percentile < 100.0,
            "percentile must be in (0.0, 100.0), got {percentile}"
        );
        Self {
            percentile,
            n_warmup_steps: 0,
            n_min_trials: 1,
            direction,
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

impl Pruner for PercentilePruner {
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

        // 4. Compute percentile threshold
        let threshold = compute_percentile(&mut values_at_step, self.percentile);

        // 5. Compare against threshold based on direction
        match self.direction {
            Direction::Minimize => current_value > threshold,
            Direction::Maximize => current_value < threshold,
        }
    }
}

/// Compute the given percentile of a non-empty slice. Sorts the slice in place.
///
/// Uses linear interpolation between the two nearest ranks.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub(crate) fn compute_percentile(values: &mut [f64], percentile: f64) -> f64 {
    assert!(!values.is_empty(), "compute_percentile: empty input");
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let len = values.len();
    if len == 1 {
        return values[0];
    }
    // Rank in [0, len-1] range
    let rank = percentile / 100.0 * (len - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let frac = rank - lower as f64;
        values[lower] * (1.0 - frac) + values[upper] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_percentile_median_odd() {
        // Percentile 50 on odd-length slice = median
        let val = compute_percentile(&mut [3.0, 1.0, 2.0], 50.0);
        assert!((val - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_percentile_median_even() {
        // Percentile 50 on even-length slice = median (interpolated)
        let val = compute_percentile(&mut [4.0, 1.0, 3.0, 2.0], 50.0);
        assert!((val - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_percentile_25() {
        // [1.0, 2.0, 3.0, 4.0], rank = 0.25 * 3 = 0.75
        // interpolate: 1.0 * 0.25 + 2.0 * 0.75 = 1.75
        let val = compute_percentile(&mut [4.0, 1.0, 3.0, 2.0], 25.0);
        assert!((val - 1.75).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_percentile_75() {
        // [1.0, 2.0, 3.0, 4.0], rank = 0.75 * 3 = 2.25
        // interpolate: 3.0 * 0.75 + 4.0 * 0.25 = 3.25
        let val = compute_percentile(&mut [4.0, 1.0, 3.0, 2.0], 75.0);
        assert!((val - 3.25).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_percentile_single() {
        let val = compute_percentile(&mut [5.0], 50.0);
        assert!((val - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "percentile must be in (0.0, 100.0)")]
    fn new_rejects_zero() {
        let _ = PercentilePruner::new(0.0, Direction::Minimize);
    }

    #[test]
    #[should_panic(expected = "percentile must be in (0.0, 100.0)")]
    fn new_rejects_hundred() {
        let _ = PercentilePruner::new(100.0, Direction::Minimize);
    }

    fn make_completed_trial(id: u64, values: &[(u64, f64)]) -> CompletedTrial {
        use std::collections::HashMap;

        use crate::parameter::ParamId;

        CompletedTrial::with_intermediate_values(
            id,
            HashMap::<ParamId, crate::parameter::ParamValue>::new(),
            HashMap::new(),
            HashMap::new(),
            0.0,
            values.to_vec(),
            HashMap::new(),
        )
    }

    #[test]
    fn percentile_50_matches_median_behavior() {
        let pruner = PercentilePruner::new(50.0, Direction::Minimize);
        let completed = vec![
            make_completed_trial(0, &[(0, 1.0), (1, 2.0)]),
            make_completed_trial(1, &[(0, 3.0), (1, 4.0)]),
            make_completed_trial(2, &[(0, 5.0), (1, 6.0)]),
        ];
        // Median at step 1 is 4.0
        // Value 5.0 > 4.0 → prune
        assert!(pruner.should_prune(3, 1, &[(0, 3.0), (1, 5.0)], &completed));
        // Value 3.0 < 4.0 → keep
        assert!(!pruner.should_prune(3, 1, &[(0, 3.0), (1, 3.0)], &completed));
    }

    #[test]
    fn percentile_25_is_more_aggressive() {
        let pruner_25 = PercentilePruner::new(25.0, Direction::Minimize);
        let pruner_75 = PercentilePruner::new(75.0, Direction::Minimize);
        let completed = vec![
            make_completed_trial(0, &[(0, 1.0)]),
            make_completed_trial(1, &[(0, 2.0)]),
            make_completed_trial(2, &[(0, 3.0)]),
            make_completed_trial(3, &[(0, 4.0)]),
        ];
        // 25th percentile at step 0: 1.75
        // 75th percentile at step 0: 3.25
        // Value 2.5: above 25th (prune), below 75th (keep)
        assert!(pruner_25.should_prune(4, 0, &[(0, 2.5)], &completed));
        assert!(!pruner_75.should_prune(4, 0, &[(0, 2.5)], &completed));
    }

    #[test]
    fn warmup_prevents_pruning() {
        let pruner = PercentilePruner::new(50.0, Direction::Minimize).n_warmup_steps(5);
        let completed = vec![make_completed_trial(0, &[(0, 1.0)])];
        // Step 3 < warmup 5 → no prune even with bad value
        assert!(!pruner.should_prune(1, 3, &[(3, 100.0)], &completed));
    }

    #[test]
    fn n_min_trials_prevents_pruning() {
        let pruner = PercentilePruner::new(50.0, Direction::Minimize).n_min_trials(5);
        let completed = vec![
            make_completed_trial(0, &[(0, 1.0)]),
            make_completed_trial(1, &[(0, 2.0)]),
        ];
        // Only 2 trials, need 5 → no prune
        assert!(!pruner.should_prune(2, 0, &[(0, 100.0)], &completed));
    }

    #[test]
    fn maximize_direction() {
        let pruner = PercentilePruner::new(50.0, Direction::Maximize);
        let completed = vec![
            make_completed_trial(0, &[(0, 1.0)]),
            make_completed_trial(1, &[(0, 3.0)]),
            make_completed_trial(2, &[(0, 5.0)]),
        ];
        // Median at step 0 is 3.0
        // Value 2.0 < 3.0 → prune (maximize wants higher)
        assert!(pruner.should_prune(3, 0, &[(0, 2.0)], &completed));
        // Value 4.0 > 3.0 → keep
        assert!(!pruner.should_prune(3, 0, &[(0, 4.0)], &completed));
    }

    #[test]
    fn near_boundary_percentiles() {
        let pruner_low = PercentilePruner::new(1.0, Direction::Minimize);
        let pruner_high = PercentilePruner::new(99.0, Direction::Minimize);
        let completed = vec![
            make_completed_trial(0, &[(0, 1.0)]),
            make_completed_trial(1, &[(0, 2.0)]),
            make_completed_trial(2, &[(0, 3.0)]),
            make_completed_trial(3, &[(0, 100.0)]),
        ];
        // Percentile 1 is very aggressive (threshold near 1.0)
        // Value 1.5 should be pruned
        assert!(pruner_low.should_prune(4, 0, &[(0, 1.5)], &completed));
        // Percentile 99 is very lenient (threshold near 100.0)
        // Value 50.0 should not be pruned
        assert!(!pruner_high.should_prune(4, 0, &[(0, 50.0)], &completed));
    }
}
