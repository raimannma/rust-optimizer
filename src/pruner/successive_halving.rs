//! Successive Halving (SHA) pruner — budget-aware pruning at exponential rungs.
//!
//! Trials are evaluated at exponentially-spaced "rungs" (checkpoints). At each
//! rung, only the top 1/η fraction of trials survive to the next rung. This
//! is a principled way to allocate compute budget: give many trials a small
//! budget, then progressively invest more in the best ones.
//!
//! For example, with `min_resource=1`, `max_resource=81`, `reduction_factor=3`:
//!
//! | Rung | Step | Survivors |
//! |------|------|-----------|
//! | 0 | 1 | top 1/3 |
//! | 1 | 3 | top 1/3 |
//! | 2 | 9 | top 1/3 |
//! | 3 | 27 | top 1/3 |
//! | 4 | 81 | all (full budget) |
//!
//! # When to use
//!
//! - When your objective has a natural "budget" dimension (epochs, iterations)
//! - When early performance is a reasonable predictor of final performance
//! - When you want a principled alternative to median pruning
//!
//! If you're unsure about the right `min_resource`, consider
//! [`HyperbandPruner`](super::HyperbandPruner) which runs multiple brackets
//! to hedge against that choice.
//!
//! # Configuration
//!
//! | Option | Default | Description |
//! |--------|---------|-------------|
//! | `min_resource` | 1 | Step at which the first rung is placed |
//! | `max_resource` | 81 | Full budget (final rung, no pruning) |
//! | `reduction_factor` | 3 | At each rung, keep top 1/η trials |
//! | `min_early_stopping_rate` | 0 | Skip the first N rungs |
//! | `direction` | `Minimize` | Optimization direction |
//!
//! # Example
//!
//! ```
//! use optimizer::Direction;
//! use optimizer::pruner::SuccessiveHalvingPruner;
//!
//! let pruner = SuccessiveHalvingPruner::new()
//!     .min_resource(1)
//!     .max_resource(81)
//!     .reduction_factor(3)
//!     .direction(Direction::Minimize);
//! ```

use super::Pruner;
use crate::sampler::CompletedTrial;
use crate::types::{Direction, TrialState};

/// Successive Halving pruner based on the SHA algorithm.
///
/// Trials are evaluated at exponentially-spaced "rungs". At each rung,
/// only the top 1/eta fraction of trials survive to the next rung.
///
/// For example, with `min_resource=1`, `max_resource=81`, `reduction_factor=3`:
/// - Rung 0: evaluate at step 1, keep top 1/3
/// - Rung 1: evaluate at step 3, keep top 1/3
/// - Rung 2: evaluate at step 9, keep top 1/3
/// - Rung 3: evaluate at step 27, keep top 1/3
/// - Rung 4: evaluate at step 81 (full budget)
///
/// # Examples
///
/// ```
/// use optimizer::Direction;
/// use optimizer::pruner::SuccessiveHalvingPruner;
///
/// let pruner = SuccessiveHalvingPruner::new()
///     .min_resource(1)
///     .max_resource(81)
///     .reduction_factor(3)
///     .direction(Direction::Minimize);
/// ```
pub struct SuccessiveHalvingPruner {
    min_resource: u64,
    max_resource: u64,
    reduction_factor: u64,
    min_early_stopping_rate: u64,
    direction: Direction,
}

impl SuccessiveHalvingPruner {
    /// Create a new `SuccessiveHalvingPruner` with default parameters.
    ///
    /// Defaults: `min_resource=1`, `max_resource=81`, `reduction_factor=3`,
    /// `min_early_stopping_rate=0`, `direction=Minimize`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_resource: 1,
            max_resource: 81,
            reduction_factor: 3,
            min_early_stopping_rate: 0,
            direction: Direction::Minimize,
        }
    }

    /// Set the minimum resource (budget) per trial.
    ///
    /// # Panics
    ///
    /// Panics if `r` is 0.
    #[must_use]
    pub fn min_resource(mut self, r: u64) -> Self {
        assert!(r > 0, "min_resource must be > 0, got {r}");
        self.min_resource = r;
        self
    }

    /// Set the maximum resource (budget) per trial.
    ///
    /// # Panics
    ///
    /// Panics if `r` is 0.
    #[must_use]
    pub fn max_resource(mut self, r: u64) -> Self {
        assert!(r > 0, "max_resource must be > 0, got {r}");
        self.max_resource = r;
        self
    }

    /// Set the reduction factor (eta). At each rung, the top 1/eta trials survive.
    ///
    /// # Panics
    ///
    /// Panics if `eta` is less than 2.
    #[must_use]
    pub fn reduction_factor(mut self, eta: u64) -> Self {
        assert!(eta >= 2, "reduction_factor must be >= 2, got {eta}");
        self.reduction_factor = eta;
        self
    }

    /// Set the minimum early stopping rate. Skips the first N rungs.
    #[must_use]
    pub fn min_early_stopping_rate(mut self, n: u64) -> Self {
        self.min_early_stopping_rate = n;
        self
    }

    /// Set the optimization direction.
    #[must_use]
    pub fn direction(mut self, d: Direction) -> Self {
        self.direction = d;
        self
    }

    /// Compute the rung steps: `[min_resource * eta^(s), ...]` up to `max_resource`,
    /// skipping the first `min_early_stopping_rate` rungs.
    fn rung_steps(&self) -> Vec<u64> {
        let eta = self.reduction_factor;
        let mut steps = Vec::new();
        let mut rung: u32 = 0;
        while let Some(power) = eta.checked_pow(rung) {
            let step = self.min_resource.saturating_mul(power);
            if step > self.max_resource {
                break;
            }
            if u64::from(rung) >= self.min_early_stopping_rate {
                steps.push(step);
            }
            rung += 1;
        }
        steps
    }
}

impl Default for SuccessiveHalvingPruner {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(clippy::cast_precision_loss)]
impl Pruner for SuccessiveHalvingPruner {
    fn should_prune(
        &self,
        _trial_id: u64,
        step: u64,
        intermediate_values: &[(u64, f64)],
        completed_trials: &[CompletedTrial],
    ) -> bool {
        let rungs = self.rung_steps();

        // Find the highest rung step <= current step
        let Some(&rung_step) = rungs.iter().rev().find(|&&r| r <= step) else {
            // No rung matches (before the first rung) → don't prune
            return false;
        };

        // If this is the last rung (full budget), don't prune
        if rung_step >= self.max_resource {
            return false;
        }

        // Get the current trial's value at this rung step
        let Some(&(_, current_value)) = intermediate_values.iter().find(|(s, _)| *s == rung_step)
        else {
            // Trial hasn't reported a value at this exact rung step.
            // Use the latest intermediate value at or before the rung step instead.
            let Some(&(_, current_value)) = intermediate_values
                .iter()
                .rev()
                .find(|(s, _)| *s <= rung_step)
            else {
                return false;
            };
            return self.is_pruned_at_rung(current_value, rung_step, completed_trials);
        };

        self.is_pruned_at_rung(current_value, rung_step, completed_trials)
    }
}

impl SuccessiveHalvingPruner {
    /// Determine whether a trial with `current_value` should be pruned at the given rung.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn is_pruned_at_rung(
        &self,
        current_value: f64,
        rung_step: u64,
        completed_trials: &[CompletedTrial],
    ) -> bool {
        let eta = self.reduction_factor as usize;

        // Collect values at this rung step from all trials that reached it
        let mut values_at_rung: Vec<f64> = completed_trials
            .iter()
            .filter(|t| t.state == TrialState::Complete || t.state == TrialState::Pruned)
            .filter_map(|t| {
                t.intermediate_values
                    .iter()
                    .find(|(s, _)| *s == rung_step)
                    .map(|(_, v)| *v)
            })
            .collect();

        // Need at least eta trials to make a meaningful comparison
        // (with fewer trials, we can't determine the top 1/eta fraction)
        if values_at_rung.len() < eta {
            return false;
        }

        // Include the current trial's value for ranking
        values_at_rung.push(current_value);

        // Sort based on direction: best values first
        values_at_rung
            .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        if self.direction == Direction::Maximize {
            values_at_rung.reverse();
        }

        // Keep top 1/eta fraction
        let n_keep = (values_at_rung.len() as f64 / eta as f64).ceil() as usize;
        let threshold_idx = n_keep.max(1) - 1;
        let threshold = values_at_rung[threshold_idx];

        // Prune if current value is worse than the threshold
        match self.direction {
            Direction::Minimize => current_value > threshold,
            Direction::Maximize => current_value < threshold,
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;

    fn make_trial(id: u64, values: &[(u64, f64)]) -> CompletedTrial {
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

    fn make_pruned_trial(id: u64, values: &[(u64, f64)]) -> CompletedTrial {
        let mut t = make_trial(id, values);
        t.state = TrialState::Pruned;
        t
    }

    #[test]
    fn rung_steps_default() {
        let pruner = SuccessiveHalvingPruner::new();
        let rungs = pruner.rung_steps();
        // min=1, max=81, eta=3 → 1, 3, 9, 27, 81
        assert_eq!(rungs, vec![1, 3, 9, 27, 81]);
    }

    #[test]
    fn rung_steps_custom() {
        let pruner = SuccessiveHalvingPruner::new()
            .min_resource(2)
            .max_resource(32)
            .reduction_factor(2);
        let rungs = pruner.rung_steps();
        // 2, 4, 8, 16, 32
        assert_eq!(rungs, vec![2, 4, 8, 16, 32]);
    }

    #[test]
    fn rung_steps_with_early_stopping_rate() {
        let pruner = SuccessiveHalvingPruner::new().min_early_stopping_rate(2);
        let rungs = pruner.rung_steps();
        // Skip rung 0 (step=1) and rung 1 (step=3), keep rung 2+ (9, 27, 81)
        assert_eq!(rungs, vec![9, 27, 81]);
    }

    #[test]
    fn no_prune_before_first_rung() {
        let pruner = SuccessiveHalvingPruner::new()
            .min_resource(10)
            .max_resource(100)
            .reduction_factor(3);
        let completed = vec![
            make_trial(0, &[(5, 1.0)]),
            make_trial(1, &[(5, 2.0)]),
            make_trial(2, &[(5, 3.0)]),
        ];
        // Step 5 is before the first rung (10)
        assert!(!pruner.should_prune(3, 5, &[(5, 100.0)], &completed));
    }

    #[test]
    fn no_prune_with_single_trial() {
        let pruner = SuccessiveHalvingPruner::new();
        let completed = vec![make_trial(0, &[(1, 5.0)]), make_trial(1, &[(1, 3.0)])];
        // Only 2 completed trials at rung + 1 current = 3 total, threshold = ceil(3/3) = 1
        // With eta=3, we need at least 3 completed trials
        assert!(!pruner.should_prune(2, 1, &[(1, 10.0)], &completed));
    }

    #[test]
    fn prune_worst_trials_at_rung() {
        let pruner = SuccessiveHalvingPruner::new().direction(Direction::Minimize);

        // 9 completed trials at rung step=1, with values 1..=9
        let completed: Vec<_> = (0..9)
            .map(|i| make_trial(i, &[(1, (i + 1) as f64)]))
            .collect();

        // With eta=3, keep top 1/3. 10 total values → ceil(10/3) = 4 kept
        // Best 4 values: 1, 2, 3, 4. Threshold = 4.0
        // Value 3.0 → keep (in top 1/3)
        assert!(!pruner.should_prune(9, 1, &[(1, 3.0)], &completed));
        // Value 5.0 → prune (not in top 1/3)
        assert!(pruner.should_prune(9, 1, &[(1, 5.0)], &completed));
    }

    #[test]
    fn top_fraction_survives() {
        let pruner = SuccessiveHalvingPruner::new().direction(Direction::Minimize);

        // 6 completed trials at step=1
        let completed: Vec<_> = (0..6)
            .map(|i| make_trial(i, &[(1, (i + 1) as f64)]))
            .collect();

        // 7 total (6 + current). ceil(7/3) = 3 keep. Threshold = 3.0
        // Value 2.0 → keep
        assert!(!pruner.should_prune(6, 1, &[(1, 2.0)], &completed));
        // Value 3.0 → keep (at threshold)
        assert!(!pruner.should_prune(6, 1, &[(1, 3.0)], &completed));
        // Value 4.0 → prune
        assert!(pruner.should_prune(6, 1, &[(1, 4.0)], &completed));
    }

    #[test]
    fn maximize_direction() {
        let pruner = SuccessiveHalvingPruner::new().direction(Direction::Maximize);

        let completed: Vec<_> = (0..6)
            .map(|i| make_trial(i, &[(1, (i + 1) as f64)]))
            .collect();

        // 7 total. For maximize, best = highest. ceil(7/3)=3. Top 3: 6,5,4. Threshold=4.0
        // Value 5.0 → keep
        assert!(!pruner.should_prune(6, 1, &[(1, 5.0)], &completed));
        // Value 4.0 → keep (at threshold)
        assert!(!pruner.should_prune(6, 1, &[(1, 4.0)], &completed));
        // Value 3.0 → prune
        assert!(pruner.should_prune(6, 1, &[(1, 3.0)], &completed));
    }

    #[test]
    fn reduction_factor_2() {
        let pruner = SuccessiveHalvingPruner::new()
            .reduction_factor(2)
            .min_resource(1)
            .max_resource(16)
            .direction(Direction::Minimize);

        // Rungs: 1, 2, 4, 8, 16
        assert_eq!(pruner.rung_steps(), vec![1, 2, 4, 8, 16]);

        // 4 completed trials at rung step=1
        let completed: Vec<_> = (0..4)
            .map(|i| make_trial(i, &[(1, (i + 1) as f64)]))
            .collect();

        // With eta=2, 5 total. ceil(5/2) = 3 keep. Threshold = 3.0
        // Value 3.0 → keep
        assert!(!pruner.should_prune(4, 1, &[(1, 3.0)], &completed));
        // Value 4.0 → prune
        assert!(pruner.should_prune(4, 1, &[(1, 4.0)], &completed));
    }

    #[test]
    fn reduction_factor_4() {
        let pruner = SuccessiveHalvingPruner::new()
            .reduction_factor(4)
            .min_resource(1)
            .max_resource(64)
            .direction(Direction::Minimize);

        // Rungs: 1, 4, 16, 64
        assert_eq!(pruner.rung_steps(), vec![1, 4, 16, 64]);

        // 12 completed trials at rung step=1
        let completed: Vec<_> = (0..12)
            .map(|i| make_trial(i, &[(1, (i + 1) as f64)]))
            .collect();

        // With eta=4, 13 total. ceil(13/4) = 4 keep. Threshold = 4.0
        // Value 4.0 → keep
        assert!(!pruner.should_prune(12, 1, &[(1, 4.0)], &completed));
        // Value 5.0 → prune
        assert!(pruner.should_prune(12, 1, &[(1, 5.0)], &completed));
    }

    #[test]
    fn non_contiguous_steps() {
        let pruner = SuccessiveHalvingPruner::new().direction(Direction::Minimize);

        // Trials reporting at rung step=3 (not step=1)
        let completed: Vec<_> = (0..6)
            .map(|i| make_trial(i, &[(3, (i + 1) as f64)]))
            .collect();

        // Current trial reports at step 5 (between rung 3 and rung 9)
        // Highest rung <= 5 is 3. Use value at rung step 3.
        // Trial has value at step 3 → use it
        assert!(!pruner.should_prune(6, 5, &[(3, 2.0)], &completed));
        assert!(pruner.should_prune(6, 5, &[(3, 5.0)], &completed));
    }

    #[test]
    fn no_prune_at_max_resource() {
        let pruner = SuccessiveHalvingPruner::new();
        let completed: Vec<_> = (0..9)
            .map(|i| make_trial(i, &[(81, (i + 1) as f64)]))
            .collect();

        // At the max resource rung, never prune (trial should complete)
        assert!(!pruner.should_prune(9, 81, &[(81, 100.0)], &completed));
    }

    #[test]
    fn includes_pruned_trials_in_comparison() {
        let pruner = SuccessiveHalvingPruner::new().direction(Direction::Minimize);

        // Mix of completed and pruned trials at rung step=1
        let completed = vec![
            make_trial(0, &[(1, 1.0)]),
            make_trial(1, &[(1, 2.0)]),
            make_pruned_trial(2, &[(1, 8.0)]),
            make_pruned_trial(3, &[(1, 9.0)]),
            make_pruned_trial(4, &[(1, 10.0)]),
        ];

        // 6 total. ceil(6/3) = 2 keep. Threshold = 2.0
        // Value 2.0 → keep
        assert!(!pruner.should_prune(5, 1, &[(1, 2.0)], &completed));
        // Value 3.0 → prune
        assert!(pruner.should_prune(5, 1, &[(1, 3.0)], &completed));
    }

    #[test]
    #[should_panic(expected = "min_resource must be > 0")]
    fn rejects_zero_min_resource() {
        let _ = SuccessiveHalvingPruner::new().min_resource(0);
    }

    #[test]
    #[should_panic(expected = "max_resource must be > 0")]
    fn rejects_zero_max_resource() {
        let _ = SuccessiveHalvingPruner::new().max_resource(0);
    }

    #[test]
    #[should_panic(expected = "reduction_factor must be >= 2")]
    fn rejects_reduction_factor_one() {
        let _ = SuccessiveHalvingPruner::new().reduction_factor(1);
    }
}
