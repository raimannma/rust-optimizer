//! `HyperBand` pruner — adaptive budget scheduling with multiple SHA brackets.
//!
//! `HyperBand` addresses the main weakness of
//! [`SuccessiveHalvingPruner`](super::SuccessiveHalvingPruner): sensitivity to
//! the `min_resource` setting. It runs multiple Successive Halving brackets in
//! parallel, each with a different trade-off between the number of trials and
//! the starting budget:
//!
//! - **Bracket 0**: many trials, small starting budget (aggressive early pruning)
//! - **Bracket `s_max`**: few trials, full budget (no pruning)
//!
//! Trials are assigned to brackets in round-robin order. This ensures that
//! the overall search is robust regardless of how informative early steps are.
//!
//! # When to use
//!
//! - When you don't know how many epochs/steps are needed before performance
//!   becomes predictive
//! - As a drop-in upgrade over [`SuccessiveHalvingPruner`](super::SuccessiveHalvingPruner)
//!   when you can afford more total trials
//! - For large-scale hyperparameter searches where compute savings matter most
//!
//! # Configuration
//!
//! | Option | Default | Description |
//! |--------|---------|-------------|
//! | `min_resource` | 1 | Smallest budget for the most aggressive bracket |
//! | `max_resource` | 81 | Full budget (last rung in every bracket) |
//! | `reduction_factor` | 3 | At each rung, keep top 1/η trials |
//! | `direction` | `Minimize` | Optimization direction |
//!
//! # Example
//!
//! ```
//! use optimizer::Direction;
//! use optimizer::pruner::HyperbandPruner;
//!
//! let pruner = HyperbandPruner::new()
//!     .min_resource(1)
//!     .max_resource(81)
//!     .reduction_factor(3)
//!     .direction(Direction::Minimize);
//! ```

use core::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::sync::Mutex;

use super::Pruner;
use crate::sampler::CompletedTrial;
use crate::types::{Direction, TrialState};

/// `HyperBand` pruner that manages multiple Successive Halving brackets.
///
/// Hyperband addresses SHA's sensitivity to the `min_resource` choice by
/// running multiple brackets, each with a different tradeoff between the
/// number of configurations and the starting budget:
///
/// - Bracket 0: many trials, very small starting budget (aggressive pruning)
/// - Bracket 1: fewer trials, larger starting budget (moderate pruning)
/// - ...
/// - Bracket `s_max`: few trials, full budget (no pruning)
///
/// Trials are assigned to brackets in round-robin fashion. Each bracket
/// runs SHA with its own `min_resource` and rung schedule.
///
/// # Examples
///
/// ```
/// use optimizer::Direction;
/// use optimizer::pruner::HyperbandPruner;
///
/// let pruner = HyperbandPruner::new()
///     .min_resource(1)
///     .max_resource(81)
///     .reduction_factor(3)
///     .direction(Direction::Minimize);
/// ```
pub struct HyperbandPruner {
    min_resource: u64,
    max_resource: u64,
    reduction_factor: u64,
    direction: Direction,
    /// Tracks which bracket each trial belongs to.
    trial_brackets: Mutex<HashMap<u64, usize>>,
    /// Counter for round-robin bracket assignment.
    next_bracket: AtomicU64,
}

impl HyperbandPruner {
    /// Create a new `HyperbandPruner` with default parameters.
    ///
    /// Defaults: `min_resource=1`, `max_resource=81`, `reduction_factor=3`,
    /// `direction=Minimize`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_resource: 1,
            max_resource: 81,
            reduction_factor: 3,
            direction: Direction::Minimize,
            trial_brackets: Mutex::new(HashMap::new()),
            next_bracket: AtomicU64::new(0),
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

    /// Set the optimization direction.
    #[must_use]
    pub fn direction(mut self, d: Direction) -> Self {
        self.direction = d;
        self
    }

    /// Compute `s_max = floor(log(max_resource / min_resource) / log(eta))`.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn s_max(&self) -> u64 {
        let eta = self.reduction_factor as f64;
        let ratio = self.max_resource as f64 / self.min_resource as f64;
        (ratio.ln() / eta.ln()).floor() as u64
    }

    /// Compute the rung steps for a given bracket `s`.
    ///
    /// For bracket `s`, the starting resource is `max_resource / eta^(s_max - s)`,
    /// and rungs are spaced at powers of eta from there up to `max_resource`.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn rung_steps_for_bracket(&self, bracket: usize) -> Vec<u64> {
        let s_max = self.s_max();
        let eta = self.reduction_factor as f64;

        // Starting resource for this bracket
        let exponent = s_max.saturating_sub(bracket as u64);
        let min_resource_bracket =
            (self.max_resource as f64 / eta.powi(exponent as i32)).ceil() as u64;

        let mut steps = Vec::new();
        let mut rung: u32 = 0;
        while let Some(power) = self.reduction_factor.checked_pow(rung) {
            let step = min_resource_bracket.saturating_mul(power);
            if step > self.max_resource {
                break;
            }
            steps.push(step);
            rung += 1;
        }
        steps
    }

    /// Assign a trial to a bracket (round-robin) and return the bracket index.
    #[allow(clippy::cast_possible_truncation)]
    fn assign_bracket(&self, trial_id: u64) -> usize {
        let n_brackets = (self.s_max() + 1) as usize;
        let mut map = self.trial_brackets.lock().expect("lock poisoned");
        *map.entry(trial_id).or_insert_with(|| {
            let idx = self.next_bracket.fetch_add(1, Ordering::Relaxed);
            (idx as usize) % n_brackets
        })
    }
}

impl Default for HyperbandPruner {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(clippy::cast_precision_loss)]
impl Pruner for HyperbandPruner {
    fn should_prune(
        &self,
        trial_id: u64,
        step: u64,
        intermediate_values: &[(u64, f64)],
        completed_trials: &[CompletedTrial],
    ) -> bool {
        let bracket = self.assign_bracket(trial_id);
        let rungs = self.rung_steps_for_bracket(bracket);

        // Find the highest rung step <= current step
        let Some(&rung_step) = rungs.iter().rev().find(|&&r| r <= step) else {
            return false;
        };

        // Never prune at the last rung (full budget)
        if rung_step >= self.max_resource {
            return false;
        }

        // Get the current trial's value at this rung step
        let current_value =
            if let Some(&(_, v)) = intermediate_values.iter().find(|(s, _)| *s == rung_step) {
                v
            } else if let Some(&(_, v)) = intermediate_values
                .iter()
                .rev()
                .find(|(s, _)| *s <= rung_step)
            {
                v
            } else {
                return false;
            };

        self.is_pruned_at_rung(current_value, rung_step, bracket, completed_trials)
    }
}

impl HyperbandPruner {
    /// Determine whether a trial should be pruned at the given rung within its bracket.
    ///
    /// Only compares against other trials in the same bracket.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn is_pruned_at_rung(
        &self,
        current_value: f64,
        rung_step: u64,
        bracket: usize,
        completed_trials: &[CompletedTrial],
    ) -> bool {
        let eta = self.reduction_factor as usize;

        // Collect values at this rung step from trials in the same bracket
        let map = self.trial_brackets.lock().expect("lock poisoned");
        let mut values_at_rung: Vec<f64> = completed_trials
            .iter()
            .filter(|t| t.state == TrialState::Complete || t.state == TrialState::Pruned)
            .filter(|t| map.get(&t.id).copied() == Some(bracket))
            .filter_map(|t| {
                t.intermediate_values
                    .iter()
                    .find(|(s, _)| *s == rung_step)
                    .map(|(_, v)| *v)
            })
            .collect();
        drop(map);

        // Need at least eta trials to make a meaningful comparison
        if values_at_rung.len() < eta {
            return false;
        }

        values_at_rung.push(current_value);

        values_at_rung
            .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        if self.direction == Direction::Maximize {
            values_at_rung.reverse();
        }

        let n_keep = (values_at_rung.len() as f64 / eta as f64).ceil() as usize;
        let threshold_idx = n_keep.max(1) - 1;
        let threshold = values_at_rung[threshold_idx];

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
    fn s_max_default() {
        let pruner = HyperbandPruner::new();
        // s_max = floor(ln(81/1) / ln(3)) = floor(4.0) = 4
        assert_eq!(pruner.s_max(), 4);
    }

    #[test]
    fn s_max_custom() {
        let pruner = HyperbandPruner::new()
            .min_resource(1)
            .max_resource(16)
            .reduction_factor(2);
        // s_max = floor(ln(16) / ln(2)) = floor(4.0) = 4
        assert_eq!(pruner.s_max(), 4);
    }

    #[test]
    fn bracket_count() {
        let pruner = HyperbandPruner::new();
        // s_max=4, so brackets 0..=4 → 5 brackets
        assert_eq!(pruner.s_max() + 1, 5);
    }

    #[test]
    fn rung_steps_bracket_0_default() {
        let pruner = HyperbandPruner::new();
        // Bracket 0: min_resource_bracket = ceil(81 / 3^4) = ceil(81/81) = 1
        // Rungs: 1, 3, 9, 27, 81
        assert_eq!(pruner.rung_steps_for_bracket(0), vec![1, 3, 9, 27, 81]);
    }

    #[test]
    fn rung_steps_bracket_2_default() {
        let pruner = HyperbandPruner::new();
        // Bracket 2: min_resource_bracket = ceil(81 / 3^(4-2)) = ceil(81/9) = 9
        // Rungs: 9, 27, 81
        assert_eq!(pruner.rung_steps_for_bracket(2), vec![9, 27, 81]);
    }

    #[test]
    fn rung_steps_bracket_4_default() {
        let pruner = HyperbandPruner::new();
        // Bracket 4 (s_max): min_resource_bracket = ceil(81 / 3^0) = 81
        // Rungs: 81 only (no pruning, full budget)
        assert_eq!(pruner.rung_steps_for_bracket(4), vec![81]);
    }

    #[test]
    fn rung_steps_eta2() {
        let pruner = HyperbandPruner::new()
            .min_resource(1)
            .max_resource(16)
            .reduction_factor(2);
        // s_max = 4
        // Bracket 0: min=ceil(16/2^4)=1, rungs: 1,2,4,8,16
        assert_eq!(pruner.rung_steps_for_bracket(0), vec![1, 2, 4, 8, 16]);
        // Bracket 2: min=ceil(16/2^2)=4, rungs: 4,8,16
        assert_eq!(pruner.rung_steps_for_bracket(2), vec![4, 8, 16]);
        // Bracket 4: min=16, rungs: 16
        assert_eq!(pruner.rung_steps_for_bracket(4), vec![16]);
    }

    #[test]
    fn round_robin_bracket_assignment() {
        let pruner = HyperbandPruner::new(); // 5 brackets (0..=4)
        // Trials get assigned in round-robin: 0→0, 1→1, 2→2, 3→3, 4→4, 5→0, ...
        assert_eq!(pruner.assign_bracket(100), 0);
        assert_eq!(pruner.assign_bracket(101), 1);
        assert_eq!(pruner.assign_bracket(102), 2);
        assert_eq!(pruner.assign_bracket(103), 3);
        assert_eq!(pruner.assign_bracket(104), 4);
        assert_eq!(pruner.assign_bracket(105), 0); // wraps around

        // Repeated calls for same trial return same bracket
        assert_eq!(pruner.assign_bracket(100), 0);
        assert_eq!(pruner.assign_bracket(103), 3);
    }

    #[test]
    fn no_prune_before_first_rung() {
        let pruner = HyperbandPruner::new().direction(Direction::Minimize);
        // Assign trial 0 to bracket 0 (rungs: 1, 3, 9, 27, 81)
        pruner.assign_bracket(0);

        // Register completed trials in bracket 0
        let mut completed = Vec::new();
        for i in 1..=9 {
            pruner.assign_bracket(i);
            completed.push(make_trial(i, &[(1, i as f64)]));
        }

        // Trial at step 0 (before rung 1) → don't prune
        assert!(!pruner.should_prune(0, 0, &[(0, 100.0)], &completed));
    }

    #[test]
    fn no_prune_at_max_resource() {
        let pruner = HyperbandPruner::new().direction(Direction::Minimize);

        // Put all trials in bracket 0
        let mut completed = Vec::new();
        for i in 0..9 {
            pruner.assign_bracket(i);
            completed.push(make_trial(i, &[(81, (i + 1) as f64)]));
        }

        let trial_id = 9;
        pruner.assign_bracket(trial_id);
        // At max_resource (81), never prune
        assert!(!pruner.should_prune(trial_id, 81, &[(81, 100.0)], &completed));
    }

    #[test]
    fn prune_worst_in_bracket_minimize() {
        let pruner = HyperbandPruner::new().direction(Direction::Minimize);

        // Force all trials into bracket 0 by assigning sequentially
        // With 5 brackets, trials 0,5,10,... go to bracket 0
        let bracket_0_ids: Vec<u64> = (0..5).map(|i| i * 5).collect();
        // Assign all 25 trial IDs to fill brackets
        for i in 0..25 {
            pruner.assign_bracket(i);
        }

        // Create 9 completed trials in bracket 0 at rung step=1
        let completed: Vec<_> = bracket_0_ids
            .iter()
            .take(3)
            .enumerate()
            .map(|(idx, &id)| make_trial(id, &[(1, (idx + 1) as f64)]))
            .collect();

        // Trial 25 → bracket 0 (25 % 5 == 0)
        let test_id = 25;
        pruner.assign_bracket(test_id);
        assert_eq!(pruner.assign_bracket(test_id), 0);

        // 3 completed + 1 current = 4. eta=3. ceil(4/3)=2. Threshold = 2.0
        // Value 2.0 → keep
        assert!(!pruner.should_prune(test_id, 1, &[(1, 2.0)], &completed));
        // Value 3.0 → prune
        assert!(pruner.should_prune(test_id, 1, &[(1, 3.0)], &completed));
    }

    #[test]
    fn prune_worst_in_bracket_maximize() {
        let pruner = HyperbandPruner::new().direction(Direction::Maximize);

        // Assign trials so they end up in bracket 0
        for i in 0..25 {
            pruner.assign_bracket(i);
        }

        let completed: Vec<_> = [0u64, 5, 10]
            .iter()
            .enumerate()
            .map(|(idx, &id)| make_trial(id, &[(1, (idx + 1) as f64)]))
            .collect();

        let test_id = 25;
        pruner.assign_bracket(test_id);

        // For maximize, best = highest. Values: 1,2,3 + current
        // Value 2.0 → keep (threshold = 2.0 when sorted desc: 3,2,current,1)
        assert!(!pruner.should_prune(test_id, 1, &[(1, 2.0)], &completed));
        // Value 1.0 → prune
        assert!(pruner.should_prune(test_id, 1, &[(1, 0.5)], &completed));
    }

    #[test]
    fn different_brackets_have_different_aggressiveness() {
        let pruner = HyperbandPruner::new()
            .min_resource(1)
            .max_resource(81)
            .reduction_factor(3)
            .direction(Direction::Minimize);

        let rungs_0 = pruner.rung_steps_for_bracket(0);
        let rungs_2 = pruner.rung_steps_for_bracket(2);
        let rungs_4 = pruner.rung_steps_for_bracket(4);

        // Bracket 0 has the most rungs (most aggressive)
        assert!(rungs_0.len() > rungs_2.len());
        // Bracket 4 has just 1 rung (no pruning)
        assert_eq!(rungs_4.len(), 1);
        // Bracket 0 starts earliest
        assert!(rungs_0[0] < rungs_2[0]);
    }

    #[test]
    fn trials_in_different_brackets_independent() {
        let pruner = HyperbandPruner::new().direction(Direction::Minimize);

        // Assign trials: bracket 0 gets IDs 0,5,10,15,20
        for i in 0..25 {
            pruner.assign_bracket(i);
        }

        // Bracket 0 trials: bad values at rung step=1
        let bracket_0_trials: Vec<_> = [0u64, 5, 10]
            .iter()
            .map(|&id| make_trial(id, &[(1, 100.0)]))
            .collect();

        // Bracket 1 trials: good values at rung step=1
        let bracket_1_trials: Vec<_> = [1u64, 6, 11]
            .iter()
            .map(|&id| make_trial(id, &[(1, 1.0)]))
            .collect();

        let mut all_trials = bracket_0_trials;
        all_trials.extend(bracket_1_trials);

        // A new bracket-0 trial with value 50 should be compared against
        // bracket-0 peers (100,100,100), not bracket-1 peers (1,1,1)
        let test_id = 25; // bracket 0
        pruner.assign_bracket(test_id);
        // 3 peers at 100.0 + current at 50.0. ceil(4/3)=2. Sorted: 50,100,100,100. Threshold=100.0
        // Value 50.0 < 100.0 → keep
        assert!(!pruner.should_prune(test_id, 1, &[(1, 50.0)], &all_trials));
    }

    #[test]
    fn includes_pruned_trials() {
        let pruner = HyperbandPruner::new().direction(Direction::Minimize);

        for i in 0..25 {
            pruner.assign_bracket(i);
        }

        let completed = vec![
            make_trial(0, &[(1, 1.0)]),
            make_pruned_trial(5, &[(1, 8.0)]),
            make_pruned_trial(10, &[(1, 9.0)]),
        ];

        let test_id = 25;
        pruner.assign_bracket(test_id);

        // Values: 1.0, 8.0, 9.0 + current. eta=3.
        // Value 1.0 → keep
        assert!(!pruner.should_prune(test_id, 1, &[(1, 1.0)], &completed));
        // Value 5.0 → prune (sorted: 1,5,8,9 → keep ceil(4/3)=2 → threshold=5.0, 5.0 not > 5.0 → keep)
        assert!(!pruner.should_prune(test_id, 1, &[(1, 5.0)], &completed));
        // Value 6.0 → prune (sorted: 1,6,8,9 → threshold=6.0, 6.0 not > 6.0 → keep)
        assert!(!pruner.should_prune(test_id, 1, &[(1, 6.0)], &completed));
        // Value 9.5 → prune
        assert!(pruner.should_prune(test_id, 1, &[(1, 9.5)], &completed));
    }

    #[test]
    #[should_panic(expected = "min_resource must be > 0")]
    fn rejects_zero_min_resource() {
        let _ = HyperbandPruner::new().min_resource(0);
    }

    #[test]
    #[should_panic(expected = "max_resource must be > 0")]
    fn rejects_zero_max_resource() {
        let _ = HyperbandPruner::new().max_resource(0);
    }

    #[test]
    #[should_panic(expected = "reduction_factor must be >= 2")]
    fn rejects_reduction_factor_one() {
        let _ = HyperbandPruner::new().reduction_factor(1);
    }
}
