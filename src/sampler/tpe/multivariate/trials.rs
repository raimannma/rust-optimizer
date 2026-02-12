//! Trial processing for the multivariate TPE sampler.
//!
//! Contains constant-liar imputation, trial filtering, good/bad splitting,
//! observation extraction, and categorical index extraction.

use std::collections::HashMap;

use crate::distribution::Distribution;
use crate::param::ParamValue;
use crate::parameter::ParamId;
use crate::sampler::{CompletedTrial, PendingTrial};

use super::{ConstantLiarStrategy, MultivariateTpeSampler};

impl MultivariateTpeSampler {
    /// Imputes objective values for pending trials based on the constant liar strategy.
    ///
    /// In parallel optimization, multiple trials may be running simultaneously. This method
    /// assigns "lie" values to pending trials so they can be included in the model fitting,
    /// which helps avoid redundant exploration of the same region.
    ///
    /// # Arguments
    ///
    /// * `pending_trials` - Trials that are currently running and have no objective value yet.
    /// * `completed_trials` - Trials that have completed and have objective values.
    ///
    /// # Returns
    ///
    /// A vector of `CompletedTrial` objects containing both the original completed trials
    /// and the pending trials with imputed values. If the strategy is `None`, returns
    /// only the completed trials (pending trials are ignored).
    #[must_use]
    pub fn impute_pending_trials(
        &self,
        pending_trials: &[PendingTrial],
        completed_trials: &[CompletedTrial],
    ) -> Vec<CompletedTrial> {
        // Start with a copy of completed trials
        let mut result: Vec<CompletedTrial> = completed_trials.to_vec();

        // If strategy is None or no pending trials, just return completed trials
        if matches!(self.constant_liar, ConstantLiarStrategy::None) || pending_trials.is_empty() {
            return result;
        }

        // Compute the imputation value based on strategy
        let imputed_value = self.compute_imputation_value(completed_trials);

        // Convert pending trials to completed trials with imputed values
        for pending in pending_trials {
            result.push(CompletedTrial::new(
                pending.id,
                pending.params.clone(),
                pending.distributions.clone(),
                HashMap::new(),
                imputed_value,
            ));
        }

        result
    }

    /// Computes the imputation value based on the constant liar strategy.
    ///
    /// This is a helper method used by [`impute_pending_trials`](Self::impute_pending_trials).
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn compute_imputation_value(&self, completed_trials: &[CompletedTrial]) -> f64 {
        match self.constant_liar {
            ConstantLiarStrategy::None => 0.0, // This case is handled before calling this method
            ConstantLiarStrategy::Mean => {
                if completed_trials.is_empty() {
                    0.0
                } else {
                    let sum: f64 = completed_trials.iter().map(|t| t.value).sum();
                    sum / completed_trials.len() as f64
                }
            }
            ConstantLiarStrategy::Best => {
                // Best means minimum for minimization problems
                completed_trials
                    .iter()
                    .map(|t| t.value)
                    .fold(f64::INFINITY, f64::min)
            }
            ConstantLiarStrategy::Worst => {
                // Worst means maximum for minimization problems
                completed_trials
                    .iter()
                    .map(|t| t.value)
                    .fold(f64::NEG_INFINITY, f64::max)
            }
            ConstantLiarStrategy::Custom(v) => v,
        }
    }

    /// Filters trials to those containing all parameters in the search space.
    ///
    /// Only trials that contain ALL parameters in the search space are included,
    /// ensuring we can model the joint distribution over all parameters.
    #[must_use]
    pub fn filter_trials<'a>(
        &self,
        history: &'a [CompletedTrial],
        search_space: &HashMap<ParamId, Distribution>,
    ) -> Vec<&'a CompletedTrial> {
        history
            .iter()
            .filter(|trial| {
                // Include trial only if it has ALL parameters in the search space
                search_space
                    .keys()
                    .all(|param_id| trial.params.contains_key(param_id))
            })
            .collect()
    }

    /// Splits filtered trials into good and bad groups based on the gamma quantile.
    ///
    /// The gamma value is computed dynamically using the configured [`GammaStrategy`].
    /// Trials are sorted by objective value (ascending for minimization), and the
    /// gamma quantile determines the split point.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    #[must_use]
    pub fn split_trials<'a>(
        &self,
        trials: &[&'a CompletedTrial],
    ) -> (Vec<&'a CompletedTrial>, Vec<&'a CompletedTrial>) {
        if trials.is_empty() {
            return (vec![], vec![]);
        }

        // Sort trials by objective value (ascending for minimization)
        let mut sorted_indices: Vec<usize> = (0..trials.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            trials[a]
                .value
                .partial_cmp(&trials[b].value)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        // Compute gamma using the strategy and clamp to valid range
        let gamma = self
            .gamma_strategy
            .gamma(trials.len())
            .clamp(f64::EPSILON, 1.0 - f64::EPSILON);

        // Calculate the split point (gamma quantile)
        // Ensure at least 1 trial in each group if possible
        let n_good = ((trials.len() as f64 * gamma).ceil() as usize)
            .max(1)
            .min(trials.len().saturating_sub(1));

        // Handle edge case: if we have only 1 trial, put it in good
        if trials.len() == 1 {
            return (vec![trials[0]], vec![]);
        }

        let good: Vec<_> = sorted_indices[..n_good]
            .iter()
            .map(|&i| trials[i])
            .collect();
        let bad: Vec<_> = sorted_indices[n_good..]
            .iter()
            .map(|&i| trials[i])
            .collect();

        (good, bad)
    }

    /// Extracts parameter values from trials as a numeric observation matrix.
    ///
    /// Each row in the output represents one trial's parameter values in the specified order.
    /// Categorical parameters are skipped.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn extract_observations(
        &self,
        trials: &[&CompletedTrial],
        param_order: &[ParamId],
    ) -> Vec<Vec<f64>> {
        trials
            .iter()
            .map(|trial| {
                param_order
                    .iter()
                    .filter_map(|param_id| {
                        trial.params.get(param_id).and_then(|value| match value {
                            crate::param::ParamValue::Float(f) => Some(*f),
                            crate::param::ParamValue::Int(i) => Some(*i as f64),
                            crate::param::ParamValue::Categorical(_) => None, // Skip categorical
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Extracts categorical indices from trials for a specific parameter.
    pub(crate) fn extract_categorical_indices(
        trials: &[&CompletedTrial],
        param_id: ParamId,
    ) -> Vec<usize> {
        trials
            .iter()
            .filter_map(|trial| {
                trial.params.get(&param_id).and_then(|value| {
                    if let ParamValue::Categorical(idx) = value {
                        Some(*idx)
                    } else {
                        None
                    }
                })
            })
            .collect()
    }
}
