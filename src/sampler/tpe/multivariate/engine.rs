//! Core multivariate TPE sampling logic.
//!
//! Contains the main sampling engine: group decomposition, single-group multivariate
//! TPE, candidate selection, independent fallbacks, and value conversion.

use std::collections::HashMap;

use crate::distribution::Distribution;
use crate::param::ParamValue;
use crate::parameter::ParamId;
use crate::sampler::CompletedTrial;

use super::MultivariateTpeSampler;

impl MultivariateTpeSampler {
    /// Samples parameters by decomposing the search space into independent groups.
    ///
    /// When `group=true`, this method analyzes the trial history to identify groups of
    /// parameters that always appear together, then samples each group independently
    /// using multivariate TPE. This is more efficient when parameters naturally partition
    /// into independent subsets (e.g., due to conditional search spaces).
    ///
    /// # Arguments
    ///
    /// * `search_space` - The full search space containing all parameters to sample.
    /// * `history` - Completed trials from the optimization history.
    ///
    /// # Returns
    ///
    /// A `HashMap` mapping parameter names to their sampled values.
    pub(crate) fn sample_with_groups(
        &self,
        search_space: &HashMap<ParamId, Distribution>,
        history: &[CompletedTrial],
    ) -> HashMap<ParamId, ParamValue> {
        use std::collections::HashSet;

        use crate::sampler::tpe::GroupDecomposedSearchSpace;

        // Decompose the search space into independent parameter groups
        let groups = GroupDecomposedSearchSpace::calculate(history);

        let mut result: HashMap<ParamId, ParamValue> = HashMap::new();

        // Sample each group independently
        for group in &groups {
            // Build a sub-search space for this group
            let group_search_space: HashMap<ParamId, Distribution> = search_space
                .iter()
                .filter(|(id, _)| group.contains(id))
                .map(|(id, dist)| (*id, dist.clone()))
                .collect();

            if group_search_space.is_empty() {
                continue;
            }

            // Filter history to trials that have at least one parameter in this group
            let group_history: Vec<&CompletedTrial> = history
                .iter()
                .filter(|trial| {
                    trial
                        .distributions
                        .keys()
                        .any(|param_id| group.contains(param_id))
                })
                .collect();

            // Build completed trials from references for the group
            // We need to create a temporary slice for sample_group_internal
            let group_history_owned: Vec<CompletedTrial> =
                group_history.iter().map(|t| (*t).clone()).collect();

            // Sample this group using multivariate TPE
            let mut rng = self.rng.lock();
            let group_result =
                self.sample_single_group(&group_search_space, &group_history_owned, &mut rng);
            drop(rng);

            // Merge group results into the main result
            for (id, value) in group_result {
                result.insert(id, value);
            }
        }

        // Handle parameters not in any group (sample independently)
        let grouped_params: HashSet<ParamId> = groups.iter().flatten().copied().collect();
        let ungrouped_params: HashMap<ParamId, Distribution> = search_space
            .iter()
            .filter(|(id, _)| !grouped_params.contains(id) && !result.contains_key(id))
            .map(|(id, dist)| (*id, dist.clone()))
            .collect();

        if !ungrouped_params.is_empty() {
            // Sample ungrouped parameters uniformly (no history for them)
            let mut rng = self.rng.lock();
            for (id, dist) in &ungrouped_params {
                let value = crate::sampler::common::sample_random(&mut rng, dist);
                result.insert(*id, value);
            }
        }

        result
    }

    /// Validates observations and fits multivariate KDEs for good and bad groups.
    ///
    /// Returns `None` if observations are invalid or KDE construction fails.
    fn try_fit_kdes(
        good_obs: Vec<Vec<f64>>,
        bad_obs: Vec<Vec<f64>>,
        expected_dims: usize,
    ) -> Option<(crate::kde::MultivariateKDE, crate::kde::MultivariateKDE)> {
        use crate::kde::MultivariateKDE;

        let valid = !good_obs.is_empty()
            && !bad_obs.is_empty()
            && good_obs.iter().all(|obs| obs.len() == expected_dims)
            && bad_obs.iter().all(|obs| obs.len() == expected_dims);

        if !valid {
            return None;
        }

        let good_kde = MultivariateKDE::new(good_obs).ok()?;
        let bad_kde = MultivariateKDE::new(bad_obs).ok()?;
        Some((good_kde, bad_kde))
    }

    /// Samples parameters as a single group using multivariate TPE.
    ///
    /// This is the core multivariate TPE sampling logic, used both in non-grouped mode
    /// and for sampling individual groups in grouped mode.
    ///
    /// # Arguments
    ///
    /// * `search_space` - The search space for this group of parameters.
    /// * `history` - Completed trials to use for model fitting.
    /// * `rng` - Random number generator (caller must hold lock).
    ///
    /// # Returns
    ///
    /// A `HashMap` mapping parameter names to their sampled values.
    pub(crate) fn sample_single_group(
        &self,
        search_space: &HashMap<ParamId, Distribution>,
        history: &[CompletedTrial],
        rng: &mut fastrand::Rng,
    ) -> HashMap<ParamId, ParamValue> {
        use crate::sampler::tpe::IntersectionSearchSpace;
        use crate::sampler::tpe::common;

        // Early returns for cases requiring random sampling
        if history.len() < self.n_startup_trials {
            return self.sample_all_uniform(search_space, rng);
        }

        let intersection = IntersectionSearchSpace::calculate(history);
        if intersection.is_empty() {
            return self.sample_all_independent_with_rng(search_space, history, rng);
        }

        let filtered = self.filter_trials(history, &intersection);
        if filtered.len() < 2 {
            return self.sample_all_independent_with_rng(search_space, history, rng);
        }

        let (good, bad) = self.split_trials(&filtered);

        // Sample categorical parameters using TPE with l(x)/g(x) ratio
        let mut result: HashMap<ParamId, ParamValue> = HashMap::new();
        for (param_id, dist) in &intersection {
            if let Distribution::Categorical(d) = dist {
                let good_indices = Self::extract_categorical_indices(&good, *param_id);
                let bad_indices = Self::extract_categorical_indices(&bad, *param_id);
                let idx =
                    common::sample_tpe_categorical(d.n_choices, &good_indices, &bad_indices, rng);
                result.insert(*param_id, ParamValue::Categorical(idx));
            }
        }

        // Collect continuous parameters
        let mut param_order: Vec<ParamId> = intersection
            .iter()
            .filter(|(_, dist)| !matches!(dist, Distribution::Categorical(_)))
            .map(|(id, _)| *id)
            .collect();

        if param_order.is_empty() {
            self.fill_remaining_independent_with_rng(
                search_space,
                &intersection,
                history,
                &mut result,
                rng,
            );
            return result;
        }

        param_order.sort_by_key(|id| format!("{id}"));

        // Extract observations, validate, and fit KDEs
        let good_obs = self.extract_observations(&good, &param_order);
        let bad_obs = self.extract_observations(&bad, &param_order);

        let Some((good_kde, bad_kde)) = Self::try_fit_kdes(good_obs, bad_obs, param_order.len())
        else {
            self.fill_remaining_independent_with_rng(
                search_space,
                &intersection,
                history,
                &mut result,
                rng,
            );
            return result;
        };

        let selected = self.select_candidate_with_rng(&good_kde, &bad_kde, rng);

        // Map selected values to parameter ids
        for (idx, param_id) in param_order.iter().enumerate() {
            if let Some(dist) = intersection.get(param_id) {
                let value = selected[idx];
                let param_value = self.convert_to_param_value(value, dist);
                if let Some(pv) = param_value {
                    result.insert(*param_id, pv);
                }
            }
        }

        // Fill remaining parameters using independent TPE sampling
        self.fill_remaining_independent_with_rng(
            search_space,
            &intersection,
            history,
            &mut result,
            rng,
        );
        result
    }

    /// Converts a raw f64 value to a `ParamValue` based on the distribution.
    #[allow(clippy::unused_self)]
    pub(crate) fn convert_to_param_value(
        &self,
        value: f64,
        dist: &Distribution,
    ) -> Option<ParamValue> {
        match dist {
            Distribution::Float(d) => {
                let clamped = value.clamp(d.low, d.high);
                let stepped = if let Some(step) = d.step {
                    let steps = ((clamped - d.low) / step).round();
                    (d.low + steps * step).clamp(d.low, d.high)
                } else {
                    clamped
                };
                Some(ParamValue::Float(stepped))
            }
            Distribution::Int(d) => {
                #[allow(clippy::cast_possible_truncation)]
                let int_value = value.round() as i64;
                let clamped = int_value.clamp(d.low, d.high);
                let stepped = if let Some(step) = d.step {
                    let steps = (clamped - d.low) / step;
                    (d.low + steps * step).clamp(d.low, d.high)
                } else {
                    clamped
                };
                Some(ParamValue::Int(stepped))
            }
            Distribution::Categorical(_) => None,
        }
    }

    /// Selects the best candidate from a set of samples using the joint acquisition function.
    ///
    /// This method implements the core TPE selection criterion: it generates candidates
    /// from the "good" KDE (l(x)) and selects the one that maximizes the ratio l(x)/g(x),
    /// which is equivalent to maximizing `log(l(x)) - log(g(x))`.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn select_candidate(
        &self,
        good_kde: &crate::kde::MultivariateKDE,
        bad_kde: &crate::kde::MultivariateKDE,
    ) -> Vec<f64> {
        let mut rng = self.rng.lock();

        // Generate candidates from the good distribution
        let candidates: Vec<Vec<f64>> = (0..self.n_ei_candidates)
            .map(|_| good_kde.sample(&mut rng))
            .collect();

        // Compute log(l(x)) - log(g(x)) for each candidate
        // This is equivalent to log(l(x)/g(x)) which we want to maximize
        let log_ratios: Vec<f64> = candidates
            .iter()
            .map(|candidate| {
                let log_l = good_kde.log_pdf(candidate);
                let log_g = bad_kde.log_pdf(candidate);
                log_l - log_g
            })
            .collect();

        // Find the candidate with the maximum log ratio
        let mut best_idx = 0;
        let mut best_ratio = f64::NEG_INFINITY;

        for (idx, &ratio) in log_ratios.iter().enumerate() {
            // Handle NaN by treating it as worse than any finite value
            if ratio > best_ratio || (best_ratio.is_nan() && !ratio.is_nan()) {
                best_ratio = ratio;
                best_idx = idx;
            }
        }

        candidates.into_iter().nth(best_idx).unwrap_or_default()
    }

    /// Selects the best candidate using an external RNG.
    ///
    /// This variant accepts an external RNG, used when the caller already holds the lock.
    pub(crate) fn select_candidate_with_rng(
        &self,
        good_kde: &crate::kde::MultivariateKDE,
        bad_kde: &crate::kde::MultivariateKDE,
        rng: &mut fastrand::Rng,
    ) -> Vec<f64> {
        // Generate candidates from the good distribution
        let candidates: Vec<Vec<f64>> = (0..self.n_ei_candidates)
            .map(|_| good_kde.sample(rng))
            .collect();

        // Compute log(l(x)) - log(g(x)) for each candidate
        let log_ratios: Vec<f64> = candidates
            .iter()
            .map(|candidate| {
                let log_l = good_kde.log_pdf(candidate);
                let log_g = bad_kde.log_pdf(candidate);
                log_l - log_g
            })
            .collect();

        // Find the candidate with the maximum log ratio
        let mut best_idx = 0;
        let mut best_ratio = f64::NEG_INFINITY;

        for (idx, &ratio) in log_ratios.iter().enumerate() {
            if ratio > best_ratio || (best_ratio.is_nan() && !ratio.is_nan()) {
                best_ratio = ratio;
                best_idx = idx;
            }
        }

        candidates.into_iter().nth(best_idx).unwrap_or_default()
    }

    /// Fills remaining parameters using independent TPE sampling with an external RNG.
    ///
    /// This variant accepts an external RNG, used when the caller already holds the lock.
    pub(crate) fn fill_remaining_independent_with_rng(
        &self,
        search_space: &HashMap<ParamId, Distribution>,
        _intersection: &HashMap<ParamId, Distribution>,
        history: &[CompletedTrial],
        result: &mut HashMap<ParamId, ParamValue>,
        rng: &mut fastrand::Rng,
    ) {
        // Identify parameters not in result (and not in intersection)
        let missing_params: Vec<(&ParamId, &Distribution)> = search_space
            .iter()
            .filter(|(id, _)| !result.contains_key(id))
            .collect();

        if missing_params.is_empty() {
            return;
        }

        // Split trials for independent sampling
        let (good_trials, bad_trials) = self.split_trials(&history.iter().collect::<Vec<_>>());

        for (param_id, dist) in missing_params {
            let value =
                self.sample_independent_tpe(*param_id, dist, &good_trials, &bad_trials, rng);
            result.insert(*param_id, value);
        }
    }

    /// Samples all parameters using independent TPE sampling.
    ///
    /// This is used as a complete fallback when no intersection search space exists.
    #[cfg(test)]
    pub(crate) fn sample_all_independent(
        &self,
        search_space: &HashMap<ParamId, Distribution>,
        history: &[CompletedTrial],
    ) -> HashMap<ParamId, ParamValue> {
        // Split trials for independent sampling
        let (good_trials, bad_trials) = self.split_trials(&history.iter().collect::<Vec<_>>());

        let mut rng = self.rng.lock();
        let mut result = HashMap::new();

        for (param_id, dist) in search_space {
            let value =
                self.sample_independent_tpe(*param_id, dist, &good_trials, &bad_trials, &mut rng);
            result.insert(*param_id, value);
        }

        result
    }

    /// Samples all parameters using independent TPE sampling with an external RNG.
    ///
    /// This variant accepts an external RNG, used when the caller already holds the lock.
    pub(crate) fn sample_all_independent_with_rng(
        &self,
        search_space: &HashMap<ParamId, Distribution>,
        history: &[CompletedTrial],
        rng: &mut fastrand::Rng,
    ) -> HashMap<ParamId, ParamValue> {
        // Split trials for independent sampling
        let (good_trials, bad_trials) = self.split_trials(&history.iter().collect::<Vec<_>>());

        let mut result = HashMap::new();

        for (param_id, dist) in search_space {
            let value =
                self.sample_independent_tpe(*param_id, dist, &good_trials, &bad_trials, rng);
            result.insert(*param_id, value);
        }

        result
    }

    /// Samples a single parameter using independent TPE.
    ///
    /// This method extracts values for the given parameter from good and bad trials,
    /// fits univariate KDEs, and samples using the TPE acquisition function.
    pub(crate) fn sample_independent_tpe(
        &self,
        param_id: ParamId,
        distribution: &Distribution,
        good_trials: &[&CompletedTrial],
        bad_trials: &[&CompletedTrial],
        rng: &mut fastrand::Rng,
    ) -> ParamValue {
        match distribution {
            Distribution::Float(d) => {
                self.sample_independent_float(param_id, d, good_trials, bad_trials, rng)
            }
            Distribution::Int(d) => {
                self.sample_independent_int(param_id, d, good_trials, bad_trials, rng)
            }
            Distribution::Categorical(d) => {
                self.sample_independent_categorical(param_id, d, good_trials, bad_trials, rng)
            }
        }
    }

    fn sample_independent_float(
        &self,
        param_id: ParamId,
        d: &crate::distribution::FloatDistribution,
        good_trials: &[&CompletedTrial],
        bad_trials: &[&CompletedTrial],
        rng: &mut fastrand::Rng,
    ) -> ParamValue {
        use crate::sampler::tpe::common;

        let good_values: Vec<f64> = good_trials
            .iter()
            .filter_map(|t| t.params.get(&param_id))
            .filter_map(|v| match v {
                ParamValue::Float(f) => Some(*f),
                _ => None,
            })
            .filter(|&v| v >= d.low && v <= d.high)
            .collect();

        let bad_values: Vec<f64> = bad_trials
            .iter()
            .filter_map(|t| t.params.get(&param_id))
            .filter_map(|v| match v {
                ParamValue::Float(f) => Some(*f),
                _ => None,
            })
            .filter(|&v| v >= d.low && v <= d.high)
            .collect();

        if good_values.is_empty() || bad_values.is_empty() {
            return crate::sampler::common::sample_random(rng, &Distribution::Float(d.clone()));
        }

        let value =
            common::sample_tpe_float(d, good_values, bad_values, self.n_ei_candidates, None, rng);
        ParamValue::Float(value)
    }

    fn sample_independent_int(
        &self,
        param_id: ParamId,
        d: &crate::distribution::IntDistribution,
        good_trials: &[&CompletedTrial],
        bad_trials: &[&CompletedTrial],
        rng: &mut fastrand::Rng,
    ) -> ParamValue {
        use crate::sampler::tpe::common;

        let good_values: Vec<i64> = good_trials
            .iter()
            .filter_map(|t| t.params.get(&param_id))
            .filter_map(|v| match v {
                ParamValue::Int(i) => Some(*i),
                _ => None,
            })
            .filter(|&v| v >= d.low && v <= d.high)
            .collect();

        let bad_values: Vec<i64> = bad_trials
            .iter()
            .filter_map(|t| t.params.get(&param_id))
            .filter_map(|v| match v {
                ParamValue::Int(i) => Some(*i),
                _ => None,
            })
            .filter(|&v| v >= d.low && v <= d.high)
            .collect();

        if good_values.is_empty() || bad_values.is_empty() {
            return crate::sampler::common::sample_random(rng, &Distribution::Int(d.clone()));
        }

        let value =
            common::sample_tpe_int(d, good_values, bad_values, self.n_ei_candidates, None, rng);
        ParamValue::Int(value)
    }

    #[allow(clippy::unused_self)]
    fn sample_independent_categorical(
        &self,
        param_id: ParamId,
        d: &crate::distribution::CategoricalDistribution,
        good_trials: &[&CompletedTrial],
        bad_trials: &[&CompletedTrial],
        rng: &mut fastrand::Rng,
    ) -> ParamValue {
        use crate::sampler::tpe::common;

        let good_indices: Vec<usize> = good_trials
            .iter()
            .filter_map(|t| t.params.get(&param_id))
            .filter_map(|v| match v {
                ParamValue::Categorical(i) => Some(*i),
                _ => None,
            })
            .filter(|&i| i < d.n_choices)
            .collect();

        let bad_indices: Vec<usize> = bad_trials
            .iter()
            .filter_map(|t| t.params.get(&param_id))
            .filter_map(|v| match v {
                ParamValue::Categorical(i) => Some(*i),
                _ => None,
            })
            .filter(|&i| i < d.n_choices)
            .collect();

        if good_indices.is_empty() || bad_indices.is_empty() {
            return crate::sampler::common::sample_random(
                rng,
                &Distribution::Categorical(d.clone()),
            );
        }

        let idx = common::sample_tpe_categorical(d.n_choices, &good_indices, &bad_indices, rng);
        ParamValue::Categorical(idx)
    }
}
