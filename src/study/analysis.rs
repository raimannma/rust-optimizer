use crate::sampler::CompletedTrial;
use crate::types::TrialState;

use super::Study;

impl<V> Study<V>
where
    V: PartialOrd,
{
    /// Return the trial with the best objective value.
    ///
    /// The "best" trial depends on the optimization direction:
    /// - `Direction::Minimize`: Returns the trial with the lowest objective value.
    /// - `Direction::Maximize`: Returns the trial with the highest objective value.
    ///
    /// When constraints are present, feasible trials always rank above infeasible
    /// trials. Among infeasible trials, those with lower total constraint violation
    /// are preferred.
    ///
    /// # Errors
    ///
    /// Returns `Error::NoCompletedTrials` if no trials have been completed.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    ///
    /// // Error when no trials completed
    /// assert!(study.best_trial().is_err());
    ///
    /// let x_param = FloatParam::new(0.0, 1.0);
    ///
    /// let mut trial1 = study.create_trial();
    /// let _ = x_param.suggest(&mut trial1);
    /// study.complete_trial(trial1, 0.8);
    ///
    /// let mut trial2 = study.create_trial();
    /// let _ = x_param.suggest(&mut trial2);
    /// study.complete_trial(trial2, 0.3);
    ///
    /// let best = study.best_trial().unwrap();
    /// assert_eq!(best.value, 0.3); // Minimize: lower is better
    /// ```
    pub fn best_trial(&self) -> crate::Result<CompletedTrial<V>>
    where
        V: Clone,
    {
        let trials = self.storage.trials_arc().read();
        let direction = self.direction;

        let best = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .max_by(|a, b| Self::compare_trials(a, b, direction))
            .ok_or(crate::Error::NoCompletedTrials)?;

        Ok(best.clone())
    }

    /// Return the best objective value found so far.
    ///
    /// The "best" value depends on the optimization direction:
    /// - `Direction::Minimize`: Returns the lowest objective value.
    /// - `Direction::Maximize`: Returns the highest objective value.
    ///
    /// # Errors
    ///
    /// Returns `Error::NoCompletedTrials` if no trials have been completed.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Maximize);
    ///
    /// // Error when no trials completed
    /// assert!(study.best_value().is_err());
    ///
    /// let x_param = FloatParam::new(0.0, 1.0);
    ///
    /// let mut trial1 = study.create_trial();
    /// let _ = x_param.suggest(&mut trial1);
    /// study.complete_trial(trial1, 0.3);
    ///
    /// let mut trial2 = study.create_trial();
    /// let _ = x_param.suggest(&mut trial2);
    /// study.complete_trial(trial2, 0.8);
    ///
    /// let best = study.best_value().unwrap();
    /// assert_eq!(best, 0.8); // Maximize: higher is better
    /// ```
    pub fn best_value(&self) -> crate::Result<V>
    where
        V: Clone,
    {
        self.best_trial().map(|trial| trial.value)
    }

    /// Return the top `n` trials sorted by objective value.
    ///
    /// For `Direction::Minimize`, returns trials with the lowest values.
    /// For `Direction::Maximize`, returns trials with the highest values.
    /// Only includes completed trials (not failed or pruned).
    ///
    /// If fewer than `n` completed trials exist, returns all of them.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let x = FloatParam::new(0.0, 10.0);
    ///
    /// for val in [5.0, 1.0, 3.0] {
    ///     let mut t = study.create_trial();
    ///     let _ = x.suggest(&mut t);
    ///     study.complete_trial(t, val);
    /// }
    ///
    /// let top2 = study.top_trials(2);
    /// assert_eq!(top2.len(), 2);
    /// assert!(top2[0].value <= top2[1].value);
    /// ```
    #[must_use]
    pub fn top_trials(&self, n: usize) -> Vec<CompletedTrial<V>>
    where
        V: Clone,
    {
        let trials = self.storage.trials_arc().read();
        let direction = self.direction;
        // Sort indices instead of cloning all trials, then clone only the top N.
        let mut indices: Vec<usize> = trials
            .iter()
            .enumerate()
            .filter(|(_, t)| t.state == TrialState::Complete)
            .map(|(i, _)| i)
            .collect();
        // Sort best-first: reverse the compare_trials ordering (which is designed for max_by)
        indices.sort_by(|&a, &b| Self::compare_trials(&trials[b], &trials[a], direction));
        indices.truncate(n);
        indices.iter().map(|&i| trials[i].clone()).collect()
    }
}

impl<V> Study<V>
where
    V: PartialOrd + Clone + Into<f64>,
{
    /// Compute parameter importance scores using Spearman rank correlation.
    ///
    /// For each parameter, the absolute Spearman correlation between its values
    /// and the objective values is computed across all completed trials. Scores
    /// are normalized so they sum to 1.0 and sorted in descending order.
    ///
    /// Parameters that appear in fewer than 2 trials are omitted.
    /// Returns an empty `Vec` if the study has fewer than 2 completed trials.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let x = FloatParam::new(0.0, 10.0).name("x");
    ///
    /// study
    ///     .optimize(20, |trial: &mut optimizer::Trial| {
    ///         let xv = x.suggest(trial)?;
    ///         Ok::<_, optimizer::Error>(xv * xv)
    ///     })
    ///     .unwrap();
    ///
    /// let importance = study.param_importance();
    /// assert_eq!(importance.len(), 1);
    /// assert_eq!(importance[0].0, "x");
    /// ```
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn param_importance(&self) -> Vec<(String, f64)> {
        use std::collections::BTreeSet;

        use crate::importance::spearman;
        use crate::param::ParamValue;
        use crate::types::TrialState;

        let trials = self.storage.trials_arc().read();
        let complete: Vec<_> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();

        if complete.len() < 2 {
            return Vec::new();
        }

        // Collect all parameter IDs across trials.
        let all_param_ids: BTreeSet<_> = complete.iter().flat_map(|t| t.params.keys()).collect();

        let mut scores: Vec<(String, f64)> = Vec::with_capacity(all_param_ids.len());

        for &param_id in &all_param_ids {
            // Collect (param_value_f64, objective_f64) for trials that have this param.
            let mut param_vals = Vec::with_capacity(complete.len());
            let mut obj_vals = Vec::with_capacity(complete.len());

            for trial in &complete {
                if let Some(pv) = trial.params.get(param_id) {
                    let f = match *pv {
                        ParamValue::Float(v) => v,
                        ParamValue::Int(v) => v as f64,
                        ParamValue::Categorical(v) => v as f64,
                    };
                    param_vals.push(f);
                    obj_vals.push(trial.value.clone().into());
                }
            }

            if param_vals.len() < 2 {
                continue;
            }

            let corr = spearman(&param_vals, &obj_vals).abs();

            // Determine label: use param_labels if available, else "param_{id}".
            let label = complete
                .iter()
                .find_map(|t| t.param_labels.get(param_id))
                .map_or_else(|| param_id.to_string(), Clone::clone);

            scores.push((label, corr));
        }

        // Normalize so scores sum to 1.0.
        let sum: f64 = scores.iter().map(|(_, s)| *s).sum();
        if sum > 0.0 {
            for entry in &mut scores {
                entry.1 /= sum;
            }
        }

        // Sort descending by score.
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        scores
    }

    /// Compute parameter importance using fANOVA (functional ANOVA) with
    /// default configuration.
    ///
    /// Fits a random forest to the trial data and decomposes variance into
    /// per-parameter main effects and pairwise interaction effects. This is
    /// more accurate than correlation-based importance ([`Self::param_importance`])
    /// and can detect non-linear relationships and parameter interactions.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::NoCompletedTrials`] if fewer than 2 trials have completed.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let x = FloatParam::new(0.0, 10.0).name("x");
    /// let y = FloatParam::new(0.0, 10.0).name("y");
    ///
    /// study
    ///     .optimize(30, |trial: &mut optimizer::Trial| {
    ///         let xv = x.suggest(trial)?;
    ///         let yv = y.suggest(trial)?;
    ///         Ok::<_, optimizer::Error>(xv * xv + 0.1 * yv)
    ///     })
    ///     .unwrap();
    ///
    /// let result = study.fanova().unwrap();
    /// assert!(!result.main_effects.is_empty());
    /// ```
    pub fn fanova(&self) -> crate::Result<crate::fanova::FanovaResult> {
        self.fanova_with_config(&crate::fanova::FanovaConfig::default())
    }

    /// Compute parameter importance using fANOVA with custom configuration.
    ///
    /// See [`Self::fanova`] for details. The [`FanovaConfig`](crate::fanova::FanovaConfig)
    /// allows tuning the number of trees, tree depth, and random seed.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::NoCompletedTrials`] if fewer than 2 trials have completed.
    #[allow(clippy::cast_precision_loss)]
    pub fn fanova_with_config(
        &self,
        config: &crate::fanova::FanovaConfig,
    ) -> crate::Result<crate::fanova::FanovaResult> {
        use std::collections::BTreeSet;

        use crate::fanova::compute_fanova;
        use crate::param::ParamValue;
        use crate::types::TrialState;

        let trials = self.storage.trials_arc().read();
        let complete: Vec<_> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();

        if complete.len() < 2 {
            return Err(crate::Error::NoCompletedTrials);
        }

        // Collect all parameter IDs in a stable order.
        let all_param_ids: Vec<_> = {
            let set: BTreeSet<_> = complete.iter().flat_map(|t| t.params.keys()).collect();
            set.into_iter().collect()
        };

        if all_param_ids.is_empty() {
            return Ok(crate::fanova::FanovaResult {
                main_effects: Vec::new(),
                interactions: Vec::new(),
            });
        }

        // Build feature matrix (only trials that have all parameters).
        let mut data = Vec::with_capacity(complete.len());
        let mut targets = Vec::with_capacity(complete.len());

        for trial in &complete {
            let mut row = Vec::with_capacity(all_param_ids.len());
            let mut has_all = true;

            for &pid in &all_param_ids {
                if let Some(pv) = trial.params.get(pid) {
                    row.push(match *pv {
                        ParamValue::Float(v) => v,
                        ParamValue::Int(v) => v as f64,
                        ParamValue::Categorical(v) => v as f64,
                    });
                } else {
                    has_all = false;
                    break;
                }
            }

            if has_all {
                data.push(row);
                targets.push(trial.value.clone().into());
            }
        }

        if data.len() < 2 {
            return Err(crate::Error::NoCompletedTrials);
        }

        // Build feature names from parameter labels.
        let feature_names: Vec<String> = all_param_ids
            .iter()
            .map(|&pid| {
                complete
                    .iter()
                    .find_map(|t| t.param_labels.get(pid))
                    .map_or_else(|| pid.to_string(), Clone::clone)
            })
            .collect();

        Ok(compute_fanova(&data, &targets, &feature_names, config))
    }
}
