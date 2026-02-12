//! Trial lifecycle management for optimization runs.
//!
//! A [`Trial`] represents a single evaluation of the objective function. The study
//! creates trials, the objective function samples parameters from them via
//! [`Parameter::suggest`](crate::parameter::Parameter::suggest), and reports
//! intermediate values for pruning decisions.
//!
//! # Lifecycle
//!
//! 1. **Created** — `Study` creates a trial with [`Trial::new`] or internally via
//!    `Trial::with_sampler`.
//! 2. **Running** — The objective calls [`Trial::suggest_param`] to sample parameters
//!    and optionally [`Trial::report`] / [`Trial::should_prune`] for early stopping.
//! 3. **Completed / Failed / Pruned** — The study marks the trial's final state.
//!
//! # User Attributes
//!
//! Trials support arbitrary key-value metadata via [`Trial::set_user_attr`] and
//! [`Trial::user_attr`], useful for logging hyperparameters, hardware info, or
//! debug notes alongside the optimization results.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::distribution::Distribution;
use crate::error::{Error, Result};
use crate::param::ParamValue;
use crate::parameter::{ParamId, Parameter};
use crate::pruner::Pruner;
use crate::sampler::{CompletedTrial, Sampler};
use crate::types::TrialState;

/// A user attribute value that can be stored on a trial.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AttrValue {
    /// A floating-point attribute.
    Float(f64),
    /// An integer attribute.
    Int(i64),
    /// A string attribute.
    String(String),
    /// A boolean attribute.
    Bool(bool),
}

impl From<f64> for AttrValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<i64> for AttrValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<String> for AttrValue {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl From<&str> for AttrValue {
    fn from(v: &str) -> Self {
        Self::String(v.to_owned())
    }
}

impl From<bool> for AttrValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

/// A single evaluation of the objective function.
///
/// Each trial has a unique ID and stores the sampled parameters along with
/// their distributions. The trial progresses through states:
/// `Running` → `Complete` / `Failed` / `Pruned`.
///
/// Trials use a [`Sampler`](crate::sampler::Sampler) to generate parameter
/// values. When created through [`Study::create_trial`](crate::Study::create_trial),
/// the trial receives the study's sampler and access to the history of
/// completed trials for informed sampling.
///
/// # Examples
///
/// ```
/// use optimizer::Trial;
/// use optimizer::parameter::{FloatParam, Parameter};
///
/// let mut trial = Trial::new(0);
/// let x = FloatParam::new(-5.0, 5.0).suggest(&mut trial).unwrap();
/// ```
#[derive(Clone)]
pub struct Trial {
    /// Unique identifier for this trial.
    id: u64,
    /// Current state of the trial.
    state: TrialState,
    /// Sampled parameter values, keyed by parameter id.
    params: HashMap<ParamId, ParamValue>,
    /// Parameter distributions, keyed by parameter id.
    distributions: HashMap<ParamId, Distribution>,
    /// Human-readable labels for parameters, keyed by parameter id.
    param_labels: HashMap<ParamId, String>,
    /// The sampler to use for generating parameter values.
    sampler: Option<Arc<dyn Sampler>>,
    /// Access to the history of completed trials (shared with Study).
    history: Option<Arc<RwLock<Vec<CompletedTrial<f64>>>>>,
    /// Intermediate objective values reported at each step.
    intermediate_values: Vec<(u64, f64)>,
    /// The pruner used to decide whether to stop this trial early.
    pruner: Option<Arc<dyn Pruner>>,
    /// User-defined attributes for logging, debugging, and analysis.
    user_attrs: HashMap<String, AttrValue>,
    /// Pre-filled parameter values from enqueue (used instead of sampling).
    fixed_params: HashMap<ParamId, ParamValue>,
    /// Constraint values for this trial (<=0.0 means feasible).
    constraint_values: Vec<f64>,
}

impl core::fmt::Debug for Trial {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Trial")
            .field("id", &self.id)
            .field("state", &self.state)
            .field("params", &self.params)
            .field("distributions", &self.distributions)
            .field("param_labels", &self.param_labels)
            .field("has_sampler", &self.sampler.is_some())
            .field("has_history", &self.history.is_some())
            .field("intermediate_values", &self.intermediate_values)
            .field("has_pruner", &self.pruner.is_some())
            .field("user_attrs", &self.user_attrs)
            .field("fixed_params", &self.fixed_params)
            .field("constraint_values", &self.constraint_values)
            .finish()
    }
}

impl Trial {
    /// Create a new trial with the given ID.
    ///
    /// The trial starts in the `Running` state with no parameters sampled.
    /// This constructor creates a trial without a sampler, which will fall
    /// back to random sampling for [`suggest_param`](Self::suggest_param) calls.
    ///
    /// For trials that use the study's sampler, the study creates them
    /// internally via `Trial::with_sampler`.
    ///
    /// # Arguments
    ///
    /// * `id` - A unique identifier for this trial.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::Trial;
    ///
    /// let trial = Trial::new(0);
    /// assert_eq!(trial.id(), 0);
    /// ```
    #[must_use]
    pub fn new(id: u64) -> Self {
        Self {
            id,
            state: TrialState::Running,
            params: HashMap::new(),
            distributions: HashMap::new(),
            param_labels: HashMap::new(),
            sampler: None,
            history: None,
            intermediate_values: Vec::new(),
            pruner: None,
            user_attrs: HashMap::new(),
            fixed_params: HashMap::new(),
            constraint_values: Vec::new(),
        }
    }

    /// Create a new trial with a sampler and access to trial history.
    ///
    /// Used internally by `Study::create_trial()` to create trials that use
    /// the study's sampler for informed parameter suggestions.
    ///
    /// # Arguments
    ///
    /// * `id` - A unique identifier for this trial.
    /// * `sampler` - The sampler to use for generating parameter values.
    /// * `history` - Shared access to the history of completed trials.
    pub(crate) fn with_sampler(
        id: u64,
        sampler: Arc<dyn Sampler>,
        history: Arc<RwLock<Vec<CompletedTrial<f64>>>>,
        pruner: Arc<dyn Pruner>,
    ) -> Self {
        Self {
            id,
            state: TrialState::Running,
            params: HashMap::new(),
            distributions: HashMap::new(),
            param_labels: HashMap::new(),
            sampler: Some(sampler),
            history: Some(history),
            intermediate_values: Vec::new(),
            pruner: Some(pruner),
            user_attrs: HashMap::new(),
            fixed_params: HashMap::new(),
            constraint_values: Vec::new(),
        }
    }

    /// Set pre-filled parameters on this trial.
    ///
    /// When [`suggest_param`](Self::suggest_param) is called for a parameter
    /// that has a fixed value, the fixed value is used instead of sampling.
    pub(crate) fn set_fixed_params(&mut self, params: HashMap<ParamId, ParamValue>) {
        self.fixed_params = params;
    }

    /// Sample a value from the given distribution using the sampler.
    ///
    /// If the trial has a sampler, delegates to the sampler's sample method
    /// with the history of completed trials. Otherwise, falls back to
    /// [`RandomSampler`](crate::sampler::random::RandomSampler).
    fn sample_value(&self, distribution: &Distribution) -> ParamValue {
        if let (Some(sampler), Some(history)) = (&self.sampler, &self.history) {
            let history_guard = history.read();
            sampler.sample(distribution, self.id, &history_guard)
        } else {
            // Fallback to RandomSampler when no sampler is configured
            use crate::sampler::random::RandomSampler;
            let fallback = RandomSampler::new();
            fallback.sample(distribution, self.id, &[])
        }
    }

    /// Return the unique ID of this trial.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Return the current state of this trial.
    #[must_use]
    pub fn state(&self) -> TrialState {
        self.state
    }

    /// Return a reference to the sampled parameters, keyed by [`ParamId`](crate::parameter::ParamId).
    #[must_use]
    pub fn params(&self) -> &HashMap<ParamId, ParamValue> {
        &self.params
    }

    /// Return a reference to the parameter distributions, keyed by [`ParamId`](crate::parameter::ParamId).
    #[must_use]
    pub fn distributions(&self) -> &HashMap<ParamId, Distribution> {
        &self.distributions
    }

    /// Return a reference to the parameter labels, keyed by [`ParamId`](crate::parameter::ParamId).
    #[must_use]
    pub fn param_labels(&self) -> &HashMap<ParamId, String> {
        &self.param_labels
    }

    /// Report an intermediate objective value at a given step.
    ///
    /// Call this during iterative training (e.g., once per epoch) so the
    /// [`Pruner`](crate::pruner::Pruner) can decide whether to stop the trial
    /// early. Steps should be monotonically increasing; duplicate steps
    /// overwrite the previous value.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::Trial;
    ///
    /// let mut trial = Trial::new(0);
    /// for epoch in 0..10 {
    ///     let loss = 1.0 / (epoch as f64 + 1.0);
    ///     trial.report(epoch, loss);
    /// }
    /// assert_eq!(trial.intermediate_values().len(), 10);
    /// ```
    pub fn report(&mut self, step: u64, value: f64) {
        if let Some(entry) = self
            .intermediate_values
            .iter_mut()
            .find(|(s, _)| *s == step)
        {
            entry.1 = value;
        } else {
            self.intermediate_values.push((step, value));
        }
    }

    /// Ask whether this trial should be pruned at the current step.
    ///
    /// Return `true` if the pruner recommends stopping this trial based on
    /// the intermediate values reported so far. When `true`, the objective
    /// should return early with `Err(TrialPruned)?`.
    ///
    /// Always returns `false` when no pruner is configured.
    #[must_use]
    pub fn should_prune(&self) -> bool {
        let (Some(pruner), Some(history)) = (&self.pruner, &self.history) else {
            return false;
        };
        let Some(&(step, _)) = self.intermediate_values.last() else {
            return false;
        };
        let history_guard = history.read();
        let prune = pruner.should_prune(self.id, step, &self.intermediate_values, &history_guard);
        if prune {
            trace_info!(trial_id = self.id, step, "pruner recommends stopping");
        }
        prune
    }

    /// Return all intermediate values reported so far as `(step, value)` pairs.
    #[must_use]
    pub fn intermediate_values(&self) -> &[(u64, f64)] {
        &self.intermediate_values
    }

    /// Set a user attribute on this trial.
    ///
    /// User attributes are arbitrary key-value pairs for logging, debugging,
    /// or analysis. Values can be `f64`, `i64`, `String`, `&str`, or `bool`
    /// (anything implementing `Into<AttrValue>`).
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::Trial;
    ///
    /// let mut trial = Trial::new(0);
    /// trial.set_user_attr("gpu", "A100");
    /// trial.set_user_attr("batch_size", 64_i64);
    /// trial.set_user_attr("accuracy", 0.95);
    /// ```
    pub fn set_user_attr(&mut self, key: impl Into<String>, value: impl Into<AttrValue>) {
        self.user_attrs.insert(key.into(), value.into());
    }

    /// Return a user attribute by key, or `None` if it does not exist.
    #[must_use]
    pub fn user_attr(&self, key: &str) -> Option<&AttrValue> {
        self.user_attrs.get(key)
    }

    /// Return all user attributes as a map.
    #[must_use]
    pub fn user_attrs(&self) -> &HashMap<String, AttrValue> {
        &self.user_attrs
    }

    /// Set constraint values for this trial.
    ///
    /// Each element represents one constraint. A value ≤ 0.0 means the
    /// constraint is satisfied (feasible); a value > 0.0 means violated.
    /// Constrained samplers (e.g., NSGA-II with constraints) use these values
    /// to prefer feasible solutions.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::Trial;
    ///
    /// let mut trial = Trial::new(0);
    /// // Two constraints: first satisfied, second violated
    /// trial.set_constraints(vec![-0.5, 0.3]);
    /// assert_eq!(trial.constraint_values(), &[-0.5, 0.3]);
    /// ```
    pub fn set_constraints(&mut self, values: Vec<f64>) {
        self.constraint_values = values;
    }

    /// Return the constraint values for this trial.
    #[must_use]
    pub fn constraint_values(&self) -> &[f64] {
        &self.constraint_values
    }

    /// Set the trial state to `Complete`.
    pub(crate) fn set_complete(&mut self) {
        self.state = TrialState::Complete;
    }

    /// Set the trial state to `Failed`.
    pub(crate) fn set_failed(&mut self) {
        self.state = TrialState::Failed;
    }

    /// Set the trial state to `Pruned`.
    pub(crate) fn set_pruned(&mut self) {
        self.state = TrialState::Pruned;
    }

    /// Suggest a parameter value using a [`Parameter`] definition.
    ///
    /// This is the primary entry point for sampling parameters. It handles
    /// validation, caching, conflict detection, sampling, and conversion.
    ///
    /// # Arguments
    ///
    /// * `param` - The parameter definition.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The parameter fails validation
    /// - The parameter conflicts with a previously suggested parameter of the same id
    /// - Sampling or conversion fails
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::Trial;
    /// use optimizer::parameter::{BoolParam, FloatParam, IntParam, Parameter};
    ///
    /// let x_param = FloatParam::new(0.0, 1.0);
    /// let n_param = IntParam::new(1, 10);
    /// let flag_param = BoolParam::new();
    ///
    /// let mut trial = Trial::new(0);
    ///
    /// let x = trial.suggest_param(&x_param).unwrap();
    /// let n = trial.suggest_param(&n_param).unwrap();
    /// let flag = trial.suggest_param(&flag_param).unwrap();
    /// ```
    pub fn suggest_param<P: Parameter>(&mut self, param: &P) -> Result<P::Value> {
        param.validate()?;

        let param_id = param.id();
        let distribution = param.distribution();

        // Check if parameter already exists
        if let Some(existing_dist) = self.distributions.get(&param_id) {
            if *existing_dist == distribution {
                // Same distribution, return cached value
                if let Some(value) = self.params.get(&param_id) {
                    return param.cast_param_value(value);
                }
            }
            // Distribution exists but doesn't match
            return Err(Error::ParameterConflict {
                name: param.label(),
                reason: "parameter was previously sampled with different configuration or type"
                    .to_string(),
            });
        }

        // Check for a pre-filled (enqueued) value for this parameter
        let value = if let Some(fixed_value) = self.fixed_params.remove(&param_id) {
            fixed_value
        } else {
            // Sample using the sampler
            self.sample_value(&distribution)
        };

        let result = param.cast_param_value(&value)?;

        trace_debug!(
            trial_id = self.id,
            param = %param.label(),
            value = %value,
            "parameter sampled"
        );

        // Store distribution, value, and label
        self.distributions.insert(param_id, distribution);
        self.params.insert(param_id, value);
        self.param_labels.insert(param_id, param.label());

        Ok(result)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use crate::parameter::{BoolParam, CategoricalParam, FloatParam, IntParam, Parameter};
    use crate::types::TrialState;

    #[test]
    fn trial_state() {
        // from test_trial_state (L643 of integration.rs)
        let trial = super::Trial::new(0);
        assert_eq!(trial.state(), TrialState::Running);
    }

    #[test]
    fn trial_params_access() {
        // from test_trial_params_access (L651)
        let x_param = FloatParam::new(0.0, 1.0);
        let n_param = IntParam::new(1, 10);
        let mut trial = super::Trial::new(0);

        x_param.suggest(&mut trial).unwrap();
        n_param.suggest(&mut trial).unwrap();

        let params = trial.params();
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn trial_debug_format() {
        // from test_trial_debug_format (L792)
        let param = FloatParam::new(0.0, 1.0);
        let mut trial = super::Trial::new(42);
        param.suggest(&mut trial).unwrap();

        let debug_str = format!("{trial:?}");

        assert!(debug_str.contains("Trial"));
        assert!(debug_str.contains("42"));
        assert!(debug_str.contains("has_sampler"));
    }

    #[test]
    fn distributions_access() {
        // from test_distributions_access (L960)
        let x_param = FloatParam::new(0.0, 1.0);
        let n_param = IntParam::new(1, 10);
        let opt_param = CategoricalParam::new(vec!["a", "b", "c"]);
        let mut trial = super::Trial::new(0);

        x_param.suggest(&mut trial).unwrap();
        n_param.suggest(&mut trial).unwrap();
        opt_param.suggest(&mut trial).unwrap();

        let dists = trial.distributions();
        assert_eq!(dists.len(), 3);
    }

    #[test]
    fn multiple_parameters_independent_caching() {
        // from test_multiple_parameters_independent_caching (L356)
        let x_param = FloatParam::new(0.0, 1.0);
        let y_param = FloatParam::new(0.0, 1.0);
        let n_param = IntParam::new(1, 10);
        let opt_param = CategoricalParam::new(vec!["a", "b"]);
        let mut trial = super::Trial::new(0);

        let x = x_param.suggest(&mut trial).unwrap();
        let y = y_param.suggest(&mut trial).unwrap();
        let n = n_param.suggest(&mut trial).unwrap();
        let opt = opt_param.suggest(&mut trial).unwrap();

        assert_eq!(x, x_param.suggest(&mut trial).unwrap());
        assert_eq!(y, y_param.suggest(&mut trial).unwrap());
        assert_eq!(n, n_param.suggest(&mut trial).unwrap());
        assert_eq!(opt, opt_param.suggest(&mut trial).unwrap());
    }

    #[test]
    fn suggest_bool_multiple_parameters() {
        // from test_suggest_bool_multiple_parameters (L1131)
        let dropout_param = BoolParam::new();
        let batchnorm_param = BoolParam::new();
        let skip_param = BoolParam::new();
        let mut trial = super::Trial::new(0);

        let a = dropout_param.suggest(&mut trial).unwrap();
        let b = batchnorm_param.suggest(&mut trial).unwrap();
        let c = skip_param.suggest(&mut trial).unwrap();

        assert_eq!(a, dropout_param.suggest(&mut trial).unwrap());
        assert_eq!(b, batchnorm_param.suggest(&mut trial).unwrap());
        assert_eq!(c, skip_param.suggest(&mut trial).unwrap());
    }

    #[test]
    fn param_name() {
        // from test_param_name (L1312)
        let param = FloatParam::new(0.0, 1.0).name("learning_rate");
        let mut trial = super::Trial::new(0);
        param.suggest(&mut trial).unwrap();

        let labels = trial.param_labels();
        let label = labels.values().next().unwrap();
        assert_eq!(label, "learning_rate");
    }

    #[test]
    fn step_float_snaps_to_grid() {
        // from test_step_float_snaps_to_grid (L676)
        let param = FloatParam::new(0.0, 1.0).step(0.25);
        let mut trial = super::Trial::new(0);

        let x = param.suggest(&mut trial).unwrap();

        let valid_values = [0.0, 0.25, 0.5, 0.75, 1.0];
        let is_valid = valid_values.iter().any(|&v| (x - v).abs() < 1e-10);
        assert!(is_valid, "stepped float {x} should snap to grid");
    }

    #[test]
    fn step_int_snaps_to_grid() {
        // from test_step_int_snaps_to_grid (L689)
        let param = IntParam::new(0, 100).step(25);
        let mut trial = super::Trial::new(0);

        let n = param.suggest(&mut trial).unwrap();

        assert!(
            n % 25 == 0 && (0..=100).contains(&n),
            "stepped int {n} should snap to grid"
        );
    }

    #[test]
    fn int_bounds_with_low_equals_high() {
        // from test_int_bounds_with_low_equals_high (L1086)
        let mut trial = super::Trial::new(0);

        let n_param = IntParam::new(5, 5);
        let n = n_param.suggest(&mut trial).unwrap();
        assert_eq!(n, 5);

        let x_param = FloatParam::new(3.0, 3.0);
        let x = x_param.suggest(&mut trial).unwrap();
        assert_eq!(x, 3.0);
    }

    #[test]
    fn single_value_float_range() {
        // from test_single_value_float_range (L1296)
        let param = FloatParam::new(4.2, 4.2);
        let mut trial = super::Trial::new(0);

        let x = param.suggest(&mut trial).unwrap();
        assert!(
            (x - 4.2).abs() < f64::EPSILON,
            "single-value range should return that value"
        );
    }
}
