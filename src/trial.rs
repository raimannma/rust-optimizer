//! Trial implementation for tracking sampled parameters and trial state.

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

/// A trial represents a single evaluation of the objective function.
///
/// Each trial has a unique ID and stores the sampled parameters along with
/// their distributions. The trial progresses through states: Running -> Complete/Failed.
///
/// Trials use a sampler to generate parameter values. When created through
/// `Study::create_trial()`, the trial receives the study's sampler and access
/// to the history of completed trials for informed sampling.
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
            .finish()
    }
}

impl Trial {
    /// Creates a new trial with the given ID.
    ///
    /// The trial starts in the `Running` state with no parameters sampled.
    /// This constructor creates a trial without a sampler, which will use
    /// local random sampling for suggest methods.
    ///
    /// For trials that use the study's sampler, use `Trial::with_sampler` instead.
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
        }
    }

    /// Creates a new trial with a sampler and access to trial history.
    ///
    /// This constructor is used by `Study::create_trial()` to create trials
    /// that use the study's sampler for informed parameter suggestions.
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
        }
    }

    /// Sets pre-filled parameters on this trial.
    ///
    /// When `suggest_param` is called for a parameter that has a fixed value,
    /// the fixed value is used instead of sampling.
    pub(crate) fn set_fixed_params(&mut self, params: HashMap<ParamId, ParamValue>) {
        self.fixed_params = params;
    }

    /// Samples a value from the given distribution using the sampler.
    ///
    /// If the trial has a sampler, it delegates to the sampler's sample method
    /// with the history of completed trials. Otherwise, it uses the `RandomSampler`
    /// as a fallback.
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

    /// Returns the unique ID of this trial.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Returns the current state of this trial.
    #[must_use]
    pub fn state(&self) -> TrialState {
        self.state
    }

    /// Returns a reference to the sampled parameters.
    #[must_use]
    pub fn params(&self) -> &HashMap<ParamId, ParamValue> {
        &self.params
    }

    /// Returns a reference to the parameter distributions.
    #[must_use]
    pub fn distributions(&self) -> &HashMap<ParamId, Distribution> {
        &self.distributions
    }

    /// Returns a reference to the parameter labels.
    #[must_use]
    pub fn param_labels(&self) -> &HashMap<ParamId, String> {
        &self.param_labels
    }

    /// Reports an intermediate objective value at a given step.
    ///
    /// Steps should be monotonically increasing (e.g., epoch number).
    /// Duplicate steps overwrite the previous value.
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
    /// Returns `true` if the pruner recommends stopping this trial.
    /// The caller should return `Err(TrialPruned)` from the objective.
    #[must_use]
    pub fn should_prune(&self) -> bool {
        let (Some(pruner), Some(history)) = (&self.pruner, &self.history) else {
            return false;
        };
        let Some(&(step, _)) = self.intermediate_values.last() else {
            return false;
        };
        let history_guard = history.read();
        pruner.should_prune(self.id, step, &self.intermediate_values, &history_guard)
    }

    /// Returns all intermediate values reported so far.
    #[must_use]
    pub fn intermediate_values(&self) -> &[(u64, f64)] {
        &self.intermediate_values
    }

    /// Sets a user attribute on this trial.
    pub fn set_user_attr(&mut self, key: impl Into<String>, value: impl Into<AttrValue>) {
        self.user_attrs.insert(key.into(), value.into());
    }

    /// Gets a user attribute by key.
    #[must_use]
    pub fn user_attr(&self, key: &str) -> Option<&AttrValue> {
        self.user_attrs.get(key)
    }

    /// Returns all user attributes.
    #[must_use]
    pub fn user_attrs(&self) -> &HashMap<String, AttrValue> {
        &self.user_attrs
    }

    /// Sets the trial state to Complete.
    pub(crate) fn set_complete(&mut self) {
        self.state = TrialState::Complete;
    }

    /// Sets the trial state to Failed.
    pub(crate) fn set_failed(&mut self) {
        self.state = TrialState::Failed;
    }

    /// Sets the trial state to Pruned.
    pub(crate) fn set_pruned(&mut self) {
        self.state = TrialState::Pruned;
    }

    /// Suggests a parameter value using a [`Parameter`] definition.
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

        // Store distribution, value, and label
        self.distributions.insert(param_id, distribution);
        self.params.insert(param_id, value);
        self.param_labels.insert(param_id, param.label());

        Ok(result)
    }
}
