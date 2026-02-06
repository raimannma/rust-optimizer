//! Trial implementation for tracking sampled parameters and trial state.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::distribution::Distribution;
use crate::error::{Error, Result};
use crate::param::ParamValue;
use crate::parameter::{ParamId, Parameter};
use crate::sampler::{CompletedTrial, Sampler};
use crate::types::TrialState;

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
    ) -> Self {
        Self {
            id,
            state: TrialState::Running,
            params: HashMap::new(),
            distributions: HashMap::new(),
            param_labels: HashMap::new(),
            sampler: Some(sampler),
            history: Some(history),
        }
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

    /// Sets the trial state to Complete.
    pub(crate) fn set_complete(&mut self) {
        self.state = TrialState::Complete;
    }

    /// Sets the trial state to Failed.
    pub(crate) fn set_failed(&mut self) {
        self.state = TrialState::Failed;
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

        // Sample using the sampler
        let value = self.sample_value(&distribution);
        let result = param.cast_param_value(&value)?;

        // Store distribution, value, and label
        self.distributions.insert(param_id, distribution);
        self.params.insert(param_id, value);
        self.param_labels.insert(param_id, param.label());

        Ok(result)
    }
}
