//! Trial implementation for tracking sampled parameters and trial state.

use core::ops::{Range, RangeInclusive};
use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::distribution::{
    CategoricalDistribution, Distribution, FloatDistribution, IntDistribution,
};
use crate::error::{Error, Result};
use crate::param::ParamValue;
use crate::sampler::{CompletedTrial, Sampler};
use crate::types::TrialState;

/// A trait for types that can be used with [`Trial::suggest_range`].
///
/// This trait is implemented for [`Range`] and [`RangeInclusive`] over `f64` and `i64`.
/// It allows using Rust's range syntax directly with the optimizer.
///
/// # Supported Range Types
///
/// | Range Type | Example | Description |
/// |------------|---------|-------------|
/// | `Range<f64>` | `0.0..1.0` | Float range (end-exclusive, treated as inclusive for continuous sampling) |
/// | `RangeInclusive<f64>` | `0.0..=1.0` | Float range (end-inclusive) |
/// | `Range<i64>` | `1..10` | Integer range from 1 to 9 (end-exclusive) |
/// | `RangeInclusive<i64>` | `1..=10` | Integer range from 1 to 10 (end-inclusive) |
pub trait SuggestableRange {
    /// The output type when suggesting from this range.
    type Output;

    /// Suggests a value from this range using the given trial.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is invalid (e.g., empty or low > high).
    fn suggest(self, trial: &mut Trial, name: String) -> Result<Self::Output>;
}

impl SuggestableRange for Range<f64> {
    type Output = f64;

    fn suggest(self, trial: &mut Trial, name: String) -> Result<f64> {
        trial.suggest_float(name, self.start, self.end)
    }
}

impl SuggestableRange for RangeInclusive<f64> {
    type Output = f64;

    fn suggest(self, trial: &mut Trial, name: String) -> Result<f64> {
        trial.suggest_float(name, *self.start(), *self.end())
    }
}

impl SuggestableRange for Range<i64> {
    type Output = i64;

    fn suggest(self, trial: &mut Trial, name: String) -> Result<i64> {
        // Range is exclusive on the end, so subtract 1
        trial.suggest_int(name, self.start, self.end.saturating_sub(1))
    }
}

impl SuggestableRange for RangeInclusive<i64> {
    type Output = i64;

    fn suggest(self, trial: &mut Trial, name: String) -> Result<i64> {
        trial.suggest_int(name, *self.start(), *self.end())
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
    /// Sampled parameter values, keyed by parameter name.
    params: HashMap<String, ParamValue>,
    /// Parameter distributions, keyed by parameter name.
    distributions: HashMap<String, Distribution>,
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
    /// local random sampling for suggest_* methods.
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
    pub fn params(&self) -> &HashMap<String, ParamValue> {
        &self.params
    }

    /// Returns a reference to the parameter distributions.
    #[must_use]
    pub fn distributions(&self) -> &HashMap<String, Distribution> {
        &self.distributions
    }

    /// Sets the trial state to Complete.
    pub(crate) fn set_complete(&mut self) {
        self.state = TrialState::Complete;
    }

    /// Sets the trial state to Failed.
    pub(crate) fn set_failed(&mut self) {
        self.state = TrialState::Failed;
    }

    /// Suggests a float parameter with the given bounds.
    ///
    /// If the parameter has already been sampled with the same bounds, the cached value is returned.
    /// If the parameter was sampled with different bounds, a `ParameterConflict` error is returned.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the parameter.
    /// * `low` - The lower bound (inclusive).
    /// * `high` - The upper bound (inclusive).
    ///
    /// # Errors
    ///
    /// Returns `InvalidBounds` if `low > high`.
    /// Returns `ParameterConflict` if the parameter was previously sampled with different bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::Trial;
    ///
    /// let mut trial = Trial::new(0);
    /// let x = trial.suggest_float("x", 0.0, 1.0).unwrap();
    /// assert!(x >= 0.0 && x <= 1.0);
    ///
    /// // Calling again with same bounds returns cached value
    /// let x2 = trial.suggest_float("x", 0.0, 1.0).unwrap();
    /// assert_eq!(x, x2);
    /// ```
    pub fn suggest_float(&mut self, name: impl Into<String>, low: f64, high: f64) -> Result<f64> {
        if low > high {
            return Err(Error::InvalidBounds { low, high });
        }

        let name = name.into();
        let distribution = FloatDistribution {
            low,
            high,
            log_scale: false,
            step: None,
        };

        // Check if parameter already exists
        if let Some(existing_dist) = self.distributions.get(&name) {
            // Verify the distribution matches
            if let Distribution::Float(existing) = existing_dist
                && (existing.low - low).abs() < f64::EPSILON
                && (existing.high - high).abs() < f64::EPSILON
                && !existing.log_scale
                && existing.step.is_none()
            {
                // Same distribution, return cached value
                if let Some(ParamValue::Float(value)) = self.params.get(&name) {
                    return Ok(*value);
                }
            }
            // Distribution exists but doesn't match
            return Err(Error::ParameterConflict {
                name,
                reason: "parameter was previously sampled with different bounds or type"
                    .to_string(),
            });
        }

        // Sample using the sampler
        let dist = Distribution::Float(distribution);
        let ParamValue::Float(value) = self.sample_value(&dist) else {
            return Err(Error::Internal(
                "Float distribution should return Float value",
            ));
        };

        // Store distribution and value
        self.distributions.insert(name.clone(), dist);
        self.params.insert(name, ParamValue::Float(value));

        Ok(value)
    }

    /// Suggests a float parameter sampled on a logarithmic scale.
    ///
    /// The value is sampled uniformly in log space, which is useful for parameters
    /// that span multiple orders of magnitude (e.g., learning rates).
    ///
    /// If the parameter has already been sampled with the same bounds and `log_scale=true`,
    /// the cached value is returned. If the parameter was sampled with different configuration,
    /// a `ParameterConflict` error is returned.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the parameter.
    /// * `low` - The lower bound (inclusive, must be positive).
    /// * `high` - The upper bound (inclusive).
    ///
    /// # Errors
    ///
    /// Returns `InvalidLogBounds` if `low <= 0`.
    /// Returns `InvalidBounds` if `low > high`.
    /// Returns `ParameterConflict` if the parameter was previously sampled with different configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::Trial;
    ///
    /// let mut trial = Trial::new(0);
    /// let lr = trial
    ///     .suggest_float_log("learning_rate", 1e-5, 1e-1)
    ///     .unwrap();
    /// assert!(lr >= 1e-5 && lr <= 1e-1);
    ///
    /// // Calling again with same bounds returns cached value
    /// let lr2 = trial
    ///     .suggest_float_log("learning_rate", 1e-5, 1e-1)
    ///     .unwrap();
    /// assert_eq!(lr, lr2);
    /// ```
    pub fn suggest_float_log(
        &mut self,
        name: impl Into<String>,
        low: f64,
        high: f64,
    ) -> Result<f64> {
        if low <= 0.0 {
            return Err(Error::InvalidLogBounds);
        }

        if low > high {
            return Err(Error::InvalidBounds { low, high });
        }

        let name = name.into();
        let distribution = FloatDistribution {
            low,
            high,
            log_scale: true,
            step: None,
        };

        // Check if parameter already exists
        if let Some(existing_dist) = self.distributions.get(&name) {
            // Verify the distribution matches
            if let Distribution::Float(existing) = existing_dist
                && (existing.low - low).abs() < f64::EPSILON
                && (existing.high - high).abs() < f64::EPSILON
                && existing.log_scale
                && existing.step.is_none()
            {
                // Same distribution, return cached value
                if let Some(ParamValue::Float(value)) = self.params.get(&name) {
                    return Ok(*value);
                }
            }
            // Distribution exists but doesn't match
            return Err(Error::ParameterConflict {
                name,
                reason: "parameter was previously sampled with different bounds or type"
                    .to_string(),
            });
        }

        // Sample using the sampler (sampler handles log-scale transformation)
        let dist = Distribution::Float(distribution);
        let ParamValue::Float(value) = self.sample_value(&dist) else {
            return Err(Error::Internal(
                "Float distribution should return Float value",
            ));
        };

        // Store distribution and value
        self.distributions.insert(name.clone(), dist);
        self.params.insert(name, ParamValue::Float(value));

        Ok(value)
    }

    /// Suggests a float parameter that snaps to a step grid.
    ///
    /// The value is sampled from the discrete set {low, low + step, low + 2*step, ...}
    /// where each value is <= high.
    ///
    /// If the parameter has already been sampled with the same configuration,
    /// the cached value is returned. If the parameter was sampled with different configuration,
    /// a `ParameterConflict` error is returned.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the parameter.
    /// * `low` - The lower bound (inclusive).
    /// * `high` - The upper bound (inclusive).
    /// * `step` - The step size (must be positive).
    ///
    /// # Errors
    ///
    /// Returns `InvalidStep` if `step <= 0`.
    /// Returns `InvalidBounds` if `low > high`.
    /// Returns `ParameterConflict` if the parameter was previously sampled with different configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::Trial;
    ///
    /// let mut trial = Trial::new(0);
    /// let x = trial.suggest_float_step("x", 0.0, 1.0, 0.25).unwrap();
    /// // x will be one of: 0.0, 0.25, 0.5, 0.75, 1.0
    /// assert!(x >= 0.0 && x <= 1.0);
    /// assert!((x / 0.25).fract().abs() < 1e-10 || (x / 0.25).fract().abs() > 1.0 - 1e-10);
    ///
    /// // Calling again with same bounds returns cached value
    /// let x2 = trial.suggest_float_step("x", 0.0, 1.0, 0.25).unwrap();
    /// assert_eq!(x, x2);
    /// ```
    pub fn suggest_float_step(
        &mut self,
        name: impl Into<String>,
        low: f64,
        high: f64,
        step: f64,
    ) -> Result<f64> {
        if step <= 0.0 {
            return Err(Error::InvalidStep);
        }

        if low > high {
            return Err(Error::InvalidBounds { low, high });
        }

        let name = name.into();
        let distribution = FloatDistribution {
            low,
            high,
            log_scale: false,
            step: Some(step),
        };

        // Check if parameter already exists
        if let Some(existing_dist) = self.distributions.get(&name) {
            // Verify the distribution matches
            if let Distribution::Float(existing) = existing_dist
                && (existing.low - low).abs() < f64::EPSILON
                && (existing.high - high).abs() < f64::EPSILON
                && !existing.log_scale
                && existing.step == Some(step)
            {
                // Same distribution, return cached value
                if let Some(ParamValue::Float(value)) = self.params.get(&name) {
                    return Ok(*value);
                }
            }
            // Distribution exists but doesn't match
            return Err(Error::ParameterConflict {
                name,
                reason: "parameter was previously sampled with different bounds or type"
                    .to_string(),
            });
        }

        // Sample using the sampler (sampler handles step-grid)
        let dist = Distribution::Float(distribution);
        let ParamValue::Float(value) = self.sample_value(&dist) else {
            return Err(Error::Internal(
                "Float distribution should return Float value",
            ));
        };

        // Store distribution and value
        self.distributions.insert(name.clone(), dist);
        self.params.insert(name, ParamValue::Float(value));

        Ok(value)
    }

    /// Suggests an integer parameter with the given bounds.
    ///
    /// The value is sampled uniformly from the range [low, high] inclusive.
    ///
    /// If the parameter has already been sampled with the same bounds, the cached value is returned.
    /// If the parameter was sampled with different bounds, a `ParameterConflict` error is returned.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the parameter.
    /// * `low` - The lower bound (inclusive).
    /// * `high` - The upper bound (inclusive).
    ///
    /// # Errors
    ///
    /// Returns `InvalidBounds` if `low > high`.
    /// Returns `ParameterConflict` if the parameter was previously sampled with different bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::Trial;
    ///
    /// let mut trial = Trial::new(0);
    /// let n = trial.suggest_int("n_layers", 1, 10).unwrap();
    /// assert!(n >= 1 && n <= 10);
    ///
    /// // Calling again with same bounds returns cached value
    /// let n2 = trial.suggest_int("n_layers", 1, 10).unwrap();
    /// assert_eq!(n, n2);
    /// ```
    #[allow(clippy::cast_precision_loss)]
    pub fn suggest_int(&mut self, name: impl Into<String>, low: i64, high: i64) -> Result<i64> {
        if low > high {
            return Err(Error::InvalidBounds {
                low: low as f64,
                high: high as f64,
            });
        }

        let name = name.into();
        let distribution = IntDistribution {
            low,
            high,
            log_scale: false,
            step: None,
        };

        // Check if parameter already exists
        if let Some(existing_dist) = self.distributions.get(&name) {
            // Verify the distribution matches
            if let Distribution::Int(existing) = existing_dist
                && existing.low == low
                && existing.high == high
                && !existing.log_scale
                && existing.step.is_none()
            {
                // Same distribution, return cached value
                if let Some(ParamValue::Int(value)) = self.params.get(&name) {
                    return Ok(*value);
                }
            }
            // Distribution exists but doesn't match
            return Err(Error::ParameterConflict {
                name,
                reason: "parameter was previously sampled with different bounds or type"
                    .to_string(),
            });
        }

        // Sample using the sampler
        let dist = Distribution::Int(distribution);
        let ParamValue::Int(value) = self.sample_value(&dist) else {
            return Err(Error::Internal("Int distribution should return Int value"));
        };

        // Store distribution and value
        self.distributions.insert(name.clone(), dist);
        self.params.insert(name, ParamValue::Int(value));

        Ok(value)
    }

    /// Suggests an integer parameter sampled on a logarithmic scale.
    ///
    /// The value is sampled uniformly in log space, which is useful for parameters
    /// that span multiple orders of magnitude (e.g., batch sizes).
    ///
    /// If the parameter has already been sampled with the same bounds and `log_scale=true`,
    /// the cached value is returned. If the parameter was sampled with different configuration,
    /// a `ParameterConflict` error is returned.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the parameter.
    /// * `low` - The lower bound (inclusive, must be >= 1).
    /// * `high` - The upper bound (inclusive).
    ///
    /// # Errors
    ///
    /// Returns `InvalidLogBounds` if `low < 1`.
    /// Returns `InvalidBounds` if `low > high`.
    /// Returns `ParameterConflict` if the parameter was previously sampled with different configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::Trial;
    ///
    /// let mut trial = Trial::new(0);
    /// let batch_size = trial.suggest_int_log("batch_size", 1, 1024).unwrap();
    /// assert!(batch_size >= 1 && batch_size <= 1024);
    ///
    /// // Calling again with same bounds returns cached value
    /// let batch_size2 = trial.suggest_int_log("batch_size", 1, 1024).unwrap();
    /// assert_eq!(batch_size, batch_size2);
    /// ```
    #[allow(clippy::cast_precision_loss)]
    pub fn suggest_int_log(&mut self, name: impl Into<String>, low: i64, high: i64) -> Result<i64> {
        if low < 1 {
            return Err(Error::InvalidLogBounds);
        }

        if low > high {
            return Err(Error::InvalidBounds {
                low: low as f64,
                high: high as f64,
            });
        }

        let name = name.into();
        let distribution = IntDistribution {
            low,
            high,
            log_scale: true,
            step: None,
        };

        // Check if parameter already exists
        if let Some(existing_dist) = self.distributions.get(&name) {
            // Verify the distribution matches
            if let Distribution::Int(existing) = existing_dist
                && existing.low == low
                && existing.high == high
                && existing.log_scale
                && existing.step.is_none()
            {
                // Same distribution, return cached value
                if let Some(ParamValue::Int(value)) = self.params.get(&name) {
                    return Ok(*value);
                }
            }
            // Distribution exists but doesn't match
            return Err(Error::ParameterConflict {
                name,
                reason: "parameter was previously sampled with different bounds or type"
                    .to_string(),
            });
        }

        // Sample using the sampler (sampler handles log-scale transformation)
        let dist = Distribution::Int(distribution);
        let ParamValue::Int(value) = self.sample_value(&dist) else {
            return Err(Error::Internal("Int distribution should return Int value"));
        };

        // Store distribution and value
        self.distributions.insert(name.clone(), dist);
        self.params.insert(name, ParamValue::Int(value));

        Ok(value)
    }

    /// Suggests an integer parameter that snaps to a step grid.
    ///
    /// The value is sampled from the discrete set {low, low + step, low + 2*step, ...}
    /// where each value is <= high.
    ///
    /// If the parameter has already been sampled with the same configuration,
    /// the cached value is returned. If the parameter was sampled with different configuration,
    /// a `ParameterConflict` error is returned.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the parameter.
    /// * `low` - The lower bound (inclusive).
    /// * `high` - The upper bound (inclusive).
    /// * `step` - The step size (must be positive).
    ///
    /// # Errors
    ///
    /// Returns `InvalidStep` if `step <= 0`.
    /// Returns `InvalidBounds` if `low > high`.
    /// Returns `ParameterConflict` if the parameter was previously sampled with different configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::Trial;
    ///
    /// let mut trial = Trial::new(0);
    /// let n = trial
    ///     .suggest_int_step("n_estimators", 100, 500, 50)
    ///     .unwrap();
    /// // n will be one of: 100, 150, 200, 250, 300, 350, 400, 450, 500
    /// assert!(n >= 100 && n <= 500);
    /// assert!((n - 100) % 50 == 0);
    ///
    /// // Calling again with same bounds returns cached value
    /// let n2 = trial
    ///     .suggest_int_step("n_estimators", 100, 500, 50)
    ///     .unwrap();
    /// assert_eq!(n, n2);
    /// ```
    #[allow(clippy::cast_precision_loss)]
    pub fn suggest_int_step(
        &mut self,
        name: impl Into<String>,
        low: i64,
        high: i64,
        step: i64,
    ) -> Result<i64> {
        if step <= 0 {
            return Err(Error::InvalidStep);
        }

        if low > high {
            return Err(Error::InvalidBounds {
                low: low as f64,
                high: high as f64,
            });
        }

        let name = name.into();
        let distribution = IntDistribution {
            low,
            high,
            log_scale: false,
            step: Some(step),
        };

        // Check if parameter already exists
        if let Some(existing_dist) = self.distributions.get(&name) {
            // Verify the distribution matches
            if let Distribution::Int(existing) = existing_dist
                && existing.low == low
                && existing.high == high
                && !existing.log_scale
                && existing.step == Some(step)
            {
                // Same distribution, return cached value
                if let Some(ParamValue::Int(value)) = self.params.get(&name) {
                    return Ok(*value);
                }
            }
            // Distribution exists but doesn't match
            return Err(Error::ParameterConflict {
                name,
                reason: "parameter was previously sampled with different bounds or type"
                    .to_string(),
            });
        }

        // Sample using the sampler (sampler handles step-grid)
        let dist = Distribution::Int(distribution);
        let ParamValue::Int(value) = self.sample_value(&dist) else {
            return Err(Error::Internal("Int distribution should return Int value"));
        };

        // Store distribution and value
        self.distributions.insert(name.clone(), dist);
        self.params.insert(name, ParamValue::Int(value));

        Ok(value)
    }

    /// Suggests a categorical parameter from the given choices.
    ///
    /// The value is selected uniformly at random from the provided choices.
    /// Internally, the index of the selected choice is stored.
    ///
    /// If the parameter has already been sampled with the same number of choices,
    /// the cached value (same index) is returned. If the parameter was sampled with
    /// a different number of choices, a `ParameterConflict` error is returned.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the parameter.
    /// * `choices` - A slice of choices to select from.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The type of the choices. Only requires `Clone`.
    ///
    /// # Errors
    ///
    /// Returns `EmptyChoices` if `choices` is empty.
    /// Returns `ParameterConflict` if the parameter was previously sampled with a different number of choices.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::Trial;
    ///
    /// let mut trial = Trial::new(0);
    /// let optimizer = trial
    ///     .suggest_categorical("optimizer", &["sgd", "adam", "rmsprop"])
    ///     .unwrap();
    /// assert!(["sgd", "adam", "rmsprop"].contains(&optimizer));
    ///
    /// // Calling again with same choices returns cached value
    /// let optimizer2 = trial
    ///     .suggest_categorical("optimizer", &["sgd", "adam", "rmsprop"])
    ///     .unwrap();
    /// assert_eq!(optimizer, optimizer2);
    /// ```
    pub fn suggest_categorical<T: Clone>(
        &mut self,
        name: impl Into<String>,
        choices: &[T],
    ) -> Result<T> {
        if choices.is_empty() {
            return Err(Error::EmptyChoices);
        }

        let name = name.into();
        let n_choices = choices.len();
        let distribution = CategoricalDistribution { n_choices };

        // Check if parameter already exists
        if let Some(existing_dist) = self.distributions.get(&name) {
            // Verify the distribution matches
            if let Distribution::Categorical(existing) = existing_dist
                && existing.n_choices == n_choices
            {
                // Same distribution, return cached value
                if let Some(ParamValue::Categorical(index)) = self.params.get(&name) {
                    return Ok(choices[*index].clone());
                }
            }
            // Distribution exists but doesn't match
            return Err(Error::ParameterConflict {
                name,
                reason: "parameter was previously sampled with different number of choices or type"
                    .to_string(),
            });
        }

        // Sample using the sampler
        let dist = Distribution::Categorical(distribution);
        let ParamValue::Categorical(index) = self.sample_value(&dist) else {
            return Err(Error::Internal(
                "Categorical distribution should return Categorical value",
            ));
        };

        // Store distribution and value (store the index)
        self.distributions.insert(name.clone(), dist);
        self.params.insert(name, ParamValue::Categorical(index));

        Ok(choices[index].clone())
    }

    /// Suggests a boolean parameter.
    ///
    /// The value is selected uniformly at random from `{false, true}`.
    /// This is equivalent to calling `suggest_categorical(name, &[false, true])`.
    ///
    /// If the parameter has already been sampled, the cached value is returned.
    /// If the parameter was sampled with a different type, a `ParameterConflict` error is returned.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the parameter.
    ///
    /// # Errors
    ///
    /// Returns `ParameterConflict` if the parameter was previously sampled with a different type.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::Trial;
    ///
    /// let mut trial = Trial::new(0);
    /// let use_dropout = trial.suggest_bool("use_dropout").unwrap();
    /// assert!(use_dropout == true || use_dropout == false);
    ///
    /// // Calling again returns cached value
    /// let use_dropout2 = trial.suggest_bool("use_dropout").unwrap();
    /// assert_eq!(use_dropout, use_dropout2);
    /// ```
    pub fn suggest_bool(&mut self, name: impl Into<String>) -> Result<bool> {
        self.suggest_categorical(name, &[false, true])
    }

    /// Suggests a parameter value from a range.
    ///
    /// This method accepts both [`Range`] (`..`) and [`RangeInclusive`] (`..=`)
    /// for both `f64` and `i64` types, allowing natural Rust range syntax.
    ///
    /// For integer ranges, note that `Range` (`..`) is end-exclusive while
    /// `RangeInclusive` (`..=`) is end-inclusive, matching Rust's semantics.
    ///
    /// If the parameter has already been sampled with the same bounds, the cached value is returned.
    /// If the parameter was sampled with different bounds, a `ParameterConflict` error is returned.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the parameter.
    /// * `range` - The range to sample from.
    ///
    /// # Type Parameters
    ///
    /// * `R` - A range type implementing [`SuggestableRange`].
    ///
    /// # Errors
    ///
    /// Returns `InvalidBounds` if the range is invalid (e.g., low > high or empty integer range).
    /// Returns `ParameterConflict` if the parameter was previously sampled with different bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::Trial;
    ///
    /// let mut trial = Trial::new(0);
    ///
    /// // Float ranges
    /// let x = trial.suggest_range("x", 0.0..1.0).unwrap();
    /// assert!(x >= 0.0 && x <= 1.0);
    ///
    /// let y = trial.suggest_range("y", 0.0..=1.0).unwrap();
    /// assert!(y >= 0.0 && y <= 1.0);
    ///
    /// // Integer ranges
    /// let n = trial.suggest_range("n", 1_i64..10).unwrap(); // 1 to 9 inclusive
    /// assert!(n >= 1 && n <= 9);
    ///
    /// let m = trial.suggest_range("m", 1_i64..=10).unwrap(); // 1 to 10 inclusive
    /// assert!(m >= 1 && m <= 10);
    ///
    /// // Calling again with same range returns cached value
    /// let x2 = trial.suggest_range("x", 0.0..1.0).unwrap();
    /// assert_eq!(x, x2);
    /// ```
    pub fn suggest_range<R: SuggestableRange>(
        &mut self,
        name: impl Into<String>,
        range: R,
    ) -> Result<R::Output> {
        range.suggest(self, name.into())
    }
}
