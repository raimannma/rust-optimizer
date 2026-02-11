//! Sampler trait and implementations for parameter sampling.

pub mod bohb;
#[cfg(feature = "cma-es")]
pub mod cma_es;
pub mod grid;
pub mod motpe;
pub mod nsga2;
pub mod random;
#[cfg(feature = "sobol")]
pub mod sobol;
pub mod tpe;

use std::collections::HashMap;

use crate::distribution::Distribution;
use crate::param::ParamValue;
use crate::parameter::{ParamId, Parameter};
use crate::trial::AttrValue;
use crate::types::TrialState;

/// A completed trial with its parameters, distributions, and objective value.
///
/// This struct stores the results of a completed trial, including all sampled
/// parameter values, their distributions, and the objective value returned
/// by the objective function.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CompletedTrial<V = f64> {
    /// The unique identifier for this trial.
    pub id: u64,
    /// The sampled parameter values, keyed by parameter id.
    pub params: HashMap<ParamId, ParamValue>,
    /// The parameter distributions used, keyed by parameter id.
    pub distributions: HashMap<ParamId, Distribution>,
    /// Human-readable labels for parameters, keyed by parameter id.
    pub param_labels: HashMap<ParamId, String>,
    /// The objective value returned by the objective function.
    pub value: V,
    /// Intermediate objective values reported during the trial.
    pub intermediate_values: Vec<(u64, f64)>,
    /// The state of the trial (Complete, Pruned, or Failed).
    pub state: TrialState,
    /// User-defined attributes stored during the trial.
    pub user_attrs: HashMap<String, AttrValue>,
    /// Constraint values for this trial (<=0.0 means feasible).
    #[cfg_attr(feature = "serde", serde(default))]
    pub constraints: Vec<f64>,
}

impl<V> CompletedTrial<V> {
    /// Creates a new completed trial.
    pub fn new(
        id: u64,
        params: HashMap<ParamId, ParamValue>,
        distributions: HashMap<ParamId, Distribution>,
        param_labels: HashMap<ParamId, String>,
        value: V,
    ) -> Self {
        Self {
            id,
            params,
            distributions,
            param_labels,
            value,
            intermediate_values: Vec::new(),
            state: TrialState::Complete,
            user_attrs: HashMap::new(),
            constraints: Vec::new(),
        }
    }

    /// Creates a new completed trial with intermediate values and user attributes.
    pub fn with_intermediate_values(
        id: u64,
        params: HashMap<ParamId, ParamValue>,
        distributions: HashMap<ParamId, Distribution>,
        param_labels: HashMap<ParamId, String>,
        value: V,
        intermediate_values: Vec<(u64, f64)>,
        user_attrs: HashMap<String, AttrValue>,
    ) -> Self {
        Self {
            id,
            params,
            distributions,
            param_labels,
            value,
            intermediate_values,
            state: TrialState::Complete,
            user_attrs,
            constraints: Vec::new(),
        }
    }

    /// Returns the typed value for the given parameter.
    ///
    /// Looks up the parameter by its unique id and casts the stored
    /// [`ParamValue`] to the parameter's typed value.
    ///
    /// Returns `None` if the parameter was not used in this trial.
    ///
    /// # Panics
    ///
    /// Panics if the stored value is incompatible with the parameter type
    /// (e.g., a `Float` value stored for an `IntParam`). This indicates
    /// a bug in the program, not a runtime error.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let x = FloatParam::new(-10.0, 10.0);
    ///
    /// study
    ///     .optimize(5, |trial| {
    ///         let val = x.suggest(trial)?;
    ///         Ok::<_, optimizer::Error>(val * val)
    ///     })
    ///     .unwrap();
    ///
    /// let best = study.best_trial().unwrap();
    /// let x_val: f64 = best.get(&x).unwrap();
    /// assert!((-10.0..=10.0).contains(&x_val));
    /// ```
    pub fn get<P: Parameter>(&self, param: &P) -> Option<P::Value> {
        self.params.get(&param.id()).map(|v| {
            param
                .cast_param_value(v)
                .expect("parameter type mismatch: stored value incompatible with parameter")
        })
    }

    /// Returns `true` if all constraints are satisfied (values <= 0.0).
    ///
    /// A trial with no constraints is considered feasible.
    #[must_use]
    pub fn is_feasible(&self) -> bool {
        self.constraints.iter().all(|&c| c <= 0.0)
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
}

/// A pending (running) trial with its parameters and distributions, but no objective value yet.
///
/// This struct represents a trial that has been started and has sampled parameters,
/// but is still running and hasn't returned an objective value. It is used with the
/// constant liar strategy for parallel optimization.
#[derive(Clone, Debug)]
pub struct PendingTrial {
    /// The unique identifier for this trial.
    pub id: u64,
    /// The sampled parameter values, keyed by parameter id.
    pub params: HashMap<ParamId, ParamValue>,
    /// The parameter distributions used, keyed by parameter id.
    pub distributions: HashMap<ParamId, Distribution>,
    /// Human-readable labels for parameters, keyed by parameter id.
    pub param_labels: HashMap<ParamId, String>,
}

impl PendingTrial {
    /// Creates a new pending trial.
    #[must_use]
    pub fn new(
        id: u64,
        params: HashMap<ParamId, ParamValue>,
        distributions: HashMap<ParamId, Distribution>,
        param_labels: HashMap<ParamId, String>,
    ) -> Self {
        Self {
            id,
            params,
            distributions,
            param_labels,
        }
    }
}

/// Trait for pluggable parameter sampling strategies.
///
/// Samplers are responsible for generating parameter values based on
/// the distribution and historical trial data. The trait requires
/// `Send + Sync` to support concurrent and async optimization.
pub trait Sampler: Send + Sync {
    /// Samples a parameter value from the given distribution.
    ///
    /// # Arguments
    ///
    /// * `distribution` - The parameter distribution to sample from.
    /// * `trial_id` - The unique ID of the trial being sampled for.
    /// * `history` - Historical completed trials for informed sampling.
    ///
    /// # Returns
    ///
    /// A `ParamValue` sampled from the distribution.
    fn sample(
        &self,
        distribution: &Distribution,
        trial_id: u64,
        history: &[CompletedTrial],
    ) -> ParamValue;
}
