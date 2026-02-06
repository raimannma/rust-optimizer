//! Sampler trait and implementations for parameter sampling.

pub mod grid;
pub mod random;
pub mod tpe;

use std::collections::HashMap;

use crate::distribution::Distribution;
use crate::param::ParamValue;
use crate::parameter::ParamId;

/// A completed trial with its parameters, distributions, and objective value.
///
/// This struct stores the results of a completed trial, including all sampled
/// parameter values, their distributions, and the objective value returned
/// by the objective function.
#[derive(Clone, Debug)]
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
        }
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
