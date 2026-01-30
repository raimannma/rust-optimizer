//! Sampler trait and implementations for parameter sampling.

pub mod grid;
pub mod random;
pub mod tpe;

use std::collections::HashMap;

use crate::distribution::Distribution;
use crate::param::ParamValue;

/// A completed trial with its parameters, distributions, and objective value.
///
/// This struct stores the results of a completed trial, including all sampled
/// parameter values, their distributions, and the objective value returned
/// by the objective function.
#[derive(Clone, Debug)]
pub struct CompletedTrial<V = f64> {
    /// The unique identifier for this trial.
    pub id: u64,
    /// The sampled parameter values, keyed by parameter name.
    pub params: HashMap<String, ParamValue>,
    /// The parameter distributions used, keyed by parameter name.
    pub distributions: HashMap<String, Distribution>,
    /// The objective value returned by the objective function.
    pub value: V,
}

impl<V> CompletedTrial<V> {
    /// Creates a new completed trial.
    pub fn new(
        id: u64,
        params: HashMap<String, ParamValue>,
        distributions: HashMap<String, Distribution>,
        value: V,
    ) -> Self {
        Self {
            id,
            params,
            distributions,
            value,
        }
    }
}

/// Trait for pluggable parameter sampling strategies.
///
/// Samplers are responsible for generating parameter values based on
/// the distribution and historical trial data. The trait requires
/// `Send + Sync` to support concurrent and async optimization.
///
/// # Examples
///
/// Implementing a custom sampler:
///
/// ```ignore
/// use optimizer::{Sampler, ParamValue, Distribution, CompletedTrial};
///
/// struct MySampler;
///
/// impl Sampler for MySampler {
///     fn sample(
///         &self,
///         distribution: &Distribution,
///         trial_id: u64,
///         history: &[CompletedTrial],
///     ) -> ParamValue {
///         // Custom sampling logic here
///         todo!()
///     }
/// }
/// ```
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
