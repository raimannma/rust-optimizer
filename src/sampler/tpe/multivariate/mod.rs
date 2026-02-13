//! Multivariate Tree-Parzen Estimator (TPE) sampler implementation.
//!
//! Unlike the standard [`TpeSampler`] which samples parameters
//! independently, the `MultivariateTpeSampler` models joint distributions over multiple
//! parameters. This allows it to capture correlations between parameters, which can
//! significantly improve optimization performance on problems where parameters interact.
//!
//! # When to Use Multivariate TPE
//!
//! Use `MultivariateTpeSampler` when:
//! - Parameters have strong correlations (e.g., Rosenbrock function)
//! - The optimal value of one parameter depends on another
//! - You have a fixed search space across all trials
//!
//! Use the standard [`TpeSampler`] when:
//! - Parameters are independent
//! - The search space varies dynamically between trials
//! - You want simpler, faster optimization
//!
//! # Configuration
//!
//! - `gamma_strategy`: Controls what fraction of trials are considered "good"
//! - `n_startup_trials`: Number of random trials before multivariate TPE kicks in
//! - `n_ei_candidates`: Number of candidates to evaluate when selecting the next point
//! - `group`: When true, decomposes search space into independent groups
//! - `constant_liar`: Strategy for imputing values to pending trials in parallel optimization
//!
//! # Fallback Behavior for Dynamic Search Spaces
//!
//! When the search space varies between trials (e.g., conditional parameters), the sampler
//! gracefully falls back to independent sampling. The fallback hierarchy is:
//!
//! 1. **Multivariate TPE** for parameters that appear in ALL completed trials (the intersection
//!    search space). These parameters are modeled jointly using multivariate KDE.
//!
//! 2. **Independent TPE** for parameters that appear in some trials but not the intersection.
//!    Each such parameter is modeled with its own univariate KDE.
//!
//! 3. **Uniform random sampling** for new parameters that have never been seen, or during
//!    the startup phase before enough trials are collected.
//!
//! # Group Decomposition
//!
//! When `group` is enabled, the sampler analyzes parameter co-occurrence across trials and
//! decomposes the search space into independent groups. Parameters that always appear
//! together form a group, and each group is sampled using multivariate TPE independently.
//!
//! This can improve efficiency when:
//! - Some parameters are truly independent of others
//! - The search space has conditional structure where certain parameters only appear together
//!
//! # Parallel Optimization with Constant Liar
//!
//! For parallel optimization where multiple trials run simultaneously, the constant liar
//! strategy helps avoid redundant exploration. When new samples are requested while trials
//! are pending, the sampler can impute "lie" values for pending trials, allowing them to
//! influence the model.
//!
//! Available strategies via [`ConstantLiarStrategy`]:
//! - `None`: Ignore pending trials (default, suitable for sequential optimization)
//! - `Mean`: Impute the mean of completed trial values
//! - `Best`: Impute the best (minimum) completed value - pessimistic, encourages exploration
//! - `Worst`: Impute the worst (maximum) completed value - optimistic, encourages exploitation
//! - `Custom(f64)`: Impute a specific user-defined value
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use optimizer::sampler::tpe::MultivariateTpeSampler;
//!
//! let sampler = MultivariateTpeSampler::builder()
//!     .gamma(0.15)
//!     .n_startup_trials(20)
//!     .n_ei_candidates(32)
//!     .seed(42)
//!     .build()
//!     .unwrap();
//! ```
//!
//! ## With Custom Gamma Strategy and Group Decomposition
//!
//! ```
//! use optimizer::sampler::tpe::{MultivariateTpeSampler, SqrtGamma};
//!
//! let sampler = MultivariateTpeSampler::builder()
//!     .gamma_strategy(SqrtGamma::default())
//!     .group(true)
//!     .build()
//!     .unwrap();
//! ```
//!
//! ## Parallel Optimization with Constant Liar
//!
//! ```
//! use optimizer::sampler::tpe::{ConstantLiarStrategy, MultivariateTpeSampler};
//!
//! // Use mean imputation for parallel workers
//! let sampler = MultivariateTpeSampler::builder()
//!     .constant_liar(ConstantLiarStrategy::Mean)
//!     .n_startup_trials(10)
//!     .build()
//!     .unwrap();
//! ```
//!
//! ## Suppressing Fallback Warnings
//!
//! ```
//! use optimizer::sampler::tpe::MultivariateTpeSampler;
//!
//! let sampler = MultivariateTpeSampler::builder().build().unwrap();
//! ```

mod engine;
mod trials;

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use super::{FixedGamma, GammaStrategy};
use crate::distribution::Distribution;
use crate::error::Result;
use crate::param::ParamValue;
use crate::parameter::ParamId;
use crate::sampler::{CompletedTrial, Sampler};

/// Strategy for imputing objective values for pending/running trials during parallel optimization.
///
/// In parallel optimization, multiple trials may be running simultaneously. The constant liar
/// strategy assigns "lie" values to pending trials so they can be included in the model fitting,
/// which helps avoid redundant exploration of the same region.
///
/// # Variants
///
/// - `None`: No imputation; pending trials are ignored (default)
/// - `Mean`: Impute with the mean of completed trial values
/// - `Best`: Impute with the best (minimum for minimization) completed value
/// - `Worst`: Impute with the worst (maximum for minimization) completed value
/// - `Custom(f64)`: Impute with a specific user-provided value
///
/// # Examples
///
/// ```
/// use optimizer::sampler::tpe::ConstantLiarStrategy;
///
/// // Use mean imputation for parallel optimization
/// let strategy = ConstantLiarStrategy::Mean;
///
/// // Use a custom lie value
/// let strategy = ConstantLiarStrategy::Custom(0.5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ConstantLiarStrategy {
    /// No imputation; pending trials are ignored.
    #[default]
    None,
    /// Impute with the mean of completed trial values.
    Mean,
    /// Impute with the best (minimum for minimization) completed value.
    Best,
    /// Impute with the worst (maximum for minimization) completed value.
    Worst,
    /// Impute with a specific user-provided value.
    Custom(f64),
}

/// A Multivariate Tree-Parzen Estimator (TPE) sampler for Bayesian optimization.
///
/// Unlike the standard [`super::TpeSampler`], which samples each parameter
/// independently, this sampler models joint distributions over all parameters
/// using multivariate KDE. This captures correlations between parameters and
/// can significantly improve optimization on problems where parameters interact
/// (e.g., Rosenbrock, coupled hyperparameters).
///
/// When the search space varies between trials (conditional parameters), the
/// sampler automatically falls back to independent TPE or uniform sampling for
/// parameters outside the intersection search space.
///
/// # When to use
///
/// - Parameters are correlated or interact with each other.
/// - The search space is mostly fixed across trials.
/// - You need parallel optimization (enable [`ConstantLiarStrategy`]).
///
/// Prefer [`super::TpeSampler`] when parameters are independent or the search space changes
/// dynamically.
///
/// # Examples
///
/// ```
/// use optimizer::parameter::{FloatParam, Parameter};
/// use optimizer::sampler::tpe::MultivariateTpeSampler;
/// use optimizer::{Direction, Study};
///
/// let sampler = MultivariateTpeSampler::builder()
///     .gamma(0.15)
///     .n_startup_trials(20)
///     .seed(42)
///     .build()
///     .unwrap();
///
/// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
/// let x = FloatParam::new(-5.0, 5.0);
/// let y = FloatParam::new(-5.0, 5.0);
///
/// study
///     .optimize(50, |trial: &mut optimizer::Trial| {
///         let xv = x.suggest(trial)?;
///         let yv = y.suggest(trial)?;
///         Ok::<_, optimizer::Error>(xv * xv + yv * yv)
///     })
///     .unwrap();
///
/// assert!(study.best_value().unwrap() < 1.0);
/// ```
/// Cached joint sample for a specific trial.
struct JointSampleCache {
    trial_id: u64,
    search_space: HashMap<ParamId, Distribution>,
    sample: HashMap<ParamId, ParamValue>,
}

pub struct MultivariateTpeSampler {
    /// Strategy for computing the gamma quantile.
    gamma_strategy: Arc<dyn GammaStrategy>,
    /// Number of trials before multivariate TPE kicks in (uses random sampling before this).
    n_startup_trials: usize,
    /// Number of candidate samples to evaluate when selecting the next point.
    n_ei_candidates: usize,
    /// Whether to decompose search space into independent groups based on parameter co-occurrence.
    group: bool,
    /// Strategy for imputing objective values for pending trials in parallel optimization.
    constant_liar: ConstantLiarStrategy,
    /// Thread-safe RNG for sampling.
    rng: Mutex<fastrand::Rng>,
    /// Cache for joint samples to maintain consistency across parameters within the same trial.
    joint_sample_cache: Mutex<Option<JointSampleCache>>,
}

impl MultivariateTpeSampler {
    /// Creates a new Multivariate TPE sampler with default settings.
    ///
    /// Default settings:
    /// - gamma strategy: [`FixedGamma`] with gamma = 0.25
    /// - `n_startup_trials`: 10 (random sampling for first 10 trials)
    /// - `n_ei_candidates`: 24 (evaluate 24 candidates per sample)
    /// - `group`: false (no group decomposition)
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::MultivariateTpeSampler;
    ///
    /// let sampler = MultivariateTpeSampler::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            gamma_strategy: Arc::new(FixedGamma::default()),
            n_startup_trials: 10,
            n_ei_candidates: 24,
            group: false,
            constant_liar: ConstantLiarStrategy::None,
            rng: Mutex::new(fastrand::Rng::new()),
            joint_sample_cache: Mutex::new(None),
        }
    }

    /// Creates a builder for configuring a Multivariate TPE sampler.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::MultivariateTpeSampler;
    ///
    /// let sampler = MultivariateTpeSampler::builder()
    ///     .gamma(0.15)
    ///     .n_startup_trials(20)
    ///     .n_ei_candidates(32)
    ///     .seed(42)
    ///     .build()
    ///     .unwrap();
    /// ```
    #[must_use]
    pub fn builder() -> MultivariateTpeSamplerBuilder {
        MultivariateTpeSamplerBuilder::new()
    }

    /// Returns the gamma strategy used by this sampler.
    #[must_use]
    pub fn gamma_strategy(&self) -> &dyn GammaStrategy {
        self.gamma_strategy.as_ref()
    }

    /// Returns the number of startup trials.
    #[must_use]
    pub fn n_startup_trials(&self) -> usize {
        self.n_startup_trials
    }

    /// Returns the number of EI candidates.
    #[must_use]
    pub fn n_ei_candidates(&self) -> usize {
        self.n_ei_candidates
    }

    /// Returns whether group decomposition is enabled.
    #[must_use]
    pub fn group(&self) -> bool {
        self.group
    }

    /// Returns the constant liar strategy for parallel optimization.
    #[must_use]
    pub fn constant_liar(&self) -> &ConstantLiarStrategy {
        &self.constant_liar
    }

    /// Samples all parameters uniformly at random.
    ///
    /// This is a fallback method used when multivariate TPE cannot be applied.
    #[allow(clippy::unused_self)]
    fn sample_all_uniform(
        &self,
        search_space: &HashMap<ParamId, Distribution>,
        rng: &mut fastrand::Rng,
    ) -> HashMap<ParamId, ParamValue> {
        search_space
            .iter()
            .map(|(id, dist)| (*id, crate::sampler::common::sample_random(rng, dist)))
            .collect()
    }

    /// Samples parameters jointly using multivariate TPE.
    ///
    /// This method samples all parameters in the search space jointly, capturing
    /// correlations between them. During the startup phase (before `n_startup_trials`
    /// have completed), it falls back to uniform random sampling.
    ///
    /// # Arguments
    ///
    /// * `search_space` - A map of parameter names to their distributions.
    /// * `history` - Historical completed trials for informed sampling.
    ///
    /// # Returns
    ///
    /// A `HashMap<ParamId, ParamValue>` containing sampled values for all parameters
    /// in the search space.
    ///
    /// # Algorithm
    ///
    /// 1. During startup phase: sample each parameter uniformly at random
    /// 2. After startup:
    ///    - Compute intersection search space (parameters common to all trials)
    ///    - Filter trials to those containing all intersection parameters
    ///    - Split into good/bad trials based on gamma quantile
    ///    - Fit multivariate KDEs on continuous parameters (Float and Int)
    ///    - Select best candidate using l(x)/g(x) acquisition function
    ///    - Sample categorical parameters independently (for now)
    ///
    /// # Notes
    ///
    /// - Categorical parameters are currently sampled independently, not jointly.
    /// - If there are not enough continuous parameters for multivariate modeling,
    ///   falls back to independent sampling.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use std::collections::HashMap;
    /// use optimizer::sampler::tpe::MultivariateTpeSampler;
    /// use optimizer::parameter::ParamId;
    /// use optimizer::distribution::{Distribution, FloatDistribution};
    ///
    /// let sampler = MultivariateTpeSampler::builder()
    ///     .n_startup_trials(10)
    ///     .seed(42)
    ///     .build()
    ///     .unwrap();
    ///
    /// let x_id = ParamId::new();
    /// let y_id = ParamId::new();
    /// let mut search_space = HashMap::new();
    /// search_space.insert(x_id, Distribution::Float(FloatDistribution {
    ///     low: 0.0, high: 1.0, log_scale: false, step: None,
    /// }));
    /// search_space.insert(y_id, Distribution::Float(FloatDistribution {
    ///     low: 0.0, high: 1.0, log_scale: false, step: None,
    /// }));
    ///
    /// let history = vec![]; // No history yet
    /// let params = sampler.sample_joint(&search_space, &history);
    /// ```
    #[must_use]
    pub fn sample_joint(
        &self,
        search_space: &HashMap<ParamId, Distribution>,
        history: &[CompletedTrial],
    ) -> HashMap<ParamId, ParamValue> {
        let mut rng = self.rng.lock();

        // Early returns for cases requiring random sampling
        if history.len() < self.n_startup_trials {
            return self.sample_all_uniform(search_space, &mut rng);
        }

        // If group mode is enabled, decompose search space into independent groups
        if self.group {
            drop(rng);
            return self.sample_with_groups(search_space, history);
        }

        // Non-grouped mode: use the original single-group logic
        self.sample_single_group(search_space, history, &mut rng)
    }
}

impl Default for MultivariateTpeSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for MultivariateTpeSampler {
    /// Samples a parameter value from the given distribution using multivariate TPE.
    ///
    /// This method integrates with the multivariate sampling strategy by:
    /// 1. On first call for a trial: generating a joint sample for all parameters
    /// 2. On subsequent calls for the same trial: returning cached values
    ///
    /// This ensures consistency across parameters within the same trial while
    /// still conforming to the single-parameter [`Sampler`] interface.
    ///
    /// # Arguments
    ///
    /// * `distribution` - The parameter distribution to sample from.
    /// * `trial_id` - The unique ID of the trial being sampled for.
    /// * `history` - Historical completed trials for informed sampling.
    ///
    /// # Returns
    ///
    /// A [`ParamValue`] sampled from the distribution.
    ///
    /// # Note
    ///
    /// Since the [`Sampler`] trait doesn't provide the parameter name, this method
    /// must infer the parameter from the distribution. For best results with
    /// multivariate sampling, use [`sample_joint`](Self::sample_joint) directly
    /// when you have access to the full search space.
    fn sample(
        &self,
        distribution: &Distribution,
        trial_id: u64,
        history: &[CompletedTrial],
    ) -> ParamValue {
        // Check if we have a cached joint sample for this trial
        {
            let cache = self.joint_sample_cache.lock();
            if let Some(ref c) = *cache
                && c.trial_id == trial_id
            {
                // Try to find a matching parameter from the cached sample
                if let Some(value) =
                    Self::find_matching_param(distribution, &c.search_space, &c.sample)
                {
                    return value;
                }
            }
        }

        // Build the search space from history to get parameter names
        let search_space = Self::build_search_space_from_history(distribution, history);

        // Generate a joint sample
        let joint_sample = self.sample_joint(&search_space, history);

        // Cache the joint sample for this trial
        let result = Self::find_matching_param(distribution, &search_space, &joint_sample);
        {
            let mut cache = self.joint_sample_cache.lock();
            *cache = Some(JointSampleCache {
                trial_id,
                search_space,
                sample: joint_sample,
            });
        }

        // Find and return the value for the requested distribution
        result.unwrap_or_else(|| {
            // Fallback to uniform sampling if no match found
            let mut rng = self.rng.lock();
            crate::sampler::common::sample_random(&mut rng, distribution)
        })
    }
}

impl MultivariateTpeSampler {
    /// Finds a matching parameter value from the cached sample based on exact
    /// distribution equality.
    fn find_matching_param(
        distribution: &Distribution,
        search_space: &HashMap<ParamId, Distribution>,
        cached_sample: &HashMap<ParamId, ParamValue>,
    ) -> Option<ParamValue> {
        for (id, dist) in search_space {
            if dist == distribution
                && let Some(value) = cached_sample.get(id)
            {
                return Some(value.clone());
            }
        }
        None
    }

    /// Builds a search space from history and the current distribution.
    ///
    /// This is an associated function that extracts parameter distributions
    /// from historical trials and includes the current distribution to form
    /// a complete search space.
    fn build_search_space_from_history(
        current_distribution: &Distribution,
        history: &[CompletedTrial],
    ) -> HashMap<ParamId, Distribution> {
        let mut search_space = HashMap::new();

        // Collect distributions from history
        for trial in history {
            for (param_id, dist) in &trial.distributions {
                search_space
                    .entry(*param_id)
                    .or_insert_with(|| dist.clone());
            }
        }

        // If the search space is empty, create a placeholder for the current distribution
        if search_space.is_empty() {
            search_space.insert(ParamId::new(), current_distribution.clone());
        }

        search_space
    }
}

/// Builder for configuring a [`MultivariateTpeSampler`].
///
/// This builder allows fluent configuration of multivariate TPE hyperparameters.
///
/// # Examples
///
/// Using a fixed gamma value:
///
/// ```
/// use optimizer::sampler::tpe::MultivariateTpeSamplerBuilder;
///
/// let sampler = MultivariateTpeSamplerBuilder::new()
///     .gamma(0.15)
///     .n_startup_trials(20)
///     .n_ei_candidates(32)
///     .seed(42)
///     .build()
///     .unwrap();
/// ```
///
/// Using a custom gamma strategy:
///
/// ```
/// use optimizer::sampler::tpe::{MultivariateTpeSamplerBuilder, SqrtGamma};
///
/// let sampler = MultivariateTpeSamplerBuilder::new()
///     .gamma_strategy(SqrtGamma::default())
///     .group(true)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MultivariateTpeSamplerBuilder {
    gamma_strategy: Box<dyn GammaStrategy>,
    /// Raw gamma value for deferred validation (Some if `gamma()` was called).
    raw_gamma: Option<f64>,
    n_startup_trials: usize,
    n_ei_candidates: usize,
    group: bool,
    constant_liar: ConstantLiarStrategy,
    seed: Option<u64>,
}

impl MultivariateTpeSamplerBuilder {
    /// Creates a new builder with default settings.
    ///
    /// Default settings:
    /// - gamma strategy: [`FixedGamma`] with gamma = 0.25
    /// - `n_startup_trials`: 10 (random sampling for first 10 trials)
    /// - `n_ei_candidates`: 24 (evaluate 24 candidates per sample)
    /// - `group`: false (no group decomposition)
    /// - `constant_liar`: None (no imputation for pending trials)
    /// - seed: None (use OS-provided entropy)
    #[must_use]
    pub fn new() -> Self {
        Self {
            gamma_strategy: Box::new(FixedGamma::default()),
            raw_gamma: None,
            n_startup_trials: 10,
            n_ei_candidates: 24,
            group: false,
            constant_liar: ConstantLiarStrategy::None,
            seed: None,
        }
    }

    /// Sets a fixed gamma value for splitting trials into good/bad groups.
    ///
    /// This is a convenience method that creates a [`FixedGamma`] strategy.
    /// For more advanced gamma strategies, use [`gamma_strategy`](Self::gamma_strategy).
    ///
    /// A gamma of 0.25 means the top 25% of trials (by objective value) are
    /// considered "good" and used to build the l(x) distribution.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Quantile value, must be in (0.0, 1.0).
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::MultivariateTpeSamplerBuilder;
    ///
    /// let sampler = MultivariateTpeSamplerBuilder::new()
    ///     .gamma(0.10)  // Use top 10% as "good" trials
    ///     .build()
    ///     .unwrap();
    /// ```
    ///
    /// # Note
    ///
    /// Validation happens at `build()` time. If gamma is not in (0.0, 1.0),
    /// `build()` will return `Err(Error::InvalidGamma)`.
    #[must_use]
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.raw_gamma = Some(gamma);
        self
    }

    /// Sets a custom gamma strategy for splitting trials into good/bad groups.
    ///
    /// The gamma strategy determines what fraction of trials are considered
    /// "good" based on the number of completed trials. This allows the gamma
    /// value to adapt dynamically during optimization.
    ///
    /// # Arguments
    ///
    /// * `strategy` - A type implementing [`GammaStrategy`].
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::{LinearGamma, MultivariateTpeSamplerBuilder, SqrtGamma};
    ///
    /// // Square root strategy (Optuna-style)
    /// let sampler = MultivariateTpeSamplerBuilder::new()
    ///     .gamma_strategy(SqrtGamma::default())
    ///     .build()
    ///     .unwrap();
    ///
    /// // Linear interpolation strategy
    /// let sampler = MultivariateTpeSamplerBuilder::new()
    ///     .gamma_strategy(LinearGamma::new(0.1, 0.3, 50).unwrap())
    ///     .build()
    ///     .unwrap();
    /// ```
    #[must_use]
    pub fn gamma_strategy<G: GammaStrategy + 'static>(mut self, strategy: G) -> Self {
        self.gamma_strategy = Box::new(strategy);
        self.raw_gamma = None; // Clear any raw gamma set by gamma()
        self
    }

    /// Sets the number of startup trials before TPE sampling begins.
    ///
    /// During the startup phase, the sampler uses uniform random sampling
    /// to gather initial data. Once `n_startup_trials` have completed,
    /// multivariate TPE-based sampling begins.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of random trials before TPE kicks in.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::MultivariateTpeSamplerBuilder;
    ///
    /// let sampler = MultivariateTpeSamplerBuilder::new()
    ///     .n_startup_trials(20)  // Random sample first 20 trials
    ///     .build()
    ///     .unwrap();
    /// ```
    #[must_use]
    pub fn n_startup_trials(mut self, n: usize) -> Self {
        self.n_startup_trials = n;
        self
    }

    /// Sets the number of EI (Expected Improvement) candidates to evaluate.
    ///
    /// When sampling a new point, multivariate TPE generates this many candidates
    /// from the l(x) distribution and selects the one with the highest l(x)/g(x)
    /// ratio.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of candidates to evaluate per sample.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::MultivariateTpeSamplerBuilder;
    ///
    /// let sampler = MultivariateTpeSamplerBuilder::new()
    ///     .n_ei_candidates(48)  // Evaluate more candidates
    ///     .build()
    ///     .unwrap();
    /// ```
    #[must_use]
    pub fn n_ei_candidates(mut self, n: usize) -> Self {
        self.n_ei_candidates = n;
        self
    }

    /// Enables or disables group decomposition.
    ///
    /// When enabled, the sampler analyzes parameter co-occurrence in trial history
    /// and decomposes the search space into independent groups. Each group is then
    /// sampled using multivariate TPE independently, which can improve efficiency
    /// when some parameters are truly independent.
    ///
    /// # Arguments
    ///
    /// * `group` - Whether to enable group decomposition.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::MultivariateTpeSamplerBuilder;
    ///
    /// let sampler = MultivariateTpeSamplerBuilder::new()
    ///     .group(true)  // Enable group decomposition
    ///     .build()
    ///     .unwrap();
    /// ```
    #[must_use]
    pub fn group(mut self, group: bool) -> Self {
        self.group = group;
        self
    }

    /// Sets the constant liar strategy for parallel optimization.
    ///
    /// The constant liar strategy determines how to impute objective values for
    /// pending (not-yet-completed) trials. This is useful in parallel optimization
    /// where multiple trials may be running simultaneously.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The [`ConstantLiarStrategy`] to use.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::{ConstantLiarStrategy, MultivariateTpeSamplerBuilder};
    ///
    /// // Use mean imputation for pending trials
    /// let sampler = MultivariateTpeSamplerBuilder::new()
    ///     .constant_liar(ConstantLiarStrategy::Mean)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Use worst-case imputation (pessimistic)
    /// let sampler = MultivariateTpeSamplerBuilder::new()
    ///     .constant_liar(ConstantLiarStrategy::Worst)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Use a custom imputation value
    /// let sampler = MultivariateTpeSamplerBuilder::new()
    ///     .constant_liar(ConstantLiarStrategy::Custom(0.5))
    ///     .build()
    ///     .unwrap();
    /// ```
    #[must_use]
    pub fn constant_liar(mut self, strategy: ConstantLiarStrategy) -> Self {
        self.constant_liar = strategy;
        self
    }

    /// Sets a seed for reproducible sampling.
    ///
    /// # Arguments
    ///
    /// * `seed` - Seed value for the random number generator.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::MultivariateTpeSamplerBuilder;
    ///
    /// let sampler = MultivariateTpeSamplerBuilder::new()
    ///     .seed(42)  // Reproducible results
    ///     .build()
    ///     .unwrap();
    /// ```
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Builds the configured [`MultivariateTpeSampler`].
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidGamma` if a fixed gamma value was set and is not in (0.0, 1.0).
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::MultivariateTpeSamplerBuilder;
    ///
    /// let sampler = MultivariateTpeSamplerBuilder::new()
    ///     .gamma(0.15)
    ///     .n_startup_trials(20)
    ///     .n_ei_candidates(32)
    ///     .seed(42)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn build(self) -> Result<MultivariateTpeSampler> {
        // Determine the gamma strategy to use
        let gamma_strategy: Arc<dyn GammaStrategy> = if let Some(raw) = self.raw_gamma {
            // Validate and create FixedGamma from raw value
            Arc::new(FixedGamma::new(raw)?)
        } else {
            Arc::from(self.gamma_strategy)
        };

        let rng = match self.seed {
            Some(s) => fastrand::Rng::with_seed(s),
            None => fastrand::Rng::new(),
        };

        Ok(MultivariateTpeSampler {
            gamma_strategy,
            n_startup_trials: self.n_startup_trials,
            n_ei_candidates: self.n_ei_candidates,
            group: self.group,
            constant_liar: self.constant_liar,
            rng: Mutex::new(rng),
            joint_sample_cache: Mutex::new(None),
        })
    }
}

impl Default for MultivariateTpeSamplerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::sampler::tpe::{HyperoptGamma, LinearGamma, SqrtGamma};

    #[test]
    fn test_multivariate_tpe_sampler_new() {
        let sampler = MultivariateTpeSampler::new();

        // Check default values
        assert!(
            (sampler.gamma_strategy().gamma(0) - 0.25).abs() < f64::EPSILON,
            "Default gamma should be 0.25"
        );
        assert_eq!(sampler.n_startup_trials(), 10);
        assert_eq!(sampler.n_ei_candidates(), 24);
        assert!(!sampler.group());
    }

    #[test]
    fn test_multivariate_tpe_sampler_default() {
        let sampler = MultivariateTpeSampler::default();

        // Default should match new()
        assert!(
            (sampler.gamma_strategy().gamma(0) - 0.25).abs() < f64::EPSILON,
            "Default gamma should be 0.25"
        );
        assert_eq!(sampler.n_startup_trials(), 10);
        assert_eq!(sampler.n_ei_candidates(), 24);
        assert!(!sampler.group());
    }

    // ========================================================================
    // Builder Tests
    // ========================================================================

    #[test]
    fn test_builder_default() {
        let builder = MultivariateTpeSamplerBuilder::new();
        let sampler = builder.build().unwrap();

        assert!(
            (sampler.gamma_strategy().gamma(0) - 0.25).abs() < f64::EPSILON,
            "Default gamma should be 0.25"
        );
        assert_eq!(sampler.n_startup_trials(), 10);
        assert_eq!(sampler.n_ei_candidates(), 24);
        assert!(!sampler.group());
    }

    #[test]
    fn test_builder_default_impl() {
        let builder = MultivariateTpeSamplerBuilder::default();
        let sampler = builder.build().unwrap();

        assert!(
            (sampler.gamma_strategy().gamma(0) - 0.25).abs() < f64::EPSILON,
            "Default gamma should be 0.25"
        );
        assert_eq!(sampler.n_startup_trials(), 10);
    }

    #[test]
    fn test_builder_gamma() {
        let sampler = MultivariateTpeSamplerBuilder::new()
            .gamma(0.15)
            .build()
            .unwrap();

        assert!(
            (sampler.gamma_strategy().gamma(0) - 0.15).abs() < f64::EPSILON,
            "Gamma should be 0.15"
        );
        // Gamma should be constant across trial counts (FixedGamma)
        assert!(
            (sampler.gamma_strategy().gamma(100) - 0.15).abs() < f64::EPSILON,
            "Gamma should be constant"
        );
    }

    #[test]
    fn test_builder_gamma_invalid_zero() {
        let result = MultivariateTpeSamplerBuilder::new().gamma(0.0).build();
        assert!(matches!(result, Err(Error::InvalidGamma(_))));
    }

    #[test]
    fn test_builder_gamma_invalid_one() {
        let result = MultivariateTpeSamplerBuilder::new().gamma(1.0).build();
        assert!(matches!(result, Err(Error::InvalidGamma(_))));
    }

    #[test]
    fn test_builder_gamma_invalid_negative() {
        let result = MultivariateTpeSamplerBuilder::new().gamma(-0.1).build();
        assert!(matches!(result, Err(Error::InvalidGamma(_))));
    }

    #[test]
    fn test_builder_gamma_invalid_greater_than_one() {
        let result = MultivariateTpeSamplerBuilder::new().gamma(1.5).build();
        assert!(matches!(result, Err(Error::InvalidGamma(_))));
    }

    #[test]
    fn test_builder_gamma_strategy_sqrt() {
        let sampler = MultivariateTpeSamplerBuilder::new()
            .gamma_strategy(SqrtGamma::default())
            .build()
            .unwrap();

        // SqrtGamma decreases with more trials
        let g10 = sampler.gamma_strategy().gamma(10);
        let g100 = sampler.gamma_strategy().gamma(100);
        assert!(g10 > g100, "SqrtGamma should decrease with more trials");
    }

    #[test]
    fn test_builder_gamma_strategy_linear() {
        let sampler = MultivariateTpeSamplerBuilder::new()
            .gamma_strategy(LinearGamma::new(0.1, 0.4, 100).unwrap())
            .build()
            .unwrap();

        assert!((sampler.gamma_strategy().gamma(0) - 0.1).abs() < f64::EPSILON);
        assert!((sampler.gamma_strategy().gamma(50) - 0.25).abs() < f64::EPSILON);
        assert!((sampler.gamma_strategy().gamma(100) - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_builder_gamma_strategy_hyperopt() {
        let sampler = MultivariateTpeSamplerBuilder::new()
            .gamma_strategy(HyperoptGamma::default())
            .build()
            .unwrap();

        // HyperoptGamma decreases with more trials
        let g50 = sampler.gamma_strategy().gamma(50);
        let g200 = sampler.gamma_strategy().gamma(200);
        assert!(g50 > g200, "HyperoptGamma should decrease with more trials");
    }

    #[test]
    fn test_builder_n_startup_trials() {
        let sampler = MultivariateTpeSamplerBuilder::new()
            .n_startup_trials(20)
            .build()
            .unwrap();

        assert_eq!(sampler.n_startup_trials(), 20);
    }

    #[test]
    fn test_builder_n_ei_candidates() {
        let sampler = MultivariateTpeSamplerBuilder::new()
            .n_ei_candidates(48)
            .build()
            .unwrap();

        assert_eq!(sampler.n_ei_candidates(), 48);
    }

    #[test]
    fn test_builder_group() {
        let sampler = MultivariateTpeSamplerBuilder::new()
            .group(true)
            .build()
            .unwrap();

        assert!(sampler.group());
    }

    #[test]
    fn test_builder_seed() {
        // Two samplers with the same seed should produce the same sequence
        // We can't directly test RNG output, but we verify build succeeds
        let sampler1 = MultivariateTpeSamplerBuilder::new()
            .seed(42)
            .build()
            .unwrap();

        let sampler2 = MultivariateTpeSamplerBuilder::new()
            .seed(42)
            .build()
            .unwrap();

        // Verify both built successfully with same config
        assert_eq!(sampler1.n_startup_trials(), sampler2.n_startup_trials());
    }

    #[test]
    fn test_builder_all_options() {
        let sampler = MultivariateTpeSamplerBuilder::new()
            .gamma(0.20)
            .n_startup_trials(15)
            .n_ei_candidates(32)
            .group(true)
            .seed(12345)
            .build()
            .unwrap();

        assert!((sampler.gamma_strategy().gamma(0) - 0.20).abs() < f64::EPSILON);
        assert_eq!(sampler.n_startup_trials(), 15);
        assert_eq!(sampler.n_ei_candidates(), 32);
        assert!(sampler.group());
    }

    #[test]
    fn test_builder_gamma_overrides_gamma_strategy() {
        // When gamma() is called after gamma_strategy(), it should take precedence
        let sampler = MultivariateTpeSamplerBuilder::new()
            .gamma_strategy(SqrtGamma::default())
            .gamma(0.15) // This should override
            .build()
            .unwrap();

        // Should use fixed gamma of 0.15
        assert!((sampler.gamma_strategy().gamma(0) - 0.15).abs() < f64::EPSILON);
        assert!((sampler.gamma_strategy().gamma(100) - 0.15).abs() < f64::EPSILON);
    }

    #[test]
    fn test_builder_gamma_strategy_overrides_gamma() {
        // When gamma_strategy() is called after gamma(), it should take precedence
        let sampler = MultivariateTpeSamplerBuilder::new()
            .gamma(0.15)
            .gamma_strategy(SqrtGamma::default()) // This should override
            .build()
            .unwrap();

        // Should use SqrtGamma - gamma decreases with trials
        let g10 = sampler.gamma_strategy().gamma(10);
        let g100 = sampler.gamma_strategy().gamma(100);
        assert!(g10 > g100, "SqrtGamma should decrease with more trials");
    }

    #[test]
    fn test_builder_via_sampler() {
        let sampler = MultivariateTpeSampler::builder()
            .gamma(0.10)
            .n_startup_trials(25)
            .n_ei_candidates(64)
            .group(true)
            .build()
            .unwrap();

        assert!((sampler.gamma_strategy().gamma(0) - 0.10).abs() < f64::EPSILON);
        assert_eq!(sampler.n_startup_trials(), 25);
        assert_eq!(sampler.n_ei_candidates(), 64);
        assert!(sampler.group());
    }

    #[test]
    fn test_builder_custom_gamma_strategy() {
        #[derive(Debug, Clone)]
        struct ConstantGamma(f64);

        impl GammaStrategy for ConstantGamma {
            fn gamma(&self, _n_trials: usize) -> f64 {
                self.0
            }

            fn clone_box(&self) -> Box<dyn GammaStrategy> {
                Box::new(self.clone())
            }
        }

        let sampler = MultivariateTpeSamplerBuilder::new()
            .gamma_strategy(ConstantGamma(0.33))
            .build()
            .unwrap();

        assert!((sampler.gamma_strategy().gamma(0) - 0.33).abs() < f64::EPSILON);
        assert!((sampler.gamma_strategy().gamma(100) - 0.33).abs() < f64::EPSILON);
    }

    #[test]
    fn test_builder_clone() {
        let builder = MultivariateTpeSamplerBuilder::new()
            .gamma(0.20)
            .n_startup_trials(15);

        let builder2 = builder.clone();

        let sampler1 = builder.build().unwrap();
        let sampler2 = builder2.build().unwrap();

        assert!((sampler1.gamma_strategy().gamma(0) - 0.20).abs() < f64::EPSILON);
        assert!((sampler2.gamma_strategy().gamma(0) - 0.20).abs() < f64::EPSILON);
        assert_eq!(sampler1.n_startup_trials(), 15);
        assert_eq!(sampler2.n_startup_trials(), 15);
    }

    #[test]
    fn test_builder_constant_liar_default() {
        let sampler = MultivariateTpeSamplerBuilder::new().build().unwrap();

        assert_eq!(*sampler.constant_liar(), ConstantLiarStrategy::None);
    }

    #[test]
    fn test_builder_constant_liar_mean() {
        let sampler = MultivariateTpeSamplerBuilder::new()
            .constant_liar(ConstantLiarStrategy::Mean)
            .build()
            .unwrap();

        assert_eq!(*sampler.constant_liar(), ConstantLiarStrategy::Mean);
    }

    #[test]
    fn test_builder_constant_liar_best() {
        let sampler = MultivariateTpeSamplerBuilder::new()
            .constant_liar(ConstantLiarStrategy::Best)
            .build()
            .unwrap();

        assert_eq!(*sampler.constant_liar(), ConstantLiarStrategy::Best);
    }

    #[test]
    fn test_builder_constant_liar_worst() {
        let sampler = MultivariateTpeSamplerBuilder::new()
            .constant_liar(ConstantLiarStrategy::Worst)
            .build()
            .unwrap();

        assert_eq!(*sampler.constant_liar(), ConstantLiarStrategy::Worst);
    }

    #[test]
    fn test_builder_constant_liar_custom() {
        let sampler = MultivariateTpeSamplerBuilder::new()
            .constant_liar(ConstantLiarStrategy::Custom(0.5))
            .build()
            .unwrap();

        match sampler.constant_liar() {
            ConstantLiarStrategy::Custom(v) => {
                assert!((v - 0.5).abs() < f64::EPSILON);
            }
            _ => panic!("Expected Custom variant"),
        }
    }

    #[test]
    fn test_builder_constant_liar_with_other_options() {
        let sampler = MultivariateTpeSamplerBuilder::new()
            .gamma(0.20)
            .n_startup_trials(15)
            .n_ei_candidates(32)
            .constant_liar(ConstantLiarStrategy::Worst)
            .seed(42)
            .build()
            .unwrap();

        assert!((sampler.gamma_strategy().gamma(0) - 0.20).abs() < f64::EPSILON);
        assert_eq!(sampler.n_startup_trials(), 15);
        assert_eq!(sampler.n_ei_candidates(), 32);
        assert_eq!(*sampler.constant_liar(), ConstantLiarStrategy::Worst);
    }

    // ========================================================================
    // impute_pending_trials Tests
    // ========================================================================

    mod impute_pending_trials_tests {
        use std::collections::HashMap;

        use super::*;
        use crate::distribution::FloatDistribution;
        use crate::param::ParamValue;
        use crate::parameter::ParamId;
        use crate::sampler::{CompletedTrial, PendingTrial};

        fn float_dist() -> Distribution {
            Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            })
        }

        fn create_completed_trial(id: u64, x_value: f64, objective: f64) -> CompletedTrial {
            let x_id = ParamId::new();
            let mut params = HashMap::new();
            params.insert(x_id, ParamValue::Float(x_value));
            let mut distributions = HashMap::new();
            distributions.insert(x_id, float_dist());
            CompletedTrial::new(id, params, distributions, HashMap::new(), objective)
        }

        fn create_pending_trial(id: u64, x_value: f64) -> PendingTrial {
            let x_id = ParamId::new();
            let mut params = HashMap::new();
            params.insert(x_id, ParamValue::Float(x_value));
            let mut distributions = HashMap::new();
            distributions.insert(x_id, float_dist());
            PendingTrial::new(id, params, distributions, HashMap::new())
        }

        #[test]
        fn test_impute_none_strategy_ignores_pending() {
            let sampler = MultivariateTpeSamplerBuilder::new()
                .constant_liar(ConstantLiarStrategy::None)
                .build()
                .unwrap();

            let completed = vec![
                create_completed_trial(0, 0.2, 1.0),
                create_completed_trial(1, 0.8, 3.0),
            ];
            let pending = vec![create_pending_trial(2, 0.5)];

            let result = sampler.impute_pending_trials(&pending, &completed);

            // Should only have the completed trials
            assert_eq!(result.len(), 2);
            assert!(result.iter().all(|t| t.id == 0 || t.id == 1));
        }

        #[test]
        fn test_impute_mean_strategy() {
            let sampler = MultivariateTpeSamplerBuilder::new()
                .constant_liar(ConstantLiarStrategy::Mean)
                .build()
                .unwrap();

            let completed = vec![
                create_completed_trial(0, 0.2, 1.0),
                create_completed_trial(1, 0.8, 3.0),
            ];
            let pending = vec![create_pending_trial(2, 0.5)];

            let result = sampler.impute_pending_trials(&pending, &completed);

            // Should have 3 trials
            assert_eq!(result.len(), 3);

            // The pending trial should have mean value (1.0 + 3.0) / 2 = 2.0
            let imputed = result.iter().find(|t| t.id == 2).unwrap();
            assert!((imputed.value - 2.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_impute_best_strategy() {
            let sampler = MultivariateTpeSamplerBuilder::new()
                .constant_liar(ConstantLiarStrategy::Best)
                .build()
                .unwrap();

            let completed = vec![
                create_completed_trial(0, 0.2, 1.0),
                create_completed_trial(1, 0.8, 3.0),
                create_completed_trial(2, 0.5, 2.0),
            ];
            let pending = vec![create_pending_trial(3, 0.6)];

            let result = sampler.impute_pending_trials(&pending, &completed);

            assert_eq!(result.len(), 4);

            // Best (minimum) is 1.0
            let imputed = result.iter().find(|t| t.id == 3).unwrap();
            assert!((imputed.value - 1.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_impute_worst_strategy() {
            let sampler = MultivariateTpeSamplerBuilder::new()
                .constant_liar(ConstantLiarStrategy::Worst)
                .build()
                .unwrap();

            let completed = vec![
                create_completed_trial(0, 0.2, 1.0),
                create_completed_trial(1, 0.8, 3.0),
                create_completed_trial(2, 0.5, 2.0),
            ];
            let pending = vec![create_pending_trial(3, 0.6)];

            let result = sampler.impute_pending_trials(&pending, &completed);

            assert_eq!(result.len(), 4);

            // Worst (maximum) is 3.0
            let imputed = result.iter().find(|t| t.id == 3).unwrap();
            assert!((imputed.value - 3.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_impute_custom_strategy() {
            let custom_value = 42.0;
            let sampler = MultivariateTpeSamplerBuilder::new()
                .constant_liar(ConstantLiarStrategy::Custom(custom_value))
                .build()
                .unwrap();

            let completed = vec![
                create_completed_trial(0, 0.2, 1.0),
                create_completed_trial(1, 0.8, 3.0),
            ];
            let pending = vec![create_pending_trial(2, 0.5)];

            let result = sampler.impute_pending_trials(&pending, &completed);

            assert_eq!(result.len(), 3);

            // Should use custom value regardless of completed values
            let imputed = result.iter().find(|t| t.id == 2).unwrap();
            assert!((imputed.value - custom_value).abs() < f64::EPSILON);
        }

        #[test]
        fn test_impute_multiple_pending_trials() {
            let sampler = MultivariateTpeSamplerBuilder::new()
                .constant_liar(ConstantLiarStrategy::Mean)
                .build()
                .unwrap();

            let completed = vec![
                create_completed_trial(0, 0.2, 1.0),
                create_completed_trial(1, 0.8, 5.0),
            ];
            let pending = vec![
                create_pending_trial(2, 0.3),
                create_pending_trial(3, 0.7),
                create_pending_trial(4, 0.5),
            ];

            let result = sampler.impute_pending_trials(&pending, &completed);

            // Should have 5 trials total
            assert_eq!(result.len(), 5);

            // All pending trials should have mean value (1.0 + 5.0) / 2 = 3.0
            let mean_value = 3.0;
            for id in [2, 3, 4] {
                let imputed = result.iter().find(|t| t.id == id).unwrap();
                assert!(
                    (imputed.value - mean_value).abs() < f64::EPSILON,
                    "Trial {id} should have imputed value {mean_value}"
                );
            }
        }

        #[test]
        fn test_impute_empty_pending_trials() {
            let sampler = MultivariateTpeSamplerBuilder::new()
                .constant_liar(ConstantLiarStrategy::Mean)
                .build()
                .unwrap();

            let completed = vec![
                create_completed_trial(0, 0.2, 1.0),
                create_completed_trial(1, 0.8, 3.0),
            ];
            let pending: Vec<PendingTrial> = vec![];

            let result = sampler.impute_pending_trials(&pending, &completed);

            // Should just return the completed trials unchanged
            assert_eq!(result.len(), 2);
        }

        #[test]
        fn test_impute_empty_completed_trials_mean() {
            let sampler = MultivariateTpeSamplerBuilder::new()
                .constant_liar(ConstantLiarStrategy::Mean)
                .build()
                .unwrap();

            let completed: Vec<CompletedTrial> = vec![];
            let pending = vec![create_pending_trial(0, 0.5)];

            let result = sampler.impute_pending_trials(&pending, &completed);

            // Should have 1 trial with imputed value 0.0 (mean of empty is 0)
            assert_eq!(result.len(), 1);
            assert!((result[0].value - 0.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_impute_empty_completed_trials_best() {
            let sampler = MultivariateTpeSamplerBuilder::new()
                .constant_liar(ConstantLiarStrategy::Best)
                .build()
                .unwrap();

            let completed: Vec<CompletedTrial> = vec![];
            let pending = vec![create_pending_trial(0, 0.5)];

            let result = sampler.impute_pending_trials(&pending, &completed);

            // Best of empty is INFINITY
            assert_eq!(result.len(), 1);
            assert!(result[0].value.is_infinite() && result[0].value > 0.0);
        }

        #[test]
        fn test_impute_empty_completed_trials_worst() {
            let sampler = MultivariateTpeSamplerBuilder::new()
                .constant_liar(ConstantLiarStrategy::Worst)
                .build()
                .unwrap();

            let completed: Vec<CompletedTrial> = vec![];
            let pending = vec![create_pending_trial(0, 0.5)];

            let result = sampler.impute_pending_trials(&pending, &completed);

            // Worst of empty is NEG_INFINITY
            assert_eq!(result.len(), 1);
            assert!(result[0].value.is_infinite() && result[0].value < 0.0);
        }

        #[test]
        fn test_impute_empty_completed_trials_custom() {
            let custom_value = 100.0;
            let sampler = MultivariateTpeSamplerBuilder::new()
                .constant_liar(ConstantLiarStrategy::Custom(custom_value))
                .build()
                .unwrap();

            let completed: Vec<CompletedTrial> = vec![];
            let pending = vec![create_pending_trial(0, 0.5)];

            let result = sampler.impute_pending_trials(&pending, &completed);

            // Custom value is used regardless of completed trials
            assert_eq!(result.len(), 1);
            assert!((result[0].value - custom_value).abs() < f64::EPSILON);
        }

        #[test]
        fn test_impute_preserves_pending_trial_params() {
            let sampler = MultivariateTpeSamplerBuilder::new()
                .constant_liar(ConstantLiarStrategy::Mean)
                .build()
                .unwrap();

            let completed = vec![create_completed_trial(0, 0.2, 1.0)];

            // Create a pending trial with specific parameter value
            let x_id = ParamId::new();
            let mut params = HashMap::new();
            params.insert(x_id, ParamValue::Float(0.777));
            let mut distributions = HashMap::new();
            distributions.insert(x_id, float_dist());
            let pending = vec![PendingTrial::new(1, params, distributions, HashMap::new())];

            let result = sampler.impute_pending_trials(&pending, &completed);

            assert_eq!(result.len(), 2);

            let imputed = result.iter().find(|t| t.id == 1).unwrap();

            // Parameter value should be preserved
            if let Some(ParamValue::Float(v)) = imputed.params.get(&x_id) {
                assert!((*v - 0.777).abs() < f64::EPSILON);
            } else {
                panic!("Expected Float parameter");
            }

            // Distribution should be preserved
            assert!(imputed.distributions.contains_key(&x_id));
        }

        #[test]
        fn test_impute_with_negative_values() {
            let sampler = MultivariateTpeSamplerBuilder::new()
                .constant_liar(ConstantLiarStrategy::Mean)
                .build()
                .unwrap();

            let completed = vec![
                create_completed_trial(0, 0.2, -5.0),
                create_completed_trial(1, 0.8, 3.0),
            ];
            let pending = vec![create_pending_trial(2, 0.5)];

            let result = sampler.impute_pending_trials(&pending, &completed);

            // Mean is (-5.0 + 3.0) / 2 = -1.0
            let imputed = result.iter().find(|t| t.id == 2).unwrap();
            assert!((imputed.value - (-1.0)).abs() < f64::EPSILON);
        }

        #[test]
        fn test_impute_best_with_negative_values() {
            let sampler = MultivariateTpeSamplerBuilder::new()
                .constant_liar(ConstantLiarStrategy::Best)
                .build()
                .unwrap();

            let completed = vec![
                create_completed_trial(0, 0.2, -5.0),
                create_completed_trial(1, 0.8, 3.0),
            ];
            let pending = vec![create_pending_trial(2, 0.5)];

            let result = sampler.impute_pending_trials(&pending, &completed);

            // Best (minimum) is -5.0
            let imputed = result.iter().find(|t| t.id == 2).unwrap();
            assert!((imputed.value - (-5.0)).abs() < f64::EPSILON);
        }
    }

    // ========================================================================
    // filter_trials Tests
    // ========================================================================

    mod filter_trials_tests {
        use std::collections::HashMap;

        use super::*;
        use crate::distribution::{FloatDistribution, IntDistribution};
        use crate::param::ParamValue;
        use crate::parameter::ParamId;
        use crate::sampler::CompletedTrial;

        fn create_trial(
            id: u64,
            params: Vec<(ParamId, ParamValue, Distribution)>,
            value: f64,
        ) -> CompletedTrial {
            let mut param_map = HashMap::new();
            let mut dist_map = HashMap::new();
            for (param_id, pv, dist) in params {
                param_map.insert(param_id, pv);
                dist_map.insert(param_id, dist);
            }
            CompletedTrial::new(id, param_map, dist_map, HashMap::new(), value)
        }

        fn float_dist() -> Distribution {
            Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            })
        }

        fn int_dist() -> Distribution {
            Distribution::Int(IntDistribution {
                low: 1,
                high: 10,
                log_scale: false,
                step: None,
            })
        }

        #[test]
        fn test_filter_trials_empty_history() {
            let sampler = MultivariateTpeSampler::new();
            let history: Vec<CompletedTrial> = vec![];
            let search_space: HashMap<ParamId, Distribution> = HashMap::new();

            let filtered = sampler.filter_trials(&history, &search_space);
            assert!(filtered.is_empty());
        }

        #[test]
        fn test_filter_trials_empty_search_space() {
            let sampler = MultivariateTpeSampler::new();
            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let history = vec![
                create_trial(0, vec![(x_id, ParamValue::Float(0.5), float_dist())], 1.0),
                create_trial(1, vec![(y_id, ParamValue::Float(0.3), float_dist())], 0.5),
            ];
            let search_space: HashMap<ParamId, Distribution> = HashMap::new();

            // With empty search space, all trials should pass (vacuously true)
            let filtered = sampler.filter_trials(&history, &search_space);
            assert_eq!(filtered.len(), 2);
        }

        #[test]
        fn test_filter_trials_all_match() {
            let sampler = MultivariateTpeSampler::new();
            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let history = vec![
                create_trial(
                    0,
                    vec![
                        (x_id, ParamValue::Float(0.5), float_dist()),
                        (y_id, ParamValue::Float(0.3), float_dist()),
                    ],
                    1.0,
                ),
                create_trial(
                    1,
                    vec![
                        (x_id, ParamValue::Float(0.7), float_dist()),
                        (y_id, ParamValue::Float(0.2), float_dist()),
                    ],
                    0.5,
                ),
            ];

            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist());
            search_space.insert(y_id, float_dist());

            let filtered = sampler.filter_trials(&history, &search_space);
            assert_eq!(filtered.len(), 2);
            assert_eq!(filtered[0].id, 0);
            assert_eq!(filtered[1].id, 1);
        }

        #[test]
        fn test_filter_trials_partial_match() {
            let sampler = MultivariateTpeSampler::new();
            let x_id = ParamId::new();
            let y_id = ParamId::new();

            // Trial 0: has x and y
            // Trial 1: has only x
            // Trial 2: has x and y
            let history = vec![
                create_trial(
                    0,
                    vec![
                        (x_id, ParamValue::Float(0.5), float_dist()),
                        (y_id, ParamValue::Float(0.3), float_dist()),
                    ],
                    1.0,
                ),
                create_trial(1, vec![(x_id, ParamValue::Float(0.7), float_dist())], 0.5),
                create_trial(
                    2,
                    vec![
                        (x_id, ParamValue::Float(0.6), float_dist()),
                        (y_id, ParamValue::Float(0.4), float_dist()),
                    ],
                    0.8,
                ),
            ];

            // Search space requires both x and y
            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist());
            search_space.insert(y_id, float_dist());

            let filtered = sampler.filter_trials(&history, &search_space);

            // Only trials 0 and 2 should match (trial 1 is missing y)
            assert_eq!(filtered.len(), 2);
            assert_eq!(filtered[0].id, 0);
            assert_eq!(filtered[1].id, 2);
        }

        #[test]
        fn test_filter_trials_none_match() {
            let sampler = MultivariateTpeSampler::new();
            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let z_id = ParamId::new();

            // All trials have x, but search space requires both x and z
            let history = vec![
                create_trial(
                    0,
                    vec![
                        (x_id, ParamValue::Float(0.5), float_dist()),
                        (y_id, ParamValue::Float(0.3), float_dist()),
                    ],
                    1.0,
                ),
                create_trial(
                    1,
                    vec![
                        (x_id, ParamValue::Float(0.7), float_dist()),
                        (y_id, ParamValue::Float(0.2), float_dist()),
                    ],
                    0.5,
                ),
            ];

            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist());
            search_space.insert(z_id, float_dist()); // No trial has z

            let filtered = sampler.filter_trials(&history, &search_space);

            // No trials should match since none have z
            assert!(filtered.is_empty());
        }

        #[test]
        fn test_filter_trials_mixed_param_types() {
            let sampler = MultivariateTpeSampler::new();
            let lr_id = ParamId::new();
            let layers_id = ParamId::new();
            let dropout_id = ParamId::new();

            // Trials with mixed parameter types
            let history = vec![
                create_trial(
                    0,
                    vec![
                        (lr_id, ParamValue::Float(0.01), float_dist()),
                        (layers_id, ParamValue::Int(3), int_dist()),
                    ],
                    1.0,
                ),
                create_trial(
                    1,
                    vec![
                        (lr_id, ParamValue::Float(0.001), float_dist()),
                        (layers_id, ParamValue::Int(5), int_dist()),
                        (dropout_id, ParamValue::Float(0.2), float_dist()), // Extra param
                    ],
                    0.8,
                ),
                create_trial(
                    2,
                    vec![
                        (lr_id, ParamValue::Float(0.005), float_dist()),
                        // Missing n_layers
                        (dropout_id, ParamValue::Float(0.1), float_dist()),
                    ],
                    0.9,
                ),
            ];

            // Search space requires learning_rate and n_layers
            let mut search_space = HashMap::new();
            search_space.insert(lr_id, float_dist());
            search_space.insert(layers_id, int_dist());

            let filtered = sampler.filter_trials(&history, &search_space);

            // Trials 0 and 1 have both params, trial 2 is missing n_layers
            assert_eq!(filtered.len(), 2);
            assert_eq!(filtered[0].id, 0);
            assert_eq!(filtered[1].id, 1);
        }

        #[test]
        fn test_filter_trials_superset_params_accepted() {
            let sampler = MultivariateTpeSampler::new();
            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let z_id = ParamId::new();
            let w_id = ParamId::new();

            // All trials have more params than the search space requires
            let history = vec![
                create_trial(
                    0,
                    vec![
                        (x_id, ParamValue::Float(0.5), float_dist()),
                        (y_id, ParamValue::Float(0.3), float_dist()),
                        (z_id, ParamValue::Float(0.1), float_dist()),
                    ],
                    1.0,
                ),
                create_trial(
                    1,
                    vec![
                        (x_id, ParamValue::Float(0.7), float_dist()),
                        (y_id, ParamValue::Float(0.2), float_dist()),
                        (w_id, ParamValue::Float(0.9), float_dist()), // Different extra param
                    ],
                    0.5,
                ),
            ];

            // Search space only requires x
            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist());

            let filtered = sampler.filter_trials(&history, &search_space);

            // Both trials should be accepted (they have x, even though they have extras)
            assert_eq!(filtered.len(), 2);
        }

        #[test]
        fn test_filter_trials_preserves_order() {
            let sampler = MultivariateTpeSampler::new();
            let x_id = ParamId::new();

            let history: Vec<CompletedTrial> = (0..10)
                .map(|i| create_trial(i, vec![(x_id, ParamValue::Float(0.5), float_dist())], 1.0))
                .collect();

            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist());

            let filtered = sampler.filter_trials(&history, &search_space);

            // Order should be preserved
            for (i, trial) in filtered.iter().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                let expected_id = i as u64;
                assert_eq!(trial.id, expected_id);
            }
        }

        #[test]
        fn test_filter_trials_single_param_search_space() {
            let sampler = MultivariateTpeSampler::new();
            let x_id = ParamId::new();
            let y_id = ParamId::new();

            // Some trials have the required param, some don't
            let history = vec![
                create_trial(
                    0,
                    vec![
                        (x_id, ParamValue::Float(0.5), float_dist()),
                        (y_id, ParamValue::Float(0.3), float_dist()),
                    ],
                    1.0,
                ),
                create_trial(1, vec![(y_id, ParamValue::Float(0.7), float_dist())], 0.5),
                create_trial(2, vec![(x_id, ParamValue::Float(0.6), float_dist())], 0.8),
            ];

            // Search space only requires x
            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist());

            let filtered = sampler.filter_trials(&history, &search_space);

            // Trials 0 and 2 have x
            assert_eq!(filtered.len(), 2);
            assert_eq!(filtered[0].id, 0);
            assert_eq!(filtered[1].id, 2);
        }
    }

    // ========================================================================
    // split_trials Tests
    // ========================================================================

    #[allow(clippy::cast_precision_loss)]
    mod split_trials_tests {
        use std::collections::HashMap;

        use super::*;
        use crate::distribution::FloatDistribution;
        use crate::param::ParamValue;
        use crate::parameter::ParamId;
        use crate::sampler::CompletedTrial;

        fn create_trial(id: u64, value: f64) -> CompletedTrial {
            let float_dist = Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            });
            let x_id = ParamId::new();
            let mut params = HashMap::new();
            params.insert(x_id, ParamValue::Float(0.5));
            let mut distributions = HashMap::new();
            distributions.insert(x_id, float_dist);
            CompletedTrial::new(id, params, distributions, HashMap::new(), value)
        }

        #[test]
        fn test_split_trials_empty() {
            let sampler = MultivariateTpeSampler::new();
            let trials: Vec<&CompletedTrial> = vec![];

            let (good, bad) = sampler.split_trials(&trials);

            assert!(good.is_empty());
            assert!(bad.is_empty());
        }

        #[test]
        fn test_split_trials_single_trial() {
            let sampler = MultivariateTpeSampler::new();
            let trial = create_trial(0, 1.0);
            let trials: Vec<&CompletedTrial> = vec![&trial];

            let (good, bad) = sampler.split_trials(&trials);

            // Single trial should go to good
            assert_eq!(good.len(), 1);
            assert!(bad.is_empty());
            assert_eq!(good[0].id, 0);
        }

        #[test]
        fn test_split_trials_two_trials() {
            let sampler = MultivariateTpeSampler::new();
            let good_trial = create_trial(0, 0.5); // Better (lower)
            let bad_trial = create_trial(1, 1.0); // Worse (higher)
            let trial_refs: Vec<&CompletedTrial> = vec![&good_trial, &bad_trial];

            let (good, bad) = sampler.split_trials(&trial_refs);

            // With 2 trials and gamma=0.25, ceil(2*0.25)=1, so 1 good, 1 bad
            assert_eq!(good.len(), 1);
            assert_eq!(bad.len(), 1);
            assert_eq!(good[0].id, 0); // Lower value in good
            assert_eq!(bad[0].id, 1); // Higher value in bad
        }

        #[test]
        fn test_split_trials_many_trials_default_gamma() {
            // Default gamma is 0.25, so with 20 trials, ceil(20*0.25)=5 good
            let sampler = MultivariateTpeSampler::new();

            // Create 20 trials with values 0..20
            let trial_data: Vec<CompletedTrial> =
                (0..20).map(|i| create_trial(i, i as f64)).collect();
            let trial_refs: Vec<&CompletedTrial> = trial_data.iter().collect();

            let (good, bad) = sampler.split_trials(&trial_refs);

            // With gamma=0.25 and 20 trials: ceil(20 * 0.25) = 5 good
            assert_eq!(good.len(), 5);
            assert_eq!(bad.len(), 15);

            // Good trials should have lowest values (0, 1, 2, 3, 4)
            for trial in &good {
                assert!(
                    trial.value < 5.0,
                    "Good trial has value {}, expected < 5.0",
                    trial.value
                );
            }

            // Bad trials should have higher values (5..20)
            for trial in &bad {
                assert!(
                    trial.value >= 5.0,
                    "Bad trial has value {}, expected >= 5.0",
                    trial.value
                );
            }
        }

        #[test]
        fn test_split_trials_custom_gamma() {
            // Create sampler with gamma = 0.10
            let sampler = MultivariateTpeSampler::builder()
                .gamma(0.10)
                .build()
                .unwrap();

            // Create 20 trials with values 0..20
            let trial_data: Vec<CompletedTrial> =
                (0..20).map(|i| create_trial(i, i as f64)).collect();
            let trial_refs: Vec<&CompletedTrial> = trial_data.iter().collect();

            let (good, bad) = sampler.split_trials(&trial_refs);

            // With gamma=0.10 and 20 trials: ceil(20 * 0.10) = 2 good
            assert_eq!(good.len(), 2);
            assert_eq!(bad.len(), 18);

            // Good trials should have lowest values (0, 1)
            for trial in &good {
                assert!(
                    trial.value < 2.0,
                    "Good trial has value {}, expected < 2.0",
                    trial.value
                );
            }
        }

        #[test]
        fn test_split_trials_unsorted_input() {
            let sampler = MultivariateTpeSampler::new();

            // Create trials in non-sorted order
            let trial_data = [
                create_trial(0, 5.0),
                create_trial(1, 1.0),
                create_trial(2, 8.0),
                create_trial(3, 0.5),
                create_trial(4, 3.0),
            ];
            let trials: Vec<&CompletedTrial> = trial_data.iter().collect();

            let (good, bad) = sampler.split_trials(&trials);

            // With 5 trials and gamma=0.25: ceil(5*0.25)=2 good, 3 bad
            assert_eq!(good.len(), 2);
            assert_eq!(bad.len(), 3);

            // Good should contain trials 3 (0.5) and 1 (1.0) - lowest values
            let good_ids: Vec<u64> = good.iter().map(|t| t.id).collect();
            assert!(
                good_ids.contains(&3),
                "Trial 3 (value=0.5) should be in good"
            );
            assert!(
                good_ids.contains(&1),
                "Trial 1 (value=1.0) should be in good"
            );

            // Bad should contain trials 0 (5.0), 2 (8.0), 4 (3.0)
            let bad_ids: Vec<u64> = bad.iter().map(|t| t.id).collect();
            assert!(bad_ids.contains(&0), "Trial 0 (value=5.0) should be in bad");
            assert!(bad_ids.contains(&2), "Trial 2 (value=8.0) should be in bad");
            assert!(bad_ids.contains(&4), "Trial 4 (value=3.0) should be in bad");
        }

        #[test]
        fn test_split_trials_with_ties() {
            let sampler = MultivariateTpeSampler::new();

            // Create trials with tied values
            let trial_data = [
                create_trial(0, 1.0),
                create_trial(1, 1.0),
                create_trial(2, 2.0),
                create_trial(3, 2.0),
            ];
            let trials: Vec<&CompletedTrial> = trial_data.iter().collect();

            let (good, bad) = sampler.split_trials(&trials);

            // With 4 trials and gamma=0.25: ceil(4*0.25)=1 good, 3 bad
            assert_eq!(good.len(), 1);
            assert_eq!(bad.len(), 3);

            // Good should contain one of the trials with value 1.0
            assert!(
                (good[0].value - 1.0).abs() < f64::EPSILON,
                "Good trial should have value 1.0"
            );
        }

        #[test]
        fn test_split_trials_ensures_both_groups_nonempty() {
            // Even with extreme gamma values, we should have at least 1 in each group
            // when there are at least 2 trials

            // Very small gamma (would give 0 good without clamping)
            let sampler = MultivariateTpeSampler::builder()
                .gamma(0.01)
                .build()
                .unwrap();

            let trial_data = [create_trial(0, 0.5), create_trial(1, 1.0)];
            let trials: Vec<&CompletedTrial> = trial_data.iter().collect();

            let (good, bad) = sampler.split_trials(&trials);

            // Should still have 1 in each group
            assert_eq!(good.len(), 1);
            assert_eq!(bad.len(), 1);
        }

        #[test]
        fn test_split_trials_large_gamma() {
            // Large gamma (would give all good without clamping)
            let sampler = MultivariateTpeSampler::builder()
                .gamma(0.99)
                .build()
                .unwrap();

            let trial_data: Vec<CompletedTrial> =
                (0..10).map(|i| create_trial(i, i as f64)).collect();
            let trial_refs: Vec<&CompletedTrial> = trial_data.iter().collect();

            let (good, bad) = sampler.split_trials(&trial_refs);

            // With gamma=0.99: ceil(10*0.99)=10, but min ensures at least 1 in bad
            // Actually: n_good = min(10, 10-1) = 9
            assert_eq!(good.len(), 9);
            assert_eq!(bad.len(), 1);

            // Bad should contain the trial with highest value
            assert!(
                (bad[0].value - 9.0).abs() < f64::EPSILON,
                "Bad trial should have highest value"
            );
        }

        #[test]
        fn test_split_trials_nan_handling() {
            let sampler = MultivariateTpeSampler::new();

            // Create trials with NaN values - they should be sorted consistently
            let trial_data = [
                create_trial(0, 1.0),
                create_trial(1, f64::NAN),
                create_trial(2, 0.5),
                create_trial(3, 2.0),
            ];
            let trials: Vec<&CompletedTrial> = trial_data.iter().collect();

            let (good, bad) = sampler.split_trials(&trials);

            // Should complete without panic
            // Total should equal input
            assert_eq!(good.len() + bad.len(), 4);
        }

        #[test]
        fn test_split_trials_preserves_trial_references() {
            let sampler = MultivariateTpeSampler::new();

            let trial_data: Vec<CompletedTrial> =
                (0..5).map(|i| create_trial(i, i as f64)).collect();
            let trial_refs: Vec<&CompletedTrial> = trial_data.iter().collect();

            let (good, bad) = sampler.split_trials(&trial_refs);

            // Verify that references point to the original data
            for trial in good.iter().chain(bad.iter()) {
                // Find the original trial by ID
                let original = trial_data.iter().find(|t| t.id == trial.id).unwrap();
                assert!(
                    core::ptr::eq(*trial, original),
                    "Reference should point to original trial"
                );
            }
        }

        #[test]
        fn test_split_trials_integration_with_filter() {
            // Test that split_trials works correctly with filter_trials output
            let sampler = MultivariateTpeSampler::new();
            let x_id = ParamId::new();
            let y_id = ParamId::new();

            let float_dist = Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            });

            // Create trials with varying parameters
            let mut trials_vec = vec![];
            for i in 0..10 {
                let mut params = HashMap::new();
                params.insert(x_id, ParamValue::Float(i as f64 / 10.0));
                params.insert(y_id, ParamValue::Float(i as f64 / 10.0));
                let mut distributions = HashMap::new();
                distributions.insert(x_id, float_dist.clone());
                distributions.insert(y_id, float_dist.clone());
                trials_vec.push(CompletedTrial::new(
                    i,
                    params,
                    distributions,
                    HashMap::new(),
                    i as f64,
                ));
            }

            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist.clone());
            search_space.insert(y_id, float_dist);

            // Filter trials
            let filtered = sampler.filter_trials(&trials_vec, &search_space);
            assert_eq!(filtered.len(), 10);

            // Split filtered trials
            let (good, bad) = sampler.split_trials(&filtered);

            // With 10 trials and gamma=0.25: ceil(10*0.25)=3 good
            assert_eq!(good.len(), 3);
            assert_eq!(bad.len(), 7);

            // Verify good trials have lowest values
            for trial in &good {
                assert!(trial.value < 3.0);
            }
        }
    }

    // ========================================================================
    // extract_observations Tests
    // ========================================================================

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]
    mod extract_observations_tests {
        use std::collections::HashMap;

        use super::*;
        use crate::distribution::{CategoricalDistribution, FloatDistribution, IntDistribution};
        use crate::param::ParamValue;
        use crate::parameter::ParamId;
        use crate::sampler::CompletedTrial;

        fn float_dist() -> Distribution {
            Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            })
        }

        fn int_dist() -> Distribution {
            Distribution::Int(IntDistribution {
                low: 1,
                high: 100,
                log_scale: false,
                step: None,
            })
        }

        fn categorical_dist(n: usize) -> Distribution {
            Distribution::Categorical(CategoricalDistribution { n_choices: n })
        }

        fn create_trial(
            id: u64,
            params: Vec<(ParamId, ParamValue, Distribution)>,
            value: f64,
        ) -> CompletedTrial {
            let mut param_map = HashMap::new();
            let mut dist_map = HashMap::new();
            for (param_id, pv, dist) in params {
                param_map.insert(param_id, pv);
                dist_map.insert(param_id, dist);
            }
            CompletedTrial::new(id, param_map, dist_map, HashMap::new(), value)
        }

        #[test]
        fn test_extract_observations_empty_trials() {
            let sampler = MultivariateTpeSampler::new();
            let trials: Vec<&CompletedTrial> = vec![];
            let param_order = vec![ParamId::new(), ParamId::new()];

            let observations = sampler.extract_observations(&trials, &param_order);

            assert!(observations.is_empty());
        }

        #[test]
        fn test_extract_observations_empty_param_order() {
            let sampler = MultivariateTpeSampler::new();
            let x_id = ParamId::new();
            let trial = create_trial(0, vec![(x_id, ParamValue::Float(0.5), float_dist())], 1.0);
            let trials: Vec<&CompletedTrial> = vec![&trial];
            let param_order: Vec<ParamId> = vec![];

            let observations = sampler.extract_observations(&trials, &param_order);

            // Should have one row (for the trial) with zero columns
            assert_eq!(observations.len(), 1);
            assert!(observations[0].is_empty());
        }

        #[test]
        fn test_extract_observations_single_float_param() {
            let sampler = MultivariateTpeSampler::new();
            let x_id = ParamId::new();
            let trial_data = [
                create_trial(0, vec![(x_id, ParamValue::Float(0.1), float_dist())], 1.0),
                create_trial(1, vec![(x_id, ParamValue::Float(0.5), float_dist())], 0.5),
                create_trial(2, vec![(x_id, ParamValue::Float(0.9), float_dist())], 0.8),
            ];
            let trials: Vec<&CompletedTrial> = trial_data.iter().collect();
            let param_order = vec![x_id];

            let observations = sampler.extract_observations(&trials, &param_order);

            assert_eq!(observations.len(), 3);
            assert_eq!(observations[0].len(), 1);
            assert!((observations[0][0] - 0.1).abs() < f64::EPSILON);
            assert!((observations[1][0] - 0.5).abs() < f64::EPSILON);
            assert!((observations[2][0] - 0.9).abs() < f64::EPSILON);
        }

        #[test]
        fn test_extract_observations_multiple_float_params() {
            let sampler = MultivariateTpeSampler::new();
            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let trial_data = [
                create_trial(
                    0,
                    vec![
                        (x_id, ParamValue::Float(0.1), float_dist()),
                        (y_id, ParamValue::Float(0.2), float_dist()),
                    ],
                    1.0,
                ),
                create_trial(
                    1,
                    vec![
                        (x_id, ParamValue::Float(0.3), float_dist()),
                        (y_id, ParamValue::Float(0.4), float_dist()),
                    ],
                    0.5,
                ),
            ];
            let trials: Vec<&CompletedTrial> = trial_data.iter().collect();
            let param_order = vec![x_id, y_id];

            let observations = sampler.extract_observations(&trials, &param_order);

            assert_eq!(observations.len(), 2);
            assert_eq!(observations[0].len(), 2);
            assert!((observations[0][0] - 0.1).abs() < f64::EPSILON);
            assert!((observations[0][1] - 0.2).abs() < f64::EPSILON);
            assert!((observations[1][0] - 0.3).abs() < f64::EPSILON);
            assert!((observations[1][1] - 0.4).abs() < f64::EPSILON);
        }

        #[test]
        fn test_extract_observations_respects_param_order() {
            let sampler = MultivariateTpeSampler::new();
            let a_id = ParamId::new();
            let b_id = ParamId::new();
            let c_id = ParamId::new();
            let trial = create_trial(
                0,
                vec![
                    (a_id, ParamValue::Float(1.0), float_dist()),
                    (b_id, ParamValue::Float(2.0), float_dist()),
                    (c_id, ParamValue::Float(3.0), float_dist()),
                ],
                1.0,
            );
            let trials: Vec<&CompletedTrial> = vec![&trial];

            // Different orderings
            let order_abc = vec![a_id, b_id, c_id];
            let order_cba = vec![c_id, b_id, a_id];
            let order_bac = vec![b_id, a_id, c_id];

            let obs_abc = sampler.extract_observations(&trials, &order_abc);
            let obs_cba = sampler.extract_observations(&trials, &order_cba);
            let obs_bac = sampler.extract_observations(&trials, &order_bac);

            assert!((obs_abc[0][0] - 1.0).abs() < f64::EPSILON);
            assert!((obs_abc[0][1] - 2.0).abs() < f64::EPSILON);
            assert!((obs_abc[0][2] - 3.0).abs() < f64::EPSILON);

            assert!((obs_cba[0][0] - 3.0).abs() < f64::EPSILON);
            assert!((obs_cba[0][1] - 2.0).abs() < f64::EPSILON);
            assert!((obs_cba[0][2] - 1.0).abs() < f64::EPSILON);

            assert!((obs_bac[0][0] - 2.0).abs() < f64::EPSILON);
            assert!((obs_bac[0][1] - 1.0).abs() < f64::EPSILON);
            assert!((obs_bac[0][2] - 3.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_extract_observations_int_conversion() {
            let sampler = MultivariateTpeSampler::new();
            let n_layers_id = ParamId::new();
            let trial_data = [
                create_trial(0, vec![(n_layers_id, ParamValue::Int(3), int_dist())], 1.0),
                create_trial(1, vec![(n_layers_id, ParamValue::Int(5), int_dist())], 0.5),
                create_trial(2, vec![(n_layers_id, ParamValue::Int(10), int_dist())], 0.8),
            ];
            let trials: Vec<&CompletedTrial> = trial_data.iter().collect();
            let param_order = vec![n_layers_id];

            let observations = sampler.extract_observations(&trials, &param_order);

            assert_eq!(observations.len(), 3);
            assert!((observations[0][0] - 3.0).abs() < f64::EPSILON);
            assert!((observations[1][0] - 5.0).abs() < f64::EPSILON);
            assert!((observations[2][0] - 10.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_extract_observations_mixed_float_and_int() {
            let sampler = MultivariateTpeSampler::new();
            let lr_id = ParamId::new();
            let n_layers_id = ParamId::new();
            let batch_size_id = ParamId::new();
            let trial_data = [
                create_trial(
                    0,
                    vec![
                        (lr_id, ParamValue::Float(0.01), float_dist()),
                        (n_layers_id, ParamValue::Int(3), int_dist()),
                        (batch_size_id, ParamValue::Int(32), int_dist()),
                    ],
                    1.0,
                ),
                create_trial(
                    1,
                    vec![
                        (lr_id, ParamValue::Float(0.001), float_dist()),
                        (n_layers_id, ParamValue::Int(5), int_dist()),
                        (batch_size_id, ParamValue::Int(64), int_dist()),
                    ],
                    0.5,
                ),
            ];
            let trials: Vec<&CompletedTrial> = trial_data.iter().collect();
            let param_order = vec![lr_id, n_layers_id, batch_size_id];

            let observations = sampler.extract_observations(&trials, &param_order);

            assert_eq!(observations.len(), 2);
            assert_eq!(observations[0].len(), 3);

            assert!((observations[0][0] - 0.01).abs() < f64::EPSILON);
            assert!((observations[0][1] - 3.0).abs() < f64::EPSILON);
            assert!((observations[0][2] - 32.0).abs() < f64::EPSILON);

            assert!((observations[1][0] - 0.001).abs() < f64::EPSILON);
            assert!((observations[1][1] - 5.0).abs() < f64::EPSILON);
            assert!((observations[1][2] - 64.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_extract_observations_skips_categorical() {
            let sampler = MultivariateTpeSampler::new();
            let x_id = ParamId::new();
            let optimizer_id = ParamId::new();
            let y_id = ParamId::new();
            let trial_data = [
                create_trial(
                    0,
                    vec![
                        (x_id, ParamValue::Float(0.5), float_dist()),
                        (
                            optimizer_id,
                            ParamValue::Categorical(1),
                            categorical_dist(3),
                        ),
                        (y_id, ParamValue::Float(0.3), float_dist()),
                    ],
                    1.0,
                ),
                create_trial(
                    1,
                    vec![
                        (x_id, ParamValue::Float(0.7), float_dist()),
                        (
                            optimizer_id,
                            ParamValue::Categorical(0),
                            categorical_dist(3),
                        ),
                        (y_id, ParamValue::Float(0.2), float_dist()),
                    ],
                    0.5,
                ),
            ];
            let trials: Vec<&CompletedTrial> = trial_data.iter().collect();

            // Include categorical in param order - it should be skipped
            let param_order = vec![x_id, optimizer_id, y_id];

            let observations = sampler.extract_observations(&trials, &param_order);

            // Each observation should have only 2 values (x and y), not 3
            assert_eq!(observations.len(), 2);
            assert_eq!(observations[0].len(), 2);
            assert_eq!(observations[1].len(), 2);

            // First row: x=0.5, y=0.3
            assert!((observations[0][0] - 0.5).abs() < f64::EPSILON);
            assert!((observations[0][1] - 0.3).abs() < f64::EPSILON);

            // Second row: x=0.7, y=0.2
            assert!((observations[1][0] - 0.7).abs() < f64::EPSILON);
            assert!((observations[1][1] - 0.2).abs() < f64::EPSILON);
        }

        #[test]
        fn test_extract_observations_only_categorical_in_order() {
            let sampler = MultivariateTpeSampler::new();
            let x_id = ParamId::new();
            let optimizer_id = ParamId::new();
            let trial = create_trial(
                0,
                vec![
                    (x_id, ParamValue::Float(0.5), float_dist()),
                    (
                        optimizer_id,
                        ParamValue::Categorical(1),
                        categorical_dist(3),
                    ),
                ],
                1.0,
            );
            let trials: Vec<&CompletedTrial> = vec![&trial];

            // Only request categorical param
            let param_order = vec![optimizer_id];

            let observations = sampler.extract_observations(&trials, &param_order);

            // Should have one row with zero columns (categorical skipped)
            assert_eq!(observations.len(), 1);
            assert!(observations[0].is_empty());
        }

        #[test]
        fn test_extract_observations_missing_param() {
            let sampler = MultivariateTpeSampler::new();
            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let trial_data = [
                create_trial(
                    0,
                    vec![
                        (x_id, ParamValue::Float(0.5), float_dist()),
                        (y_id, ParamValue::Float(0.3), float_dist()),
                    ],
                    1.0,
                ),
                // This trial is missing y
                create_trial(1, vec![(x_id, ParamValue::Float(0.7), float_dist())], 0.5),
            ];
            let trials: Vec<&CompletedTrial> = trial_data.iter().collect();
            let param_order = vec![x_id, y_id];

            let observations = sampler.extract_observations(&trials, &param_order);

            // First trial should have both params
            assert_eq!(observations[0].len(), 2);
            // Second trial is missing y, so it only has x
            assert_eq!(observations[1].len(), 1);
        }

        #[test]
        fn test_extract_observations_large_int_values() {
            let sampler = MultivariateTpeSampler::new();
            let small_int_id = ParamId::new();
            let medium_int_id = ParamId::new();
            let negative_int_id = ParamId::new();
            // Test with large integer values to verify precision
            let trial = create_trial(
                0,
                vec![
                    (small_int_id, ParamValue::Int(1), int_dist()),
                    (medium_int_id, ParamValue::Int(1_000_000), int_dist()),
                    (negative_int_id, ParamValue::Int(-42), int_dist()),
                ],
                1.0,
            );
            let trials: Vec<&CompletedTrial> = vec![&trial];
            let param_order = vec![small_int_id, medium_int_id, negative_int_id];

            let observations = sampler.extract_observations(&trials, &param_order);

            assert_eq!(observations.len(), 1);
            assert_eq!(observations[0].len(), 3);
            assert!((observations[0][0] - 1.0).abs() < f64::EPSILON);
            assert!((observations[0][1] - 1_000_000.0).abs() < f64::EPSILON);
            assert!((observations[0][2] - (-42.0)).abs() < f64::EPSILON);
        }

        #[test]
        fn test_extract_observations_many_trials() {
            let sampler = MultivariateTpeSampler::new();
            let x_id = ParamId::new();
            let y_id = ParamId::new();

            // Create 100 trials with predictable values
            let trial_data: Vec<CompletedTrial> = (0..100)
                .map(|i| {
                    create_trial(
                        i,
                        vec![
                            (x_id, ParamValue::Float(i as f64 / 100.0), float_dist()),
                            (y_id, ParamValue::Int(i as i64), int_dist()),
                        ],
                        i as f64,
                    )
                })
                .collect();
            let trials: Vec<&CompletedTrial> = trial_data.iter().collect();
            let param_order = vec![x_id, y_id];

            let observations = sampler.extract_observations(&trials, &param_order);

            assert_eq!(observations.len(), 100);
            for (i, obs) in observations.iter().enumerate() {
                assert_eq!(obs.len(), 2);
                assert!((obs[0] - i as f64 / 100.0).abs() < f64::EPSILON);
                assert!((obs[1] - i as f64).abs() < f64::EPSILON);
            }
        }

        #[test]
        fn test_extract_observations_subset_of_params() {
            let sampler = MultivariateTpeSampler::new();
            let a_id = ParamId::new();
            let b_id = ParamId::new();
            let c_id = ParamId::new();
            let d_id = ParamId::new();
            let trial = create_trial(
                0,
                vec![
                    (a_id, ParamValue::Float(1.0), float_dist()),
                    (b_id, ParamValue::Float(2.0), float_dist()),
                    (c_id, ParamValue::Float(3.0), float_dist()),
                    (d_id, ParamValue::Float(4.0), float_dist()),
                ],
                1.0,
            );
            let trials: Vec<&CompletedTrial> = vec![&trial];

            // Only extract a subset of params
            let param_order = vec![b_id, d_id];

            let observations = sampler.extract_observations(&trials, &param_order);

            assert_eq!(observations.len(), 1);
            assert_eq!(observations[0].len(), 2);
            assert!((observations[0][0] - 2.0).abs() < f64::EPSILON); // b
            assert!((observations[0][1] - 4.0).abs() < f64::EPSILON); // d
        }

        #[test]
        fn test_extract_observations_integration_with_pipeline() {
            // Test full pipeline: filter -> split -> extract
            let sampler = MultivariateTpeSampler::new();

            let x_id = ParamId::new();
            let n_id = ParamId::new();
            let float_dist_val = float_dist();
            let int_dist_val = int_dist();

            let trial_data: Vec<CompletedTrial> = (0..20)
                .map(|i| {
                    let mut params = HashMap::new();
                    params.insert(x_id, ParamValue::Float(i as f64 / 20.0));
                    params.insert(n_id, ParamValue::Int(i as i64));
                    let mut distributions = HashMap::new();
                    distributions.insert(x_id, float_dist_val.clone());
                    distributions.insert(n_id, int_dist_val.clone());
                    CompletedTrial::new(i, params, distributions, HashMap::new(), i as f64)
                })
                .collect();

            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist_val);
            search_space.insert(n_id, int_dist_val);

            // Filter
            let filtered = sampler.filter_trials(&trial_data, &search_space);
            assert_eq!(filtered.len(), 20);

            // Split
            let (good, bad) = sampler.split_trials(&filtered);
            assert_eq!(good.len(), 5); // gamma=0.25, ceil(20*0.25)=5
            assert_eq!(bad.len(), 15);

            // Extract from good trials
            let param_order = vec![x_id, n_id];
            let good_obs = sampler.extract_observations(&good, &param_order);
            let bad_obs = sampler.extract_observations(&bad, &param_order);

            assert_eq!(good_obs.len(), 5);
            assert_eq!(bad_obs.len(), 15);

            // Good observations should all have low values (trials 0-4)
            for obs in &good_obs {
                assert_eq!(obs.len(), 2);
                assert!(obs[0] < 0.25); // x < 5/20 = 0.25
                assert!(obs[1] < 5.0); // n < 5
            }

            // Bad observations should have higher values
            for obs in &bad_obs {
                assert_eq!(obs.len(), 2);
                assert!(obs[0] >= 0.25 || obs[1] >= 5.0);
            }
        }
    }

    // ========================================================================
    // select_candidate Tests
    // ========================================================================

    mod select_candidate_tests {
        use super::*;
        use crate::kde::MultivariateKDE;

        #[test]
        fn test_select_candidate_basic() {
            // Create a sampler with a fixed seed for reproducibility
            let sampler = MultivariateTpeSampler::builder()
                .n_ei_candidates(24)
                .seed(42)
                .build()
                .unwrap();

            // Good samples clustered around (0.1, 0.1)
            let good_samples = vec![
                vec![0.1, 0.1],
                vec![0.15, 0.12],
                vec![0.08, 0.11],
                vec![0.12, 0.09],
                vec![0.11, 0.13],
            ];

            // Bad samples clustered around (0.9, 0.9)
            let bad_samples = vec![
                vec![0.9, 0.9],
                vec![0.85, 0.88],
                vec![0.92, 0.91],
                vec![0.88, 0.93],
                vec![0.91, 0.87],
            ];

            let good_kde = MultivariateKDE::new(good_samples).unwrap();
            let bad_kde = MultivariateKDE::new(bad_samples).unwrap();

            let selected = sampler.select_candidate(&good_kde, &bad_kde);

            // The selected candidate should have 2 dimensions
            assert_eq!(selected.len(), 2);

            // The selected point should be closer to the good region than the bad region
            // (though not always perfectly so due to stochasticity)
            let dist_to_good = ((selected[0] - 0.1).powi(2) + (selected[1] - 0.1).powi(2)).sqrt();
            let dist_to_bad = ((selected[0] - 0.9).powi(2) + (selected[1] - 0.9).powi(2)).sqrt();

            assert!(
                dist_to_good < dist_to_bad,
                "Selected point ({}, {}) is closer to bad region than good region",
                selected[0],
                selected[1]
            );
        }

        #[test]
        fn test_select_candidate_returns_correct_dimension() {
            let sampler = MultivariateTpeSampler::builder()
                .n_ei_candidates(10)
                .seed(123)
                .build()
                .unwrap();

            // 3D case
            let good_samples = vec![
                vec![0.1, 0.2, 0.3],
                vec![0.15, 0.25, 0.35],
                vec![0.12, 0.22, 0.32],
            ];
            let bad_samples = vec![
                vec![0.8, 0.7, 0.6],
                vec![0.85, 0.75, 0.65],
                vec![0.82, 0.72, 0.62],
            ];

            let good_kde = MultivariateKDE::new(good_samples).unwrap();
            let bad_kde = MultivariateKDE::new(bad_samples).unwrap();

            let selected = sampler.select_candidate(&good_kde, &bad_kde);
            assert_eq!(selected.len(), 3);
        }

        #[test]
        fn test_select_candidate_one_dimension() {
            let sampler = MultivariateTpeSampler::builder()
                .n_ei_candidates(20)
                .seed(456)
                .build()
                .unwrap();

            // 1D case - good samples near 0, bad samples near 10
            let good_samples = vec![vec![0.0], vec![0.5], vec![1.0], vec![0.3], vec![0.7]];
            let bad_samples = vec![vec![8.0], vec![9.0], vec![10.0], vec![8.5], vec![9.5]];

            let good_kde = MultivariateKDE::new(good_samples).unwrap();
            let bad_kde = MultivariateKDE::new(bad_samples).unwrap();

            let selected = sampler.select_candidate(&good_kde, &bad_kde);
            assert_eq!(selected.len(), 1);

            // Selected value should be closer to the good region
            assert!(
                selected[0] < 5.0,
                "Selected value {} should be closer to good region (< 5.0)",
                selected[0]
            );
        }

        #[test]
        fn test_select_candidate_reproducibility() {
            // Two samplers with the same seed should produce the same results
            let sampler1 = MultivariateTpeSampler::builder()
                .n_ei_candidates(24)
                .seed(999)
                .build()
                .unwrap();

            let sampler2 = MultivariateTpeSampler::builder()
                .n_ei_candidates(24)
                .seed(999)
                .build()
                .unwrap();

            let good_samples = vec![vec![1.0, 2.0], vec![1.5, 2.5], vec![1.2, 2.2]];
            let bad_samples = vec![vec![8.0, 9.0], vec![8.5, 9.5], vec![8.2, 9.2]];

            let good_kde = MultivariateKDE::new(good_samples.clone()).unwrap();
            let bad_kde = MultivariateKDE::new(bad_samples.clone()).unwrap();

            let selected1 = sampler1.select_candidate(&good_kde, &bad_kde);

            // Need to recreate KDEs for second sampler since we consumed them
            let good_kde2 = MultivariateKDE::new(good_samples).unwrap();
            let bad_kde2 = MultivariateKDE::new(bad_samples).unwrap();

            let selected2 = sampler2.select_candidate(&good_kde2, &bad_kde2);

            // With same seed, should get same result
            assert!(
                (selected1[0] - selected2[0]).abs() < f64::EPSILON,
                "Dimension 0: {} vs {}",
                selected1[0],
                selected2[0]
            );
            assert!(
                (selected1[1] - selected2[1]).abs() < f64::EPSILON,
                "Dimension 1: {} vs {}",
                selected1[1],
                selected2[1]
            );
        }

        #[test]
        fn test_select_candidate_with_single_candidate() {
            // With n_ei_candidates=1, should still work
            let sampler = MultivariateTpeSampler::builder()
                .n_ei_candidates(1)
                .seed(789)
                .build()
                .unwrap();

            let good_samples = vec![vec![0.0, 0.0], vec![0.1, 0.1]];
            let bad_samples = vec![vec![5.0, 5.0], vec![5.1, 5.1]];

            let good_kde = MultivariateKDE::new(good_samples).unwrap();
            let bad_kde = MultivariateKDE::new(bad_samples).unwrap();

            let selected = sampler.select_candidate(&good_kde, &bad_kde);
            assert_eq!(selected.len(), 2);
        }

        #[test]
        fn test_select_candidate_many_candidates() {
            // With many candidates, should find a good point
            let sampler = MultivariateTpeSampler::builder()
                .n_ei_candidates(100)
                .seed(111)
                .build()
                .unwrap();

            let good_samples = vec![
                vec![0.0, 0.0],
                vec![0.1, 0.1],
                vec![0.2, 0.2],
                vec![0.05, 0.15],
                vec![0.15, 0.05],
            ];
            let bad_samples = vec![
                vec![10.0, 10.0],
                vec![10.1, 10.1],
                vec![10.2, 10.2],
                vec![10.05, 10.15],
                vec![10.15, 10.05],
            ];

            let good_kde = MultivariateKDE::new(good_samples).unwrap();
            let bad_kde = MultivariateKDE::new(bad_samples).unwrap();

            let selected = sampler.select_candidate(&good_kde, &bad_kde);

            // With more candidates, should definitely find a point in the good region
            assert!(
                selected[0] < 5.0 && selected[1] < 5.0,
                "Selected point ({}, {}) should be in good region",
                selected[0],
                selected[1]
            );
        }

        #[test]
        fn test_select_candidate_overlapping_distributions() {
            // When distributions overlap, selection should still work
            let sampler = MultivariateTpeSampler::builder()
                .n_ei_candidates(24)
                .seed(222)
                .build()
                .unwrap();

            // Overlapping distributions - good centered at 0, bad centered at 1
            let good_samples = vec![
                vec![0.0, 0.0],
                vec![0.5, 0.5],
                vec![-0.5, -0.5],
                vec![0.3, -0.3],
                vec![-0.3, 0.3],
            ];
            let bad_samples = vec![
                vec![1.0, 1.0],
                vec![1.5, 1.5],
                vec![0.5, 0.5], // This overlaps with good
                vec![1.3, 0.7],
                vec![0.7, 1.3],
            ];

            let good_kde = MultivariateKDE::new(good_samples).unwrap();
            let bad_kde = MultivariateKDE::new(bad_samples).unwrap();

            let selected = sampler.select_candidate(&good_kde, &bad_kde);

            // Should still return a valid point
            assert_eq!(selected.len(), 2);
            assert!(selected[0].is_finite());
            assert!(selected[1].is_finite());
        }

        #[test]
        fn test_select_candidate_high_dimensional() {
            // Test with higher dimensions (5D)
            let sampler = MultivariateTpeSampler::builder()
                .n_ei_candidates(50)
                .seed(333)
                .build()
                .unwrap();

            // Good samples near origin
            let good_samples: Vec<Vec<f64>> = (0..10)
                .map(|i| {
                    let offset = f64::from(i) * 0.01;
                    vec![offset, offset, offset, offset, offset]
                })
                .collect();

            // Bad samples far from origin
            let bad_samples: Vec<Vec<f64>> = (0..10)
                .map(|i| {
                    let offset = 10.0 + f64::from(i) * 0.01;
                    vec![offset, offset, offset, offset, offset]
                })
                .collect();

            let good_kde = MultivariateKDE::new(good_samples).unwrap();
            let bad_kde = MultivariateKDE::new(bad_samples).unwrap();

            let selected = sampler.select_candidate(&good_kde, &bad_kde);

            assert_eq!(selected.len(), 5);

            // All dimensions should be closer to 0 than to 10
            for (dim, &val) in selected.iter().enumerate() {
                assert!(
                    val.abs() < 5.0,
                    "Dimension {dim} value {val} should be closer to origin"
                );
            }
        }

        #[test]
        fn test_select_candidate_integration_with_pipeline() {
            // Full integration test: extract observations -> fit KDEs -> select candidate
            use crate::distribution::FloatDistribution;
            use crate::param::ParamValue;
            use crate::parameter::ParamId;

            let sampler = MultivariateTpeSampler::builder()
                .gamma(0.25)
                .n_ei_candidates(24)
                .seed(444)
                .build()
                .unwrap();

            let float_dist = Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 10.0,
                log_scale: false,
                step: None,
            });

            let x_id = ParamId::new();
            let y_id = ParamId::new();

            // Create trials with objective values equal to x + y
            // So good trials have low x and y values
            let trial_data: Vec<CompletedTrial> = (0..20)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let x = (i as f64) / 2.0;
                    #[allow(clippy::cast_precision_loss)]
                    let y = (i as f64) / 2.0;
                    let mut params = HashMap::new();
                    params.insert(x_id, ParamValue::Float(x));
                    params.insert(y_id, ParamValue::Float(y));
                    let mut distributions = HashMap::new();
                    distributions.insert(x_id, float_dist.clone());
                    distributions.insert(y_id, float_dist.clone());
                    CompletedTrial::new(i, params, distributions, HashMap::new(), x + y)
                })
                .collect();

            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist.clone());
            search_space.insert(y_id, float_dist);

            // Filter and split
            let filtered = sampler.filter_trials(&trial_data, &search_space);
            let (good, bad) = sampler.split_trials(&filtered);

            // Extract observations
            let param_order = vec![x_id, y_id];
            let good_obs = sampler.extract_observations(&good, &param_order);
            let bad_obs = sampler.extract_observations(&bad, &param_order);

            // Fit KDEs
            let good_kde = MultivariateKDE::new(good_obs).unwrap();
            let bad_kde = MultivariateKDE::new(bad_obs).unwrap();

            // Select candidate
            let selected = sampler.select_candidate(&good_kde, &bad_kde);

            assert_eq!(selected.len(), 2);

            // Selected point should be in the "good" region (low x, low y)
            // Good trials have x, y in [0, 2.5), bad trials have x, y in [2.5, 10)
            assert!(
                selected[0] < 5.0 && selected[1] < 5.0,
                "Selected point ({}, {}) should be in good region",
                selected[0],
                selected[1]
            );
        }
    }

    // ========================================================================
    // sample_joint Tests
    // ========================================================================

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_lossless,
        clippy::collapsible_if
    )]
    mod sample_joint_tests {
        use std::collections::HashMap;

        use super::*;
        use crate::distribution::{CategoricalDistribution, FloatDistribution, IntDistribution};
        use crate::param::ParamValue;
        use crate::parameter::ParamId;
        use crate::sampler::CompletedTrial;

        fn float_dist(low: f64, high: f64) -> Distribution {
            Distribution::Float(FloatDistribution {
                low,
                high,
                log_scale: false,
                step: None,
            })
        }

        fn int_dist(low: i64, high: i64) -> Distribution {
            Distribution::Int(IntDistribution {
                low,
                high,
                log_scale: false,
                step: None,
            })
        }

        fn categorical_dist(n: usize) -> Distribution {
            Distribution::Categorical(CategoricalDistribution { n_choices: n })
        }

        fn create_trial(
            id: u64,
            params: Vec<(ParamId, ParamValue, Distribution)>,
            value: f64,
        ) -> CompletedTrial {
            let mut param_map = HashMap::new();
            let mut dist_map = HashMap::new();
            for (param_id, pv, dist) in params {
                param_map.insert(param_id, pv);
                dist_map.insert(param_id, dist);
            }
            CompletedTrial::new(id, param_map, dist_map, HashMap::new(), value)
        }

        #[test]
        fn test_sample_joint_empty_history_returns_all_params() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(10)
                .seed(42)
                .build()
                .unwrap();

            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist(0.0, 1.0));
            search_space.insert(y_id, float_dist(0.0, 1.0));

            let history: Vec<CompletedTrial> = vec![];

            let result = sampler.sample_joint(&search_space, &history);

            assert_eq!(result.len(), 2);
            assert!(result.contains_key(&x_id));
            assert!(result.contains_key(&y_id));
        }

        #[test]
        fn test_sample_joint_startup_phase_uses_random() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(10)
                .seed(42)
                .build()
                .unwrap();

            let x_id = ParamId::new();
            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist(0.0, 1.0));

            // Create 5 trials (less than n_startup_trials=10)
            let history: Vec<CompletedTrial> = (0..5)
                .map(|i| {
                    create_trial(
                        i,
                        vec![(
                            x_id,
                            ParamValue::Float(i as f64 / 10.0),
                            float_dist(0.0, 1.0),
                        )],
                        i as f64,
                    )
                })
                .collect();

            let result = sampler.sample_joint(&search_space, &history);

            assert!(result.contains_key(&x_id));
            if let Some(ParamValue::Float(v)) = result.get(&x_id) {
                assert!(*v >= 0.0 && *v <= 1.0);
            } else {
                panic!("Expected Float value for x");
            }
        }

        #[test]
        fn test_sample_joint_after_startup_uses_tpe() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .n_ei_candidates(24)
                .seed(42)
                .build()
                .unwrap();

            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist(0.0, 10.0));
            search_space.insert(y_id, float_dist(0.0, 10.0));

            // Create 20 trials with good values near origin
            let history: Vec<CompletedTrial> = (0..20)
                .map(|i| {
                    let x = i as f64 / 2.0;
                    let y = i as f64 / 2.0;
                    create_trial(
                        i,
                        vec![
                            (x_id, ParamValue::Float(x), float_dist(0.0, 10.0)),
                            (y_id, ParamValue::Float(y), float_dist(0.0, 10.0)),
                        ],
                        x + y, // Objective: lower is better
                    )
                })
                .collect();

            let result = sampler.sample_joint(&search_space, &history);

            assert_eq!(result.len(), 2);
            assert!(result.contains_key(&x_id));
            assert!(result.contains_key(&y_id));

            // Values should be in the distribution bounds
            if let Some(ParamValue::Float(x)) = result.get(&x_id) {
                assert!(*x >= 0.0 && *x <= 10.0, "x={x} should be within [0, 10]");
            }
            if let Some(ParamValue::Float(y)) = result.get(&y_id) {
                assert!(*y >= 0.0 && *y <= 10.0, "y={y} should be within [0, 10]");
            }
        }

        #[test]
        fn test_sample_joint_biases_toward_good_region() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .n_ei_candidates(50)
                .seed(123)
                .build()
                .unwrap();

            let x_id = ParamId::new();
            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist(0.0, 10.0));

            // Create trials: good ones near 0, bad ones near 10
            let mut history: Vec<CompletedTrial> = vec![];
            // Good trials (low value = good)
            for i in 0..5 {
                history.push(create_trial(
                    i,
                    vec![(
                        x_id,
                        ParamValue::Float(i as f64 * 0.5),
                        float_dist(0.0, 10.0),
                    )],
                    i as f64 * 0.5, // Low objective
                ));
            }
            // Bad trials
            for i in 5..15 {
                history.push(create_trial(
                    i,
                    vec![(
                        x_id,
                        ParamValue::Float(5.0 + (i as f64 - 5.0) * 0.5),
                        float_dist(0.0, 10.0),
                    )],
                    5.0 + (i as f64 - 5.0) * 0.5, // High objective
                ));
            }

            // Sample multiple times and check bias
            let mut low_count = 0;
            for _ in 0..20 {
                let result = sampler.sample_joint(&search_space, &history);
                if let Some(ParamValue::Float(x)) = result.get(&x_id) {
                    if *x < 5.0 {
                        low_count += 1;
                    }
                }
            }

            // Should be biased toward lower values
            assert!(
                low_count > 10,
                "Expected more samples in good region, got {low_count}/20"
            );
        }

        #[test]
        fn test_sample_joint_with_int_params() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .seed(42)
                .build()
                .unwrap();

            let n_layers_id = ParamId::new();
            let mut search_space = HashMap::new();
            search_space.insert(n_layers_id, int_dist(1, 10));

            // Create history
            let history: Vec<CompletedTrial> = (0..10)
                .map(|i| {
                    create_trial(
                        i,
                        vec![(n_layers_id, ParamValue::Int(i as i64 + 1), int_dist(1, 10))],
                        i as f64,
                    )
                })
                .collect();

            let result = sampler.sample_joint(&search_space, &history);

            assert!(result.contains_key(&n_layers_id));
            if let Some(ParamValue::Int(v)) = result.get(&n_layers_id) {
                assert!(*v >= 1 && *v <= 10, "n_layers={v} should be within [1, 10]");
            } else {
                panic!("Expected Int value for n_layers");
            }
        }

        #[test]
        fn test_sample_joint_with_mixed_params() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .seed(42)
                .build()
                .unwrap();

            let lr_id = ParamId::new();
            let n_layers_id = ParamId::new();
            let mut search_space = HashMap::new();
            search_space.insert(lr_id, float_dist(0.0, 1.0));
            search_space.insert(n_layers_id, int_dist(1, 5));

            // Create history with both params
            let history: Vec<CompletedTrial> = (0..15)
                .map(|i| {
                    create_trial(
                        i,
                        vec![
                            (
                                lr_id,
                                ParamValue::Float(i as f64 / 15.0),
                                float_dist(0.0, 1.0),
                            ),
                            (
                                n_layers_id,
                                ParamValue::Int((i % 5) as i64 + 1),
                                int_dist(1, 5),
                            ),
                        ],
                        i as f64,
                    )
                })
                .collect();

            let result = sampler.sample_joint(&search_space, &history);

            assert_eq!(result.len(), 2);
            assert!(result.contains_key(&lr_id));
            assert!(result.contains_key(&n_layers_id));
        }

        #[test]
        fn test_sample_joint_with_categorical_params() {
            // Categorical params are currently sampled independently
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .seed(42)
                .build()
                .unwrap();

            let x_id = ParamId::new();
            let optimizer_id = ParamId::new();
            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist(0.0, 1.0));
            search_space.insert(optimizer_id, categorical_dist(3));

            let history: Vec<CompletedTrial> = (0..15)
                .map(|i| {
                    create_trial(
                        i,
                        vec![
                            (
                                x_id,
                                ParamValue::Float(i as f64 / 15.0),
                                float_dist(0.0, 1.0),
                            ),
                            (
                                optimizer_id,
                                ParamValue::Categorical(i as usize % 3),
                                categorical_dist(3),
                            ),
                        ],
                        i as f64,
                    )
                })
                .collect();

            let result = sampler.sample_joint(&search_space, &history);

            assert_eq!(result.len(), 2);
            assert!(result.contains_key(&x_id));
            assert!(result.contains_key(&optimizer_id));

            if let Some(ParamValue::Categorical(v)) = result.get(&optimizer_id) {
                assert!(*v < 3, "optimizer={v} should be in [0, 3)");
            } else {
                panic!("Expected Categorical value for optimizer");
            }
        }

        #[test]
        fn test_sample_joint_reproducibility() {
            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let search_space = {
                let mut s = HashMap::new();
                s.insert(x_id, float_dist(0.0, 1.0));
                s.insert(y_id, float_dist(0.0, 1.0));
                s
            };

            let history: Vec<CompletedTrial> = (0..15)
                .map(|i| {
                    create_trial(
                        i,
                        vec![
                            (
                                x_id,
                                ParamValue::Float(i as f64 / 15.0),
                                float_dist(0.0, 1.0),
                            ),
                            (
                                y_id,
                                ParamValue::Float(i as f64 / 15.0),
                                float_dist(0.0, 1.0),
                            ),
                        ],
                        i as f64,
                    )
                })
                .collect();

            let sampler1 = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .seed(999)
                .build()
                .unwrap();

            let sampler2 = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .seed(999)
                .build()
                .unwrap();

            let result1 = sampler1.sample_joint(&search_space, &history);
            let result2 = sampler2.sample_joint(&search_space, &history);

            // With same seed, should get same results
            assert_eq!(result1, result2);
        }

        #[test]
        fn test_sample_joint_clamps_to_bounds() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(2)
                .seed(42)
                .build()
                .unwrap();

            let x_id = ParamId::new();
            let mut search_space = HashMap::new();
            // Narrow distribution
            search_space.insert(x_id, float_dist(0.0, 0.1));

            // Create trials at the edge
            let history: Vec<CompletedTrial> = (0..10)
                .map(|i| {
                    create_trial(
                        i,
                        vec![(
                            x_id,
                            ParamValue::Float(i as f64 / 100.0),
                            float_dist(0.0, 0.1),
                        )],
                        i as f64,
                    )
                })
                .collect();

            // Sample multiple times
            for _ in 0..10 {
                let result = sampler.sample_joint(&search_space, &history);
                if let Some(ParamValue::Float(x)) = result.get(&x_id) {
                    assert!(
                        *x >= 0.0 && *x <= 0.1,
                        "x={x} should be clamped to [0.0, 0.1]"
                    );
                }
            }
        }

        #[test]
        fn test_sample_joint_handles_empty_intersection() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(2)
                .seed(42)
                .build()
                .unwrap();

            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist(0.0, 1.0));
            search_space.insert(y_id, float_dist(0.0, 1.0));

            // Create trials with non-overlapping parameters
            let history = vec![
                create_trial(
                    0,
                    vec![(x_id, ParamValue::Float(0.5), float_dist(0.0, 1.0))],
                    1.0,
                ),
                create_trial(
                    1,
                    vec![(y_id, ParamValue::Float(0.5), float_dist(0.0, 1.0))],
                    1.0,
                ),
            ];

            // Should fall back to random sampling since no common params
            let result = sampler.sample_joint(&search_space, &history);

            assert_eq!(result.len(), 2);
            assert!(result.contains_key(&x_id));
            assert!(result.contains_key(&y_id));
        }

        #[test]
        fn test_sample_joint_integration_many_dimensions() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(10)
                .n_ei_candidates(24)
                .seed(42)
                .build()
                .unwrap();

            // 5D search space
            let dim_ids: Vec<ParamId> = (0..5).map(|_| ParamId::new()).collect();
            let mut search_space = HashMap::new();
            for &id in &dim_ids {
                search_space.insert(id, float_dist(0.0, 10.0));
            }

            // Create history
            let history: Vec<CompletedTrial> = (0..30)
                .map(|trial_id| {
                    let mut param_map = HashMap::new();
                    let mut dist_map = HashMap::new();
                    for (dim, &id) in dim_ids.iter().enumerate() {
                        let value = (trial_id as f64 + dim as f64) / 3.0;
                        param_map.insert(id, ParamValue::Float(value));
                        dist_map.insert(id, float_dist(0.0, 10.0));
                    }
                    CompletedTrial::new(
                        trial_id,
                        param_map,
                        dist_map,
                        HashMap::new(),
                        trial_id as f64,
                    )
                })
                .collect();

            let result = sampler.sample_joint(&search_space, &history);

            assert_eq!(result.len(), 5);
            for (i, &id) in dim_ids.iter().enumerate() {
                assert!(result.contains_key(&id), "Missing parameter x{i}");
                if let Some(ParamValue::Float(v)) = result.get(&id) {
                    assert!(*v >= 0.0 && *v <= 10.0, "x{i}={v} should be within [0, 10]");
                }
            }
        }

        // ====================================================================
        // Categorical Parameter Tests (US-014)
        // ====================================================================

        #[test]
        fn test_sample_tpe_categorical_basic() {
            let mut rng = fastrand::Rng::with_seed(42);

            // Category 0 is good (appears more in good trials)
            let good_indices = vec![0, 0, 0, 1];
            let bad_indices = vec![1, 1, 2, 2];

            // Sample many times and check bias toward good category
            let mut counts = [0usize; 3];
            for _ in 0..1000 {
                let idx = crate::sampler::tpe::common::sample_tpe_categorical(
                    3,
                    &good_indices,
                    &bad_indices,
                    &mut rng,
                );
                counts[idx] += 1;
            }

            // Category 0 should be sampled most frequently
            assert!(
                counts[0] > counts[1],
                "Category 0 (good) should be sampled more than category 1: {} vs {}",
                counts[0],
                counts[1]
            );
            assert!(
                counts[0] > counts[2],
                "Category 0 (good) should be sampled more than category 2: {} vs {}",
                counts[0],
                counts[2]
            );
        }

        #[test]
        fn test_sample_tpe_categorical_laplace_smoothing() {
            let mut rng = fastrand::Rng::with_seed(42);

            // Category 2 never appears, but should still be sampled due to Laplace smoothing
            let good_indices = vec![0, 0, 1];
            let bad_indices = vec![0, 1, 1];

            let mut sampled_two = false;
            for _ in 0..1000 {
                let idx = crate::sampler::tpe::common::sample_tpe_categorical(
                    3,
                    &good_indices,
                    &bad_indices,
                    &mut rng,
                );
                if idx == 2 {
                    sampled_two = true;
                    break;
                }
            }

            assert!(
                sampled_two,
                "Category 2 should be sampled occasionally due to Laplace smoothing"
            );
        }

        #[test]
        fn test_sample_tpe_categorical_empty_good() {
            let mut rng = fastrand::Rng::with_seed(42);

            // Empty good group - all categories should have equal probability
            let good_indices: Vec<usize> = vec![];
            let bad_indices = vec![0, 1, 2];

            let mut counts = [0usize; 3];
            for _ in 0..1000 {
                let idx = crate::sampler::tpe::common::sample_tpe_categorical(
                    3,
                    &good_indices,
                    &bad_indices,
                    &mut rng,
                );
                counts[idx] += 1;
            }

            // All categories should be sampled (roughly uniformly with Laplace)
            assert!(counts[0] > 0, "Category 0 should be sampled");
            assert!(counts[1] > 0, "Category 1 should be sampled");
            assert!(counts[2] > 0, "Category 2 should be sampled");
        }

        #[test]
        fn test_sample_tpe_categorical_all_indices_valid() {
            let mut rng = fastrand::Rng::with_seed(42);

            let n_choices = 4;
            let good_indices = vec![0, 1, 2, 3];
            let bad_indices = vec![0, 1, 2, 3];

            // All samples should be valid indices
            for _ in 0..100 {
                let idx = crate::sampler::tpe::common::sample_tpe_categorical(
                    n_choices,
                    &good_indices,
                    &bad_indices,
                    &mut rng,
                );
                assert!(idx < n_choices, "Index {idx} should be < {n_choices}");
            }
        }

        #[test]
        fn test_extract_categorical_indices_basic() {
            let cat_id = ParamId::new();
            let trials = [
                create_trial(
                    0,
                    vec![(cat_id, ParamValue::Categorical(1), categorical_dist(3))],
                    0.5,
                ),
                create_trial(
                    1,
                    vec![(cat_id, ParamValue::Categorical(0), categorical_dist(3))],
                    1.0,
                ),
                create_trial(
                    2,
                    vec![(cat_id, ParamValue::Categorical(2), categorical_dist(3))],
                    1.5,
                ),
            ];

            let trial_refs: Vec<&CompletedTrial> = trials.iter().collect();
            let indices = MultivariateTpeSampler::extract_categorical_indices(&trial_refs, cat_id);

            assert_eq!(indices, vec![1, 0, 2]);
        }

        #[test]
        fn test_extract_categorical_indices_missing_param() {
            let cat_id = ParamId::new();
            let other_id = ParamId::new();
            let trials = [
                create_trial(
                    0,
                    vec![(cat_id, ParamValue::Categorical(1), categorical_dist(3))],
                    0.5,
                ),
                create_trial(
                    1,
                    vec![(other_id, ParamValue::Float(1.0), float_dist(0.0, 2.0))],
                    1.0,
                ),
            ];

            let trial_refs: Vec<&CompletedTrial> = trials.iter().collect();
            let indices = MultivariateTpeSampler::extract_categorical_indices(&trial_refs, cat_id);

            // Only the trial with cat should be included
            assert_eq!(indices, vec![1]);
        }

        #[test]
        fn test_extract_categorical_indices_wrong_type() {
            let param_id = ParamId::new();
            let trials = [create_trial(
                0,
                vec![(param_id, ParamValue::Float(1.0), float_dist(0.0, 2.0))],
                0.5,
            )];

            let trial_refs: Vec<&CompletedTrial> = trials.iter().collect();
            let indices =
                MultivariateTpeSampler::extract_categorical_indices(&trial_refs, param_id);

            // Should be empty since param is Float, not Categorical
            assert!(indices.is_empty());
        }

        #[test]
        fn test_sample_joint_categorical_tpe_sampling() {
            let cat_id = ParamId::new();
            let mut search_space = HashMap::new();
            search_space.insert(cat_id, categorical_dist(3));

            // Create history where category 0 is consistently good
            let mut history = Vec::new();
            // Good trials (low objective): category 0
            for i in 0..5 {
                history.push(create_trial(
                    i,
                    vec![(cat_id, ParamValue::Categorical(0), categorical_dist(3))],
                    0.1,
                ));
            }
            // Bad trials (high objective): categories 1 and 2
            for i in 5u64..15 {
                let cat = if i.is_multiple_of(2) { 1 } else { 2 };
                history.push(create_trial(
                    i,
                    vec![(cat_id, ParamValue::Categorical(cat), categorical_dist(3))],
                    10.0,
                ));
            }

            // Sample many times and count
            let mut counts = [0usize; 3];
            for seed in 0..100 {
                let sampler_test = MultivariateTpeSampler::builder()
                    .n_startup_trials(5)
                    .seed(seed)
                    .build()
                    .unwrap();
                let result = sampler_test.sample_joint(&search_space, &history);
                if let Some(ParamValue::Categorical(idx)) = result.get(&cat_id) {
                    counts[*idx] += 1;
                }
            }

            // Category 0 should be sampled most frequently (it's in the good group)
            assert!(
                counts[0] > counts[1] && counts[0] > counts[2],
                "Category 0 (good) should be sampled most: {counts:?}"
            );
        }

        #[test]
        fn test_sample_joint_mixed_continuous_and_categorical() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .seed(42)
                .build()
                .unwrap();

            let x_id = ParamId::new();
            let cat_id = ParamId::new();
            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist(0.0, 1.0));
            search_space.insert(cat_id, categorical_dist(3));

            // Create history with both types
            let mut history = Vec::new();
            // Good trials: low x values, category 0
            for i in 0..5 {
                history.push(create_trial(
                    i,
                    vec![
                        (
                            x_id,
                            ParamValue::Float(f64::from(i as u32) * 0.05),
                            float_dist(0.0, 1.0),
                        ),
                        (cat_id, ParamValue::Categorical(0), categorical_dist(3)),
                    ],
                    f64::from(i as u32) * 0.1,
                ));
            }
            // Bad trials: high x values, categories 1 and 2
            for i in 5u64..15 {
                let cat = if i.is_multiple_of(2) { 1 } else { 2 };
                history.push(create_trial(
                    i,
                    vec![
                        (
                            x_id,
                            ParamValue::Float(0.5 + f64::from(i as u32 - 5) * 0.05),
                            float_dist(0.0, 1.0),
                        ),
                        (cat_id, ParamValue::Categorical(cat), categorical_dist(3)),
                    ],
                    5.0 + f64::from(i as u32),
                ));
            }

            let result = sampler.sample_joint(&search_space, &history);

            // Both parameters should be present
            assert!(result.contains_key(&x_id));
            assert!(result.contains_key(&cat_id));

            // x should be Float
            assert!(
                matches!(result.get(&x_id), Some(ParamValue::Float(_))),
                "x should be Float"
            );

            // cat should be Categorical
            assert!(
                matches!(result.get(&cat_id), Some(ParamValue::Categorical(_))),
                "cat should be Categorical"
            );
        }

        #[test]
        fn test_sample_joint_only_categorical_params() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .seed(42)
                .build()
                .unwrap();

            let cat1_id = ParamId::new();
            let cat2_id = ParamId::new();
            let mut search_space = HashMap::new();
            search_space.insert(cat1_id, categorical_dist(2));
            search_space.insert(cat2_id, categorical_dist(3));

            // Create history with only categorical parameters
            let mut history = Vec::new();
            for i in 0u64..15 {
                let cat1 = usize::from(i >= 5);
                let cat2 = (i % 3) as usize;
                let value = if i < 5 { 0.1 } else { 10.0 };
                history.push(create_trial(
                    i,
                    vec![
                        (cat1_id, ParamValue::Categorical(cat1), categorical_dist(2)),
                        (cat2_id, ParamValue::Categorical(cat2), categorical_dist(3)),
                    ],
                    value,
                ));
            }

            let result = sampler.sample_joint(&search_space, &history);

            assert_eq!(result.len(), 2);
            assert!(matches!(
                result.get(&cat1_id),
                Some(ParamValue::Categorical(_))
            ));
            assert!(matches!(
                result.get(&cat2_id),
                Some(ParamValue::Categorical(_))
            ));
        }
    }

    // ========================================================================
    // Sampler Trait Implementation Tests
    // ========================================================================
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    mod sampler_trait_tests {
        use super::*;
        use crate::parameter::ParamId;

        fn float_dist(low: f64, high: f64) -> Distribution {
            Distribution::Float(crate::distribution::FloatDistribution {
                low,
                high,
                log_scale: false,
                step: None,
            })
        }

        fn int_dist(low: i64, high: i64) -> Distribution {
            Distribution::Int(crate::distribution::IntDistribution {
                low,
                high,
                log_scale: false,
                step: None,
            })
        }

        fn categorical_dist(n_choices: usize) -> Distribution {
            Distribution::Categorical(crate::distribution::CategoricalDistribution { n_choices })
        }

        fn create_trial(
            id: u64,
            params: Vec<(ParamId, ParamValue, Distribution)>,
            value: f64,
        ) -> CompletedTrial {
            let mut param_map = HashMap::new();
            let mut dist_map = HashMap::new();
            for (param_id, param, dist) in params {
                param_map.insert(param_id, param);
                dist_map.insert(param_id, dist);
            }
            CompletedTrial::new(id, param_map, dist_map, HashMap::new(), value)
        }

        #[test]
        fn test_sampler_trait_basic() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .seed(42)
                .build()
                .unwrap();

            let dist = float_dist(0.0, 1.0);
            let history: Vec<CompletedTrial> = vec![];

            // During startup phase, should sample uniformly
            let value = sampler.sample(&dist, 0, &history);
            match value {
                ParamValue::Float(v) => {
                    assert!((0.0..=1.0).contains(&v), "Value {v} should be in [0, 1]");
                }
                _ => panic!("Expected Float value"),
            }
        }

        #[test]
        fn test_sampler_trait_with_history() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .n_ei_candidates(24)
                .seed(42)
                .build()
                .unwrap();

            let dist = float_dist(0.0, 1.0);

            // Create enough history to trigger TPE sampling
            let x_id = ParamId::new();
            let history: Vec<CompletedTrial> = (0..10)
                .map(|i| {
                    create_trial(
                        i,
                        vec![(
                            x_id,
                            ParamValue::Float(f64::from(i as u32) / 10.0),
                            float_dist(0.0, 1.0),
                        )],
                        f64::from(i as u32),
                    )
                })
                .collect();

            let value = sampler.sample(&dist, 10, &history);
            match value {
                ParamValue::Float(v) => {
                    assert!((0.0..=1.0).contains(&v), "Value {v} should be in [0, 1]");
                }
                _ => panic!("Expected Float value"),
            }
        }

        #[test]
        fn test_sampler_trait_cache_consistency() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .seed(42)
                .build()
                .unwrap();

            let dist = float_dist(0.0, 1.0);
            let history: Vec<CompletedTrial> = vec![];

            // Sample twice with the same trial_id
            let value1 = sampler.sample(&dist, 0, &history);
            let value2 = sampler.sample(&dist, 0, &history);

            // Should return cached value for same trial_id
            match (&value1, &value2) {
                (ParamValue::Float(v1), ParamValue::Float(v2)) => {
                    assert!(
                        (*v1 - *v2).abs() < f64::EPSILON,
                        "Same trial_id should return same value: {v1} vs {v2}"
                    );
                }
                _ => panic!("Expected Float values"),
            }
        }

        #[test]
        fn test_sampler_trait_different_trial_ids() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .seed(42)
                .build()
                .unwrap();

            let dist = float_dist(0.0, 1.0);
            let history: Vec<CompletedTrial> = vec![];

            // Sample with different trial_ids
            let value1 = sampler.sample(&dist, 0, &history);
            let value2 = sampler.sample(&dist, 1, &history);

            // Values may differ (new joint sample for each trial_id)
            // Just verify both are valid
            match (&value1, &value2) {
                (ParamValue::Float(v1), ParamValue::Float(v2)) => {
                    assert!((0.0..=1.0).contains(v1), "Value {v1} should be in [0, 1]");
                    assert!((0.0..=1.0).contains(v2), "Value {v2} should be in [0, 1]");
                }
                _ => panic!("Expected Float values"),
            }
        }

        #[test]
        fn test_sampler_trait_int_distribution() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .seed(42)
                .build()
                .unwrap();

            let dist = int_dist(0, 10);
            let history: Vec<CompletedTrial> = vec![];

            let value = sampler.sample(&dist, 0, &history);
            match value {
                ParamValue::Int(v) => {
                    assert!((0..=10).contains(&v), "Value {v} should be in [0, 10]");
                }
                _ => panic!("Expected Int value"),
            }
        }

        #[test]
        fn test_sampler_trait_categorical_distribution() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .seed(42)
                .build()
                .unwrap();

            let dist = categorical_dist(3);
            let history: Vec<CompletedTrial> = vec![];

            let value = sampler.sample(&dist, 0, &history);
            match value {
                ParamValue::Categorical(v) => {
                    assert!(v < 3, "Value {v} should be < 3");
                }
                _ => panic!("Expected Categorical value"),
            }
        }

        #[test]
        fn test_sampler_trait_with_multivariate_history() {
            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .n_ei_candidates(24)
                .seed(42)
                .build()
                .unwrap();

            let dist_x = float_dist(0.0, 1.0);
            let dist_y = float_dist(0.0, 1.0);

            // Create history with two parameters
            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let history: Vec<CompletedTrial> = (0..10)
                .map(|i| {
                    create_trial(
                        i,
                        vec![
                            (
                                x_id,
                                ParamValue::Float(f64::from(i as u32) / 10.0),
                                float_dist(0.0, 1.0),
                            ),
                            (
                                y_id,
                                ParamValue::Float(f64::from((10 - i) as u32) / 10.0),
                                float_dist(0.0, 1.0),
                            ),
                        ],
                        f64::from(i as u32),
                    )
                })
                .collect();

            // Sample x and y for the same trial
            let value_x = sampler.sample(&dist_x, 10, &history);
            let value_y = sampler.sample(&dist_y, 10, &history);

            // Both should be valid Float values
            match (&value_x, &value_y) {
                (ParamValue::Float(vx), ParamValue::Float(vy)) => {
                    assert!((0.0..=1.0).contains(vx), "Value {vx} should be in [0, 1]");
                    assert!((0.0..=1.0).contains(vy), "Value {vy} should be in [0, 1]");
                }
                _ => panic!("Expected Float values"),
            }
        }

        #[test]
        fn test_find_matching_param_float() {
            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let dist = float_dist(0.0, 1.0);
            let mut space = HashMap::new();
            space.insert(x_id, dist.clone());
            space.insert(y_id, float_dist(2.0, 3.0));
            let mut cached = HashMap::new();
            cached.insert(x_id, ParamValue::Float(0.5));
            cached.insert(y_id, ParamValue::Float(2.8));

            let result = MultivariateTpeSampler::find_matching_param(&dist, &space, &cached);

            assert!(result.is_some());
            if let Some(ParamValue::Float(v)) = result {
                assert!((0.0..=1.0).contains(&v));
            }
        }

        #[test]
        fn test_find_matching_param_int() {
            let n_id = ParamId::new();
            let dist = int_dist(0, 10);
            let mut space = HashMap::new();
            space.insert(n_id, dist.clone());
            let mut cached = HashMap::new();
            cached.insert(n_id, ParamValue::Int(5));

            let result = MultivariateTpeSampler::find_matching_param(&dist, &space, &cached);

            assert!(result.is_some());
            if let Some(ParamValue::Int(v)) = result {
                assert!((0..=10).contains(&v));
            }
        }

        #[test]
        fn test_find_matching_param_categorical() {
            let choice_id = ParamId::new();
            let dist = categorical_dist(3);
            let mut space = HashMap::new();
            space.insert(choice_id, dist.clone());
            let mut cached = HashMap::new();
            cached.insert(choice_id, ParamValue::Categorical(1));

            let result = MultivariateTpeSampler::find_matching_param(&dist, &space, &cached);

            assert!(result.is_some());
            if let Some(ParamValue::Categorical(v)) = result {
                assert!(v < 3);
            }
        }

        #[test]
        fn test_find_matching_param_no_match() {
            let x_id = ParamId::new();
            let mut space = HashMap::new();
            space.insert(x_id, float_dist(0.0, 1.0));
            let mut cached = HashMap::new();
            cached.insert(x_id, ParamValue::Float(0.5));

            // Looking for Int, but only Float in search space
            let dist = int_dist(0, 10);
            let result = MultivariateTpeSampler::find_matching_param(&dist, &space, &cached);

            assert!(result.is_none());
        }

        #[test]
        fn test_find_matching_param_out_of_bounds() {
            let x_id = ParamId::new();
            // Search space has a different distribution than what we're looking for
            let mut space = HashMap::new();
            space.insert(x_id, float_dist(0.0, 10.0));
            let mut cached = HashMap::new();
            cached.insert(x_id, ParamValue::Float(5.0));

            let dist = float_dist(0.0, 1.0);
            let result = MultivariateTpeSampler::find_matching_param(&dist, &space, &cached);

            assert!(result.is_none());
        }

        #[test]
        fn test_build_search_space_from_history() {
            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let history = vec![
                create_trial(
                    0,
                    vec![
                        (x_id, ParamValue::Float(0.5), float_dist(0.0, 1.0)),
                        (y_id, ParamValue::Int(5), int_dist(0, 10)),
                    ],
                    1.0,
                ),
                create_trial(
                    1,
                    vec![
                        (x_id, ParamValue::Float(0.3), float_dist(0.0, 1.0)),
                        (y_id, ParamValue::Int(3), int_dist(0, 10)),
                    ],
                    2.0,
                ),
            ];

            let current_dist = float_dist(0.0, 1.0);
            let search_space =
                MultivariateTpeSampler::build_search_space_from_history(&current_dist, &history);

            assert!(search_space.contains_key(&x_id));
            assert!(search_space.contains_key(&y_id));
        }

        #[test]
        fn test_build_search_space_empty_history() {
            let history: Vec<CompletedTrial> = vec![];

            let current_dist = float_dist(0.0, 1.0);
            let search_space =
                MultivariateTpeSampler::build_search_space_from_history(&current_dist, &history);

            // Should have a placeholder for current distribution
            assert!(!search_space.is_empty());
            // The placeholder uses ParamId::new(), so just check it has exactly 1 entry
            assert_eq!(search_space.len(), 1);
        }
    }

    // ========================================================================
    // Independent Fallback Sampling Tests
    // ========================================================================

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_lossless
    )]
    mod independent_fallback_tests {
        use super::*;
        use crate::distribution::{CategoricalDistribution, FloatDistribution, IntDistribution};
        use crate::parameter::ParamId;

        fn create_trial(
            id: u64,
            params: Vec<(ParamId, ParamValue, Distribution)>,
            value: f64,
        ) -> CompletedTrial {
            let mut param_map = HashMap::new();
            let mut dist_map = HashMap::new();
            for (param_id, pv, dist) in params {
                param_map.insert(param_id, pv);
                dist_map.insert(param_id, dist);
            }
            CompletedTrial::new(id, param_map, dist_map, HashMap::new(), value)
        }

        fn float_dist(low: f64, high: f64) -> Distribution {
            Distribution::Float(FloatDistribution {
                low,
                high,
                log_scale: false,
                step: None,
            })
        }

        fn int_dist(low: i64, high: i64) -> Distribution {
            Distribution::Int(IntDistribution {
                low,
                high,
                log_scale: false,
                step: None,
            })
        }

        fn categorical_dist(n_choices: usize) -> Distribution {
            Distribution::Categorical(CategoricalDistribution { n_choices })
        }

        #[test]
        fn test_fallback_on_empty_intersection() {
            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let z_id = ParamId::new();
            // Create trials with completely different parameters
            let history = vec![
                create_trial(
                    0,
                    vec![(x_id, ParamValue::Float(0.1), float_dist(0.0, 1.0))],
                    1.0,
                ),
                create_trial(
                    1,
                    vec![(y_id, ParamValue::Float(0.2), float_dist(0.0, 1.0))],
                    2.0,
                ),
                create_trial(
                    2,
                    vec![(z_id, ParamValue::Float(0.3), float_dist(0.0, 1.0))],
                    3.0,
                ),
            ];

            // Request parameters that none of the trials have in common
            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist(0.0, 1.0));
            search_space.insert(y_id, float_dist(0.0, 1.0));

            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(1)
                .seed(42)
                .build()
                .unwrap();

            // This should fall back to independent sampling since no common params
            let result = sampler.sample_joint(&search_space, &history);

            // Should have all requested parameters
            assert!(result.contains_key(&x_id));
            assert!(result.contains_key(&y_id));

            // Values should be within bounds
            if let ParamValue::Float(x) = result.get(&x_id).unwrap() {
                assert!((0.0..=1.0).contains(x));
            } else {
                panic!("Expected Float for x");
            }
            if let ParamValue::Float(y) = result.get(&y_id).unwrap() {
                assert!((0.0..=1.0).contains(y));
            } else {
                panic!("Expected Float for y");
            }
        }

        #[test]
        fn test_fallback_fills_missing_params() {
            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let z_id = ParamId::new();
            // Trials have x and y, but we request x, y, and z
            let history = vec![
                create_trial(
                    0,
                    vec![
                        (x_id, ParamValue::Float(0.2), float_dist(0.0, 1.0)),
                        (y_id, ParamValue::Float(0.3), float_dist(0.0, 1.0)),
                    ],
                    1.0,
                ),
                create_trial(
                    1,
                    vec![
                        (x_id, ParamValue::Float(0.4), float_dist(0.0, 1.0)),
                        (y_id, ParamValue::Float(0.5), float_dist(0.0, 1.0)),
                    ],
                    2.0,
                ),
                create_trial(
                    2,
                    vec![
                        (x_id, ParamValue::Float(0.6), float_dist(0.0, 1.0)),
                        (y_id, ParamValue::Float(0.7), float_dist(0.0, 1.0)),
                    ],
                    3.0,
                ),
            ];

            // Request x, y (which are in intersection) and z (which is not)
            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist(0.0, 1.0));
            search_space.insert(y_id, float_dist(0.0, 1.0));
            search_space.insert(z_id, float_dist(0.0, 1.0));

            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(1)
                .seed(42)
                .build()
                .unwrap();

            let result = sampler.sample_joint(&search_space, &history);

            // Should have all three parameters
            assert!(result.contains_key(&x_id));
            assert!(result.contains_key(&y_id));
            assert!(result.contains_key(&z_id));

            // All values should be within bounds
            for value in result.values() {
                if let ParamValue::Float(v) = value {
                    assert!(
                        (0.0..=1.0).contains(v),
                        "Parameter has value {v} out of bounds"
                    );
                }
            }
        }

        #[test]
        fn test_independent_tpe_sampling_with_int() {
            let n_id = ParamId::new();
            let m_id = ParamId::new();
            // Create trials with int parameters
            let history: Vec<CompletedTrial> = (0..20)
                .map(|i| {
                    create_trial(
                        i,
                        vec![(n_id, ParamValue::Int((i % 10) as i64), int_dist(0, 10))],
                        (i as f64) * 0.1,
                    )
                })
                .collect();

            let mut search_space = HashMap::new();
            search_space.insert(n_id, int_dist(0, 10));
            search_space.insert(m_id, int_dist(0, 5)); // Not in history

            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .seed(42)
                .build()
                .unwrap();

            let result = sampler.sample_joint(&search_space, &history);

            assert!(result.contains_key(&n_id));
            assert!(result.contains_key(&m_id));

            if let ParamValue::Int(n) = result.get(&n_id).unwrap() {
                assert!((0..=10).contains(n));
            } else {
                panic!("Expected Int for n");
            }
            if let ParamValue::Int(m) = result.get(&m_id).unwrap() {
                assert!((0..=5).contains(m));
            } else {
                panic!("Expected Int for m");
            }
        }

        #[test]
        fn test_independent_tpe_sampling_with_categorical() {
            let cat_id = ParamId::new();
            let other_cat_id = ParamId::new();
            // Create trials with categorical parameters
            let history: Vec<CompletedTrial> = (0..20)
                .map(|i| {
                    create_trial(
                        i,
                        vec![(
                            cat_id,
                            ParamValue::Categorical(i as usize % 3),
                            categorical_dist(3),
                        )],
                        (i as f64) * 0.1,
                    )
                })
                .collect();

            let mut search_space = HashMap::new();
            search_space.insert(cat_id, categorical_dist(3));
            search_space.insert(other_cat_id, categorical_dist(4)); // Not in history

            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .seed(42)
                .build()
                .unwrap();

            let result = sampler.sample_joint(&search_space, &history);

            assert!(result.contains_key(&cat_id));
            assert!(result.contains_key(&other_cat_id));

            if let ParamValue::Categorical(cat) = result.get(&cat_id).unwrap() {
                assert!(*cat < 3);
            } else {
                panic!("Expected Categorical for cat");
            }
            if let ParamValue::Categorical(other) = result.get(&other_cat_id).unwrap() {
                assert!(*other < 4);
            } else {
                panic!("Expected Categorical for other_cat");
            }
        }

        #[test]
        fn test_sample_all_independent_uses_tpe() {
            let x_id = ParamId::new();
            // Create trials with a clear pattern - good values clustered low
            let history: Vec<CompletedTrial> = (0..30)
                .map(|i| {
                    let x = if i < 10 {
                        0.1 + (i as f64) * 0.02 // Good trials: 0.1-0.28
                    } else {
                        0.6 + ((i - 10) as f64) * 0.02 // Bad trials: 0.6-0.98
                    };
                    let value = if i < 10 { 0.0 } else { 1.0 };
                    create_trial(
                        0,
                        vec![(x_id, ParamValue::Float(x), float_dist(0.0, 1.0))],
                        value,
                    )
                })
                .collect();

            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist(0.0, 1.0));

            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .n_ei_candidates(48)
                .seed(123)
                .build()
                .unwrap();

            // Sample multiple times and check that we tend to sample from the good region
            let mut low_count = 0;
            let n_samples = 100;
            for _ in 0..n_samples {
                let result = sampler.sample_all_independent(&search_space, &history);
                if let Some(ParamValue::Float(x)) = result.get(&x_id)
                    && *x < 0.5
                {
                    low_count += 1;
                }
            }

            // TPE should bias towards lower values where good trials are clustered
            // With random sampling we'd expect ~50%, with TPE we should see more bias
            assert!(
                low_count > 60,
                "Expected TPE to bias towards good region, but only got {low_count} out of {n_samples} in low region"
            );
        }

        #[test]
        fn test_fallback_with_few_filtered_trials() {
            let x_id = ParamId::new();
            let y_id = ParamId::new();
            // Create trials where only 1 has all the required parameters
            let history = vec![
                create_trial(
                    0,
                    vec![
                        (x_id, ParamValue::Float(0.1), float_dist(0.0, 1.0)),
                        (y_id, ParamValue::Float(0.2), float_dist(0.0, 1.0)),
                    ],
                    1.0,
                ),
                create_trial(
                    1,
                    vec![(x_id, ParamValue::Float(0.3), float_dist(0.0, 1.0))],
                    2.0,
                ),
                create_trial(
                    2,
                    vec![(x_id, ParamValue::Float(0.5), float_dist(0.0, 1.0))],
                    3.0,
                ),
            ];

            // Request both x and y - only trial 0 has both
            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist(0.0, 1.0));
            search_space.insert(y_id, float_dist(0.0, 1.0));

            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(1)
                .seed(42)
                .build()
                .unwrap();

            // Should fall back since only 1 trial has both params (need at least 2)
            let result = sampler.sample_joint(&search_space, &history);

            assert!(result.contains_key(&x_id));
            assert!(result.contains_key(&y_id));
        }

        #[test]
        fn test_fill_remaining_uniform_fallback() {
            let x_id = ParamId::new();
            // During startup phase, should use uniform sampling
            let history: Vec<CompletedTrial> = vec![]; // No history

            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist(0.0, 1.0));

            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(10) // High startup means we'll use random
                .seed(42)
                .build()
                .unwrap();

            // With no history and high startup threshold, should sample uniformly
            let result = sampler.sample_joint(&search_space, &history);

            assert!(result.contains_key(&x_id));
            if let ParamValue::Float(x) = result.get(&x_id).unwrap() {
                assert!((0.0..=1.0).contains(x));
            }
        }

        #[test]
        fn test_mixed_params_intersection_and_not() {
            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let z_id = ParamId::new();
            let cat_id = ParamId::new();
            // Create history with x,y - both will be in intersection
            let history: Vec<CompletedTrial> = (0..15)
                .map(|i| {
                    create_trial(
                        i,
                        vec![
                            (
                                x_id,
                                ParamValue::Float((i as f64) * 0.05),
                                float_dist(0.0, 1.0),
                            ),
                            (y_id, ParamValue::Int(i as i64 % 5), int_dist(0, 10)),
                        ],
                        (i as f64) * 0.1,
                    )
                })
                .collect();

            // Request x, y (in intersection) and z, cat (not in intersection)
            let mut search_space = HashMap::new();
            search_space.insert(x_id, float_dist(0.0, 1.0));
            search_space.insert(y_id, int_dist(0, 10));
            search_space.insert(z_id, float_dist(-1.0, 1.0));
            search_space.insert(cat_id, categorical_dist(5));

            let sampler = MultivariateTpeSampler::builder()
                .n_startup_trials(5)
                .seed(42)
                .build()
                .unwrap();

            let result = sampler.sample_joint(&search_space, &history);

            // All parameters should be present
            assert!(result.contains_key(&x_id));
            assert!(result.contains_key(&y_id));
            assert!(result.contains_key(&z_id));
            assert!(result.contains_key(&cat_id));

            // Validate bounds
            if let ParamValue::Float(x) = result.get(&x_id).unwrap() {
                assert!((0.0..=1.0).contains(x));
            }
            if let ParamValue::Int(y) = result.get(&y_id).unwrap() {
                assert!((0..=10).contains(y));
            }
            if let ParamValue::Float(z) = result.get(&z_id).unwrap() {
                assert!((-1.0..=1.0).contains(z));
            }
            if let ParamValue::Categorical(cat) = result.get(&cat_id).unwrap() {
                assert!(*cat < 5);
            }
        }
    }

    /// Tests for group decomposition integration (US-020).
    mod group_sampling_tests {
        use super::*;
        use crate::distribution::{CategoricalDistribution, FloatDistribution, IntDistribution};
        use crate::parameter::ParamId;

        fn create_trial(
            id: u64,
            params: Vec<(ParamId, ParamValue, Distribution)>,
            value: f64,
        ) -> CompletedTrial {
            let mut param_map = HashMap::new();
            let mut dist_map = HashMap::new();
            for (param_id, pv, dist) in params {
                param_map.insert(param_id, pv);
                dist_map.insert(param_id, dist);
            }
            CompletedTrial::new(id, param_map, dist_map, HashMap::new(), value)
        }

        #[test]
        fn test_group_mode_disabled_samples_all_together() {
            // When group=false, should sample all params jointly
            let sampler = MultivariateTpeSamplerBuilder::new()
                .group(false)
                .n_startup_trials(3)
                .seed(42)
                .build()
                .unwrap();

            let dist = Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            });

            let x_id = ParamId::new();
            let y_id = ParamId::new();

            // Create history with all params together
            let history: Vec<CompletedTrial> = (0..10)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let v = (i as f64) * 0.1;
                    create_trial(
                        i,
                        vec![
                            (x_id, ParamValue::Float(v), dist.clone()),
                            (y_id, ParamValue::Float(v + 0.05), dist.clone()),
                        ],
                        v * v,
                    )
                })
                .collect();

            let mut search_space = HashMap::new();
            search_space.insert(x_id, dist.clone());
            search_space.insert(y_id, dist);

            let result = sampler.sample_joint(&search_space, &history);

            assert!(result.contains_key(&x_id));
            assert!(result.contains_key(&y_id));
            assert_eq!(result.len(), 2);
        }

        #[test]
        fn test_group_mode_enabled_samples_groups_independently() {
            // When group=true, should decompose into groups based on co-occurrence
            let sampler = MultivariateTpeSamplerBuilder::new()
                .group(true)
                .n_startup_trials(3)
                .seed(42)
                .build()
                .unwrap();

            let dist = Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            });

            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let a_id = ParamId::new();
            let b_id = ParamId::new();

            // Create history with two independent groups:
            // Group 1: x, y appear together
            // Group 2: a, b appear together
            // Groups never co-occur
            let mut history = Vec::new();
            for i in 0..5 {
                #[allow(clippy::cast_precision_loss)]
                let v = (i as f64) * 0.1;
                history.push(create_trial(
                    i,
                    vec![
                        (x_id, ParamValue::Float(v), dist.clone()),
                        (y_id, ParamValue::Float(v + 0.05), dist.clone()),
                    ],
                    v * v,
                ));
            }
            for i in 5..10 {
                #[allow(clippy::cast_precision_loss)]
                let v = (i as f64) * 0.05;
                history.push(create_trial(
                    i,
                    vec![
                        (a_id, ParamValue::Float(v), dist.clone()),
                        (b_id, ParamValue::Float(v + 0.1), dist.clone()),
                    ],
                    v + 0.5,
                ));
            }

            // Search space contains params from both groups
            let mut search_space = HashMap::new();
            search_space.insert(x_id, dist.clone());
            search_space.insert(y_id, dist.clone());
            search_space.insert(a_id, dist.clone());
            search_space.insert(b_id, dist);

            let result = sampler.sample_joint(&search_space, &history);

            // All params should be sampled
            assert!(result.contains_key(&x_id));
            assert!(result.contains_key(&y_id));
            assert!(result.contains_key(&a_id));
            assert!(result.contains_key(&b_id));
            assert_eq!(result.len(), 4);

            // Values should be within bounds
            for value in result.values() {
                if let ParamValue::Float(v) = value {
                    assert!((0.0..=1.0).contains(v));
                }
            }
        }

        #[test]
        fn test_group_mode_with_single_group() {
            // When all params co-occur, group mode should behave like non-group mode
            let sampler_grouped = MultivariateTpeSamplerBuilder::new()
                .group(true)
                .n_startup_trials(3)
                .seed(123)
                .build()
                .unwrap();

            let sampler_ungrouped = MultivariateTpeSamplerBuilder::new()
                .group(false)
                .n_startup_trials(3)
                .seed(123)
                .build()
                .unwrap();

            let dist = Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            });

            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let z_id = ParamId::new();

            // All params appear together in all trials
            let history: Vec<CompletedTrial> = (0..10)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let v = (i as f64) * 0.1;
                    create_trial(
                        i,
                        vec![
                            (x_id, ParamValue::Float(v), dist.clone()),
                            (y_id, ParamValue::Float(v + 0.05), dist.clone()),
                            (z_id, ParamValue::Float(v + 0.1), dist.clone()),
                        ],
                        v * v,
                    )
                })
                .collect();

            let mut search_space = HashMap::new();
            search_space.insert(x_id, dist.clone());
            search_space.insert(y_id, dist.clone());
            search_space.insert(z_id, dist);

            let result_grouped = sampler_grouped.sample_joint(&search_space, &history);
            let result_ungrouped = sampler_ungrouped.sample_joint(&search_space, &history);

            // Both should produce valid samples with all params
            assert_eq!(result_grouped.len(), 3);
            assert_eq!(result_ungrouped.len(), 3);

            // Both should have valid bounds
            for result in [&result_grouped, &result_ungrouped] {
                for value in result.values() {
                    if let ParamValue::Float(v) = value {
                        assert!((0.0..=1.0).contains(v));
                    }
                }
            }
        }

        #[test]
        fn test_group_mode_handles_isolated_params() {
            // Test handling of parameters that only appear alone
            let sampler = MultivariateTpeSamplerBuilder::new()
                .group(true)
                .n_startup_trials(3)
                .seed(42)
                .build()
                .unwrap();

            let dist = Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            });

            let x_id = ParamId::new();
            let y_id = ParamId::new();
            let z_id = ParamId::new();

            // Each param appears alone (forms its own isolated group)
            let history = vec![
                create_trial(0, vec![(x_id, ParamValue::Float(0.3), dist.clone())], 1.0),
                create_trial(1, vec![(y_id, ParamValue::Float(0.5), dist.clone())], 0.5),
                create_trial(2, vec![(z_id, ParamValue::Float(0.7), dist.clone())], 0.8),
                create_trial(3, vec![(x_id, ParamValue::Float(0.2), dist.clone())], 1.2),
                create_trial(4, vec![(y_id, ParamValue::Float(0.6), dist.clone())], 0.4),
                create_trial(5, vec![(z_id, ParamValue::Float(0.8), dist.clone())], 0.7),
            ];

            let mut search_space = HashMap::new();
            search_space.insert(x_id, dist.clone());
            search_space.insert(y_id, dist.clone());
            search_space.insert(z_id, dist);

            let result = sampler.sample_joint(&search_space, &history);

            // All params should be sampled
            assert!(result.contains_key(&x_id));
            assert!(result.contains_key(&y_id));
            assert!(result.contains_key(&z_id));

            // Values should be within bounds
            for value in result.values() {
                if let ParamValue::Float(v) = value {
                    assert!((0.0..=1.0).contains(v));
                }
            }
        }

        #[test]
        fn test_group_mode_during_startup() {
            // During startup phase, should sample uniformly regardless of group mode
            let sampler = MultivariateTpeSamplerBuilder::new()
                .group(true)
                .n_startup_trials(10)
                .seed(42)
                .build()
                .unwrap();

            let dist = Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            });

            let x_id = ParamId::new();
            let y_id = ParamId::new();

            // Only 5 trials (less than n_startup_trials=10)
            let history: Vec<CompletedTrial> = (0..5)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let v = (i as f64) * 0.1;
                    create_trial(i, vec![(x_id, ParamValue::Float(v), dist.clone())], v * v)
                })
                .collect();

            let mut search_space = HashMap::new();
            search_space.insert(x_id, dist.clone());
            search_space.insert(y_id, dist);

            let result = sampler.sample_joint(&search_space, &history);

            // Should sample uniformly for all params
            assert!(result.contains_key(&x_id));
            assert!(result.contains_key(&y_id));
        }

        #[test]
        fn test_group_mode_with_mixed_distribution_types() {
            // Test group mode with Float, Int, and Categorical distributions
            let sampler = MultivariateTpeSamplerBuilder::new()
                .group(true)
                .n_startup_trials(3)
                .seed(42)
                .build()
                .unwrap();

            let dist_float = Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            });
            let dist_int = Distribution::Int(IntDistribution {
                low: 0,
                high: 10,
                log_scale: false,
                step: None,
            });
            let dist_cat = Distribution::Categorical(CategoricalDistribution { n_choices: 3 });

            let lr_id = ParamId::new();
            let layers_id = ParamId::new();
            let opt_id = ParamId::new();

            // Group 1: float, int co-occur
            // Group 2: categorical alone
            let mut history = Vec::new();
            for i in 0..5 {
                #[allow(clippy::cast_precision_loss)]
                let v = (i as f64) * 0.1;
                #[allow(clippy::cast_possible_wrap)]
                let int_v = (i % 10) as i64;
                history.push(create_trial(
                    i,
                    vec![
                        (lr_id, ParamValue::Float(v), dist_float.clone()),
                        (layers_id, ParamValue::Int(int_v), dist_int.clone()),
                    ],
                    v * v,
                ));
            }
            for i in 5..10 {
                history.push(create_trial(
                    i,
                    vec![(
                        opt_id,
                        ParamValue::Categorical((i % 3) as usize),
                        dist_cat.clone(),
                    )],
                    1.0,
                ));
            }

            let mut search_space = HashMap::new();
            search_space.insert(lr_id, dist_float);
            search_space.insert(layers_id, dist_int);
            search_space.insert(opt_id, dist_cat);

            let result = sampler.sample_joint(&search_space, &history);

            assert!(result.contains_key(&lr_id));
            assert!(result.contains_key(&layers_id));
            assert!(result.contains_key(&opt_id));

            // Check types
            assert!(matches!(result.get(&lr_id), Some(ParamValue::Float(_))));
            assert!(matches!(result.get(&layers_id), Some(ParamValue::Int(_))));
            assert!(matches!(
                result.get(&opt_id),
                Some(ParamValue::Categorical(_))
            ));
        }

        #[test]
        fn test_group_mode_empty_history() {
            // With empty history (startup phase), should sample uniformly
            let sampler = MultivariateTpeSamplerBuilder::new()
                .group(true)
                .n_startup_trials(5)
                .seed(42)
                .build()
                .unwrap();

            let dist = Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            });

            let x_id = ParamId::new();
            let y_id = ParamId::new();

            let history: Vec<CompletedTrial> = vec![];

            let mut search_space = HashMap::new();
            search_space.insert(x_id, dist.clone());
            search_space.insert(y_id, dist);

            let result = sampler.sample_joint(&search_space, &history);

            assert!(result.contains_key(&x_id));
            assert!(result.contains_key(&y_id));
        }

        #[test]
        fn test_non_grouped_sampling_with_different_groups() {
            // When group=false but history has separate groups,
            // intersection will be empty and fall back to independent
            let sampler = MultivariateTpeSamplerBuilder::new()
                .group(false)
                .n_startup_trials(3)
                .seed(42)
                .build()
                .unwrap();

            let dist = Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            });

            let x_id = ParamId::new();
            let y_id = ParamId::new();

            // Two non-overlapping groups
            let mut history = Vec::new();
            for i in 0..5 {
                #[allow(clippy::cast_precision_loss)]
                let v = (i as f64) * 0.1;
                history.push(create_trial(
                    i,
                    vec![(x_id, ParamValue::Float(v), dist.clone())],
                    v,
                ));
            }
            for i in 5..10 {
                #[allow(clippy::cast_precision_loss)]
                let v = (i as f64) * 0.05;
                history.push(create_trial(
                    i,
                    vec![(y_id, ParamValue::Float(v), dist.clone())],
                    v,
                ));
            }

            let mut search_space = HashMap::new();
            search_space.insert(x_id, dist.clone());
            search_space.insert(y_id, dist);

            let result = sampler.sample_joint(&search_space, &history);

            // Should still produce valid results via fallback
            assert!(result.contains_key(&x_id));
            assert!(result.contains_key(&y_id));
        }

        #[test]
        fn test_group_mode_deterministic_with_seed() {
            // Same seed should produce same results
            let dist = Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            });

            let x_id = ParamId::new();
            let y_id = ParamId::new();

            let history: Vec<CompletedTrial> = (0..10)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let v = (i as f64) * 0.1;
                    create_trial(
                        i,
                        vec![
                            (x_id, ParamValue::Float(v), dist.clone()),
                            (y_id, ParamValue::Float(v + 0.05), dist.clone()),
                        ],
                        v * v,
                    )
                })
                .collect();

            let mut search_space = HashMap::new();
            search_space.insert(x_id, dist.clone());
            search_space.insert(y_id, dist);

            // Same seed should produce same results
            let sampler1 = MultivariateTpeSamplerBuilder::new()
                .group(true)
                .seed(999)
                .build()
                .unwrap();

            let sampler2 = MultivariateTpeSamplerBuilder::new()
                .group(true)
                .seed(999)
                .build()
                .unwrap();

            let result1 = sampler1.sample_joint(&search_space, &history);
            let result2 = sampler2.sample_joint(&search_space, &history);

            assert_eq!(result1, result2);
        }

        #[test]
        #[allow(clippy::cast_precision_loss)]
        fn test_group_mode_handles_ungrouped_params() {
            // Test that params not in any group get sampled independently
            let sampler = MultivariateTpeSamplerBuilder::new()
                .group(true)
                .n_startup_trials(3)
                .seed(42)
                .build()
                .unwrap();

            let dist = Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            });

            let x_id = ParamId::new();
            let y_id = ParamId::new();

            // History only has x, but search space has x and y
            let history: Vec<CompletedTrial> = (0..10)
                .map(|i| {
                    let v = (i as f64) * 0.1;
                    create_trial(i, vec![(x_id, ParamValue::Float(v), dist.clone())], v * v)
                })
                .collect();

            let mut search_space = HashMap::new();
            search_space.insert(x_id, dist.clone());
            search_space.insert(y_id, dist); // Not in history

            let result = sampler.sample_joint(&search_space, &history);

            // Both should be sampled
            assert!(result.contains_key(&x_id));
            assert!(result.contains_key(&y_id));

            // y should be sampled uniformly (not in any group)
            if let ParamValue::Float(v) = result.get(&y_id).unwrap() {
                assert!((0.0..=1.0).contains(v));
            }
        }
    }
}
