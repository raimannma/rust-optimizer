//! Tree-Parzen Estimator (TPE) sampler implementation.
//!
//! TPE is a Bayesian optimization algorithm that models the objective function
//! using two probability distributions: one for promising (good) parameter values
//! and one for unpromising (bad) parameter values.
//!
//! # Gamma Strategies
//!
//! The gamma parameter controls what fraction of trials are considered "good".
//! This module provides several built-in strategies via the [`GammaStrategy`] trait:
//!
//! - [`FixedGamma`]: Constant gamma value (default: 0.25)
//! - [`LinearGamma`]: Linear interpolation between min and max based on trial count
//! - [`SqrtGamma`]: Gamma decreases as 1/√n (similar to Optuna)
//! - [`HyperoptGamma`]: Hyperopt-style adaptive gamma
//!
//! You can also implement your own strategy by implementing the [`GammaStrategy`] trait.
//!
//! # Examples
//!
//! Using a built-in gamma strategy:
//!
//! ```
//! use optimizer::sampler::tpe::{SqrtGamma, TpeSampler};
//!
//! let sampler = TpeSampler::builder()
//!     .gamma_strategy(SqrtGamma::default())
//!     .build()
//!     .unwrap();
//! ```
//!
//! Implementing a custom gamma strategy:
//!
//! ```
//! use optimizer::sampler::tpe::{GammaStrategy, TpeSampler};
//!
//! #[derive(Debug, Clone)]
//! struct MyGamma {
//!     base: f64,
//! }
//!
//! impl GammaStrategy for MyGamma {
//!     fn gamma(&self, n_trials: usize) -> f64 {
//!         (self.base + 0.01 * n_trials as f64).min(0.5)
//!     }
//!
//!     fn clone_box(&self) -> Box<dyn GammaStrategy> {
//!         Box::new(self.clone())
//!     }
//! }
//!
//! let sampler = TpeSampler::builder()
//!     .gamma_strategy(MyGamma { base: 0.1 })
//!     .build()
//!     .unwrap();
//! ```

use core::fmt::Debug;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::distribution::Distribution;
use crate::error::{Error, Result};
use crate::param::ParamValue;
use crate::rng_util;
use crate::sampler::common;
use crate::sampler::tpe::gamma::{FixedGamma, GammaStrategy};
use crate::sampler::{CompletedTrial, Sampler};

use super::common as tpe_common;

// ============================================================================
// Gamma Strategy Trait and Implementations
// ============================================================================

// ============================================================================
// TPE Sampler
// ============================================================================

/// A Tree-Parzen Estimator (TPE) sampler for Bayesian optimization.
///
/// TPE works by splitting completed trials into two groups based on their
/// objective values: good trials (below the gamma quantile) and bad trials
/// (above the gamma quantile). It then fits kernel density estimators (KDE)
/// to each group and samples new points that maximize the ratio l(x)/g(x),
/// where l(x) is the density of good trials and g(x) is the density of bad trials.
///
/// During the startup phase (when fewer than `n_startup_trials` are completed),
/// TPE falls back to random sampling to gather initial data.
///
/// # Gamma Strategies
///
/// The gamma quantile can be configured using different strategies via the
/// [`GammaStrategy`] trait.
///
/// # Examples
///
/// ```
/// use optimizer::sampler::tpe::TpeSampler;
///
/// // Create with default settings (FixedGamma at 0.25)
/// let sampler = TpeSampler::new();
///
/// // Create with custom settings using the builder
/// let sampler = TpeSampler::builder()
///     .gamma(0.15)  // Shorthand for FixedGamma::new(0.15)
///     .n_startup_trials(20)
///     .n_ei_candidates(32)
///     .seed(42)
///     .build()
///     .unwrap();
/// ```
///
/// Using a different gamma strategy:
///
/// ```
/// use optimizer::sampler::tpe::{SqrtGamma, TpeSampler};
///
/// let sampler = TpeSampler::builder()
///     .gamma_strategy(SqrtGamma::default())
///     .build()
///     .unwrap();
/// ```
pub struct TpeSampler {
    /// Strategy for computing the gamma quantile.
    gamma_strategy: Arc<dyn GammaStrategy>,
    /// Number of trials before TPE kicks in (uses random sampling before this).
    n_startup_trials: usize,
    /// Number of candidate samples to evaluate when selecting the next point.
    n_ei_candidates: usize,
    /// Optional fixed bandwidth for KDE. If None, uses Scott's rule.
    kde_bandwidth: Option<f64>,
    /// Base seed for deterministic per-call RNG derivation (no mutex needed).
    seed: u64,
    /// Monotonic counter to disambiguate calls with identical (`trial_id`, distribution).
    call_seq: AtomicU64,
}

impl TpeSampler {
    /// Creates a new TPE sampler with default settings.
    ///
    /// Default settings:
    /// - gamma strategy: [`FixedGamma`] with gamma = 0.25
    /// - `n_startup_trials`: 10 (random sampling for first 10 trials)
    /// - `n_ei_candidates`: 24 (evaluate 24 candidates per sample)
    /// - `kde_bandwidth`: None (uses Scott's rule for automatic bandwidth)
    #[must_use]
    pub fn new() -> Self {
        Self {
            gamma_strategy: Arc::new(FixedGamma::default()),
            n_startup_trials: 10,
            n_ei_candidates: 24,
            kde_bandwidth: None,
            seed: fastrand::u64(..),
            call_seq: AtomicU64::new(0),
        }
    }

    /// Creates a builder for configuring a TPE sampler.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::TpeSampler;
    ///
    /// let sampler = TpeSampler::builder()
    ///     .gamma(0.15)
    ///     .n_startup_trials(20)
    ///     .n_ei_candidates(32)
    ///     .seed(42)
    ///     .build()
    ///     .unwrap();
    /// ```
    #[must_use]
    pub fn builder() -> TpeSamplerBuilder {
        TpeSamplerBuilder::new()
    }

    /// Creates a new TPE sampler with custom configuration.
    ///
    /// This method uses a fixed gamma value. For more advanced gamma strategies,
    /// use [`TpeSampler::with_strategy`] or the builder pattern with
    /// [`TpeSamplerBuilder::gamma_strategy`].
    ///
    /// # Arguments
    ///
    /// * `gamma` - Fraction of trials to consider "good" (0.0 to 1.0).
    /// * `n_startup_trials` - Number of random trials before TPE sampling.
    /// * `n_ei_candidates` - Number of candidates to evaluate per sample.
    /// * `kde_bandwidth` - Optional fixed bandwidth for KDE. If None, uses Scott's rule.
    /// * `seed` - Optional seed for reproducibility.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidGamma` if gamma is not in (0.0, 1.0).
    /// Returns `Error::InvalidBandwidth` if `kde_bandwidth` is Some but not positive.
    pub fn with_config(
        gamma: f64,
        n_startup_trials: usize,
        n_ei_candidates: usize,
        kde_bandwidth: Option<f64>,
        seed: Option<u64>,
    ) -> Result<Self> {
        let gamma_strategy = FixedGamma::new(gamma)?;
        Self::with_strategy(
            gamma_strategy,
            n_startup_trials,
            n_ei_candidates,
            kde_bandwidth,
            seed,
        )
    }

    /// Creates a new TPE sampler with a custom gamma strategy.
    ///
    /// # Arguments
    ///
    /// * `gamma_strategy` - The strategy for computing the gamma quantile.
    /// * `n_startup_trials` - Number of random trials before TPE sampling.
    /// * `n_ei_candidates` - Number of candidates to evaluate per sample.
    /// * `kde_bandwidth` - Optional fixed bandwidth for KDE. If None, uses Scott's rule.
    /// * `seed` - Optional seed for reproducibility.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidBandwidth` if `kde_bandwidth` is Some but not positive.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::{SqrtGamma, TpeSampler};
    ///
    /// let sampler = TpeSampler::with_strategy(
    ///     SqrtGamma::default(),
    ///     10,       // n_startup_trials
    ///     24,       // n_ei_candidates
    ///     None,     // kde_bandwidth
    ///     Some(42), // seed
    /// )
    /// .unwrap();
    /// ```
    pub fn with_strategy<G: GammaStrategy + 'static>(
        gamma_strategy: G,
        n_startup_trials: usize,
        n_ei_candidates: usize,
        kde_bandwidth: Option<f64>,
        seed: Option<u64>,
    ) -> Result<Self> {
        if let Some(bw) = kde_bandwidth
            && bw <= 0.0
        {
            return Err(Error::InvalidBandwidth(bw));
        }

        Ok(Self {
            gamma_strategy: Arc::new(gamma_strategy),
            n_startup_trials,
            n_ei_candidates,
            kde_bandwidth,
            seed: seed.unwrap_or_else(|| fastrand::u64(..)),
            call_seq: AtomicU64::new(0),
        })
    }

    /// Returns the gamma strategy used by this sampler.
    #[must_use]
    pub fn gamma_strategy(&self) -> &dyn GammaStrategy {
        self.gamma_strategy.as_ref()
    }

    /// Splits trials into good and bad groups based on the gamma quantile.
    ///
    /// The gamma value is computed dynamically using the configured [`GammaStrategy`].
    ///
    /// Returns (`good_trials`, `bad_trials`) where `good_trials` contains trials
    /// with values below the gamma quantile (for minimization).
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    #[must_use]
    fn split_trials<'a>(
        &self,
        history: &'a [CompletedTrial],
    ) -> (Vec<&'a CompletedTrial>, Vec<&'a CompletedTrial>) {
        if history.is_empty() {
            return (vec![], vec![]);
        }

        // Compute gamma using the strategy and clamp to valid range
        let gamma = self
            .gamma_strategy
            .gamma(history.len())
            .clamp(f64::EPSILON, 1.0 - f64::EPSILON);

        // Calculate the split point (gamma quantile)
        // Ensure at least 1 trial in each group if possible
        let n_good = ((history.len() as f64 * gamma).ceil() as usize)
            .max(1)
            .min(history.len() - 1);

        // Use quickselect (O(n)) to partition indices instead of full sort (O(n log n)).
        // We only need to know which trials are in the top gamma-quantile, not their order.
        let mut indices: Vec<usize> = (0..history.len()).collect();
        if n_good > 0 {
            indices.select_nth_unstable_by(n_good - 1, |&a, &b| {
                history[a]
                    .value
                    .partial_cmp(&history[b].value)
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
        }

        let good: Vec<_> = indices[..n_good].iter().map(|&i| &history[i]).collect();
        let bad: Vec<_> = indices[n_good..].iter().map(|&i| &history[i]).collect();

        (good, bad)
    }
}

impl Default for TpeSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for configuring a [`TpeSampler`].
///
/// This builder allows fluent configuration of TPE hyperparameters.
///
/// # Examples
///
/// Using a fixed gamma value:
///
/// ```
/// use optimizer::sampler::tpe::TpeSamplerBuilder;
///
/// let sampler = TpeSamplerBuilder::new()
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
/// use optimizer::sampler::tpe::{SqrtGamma, TpeSamplerBuilder};
///
/// let sampler = TpeSamplerBuilder::new()
///     .gamma_strategy(SqrtGamma::default())
///     .n_startup_trials(20)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TpeSamplerBuilder {
    gamma_strategy: Box<dyn GammaStrategy>,
    /// Raw gamma value for deferred validation (Some if `gamma()` was called)
    raw_gamma: Option<f64>,
    n_startup_trials: usize,
    n_ei_candidates: usize,
    kde_bandwidth: Option<f64>,
    seed: Option<u64>,
}

impl TpeSamplerBuilder {
    /// Creates a new builder with default settings.
    ///
    /// Default settings:
    /// - gamma strategy: [`FixedGamma`] with gamma = 0.25
    /// - `n_startup_trials`: 10 (random sampling for first 10 trials)
    /// - `n_ei_candidates`: 24 (evaluate 24 candidates per sample)
    /// - `kde_bandwidth`: None (uses Scott's rule for automatic bandwidth)
    /// - seed: None (use OS-provided entropy)
    #[must_use]
    pub fn new() -> Self {
        Self {
            gamma_strategy: Box::new(FixedGamma::default()),
            raw_gamma: None,
            n_startup_trials: 10,
            n_ei_candidates: 24,
            kde_bandwidth: None,
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
    /// use optimizer::sampler::tpe::TpeSamplerBuilder;
    ///
    /// let sampler = TpeSamplerBuilder::new()
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
        // We defer validation to build() time for consistency with the existing API
        // Store the raw value for validation later
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
    /// Using built-in strategies:
    ///
    /// ```
    /// use optimizer::sampler::tpe::{LinearGamma, SqrtGamma, TpeSamplerBuilder};
    ///
    /// // Square root strategy (Optuna-style)
    /// let sampler = TpeSamplerBuilder::new()
    ///     .gamma_strategy(SqrtGamma::default())
    ///     .build()
    ///     .unwrap();
    ///
    /// // Linear interpolation strategy
    /// let sampler = TpeSamplerBuilder::new()
    ///     .gamma_strategy(LinearGamma::new(0.1, 0.3, 50).unwrap())
    ///     .build()
    ///     .unwrap();
    /// ```
    ///
    /// Using a custom strategy:
    ///
    /// ```
    /// use optimizer::sampler::tpe::{GammaStrategy, TpeSamplerBuilder};
    ///
    /// #[derive(Debug, Clone)]
    /// struct MyGamma;
    ///
    /// impl GammaStrategy for MyGamma {
    ///     fn gamma(&self, n_trials: usize) -> f64 {
    ///         0.25 // Always return 0.25
    ///     }
    ///     fn clone_box(&self) -> Box<dyn GammaStrategy> {
    ///         Box::new(self.clone())
    ///     }
    /// }
    ///
    /// let sampler = TpeSamplerBuilder::new()
    ///     .gamma_strategy(MyGamma)
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
    /// TPE-based sampling begins.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of random trials before TPE kicks in.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::TpeSamplerBuilder;
    ///
    /// let sampler = TpeSamplerBuilder::new()
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
    /// When sampling a new point, TPE generates this many candidates from
    /// the l(x) distribution and selects the one with the highest l(x)/g(x)
    /// ratio.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of candidates to evaluate per sample.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::TpeSamplerBuilder;
    ///
    /// let sampler = TpeSamplerBuilder::new()
    ///     .n_ei_candidates(48)  // Evaluate more candidates
    ///     .build()
    ///     .unwrap();
    /// ```
    #[must_use]
    pub fn n_ei_candidates(mut self, n: usize) -> Self {
        self.n_ei_candidates = n;
        self
    }

    /// Sets a fixed bandwidth for the kernel density estimator.
    ///
    /// By default, TPE uses Scott's rule to automatically select the bandwidth
    /// based on the sample data. Use this method to override with a fixed value.
    ///
    /// Smaller bandwidths give more localized, peaky distributions.
    /// Larger bandwidths give smoother, more spread-out distributions.
    ///
    /// # Arguments
    ///
    /// * `bandwidth` - The fixed bandwidth (standard deviation) for Gaussian kernels.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::TpeSamplerBuilder;
    ///
    /// let sampler = TpeSamplerBuilder::new()
    ///     .kde_bandwidth(0.5)  // Fixed bandwidth of 0.5
    ///     .build()
    ///     .unwrap();
    /// ```
    ///
    /// # Note
    ///
    /// Validation happens at `build()` time. If bandwidth is not positive,
    /// `build()` will return `Err(Error::InvalidBandwidth)`.
    #[must_use]
    pub fn kde_bandwidth(mut self, bandwidth: f64) -> Self {
        self.kde_bandwidth = Some(bandwidth);
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
    /// use optimizer::sampler::tpe::TpeSamplerBuilder;
    ///
    /// let sampler = TpeSamplerBuilder::new()
    ///     .seed(42)  // Reproducible results
    ///     .build()
    ///     .unwrap();
    /// ```
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Builds the configured [`TpeSampler`].
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidGamma` if a fixed gamma value was set and is not in (0.0, 1.0).
    /// Returns `Error::InvalidBandwidth` if `kde_bandwidth` is Some but not positive.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::TpeSamplerBuilder;
    ///
    /// let sampler = TpeSamplerBuilder::new()
    ///     .gamma(0.15)
    ///     .n_startup_trials(20)
    ///     .n_ei_candidates(32)
    ///     .seed(42)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn build(self) -> Result<TpeSampler> {
        // Determine the gamma strategy to use
        let gamma_strategy: Arc<dyn GammaStrategy> = if let Some(raw) = self.raw_gamma {
            // Validate and create FixedGamma from raw value
            Arc::new(FixedGamma::new(raw)?)
        } else {
            Arc::from(self.gamma_strategy)
        };

        // Validate bandwidth
        if let Some(bw) = self.kde_bandwidth
            && bw <= 0.0
        {
            return Err(Error::InvalidBandwidth(bw));
        }

        Ok(TpeSampler {
            gamma_strategy,
            n_startup_trials: self.n_startup_trials,
            n_ei_candidates: self.n_ei_candidates,
            kde_bandwidth: self.kde_bandwidth,
            seed: self.seed.unwrap_or_else(|| fastrand::u64(..)),
            call_seq: AtomicU64::new(0),
        })
    }
}

impl Default for TpeSamplerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TpeSampler {
    fn sample_float(
        &self,
        d: &crate::distribution::FloatDistribution,
        good_trials: &[&CompletedTrial],
        bad_trials: &[&CompletedTrial],
        rng: &mut fastrand::Rng,
    ) -> ParamValue {
        let target_dist = Distribution::Float(d.clone());
        let good_values: Vec<f64> = good_trials
            .iter()
            .filter_map(|t| {
                t.distributions.iter().find_map(|(id, dist)| {
                    if *dist == target_dist {
                        t.params.get(id).and_then(|v| match v {
                            ParamValue::Float(f) => Some(*f),
                            _ => None,
                        })
                    } else {
                        None
                    }
                })
            })
            .collect();

        let bad_values: Vec<f64> = bad_trials
            .iter()
            .filter_map(|t| {
                t.distributions.iter().find_map(|(id, dist)| {
                    if *dist == target_dist {
                        t.params.get(id).and_then(|v| match v {
                            ParamValue::Float(f) => Some(*f),
                            _ => None,
                        })
                    } else {
                        None
                    }
                })
            })
            .collect();

        if good_values.is_empty() || bad_values.is_empty() {
            return ParamValue::Float(rng_util::f64_range(rng, d.low, d.high));
        }

        let value = tpe_common::sample_tpe_float(
            d,
            good_values,
            bad_values,
            self.n_ei_candidates,
            self.kde_bandwidth,
            rng,
        );
        ParamValue::Float(value)
    }

    fn sample_int(
        &self,
        d: &crate::distribution::IntDistribution,
        good_trials: &[&CompletedTrial],
        bad_trials: &[&CompletedTrial],
        rng: &mut fastrand::Rng,
    ) -> ParamValue {
        let target_dist = Distribution::Int(d.clone());
        let good_values: Vec<i64> = good_trials
            .iter()
            .filter_map(|t| {
                t.distributions.iter().find_map(|(id, dist)| {
                    if *dist == target_dist {
                        t.params.get(id).and_then(|v| match v {
                            ParamValue::Int(i) => Some(*i),
                            _ => None,
                        })
                    } else {
                        None
                    }
                })
            })
            .collect();

        let bad_values: Vec<i64> = bad_trials
            .iter()
            .filter_map(|t| {
                t.distributions.iter().find_map(|(id, dist)| {
                    if *dist == target_dist {
                        t.params.get(id).and_then(|v| match v {
                            ParamValue::Int(i) => Some(*i),
                            _ => None,
                        })
                    } else {
                        None
                    }
                })
            })
            .collect();

        if good_values.is_empty() || bad_values.is_empty() {
            return common::sample_random(rng, &Distribution::Int(d.clone()));
        }

        let value = tpe_common::sample_tpe_int(
            d,
            good_values,
            bad_values,
            self.n_ei_candidates,
            self.kde_bandwidth,
            rng,
        );
        ParamValue::Int(value)
    }

    #[allow(clippy::unused_self)]
    fn sample_categorical(
        &self,
        d: &crate::distribution::CategoricalDistribution,
        good_trials: &[&CompletedTrial],
        bad_trials: &[&CompletedTrial],
        rng: &mut fastrand::Rng,
    ) -> ParamValue {
        let target_dist = Distribution::Categorical(d.clone());
        let good_indices: Vec<usize> = good_trials
            .iter()
            .filter_map(|t| {
                t.distributions.iter().find_map(|(id, dist)| {
                    if *dist == target_dist {
                        t.params.get(id).and_then(|v| match v {
                            ParamValue::Categorical(i) => Some(*i),
                            _ => None,
                        })
                    } else {
                        None
                    }
                })
            })
            .collect();

        let bad_indices: Vec<usize> = bad_trials
            .iter()
            .filter_map(|t| {
                t.distributions.iter().find_map(|(id, dist)| {
                    if *dist == target_dist {
                        t.params.get(id).and_then(|v| match v {
                            ParamValue::Categorical(i) => Some(*i),
                            _ => None,
                        })
                    } else {
                        None
                    }
                })
            })
            .collect();

        if good_indices.is_empty() || bad_indices.is_empty() {
            return common::sample_random(rng, &Distribution::Categorical(d.clone()));
        }

        let index =
            tpe_common::sample_tpe_categorical(d.n_choices, &good_indices, &bad_indices, rng);
        ParamValue::Categorical(index)
    }
}

impl Sampler for TpeSampler {
    fn sample(
        &self,
        distribution: &Distribution,
        trial_id: u64,
        history: &[CompletedTrial],
    ) -> ParamValue {
        let seq = self.call_seq.fetch_add(1, Ordering::Relaxed);
        let mut rng = fastrand::Rng::with_seed(rng_util::mix_seed(
            self.seed,
            trial_id,
            rng_util::distribution_fingerprint(distribution).wrapping_add(seq),
        ));

        // Fall back to random sampling during startup phase
        if history.len() < self.n_startup_trials {
            return common::sample_random(&mut rng, distribution);
        }

        // Split trials into good and bad groups
        let (good_trials, bad_trials) = self.split_trials(history);

        // Need at least 1 trial in each group for TPE
        if good_trials.is_empty() || bad_trials.is_empty() {
            return common::sample_random(&mut rng, distribution);
        }

        match distribution {
            Distribution::Float(d) => self.sample_float(d, &good_trials, &bad_trials, &mut rng),
            Distribution::Int(d) => self.sample_int(d, &good_trials, &bad_trials, &mut rng),
            Distribution::Categorical(d) => {
                self.sample_categorical(d, &good_trials, &bad_trials, &mut rng)
            }
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::similar_names,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::distribution::{CategoricalDistribution, FloatDistribution, IntDistribution};
    use crate::parameter::ParamId;

    fn create_trial(
        id: u64,
        value: f64,
        params: Vec<(ParamId, ParamValue, Distribution)>,
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
    fn test_tpe_sampler_new() {
        let sampler = TpeSampler::new();
        // Default uses FixedGamma with 0.25
        assert!((sampler.gamma_strategy().gamma(0) - 0.25).abs() < f64::EPSILON);
        assert_eq!(sampler.n_startup_trials, 10);
        assert_eq!(sampler.n_ei_candidates, 24);
    }

    #[test]
    fn test_tpe_sampler_with_config() {
        let sampler = TpeSampler::with_config(0.15, 20, 32, None, Some(42)).unwrap();
        // with_config uses FixedGamma
        assert!((sampler.gamma_strategy().gamma(0) - 0.15).abs() < f64::EPSILON);
        assert_eq!(sampler.n_startup_trials, 20);
        assert_eq!(sampler.n_ei_candidates, 32);
    }

    #[test]
    fn test_tpe_sampler_invalid_gamma_zero() {
        let result = TpeSampler::with_config(0.0, 10, 24, None, None);
        assert!(matches!(result, Err(Error::InvalidGamma(_))));
    }

    #[test]
    fn test_tpe_sampler_invalid_gamma_one() {
        let result = TpeSampler::with_config(1.0, 10, 24, None, None);
        assert!(matches!(result, Err(Error::InvalidGamma(_))));
    }

    #[test]
    fn test_tpe_startup_random_sampling() {
        let sampler = TpeSampler::with_config(0.25, 10, 24, None, Some(42)).unwrap();
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        // With fewer than n_startup_trials, should use random sampling
        let history: Vec<CompletedTrial> = vec![];

        for i in 0..100 {
            let value = sampler.sample(&dist, i, &history);
            if let ParamValue::Float(v) = value {
                assert!((0.0..=1.0).contains(&v));
            } else {
                panic!("Expected Float value");
            }
        }
    }

    #[test]
    fn test_tpe_split_trials() {
        let sampler = TpeSampler::with_config(0.25, 10, 24, None, Some(42)).unwrap();

        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        // Create 20 trials with values 0..20
        let x_id = ParamId::new();
        let history: Vec<CompletedTrial> = (0..20)
            .map(|i| {
                create_trial(
                    i as u64,
                    f64::from(i),
                    vec![(x_id, ParamValue::Float(f64::from(i) / 20.0), dist.clone())],
                )
            })
            .collect();

        let (good, bad) = sampler.split_trials(&history);

        // With gamma=0.25 and 20 trials, should have 5 good and 15 bad
        assert_eq!(good.len(), 5);
        assert_eq!(bad.len(), 15);

        // Good trials should have lowest values
        for trial in &good {
            assert!(trial.value < 5.0);
        }
    }

    #[test]
    fn test_tpe_samples_float_with_history() {
        let sampler = TpeSampler::with_config(0.25, 5, 24, None, Some(42)).unwrap();

        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        // Create history where low values (near 0.2) are "good"
        let x_id = ParamId::new();
        let history: Vec<CompletedTrial> = (0..20)
            .map(|i| {
                let x = f64::from(i) / 20.0;
                // Objective is (x - 0.2)^2, minimized at x=0.2
                let value = (x - 0.2).powi(2);
                create_trial(
                    i as u64,
                    value,
                    vec![(x_id, ParamValue::Float(x), dist.clone())],
                )
            })
            .collect();

        // TPE should bias toward values near 0.2
        let mut samples = vec![];
        for i in 0..100 {
            let value = sampler.sample(&dist, 100 + i, &history);
            if let ParamValue::Float(v) = value {
                samples.push(v);
            }
        }

        // Calculate mean of samples - should be closer to 0.2 than 0.5
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(
            mean < 0.5,
            "Mean {mean} should be less than 0.5 (biased toward good region near 0.2)"
        );
    }

    #[test]
    fn test_tpe_categorical_sampling() {
        let sampler = TpeSampler::with_config(0.25, 5, 24, None, Some(42)).unwrap();

        let dist = Distribution::Categorical(CategoricalDistribution { n_choices: 4 });

        // Create history where category 1 is consistently good
        let cat_id = ParamId::new();
        let history: Vec<CompletedTrial> = (0..20)
            .map(|i| {
                let category = i % 4;
                // Category 1 has best (lowest) objective value
                let value = if category == 1 { 0.0 } else { 1.0 };
                create_trial(
                    i as u64,
                    value,
                    vec![(
                        cat_id,
                        ParamValue::Categorical(category as usize),
                        dist.clone(),
                    )],
                )
            })
            .collect();

        // TPE should favor category 1
        let mut counts = vec![0usize; 4];
        for i in 0..100 {
            let value = sampler.sample(&dist, 100 + i, &history);
            if let ParamValue::Categorical(idx) = value {
                counts[idx] += 1;
            }
        }

        // Category 1 should be sampled more often
        assert!(
            counts[1] > counts[0] && counts[1] > counts[2] && counts[1] > counts[3],
            "Category 1 should be most common: {counts:?}"
        );
    }

    #[test]
    fn test_tpe_int_sampling() {
        let sampler = TpeSampler::with_config(0.25, 5, 24, None, Some(42)).unwrap();

        let dist = Distribution::Int(IntDistribution {
            low: 0,
            high: 100,
            log_scale: false,
            step: None,
        });

        // Create history where values near 30 are good
        let x_id = ParamId::new();
        let history: Vec<CompletedTrial> = (0..20)
            .map(|i| {
                let x = i * 5; // 0, 5, 10, ..., 95
                let value = ((x as f64) - 30.0).powi(2);
                create_trial(
                    i as u64,
                    value,
                    vec![(x_id, ParamValue::Int(x), dist.clone())],
                )
            })
            .collect();

        // TPE should bias toward values near 30
        for i in 0..50 {
            let value = sampler.sample(&dist, 100 + i, &history);
            if let ParamValue::Int(v) = value {
                assert!((0..=100).contains(&v), "Value {v} out of range");
            } else {
                panic!("Expected Int value");
            }
        }
    }

    #[test]
    fn test_tpe_reproducibility() {
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        let x_id = ParamId::new();
        let history: Vec<CompletedTrial> = (0..20)
            .map(|i| {
                create_trial(
                    i as u64,
                    f64::from(i),
                    vec![(x_id, ParamValue::Float(f64::from(i) / 20.0), dist.clone())],
                )
            })
            .collect();

        let sampler1 = TpeSampler::with_config(0.25, 5, 24, None, Some(12345)).unwrap();
        let sampler2 = TpeSampler::with_config(0.25, 5, 24, None, Some(12345)).unwrap();

        for i in 0..10 {
            let v1 = sampler1.sample(&dist, i, &history);
            let v2 = sampler2.sample(&dist, i, &history);
            assert_eq!(v1, v2, "Samples should be identical with same seed");
        }
    }

    #[test]
    fn test_tpe_sampler_builder_default() {
        let builder = TpeSamplerBuilder::new();
        let sampler = builder.build().unwrap();
        assert!((sampler.gamma_strategy().gamma(0) - 0.25).abs() < f64::EPSILON);
        assert_eq!(sampler.n_startup_trials, 10);
        assert_eq!(sampler.n_ei_candidates, 24);
    }

    #[test]
    fn test_tpe_sampler_builder_custom() {
        let sampler = TpeSamplerBuilder::new()
            .gamma(0.15)
            .n_startup_trials(20)
            .n_ei_candidates(32)
            .seed(42)
            .build()
            .unwrap();
        assert!((sampler.gamma_strategy().gamma(0) - 0.15).abs() < f64::EPSILON);
        assert_eq!(sampler.n_startup_trials, 20);
        assert_eq!(sampler.n_ei_candidates, 32);
    }

    #[test]
    fn test_tpe_sampler_builder_via_sampler() {
        let sampler = TpeSampler::builder()
            .gamma(0.10)
            .n_startup_trials(15)
            .n_ei_candidates(48)
            .build()
            .unwrap();
        assert!((sampler.gamma_strategy().gamma(0) - 0.10).abs() < f64::EPSILON);
        assert_eq!(sampler.n_startup_trials, 15);
        assert_eq!(sampler.n_ei_candidates, 48);
    }

    #[test]
    fn test_tpe_sampler_builder_partial() {
        // Test setting only some options
        let sampler = TpeSamplerBuilder::new().gamma(0.20).build().unwrap();
        assert!((sampler.gamma_strategy().gamma(0) - 0.20).abs() < f64::EPSILON);
        assert_eq!(sampler.n_startup_trials, 10); // default
        assert_eq!(sampler.n_ei_candidates, 24); // default
    }

    #[test]
    fn test_tpe_sampler_builder_invalid_gamma() {
        let result = TpeSamplerBuilder::new().gamma(1.5).build();
        assert!(matches!(result, Err(Error::InvalidGamma(_))));
    }

    #[test]
    fn test_tpe_sampler_builder_reproducibility() {
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        let x_id = ParamId::new();
        let history: Vec<CompletedTrial> = (0..20u32)
            .map(|i| {
                create_trial(
                    u64::from(i),
                    f64::from(i),
                    vec![(x_id, ParamValue::Float(f64::from(i) / 20.0), dist.clone())],
                )
            })
            .collect();

        let sampler1 = TpeSampler::builder()
            .seed(99999)
            .n_startup_trials(5)
            .build()
            .unwrap();
        let sampler2 = TpeSampler::builder()
            .seed(99999)
            .n_startup_trials(5)
            .build()
            .unwrap();

        for i in 0..10 {
            let v1 = sampler1.sample(&dist, i, &history);
            let v2 = sampler2.sample(&dist, i, &history);
            assert_eq!(
                v1, v2,
                "Builder-created samplers with same seed should be identical"
            );
        }
    }
}
