//! Multi-Objective Tree-Parzen Estimator (MOTPE) sampler.
//!
//! MOTPE extends TPE to multi-objective optimization by replacing the gamma-based
//! split with Pareto non-dominated sorting. This lets the sampler propose
//! parameters that push the Pareto front forward across all objectives
//! simultaneously.
//!
//! # Algorithm
//!
//! In single-objective TPE, trials are sorted by value and split at a gamma
//! percentile into good/bad groups. MOTPE replaces this with:
//!
//! 1. Compute non-dominated sorting on all completed trials.
//! 2. Use the Pareto front (rank 0) as "good" trials.
//! 3. Use dominated trials (rank 1+) as "bad" trials.
//! 4. Build KDE l(x) from good, g(x) from bad.
//! 5. Sample candidates and score by l(x)/g(x).
//!
//! # When to use
//!
//! - You have 2+ objectives and want model-guided search (not pure evolutionary).
//! - Your objectives are relatively smooth and continuous.
//! - You want a Pareto-aware version of TPE without the overhead of full
//!   population-based algorithms like NSGA-II or NSGA-III.
//!
//! For single-objective problems, use [`TpeSampler`](super::tpe::TpeSampler) instead.
//! For many-objective (3+) problems with reference-point decomposition, consider
//! NSGA-III or MOEA/D.
//!
//! # Configuration
//!
//! - `n_startup_trials` — number of random trials before MOTPE kicks in (default: 11)
//! - `n_ei_candidates` — candidates evaluated per sample (default: 24)
//! - `kde_bandwidth` — optional fixed KDE bandwidth; `None` uses Scott's rule
//! - `seed` — optional seed for reproducibility
//!
//! # Examples
//!
//! ```
//! use optimizer::Direction;
//! use optimizer::multi_objective::MultiObjectiveStudy;
//! use optimizer::parameter::{FloatParam, Parameter};
//! use optimizer::sampler::motpe::MotpeSampler;
//!
//! let sampler = MotpeSampler::builder().seed(42).build();
//! let study =
//!     MultiObjectiveStudy::with_sampler(vec![Direction::Minimize, Direction::Minimize], sampler);
//!
//! let x = FloatParam::new(0.0, 1.0);
//! study
//!     .optimize(30, |trial: &mut optimizer::Trial| {
//!         let xv = x.suggest(trial)?;
//!         Ok::<_, optimizer::Error>(vec![xv, 1.0 - xv])
//!     })
//!     .unwrap();
//!
//! let front = study.pareto_front();
//! assert!(!front.is_empty());
//! ```

use parking_lot::Mutex;

use crate::distribution::Distribution;
use crate::kde::KernelDensityEstimator;
use crate::multi_objective::{MultiObjectiveSampler, MultiObjectiveTrial};
use crate::param::ParamValue;
use crate::types::{Direction, TrialState};
use crate::{pareto, rng_util};

/// Multi-Objective TPE (MOTPE) sampler for multi-objective Bayesian optimization.
///
/// Use Pareto non-dominated sorting to split completed trials into "good"
/// (non-dominated, rank 0) and "bad" (dominated) groups, then fit kernel
/// density estimators to each group and sample new points that maximize
/// l(x)/g(x).
///
/// During the startup phase (fewer than `n_startup_trials` completed),
/// MOTPE falls back to random sampling.
///
/// # When to use
///
/// Use `MotpeSampler` when optimizing 2+ objectives and you want a
/// model-guided sampler that adapts proposals based on the current
/// Pareto front. For single-objective problems, use
/// [`TpeSampler`](super::tpe::TpeSampler) instead.
///
/// # Examples
///
/// ```
/// use optimizer::Direction;
/// use optimizer::multi_objective::MultiObjectiveStudy;
/// use optimizer::parameter::{FloatParam, Parameter};
/// use optimizer::sampler::motpe::MotpeSampler;
///
/// let sampler = MotpeSampler::builder()
///     .n_startup_trials(10)
///     .n_ei_candidates(24)
///     .seed(42)
///     .build();
///
/// let study =
///     MultiObjectiveStudy::with_sampler(vec![Direction::Minimize, Direction::Minimize], sampler);
///
/// let x = FloatParam::new(0.0, 1.0);
/// study
///     .optimize(30, |trial: &mut optimizer::Trial| {
///         let xv = x.suggest(trial)?;
///         Ok::<_, optimizer::Error>(vec![xv, 1.0 - xv])
///     })
///     .unwrap();
///
/// let front = study.pareto_front();
/// assert!(!front.is_empty());
/// ```
pub struct MotpeSampler {
    /// Number of trials before MOTPE kicks in (uses random sampling before this).
    n_startup_trials: usize,
    /// Number of candidate samples to evaluate when selecting the next point.
    n_ei_candidates: usize,
    /// Optional fixed bandwidth for KDE. If None, uses Scott's rule.
    kde_bandwidth: Option<f64>,
    /// Thread-safe RNG for sampling.
    rng: Mutex<fastrand::Rng>,
}

impl MotpeSampler {
    /// Creates a new MOTPE sampler with default settings.
    ///
    /// Defaults:
    /// - `n_startup_trials`: 11
    /// - `n_ei_candidates`: 24
    /// - `kde_bandwidth`: None (Scott's rule)
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_startup_trials: 11,
            n_ei_candidates: 24,
            kde_bandwidth: None,
            rng: Mutex::new(fastrand::Rng::new()),
        }
    }

    /// Creates a new MOTPE sampler with a fixed seed.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            n_startup_trials: 11,
            n_ei_candidates: 24,
            kde_bandwidth: None,
            rng: Mutex::new(fastrand::Rng::with_seed(seed)),
        }
    }

    /// Creates a builder for configuring a MOTPE sampler.
    #[must_use]
    pub fn builder() -> MotpeSamplerBuilder {
        MotpeSamplerBuilder::new()
    }

    /// Splits trials into good (non-dominated) and bad (dominated) groups
    /// using Pareto non-dominated sorting.
    fn split_trials<'a>(
        history: &'a [MultiObjectiveTrial],
        directions: &[Direction],
    ) -> (Vec<&'a MultiObjectiveTrial>, Vec<&'a MultiObjectiveTrial>) {
        let complete: Vec<(usize, &MultiObjectiveTrial)> = history
            .iter()
            .enumerate()
            .filter(|(_, t)| t.state == TrialState::Complete)
            .collect();

        if complete.is_empty() {
            return (vec![], vec![]);
        }

        let values: Vec<Vec<f64>> = complete.iter().map(|(_, t)| t.values.clone()).collect();
        let constraints: Vec<Vec<f64>> = complete
            .iter()
            .map(|(_, t)| t.constraints.clone())
            .collect();
        let has_constraints = constraints.iter().any(|c| !c.is_empty());

        let fronts = if has_constraints {
            pareto::fast_non_dominated_sort_constrained(&values, directions, &constraints)
        } else {
            pareto::fast_non_dominated_sort(&values, directions)
        };

        if fronts.is_empty() {
            return (vec![], vec![]);
        }

        // Front 0 = good (non-dominated), everything else = bad
        let good: Vec<&MultiObjectiveTrial> = fronts[0].iter().map(|&i| complete[i].1).collect();

        let bad: Vec<&MultiObjectiveTrial> = fronts[1..]
            .iter()
            .flatten()
            .map(|&i| complete[i].1)
            .collect();

        (good, bad)
    }

    /// Samples uniformly from a distribution (used during startup phase).
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::unused_self
    )]
    fn sample_uniform(distribution: &Distribution, rng: &mut fastrand::Rng) -> ParamValue {
        match distribution {
            Distribution::Float(d) => {
                let value = if d.log_scale {
                    let log_low = d.low.ln();
                    let log_high = d.high.ln();
                    rng_util::f64_range(rng, log_low, log_high).exp()
                } else if let Some(step) = d.step {
                    let n_steps = ((d.high - d.low) / step).floor() as i64;
                    let k = rng.i64(0..=n_steps);
                    d.low + (k as f64) * step
                } else {
                    rng_util::f64_range(rng, d.low, d.high)
                };
                ParamValue::Float(value)
            }
            Distribution::Int(d) => {
                let value = if d.log_scale {
                    let log_low = (d.low as f64).ln();
                    let log_high = (d.high as f64).ln();
                    let raw = rng_util::f64_range(rng, log_low, log_high).exp().round() as i64;
                    raw.clamp(d.low, d.high)
                } else if let Some(step) = d.step {
                    let n_steps = (d.high - d.low) / step;
                    let k = rng.i64(0..=n_steps);
                    d.low + k * step
                } else {
                    rng.i64(d.low..=d.high)
                };
                ParamValue::Int(value)
            }
            Distribution::Categorical(d) => ParamValue::Categorical(rng.usize(0..d.n_choices)),
        }
    }

    /// Samples using TPE for float distributions.
    #[allow(clippy::too_many_arguments)]
    fn sample_tpe_float(
        &self,
        low: f64,
        high: f64,
        log_scale: bool,
        step: Option<f64>,
        good_values: Vec<f64>,
        bad_values: Vec<f64>,
        rng: &mut fastrand::Rng,
    ) -> f64 {
        // Transform to internal space (log space if needed)
        let (internal_low, internal_high, good_internal, bad_internal) = if log_scale {
            let i_low = low.ln();
            let i_high = high.ln();
            let g: Vec<f64> = good_values.iter().map(|&v| v.ln()).collect();
            let b: Vec<f64> = bad_values.iter().map(|&v| v.ln()).collect();
            (i_low, i_high, g, b)
        } else {
            (low, high, good_values, bad_values)
        };

        // Fit KDEs to good and bad groups
        let l_kde = match self.kde_bandwidth {
            Some(bw) => KernelDensityEstimator::with_bandwidth(good_internal, bw),
            None => KernelDensityEstimator::new(good_internal),
        };
        let g_kde = match self.kde_bandwidth {
            Some(bw) => KernelDensityEstimator::with_bandwidth(bad_internal, bw),
            None => KernelDensityEstimator::new(bad_internal),
        };

        // If KDE construction fails, fall back to uniform sampling
        let (Ok(l_kde), Ok(g_kde)) = (l_kde, g_kde) else {
            return rng_util::f64_range(rng, low, high);
        };

        // Generate candidates from l(x) and select the one with best l(x)/g(x)
        let mut best_candidate = internal_low;
        let mut best_ratio = f64::NEG_INFINITY;

        for _ in 0..self.n_ei_candidates {
            let candidate = l_kde.sample(rng).clamp(internal_low, internal_high);

            let l_density = l_kde.pdf(candidate);
            let g_density = g_kde.pdf(candidate);

            let ratio = if g_density < f64::EPSILON {
                if l_density > f64::EPSILON {
                    f64::INFINITY
                } else {
                    0.0
                }
            } else {
                l_density / g_density
            };

            if ratio > best_ratio {
                best_ratio = ratio;
                best_candidate = candidate;
            }
        }

        // Transform back from internal space
        let mut value = if log_scale {
            best_candidate.exp()
        } else {
            best_candidate
        };

        // Apply step constraint if present
        if let Some(step) = step {
            let k = ((value - low) / step).round();
            value = low + k * step;
        }

        value.clamp(low, high)
    }

    /// Samples using TPE for integer distributions.
    #[allow(
        clippy::too_many_arguments,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation
    )]
    fn sample_tpe_int(
        &self,
        low: i64,
        high: i64,
        log_scale: bool,
        step: Option<i64>,
        good_values: &[i64],
        bad_values: &[i64],
        rng: &mut fastrand::Rng,
    ) -> i64 {
        let good_floats: Vec<f64> = good_values.iter().map(|&v| v as f64).collect();
        let bad_floats: Vec<f64> = bad_values.iter().map(|&v| v as f64).collect();

        let float_value = self.sample_tpe_float(
            low as f64,
            high as f64,
            log_scale,
            step.map(|s| s as f64),
            good_floats,
            bad_floats,
            rng,
        );

        let int_value = float_value.round() as i64;
        let int_value = if let Some(step) = step {
            let k = ((int_value - low) as f64 / step as f64).round() as i64;
            low + k * step
        } else {
            int_value
        };

        int_value.clamp(low, high)
    }

    /// Samples using TPE for categorical distributions.
    #[allow(clippy::cast_precision_loss)]
    fn sample_tpe_categorical(
        n_choices: usize,
        good_indices: &[usize],
        bad_indices: &[usize],
        rng: &mut fastrand::Rng,
    ) -> usize {
        let mut good_counts = vec![0usize; n_choices];
        let mut bad_counts = vec![0usize; n_choices];

        for &idx in good_indices {
            if idx < n_choices {
                good_counts[idx] += 1;
            }
        }
        for &idx in bad_indices {
            if idx < n_choices {
                bad_counts[idx] += 1;
            }
        }

        // Laplace smoothing
        let good_total = good_indices.len() as f64 + n_choices as f64;
        let bad_total = bad_indices.len() as f64 + n_choices as f64;

        let mut weights = vec![0.0f64; n_choices];
        for i in 0..n_choices {
            let l_prob = (good_counts[i] as f64 + 1.0) / good_total;
            let g_prob = (bad_counts[i] as f64 + 1.0) / bad_total;
            weights[i] = l_prob / g_prob;
        }

        // Sample proportionally to weights
        let total_weight: f64 = weights.iter().sum();
        let threshold = rng.f64() * total_weight;

        let mut cumulative = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if cumulative >= threshold {
                return i;
            }
        }

        n_choices - 1
    }
}

impl Default for MotpeSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiObjectiveSampler for MotpeSampler {
    #[allow(clippy::too_many_lines)]
    fn sample(
        &self,
        distribution: &Distribution,
        _trial_id: u64,
        history: &[MultiObjectiveTrial],
        directions: &[Direction],
    ) -> ParamValue {
        let mut rng = self.rng.lock();

        // Fall back to random sampling during startup phase
        let n_complete = history
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .count();
        if n_complete < self.n_startup_trials {
            return Self::sample_uniform(distribution, &mut rng);
        }

        // Split trials into good (Pareto front) and bad (dominated)
        let (good_trials, bad_trials) = Self::split_trials(history, directions);

        if good_trials.is_empty() || bad_trials.is_empty() {
            return Self::sample_uniform(distribution, &mut rng);
        }

        match distribution {
            Distribution::Float(d) => {
                let good_values: Vec<f64> = good_trials
                    .iter()
                    .flat_map(|t| t.params.values())
                    .filter_map(|v| match v {
                        ParamValue::Float(f) => Some(*f),
                        _ => None,
                    })
                    .filter(|&v| v >= d.low && v <= d.high)
                    .collect();

                let bad_values: Vec<f64> = bad_trials
                    .iter()
                    .flat_map(|t| t.params.values())
                    .filter_map(|v| match v {
                        ParamValue::Float(f) => Some(*f),
                        _ => None,
                    })
                    .filter(|&v| v >= d.low && v <= d.high)
                    .collect();

                if good_values.is_empty() || bad_values.is_empty() {
                    return Self::sample_uniform(distribution, &mut rng);
                }

                let value = self.sample_tpe_float(
                    d.low,
                    d.high,
                    d.log_scale,
                    d.step,
                    good_values,
                    bad_values,
                    &mut rng,
                );
                ParamValue::Float(value)
            }
            Distribution::Int(d) => {
                let good_values: Vec<i64> = good_trials
                    .iter()
                    .flat_map(|t| t.params.values())
                    .filter_map(|v| match v {
                        ParamValue::Int(i) => Some(*i),
                        _ => None,
                    })
                    .filter(|&v| v >= d.low && v <= d.high)
                    .collect();

                let bad_values: Vec<i64> = bad_trials
                    .iter()
                    .flat_map(|t| t.params.values())
                    .filter_map(|v| match v {
                        ParamValue::Int(i) => Some(*i),
                        _ => None,
                    })
                    .filter(|&v| v >= d.low && v <= d.high)
                    .collect();

                if good_values.is_empty() || bad_values.is_empty() {
                    return Self::sample_uniform(distribution, &mut rng);
                }

                let value = self.sample_tpe_int(
                    d.low,
                    d.high,
                    d.log_scale,
                    d.step,
                    &good_values,
                    &bad_values,
                    &mut rng,
                );
                ParamValue::Int(value)
            }
            Distribution::Categorical(d) => {
                let good_indices: Vec<usize> = good_trials
                    .iter()
                    .flat_map(|t| t.params.values())
                    .filter_map(|v| match v {
                        ParamValue::Categorical(i) => Some(*i),
                        _ => None,
                    })
                    .filter(|&i| i < d.n_choices)
                    .collect();

                let bad_indices: Vec<usize> = bad_trials
                    .iter()
                    .flat_map(|t| t.params.values())
                    .filter_map(|v| match v {
                        ParamValue::Categorical(i) => Some(*i),
                        _ => None,
                    })
                    .filter(|&i| i < d.n_choices)
                    .collect();

                if good_indices.is_empty() || bad_indices.is_empty() {
                    return Self::sample_uniform(distribution, &mut rng);
                }

                let index = Self::sample_tpe_categorical(
                    d.n_choices,
                    &good_indices,
                    &bad_indices,
                    &mut rng,
                );
                ParamValue::Categorical(index)
            }
        }
    }
}

/// Builder for configuring a [`MotpeSampler`].
///
/// # Defaults
///
/// - `n_startup_trials`: 11
/// - `n_ei_candidates`: 24
/// - `kde_bandwidth`: None (Scott's rule)
/// - `seed`: None (OS entropy)
///
/// # Examples
///
/// ```
/// use optimizer::sampler::motpe::MotpeSamplerBuilder;
///
/// let sampler = MotpeSamplerBuilder::new()
///     .n_startup_trials(15)
///     .n_ei_candidates(32)
///     .seed(42)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct MotpeSamplerBuilder {
    n_startup_trials: usize,
    n_ei_candidates: usize,
    kde_bandwidth: Option<f64>,
    seed: Option<u64>,
}

impl MotpeSamplerBuilder {
    /// Creates a new builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_startup_trials: 11,
            n_ei_candidates: 24,
            kde_bandwidth: None,
            seed: None,
        }
    }

    /// Sets the number of startup trials before MOTPE sampling begins.
    #[must_use]
    pub fn n_startup_trials(mut self, n: usize) -> Self {
        self.n_startup_trials = n;
        self
    }

    /// Sets the number of EI candidates to evaluate per sample.
    #[must_use]
    pub fn n_ei_candidates(mut self, n: usize) -> Self {
        self.n_ei_candidates = n;
        self
    }

    /// Sets a fixed bandwidth for the kernel density estimator.
    ///
    /// By default, Scott's rule is used for automatic bandwidth selection.
    #[must_use]
    pub fn kde_bandwidth(mut self, bandwidth: f64) -> Self {
        self.kde_bandwidth = Some(bandwidth);
        self
    }

    /// Sets a seed for reproducible sampling.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Builds the configured [`MotpeSampler`].
    #[must_use]
    pub fn build(self) -> MotpeSampler {
        let rng = match self.seed {
            Some(s) => fastrand::Rng::with_seed(s),
            None => fastrand::Rng::new(),
        };

        MotpeSampler {
            n_startup_trials: self.n_startup_trials,
            n_ei_candidates: self.n_ei_candidates,
            kde_bandwidth: self.kde_bandwidth,
            rng: Mutex::new(rng),
        }
    }
}

impl Default for MotpeSamplerBuilder {
    fn default() -> Self {
        Self::new()
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
    use crate::trial::Trial;

    fn create_mo_trial(
        id: u64,
        values: Vec<f64>,
        params: Vec<(ParamId, ParamValue, Distribution)>,
    ) -> MultiObjectiveTrial {
        let mut param_map = HashMap::new();
        let mut dist_map = HashMap::new();
        let label_map = HashMap::new();
        for (param_id, pv, dist) in params {
            param_map.insert(param_id, pv);
            dist_map.insert(param_id, dist);
        }
        MultiObjectiveTrial {
            id,
            params: param_map,
            distributions: dist_map,
            param_labels: label_map,
            values,
            state: TrialState::Complete,
            user_attrs: HashMap::new(),
            constraints: Vec::new(),
        }
    }

    #[test]
    fn test_motpe_startup_random_sampling() {
        let sampler = MotpeSampler::with_seed(42);
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });
        let directions = [Direction::Minimize, Direction::Minimize];

        // With no history, should use random sampling
        let history: Vec<MultiObjectiveTrial> = vec![];
        for _ in 0..50 {
            let value = sampler.sample(&dist, 0, &history, &directions);
            if let ParamValue::Float(v) = value {
                assert!((0.0..=1.0).contains(&v));
            } else {
                panic!("Expected Float value");
            }
        }
    }

    #[test]
    fn test_motpe_split_pareto() {
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });
        let directions = [Direction::Minimize, Direction::Minimize];
        let x_id = ParamId::new();

        // Create trials: Pareto front = {(0.1, 0.9), (0.5, 0.5), (0.9, 0.1)}
        // Dominated = {(0.6, 0.8), (0.8, 0.7)}
        let history = vec![
            create_mo_trial(
                0,
                vec![0.1, 0.9],
                vec![(x_id, ParamValue::Float(0.1), dist.clone())],
            ),
            create_mo_trial(
                1,
                vec![0.5, 0.5],
                vec![(x_id, ParamValue::Float(0.5), dist.clone())],
            ),
            create_mo_trial(
                2,
                vec![0.9, 0.1],
                vec![(x_id, ParamValue::Float(0.9), dist.clone())],
            ),
            create_mo_trial(
                3,
                vec![0.6, 0.8],
                vec![(x_id, ParamValue::Float(0.6), dist.clone())],
            ),
            create_mo_trial(
                4,
                vec![0.8, 0.7],
                vec![(x_id, ParamValue::Float(0.8), dist.clone())],
            ),
        ];

        let (good, bad) = MotpeSampler::split_trials(&history, &directions);
        assert_eq!(good.len(), 3, "Pareto front should have 3 members");
        assert_eq!(bad.len(), 2, "2 dominated trials");
    }

    #[test]
    fn test_motpe_samples_float() {
        let sampler = MotpeSampler::builder()
            .n_startup_trials(5)
            .n_ei_candidates(24)
            .seed(42)
            .build();

        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });
        let directions = [Direction::Minimize, Direction::Minimize];
        let x_id = ParamId::new();

        // Build history where values near 0.3 are on the Pareto front
        let mut history = Vec::new();
        for i in 0..20 {
            let x = f64::from(i) / 20.0;
            // Pareto front: f1 = (x - 0.3)^2, f2 = (x - 0.3)^2 + 0.1
            // Best solutions cluster around x = 0.3
            let f1 = (x - 0.3).powi(2);
            let f2 = (x - 0.7).powi(2);
            history.push(create_mo_trial(
                i as u64,
                vec![f1, f2],
                vec![(x_id, ParamValue::Float(x), dist.clone())],
            ));
        }

        // MOTPE should produce values within [0, 1]
        for i in 0..50 {
            let value = sampler.sample(&dist, 100 + i, &history, &directions);
            if let ParamValue::Float(v) = value {
                assert!((0.0..=1.0).contains(&v), "Value {v} out of range");
            } else {
                panic!("Expected Float value");
            }
        }
    }

    #[test]
    fn test_motpe_int_sampling() {
        let sampler = MotpeSampler::builder().n_startup_trials(5).seed(42).build();

        let dist = Distribution::Int(IntDistribution {
            low: 0,
            high: 100,
            log_scale: false,
            step: None,
        });
        let directions = [Direction::Minimize, Direction::Minimize];
        let x_id = ParamId::new();

        let mut history = Vec::new();
        for i in 0..20 {
            let x = i * 5;
            let f1 = ((x as f64) - 30.0).powi(2);
            let f2 = ((x as f64) - 70.0).powi(2);
            history.push(create_mo_trial(
                i as u64,
                vec![f1, f2],
                vec![(x_id, ParamValue::Int(x), dist.clone())],
            ));
        }

        for i in 0..50 {
            let value = sampler.sample(&dist, 100 + i, &history, &directions);
            if let ParamValue::Int(v) = value {
                assert!((0..=100).contains(&v), "Value {v} out of range");
            } else {
                panic!("Expected Int value");
            }
        }
    }

    #[test]
    fn test_motpe_categorical_sampling() {
        let sampler = MotpeSampler::builder().n_startup_trials(5).seed(42).build();

        let dist = Distribution::Categorical(CategoricalDistribution { n_choices: 3 });
        let directions = [Direction::Minimize, Direction::Minimize];
        let cat_id = ParamId::new();

        // Category 1 is on the Pareto front, others are dominated
        let mut history = Vec::new();
        for i in 0..15 {
            let category = i % 3;
            let (f1, f2) = match category {
                0 => (0.8, 0.8), // dominated
                1 => (0.1, 0.9), // Pareto front
                2 => (0.9, 0.1), // Pareto front
                _ => unreachable!(),
            };
            history.push(create_mo_trial(
                i as u64,
                vec![f1, f2],
                vec![(
                    cat_id,
                    ParamValue::Categorical(category as usize),
                    dist.clone(),
                )],
            ));
        }

        let mut counts = vec![0usize; 3];
        for i in 0..200 {
            let value = sampler.sample(&dist, 100 + i, &history, &directions);
            if let ParamValue::Categorical(idx) = value {
                assert!(idx < 3, "Category {idx} out of range");
                counts[idx] += 1;
            } else {
                panic!("Expected Categorical value");
            }
        }

        // Categories 1 and 2 (on Pareto front) should dominate category 0
        assert!(
            counts[1] + counts[2] > counts[0],
            "Pareto-front categories should be sampled more: {counts:?}"
        );
    }

    #[test]
    fn test_motpe_reproducibility() {
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });
        let directions = [Direction::Minimize, Direction::Minimize];
        let x_id = ParamId::new();

        let history: Vec<MultiObjectiveTrial> = (0..20)
            .map(|i| {
                let x = f64::from(i) / 20.0;
                create_mo_trial(
                    i as u64,
                    vec![x, 1.0 - x],
                    vec![(x_id, ParamValue::Float(x), dist.clone())],
                )
            })
            .collect();

        let sampler1 = MotpeSampler::builder()
            .seed(12345)
            .n_startup_trials(5)
            .build();
        let sampler2 = MotpeSampler::builder()
            .seed(12345)
            .n_startup_trials(5)
            .build();

        for i in 0..10 {
            let v1 = sampler1.sample(&dist, i, &history, &directions);
            let v2 = sampler2.sample(&dist, i, &history, &directions);
            assert_eq!(v1, v2, "Samples should be identical with same seed");
        }
    }

    #[test]
    fn test_motpe_with_study() {
        use crate::multi_objective::MultiObjectiveStudy;
        use crate::parameter::{FloatParam, Parameter};

        let sampler = MotpeSampler::builder().seed(42).build();
        let study = MultiObjectiveStudy::with_sampler(
            vec![Direction::Minimize, Direction::Minimize],
            sampler,
        );

        let x = FloatParam::new(0.0, 1.0);
        study
            .optimize(30, |trial: &mut Trial| {
                let xv = x.suggest(trial)?;
                Ok::<_, crate::Error>(vec![xv, 1.0 - xv])
            })
            .unwrap();

        let front = study.pareto_front();
        assert!(!front.is_empty(), "Should have Pareto-optimal solutions");

        // All front solutions should have values summing to ~1.0
        for trial in &front {
            let sum: f64 = trial.values.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Pareto front values should sum to ~1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_motpe_builder_defaults() {
        let sampler = MotpeSamplerBuilder::new().build();
        assert_eq!(sampler.n_startup_trials, 11);
        assert_eq!(sampler.n_ei_candidates, 24);
        assert!(sampler.kde_bandwidth.is_none());
    }

    #[test]
    fn test_motpe_builder_custom() {
        let sampler = MotpeSamplerBuilder::new()
            .n_startup_trials(20)
            .n_ei_candidates(48)
            .kde_bandwidth(0.5)
            .seed(99)
            .build();
        assert_eq!(sampler.n_startup_trials, 20);
        assert_eq!(sampler.n_ei_candidates, 48);
        assert_eq!(sampler.kde_bandwidth, Some(0.5));
    }
}
