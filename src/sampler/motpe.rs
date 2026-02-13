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

use core::sync::atomic::{AtomicU64, Ordering};

use crate::distribution::Distribution;
use crate::multi_objective::{MultiObjectiveSampler, MultiObjectiveTrial};
use crate::param::ParamValue;
use crate::sampler::common;
use crate::sampler::tpe::common as tpe_common;
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
    /// Base seed for deterministic per-call RNG derivation (no mutex needed).
    seed: u64,
    /// Monotonic counter to disambiguate calls with identical (`trial_id`, distribution).
    call_seq: AtomicU64,
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
            seed: fastrand::u64(..),
            call_seq: AtomicU64::new(0),
        }
    }

    /// Creates a new MOTPE sampler with a fixed seed.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            n_startup_trials: 11,
            n_ei_candidates: 24,
            kde_bandwidth: None,
            seed,
            call_seq: AtomicU64::new(0),
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
}

impl Default for MotpeSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl MotpeSampler {
    fn sample_float(
        &self,
        d: &crate::distribution::FloatDistribution,
        good_trials: &[&MultiObjectiveTrial],
        bad_trials: &[&MultiObjectiveTrial],
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
        good_trials: &[&MultiObjectiveTrial],
        bad_trials: &[&MultiObjectiveTrial],
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
        good_trials: &[&MultiObjectiveTrial],
        bad_trials: &[&MultiObjectiveTrial],
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

impl MultiObjectiveSampler for MotpeSampler {
    fn sample(
        &self,
        distribution: &Distribution,
        trial_id: u64,
        history: &[MultiObjectiveTrial],
        directions: &[Direction],
    ) -> ParamValue {
        let seq = self.call_seq.fetch_add(1, Ordering::Relaxed);
        let mut rng = fastrand::Rng::with_seed(rng_util::mix_seed(
            self.seed,
            trial_id,
            rng_util::distribution_fingerprint(distribution).wrapping_add(seq),
        ));

        // Fall back to random sampling during startup phase
        let n_complete = history
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .count();
        if n_complete < self.n_startup_trials {
            return common::sample_random(&mut rng, distribution);
        }

        // Split trials into good (Pareto front) and bad (dominated)
        let (good_trials, bad_trials) = Self::split_trials(history, directions);

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
        MotpeSampler {
            n_startup_trials: self.n_startup_trials,
            n_ei_candidates: self.n_ei_candidates,
            kde_bandwidth: self.kde_bandwidth,
            seed: self.seed.unwrap_or_else(|| fastrand::u64(..)),
            call_seq: AtomicU64::new(0),
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
        for i in 0..50 {
            let value = sampler.sample(&dist, i, &history, &directions);
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
