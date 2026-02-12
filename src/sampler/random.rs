//! Random sampler — uniform independent sampling.
//!
//! [`RandomSampler`] draws each parameter value independently and uniformly
//! at random, ignoring trial history entirely. It respects log-scale and
//! step-size constraints defined by the parameter distribution.
//!
//! # When to use
//!
//! - **Baseline comparison** — run Random alongside smarter samplers to
//!   quantify their benefit.
//! - **Startup phase** — many model-based samplers (TPE, GP, CMA-ES) use
//!   random sampling for their first *n* trials before fitting a surrogate.
//! - **Very high dimensions** — when the search space is too large for
//!   structured exploration, random search with enough budget can be
//!   surprisingly competitive.
//!
//! For better uniform coverage without model fitting, consider
//! `SobolSampler` (requires the `sobol`
//! feature flag).
//!
//! # Example
//!
//! ```
//! use optimizer::prelude::*;
//! use optimizer::sampler::random::RandomSampler;
//!
//! let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
//! ```

use core::sync::atomic::{AtomicU64, Ordering};

use crate::distribution::Distribution;
use crate::multi_objective::{MultiObjectiveSampler, MultiObjectiveTrial};
use crate::param::ParamValue;
use crate::rng_util;
use crate::sampler::{CompletedTrial, Sampler};
use crate::types::Direction;

/// Uniform independent random sampler.
///
/// Sample each parameter value uniformly at random, respecting log-scale and
/// step-size constraints. Trial history is ignored — every sample is drawn
/// independently.
///
/// This is the default sampler used by [`Study::new`](crate::Study::new)
/// and during the startup phase of model-based samplers such as
/// [`TpeSampler`](super::tpe::TpeSampler).
///
/// # Examples
///
/// ```
/// use optimizer::sampler::random::RandomSampler;
///
/// // Create with default RNG
/// let sampler = RandomSampler::new();
///
/// // Create with a fixed seed for reproducibility
/// let sampler = RandomSampler::with_seed(42);
/// ```
pub struct RandomSampler {
    seed: u64,
    /// Monotonic counter to disambiguate calls with identical (`trial_id`, distribution).
    call_seq: AtomicU64,
}

impl RandomSampler {
    /// Creates a new random sampler with a default random seed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            seed: fastrand::u64(..),
            call_seq: AtomicU64::new(0),
        }
    }

    /// Creates a new random sampler with a fixed seed for reproducibility.
    ///
    /// Using the same seed will produce the same sequence of sampled values.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            seed,
            call_seq: AtomicU64::new(0),
        }
    }
}

/// Default multi-objective sampler that delegates to [`RandomSampler`].
pub(crate) struct RandomMultiObjectiveSampler(RandomSampler);

impl RandomMultiObjectiveSampler {
    pub(crate) fn new() -> Self {
        Self(RandomSampler::new())
    }
}

impl MultiObjectiveSampler for RandomMultiObjectiveSampler {
    fn sample(
        &self,
        distribution: &Distribution,
        trial_id: u64,
        _history: &[MultiObjectiveTrial],
        _directions: &[Direction],
    ) -> ParamValue {
        self.0.sample(distribution, trial_id, &[])
    }
}

impl Default for RandomSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for RandomSampler {
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn sample(
        &self,
        distribution: &Distribution,
        trial_id: u64,
        _history: &[CompletedTrial],
    ) -> ParamValue {
        let seq = self.call_seq.fetch_add(1, Ordering::Relaxed);
        let mut rng = fastrand::Rng::with_seed(rng_util::mix_seed(
            self.seed,
            trial_id,
            rng_util::distribution_fingerprint(distribution).wrapping_add(seq),
        ));

        super::common::sample_random(&mut rng, distribution)
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::distribution::{CategoricalDistribution, FloatDistribution, IntDistribution};

    #[test]
    fn test_random_sampler_float() {
        let sampler = RandomSampler::with_seed(42);
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        for i in 0..100 {
            let value = sampler.sample(&dist, i, &[]);
            if let ParamValue::Float(v) = value {
                assert!((0.0..=1.0).contains(&v));
            } else {
                panic!("Expected Float value");
            }
        }
    }

    #[test]
    fn test_random_sampler_float_log() {
        let sampler = RandomSampler::with_seed(42);
        let dist = Distribution::Float(FloatDistribution {
            low: 1e-5,
            high: 1.0,
            log_scale: true,
            step: None,
        });

        for i in 0..100 {
            let value = sampler.sample(&dist, i, &[]);
            if let ParamValue::Float(v) = value {
                assert!((1e-5..=1.0).contains(&v));
            } else {
                panic!("Expected Float value");
            }
        }
    }

    #[test]
    fn test_random_sampler_float_step() {
        let sampler = RandomSampler::with_seed(42);
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: Some(0.25),
        });

        for i in 0..100 {
            let value = sampler.sample(&dist, i, &[]);
            if let ParamValue::Float(v) = value {
                assert!((0.0..=1.0).contains(&v));
                // Check it's on the step grid
                let k = ((v - 0.0) / 0.25).round() as i64;
                let expected = 0.0 + k as f64 * 0.25;
                assert!((v - expected).abs() < 1e-10);
            } else {
                panic!("Expected Float value");
            }
        }
    }

    #[test]
    fn test_random_sampler_int() {
        let sampler = RandomSampler::with_seed(42);
        let dist = Distribution::Int(IntDistribution {
            low: 0,
            high: 10,
            log_scale: false,
            step: None,
        });

        for i in 0..100 {
            let value = sampler.sample(&dist, i, &[]);
            if let ParamValue::Int(v) = value {
                assert!((0..=10).contains(&v));
            } else {
                panic!("Expected Int value");
            }
        }
    }

    #[test]
    fn test_random_sampler_int_log() {
        let sampler = RandomSampler::with_seed(42);
        let dist = Distribution::Int(IntDistribution {
            low: 1,
            high: 1000,
            log_scale: true,
            step: None,
        });

        for i in 0..100 {
            let value = sampler.sample(&dist, i, &[]);
            if let ParamValue::Int(v) = value {
                assert!((1..=1000).contains(&v));
            } else {
                panic!("Expected Int value");
            }
        }
    }

    #[test]
    fn test_random_sampler_int_step() {
        let sampler = RandomSampler::with_seed(42);
        let dist = Distribution::Int(IntDistribution {
            low: 0,
            high: 10,
            log_scale: false,
            step: Some(2),
        });

        for i in 0..100 {
            let value = sampler.sample(&dist, i, &[]);
            if let ParamValue::Int(v) = value {
                assert!((0..=10).contains(&v));
                // Check it's on the step grid: 0, 2, 4, 6, 8, 10
                assert!(v % 2 == 0);
            } else {
                panic!("Expected Int value");
            }
        }
    }

    #[test]
    fn test_random_sampler_categorical() {
        let sampler = RandomSampler::with_seed(42);
        let dist = Distribution::Categorical(CategoricalDistribution { n_choices: 5 });

        for i in 0..100 {
            let value = sampler.sample(&dist, i, &[]);
            if let ParamValue::Categorical(idx) = value {
                assert!(idx < 5);
            } else {
                panic!("Expected Categorical value");
            }
        }
    }

    #[test]
    fn test_random_sampler_reproducibility() {
        let sampler1 = RandomSampler::with_seed(42);
        let sampler2 = RandomSampler::with_seed(42);
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        for i in 0..10 {
            let v1 = sampler1.sample(&dist, i, &[]);
            let v2 = sampler2.sample(&dist, i, &[]);
            assert_eq!(v1, v2);
        }
    }
}
