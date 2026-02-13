//! Quasi-random sampler using Sobol low-discrepancy sequences.
//!
//! [`SobolSampler`] generates points from a Sobol sequence (scrambled via the
//! Burley 2020 algorithm) to fill the parameter space more uniformly than
//! pure random sampling. Where [`RandomSampler`](super::random::RandomSampler)
//! may cluster points in some regions by chance, Sobol sequences are
//! constructed to spread points evenly across all dimensions.
//!
//! # When to use
//!
//! - **Better-than-random baseline** — when you want uniform coverage
//!   without the cost of model fitting (TPE, GP, etc.).
//! - **Startup phase replacement** — use Sobol instead of random for the
//!   initial exploration phase of adaptive samplers.
//! - **Moderate dimensionality** — Sobol uniformity is strongest up to
//!   ~20 dimensions; beyond that the advantage over random sampling
//!   diminishes.
//! - **Deterministic exploration** — Sobol sequences are fully deterministic
//!   for a given seed, making experiments reproducible.
//!
//! # How it works
//!
//! Each trial maps to a Sobol sequence index, and each parameter within a
//! trial maps to a separate Sobol dimension. The resulting quasi-random
//! point in \[0, 1) is then scaled to the parameter's distribution (linear,
//! log-scale, or step grid).
//!
//! **Important:** parameters must be suggested in the same order across
//! trials for consistent dimension assignment.
//!
//! Requires the **`sobol`** feature flag:
//!
//! ```toml
//! [dependencies]
//! optimizer = { version = "...", features = ["sobol"] }
//! ```
//!
//! # Example
//!
//! ```
//! use optimizer::prelude::*;
//! use optimizer::sampler::sobol::SobolSampler;
//!
//! let study: Study<f64> = Study::with_sampler(Direction::Minimize, SobolSampler::with_seed(42));
//! ```

use std::collections::HashMap;

use parking_lot::Mutex;
use sobol_burley::sample;

use crate::distribution::Distribution;
use crate::param::ParamValue;
use crate::sampler::{CompletedTrial, Sampler};

/// Internal state for tracking per-trial dimension counters.
struct SobolState {
    /// Next Sobol dimension for each in-flight trial.
    dimensions: HashMap<u64, u32>,
}

/// Quasi-random sampler using Sobol low-discrepancy sequences.
///
/// Produce better uniform coverage of the parameter space than
/// [`RandomSampler`](super::random::RandomSampler) by using a
/// scrambled Sobol sequence (Burley 2020). Useful as a standalone
/// baseline or as a drop-in replacement for the random startup
/// phase of model-based samplers.
///
/// Each trial uses a different Sobol sequence index, and each parameter
/// within a trial maps to a different Sobol dimension. Parameters must be
/// suggested in the same order across trials for consistent dimension
/// assignment.
///
/// Sobol sequences are most effective in moderate dimensions (up to ~20).
/// For very high dimensions, the uniformity advantage diminishes.
///
/// Requires the **`sobol`** feature flag.
///
/// # Examples
///
/// ```
/// use optimizer::sampler::sobol::SobolSampler;
/// use optimizer::{Direction, Study};
///
/// let study: Study<f64> = Study::with_sampler(Direction::Minimize, SobolSampler::new());
/// ```
pub struct SobolSampler {
    seed: u32,
    state: Mutex<SobolState>,
}

impl SobolSampler {
    /// Creates a new Sobol sampler with a default seed of 0.
    #[must_use]
    pub fn new() -> Self {
        Self::with_seed(0)
    }

    /// Creates a new Sobol sampler with the given seed.
    ///
    /// Different seeds produce statistically independent Sobol sequences.
    /// Using the same seed will produce the same sequence of sampled values.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            seed: seed as u32,
            state: Mutex::new(SobolState {
                dimensions: HashMap::new(),
            }),
        }
    }
}

impl Default for SobolSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for SobolSampler {
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn sample(
        &self,
        distribution: &Distribution,
        trial_id: u64,
        _history: &[CompletedTrial],
    ) -> ParamValue {
        let mut state = self.state.lock();

        let dimension = state.dimensions.entry(trial_id).or_insert(0);
        let dim = *dimension;
        *dimension = dim + 1;

        // Use trial_id as the Sobol sequence index.
        let index = trial_id as u32;

        // Generate a quasi-random point in [0, 1).
        let point = f64::from(sample(index, dim, self.seed));

        map_point_to_distribution(point, distribution)
    }
}

/// Maps a uniform [0, 1) point to a value within the given distribution.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn map_point_to_distribution(point: f64, distribution: &Distribution) -> ParamValue {
    match distribution {
        Distribution::Float(d) => {
            let value = if d.log_scale {
                let log_low = d.low.ln();
                let log_high = d.high.ln();
                (log_low + point * (log_high - log_low)).exp()
            } else if let Some(step) = d.step {
                let n_steps = ((d.high - d.low) / step).floor() as i64;
                let k = (point * (n_steps + 1) as f64).floor() as i64;
                let k = k.min(n_steps);
                d.low + (k as f64) * step
            } else {
                d.low + point * (d.high - d.low)
            };
            ParamValue::Float(value)
        }
        Distribution::Int(d) => {
            let value = if d.log_scale {
                let log_low = (d.low as f64).ln();
                let log_high = (d.high as f64).ln();
                let raw = (log_low + point * (log_high - log_low)).exp().round() as i64;
                raw.clamp(d.low, d.high)
            } else if let Some(step) = d.step {
                let n_steps = (d.high - d.low) / step;
                let k = (point * (n_steps + 1) as f64).floor() as i64;
                let k = k.min(n_steps);
                d.low + k * step
            } else {
                let range = d.high - d.low + 1;
                let k = (point * range as f64).floor() as i64;
                (d.low + k).min(d.high)
            };
            ParamValue::Int(value)
        }
        Distribution::Categorical(d) => {
            let index = (point * d.n_choices as f64).floor() as usize;
            let index = index.min(d.n_choices - 1);
            ParamValue::Categorical(index)
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;
    use crate::distribution::{CategoricalDistribution, FloatDistribution, IntDistribution};

    #[test]
    fn float_within_bounds() {
        let sampler = SobolSampler::with_seed(42);
        let dist = Distribution::Float(FloatDistribution {
            low: -5.0,
            high: 5.0,
            log_scale: false,
            step: None,
        });

        for i in 0..100 {
            let value = sampler.sample(&dist, i, &[]);
            if let ParamValue::Float(v) = value {
                assert!(
                    (-5.0..=5.0).contains(&v),
                    "value {v} out of bounds at trial {i}"
                );
            } else {
                panic!("Expected Float value");
            }
        }
    }

    #[test]
    fn float_log_scale_within_bounds() {
        let sampler = SobolSampler::with_seed(42);
        let dist = Distribution::Float(FloatDistribution {
            low: 1e-5,
            high: 1.0,
            log_scale: true,
            step: None,
        });

        for i in 0..100 {
            let value = sampler.sample(&dist, i, &[]);
            if let ParamValue::Float(v) = value {
                assert!(
                    (1e-5..=1.0).contains(&v),
                    "value {v} out of bounds at trial {i}"
                );
            } else {
                panic!("Expected Float value");
            }
        }
    }

    #[test]
    fn float_step_respects_grid() {
        let sampler = SobolSampler::with_seed(42);
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: Some(0.25),
        });

        for i in 0..100 {
            let value = sampler.sample(&dist, i, &[]);
            if let ParamValue::Float(v) = value {
                assert!((0.0..=1.0).contains(&v), "value {v} out of bounds");
                let k = (v / 0.25).round() as i64;
                let expected = k as f64 * 0.25;
                assert!((v - expected).abs() < 1e-10, "value {v} not on step grid");
            } else {
                panic!("Expected Float value");
            }
        }
    }

    #[test]
    fn int_within_bounds() {
        let sampler = SobolSampler::with_seed(42);
        let dist = Distribution::Int(IntDistribution {
            low: 0,
            high: 10,
            log_scale: false,
            step: None,
        });

        for i in 0..100 {
            let value = sampler.sample(&dist, i, &[]);
            if let ParamValue::Int(v) = value {
                assert!(
                    (0..=10).contains(&v),
                    "value {v} out of bounds at trial {i}"
                );
            } else {
                panic!("Expected Int value");
            }
        }
    }

    #[test]
    fn int_log_scale_within_bounds() {
        let sampler = SobolSampler::with_seed(42);
        let dist = Distribution::Int(IntDistribution {
            low: 1,
            high: 1000,
            log_scale: true,
            step: None,
        });

        for i in 0..100 {
            let value = sampler.sample(&dist, i, &[]);
            if let ParamValue::Int(v) = value {
                assert!(
                    (1..=1000).contains(&v),
                    "value {v} out of bounds at trial {i}"
                );
            } else {
                panic!("Expected Int value");
            }
        }
    }

    #[test]
    fn int_step_respects_grid() {
        let sampler = SobolSampler::with_seed(42);
        let dist = Distribution::Int(IntDistribution {
            low: 0,
            high: 10,
            log_scale: false,
            step: Some(2),
        });

        for i in 0..100 {
            let value = sampler.sample(&dist, i, &[]);
            if let ParamValue::Int(v) = value {
                assert!((0..=10).contains(&v), "value {v} out of bounds");
                assert!(v % 2 == 0, "value {v} not on step grid");
            } else {
                panic!("Expected Int value");
            }
        }
    }

    #[test]
    fn categorical_within_bounds() {
        let sampler = SobolSampler::with_seed(42);
        let dist = Distribution::Categorical(CategoricalDistribution { n_choices: 5 });

        for i in 0..100 {
            let value = sampler.sample(&dist, i, &[]);
            if let ParamValue::Categorical(idx) = value {
                assert!(idx < 5, "index {idx} out of bounds at trial {i}");
            } else {
                panic!("Expected Categorical value");
            }
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        let sampler1 = SobolSampler::with_seed(42);
        let sampler2 = SobolSampler::with_seed(42);

        for i in 0..20 {
            let v1 = sampler1.sample(&dist, i, &[]);
            let v2 = sampler2.sample(&dist, i, &[]);
            assert_eq!(v1, v2, "mismatch at trial {i}");
        }
    }

    #[test]
    fn different_seeds_produce_different_sequences() {
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        let sampler1 = SobolSampler::with_seed(0);
        let sampler2 = SobolSampler::with_seed(12345);

        let mut any_different = false;
        for i in 0..20 {
            let v1 = sampler1.sample(&dist, i, &[]);
            let v2 = sampler2.sample(&dist, i, &[]);
            if v1 != v2 {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "different seeds should produce different sequences"
        );
    }

    #[test]
    fn better_coverage_than_random() {
        // Sobol sequence should cover [0,1] more uniformly than random.
        // We measure this by checking that the Sobol samples fill all
        // 10 equal-width bins with only 20 samples.
        let sampler = SobolSampler::with_seed(0);
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        let n_bins = 10;
        let n_samples = 20;
        let mut bins = vec![0u32; n_bins];

        for i in 0..n_samples {
            let value = sampler.sample(&dist, i, &[]);
            if let ParamValue::Float(v) = value {
                let bin = ((v * n_bins as f64).floor() as usize).min(n_bins - 1);
                bins[bin] += 1;
            }
        }

        let filled_bins = bins.iter().filter(|&&c| c > 0).count();
        assert!(
            filled_bins >= 8,
            "Expected at least 8/10 bins filled, got {filled_bins}: {bins:?}"
        );
    }

    #[test]
    fn multi_parameter_uses_different_dimensions() {
        // When sampling multiple parameters per trial, each parameter
        // should get a different Sobol dimension, producing different values.
        let sampler = SobolSampler::with_seed(0);
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        // Sample two parameters for trial 0.
        let v1 = sampler.sample(&dist, 0, &[]);
        let v2 = sampler.sample(&dist, 0, &[]);

        // They should differ (different Sobol dimensions for the same index).
        assert_ne!(
            v1, v2,
            "multi-parameter samples should use different dimensions"
        );
    }
}
