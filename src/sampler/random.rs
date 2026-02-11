//! Random sampler implementation.

use parking_lot::Mutex;

use crate::distribution::Distribution;
use crate::param::ParamValue;
use crate::rng_util;
use crate::sampler::{CompletedTrial, Sampler};

/// A simple random sampler that samples uniformly from distributions.
///
/// This sampler ignores the trial history and samples uniformly at random,
/// respecting log scale and step size constraints. It serves as a baseline
/// sampler and is used during the startup phase of more sophisticated samplers.
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
    rng: Mutex<fastrand::Rng>,
}

impl RandomSampler {
    /// Creates a new random sampler with a default random seed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rng: Mutex::new(fastrand::Rng::new()),
        }
    }

    /// Creates a new random sampler with a fixed seed for reproducibility.
    ///
    /// Using the same seed will produce the same sequence of sampled values.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: Mutex::new(fastrand::Rng::with_seed(seed)),
        }
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
        _trial_id: u64,
        _history: &[CompletedTrial],
    ) -> ParamValue {
        let mut rng = self.rng.lock();

        match distribution {
            Distribution::Float(d) => {
                let value = if d.log_scale {
                    // Sample uniformly in log space
                    let log_low = d.low.ln();
                    let log_high = d.high.ln();
                    let log_value = rng_util::f64_range(&mut rng, log_low, log_high);
                    log_value.exp()
                } else if let Some(step) = d.step {
                    // Sample from step grid
                    let n_steps = ((d.high - d.low) / step).floor() as i64;
                    let k = rng.i64(0..=n_steps);
                    d.low + (k as f64) * step
                } else {
                    // Uniform sampling
                    rng_util::f64_range(&mut rng, d.low, d.high)
                };
                ParamValue::Float(value)
            }
            Distribution::Int(d) => {
                let value = if d.log_scale {
                    // Sample uniformly in log space, then round
                    let log_low = (d.low as f64).ln();
                    let log_high = (d.high as f64).ln();
                    let log_value = rng_util::f64_range(&mut rng, log_low, log_high);
                    let raw = log_value.exp().round() as i64;
                    // Clamp to bounds since rounding might push outside
                    raw.clamp(d.low, d.high)
                } else if let Some(step) = d.step {
                    // Sample from step grid
                    let n_steps = (d.high - d.low) / step;
                    let k = rng.i64(0..=n_steps);
                    d.low + k * step
                } else {
                    // Uniform sampling
                    rng.i64(d.low..=d.high)
                };
                ParamValue::Int(value)
            }
            Distribution::Categorical(d) => {
                let index = rng.usize(0..d.n_choices);
                ParamValue::Categorical(index)
            }
        }
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

        for _ in 0..100 {
            let value = sampler.sample(&dist, 0, &[]);
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

        for _ in 0..100 {
            let value = sampler.sample(&dist, 0, &[]);
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

        for _ in 0..100 {
            let value = sampler.sample(&dist, 0, &[]);
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

        for _ in 0..100 {
            let value = sampler.sample(&dist, 0, &[]);
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

        for _ in 0..100 {
            let value = sampler.sample(&dist, 0, &[]);
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

        for _ in 0..100 {
            let value = sampler.sample(&dist, 0, &[]);
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

        for _ in 0..100 {
            let value = sampler.sample(&dist, 0, &[]);
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

        for _ in 0..10 {
            let v1 = sampler1.sample(&dist, 0, &[]);
            let v2 = sampler2.sample(&dist, 0, &[]);
            assert_eq!(v1, v2);
        }
    }
}
