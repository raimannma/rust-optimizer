//! Kernel Density Estimation for continuous parameters.
//!
//! This module provides a Gaussian kernel density estimator used by the TPE
//! sampler to model probability distributions over good and bad trial regions.

use rand::Rng;

use crate::error::{Error, Result};

/// A Gaussian kernel density estimator for continuous distributions.
///
/// KDE estimates a probability density function from a set of samples by
/// placing Gaussian kernels centered at each sample point. This is used
/// in TPE to model the distributions l(x) (good trials) and g(x) (bad trials).
///
/// # Examples
///
/// ```ignore
/// use crate::kde::KernelDensityEstimator;
///
/// let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let kde = KernelDensityEstimator::new(samples);
///
/// // Get probability density at a point
/// let density = kde.pdf(2.5);
/// assert!(density > 0.0);
///
/// // Sample from the estimated distribution
/// let mut rng = rand::rng();
/// let sample = kde.sample(&mut rng);
/// ```
#[derive(Clone, Debug)]
pub(crate) struct KernelDensityEstimator {
    /// The sample points used to construct the KDE.
    samples: Vec<f64>,
    /// The bandwidth (standard deviation) of the Gaussian kernels.
    bandwidth: f64,
}

impl KernelDensityEstimator {
    /// Creates a new KDE with automatic bandwidth selection using Scott's rule.
    ///
    /// Scott's rule sets bandwidth = n^(-1/5) * `std_dev`, which works well
    /// for unimodal distributions close to normal.
    ///
    /// # Errors
    ///
    /// Returns `Error::EmptySamples` if `samples` is empty.
    pub(crate) fn new(samples: Vec<f64>) -> Result<Self> {
        if samples.is_empty() {
            return Err(Error::EmptySamples);
        }

        let bandwidth = Self::scotts_rule(&samples);
        Ok(Self { samples, bandwidth })
    }

    /// Creates a new KDE with a specified bandwidth.
    ///
    /// Use this when you want explicit control over the smoothing parameter.
    ///
    /// # Errors
    ///
    /// Returns `Error::EmptySamples` if `samples` is empty.
    /// Returns `Error::InvalidBandwidth` if `bandwidth` is not positive.
    pub(crate) fn with_bandwidth(samples: Vec<f64>, bandwidth: f64) -> Result<Self> {
        if samples.is_empty() {
            return Err(Error::EmptySamples);
        }
        if bandwidth <= 0.0 {
            return Err(Error::InvalidBandwidth(bandwidth));
        }

        Ok(Self { samples, bandwidth })
    }

    /// Computes bandwidth using Scott's rule.
    ///
    /// Scott's rule: h = n^(-1/5) * sigma
    /// where sigma is the sample standard deviation.
    #[allow(clippy::cast_precision_loss)]
    fn scotts_rule(samples: &[f64]) -> f64 {
        let n = samples.len() as f64;
        let std_dev = Self::sample_std_dev(samples);

        // For degenerate case where all samples are identical,
        // use a small positive bandwidth
        if std_dev < f64::EPSILON {
            return 1.0;
        }

        n.powf(-0.2) * std_dev
    }

    /// Computes the sample standard deviation.
    #[allow(clippy::cast_precision_loss)]
    fn sample_std_dev(samples: &[f64]) -> f64 {
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        variance.sqrt()
    }

    /// Returns the probability density at point `x`.
    ///
    /// The density is computed as the average of Gaussian kernels centered
    /// at each sample point:
    ///
    /// f(x) = (1/n) * `sum_i` K((x - `x_i`) / h)
    ///
    /// where K is the standard Gaussian kernel and h is the bandwidth.
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn pdf(&self, x: f64) -> f64 {
        let n = self.samples.len() as f64;
        let inv_bandwidth = 1.0 / self.bandwidth;
        let normalization = inv_bandwidth / (2.0 * core::f64::consts::PI).sqrt();

        let density: f64 = self
            .samples
            .iter()
            .map(|&xi| {
                let z = (x - xi) * inv_bandwidth;
                normalization * (-0.5 * z * z).exp()
            })
            .sum();

        density / n
    }

    /// Samples a value from the estimated density distribution.
    ///
    /// Sampling works by:
    /// 1. Uniformly selecting one of the kernel centers (samples)
    /// 2. Adding Gaussian noise with the bandwidth as standard deviation
    pub(crate) fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        // Select a random sample to center the kernel on
        let idx = rng.random_range(0..self.samples.len());
        let center = self.samples[idx];

        // Add Gaussian noise with bandwidth as standard deviation
        // Using Box-Muller transform for Gaussian sampling
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();

        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos();
        center + z * self.bandwidth
    }

    /// Returns the bandwidth of this KDE.
    #[cfg(test)]
    pub(crate) fn bandwidth(&self) -> f64 {
        self.bandwidth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kde_pdf_basic() {
        let samples = vec![0.0, 1.0, 2.0];
        let kde = KernelDensityEstimator::new(samples).unwrap();

        // Density should be positive everywhere
        assert!(kde.pdf(0.0) > 0.0);
        assert!(kde.pdf(1.0) > 0.0);
        assert!(kde.pdf(2.0) > 0.0);

        // Density should be higher near sample points
        let mid_density = kde.pdf(1.0);
        let far_density = kde.pdf(10.0);
        assert!(mid_density > far_density);
    }

    #[test]
    fn test_kde_pdf_integrates_to_one() {
        let samples = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let kde = KernelDensityEstimator::new(samples).unwrap();

        // Numerical integration over a wide range
        let n_points = 10000;
        let low = -10.0;
        let high = 15.0;
        let dx = (high - low) / f64::from(n_points);

        let integral: f64 = (0..n_points)
            .map(|i| {
                let x = low + (f64::from(i) + 0.5) * dx;
                kde.pdf(x) * dx
            })
            .sum();

        // Should be approximately 1.0 (within numerical error)
        assert!(
            (integral - 1.0).abs() < 0.01,
            "Integral = {integral}, expected ~1.0"
        );
    }

    #[test]
    fn test_kde_with_bandwidth() {
        let samples = vec![0.0, 1.0, 2.0];
        let kde = KernelDensityEstimator::with_bandwidth(samples, 0.5).unwrap();

        assert!((kde.bandwidth() - 0.5).abs() < f64::EPSILON);
        assert!(kde.pdf(1.0) > 0.0);
    }

    #[test]
    fn test_kde_sample_in_reasonable_range() {
        let samples = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let kde = KernelDensityEstimator::new(samples).unwrap();
        let mut rng = rand::rng();

        // Samples should generally be in a reasonable range around the data
        for _ in 0..100 {
            let s = kde.sample(&mut rng);
            // With high probability, samples should be within a few bandwidths
            // of the data range. Use a generous range to avoid flaky tests.
            assert!(s > -10.0 && s < 15.0, "Sample {s} outside expected range");
        }
    }

    #[test]
    fn test_kde_single_sample() {
        let samples = vec![5.0];
        let kde = KernelDensityEstimator::new(samples).unwrap();

        // Should have positive density near the sample
        assert!(kde.pdf(5.0) > 0.0);
        assert!(kde.pdf(4.5) > 0.0);
    }

    #[test]
    fn test_kde_identical_samples() {
        let samples = vec![3.0, 3.0, 3.0, 3.0];
        let kde = KernelDensityEstimator::new(samples).unwrap();

        // Should handle degenerate case with identical samples
        assert!(kde.bandwidth() > 0.0);
        assert!(kde.pdf(3.0) > 0.0);
    }

    #[test]
    fn test_scotts_rule_bandwidth() {
        let samples = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let kde = KernelDensityEstimator::new(samples).unwrap();

        // n = 10, n^(-1/5) ≈ 0.631
        // std_dev ≈ 2.87
        // bandwidth ≈ 0.631 * 2.87 ≈ 1.81
        let bandwidth = kde.bandwidth();
        assert!(
            bandwidth > 1.0 && bandwidth < 3.0,
            "Bandwidth {bandwidth} outside expected range"
        );
    }

    #[test]
    fn test_kde_empty_samples() {
        let samples: Vec<f64> = vec![];
        let result = KernelDensityEstimator::new(samples);
        assert!(matches!(result, Err(Error::EmptySamples)));
    }

    #[test]
    fn test_kde_zero_bandwidth() {
        let samples = vec![1.0, 2.0, 3.0];
        let result = KernelDensityEstimator::with_bandwidth(samples, 0.0);
        assert!(matches!(result, Err(Error::InvalidBandwidth(_))));
    }

    #[test]
    fn test_kde_negative_bandwidth() {
        let samples = vec![1.0, 2.0, 3.0];
        let result = KernelDensityEstimator::with_bandwidth(samples, -1.0);
        assert!(matches!(result, Err(Error::InvalidBandwidth(_))));
    }
}
