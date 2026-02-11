//! Multivariate Kernel Density Estimation for joint parameter distributions.
//!
//! This module provides a multivariate kernel density estimator that captures
//! dependencies between parameters. Unlike the univariate KDE which models each
//! parameter independently, the multivariate KDE models the joint distribution
//! to better capture correlations between parameters.

use crate::error::{Error, Result};

/// A multivariate Gaussian kernel density estimator for joint distributions.
///
/// `MultivariateKDE` estimates a joint probability density function from a set
/// of multi-dimensional samples. This is used in multivariate TPE to model
/// the joint distributions l(x) and g(x) across multiple parameters simultaneously,
/// capturing their dependencies.
///
/// # Examples
///
/// ```ignore
/// use crate::kde::MultivariateKDE;
///
/// // Create samples: 3 samples with 2 dimensions each
/// let samples = vec![
///     vec![1.0, 2.0],
///     vec![1.5, 2.5],
///     vec![2.0, 3.0],
/// ];
/// let kde = MultivariateKDE::new(samples).unwrap();
///
/// // Get dimensionality
/// assert_eq!(kde.n_dims(), 2);
/// ```
#[allow(dead_code)] // Fields and methods will be used in subsequent stories (US-003, US-004)
#[derive(Clone, Debug)]
pub(crate) struct MultivariateKDE {
    /// The sample points used to construct the KDE.
    /// Each inner Vec is one sample with `n_dims` values.
    samples: Vec<Vec<f64>>,
    /// The bandwidth (standard deviation) for each dimension.
    /// Uses a diagonal bandwidth matrix (independent bandwidths per dimension).
    bandwidths: Vec<f64>,
    /// The number of dimensions.
    n_dims: usize,
}

#[allow(dead_code)] // Methods will be used in subsequent stories (US-003, US-004)
impl MultivariateKDE {
    /// Creates a new multivariate KDE with automatic bandwidth selection using Scott's rule.
    ///
    /// Scott's rule for multivariate KDE sets bandwidth per dimension as:
    /// `h_j = n^(-1/(d+4)) * sigma_j`
    ///
    /// where n is the number of samples, d is the dimensionality, and `sigma_j` is the
    /// standard deviation of the j-th dimension.
    ///
    /// # Errors
    ///
    /// Returns `Error::EmptySamples` if `samples` is empty.
    /// Returns `Error::DimensionMismatch` if samples have inconsistent dimensions.
    /// Returns `Error::ZeroDimensions` if samples have zero dimensions.
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn new(samples: Vec<Vec<f64>>) -> Result<Self> {
        if samples.is_empty() {
            return Err(Error::EmptySamples);
        }

        let n_dims = samples[0].len();
        if n_dims == 0 {
            return Err(Error::ZeroDimensions);
        }

        // Check all samples have the same dimensionality
        for (i, sample) in samples.iter().enumerate() {
            if sample.len() != n_dims {
                return Err(Error::DimensionMismatch {
                    expected: n_dims,
                    got: sample.len(),
                    sample_index: i,
                });
            }
        }

        let bandwidths = Self::scotts_rule_multivariate(&samples, n_dims);

        Ok(Self {
            samples,
            bandwidths,
            n_dims,
        })
    }

    /// Creates a new multivariate KDE with specified bandwidths.
    ///
    /// Use this when you want explicit control over the smoothing parameters.
    ///
    /// # Errors
    ///
    /// Returns `Error::EmptySamples` if `samples` is empty.
    /// Returns `Error::DimensionMismatch` if samples have inconsistent dimensions.
    /// Returns `Error::ZeroDimensions` if samples have zero dimensions.
    /// Returns `Error::BandwidthDimensionMismatch` if bandwidths length doesn't match dimensions.
    /// Returns `Error::InvalidBandwidth` if any bandwidth is not positive.
    pub(crate) fn with_bandwidths(samples: Vec<Vec<f64>>, bandwidths: Vec<f64>) -> Result<Self> {
        if samples.is_empty() {
            return Err(Error::EmptySamples);
        }

        let n_dims = samples[0].len();
        if n_dims == 0 {
            return Err(Error::ZeroDimensions);
        }

        // Check all samples have the same dimensionality
        for (i, sample) in samples.iter().enumerate() {
            if sample.len() != n_dims {
                return Err(Error::DimensionMismatch {
                    expected: n_dims,
                    got: sample.len(),
                    sample_index: i,
                });
            }
        }

        // Check bandwidths length matches dimensions
        if bandwidths.len() != n_dims {
            return Err(Error::BandwidthDimensionMismatch {
                expected: n_dims,
                got: bandwidths.len(),
            });
        }

        // Check all bandwidths are positive
        for &bw in &bandwidths {
            if bw <= 0.0 {
                return Err(Error::InvalidBandwidth(bw));
            }
        }

        Ok(Self {
            samples,
            bandwidths,
            n_dims,
        })
    }

    /// Computes bandwidths using Scott's rule for multivariate KDE.
    ///
    /// Scott's rule for d dimensions: `h_j = n^(-1/(d+4)) * sigma_j`
    #[allow(clippy::cast_precision_loss)]
    fn scotts_rule_multivariate(samples: &[Vec<f64>], n_dims: usize) -> Vec<f64> {
        let n = samples.len() as f64;
        let d = n_dims as f64;

        // Scott's rule exponent for multivariate: -1/(d+4)
        let exponent = -1.0 / (d + 4.0);
        let scale_factor = n.powf(exponent);

        (0..n_dims)
            .map(|dim| {
                let std_dev = Self::dimension_std_dev(samples, dim);
                // For degenerate case where all samples are identical in this dimension,
                // use a small positive bandwidth
                if std_dev < f64::EPSILON {
                    1.0
                } else {
                    scale_factor * std_dev
                }
            })
            .collect()
    }

    /// Computes the sample standard deviation for a single dimension.
    #[allow(clippy::cast_precision_loss)]
    fn dimension_std_dev(samples: &[Vec<f64>], dim: usize) -> f64 {
        let n = samples.len() as f64;
        let values: Vec<f64> = samples.iter().map(|s| s[dim]).collect();
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        variance.sqrt()
    }

    /// Returns the number of dimensions.
    pub(crate) fn n_dims(&self) -> usize {
        self.n_dims
    }

    /// Returns the number of samples.
    #[allow(dead_code)] // Will be used in subsequent stories
    pub(crate) fn n_samples(&self) -> usize {
        self.samples.len()
    }

    /// Returns the bandwidths for each dimension.
    #[cfg(test)]
    pub(crate) fn bandwidths(&self) -> &[f64] {
        &self.bandwidths
    }

    /// Returns a reference to the samples.
    #[allow(dead_code)] // Will be used in subsequent stories
    pub(crate) fn samples(&self) -> &[Vec<f64>] {
        &self.samples
    }

    /// Returns the log probability density at point `x`.
    ///
    /// This computes the log-density for numerical stability, using the formula:
    ///
    /// `log f(x) = log((1/n) * Σ_i Π_j K_hj((x_j - x_ij) / h_j))`
    ///
    /// The computation uses the log-sum-exp trick for numerical stability:
    ///
    /// `log(Σ exp(a_i)) = max(a) + log(Σ exp(a_i - max(a)))`
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.n_dims`.
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn log_pdf(&self, x: &[f64]) -> f64 {
        assert_eq!(
            x.len(),
            self.n_dims,
            "Point dimension {} doesn't match KDE dimension {}",
            x.len(),
            self.n_dims
        );

        let n = self.samples.len() as f64;

        // Precompute log normalization constant for each dimension
        // For a Gaussian kernel: K_h(z) = (1/(h*sqrt(2*pi))) * exp(-0.5*z^2)
        // log(K_h(z)) = -log(h) - 0.5*log(2*pi) - 0.5*z^2
        let log_2pi = (2.0 * core::f64::consts::PI).ln();
        let log_norm_per_dim: Vec<f64> = self
            .bandwidths
            .iter()
            .map(|&h| -h.ln() - 0.5 * log_2pi)
            .collect();

        // Compute log of kernel contribution for each sample
        // log(prod_j K_hj(z_j)) = sum_j log(K_hj(z_j))
        let log_kernels: Vec<f64> = self
            .samples
            .iter()
            .map(|sample| {
                let mut log_kernel_sum = 0.0;
                for j in 0..self.n_dims {
                    let z = (x[j] - sample[j]) / self.bandwidths[j];
                    log_kernel_sum += log_norm_per_dim[j] - 0.5 * z * z;
                }
                log_kernel_sum
            })
            .collect();

        // Use log-sum-exp trick for numerical stability
        // log(sum(exp(log_kernels))) = max + log(sum(exp(log_kernels - max)))
        let max_log_kernel = log_kernels
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        // Handle case where all log_kernels are -inf (extremely unlikely point)
        if max_log_kernel.is_infinite() && max_log_kernel < 0.0 {
            return f64::NEG_INFINITY;
        }

        let sum_exp: f64 = log_kernels
            .iter()
            .map(|&lk| (lk - max_log_kernel).exp())
            .sum();

        // log((1/n) * sum) = -log(n) + log(sum)
        -n.ln() + max_log_kernel + sum_exp.ln()
    }

    /// Returns the probability density at point `x`.
    ///
    /// The density is computed as the average of multivariate Gaussian kernels
    /// centered at each sample point:
    ///
    /// `f(x) = (1/n) * Σ_i Π_j K_hj((x_j - x_ij) / h_j)`
    ///
    /// where `K_hj` is the univariate Gaussian kernel with bandwidth `h_j`.
    ///
    /// This method computes in log-space for numerical stability and then
    /// exponentiates the result.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.n_dims`.
    pub(crate) fn pdf(&self, x: &[f64]) -> f64 {
        self.log_pdf(x).exp()
    }

    /// Samples a point from the estimated joint density distribution.
    ///
    /// Sampling works by:
    /// 1. Uniformly selecting one of the kernel centers (samples)
    /// 2. Adding independent Gaussian noise to each dimension with that
    ///    dimension's bandwidth as the standard deviation
    ///
    /// This is equivalent to sampling from a mixture of multivariate Gaussians
    /// with diagonal covariance matrices, where each mixture component is
    /// centered at a sample point.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of length `n_dims` representing a sample from the KDE.
    pub(crate) fn sample(&self, rng: &mut fastrand::Rng) -> Vec<f64> {
        // Select a random sample to center the kernel on
        let idx = rng.usize(0..self.samples.len());
        let center = &self.samples[idx];

        // Add independent Gaussian noise to each dimension
        // Using Box-Muller transform for Gaussian sampling
        center
            .iter()
            .zip(self.bandwidths.iter())
            .map(|(&center_j, &bandwidth_j)| {
                let u1: f64 = rng.f64();
                let u2: f64 = rng.f64();

                // Box-Muller transform: generates standard normal variate
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos();

                // Scale by bandwidth and shift by center
                center_j + z * bandwidth_j
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multivariate_kde_new_basic() {
        let samples = vec![vec![1.0, 2.0], vec![1.5, 2.5], vec![2.0, 3.0]];
        let kde = MultivariateKDE::new(samples).unwrap();

        assert_eq!(kde.n_dims(), 2);
        assert_eq!(kde.n_samples(), 3);
        assert_eq!(kde.bandwidths().len(), 2);
    }

    #[test]
    fn test_multivariate_kde_new_single_sample() {
        let samples = vec![vec![1.0, 2.0, 3.0]];
        let kde = MultivariateKDE::new(samples).unwrap();

        assert_eq!(kde.n_dims(), 3);
        assert_eq!(kde.n_samples(), 1);
        // With single sample, bandwidths should default to 1.0 (degenerate case)
        for &bw in kde.bandwidths() {
            assert!((bw - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_multivariate_kde_new_single_dimension() {
        let samples = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let kde = MultivariateKDE::new(samples).unwrap();

        assert_eq!(kde.n_dims(), 1);
        assert_eq!(kde.n_samples(), 5);
        assert_eq!(kde.bandwidths().len(), 1);
        // Bandwidth should be positive
        assert!(kde.bandwidths()[0] > 0.0);
    }

    #[test]
    fn test_multivariate_kde_scotts_rule() {
        // Create samples with known statistics
        // 10 samples, 2 dimensions
        let samples: Vec<Vec<f64>> = (0..10)
            .map(|i| {
                let x = f64::from(i);
                vec![x, x * 2.0] // Second dimension has 2x variance
            })
            .collect();
        let kde = MultivariateKDE::new(samples).unwrap();

        // Scott's rule: h = n^(-1/(d+4)) * sigma
        // n=10, d=2: exponent = -1/6 ≈ -0.167
        // 10^(-1/6) ≈ 0.681
        // First dim std_dev ≈ 2.87, second ≈ 5.74
        // Expected bandwidths: ~1.95 and ~3.91

        let bw = kde.bandwidths();
        assert!(
            bw[0] > 1.0 && bw[0] < 3.0,
            "First bandwidth {} unexpected",
            bw[0]
        );
        assert!(
            bw[1] > 2.0 && bw[1] < 6.0,
            "Second bandwidth {} unexpected",
            bw[1]
        );
        // Second bandwidth should be approximately 2x the first
        assert!(
            (bw[1] / bw[0] - 2.0).abs() < 0.1,
            "Ratio {} not close to 2",
            bw[1] / bw[0]
        );
    }

    #[test]
    fn test_multivariate_kde_with_bandwidths() {
        let samples = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let bandwidths = vec![0.5, 1.0];
        let kde = MultivariateKDE::with_bandwidths(samples, bandwidths).unwrap();

        assert_eq!(kde.n_dims(), 2);
        assert!((kde.bandwidths()[0] - 0.5).abs() < f64::EPSILON);
        assert!((kde.bandwidths()[1] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_multivariate_kde_empty_samples() {
        let samples: Vec<Vec<f64>> = vec![];
        let result = MultivariateKDE::new(samples);
        assert!(matches!(result, Err(Error::EmptySamples)));
    }

    #[test]
    fn test_multivariate_kde_zero_dimensions() {
        let samples = vec![vec![], vec![]];
        let result = MultivariateKDE::new(samples);
        assert!(matches!(result, Err(Error::ZeroDimensions)));
    }

    #[test]
    fn test_multivariate_kde_dimension_mismatch() {
        let samples = vec![vec![1.0, 2.0], vec![3.0]]; // Second sample has wrong dimensions
        let result = MultivariateKDE::new(samples);
        assert!(matches!(
            result,
            Err(Error::DimensionMismatch {
                expected: 2,
                got: 1,
                sample_index: 1
            })
        ));
    }

    #[test]
    fn test_multivariate_kde_with_bandwidths_wrong_length() {
        let samples = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let bandwidths = vec![0.5]; // Only 1 bandwidth for 2 dimensions
        let result = MultivariateKDE::with_bandwidths(samples, bandwidths);
        assert!(matches!(
            result,
            Err(Error::BandwidthDimensionMismatch {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn test_multivariate_kde_with_bandwidths_zero() {
        let samples = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let bandwidths = vec![0.5, 0.0]; // Second bandwidth is zero
        let result = MultivariateKDE::with_bandwidths(samples, bandwidths);
        assert!(matches!(result, Err(Error::InvalidBandwidth(bw)) if bw == 0.0));
    }

    #[test]
    fn test_multivariate_kde_with_bandwidths_negative() {
        let samples = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let bandwidths = vec![0.5, -1.0]; // Negative bandwidth
        let result = MultivariateKDE::with_bandwidths(samples, bandwidths);
        assert!(
            matches!(result, Err(Error::InvalidBandwidth(bw)) if (bw - (-1.0)).abs() < f64::EPSILON)
        );
    }

    #[test]
    fn test_multivariate_kde_identical_samples() {
        // All samples identical - should handle degenerate case
        let samples = vec![vec![5.0, 10.0], vec![5.0, 10.0], vec![5.0, 10.0]];
        let kde = MultivariateKDE::new(samples).unwrap();

        // Bandwidths should default to 1.0 for degenerate dimensions
        for &bw in kde.bandwidths() {
            assert!((bw - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_multivariate_kde_high_dimensional() {
        // Test with higher dimensions
        let samples: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                let x = f64::from(i);
                vec![x, x * 0.5, x * 2.0, x * 0.1, x * 10.0]
            })
            .collect();
        let kde = MultivariateKDE::new(samples).unwrap();

        assert_eq!(kde.n_dims(), 5);
        assert_eq!(kde.n_samples(), 20);
        assert_eq!(kde.bandwidths().len(), 5);

        // All bandwidths should be positive
        for &bw in kde.bandwidths() {
            assert!(bw > 0.0);
        }
    }

    #[test]
    fn test_multivariate_kde_samples_accessor() {
        let samples = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let kde = MultivariateKDE::new(samples.clone()).unwrap();

        assert_eq!(kde.samples(), &samples);
    }

    // ==================== PDF Tests ====================

    #[test]
    fn test_multivariate_kde_pdf_basic() {
        let samples = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let kde = MultivariateKDE::new(samples).unwrap();

        // Density should be positive everywhere
        assert!(kde.pdf(&[0.0, 0.0]) > 0.0);
        assert!(kde.pdf(&[1.0, 1.0]) > 0.0);
        assert!(kde.pdf(&[2.0, 2.0]) > 0.0);
        assert!(kde.pdf(&[0.5, 0.5]) > 0.0);

        // Density should be higher near sample points
        let near_density = kde.pdf(&[1.0, 1.0]);
        let far_density = kde.pdf(&[10.0, 10.0]);
        assert!(near_density > far_density);
    }

    #[test]
    fn test_multivariate_kde_pdf_with_custom_bandwidths() {
        let samples = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let bandwidths = vec![0.5, 0.5];
        let kde = MultivariateKDE::with_bandwidths(samples, bandwidths).unwrap();

        // Density should be positive
        assert!(kde.pdf(&[0.5, 0.5]) > 0.0);
        assert!(kde.pdf(&[0.0, 0.0]) > 0.0);
    }

    #[test]
    fn test_multivariate_kde_pdf_single_sample() {
        let samples = vec![vec![5.0, 10.0]];
        let kde = MultivariateKDE::new(samples).unwrap();

        // Should have positive density near the sample
        assert!(kde.pdf(&[5.0, 10.0]) > 0.0);
        assert!(kde.pdf(&[4.5, 9.5]) > 0.0);
    }

    #[test]
    fn test_multivariate_kde_log_pdf_consistency() {
        let samples = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let kde = MultivariateKDE::new(samples).unwrap();

        // log_pdf and pdf should be consistent: exp(log_pdf(x)) == pdf(x)
        let test_points = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
            vec![3.0, 3.0],
        ];

        for point in test_points {
            let log_p = kde.log_pdf(&point);
            let p = kde.pdf(&point);
            let p_from_log = log_p.exp();
            assert!(
                (p - p_from_log).abs() < 1e-10,
                "pdf={p}, exp(log_pdf)={p_from_log}"
            );
        }
    }

    #[test]
    fn test_multivariate_kde_pdf_integrates_to_one_1d() {
        // Test with 1D case first (should match univariate KDE behavior)
        let samples = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let kde = MultivariateKDE::new(samples).unwrap();

        // Numerical integration over a wide range
        let n_points = 1000;
        let low = -10.0;
        let high = 15.0;
        let dx = (high - low) / f64::from(n_points);

        let integral: f64 = (0..n_points)
            .map(|i| {
                let x = low + (f64::from(i) + 0.5) * dx;
                kde.pdf(&[x]) * dx
            })
            .sum();

        // Should be approximately 1.0 (within numerical error)
        assert!(
            (integral - 1.0).abs() < 0.02,
            "1D integral = {integral}, expected ~1.0"
        );
    }

    #[test]
    fn test_multivariate_kde_pdf_integrates_to_one_2d() {
        // Test with 2D case
        let samples = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        let kde = MultivariateKDE::new(samples).unwrap();

        // Numerical integration over a 2D grid
        let n_points = 100; // 100x100 = 10000 points
        let low = -5.0;
        let high = 6.0;
        let dx = (high - low) / f64::from(n_points);

        let mut integral = 0.0;
        for i in 0..n_points {
            for j in 0..n_points {
                let x = low + (f64::from(i) + 0.5) * dx;
                let y = low + (f64::from(j) + 0.5) * dx;
                integral += kde.pdf(&[x, y]) * dx * dx;
            }
        }

        // Should be approximately 1.0 (within numerical error)
        // 2D integration has more error, so use larger tolerance
        assert!(
            (integral - 1.0).abs() < 0.05,
            "2D integral = {integral}, expected ~1.0"
        );
    }

    #[test]
    fn test_multivariate_kde_pdf_symmetry() {
        // With symmetric samples, PDF should be symmetric
        let samples = vec![
            vec![1.0, 0.0],
            vec![-1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, -1.0],
        ];
        let kde = MultivariateKDE::new(samples).unwrap();

        // Density at symmetric points should be equal
        let d1 = kde.pdf(&[0.5, 0.0]);
        let d2 = kde.pdf(&[-0.5, 0.0]);
        assert!(
            (d1 - d2).abs() < 1e-10,
            "Symmetric points have different densities: {d1} vs {d2}"
        );

        let d3 = kde.pdf(&[0.0, 0.5]);
        let d4 = kde.pdf(&[0.0, -0.5]);
        assert!(
            (d3 - d4).abs() < 1e-10,
            "Symmetric points have different densities: {d3} vs {d4}"
        );
    }

    #[test]
    fn test_multivariate_kde_pdf_high_dimensional() {
        // Test with higher dimensions
        let samples: Vec<Vec<f64>> = (0..10)
            .map(|i| {
                let x = f64::from(i) * 0.1;
                vec![x, x, x, x, x] // 5D
            })
            .collect();
        let kde = MultivariateKDE::new(samples).unwrap();

        // Density should be positive
        assert!(kde.pdf(&[0.5, 0.5, 0.5, 0.5, 0.5]) > 0.0);
        assert!(kde.pdf(&[0.0, 0.0, 0.0, 0.0, 0.0]) > 0.0);

        // Log PDF should be finite
        let log_p = kde.log_pdf(&[0.5, 0.5, 0.5, 0.5, 0.5]);
        assert!(log_p.is_finite());
    }

    #[test]
    fn test_multivariate_kde_pdf_numerical_stability() {
        // Test numerical stability with points far from samples
        let samples = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let kde = MultivariateKDE::new(samples).unwrap();

        // Very far point should have very small but non-negative density
        let far_pdf = kde.pdf(&[100.0, 100.0]);
        assert!(far_pdf >= 0.0);
        assert!(far_pdf.is_finite() || far_pdf == 0.0);

        // Log PDF should be finite (or -inf for zero density)
        let far_log_pdf = kde.log_pdf(&[100.0, 100.0]);
        assert!(far_log_pdf.is_finite() || far_log_pdf.is_infinite());
    }

    #[test]
    #[should_panic(expected = "Point dimension")]
    fn test_multivariate_kde_pdf_wrong_dimension() {
        let samples = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let kde = MultivariateKDE::new(samples).unwrap();

        // Should panic with wrong dimension
        let _ = kde.pdf(&[0.0]); // Only 1 value for 2D KDE
    }

    // ==================== Sampling Tests ====================

    #[test]
    fn test_multivariate_kde_sample_basic() {
        let samples = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let kde = MultivariateKDE::new(samples).unwrap();
        let mut rng = fastrand::Rng::new();

        // Sample should have correct dimensionality
        let sample = kde.sample(&mut rng);
        assert_eq!(sample.len(), 2);
    }

    #[test]
    fn test_multivariate_kde_sample_in_reasonable_range() {
        let samples = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ];
        let kde = MultivariateKDE::new(samples).unwrap();
        let mut rng = fastrand::Rng::new();

        // Samples should generally be in a reasonable range around the data
        for _ in 0..100 {
            let s = kde.sample(&mut rng);
            // With high probability, samples should be within a few bandwidths
            // of the data range. Use a generous range to avoid flaky tests.
            assert!(
                s[0] > -10.0 && s[0] < 15.0,
                "Sample dimension 0: {} outside expected range",
                s[0]
            );
            assert!(
                s[1] > -10.0 && s[1] < 15.0,
                "Sample dimension 1: {} outside expected range",
                s[1]
            );
        }
    }

    #[test]
    fn test_multivariate_kde_sample_single_sample() {
        // When KDE has only one sample, all samples should be centered around it
        let samples = vec![vec![5.0, 10.0]];
        let kde = MultivariateKDE::new(samples).unwrap();
        let mut rng = fastrand::Rng::new();

        // Generate many samples and check they cluster around (5.0, 10.0)
        let n_samples = 100;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for _ in 0..n_samples {
            let s = kde.sample(&mut rng);
            sum_x += s[0];
            sum_y += s[1];
        }

        let mean_x = sum_x / f64::from(n_samples);
        let mean_y = sum_y / f64::from(n_samples);

        // Mean should be close to the single sample point
        // With 100 samples and bandwidth=1.0, we expect mean within ~0.3 of center
        assert!(
            (mean_x - 5.0).abs() < 1.0,
            "Mean x={mean_x}, expected close to 5.0"
        );
        assert!(
            (mean_y - 10.0).abs() < 1.0,
            "Mean y={mean_y}, expected close to 10.0"
        );
    }

    #[test]
    fn test_multivariate_kde_sample_high_dimensional() {
        // Test sampling in higher dimensions
        let samples: Vec<Vec<f64>> = (0..10)
            .map(|i| {
                let x = f64::from(i) * 0.5;
                vec![x, x * 2.0, x * 0.5, x + 1.0, x - 1.0] // 5D
            })
            .collect();
        let kde = MultivariateKDE::new(samples).unwrap();
        let mut rng = fastrand::Rng::new();

        // Sample should have correct dimensionality
        for _ in 0..50 {
            let sample = kde.sample(&mut rng);
            assert_eq!(sample.len(), 5);
            // All values should be finite
            for &val in &sample {
                assert!(val.is_finite(), "Sample value is not finite: {val}");
            }
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_multivariate_kde_sample_respects_bandwidth() {
        // Create samples all at origin with large custom bandwidth
        let data = vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]];
        let bandwidths = vec![0.1, 10.0]; // Small bandwidth in x, large in y
        let kde = MultivariateKDE::with_bandwidths(data, bandwidths).unwrap();
        let mut rng = fastrand::Rng::new();

        // Generate samples and check variance in each dimension
        let n_samples = 1000;
        let mut values_x: Vec<f64> = Vec::with_capacity(n_samples);
        let mut values_y: Vec<f64> = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let s = kde.sample(&mut rng);
            values_x.push(s[0]);
            values_y.push(s[1]);
        }

        // Compute sample variances
        let n = n_samples as f64;
        let mean_x: f64 = values_x.iter().sum::<f64>() / n;
        let mean_y: f64 = values_y.iter().sum::<f64>() / n;

        let var_x: f64 = values_x.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>() / n;
        let var_y: f64 = values_y.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>() / n;

        // Variance should be approximately bandwidth^2
        // x: bandwidth=0.1, expected variance ~0.01
        // y: bandwidth=10.0, expected variance ~100.0
        assert!(
            var_x < 0.05,
            "X variance {var_x} too large for bandwidth 0.1"
        );
        assert!(
            var_y > 50.0 && var_y < 200.0,
            "Y variance {var_y} unexpected for bandwidth 10.0"
        );
    }

    #[test]
    fn test_multivariate_kde_sample_distribution_shape() {
        // Create samples along a diagonal line
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ];
        let kde = MultivariateKDE::new(data).unwrap();
        let mut rng = fastrand::Rng::new();

        // Sample many points and verify the mean is near the center
        let n_samples = 500;
        let mut sum = [0.0, 0.0];

        for _ in 0..n_samples {
            let s = kde.sample(&mut rng);
            sum[0] += s[0];
            sum[1] += s[1];
        }

        let mean_x = sum[0] / f64::from(n_samples);
        let mean_y = sum[1] / f64::from(n_samples);

        // Mean should be close to (2.0, 2.0) - the center of the samples
        assert!(
            (mean_x - 2.0).abs() < 0.5,
            "Mean x={mean_x}, expected close to 2.0"
        );
        assert!(
            (mean_y - 2.0).abs() < 0.5,
            "Mean y={mean_y}, expected close to 2.0"
        );
    }

    #[test]
    fn test_multivariate_kde_sample_deterministic_with_seeded_rng() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let kde = MultivariateKDE::new(data).unwrap();

        // Use a seeded RNG for reproducibility
        let mut rng1 = fastrand::Rng::with_seed(42);
        let mut rng2 = fastrand::Rng::with_seed(42);

        // Same seed should produce same samples
        let result1 = kde.sample(&mut rng1);
        let result2 = kde.sample(&mut rng2);

        assert!(
            (result1[0] - result2[0]).abs() < f64::EPSILON,
            "Samples with same seed differ in dimension 0"
        );
        assert!(
            (result1[1] - result2[1]).abs() < f64::EPSILON,
            "Samples with same seed differ in dimension 1"
        );
    }
}
