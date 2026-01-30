//! Grid search sampler implementation.
//!
//! `GridSearchSampler` performs exhaustive grid search over the parameter space,
//! systematically evaluating all combinations of discretized parameter values.

use std::collections::HashMap;

use parking_lot::Mutex;

use crate::distribution::{
    CategoricalDistribution, Distribution, FloatDistribution, IntDistribution,
};
use crate::param::ParamValue;
use crate::sampler::{CompletedTrial, Sampler};

/// Generates grid points for an integer distribution.
///
/// # Behavior
///
/// - If `step` is `Some(s)`: generates points at `low, low+s, low+2*s, ...` up to `high`.
/// - If `step` is `None` and `log_scale` is `false`: generates `n_points` evenly spaced
///   integers from `low` to `high`.
/// - If `step` is `None` and `log_scale` is `true`: generates `n_points` evenly spaced
///   in log space, rounded to integers.
///
/// All grid points are clamped to `[low, high]` bounds and deduplicated.
///
/// # Arguments
///
/// * `dist` - The integer distribution defining bounds, step, and log scale.
/// * `n_points` - Number of points to generate when auto-discretizing (ignored if step is set).
///
/// # Returns
///
/// A vector of unique grid points in ascending order.
#[must_use]
pub fn generate_int_grid_points(dist: &IntDistribution, n_points: usize) -> Vec<i64> {
    let low = dist.low;
    let high = dist.high;

    if low > high {
        return vec![];
    }

    if low == high {
        return vec![low];
    }

    let points: Vec<i64> = if let Some(step) = dist.step {
        // Generate points at low, low+step, low+2*step, ... up to high
        if step <= 0 {
            return vec![low];
        }
        let mut result = Vec::new();
        let mut current = low;
        while current <= high {
            result.push(current);
            current = current.saturating_add(step);
            // Prevent infinite loop if saturating_add doesn't change the value
            if result.last() == Some(&current) {
                break;
            }
        }
        result
    } else if dist.log_scale {
        // Generate n_points evenly spaced in log space, rounded to integers
        // For log scale, low must be positive
        if low <= 0 {
            // Fall back to linear for non-positive values
            generate_linear_int_points(low, high, n_points)
        } else {
            generate_log_int_points(low, high, n_points)
        }
    } else {
        // Generate n_points evenly spaced integers from low to high
        generate_linear_int_points(low, high, n_points)
    };

    // Clamp all points to [low, high] and deduplicate
    let mut clamped: Vec<i64> = points.into_iter().map(|p| p.clamp(low, high)).collect();

    // Remove duplicates while preserving order
    clamped.sort_unstable();
    clamped.dedup();

    clamped
}

/// Generates evenly spaced integers from low to high (linear scale).
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn generate_linear_int_points(low: i64, high: i64, n_points: usize) -> Vec<i64> {
    if n_points == 0 {
        return vec![];
    }
    if n_points == 1 {
        return vec![low];
    }

    let range = high - low;
    let mut result = Vec::with_capacity(n_points);

    for i in 0..n_points {
        // Calculate position as a fraction of the range
        let fraction = i as f64 / (n_points - 1) as f64;
        let value = low as f64 + fraction * range as f64;
        result.push(value.round() as i64);
    }

    result
}

/// Generates evenly spaced integers in log space.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn generate_log_int_points(low: i64, high: i64, n_points: usize) -> Vec<i64> {
    debug_assert!(low > 0, "log scale requires positive low bound");

    if n_points == 0 {
        return vec![];
    }
    if n_points == 1 {
        return vec![low];
    }

    let log_low = (low as f64).ln();
    let log_high = (high as f64).ln();
    let mut result = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let fraction = i as f64 / (n_points - 1) as f64;
        let log_value = log_low + fraction * (log_high - log_low);
        let value = log_value.exp().round() as i64;
        result.push(value);
    }

    result
}

/// Generates grid points for a float distribution.
///
/// # Behavior
///
/// - If `step` is `Some(s)`: generates points at `low, low+s, low+2*s, ...` up to `high`.
///   Step size overrides `log_scale` behavior.
/// - If `step` is `None` and `log_scale` is `false`: generates `n_points` evenly spaced
///   floats from `low` to `high`.
/// - If `step` is `None` and `log_scale` is `true`: generates `n_points` evenly spaced
///   in log space.
///
/// All grid points are within `[low, high]` bounds.
///
/// # Arguments
///
/// * `dist` - The float distribution defining bounds, step, and log scale.
/// * `n_points` - Number of points to generate when auto-discretizing (ignored if step is set).
///
/// # Returns
///
/// A vector of grid points in ascending order.
#[must_use]
pub fn generate_float_grid_points(dist: &FloatDistribution, n_points: usize) -> Vec<f64> {
    let low = dist.low;
    let high = dist.high;

    if low > high {
        return vec![];
    }

    if (low - high).abs() < f64::EPSILON {
        return vec![low];
    }

    let points: Vec<f64> = if let Some(step) = dist.step {
        // Step overrides log_scale - generate points at low, low+step, low+2*step, ... up to high
        if step <= 0.0 {
            return vec![low];
        }
        let mut result = Vec::new();
        let mut current = low;
        while current <= high + f64::EPSILON {
            result.push(current.clamp(low, high));
            current += step;
            // Prevent infinite loop from floating point issues
            if result.len() > 1_000_000 {
                break;
            }
        }
        result
    } else if dist.log_scale {
        // Generate n_points evenly spaced in log space
        // For log scale, low must be positive
        if low <= 0.0 {
            // Fall back to linear for non-positive values
            generate_linear_float_points(low, high, n_points)
        } else {
            generate_log_float_points(low, high, n_points)
        }
    } else {
        // Generate n_points evenly spaced floats from low to high
        generate_linear_float_points(low, high, n_points)
    };

    // Clamp all points to [low, high] bounds
    points.into_iter().map(|p| p.clamp(low, high)).collect()
}

/// Generates evenly spaced floats from low to high (linear scale).
#[allow(clippy::cast_precision_loss)]
fn generate_linear_float_points(low: f64, high: f64, n_points: usize) -> Vec<f64> {
    if n_points == 0 {
        return vec![];
    }
    if n_points == 1 {
        return vec![low];
    }

    let range = high - low;
    let mut result = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let fraction = i as f64 / (n_points - 1) as f64;
        let value = low + fraction * range;
        result.push(value);
    }

    result
}

/// Generates evenly spaced floats in log space.
#[allow(clippy::cast_precision_loss)]
fn generate_log_float_points(low: f64, high: f64, n_points: usize) -> Vec<f64> {
    debug_assert!(low > 0.0, "log scale requires positive low bound");

    if n_points == 0 {
        return vec![];
    }
    if n_points == 1 {
        return vec![low];
    }

    let log_low = low.ln();
    let log_high = high.ln();
    let mut result = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let fraction = i as f64 / (n_points - 1) as f64;
        let log_value = log_low + fraction * (log_high - log_low);
        let value = log_value.exp();
        result.push(value);
    }

    result
}

/// Generates grid points for a categorical distribution.
///
/// Returns a vector containing all choice indices `[0, 1, 2, ..., n_choices-1]`.
/// All choices are included in the grid for exhaustive evaluation.
///
/// # Arguments
///
/// * `dist` - The categorical distribution defining the number of choices.
///
/// # Returns
///
/// A vector of all choice indices from 0 to `n_choices - 1`.
///
/// # Examples
///
/// ```ignore
/// use optimizer::sampler::grid::generate_categorical_grid_points;
///
/// // CategoricalDistribution is internal; this shows intended usage
/// let dist = CategoricalDistribution { n_choices: 3 };
/// let points = generate_categorical_grid_points(&dist);
/// assert_eq!(points, vec![0, 1, 2]);
/// ```
#[must_use]
pub fn generate_categorical_grid_points(dist: &CategoricalDistribution) -> Vec<usize> {
    (0..dist.n_choices).collect()
}

/// Cached grid points for a distribution.
#[derive(Debug, Clone)]
struct CachedGrid {
    /// The generated grid points as `ParamValue` instances.
    points: Vec<ParamValue>,
    /// Current index into the grid.
    current_index: usize,
}

/// Internal state for tracking grid position per distribution.
#[derive(Debug, Default)]
struct GridState {
    /// Cached grids for each distribution, keyed by distribution identifier.
    grids: HashMap<String, CachedGrid>,
}

/// A grid search sampler that exhaustively evaluates all grid points.
///
/// `GridSearchSampler` divides the parameter space into a grid and systematically
/// samples each point. This is useful when you want to evaluate all combinations
/// of parameter values, especially for discrete or small parameter spaces.
///
/// # Grid Exhaustion
///
/// The sampler tracks its position in the grid for each distribution independently.
/// When all grid points for a distribution have been sampled, subsequent calls to
/// `sample()` for that distribution will **panic** with the message:
/// `"GridSearchSampler: all grid points exhausted"`.
///
/// To avoid panics, use [`is_exhausted()`](Self::is_exhausted) to check if all
/// points have been sampled before calling `sample()`. You can also use
/// [`grid_size()`](Self::grid_size) to determine the total number of grid points
/// that will be sampled.
///
/// # Thread Safety
///
/// `GridSearchSampler` is thread-safe (`Send + Sync`) and uses internal locking
/// to ensure safe concurrent access to grid state.
///
/// # Examples
///
/// ```
/// use optimizer::sampler::grid::GridSearchSampler;
///
/// // Create with default settings (10 points per parameter)
/// let sampler = GridSearchSampler::new();
///
/// // Create with custom settings using the builder
/// let sampler = GridSearchSampler::builder().n_points_per_param(20).build();
/// ```
pub struct GridSearchSampler {
    /// Number of grid points per parameter (used when auto-discretizing).
    n_points_per_param: usize,
    /// Thread-safe internal state for tracking grid positions.
    state: Mutex<GridState>,
}

impl GridSearchSampler {
    /// Creates a new grid search sampler with default settings.
    ///
    /// Default settings:
    /// - `n_points_per_param`: 10 (each continuous parameter is discretized to 10 points)
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_points_per_param: 10,
            state: Mutex::new(GridState::default()),
        }
    }

    /// Creates a builder for configuring a `GridSearchSampler`.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::grid::GridSearchSampler;
    ///
    /// let sampler = GridSearchSampler::builder().n_points_per_param(20).build();
    /// ```
    #[must_use]
    pub fn builder() -> GridSearchSamplerBuilder {
        GridSearchSamplerBuilder::new()
    }
}

impl Default for GridSearchSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl GridSearchSampler {
    /// Returns `true` if all grid points for all tracked distributions have been sampled.
    ///
    /// A distribution is considered exhausted when its `current_index` equals the number
    /// of grid points. This method returns `true` only when **all** tracked distributions
    /// are exhausted.
    ///
    /// Note that a newly created sampler with no distributions sampled yet will return
    /// `true` (vacuously exhausted). After the first `sample()` call, the distribution
    /// is tracked and `is_exhausted()` will return `false` until all its points are used.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::grid::GridSearchSampler;
    ///
    /// let sampler = GridSearchSampler::new();
    /// // Initially exhausted (no distributions tracked yet)
    /// assert!(sampler.is_exhausted());
    /// ```
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        let state = self.state.lock();

        // If no distributions are tracked yet, consider it vacuously exhausted
        if state.grids.is_empty() {
            return true;
        }

        // All distributions must be exhausted
        state
            .grids
            .values()
            .all(|grid| grid.current_index >= grid.points.len())
    }

    /// Returns the total number of grid points across all tracked distributions.
    ///
    /// This method sums the number of grid points for each distribution that has been
    /// sampled at least once. Before any `sample()` calls, this returns 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::grid::GridSearchSampler;
    ///
    /// let sampler = GridSearchSampler::new();
    /// // No distributions tracked yet
    /// assert_eq!(sampler.grid_size(), 0);
    /// ```
    #[must_use]
    pub fn grid_size(&self) -> usize {
        let state = self.state.lock();
        state.grids.values().map(|grid| grid.points.len()).sum()
    }
}

/// Builder for configuring a [`GridSearchSampler`].
///
/// # Examples
///
/// ```
/// use optimizer::sampler::grid::GridSearchSamplerBuilder;
///
/// let sampler = GridSearchSamplerBuilder::new()
///     .n_points_per_param(20)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct GridSearchSamplerBuilder {
    n_points_per_param: usize,
}

impl GridSearchSamplerBuilder {
    /// Creates a new builder with default settings.
    ///
    /// Default settings:
    /// - `n_points_per_param`: 10
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_points_per_param: 10,
        }
    }

    /// Sets the number of grid points per parameter for auto-discretization.
    ///
    /// This value is used when a parameter distribution doesn't have an explicit
    /// step size. The parameter range is divided into `n` evenly spaced points.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of points per parameter.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::grid::GridSearchSamplerBuilder;
    ///
    /// let sampler = GridSearchSamplerBuilder::new()
    ///     .n_points_per_param(20)
    ///     .build();
    /// ```
    #[must_use]
    pub fn n_points_per_param(mut self, n: usize) -> Self {
        self.n_points_per_param = n;
        self
    }

    /// Builds the configured [`GridSearchSampler`].
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::grid::GridSearchSamplerBuilder;
    ///
    /// let sampler = GridSearchSamplerBuilder::new()
    ///     .n_points_per_param(20)
    ///     .build();
    /// ```
    #[must_use]
    pub fn build(self) -> GridSearchSampler {
        GridSearchSampler {
            n_points_per_param: self.n_points_per_param,
            state: Mutex::new(GridState::default()),
        }
    }
}

impl Default for GridSearchSamplerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Creates a unique identifier string from a distribution.
///
/// This is used as a key in the internal state to track grid position per distribution.
fn distribution_key(dist: &Distribution) -> String {
    match dist {
        Distribution::Float(d) => {
            format!(
                "float:{}:{}:{}:{}",
                d.low,
                d.high,
                d.log_scale,
                d.step.map_or("none".to_string(), |s| s.to_string())
            )
        }
        Distribution::Int(d) => {
            format!(
                "int:{}:{}:{}:{}",
                d.low,
                d.high,
                d.log_scale,
                d.step.map_or("none".to_string(), |s| s.to_string())
            )
        }
        Distribution::Categorical(d) => {
            format!("cat:{}", d.n_choices)
        }
    }
}

impl Sampler for GridSearchSampler {
    fn sample(
        &self,
        distribution: &Distribution,
        _trial_id: u64,
        _history: &[CompletedTrial],
    ) -> ParamValue {
        let mut state = self.state.lock();
        let key = distribution_key(distribution);

        // Get or create the cached grid for this distribution
        let cached = state.grids.entry(key).or_insert_with(|| {
            let points = match distribution {
                Distribution::Float(d) => generate_float_grid_points(d, self.n_points_per_param)
                    .into_iter()
                    .map(ParamValue::Float)
                    .collect(),
                Distribution::Int(d) => generate_int_grid_points(d, self.n_points_per_param)
                    .into_iter()
                    .map(ParamValue::Int)
                    .collect(),
                Distribution::Categorical(d) => generate_categorical_grid_points(d)
                    .into_iter()
                    .map(ParamValue::Categorical)
                    .collect(),
            };
            CachedGrid {
                points,
                current_index: 0,
            }
        });

        // Check if all points have been exhausted
        assert!(
            cached.current_index < cached.points.len(),
            "GridSearchSampler: all grid points exhausted"
        );

        // Get the current grid point and advance the index
        let value = cached.points[cached.current_index].clone();
        cached.current_index += 1;

        value
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
mod tests {
    use super::*;

    // ==================== Int Distribution Tests ====================

    #[test]
    fn test_int_grid_with_step() {
        let dist = IntDistribution {
            low: 0,
            high: 10,
            log_scale: false,
            step: Some(2),
        };
        let points = generate_int_grid_points(&dist, 10);
        // Should generate: 0, 2, 4, 6, 8, 10
        assert_eq!(points, vec![0, 2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_int_grid_with_step_not_exact_multiple() {
        let dist = IntDistribution {
            low: 0,
            high: 9,
            log_scale: false,
            step: Some(2),
        };
        let points = generate_int_grid_points(&dist, 10);
        // Should generate: 0, 2, 4, 6, 8 (stops at 8 because next step would be 10 > 9)
        assert_eq!(points, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_int_grid_without_step_linear() {
        let dist = IntDistribution {
            low: 0,
            high: 100,
            log_scale: false,
            step: None,
        };
        let points = generate_int_grid_points(&dist, 5);
        // Should generate 5 evenly spaced points: 0, 25, 50, 75, 100
        assert_eq!(points, vec![0, 25, 50, 75, 100]);
    }

    #[test]
    fn test_int_grid_without_step_linear_10_points() {
        let dist = IntDistribution {
            low: 0,
            high: 9,
            log_scale: false,
            step: None,
        };
        let points = generate_int_grid_points(&dist, 10);
        // Should generate all integers from 0 to 9
        assert_eq!(points, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_int_grid_with_log_scale() {
        let dist = IntDistribution {
            low: 1,
            high: 1000,
            log_scale: true,
            step: None,
        };
        let points = generate_int_grid_points(&dist, 4);
        // Log scale: should cover range exponentially
        // Points should be roughly: 1, 10, 100, 1000
        assert_eq!(points.len(), 4);
        assert_eq!(points[0], 1);
        assert_eq!(*points.last().unwrap(), 1000);
        // Middle points should be between bounds
        for p in &points {
            assert!(*p >= 1 && *p <= 1000);
        }
    }

    #[test]
    fn test_int_grid_log_scale_non_positive_fallback() {
        // Log scale with non-positive low should fall back to linear
        let dist = IntDistribution {
            low: 0,
            high: 100,
            log_scale: true,
            step: None,
        };
        let points = generate_int_grid_points(&dist, 5);
        // Falls back to linear: 0, 25, 50, 75, 100
        assert_eq!(points, vec![0, 25, 50, 75, 100]);
    }

    #[test]
    fn test_int_grid_single_point() {
        let dist = IntDistribution {
            low: 5,
            high: 5,
            log_scale: false,
            step: None,
        };
        let points = generate_int_grid_points(&dist, 10);
        assert_eq!(points, vec![5]);
    }

    #[test]
    fn test_int_grid_invalid_bounds() {
        let dist = IntDistribution {
            low: 10,
            high: 5,
            log_scale: false,
            step: None,
        };
        let points = generate_int_grid_points(&dist, 10);
        assert!(points.is_empty());
    }

    // ==================== Float Distribution Tests ====================

    #[test]
    fn test_float_grid_with_step() {
        let dist = FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: Some(0.25),
        };
        let points = generate_float_grid_points(&dist, 10);
        // Should generate: 0.0, 0.25, 0.5, 0.75, 1.0
        assert_eq!(points.len(), 5);
        assert!((points[0] - 0.0).abs() < f64::EPSILON);
        assert!((points[1] - 0.25).abs() < f64::EPSILON);
        assert!((points[2] - 0.5).abs() < f64::EPSILON);
        assert!((points[3] - 0.75).abs() < f64::EPSILON);
        assert!((points[4] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_float_grid_step_overrides_log_scale() {
        // Step should override log_scale
        let dist = FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: true, // This should be ignored
            step: Some(0.5),
        };
        let points = generate_float_grid_points(&dist, 10);
        // Should generate: 0.0, 0.5, 1.0 (step overrides log_scale)
        assert_eq!(points.len(), 3);
        assert!((points[0] - 0.0).abs() < f64::EPSILON);
        assert!((points[1] - 0.5).abs() < f64::EPSILON);
        assert!((points[2] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_float_grid_without_step_linear() {
        let dist = FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        };
        let points = generate_float_grid_points(&dist, 5);
        // Should generate 5 evenly spaced points: 0.0, 0.25, 0.5, 0.75, 1.0
        assert_eq!(points.len(), 5);
        assert!((points[0] - 0.0).abs() < f64::EPSILON);
        assert!((points[1] - 0.25).abs() < f64::EPSILON);
        assert!((points[2] - 0.5).abs() < f64::EPSILON);
        assert!((points[3] - 0.75).abs() < f64::EPSILON);
        assert!((points[4] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_float_grid_with_log_scale() {
        let dist = FloatDistribution {
            low: 1e-4,
            high: 1.0,
            log_scale: true,
            step: None,
        };
        let points = generate_float_grid_points(&dist, 5);
        // Log scale: should cover range exponentially
        // Points should be roughly: 1e-4, 1e-3, 1e-2, 1e-1, 1.0
        assert_eq!(points.len(), 5);
        assert!((points[0] - 1e-4).abs() < 1e-10);
        assert!((points[4] - 1.0).abs() < 1e-10);
        // Middle points should be between bounds
        for p in &points {
            assert!(*p >= 1e-4 && *p <= 1.0);
        }
        // In log scale, ratio between consecutive points should be roughly equal
        let ratio1 = points[1] / points[0];
        let ratio2 = points[2] / points[1];
        assert!((ratio1 - ratio2).abs() / ratio1 < 0.01);
    }

    #[test]
    fn test_float_grid_log_scale_non_positive_fallback() {
        // Log scale with non-positive low should fall back to linear
        let dist = FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: true,
            step: None,
        };
        let points = generate_float_grid_points(&dist, 5);
        // Falls back to linear: 0.0, 0.25, 0.5, 0.75, 1.0
        assert_eq!(points.len(), 5);
        assert!((points[0] - 0.0).abs() < f64::EPSILON);
        assert!((points[4] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_float_grid_single_point() {
        let dist = FloatDistribution {
            low: 0.5,
            high: 0.5,
            log_scale: false,
            step: None,
        };
        let points = generate_float_grid_points(&dist, 10);
        assert_eq!(points.len(), 1);
        assert!((points[0] - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_float_grid_invalid_bounds() {
        let dist = FloatDistribution {
            low: 1.0,
            high: 0.0,
            log_scale: false,
            step: None,
        };
        let points = generate_float_grid_points(&dist, 10);
        assert!(points.is_empty());
    }

    // ==================== Categorical Distribution Tests ====================

    #[test]
    fn test_categorical_grid() {
        let dist = CategoricalDistribution { n_choices: 5 };
        let points = generate_categorical_grid_points(&dist);
        assert_eq!(points, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_categorical_grid_single_choice() {
        let dist = CategoricalDistribution { n_choices: 1 };
        let points = generate_categorical_grid_points(&dist);
        assert_eq!(points, vec![0]);
    }

    #[test]
    fn test_categorical_grid_empty() {
        let dist = CategoricalDistribution { n_choices: 0 };
        let points = generate_categorical_grid_points(&dist);
        assert!(points.is_empty());
    }

    // ==================== Sampler Exhaustion Tests ====================

    #[test]
    fn test_sampler_exhausts_after_expected_samples() {
        let sampler = GridSearchSampler::new();
        let dist = Distribution::Categorical(CategoricalDistribution { n_choices: 3 });

        // Sample all 3 points
        for _ in 0..3 {
            let _ = sampler.sample(&dist, 0, &[]);
        }

        // Check exhaustion
        assert!(sampler.is_exhausted());
    }

    #[test]
    fn test_sampler_exhaustion_with_int_distribution() {
        let sampler = GridSearchSampler::builder().n_points_per_param(5).build();
        let dist = Distribution::Int(IntDistribution {
            low: 0,
            high: 100,
            log_scale: false,
            step: None,
        });

        // Sample all 5 points
        for _ in 0..5 {
            let _ = sampler.sample(&dist, 0, &[]);
        }

        assert!(sampler.is_exhausted());
        assert_eq!(sampler.grid_size(), 5);
    }

    #[test]
    #[should_panic(expected = "GridSearchSampler: all grid points exhausted")]
    fn test_sampler_panics_after_exhaustion() {
        let sampler = GridSearchSampler::new();
        let dist = Distribution::Categorical(CategoricalDistribution { n_choices: 2 });

        // Sample all 2 points
        sampler.sample(&dist, 0, &[]);
        sampler.sample(&dist, 0, &[]);

        // This should panic
        sampler.sample(&dist, 0, &[]);
    }

    // ==================== is_exhausted() Tests ====================

    #[test]
    fn test_is_exhausted_before_sampling() {
        let sampler = GridSearchSampler::new();
        // Newly created sampler is vacuously exhausted (no distributions tracked)
        assert!(sampler.is_exhausted());
    }

    #[test]
    fn test_is_exhausted_during_sampling() {
        let sampler = GridSearchSampler::new();
        let dist = Distribution::Categorical(CategoricalDistribution { n_choices: 3 });

        // After first sample, not exhausted
        sampler.sample(&dist, 0, &[]);
        assert!(!sampler.is_exhausted());

        // After second sample, still not exhausted
        sampler.sample(&dist, 0, &[]);
        assert!(!sampler.is_exhausted());

        // After third sample, exhausted
        sampler.sample(&dist, 0, &[]);
        assert!(sampler.is_exhausted());
    }

    #[test]
    fn test_is_exhausted_multiple_distributions() {
        let sampler = GridSearchSampler::new();
        // Use different n_choices so they have different distribution keys
        let dist1 = Distribution::Categorical(CategoricalDistribution { n_choices: 2 });
        let dist2 = Distribution::Categorical(CategoricalDistribution { n_choices: 3 });

        // Exhaust first distribution
        sampler.sample(&dist1, 0, &[]);
        sampler.sample(&dist1, 0, &[]);

        // Not exhausted yet because dist2 is not exhausted
        sampler.sample(&dist2, 0, &[]);
        assert!(!sampler.is_exhausted());

        // Continue sampling dist2
        sampler.sample(&dist2, 0, &[]);
        assert!(!sampler.is_exhausted());

        // Exhaust second distribution
        sampler.sample(&dist2, 0, &[]);
        assert!(sampler.is_exhausted());
    }

    // ==================== Builder Pattern Tests ====================

    #[test]
    fn test_builder_default() {
        let sampler = GridSearchSampler::builder().build();
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        // Default is 10 points per param
        for _ in 0..10 {
            let _ = sampler.sample(&dist, 0, &[]);
        }
        assert!(sampler.is_exhausted());
    }

    #[test]
    fn test_builder_custom_n_points() {
        let sampler = GridSearchSampler::builder().n_points_per_param(3).build();
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        // Should have 3 points
        for _ in 0..3 {
            let _ = sampler.sample(&dist, 0, &[]);
        }
        assert!(sampler.is_exhausted());
        assert_eq!(sampler.grid_size(), 3);
    }

    #[test]
    fn test_new_default() {
        let sampler = GridSearchSampler::new();
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        // Default is 10 points per param
        for _ in 0..10 {
            let _ = sampler.sample(&dist, 0, &[]);
        }
        assert!(sampler.is_exhausted());
    }

    // ==================== Reproducibility Tests ====================

    #[test]
    fn test_reproducibility_same_grid_order() {
        // Two samplers with the same configuration should produce the same grid order
        let sampler1 = GridSearchSampler::builder().n_points_per_param(5).build();
        let sampler2 = GridSearchSampler::builder().n_points_per_param(5).build();

        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        // Both should produce the same sequence
        for _ in 0..5 {
            let v1 = sampler1.sample(&dist, 0, &[]);
            let v2 = sampler2.sample(&dist, 0, &[]);
            assert_eq!(v1, v2);
        }
    }

    #[test]
    fn test_reproducibility_int_distribution() {
        let sampler1 = GridSearchSampler::new();
        let sampler2 = GridSearchSampler::new();

        let dist = Distribution::Int(IntDistribution {
            low: 0,
            high: 10,
            log_scale: false,
            step: Some(2),
        });

        // Both should produce: 0, 2, 4, 6, 8, 10
        let expected = vec![0, 2, 4, 6, 8, 10];
        for exp in &expected {
            let v1 = sampler1.sample(&dist, 0, &[]);
            let v2 = sampler2.sample(&dist, 0, &[]);
            assert_eq!(v1, ParamValue::Int(*exp));
            assert_eq!(v2, ParamValue::Int(*exp));
        }
    }

    #[test]
    fn test_reproducibility_categorical() {
        let sampler1 = GridSearchSampler::new();
        let sampler2 = GridSearchSampler::new();

        let dist = Distribution::Categorical(CategoricalDistribution { n_choices: 4 });

        // Both should produce: 0, 1, 2, 3
        for i in 0..4 {
            let v1 = sampler1.sample(&dist, 0, &[]);
            let v2 = sampler2.sample(&dist, 0, &[]);
            assert_eq!(v1, ParamValue::Categorical(i));
            assert_eq!(v2, ParamValue::Categorical(i));
        }
    }

    // ==================== Grid Size Tests ====================

    #[test]
    fn test_grid_size_empty() {
        let sampler = GridSearchSampler::new();
        assert_eq!(sampler.grid_size(), 0);
    }

    #[test]
    fn test_grid_size_single_distribution() {
        let sampler = GridSearchSampler::builder().n_points_per_param(5).build();
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        // Before sampling
        assert_eq!(sampler.grid_size(), 0);

        // After first sample, grid is created
        sampler.sample(&dist, 0, &[]);
        assert_eq!(sampler.grid_size(), 5);
    }

    #[test]
    fn test_grid_size_multiple_distributions() {
        let sampler = GridSearchSampler::builder().n_points_per_param(3).build();
        let dist1 = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });
        let dist2 = Distribution::Categorical(CategoricalDistribution { n_choices: 5 });

        // Sample from first distribution
        sampler.sample(&dist1, 0, &[]);
        assert_eq!(sampler.grid_size(), 3);

        // Sample from second distribution
        sampler.sample(&dist2, 0, &[]);
        assert_eq!(sampler.grid_size(), 3 + 5);
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_int_step_larger_than_range() {
        let dist = IntDistribution {
            low: 0,
            high: 5,
            log_scale: false,
            step: Some(10),
        };
        let points = generate_int_grid_points(&dist, 10);
        // Only the starting point should be generated
        assert_eq!(points, vec![0]);
    }

    #[test]
    fn test_float_step_larger_than_range() {
        let dist = FloatDistribution {
            low: 0.0,
            high: 0.5,
            log_scale: false,
            step: Some(1.0),
        };
        let points = generate_float_grid_points(&dist, 10);
        // Only the starting point should be generated
        assert_eq!(points.len(), 1);
        assert!((points[0] - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_n_points_one() {
        let dist = IntDistribution {
            low: 0,
            high: 100,
            log_scale: false,
            step: None,
        };
        let points = generate_int_grid_points(&dist, 1);
        // Should return just the low bound
        assert_eq!(points, vec![0]);
    }

    #[test]
    fn test_n_points_zero() {
        let dist = IntDistribution {
            low: 0,
            high: 100,
            log_scale: false,
            step: None,
        };
        let points = generate_int_grid_points(&dist, 0);
        assert!(points.is_empty());
    }
}
