use core::fmt::Debug;

use crate::Error;

/// A strategy for computing the gamma quantile in TPE.
///
/// The gamma value determines what fraction of trials are considered "good"
/// when splitting the trial history. Different strategies can adapt this
/// fraction based on the number of completed trials.
///
/// # Implementation Notes
///
/// - The returned gamma must be in the range (0.0, 1.0)
/// - Implementations should be deterministic for reproducibility
/// - The `clone_box` method enables trait object cloning
///
/// # Examples
///
/// ```
/// use optimizer::sampler::tpe::GammaStrategy;
///
/// #[derive(Debug, Clone)]
/// struct ConstantGamma(f64);
///
/// impl GammaStrategy for ConstantGamma {
///     fn gamma(&self, _n_trials: usize) -> f64 {
///         self.0
///     }
///
///     fn clone_box(&self) -> Box<dyn GammaStrategy> {
///         Box::new(self.clone())
///     }
/// }
/// ```
pub trait GammaStrategy: Send + Sync + Debug {
    /// Computes the gamma quantile based on the number of completed trials.
    ///
    /// # Arguments
    ///
    /// * `n_trials` - The number of completed trials in the history.
    ///
    /// # Returns
    ///
    /// A gamma value in the range (0.0, 1.0). Values outside this range
    /// will be clamped by the sampler.
    fn gamma(&self, n_trials: usize) -> f64;

    /// Creates a boxed clone of this strategy.
    ///
    /// This method enables cloning of trait objects, which is necessary
    /// for the builder pattern and sampler configuration.
    fn clone_box(&self) -> Box<dyn GammaStrategy>;
}

impl Clone for Box<dyn GammaStrategy> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// A fixed gamma strategy that returns a constant value.
///
/// This is the simplest strategy and the default behavior of TPE.
/// The gamma value remains constant regardless of the number of trials.
///
/// # Examples
///
/// ```
/// use optimizer::sampler::tpe::{FixedGamma, TpeSampler};
///
/// // Use 15% of trials as "good"
/// let sampler = TpeSampler::builder()
///     .gamma_strategy(FixedGamma::new(0.15).unwrap())
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FixedGamma {
    gamma: f64,
}

impl FixedGamma {
    /// Creates a new fixed gamma strategy.
    ///
    /// # Arguments
    ///
    /// * `gamma` - The constant gamma value to use.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidGamma` if gamma is not in (0.0, 1.0).
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::FixedGamma;
    ///
    /// let strategy = FixedGamma::new(0.25).unwrap();
    /// assert!((strategy.value() - 0.25).abs() < f64::EPSILON);
    /// ```
    pub fn new(gamma: f64) -> crate::Result<Self> {
        if gamma <= 0.0 || gamma >= 1.0 {
            return Err(Error::InvalidGamma(gamma));
        }
        Ok(Self { gamma })
    }

    /// Returns the fixed gamma value.
    #[must_use]
    pub fn value(&self) -> f64 {
        self.gamma
    }
}

impl Default for FixedGamma {
    /// Creates a fixed gamma strategy with the default value of 0.25.
    fn default() -> Self {
        Self { gamma: 0.25 }
    }
}

impl GammaStrategy for FixedGamma {
    fn gamma(&self, _n_trials: usize) -> f64 {
        self.gamma
    }

    fn clone_box(&self) -> Box<dyn GammaStrategy> {
        Box::new(*self)
    }
}

/// A linear gamma strategy that interpolates between min and max values.
///
/// The gamma value increases linearly from `gamma_min` to `gamma_max` as the
/// number of trials grows from 0 to `n_trials_max`. Beyond `n_trials_max`,
/// gamma remains at `gamma_max`.
///
/// This strategy is useful when you want to be more explorative early on
/// (smaller gamma = fewer "good" trials) and more exploitative later
/// (larger gamma = more "good" trials).
///
/// # Formula
///
/// ```text
/// gamma = gamma_min + (gamma_max - gamma_min) * min(n_trials / n_trials_max, 1.0)
/// ```
///
/// # Examples
///
/// ```
/// use optimizer::sampler::tpe::{GammaStrategy, LinearGamma, TpeSampler};
///
/// let strategy = LinearGamma::new(0.1, 0.4, 100).unwrap();
///
/// // At 0 trials: gamma = 0.1
/// assert!((strategy.gamma(0) - 0.1).abs() < f64::EPSILON);
///
/// // At 50 trials: gamma = 0.25 (midpoint)
/// assert!((strategy.gamma(50) - 0.25).abs() < f64::EPSILON);
///
/// // At 100+ trials: gamma = 0.4
/// assert!((strategy.gamma(100) - 0.4).abs() < f64::EPSILON);
/// assert!((strategy.gamma(200) - 0.4).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LinearGamma {
    gamma_min: f64,
    gamma_max: f64,
    n_trials_max: usize,
}

impl LinearGamma {
    /// Creates a new linear gamma strategy.
    ///
    /// # Arguments
    ///
    /// * `gamma_min` - The minimum gamma value (at 0 trials).
    /// * `gamma_max` - The maximum gamma value (at `n_trials_max` trials).
    /// * `n_trials_max` - The number of trials at which gamma reaches its maximum.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidGamma` if:
    /// - `gamma_min` is not in (0.0, 1.0)
    /// - `gamma_max` is not in (0.0, 1.0)
    /// - `gamma_min > gamma_max`
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::LinearGamma;
    ///
    /// // Gamma goes from 0.1 to 0.3 over 50 trials
    /// let strategy = LinearGamma::new(0.1, 0.3, 50).unwrap();
    /// ```
    pub fn new(gamma_min: f64, gamma_max: f64, n_trials_max: usize) -> crate::Result<Self> {
        if gamma_min <= 0.0 || gamma_min >= 1.0 {
            return Err(Error::InvalidGamma(gamma_min));
        }
        if gamma_max <= 0.0 || gamma_max >= 1.0 {
            return Err(Error::InvalidGamma(gamma_max));
        }
        if gamma_min > gamma_max {
            return Err(Error::InvalidGamma(gamma_min));
        }
        Ok(Self {
            gamma_min,
            gamma_max,
            n_trials_max,
        })
    }

    /// Returns the minimum gamma value.
    #[must_use]
    pub fn gamma_min(&self) -> f64 {
        self.gamma_min
    }

    /// Returns the maximum gamma value.
    #[must_use]
    pub fn gamma_max(&self) -> f64 {
        self.gamma_max
    }

    /// Returns the number of trials at which gamma reaches its maximum.
    #[must_use]
    pub fn n_trials_max(&self) -> usize {
        self.n_trials_max
    }
}

impl Default for LinearGamma {
    /// Creates a linear gamma strategy with default values:
    /// - `gamma_min`: 0.10
    /// - `gamma_max`: 0.25
    /// - `n_trials_max`: 100
    fn default() -> Self {
        Self {
            gamma_min: 0.10,
            gamma_max: 0.25,
            n_trials_max: 100,
        }
    }
}

impl GammaStrategy for LinearGamma {
    #[allow(clippy::cast_precision_loss)]
    fn gamma(&self, n_trials: usize) -> f64 {
        if self.n_trials_max == 0 {
            return self.gamma_max;
        }
        let t = (n_trials as f64 / self.n_trials_max as f64).min(1.0);
        self.gamma_min + (self.gamma_max - self.gamma_min) * t
    }

    fn clone_box(&self) -> Box<dyn GammaStrategy> {
        Box::new(*self)
    }
}

/// A square root gamma strategy inspired by Optuna's default behavior.
///
/// The gamma value is computed based on the inverse square root of the number
/// of trials, providing a balance between exploration and exploitation that
/// naturally adapts as more data becomes available.
///
/// # Formula
///
/// ```text
/// n_good = max(1, floor(gamma_factor / sqrt(n_trials)))
/// gamma = min(gamma_max, n_good / n_trials)
/// ```
///
/// When `n_trials` is 0, returns `gamma_max`.
///
/// # Examples
///
/// ```
/// use optimizer::sampler::tpe::{GammaStrategy, SqrtGamma, TpeSampler};
///
/// let strategy = SqrtGamma::default();
///
/// // Gamma decreases as trials increase
/// let g10 = strategy.gamma(10);
/// let g100 = strategy.gamma(100);
/// assert!(g10 > g100, "Gamma should decrease with more trials");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SqrtGamma {
    gamma_factor: f64,
    gamma_max: f64,
}

impl SqrtGamma {
    /// Creates a new square root gamma strategy.
    ///
    /// # Arguments
    ///
    /// * `gamma_factor` - The factor controlling how quickly gamma decreases.
    ///   Higher values mean more "good" trials at any given point.
    /// * `gamma_max` - The maximum gamma value (used when `n_trials` is small).
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidGamma` if:
    /// - `gamma_factor` is not positive
    /// - `gamma_max` is not in (0.0, 1.0)
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::SqrtGamma;
    ///
    /// let strategy = SqrtGamma::new(1.0, 0.25).unwrap();
    /// ```
    pub fn new(gamma_factor: f64, gamma_max: f64) -> crate::Result<Self> {
        if gamma_factor <= 0.0 {
            return Err(Error::InvalidGamma(gamma_factor));
        }
        if gamma_max <= 0.0 || gamma_max >= 1.0 {
            return Err(Error::InvalidGamma(gamma_max));
        }
        Ok(Self {
            gamma_factor,
            gamma_max,
        })
    }

    /// Returns the gamma factor.
    #[must_use]
    pub fn gamma_factor(&self) -> f64 {
        self.gamma_factor
    }

    /// Returns the maximum gamma value.
    #[must_use]
    pub fn gamma_max(&self) -> f64 {
        self.gamma_max
    }
}

impl Default for SqrtGamma {
    /// Creates a square root gamma strategy with default values:
    /// - `gamma_factor`: 1.0
    /// - `gamma_max`: 0.25
    fn default() -> Self {
        Self {
            gamma_factor: 1.0,
            gamma_max: 0.25,
        }
    }
}

impl GammaStrategy for SqrtGamma {
    #[allow(clippy::cast_precision_loss)]
    fn gamma(&self, n_trials: usize) -> f64 {
        if n_trials == 0 {
            return self.gamma_max;
        }
        let n_good = (self.gamma_factor / (n_trials as f64).sqrt()).max(1.0);
        (n_good / n_trials as f64).min(self.gamma_max)
    }

    fn clone_box(&self) -> Box<dyn GammaStrategy> {
        Box::new(*self)
    }
}

/// A Hyperopt-style gamma strategy.
///
/// This strategy computes gamma as `min(gamma_max, (gamma_base + 1) / n_trials)`,
/// which is inspired by the original Hyperopt TPE implementation.
///
/// # Formula
///
/// ```text
/// gamma = min(gamma_max, (gamma_base + 1) / n_trials)
/// ```
///
/// When `n_trials` is 0, returns `gamma_max`.
///
/// # Examples
///
/// ```
/// use optimizer::sampler::tpe::{GammaStrategy, HyperoptGamma};
///
/// // With gamma_base=24 and gamma_max=0.5:
/// // - At n=25: gamma = min(0.5, 25/25) = 0.5 (capped)
/// // - At n=100: gamma = min(0.5, 25/100) = 0.25
/// let strategy = HyperoptGamma::new(24.0, 0.5).unwrap();
///
/// // Early trials have higher gamma
/// let g50 = strategy.gamma(50);
/// let g200 = strategy.gamma(200);
/// assert!(g50 > g200, "Gamma should decrease with more trials");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct HyperoptGamma {
    gamma_base: f64,
    gamma_max: f64,
}

impl HyperoptGamma {
    /// Creates a new Hyperopt-style gamma strategy.
    ///
    /// # Arguments
    ///
    /// * `gamma_base` - The base value added to 1 in the numerator.
    /// * `gamma_max` - The maximum gamma value.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidGamma` if:
    /// - `gamma_base` is negative
    /// - `gamma_max` is not in (0.0, 1.0)
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::HyperoptGamma;
    ///
    /// let strategy = HyperoptGamma::new(24.0, 0.25).unwrap();
    /// ```
    pub fn new(gamma_base: f64, gamma_max: f64) -> crate::Result<Self> {
        if gamma_base < 0.0 {
            return Err(Error::InvalidGamma(gamma_base));
        }
        if gamma_max <= 0.0 || gamma_max >= 1.0 {
            return Err(Error::InvalidGamma(gamma_max));
        }
        Ok(Self {
            gamma_base,
            gamma_max,
        })
    }

    /// Returns the gamma base value.
    #[must_use]
    pub fn gamma_base(&self) -> f64 {
        self.gamma_base
    }

    /// Returns the maximum gamma value.
    #[must_use]
    pub fn gamma_max(&self) -> f64 {
        self.gamma_max
    }
}

impl Default for HyperoptGamma {
    /// Creates a Hyperopt-style gamma strategy with default values:
    /// - `gamma_base`: 24.0
    /// - `gamma_max`: 0.25
    fn default() -> Self {
        Self {
            gamma_base: 24.0,
            gamma_max: 0.25,
        }
    }
}

impl GammaStrategy for HyperoptGamma {
    #[allow(clippy::cast_precision_loss)]
    fn gamma(&self, n_trials: usize) -> f64 {
        if n_trials == 0 {
            return self.gamma_max;
        }
        ((self.gamma_base + 1.0) / n_trials as f64).min(self.gamma_max)
    }

    fn clone_box(&self) -> Box<dyn GammaStrategy> {
        Box::new(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::tpe::TpeSampler;

    #[test]
    fn test_fixed_gamma_default() {
        let strategy = FixedGamma::default();
        assert!((strategy.gamma(0) - 0.25).abs() < f64::EPSILON);
        assert!((strategy.gamma(100) - 0.25).abs() < f64::EPSILON);
        assert!((strategy.value() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fixed_gamma_custom() {
        let strategy = FixedGamma::new(0.15).unwrap();
        assert!((strategy.gamma(0) - 0.15).abs() < f64::EPSILON);
        assert!((strategy.gamma(50) - 0.15).abs() < f64::EPSILON);
        assert!((strategy.gamma(1000) - 0.15).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fixed_gamma_invalid() {
        assert!(FixedGamma::new(0.0).is_err());
        assert!(FixedGamma::new(1.0).is_err());
        assert!(FixedGamma::new(-0.1).is_err());
        assert!(FixedGamma::new(1.5).is_err());
    }

    #[test]
    fn test_linear_gamma_default() {
        let strategy = LinearGamma::default();
        assert!((strategy.gamma(0) - 0.10).abs() < f64::EPSILON);
        assert!((strategy.gamma(50) - 0.175).abs() < f64::EPSILON); // midpoint
        assert!((strategy.gamma(100) - 0.25).abs() < f64::EPSILON);
        assert!((strategy.gamma(200) - 0.25).abs() < f64::EPSILON); // capped
    }

    #[test]
    fn test_linear_gamma_custom() {
        let strategy = LinearGamma::new(0.1, 0.4, 100).unwrap();
        assert!((strategy.gamma(0) - 0.1).abs() < f64::EPSILON);
        assert!((strategy.gamma(50) - 0.25).abs() < f64::EPSILON);
        assert!((strategy.gamma(100) - 0.4).abs() < f64::EPSILON);
        assert!((strategy.gamma(200) - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_linear_gamma_invalid() {
        assert!(LinearGamma::new(0.0, 0.5, 100).is_err());
        assert!(LinearGamma::new(0.1, 1.0, 100).is_err());
        assert!(LinearGamma::new(0.5, 0.2, 100).is_err()); // min > max
    }

    #[test]
    fn test_sqrt_gamma_default() {
        let strategy = SqrtGamma::default();
        // At n=0, returns gamma_max
        assert!((strategy.gamma(0) - 0.25).abs() < f64::EPSILON);

        // gamma decreases with more trials
        let g10 = strategy.gamma(10);
        let g100 = strategy.gamma(100);
        assert!(g10 > g100);
    }

    #[test]
    fn test_sqrt_gamma_custom() {
        let strategy = SqrtGamma::new(2.0, 0.5).unwrap();
        assert!((strategy.gamma(0) - 0.5).abs() < f64::EPSILON);

        // At n=4: n_good = max(1, 2/2) = 1, gamma = 1/4 = 0.25
        let g4 = strategy.gamma(4);
        assert!((g4 - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sqrt_gamma_invalid() {
        assert!(SqrtGamma::new(0.0, 0.25).is_err()); // factor must be positive
        assert!(SqrtGamma::new(-1.0, 0.25).is_err());
        assert!(SqrtGamma::new(1.0, 0.0).is_err());
        assert!(SqrtGamma::new(1.0, 1.0).is_err());
    }

    #[test]
    fn test_hyperopt_gamma_default() {
        let strategy = HyperoptGamma::default();
        // At n=0, returns gamma_max
        assert!((strategy.gamma(0) - 0.25).abs() < f64::EPSILON);

        // At n=100: (24+1)/100 = 0.25, so capped to 0.25
        assert!((strategy.gamma(100) - 0.25).abs() < f64::EPSILON);

        // At n=200: (24+1)/200 = 0.125
        assert!((strategy.gamma(200) - 0.125).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hyperopt_gamma_custom() {
        let strategy = HyperoptGamma::new(9.0, 0.5).unwrap();
        // At n=20: (9+1)/20 = 0.5, capped to 0.5
        assert!((strategy.gamma(20) - 0.5).abs() < f64::EPSILON);

        // At n=100: (9+1)/100 = 0.1
        assert!((strategy.gamma(100) - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hyperopt_gamma_invalid() {
        assert!(HyperoptGamma::new(-1.0, 0.25).is_err());
        assert!(HyperoptGamma::new(24.0, 0.0).is_err());
        assert!(HyperoptGamma::new(24.0, 1.0).is_err());
    }

    #[test]
    fn test_gamma_strategy_clone_box() {
        let fixed: Box<dyn GammaStrategy> = Box::new(FixedGamma::new(0.3).unwrap());
        let cloned = fixed.clone();
        assert!((cloned.gamma(0) - 0.3).abs() < f64::EPSILON);

        let linear: Box<dyn GammaStrategy> = Box::new(LinearGamma::default());
        let cloned = linear.clone();
        assert!((cloned.gamma(0) - 0.10).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tpe_with_linear_gamma_strategy() {
        let sampler = TpeSampler::builder()
            .gamma_strategy(LinearGamma::new(0.1, 0.3, 50).unwrap())
            .n_startup_trials(5)
            .seed(42)
            .build()
            .unwrap();

        // Verify the strategy is applied
        let g = sampler.gamma_strategy().gamma(25);
        assert!((g - 0.2).abs() < f64::EPSILON); // midpoint of 0.1 to 0.3
    }

    #[test]
    fn test_gamma_overrides_gamma_strategy() {
        // When gamma() is called after gamma_strategy(), it should take precedence
        let sampler = TpeSampler::builder()
            .gamma_strategy(SqrtGamma::default())
            .gamma(0.15) // This should override
            .build()
            .unwrap();

        // Should use fixed gamma of 0.15
        assert!((sampler.gamma_strategy().gamma(0) - 0.15).abs() < f64::EPSILON);
        assert!((sampler.gamma_strategy().gamma(100) - 0.15).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gamma_strategy_overrides_gamma() {
        // When gamma_strategy() is called after gamma(), it should take precedence
        let sampler = TpeSampler::builder()
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
    fn test_custom_gamma_strategy() {
        #[derive(Debug, Clone)]
        struct DoubleGamma;

        impl GammaStrategy for DoubleGamma {
            fn gamma(&self, n_trials: usize) -> f64 {
                // Double the trial count-based calculation, capped at 0.5
                #[allow(clippy::cast_precision_loss)]
                (0.01 * n_trials as f64).min(0.5)
            }

            fn clone_box(&self) -> Box<dyn GammaStrategy> {
                Box::new(self.clone())
            }
        }

        let sampler = TpeSampler::builder()
            .gamma_strategy(DoubleGamma)
            .build()
            .unwrap();

        assert!((sampler.gamma_strategy().gamma(10) - 0.1).abs() < f64::EPSILON);
        assert!((sampler.gamma_strategy().gamma(50) - 0.5).abs() < f64::EPSILON);
        assert!((sampler.gamma_strategy().gamma(100) - 0.5).abs() < f64::EPSILON);
    }
}
