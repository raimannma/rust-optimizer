//! Parameter trait and five built-in parameter types.
//!
//! The [`Parameter`] trait provides a unified way to define search-space
//! dimensions and sample values from a [`Trial`]. Five implementations
//! cover the most common hyperparameter types:
//!
//! | Type | Sampled value | Typical use |
//! |------|---------------|-------------|
//! | [`FloatParam`] | `f64` | Learning rate, dropout probability |
//! | [`IntParam`] | `i64` | Layer count, batch size |
//! | [`CategoricalParam`] | `T: Clone` | Optimizer name, activation function |
//! | [`BoolParam`] | `bool` | Feature toggle |
//! | [`EnumParam`] | `T: Categorical` | Typed enum variant selection |
//!
//! All five types support `.name()` for a human-readable label and
//! `.suggest(&mut trial)` as a shorthand for `trial.suggest_param(&param)`.
//!
//! # Example
//!
//! ```
//! use optimizer::Trial;
//! use optimizer::parameter::{BoolParam, FloatParam, IntParam, Parameter};
//!
//! let mut trial = Trial::new(0);
//!
//! let lr = FloatParam::new(1e-5, 1e-1)
//!     .log_scale()
//!     .name("learning_rate")
//!     .suggest(&mut trial)
//!     .unwrap();
//! let layers = IntParam::new(1, 10)
//!     .name("n_layers")
//!     .suggest(&mut trial)
//!     .unwrap();
//! let dropout = BoolParam::new()
//!     .name("use_dropout")
//!     .suggest(&mut trial)
//!     .unwrap();
//! ```

use core::fmt::Debug;
use core::ops::RangeInclusive;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::distribution::{
    CategoricalDistribution, Distribution, FloatDistribution, IntDistribution,
};
use crate::error::{Error, Result};
pub use crate::param::ParamValue;
use crate::trial::Trial;

static NEXT_PARAM_ID: AtomicU64 = AtomicU64::new(0);

/// A unique identifier for a parameter instance.
///
/// Each parameter is assigned a unique `ParamId` at creation time. Cloning a parameter
/// copies its `ParamId`, so clones refer to the same logical parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParamId(u64);

impl ParamId {
    /// Create a new unique `ParamId`.
    pub fn new() -> Self {
        Self(NEXT_PARAM_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for ParamId {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Display for ParamId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "param_{}", self.0)
    }
}

/// Define a parameter type that can be suggested by a [`Trial`].
///
/// Implementors specify the distribution to sample from and how to convert
/// the raw [`ParamValue`] back into a typed value. See the five built-in
/// implementations: [`FloatParam`], [`IntParam`], [`CategoricalParam`],
/// [`BoolParam`], and [`EnumParam`].
pub trait Parameter: Debug {
    /// The typed value returned after sampling.
    type Value;

    /// Return the unique identifier for this parameter.
    fn id(&self) -> ParamId;

    /// Return the distribution that this parameter samples from.
    fn distribution(&self) -> Distribution;

    /// Convert a raw [`ParamValue`] into the typed value.
    ///
    /// # Errors
    ///
    /// Return an error if the `ParamValue` variant does not match what this parameter expects.
    fn cast_param_value(&self, param_value: &ParamValue) -> Result<Self::Value>;

    /// Validate the parameter configuration.
    ///
    /// Called before sampling. The default implementation accepts all configurations.
    ///
    /// # Errors
    ///
    /// Return an error if the parameter configuration is invalid.
    fn validate(&self) -> Result<()> {
        Ok(())
    }

    /// Return a human-readable label for this parameter.
    ///
    /// Defaults to the `Debug` output of the parameter. Override with
    /// the `.name()` builder method on concrete types.
    fn label(&self) -> String {
        format!("{self:?}")
    }

    /// Suggest a value for this parameter from the given trial.
    ///
    /// This is a convenience method that delegates to [`Trial::suggest_param`].
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::Trial;
    /// use optimizer::parameter::{FloatParam, Parameter};
    ///
    /// let mut trial = Trial::new(0);
    /// let param = FloatParam::new(-5.0, 5.0).name("x");
    /// let value: f64 = param.suggest(&mut trial).unwrap();
    /// assert!((-5.0..=5.0).contains(&value));
    /// ```
    ///
    /// # Errors
    ///
    /// Return an error if validation fails, the parameter conflicts with
    /// a previously suggested parameter of the same id, or sampling fails.
    fn suggest(&self, trial: &mut Trial) -> Result<Self::Value>
    where
        Self: Sized,
    {
        trial.suggest_param(self)
    }
}

/// A floating-point parameter with optional log-scale and step size.
///
/// # Examples
///
/// ```
/// use optimizer::Trial;
/// use optimizer::parameter::{FloatParam, Parameter};
///
/// let mut trial = Trial::new(0);
///
/// // Simple range
/// let x = FloatParam::new(0.0, 1.0).suggest(&mut trial).unwrap();
///
/// // Log-scale with a human-readable name
/// let lr = FloatParam::new(1e-5, 1e-1)
///     .log_scale()
///     .name("learning_rate")
///     .suggest(&mut trial)
///     .unwrap();
///
/// // Stepped (values will be multiples of 0.25)
/// let step = FloatParam::new(0.0, 1.0)
///     .step(0.25)
///     .suggest(&mut trial)
///     .unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct FloatParam {
    id: ParamId,
    low: f64,
    high: f64,
    log_scale: bool,
    step: Option<f64>,
    name: Option<String>,
}

impl FloatParam {
    /// Create a new float parameter sampling uniformly from `[low, high]`.
    #[must_use]
    pub fn new(low: f64, high: f64) -> Self {
        Self {
            id: ParamId::new(),
            low,
            high,
            log_scale: false,
            step: None,
            name: None,
        }
    }

    /// Enable log-scale sampling (bounds must be positive).
    #[must_use]
    pub fn log_scale(mut self) -> Self {
        self.log_scale = true;
        self
    }

    /// Set a step size for discretized sampling.
    #[must_use]
    pub fn step(mut self, step: f64) -> Self {
        self.step = Some(step);
        self
    }

    /// Set a human-readable name for this parameter.
    ///
    /// When set, this name is used as the parameter's label instead of
    /// the default `Debug` output.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl From<RangeInclusive<f64>> for FloatParam {
    fn from(range: RangeInclusive<f64>) -> Self {
        FloatParam::new(*range.start(), *range.end())
    }
}

impl Parameter for FloatParam {
    type Value = f64;

    fn id(&self) -> ParamId {
        self.id
    }

    fn distribution(&self) -> Distribution {
        Distribution::Float(FloatDistribution {
            low: self.low,
            high: self.high,
            log_scale: self.log_scale,
            step: self.step,
        })
    }

    fn cast_param_value(&self, param_value: &ParamValue) -> Result<f64> {
        match param_value {
            ParamValue::Float(v) => Ok(*v),
            _ => Err(Error::Internal(
                "Float distribution should return Float value",
            )),
        }
    }

    fn validate(&self) -> Result<()> {
        if !self.low.is_finite() || !self.high.is_finite() {
            return Err(Error::InvalidBounds {
                low: self.low,
                high: self.high,
            });
        }
        if self.low > self.high {
            return Err(Error::InvalidBounds {
                low: self.low,
                high: self.high,
            });
        }
        if self.log_scale && self.low <= 0.0 {
            return Err(Error::InvalidLogBounds);
        }
        if let Some(step) = self.step
            && (!step.is_finite() || step <= 0.0)
        {
            return Err(Error::InvalidStep);
        }
        Ok(())
    }

    fn label(&self) -> String {
        self.name.clone().unwrap_or_else(|| format!("{self:?}"))
    }
}

/// An integer parameter with optional log-scale and step size.
///
/// # Examples
///
/// ```
/// use optimizer::Trial;
/// use optimizer::parameter::{IntParam, Parameter};
///
/// let mut trial = Trial::new(0);
///
/// // Simple range
/// let n = IntParam::new(1, 10)
///     .name("n_layers")
///     .suggest(&mut trial)
///     .unwrap();
///
/// // Log-scale
/// let batch = IntParam::new(1, 1024)
///     .log_scale()
///     .name("batch_size")
///     .suggest(&mut trial)
///     .unwrap();
///
/// // Stepped (multiples of 32)
/// let units = IntParam::new(32, 512).step(32).suggest(&mut trial).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct IntParam {
    id: ParamId,
    low: i64,
    high: i64,
    log_scale: bool,
    step: Option<i64>,
    name: Option<String>,
}

impl IntParam {
    /// Create a new integer parameter sampling uniformly from `[low, high]`.
    #[must_use]
    pub fn new(low: i64, high: i64) -> Self {
        Self {
            id: ParamId::new(),
            low,
            high,
            log_scale: false,
            step: None,
            name: None,
        }
    }

    /// Enable log-scale sampling (bounds must be ≥ 1).
    #[must_use]
    pub fn log_scale(mut self) -> Self {
        self.log_scale = true;
        self
    }

    /// Set a step size for discretized sampling.
    #[must_use]
    pub fn step(mut self, step: i64) -> Self {
        self.step = Some(step);
        self
    }

    /// Set a human-readable name for this parameter.
    ///
    /// When set, this name is used as the parameter's label instead of
    /// the default `Debug` output.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl From<RangeInclusive<i64>> for IntParam {
    fn from(range: RangeInclusive<i64>) -> Self {
        IntParam::new(*range.start(), *range.end())
    }
}

impl Parameter for IntParam {
    type Value = i64;

    fn id(&self) -> ParamId {
        self.id
    }

    fn distribution(&self) -> Distribution {
        Distribution::Int(IntDistribution {
            low: self.low,
            high: self.high,
            log_scale: self.log_scale,
            step: self.step,
        })
    }

    #[allow(clippy::cast_precision_loss)]
    fn cast_param_value(&self, param_value: &ParamValue) -> Result<i64> {
        match param_value {
            ParamValue::Int(v) => Ok(*v),
            _ => Err(Error::Internal("Int distribution should return Int value")),
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn validate(&self) -> Result<()> {
        if self.low > self.high {
            return Err(Error::InvalidBounds {
                low: self.low as f64,
                high: self.high as f64,
            });
        }
        if self.log_scale && self.low < 1 {
            return Err(Error::InvalidLogBounds);
        }
        if let Some(step) = self.step
            && step <= 0
        {
            return Err(Error::InvalidStep);
        }
        Ok(())
    }

    fn label(&self) -> String {
        self.name.clone().unwrap_or_else(|| format!("{self:?}"))
    }
}

/// A categorical parameter that selects from a list of choices.
///
/// The generic type `T` is the element type of the choices vector.
/// The sampler picks an index and the corresponding element is returned.
///
/// # Examples
///
/// ```
/// use optimizer::Trial;
/// use optimizer::parameter::{CategoricalParam, Parameter};
///
/// let mut trial = Trial::new(0);
/// let opt = CategoricalParam::new(vec!["sgd", "adam", "rmsprop"])
///     .name("optimizer")
///     .suggest(&mut trial)
///     .unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct CategoricalParam<T: Clone> {
    id: ParamId,
    choices: Vec<T>,
    name: Option<String>,
}

impl<T: Clone> CategoricalParam<T> {
    /// Create a new categorical parameter with the given choices.
    #[must_use]
    pub fn new(choices: Vec<T>) -> Self {
        Self {
            id: ParamId::new(),
            choices,
            name: None,
        }
    }

    /// Set a human-readable name for this parameter.
    ///
    /// When set, this name is used as the parameter's label instead of
    /// the default `Debug` output.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl<T: Clone + Debug> Parameter for CategoricalParam<T> {
    type Value = T;

    fn id(&self) -> ParamId {
        self.id
    }

    fn distribution(&self) -> Distribution {
        Distribution::Categorical(CategoricalDistribution {
            n_choices: self.choices.len(),
        })
    }

    fn cast_param_value(&self, param_value: &ParamValue) -> Result<T> {
        match param_value {
            ParamValue::Categorical(index) => self
                .choices
                .get(*index)
                .cloned()
                .ok_or(Error::Internal("categorical index out of bounds")),
            _ => Err(Error::Internal(
                "Categorical distribution should return Categorical value",
            )),
        }
    }

    fn validate(&self) -> Result<()> {
        if self.choices.is_empty() {
            return Err(Error::EmptyChoices);
        }
        Ok(())
    }

    fn label(&self) -> String {
        self.name.clone().unwrap_or_else(|| format!("{self:?}"))
    }
}

/// A boolean parameter (equivalent to a two-choice categorical: `false` / `true`).
///
/// # Examples
///
/// ```
/// use optimizer::Trial;
/// use optimizer::parameter::{BoolParam, Parameter};
///
/// let mut trial = Trial::new(0);
/// let use_dropout = BoolParam::new()
///     .name("use_dropout")
///     .suggest(&mut trial)
///     .unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct BoolParam {
    id: ParamId,
    name: Option<String>,
}

impl BoolParam {
    /// Create a new boolean parameter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: ParamId::new(),
            name: None,
        }
    }

    /// Set a human-readable name for this parameter.
    ///
    /// When set, this name is used as the parameter's label instead of
    /// the default `Debug` output.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl Default for BoolParam {
    fn default() -> Self {
        Self::new()
    }
}

impl Parameter for BoolParam {
    type Value = bool;

    fn id(&self) -> ParamId {
        self.id
    }

    fn distribution(&self) -> Distribution {
        Distribution::Categorical(CategoricalDistribution { n_choices: 2 })
    }

    fn cast_param_value(&self, param_value: &ParamValue) -> Result<bool> {
        match param_value {
            ParamValue::Categorical(index) if *index < 2 => Ok(*index != 0),
            ParamValue::Categorical(_) => Err(Error::Internal("bool index out of bounds")),
            _ => Err(Error::Internal(
                "Categorical distribution should return Categorical value",
            )),
        }
    }

    fn label(&self) -> String {
        self.name.clone().unwrap_or_else(|| format!("{self:?}"))
    }
}

/// Map an enum type to sequential indices for use as a categorical parameter.
///
/// This trait converts enum variants to sequential indices and back. It can
/// be derived automatically for fieldless enums using `#[derive(Categorical)]`
/// when the `derive` feature is enabled.
///
/// # Example
///
/// Manual implementation:
///
/// ```
/// use optimizer::parameter::Categorical;
///
/// #[derive(Clone)]
/// enum Activation {
///     Relu,
///     Sigmoid,
///     Tanh,
/// }
///
/// impl Categorical for Activation {
///     const N_CHOICES: usize = 3;
///
///     fn from_index(index: usize) -> Self {
///         match index {
///             0 => Activation::Relu,
///             1 => Activation::Sigmoid,
///             2 => Activation::Tanh,
///             _ => panic!("invalid index"),
///         }
///     }
///
///     fn to_index(&self) -> usize {
///         match self {
///             Activation::Relu => 0,
///             Activation::Sigmoid => 1,
///             Activation::Tanh => 2,
///         }
///     }
/// }
/// ```
pub trait Categorical: Sized + Clone {
    /// The number of variants in the enum.
    const N_CHOICES: usize;

    /// Create an instance from a variant index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= N_CHOICES`.
    fn from_index(index: usize) -> Self;

    /// Return the index of this variant.
    fn to_index(&self) -> usize;
}

/// A parameter that selects from the variants of an enum implementing [`Categorical`].
///
/// Prefer this over [`CategoricalParam`] when the choices map to a Rust enum,
/// because the returned value is already the correct variant — no string
/// matching required.
///
/// # Examples
///
/// ```
/// use optimizer::Trial;
/// use optimizer::parameter::{Categorical, EnumParam, Parameter};
///
/// #[derive(Clone, Debug)]
/// enum Optimizer {
///     Sgd,
///     Adam,
///     Rmsprop,
/// }
///
/// impl Categorical for Optimizer {
///     const N_CHOICES: usize = 3;
///     fn from_index(index: usize) -> Self {
///         match index {
///             0 => Optimizer::Sgd,
///             1 => Optimizer::Adam,
///             2 => Optimizer::Rmsprop,
///             _ => panic!("invalid index"),
///         }
///     }
///     fn to_index(&self) -> usize {
///         match self {
///             Optimizer::Sgd => 0,
///             Optimizer::Adam => 1,
///             Optimizer::Rmsprop => 2,
///         }
///     }
/// }
///
/// let mut trial = Trial::new(0);
/// let opt = EnumParam::<Optimizer>::new()
///     .name("optimizer")
///     .suggest(&mut trial)
///     .unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct EnumParam<T: Categorical> {
    id: ParamId,
    name: Option<String>,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Categorical> EnumParam<T> {
    /// Create a new enum parameter over all variants of `T`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: ParamId::new(),
            name: None,
            _marker: core::marker::PhantomData,
        }
    }

    /// Set a human-readable name for this parameter.
    ///
    /// When set, this name is used as the parameter's label instead of
    /// the default `Debug` output.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl<T: Categorical> Default for EnumParam<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Categorical + Debug> Parameter for EnumParam<T> {
    type Value = T;

    fn id(&self) -> ParamId {
        self.id
    }

    fn distribution(&self) -> Distribution {
        Distribution::Categorical(CategoricalDistribution {
            n_choices: T::N_CHOICES,
        })
    }

    fn cast_param_value(&self, param_value: &ParamValue) -> Result<T> {
        match param_value {
            ParamValue::Categorical(index) if *index < T::N_CHOICES => Ok(T::from_index(*index)),
            ParamValue::Categorical(_) => Err(Error::Internal("categorical index out of bounds")),
            _ => Err(Error::Internal(
                "Categorical distribution should return Categorical value",
            )),
        }
    }

    fn label(&self) -> String {
        self.name.clone().unwrap_or_else(|| format!("{self:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_param_distribution() {
        let param = FloatParam::new(0.0, 1.0);
        assert_eq!(
            param.distribution(),
            Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            })
        );
    }

    #[test]
    fn float_param_log_scale() {
        let param = FloatParam::new(1e-5, 1e-1).log_scale();
        assert_eq!(
            param.distribution(),
            Distribution::Float(FloatDistribution {
                low: 1e-5,
                high: 1e-1,
                log_scale: true,
                step: None,
            })
        );
    }

    #[test]
    fn float_param_step() {
        let param = FloatParam::new(0.0, 1.0).step(0.25);
        assert_eq!(
            param.distribution(),
            Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: Some(0.25),
            })
        );
    }

    #[test]
    fn float_param_validate_invalid_bounds() {
        let param = FloatParam::new(1.0, 0.0);
        assert!(param.validate().is_err());
    }

    #[test]
    fn float_param_validate_invalid_log() {
        let param = FloatParam::new(-1.0, 1.0).log_scale();
        assert!(param.validate().is_err());
    }

    #[test]
    fn float_param_validate_invalid_step() {
        let param = FloatParam::new(0.0, 1.0).step(-0.1);
        assert!(param.validate().is_err());
    }

    #[test]
    fn float_param_validate_nan() {
        assert!(FloatParam::new(f64::NAN, 1.0).validate().is_err());
        assert!(FloatParam::new(0.0, f64::NAN).validate().is_err());
        assert!(FloatParam::new(f64::NAN, f64::NAN).validate().is_err());
    }

    #[test]
    fn float_param_validate_infinity() {
        assert!(FloatParam::new(f64::INFINITY, 1.0).validate().is_err());
        assert!(FloatParam::new(0.0, f64::NEG_INFINITY).validate().is_err());
    }

    #[test]
    fn float_param_validate_nan_step() {
        assert!(FloatParam::new(0.0, 1.0).step(f64::NAN).validate().is_err());
        assert!(
            FloatParam::new(0.0, 1.0)
                .step(f64::INFINITY)
                .validate()
                .is_err()
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn float_param_cast_param_value() {
        let param = FloatParam::new(0.0, 1.0);
        assert_eq!(
            param.cast_param_value(&ParamValue::Float(0.5)).unwrap(),
            0.5
        );
        assert!(param.cast_param_value(&ParamValue::Int(1)).is_err());
    }

    #[test]
    fn int_param_distribution() {
        let param = IntParam::new(1, 100);
        assert_eq!(
            param.distribution(),
            Distribution::Int(IntDistribution {
                low: 1,
                high: 100,
                log_scale: false,
                step: None,
            })
        );
    }

    #[test]
    fn int_param_log_scale() {
        let param = IntParam::new(1, 1024).log_scale();
        assert_eq!(
            param.distribution(),
            Distribution::Int(IntDistribution {
                low: 1,
                high: 1024,
                log_scale: true,
                step: None,
            })
        );
    }

    #[test]
    fn int_param_step() {
        let param = IntParam::new(100, 500).step(50);
        assert_eq!(
            param.distribution(),
            Distribution::Int(IntDistribution {
                low: 100,
                high: 500,
                log_scale: false,
                step: Some(50),
            })
        );
    }

    #[test]
    fn int_param_validate_invalid_bounds() {
        let param = IntParam::new(10, 1);
        assert!(param.validate().is_err());
    }

    #[test]
    fn int_param_validate_invalid_log() {
        let param = IntParam::new(0, 10).log_scale();
        assert!(param.validate().is_err());
    }

    #[test]
    fn int_param_validate_invalid_step() {
        let param = IntParam::new(0, 10).step(-1);
        assert!(param.validate().is_err());
    }

    #[test]
    fn int_param_cast_param_value() {
        let param = IntParam::new(1, 10);
        assert_eq!(param.cast_param_value(&ParamValue::Int(5)).unwrap(), 5);
        assert!(param.cast_param_value(&ParamValue::Float(1.0)).is_err());
    }

    #[test]
    fn categorical_param_distribution() {
        let param = CategoricalParam::new(vec!["a", "b", "c"]);
        assert_eq!(
            param.distribution(),
            Distribution::Categorical(CategoricalDistribution { n_choices: 3 })
        );
    }

    #[test]
    fn categorical_param_validate_empty() {
        let param = CategoricalParam::<&str>::new(vec![]);
        assert!(param.validate().is_err());
    }

    #[test]
    fn categorical_param_cast_param_value() {
        let param = CategoricalParam::new(vec!["sgd", "adam", "rmsprop"]);
        assert_eq!(
            param.cast_param_value(&ParamValue::Categorical(1)).unwrap(),
            "adam"
        );
        assert!(param.cast_param_value(&ParamValue::Float(1.0)).is_err());
    }

    #[test]
    fn categorical_param_cast_out_of_bounds() {
        let param = CategoricalParam::new(vec!["sgd", "adam", "rmsprop"]);
        assert!(param.cast_param_value(&ParamValue::Categorical(3)).is_err());
        assert!(
            param
                .cast_param_value(&ParamValue::Categorical(usize::MAX))
                .is_err()
        );
    }

    #[test]
    fn bool_param_distribution() {
        let param = BoolParam::new();
        assert_eq!(
            param.distribution(),
            Distribution::Categorical(CategoricalDistribution { n_choices: 2 })
        );
    }

    #[test]
    fn bool_param_cast_param_value() {
        let param = BoolParam::new();
        assert!(!param.cast_param_value(&ParamValue::Categorical(0)).unwrap());
        assert!(param.cast_param_value(&ParamValue::Categorical(1)).unwrap());
        assert!(param.cast_param_value(&ParamValue::Float(1.0)).is_err());
    }

    #[test]
    fn bool_param_cast_out_of_bounds() {
        let param = BoolParam::new();
        assert!(param.cast_param_value(&ParamValue::Categorical(2)).is_err());
        assert!(param.cast_param_value(&ParamValue::Categorical(5)).is_err());
    }

    #[derive(Clone, Debug, PartialEq)]
    enum TestEnum {
        A,
        B,
        C,
    }

    impl Categorical for TestEnum {
        const N_CHOICES: usize = 3;

        fn from_index(index: usize) -> Self {
            match index {
                0 => TestEnum::A,
                1 => TestEnum::B,
                2 => TestEnum::C,
                _ => panic!("invalid index"),
            }
        }

        fn to_index(&self) -> usize {
            match self {
                TestEnum::A => 0,
                TestEnum::B => 1,
                TestEnum::C => 2,
            }
        }
    }

    #[test]
    fn enum_param_distribution() {
        let param = EnumParam::<TestEnum>::new();
        assert_eq!(
            param.distribution(),
            Distribution::Categorical(CategoricalDistribution { n_choices: 3 })
        );
    }

    #[test]
    fn enum_param_cast_param_value() {
        let param = EnumParam::<TestEnum>::new();
        assert_eq!(
            param.cast_param_value(&ParamValue::Categorical(0)).unwrap(),
            TestEnum::A
        );
        assert_eq!(
            param.cast_param_value(&ParamValue::Categorical(2)).unwrap(),
            TestEnum::C
        );
        assert!(param.cast_param_value(&ParamValue::Float(1.0)).is_err());
    }

    #[test]
    fn enum_param_cast_out_of_bounds() {
        let param = EnumParam::<TestEnum>::new();
        assert!(param.cast_param_value(&ParamValue::Categorical(3)).is_err());
        assert!(
            param
                .cast_param_value(&ParamValue::Categorical(usize::MAX))
                .is_err()
        );
    }

    #[test]
    fn float_param_suggest_via_trial() {
        let param = FloatParam::new(0.0, 1.0);
        let mut trial = Trial::new(0);
        let x = param.suggest(&mut trial).unwrap();
        assert!((0.0..=1.0).contains(&x));

        // Cached value (same param id) - exact equality expected for cached values
        let x2 = param.suggest(&mut trial).unwrap();
        assert!((x - x2).abs() < f64::EPSILON);
    }

    #[test]
    fn int_param_suggest_via_trial() {
        let param = IntParam::new(1, 10);
        let mut trial = Trial::new(0);
        let n = param.suggest(&mut trial).unwrap();
        assert!((1..=10).contains(&n));

        // Cached value
        let n2 = param.suggest(&mut trial).unwrap();
        assert_eq!(n, n2);
    }

    #[test]
    fn categorical_param_suggest_via_trial() {
        let choices = vec!["sgd", "adam", "rmsprop"];
        let param = CategoricalParam::new(choices.clone());
        let mut trial = Trial::new(0);
        let opt = param.suggest(&mut trial).unwrap();
        assert!(choices.contains(&opt));

        // Cached value
        let opt2 = param.suggest(&mut trial).unwrap();
        assert_eq!(opt, opt2);
    }

    #[test]
    fn bool_param_suggest_via_trial() {
        let param = BoolParam::new();
        let mut trial = Trial::new(0);
        let val = param.suggest(&mut trial).unwrap();
        let _ = val; // just check it doesn't error

        // Cached value
        let val2 = param.suggest(&mut trial).unwrap();
        assert_eq!(val, val2);
    }

    #[test]
    fn enum_param_suggest_via_trial() {
        let param = EnumParam::<TestEnum>::new();
        let mut trial = Trial::new(0);
        let val = param.suggest(&mut trial).unwrap();
        assert!([TestEnum::A, TestEnum::B, TestEnum::C].contains(&val));

        // Cached value
        let val2 = param.suggest(&mut trial).unwrap();
        assert_eq!(val, val2);
    }

    #[test]
    fn parameter_conflict_detection() {
        let param_float = FloatParam::new(0.0, 1.0);
        let param_int = IntParam::new(0, 10);
        let mut trial = Trial::new(0);
        let _ = param_float.suggest(&mut trial).unwrap();

        // Different param with different distribution but same id won't happen
        // since each param gets a unique id. But different param object = different id = no conflict.
        let result = param_int.suggest(&mut trial);
        assert!(result.is_ok()); // Different id, no conflict
    }

    #[test]
    fn float_param_validation_prevents_suggest() {
        let param = FloatParam::new(1.0, 0.0);
        let mut trial = Trial::new(0);
        let result = param.suggest(&mut trial);
        assert!(result.is_err());
    }

    #[test]
    fn categorical_trait_roundtrip() {
        for i in 0..TestEnum::N_CHOICES {
            let val = TestEnum::from_index(i);
            assert_eq!(val.to_index(), i);
        }
    }

    #[test]
    fn param_id_uniqueness() {
        let id1 = ParamId::new();
        let id2 = ParamId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn param_clone_preserves_id() {
        let param = FloatParam::new(0.0, 1.0);
        let cloned = param.clone();
        assert_eq!(param.id(), cloned.id());
    }

    #[test]
    fn float_param_from_range() {
        let param = FloatParam::from(0.0..=1.0);
        assert_eq!(
            param.distribution(),
            Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            })
        );
        assert_eq!(param.label(), format!("{param:?}"));
    }

    #[test]
    fn int_param_from_range() {
        let param = IntParam::from(1..=10);
        assert_eq!(
            param.distribution(),
            Distribution::Int(IntDistribution {
                low: 1,
                high: 10,
                log_scale: false,
                step: None,
            })
        );
        assert_eq!(param.label(), format!("{param:?}"));
    }
}
