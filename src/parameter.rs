//! Central parameter trait and built-in parameter types.
//!
//! The [`Parameter`] trait provides a unified way to define parameter types
//! and suggest values from a [`Trial`]. Built-in implementations
//! cover floats, integers, categoricals, booleans, and enum types.
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
//!     .suggest(&mut trial)
//!     .unwrap();
//! let layers = IntParam::new(1, 10).suggest(&mut trial).unwrap();
//! let dropout = BoolParam::new().suggest(&mut trial).unwrap();
//! ```

use core::fmt::Debug;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::distribution::{
    CategoricalDistribution, Distribution, FloatDistribution, IntDistribution,
};
use crate::error::{Error, Result};
use crate::param::ParamValue;
use crate::trial::Trial;

static NEXT_PARAM_ID: AtomicU64 = AtomicU64::new(0);

/// A unique identifier for a parameter instance.
///
/// Each parameter is assigned a unique `ParamId` at creation time. Cloning a parameter
/// copies its `ParamId`, so clones refer to the same logical parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParamId(u64);

impl ParamId {
    /// Creates a new unique `ParamId`.
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

/// A trait for defining parameter types that can be suggested by a [`Trial`].
///
/// Implementors specify the distribution to sample from and how to convert
/// the raw [`ParamValue`] back into a typed value.
pub trait Parameter: Debug {
    /// The typed value returned after sampling.
    type Value;

    /// Returns the unique identifier for this parameter.
    fn id(&self) -> ParamId;

    /// Returns the distribution that this parameter samples from.
    fn distribution(&self) -> Distribution;

    /// Converts a raw [`ParamValue`] into the typed value.
    ///
    /// # Errors
    ///
    /// Returns an error if the `ParamValue` variant doesn't match what this parameter expects.
    fn cast_param_value(&self, param_value: &ParamValue) -> Result<Self::Value>;

    /// Validates the parameter configuration.
    ///
    /// Called before sampling. The default implementation accepts all configurations.
    ///
    /// # Errors
    ///
    /// Returns an error if the parameter configuration is invalid.
    fn validate(&self) -> Result<()> {
        Ok(())
    }

    /// Returns a human-readable label for this parameter.
    ///
    /// Defaults to the `Debug` output of the parameter.
    fn label(&self) -> String {
        format!("{self:?}")
    }

    /// Suggests a value for this parameter from the given trial.
    ///
    /// This is a convenience method that delegates to [`Trial::suggest_param`].
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails, the parameter conflicts with
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
/// # Example
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
/// // Log-scale
/// let lr = FloatParam::new(1e-5, 1e-1)
///     .log_scale()
///     .suggest(&mut trial)
///     .unwrap();
///
/// // Stepped
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
}

impl FloatParam {
    /// Creates a new float parameter with the given bounds.
    #[must_use]
    pub fn new(low: f64, high: f64) -> Self {
        Self {
            id: ParamId::new(),
            low,
            high,
            log_scale: false,
            step: None,
        }
    }

    /// Enables log-scale sampling.
    #[must_use]
    pub fn log_scale(mut self) -> Self {
        self.log_scale = true;
        self
    }

    /// Sets a step size for discretized sampling.
    #[must_use]
    pub fn step(mut self, step: f64) -> Self {
        self.step = Some(step);
        self
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
            && step <= 0.0
        {
            return Err(Error::InvalidStep);
        }
        Ok(())
    }
}

/// An integer parameter with optional log-scale and step size.
///
/// # Example
///
/// ```
/// use optimizer::Trial;
/// use optimizer::parameter::{IntParam, Parameter};
///
/// let mut trial = Trial::new(0);
///
/// // Simple range
/// let n = IntParam::new(1, 10).suggest(&mut trial).unwrap();
///
/// // Log-scale
/// let batch = IntParam::new(1, 1024)
///     .log_scale()
///     .suggest(&mut trial)
///     .unwrap();
///
/// // Stepped
/// let units = IntParam::new(32, 512).step(32).suggest(&mut trial).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct IntParam {
    id: ParamId,
    low: i64,
    high: i64,
    log_scale: bool,
    step: Option<i64>,
}

impl IntParam {
    /// Creates a new integer parameter with the given bounds.
    #[must_use]
    pub fn new(low: i64, high: i64) -> Self {
        Self {
            id: ParamId::new(),
            low,
            high,
            log_scale: false,
            step: None,
        }
    }

    /// Enables log-scale sampling.
    #[must_use]
    pub fn log_scale(mut self) -> Self {
        self.log_scale = true;
        self
    }

    /// Sets a step size for discretized sampling.
    #[must_use]
    pub fn step(mut self, step: i64) -> Self {
        self.step = Some(step);
        self
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
}

/// A categorical parameter that selects from a list of choices.
///
/// # Example
///
/// ```
/// use optimizer::Trial;
/// use optimizer::parameter::{CategoricalParam, Parameter};
///
/// let mut trial = Trial::new(0);
/// let opt = CategoricalParam::new(vec!["sgd", "adam", "rmsprop"])
///     .suggest(&mut trial)
///     .unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct CategoricalParam<T: Clone> {
    id: ParamId,
    choices: Vec<T>,
}

impl<T: Clone> CategoricalParam<T> {
    /// Creates a new categorical parameter with the given choices.
    #[must_use]
    pub fn new(choices: Vec<T>) -> Self {
        Self {
            id: ParamId::new(),
            choices,
        }
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
            ParamValue::Categorical(index) => Ok(self.choices[*index].clone()),
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
}

/// A boolean parameter (equivalent to a categorical with `[false, true]`).
///
/// # Example
///
/// ```
/// use optimizer::Trial;
/// use optimizer::parameter::{BoolParam, Parameter};
///
/// let mut trial = Trial::new(0);
/// let dropout = BoolParam::new().suggest(&mut trial).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct BoolParam {
    id: ParamId,
}

impl BoolParam {
    /// Creates a new boolean parameter.
    #[must_use]
    pub fn new() -> Self {
        Self { id: ParamId::new() }
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
            ParamValue::Categorical(index) => Ok(*index != 0),
            _ => Err(Error::Internal(
                "Categorical distribution should return Categorical value",
            )),
        }
    }
}

/// A trait for enum types that can be used as categorical parameters.
///
/// This trait maps enum variants to sequential indices and back. It can be
/// derived automatically for fieldless enums using `#[derive(Categorical)]`
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

    /// Creates an instance from a variant index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= N_CHOICES`.
    fn from_index(index: usize) -> Self;

    /// Returns the index of this variant.
    fn to_index(&self) -> usize;
}

/// A parameter that selects from the variants of an enum implementing [`Categorical`].
///
/// # Example
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
/// let opt = EnumParam::<Optimizer>::new().suggest(&mut trial).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct EnumParam<T: Categorical> {
    id: ParamId,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Categorical> EnumParam<T> {
    /// Creates a new enum parameter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: ParamId::new(),
            _marker: core::marker::PhantomData,
        }
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
            ParamValue::Categorical(index) => Ok(T::from_index(*index)),
            _ => Err(Error::Internal(
                "Categorical distribution should return Categorical value",
            )),
        }
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
}
