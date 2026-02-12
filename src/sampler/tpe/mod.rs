//! Tree-Parzen Estimator (TPE) sampler family for Bayesian optimization.
//!
//! TPE is a sequential model-based optimization algorithm that models P(x|y) instead
//! of P(y|x). It splits completed trials into "good" (below the gamma quantile) and
//! "bad" groups, fits a kernel density estimator (KDE) to each, and proposes new
//! points by maximizing the l(x)/g(x) ratio — an approximation of Expected Improvement.
//!
//! # Samplers
//!
//! | Sampler | Models parameters | Best for |
//! |---------|-------------------|----------|
//! | [`TpeSampler`] | Independently | General-purpose single-objective optimization |
//! | [`MultivariateTpeSampler`] | Jointly | Problems with correlated parameters |
//!
//! # Gamma strategies
//!
//! The gamma quantile controls how many trials are considered "good". This module
//! provides four built-in strategies via the [`GammaStrategy`] trait:
//!
//! | Strategy | Formula | Default |
//! |----------|---------|---------|
//! | [`FixedGamma`] | Constant value | gamma = 0.25 |
//! | [`LinearGamma`] | Linear ramp from min to max | 0.10 → 0.25 over 100 trials |
//! | [`SqrtGamma`] | 1/√n decay (Optuna-style) | factor = 1.0, max = 0.25 |
//! | [`HyperoptGamma`] | (base+1)/n (Hyperopt-style) | base = 24, max = 0.25 |
//!
//! You can also implement [`GammaStrategy`] for a custom splitting rule.
//!
//! # Search-space utilities
//!
//! The [`search_space`] submodule provides [`IntersectionSearchSpace`] for computing
//! the common parameter set across trials, and [`GroupDecomposedSearchSpace`] for
//! splitting parameters into independent groups based on co-occurrence.
//!
//! # Examples
//!
//! Basic TPE with default settings:
//!
//! ```
//! use optimizer::sampler::tpe::TpeSampler;
//! use optimizer::{Direction, Study};
//!
//! let study: Study<f64> = Study::with_sampler(Direction::Minimize, TpeSampler::new());
//! ```
//!
//! Multivariate TPE for correlated parameters:
//!
//! ```
//! use optimizer::sampler::tpe::MultivariateTpeSampler;
//! use optimizer::{Direction, Study};
//!
//! let sampler = MultivariateTpeSampler::builder()
//!     .gamma(0.15)
//!     .n_startup_trials(20)
//!     .group(true)
//!     .seed(42)
//!     .build()
//!     .unwrap();
//! let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
//! ```

pub(crate) mod common;
mod gamma;
mod multivariate;
mod sampler;
pub mod search_space;

pub use gamma::{FixedGamma, GammaStrategy, HyperoptGamma, LinearGamma, SqrtGamma};
pub use multivariate::{
    ConstantLiarStrategy, MultivariateTpeSampler, MultivariateTpeSamplerBuilder,
};
pub use sampler::{TpeSampler, TpeSamplerBuilder};
pub use search_space::{GroupDecomposedSearchSpace, IntersectionSearchSpace};
