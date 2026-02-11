#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![deny(unreachable_pub)]
#![deny(clippy::correctness)]
#![deny(clippy::suspicious)]
#![deny(clippy::style)]
#![deny(clippy::complexity)]
#![deny(clippy::perf)]
#![deny(clippy::pedantic)]
#![deny(clippy::std_instead_of_core)]

//! A black-box optimization library with multiple sampling strategies.
//!
//! This library provides an Optuna-like API for hyperparameter optimization
//! with support for multiple sampling algorithms:
//!
//! - **Random Search** - Simple random sampling for baseline comparisons
//! - **TPE (Tree-Parzen Estimator)** - Bayesian optimization for efficient search
//! - **Grid Search** - Exhaustive search over a specified parameter grid
//! - **Sobol (QMC)** - Quasi-random sampling for better space coverage (requires `sobol` feature)
//! - **CMA-ES** - Covariance Matrix Adaptation Evolution Strategy for continuous optimization (requires `cma-es` feature)
//! - **BOHB** - Bayesian Optimization + `HyperBand` for budget-aware TPE sampling
//! - **NSGA-II** - Non-dominated Sorting Genetic Algorithm II for multi-objective optimization
//! - **MOTPE** - Multi-Objective Tree-Parzen Estimator for Bayesian multi-objective optimization
//!
//! Additional features include:
//!
//! - Float, integer, and categorical parameter types
//! - Log-scale and stepped parameter sampling
//! - Synchronous and async optimization
//! - Parallel trial evaluation with bounded concurrency
//!
//! # Quick Start
//!
//! ```
//! use optimizer::prelude::*;
//!
//! // Create a study with TPE sampler
//! let sampler = TpeSampler::builder().seed(42).build().unwrap();
//! let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
//!
//! // Define parameter search space
//! let x = FloatParam::new(-10.0, 10.0).name("x");
//!
//! // Optimize x^2 for 20 trials
//! study
//!     .optimize(20, |trial| {
//!         let x_val = x.suggest(trial)?;
//!         Ok::<_, Error>(x_val * x_val)
//!     })
//!     .unwrap();
//!
//! // Get the best result
//! let best = study.best_trial().unwrap();
//! println!("x = {}", best.get(&x).unwrap());
//! ```
//!
//! # Creating a Study
//!
//! A [`Study`] manages optimization trials. Create one with an optimization direction:
//!
//! ```
//! use optimizer::sampler::random::RandomSampler;
//! use optimizer::sampler::tpe::TpeSampler;
//! use optimizer::{Direction, Study};
//!
//! // Minimize with default random sampler
//! let study: Study<f64> = Study::new(Direction::Minimize);
//!
//! // Maximize with TPE sampler
//! let study: Study<f64> = Study::with_sampler(Direction::Maximize, TpeSampler::new());
//!
//! // With seeded sampler for reproducibility
//! let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
//! ```
//!
//! # Suggesting Parameters
//!
//! Within the objective function, use parameter types to suggest values:
//!
//! ```
//! use optimizer::parameter::{BoolParam, CategoricalParam, FloatParam, IntParam, Parameter};
//! use optimizer::{Direction, Study};
//!
//! let study: Study<f64> = Study::new(Direction::Minimize);
//!
//! // Define parameter search spaces
//! let x_param = FloatParam::new(0.0, 1.0);
//! let lr_param = FloatParam::new(1e-5, 1e-1).log_scale();
//! let step_param = FloatParam::new(0.0, 1.0).step(0.1);
//! let n_param = IntParam::new(1, 10);
//! let batch_param = IntParam::new(16, 256).log_scale();
//! let units_param = IntParam::new(32, 512).step(32);
//! let flag_param = BoolParam::new();
//! let optimizer_param = CategoricalParam::new(vec!["sgd", "adam", "rmsprop"]);
//!
//! study
//!     .optimize(10, |trial| {
//!         let x = x_param.suggest(trial)?;
//!         let lr = lr_param.suggest(trial)?;
//!         let step = step_param.suggest(trial)?;
//!         let n = n_param.suggest(trial)?;
//!         let batch = batch_param.suggest(trial)?;
//!         let units = units_param.suggest(trial)?;
//!         let flag = flag_param.suggest(trial)?;
//!         let optimizer = optimizer_param.suggest(trial)?;
//!
//!         Ok::<_, optimizer::Error>(x * n as f64)
//!     })
//!     .unwrap();
//! ```
//!
//! # Available Samplers
//!
//! ## Random Search
//!
//! The simplest sampling strategy, useful for baselines:
//!
//! ```
//! use optimizer::sampler::random::RandomSampler;
//! use optimizer::{Direction, Study};
//!
//! let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
//! ```
//!
//! ## TPE (Tree-Parzen Estimator)
//!
//! Bayesian optimization that learns from previous trials:
//!
//! ```
//! use optimizer::sampler::tpe::TpeSampler;
//!
//! let sampler = TpeSampler::builder()
//!     .gamma(0.15)           // Quantile for good/bad split
//!     .n_startup_trials(20)  // Random trials before TPE
//!     .n_ei_candidates(32)   // Candidates to evaluate
//!     .seed(42)              // Reproducibility
//!     .build()
//!     .unwrap();
//! ```
//!
//! ## Grid Search
//!
//! Exhaustive search over a discretized parameter space:
//!
//! ```
//! use optimizer::sampler::grid::GridSearchSampler;
//! use optimizer::{Direction, Study};
//!
//! let sampler = GridSearchSampler::builder()
//!     .n_points_per_param(10)  // Points per parameter dimension
//!     .build();
//!
//! let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
//! ```
//!
//! # Async and Parallel Optimization
//!
//! With the `async` feature enabled, you can run trials asynchronously:
//!
//! ```ignore
//! use optimizer::{Study, Direction};
//! use optimizer::parameter::{FloatParam, Parameter};
//!
//! let x_param = FloatParam::new(0.0, 1.0);
//!
//! // Sequential async
//! study.optimize_async(10, |mut trial| {
//!     let x_param = x_param.clone();
//!     async move {
//!         let x = x_param.suggest(&mut trial)?;
//!         Ok((trial, x * x))
//!     }
//! }).await?;
//!
//! // Parallel with bounded concurrency
//! study.optimize_parallel(10, 4, |mut trial| {
//!     let x_param = x_param.clone();
//!     async move {
//!         let x = x_param.suggest(&mut trial)?;
//!         Ok((trial, x * x))
//!     }
//! }).await?;
//! ```
//!
//! # Feature Flags
//!
//! - `async`: Enable async optimization methods (requires tokio)
//! - `derive`: Enable `#[derive(Categorical)]` for enum parameters
//! - `serde`: Enable `Serialize`/`Deserialize` on public types and `Study::save()`/`Study::load()`
//! - `sobol`: Enable the Sobol quasi-random sampler for better space coverage
//! - `cma-es`: Enable the CMA-ES sampler for continuous optimization
//! - `visualization`: Generate self-contained HTML reports with interactive Plotly.js charts
//! - `tracing`: Emit structured log events via the [`tracing`](https://docs.rs/tracing) crate at key optimization points

/// Emit a `tracing::info!` event when the `tracing` feature is enabled.
/// No-op otherwise.
#[cfg(feature = "tracing")]
macro_rules! trace_info {
    ($($arg:tt)*) => { tracing::info!($($arg)*) };
}

#[cfg(not(feature = "tracing"))]
macro_rules! trace_info {
    ($($arg:tt)*) => {};
}

/// Emit a `tracing::debug!` event when the `tracing` feature is enabled.
/// No-op otherwise.
#[cfg(feature = "tracing")]
macro_rules! trace_debug {
    ($($arg:tt)*) => { tracing::debug!($($arg)*) };
}

#[cfg(not(feature = "tracing"))]
macro_rules! trace_debug {
    ($($arg:tt)*) => {};
}

mod distribution;
mod error;
mod importance;
mod kde;
pub mod multi_objective;
mod param;
pub mod parameter;
pub mod pareto;
pub mod pruner;
pub mod sampler;
mod study;
mod trial;
mod types;
#[cfg(feature = "visualization")]
mod visualization;

pub use error::{Error, Result, TrialPruned};
pub use multi_objective::{MultiObjectiveSampler, MultiObjectiveStudy, MultiObjectiveTrial};
#[cfg(feature = "derive")]
pub use optimizer_derive::Categorical;
pub use param::ParamValue;
pub use parameter::{
    BoolParam, Categorical, CategoricalParam, EnumParam, FloatParam, IntParam, ParamId, Parameter,
};
pub use pruner::{
    HyperbandPruner, MedianPruner, NopPruner, PatientPruner, PercentilePruner, Pruner,
    SuccessiveHalvingPruner, ThresholdPruner, WilcoxonPruner,
};
pub use sampler::CompletedTrial;
pub use sampler::bohb::BohbSampler;
#[cfg(feature = "cma-es")]
pub use sampler::cma_es::CmaEsSampler;
pub use sampler::grid::GridSearchSampler;
pub use sampler::motpe::MotpeSampler;
pub use sampler::nsga2::Nsga2Sampler;
pub use sampler::random::RandomSampler;
#[cfg(feature = "sobol")]
pub use sampler::sobol::SobolSampler;
pub use sampler::tpe::TpeSampler;
pub use study::Study;
#[cfg(feature = "serde")]
pub use study::StudySnapshot;
pub use trial::{AttrValue, Trial};
pub use types::{Direction, TrialState};
#[cfg(feature = "visualization")]
pub use visualization::generate_html_report;

/// Convenient wildcard import for the most common types.
///
/// ```
/// use optimizer::prelude::*;
/// ```
pub mod prelude {
    #[cfg(feature = "derive")]
    pub use optimizer_derive::Categorical as DeriveCategory;

    pub use crate::error::{Error, Result, TrialPruned};
    pub use crate::multi_objective::{MultiObjectiveStudy, MultiObjectiveTrial};
    pub use crate::param::ParamValue;
    pub use crate::parameter::{
        BoolParam, Categorical, CategoricalParam, EnumParam, FloatParam, IntParam, Parameter,
    };
    pub use crate::pruner::{
        HyperbandPruner, MedianPruner, NopPruner, PatientPruner, PercentilePruner, Pruner,
        SuccessiveHalvingPruner, ThresholdPruner,
    };
    pub use crate::sampler::CompletedTrial;
    pub use crate::sampler::bohb::BohbSampler;
    #[cfg(feature = "cma-es")]
    pub use crate::sampler::cma_es::CmaEsSampler;
    pub use crate::sampler::grid::GridSearchSampler;
    pub use crate::sampler::motpe::MotpeSampler;
    pub use crate::sampler::nsga2::Nsga2Sampler;
    pub use crate::sampler::random::RandomSampler;
    #[cfg(feature = "sobol")]
    pub use crate::sampler::sobol::SobolSampler;
    pub use crate::sampler::tpe::TpeSampler;
    pub use crate::study::Study;
    #[cfg(feature = "serde")]
    pub use crate::study::StudySnapshot;
    pub use crate::trial::{AttrValue, Trial};
    pub use crate::types::Direction;
    #[cfg(feature = "visualization")]
    pub use crate::visualization::generate_html_report;
}
