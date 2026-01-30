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
//! use optimizer::sampler::tpe::TpeSampler;
//! use optimizer::{Direction, Study};
//!
//! // Create a study with TPE sampler
//! let sampler = TpeSampler::builder().seed(42).build().unwrap();
//! let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
//!
//! // Optimize x^2 for 20 trials
//! study
//!     .optimize_with_sampler(20, |trial| {
//!         let x = trial.suggest_float("x", -10.0, 10.0)?;
//!         Ok::<_, optimizer::Error>(x * x)
//!     })
//!     .unwrap();
//!
//! // Get the best result
//! let best = study.best_trial().unwrap();
//! println!("Best value: {} at x={:?}", best.value, best.params);
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
//! Within the objective function, use [`Trial`] to suggest parameter values:
//!
//! ```
//! use optimizer::{Direction, Study};
//!
//! let study: Study<f64> = Study::new(Direction::Minimize);
//!
//! study
//!     .optimize(10, |trial| {
//!         // Float parameters
//!         let x = trial.suggest_float("x", 0.0, 1.0)?;
//!         let lr = trial.suggest_float_log("learning_rate", 1e-5, 1e-1)?;
//!         let step = trial.suggest_float_step("step", 0.0, 1.0, 0.1)?;
//!
//!         // Integer parameters
//!         let n = trial.suggest_int("n_layers", 1, 10)?;
//!         let batch = trial.suggest_int_log("batch_size", 16, 256)?;
//!         let units = trial.suggest_int_step("units", 32, 512, 32)?;
//!
//!         // Categorical parameters
//!         let optimizer = trial.suggest_categorical("optimizer", &["sgd", "adam", "rmsprop"])?;
//!
//!         // Return objective value
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
//!
//! // Sequential async
//! study.optimize_async(10, |mut trial| async move {
//!     let x = trial.suggest_float("x", 0.0, 1.0)?;
//!     Ok((trial, x * x))
//! }).await?;
//!
//! // Parallel with bounded concurrency
//! study.optimize_parallel(10, 4, |mut trial| async move {
//!     let x = trial.suggest_float("x", 0.0, 1.0)?;
//!     Ok((trial, x * x))
//! }).await?;
//! ```
//!
//! # Feature Flags
//!
//! - `async`: Enable async optimization methods (requires tokio)

mod distribution;
mod error;
mod kde;
mod param;
pub mod sampler;
mod study;
mod trial;
mod types;

pub use error::{Error, Result};
pub use param::ParamValue;
pub use study::Study;
pub use trial::Trial;
pub use types::{Direction, TrialState};
