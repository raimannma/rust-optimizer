//! A Tree-Parzen Estimator (TPE) library for black-box optimization.
//!
//! This library provides an Optuna-like API for hyperparameter optimization
//! using the Tree-Parzen Estimator algorithm. It supports:
//!
//! - Float, integer, and categorical parameter types
//! - Log-scale and stepped parameter sampling
//! - Synchronous and async optimization
//! - Parallel trial evaluation with bounded concurrency
//!
//! # Quick Start
//!
//! ```
//! use optimizer::{Direction, Study, TpeSampler};
//!
//! // Create a study with TPE sampler
//! let sampler = TpeSampler::builder().seed(42).build();
//! let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
//!
//! // Optimize x^2 for 20 trials
//! study
//!     .optimize_with_sampler(20, |trial| {
//!         let x = trial.suggest_float("x", -10.0, 10.0)?;
//!         Ok::<_, optimizer::TpeError>(x * x)
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
//! use optimizer::{Direction, RandomSampler, Study, TpeSampler};
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
//!         Ok::<_, optimizer::TpeError>(x * n as f64)
//!     })
//!     .unwrap();
//! ```
//!
//! # Configuring TPE
//!
//! The [`TpeSampler`] can be configured using the builder pattern:
//!
//! ```
//! use optimizer::TpeSampler;
//!
//! let sampler = TpeSampler::builder()
//!     .gamma(0.15)           // Quantile for good/bad split
//!     .n_startup_trials(20)  // Random trials before TPE
//!     .n_ei_candidates(32)   // Candidates to evaluate
//!     .seed(42)              // Reproducibility
//!     .build();
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
mod sampler;
mod study;
mod trial;
mod types;

pub use error::{Result, TpeError};
pub use sampler::{CompletedTrial, RandomSampler, Sampler, TpeSampler, TpeSamplerBuilder};
pub use study::Study;
pub use trial::Trial;
pub use types::{Direction, TrialState};
