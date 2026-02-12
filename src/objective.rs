//! The [`Objective`] trait defines what gets optimized.
//!
//! For simple closures, pass them directly to
//! [`Study::optimize`](crate::Study::optimize):
//!
//! ```
//! use optimizer::prelude::*;
//!
//! let study: Study<f64> = Study::new(Direction::Minimize);
//! let x = FloatParam::new(-10.0, 10.0).name("x");
//!
//! study
//!     .optimize(50, |trial| {
//!         let v = x.suggest(trial)?;
//!         Ok::<_, Error>((v - 3.0).powi(2))
//!     })
//!     .unwrap();
//! ```
//!
//! For richer control — early stopping, retries, or per-trial logging —
//! implement [`Objective`] on a struct and pass it to
//! [`Study::optimize_with`](crate::Study::optimize_with):
//!
//! ```
//! use std::ops::ControlFlow;
//!
//! use optimizer::Objective;
//! use optimizer::prelude::*;
//!
//! struct QuadraticWithEarlyStopping {
//!     x: FloatParam,
//!     target: f64,
//! }
//!
//! impl Objective<f64> for QuadraticWithEarlyStopping {
//!     type Error = Error;
//!
//!     fn evaluate(&self, trial: &mut Trial) -> Result<f64> {
//!         let v = self.x.suggest(trial)?;
//!         Ok((v - 3.0).powi(2))
//!     }
//!
//!     fn after_trial(&self, _study: &Study<f64>, trial: &CompletedTrial<f64>) -> ControlFlow<()> {
//!         if trial.value < self.target {
//!             ControlFlow::Break(())
//!         } else {
//!             ControlFlow::Continue(())
//!         }
//!     }
//! }
//!
//! let study: Study<f64> = Study::new(Direction::Minimize);
//! let obj = QuadraticWithEarlyStopping {
//!     x: FloatParam::new(-10.0, 10.0).name("x"),
//!     target: 1.0,
//! };
//! study.optimize_with(200, obj).unwrap();
//! assert!(study.best_value().unwrap() < 1.0);
//! ```

use core::ops::ControlFlow;

use crate::sampler::CompletedTrial;
use crate::study::Study;
use crate::trial::Trial;

/// Defines an objective function with lifecycle hooks for optimization.
///
/// The only required method is [`evaluate`](Objective::evaluate), which
/// computes the objective value for a given trial. Optional hooks provide
/// early stopping ([`before_trial`](Objective::before_trial),
/// [`after_trial`](Objective::after_trial)) and automatic retries
/// ([`max_retries`](Objective::max_retries)).
///
/// # When to use `Objective` vs a closure
///
/// - **Closure** — pass directly to [`Study::optimize`](crate::Study::optimize)
///   for simple evaluate-only objectives.
/// - **`Objective` struct** — implement this trait when you need hooks
///   (`before_trial`, `after_trial`) or retries.
///
/// # Thread safety
///
/// The async optimization methods (`optimize_async`, `optimize_parallel`)
/// additionally require `Send + Sync + 'static` on the objective. The
/// sync `optimize` method has no thread-safety requirements.
pub trait Objective<V: PartialOrd = f64> {
    /// The error type returned by [`evaluate`](Objective::evaluate).
    type Error: ToString + 'static;

    /// Evaluate the objective function for a single trial.
    ///
    /// Sample parameters from `trial` via
    /// [`Parameter::suggest`](crate::parameter::Parameter::suggest) and
    /// return the objective value. Return `Err(TrialPruned)` to prune a
    /// trial early.
    ///
    /// # Errors
    ///
    /// Any error whose type implements `ToString`. Pruning errors
    /// (`Error::TrialPruned` or `TrialPruned`) are handled specially —
    /// the trial is recorded as pruned rather than failed.
    fn evaluate(&self, trial: &mut Trial) -> Result<V, Self::Error>;

    /// Called before each trial is created.
    ///
    /// Return `ControlFlow::Break(())` to stop the optimization loop
    /// before the next trial starts.
    ///
    /// Default: always continues.
    fn before_trial(&self, _study: &Study<V>) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }

    /// Called after each **completed** trial (not failed or pruned).
    ///
    /// Return `ControlFlow::Break(())` to stop the optimization loop.
    ///
    /// Default: always continues.
    fn after_trial(&self, _study: &Study<V>, _trial: &CompletedTrial<V>) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }

    /// Maximum number of retries for a failed trial.
    ///
    /// When `evaluate` returns a non-pruning error and retries remain,
    /// the same parameter configuration is re-evaluated. Set to `0`
    /// (the default) to disable retries.
    fn max_retries(&self) -> usize {
        0
    }
}
