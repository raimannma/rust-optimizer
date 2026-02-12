//! The [`Objective`] trait defines what gets optimized.
//!
//! # Closures work directly
//!
//! Any `Fn(&mut Trial) -> Result<V, E>` closure automatically implements
//! [`Objective`], so you can pass closures straight to
//! [`Study::optimize`](crate::Study::optimize):
//!
//! ```
//! use optimizer::prelude::*;
//!
//! let study: Study<f64> = Study::new(Direction::Minimize);
//! let x = FloatParam::new(-10.0, 10.0).name("x");
//!
//! study
//!     .optimize(50, |trial: &mut optimizer::Trial| {
//!         let v = x.suggest(trial)?;
//!         Ok::<_, Error>((v - 3.0).powi(2))
//!     })
//!     .unwrap();
//! ```
//!
//! # Structs for lifecycle hooks
//!
//! For richer control — early stopping or per-trial logging — implement
//! [`Objective`] on a struct and pass it to the same
//! [`Study::optimize`](crate::Study::optimize) method:
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
//! study.optimize(200, obj).unwrap();
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
/// [`after_trial`](Objective::after_trial)).
///
/// # Closures implement `Objective` automatically
///
/// A blanket implementation covers all `Fn(&mut Trial) -> Result<V, E>`
/// closures, so you can pass closures directly to
/// [`Study::optimize`](crate::Study::optimize) without wrapping them.
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
}

/// Blanket implementation: any `Fn(&mut Trial) -> Result<V, E>` is an
/// `Objective` with no lifecycle hooks.
impl<F, V, E> Objective<V> for F
where
    F: Fn(&mut Trial) -> Result<V, E>,
    V: PartialOrd,
    E: ToString + 'static,
{
    type Error = E;

    fn evaluate(&self, trial: &mut Trial) -> Result<V, E> {
        self(trial)
    }
}
