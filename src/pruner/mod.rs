//! Pruner trait and implementations for trial pruning.
//!
//! Pruners decide whether to stop (prune) a trial early based on its
//! intermediate values compared to other trials. This is useful for
//! discarding unpromising trials before they complete, saving compute.

mod median;
mod nop;
mod threshold;

pub use median::MedianPruner;
pub use nop::NopPruner;
pub use threshold::ThresholdPruner;

use crate::sampler::CompletedTrial;

/// Trait for pluggable trial pruning strategies.
///
/// Pruners are consulted after each intermediate value is reported to
/// decide whether the trial should be stopped early. The trait requires
/// `Send + Sync` to support concurrent and async optimization.
///
/// # Implementing a custom pruner
///
/// ```
/// use optimizer::pruner::Pruner;
/// use optimizer::sampler::CompletedTrial;
///
/// struct MyPruner {
///     threshold: f64,
/// }
///
/// impl Pruner for MyPruner {
///     fn should_prune(
///         &self,
///         _trial_id: u64,
///         _step: u64,
///         intermediate_values: &[(u64, f64)],
///         _completed_trials: &[CompletedTrial],
///     ) -> bool {
///         // Prune if the latest value exceeds the threshold
///         intermediate_values
///             .last()
///             .is_some_and(|&(_, v)| v > self.threshold)
///     }
/// }
/// ```
pub trait Pruner: Send + Sync {
    /// Decide whether to prune a trial at the given step.
    ///
    /// # Arguments
    ///
    /// * `trial_id` - The current trial's ID.
    /// * `step` - The step at which the intermediate value was reported.
    /// * `intermediate_values` - All `(step, value)` pairs reported so far for this trial.
    /// * `completed_trials` - History of all completed trials (for comparison).
    fn should_prune(
        &self,
        trial_id: u64,
        step: u64,
        intermediate_values: &[(u64, f64)],
        completed_trials: &[CompletedTrial],
    ) -> bool;
}
