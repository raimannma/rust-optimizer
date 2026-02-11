use super::Pruner;
use crate::sampler::CompletedTrial;

/// A pruner that never prunes. This is the default when no pruner is configured.
pub struct NopPruner;

impl Pruner for NopPruner {
    fn should_prune(
        &self,
        _trial_id: u64,
        _step: u64,
        _intermediate_values: &[(u64, f64)],
        _completed_trials: &[CompletedTrial],
    ) -> bool {
        false
    }
}
