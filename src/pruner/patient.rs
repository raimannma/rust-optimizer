use std::collections::HashMap;
use std::sync::Mutex;

use super::Pruner;
use crate::sampler::CompletedTrial;

/// Wraps another pruner and adds a patience window.
///
/// The inner pruner must recommend pruning for `patience` consecutive
/// steps before this pruner actually prunes the trial. This is useful
/// to prevent premature pruning when intermediate values are noisy.
///
/// # Examples
///
/// ```
/// use optimizer::pruner::{PatientPruner, ThresholdPruner};
///
/// // Only prune after the threshold pruner recommends pruning 3 times in a row
/// let inner = ThresholdPruner::new().upper(100.0);
/// let pruner = PatientPruner::new(inner, 3);
/// ```
pub struct PatientPruner {
    inner: Box<dyn Pruner>,
    patience: u64,
    /// Track consecutive prune recommendations per trial.
    consecutive_counts: Mutex<HashMap<u64, u64>>,
}

impl PatientPruner {
    /// Create a new `PatientPruner` wrapping the given inner pruner.
    ///
    /// The inner pruner must recommend pruning for `patience` consecutive
    /// calls before this pruner returns `true`.
    pub fn new(inner: impl Pruner + 'static, patience: u64) -> Self {
        Self {
            inner: Box::new(inner),
            patience,
            consecutive_counts: Mutex::new(HashMap::new()),
        }
    }
}

impl Pruner for PatientPruner {
    fn should_prune(
        &self,
        trial_id: u64,
        step: u64,
        intermediate_values: &[(u64, f64)],
        completed_trials: &[CompletedTrial],
    ) -> bool {
        let inner_says_prune =
            self.inner
                .should_prune(trial_id, step, intermediate_values, completed_trials);
        let mut counts = self.consecutive_counts.lock().expect("lock poisoned");
        let count = counts.entry(trial_id).or_insert(0);
        if inner_says_prune {
            *count += 1;
            *count >= self.patience
        } else {
            *count = 0;
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruner::ThresholdPruner;

    /// A test pruner that always returns the given value.
    struct ConstPruner(bool);

    impl Pruner for ConstPruner {
        fn should_prune(
            &self,
            _trial_id: u64,
            _step: u64,
            _intermediate_values: &[(u64, f64)],
            _completed_trials: &[CompletedTrial],
        ) -> bool {
            self.0
        }
    }

    /// A pruner that returns values from a sequence.
    struct SequencePruner(Mutex<Vec<bool>>);

    impl Pruner for SequencePruner {
        fn should_prune(
            &self,
            _trial_id: u64,
            _step: u64,
            _intermediate_values: &[(u64, f64)],
            _completed_trials: &[CompletedTrial],
        ) -> bool {
            self.0.lock().expect("lock poisoned").remove(0)
        }
    }

    fn call(pruner: &PatientPruner, trial_id: u64, step: u64) -> bool {
        pruner.should_prune(trial_id, step, &[(step, 0.0)], &[])
    }

    #[test]
    fn patience_1_behaves_like_inner() {
        let pruner = PatientPruner::new(ConstPruner(true), 1);
        assert!(call(&pruner, 0, 0));
        assert!(call(&pruner, 0, 1));

        let pruner = PatientPruner::new(ConstPruner(false), 1);
        assert!(!call(&pruner, 0, 0));
        assert!(!call(&pruner, 0, 1));
    }

    #[test]
    fn patience_3_requires_consecutive_recommendations() {
        let pruner = PatientPruner::new(ConstPruner(true), 3);
        assert!(!call(&pruner, 0, 0)); // count=1
        assert!(!call(&pruner, 0, 1)); // count=2
        assert!(call(&pruner, 0, 2)); // count=3 → prune
    }

    #[test]
    fn counter_resets_on_no_prune() {
        // Sequence: prune, prune, no-prune, prune, prune, prune
        let seq = vec![true, true, false, true, true, true];
        let pruner = PatientPruner::new(SequencePruner(Mutex::new(seq)), 3);

        assert!(!call(&pruner, 0, 0)); // count=1
        assert!(!call(&pruner, 0, 1)); // count=2
        assert!(!call(&pruner, 0, 2)); // reset → count=0
        assert!(!call(&pruner, 0, 3)); // count=1
        assert!(!call(&pruner, 0, 4)); // count=2
        assert!(call(&pruner, 0, 5)); // count=3 → prune
    }

    #[test]
    fn independent_per_trial() {
        let pruner = PatientPruner::new(ConstPruner(true), 2);
        assert!(!call(&pruner, 0, 0)); // trial 0: count=1
        assert!(!call(&pruner, 1, 0)); // trial 1: count=1
        assert!(call(&pruner, 0, 1)); // trial 0: count=2 → prune
        assert!(!call(&pruner, 2, 0)); // trial 2: count=1
        assert!(call(&pruner, 1, 1)); // trial 1: count=2 → prune
    }

    #[test]
    fn works_with_threshold_pruner() {
        let inner = ThresholdPruner::new().upper(10.0);
        let pruner = PatientPruner::new(inner, 2);

        // Value below threshold → inner says no
        assert!(!pruner.should_prune(0, 0, &[(0, 5.0)], &[]));
        // Value above threshold → inner says yes, count=1
        assert!(!pruner.should_prune(0, 1, &[(0, 5.0), (1, 15.0)], &[]));
        // Value above threshold again → count=2 → prune
        assert!(pruner.should_prune(0, 2, &[(0, 5.0), (1, 15.0), (2, 20.0)], &[]));
    }
}
