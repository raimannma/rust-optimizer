use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

use super::Storage;
use crate::sampler::CompletedTrial;

/// In-memory trial storage (the default).
///
/// This is a thin wrapper around `Arc<RwLock<Vec<CompletedTrial<V>>>>`.
pub struct MemoryStorage<V> {
    trials: Arc<RwLock<Vec<CompletedTrial<V>>>>,
    next_id: AtomicU64,
}

impl<V> MemoryStorage<V> {
    /// Creates a new, empty in-memory store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            trials: Arc::new(RwLock::new(Vec::new())),
            next_id: AtomicU64::new(0),
        }
    }

    /// Creates an in-memory store pre-populated with `trials`.
    #[must_use]
    pub fn with_trials(trials: Vec<CompletedTrial<V>>) -> Self {
        let next_id = trials.iter().map(|t| t.id).max().map_or(0, |id| id + 1);
        Self {
            trials: Arc::new(RwLock::new(trials)),
            next_id: AtomicU64::new(next_id),
        }
    }

    /// Ensures the ID counter is at least `min_value`.
    pub(crate) fn bump_next_id(&self, min_value: u64) {
        self.next_id.fetch_max(min_value, Ordering::SeqCst);
    }
}

impl<V> Default for MemoryStorage<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Send + Sync> Storage<V> for MemoryStorage<V> {
    fn push(&self, trial: CompletedTrial<V>) {
        self.trials.write().push(trial);
    }

    fn trials_arc(&self) -> &Arc<RwLock<Vec<CompletedTrial<V>>>> {
        &self.trials
    }

    fn next_trial_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }
}
