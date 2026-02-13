//! In-memory storage backend.
//!
//! [`MemoryStorage`] is the default backend used by every
//! [`Study`](crate::Study).  Trials are stored in a
//! `Vec<CompletedTrial<V>>` behind a [`parking_lot::RwLock`] for
//! thread-safe access.
//!
//! # When to use
//!
//! - **Single-process** studies where persistence is not needed.
//! - **Testing** or **prototyping** — zero configuration required.
//! - When you want the **fastest** possible read/write performance
//!   (no disk I/O).
//!
//! For persistent storage that survives process restarts, see
//! `JournalStorage` (requires the `journal` feature).
//!
//! # Example
//!
//! ```
//! use optimizer::prelude::*;
//! use optimizer::storage::MemoryStorage;
//!
//! // Explicit memory storage (equivalent to the default)
//! let storage = MemoryStorage::<f64>::new();
//! let study = Study::builder().minimize().storage(storage).build();
//! ```

use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

use super::Storage;
use crate::sampler::CompletedTrial;

/// In-memory trial storage (the default).
///
/// Wrap a `Vec<CompletedTrial<V>>` behind a read-write lock so that
/// trials can be appended from any thread.  This is the backend that
/// [`Study`](crate::Study) uses when no explicit storage is provided.
///
/// Use [`with_trials`](Self::with_trials) to seed a study with
/// previously collected data.
///
/// # Example
///
/// ```
/// use optimizer::storage::{MemoryStorage, Storage};
///
/// let storage = MemoryStorage::<f64>::new();
/// assert_eq!(storage.trials_arc().read().len(), 0);
/// ```
pub struct MemoryStorage<V> {
    trials: Arc<RwLock<Vec<CompletedTrial<V>>>>,
    next_id: AtomicU64,
}

impl<V> MemoryStorage<V> {
    /// Create a new, empty in-memory store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            trials: Arc::new(RwLock::new(Vec::new())),
            next_id: AtomicU64::new(0),
        }
    }

    /// Create an in-memory store pre-populated with `trials`.
    ///
    /// The internal ID counter is set to one past the highest trial ID
    /// so that subsequent trials receive unique IDs.
    #[must_use]
    pub fn with_trials(trials: Vec<CompletedTrial<V>>) -> Self {
        let next_id = trials.iter().map(|t| t.id).max().map_or(0, |id| id + 1);
        Self {
            trials: Arc::new(RwLock::new(trials)),
            next_id: AtomicU64::new(next_id),
        }
    }

    /// Ensure the ID counter is at least `min_value`.
    #[cfg(feature = "journal")]
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

    fn peek_next_trial_id(&self) -> u64 {
        self.next_id.load(Ordering::SeqCst)
    }
}
