//! Trial storage backends.
//!
//! The [`Storage`] trait defines how completed trials are stored and
//! accessed.  [`MemoryStorage`] keeps trials in memory (the default).
//! With the `journal` feature enabled, [`JournalStorage`] appends
//! trials to a JSONL file with file-level locking so multiple
//! processes can safely share state.

#[cfg(feature = "journal")]
mod journal;
#[cfg(feature = "sqlite")]
mod sqlite;

use std::sync::Arc;

#[cfg(feature = "journal")]
pub use journal::JournalStorage;
use parking_lot::RwLock;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteStorage;

mod memory;
pub use memory::MemoryStorage;

use crate::sampler::CompletedTrial;

/// Trait for storing and retrieving completed trials.
///
/// Every [`Study`](crate::Study) owns an `Arc<dyn Storage<V>>`.  The
/// default implementation is [`MemoryStorage`], which keeps trials in
/// a plain `Vec` behind a read-write lock.
///
/// Implementations must be safe to use from multiple threads.
pub trait Storage<V>: Send + Sync {
    /// Append a completed trial to the store.
    fn push(&self, trial: CompletedTrial<V>);

    /// Return a reference to the in-memory trial buffer.
    ///
    /// All implementations must maintain an `Arc<RwLock<Vec<…>>>` that
    /// reflects the current set of trials.  Callers may acquire a read
    /// lock for efficient, allocation-free access.
    fn trials_arc(&self) -> &Arc<RwLock<Vec<CompletedTrial<V>>>>;

    /// Atomically returns the next unique trial ID.
    ///
    /// Each call increments an internal counter so that consecutive
    /// calls always produce distinct IDs.
    fn next_trial_id(&self) -> u64;

    /// Reload from an external source (e.g. a file written by another
    /// process).  Returns `true` if the in-memory buffer was updated.
    ///
    /// The default implementation is a no-op that returns `false`.
    fn refresh(&self) -> bool {
        false
    }
}
