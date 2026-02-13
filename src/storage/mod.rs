//! Trial storage backends.
//!
//! The [`Storage`] trait defines how completed trials are persisted and
//! retrieved.  Every [`Study`](crate::Study) owns an `Arc<dyn Storage<V>>`
//! so storage is transparently shared across threads.
//!
//! # Available backends
//!
//! | Backend | Description | Feature flag |
//! |---------|-------------|-------------|
//! | [`MemoryStorage`] | In-memory `Vec` behind a read-write lock (the default) | — |
//! | `JournalStorage` | JSONL file with `fs2` file locking for multi-process sharing | `journal` |
//!
//! # When to swap backends
//!
//! The default [`MemoryStorage`] is sufficient for single-process studies
//! where persistence is not needed.  Switch to `JournalStorage` when you
//! want to:
//!
//! - **Resume** a study after a process restart.
//! - **Share state** across multiple processes writing to the same file.
//! - **Inspect** trial history in a human-readable JSONL file.
//!
//! # Implementing a custom backend
//!
//! Implement the [`Storage`] trait to plug in your own backend (e.g. a
//! database).  The trait requires four methods: [`push`](Storage::push),
//! [`trials_arc`](Storage::trials_arc), [`next_trial_id`](Storage::next_trial_id),
//! and optionally [`refresh`](Storage::refresh) for external data sources.
//!
//! Inject your storage into a study via the builder:
//!
//! ```
//! use optimizer::prelude::*;
//! use optimizer::storage::MemoryStorage;
//!
//! let storage = MemoryStorage::<f64>::new();
//! let study = Study::builder().minimize().storage(storage).build();
//! ```

#[cfg(feature = "journal")]
mod journal;

use std::sync::Arc;

#[cfg(feature = "journal")]
pub use journal::JournalStorage;
use parking_lot::RwLock;

mod memory;
pub use memory::MemoryStorage;

use crate::sampler::CompletedTrial;

/// Trait for storing and retrieving completed trials.
///
/// Every [`Study`](crate::Study) owns an `Arc<dyn Storage<V>>`.  The
/// default implementation is [`MemoryStorage`], which keeps trials in
/// a plain `Vec` behind a read-write lock.
///
/// Implementations must be `Send + Sync` because a study may be shared
/// across threads (e.g. via `Study::optimize_parallel`).
pub trait Storage<V>: Send + Sync {
    /// Append a completed trial to the store.
    fn push(&self, trial: CompletedTrial<V>);

    /// Return a reference to the in-memory trial buffer.
    ///
    /// All implementations must maintain an `Arc<RwLock<Vec<…>>>` that
    /// reflects the current set of trials.  Callers may acquire a read
    /// lock for efficient, allocation-free access.
    fn trials_arc(&self) -> &Arc<RwLock<Vec<CompletedTrial<V>>>>;

    /// Atomically return the next unique trial ID.
    ///
    /// Each call increments an internal counter so that consecutive
    /// calls always produce distinct IDs.
    fn next_trial_id(&self) -> u64;

    /// Return the current value of the next-trial-ID counter without incrementing.
    ///
    /// This is used for persistence (e.g. `Study::save`) to capture the
    /// counter's exact position, including IDs assigned to failed trials
    /// that are not stored.
    fn peek_next_trial_id(&self) -> u64;

    /// Reload from an external source (e.g. a file written by another
    /// process).  Return `true` if the in-memory buffer was updated.
    ///
    /// The default implementation is a no-op that returns `false`.
    fn refresh(&self) -> bool {
        false
    }
}
