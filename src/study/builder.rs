use core::marker::PhantomData;
use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::pruner::{NopPruner, Pruner};
use crate::sampler::Sampler;
use crate::sampler::random::RandomSampler;
use crate::types::Direction;

use super::Study;

/// A builder for constructing [`Study`] instances with a fluent API.
///
/// Created via [`Study::builder()`]. Collects sampler, pruner, direction,
/// and storage options before constructing the study.
///
/// # Defaults
///
/// - Direction: [`Minimize`](Direction::Minimize)
/// - Sampler: [`RandomSampler`]
/// - Pruner: [`NopPruner`]
/// - Storage: [`MemoryStorage`](crate::storage::MemoryStorage)
///
/// # Examples
///
/// ```
/// use optimizer::prelude::*;
///
/// let study: Study<f64> = Study::builder()
///     .maximize()
///     .sampler(TpeSampler::new())
///     .pruner(MedianPruner::new(Direction::Maximize).n_warmup_steps(5))
///     .build();
///
/// assert_eq!(study.direction(), Direction::Maximize);
/// ```
pub struct StudyBuilder<V: PartialOrd = f64> {
    direction: Direction,
    sampler: Option<Box<dyn Sampler>>,
    pruner: Option<Box<dyn Pruner>>,
    storage: Option<Box<dyn crate::storage::Storage<V>>>,
    _marker: PhantomData<V>,
}

impl<V: PartialOrd> StudyBuilder<V> {
    /// Create a new builder with default settings.
    pub(super) fn new() -> Self {
        Self {
            direction: Direction::Minimize,
            sampler: None,
            pruner: None,
            storage: None,
            _marker: PhantomData,
        }
    }

    /// Set the optimization direction to minimize (the default).
    #[must_use]
    pub fn minimize(mut self) -> Self {
        self.direction = Direction::Minimize;
        self
    }

    /// Set the optimization direction to maximize.
    #[must_use]
    pub fn maximize(mut self) -> Self {
        self.direction = Direction::Maximize;
        self
    }

    /// Set the optimization direction explicitly.
    #[must_use]
    pub fn direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    /// Set the sampler used for parameter suggestions.
    ///
    /// Defaults to [`RandomSampler`] if not specified.
    #[must_use]
    pub fn sampler(mut self, sampler: impl Sampler + 'static) -> Self {
        self.sampler = Some(Box::new(sampler));
        self
    }

    /// Set the pruner used for early stopping of trials.
    ///
    /// Defaults to [`NopPruner`] (no pruning) if not specified.
    #[must_use]
    pub fn pruner(mut self, pruner: impl Pruner + 'static) -> Self {
        self.pruner = Some(Box::new(pruner));
        self
    }

    /// Set a custom storage backend.
    ///
    /// Defaults to [`MemoryStorage`](crate::storage::MemoryStorage) if not specified.
    #[must_use]
    pub fn storage(mut self, storage: impl crate::storage::Storage<V> + 'static) -> Self {
        self.storage = Some(Box::new(storage));
        self
    }

    /// Build the [`Study`] with the configured options.
    #[must_use]
    pub fn build(self) -> Study<V>
    where
        V: Send + Sync + 'static,
    {
        let sampler = self
            .sampler
            .unwrap_or_else(|| Box::new(RandomSampler::new()));
        let pruner = self.pruner.unwrap_or_else(|| Box::new(NopPruner));
        let storage = self
            .storage
            .unwrap_or_else(|| Box::new(crate::storage::MemoryStorage::<V>::new()));

        let sampler: Arc<dyn Sampler> = Arc::from(sampler);
        let pruner: Arc<dyn Pruner> = Arc::from(pruner);
        let storage: Arc<dyn crate::storage::Storage<V>> = Arc::from(storage);
        let trial_factory = Study::make_trial_factory(&sampler, &storage, &pruner);

        Study {
            direction: self.direction,
            sampler,
            pruner,
            storage,
            trial_factory,
            enqueued_params: Arc::new(Mutex::new(VecDeque::new())),
        }
    }
}
