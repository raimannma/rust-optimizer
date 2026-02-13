//! Study implementation for managing optimization trials.

use core::any::Any;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use crate::param::ParamValue;
use crate::parameter::ParamId;
use crate::pruner::{NopPruner, Pruner};
use crate::sampler::random::RandomSampler;
use crate::sampler::{CompletedTrial, Sampler};
use crate::trial::Trial;
use crate::types::{Direction, TrialState};

mod analysis;
mod builder;
mod export;
mod iter;
mod optimize;
mod persistence;

#[cfg(feature = "async")]
mod async_impl;

pub use builder::StudyBuilder;
#[cfg(feature = "serde")]
pub use persistence::StudySnapshot;

/// A study manages the optimization process, tracking trials and their results.
///
/// The study is parameterized by the objective value type `V`, which defaults to `f64`.
/// The only constraint on `V` is `PartialOrd`, allowing comparison of objective values
/// to determine which trial is best.
///
/// When `V = f64`, the study passes trial history to the sampler for informed
/// parameter suggestions (e.g., TPE sampler uses history to guide sampling).
///
/// # Examples
///
/// ```
/// use optimizer::{Direction, Study};
///
/// // Create a study to minimize an objective function
/// let study: Study<f64> = Study::new(Direction::Minimize);
/// assert_eq!(study.direction(), Direction::Minimize);
/// ```
pub struct Study<V = f64>
where
    V: PartialOrd,
{
    /// The optimization direction.
    pub(crate) direction: Direction,
    /// The sampler used to generate parameter values.
    pub(crate) sampler: Arc<dyn Sampler>,
    /// The pruner used to decide whether to stop trials early.
    pub(crate) pruner: Arc<dyn Pruner>,
    /// Trial storage backend (default: [`MemoryStorage`](crate::storage::MemoryStorage)).
    pub(crate) storage: Arc<dyn crate::storage::Storage<V>>,
    /// Optional factory for creating sampler-aware trials.
    /// Set automatically for `Study<f64>` so that `create_trial()` and all
    /// optimization methods use the sampler without requiring `_with_sampler` suffixes.
    pub(crate) trial_factory: Option<Arc<dyn Fn(u64) -> Trial + Send + Sync>>,
    /// Queue of parameter configurations to evaluate next.
    pub(crate) enqueued_params: Arc<Mutex<VecDeque<HashMap<ParamId, ParamValue>>>>,
}

impl<V> Study<V>
where
    V: PartialOrd,
{
    /// Create a new study with the given optimization direction.
    ///
    /// Uses the default `RandomSampler` for parameter sampling.
    ///
    /// # Arguments
    ///
    /// * `direction` - Whether to minimize or maximize the objective function.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// assert_eq!(study.direction(), Direction::Minimize);
    /// ```
    #[must_use]
    pub fn new(direction: Direction) -> Self
    where
        V: Send + Sync + 'static,
    {
        Self::with_sampler(direction, RandomSampler::new())
    }

    /// Return a [`StudyBuilder`] for constructing a study with a fluent API.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::prelude::*;
    ///
    /// let study: Study<f64> = Study::builder()
    ///     .minimize()
    ///     .sampler(TpeSampler::new())
    ///     .pruner(NopPruner)
    ///     .build();
    /// ```
    #[must_use]
    pub fn builder() -> StudyBuilder<V> {
        StudyBuilder::new()
    }

    /// Create a study that minimizes the objective value.
    ///
    /// This is a shorthand for `Study::with_sampler(Direction::Minimize, sampler)`.
    ///
    /// # Arguments
    ///
    /// * `sampler` - The sampler to use for parameter sampling.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::Study;
    /// use optimizer::sampler::tpe::TpeSampler;
    ///
    /// let study: Study<f64> = Study::minimize(TpeSampler::new());
    /// assert_eq!(study.direction(), optimizer::Direction::Minimize);
    /// ```
    #[must_use]
    pub fn minimize(sampler: impl Sampler + 'static) -> Self
    where
        V: Send + Sync + 'static,
    {
        Self::with_sampler(Direction::Minimize, sampler)
    }

    /// Create a study that maximizes the objective value.
    ///
    /// This is a shorthand for `Study::with_sampler(Direction::Maximize, sampler)`.
    ///
    /// # Arguments
    ///
    /// * `sampler` - The sampler to use for parameter sampling.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::Study;
    /// use optimizer::sampler::tpe::TpeSampler;
    ///
    /// let study: Study<f64> = Study::maximize(TpeSampler::new());
    /// assert_eq!(study.direction(), optimizer::Direction::Maximize);
    /// ```
    #[must_use]
    pub fn maximize(sampler: impl Sampler + 'static) -> Self
    where
        V: Send + Sync + 'static,
    {
        Self::with_sampler(Direction::Maximize, sampler)
    }

    /// Create a new study with a custom sampler.
    ///
    /// # Arguments
    ///
    /// * `direction` - Whether to minimize or maximize the objective function.
    /// * `sampler` - The sampler to use for parameter sampling.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::random::RandomSampler;
    /// use optimizer::{Direction, Study};
    ///
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Maximize, sampler);
    /// assert_eq!(study.direction(), Direction::Maximize);
    /// ```
    pub fn with_sampler(direction: Direction, sampler: impl Sampler + 'static) -> Self
    where
        V: Send + Sync + 'static,
    {
        Self::with_sampler_and_storage(
            direction,
            sampler,
            crate::storage::MemoryStorage::<V>::new(),
        )
    }

    /// Build a trial factory for sampler integration when `V = f64`.
    pub(crate) fn make_trial_factory(
        sampler: &Arc<dyn Sampler>,
        storage: &Arc<dyn crate::storage::Storage<V>>,
        pruner: &Arc<dyn Pruner>,
    ) -> Option<Arc<dyn Fn(u64) -> Trial + Send + Sync>>
    where
        V: 'static,
    {
        // Try to downcast the storage's trial buffer to the f64 specialization.
        // This succeeds only when V = f64, enabling automatic sampler integration.
        let trials_arc = storage.trials_arc();
        let any_ref: &dyn Any = trials_arc;
        let f64_trials: Option<&Arc<RwLock<Vec<CompletedTrial<f64>>>>> = any_ref.downcast_ref();

        f64_trials.map(|trials| {
            let sampler = Arc::clone(sampler);
            let trials = Arc::clone(trials);
            let pruner = Arc::clone(pruner);
            let factory: Arc<dyn Fn(u64) -> Trial + Send + Sync> = Arc::new(move |id| {
                Trial::with_sampler(
                    id,
                    Arc::clone(&sampler),
                    Arc::clone(&trials),
                    Arc::clone(&pruner),
                )
            });
            factory
        })
    }

    /// Create a study with a custom sampler and storage backend.
    ///
    /// This is the most general constructor — all other constructors
    /// delegate to this one. Use it when you need a non-default storage
    /// backend (e.g., `JournalStorage`).
    ///
    /// # Arguments
    ///
    /// * `direction` - Whether to minimize or maximize the objective function.
    /// * `sampler` - The sampler to use for parameter sampling.
    /// * `storage` - The storage backend for completed trials.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::random::RandomSampler;
    /// use optimizer::storage::MemoryStorage;
    /// use optimizer::{Direction, Study};
    ///
    /// let storage = MemoryStorage::<f64>::new();
    /// let study = Study::with_sampler_and_storage(Direction::Minimize, RandomSampler::new(), storage);
    /// ```
    pub fn with_sampler_and_storage(
        direction: Direction,
        sampler: impl Sampler + 'static,
        storage: impl crate::storage::Storage<V> + 'static,
    ) -> Self
    where
        V: 'static,
    {
        let sampler: Arc<dyn Sampler> = Arc::new(sampler);
        let pruner: Arc<dyn Pruner> = Arc::new(NopPruner);
        let storage: Arc<dyn crate::storage::Storage<V>> = Arc::new(storage);
        let trial_factory = Self::make_trial_factory(&sampler, &storage, &pruner);

        Self {
            direction,
            sampler,
            pruner,
            storage,
            trial_factory,
            enqueued_params: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Return the optimization direction.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Creates a study with a custom sampler and pruner.
    ///
    /// Uses the default [`MemoryStorage`](crate::storage::MemoryStorage) backend.
    ///
    /// # Arguments
    ///
    /// * `direction` - Whether to minimize or maximize the objective function.
    /// * `sampler` - The sampler to use for parameter sampling.
    /// * `pruner` - The pruner to use for trial pruning.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::pruner::NopPruner;
    /// use optimizer::sampler::random::RandomSampler;
    /// use optimizer::{Direction, Study};
    ///
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler_and_pruner(Direction::Minimize, sampler, NopPruner);
    /// ```
    pub fn with_sampler_and_pruner(
        direction: Direction,
        sampler: impl Sampler + 'static,
        pruner: impl Pruner + 'static,
    ) -> Self
    where
        V: Send + Sync + 'static,
    {
        let sampler: Arc<dyn Sampler> = Arc::new(sampler);
        let pruner: Arc<dyn Pruner> = Arc::new(pruner);
        let storage: Arc<dyn crate::storage::Storage<V>> =
            Arc::new(crate::storage::MemoryStorage::<V>::new());
        let trial_factory = Self::make_trial_factory(&sampler, &storage, &pruner);

        Self {
            direction,
            sampler,
            pruner,
            storage,
            trial_factory,
            enqueued_params: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Replace the sampler used for future parameter suggestions.
    ///
    /// The new sampler takes effect for all subsequent calls to
    /// [`create_trial`](Self::create_trial), [`ask`](Self::ask), and the
    /// `optimize*` family. Already-completed trials are unaffected.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::tpe::TpeSampler;
    /// use optimizer::{Direction, Study};
    ///
    /// let mut study: Study<f64> = Study::new(Direction::Minimize);
    /// study.set_sampler(TpeSampler::new());
    /// ```
    pub fn set_sampler(&mut self, sampler: impl Sampler + 'static)
    where
        V: 'static,
    {
        self.sampler = Arc::new(sampler);
        self.trial_factory = Self::make_trial_factory(&self.sampler, &self.storage, &self.pruner);
    }

    /// Replace the pruner used for future trials.
    ///
    /// The new pruner takes effect for all trials created after this call.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::prelude::*;
    ///
    /// let mut study: Study<f64> = Study::new(Direction::Minimize);
    /// study.set_pruner(MedianPruner::new(Direction::Minimize));
    /// ```
    pub fn set_pruner(&mut self, pruner: impl Pruner + 'static)
    where
        V: 'static,
    {
        self.pruner = Arc::new(pruner);
        self.trial_factory = Self::make_trial_factory(&self.sampler, &self.storage, &self.pruner);
    }

    /// Return a reference to the study's current pruner.
    #[must_use]
    pub fn pruner(&self) -> &dyn Pruner {
        &*self.pruner
    }

    /// Enqueue a specific parameter configuration to be evaluated next.
    ///
    /// The next call to [`ask()`](Self::ask) or the next trial in [`optimize()`](Self::optimize)
    /// will use these exact parameters instead of sampling from the sampler.
    ///
    /// Multiple configurations can be enqueued; they are evaluated in FIFO order.
    /// If an enqueued configuration is missing a parameter that the objective calls
    /// `suggest()` on, that parameter falls back to normal sampling.
    ///
    /// # Arguments
    ///
    /// * `params` - A map from parameter IDs to the values to use.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// use optimizer::parameter::{FloatParam, IntParam, ParamValue, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let x = FloatParam::new(0.0, 10.0);
    /// let y = IntParam::new(1, 100);
    ///
    /// // Evaluate these specific configurations first
    /// study.enqueue(HashMap::from([
    ///     (x.id(), ParamValue::Float(0.001)),
    ///     (y.id(), ParamValue::Int(3)),
    /// ]));
    ///
    /// // Next trial will use x=0.001, y=3
    /// let mut trial = study.ask();
    /// assert_eq!(x.suggest(&mut trial).unwrap(), 0.001);
    /// assert_eq!(y.suggest(&mut trial).unwrap(), 3);
    /// ```
    pub fn enqueue(&self, params: HashMap<ParamId, ParamValue>) {
        self.enqueued_params.lock().push_back(params);
    }

    /// Return the trial ID of the current best trial from the given slice.
    #[cfg(feature = "tracing")]
    pub(crate) fn best_id(&self, trials: &[CompletedTrial<V>]) -> Option<u64> {
        let direction = self.direction;
        trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .max_by(|a, b| Self::compare_trials(a, b, direction))
            .map(|t| t.id)
    }

    /// Return the number of enqueued parameter configurations.
    ///
    /// See [`enqueue`](Self::enqueue) for how to add configurations.
    #[must_use]
    pub fn n_enqueued(&self) -> usize {
        self.enqueued_params.lock().len()
    }

    /// Generate the next unique trial ID.
    pub(crate) fn next_trial_id(&self) -> u64 {
        self.storage.next_trial_id()
    }

    /// Create a new trial with a unique ID.
    ///
    /// The trial starts in the `Running` state and can be used to suggest
    /// parameter values. After the objective function is evaluated, call
    /// `complete_trial` or `fail_trial` to record the result.
    ///
    /// For `Study<f64>`, this method automatically integrates with the study's
    /// sampler and trial history, so there is no need to call a separate
    /// `create_trial_with_sampler()` method.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let trial = study.create_trial();
    /// assert_eq!(trial.id(), 0);
    ///
    /// let trial2 = study.create_trial();
    /// assert_eq!(trial2.id(), 1);
    /// ```
    #[must_use]
    pub fn create_trial(&self) -> Trial {
        self.storage.refresh();

        let id = self.next_trial_id();
        let mut trial = if let Some(factory) = &self.trial_factory {
            factory(id)
        } else {
            Trial::new(id)
        };

        // If there are enqueued params, inject them into this trial
        if let Some(fixed_params) = self.enqueued_params.lock().pop_front() {
            trial.set_fixed_params(fixed_params);
        }

        trial
    }

    /// Record a completed trial with its objective value.
    ///
    /// This method stores the trial's parameters, distributions, and objective
    /// value in the study's history. The stored data is used by samplers to
    /// inform future parameter suggestions.
    ///
    /// # Arguments
    ///
    /// * `trial` - The trial that was evaluated.
    /// * `value` - The objective value returned by the objective function.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let x_param = FloatParam::new(0.0, 1.0);
    /// let mut trial = study.create_trial();
    /// let x = x_param.suggest(&mut trial).unwrap();
    /// let objective_value = x * x;
    /// study.complete_trial(trial, objective_value);
    ///
    /// assert_eq!(study.n_trials(), 1);
    /// ```
    pub fn complete_trial(&self, trial: Trial, value: V) {
        let completed = trial.into_completed(value, TrialState::Complete);
        self.storage.push(completed);
    }

    /// Record a failed trial with an error message.
    ///
    /// Failed trials are not stored in the study's history and do not
    /// contribute to future sampling decisions. This method is useful
    /// when the objective function raises an error that should not stop
    /// the optimization process.
    ///
    /// # Arguments
    ///
    /// * `trial` - The trial that failed.
    /// * `_error` - An error message describing why the trial failed.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let trial = study.create_trial();
    /// study.fail_trial(trial, "objective function raised an exception");
    ///
    /// // Failed trials are not counted
    /// assert_eq!(study.n_trials(), 0);
    /// ```
    pub fn fail_trial(&self, mut trial: Trial, _error: impl ToString) {
        trial.set_failed();
        // Failed trials are not stored in completed_trials
        // They could be stored in a separate list for debugging if needed
    }

    /// Request a new trial with suggested parameters.
    ///
    /// This is the first half of the ask-and-tell interface. After calling
    /// `ask()`, use parameter types to suggest values on the returned trial,
    /// evaluate your objective externally, then pass the trial back to
    /// [`tell()`](Self::tell) with the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let x = FloatParam::new(0.0, 10.0);
    ///
    /// let mut trial = study.ask();
    /// let x_val = x.suggest(&mut trial).unwrap();
    /// let value = x_val * x_val;
    /// study.tell(trial, Ok::<_, &str>(value));
    /// ```
    #[must_use]
    pub fn ask(&self) -> Trial {
        self.create_trial()
    }

    /// Report the result of a trial obtained from [`ask()`](Self::ask).
    ///
    /// Pass `Ok(value)` for a successful evaluation or `Err(reason)` for a
    /// failure. Failed trials are not stored in the study's history.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    ///
    /// let trial = study.ask();
    /// study.tell(trial, Ok::<_, &str>(42.0));
    /// assert_eq!(study.n_trials(), 1);
    ///
    /// let trial = study.ask();
    /// study.tell(trial, Err::<f64, _>("evaluation failed"));
    /// assert_eq!(study.n_trials(), 1); // failed trials not counted
    /// ```
    pub fn tell(&self, trial: Trial, value: core::result::Result<V, impl ToString>) {
        match value {
            Ok(v) => self.complete_trial(trial, v),
            Err(e) => self.fail_trial(trial, e),
        }
    }

    /// Record a pruned trial, preserving its intermediate values.
    ///
    /// Pruned trials are stored alongside completed trials so that samplers
    /// can optionally learn from partial evaluations. The trial's state is
    /// set to [`Pruned`](crate::TrialState::Pruned).
    ///
    /// In practice you rarely call this directly — returning
    /// `Err(TrialPruned)` from an objective function handles pruning
    /// automatically.
    ///
    /// # Arguments
    ///
    /// * `trial` - The trial that was pruned.
    pub fn prune_trial(&self, trial: Trial)
    where
        V: Default,
    {
        let completed = trial.into_completed(V::default(), TrialState::Pruned);
        self.storage.push(completed);
    }

    /// Return all completed trials as a `Vec`.
    ///
    /// The returned vector contains clones of `CompletedTrial` values, which contain
    /// the trial's parameters, distributions, and objective value.
    ///
    /// Note: This method acquires a read lock on the completed trials, so the
    /// returned vector is a clone of the internal storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let x_param = FloatParam::new(0.0, 1.0);
    /// let mut trial = study.create_trial();
    /// let _ = x_param.suggest(&mut trial);
    /// study.complete_trial(trial, 0.5);
    ///
    /// for completed in study.trials() {
    ///     println!("Trial {} has value {:?}", completed.id, completed.value);
    /// }
    /// ```
    #[must_use]
    pub fn trials(&self) -> Vec<CompletedTrial<V>>
    where
        V: Clone,
    {
        self.storage.trials_arc().read().clone()
    }

    /// Return the number of completed trials.
    ///
    /// Pruned and failed trials are not counted. Use
    /// [`n_pruned_trials()`](Self::n_pruned_trials) for the pruned count.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// assert_eq!(study.n_trials(), 0);
    ///
    /// let x_param = FloatParam::new(0.0, 1.0);
    /// let mut trial = study.create_trial();
    /// let _ = x_param.suggest(&mut trial);
    /// study.complete_trial(trial, 0.5);
    /// assert_eq!(study.n_trials(), 1);
    /// ```
    #[must_use]
    pub fn n_trials(&self) -> usize {
        self.storage
            .trials_arc()
            .read()
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .count()
    }

    /// Return the number of pruned trials.
    ///
    /// Pruned trials are those that were stopped early by the pruner.
    #[must_use]
    pub fn n_pruned_trials(&self) -> usize {
        self.storage
            .trials_arc()
            .read()
            .iter()
            .filter(|t| t.state == TrialState::Pruned)
            .count()
    }

    /// Compare two completed trials using constraint-aware ranking.
    ///
    /// 1. Feasible trials always rank above infeasible trials.
    /// 2. Among feasible trials, rank by objective value (respecting direction).
    /// 3. Among infeasible trials, rank by total constraint violation (lower is better).
    pub(crate) fn compare_trials(
        a: &CompletedTrial<V>,
        b: &CompletedTrial<V>,
        direction: Direction,
    ) -> core::cmp::Ordering {
        match (a.is_feasible(), b.is_feasible()) {
            (true, false) => core::cmp::Ordering::Greater,
            (false, true) => core::cmp::Ordering::Less,
            (false, false) => {
                let va: f64 = a.constraints.iter().map(|c| c.max(0.0)).sum();
                let vb: f64 = b.constraints.iter().map(|c| c.max(0.0)).sum();
                vb.partial_cmp(&va).unwrap_or(core::cmp::Ordering::Equal)
            }
            (true, true) => {
                let ordering = a.value.partial_cmp(&b.value);
                match direction {
                    Direction::Minimize => {
                        ordering.map_or(core::cmp::Ordering::Equal, core::cmp::Ordering::reverse)
                    }
                    Direction::Maximize => ordering.unwrap_or(core::cmp::Ordering::Equal),
                }
            }
        }
    }
}

impl<V: PartialOrd + Send + Sync + 'static> Study<V> {
    /// Create a study with a custom sampler, pruner, and storage backend.
    ///
    /// The most flexible constructor, allowing full control over all components.
    ///
    /// # Arguments
    ///
    /// * `direction` - Whether to minimize or maximize the objective function.
    /// * `sampler` - The sampler to use for parameter sampling.
    /// * `pruner` - The pruner to use for trial pruning.
    /// * `storage` - The storage backend for completed trials.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::prelude::*;
    /// use optimizer::storage::MemoryStorage;
    ///
    /// let study = Study::with_sampler_pruner_and_storage(
    ///     Direction::Minimize,
    ///     TpeSampler::new(),
    ///     MedianPruner::new(Direction::Minimize),
    ///     MemoryStorage::<f64>::new(),
    /// );
    /// ```
    pub fn with_sampler_pruner_and_storage(
        direction: Direction,
        sampler: impl Sampler + 'static,
        pruner: impl Pruner + 'static,
        storage: impl crate::storage::Storage<V> + 'static,
    ) -> Self {
        let sampler: Arc<dyn Sampler> = Arc::new(sampler);
        let pruner: Arc<dyn Pruner> = Arc::new(pruner);
        let storage: Arc<dyn crate::storage::Storage<V>> = Arc::new(storage);
        let trial_factory = Self::make_trial_factory(&sampler, &storage, &pruner);

        Self {
            direction,
            sampler,
            pruner,
            storage,
            trial_factory,
            enqueued_params: Arc::new(Mutex::new(VecDeque::new())),
        }
    }
}

/// Returns `true` if the error represents a pruned trial.
///
/// Checks via `Any` downcasting whether `e` is `Error::TrialPruned` or
/// the standalone `TrialPruned` struct.
pub(super) fn is_trial_pruned<E: 'static>(e: &E) -> bool {
    let any: &dyn Any = e;
    if let Some(err) = any.downcast_ref::<crate::Error>() {
        matches!(err, crate::Error::TrialPruned)
    } else {
        any.downcast_ref::<crate::error::TrialPruned>().is_some()
    }
}
