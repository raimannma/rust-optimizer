//! Study implementation for managing optimization trials.

use core::any::Any;
use core::fmt;
use core::marker::PhantomData;
use core::ops::ControlFlow;
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
    direction: Direction,
    /// The sampler used to generate parameter values.
    sampler: Arc<dyn Sampler>,
    /// The pruner used to decide whether to stop trials early.
    pruner: Arc<dyn Pruner>,
    /// Trial storage backend (default: [`MemoryStorage`](crate::storage::MemoryStorage)).
    storage: Arc<dyn crate::storage::Storage<V>>,
    /// Optional factory for creating sampler-aware trials.
    /// Set automatically for `Study<f64>` so that `create_trial()` and all
    /// optimization methods use the sampler without requiring `_with_sampler` suffixes.
    trial_factory: Option<Arc<dyn Fn(u64) -> Trial + Send + Sync>>,
    /// Queue of parameter configurations to evaluate next.
    enqueued_params: Arc<Mutex<VecDeque<HashMap<ParamId, ParamValue>>>>,
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
        StudyBuilder {
            direction: Direction::Minimize,
            sampler: None,
            pruner: None,
            storage: None,
            _marker: PhantomData,
        }
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
    fn make_trial_factory(
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
    /// backend (e.g., [`JournalStorage`](crate::storage::JournalStorage)).
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
    fn best_id(&self, trials: &[CompletedTrial<V>]) -> Option<u64> {
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
    /// Failed trials are not counted.
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
        self.storage.trials_arc().read().len()
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
    fn compare_trials(
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

    /// Return the trial with the best objective value.
    ///
    /// The "best" trial depends on the optimization direction:
    /// - `Direction::Minimize`: Returns the trial with the lowest objective value.
    /// - `Direction::Maximize`: Returns the trial with the highest objective value.
    ///
    /// When constraints are present, feasible trials always rank above infeasible
    /// trials. Among infeasible trials, those with lower total constraint violation
    /// are preferred.
    ///
    /// # Errors
    ///
    /// Returns `Error::NoCompletedTrials` if no trials have been completed.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    ///
    /// // Error when no trials completed
    /// assert!(study.best_trial().is_err());
    ///
    /// let x_param = FloatParam::new(0.0, 1.0);
    ///
    /// let mut trial1 = study.create_trial();
    /// let _ = x_param.suggest(&mut trial1);
    /// study.complete_trial(trial1, 0.8);
    ///
    /// let mut trial2 = study.create_trial();
    /// let _ = x_param.suggest(&mut trial2);
    /// study.complete_trial(trial2, 0.3);
    ///
    /// let best = study.best_trial().unwrap();
    /// assert_eq!(best.value, 0.3); // Minimize: lower is better
    /// ```
    pub fn best_trial(&self) -> crate::Result<CompletedTrial<V>>
    where
        V: Clone,
    {
        let trials = self.storage.trials_arc().read();
        let direction = self.direction;

        let best = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .max_by(|a, b| Self::compare_trials(a, b, direction))
            .ok_or(crate::Error::NoCompletedTrials)?;

        Ok(best.clone())
    }

    /// Return the best objective value found so far.
    ///
    /// The "best" value depends on the optimization direction:
    /// - `Direction::Minimize`: Returns the lowest objective value.
    /// - `Direction::Maximize`: Returns the highest objective value.
    ///
    /// # Errors
    ///
    /// Returns `Error::NoCompletedTrials` if no trials have been completed.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Maximize);
    ///
    /// // Error when no trials completed
    /// assert!(study.best_value().is_err());
    ///
    /// let x_param = FloatParam::new(0.0, 1.0);
    ///
    /// let mut trial1 = study.create_trial();
    /// let _ = x_param.suggest(&mut trial1);
    /// study.complete_trial(trial1, 0.3);
    ///
    /// let mut trial2 = study.create_trial();
    /// let _ = x_param.suggest(&mut trial2);
    /// study.complete_trial(trial2, 0.8);
    ///
    /// let best = study.best_value().unwrap();
    /// assert_eq!(best, 0.8); // Maximize: higher is better
    /// ```
    pub fn best_value(&self) -> crate::Result<V>
    where
        V: Clone,
    {
        self.best_trial().map(|trial| trial.value)
    }

    /// Return the top `n` trials sorted by objective value.
    ///
    /// For `Direction::Minimize`, returns trials with the lowest values.
    /// For `Direction::Maximize`, returns trials with the highest values.
    /// Only includes completed trials (not failed or pruned).
    ///
    /// If fewer than `n` completed trials exist, returns all of them.
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
    /// for val in [5.0, 1.0, 3.0] {
    ///     let mut t = study.create_trial();
    ///     let _ = x.suggest(&mut t);
    ///     study.complete_trial(t, val);
    /// }
    ///
    /// let top2 = study.top_trials(2);
    /// assert_eq!(top2.len(), 2);
    /// assert!(top2[0].value <= top2[1].value);
    /// ```
    #[must_use]
    pub fn top_trials(&self, n: usize) -> Vec<CompletedTrial<V>>
    where
        V: Clone,
    {
        let trials = self.storage.trials_arc().read();
        let direction = self.direction;
        // Sort indices instead of cloning all trials, then clone only the top N.
        let mut indices: Vec<usize> = trials
            .iter()
            .enumerate()
            .filter(|(_, t)| t.state == TrialState::Complete)
            .map(|(i, _)| i)
            .collect();
        // Sort best-first: reverse the compare_trials ordering (which is designed for max_by)
        indices.sort_by(|&a, &b| Self::compare_trials(&trials[b], &trials[a], direction));
        indices.truncate(n);
        indices.iter().map(|&i| trials[i].clone()).collect()
    }

    /// Run optimization with an objective.
    ///
    /// Accepts any [`Objective`](crate::Objective) implementation, including
    /// plain closures (`Fn(&mut Trial) -> Result<V, E>`) thanks to the
    /// blanket impl. Struct-based objectives can override
    /// [`before_trial`](crate::Objective::before_trial) and
    /// [`after_trial`](crate::Objective::after_trial) for early stopping.
    ///
    /// Runs up to `n_trials` evaluations sequentially.
    ///
    /// # Errors
    ///
    /// Returns `Error::NoCompletedTrials` if no trials completed successfully.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::sampler::random::RandomSampler;
    /// use optimizer::{Direction, Study};
    ///
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    /// let x_param = FloatParam::new(-10.0, 10.0);
    ///
    /// study
    ///     .optimize(10, |trial: &mut optimizer::Trial| {
    ///         let x = x_param.suggest(trial)?;
    ///         Ok::<_, optimizer::Error>(x * x)
    ///     })
    ///     .unwrap();
    ///
    /// assert!(study.n_trials() > 0);
    /// assert!(study.best_value().unwrap() >= 0.0);
    /// ```
    #[allow(clippy::needless_pass_by_value)]
    pub fn optimize(
        &self,
        n_trials: usize,
        objective: impl crate::objective::Objective<V>,
    ) -> crate::Result<()>
    where
        V: Clone + Default,
    {
        #[cfg(feature = "tracing")]
        let _span =
            tracing::info_span!("optimize", n_trials, direction = ?self.direction).entered();

        for _ in 0..n_trials {
            if let ControlFlow::Break(()) = objective.before_trial(self) {
                break;
            }

            let mut trial = self.create_trial();
            match objective.evaluate(&mut trial) {
                Ok(value) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = trial.id();

                    let completed = trial.into_completed(value, TrialState::Complete);

                    // Fire after_trial hook before pushing to storage
                    let flow = objective.after_trial(self, &completed);
                    self.storage.push(completed);

                    #[cfg(feature = "tracing")]
                    {
                        tracing::info!(trial_id, "trial completed");
                        let trials = self.storage.trials_arc().read();
                        if trials
                            .iter()
                            .filter(|t| t.state == TrialState::Complete)
                            .count()
                            == 1
                            || trials.last().map(|t| t.id) == self.best_id(&trials)
                        {
                            tracing::info!(trial_id, "new best value found");
                        }
                    }

                    if let ControlFlow::Break(()) = flow {
                        return Ok(());
                    }
                }
                Err(e) if is_trial_pruned(&e) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = trial.id();
                    self.prune_trial(trial);
                    trace_info!(trial_id, "trial pruned");
                }
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = trial.id();
                    self.fail_trial(trial, e.to_string());
                    trace_debug!(trial_id, "trial failed");
                }
            }
        }

        // Return error if no trials completed successfully
        let has_complete = self
            .storage
            .trials_arc()
            .read()
            .iter()
            .any(|t| t.state == TrialState::Complete);
        if !has_complete {
            return Err(crate::Error::NoCompletedTrials);
        }

        Ok(())
    }

    /// Run async optimization with an objective.
    ///
    /// Like [`optimize`](Self::optimize), but each evaluation is wrapped in
    /// [`spawn_blocking`](tokio::task::spawn_blocking), keeping the async
    /// runtime responsive for CPU-bound objectives. Trials run sequentially.
    ///
    /// Accepts any [`Objective`](crate::Objective) implementation, including
    /// plain closures. Struct-based objectives can override lifecycle hooks.
    ///
    /// # Errors
    ///
    /// Returns `Error::NoCompletedTrials` if no trials completed successfully.
    /// Returns `Error::TaskError` if a spawned blocking task panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::sampler::random::RandomSampler;
    /// use optimizer::{Direction, Study};
    ///
    /// # #[cfg(feature = "async")]
    /// # async fn example() -> optimizer::Result<()> {
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    /// let x_param = FloatParam::new(-10.0, 10.0);
    ///
    /// study
    ///     .optimize_async(10, move |trial: &mut optimizer::Trial| {
    ///         let x = x_param.suggest(trial)?;
    ///         Ok::<_, optimizer::Error>(x * x)
    ///     })
    ///     .await?;
    ///
    /// assert!(study.n_trials() > 0);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async")]
    pub async fn optimize_async<O>(&self, n_trials: usize, objective: O) -> crate::Result<()>
    where
        O: crate::objective::Objective<V> + Send + Sync + 'static,
        O::Error: Send,
        V: Clone + Default + Send + 'static,
    {
        #[cfg(feature = "tracing")]
        let _span =
            tracing::info_span!("optimize_async", n_trials, direction = ?self.direction).entered();

        let objective = Arc::new(objective);

        for _ in 0..n_trials {
            if let ControlFlow::Break(()) = objective.before_trial(self) {
                break;
            }

            let obj = Arc::clone(&objective);
            let mut trial = self.create_trial();
            let result = tokio::task::spawn_blocking(move || {
                let res = obj.evaluate(&mut trial);
                (trial, res)
            })
            .await
            .map_err(|e| crate::Error::TaskError(e.to_string()))?;

            match result {
                (t, Ok(value)) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = t.id();

                    let completed = t.into_completed(value, TrialState::Complete);
                    let flow = objective.after_trial(self, &completed);
                    self.storage.push(completed);
                    trace_info!(trial_id, "trial completed");

                    if let ControlFlow::Break(()) = flow {
                        return Ok(());
                    }
                }
                (t, Err(e)) if is_trial_pruned(&e) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = t.id();
                    self.prune_trial(t);
                    trace_info!(trial_id, "trial pruned");
                }
                (t, Err(e)) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = t.id();
                    self.fail_trial(t, e.to_string());
                    trace_debug!(trial_id, "trial failed");
                }
            }
        }

        let has_complete = self
            .storage
            .trials_arc()
            .read()
            .iter()
            .any(|t| t.state == TrialState::Complete);
        if !has_complete {
            return Err(crate::Error::NoCompletedTrials);
        }

        Ok(())
    }

    /// Run parallel optimization with an objective.
    ///
    /// Spawns up to `concurrency` evaluations concurrently using
    /// [`spawn_blocking`](tokio::task::spawn_blocking). Results are
    /// collected via a [`JoinSet`](tokio::task::JoinSet).
    ///
    /// Accepts any [`Objective`](crate::Objective) implementation, including
    /// plain closures. The [`after_trial`](crate::Objective::after_trial)
    /// hook fires as each result arrives — returning `Break` stops spawning
    /// new trials while in-flight tasks drain.
    ///
    /// # Errors
    ///
    /// Returns `Error::NoCompletedTrials` if no trials completed successfully.
    /// Returns `Error::TaskError` if the semaphore is closed or a spawned task panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::sampler::random::RandomSampler;
    /// use optimizer::{Direction, Study};
    ///
    /// # #[cfg(feature = "async")]
    /// # async fn example() -> optimizer::Result<()> {
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    /// let x_param = FloatParam::new(-10.0, 10.0);
    ///
    /// study
    ///     .optimize_parallel(10, 4, move |trial: &mut optimizer::Trial| {
    ///         let x = x_param.suggest(trial)?;
    ///         Ok::<_, optimizer::Error>(x * x)
    ///     })
    ///     .await?;
    ///
    /// assert_eq!(study.n_trials(), 10);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async")]
    #[allow(clippy::missing_panics_doc, clippy::too_many_lines)]
    pub async fn optimize_parallel<O>(
        &self,
        n_trials: usize,
        concurrency: usize,
        objective: O,
    ) -> crate::Result<()>
    where
        O: crate::objective::Objective<V> + Send + Sync + 'static,
        O::Error: Send,
        V: Clone + Default + Send + 'static,
    {
        use tokio::sync::Semaphore;
        use tokio::task::JoinSet;

        #[cfg(feature = "tracing")]
        let _span = tracing::info_span!("optimize_parallel", n_trials, concurrency, direction = ?self.direction).entered();

        let objective = Arc::new(objective);
        let semaphore = Arc::new(Semaphore::new(concurrency));
        let mut join_set: JoinSet<(Trial, Result<V, O::Error>)> = JoinSet::new();
        let mut spawned = 0;

        'spawn: while spawned < n_trials {
            if let ControlFlow::Break(()) = objective.before_trial(self) {
                break;
            }

            // If the join set is full, drain one result to free a slot.
            while join_set.len() >= concurrency {
                let result = join_set
                    .join_next()
                    .await
                    .expect("join_set should not be empty")
                    .map_err(|e| crate::Error::TaskError(e.to_string()))?;
                match result {
                    (t, Ok(value)) => {
                        #[cfg(feature = "tracing")]
                        let trial_id = t.id();

                        let completed = t.into_completed(value, TrialState::Complete);
                        let flow = objective.after_trial(self, &completed);
                        self.storage.push(completed);
                        trace_info!(trial_id, "trial completed");

                        if let ControlFlow::Break(()) = flow {
                            break 'spawn;
                        }
                    }
                    (t, Err(e)) => {
                        #[cfg(feature = "tracing")]
                        let trial_id = t.id();
                        if is_trial_pruned(&e) {
                            self.prune_trial(t);
                            trace_info!(trial_id, "trial pruned");
                        } else {
                            self.fail_trial(t, e.to_string());
                            trace_debug!(trial_id, "trial failed");
                        }
                    }
                }
            }

            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|e| crate::Error::TaskError(e.to_string()))?;

            let mut trial = self.create_trial();
            let obj = Arc::clone(&objective);
            join_set.spawn(async move {
                let result = tokio::task::spawn_blocking(move || {
                    let res = obj.evaluate(&mut trial);
                    (trial, res)
                })
                .await
                .expect("spawn_blocking should not panic");
                drop(permit);
                result
            });
            spawned += 1;
        }

        // Drain remaining in-flight tasks.
        while let Some(result) = join_set.join_next().await {
            let result = result.map_err(|e| crate::Error::TaskError(e.to_string()))?;
            match result {
                (t, Ok(value)) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = t.id();

                    let completed = t.into_completed(value, TrialState::Complete);
                    // Still fire after_trial for bookkeeping, but don't break — we're draining.
                    let _ = objective.after_trial(self, &completed);
                    self.storage.push(completed);
                    trace_info!(trial_id, "trial completed");
                }
                (t, Err(e)) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = t.id();
                    if is_trial_pruned(&e) {
                        self.prune_trial(t);
                        trace_info!(trial_id, "trial pruned");
                    } else {
                        self.fail_trial(t, e.to_string());
                        trace_debug!(trial_id, "trial failed");
                    }
                }
            }
        }

        let has_complete = self
            .storage
            .trials_arc()
            .read()
            .iter()
            .any(|t| t.state == TrialState::Complete);
        if !has_complete {
            return Err(crate::Error::NoCompletedTrials);
        }

        Ok(())
    }
}

impl<V> Study<V>
where
    V: PartialOrd + Clone + fmt::Display,
{
    /// Write completed trials to a writer in CSV format.
    ///
    /// Columns: `trial_id`, `value`, `state`, then one column per unique
    /// parameter label, then one column per unique user-attribute key.
    ///
    /// Parameters without labels use a generated name (`param_<id>`).
    /// Pruned trials have an empty `value` cell.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let x = FloatParam::new(0.0, 10.0).name("x");
    ///
    /// let mut trial = study.create_trial();
    /// let _ = x.suggest(&mut trial);
    /// study.complete_trial(trial, 0.42);
    ///
    /// let mut buf = Vec::new();
    /// study.to_csv(&mut buf).unwrap();
    /// let csv = String::from_utf8(buf).unwrap();
    /// assert!(csv.contains("trial_id"));
    /// ```
    pub fn to_csv(&self, mut writer: impl std::io::Write) -> std::io::Result<()> {
        use std::collections::BTreeMap;

        let trials = self.storage.trials_arc().read();

        // Collect all unique parameter labels (sorted for deterministic column order).
        let mut param_columns: BTreeMap<ParamId, String> = BTreeMap::new();
        for trial in trials.iter() {
            for &id in trial.params.keys() {
                param_columns.entry(id).or_insert_with(|| {
                    trial
                        .param_labels
                        .get(&id)
                        .cloned()
                        .unwrap_or_else(|| id.to_string())
                });
            }
        }
        // Fill in labels from other trials that might have better labels.
        for trial in trials.iter() {
            for (&id, label) in &trial.param_labels {
                param_columns.entry(id).or_insert_with(|| label.clone());
            }
        }

        // Collect all unique attribute keys (sorted).
        let mut attr_keys: Vec<String> = Vec::new();
        for trial in trials.iter() {
            for key in trial.user_attrs.keys() {
                if !attr_keys.contains(key) {
                    attr_keys.push(key.clone());
                }
            }
        }
        attr_keys.sort();

        let param_ids: Vec<ParamId> = param_columns.keys().copied().collect();

        // Write header.
        write!(writer, "trial_id,value,state")?;
        for id in &param_ids {
            write!(writer, ",{}", csv_escape(&param_columns[id]))?;
        }
        for key in &attr_keys {
            write!(writer, ",{}", csv_escape(key))?;
        }
        writeln!(writer)?;

        // Write one row per trial.
        for trial in trials.iter() {
            write!(writer, "{}", trial.id)?;

            // Value: empty for pruned trials.
            if trial.state == TrialState::Complete {
                write!(writer, ",{}", trial.value)?;
            } else {
                write!(writer, ",")?;
            }

            write!(
                writer,
                ",{}",
                match trial.state {
                    TrialState::Complete => "Complete",
                    TrialState::Pruned => "Pruned",
                    TrialState::Failed => "Failed",
                    TrialState::Running => "Running",
                }
            )?;

            for id in &param_ids {
                if let Some(pv) = trial.params.get(id) {
                    write!(writer, ",{pv}")?;
                } else {
                    write!(writer, ",")?;
                }
            }

            for key in &attr_keys {
                if let Some(attr) = trial.user_attrs.get(key) {
                    write!(writer, ",{}", csv_escape(&format_attr(attr)))?;
                } else {
                    write!(writer, ",")?;
                }
            }

            writeln!(writer)?;
        }

        Ok(())
    }

    /// Export completed trials to a CSV file at the given path.
    ///
    /// Convenience wrapper around [`to_csv`](Self::to_csv) that creates a
    /// buffered file writer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be created or written.
    pub fn export_csv(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        self.to_csv(std::io::BufWriter::new(file))
    }

    /// Return a human-readable summary of the study.
    ///
    /// The summary includes:
    /// - Optimization direction and total trial count
    /// - Breakdown by state (complete, pruned) when applicable
    /// - Best trial value and parameters (if any completed trials exist)
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let x = FloatParam::new(0.0, 10.0).name("x");
    ///
    /// let mut trial = study.create_trial();
    /// let _ = x.suggest(&mut trial).unwrap();
    /// study.complete_trial(trial, 0.42);
    ///
    /// let summary = study.summary();
    /// assert!(summary.contains("Minimize"));
    /// assert!(summary.contains("0.42"));
    /// ```
    #[must_use]
    pub fn summary(&self) -> String {
        use fmt::Write;

        let trials = self.storage.trials_arc().read();
        let n_complete = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .count();
        let n_pruned = trials
            .iter()
            .filter(|t| t.state == TrialState::Pruned)
            .count();

        let direction_str = match self.direction {
            Direction::Minimize => "Minimize",
            Direction::Maximize => "Maximize",
        };

        let mut s = format!("Study: {direction_str} | {n} trials", n = trials.len());
        if n_pruned > 0 {
            let _ = write!(s, " ({n_complete} complete, {n_pruned} pruned)");
        }

        drop(trials);

        if let Ok(best) = self.best_trial() {
            let _ = write!(s, "\nBest value: {} (trial #{})", best.value, best.id);
            if !best.params.is_empty() {
                s.push_str("\nBest parameters:");
                let mut params: Vec<_> = best.params.iter().collect();
                params.sort_by_key(|(id, _)| *id);
                for (id, value) in params {
                    let label = best.param_labels.get(id).map_or("?", String::as_str);
                    let _ = write!(s, "\n  {label} = {value}");
                }
            }
        }

        s
    }
}

impl<V> Study<V>
where
    V: PartialOrd + Clone,
{
    /// Return an iterator over all completed trials.
    ///
    /// This clones the internal trial list, so it is suitable for
    /// analysis and iteration but not for hot paths.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let trial = study.create_trial();
    /// study.complete_trial(trial, 1.0);
    ///
    /// for t in study.iter() {
    ///     println!("Trial {} → {}", t.id, t.value);
    /// }
    /// ```
    #[must_use]
    pub fn iter(&self) -> std::vec::IntoIter<CompletedTrial<V>> {
        self.trials().into_iter()
    }
}

impl<V> Study<V>
where
    V: PartialOrd + Clone + Into<f64>,
{
    /// Compute parameter importance scores using Spearman rank correlation.
    ///
    /// For each parameter, the absolute Spearman correlation between its values
    /// and the objective values is computed across all completed trials. Scores
    /// are normalized so they sum to 1.0 and sorted in descending order.
    ///
    /// Parameters that appear in fewer than 2 trials are omitted.
    /// Returns an empty `Vec` if the study has fewer than 2 completed trials.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let x = FloatParam::new(0.0, 10.0).name("x");
    ///
    /// study
    ///     .optimize(20, |trial: &mut optimizer::Trial| {
    ///         let xv = x.suggest(trial)?;
    ///         Ok::<_, optimizer::Error>(xv * xv)
    ///     })
    ///     .unwrap();
    ///
    /// let importance = study.param_importance();
    /// assert_eq!(importance.len(), 1);
    /// assert_eq!(importance[0].0, "x");
    /// ```
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn param_importance(&self) -> Vec<(String, f64)> {
        use std::collections::BTreeSet;

        use crate::importance::spearman;
        use crate::param::ParamValue;
        use crate::types::TrialState;

        let trials = self.storage.trials_arc().read();
        let complete: Vec<_> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();

        if complete.len() < 2 {
            return Vec::new();
        }

        // Collect all parameter IDs across trials.
        let all_param_ids: BTreeSet<_> = complete.iter().flat_map(|t| t.params.keys()).collect();

        let mut scores: Vec<(String, f64)> = Vec::new();

        for &param_id in &all_param_ids {
            // Collect (param_value_f64, objective_f64) for trials that have this param.
            let mut param_vals = Vec::new();
            let mut obj_vals = Vec::new();

            for trial in &complete {
                if let Some(pv) = trial.params.get(param_id) {
                    let f = match *pv {
                        ParamValue::Float(v) => v,
                        ParamValue::Int(v) => v as f64,
                        ParamValue::Categorical(v) => v as f64,
                    };
                    param_vals.push(f);
                    obj_vals.push(trial.value.clone().into());
                }
            }

            if param_vals.len() < 2 {
                continue;
            }

            let corr = spearman(&param_vals, &obj_vals).abs();

            // Determine label: use param_labels if available, else "param_{id}".
            let label = complete
                .iter()
                .find_map(|t| t.param_labels.get(param_id))
                .map_or_else(|| param_id.to_string(), Clone::clone);

            scores.push((label, corr));
        }

        // Normalize so scores sum to 1.0.
        let sum: f64 = scores.iter().map(|(_, s)| *s).sum();
        if sum > 0.0 {
            for entry in &mut scores {
                entry.1 /= sum;
            }
        }

        // Sort descending by score.
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        scores
    }

    /// Compute parameter importance using fANOVA (functional ANOVA) with
    /// default configuration.
    ///
    /// Fits a random forest to the trial data and decomposes variance into
    /// per-parameter main effects and pairwise interaction effects. This is
    /// more accurate than correlation-based importance ([`Self::param_importance`])
    /// and can detect non-linear relationships and parameter interactions.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::NoCompletedTrials`] if fewer than 2 trials have completed.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let x = FloatParam::new(0.0, 10.0).name("x");
    /// let y = FloatParam::new(0.0, 10.0).name("y");
    ///
    /// study
    ///     .optimize(30, |trial: &mut optimizer::Trial| {
    ///         let xv = x.suggest(trial)?;
    ///         let yv = y.suggest(trial)?;
    ///         Ok::<_, optimizer::Error>(xv * xv + 0.1 * yv)
    ///     })
    ///     .unwrap();
    ///
    /// let result = study.fanova().unwrap();
    /// assert!(!result.main_effects.is_empty());
    /// ```
    pub fn fanova(&self) -> crate::Result<crate::fanova::FanovaResult> {
        self.fanova_with_config(&crate::fanova::FanovaConfig::default())
    }

    /// Compute parameter importance using fANOVA with custom configuration.
    ///
    /// See [`Self::fanova`] for details. The [`FanovaConfig`](crate::fanova::FanovaConfig)
    /// allows tuning the number of trees, tree depth, and random seed.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::NoCompletedTrials`] if fewer than 2 trials have completed.
    #[allow(clippy::cast_precision_loss)]
    pub fn fanova_with_config(
        &self,
        config: &crate::fanova::FanovaConfig,
    ) -> crate::Result<crate::fanova::FanovaResult> {
        use std::collections::BTreeSet;

        use crate::fanova::compute_fanova;
        use crate::param::ParamValue;
        use crate::types::TrialState;

        let trials = self.storage.trials_arc().read();
        let complete: Vec<_> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();

        if complete.len() < 2 {
            return Err(crate::Error::NoCompletedTrials);
        }

        // Collect all parameter IDs in a stable order.
        let all_param_ids: Vec<_> = {
            let set: BTreeSet<_> = complete.iter().flat_map(|t| t.params.keys()).collect();
            set.into_iter().collect()
        };

        if all_param_ids.is_empty() {
            return Ok(crate::fanova::FanovaResult {
                main_effects: Vec::new(),
                interactions: Vec::new(),
            });
        }

        // Build feature matrix (only trials that have all parameters).
        let mut data = Vec::new();
        let mut targets = Vec::new();

        for trial in &complete {
            let mut row = Vec::with_capacity(all_param_ids.len());
            let mut has_all = true;

            for &pid in &all_param_ids {
                if let Some(pv) = trial.params.get(pid) {
                    row.push(match *pv {
                        ParamValue::Float(v) => v,
                        ParamValue::Int(v) => v as f64,
                        ParamValue::Categorical(v) => v as f64,
                    });
                } else {
                    has_all = false;
                    break;
                }
            }

            if has_all {
                data.push(row);
                targets.push(trial.value.clone().into());
            }
        }

        if data.len() < 2 {
            return Err(crate::Error::NoCompletedTrials);
        }

        // Build feature names from parameter labels.
        let feature_names: Vec<String> = all_param_ids
            .iter()
            .map(|&pid| {
                complete
                    .iter()
                    .find_map(|t| t.param_labels.get(pid))
                    .map_or_else(|| pid.to_string(), Clone::clone)
            })
            .collect();

        Ok(compute_fanova(&data, &targets, &feature_names, config))
    }
}

impl<V> IntoIterator for &Study<V>
where
    V: PartialOrd + Clone,
{
    type Item = CompletedTrial<V>;
    type IntoIter = std::vec::IntoIter<CompletedTrial<V>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<V> fmt::Display for Study<V>
where
    V: PartialOrd + Clone + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary())
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

#[cfg(feature = "journal")]
impl<V> Study<V>
where
    V: PartialOrd + Send + Sync + serde::Serialize + serde::de::DeserializeOwned + 'static,
{
    /// Create a study backed by a JSONL journal file.
    ///
    /// Any existing trials in the file are loaded into memory and the
    /// trial ID counter is set to one past the highest stored ID. New
    /// trials are written through to the file on completion.
    ///
    /// # Arguments
    ///
    /// * `direction` - Whether to minimize or maximize the objective function.
    /// * `sampler` - The sampler to use for parameter sampling.
    /// * `path` - Path to the JSONL journal file (created if absent).
    ///
    /// # Errors
    ///
    /// Returns a [`Storage`](crate::Error::Storage) error if loading fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use optimizer::sampler::tpe::TpeSampler;
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> =
    ///     Study::with_journal(Direction::Minimize, TpeSampler::new(), "trials.jsonl").unwrap();
    /// ```
    pub fn with_journal(
        direction: Direction,
        sampler: impl Sampler + 'static,
        path: impl AsRef<std::path::Path>,
    ) -> crate::Result<Self> {
        let storage = crate::storage::JournalStorage::<V>::open(path)?;
        Ok(Self::with_sampler_and_storage(direction, sampler, storage))
    }
}

impl Study<f64> {
    /// Generate an HTML report with interactive Plotly.js charts.
    ///
    /// Create a self-contained HTML file that can be opened in any browser.
    /// See [`generate_html_report`](crate::visualization::generate_html_report)
    /// for details on the included charts.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be created or written.
    pub fn export_html(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        crate::visualization::generate_html_report(self, path)
    }
}

/// A serializable snapshot of a study's state.
///
/// Since [`Study`] contains non-serializable fields (samplers, atomics, etc.),
/// this struct captures the essential state needed to save and restore a study.
///
/// # Schema versioning
///
/// The `version` field enables future schema evolution without breaking existing files.
/// The current version is `1`.
///
/// # Sampler state
///
/// Sampler state is **not** included in the snapshot. After loading, the study
/// uses a default `RandomSampler`. Call [`Study::set_sampler`] to restore
/// the desired sampler configuration.
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct StudySnapshot<V> {
    /// Schema version for forward compatibility.
    pub version: u32,
    /// The optimization direction.
    pub direction: Direction,
    /// All completed (and pruned) trials.
    pub trials: Vec<CompletedTrial<V>>,
    /// The next trial ID to assign.
    pub next_trial_id: u64,
    /// Optional metadata (creation timestamp, sampler description, etc.).
    pub metadata: HashMap<String, String>,
}

#[cfg(feature = "serde")]
impl<V: PartialOrd + Clone + serde::Serialize> Study<V> {
    /// Export trials as a pretty-printed JSON array to a file.
    ///
    /// Each element in the array is a serialized [`CompletedTrial`].
    /// Requires the `serde` feature.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be created or written.
    pub fn export_json(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let trials = self.trials();
        serde_json::to_writer_pretty(file, &trials).map_err(std::io::Error::other)
    }

    /// Save the study state to a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be created or written.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        let trials = self.trials();
        let next_trial_id = trials.iter().map(|t| t.id).max().map_or(0, |id| id + 1);
        let snapshot = StudySnapshot {
            version: 1,
            direction: self.direction,
            trials,
            next_trial_id,
            metadata: HashMap::new(),
        };

        // Atomic write: write to a temp file in the same directory, then rename.
        // This prevents corrupt files if the process crashes mid-write.
        let parent = path.parent().unwrap_or(std::path::Path::new("."));
        let tmp_path = parent.join(format!(
            ".{}.tmp",
            path.file_name().unwrap_or_default().to_string_lossy()
        ));
        let file = std::fs::File::create(&tmp_path)?;
        serde_json::to_writer_pretty(file, &snapshot).map_err(std::io::Error::other)?;
        std::fs::rename(&tmp_path, path)
    }
}

#[cfg(feature = "serde")]
impl<V: PartialOrd + Send + Sync + Clone + serde::de::DeserializeOwned + 'static> Study<V> {
    /// Load a study from a JSON file.
    ///
    /// The loaded study uses a `RandomSampler` by default. Call
    /// [`set_sampler()`](Self::set_sampler) to restore the original sampler
    /// configuration.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be read or parsed.
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let snapshot: StudySnapshot<V> = serde_json::from_reader(file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let storage = crate::storage::MemoryStorage::with_trials(snapshot.trials);
        Ok(Self::with_sampler_and_storage(
            snapshot.direction,
            RandomSampler::new(),
            storage,
        ))
    }
}

/// Returns `true` if the error represents a pruned trial.
///
/// Checks via `Any` downcasting whether `e` is `Error::TrialPruned` or
/// the standalone `TrialPruned` struct.
fn is_trial_pruned<E: 'static>(e: &E) -> bool {
    let any: &dyn Any = e;
    if let Some(err) = any.downcast_ref::<crate::Error>() {
        matches!(err, crate::Error::TrialPruned)
    } else {
        any.downcast_ref::<crate::error::TrialPruned>().is_some()
    }
}

/// Escape a string for CSV output. If the value contains a comma, quote, or
/// newline, wrap it in double-quotes and double any embedded quotes.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Format an `AttrValue` as a string for CSV cells.
fn format_attr(attr: &crate::trial::AttrValue) -> String {
    use crate::trial::AttrValue;
    match attr {
        AttrValue::Float(v) => v.to_string(),
        AttrValue::Int(v) => v.to_string(),
        AttrValue::String(v) => v.clone(),
        AttrValue::Bool(v) => v.to_string(),
    }
}
