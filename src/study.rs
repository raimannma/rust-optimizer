//! Study implementation for managing optimization trials.

use core::any::Any;
use core::fmt;
#[cfg(feature = "async")]
use core::future::Future;
use core::marker::PhantomData;
use core::ops::ControlFlow;
use core::time::Duration;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

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
    /// Creates a new study with the given optimization direction.
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

    /// Returns a [`StudyBuilder`] for constructing a study with a fluent API.
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

    /// Creates a study that minimizes the objective value.
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

    /// Creates a study that maximizes the objective value.
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

    /// Creates a new study with a custom sampler.
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

    /// Builds a trial factory for sampler integration when `V = f64`.
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

    /// Creates a study with a custom sampler and storage backend.
    ///
    /// This is the most general constructor — all other constructors
    /// delegate to this one.
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

    /// Returns the optimization direction.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Sets a new sampler for the study.
    ///
    /// # Arguments
    ///
    /// * `sampler` - The sampler to use for parameter sampling.
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
    /// Creates a new study with a custom sampler and pruner.
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

    pub fn set_sampler(&mut self, sampler: impl Sampler + 'static)
    where
        V: 'static,
    {
        self.sampler = Arc::new(sampler);
        self.trial_factory = Self::make_trial_factory(&self.sampler, &self.storage, &self.pruner);
    }

    /// Sets a new pruner for the study.
    ///
    /// # Arguments
    ///
    /// * `pruner` - The pruner to use for trial pruning.
    pub fn set_pruner(&mut self, pruner: impl Pruner + 'static)
    where
        V: 'static,
    {
        self.pruner = Arc::new(pruner);
        self.trial_factory = Self::make_trial_factory(&self.sampler, &self.storage, &self.pruner);
    }

    /// Returns a reference to the study's pruner.
    #[must_use]
    pub fn pruner(&self) -> &dyn Pruner {
        &*self.pruner
    }

    /// Enqueues a specific parameter configuration to be evaluated next.
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
    /// use optimizer::parameter::{FloatParam, IntParam, Parameter};
    /// use optimizer::{Direction, ParamValue, Study};
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

    /// Returns the trial ID of the current best trial from the given slice.
    #[cfg(feature = "tracing")]
    fn best_id(&self, trials: &[CompletedTrial<V>]) -> Option<u64> {
        let direction = self.direction;
        trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .max_by(|a, b| Self::compare_trials(a, b, direction))
            .map(|t| t.id)
    }

    /// Creates a new trial with pre-set parameter values.
    ///
    /// The trial gets a new unique ID but reuses the given parameters. When
    /// `suggest_param` is called on the resulting trial, fixed values are
    /// returned instead of sampling.
    fn create_trial_with_params(&self, params: HashMap<ParamId, ParamValue>) -> Trial {
        let id = self.next_trial_id();
        let mut trial = if let Some(factory) = &self.trial_factory {
            factory(id)
        } else {
            Trial::new(id)
        };
        trial.set_fixed_params(params);
        trial
    }

    /// Returns the number of enqueued parameter configurations.
    #[must_use]
    pub fn n_enqueued(&self) -> usize {
        self.enqueued_params.lock().len()
    }

    /// Generates the next unique trial ID.
    pub(crate) fn next_trial_id(&self) -> u64 {
        self.storage.next_trial_id()
    }

    /// Creates a new trial with a unique ID.
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

    /// Records a completed trial with its objective value.
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
    pub fn complete_trial(&self, mut trial: Trial, value: V) {
        trial.set_complete();
        let mut completed = CompletedTrial::with_intermediate_values(
            trial.id(),
            trial.params().clone(),
            trial.distributions().clone(),
            trial.param_labels().clone(),
            value,
            trial.intermediate_values().to_vec(),
            trial.user_attrs().clone(),
        );
        completed.state = TrialState::Complete;
        completed.constraints = trial.constraint_values().to_vec();

        self.storage.push(completed);
    }

    /// Records a failed trial with an error message.
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

    /// Records a pruned trial, preserving its intermediate values.
    ///
    /// Pruned trials are stored alongside completed trials so that samplers
    /// can optionally learn from partial evaluations. The trial's state is
    /// set to `Pruned`.
    ///
    /// # Arguments
    ///
    /// * `trial` - The trial that was pruned.
    pub fn prune_trial(&self, mut trial: Trial)
    where
        V: Default,
    {
        trial.set_pruned();
        let mut completed = CompletedTrial::with_intermediate_values(
            trial.id(),
            trial.params().clone(),
            trial.distributions().clone(),
            trial.param_labels().clone(),
            V::default(),
            trial.intermediate_values().to_vec(),
            trial.user_attrs().clone(),
        );
        completed.state = TrialState::Pruned;
        completed.constraints = trial.constraint_values().to_vec();

        self.storage.push(completed);
    }

    /// Returns an iterator over all completed trials.
    ///
    /// The iterator yields references to `CompletedTrial` values, which contain
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

    /// Returns the number of completed trials.
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

    /// Returns the number of pruned trials.
    #[must_use]
    pub fn n_pruned_trials(&self) -> usize {
        self.storage
            .trials_arc()
            .read()
            .iter()
            .filter(|t| t.state == TrialState::Pruned)
            .count()
    }

    /// Compares two completed trials using constraint-aware ranking.
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

    /// Returns the trial with the best objective value.
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

    /// Returns the best objective value found so far.
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

    /// Returns the top `n` trials sorted by objective value.
    ///
    /// For `Direction::Minimize`, returns trials with the lowest values.
    /// For `Direction::Maximize`, returns trials with the highest values.
    /// Only includes completed trials (not failed or pruned).
    ///
    /// If fewer than `n` completed trials exist, returns all of them.
    #[must_use]
    pub fn top_trials(&self, n: usize) -> Vec<CompletedTrial<V>>
    where
        V: Clone,
    {
        let trials = self.storage.trials_arc().read();
        let direction = self.direction;
        let mut completed: Vec<_> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .cloned()
            .collect();
        // Sort best-first: reverse the compare_trials ordering (which is designed for max_by)
        completed.sort_by(|a, b| Self::compare_trials(b, a, direction));
        completed.truncate(n);
        completed
    }

    /// Runs optimization with the given objective function.
    ///
    /// This method runs `n_trials` evaluations sequentially. For each trial:
    /// 1. A new trial is created
    /// 2. The objective function is called with the trial
    /// 3. If successful, the trial is recorded as completed
    /// 4. If the objective returns an error, the trial is recorded as failed
    ///
    /// Failed trials do not stop the optimization; the process continues with
    /// the next trial.
    ///
    /// # Arguments
    ///
    /// * `n_trials` - The number of trials to run.
    /// * `objective` - A closure that takes a mutable reference to a `Trial` and
    ///   returns the objective value or an error.
    ///
    /// # Errors
    ///
    /// Returns `Error::NoCompletedTrials` if all trials failed (no successful trials).
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::sampler::random::RandomSampler;
    /// use optimizer::{Direction, Study};
    ///
    /// // Minimize x^2
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    ///
    /// let x_param = FloatParam::new(-10.0, 10.0);
    ///
    /// study
    ///     .optimize(10, |trial| {
    ///         let x = x_param.suggest(trial)?;
    ///         Ok::<_, optimizer::Error>(x * x)
    ///     })
    ///     .unwrap();
    ///
    /// // At least one trial should have completed
    /// assert!(study.n_trials() > 0);
    /// let best = study.best_value().unwrap();
    /// assert!(best >= 0.0);
    /// ```
    pub fn optimize<F, E>(&self, n_trials: usize, mut objective: F) -> crate::Result<()>
    where
        F: FnMut(&mut Trial) -> core::result::Result<V, E>,
        E: ToString + 'static,
        V: Default,
    {
        #[cfg(feature = "tracing")]
        let _span =
            tracing::info_span!("optimize", n_trials, direction = ?self.direction).entered();

        for _ in 0..n_trials {
            let mut trial = self.create_trial();

            match objective(&mut trial) {
                Ok(value) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = trial.id();
                    self.complete_trial(trial, value);

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
                }
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = trial.id();
                    if is_trial_pruned(&e) {
                        self.prune_trial(trial);
                        trace_info!(trial_id, "trial pruned");
                    } else {
                        self.fail_trial(trial, e.to_string());
                        trace_debug!(trial_id, "trial failed");
                    }
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

    /// Runs optimization asynchronously with the given objective function.
    ///
    /// This method runs `n_trials` evaluations sequentially, but the objective
    /// function can be async (e.g., for I/O-bound operations like network requests
    /// or file operations).
    ///
    /// The objective function takes ownership of the `Trial` and must return it
    /// along with the result. This allows async operations to use the trial
    /// across await points.
    ///
    /// # Arguments
    ///
    /// * `n_trials` - The number of trials to run.
    /// * `objective` - A function that takes a `Trial` and returns a `Future`
    ///   that resolves to a tuple of `(Trial, Result<V, E>)`.
    ///
    /// # Errors
    ///
    /// Returns `Error::NoCompletedTrials` if all trials failed (no successful trials).
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
    /// // Minimize x^2 with async objective
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    ///
    /// let x_param = FloatParam::new(-10.0, 10.0);
    ///
    /// study
    ///     .optimize_async(10, |mut trial| {
    ///         let x_param = x_param.clone();
    ///         async move {
    ///             let x = x_param.suggest(&mut trial)?;
    ///             // Simulate async work (e.g., network request)
    ///             let value = x * x;
    ///             Ok::<_, optimizer::Error>((trial, value))
    ///         }
    ///     })
    ///     .await?;
    ///
    /// // At least one trial should have completed
    /// assert!(study.n_trials() > 0);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async")]
    pub async fn optimize_async<F, Fut, E>(
        &self,
        n_trials: usize,
        objective: F,
    ) -> crate::Result<()>
    where
        F: Fn(Trial) -> Fut,
        Fut: Future<Output = core::result::Result<(Trial, V), E>>,
        E: ToString,
    {
        #[cfg(feature = "tracing")]
        let _span =
            tracing::info_span!("optimize_async", n_trials, direction = ?self.direction).entered();

        for _ in 0..n_trials {
            let trial = self.create_trial();
            #[cfg(feature = "tracing")]
            let trial_id = trial.id();

            match objective(trial).await {
                Ok((trial, value)) => {
                    self.complete_trial(trial, value);
                    trace_info!(trial_id, "trial completed");
                }
                Err(e) => {
                    // For async, we don't have the trial back on error
                    // We'll just count this as a failed trial without recording it
                    let _ = e.to_string();
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

    /// Runs optimization with bounded parallelism for concurrent trial evaluation.
    ///
    /// This method runs up to `concurrency` trials simultaneously, allowing
    /// efficient use of async I/O-bound objective functions. A semaphore limits
    /// the number of concurrent evaluations.
    ///
    /// The objective function takes ownership of the `Trial` and must return it
    /// along with the result. This allows async operations to use the trial
    /// across await points.
    ///
    /// # Arguments
    ///
    /// * `n_trials` - The total number of trials to run.
    /// * `concurrency` - The maximum number of trials to run simultaneously.
    /// * `objective` - A function that takes a `Trial` and returns a `Future`
    ///   that resolves to a tuple of `(Trial, V)` or an error.
    ///
    /// # Errors
    ///
    /// Returns `Error::NoCompletedTrials` if all trials failed (no successful trials).
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
    /// // Minimize x^2 with parallel async evaluation
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    ///
    /// let x_param = FloatParam::new(-10.0, 10.0);
    ///
    /// study
    ///     .optimize_parallel(10, 4, move |mut trial| {
    ///         let x_param = x_param.clone();
    ///         async move {
    ///             let x = x_param.suggest(&mut trial)?;
    ///             // Async objective function (e.g., network request)
    ///             let value = x * x;
    ///             Ok::<_, optimizer::Error>((trial, value))
    ///         }
    ///     })
    ///     .await?;
    ///
    /// // All trials should have completed
    /// assert_eq!(study.n_trials(), 10);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async")]
    pub async fn optimize_parallel<F, Fut, E>(
        &self,
        n_trials: usize,
        concurrency: usize,
        objective: F,
    ) -> crate::Result<()>
    where
        F: Fn(Trial) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = core::result::Result<(Trial, V), E>> + Send,
        E: ToString + Send + 'static,
        V: Send + 'static,
    {
        use tokio::sync::Semaphore;

        #[cfg(feature = "tracing")]
        let _span = tracing::info_span!("optimize_parallel", n_trials, concurrency, direction = ?self.direction).entered();

        let semaphore = Arc::new(Semaphore::new(concurrency));
        let objective = Arc::new(objective);

        let mut handles = Vec::with_capacity(n_trials);

        for _ in 0..n_trials {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|e| crate::Error::TaskError(e.to_string()))?;
            let trial = self.create_trial();
            let objective = Arc::clone(&objective);

            let handle = tokio::spawn(async move {
                let result = objective(trial).await;
                drop(permit); // Release semaphore permit when done
                result
            });

            handles.push(handle);
        }

        // Wait for all tasks and record results
        for handle in handles {
            match handle
                .await
                .map_err(|e| crate::Error::TaskError(e.to_string()))?
            {
                Ok((trial, value)) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = trial.id();
                    self.complete_trial(trial, value);
                    trace_info!(trial_id, "trial completed");
                }
                Err(e) => {
                    let _ = e.to_string();
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

    /// Runs optimization with a callback for monitoring progress.
    ///
    /// This method is similar to `optimize`, but calls a callback function after
    /// each completed trial. The callback can inspect the study state and the
    /// completed trial, and can optionally stop optimization early by returning
    /// `ControlFlow::Break(())`.
    ///
    /// # Arguments
    ///
    /// * `n_trials` - The maximum number of trials to run.
    /// * `objective` - A closure that takes a mutable reference to a `Trial` and
    ///   returns the objective value or an error.
    /// * `callback` - A closure called after each successful trial. Returns
    ///   `ControlFlow::Continue(())` to proceed or `ControlFlow::Break(())` to stop.
    ///
    /// # Errors
    ///
    /// Returns `Error::NoCompletedTrials` if no trials completed successfully
    /// before optimization stopped (either by completing all trials or early stopping).
    /// Returns `Error::Internal` if a completed trial is not found after adding (internal invariant violation).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::ControlFlow;
    ///
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::sampler::random::RandomSampler;
    /// use optimizer::{Direction, Study};
    ///
    /// // Stop early when we find a good enough value
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    ///
    /// let x_param = FloatParam::new(-10.0, 10.0);
    ///
    /// study
    ///     .optimize_with_callback(
    ///         100,
    ///         |trial| {
    ///             let x = x_param.suggest(trial)?;
    ///             Ok::<_, optimizer::Error>(x * x)
    ///         },
    ///         |_study, completed_trial| {
    ///             // Stop early if we find a value less than 1.0
    ///             if completed_trial.value < 1.0 {
    ///                 ControlFlow::Break(())
    ///             } else {
    ///                 ControlFlow::Continue(())
    ///             }
    ///         },
    ///     )
    ///     .unwrap();
    ///
    /// // May have stopped early, but should have at least one trial
    /// assert!(study.n_trials() > 0);
    /// ```
    pub fn optimize_with_callback<F, C, E>(
        &self,
        n_trials: usize,
        mut objective: F,
        mut callback: C,
    ) -> crate::Result<()>
    where
        V: Clone + Default,
        F: FnMut(&mut Trial) -> core::result::Result<V, E>,
        C: FnMut(&Study<V>, &CompletedTrial<V>) -> ControlFlow<()>,
        E: ToString + 'static,
    {
        #[cfg(feature = "tracing")]
        let _span =
            tracing::info_span!("optimize", n_trials, direction = ?self.direction).entered();

        for _ in 0..n_trials {
            let mut trial = self.create_trial();

            match objective(&mut trial) {
                Ok(value) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = trial.id();
                    self.complete_trial(trial, value);

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

                    // Get the just-completed trial for the callback
                    let trials = self.storage.trials_arc().read();
                    let Some(completed) = trials.last() else {
                        return Err(crate::Error::Internal(
                            "completed trial not found after adding",
                        ));
                    };

                    // Call the callback and check if we should stop
                    // Note: We need to drop the read lock before calling callback
                    // to avoid potential deadlock if callback accesses the study
                    let completed_clone = completed.clone();
                    drop(trials);

                    if let ControlFlow::Break(()) = callback(self, &completed_clone) {
                        break;
                    }
                }
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = trial.id();
                    if is_trial_pruned(&e) {
                        self.prune_trial(trial);
                        trace_info!(trial_id, "trial pruned");
                    } else {
                        self.fail_trial(trial, e.to_string());
                        trace_debug!(trial_id, "trial failed");
                    }
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
    /// Runs optimization until the given duration has elapsed.
    ///
    /// Trials that are already running when the timeout is reached will
    /// complete — we never interrupt mid-trial. The actual elapsed time
    /// may therefore slightly exceed the specified duration.
    ///
    /// # Arguments
    ///
    /// * `duration` - The maximum wall-clock time to spend on optimization.
    /// * `objective` - A closure that takes a mutable reference to a `Trial` and
    ///   returns the objective value or an error.
    ///
    /// # Errors
    ///
    /// Returns `Error::NoCompletedTrials` if no trials completed successfully
    /// before the timeout.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::sampler::random::RandomSampler;
    /// use optimizer::{Direction, Study};
    ///
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    ///
    /// let x_param = FloatParam::new(-10.0, 10.0);
    ///
    /// study
    ///     .optimize_until(Duration::from_millis(100), |trial| {
    ///         let x = x_param.suggest(trial)?;
    ///         Ok::<_, optimizer::Error>(x * x)
    ///     })
    ///     .unwrap();
    ///
    /// assert!(study.n_trials() > 0);
    /// ```
    pub fn optimize_until<F, E>(&self, duration: Duration, mut objective: F) -> crate::Result<()>
    where
        F: FnMut(&mut Trial) -> core::result::Result<V, E>,
        E: ToString + 'static,
        V: Default,
    {
        #[cfg(feature = "tracing")]
        let _span = tracing::info_span!("optimize", duration_secs = duration.as_secs(), direction = ?self.direction).entered();

        let deadline = Instant::now() + duration;
        while Instant::now() < deadline {
            let mut trial = self.create_trial();

            match objective(&mut trial) {
                Ok(value) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = trial.id();
                    self.complete_trial(trial, value);
                    trace_info!(trial_id, "trial completed");
                }
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = trial.id();
                    if is_trial_pruned(&e) {
                        self.prune_trial(trial);
                        trace_info!(trial_id, "trial pruned");
                    } else {
                        self.fail_trial(trial, e.to_string());
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

    /// Runs optimization until the given duration has elapsed, with a callback.
    ///
    /// Like [`optimize_until`](Self::optimize_until), but calls a callback after
    /// each completed trial. The callback can stop optimization early by returning
    /// `ControlFlow::Break(())`.
    ///
    /// # Arguments
    ///
    /// * `duration` - The maximum wall-clock time to spend on optimization.
    /// * `objective` - A closure that takes a mutable reference to a `Trial` and
    ///   returns the objective value or an error.
    /// * `callback` - A closure called after each successful trial. Returns
    ///   `ControlFlow::Continue(())` to proceed or `ControlFlow::Break(())` to stop.
    ///
    /// # Errors
    ///
    /// Returns `Error::NoCompletedTrials` if no trials completed successfully.
    /// Returns `Error::Internal` if a completed trial is not found after adding.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::ControlFlow;
    /// use std::time::Duration;
    ///
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::sampler::random::RandomSampler;
    /// use optimizer::{Direction, Study};
    ///
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    ///
    /// let x_param = FloatParam::new(-10.0, 10.0);
    ///
    /// study
    ///     .optimize_until_with_callback(
    ///         Duration::from_secs(1),
    ///         |trial| {
    ///             let x = x_param.suggest(trial)?;
    ///             Ok::<_, optimizer::Error>(x * x)
    ///         },
    ///         |_study, completed_trial| {
    ///             if completed_trial.value < 1.0 {
    ///                 ControlFlow::Break(())
    ///             } else {
    ///                 ControlFlow::Continue(())
    ///             }
    ///         },
    ///     )
    ///     .unwrap();
    ///
    /// assert!(study.n_trials() > 0);
    /// ```
    pub fn optimize_until_with_callback<F, C, E>(
        &self,
        duration: Duration,
        mut objective: F,
        mut callback: C,
    ) -> crate::Result<()>
    where
        V: Clone + Default,
        F: FnMut(&mut Trial) -> core::result::Result<V, E>,
        C: FnMut(&Study<V>, &CompletedTrial<V>) -> ControlFlow<()>,
        E: ToString + 'static,
    {
        #[cfg(feature = "tracing")]
        let _span = tracing::info_span!("optimize", duration_secs = duration.as_secs(), direction = ?self.direction).entered();

        let deadline = Instant::now() + duration;
        while Instant::now() < deadline {
            let mut trial = self.create_trial();

            match objective(&mut trial) {
                Ok(value) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = trial.id();
                    self.complete_trial(trial, value);

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

                    let trials = self.storage.trials_arc().read();
                    let Some(completed) = trials.last() else {
                        return Err(crate::Error::Internal(
                            "completed trial not found after adding",
                        ));
                    };

                    let completed_clone = completed.clone();
                    drop(trials);

                    if let ControlFlow::Break(()) = callback(self, &completed_clone) {
                        break;
                    }
                }
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = trial.id();
                    if is_trial_pruned(&e) {
                        self.prune_trial(trial);
                        trace_info!(trial_id, "trial pruned");
                    } else {
                        self.fail_trial(trial, e.to_string());
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

    /// Runs optimization asynchronously until the given duration has elapsed.
    ///
    /// The async variant of [`optimize_until`](Self::optimize_until). Trials are
    /// run sequentially, but the objective function can be async.
    ///
    /// # Arguments
    ///
    /// * `duration` - The maximum wall-clock time to spend on optimization.
    /// * `objective` - A function that takes a `Trial` and returns a `Future`
    ///   that resolves to a tuple of `(Trial, Result<V, E>)`.
    ///
    /// # Errors
    ///
    /// Returns `Error::NoCompletedTrials` if no trials completed successfully.
    #[cfg(feature = "async")]
    pub async fn optimize_until_async<F, Fut, E>(
        &self,
        duration: Duration,
        objective: F,
    ) -> crate::Result<()>
    where
        F: Fn(Trial) -> Fut,
        Fut: Future<Output = core::result::Result<(Trial, V), E>>,
        E: ToString,
    {
        #[cfg(feature = "tracing")]
        let _span = tracing::info_span!("optimize_until_async", duration_secs = duration.as_secs(), direction = ?self.direction).entered();

        let deadline = Instant::now() + duration;
        while Instant::now() < deadline {
            let trial = self.create_trial();
            #[cfg(feature = "tracing")]
            let trial_id = trial.id();

            match objective(trial).await {
                Ok((trial, value)) => {
                    self.complete_trial(trial, value);
                    trace_info!(trial_id, "trial completed");
                }
                Err(e) => {
                    let _ = e.to_string();
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

    /// Runs optimization with bounded parallelism until the given duration has elapsed.
    ///
    /// The parallel variant of [`optimize_until`](Self::optimize_until). Runs up to
    /// `concurrency` trials simultaneously using async tasks. New trials are spawned
    /// as long as the deadline has not been reached; trials already running when the
    /// deadline passes will complete.
    ///
    /// # Arguments
    ///
    /// * `duration` - The maximum wall-clock time to spend spawning new trials.
    /// * `concurrency` - The maximum number of trials to run simultaneously.
    /// * `objective` - A function that takes a `Trial` and returns a `Future`
    ///   that resolves to a tuple of `(Trial, V)` or an error.
    ///
    /// # Errors
    ///
    /// Returns `Error::NoCompletedTrials` if no trials completed successfully.
    /// Returns `Error::TaskError` if the semaphore is closed or a spawned task panics.
    #[cfg(feature = "async")]
    pub async fn optimize_until_parallel<F, Fut, E>(
        &self,
        duration: Duration,
        concurrency: usize,
        objective: F,
    ) -> crate::Result<()>
    where
        F: Fn(Trial) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = core::result::Result<(Trial, V), E>> + Send,
        E: ToString + Send + 'static,
        V: Send + 'static,
    {
        use tokio::sync::Semaphore;

        #[cfg(feature = "tracing")]
        let _span = tracing::info_span!("optimize_until_parallel", duration_secs = duration.as_secs(), concurrency, direction = ?self.direction).entered();

        let deadline = Instant::now() + duration;
        let semaphore = Arc::new(Semaphore::new(concurrency));
        let objective = Arc::new(objective);

        let mut handles = Vec::new();

        while Instant::now() < deadline {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|e| crate::Error::TaskError(e.to_string()))?;
            let trial = self.create_trial();
            let objective = Arc::clone(&objective);

            let handle = tokio::spawn(async move {
                let result = objective(trial).await;
                drop(permit);
                result
            });

            handles.push(handle);
        }

        for handle in handles {
            match handle
                .await
                .map_err(|e| crate::Error::TaskError(e.to_string()))?
            {
                Ok((trial, value)) => {
                    #[cfg(feature = "tracing")]
                    let trial_id = trial.id();
                    self.complete_trial(trial, value);
                    trace_info!(trial_id, "trial completed");
                }
                Err(e) => {
                    let _ = e.to_string();
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

    /// Runs optimization with automatic retry for failed trials.
    ///
    /// If the objective function returns an error, the same parameter
    /// configuration is retried up to `max_retries` times. Only after all
    /// retries are exhausted is the trial recorded as permanently failed.
    ///
    /// `n_trials` counts unique parameter configurations, not total
    /// evaluations. A trial retried 3 times still counts as 1 toward the
    /// `n_trials` limit.
    ///
    /// # Arguments
    ///
    /// * `n_trials` - The number of unique configurations to evaluate.
    /// * `max_retries` - Maximum retry attempts per failed trial.
    /// * `objective` - A closure that takes a mutable reference to a `Trial`
    ///   and returns the objective value or an error.
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
    /// let call_count = std::cell::Cell::new(0u32);
    /// study
    ///     .optimize_with_retries(5, 2, |trial| {
    ///         let x = x_param.suggest(trial)?;
    ///         call_count.set(call_count.get() + 1);
    ///         // Fail once every other call to exercise retry
    ///         if call_count.get() % 2 == 0 {
    ///             Err::<f64, _>(optimizer::Error::Internal("transient"))
    ///         } else {
    ///             Ok(x * x)
    ///         }
    ///     })
    ///     .unwrap();
    ///
    /// assert_eq!(study.n_trials(), 5);
    /// ```
    pub fn optimize_with_retries<F, E>(
        &self,
        n_trials: usize,
        max_retries: usize,
        mut objective: F,
    ) -> crate::Result<()>
    where
        F: FnMut(&mut Trial) -> core::result::Result<V, E>,
        E: ToString + 'static,
        V: Default,
    {
        #[cfg(feature = "tracing")]
        let _span = tracing::info_span!("optimize_with_retries", n_trials, max_retries, direction = ?self.direction).entered();

        for _ in 0..n_trials {
            let mut trial = self.create_trial();
            let mut retries = 0;
            loop {
                match objective(&mut trial) {
                    Ok(value) => {
                        #[cfg(feature = "tracing")]
                        let trial_id = trial.id();
                        self.complete_trial(trial, value);
                        trace_info!(trial_id, "trial completed");
                        break;
                    }
                    Err(_) if retries < max_retries => {
                        retries += 1;
                        // Create a new trial with the same parameters
                        trial = self.create_trial_with_params(trial.params().clone());
                    }
                    Err(e) => {
                        #[cfg(feature = "tracing")]
                        let trial_id = trial.id();
                        if is_trial_pruned(&e) {
                            self.prune_trial(trial);
                            trace_info!(trial_id, "trial pruned");
                        } else {
                            self.fail_trial(trial, e.to_string());
                            trace_debug!(trial_id, "trial permanently failed");
                        }
                        break;
                    }
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
}

impl<V> Study<V>
where
    V: PartialOrd + Clone + fmt::Display,
{
    /// Export completed trials to CSV format.
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

    /// Export completed trials to a CSV file.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be created or written.
    pub fn export_csv(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        self.to_csv(std::io::BufWriter::new(file))
    }

    /// Returns a human-readable summary of the study.
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
    /// Returns an iterator over all completed trials.
    ///
    /// This clones the internal trial list, so it is suitable for
    /// analysis and iteration but not for hot paths.
    #[must_use]
    pub fn iter(&self) -> std::vec::IntoIter<CompletedTrial<V>> {
        self.trials().into_iter()
    }
}

impl<V> Study<V>
where
    V: PartialOrd + Clone + Into<f64>,
{
    /// Computes parameter importance scores using Spearman rank correlation.
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
    ///     .optimize(20, |trial| {
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

    /// Computes parameter importance using fANOVA (functional ANOVA) with
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
    pub fn fanova(&self) -> crate::Result<crate::fanova::FanovaResult> {
        self.fanova_with_config(&crate::fanova::FanovaConfig::default())
    }

    /// Computes parameter importance using fANOVA with custom configuration.
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

// Specialized implementation for Study<f64> that provides deprecated `_with_sampler` aliases.
//
// For Study<f64>, the generic methods from `impl<V> Study<V>` (like `optimize()`,
// `create_trial()`) now automatically use the sampler via the `trial_factory`.
// The `_with_sampler` method names are deprecated in favor of the generic names.
#[allow(clippy::missing_errors_doc)]
impl Study<f64> {
    /// Deprecated: use `create_trial()` instead.
    ///
    /// The generic `create_trial()` now automatically integrates with the sampler
    /// for `Study<f64>`.
    #[deprecated(
        since = "0.2.0",
        note = "use `create_trial()` instead — it now uses the sampler automatically for Study<f64>"
    )]
    #[must_use]
    pub fn create_trial_with_sampler(&self) -> Trial {
        self.create_trial()
    }

    /// Deprecated: use `optimize()` instead.
    ///
    /// The generic `optimize()` now automatically integrates with the sampler
    /// for `Study<f64>`.
    #[deprecated(
        since = "0.2.0",
        note = "use `optimize()` instead — it now uses the sampler automatically for Study<f64>"
    )]
    pub fn optimize_with_sampler<F, E>(&self, n_trials: usize, objective: F) -> crate::Result<()>
    where
        F: FnMut(&mut Trial) -> core::result::Result<f64, E>,
        E: ToString + 'static,
    {
        self.optimize(n_trials, objective)
    }

    /// Deprecated: use `optimize_with_callback()` instead.
    ///
    /// The generic `optimize_with_callback()` now automatically integrates with the
    /// sampler for `Study<f64>`.
    #[deprecated(
        since = "0.2.0",
        note = "use `optimize_with_callback()` instead — it now uses the sampler automatically for Study<f64>"
    )]
    pub fn optimize_with_callback_sampler<F, C, E>(
        &self,
        n_trials: usize,
        objective: F,
        callback: C,
    ) -> crate::Result<()>
    where
        F: FnMut(&mut Trial) -> core::result::Result<f64, E>,
        C: FnMut(&Study<f64>, &CompletedTrial<f64>) -> ControlFlow<()>,
        E: ToString + 'static,
    {
        self.optimize_with_callback(n_trials, objective, callback)
    }

    /// Deprecated: use `optimize_async()` instead.
    ///
    /// The generic `optimize_async()` now automatically integrates with the sampler
    /// for `Study<f64>`.
    #[cfg(feature = "async")]
    #[deprecated(
        since = "0.2.0",
        note = "use `optimize_async()` instead — it now uses the sampler automatically for Study<f64>"
    )]
    pub async fn optimize_async_with_sampler<F, Fut, E>(
        &self,
        n_trials: usize,
        objective: F,
    ) -> crate::Result<()>
    where
        F: Fn(Trial) -> Fut,
        Fut: Future<Output = core::result::Result<(Trial, f64), E>>,
        E: ToString,
    {
        self.optimize_async(n_trials, objective).await
    }

    /// Deprecated: use `optimize_parallel()` instead.
    ///
    /// The generic `optimize_parallel()` now automatically integrates with the
    /// sampler for `Study<f64>`.
    #[cfg(feature = "async")]
    #[deprecated(
        since = "0.2.0",
        note = "use `optimize_parallel()` instead — it now uses the sampler automatically for Study<f64>"
    )]
    pub async fn optimize_parallel_with_sampler<F, Fut, E>(
        &self,
        n_trials: usize,
        concurrency: usize,
        objective: F,
    ) -> crate::Result<()>
    where
        F: Fn(Trial) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = core::result::Result<(Trial, f64), E>> + Send,
        E: ToString + Send + 'static,
    {
        self.optimize_parallel(n_trials, concurrency, objective)
            .await
    }
}

impl<V: PartialOrd + Send + Sync + 'static> Study<V> {
    /// Creates a study with a custom sampler, pruner, and storage backend.
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
    /// Sets the optimization direction to minimize.
    #[must_use]
    pub fn minimize(mut self) -> Self {
        self.direction = Direction::Minimize;
        self
    }

    /// Sets the optimization direction to maximize.
    #[must_use]
    pub fn maximize(mut self) -> Self {
        self.direction = Direction::Maximize;
        self
    }

    /// Sets the optimization direction.
    #[must_use]
    pub fn direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    /// Sets the sampler used for parameter suggestions.
    #[must_use]
    pub fn sampler(mut self, sampler: impl Sampler + 'static) -> Self {
        self.sampler = Some(Box::new(sampler));
        self
    }

    /// Sets the pruner used for early stopping of trials.
    #[must_use]
    pub fn pruner(mut self, pruner: impl Pruner + 'static) -> Self {
        self.pruner = Some(Box::new(pruner));
        self
    }

    /// Sets a custom storage backend.
    #[must_use]
    pub fn storage(mut self, storage: impl crate::storage::Storage<V> + 'static) -> Self {
        self.storage = Some(Box::new(storage));
        self
    }

    /// Builds the [`Study`] with the configured options.
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
    /// Creates a study backed by a JSONL journal file.
    ///
    /// Any existing trials in the file are loaded into memory and the
    /// trial ID counter is set to one past the highest stored ID. New
    /// trials are written through to the file on completion.
    ///
    /// # Errors
    ///
    /// Returns a [`Storage`](crate::Error::Storage) error if loading fails.
    pub fn with_journal(
        direction: Direction,
        sampler: impl Sampler + 'static,
        path: impl AsRef<std::path::Path>,
    ) -> crate::Result<Self> {
        let storage = crate::storage::JournalStorage::<V>::open(path)?;
        Ok(Self::with_sampler_and_storage(direction, sampler, storage))
    }
}

#[cfg(feature = "sqlite")]
impl<V> Study<V>
where
    V: PartialOrd + Send + Sync + serde::Serialize + serde::de::DeserializeOwned + 'static,
{
    /// Creates a study backed by a `SQLite` database.
    ///
    /// Any existing trials in the database are loaded into memory and
    /// the trial ID counter is set to one past the highest stored ID.
    /// New trials are written through to the database on completion.
    ///
    /// Uses WAL mode for concurrent readers, making it suitable for
    /// single-machine multi-process optimization.
    ///
    /// # Errors
    ///
    /// Returns a [`Storage`](crate::Error::Storage) error if the
    /// database cannot be opened.
    pub fn with_sqlite(
        direction: Direction,
        sampler: impl Sampler + 'static,
        path: impl AsRef<std::path::Path>,
    ) -> crate::Result<Self> {
        let storage = crate::storage::SqliteStorage::<V>::new(path)?;
        Ok(Self::with_sampler_and_storage(direction, sampler, storage))
    }
}

impl Study<f64> {
    /// Generates an HTML report with interactive Plotly.js charts.
    ///
    /// Creates a self-contained HTML file that can be opened in any browser.
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
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be created or written.
    pub fn export_json(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let trials = self.trials();
        serde_json::to_writer_pretty(file, &trials).map_err(std::io::Error::other)
    }

    /// Saves the study state to a JSON file.
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
impl<V: PartialOrd + Clone + Default + serde::Serialize> Study<V> {
    /// Runs optimization with automatic checkpointing every `interval` trials.
    ///
    /// This is convenience sugar over [`optimize_with_callback`](Self::optimize_with_callback)
    /// combined with [`save`](Self::save). The checkpoint is written atomically so
    /// a crash mid-write will never leave a corrupt file.
    ///
    /// # Errors
    ///
    /// Returns an error if the optimization itself fails (see
    /// [`optimize`](Self::optimize) for details). Checkpoint I/O errors are
    /// silently ignored (best-effort).
    pub fn optimize_with_checkpoint<F, E>(
        &self,
        n_trials: usize,
        checkpoint_interval: usize,
        checkpoint_path: impl AsRef<std::path::Path>,
        objective: F,
    ) -> crate::Result<()>
    where
        F: FnMut(&mut Trial) -> core::result::Result<V, E>,
        E: ToString + 'static,
    {
        let path = checkpoint_path.as_ref().to_owned();
        self.optimize_with_callback(n_trials, objective, |study, _trial| {
            if study.n_trials().is_multiple_of(checkpoint_interval) {
                let _ = study.save(&path);
            }
            ControlFlow::Continue(())
        })
    }
}

#[cfg(feature = "serde")]
impl<V: PartialOrd + Send + Sync + Clone + serde::de::DeserializeOwned + 'static> Study<V> {
    /// Loads a study from a JSON file.
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
