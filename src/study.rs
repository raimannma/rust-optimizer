//! Study implementation for managing optimization trials.

use core::any::Any;
#[cfg(feature = "async")]
use core::future::Future;
use core::ops::ControlFlow;
use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;

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
    /// Completed trials (wrapped in Arc for sharing with Trial).
    completed_trials: Arc<RwLock<Vec<CompletedTrial<V>>>>,
    /// Counter for generating unique trial IDs.
    next_trial_id: AtomicU64,
    /// Optional factory for creating sampler-aware trials.
    /// Set automatically for `Study<f64>` so that `create_trial()` and all
    /// optimization methods use the sampler without requiring `_with_sampler` suffixes.
    trial_factory: Option<Arc<dyn Fn(u64) -> Trial + Send + Sync>>,
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
        V: 'static,
    {
        Self::with_sampler(direction, RandomSampler::new())
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
        V: 'static,
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
        V: 'static,
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
        V: 'static,
    {
        let sampler: Arc<dyn Sampler> = Arc::new(sampler);
        let completed_trials = Arc::new(RwLock::new(Vec::new()));

        let pruner: Arc<dyn Pruner> = Arc::new(NopPruner);

        // For Study<f64>, set up a trial factory that provides sampler integration.
        // This uses Any downcasting to check at runtime whether V = f64.
        let trial_factory = Self::make_trial_factory(&sampler, &completed_trials, &pruner);

        Self {
            direction,
            sampler,
            pruner,
            completed_trials,
            next_trial_id: AtomicU64::new(0),
            trial_factory,
        }
    }

    /// Builds a trial factory for sampler integration when `V = f64`.
    fn make_trial_factory(
        sampler: &Arc<dyn Sampler>,
        completed_trials: &Arc<RwLock<Vec<CompletedTrial<V>>>>,
        pruner: &Arc<dyn Pruner>,
    ) -> Option<Arc<dyn Fn(u64) -> Trial + Send + Sync>>
    where
        V: 'static,
    {
        // Try to downcast the completed_trials Arc to the f64 specialization.
        // This succeeds only when V = f64, enabling automatic sampler integration.
        let any_ref: &dyn Any = completed_trials;
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

    /// Returns the optimization direction.
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
        V: 'static,
    {
        let sampler: Arc<dyn Sampler> = Arc::new(sampler);
        let pruner: Arc<dyn Pruner> = Arc::new(pruner);
        let completed_trials = Arc::new(RwLock::new(Vec::new()));
        let trial_factory = Self::make_trial_factory(&sampler, &completed_trials, &pruner);

        Self {
            direction,
            sampler,
            pruner,
            completed_trials,
            next_trial_id: AtomicU64::new(0),
            trial_factory,
        }
    }

    pub fn set_sampler(&mut self, sampler: impl Sampler + 'static)
    where
        V: 'static,
    {
        self.sampler = Arc::new(sampler);
        self.trial_factory =
            Self::make_trial_factory(&self.sampler, &self.completed_trials, &self.pruner);
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
        self.trial_factory =
            Self::make_trial_factory(&self.sampler, &self.completed_trials, &self.pruner);
    }

    /// Returns a reference to the study's pruner.
    pub fn pruner(&self) -> &dyn Pruner {
        &*self.pruner
    }

    /// Generates the next unique trial ID.
    pub(crate) fn next_trial_id(&self) -> u64 {
        self.next_trial_id.fetch_add(1, Ordering::SeqCst)
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
    pub fn create_trial(&self) -> Trial {
        let id = self.next_trial_id();
        if let Some(factory) = &self.trial_factory {
            factory(id)
        } else {
            Trial::new(id)
        }
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
        );
        completed.state = TrialState::Complete;
        self.completed_trials.write().push(completed);
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
        );
        completed.state = TrialState::Pruned;
        self.completed_trials.write().push(completed);
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
    pub fn trials(&self) -> Vec<CompletedTrial<V>>
    where
        V: Clone,
    {
        self.completed_trials.read().clone()
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
    pub fn n_trials(&self) -> usize {
        self.completed_trials.read().len()
    }

    /// Returns the number of pruned trials.
    pub fn n_pruned_trials(&self) -> usize {
        self.completed_trials
            .read()
            .iter()
            .filter(|t| t.state == TrialState::Pruned)
            .count()
    }

    /// Returns the trial with the best objective value.
    ///
    /// The "best" trial depends on the optimization direction:
    /// - `Direction::Minimize`: Returns the trial with the lowest objective value.
    /// - `Direction::Maximize`: Returns the trial with the highest objective value.
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
        let trials = self.completed_trials.read();

        let best = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .max_by(|a, b| {
                // For Minimize, we want the smallest value to be "max" in ordering
                // For Maximize, we want the largest value to be "max" in ordering
                let ordering = a.value.partial_cmp(&b.value);
                match self.direction {
                    Direction::Minimize => {
                        // Reverse ordering: smaller values are "greater" for max_by
                        ordering.map_or(core::cmp::Ordering::Equal, core::cmp::Ordering::reverse)
                    }
                    Direction::Maximize => {
                        // Normal ordering: larger values are "greater" for max_by
                        ordering.unwrap_or(core::cmp::Ordering::Equal)
                    }
                }
            })
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
    pub fn top_trials(&self, n: usize) -> Vec<CompletedTrial<V>>
    where
        V: Clone,
    {
        let trials = self.completed_trials.read();
        let mut completed: Vec<_> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .cloned()
            .collect();
        completed.sort_by(|a, b| match self.direction {
            Direction::Minimize => a
                .value
                .partial_cmp(&b.value)
                .unwrap_or(core::cmp::Ordering::Equal),
            Direction::Maximize => b
                .value
                .partial_cmp(&a.value)
                .unwrap_or(core::cmp::Ordering::Equal),
        });
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
        for _ in 0..n_trials {
            let mut trial = self.create_trial();

            match objective(&mut trial) {
                Ok(value) => {
                    self.complete_trial(trial, value);
                }
                Err(e) => {
                    if is_trial_pruned(&e) {
                        self.prune_trial(trial);
                    } else {
                        self.fail_trial(trial, e.to_string());
                    }
                }
            }
        }

        // Return error if no trials completed successfully
        let has_complete = self
            .completed_trials
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
        for _ in 0..n_trials {
            let trial = self.create_trial();

            match objective(trial).await {
                Ok((trial, value)) => {
                    self.complete_trial(trial, value);
                }
                Err(e) => {
                    // For async, we don't have the trial back on error
                    // We'll just count this as a failed trial without recording it
                    let _ = e.to_string();
                }
            }
        }

        // Return error if no trials completed successfully
        let has_complete = self
            .completed_trials
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
                    self.complete_trial(trial, value);
                }
                Err(e) => {
                    let _ = e.to_string();
                }
            }
        }

        // Return error if no trials completed successfully
        let has_complete = self
            .completed_trials
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
        for _ in 0..n_trials {
            let mut trial = self.create_trial();

            match objective(&mut trial) {
                Ok(value) => {
                    self.complete_trial(trial, value);

                    // Get the just-completed trial for the callback
                    let trials = self.completed_trials.read();
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
                    if is_trial_pruned(&e) {
                        self.prune_trial(trial);
                    } else {
                        self.fail_trial(trial, e.to_string());
                    }
                }
            }
        }

        // Return error if no trials completed successfully
        let has_complete = self
            .completed_trials
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
        let deadline = Instant::now() + duration;
        while Instant::now() < deadline {
            let mut trial = self.create_trial();

            match objective(&mut trial) {
                Ok(value) => {
                    self.complete_trial(trial, value);
                }
                Err(e) => {
                    if is_trial_pruned(&e) {
                        self.prune_trial(trial);
                    } else {
                        self.fail_trial(trial, e.to_string());
                    }
                }
            }
        }

        let has_complete = self
            .completed_trials
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
        let deadline = Instant::now() + duration;
        while Instant::now() < deadline {
            let mut trial = self.create_trial();

            match objective(&mut trial) {
                Ok(value) => {
                    self.complete_trial(trial, value);

                    let trials = self.completed_trials.read();
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
                    if is_trial_pruned(&e) {
                        self.prune_trial(trial);
                    } else {
                        self.fail_trial(trial, e.to_string());
                    }
                }
            }
        }

        let has_complete = self
            .completed_trials
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
        let deadline = Instant::now() + duration;
        while Instant::now() < deadline {
            let trial = self.create_trial();

            match objective(trial).await {
                Ok((trial, value)) => {
                    self.complete_trial(trial, value);
                }
                Err(e) => {
                    let _ = e.to_string();
                }
            }
        }

        let has_complete = self
            .completed_trials
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
                    self.complete_trial(trial, value);
                }
                Err(e) => {
                    let _ = e.to_string();
                }
            }
        }

        let has_complete = self
            .completed_trials
            .read()
            .iter()
            .any(|t| t.state == TrialState::Complete);
        if !has_complete {
            return Err(crate::Error::NoCompletedTrials);
        }

        Ok(())
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
