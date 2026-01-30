//! Study implementation for managing optimization trials.

#[cfg(feature = "async")]
use std::future::Future;
use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::sampler::{CompletedTrial, RandomSampler, Sampler};
use crate::trial::Trial;
use crate::types::Direction;

/// Helper function to create default sampler for serde deserialization.
#[cfg(feature = "serde")]
fn default_sampler() -> Arc<dyn Sampler> {
    Arc::new(RandomSampler::new())
}

/// A study manages the optimization process, tracking trials and their results.
///
/// The study is parameterized by the objective value type `V`, which defaults to `f64`.
/// The only constraint on `V` is `PartialOrd`, allowing comparison of objective values
/// to determine which trial is best.
///
/// When `V = f64`, the study passes trial history to the sampler for informed
/// parameter suggestions (e.g., TPE sampler uses history to guide sampling).
///
/// # Serialization
///
/// When the `serde` feature is enabled, the study can be serialized and deserialized.
/// The completed trials and trial ID counter are preserved, allowing optimization to
/// continue after deserialization. The sampler is not serialized; upon deserialization,
/// a default `RandomSampler` is used. Use `Study::set_sampler()` to restore a custom sampler.
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
    /// Completed trials (wrapped in Arc for sharing with Trial).
    completed_trials: Arc<RwLock<Vec<CompletedTrial<V>>>>,
    /// Counter for generating unique trial IDs.
    next_trial_id: AtomicU64,
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
    pub fn new(direction: Direction) -> Self {
        Self::with_sampler(direction, RandomSampler::new())
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
    /// use optimizer::{Direction, RandomSampler, Study};
    ///
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Maximize, sampler);
    /// assert_eq!(study.direction(), Direction::Maximize);
    /// ```
    pub fn with_sampler(direction: Direction, sampler: impl Sampler + 'static) -> Self {
        Self {
            direction,
            sampler: Arc::new(sampler),
            completed_trials: Arc::new(RwLock::new(Vec::new())),
            next_trial_id: AtomicU64::new(0),
        }
    }

    /// Returns the optimization direction.
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Sets a new sampler for the study.
    ///
    /// This method is useful after deserializing a study when you want to use
    /// a custom sampler (e.g., TPE) instead of the default `RandomSampler`.
    ///
    /// # Arguments
    ///
    /// * `sampler` - The sampler to use for parameter sampling.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::{Direction, Study, TpeSampler};
    ///
    /// // After deserializing a study, restore the TPE sampler
    /// let mut study: Study<f64> = Study::new(Direction::Minimize);
    /// study.set_sampler(TpeSampler::new());
    /// ```
    pub fn set_sampler(&mut self, sampler: impl Sampler + 'static) {
        self.sampler = Arc::new(sampler);
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
    /// Note: For `Study<f64>`, this method creates a trial without sampler
    /// integration. Use `create_trial_with_sampler()` to create trials that
    /// use the study's sampler and have access to trial history.
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
        Trial::new(id)
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
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let mut trial = study.create_trial();
    /// let x = trial.suggest_float("x", 0.0, 1.0).unwrap();
    /// let objective_value = x * x;
    /// study.complete_trial(trial, objective_value);
    ///
    /// assert_eq!(study.n_trials(), 1);
    /// ```
    pub fn complete_trial(&self, mut trial: Trial, value: V) {
        trial.set_complete();
        let completed = CompletedTrial::new(
            trial.id(),
            trial.params().clone(),
            trial.distributions().clone(),
            value,
        );
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
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let mut trial = study.create_trial();
    /// let _ = trial.suggest_float("x", 0.0, 1.0);
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
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// assert_eq!(study.n_trials(), 0);
    ///
    /// let mut trial = study.create_trial();
    /// let _ = trial.suggest_float("x", 0.0, 1.0);
    /// study.complete_trial(trial, 0.5);
    /// assert_eq!(study.n_trials(), 1);
    /// ```
    pub fn n_trials(&self) -> usize {
        self.completed_trials.read().len()
    }

    /// Returns the trial with the best objective value.
    ///
    /// The "best" trial depends on the optimization direction:
    /// - `Direction::Minimize`: Returns the trial with the lowest objective value.
    /// - `Direction::Maximize`: Returns the trial with the highest objective value.
    ///
    /// # Errors
    ///
    /// Returns `TpeError::NoCompletedTrials` if no trials have been completed.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    ///
    /// // Error when no trials completed
    /// assert!(study.best_trial().is_err());
    ///
    /// let mut trial1 = study.create_trial();
    /// let _ = trial1.suggest_float("x", 0.0, 1.0);
    /// study.complete_trial(trial1, 0.8);
    ///
    /// let mut trial2 = study.create_trial();
    /// let _ = trial2.suggest_float("x", 0.0, 1.0);
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

        if trials.is_empty() {
            return Err(crate::TpeError::NoCompletedTrials);
        }

        let best = trials
            .iter()
            .max_by(|a, b| {
                // For Minimize, we want the smallest value to be "max" in ordering
                // For Maximize, we want the largest value to be "max" in ordering
                let ordering = a.value.partial_cmp(&b.value);
                match self.direction {
                    Direction::Minimize => {
                        // Reverse ordering: smaller values are "greater" for max_by
                        ordering
                            .map(|o| o.reverse())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }
                    Direction::Maximize => {
                        // Normal ordering: larger values are "greater" for max_by
                        ordering.unwrap_or(std::cmp::Ordering::Equal)
                    }
                }
            })
            .expect("trials is not empty");

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
    /// Returns `TpeError::NoCompletedTrials` if no trials have been completed.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Maximize);
    ///
    /// // Error when no trials completed
    /// assert!(study.best_value().is_err());
    ///
    /// let mut trial1 = study.create_trial();
    /// let _ = trial1.suggest_float("x", 0.0, 1.0);
    /// study.complete_trial(trial1, 0.3);
    ///
    /// let mut trial2 = study.create_trial();
    /// let _ = trial2.suggest_float("x", 0.0, 1.0);
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
    /// Returns `TpeError::NoCompletedTrials` if all trials failed (no successful trials).
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::{Direction, RandomSampler, Study};
    ///
    /// // Minimize x^2
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    ///
    /// study
    ///     .optimize(10, |trial| {
    ///         let x = trial.suggest_float("x", -10.0, 10.0)?;
    ///         Ok::<_, optimizer::TpeError>(x * x)
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
        F: FnMut(&mut Trial) -> std::result::Result<V, E>,
        E: ToString,
    {
        for _ in 0..n_trials {
            let mut trial = self.create_trial();

            match objective(&mut trial) {
                Ok(value) => {
                    self.complete_trial(trial, value);
                }
                Err(e) => {
                    self.fail_trial(trial, e.to_string());
                }
            }
        }

        // Return error if no trials succeeded
        if self.n_trials() == 0 {
            return Err(crate::TpeError::NoCompletedTrials);
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
    /// Returns `TpeError::NoCompletedTrials` if all trials failed (no successful trials).
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::{Direction, RandomSampler, Study};
    ///
    /// # #[cfg(feature = "async")]
    /// # async fn example() -> optimizer::Result<()> {
    /// // Minimize x^2 with async objective
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    ///
    /// study
    ///     .optimize_async(10, |mut trial| async move {
    ///         let x = trial.suggest_float("x", -10.0, 10.0)?;
    ///         // Simulate async work (e.g., network request)
    ///         let value = x * x;
    ///         Ok::<_, optimizer::TpeError>((trial, value))
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
        Fut: Future<Output = std::result::Result<(Trial, V), E>>,
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

        // Return error if no trials succeeded
        if self.n_trials() == 0 {
            return Err(crate::TpeError::NoCompletedTrials);
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
    /// Returns `TpeError::NoCompletedTrials` if all trials failed (no successful trials).
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::{Direction, RandomSampler, Study};
    ///
    /// # #[cfg(feature = "async")]
    /// # async fn example() -> optimizer::Result<()> {
    /// // Minimize x^2 with parallel async evaluation
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    ///
    /// study
    ///     .optimize_parallel(10, 4, |mut trial| async move {
    ///         let x = trial.suggest_float("x", -10.0, 10.0)?;
    ///         // Async objective function (e.g., network request)
    ///         let value = x * x;
    ///         Ok::<_, optimizer::TpeError>((trial, value))
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
        Fut: Future<Output = std::result::Result<(Trial, V), E>> + Send,
        E: ToString + Send + 'static,
        V: Send + 'static,
    {
        use tokio::sync::Semaphore;

        let semaphore = Arc::new(Semaphore::new(concurrency));
        let objective = Arc::new(objective);

        let mut handles = Vec::with_capacity(n_trials);

        for _ in 0..n_trials {
            let permit = semaphore.clone().acquire_owned().await.unwrap();
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
            match handle.await.unwrap() {
                Ok((trial, value)) => {
                    self.complete_trial(trial, value);
                }
                Err(e) => {
                    let _ = e.to_string();
                }
            }
        }

        // Return error if no trials succeeded
        if self.n_trials() == 0 {
            return Err(crate::TpeError::NoCompletedTrials);
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
    /// Returns `TpeError::NoCompletedTrials` if no trials completed successfully
    /// before optimization stopped (either by completing all trials or early stopping).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::ControlFlow;
    ///
    /// use optimizer::{Direction, RandomSampler, Study};
    ///
    /// // Stop early when we find a good enough value
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    ///
    /// study
    ///     .optimize_with_callback(
    ///         100,
    ///         |trial| {
    ///             let x = trial.suggest_float("x", -10.0, 10.0)?;
    ///             Ok::<_, optimizer::TpeError>(x * x)
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
        V: Clone,
        F: FnMut(&mut Trial) -> std::result::Result<V, E>,
        C: FnMut(&Study<V>, &CompletedTrial<V>) -> ControlFlow<()>,
        E: ToString,
    {
        for _ in 0..n_trials {
            let mut trial = self.create_trial();

            match objective(&mut trial) {
                Ok(value) => {
                    self.complete_trial(trial, value);

                    // Get the just-completed trial for the callback
                    let trials = self.completed_trials.read();
                    let completed = trials.last().expect("just added a trial");

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
                    self.fail_trial(trial, e.to_string());
                }
            }
        }

        // Return error if no trials succeeded
        if self.n_trials() == 0 {
            return Err(crate::TpeError::NoCompletedTrials);
        }

        Ok(())
    }
}

// Specialized implementation for Study<f64> that provides full sampler integration.
impl Study<f64> {
    /// Creates a new trial with sampler integration.
    ///
    /// This method creates a trial that uses the study's sampler and has access
    /// to the history of completed trials for informed parameter suggestions.
    /// This is the recommended way to create trials when using `Study<f64>`.
    ///
    /// The trial's `suggest_*` methods will delegate to the sampler (e.g., TPE)
    /// which can use historical trial data to make informed sampling decisions.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::{Direction, RandomSampler, Study};
    ///
    /// // With a seeded sampler for reproducibility
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    /// let mut trial = study.create_trial_with_sampler();
    ///
    /// // Parameter suggestions now use the study's sampler and history
    /// let x = trial.suggest_float("x", 0.0, 1.0).unwrap();
    /// ```
    pub fn create_trial_with_sampler(&self) -> Trial {
        let id = self.next_trial_id();
        Trial::with_sampler(
            id,
            Arc::clone(&self.sampler),
            Arc::clone(&self.completed_trials),
        )
    }

    /// Runs optimization with full sampler integration.
    ///
    /// This method is similar to the generic `optimizer` method but creates trials
    /// using `create_trial_with_sampler()`, giving the sampler access to the history
    /// of completed trials for informed parameter suggestions.
    ///
    /// This is the recommended way to run optimization when using `Study<f64>`
    /// with advanced samplers like TPE.
    ///
    /// # Arguments
    ///
    /// * `n_trials` - The number of trials to run.
    /// * `objective` - A closure that takes a mutable reference to a `Trial` and
    ///   returns the objective value or an error.
    ///
    /// # Errors
    ///
    /// Returns `TpeError::NoCompletedTrials` if all trials failed (no successful trials).
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::{Direction, RandomSampler, Study};
    ///
    /// // Minimize x^2 with sampler integration
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    ///
    /// study
    ///     .optimize_with_sampler(10, |trial| {
    ///         let x = trial.suggest_float("x", -10.0, 10.0)?;
    ///         Ok::<_, optimizer::TpeError>(x * x)
    ///     })
    ///     .unwrap();
    ///
    /// // At least one trial should have completed
    /// assert!(study.n_trials() > 0);
    /// ```
    pub fn optimize_with_sampler<F, E>(
        &self,
        n_trials: usize,
        mut objective: F,
    ) -> crate::Result<()>
    where
        F: FnMut(&mut Trial) -> std::result::Result<f64, E>,
        E: ToString,
    {
        for _ in 0..n_trials {
            let mut trial = self.create_trial_with_sampler();

            match objective(&mut trial) {
                Ok(value) => {
                    self.complete_trial(trial, value);
                }
                Err(e) => {
                    self.fail_trial(trial, e.to_string());
                }
            }
        }

        // Return error if no trials succeeded
        if self.n_trials() == 0 {
            return Err(crate::TpeError::NoCompletedTrials);
        }

        Ok(())
    }

    /// Runs optimization with a callback and full sampler integration.
    ///
    /// This method combines the benefits of `optimize_with_sampler` (sampler access
    /// to trial history) with `optimize_with_callback` (progress monitoring and
    /// early stopping).
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
    /// Returns `TpeError::NoCompletedTrials` if no trials completed successfully.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::ControlFlow;
    ///
    /// use optimizer::{Direction, RandomSampler, Study};
    ///
    /// // Optimize with sampler integration and early stopping
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    ///
    /// study
    ///     .optimize_with_callback_sampler(
    ///         100,
    ///         |trial| {
    ///             let x = trial.suggest_float("x", -10.0, 10.0)?;
    ///             Ok::<_, optimizer::TpeError>(x * x)
    ///         },
    ///         |study, _completed_trial| {
    ///             // Stop after finding 5 good trials
    ///             if study.n_trials() >= 5 {
    ///                 ControlFlow::Break(())
    ///             } else {
    ///                 ControlFlow::Continue(())
    ///             }
    ///         },
    ///     )
    ///     .unwrap();
    ///
    /// assert!(study.n_trials() >= 5);
    /// ```
    pub fn optimize_with_callback_sampler<F, C, E>(
        &self,
        n_trials: usize,
        mut objective: F,
        mut callback: C,
    ) -> crate::Result<()>
    where
        F: FnMut(&mut Trial) -> std::result::Result<f64, E>,
        C: FnMut(&Study<f64>, &CompletedTrial<f64>) -> ControlFlow<()>,
        E: ToString,
    {
        for _ in 0..n_trials {
            let mut trial = self.create_trial_with_sampler();

            match objective(&mut trial) {
                Ok(value) => {
                    self.complete_trial(trial, value);

                    // Get the just-completed trial for the callback
                    let trials = self.completed_trials.read();
                    let completed = trials.last().expect("just added a trial");

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
                    self.fail_trial(trial, e.to_string());
                }
            }
        }

        // Return error if no trials succeeded
        if self.n_trials() == 0 {
            return Err(crate::TpeError::NoCompletedTrials);
        }

        Ok(())
    }

    /// Runs optimization asynchronously with full sampler integration.
    ///
    /// This method combines async execution with the TPE sampler's ability to use
    /// historical trial data for informed parameter suggestions.
    ///
    /// The objective function takes ownership of the `Trial` and must return it
    /// along with the result. This allows async operations to use the trial
    /// across await points.
    ///
    /// # Arguments
    ///
    /// * `n_trials` - The number of trials to run.
    /// * `objective` - A function that takes a `Trial` and returns a `Future`
    ///   that resolves to a tuple of `(Trial, Result<f64, E>)`.
    ///
    /// # Errors
    ///
    /// Returns `TpeError::NoCompletedTrials` if all trials failed (no successful trials).
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::{Direction, RandomSampler, Study};
    ///
    /// # #[cfg(feature = "async")]
    /// # async fn example() -> optimizer::Result<()> {
    /// // Minimize x^2 with async objective and sampler integration
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    ///
    /// study
    ///     .optimize_async_with_sampler(10, |mut trial| async move {
    ///         let x = trial.suggest_float("x", -10.0, 10.0)?;
    ///         // Simulate async work (e.g., network request)
    ///         let value = x * x;
    ///         Ok::<_, optimizer::TpeError>((trial, value))
    ///     })
    ///     .await?;
    ///
    /// // At least one trial should have completed
    /// assert!(study.n_trials() > 0);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async")]
    pub async fn optimize_async_with_sampler<F, Fut, E>(
        &self,
        n_trials: usize,
        objective: F,
    ) -> crate::Result<()>
    where
        F: Fn(Trial) -> Fut,
        Fut: Future<Output = std::result::Result<(Trial, f64), E>>,
        E: ToString,
    {
        for _ in 0..n_trials {
            let trial = self.create_trial_with_sampler();

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

        // Return error if no trials succeeded
        if self.n_trials() == 0 {
            return Err(crate::TpeError::NoCompletedTrials);
        }

        Ok(())
    }

    /// Runs optimization with bounded parallelism and full sampler integration.
    ///
    /// This method combines parallel async execution with the TPE sampler's ability
    /// to use historical trial data for informed parameter suggestions. Up to
    /// `concurrency` trials run simultaneously.
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
    ///   that resolves to a tuple of `(Trial, f64)` or an error.
    ///
    /// # Errors
    ///
    /// Returns `TpeError::NoCompletedTrials` if all trials failed (no successful trials).
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::{Direction, RandomSampler, Study};
    ///
    /// # #[cfg(feature = "async")]
    /// # async fn example() -> optimizer::Result<()> {
    /// // Minimize x^2 with parallel async evaluation and sampler integration
    /// let sampler = RandomSampler::with_seed(42);
    /// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    ///
    /// study
    ///     .optimize_parallel_with_sampler(10, 4, |mut trial| async move {
    ///         let x = trial.suggest_float("x", -10.0, 10.0)?;
    ///         // Async objective function (e.g., network request)
    ///         let value = x * x;
    ///         Ok::<_, optimize::TpeError>((trial, value))
    ///     })
    ///     .await?;
    ///
    /// // All trials should have completed
    /// assert_eq!(study.n_trials(), 10);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async")]
    pub async fn optimize_parallel_with_sampler<F, Fut, E>(
        &self,
        n_trials: usize,
        concurrency: usize,
        objective: F,
    ) -> crate::Result<()>
    where
        F: Fn(Trial) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = std::result::Result<(Trial, f64), E>> + Send,
        E: ToString + Send + 'static,
    {
        use tokio::sync::Semaphore;

        let semaphore = Arc::new(Semaphore::new(concurrency));
        let objective = Arc::new(objective);

        let mut handles = Vec::with_capacity(n_trials);

        for _ in 0..n_trials {
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let trial = self.create_trial_with_sampler();
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
            match handle.await.unwrap() {
                Ok((trial, value)) => {
                    self.complete_trial(trial, value);
                }
                Err(e) => {
                    let _ = e.to_string();
                }
            }
        }

        // Return error if no trials succeeded
        if self.n_trials() == 0 {
            return Err(crate::TpeError::NoCompletedTrials);
        }

        Ok(())
    }
}

// Manual Serialize implementation for Study<V> when serde feature is enabled.
#[cfg(feature = "serde")]
impl<V> Serialize for Study<V>
where
    V: PartialOrd + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("Study", 3)?;
        state.serialize_field("direction", &self.direction)?;
        // Serialize the Vec inside the Arc<RwLock<>>
        let trials = self.completed_trials.read();
        state.serialize_field("completed_trials", &*trials)?;
        state.serialize_field("next_trial_id", &self.next_trial_id.load(Ordering::SeqCst))?;
        state.end()
    }
}

// Manual Deserialize implementation for Study<V> when serde feature is enabled.
#[cfg(feature = "serde")]
impl<'de, V> Deserialize<'de> for Study<V>
where
    V: PartialOrd + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use std::fmt;
        use std::marker::PhantomData;

        use serde::de::{self, MapAccess, Visitor};

        #[derive(serde::Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Direction,
            CompletedTrials,
            NextTrialId,
        }

        struct StudyVisitor<V>(PhantomData<V>);

        impl<'de, V> Visitor<'de> for StudyVisitor<V>
        where
            V: PartialOrd + Deserialize<'de>,
        {
            type Value = Study<V>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Study")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut direction = None;
                let mut completed_trials: Option<Vec<CompletedTrial<V>>> = None;
                let mut next_trial_id = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Direction => {
                            if direction.is_some() {
                                return Err(de::Error::duplicate_field("direction"));
                            }
                            direction = Some(map.next_value()?);
                        }
                        Field::CompletedTrials => {
                            if completed_trials.is_some() {
                                return Err(de::Error::duplicate_field("completed_trials"));
                            }
                            completed_trials = Some(map.next_value()?);
                        }
                        Field::NextTrialId => {
                            if next_trial_id.is_some() {
                                return Err(de::Error::duplicate_field("next_trial_id"));
                            }
                            next_trial_id = Some(map.next_value()?);
                        }
                    }
                }

                let direction = direction.ok_or_else(|| de::Error::missing_field("direction"))?;
                let completed_trials =
                    completed_trials.ok_or_else(|| de::Error::missing_field("completed_trials"))?;
                let next_trial_id: u64 =
                    next_trial_id.ok_or_else(|| de::Error::missing_field("next_trial_id"))?;

                Ok(Study {
                    direction,
                    sampler: default_sampler(),
                    completed_trials: Arc::new(RwLock::new(completed_trials)),
                    next_trial_id: AtomicU64::new(next_trial_id),
                })
            }
        }

        const FIELDS: &[&str] = &["direction", "completed_trials", "next_trial_id"];
        deserializer.deserialize_struct("Study", FIELDS, StudyVisitor(PhantomData))
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;

    #[test]
    fn test_study_serde_round_trip() {
        // Create a study and add some trials
        let study: Study<f64> = Study::new(Direction::Minimize);

        // Run some optimization
        study
            .optimize(5, |trial| {
                let x = trial.suggest_float("x", 0.0, 10.0)?;
                let y = trial.suggest_int("y", 1, 5)?;
                Ok::<_, crate::TpeError>(x + y as f64)
            })
            .unwrap();

        // Serialize to JSON
        let serialized = serde_json::to_string(&study).unwrap();

        // Deserialize from JSON
        let deserialized: Study<f64> = serde_json::from_str(&serialized).unwrap();

        // Verify the data is preserved
        assert_eq!(deserialized.direction(), study.direction());
        assert_eq!(deserialized.n_trials(), study.n_trials());

        // Verify the best trial is the same
        let original_best = study.best_trial().unwrap();
        let deserialized_best = deserialized.best_trial().unwrap();
        assert_eq!(original_best.id, deserialized_best.id);
        // Use approximate comparison for floats due to JSON serialization precision
        assert!((original_best.value - deserialized_best.value).abs() < 1e-10);
        // Check that all param keys match
        assert_eq!(original_best.params.len(), deserialized_best.params.len());
        for (key, original_val) in &original_best.params {
            let deserialized_val = deserialized_best.params.get(key).unwrap();
            match (original_val, deserialized_val) {
                (crate::param::ParamValue::Float(a), crate::param::ParamValue::Float(b)) => {
                    assert!((a - b).abs() < 1e-10, "Float param {key} differs");
                }
                _ => assert_eq!(original_val, deserialized_val),
            }
        }

        // Verify we can continue optimization on the deserialized study
        let initial_count = deserialized.n_trials();
        deserialized
            .optimize(3, |trial| {
                let x = trial.suggest_float("x", 0.0, 10.0)?;
                let y = trial.suggest_int("y", 1, 5)?;
                Ok::<_, crate::TpeError>(x + y as f64)
            })
            .unwrap();

        // Verify new trials were added
        assert_eq!(deserialized.n_trials(), initial_count + 3);
    }

    #[test]
    fn test_study_serde_preserves_trial_ids() {
        let study: Study<f64> = Study::new(Direction::Maximize);

        // Add 5 trials
        study
            .optimize(5, |trial| {
                let x = trial.suggest_float("x", -1.0, 1.0)?;
                Ok::<_, crate::TpeError>(x * x)
            })
            .unwrap();

        // Serialize and deserialize
        let serialized = serde_json::to_string(&study).unwrap();
        let deserialized: Study<f64> = serde_json::from_str(&serialized).unwrap();

        // Create a new trial - its ID should continue from where we left off
        let new_trial = deserialized.create_trial();
        assert_eq!(new_trial.id(), 5); // Next trial should be ID 5
    }

    #[test]
    fn test_completed_trial_serde() {
        use std::collections::HashMap;

        use crate::distribution::{Distribution, FloatDistribution, IntDistribution};
        use crate::param::ParamValue;

        let mut params = HashMap::new();
        params.insert("x".to_string(), ParamValue::Float(0.5));
        params.insert("n".to_string(), ParamValue::Int(42));

        let mut distributions = HashMap::new();
        distributions.insert(
            "x".to_string(),
            Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log_scale: false,
                step: None,
            }),
        );
        distributions.insert(
            "n".to_string(),
            Distribution::Int(IntDistribution {
                low: 1,
                high: 100,
                log_scale: false,
                step: None,
            }),
        );

        let completed = CompletedTrial::new(42, params.clone(), distributions.clone(), 0.75);

        // Serialize and deserialize
        let serialized = serde_json::to_string(&completed).unwrap();
        let deserialized: CompletedTrial<f64> = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.id, 42);
        assert_eq!(deserialized.value, 0.75);
        assert_eq!(deserialized.params, params);
        assert_eq!(deserialized.distributions, distributions);
    }
}
