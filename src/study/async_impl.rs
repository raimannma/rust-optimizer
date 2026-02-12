use core::ops::ControlFlow;
use std::sync::Arc;

use crate::trial::Trial;
use crate::types::TrialState;

use super::{Study, is_trial_pruned};

impl<V> Study<V>
where
    V: PartialOrd,
{
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

        assert!(concurrency > 0, "concurrency must be at least 1");

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
