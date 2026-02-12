use core::ops::ControlFlow;

use crate::types::TrialState;

use super::{Study, is_trial_pruned};

impl<V> Study<V>
where
    V: PartialOrd,
{
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
}
