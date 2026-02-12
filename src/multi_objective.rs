//! Multi-objective optimization via a dedicated study type.
//!
//! [`MultiObjectiveStudy`] manages trials that return **multiple** objective
//! values simultaneously. It supports arbitrary numbers of objectives with
//! per-objective directions (minimize or maximize).
//!
//! # Key concepts
//!
//! In multi-objective optimization there is usually no single best solution.
//! Instead, there is a **Pareto front** — the set of solutions where no
//! objective can be improved without worsening another. Use
//! [`pareto_front()`](MultiObjectiveStudy::pareto_front) to retrieve these
//! non-dominated solutions after optimization.
//!
//! A solution **dominates** another if it is at least as good in all
//! objectives and strictly better in at least one. Solutions that are not
//! dominated by any other are called **Pareto-optimal**.
//!
//! # Samplers
//!
//! By default a random sampler is used. For smarter search, pass a
//! [`MultiObjectiveSampler`] such as [`Nsga2Sampler`](crate::sampler::Nsga2Sampler),
//! [`Nsga3Sampler`](crate::sampler::Nsga3Sampler), or
//! [`MoeadSampler`](crate::sampler::MoeadSampler) via
//! [`MultiObjectiveStudy::with_sampler`].
//!
//! # Examples
//!
//! ```
//! use optimizer::Direction;
//! use optimizer::multi_objective::MultiObjectiveStudy;
//! use optimizer::parameter::{FloatParam, Parameter};
//!
//! let study = MultiObjectiveStudy::new(vec![Direction::Minimize, Direction::Minimize]);
//! let x = FloatParam::new(0.0, 1.0);
//!
//! study
//!     .optimize(20, |trial| {
//!         let xv = x.suggest(trial)?;
//!         Ok::<_, optimizer::Error>(vec![xv, 1.0 - xv])
//!     })
//!     .unwrap();
//!
//! let front = study.pareto_front();
//! assert!(!front.is_empty());
//! ```

use core::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::distribution::Distribution;
use crate::param::ParamValue;
use crate::parameter::{ParamId, Parameter};
use crate::pruner::NopPruner;
use crate::sampler::random::RandomSampler;
use crate::sampler::{CompletedTrial, Sampler};
use crate::trial::{AttrValue, Trial};
use crate::types::{Direction, TrialState};

// ---------------------------------------------------------------------------
// MultiObjectiveTrial
// ---------------------------------------------------------------------------

/// A completed trial with multiple objective values.
///
/// Each trial stores its sampled parameter values, the vector of
/// objective values (one per objective), and optional constraint values.
/// Retrieve typed parameter values with [`get()`](Self::get) and check
/// constraint feasibility with [`is_feasible()`](Self::is_feasible).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MultiObjectiveTrial {
    /// The unique identifier for this trial.
    pub id: u64,
    /// The sampled parameter values, keyed by parameter id.
    pub params: HashMap<ParamId, ParamValue>,
    /// The parameter distributions used, keyed by parameter id.
    pub distributions: HashMap<ParamId, Distribution>,
    /// Human-readable labels for parameters, keyed by parameter id.
    pub param_labels: HashMap<ParamId, String>,
    /// The objective values (one per objective).
    pub values: Vec<f64>,
    /// The state of the trial.
    pub state: TrialState,
    /// User-defined attributes stored during the trial.
    pub user_attrs: HashMap<String, AttrValue>,
    /// Constraint values for this trial (<=0.0 means feasible).
    #[cfg_attr(feature = "serde", serde(default))]
    pub constraints: Vec<f64>,
}

impl MultiObjectiveTrial {
    /// Returns the typed value for the given parameter.
    ///
    /// Returns `None` if the parameter was not used in this trial.
    ///
    /// # Panics
    ///
    /// Panics if the stored value is incompatible with the parameter type.
    pub fn get<P: Parameter>(&self, param: &P) -> Option<P::Value> {
        self.params.get(&param.id()).map(|v| {
            param
                .cast_param_value(v)
                .expect("parameter type mismatch: stored value incompatible with parameter")
        })
    }

    /// Returns `true` if all constraints are satisfied (values <= 0.0).
    ///
    /// A trial with no constraints is considered feasible.
    #[must_use]
    pub fn is_feasible(&self) -> bool {
        self.constraints.iter().all(|&c| c <= 0.0)
    }

    /// Gets a user attribute by key.
    #[must_use]
    pub fn user_attr(&self, key: &str) -> Option<&AttrValue> {
        self.user_attrs.get(key)
    }

    /// Returns all user attributes.
    #[must_use]
    pub fn user_attrs(&self) -> &HashMap<String, AttrValue> {
        &self.user_attrs
    }
}

// ---------------------------------------------------------------------------
// MultiObjectiveSampler trait
// ---------------------------------------------------------------------------

/// Trait for samplers aware of multi-objective history.
///
/// Separate from [`Sampler`] because multi-objective algorithms (e.g.,
/// NSGA-II) need access to the full vector of objective values per trial
/// (`&[MultiObjectiveTrial]`) and the per-objective directions
/// (`&[Direction]`).
///
/// Implementations include [`Nsga2Sampler`](crate::sampler::Nsga2Sampler),
/// [`Nsga3Sampler`](crate::sampler::Nsga3Sampler), and
/// [`MoeadSampler`](crate::sampler::MoeadSampler).
pub trait MultiObjectiveSampler: Send + Sync {
    /// Samples a parameter value from the given distribution.
    fn sample(
        &self,
        distribution: &Distribution,
        trial_id: u64,
        history: &[MultiObjectiveTrial],
        directions: &[Direction],
    ) -> ParamValue;
}

// ---------------------------------------------------------------------------
// RandomMultiObjectiveSampler
// ---------------------------------------------------------------------------

/// Default MO sampler that delegates to [`RandomSampler`].
pub(crate) struct RandomMultiObjectiveSampler(RandomSampler);

impl RandomMultiObjectiveSampler {
    pub(crate) fn new() -> Self {
        Self(RandomSampler::new())
    }
}

impl MultiObjectiveSampler for RandomMultiObjectiveSampler {
    fn sample(
        &self,
        distribution: &Distribution,
        trial_id: u64,
        _history: &[MultiObjectiveTrial],
        _directions: &[Direction],
    ) -> ParamValue {
        self.0.sample(distribution, trial_id, &[])
    }
}

// ---------------------------------------------------------------------------
// MoSamplerBridge — bridges MultiObjectiveSampler to Sampler trait
// ---------------------------------------------------------------------------

/// Bridges a [`MultiObjectiveSampler`] to the [`Sampler`] trait so that
/// `Trial::with_sampler()` can use it.
struct MoSamplerBridge {
    inner: Arc<dyn MultiObjectiveSampler>,
    history: Arc<RwLock<Vec<MultiObjectiveTrial>>>,
    directions: Vec<Direction>,
}

impl Sampler for MoSamplerBridge {
    fn sample(
        &self,
        distribution: &Distribution,
        trial_id: u64,
        _history: &[CompletedTrial],
    ) -> ParamValue {
        let mo_history = self.history.read();
        self.inner
            .sample(distribution, trial_id, &mo_history, &self.directions)
    }
}

// ---------------------------------------------------------------------------
// MultiObjectiveStudy
// ---------------------------------------------------------------------------

/// A study for multi-objective optimization.
///
/// Manage trials that return multiple objective values. Supports
/// arbitrary numbers of objectives with independent minimize/maximize
/// directions. After optimization, call [`pareto_front()`](Self::pareto_front)
/// to retrieve the non-dominated solutions.
///
/// For single-objective optimization, use [`Study`](crate::Study) instead.
///
/// # Examples
///
/// ```
/// use optimizer::Direction;
/// use optimizer::multi_objective::MultiObjectiveStudy;
/// use optimizer::parameter::{FloatParam, Parameter};
///
/// // Bi-objective: minimize both
/// let study = MultiObjectiveStudy::new(vec![Direction::Minimize, Direction::Minimize]);
/// let x = FloatParam::new(0.0, 1.0);
///
/// study
///     .optimize(30, |trial| {
///         let xv = x.suggest(trial)?;
///         Ok::<_, optimizer::Error>(vec![xv, 1.0 - xv])
///     })
///     .unwrap();
///
/// let front = study.pareto_front();
/// assert!(!front.is_empty());
/// ```
pub struct MultiObjectiveStudy {
    directions: Vec<Direction>,
    sampler: Arc<dyn MultiObjectiveSampler>,
    completed_trials: Arc<RwLock<Vec<MultiObjectiveTrial>>>,
    next_trial_id: AtomicU64,
}

impl MultiObjectiveStudy {
    /// Creates a new multi-objective study with the given directions.
    ///
    /// Uses a random sampler by default.
    ///
    /// # Arguments
    ///
    /// * `directions` - One direction per objective (minimize or maximize).
    #[must_use]
    pub fn new(directions: Vec<Direction>) -> Self {
        Self {
            directions,
            sampler: Arc::new(RandomMultiObjectiveSampler::new()),
            completed_trials: Arc::new(RwLock::new(Vec::new())),
            next_trial_id: AtomicU64::new(0),
        }
    }

    /// Creates a new study with a custom multi-objective sampler.
    #[must_use]
    pub fn with_sampler(
        directions: Vec<Direction>,
        sampler: impl MultiObjectiveSampler + 'static,
    ) -> Self {
        Self {
            directions,
            sampler: Arc::new(sampler),
            completed_trials: Arc::new(RwLock::new(Vec::new())),
            next_trial_id: AtomicU64::new(0),
        }
    }

    /// Returns the optimization directions.
    #[must_use]
    pub fn directions(&self) -> &[Direction] {
        &self.directions
    }

    /// Returns the number of objectives.
    #[must_use]
    pub fn n_objectives(&self) -> usize {
        self.directions.len()
    }

    /// Returns the number of completed trials.
    #[must_use]
    pub fn n_trials(&self) -> usize {
        self.completed_trials.read().len()
    }

    /// Returns all completed trials.
    #[must_use]
    pub fn trials(&self) -> Vec<MultiObjectiveTrial> {
        self.completed_trials.read().clone()
    }

    /// Return the Pareto-optimal trials (the non-dominated front).
    ///
    /// Uses fast non-dominated sorting (Deb et al., 2002) from the
    /// [`pareto`](crate::pareto) module. Returns an empty vec if no
    /// trials have completed.
    #[must_use]
    pub fn pareto_front(&self) -> Vec<MultiObjectiveTrial> {
        let trials = self.completed_trials.read();
        let complete: Vec<_> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();

        if complete.is_empty() {
            return Vec::new();
        }

        let values: Vec<Vec<f64>> = complete.iter().map(|t| t.values.clone()).collect();
        let fronts = crate::pareto::fast_non_dominated_sort(&values, &self.directions);

        if fronts.is_empty() {
            return Vec::new();
        }

        fronts[0].iter().map(|&i| complete[i].clone()).collect()
    }

    /// Creates a new trial wired to the study's MO sampler.
    fn create_trial(&self) -> Trial {
        let id = self.next_trial_id.fetch_add(1, Ordering::SeqCst);

        let bridge: Arc<dyn Sampler> = Arc::new(MoSamplerBridge {
            inner: Arc::clone(&self.sampler),
            history: Arc::clone(&self.completed_trials),
            directions: self.directions.clone(),
        });

        // Dummy f64 history — the bridge ignores it.
        let dummy_history: Arc<RwLock<Vec<CompletedTrial<f64>>>> =
            Arc::new(RwLock::new(Vec::new()));

        Trial::with_sampler(id, bridge, dummy_history, Arc::new(NopPruner))
    }

    /// Records a completed trial.
    fn complete_trial(&self, mut trial: Trial, values: Vec<f64>) {
        trial.set_complete();
        let mo_trial = MultiObjectiveTrial {
            id: trial.id(),
            params: trial.params().clone(),
            distributions: trial.distributions().clone(),
            param_labels: trial.param_labels().clone(),
            values,
            state: TrialState::Complete,
            user_attrs: trial.user_attrs().clone(),
            constraints: trial.constraint_values().to_vec(),
        };
        self.completed_trials.write().push(mo_trial);
    }

    /// Records a failed trial (not stored in history).
    fn fail_trial(trial: &mut Trial) {
        trial.set_failed();
    }

    /// Request a new trial for the ask/tell interface.
    ///
    /// After creating the trial, suggest parameters on it, evaluate your
    /// objective externally, then pass the trial back to [`tell()`](Self::tell).
    pub fn ask(&self) -> Trial {
        self.create_trial()
    }

    /// Report the result of a trial obtained from [`ask()`](Self::ask).
    ///
    /// Pass `Ok(values)` for a successful evaluation or `Err(reason)` for a failure.
    ///
    /// # Errors
    ///
    /// Returns `ObjectiveDimensionMismatch` if the number of values doesn't
    /// match the number of directions.
    pub fn tell(
        &self,
        mut trial: Trial,
        result: core::result::Result<Vec<f64>, impl ToString>,
    ) -> crate::Result<()> {
        if let Ok(values) = result {
            if values.len() != self.directions.len() {
                return Err(crate::Error::ObjectiveDimensionMismatch {
                    expected: self.directions.len(),
                    got: values.len(),
                });
            }
            self.complete_trial(trial, values);
        } else {
            Self::fail_trial(&mut trial);
        }
        Ok(())
    }

    /// Runs multi-objective optimization for `n_trials` trials.
    ///
    /// The objective function must return a `Vec<f64>` with one value per
    /// objective.
    ///
    /// # Errors
    ///
    /// Returns `ObjectiveDimensionMismatch` if the objective returns the wrong
    /// number of values. Returns `NoCompletedTrials` if all trials fail.
    pub fn optimize<F, E>(&self, n_trials: usize, mut objective: F) -> crate::Result<()>
    where
        F: FnMut(&mut Trial) -> core::result::Result<Vec<f64>, E>,
        E: ToString,
    {
        for _ in 0..n_trials {
            let mut trial = self.create_trial();

            match objective(&mut trial) {
                Ok(values) => {
                    if values.len() != self.directions.len() {
                        return Err(crate::Error::ObjectiveDimensionMismatch {
                            expected: self.directions.len(),
                            got: values.len(),
                        });
                    }
                    self.complete_trial(trial, values);
                }
                Err(_) => {
                    Self::fail_trial(&mut trial);
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
