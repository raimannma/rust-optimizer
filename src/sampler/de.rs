//! Differential Evolution (DE) sampler.
//!
//! DE is a population-based metaheuristic that maintains a pool of candidate
//! solutions and creates new candidates through **mutation** (combining
//! difference vectors of existing members) and **binomial crossover**. A
//! trial vector replaces its parent only if it achieves a better objective
//! value, guaranteeing monotonic improvement of the population.
//!
//! # Algorithm overview
//!
//! Each generation, for every population member *xᵢ*:
//! 1. **Mutation** — create a mutant vector *v* from other population
//!    members using the selected [`DEStrategy`]:
//!    - `Rand1`:  `v = x_r1 + F * (x_r2 - x_r3)`
//!    - `Best1`:  `v = x_best + F * (x_r1 - x_r2)`
//!    - `CurrentToBest1`:  `v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)`
//! 2. **Crossover** — create a trial vector *u* by mixing *v* and *xᵢ*
//!    dimension-by-dimension with probability CR.
//! 3. **Selection** — replace *xᵢ* with *u* if `f(u) ≤ f(xᵢ)`.
//!
//! # When to use
//!
//! - **Continuous parameters** (float/int). Categorical parameters are
//!   sampled uniformly at random and do not participate in DE.
//! - **Moderate to large search spaces** — DE scales better than GP-based
//!   methods to higher dimensions, though it may need more evaluations.
//! - **Multi-modal landscapes** — the `Rand1` strategy maintains diversity
//!   and avoids premature convergence.
//! - **No feature flags required** — DE is available with default features.
//!
//! For non-separable problems in moderate dimensions, consider
//! [`CmaEsSampler`](super::cma_es::CmaEsSampler) which learns parameter
//! correlations. For expensive functions with few dimensions, consider
//! [`GpSampler`](super::gp::GpSampler).
//!
//! # Configuration
//!
//! | Option | Default | Description |
//! |--------|---------|-------------|
//! | `population_size` | `max(10n, 15)` | Candidates per generation |
//! | `mutation_factor` (F) | 0.8 | Differential amplification — higher = more exploration |
//! | `crossover_rate` (CR) | 0.9 | Probability of taking a dimension from the mutant |
//! | `strategy` | `Rand1` | Mutation strategy (see [`DEStrategy`]) |
//! | `seed` | random | RNG seed for reproducibility |
//!
//! # Examples
//!
//! ```
//! use optimizer::sampler::de::{DESampler, DEStrategy};
//! use optimizer::{Direction, Study};
//!
//! // Minimize with DE using the Best1 strategy for faster convergence
//! let sampler = DESampler::builder()
//!     .mutation_factor(0.7)
//!     .crossover_rate(0.9)
//!     .strategy(DEStrategy::Best1)
//!     .population_size(20)
//!     .seed(42)
//!     .build();
//!
//! let mut study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
//! ```

use std::collections::HashMap;

use parking_lot::Mutex;

use crate::distribution::Distribution;
use crate::param::ParamValue;
use crate::rng_util;
use crate::sampler::{CompletedTrial, Sampler};

/// Differential Evolution mutation strategy.
///
/// Controls how mutant vectors are created from the current population.
#[derive(Clone, Copy, Debug, Default)]
pub enum DEStrategy {
    /// DE/rand/1: `v = x_r1 + F * (x_r2 - x_r3)`
    ///
    /// The most robust strategy. Uses three random population members.
    #[default]
    Rand1,
    /// DE/best/1: `v = x_best + F * (x_r1 - x_r2)`
    ///
    /// Greedier strategy that biases toward the current best solution.
    Best1,
    /// DE/current-to-best/1: `v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)`
    ///
    /// Balances exploration and exploitation by blending the current
    /// individual with the best.
    CurrentToBest1,
}

/// Differential Evolution sampler for continuous global optimization.
///
/// Maintains a population of candidate solutions. New candidates are
/// created by combining (mutating + crossing over) existing members.
///
/// # Examples
///
/// ```
/// use optimizer::sampler::de::DESampler;
/// use optimizer::{Direction, Study};
///
/// // Default configuration
/// let study: Study<f64> = Study::with_sampler(Direction::Minimize, DESampler::new());
///
/// // With seed for reproducibility
/// let study: Study<f64> = Study::with_sampler(Direction::Minimize, DESampler::with_seed(42));
///
/// // Custom configuration via builder
/// use optimizer::sampler::de::DEStrategy;
/// let sampler = DESampler::builder()
///     .mutation_factor(0.8)
///     .crossover_rate(0.9)
///     .strategy(DEStrategy::Best1)
///     .population_size(30)
///     .seed(42)
///     .build();
/// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
/// ```
pub struct DESampler {
    state: Mutex<State>,
}

impl DESampler {
    /// Creates a new DE sampler with default settings and a random seed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State::new(None, 0.8, 0.9, DEStrategy::Rand1, None)),
        }
    }

    /// Creates a new DE sampler with a fixed seed for reproducibility.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            state: Mutex::new(State::new(None, 0.8, 0.9, DEStrategy::Rand1, Some(seed))),
        }
    }

    /// Creates a builder for configuring a `DESampler`.
    #[must_use]
    pub fn builder() -> DESamplerBuilder {
        DESamplerBuilder::new()
    }
}

impl Default for DESampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for configuring a [`DESampler`].
///
/// All options have sensible defaults:
/// - `population_size`: `max(10 * n_dims, 15)` (auto-computed from parameter count)
/// - `mutation_factor` (F): 0.8
/// - `crossover_rate` (CR): 0.9
/// - `strategy`: `Rand1`
/// - `seed`: random
///
/// # Examples
///
/// ```
/// use optimizer::sampler::de::{DESamplerBuilder, DEStrategy};
///
/// let sampler = DESamplerBuilder::new()
///     .mutation_factor(0.5)
///     .crossover_rate(0.7)
///     .strategy(DEStrategy::CurrentToBest1)
///     .population_size(20)
///     .seed(42)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct DESamplerBuilder {
    population_size: Option<usize>,
    mutation_factor: f64,
    crossover_rate: f64,
    strategy: DEStrategy,
    seed: Option<u64>,
}

impl Default for DESamplerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DESamplerBuilder {
    /// Creates a new builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            population_size: None,
            mutation_factor: 0.8,
            crossover_rate: 0.9,
            strategy: DEStrategy::Rand1,
            seed: None,
        }
    }

    /// Sets the population size.
    ///
    /// Number of candidate solutions maintained across generations.
    /// Larger populations improve robustness but require more evaluations
    /// per generation.
    ///
    /// Default: `max(10 * n_continuous_dims, 15)`.
    #[must_use]
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = Some(size);
        self
    }

    /// Sets the mutation factor (F).
    ///
    /// Controls the amplification of differential variation.
    /// Typical values are in `[0.5, 1.0]`. Higher values increase
    /// exploration; lower values favor exploitation.
    ///
    /// Default: 0.8.
    #[must_use]
    pub fn mutation_factor(mut self, f: f64) -> Self {
        self.mutation_factor = f;
        self
    }

    /// Sets the crossover rate (CR).
    ///
    /// Probability of each dimension being taken from the mutant vector
    /// rather than the parent. Typical values are in `[0.7, 1.0]`.
    ///
    /// Default: 0.9.
    #[must_use]
    pub fn crossover_rate(mut self, cr: f64) -> Self {
        self.crossover_rate = cr;
        self
    }

    /// Sets the mutation strategy.
    ///
    /// Default: [`DEStrategy::Rand1`].
    #[must_use]
    pub fn strategy(mut self, strategy: DEStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets the random seed for reproducibility.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Builds the configured [`DESampler`].
    #[must_use]
    pub fn build(self) -> DESampler {
        DESampler {
            state: Mutex::new(State::new(
                self.population_size,
                self.mutation_factor,
                self.crossover_rate,
                self.strategy,
                self.seed,
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Describes how a parameter dimension maps into the DE internal vector.
#[derive(Clone, Debug)]
struct DimensionInfo {
    /// The distribution for this dimension (stored for decoding).
    distribution: Distribution,
    /// Whether this dimension participates in DE (Float/Int = true, Categorical = false).
    is_continuous: bool,
    /// Internal-space bounds for continuous dimensions: `(low, high)`.
    /// For log-scale parameters these are in log-space.
    bounds: Option<(f64, f64)>,
}

/// A candidate solution produced by mutation + crossover.
#[derive(Clone, Debug)]
struct Candidate {
    /// Internal-space vector (only continuous dimensions).
    x: Vec<f64>,
    /// Values for categorical dimensions (index in `dimensions` -> categorical index).
    categorical_values: HashMap<usize, usize>,
    /// Index of the population member this candidate competes against.
    target_idx: usize,
}

/// Tracks per-trial sampling progress.
#[derive(Clone, Debug)]
struct TrialProgress {
    /// Index of the candidate assigned to this trial.
    candidate_idx: usize,
    /// Next dimension to return for this trial.
    next_dim: usize,
}

/// Phase of the DE state machine.
enum Phase {
    /// Discovering the search space structure (first trial).
    Discovery,
    /// Active sampling and evolving.
    Active,
}

/// Top-level mutable state behind the `Mutex`.
struct State {
    /// The RNG used for sampling.
    rng: fastrand::Rng,
    /// User-provided population size (None = auto).
    user_population_size: Option<usize>,
    /// Mutation factor (F).
    mutation_factor: f64,
    /// Crossover rate (CR).
    crossover_rate: f64,
    /// Mutation strategy.
    strategy: DEStrategy,
    /// Current phase.
    phase: Phase,
    /// Discovered dimension info (populated during discovery).
    dimensions: Vec<DimensionInfo>,
    /// Last `trial_id` seen during discovery.
    discovery_trial_id: Option<u64>,

    // --- Population state ---
    /// Current population (internal-space vectors, continuous dims only).
    population: Vec<Vec<f64>>,
    /// Categorical values for each population member.
    population_categorical: Vec<HashMap<usize, usize>>,
    /// Objective values for the current population.
    population_values: Vec<f64>,
    /// Index of the best population member.
    best_idx: usize,
    /// Whether the initial population has been evaluated.
    initialized: bool,
    /// Effective population size (resolved after discovery).
    population_size: usize,

    // --- Current generation ---
    /// Current generation's candidates.
    candidates: Vec<Candidate>,
    /// Mapping from `trial_id` to its progress.
    trial_progress: HashMap<u64, TrialProgress>,
    /// Number of candidates assigned so far in the current generation.
    assigned_count: usize,
    /// Trial IDs assigned in the current generation.
    generation_trial_ids: Vec<u64>,
}

impl State {
    fn new(
        user_population_size: Option<usize>,
        mutation_factor: f64,
        crossover_rate: f64,
        strategy: DEStrategy,
        seed: Option<u64>,
    ) -> Self {
        let rng = seed.map_or_else(fastrand::Rng::new, fastrand::Rng::with_seed);
        Self {
            rng,
            user_population_size,
            mutation_factor,
            crossover_rate,
            strategy,
            phase: Phase::Discovery,
            dimensions: Vec::new(),
            discovery_trial_id: None,
            population: Vec::new(),
            population_categorical: Vec::new(),
            population_values: Vec::new(),
            best_idx: 0,
            initialized: false,
            population_size: 0,
            candidates: Vec::new(),
            trial_progress: HashMap::new(),
            assigned_count: 0,
            generation_trial_ids: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute internal-space bounds for a distribution.
#[allow(clippy::cast_precision_loss)]
fn internal_bounds(distribution: &Distribution) -> Option<(f64, f64)> {
    match distribution {
        Distribution::Float(d) => {
            if d.log_scale {
                Some((d.low.ln(), d.high.ln()))
            } else {
                Some((d.low, d.high))
            }
        }
        Distribution::Int(d) => {
            if d.log_scale {
                Some(((d.low as f64).ln(), (d.high as f64).ln()))
            } else {
                Some((d.low as f64, d.high as f64))
            }
        }
        Distribution::Categorical(_) => None,
    }
}

/// Convert an internal-space value back to a `ParamValue`.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn from_internal(value: f64, distribution: &Distribution) -> ParamValue {
    match distribution {
        Distribution::Float(d) => {
            let v = if d.log_scale { value.exp() } else { value };
            let v = if let Some(step) = d.step {
                let k = ((v - d.low) / step).round();
                d.low + k * step
            } else {
                v
            };
            ParamValue::Float(v.clamp(d.low, d.high))
        }
        Distribution::Int(d) => {
            let v = if d.log_scale { value.exp() } else { value };
            let v = if let Some(step) = d.step {
                let k = ((v - d.low as f64) / step as f64).round() as i64;
                d.low + k * step
            } else {
                v.round() as i64
            };
            ParamValue::Int(v.clamp(d.low, d.high))
        }
        Distribution::Categorical(_) => {
            unreachable!("from_internal should not be called for categorical distributions")
        }
    }
}

/// Sample a random value for any distribution.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn sample_random(rng: &mut fastrand::Rng, distribution: &Distribution) -> ParamValue {
    match distribution {
        Distribution::Float(d) => {
            let value = if d.log_scale {
                let log_low = d.low.ln();
                let log_high = d.high.ln();
                rng_util::f64_range(rng, log_low, log_high).exp()
            } else if let Some(step) = d.step {
                let n_steps = ((d.high - d.low) / step).floor() as i64;
                let k = rng.i64(0..=n_steps);
                d.low + (k as f64) * step
            } else {
                rng_util::f64_range(rng, d.low, d.high)
            };
            ParamValue::Float(value)
        }
        Distribution::Int(d) => {
            let value = if d.log_scale {
                let log_low = (d.low as f64).ln();
                let log_high = (d.high as f64).ln();
                let raw = rng_util::f64_range(rng, log_low, log_high).exp().round() as i64;
                raw.clamp(d.low, d.high)
            } else if let Some(step) = d.step {
                let n_steps = (d.high - d.low) / step;
                let k = rng.i64(0..=n_steps);
                d.low + k * step
            } else {
                rng.i64(d.low..=d.high)
            };
            ParamValue::Int(value)
        }
        Distribution::Categorical(d) => ParamValue::Categorical(rng.usize(0..d.n_choices)),
    }
}

/// Sample a random value in internal space for a continuous dimension.
fn sample_random_internal(rng: &mut fastrand::Rng, bounds: (f64, f64)) -> f64 {
    rng_util::f64_range(rng, bounds.0, bounds.1)
}

/// Clamp a value to the given bounds.
fn clamp_to_bounds(value: f64, bounds: Option<(f64, f64)>) -> f64 {
    if let Some((lo, hi)) = bounds {
        value.clamp(lo, hi)
    } else {
        value
    }
}

// ---------------------------------------------------------------------------
// DE algorithm
// ---------------------------------------------------------------------------

/// Select `count` distinct random indices from `0..n`, all different from `exclude`.
fn select_random_indices(
    rng: &mut fastrand::Rng,
    n: usize,
    count: usize,
    exclude: &[usize],
) -> Vec<usize> {
    let mut selected = Vec::with_capacity(count);
    while selected.len() < count {
        let idx = rng.usize(0..n);
        if !exclude.contains(&idx) && !selected.contains(&idx) {
            selected.push(idx);
        }
    }
    selected
}

/// Generate trial vectors (mutation + crossover) for the current population.
fn generate_trial_vectors(state: &mut State) -> Vec<Candidate> {
    let n_continuous = state.dimensions.iter().filter(|d| d.is_continuous).count();
    let pop_size = state.population_size;

    let mut candidates = Vec::with_capacity(pop_size);

    for i in 0..pop_size {
        // Mutation
        let mutant = create_mutant_with_rng(state, i, n_continuous);

        // Crossover (binomial)
        let j_rand = state.rng.usize(0..n_continuous.max(1));
        let trial_x: Vec<f64> = if n_continuous > 0 {
            (0..n_continuous)
                .map(|j| {
                    let use_mutant = j == j_rand || state.rng.f64() < state.crossover_rate;
                    let val = if use_mutant {
                        mutant[j]
                    } else {
                        state.population[i][j]
                    };
                    // Clamp to bounds
                    let dim_bounds = continuous_dim_bounds(&state.dimensions, j);
                    clamp_to_bounds(val, dim_bounds)
                })
                .collect()
        } else {
            Vec::new()
        };

        // Categorical: randomly sample (DE doesn't optimize categoricals)
        let mut categorical_values = HashMap::new();
        for (dim_idx, dim) in state.dimensions.iter().enumerate() {
            if !dim.is_continuous
                && let Distribution::Categorical(cat) = &dim.distribution
            {
                categorical_values.insert(dim_idx, state.rng.usize(0..cat.n_choices));
            }
        }

        candidates.push(Candidate {
            x: trial_x,
            categorical_values,
            target_idx: i,
        });
    }

    candidates
}

/// Create a mutant vector, consuming RNG from state.
fn create_mutant_with_rng(state: &mut State, target_idx: usize, n_continuous: usize) -> Vec<f64> {
    if n_continuous == 0 {
        return Vec::new();
    }

    let pop = &state.population;
    let best_idx = state.best_idx;
    let f = state.mutation_factor;
    let pop_size = state.population_size;

    match state.strategy {
        DEStrategy::Rand1 => {
            let indices = select_random_indices(&mut state.rng, pop_size, 3, &[target_idx]);
            let (r1, r2, r3) = (indices[0], indices[1], indices[2]);
            (0..n_continuous)
                .map(|j| pop[r1][j] + f * (pop[r2][j] - pop[r3][j]))
                .collect()
        }
        DEStrategy::Best1 => {
            let indices = select_random_indices(&mut state.rng, pop_size, 2, &[target_idx]);
            let (r1, r2) = (indices[0], indices[1]);
            (0..n_continuous)
                .map(|j| pop[best_idx][j] + f * (pop[r1][j] - pop[r2][j]))
                .collect()
        }
        DEStrategy::CurrentToBest1 => {
            let indices = select_random_indices(&mut state.rng, pop_size, 2, &[target_idx]);
            let (r1, r2) = (indices[0], indices[1]);
            (0..n_continuous)
                .map(|j| {
                    pop[target_idx][j]
                        + f * (pop[best_idx][j] - pop[target_idx][j])
                        + f * (pop[r1][j] - pop[r2][j])
                })
                .collect()
        }
    }
}

/// Get the bounds for the j-th continuous dimension.
fn continuous_dim_bounds(
    dimensions: &[DimensionInfo],
    continuous_idx: usize,
) -> Option<(f64, f64)> {
    let mut ci = 0;
    for dim in dimensions {
        if dim.is_continuous {
            if ci == continuous_idx {
                return dim.bounds;
            }
            ci += 1;
        }
    }
    None
}

/// Generate the initial random population.
fn generate_initial_population(state: &mut State) -> Vec<Candidate> {
    let n_continuous = state.dimensions.iter().filter(|d| d.is_continuous).count();

    let mut candidates = Vec::with_capacity(state.population_size);

    for i in 0..state.population_size {
        let x: Vec<f64> = if n_continuous > 0 {
            let mut v = Vec::with_capacity(n_continuous);
            for dim in &state.dimensions {
                if dim.is_continuous {
                    let val = if let Some(bounds) = dim.bounds {
                        sample_random_internal(&mut state.rng, bounds)
                    } else {
                        0.0
                    };
                    v.push(val);
                }
            }
            v
        } else {
            Vec::new()
        };

        let mut categorical_values = HashMap::new();
        for (dim_idx, dim) in state.dimensions.iter().enumerate() {
            if !dim.is_continuous
                && let Distribution::Categorical(cat) = &dim.distribution
            {
                categorical_values.insert(dim_idx, state.rng.usize(0..cat.n_choices));
            }
        }

        candidates.push(Candidate {
            x,
            categorical_values,
            target_idx: i,
        });
    }

    candidates
}

// ---------------------------------------------------------------------------
// Sampler trait implementation
// ---------------------------------------------------------------------------

impl Sampler for DESampler {
    #[allow(clippy::cast_precision_loss)]
    fn sample(
        &self,
        distribution: &Distribution,
        trial_id: u64,
        history: &[CompletedTrial],
    ) -> ParamValue {
        let mut state = self.state.lock();

        match &state.phase {
            Phase::Discovery => sample_discovery(&mut state, distribution, trial_id),
            Phase::Active => sample_active(&mut state, distribution, trial_id, history),
        }
    }
}

/// Handle sampling during the discovery phase.
fn sample_discovery(state: &mut State, distribution: &Distribution, trial_id: u64) -> ParamValue {
    // Check if this is a new trial (discovery phase ended for previous trial)
    if let Some(prev_id) = state.discovery_trial_id
        && trial_id != prev_id
    {
        // First trial is done; we know the search space. Initialize DE.
        finalize_discovery(state);
        return sample_active(state, distribution, trial_id, &[]);
    }

    // Record this trial_id
    state.discovery_trial_id = Some(trial_id);

    // Record this dimension
    let is_continuous = !matches!(distribution, Distribution::Categorical(_));
    let bounds = internal_bounds(distribution);
    state.dimensions.push(DimensionInfo {
        distribution: distribution.clone(),
        is_continuous,
        bounds,
    });

    // Sample randomly for the discovery trial
    sample_random(&mut state.rng, distribution)
}

/// Finalize discovery and transition to the active phase.
#[allow(clippy::cast_precision_loss)]
fn finalize_discovery(state: &mut State) {
    let n_continuous = state.dimensions.iter().filter(|d| d.is_continuous).count();

    // Resolve population size
    state.population_size = state
        .user_population_size
        .unwrap_or_else(|| (10 * n_continuous).max(15));

    // Ensure population size is at least 4 (DE needs distinct random indices)
    state.population_size = state.population_size.max(4);

    // Generate initial random population
    state.candidates = generate_initial_population(state);
    state.assigned_count = 0;
    state.generation_trial_ids.clear();
    state.trial_progress.clear();
    state.phase = Phase::Active;
}

/// Handle sampling during the active phase.
fn sample_active(
    state: &mut State,
    distribution: &Distribution,
    trial_id: u64,
    history: &[CompletedTrial],
) -> ParamValue {
    // Check if we need to process completed trials and start a new generation
    maybe_update_generation(state, history);

    // Assign a candidate to this trial if not yet done
    if !state.trial_progress.contains_key(&trial_id) {
        assign_candidate(state, trial_id);
    }

    let progress = state.trial_progress.get_mut(&trial_id).unwrap();
    let dim_idx = progress.next_dim;
    progress.next_dim += 1;

    // Safety check
    if dim_idx >= state.dimensions.len() {
        return sample_random(&mut state.rng, distribution);
    }

    let candidate = &state.candidates[progress.candidate_idx];
    let dim_info = &state.dimensions[dim_idx];

    if dim_info.is_continuous {
        // Map from overall dimension index to continuous index
        let ci = state.dimensions[..dim_idx]
            .iter()
            .filter(|d| d.is_continuous)
            .count();
        if ci < candidate.x.len() {
            from_internal(candidate.x[ci], &dim_info.distribution)
        } else {
            sample_random(&mut state.rng, distribution)
        }
    } else {
        // Categorical: use pre-sampled value
        if let Some(&cat_idx) = candidate.categorical_values.get(&dim_idx) {
            ParamValue::Categorical(cat_idx)
        } else {
            sample_random(&mut state.rng, distribution)
        }
    }
}

/// Assign a candidate to a trial.
fn assign_candidate(state: &mut State, trial_id: u64) {
    let candidate_idx = if state.assigned_count < state.candidates.len() {
        let idx = state.assigned_count;
        state.assigned_count += 1;
        idx
    } else {
        // Overflow: generate an extra random candidate
        let n_continuous = state.dimensions.iter().filter(|d| d.is_continuous).count();
        let x: Vec<f64> = (0..n_continuous)
            .map(|j| {
                let bounds = continuous_dim_bounds(&state.dimensions, j);
                if let Some(b) = bounds {
                    sample_random_internal(&mut state.rng, b)
                } else {
                    0.0
                }
            })
            .collect();
        let mut categorical_values = HashMap::new();
        for (dim_idx, dim) in state.dimensions.iter().enumerate() {
            if !dim.is_continuous
                && let Distribution::Categorical(cat) = &dim.distribution
            {
                categorical_values.insert(dim_idx, state.rng.usize(0..cat.n_choices));
            }
        }
        state.candidates.push(Candidate {
            x,
            categorical_values,
            target_idx: 0, // overflow candidates don't compete
        });
        let idx = state.candidates.len() - 1;
        state.assigned_count = state.candidates.len();
        idx
    };

    state.trial_progress.insert(
        trial_id,
        TrialProgress {
            candidate_idx,
            next_dim: 0,
        },
    );
    state.generation_trial_ids.push(trial_id);
}

/// Check if we should process completed trials and start a new generation.
fn maybe_update_generation(state: &mut State, history: &[CompletedTrial]) {
    let pop_size = state.population_size;

    // Only update when at least pop_size candidates have been assigned
    if state.generation_trial_ids.len() < pop_size {
        return;
    }

    // Check if the first pop_size trial IDs are all completed
    let trial_ids: Vec<u64> = state
        .generation_trial_ids
        .iter()
        .take(pop_size)
        .copied()
        .collect();
    let history_map: HashMap<u64, f64> = history.iter().map(|t| (t.id, t.value)).collect();

    let all_completed = trial_ids.iter().all(|id| history_map.contains_key(id));
    if !all_completed {
        return;
    }

    let n_continuous = state.dimensions.iter().filter(|d| d.is_continuous).count();

    if state.initialized {
        // Subsequent generations: selection
        perform_selection(state, &trial_ids, &history_map);
    } else {
        // First generation: store as initial population
        initialize_population(state, &trial_ids, &history_map, n_continuous);
    }

    // Generate next generation's trial vectors
    state.candidates = if state.initialized && n_continuous > 0 {
        generate_trial_vectors(state)
    } else {
        generate_initial_population(state)
    };
    state.assigned_count = 0;
    state.generation_trial_ids.clear();
    state.trial_progress.clear();
}

/// Initialize the population from the first generation's results.
fn initialize_population(
    state: &mut State,
    trial_ids: &[u64],
    history_map: &HashMap<u64, f64>,
    _n_continuous: usize,
) {
    state.population.clear();
    state.population_categorical.clear();
    state.population_values.clear();

    let mut best_value = f64::INFINITY;
    let mut best_idx = 0;

    for (i, &trial_id) in trial_ids.iter().enumerate() {
        let progress = &state.trial_progress[&trial_id];
        let candidate = &state.candidates[progress.candidate_idx];
        let value = history_map[&trial_id];

        state.population.push(candidate.x.clone());
        state
            .population_categorical
            .push(candidate.categorical_values.clone());
        state.population_values.push(value);

        if value < best_value {
            best_value = value;
            best_idx = i;
        }
    }

    state.best_idx = best_idx;
    state.initialized = true;
}

/// Perform DE selection: replace parent if trial vector is better.
fn perform_selection(state: &mut State, trial_ids: &[u64], history_map: &HashMap<u64, f64>) {
    for &trial_id in trial_ids {
        let progress = &state.trial_progress[&trial_id];
        let candidate = &state.candidates[progress.candidate_idx];
        let trial_value = history_map[&trial_id];
        let target_idx = candidate.target_idx;

        if target_idx < state.population_size && trial_value <= state.population_values[target_idx]
        {
            state.population[target_idx] = candidate.x.clone();
            state.population_categorical[target_idx] = candidate.categorical_values.clone();
            state.population_values[target_idx] = trial_value;
        }
    }

    // Update best index
    let mut best_value = f64::INFINITY;
    let mut best_idx = 0;
    for (i, &val) in state.population_values.iter().enumerate() {
        if val < best_value {
            best_value = val;
            best_idx = i;
        }
    }
    state.best_idx = best_idx;
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::distribution::FloatDistribution;

    #[test]
    fn test_de_sampler_basic_float() {
        let sampler = DESampler::with_seed(42);
        let dist = Distribution::Float(FloatDistribution {
            low: -5.0,
            high: 5.0,
            log_scale: false,
            step: None,
        });

        // Sample many values and check bounds
        for i in 0..100 {
            let value = sampler.sample(&dist, i, &[]);
            if let ParamValue::Float(v) = value {
                assert!(
                    (-5.0..=5.0).contains(&v),
                    "value {v} out of bounds at trial {i}"
                );
            } else {
                panic!("Expected Float value");
            }
        }
    }

    #[test]
    fn test_de_sampler_reproducibility() {
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        let sample_values = |seed: u64| {
            let sampler = DESampler::with_seed(seed);
            (0..20)
                .map(|i| sampler.sample(&dist, i, &[]))
                .collect::<Vec<_>>()
        };

        let v1 = sample_values(42);
        let v2 = sample_values(42);
        assert_eq!(v1, v2, "same seed should produce same results");

        let v3 = sample_values(99);
        assert_ne!(v1, v3, "different seeds should produce different results");
    }

    #[test]
    fn test_de_strategy_default() {
        assert!(matches!(DEStrategy::default(), DEStrategy::Rand1));
    }

    #[test]
    fn test_builder_defaults() {
        let builder = DESamplerBuilder::new();
        assert!(builder.population_size.is_none());
        assert!((builder.mutation_factor - 0.8).abs() < f64::EPSILON);
        assert!((builder.crossover_rate - 0.9).abs() < f64::EPSILON);
        assert!(matches!(builder.strategy, DEStrategy::Rand1));
        assert!(builder.seed.is_none());
    }
}
