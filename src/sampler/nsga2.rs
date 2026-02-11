//! NSGA-II (Non-dominated Sorting Genetic Algorithm II) sampler.
//!
//! Implements multi-objective optimization using non-dominated sorting,
//! crowding distance, SBX crossover, and polynomial mutation.
//!
//! # Examples
//!
//! ```
//! use optimizer::Direction;
//! use optimizer::multi_objective::MultiObjectiveStudy;
//! use optimizer::parameter::{FloatParam, Parameter};
//! use optimizer::sampler::nsga2::Nsga2Sampler;
//!
//! let sampler = Nsga2Sampler::with_seed(42);
//! let study =
//!     MultiObjectiveStudy::with_sampler(vec![Direction::Minimize, Direction::Minimize], sampler);
//!
//! let x = FloatParam::new(0.0, 1.0);
//! study
//!     .optimize(50, |trial| {
//!         let xv = x.suggest(trial)?;
//!         Ok::<_, optimizer::Error>(vec![xv * xv, (xv - 1.0).powi(2)])
//!     })
//!     .unwrap();
//! ```

use std::collections::HashMap;

use parking_lot::Mutex;

use crate::distribution::Distribution;
use crate::multi_objective::MultiObjectiveTrial;
use crate::param::ParamValue;
use crate::types::Direction;
use crate::{pareto, rng_util};

/// NSGA-II sampler for multi-objective optimization.
///
/// Provides non-dominated sorting, crowding distance selection,
/// SBX crossover, and polynomial mutation.
pub struct Nsga2Sampler {
    state: Mutex<Nsga2State>,
}

impl Nsga2Sampler {
    /// Creates a new NSGA-II sampler with a random seed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(Nsga2State::new(Nsga2Config::default(), None)),
        }
    }

    /// Creates a new NSGA-II sampler with a fixed seed.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            state: Mutex::new(Nsga2State::new(Nsga2Config::default(), Some(seed))),
        }
    }

    /// Creates a builder for configuring an `Nsga2Sampler`.
    #[must_use]
    pub fn builder() -> Nsga2SamplerBuilder {
        Nsga2SamplerBuilder::default()
    }
}

impl Default for Nsga2Sampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`Nsga2Sampler`].
#[derive(Debug, Clone, Default)]
pub struct Nsga2SamplerBuilder {
    population_size: Option<usize>,
    crossover_prob: Option<f64>,
    crossover_eta: Option<f64>,
    mutation_eta: Option<f64>,
    seed: Option<u64>,
}

impl Nsga2SamplerBuilder {
    /// Sets the population size. Default: `4 + floor(3 * ln(n_params))`, minimum 4.
    #[must_use]
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = Some(size);
        self
    }

    /// Sets the crossover probability. Default: 0.9.
    #[must_use]
    pub fn crossover_prob(mut self, prob: f64) -> Self {
        self.crossover_prob = Some(prob);
        self
    }

    /// Sets the SBX distribution index. Default: 20.0.
    #[must_use]
    pub fn crossover_eta(mut self, eta: f64) -> Self {
        self.crossover_eta = Some(eta);
        self
    }

    /// Sets the polynomial mutation distribution index. Default: 20.0.
    #[must_use]
    pub fn mutation_eta(mut self, eta: f64) -> Self {
        self.mutation_eta = Some(eta);
        self
    }

    /// Sets the random seed for reproducibility.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Builds the configured [`Nsga2Sampler`].
    #[must_use]
    pub fn build(self) -> Nsga2Sampler {
        let config = Nsga2Config {
            user_population_size: self.population_size,
            crossover_prob: self.crossover_prob.unwrap_or(0.9),
            crossover_eta: self.crossover_eta.unwrap_or(20.0),
            mutation_eta: self.mutation_eta.unwrap_or(20.0),
        };
        Nsga2Sampler {
            state: Mutex::new(Nsga2State::new(config, self.seed)),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Nsga2Config {
    user_population_size: Option<usize>,
    crossover_prob: f64,
    crossover_eta: f64,
    mutation_eta: f64,
}

impl Default for Nsga2Config {
    fn default() -> Self {
        Self {
            user_population_size: None,
            crossover_prob: 0.9,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
        }
    }
}

/// Describes a parameter dimension.
#[derive(Clone, Debug)]
struct DimensionInfo {
    distribution: Distribution,
}

/// A candidate solution: one value per dimension.
#[derive(Clone, Debug)]
struct Candidate {
    params: Vec<ParamValue>,
}

/// Tracks per-trial sampling progress.
#[derive(Clone, Debug)]
struct TrialProgress {
    candidate_idx: usize,
    next_dim: usize,
}

enum Phase {
    /// First trial reveals parameter dimensions.
    Discovery,
    /// NSGA-II optimisation.
    Active,
}

struct Nsga2State {
    rng: fastrand::Rng,
    config: Nsga2Config,
    phase: Phase,
    dimensions: Vec<DimensionInfo>,
    population_size: usize,
    candidates: Vec<Candidate>,
    trial_progress: HashMap<u64, TrialProgress>,
    assigned_count: usize,
    generation_trial_ids: Vec<u64>,
    discovery_trial_id: Option<u64>,
    /// How many complete generations have been evaluated.
    generation: usize,
}

impl Nsga2State {
    fn new(config: Nsga2Config, seed: Option<u64>) -> Self {
        let rng = seed.map_or_else(fastrand::Rng::new, fastrand::Rng::with_seed);
        Self {
            rng,
            config,
            phase: Phase::Discovery,
            dimensions: Vec::new(),
            population_size: 4,
            candidates: Vec::new(),
            trial_progress: HashMap::new(),
            assigned_count: 0,
            generation_trial_ids: Vec::new(),
            discovery_trial_id: None,
            generation: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// MultiObjectiveSampler implementation
// ---------------------------------------------------------------------------

impl crate::multi_objective::MultiObjectiveSampler for Nsga2Sampler {
    fn sample(
        &self,
        distribution: &Distribution,
        trial_id: u64,
        history: &[MultiObjectiveTrial],
        directions: &[Direction],
    ) -> ParamValue {
        let mut state = self.state.lock();

        match &state.phase {
            Phase::Discovery => sample_discovery(&mut state, distribution, trial_id),
            Phase::Active => sample_active(&mut state, distribution, trial_id, history, directions),
        }
    }
}

/// Handle sampling during the discovery phase.
fn sample_discovery(
    state: &mut Nsga2State,
    distribution: &Distribution,
    trial_id: u64,
) -> ParamValue {
    if let Some(prev_id) = state.discovery_trial_id
        && trial_id != prev_id
    {
        finalize_discovery(state);
        // Assign this trial a random candidate (no history yet)
        generate_random_candidates(state);
        return sample_from_candidate(state, trial_id);
    }

    state.discovery_trial_id = Some(trial_id);
    state.dimensions.push(DimensionInfo {
        distribution: distribution.clone(),
    });

    sample_random(&mut state.rng, distribution)
}

/// Transition from discovery to active phase.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn finalize_discovery(state: &mut Nsga2State) {
    let n = state.dimensions.len();
    state.population_size = state
        .config
        .user_population_size
        .unwrap_or_else(|| (4.0 + 3.0 * (n as f64).ln().max(0.0)).floor() as usize)
        .max(4);
    state.phase = Phase::Active;
}

/// Generate `population_size` random candidates.
fn generate_random_candidates(state: &mut Nsga2State) {
    let pop = state.population_size;
    state.candidates = (0..pop)
        .map(|_| {
            let params: Vec<ParamValue> = state
                .dimensions
                .iter()
                .map(|d| sample_random(&mut state.rng, &d.distribution))
                .collect();
            Candidate { params }
        })
        .collect();
    state.assigned_count = 0;
    state.generation_trial_ids.clear();
    state.trial_progress.clear();
}

/// Active-phase sampling.
fn sample_active(
    state: &mut Nsga2State,
    _distribution: &Distribution,
    trial_id: u64,
    history: &[MultiObjectiveTrial],
    directions: &[Direction],
) -> ParamValue {
    // Check if we need to generate a new generation
    maybe_generate_new_generation(state, history, directions);

    sample_from_candidate(state, trial_id)
}

/// Assign a candidate to a trial and return the next dimension value.
fn sample_from_candidate(state: &mut Nsga2State, trial_id: u64) -> ParamValue {
    // Assign candidate if not yet done
    if !state.trial_progress.contains_key(&trial_id) {
        let candidate_idx = if state.assigned_count < state.candidates.len() {
            let idx = state.assigned_count;
            state.assigned_count += 1;
            idx
        } else {
            // Overflow: generate a random candidate
            let params: Vec<ParamValue> = state
                .dimensions
                .iter()
                .map(|d| sample_random(&mut state.rng, &d.distribution))
                .collect();
            state.candidates.push(Candidate { params });
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

    let progress = state.trial_progress.get_mut(&trial_id).unwrap();
    let dim_idx = progress.next_dim;
    progress.next_dim += 1;

    if dim_idx >= state.dimensions.len() {
        // Extra dimension: sample randomly
        return sample_random(
            &mut state.rng,
            &state.dimensions.last().unwrap().distribution,
        );
    }

    state.candidates[progress.candidate_idx].params[dim_idx].clone()
}

/// Check if all candidates in the current generation have been evaluated;
/// if so, run NSGA-II selection and generate offspring.
fn maybe_generate_new_generation(
    state: &mut Nsga2State,
    history: &[MultiObjectiveTrial],
    directions: &[Direction],
) {
    let pop_size = state.population_size;

    // Need at least pop_size assigned trials
    if state.generation_trial_ids.len() < pop_size {
        // Not enough candidates assigned yet — check if we need initial candidates
        if state.candidates.is_empty() {
            generate_random_candidates(state);
        }
        return;
    }

    // Check if the first pop_size trials are completed
    let gen_ids: Vec<u64> = state
        .generation_trial_ids
        .iter()
        .take(pop_size)
        .copied()
        .collect();
    let history_map: HashMap<u64, &MultiObjectiveTrial> =
        history.iter().map(|t| (t.id, t)).collect();

    let all_completed = gen_ids.iter().all(|id| history_map.contains_key(id));
    if !all_completed {
        return;
    }

    // Collect the evaluated population
    let evaluated: Vec<&MultiObjectiveTrial> = gen_ids
        .iter()
        .filter_map(|id| history_map.get(id).copied())
        .collect();

    // Run NSGA-II to produce offspring
    let offspring = nsga2_generate_offspring(state, &evaluated, directions);
    state.candidates = offspring;
    state.assigned_count = 0;
    state.generation_trial_ids.clear();
    state.trial_progress.clear();
    state.generation += 1;
}

// ---------------------------------------------------------------------------
// NSGA-II generation algorithm
// ---------------------------------------------------------------------------

/// Performs NSGA-II selection: non-dominated sort + crowding distance,
/// then selects `pop_size` parents from the population.
fn nsga2_select(
    state: &mut Nsga2State,
    population: &[&MultiObjectiveTrial],
    directions: &[Direction],
) -> (Vec<Vec<ParamValue>>, Vec<usize>, Vec<f64>) {
    let pop_size = state.population_size;

    let values: Vec<Vec<f64>> = population.iter().map(|t| t.values.clone()).collect();
    let constraints: Vec<Vec<f64>> = population.iter().map(|t| t.constraints.clone()).collect();
    let has_constraints = constraints.iter().any(|c| !c.is_empty());

    let fronts = if has_constraints {
        pareto::fast_non_dominated_sort_constrained(&values, directions, &constraints)
    } else {
        pareto::fast_non_dominated_sort(&values, directions)
    };

    let n = population.len();
    let mut rank = vec![0_usize; n];
    let mut crowding = vec![0.0_f64; n];

    for (front_rank, front) in fronts.iter().enumerate() {
        let cd = pareto::crowding_distance_indexed(front, &values);
        for (i, &idx) in front.iter().enumerate() {
            rank[idx] = front_rank;
            crowding[idx] = cd[i];
        }
    }

    let mut selected: Vec<usize> = Vec::with_capacity(pop_size);
    for front in &fronts {
        if selected.len() + front.len() <= pop_size {
            selected.extend_from_slice(front);
        } else {
            let remaining = pop_size - selected.len();
            let mut front_sorted: Vec<usize> = front.clone();
            front_sorted.sort_by(|&a, &b| {
                crowding[b]
                    .partial_cmp(&crowding[a])
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
            selected.extend_from_slice(&front_sorted[..remaining]);
            break;
        }
    }

    while selected.len() < pop_size {
        selected.push(state.rng.usize(0..n));
    }

    // Extract parent parameter vectors ordered by dimension
    let parents: Vec<Vec<ParamValue>> = selected
        .iter()
        .map(|&idx| extract_trial_params(population[idx], &state.dimensions, &mut state.rng))
        .collect();

    let sel_rank: Vec<usize> = selected.iter().map(|&i| rank[i]).collect();
    let sel_crowding: Vec<f64> = selected.iter().map(|&i| crowding[i]).collect();

    (parents, sel_rank, sel_crowding)
}

/// Extract parameter values from a trial, ordered by dimension index.
fn extract_trial_params(
    trial: &MultiObjectiveTrial,
    dimensions: &[DimensionInfo],
    rng: &mut fastrand::Rng,
) -> Vec<ParamValue> {
    let mut param_pairs: Vec<_> = trial.params.iter().collect();
    param_pairs.sort_by_key(|(id, _)| *id);

    dimensions
        .iter()
        .enumerate()
        .map(|(dim_idx, dim_info)| {
            if dim_idx < param_pairs.len() {
                param_pairs[dim_idx].1.clone()
            } else {
                sample_random(rng, &dim_info.distribution)
            }
        })
        .collect()
}

/// Runs NSGA-II selection and generates offspring candidates.
fn nsga2_generate_offspring(
    state: &mut Nsga2State,
    population: &[&MultiObjectiveTrial],
    directions: &[Direction],
) -> Vec<Candidate> {
    let pop_size = state.population_size;

    if population.len() < 2 {
        return (0..pop_size)
            .map(|_| {
                let params = state
                    .dimensions
                    .iter()
                    .map(|d| sample_random(&mut state.rng, &d.distribution))
                    .collect();
                Candidate { params }
            })
            .collect();
    }

    let (parents, sel_rank, sel_crowding) = nsga2_select(state, population, directions);

    let mut offspring = Vec::with_capacity(pop_size);
    while offspring.len() < pop_size {
        let p1 = tournament_select(&mut state.rng, &sel_rank, &sel_crowding, parents.len());
        let p2 = tournament_select(&mut state.rng, &sel_rank, &sel_crowding, parents.len());

        let (mut child1, mut child2) = crossover(
            &mut state.rng,
            &parents[p1],
            &parents[p2],
            &state.dimensions,
            state.config.crossover_prob,
            state.config.crossover_eta,
        );

        mutate(
            &mut state.rng,
            &mut child1,
            &state.dimensions,
            state.config.mutation_eta,
        );
        mutate(
            &mut state.rng,
            &mut child2,
            &state.dimensions,
            state.config.mutation_eta,
        );

        offspring.push(Candidate { params: child1 });
        if offspring.len() < pop_size {
            offspring.push(Candidate { params: child2 });
        }
    }

    offspring
}

// ---------------------------------------------------------------------------
// Genetic operators
// ---------------------------------------------------------------------------

/// Tournament selection: pick 2 random individuals, return index of winner.
/// Winner has lower rank; ties broken by higher crowding distance.
fn tournament_select(
    rng: &mut fastrand::Rng,
    ranks: &[usize],
    crowding: &[f64],
    n: usize,
) -> usize {
    let a = rng.usize(0..n);
    let b = rng.usize(0..n);

    if ranks[a] < ranks[b] {
        a
    } else if ranks[b] < ranks[a] {
        b
    } else if crowding[a] >= crowding[b] {
        a
    } else {
        b
    }
}

/// SBX crossover for continuous params, uniform crossover for categorical.
fn crossover(
    rng: &mut fastrand::Rng,
    parent1: &[ParamValue],
    parent2: &[ParamValue],
    dimensions: &[DimensionInfo],
    crossover_prob: f64,
    eta: f64,
) -> (Vec<ParamValue>, Vec<ParamValue>) {
    let n = parent1.len();
    let mut child1 = parent1.to_vec();
    let mut child2 = parent2.to_vec();

    let u: f64 = rng_util::f64_range(rng, 0.0, 1.0);
    if u > crossover_prob {
        return (child1, child2);
    }

    for i in 0..n {
        match (&parent1[i], &parent2[i], &dimensions[i].distribution) {
            (ParamValue::Float(p1), ParamValue::Float(p2), Distribution::Float(d)) => {
                if (p1 - p2).abs() < 1e-14 {
                    continue;
                }
                let (c1, c2) = sbx_crossover_f64(rng, *p1, *p2, d.low, d.high, eta);
                child1[i] = ParamValue::Float(c1);
                child2[i] = ParamValue::Float(c2);
            }
            (ParamValue::Int(p1), ParamValue::Int(p2), Distribution::Int(d)) => {
                if p1 == p2 {
                    continue;
                }
                #[allow(clippy::cast_precision_loss)]
                let (c1, c2) = sbx_crossover_f64(
                    rng,
                    *p1 as f64,
                    *p2 as f64,
                    d.low as f64,
                    d.high as f64,
                    eta,
                );
                #[allow(clippy::cast_possible_truncation)]
                {
                    child1[i] = ParamValue::Int((c1.round() as i64).clamp(d.low, d.high));
                    child2[i] = ParamValue::Int((c2.round() as i64).clamp(d.low, d.high));
                }
            }
            (ParamValue::Categorical(_), ParamValue::Categorical(_), _) => {
                // Uniform crossover: swap with 50% probability
                if rng_util::f64_range(rng, 0.0, 1.0) < 0.5 {
                    core::mem::swap(&mut child1[i], &mut child2[i]);
                }
            }
            _ => {}
        }
    }

    (child1, child2)
}

/// SBX crossover for a single float dimension.
fn sbx_crossover_f64(
    rng: &mut fastrand::Rng,
    p1: f64,
    p2: f64,
    low: f64,
    high: f64,
    eta: f64,
) -> (f64, f64) {
    let u: f64 = rng_util::f64_range(rng, 0.0, 1.0);

    let beta = if u <= 0.5 {
        (2.0 * u).powf(1.0 / (eta + 1.0))
    } else {
        (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
    };

    let c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2);
    let c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2);

    (c1.clamp(low, high), c2.clamp(low, high))
}

/// Polynomial mutation for each dimension.
#[allow(clippy::cast_precision_loss)]
fn mutate(
    rng: &mut fastrand::Rng,
    individual: &mut [ParamValue],
    dimensions: &[DimensionInfo],
    eta: f64,
) {
    let n = individual.len();
    if n == 0 {
        return;
    }
    let mutation_prob = 1.0 / n as f64;

    for (i, value) in individual.iter_mut().enumerate() {
        if rng_util::f64_range(rng, 0.0, 1.0) >= mutation_prob {
            continue;
        }

        match (value, &dimensions[i].distribution) {
            (v @ ParamValue::Float(_), Distribution::Float(d)) => {
                let ParamValue::Float(x) = *v else {
                    unreachable!();
                };
                let mutated = polynomial_mutation_f64(rng, x, d.low, d.high, eta);
                *v = ParamValue::Float(mutated);
            }
            (v @ ParamValue::Int(_), Distribution::Int(d)) => {
                let ParamValue::Int(x) = *v else {
                    unreachable!();
                };
                #[allow(clippy::cast_possible_truncation)]
                {
                    let mutated =
                        polynomial_mutation_f64(rng, x as f64, d.low as f64, d.high as f64, eta);
                    *v = ParamValue::Int((mutated.round() as i64).clamp(d.low, d.high));
                }
            }
            (v @ ParamValue::Categorical(_), Distribution::Categorical(d)) => {
                *v = ParamValue::Categorical(rng.usize(0..d.n_choices));
            }
            _ => {}
        }
    }
}

/// Polynomial mutation for a single float value.
fn polynomial_mutation_f64(rng: &mut fastrand::Rng, x: f64, low: f64, high: f64, eta: f64) -> f64 {
    let u: f64 = rng_util::f64_range(rng, 0.0, 1.0);
    let range = high - low;
    if range <= 0.0 {
        return x;
    }

    let delta1 = (x - low) / range;
    let delta2 = (high - x) / range;

    let delta_q = if u < 0.5 {
        let xy = 1.0 - delta1;
        let val = 2.0 * u + (1.0 - 2.0 * u) * xy.powf(eta + 1.0);
        val.powf(1.0 / (eta + 1.0)) - 1.0
    } else {
        let xy = 1.0 - delta2;
        let val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy.powf(eta + 1.0);
        1.0 - val.powf(1.0 / (eta + 1.0))
    };

    (x + delta_q * range).clamp(low, high)
}

// ---------------------------------------------------------------------------
// Random sampling helper (for discovery phase)
// ---------------------------------------------------------------------------

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
