//! NSGA-II (Non-dominated Sorting Genetic Algorithm II) sampler.
//!
//! NSGA-II is one of the most widely used evolutionary multi-objective
//! optimization algorithms. It ranks the population using **non-dominated
//! sorting** (fast O(MN²) algorithm) and breaks ties within the same
//! Pareto front using **crowding distance**, which favors solutions in
//! less-crowded regions of the objective space.
//!
//! # Algorithm
//!
//! Each generation proceeds as follows:
//!
//! 1. **Non-dominated sorting** — partition the combined parent+offspring
//!    population into Pareto fronts F₁, F₂, …
//! 2. **Crowding distance** — for each front, compute per-solution crowding
//!    distance (sum of normalized neighbor gaps in each objective).
//! 3. **Selection** — fill the next population front-by-front. When a front
//!    only partially fits, prefer solutions with higher crowding distance.
//! 4. **Binary tournament** — select parents using (rank, crowding distance)
//!    comparisons.
//! 5. **SBX crossover + polynomial mutation** — generate offspring.
//!
//! # When to use
//!
//! - Two-objective problems where you want a well-spread Pareto front.
//! - General-purpose multi-objective optimization with moderate population
//!   sizes.
//! - Problems that benefit from diversity preservation via crowding distance.
//!
//! For problems with **three or more objectives**, consider
//! [`Nsga3Sampler`](super::nsga3::Nsga3Sampler) (reference-point niching)
//! or [`MoeadSampler`](super::moead::MoeadSampler) (decomposition).
//!
//! # Configuration
//!
//! | Parameter | Builder method | Default |
//! |-----------|---------------|---------|
//! | Population size | [`population_size`](Nsga2SamplerBuilder::population_size) | `4 + floor(3 * ln(n_params))`, min 4 |
//! | Crossover probability | [`crossover_prob`](Nsga2SamplerBuilder::crossover_prob) | 0.9 |
//! | SBX distribution index | [`crossover_eta`](Nsga2SamplerBuilder::crossover_eta) | 20.0 |
//! | Mutation distribution index | [`mutation_eta`](Nsga2SamplerBuilder::mutation_eta) | 20.0 |
//! | Random seed | [`seed`](Nsga2SamplerBuilder::seed) | random |
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
//!     .optimize(50, |trial: &mut optimizer::Trial| {
//!         let xv = x.suggest(trial)?;
//!         Ok::<_, optimizer::Error>(vec![xv * xv, (xv - 1.0).powi(2)])
//!     })
//!     .unwrap();
//! ```

use parking_lot::Mutex;

use super::genetic::{
    self, Candidate, EvolutionaryState, Phase, advance_generation, collect_evaluated_generation,
    crossover, extract_trial_params, finalize_discovery, generate_random_candidates, mutate,
    sample_from_candidate, sample_random,
};
use crate::distribution::Distribution;
use crate::multi_objective::MultiObjectiveTrial;
use crate::param::ParamValue;
use crate::pareto;
use crate::types::Direction;

/// NSGA-II sampler for multi-objective optimization.
///
/// Use non-dominated sorting with crowding-distance tie-breaking to
/// evolve a well-spread Pareto front. Best suited for bi-objective
/// problems; for 3+ objectives prefer [`Nsga3Sampler`](super::nsga3::Nsga3Sampler).
///
/// Create with [`Nsga2Sampler::new`], [`Nsga2Sampler::with_seed`], or
/// [`Nsga2Sampler::builder`] for full configuration.
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

struct Nsga2State {
    evo: EvolutionaryState,
    config: Nsga2Config,
}

impl Nsga2State {
    fn new(config: Nsga2Config, seed: Option<u64>) -> Self {
        Self {
            evo: EvolutionaryState::new(seed),
            config,
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

        match &state.evo.phase {
            Phase::Discovery => {
                if let Some(value) =
                    genetic::sample_discovery(&mut state.evo, distribution, trial_id)
                {
                    return value;
                }
                // Transitioned to active phase
                let user_pop = state.config.user_population_size;
                finalize_discovery(&mut state.evo, user_pop);
                generate_random_candidates(&mut state.evo);
                sample_from_candidate(&mut state.evo, trial_id)
            }
            Phase::Active => {
                maybe_generate_new_generation(&mut state, history, directions);
                sample_from_candidate(&mut state.evo, trial_id)
            }
        }
    }
}

/// Check if all candidates in the current generation have been evaluated;
/// if so, run NSGA-II selection and generate offspring.
fn maybe_generate_new_generation(
    state: &mut Nsga2State,
    history: &[MultiObjectiveTrial],
    directions: &[Direction],
) {
    if state.evo.candidates.is_empty() {
        generate_random_candidates(&mut state.evo);
        return;
    }

    if let Some(evaluated) = collect_evaluated_generation(&state.evo, history) {
        let offspring = nsga2_generate_offspring(state, &evaluated, directions);
        advance_generation(&mut state.evo, offspring);
    }
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
    let pop_size = state.evo.population_size;

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
        selected.push(state.evo.rng.usize(0..n));
    }

    let parents: Vec<Vec<ParamValue>> = selected
        .iter()
        .map(|&idx| {
            extract_trial_params(population[idx], &state.evo.dimensions, &mut state.evo.rng)
        })
        .collect();

    let sel_rank: Vec<usize> = selected.iter().map(|&i| rank[i]).collect();
    let sel_crowding: Vec<f64> = selected.iter().map(|&i| crowding[i]).collect();

    (parents, sel_rank, sel_crowding)
}

/// Runs NSGA-II selection and generates offspring candidates.
fn nsga2_generate_offspring(
    state: &mut Nsga2State,
    population: &[&MultiObjectiveTrial],
    directions: &[Direction],
) -> Vec<Candidate> {
    let pop_size = state.evo.population_size;

    if population.len() < 2 {
        return (0..pop_size)
            .map(|_| {
                let params = state
                    .evo
                    .dimensions
                    .iter()
                    .map(|d| sample_random(&mut state.evo.rng, &d.distribution))
                    .collect();
                Candidate { params }
            })
            .collect();
    }

    let (parents, sel_rank, sel_crowding) = nsga2_select(state, population, directions);

    let mut offspring = Vec::with_capacity(pop_size);
    while offspring.len() < pop_size {
        let p1 = tournament_select(&mut state.evo.rng, &sel_rank, &sel_crowding, parents.len());
        let p2 = tournament_select(&mut state.evo.rng, &sel_rank, &sel_crowding, parents.len());

        let (mut child1, mut child2) = crossover(
            &mut state.evo.rng,
            &parents[p1],
            &parents[p2],
            &state.evo.dimensions,
            state.config.crossover_prob,
            state.config.crossover_eta,
        );

        mutate(
            &mut state.evo.rng,
            &mut child1,
            &state.evo.dimensions,
            state.config.mutation_eta,
        );
        mutate(
            &mut state.evo.rng,
            &mut child2,
            &state.evo.dimensions,
            state.config.mutation_eta,
        );

        offspring.push(Candidate { params: child1 });
        if offspring.len() < pop_size {
            offspring.push(Candidate { params: child2 });
        }
    }

    offspring
}

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
