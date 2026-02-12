//! MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) sampler.
//!
//! MOEA/D takes a fundamentally different approach from Pareto-based
//! algorithms like NSGA-II/III. It **decomposes** the multi-objective
//! problem into a set of scalar subproblems using evenly distributed
//! weight vectors (Das-Dennis points), then solves them collaboratively
//! through **neighborhood-based mating and replacement**.
//!
//! # Algorithm
//!
//! 1. **Decompose** — generate weight vectors on the unit simplex and
//!    assign one scalar subproblem per weight vector.
//! 2. **Build neighborhoods** — for each subproblem, find its T nearest
//!    neighbors by Euclidean distance between weight vectors.
//! 3. **Mate from neighborhood** — select parents from the neighborhood
//!    of each subproblem and produce offspring via SBX crossover +
//!    polynomial mutation.
//! 4. **Scalarize and update** — evaluate offspring using a scalarization
//!    function and update neighboring subproblems if the offspring improves
//!    their scalar value.
//! 5. **Update ideal point** — track the best value seen per objective.
//!
//! # Scalarization methods
//!
//! | Method | Formula | Best for |
//! |--------|---------|----------|
//! | [`Tchebycheff`](Decomposition::Tchebycheff) (default) | `max(wᵢ * \|fᵢ - zᵢ*\|)` | General purpose, handles non-convex fronts |
//! | [`WeightedSum`](Decomposition::WeightedSum) | `Σ(wᵢ * fᵢ)` | Convex Pareto fronts only |
//! | [`Pbi`](Decomposition::Pbi) | `d₁ + θ * d₂` | Fine-grained convergence/diversity control |
//!
//! # When to use
//!
//! - Problems where you want **evenly distributed** solutions along the
//!   Pareto front (one solution per weight direction).
//! - Many-objective optimization (3+ objectives) — scales well because
//!   each subproblem is a simple scalar optimization.
//! - Problems with **non-convex** Pareto fronts (use Tchebycheff or PBI).
//! - When you need explicit control over the trade-off distribution via
//!   weight vectors.
//!
//! For Pareto-based approaches, see
//! [`Nsga2Sampler`](super::nsga2::Nsga2Sampler) (crowding distance) or
//! [`Nsga3Sampler`](super::nsga3::Nsga3Sampler) (reference-point niching).
//!
//! # Configuration
//!
//! | Parameter | Builder method | Default |
//! |-----------|---------------|---------|
//! | Population size | [`population_size`](MoeadSamplerBuilder::population_size) | Number of Das-Dennis weight vectors |
//! | Neighborhood size (T) | [`neighborhood_size`](MoeadSamplerBuilder::neighborhood_size) | `min(20, pop_size)` |
//! | Decomposition method | [`decomposition`](MoeadSamplerBuilder::decomposition) | Tchebycheff |
//! | Crossover probability | [`crossover_prob`](MoeadSamplerBuilder::crossover_prob) | 1.0 |
//! | SBX distribution index | [`crossover_eta`](MoeadSamplerBuilder::crossover_eta) | 20.0 |
//! | Mutation distribution index | [`mutation_eta`](MoeadSamplerBuilder::mutation_eta) | 20.0 |
//! | Random seed | [`seed`](MoeadSamplerBuilder::seed) | random |
//!
//! # Examples
//!
//! ```
//! use optimizer::Direction;
//! use optimizer::multi_objective::MultiObjectiveStudy;
//! use optimizer::parameter::{FloatParam, Parameter};
//! use optimizer::sampler::moead::MoeadSampler;
//!
//! let sampler = MoeadSampler::with_seed(42);
//! let study =
//!     MultiObjectiveStudy::with_sampler(vec![Direction::Minimize, Direction::Minimize], sampler);
//!
//! let x = FloatParam::new(0.0, 1.0);
//! study
//!     .optimize(100, |trial: &mut optimizer::Trial| {
//!         let xv = x.suggest(trial)?;
//!         Ok::<_, optimizer::Error>(vec![xv, 1.0 - xv])
//!     })
//!     .unwrap();
//! ```

use parking_lot::Mutex;

use super::genetic::{
    self, Candidate, EvolutionaryState, Phase, advance_generation, auto_divisions,
    collect_evaluated_generation, crossover, das_dennis, extract_trial_params,
    generate_random_candidates, mutate, sample_from_candidate, sample_random,
};
use crate::distribution::Distribution;
use crate::multi_objective::MultiObjectiveTrial;
use crate::param::ParamValue;
use crate::types::Direction;

/// Decomposition (scalarization) method for [`MoeadSampler`].
///
/// Control how multi-objective values are reduced to a single scalar
/// for each subproblem. The default is [`Tchebycheff`](Self::Tchebycheff),
/// which handles both convex and non-convex Pareto fronts.
#[derive(Debug, Clone, Default)]
pub enum Decomposition {
    /// Weighted sum: `Σ(wᵢ * fᵢ)`.
    ///
    /// Simplest method but can only find solutions on convex regions
    /// of the Pareto front.
    WeightedSum,
    /// Tchebycheff: `max(wᵢ * |fᵢ - zᵢ*|)`.
    ///
    /// Handles non-convex Pareto fronts. The most commonly used
    /// decomposition method (default).
    #[default]
    Tchebycheff,
    /// Penalty-based Boundary Intersection: `d₁ + θ * d₂`.
    ///
    /// Provides fine-grained control over the convergence/diversity
    /// balance via the penalty parameter `theta`. Higher `theta`
    /// favors solutions closer to the weight direction.
    Pbi {
        /// Penalty parameter controlling the balance between convergence
        /// and diversity. Default: 5.0.
        theta: f64,
    },
}

/// MOEA/D sampler for multi-objective optimization.
///
/// Decompose a multi-objective problem into scalar subproblems using
/// weight vectors and solve them collaboratively via neighborhood-based
/// mating. Supports [`Tchebycheff`](Decomposition::Tchebycheff),
/// [`WeightedSum`](Decomposition::WeightedSum), and
/// [`Pbi`](Decomposition::Pbi) scalarization.
///
/// Create with [`MoeadSampler::new`], [`MoeadSampler::with_seed`], or
/// [`MoeadSampler::builder`] for full configuration.
pub struct MoeadSampler {
    state: Mutex<MoeadState>,
}

impl MoeadSampler {
    /// Creates a new MOEA/D sampler with a random seed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(MoeadState::new(MoeadConfig::default(), None)),
        }
    }

    /// Creates a new MOEA/D sampler with a fixed seed.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            state: Mutex::new(MoeadState::new(MoeadConfig::default(), Some(seed))),
        }
    }

    /// Creates a builder for configuring a `MoeadSampler`.
    #[must_use]
    pub fn builder() -> MoeadSamplerBuilder {
        MoeadSamplerBuilder::default()
    }
}

impl Default for MoeadSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`MoeadSampler`].
#[derive(Debug, Clone, Default)]
pub struct MoeadSamplerBuilder {
    population_size: Option<usize>,
    neighborhood_size: Option<usize>,
    decomposition: Decomposition,
    crossover_prob: Option<f64>,
    crossover_eta: Option<f64>,
    mutation_eta: Option<f64>,
    seed: Option<u64>,
}

impl MoeadSamplerBuilder {
    /// Sets the population size. If unset, equals the number of
    /// Das-Dennis weight vectors.
    #[must_use]
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = Some(size);
        self
    }

    /// Sets the neighborhood size (T). Default: `min(20, pop_size)`.
    #[must_use]
    pub fn neighborhood_size(mut self, size: usize) -> Self {
        self.neighborhood_size = Some(size);
        self
    }

    /// Sets the decomposition method. Default: Tchebycheff.
    #[must_use]
    pub fn decomposition(mut self, decomp: Decomposition) -> Self {
        self.decomposition = decomp;
        self
    }

    /// Sets the crossover probability. Default: 1.0.
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

    /// Builds the configured [`MoeadSampler`].
    #[must_use]
    pub fn build(self) -> MoeadSampler {
        let config = MoeadConfig {
            user_population_size: self.population_size,
            neighborhood_size: self.neighborhood_size,
            decomposition: self.decomposition,
            crossover_prob: self.crossover_prob.unwrap_or(1.0),
            crossover_eta: self.crossover_eta.unwrap_or(20.0),
            mutation_eta: self.mutation_eta.unwrap_or(20.0),
        };
        MoeadSampler {
            state: Mutex::new(MoeadState::new(config, self.seed)),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct MoeadConfig {
    user_population_size: Option<usize>,
    neighborhood_size: Option<usize>,
    decomposition: Decomposition,
    crossover_prob: f64,
    crossover_eta: f64,
    mutation_eta: f64,
}

impl Default for MoeadConfig {
    fn default() -> Self {
        Self {
            user_population_size: None,
            neighborhood_size: None,
            decomposition: Decomposition::default(),
            crossover_prob: 1.0,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
        }
    }
}

struct MoeadState {
    evo: EvolutionaryState,
    config: MoeadConfig,
    /// Weight vectors (Das-Dennis), one per subproblem.
    weight_vectors: Vec<Vec<f64>>,
    /// Neighborhoods: for each subproblem, indices of T nearest weight vectors.
    neighborhoods: Vec<Vec<usize>>,
    /// Ideal point z* (best per-objective in minimize-space).
    ideal_point: Vec<f64>,
    /// Current population's objective values in minimize-space (one per subproblem).
    population_values: Vec<Vec<f64>>,
    /// Current population's parameter vectors (one per subproblem).
    population_params: Vec<Vec<ParamValue>>,
    /// Whether the MOEA/D state has been initialized.
    initialized: bool,
}

impl MoeadState {
    fn new(config: MoeadConfig, seed: Option<u64>) -> Self {
        Self {
            evo: EvolutionaryState::new(seed),
            config,
            weight_vectors: Vec::new(),
            neighborhoods: Vec::new(),
            ideal_point: Vec::new(),
            population_values: Vec::new(),
            population_params: Vec::new(),
            initialized: false,
        }
    }
}

// ---------------------------------------------------------------------------
// MultiObjectiveSampler implementation
// ---------------------------------------------------------------------------

impl crate::multi_objective::MultiObjectiveSampler for MoeadSampler {
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
                initialize_moead(&mut state, directions);
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

/// Initialize MOEA/D: weight vectors, neighborhoods, ideal point.
fn initialize_moead(state: &mut MoeadState, directions: &[Direction]) {
    let n_obj = directions.len();

    // Generate weight vectors
    let divisions = auto_divisions(n_obj, state.config.user_population_size.unwrap_or(100));
    state.weight_vectors = das_dennis(n_obj, divisions);

    let pop_size = state
        .config
        .user_population_size
        .unwrap_or(state.weight_vectors.len())
        .max(4);

    // Trim or pad weight vectors to match population size
    state.weight_vectors.truncate(pop_size);
    while state.weight_vectors.len() < pop_size {
        // Duplicate random existing weight vectors
        let idx = state.evo.rng.usize(0..state.weight_vectors.len());
        let w = state.weight_vectors[idx].clone();
        state.weight_vectors.push(w);
    }

    // Compute neighborhoods
    let t = state
        .config
        .neighborhood_size
        .unwrap_or_else(|| 20.min(pop_size));
    let t = t.min(pop_size);
    state.neighborhoods = compute_neighborhoods(&state.weight_vectors, t);

    state.evo.population_size = pop_size;
    state.evo.phase = Phase::Active;
    state.ideal_point = vec![f64::INFINITY; n_obj];
    state.initialized = true;
}

/// Compute T-nearest neighborhoods by Euclidean distance between weight vectors.
fn compute_neighborhoods(weights: &[Vec<f64>], t: usize) -> Vec<Vec<usize>> {
    let n = weights.len();
    weights
        .iter()
        .map(|wi| {
            let mut distances: Vec<(usize, f64)> = (0..n)
                .map(|j| {
                    let d: f64 = wi
                        .iter()
                        .zip(&weights[j])
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    (j, d)
                })
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
            distances.into_iter().take(t).map(|(idx, _)| idx).collect()
        })
        .collect()
}

/// Convert values to minimize-space.
fn to_minimize_space(values: &[f64], directions: &[Direction]) -> Vec<f64> {
    values
        .iter()
        .zip(directions)
        .map(|(&v, d)| match d {
            Direction::Minimize => v,
            Direction::Maximize => -v,
        })
        .collect()
}

fn maybe_generate_new_generation(
    state: &mut MoeadState,
    history: &[MultiObjectiveTrial],
    directions: &[Direction],
) {
    if state.evo.candidates.is_empty() {
        generate_random_candidates(&mut state.evo);
        return;
    }

    if let Some(evaluated) = collect_evaluated_generation(&state.evo, history) {
        let offspring = moead_generate_offspring(state, &evaluated, directions);
        advance_generation(&mut state.evo, offspring);
    }
}

// ---------------------------------------------------------------------------
// Scalarization functions
// ---------------------------------------------------------------------------

/// Weighted sum scalarization: `sum(w_i * f_i)`.
fn scalarize_weighted_sum(values: &[f64], weight: &[f64]) -> f64 {
    values.iter().zip(weight).map(|(&v, &w)| w * v).sum()
}

/// Tchebycheff scalarization: `max(w_i * |f_i - z_i*|)`.
fn scalarize_tchebycheff(values: &[f64], weight: &[f64], ideal: &[f64]) -> f64 {
    values
        .iter()
        .zip(weight)
        .zip(ideal)
        .map(|((&v, &w), &z)| {
            let w = if w < 1e-6 { 1e-6 } else { w };
            w * (v - z).abs()
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

/// PBI scalarization: `d1 + theta * d2`.
///
/// d1 = projection onto weight direction, d2 = perpendicular distance.
fn scalarize_pbi(values: &[f64], weight: &[f64], ideal: &[f64], theta: f64) -> f64 {
    let n = values.len();

    // Direction from ideal to the point
    let diff: Vec<f64> = values.iter().zip(ideal).map(|(&v, &z)| v - z).collect();

    // Normalize weight vector
    let w_norm: f64 = weight.iter().map(|&w| w * w).sum::<f64>().sqrt();
    if w_norm < 1e-30 {
        return f64::INFINITY;
    }
    let w_unit: Vec<f64> = weight.iter().map(|&w| w / w_norm).collect();

    // d1 = projection of diff onto weight direction
    let d1: f64 = diff.iter().zip(&w_unit).map(|(&d, &w)| d * w).sum();

    // d2 = perpendicular distance
    let d2_sq: f64 = (0..n)
        .map(|i| {
            let proj = d1 * w_unit[i];
            (diff[i] - proj).powi(2)
        })
        .sum::<f64>();

    d1 + theta * d2_sq.sqrt()
}

/// Evaluate scalarization for a given decomposition method.
fn scalarize(values: &[f64], weight: &[f64], ideal: &[f64], decomposition: &Decomposition) -> f64 {
    match decomposition {
        Decomposition::WeightedSum => scalarize_weighted_sum(values, weight),
        Decomposition::Tchebycheff => scalarize_tchebycheff(values, weight, ideal),
        Decomposition::Pbi { theta } => scalarize_pbi(values, weight, ideal, *theta),
    }
}

// ---------------------------------------------------------------------------
// MOEA/D generation algorithm
// ---------------------------------------------------------------------------

fn moead_generate_offspring(
    state: &mut MoeadState,
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

    // Extract current population parameters and objective values
    let current_params: Vec<Vec<ParamValue>> = population
        .iter()
        .map(|t| extract_trial_params(t, &state.evo.dimensions, &mut state.evo.rng))
        .collect();

    let current_values: Vec<Vec<f64>> = population
        .iter()
        .map(|t| to_minimize_space(&t.values, directions))
        .collect();

    // Update ideal point
    for vals in &current_values {
        for (i, &v) in vals.iter().enumerate() {
            if i < state.ideal_point.len() && v < state.ideal_point[i] {
                state.ideal_point[i] = v;
            }
        }
    }

    // Assign each solution to its best subproblem via scalarization
    // and select the best solution for each subproblem as its representative
    let n_weights = state.weight_vectors.len();
    let mut best_for_subproblem: Vec<usize> = Vec::with_capacity(n_weights);

    for j in 0..n_weights {
        let mut best_idx = 0;
        let mut best_val = f64::INFINITY;
        for (k, vals) in current_values.iter().enumerate() {
            let s = scalarize(
                vals,
                &state.weight_vectors[j],
                &state.ideal_point,
                &state.config.decomposition,
            );
            if s < best_val {
                best_val = s;
                best_idx = k;
            }
        }
        best_for_subproblem.push(best_idx);
    }

    // Store current population state
    state.population_values = current_values;
    state.population_params = current_params;

    // Generate offspring: for each subproblem, mate from neighborhood
    let mut offspring = Vec::with_capacity(pop_size);

    for i in 0..pop_size.min(state.neighborhoods.len()) {
        let neighborhood = &state.neighborhoods[i];

        // Pick two parents from the neighborhood using subproblem assignments
        let n1 = neighborhood[state.evo.rng.usize(0..neighborhood.len())];
        let n2 = neighborhood[state.evo.rng.usize(0..neighborhood.len())];

        let p1_idx = best_for_subproblem[n1 % best_for_subproblem.len()];
        let p2_idx = best_for_subproblem[n2 % best_for_subproblem.len()];

        let p1 = &state.population_params[p1_idx];
        let p2 = &state.population_params[p2_idx];

        let (mut child1, _child2) = crossover(
            &mut state.evo.rng,
            p1,
            p2,
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

        offspring.push(Candidate { params: child1 });
    }

    // If pop_size > neighborhoods, fill remaining with random neighborhood crossover
    while offspring.len() < pop_size {
        let i = state.evo.rng.usize(0..state.neighborhoods.len());
        let neighborhood = &state.neighborhoods[i];
        let n1 = neighborhood[state.evo.rng.usize(0..neighborhood.len())];
        let n2 = neighborhood[state.evo.rng.usize(0..neighborhood.len())];

        let p1_idx = best_for_subproblem[n1 % best_for_subproblem.len()];
        let p2_idx = best_for_subproblem[n2 % best_for_subproblem.len()];

        let (mut child1, _) = crossover(
            &mut state.evo.rng,
            &state.population_params[p1_idx],
            &state.population_params[p2_idx],
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

        offspring.push(Candidate { params: child1 });
    }

    offspring
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalarize_weighted_sum() {
        let values = [1.0, 2.0, 3.0];
        let weight = [0.5, 0.3, 0.2];
        let result = scalarize_weighted_sum(&values, &weight);
        assert!((result - (0.5 + 0.6 + 0.6)).abs() < 1e-10);
    }

    #[test]
    fn test_scalarize_tchebycheff() {
        let values = [3.0, 2.0];
        let weight = [0.5, 0.5];
        let ideal = [1.0, 1.0];
        let result = scalarize_tchebycheff(&values, &weight, &ideal);
        // max(0.5 * |3-1|, 0.5 * |2-1|) = max(1.0, 0.5) = 1.0
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_scalarize_pbi() {
        let values = [2.0, 2.0];
        let weight = [1.0, 1.0];
        let ideal = [0.0, 0.0];
        let result = scalarize_pbi(&values, &weight, &ideal, 5.0);
        // d1 = projection of (2,2) onto (1/√2, 1/√2) = 2*√2
        // d2 = 0 (point is on the weight direction)
        let expected_d1 = 2.0 * (2.0_f64).sqrt();
        assert!((result - expected_d1).abs() < 1e-10);
    }

    #[test]
    fn test_compute_neighborhoods() {
        let weights = vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]];
        let neighborhoods = compute_neighborhoods(&weights, 2);
        assert_eq!(neighborhoods.len(), 3);
        // Each neighborhood should have 2 entries
        for n in &neighborhoods {
            assert_eq!(n.len(), 2);
        }
        // First weight [1,0] should be closest to itself and [0.5,0.5]
        assert_eq!(neighborhoods[0][0], 0); // itself
        assert_eq!(neighborhoods[0][1], 1); // nearest neighbor
    }
}
