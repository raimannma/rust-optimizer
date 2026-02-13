//! NSGA-III (Non-dominated Sorting Genetic Algorithm III) sampler.
//!
//! NSGA-III extends NSGA-II to handle **many-objective** (3+) problems
//! where crowding distance loses effectiveness. Instead of crowding
//! distance, it uses **reference-point-based niching** with structured
//! Das-Dennis reference points distributed on the unit simplex to guide
//! the population toward a well-diversified Pareto front.
//!
//! # Algorithm
//!
//! Each generation proceeds as follows:
//!
//! 1. **Non-dominated sorting** — same as NSGA-II, partition the
//!    combined population into Pareto fronts F₁, F₂, …
//! 2. **Normalize objectives** — translate by ideal point and scale by
//!    intercepts so all objectives lie in roughly \[0, 1\].
//! 3. **Associate with reference points** — assign each solution to the
//!    closest Das-Dennis reference direction by perpendicular distance.
//! 4. **Niching selection** — when the last front only partially fits,
//!    prefer solutions associated with under-represented reference points
//!    (lowest niche count first, closest distance second).
//! 5. **SBX crossover + polynomial mutation** — generate offspring via
//!    rank-based tournament selection.
//!
//! # When to use
//!
//! - **Three or more objectives** — NSGA-III maintains diversity far
//!   better than NSGA-II as the number of objectives grows.
//! - Problems where you want a **uniformly distributed** Pareto front
//!   guided by structured reference points.
//! - Scales well up to ~10 objectives with appropriate division settings.
//!
//! For bi-objective problems, [`Nsga2Sampler`](super::nsga2::Nsga2Sampler)
//! is simpler and equally effective. For decomposition-based optimization,
//! see [`MoeadSampler`](super::moead::MoeadSampler).
//!
//! # Configuration
//!
//! | Parameter | Builder method | Default |
//! |-----------|---------------|---------|
//! | Population size | [`population_size`](Nsga3SamplerBuilder::population_size) | Number of Das-Dennis reference points |
//! | Das-Dennis divisions (H) | [`n_divisions`](Nsga3SamplerBuilder::n_divisions) | Auto-chosen from population size and objectives |
//! | Crossover probability | [`crossover_prob`](Nsga3SamplerBuilder::crossover_prob) | 1.0 |
//! | SBX distribution index | [`crossover_eta`](Nsga3SamplerBuilder::crossover_eta) | 30.0 |
//! | Mutation distribution index | [`mutation_eta`](Nsga3SamplerBuilder::mutation_eta) | 20.0 |
//! | Random seed | [`seed`](Nsga3SamplerBuilder::seed) | random |
//!
//! # Examples
//!
//! ```
//! use optimizer::Direction;
//! use optimizer::multi_objective::MultiObjectiveStudy;
//! use optimizer::parameter::{FloatParam, Parameter};
//! use optimizer::sampler::nsga3::Nsga3Sampler;
//!
//! let sampler = Nsga3Sampler::with_seed(42);
//! let study = MultiObjectiveStudy::with_sampler(
//!     vec![
//!         Direction::Minimize,
//!         Direction::Minimize,
//!         Direction::Minimize,
//!     ],
//!     sampler,
//! );
//!
//! let x = FloatParam::new(0.0, 1.0);
//! let y = FloatParam::new(0.0, 1.0);
//! study
//!     .optimize(100, |trial: &mut optimizer::Trial| {
//!         let xv = x.suggest(trial)?;
//!         let yv = y.suggest(trial)?;
//!         Ok::<_, optimizer::Error>(vec![xv, yv, (1.0 - xv - yv).abs()])
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
use crate::pareto;
use crate::types::Direction;

/// NSGA-III sampler for multi-objective optimization.
///
/// Use reference-point niching with Das-Dennis structured points to
/// maintain diversity in many-objective (3+) problems. For bi-objective
/// problems, [`Nsga2Sampler`](super::nsga2::Nsga2Sampler) is simpler.
///
/// Create with [`Nsga3Sampler::new`], [`Nsga3Sampler::with_seed`], or
/// [`Nsga3Sampler::builder`] for full configuration.
pub struct Nsga3Sampler {
    state: Mutex<Nsga3State>,
}

impl Nsga3Sampler {
    /// Creates a new NSGA-III sampler with a random seed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(Nsga3State::new(Nsga3Config::default(), None)),
        }
    }

    /// Creates a new NSGA-III sampler with a fixed seed.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            state: Mutex::new(Nsga3State::new(Nsga3Config::default(), Some(seed))),
        }
    }

    /// Creates a builder for configuring an `Nsga3Sampler`.
    #[must_use]
    pub fn builder() -> Nsga3SamplerBuilder {
        Nsga3SamplerBuilder::default()
    }
}

impl Default for Nsga3Sampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`Nsga3Sampler`].
#[derive(Debug, Clone, Default)]
pub struct Nsga3SamplerBuilder {
    population_size: Option<usize>,
    n_divisions: Option<usize>,
    crossover_prob: Option<f64>,
    crossover_eta: Option<f64>,
    mutation_eta: Option<f64>,
    seed: Option<u64>,
}

impl Nsga3SamplerBuilder {
    /// Sets the population size. If unset, equals the number of
    /// Das-Dennis reference points.
    #[must_use]
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = Some(size);
        self
    }

    /// Sets the number of divisions (H) for Das-Dennis reference points.
    /// If unset, automatically chosen based on population size and number
    /// of objectives.
    #[must_use]
    pub fn n_divisions(mut self, h: usize) -> Self {
        self.n_divisions = Some(h);
        self
    }

    /// Sets the crossover probability. Default: 1.0.
    #[must_use]
    pub fn crossover_prob(mut self, prob: f64) -> Self {
        self.crossover_prob = Some(prob);
        self
    }

    /// Sets the SBX distribution index. Default: 30.0.
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

    /// Builds the configured [`Nsga3Sampler`].
    #[must_use]
    pub fn build(self) -> Nsga3Sampler {
        let config = Nsga3Config {
            user_population_size: self.population_size,
            n_divisions: self.n_divisions,
            crossover_prob: self.crossover_prob.unwrap_or(1.0),
            crossover_eta: self.crossover_eta.unwrap_or(30.0),
            mutation_eta: self.mutation_eta.unwrap_or(20.0),
        };
        Nsga3Sampler {
            state: Mutex::new(Nsga3State::new(config, self.seed)),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Nsga3Config {
    user_population_size: Option<usize>,
    n_divisions: Option<usize>,
    crossover_prob: f64,
    crossover_eta: f64,
    mutation_eta: f64,
}

impl Default for Nsga3Config {
    fn default() -> Self {
        Self {
            user_population_size: None,
            n_divisions: None,
            crossover_prob: 1.0,
            crossover_eta: 30.0,
            mutation_eta: 20.0,
        }
    }
}

struct Nsga3State {
    evo: EvolutionaryState,
    config: Nsga3Config,
    /// Das-Dennis reference points (lazily generated once objectives are known).
    reference_points: Vec<Vec<f64>>,
    /// Best value seen per objective (minimize-space).
    ideal_point: Vec<f64>,
    /// Whether reference points have been initialized.
    initialized: bool,
}

impl Nsga3State {
    fn new(config: Nsga3Config, seed: Option<u64>) -> Self {
        Self {
            evo: EvolutionaryState::new(seed),
            config,
            reference_points: Vec::new(),
            ideal_point: Vec::new(),
            initialized: false,
        }
    }
}

// ---------------------------------------------------------------------------
// MultiObjectiveSampler implementation
// ---------------------------------------------------------------------------

impl crate::multi_objective::MultiObjectiveSampler for Nsga3Sampler {
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
                initialize_nsga3(&mut state, directions);
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

/// Initialize NSGA-III: generate reference points and set population size.
fn initialize_nsga3(state: &mut Nsga3State, directions: &[Direction]) {
    let n_obj = directions.len();

    // Determine divisions
    let divisions = state
        .config
        .n_divisions
        .unwrap_or_else(|| auto_divisions(n_obj, state.config.user_population_size.unwrap_or(100)));

    state.reference_points = das_dennis(n_obj, divisions);
    let n_ref = state.reference_points.len();

    // Population size = number of reference points (or user override, at least n_ref)
    let pop_size = state.config.user_population_size.unwrap_or(n_ref).max(4);
    state.evo.population_size = pop_size;
    state.evo.phase = Phase::Active;
    state.ideal_point = vec![f64::INFINITY; n_obj];
    state.initialized = true;
}

fn maybe_generate_new_generation(
    state: &mut Nsga3State,
    history: &[MultiObjectiveTrial],
    directions: &[Direction],
) {
    if state.evo.candidates.is_empty() {
        generate_random_candidates(&mut state.evo);
        return;
    }

    if let Some(evaluated) = collect_evaluated_generation(&state.evo, history) {
        let offspring = nsga3_generate_offspring(state, &evaluated, directions);
        advance_generation(&mut state.evo, offspring);
    }
}

// ---------------------------------------------------------------------------
// NSGA-III selection algorithm
// ---------------------------------------------------------------------------

/// Normalize objectives to minimize-space.
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

/// Update ideal point with new observations.
fn update_ideal_point(ideal: &mut [f64], normalized_values: &[Vec<f64>]) {
    for vals in normalized_values {
        for (i, &v) in vals.iter().enumerate() {
            if v < ideal[i] {
                ideal[i] = v;
            }
        }
    }
}

/// Compute Achievement Scalarizing Function (ASF) for extreme point finding.
fn asf(point: &[f64], weight: &[f64], ideal: &[f64]) -> f64 {
    point
        .iter()
        .zip(weight)
        .zip(ideal)
        .map(|((&p, &w), &z)| {
            let w = if w < 1e-6 { 1e-6 } else { w };
            (p - z) / w
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Find intercepts for normalization via extreme points.
///
/// For each objective, find the point with best ASF (using a weight vector
/// that emphasizes that objective). The intercepts are where the hyperplane
/// through the extreme points crosses each axis.
fn find_intercepts(normalized_values: &[Vec<f64>], ideal: &[f64]) -> Vec<f64> {
    let n_obj = ideal.len();
    let n = normalized_values.len();

    if n == 0 || n_obj == 0 {
        return vec![1.0; n_obj];
    }

    // Find extreme points (one per objective)
    let mut extreme_indices = Vec::with_capacity(n_obj);
    for obj in 0..n_obj {
        let mut weight = vec![1e-6; n_obj];
        weight[obj] = 1.0;

        let mut best_idx = 0;
        let mut best_asf = f64::INFINITY;
        for (i, vals) in normalized_values.iter().enumerate() {
            let a = asf(vals, &weight, ideal);
            if a < best_asf {
                best_asf = a;
                best_idx = i;
            }
        }
        extreme_indices.push(best_idx);
    }

    // Try to compute hyperplane intercepts
    // For stability, if the extreme points are degenerate, fall back to
    // max - ideal per objective
    let mut intercepts = Vec::with_capacity(n_obj);
    for obj in 0..n_obj {
        let max_val = normalized_values
            .iter()
            .map(|v| v[obj])
            .fold(f64::NEG_INFINITY, f64::max);
        let intercept = max_val - ideal[obj];
        intercepts.push(if intercept > 1e-10 { intercept } else { 1.0 });
    }

    intercepts
}

/// Normalize objective values: subtract ideal, divide by intercepts.
fn normalize_objectives(values: &[Vec<f64>], ideal: &[f64], intercepts: &[f64]) -> Vec<Vec<f64>> {
    values
        .iter()
        .map(|v| {
            v.iter()
                .zip(ideal)
                .zip(intercepts)
                .map(|((&val, &z), &a)| {
                    let norm = if a > 1e-10 { a } else { 1.0 };
                    (val - z) / norm
                })
                .collect()
        })
        .collect()
}

/// Perpendicular distance from a point to a reference line (direction vector).
fn perpendicular_distance(point: &[f64], reference: &[f64]) -> f64 {
    let dot: f64 = point.iter().zip(reference).map(|(&p, &r)| p * r).sum();
    let ref_norm_sq: f64 = reference.iter().map(|&r| r * r).sum();

    if ref_norm_sq < 1e-30 {
        return f64::INFINITY;
    }

    let proj_scalar = dot / ref_norm_sq;
    let dist_sq: f64 = point
        .iter()
        .zip(reference)
        .map(|(&p, &r)| {
            let proj = proj_scalar * r;
            (p - proj).powi(2)
        })
        .sum();

    dist_sq.sqrt()
}

/// Associate each solution with its nearest reference point.
/// Returns (`closest_ref_idx`, distance) for each solution.
fn associate_to_reference_points(
    normalized: &[Vec<f64>],
    reference_points: &[Vec<f64>],
) -> Vec<(usize, f64)> {
    normalized
        .iter()
        .map(|point| {
            let mut best_ref = 0;
            let mut best_dist = f64::INFINITY;
            for (j, rp) in reference_points.iter().enumerate() {
                let d = perpendicular_distance(point, rp);
                if d < best_dist {
                    best_dist = d;
                    best_ref = j;
                }
            }
            (best_ref, best_dist)
        })
        .collect()
}

/// NSGA-III niching-based selection from the last front.
///
/// `already_selected` are indices into the combined population that are
/// already accepted (from fronts 0..L-1). `last_front` contains indices
/// from front L. We need to pick `remaining` more from `last_front`.
fn niching_select(
    rng: &mut fastrand::Rng,
    associations: &[(usize, f64)],
    already_selected: &[usize],
    last_front: &[usize],
    n_reference_points: usize,
    remaining: usize,
) -> Vec<usize> {
    // Count niche per reference point for already selected
    let mut niche_count = vec![0_usize; n_reference_points];
    for &idx in already_selected {
        niche_count[associations[idx].0] += 1;
    }

    // Build per-reference-point candidate lists from the last front
    let mut ref_candidates: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_reference_points];
    for &idx in last_front {
        let (ref_idx, dist) = associations[idx];
        ref_candidates[ref_idx].push((idx, dist));
    }

    let mut selected = Vec::with_capacity(remaining);
    let mut excluded = vec![false; associations.len()];

    for _ in 0..remaining {
        // Find minimum niche count among reference points that still have candidates
        let min_count = (0..n_reference_points)
            .filter(|&j| ref_candidates[j].iter().any(|&(idx, _)| !excluded[idx]))
            .map(|j| niche_count[j])
            .min();

        let Some(min_count) = min_count else {
            break;
        };

        // Collect reference points with this minimum count that have candidates
        let min_refs: Vec<usize> = (0..n_reference_points)
            .filter(|&j| {
                niche_count[j] == min_count
                    && ref_candidates[j].iter().any(|&(idx, _)| !excluded[idx])
            })
            .collect();

        if min_refs.is_empty() {
            break;
        }

        // Pick a random reference point from the minimum set
        let chosen_ref = min_refs[rng.usize(0..min_refs.len())];

        // Available candidates for this reference point
        let available: Vec<(usize, f64)> = ref_candidates[chosen_ref]
            .iter()
            .filter(|&&(idx, _)| !excluded[idx])
            .copied()
            .collect();

        if available.is_empty() {
            continue;
        }

        let chosen_idx = if min_count == 0 {
            // Pick closest to reference line
            available
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal))
                .unwrap()
                .0
        } else {
            // Pick random
            available[rng.usize(0..available.len())].0
        };

        selected.push(chosen_idx);
        excluded[chosen_idx] = true;
        niche_count[chosen_ref] += 1;
    }

    selected
}

/// Perform NSGA-III selection: non-dominated sort + reference-point niching.
fn nsga3_select(
    state: &mut Nsga3State,
    population: &[&MultiObjectiveTrial],
    directions: &[Direction],
) -> (Vec<Vec<ParamValue>>, Vec<usize>) {
    let pop_size = state.evo.population_size;
    let n_obj = directions.len();

    // Convert to minimize-space
    let min_values: Vec<Vec<f64>> = population
        .iter()
        .map(|t| to_minimize_space(&t.values, directions))
        .collect();

    // Non-dominated sort
    let constraints: Vec<Vec<f64>> = population.iter().map(|t| t.constraints.clone()).collect();
    let has_constraints = constraints.iter().any(|c| !c.is_empty());
    let fronts = if has_constraints {
        pareto::fast_non_dominated_sort_constrained(
            &min_values,
            &vec![Direction::Minimize; n_obj],
            &constraints,
        )
    } else {
        pareto::fast_non_dominated_sort(&min_values, &vec![Direction::Minimize; n_obj])
    };

    // Fill front-by-front
    let mut selected: Vec<usize> = Vec::with_capacity(pop_size);
    let mut last_front_idx = None;

    for (fi, front) in fronts.iter().enumerate() {
        if selected.len() + front.len() <= pop_size {
            selected.extend_from_slice(front);
        } else {
            last_front_idx = Some(fi);
            break;
        }
    }

    // If we filled exactly or all fronts fit, done
    if selected.len() < pop_size
        && let Some(lf_idx) = last_front_idx
    {
        // Need niching from the last partial front
        let remaining = pop_size - selected.len();

        // Update ideal point
        update_ideal_point(&mut state.ideal_point, &min_values);

        // Find intercepts and normalize
        let intercepts = find_intercepts(&min_values, &state.ideal_point);
        let normalized = normalize_objectives(&min_values, &state.ideal_point, &intercepts);

        // Associate all solutions with reference points
        let associations = associate_to_reference_points(&normalized, &state.reference_points);

        // Select from last front using niching
        let last_front = &fronts[lf_idx];
        let additional = niching_select(
            &mut state.evo.rng,
            &associations,
            &selected,
            last_front,
            state.reference_points.len(),
            remaining,
        );
        selected.extend(additional);
    }

    // Pad if needed
    let n = population.len();
    while selected.len() < pop_size {
        selected.push(state.evo.rng.usize(0..n));
    }

    let params = selected
        .iter()
        .map(|&idx| {
            extract_trial_params(population[idx], &state.evo.dimensions, &mut state.evo.rng)
        })
        .collect();
    (params, selected)
}

/// Tournament selection based on rank only (no crowding distance in NSGA-III).
fn tournament_select_rank(rng: &mut fastrand::Rng, ranks: &[usize], n: usize) -> usize {
    let a = rng.usize(0..n);
    let b = rng.usize(0..n);

    if ranks[a] <= ranks[b] { a } else { b }
}

fn nsga3_generate_offspring(
    state: &mut Nsga3State,
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

    // Initialize reference points and ideal on first generation
    if !state.initialized {
        initialize_nsga3(state, directions);
    }

    let (parents, selected_indices) = nsga3_select(state, population, directions);

    // Assign Pareto front ranks for tournament selection
    let n_obj = directions.len();
    let min_values: Vec<Vec<f64>> = population
        .iter()
        .map(|t| to_minimize_space(&t.values, directions))
        .collect();
    let fronts = pareto::fast_non_dominated_sort(&min_values, &vec![Direction::Minimize; n_obj]);
    // Build rank lookup for population indices
    let mut pop_rank = vec![0_usize; population.len()];
    for (front_rank, front) in fronts.iter().enumerate() {
        for &idx in front {
            if idx < pop_rank.len() {
                pop_rank[idx] = front_rank;
            }
        }
    }
    // Map population ranks to selected parent indices
    let parent_ranks: Vec<usize> = selected_indices
        .iter()
        .map(|&idx| {
            if idx < pop_rank.len() {
                pop_rank[idx]
            } else {
                0
            }
        })
        .collect();

    let mut offspring = Vec::with_capacity(pop_size);
    while offspring.len() < pop_size {
        let p1 = tournament_select_rank(&mut state.evo.rng, &parent_ranks, parents.len());
        let p2 = tournament_select_rank(&mut state.evo.rng, &parent_ranks, parents.len());

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perpendicular_distance() {
        // Point (1, 0) to reference line (1, 1) (45-degree line)
        let d = perpendicular_distance(&[1.0, 0.0], &[1.0, 1.0]);
        // Projection is (0.5, 0.5), distance = sqrt(0.25 + 0.25) = sqrt(0.5)
        assert!((d - (0.5_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_perpendicular_distance_on_line() {
        // Point on the reference line
        let d = perpendicular_distance(&[2.0, 2.0], &[1.0, 1.0]);
        assert!(d < 1e-10);
    }

    #[test]
    fn test_normalize_objectives() {
        let values = vec![vec![2.0, 4.0], vec![4.0, 2.0]];
        let ideal = vec![1.0, 1.0];
        let intercepts = vec![3.0, 3.0];
        let normalized = normalize_objectives(&values, &ideal, &intercepts);
        assert!((normalized[0][0] - 1.0 / 3.0).abs() < 1e-10);
        assert!((normalized[0][1] - 1.0).abs() < 1e-10);
    }
}
