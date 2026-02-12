//! Shared types and genetic operators for evolutionary multi-objective samplers.
//!
//! This module extracts common functionality used by NSGA-II, NSGA-III, and MOEA/D:
//! candidate management, discovery/active phase logic, SBX crossover,
//! polynomial mutation, and Das-Dennis reference point generation.

use std::collections::HashMap;

use crate::distribution::Distribution;
use crate::multi_objective::MultiObjectiveTrial;
use crate::param::ParamValue;
use crate::rng_util;

/// Describes a parameter dimension discovered during the first trial.
#[derive(Clone, Debug)]
pub(crate) struct DimensionInfo {
    pub distribution: Distribution,
}

/// A candidate solution: one value per dimension.
#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub params: Vec<ParamValue>,
}

/// Tracks per-trial sampling progress (which candidate, which dimension next).
#[derive(Clone, Debug)]
pub(crate) struct TrialProgress {
    pub candidate_idx: usize,
    pub next_dim: usize,
}

/// Phase of an evolutionary sampler.
pub(crate) enum Phase {
    /// First trial reveals parameter dimensions.
    Discovery,
    /// Evolutionary optimisation.
    Active,
}

/// Common state shared by all evolutionary multi-objective samplers.
pub(crate) struct EvolutionaryState {
    pub rng: fastrand::Rng,
    pub phase: Phase,
    pub dimensions: Vec<DimensionInfo>,
    pub population_size: usize,
    pub candidates: Vec<Candidate>,
    pub trial_progress: HashMap<u64, TrialProgress>,
    pub assigned_count: usize,
    pub generation_trial_ids: Vec<u64>,
    pub discovery_trial_id: Option<u64>,
    pub generation: usize,
}

impl EvolutionaryState {
    pub(crate) fn new(seed: Option<u64>) -> Self {
        let rng = seed.map_or_else(fastrand::Rng::new, fastrand::Rng::with_seed);
        Self {
            rng,
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
// Discovery phase helpers
// ---------------------------------------------------------------------------

/// Handle sampling during the discovery phase.
///
/// Returns `Some(value)` if the discovery phase handled the sample,
/// or `None` if it transitioned to active phase and the caller should
/// generate candidates and sample from them.
pub(crate) fn sample_discovery(
    evo: &mut EvolutionaryState,
    distribution: &Distribution,
    trial_id: u64,
) -> Option<ParamValue> {
    if let Some(prev_id) = evo.discovery_trial_id
        && trial_id != prev_id
    {
        // A new trial arrived — transition to active phase
        return None;
    }

    evo.discovery_trial_id = Some(trial_id);
    evo.dimensions.push(DimensionInfo {
        distribution: distribution.clone(),
    });

    Some(sample_random(&mut evo.rng, distribution))
}

/// Compute population size from dimensions and optional user override.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub(crate) fn compute_population_size(
    n_dims: usize,
    user_pop_size: Option<usize>,
    minimum: usize,
) -> usize {
    user_pop_size
        .unwrap_or_else(|| (4.0 + 3.0 * (n_dims as f64).ln().max(0.0)).floor() as usize)
        .max(minimum)
}

/// Transition from discovery to active phase.
pub(crate) fn finalize_discovery(evo: &mut EvolutionaryState, user_pop_size: Option<usize>) {
    evo.population_size = compute_population_size(evo.dimensions.len(), user_pop_size, 4);
    evo.phase = Phase::Active;
}

/// Generate `population_size` random candidates.
pub(crate) fn generate_random_candidates(evo: &mut EvolutionaryState) {
    let pop = evo.population_size;
    evo.candidates = (0..pop)
        .map(|_| {
            let params: Vec<ParamValue> = evo
                .dimensions
                .iter()
                .map(|d| sample_random(&mut evo.rng, &d.distribution))
                .collect();
            Candidate { params }
        })
        .collect();
    evo.assigned_count = 0;
    evo.generation_trial_ids.clear();
    evo.trial_progress.clear();
}

/// Assign a candidate to a trial and return the next dimension value.
pub(crate) fn sample_from_candidate(evo: &mut EvolutionaryState, trial_id: u64) -> ParamValue {
    if !evo.trial_progress.contains_key(&trial_id) {
        let candidate_idx = if evo.assigned_count < evo.candidates.len() {
            let idx = evo.assigned_count;
            evo.assigned_count += 1;
            idx
        } else {
            // Overflow: generate a random candidate
            let params: Vec<ParamValue> = evo
                .dimensions
                .iter()
                .map(|d| sample_random(&mut evo.rng, &d.distribution))
                .collect();
            evo.candidates.push(Candidate { params });
            let idx = evo.candidates.len() - 1;
            evo.assigned_count = evo.candidates.len();
            idx
        };

        evo.trial_progress.insert(
            trial_id,
            TrialProgress {
                candidate_idx,
                next_dim: 0,
            },
        );
        evo.generation_trial_ids.push(trial_id);
    }

    let progress = evo.trial_progress.get_mut(&trial_id).unwrap();
    let dim_idx = progress.next_dim;
    progress.next_dim += 1;

    if dim_idx >= evo.dimensions.len() {
        return sample_random(&mut evo.rng, &evo.dimensions.last().unwrap().distribution);
    }

    evo.candidates[progress.candidate_idx].params[dim_idx].clone()
}

/// Extract parameter values from a trial, ordered by dimension index.
pub(crate) fn extract_trial_params(
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

/// Install new offspring as the next generation's candidates.
pub(crate) fn advance_generation(evo: &mut EvolutionaryState, offspring: Vec<Candidate>) {
    evo.candidates = offspring;
    evo.assigned_count = 0;
    evo.generation_trial_ids.clear();
    evo.trial_progress.clear();
    evo.generation += 1;
}

/// Check if the current generation is fully evaluated and return the
/// evaluated trials if so.
pub(crate) fn collect_evaluated_generation<'a>(
    evo: &EvolutionaryState,
    history: &'a [MultiObjectiveTrial],
) -> Option<Vec<&'a MultiObjectiveTrial>> {
    let pop_size = evo.population_size;

    if evo.generation_trial_ids.len() < pop_size {
        return None;
    }

    let gen_ids: Vec<u64> = evo
        .generation_trial_ids
        .iter()
        .take(pop_size)
        .copied()
        .collect();
    let history_map: HashMap<u64, &MultiObjectiveTrial> =
        history.iter().map(|t| (t.id, t)).collect();

    if !gen_ids.iter().all(|id| history_map.contains_key(id)) {
        return None;
    }

    Some(
        gen_ids
            .iter()
            .filter_map(|id| history_map.get(id).copied())
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Genetic operators
// ---------------------------------------------------------------------------

/// SBX crossover for continuous params, uniform crossover for categorical.
pub(crate) fn crossover(
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
pub(crate) fn sbx_crossover_f64(
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
pub(crate) fn mutate(
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
pub(crate) fn polynomial_mutation_f64(
    rng: &mut fastrand::Rng,
    x: f64,
    low: f64,
    high: f64,
    eta: f64,
) -> f64 {
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

pub(crate) use super::common::sample_random;

// ---------------------------------------------------------------------------
// Das-Dennis reference point generation
// ---------------------------------------------------------------------------

/// Generate Das-Dennis (simplex-lattice) reference points.
///
/// Returns `C(H + M - 1, M - 1)` uniformly spaced points on the
/// `M`-dimensional unit simplex, where `M = n_objectives` and
/// `H = divisions`.
pub(crate) fn das_dennis(n_objectives: usize, divisions: usize) -> Vec<Vec<f64>> {
    let mut points = Vec::new();
    let mut point = vec![0.0_f64; n_objectives];
    das_dennis_recursive(
        n_objectives,
        divisions,
        0,
        divisions,
        &mut point,
        &mut points,
    );
    points
}

#[allow(clippy::cast_precision_loss)]
fn das_dennis_recursive(
    n_objectives: usize,
    divisions: usize,
    depth: usize,
    remaining: usize,
    current: &mut Vec<f64>,
    result: &mut Vec<Vec<f64>>,
) {
    if depth == n_objectives - 1 {
        current[depth] = remaining as f64 / divisions as f64;
        result.push(current.clone());
        return;
    }

    for i in 0..=remaining {
        current[depth] = i as f64 / divisions as f64;
        das_dennis_recursive(
            n_objectives,
            divisions,
            depth + 1,
            remaining - i,
            current,
            result,
        );
    }
}

/// Choose the number of divisions for Das-Dennis to get close to a target
/// population size.
///
/// The number of reference points is `C(H + M - 1, M - 1)`. This function
/// finds the smallest `H` such that the number of points >= `target_pop`.
pub(crate) fn auto_divisions(n_objectives: usize, target_pop: usize) -> usize {
    let m = n_objectives;
    for h in 1..200 {
        let n_points = n_combinations(h + m - 1, m - 1);
        if n_points >= target_pop {
            return h;
        }
    }
    12
}

/// Compute `C(n, k)` = n! / (k! * (n-k)!).
fn n_combinations(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result: usize = 1;
    for i in 0..k {
        result = result.saturating_mul(n - i) / (i + 1);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_das_dennis_2d() {
        let points = das_dennis(2, 4);
        // C(4+1, 1) = 5 points
        assert_eq!(points.len(), 5);
        for p in &points {
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "point {p:?} doesn't sum to 1");
        }
    }

    #[test]
    fn test_das_dennis_3d() {
        let points = das_dennis(3, 4);
        // C(4+2, 2) = 15 points
        assert_eq!(points.len(), 15);
        for p in &points {
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_auto_divisions() {
        // For 2 objectives targeting 10 points: H=9 gives C(10,1)=10
        let h = auto_divisions(2, 10);
        let n = n_combinations(h + 1, 1);
        assert!(n >= 10);

        // For 3 objectives targeting ~91 points: H=12 gives C(14,2)=91
        let h3 = auto_divisions(3, 91);
        let n3 = n_combinations(h3 + 2, 2);
        assert!(n3 >= 91);
    }

    #[test]
    fn test_n_combinations() {
        assert_eq!(n_combinations(5, 2), 10);
        assert_eq!(n_combinations(4, 0), 1);
        assert_eq!(n_combinations(4, 4), 1);
        assert_eq!(n_combinations(6, 3), 20);
    }
}
