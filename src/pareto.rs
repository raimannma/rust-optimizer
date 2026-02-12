//! Pareto front analysis utilities for multi-objective optimization.
//!
//! In multi-objective optimization there is generally no single best
//! solution. Instead, the goal is to find the **Pareto front** — the set
//! of solutions where no objective can be improved without worsening
//! another. This module provides tools for computing and analyzing Pareto
//! fronts.
//!
//! # Available functions
//!
//! | Function | Purpose |
//! |---|---|
//! | [`hypervolume`] | Measure the quality of a Pareto front (volume of dominated space) |
//! | [`non_dominated_sort`] | Rank solutions into successive fronts (front 0, 1, …) |
//! | [`pareto_front_indices`] | Filter to non-dominated (Pareto-optimal) solutions only |
//! | [`crowding_distance`] | Measure diversity/spread within a single front |
//!
//! # When to use
//!
//! - **Evaluating front quality**: Use [`hypervolume`] to compare two
//!   Pareto fronts — a higher hypervolume indicates a better-quality front.
//! - **Ranking all solutions**: Use [`non_dominated_sort`] to partition
//!   solutions into successive fronts, useful for selection in evolutionary
//!   algorithms.
//! - **Extracting the best solutions**: Use [`pareto_front_indices`] to get
//!   only the non-dominated set.
//! - **Diversity measurement**: Use [`crowding_distance`] to quantify how
//!   spread out solutions are within a front, which helps maintain diversity.
//!
//! Internally, this module also provides the fast non-dominated sorting
//! algorithm (Deb et al., 2002) used by
//! [`MultiObjectiveStudy::pareto_front()`](crate::multi_objective::MultiObjectiveStudy::pareto_front)
//! and [`Nsga2Sampler`](crate::sampler::Nsga2Sampler).
//!
//! # Example
//!
//! ```
//! use optimizer::Direction;
//! use optimizer::pareto::{
//!     crowding_distance, hypervolume, non_dominated_sort, pareto_front_indices,
//! };
//!
//! let solutions = vec![
//!     vec![1.0, 5.0], // Pareto-optimal
//!     vec![5.0, 1.0], // Pareto-optimal
//!     vec![3.0, 3.0], // Pareto-optimal
//!     vec![4.0, 4.0], // Dominated by (3, 3)
//! ];
//! let dirs = [Direction::Minimize, Direction::Minimize];
//!
//! // Non-dominated sorting: front 0 has indices {0, 1, 2}
//! let fronts = non_dominated_sort(&solutions, &dirs);
//! assert_eq!(fronts.len(), 2);
//!
//! // Pareto front indices (shortcut for fronts[0])
//! let mut front = pareto_front_indices(&solutions, &dirs);
//! front.sort();
//! assert_eq!(front, vec![0, 1, 2]);
//!
//! // Hypervolume with reference point (6, 6)
//! let front_values: Vec<_> = front.iter().map(|&i| solutions[i].clone()).collect();
//! let hv = hypervolume(&front_values, &[6.0, 6.0], &dirs);
//! assert!(hv > 0.0);
//!
//! // Crowding distance for diversity analysis
//! let cd = crowding_distance(&front_values, &dirs);
//! assert!(cd[0].is_infinite()); // boundary solution
//! ```

use crate::types::Direction;

/// Returns `true` if solution `a` Pareto-dominates solution `b`.
///
/// A solution dominates another if it is at least as good in all objectives
/// and strictly better in at least one, respecting the given directions.
#[allow(clippy::module_name_repetitions)]
pub(crate) fn dominates(a: &[f64], b: &[f64], directions: &[Direction]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), directions.len());

    let mut strictly_better = false;
    for ((&av, &bv), dir) in a.iter().zip(b.iter()).zip(directions.iter()) {
        let better = match dir {
            Direction::Minimize => av < bv,
            Direction::Maximize => av > bv,
        };
        let worse = match dir {
            Direction::Minimize => av > bv,
            Direction::Maximize => av < bv,
        };
        if worse {
            return false;
        }
        if better {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Constrained dominance: feasible beats infeasible, among infeasible
/// prefer lower total constraint violation, among feasible use Pareto dominance.
pub(crate) fn constrained_dominates(
    a_values: &[f64],
    b_values: &[f64],
    a_constraints: &[f64],
    b_constraints: &[f64],
    directions: &[Direction],
) -> bool {
    let a_feasible = a_constraints.iter().all(|&c| c <= 0.0);
    let b_feasible = b_constraints.iter().all(|&c| c <= 0.0);

    match (a_feasible, b_feasible) {
        (true, false) => true,
        (false, true) => false,
        (false, false) => {
            let a_violation: f64 = a_constraints.iter().map(|c| c.max(0.0)).sum();
            let b_violation: f64 = b_constraints.iter().map(|c| c.max(0.0)).sum();
            a_violation < b_violation
        }
        (true, true) => dominates(a_values, b_values, directions),
    }
}

/// Fast non-dominated sorting (Deb et al., 2002).
///
/// Returns `Vec<Vec<usize>>` where `fronts[0]` is the Pareto front,
/// each inner vec contains indices into `values`.
///
/// Complexity: O(M * N^2) where M = objectives, N = solutions.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn fast_non_dominated_sort(
    values: &[Vec<f64>],
    directions: &[Direction],
) -> Vec<Vec<usize>> {
    fast_non_dominated_sort_constrained(values, directions, &[])
}

/// Fast non-dominated sorting with constraint support.
///
/// `constraints` is either empty (no constraints) or has the same length
/// as `values`, where each entry is the constraint vector for that solution.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn fast_non_dominated_sort_constrained(
    values: &[Vec<f64>],
    directions: &[Direction],
    constraints: &[Vec<f64>],
) -> Vec<Vec<usize>> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }

    let has_constraints = !constraints.is_empty();
    let empty_constraints: Vec<f64> = Vec::new();

    // S_p: set of solutions dominated by p
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    // n_p: domination count for p
    let mut domination_count: Vec<usize> = vec![0; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let (a_c, b_c) = if has_constraints {
                (&constraints[i], &constraints[j])
            } else {
                (&empty_constraints, &empty_constraints)
            };

            let i_dom_j = if has_constraints {
                constrained_dominates(&values[i], &values[j], a_c, b_c, directions)
            } else {
                dominates(&values[i], &values[j], directions)
            };
            let j_dom_i = if has_constraints {
                constrained_dominates(&values[j], &values[i], b_c, a_c, directions)
            } else {
                dominates(&values[j], &values[i], directions)
            };

            if i_dom_j {
                dominated_by[i].push(j);
                domination_count[j] += 1;
            } else if j_dom_i {
                dominated_by[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current_front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();

    while !current_front.is_empty() {
        let mut next_front: Vec<usize> = Vec::new();
        for &p in &current_front {
            for &q in &dominated_by[p] {
                domination_count[q] -= 1;
                if domination_count[q] == 0 {
                    next_front.push(q);
                }
            }
        }
        fronts.push(current_front);
        current_front = next_front;
    }

    fronts
}

/// Crowding distance for one front (index-based, internal API).
///
/// Boundary solutions get `f64::INFINITY`. Returns one distance value per
/// solution in the front, in the same order as `front_indices`.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn crowding_distance_indexed(front_indices: &[usize], values: &[Vec<f64>]) -> Vec<f64> {
    let n = front_indices.len();
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let m = values[front_indices[0]].len(); // number of objectives
    let mut distances = vec![0.0_f64; n];

    // Helper to look up objective value for a front member.
    let val = |front_pos: usize, obj: usize| -> f64 { values[front_indices[front_pos]][obj] };

    for obj in 0..m {
        // Sort front positions by this objective
        let mut sorted: Vec<usize> = (0..n).collect();
        sorted.sort_by(|&a, &b| {
            val(a, obj)
                .partial_cmp(&val(b, obj))
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        // Boundary solutions get infinity
        distances[sorted[0]] = f64::INFINITY;
        distances[sorted[n - 1]] = f64::INFINITY;

        let range = val(sorted[n - 1], obj) - val(sorted[0], obj);
        if range > 0.0 {
            for i in 1..(n - 1) {
                distances[sorted[i]] += (val(sorted[i + 1], obj) - val(sorted[i - 1], obj)) / range;
            }
        }
    }

    distances
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the hypervolume indicator of a Pareto front.
///
/// The hypervolume is the volume of the objective space dominated by
/// the Pareto front and bounded by a reference point. A **higher**
/// hypervolume indicates a better front (closer to the ideal and more
/// spread out).
///
/// Each entry in `front` is one solution's objective values.
/// `reference_point` should be worse than all front members in every
/// objective (e.g., the worst acceptable values). Solutions that do
/// not strictly dominate the reference point are ignored.
///
/// Uses recursive slicing for dimensions > 1. Complexity grows with
/// the number of objectives and front size.
///
/// # Panics
///
/// Panics (in debug) if dimensions of `front`, `reference_point`, and
/// `directions` are inconsistent.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn hypervolume(front: &[Vec<f64>], reference_point: &[f64], directions: &[Direction]) -> f64 {
    if front.is_empty() {
        return 0.0;
    }
    let d = reference_point.len();
    debug_assert!(front.iter().all(|p| p.len() == d));
    debug_assert_eq!(d, directions.len());

    // Normalize to minimize-space (negate maximized objectives).
    let normalized: Vec<Vec<f64>> = front
        .iter()
        .map(|p| {
            p.iter()
                .zip(directions)
                .map(|(&v, dir)| match dir {
                    Direction::Minimize => v,
                    Direction::Maximize => -v,
                })
                .collect()
        })
        .collect();

    let ref_norm: Vec<f64> = reference_point
        .iter()
        .zip(directions)
        .map(|(&v, dir)| match dir {
            Direction::Minimize => v,
            Direction::Maximize => -v,
        })
        .collect();

    // Keep only points strictly dominated by the reference point.
    let filtered: Vec<Vec<f64>> = normalized
        .into_iter()
        .filter(|p| p.iter().zip(&ref_norm).all(|(&pv, &rv)| pv < rv))
        .collect();

    if filtered.is_empty() {
        return 0.0;
    }

    hv_recursive(&filtered, &ref_norm)
}

/// Recursive hypervolume via slicing on the last objective.
///
/// All points are in minimize-space and dominated by `reference`.
#[allow(clippy::cast_precision_loss)]
fn hv_recursive(points: &[Vec<f64>], reference: &[f64]) -> f64 {
    let d = reference.len();

    // Base case: 1-D hypervolume is just the gap from the best point to ref.
    if d == 1 {
        let min_val = points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        return (reference[0] - min_val).max(0.0);
    }

    // Single point: hypervolume is the product of gaps.
    if points.len() == 1 {
        return points[0]
            .iter()
            .zip(reference)
            .map(|(&p, &r)| (r - p).max(0.0))
            .product();
    }

    // Sort by last objective ascending.
    let mut sorted: Vec<&Vec<f64>> = points.iter().collect();
    sorted.sort_by(|a, b| {
        a[d - 1]
            .partial_cmp(&b[d - 1])
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    let sub_ref: Vec<f64> = reference[..d - 1].to_vec();
    let mut result = 0.0;

    for i in 0..sorted.len() {
        let height = if i + 1 < sorted.len() {
            sorted[i + 1][d - 1] - sorted[i][d - 1]
        } else {
            reference[d - 1] - sorted[i][d - 1]
        };

        if height <= 0.0 {
            continue;
        }

        // Project points[0..=i] onto the first d-1 dimensions and
        // keep only the non-dominated subset.
        let projected: Vec<Vec<f64>> = sorted[..=i].iter().map(|p| p[..d - 1].to_vec()).collect();
        let non_dom = non_dominated_minimize(&projected);

        if !non_dom.is_empty() {
            result += height * hv_recursive(&non_dom, &sub_ref);
        }
    }

    result
}

/// Return the non-dominated subset of `points` in minimize-space.
fn non_dominated_minimize(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut result = Vec::new();
    'outer: for (i, p) in points.iter().enumerate() {
        for (j, q) in points.iter().enumerate() {
            if i == j {
                continue;
            }
            // Check if q dominates p (all <=, at least one <).
            let mut all_leq = true;
            let mut any_lt = false;
            for (&qv, &pv) in q.iter().zip(p.iter()) {
                if qv > pv {
                    all_leq = false;
                    break;
                }
                if qv < pv {
                    any_lt = true;
                }
            }
            if all_leq && any_lt {
                continue 'outer;
            }
        }
        result.push(p.clone());
    }
    result
}

/// Compute non-dominated sorting of a set of solutions.
///
/// Return a vec of fronts, where `fronts[0]` is the Pareto front
/// (non-dominated solutions), `fronts[1]` is the next-best front
/// (dominated only by front 0), and so on. Each inner vec contains
/// indices into the original `solutions` slice.
///
/// Use the fast non-dominated sorting algorithm from
/// Deb et al. (2002) with O(M × N²) complexity, where M is the
/// number of objectives and N is the number of solutions.
#[must_use]
pub fn non_dominated_sort(solutions: &[Vec<f64>], directions: &[Direction]) -> Vec<Vec<usize>> {
    fast_non_dominated_sort(solutions, directions)
}

/// Filter solutions to return only non-dominated (Pareto-optimal) indices.
///
/// Equivalent to `non_dominated_sort(solutions, directions)[0]` but
/// communicates the intent more clearly. Use this when you only need
/// the Pareto front and not the full ranking.
#[must_use]
pub fn pareto_front_indices(solutions: &[Vec<f64>], directions: &[Direction]) -> Vec<usize> {
    let fronts = fast_non_dominated_sort(solutions, directions);
    fronts.into_iter().next().unwrap_or_default()
}

/// Compute crowding distance for diversity measurement.
///
/// Return one distance value per solution in `front` (same order).
/// Boundary solutions (best/worst in any objective) receive
/// [`f64::INFINITY`]. Interior solutions get a finite positive value
/// proportional to the gap between their neighbors in each objective.
///
/// Crowding distance is used by NSGA-II to prefer well-spread
/// solutions when two solutions are in the same front.
///
/// `directions` is accepted for API consistency but does not affect
/// the result, since crowding distance measures spacing regardless of
/// optimization direction.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::needless_range_loop)]
pub fn crowding_distance(front: &[Vec<f64>], _directions: &[Direction]) -> Vec<f64> {
    let n = front.len();
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let m = front[0].len();
    let mut distances = vec![0.0_f64; n];

    for obj in 0..m {
        let mut sorted: Vec<usize> = (0..n).collect();
        sorted.sort_by(|&a, &b| {
            front[a][obj]
                .partial_cmp(&front[b][obj])
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        distances[sorted[0]] = f64::INFINITY;
        distances[sorted[n - 1]] = f64::INFINITY;

        let range = front[sorted[n - 1]][obj] - front[sorted[0]][obj];
        if range > 0.0 {
            for i in 1..(n - 1) {
                distances[sorted[i]] +=
                    (front[sorted[i + 1]][obj] - front[sorted[i - 1]][obj]) / range;
            }
        }
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dominates_basic() {
        let dirs = [Direction::Minimize, Direction::Minimize];
        assert!(dominates(&[1.0, 1.0], &[2.0, 2.0], &dirs));
        assert!(!dominates(&[2.0, 2.0], &[1.0, 1.0], &dirs));
        // Equal does not dominate
        assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0], &dirs));
    }

    #[test]
    fn test_dominates_incomparable() {
        let dirs = [Direction::Minimize, Direction::Minimize];
        assert!(!dominates(&[1.0, 3.0], &[3.0, 1.0], &dirs));
        assert!(!dominates(&[3.0, 1.0], &[1.0, 3.0], &dirs));
    }

    #[test]
    fn test_dominates_maximize() {
        let dirs = [Direction::Maximize, Direction::Minimize];
        // a = (5, 1) vs b = (3, 2): a is better in both
        assert!(dominates(&[5.0, 1.0], &[3.0, 2.0], &dirs));
        assert!(!dominates(&[3.0, 2.0], &[5.0, 1.0], &dirs));
    }

    #[test]
    fn test_nds_known() {
        let values = vec![
            vec![1.0, 5.0], // front 0
            vec![5.0, 1.0], // front 0
            vec![3.0, 3.0], // front 0 (non-dominated)
            vec![4.0, 4.0], // front 1 (dominated by #2)
            vec![6.0, 6.0], // front 2
        ];
        let dirs = [Direction::Minimize, Direction::Minimize];
        let fronts = fast_non_dominated_sort(&values, &dirs);

        assert_eq!(fronts.len(), 3);
        let mut f0 = fronts[0].clone();
        f0.sort_unstable();
        assert_eq!(f0, vec![0, 1, 2]);
        assert_eq!(fronts[1], vec![3]);
        assert_eq!(fronts[2], vec![4]);
    }

    #[test]
    fn test_crowding_indexed_boundaries() {
        let values = vec![vec![1.0, 5.0], vec![3.0, 3.0], vec![5.0, 1.0]];
        let front = vec![0, 1, 2];
        let cd = crowding_distance_indexed(&front, &values);
        assert!(cd[0].is_infinite());
        assert!(cd[2].is_infinite());
        assert!(cd[1].is_finite());
        assert!(cd[1] > 0.0);
    }

    // ---- Public API tests ----

    #[test]
    fn test_hypervolume_2d_minimize() {
        // Front: (1,3), (2,2), (3,1) with ref (4,4) — all minimize
        let front = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];
        let dirs = [Direction::Minimize, Direction::Minimize];
        let hv = hypervolume(&front, &[4.0, 4.0], &dirs);
        // Strip 1: x=[1,2), h=4-3=1 → area=1
        // Strip 2: x=[2,3), h=4-2=2 → area=2
        // Strip 3: x=[3,4], h=4-1=3 → area=3
        // Total = 6
        assert!((hv - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume_2d_maximize() {
        // Front: (3,1), (2,2), (1,3) with ref (0,0) — all maximize
        let front = vec![vec![3.0, 1.0], vec![2.0, 2.0], vec![1.0, 3.0]];
        let dirs = [Direction::Maximize, Direction::Maximize];
        let hv = hypervolume(&front, &[0.0, 0.0], &dirs);
        // In negate-space: points become (-3,-1),(-2,-2),(-1,-3), ref=(0,0)
        // Same geometry as minimize test above → area = 6
        assert!((hv - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume_single_point() {
        let front = vec![vec![1.0, 1.0]];
        let dirs = [Direction::Minimize, Direction::Minimize];
        let hv = hypervolume(&front, &[3.0, 3.0], &dirs);
        // Rectangle: (3-1) * (3-1) = 4
        assert!((hv - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume_empty_front() {
        let front: Vec<Vec<f64>> = vec![];
        let dirs = [Direction::Minimize];
        assert!(hypervolume(&front, &[1.0], &dirs).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hypervolume_point_at_ref() {
        // Point not strictly better than ref → contributes nothing
        let front = vec![vec![5.0, 5.0]];
        let dirs = [Direction::Minimize, Direction::Minimize];
        let hv = hypervolume(&front, &[5.0, 5.0], &dirs);
        assert!(hv.abs() < f64::EPSILON);
    }

    #[test]
    fn test_hypervolume_3d() {
        // Single point in 3D: (1,1,1) with ref (2,2,2)
        let front = vec![vec![1.0, 1.0, 1.0]];
        let dirs = [
            Direction::Minimize,
            Direction::Minimize,
            Direction::Minimize,
        ];
        let hv = hypervolume(&front, &[2.0, 2.0, 2.0], &dirs);
        assert!((hv - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_non_dominated_sort_public() {
        let values = vec![
            vec![1.0, 5.0],
            vec![5.0, 1.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ];
        let dirs = [Direction::Minimize, Direction::Minimize];
        let fronts = non_dominated_sort(&values, &dirs);
        assert_eq!(fronts.len(), 2);
        let mut f0 = fronts[0].clone();
        f0.sort_unstable();
        assert_eq!(f0, vec![0, 1, 2]);
        assert_eq!(fronts[1], vec![3]);
    }

    #[test]
    fn test_pareto_front_indices_basic() {
        let values = vec![
            vec![1.0, 5.0],
            vec![5.0, 1.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ];
        let dirs = [Direction::Minimize, Direction::Minimize];
        let mut idx = pareto_front_indices(&values, &dirs);
        idx.sort_unstable();
        assert_eq!(idx, vec![0, 1, 2]);
    }

    #[test]
    fn test_pareto_front_indices_empty() {
        let values: Vec<Vec<f64>> = vec![];
        let dirs = [Direction::Minimize];
        assert!(pareto_front_indices(&values, &dirs).is_empty());
    }

    #[test]
    fn test_crowding_distance_public() {
        let front = vec![vec![1.0, 5.0], vec![3.0, 3.0], vec![5.0, 1.0]];
        let dirs = [Direction::Minimize, Direction::Minimize];
        let cd = crowding_distance(&front, &dirs);
        assert!(cd[0].is_infinite());
        assert!(cd[2].is_infinite());
        assert!(cd[1].is_finite());
        assert!(cd[1] > 0.0);
    }

    #[test]
    fn test_crowding_distance_single_point() {
        let front = vec![vec![2.0, 3.0]];
        let dirs = [Direction::Minimize, Direction::Minimize];
        let cd = crowding_distance(&front, &dirs);
        assert_eq!(cd.len(), 1);
        assert!(cd[0].is_infinite());
    }
}
