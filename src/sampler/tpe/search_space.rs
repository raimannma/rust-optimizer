//! Search space intersection utilities for multivariate TPE sampling.
//!
//! This module provides utilities for computing the intersection of search spaces
//! across completed trials, which is necessary for multivariate TPE sampling where
//! we need to identify parameters that appear in all trials.
//!
//! It also provides utilities for decomposing the search space into independent
//! groups based on parameter co-occurrence patterns, which enables more efficient
//! multivariate modeling when parameters naturally partition into independent subsets.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::distribution::Distribution;
use crate::parameter::ParamId;
use crate::sampler::CompletedTrial;

/// Computes the intersection of search spaces across completed trials.
///
/// In multivariate TPE sampling, we need to model the joint distribution of parameters.
/// However, trials may have different sets of parameters due to dynamic search spaces
/// (e.g., conditional parameters). The intersection search space contains only those
/// parameters that appear in ALL completed trials, allowing us to fit a joint KDE.
///
/// # Example
///
/// ```ignore
/// use std::collections::HashMap;
/// use optimizer::distribution::{Distribution, FloatDistribution};
/// use optimizer::param::ParamValue;
/// use optimizer::sampler::CompletedTrial;
/// use optimizer::sampler::tpe::IntersectionSearchSpace;
///
/// // Create two trials with overlapping parameters
/// let dist_x = Distribution::Float(FloatDistribution {
///     low: 0.0, high: 1.0, log_scale: false, step: None,
/// });
/// let dist_y = Distribution::Float(FloatDistribution {
///     low: 0.0, high: 1.0, log_scale: false, step: None,
/// });
///
/// let mut params1 = HashMap::new();
/// params1.insert("x".to_string(), ParamValue::Float(0.5));
/// params1.insert("y".to_string(), ParamValue::Float(0.3));
/// let mut dists1 = HashMap::new();
/// dists1.insert("x".to_string(), dist_x.clone());
/// dists1.insert("y".to_string(), dist_y.clone());
/// let trial1 = CompletedTrial::new(0, params1, dists1, 1.0);
///
/// let mut params2 = HashMap::new();
/// params2.insert("x".to_string(), ParamValue::Float(0.7));
/// // Note: trial2 doesn't have "y"
/// let mut dists2 = HashMap::new();
/// dists2.insert("x".to_string(), dist_x.clone());
/// let trial2 = CompletedTrial::new(1, params2, dists2, 0.5);
///
/// let trials = vec![trial1, trial2];
/// let intersection = IntersectionSearchSpace::calculate(&trials);
///
/// // Only "x" appears in both trials
/// assert!(intersection.contains_key("x"));
/// assert!(!intersection.contains_key("y"));
/// ```
#[derive(Debug, Clone, Default)]
pub struct IntersectionSearchSpace;

impl IntersectionSearchSpace {
    /// Calculates the intersection of search spaces across all completed trials.
    ///
    /// This method identifies parameters that appear in ALL completed trials and returns
    /// a mapping from parameter names to their distributions. If any parameter has
    /// different distributions across trials (e.g., different bounds), the first
    /// encountered distribution is used.
    ///
    /// # Arguments
    ///
    /// * `trials` - A slice of completed trials to compute the intersection from.
    ///
    /// # Returns
    ///
    /// A `HashMap` mapping parameter names to their distributions, containing only
    /// parameters that appear in every trial. Returns an empty map if `trials` is empty.
    ///
    /// # Notes
    ///
    /// - Parameters must appear in ALL trials to be included in the intersection.
    /// - If a parameter has different distributions in different trials, the distribution
    ///   from the first trial containing that parameter is used.
    /// - For dynamic search spaces where not all parameters are sampled in every trial,
    ///   this helps identify the "stable" set of parameters that can be modeled jointly.
    #[must_use]
    pub fn calculate(trials: &[CompletedTrial]) -> HashMap<ParamId, Distribution> {
        if trials.is_empty() {
            return HashMap::new();
        }

        // Get parameter ids from the first trial as the initial candidate set
        let first_trial = &trials[0];
        let mut candidate_params: HashSet<ParamId> =
            first_trial.distributions.keys().copied().collect();

        // Intersect with parameter sets from all other trials
        for trial in trials.iter().skip(1) {
            let trial_params: HashSet<ParamId> = trial.distributions.keys().copied().collect();
            candidate_params.retain(|param| trial_params.contains(param));
        }

        // Build the result map using distributions from the first trial
        // that contains each parameter
        let mut result = HashMap::new();
        for param_id in candidate_params {
            // Find the first trial that has this parameter and use its distribution
            for trial in trials {
                if let Some(dist) = trial.distributions.get(&param_id) {
                    result.insert(param_id, dist.clone());
                    break;
                }
            }
        }

        result
    }
}

/// Decomposes the search space into independent parameter groups based on co-occurrence.
///
/// In multivariate TPE, we model the joint distribution of parameters. However, when
/// certain parameters never appear together (e.g., they are in different branches of
/// a conditional), modeling them jointly is wasteful and can hurt performance.
///
/// `GroupDecomposedSearchSpace` analyzes trial history to identify groups of parameters
/// that always appear together. By building a co-occurrence graph (where edges connect
/// parameters that appear in the same trial) and finding its connected components, we
/// can partition parameters into independent groups that can be sampled separately.
///
/// # Example
///
/// ```ignore
/// use std::collections::HashMap;
/// use optimizer::distribution::{Distribution, FloatDistribution};
/// use optimizer::param::ParamValue;
/// use optimizer::sampler::CompletedTrial;
/// use optimizer::sampler::tpe::GroupDecomposedSearchSpace;
///
/// // Trials with two independent parameter groups:
/// // Group 1: "x", "y" always appear together
/// // Group 2: "a", "b" always appear together
/// // But "x"/"y" never appear with "a"/"b"
///
/// let dist = Distribution::Float(FloatDistribution {
///     low: 0.0, high: 1.0, log_scale: false, step: None,
/// });
///
/// let mut params1 = HashMap::new();
/// params1.insert("x".to_string(), ParamValue::Float(0.1));
/// params1.insert("y".to_string(), ParamValue::Float(0.2));
/// let mut dists1 = HashMap::new();
/// dists1.insert("x".to_string(), dist.clone());
/// dists1.insert("y".to_string(), dist.clone());
/// let trial1 = CompletedTrial::new(0, params1, dists1, 1.0);
///
/// let mut params2 = HashMap::new();
/// params2.insert("a".to_string(), ParamValue::Float(0.3));
/// params2.insert("b".to_string(), ParamValue::Float(0.4));
/// let mut dists2 = HashMap::new();
/// dists2.insert("a".to_string(), dist.clone());
/// dists2.insert("b".to_string(), dist.clone());
/// let trial2 = CompletedTrial::new(1, params2, dists2, 0.5);
///
/// let trials = vec![trial1, trial2];
/// let groups = GroupDecomposedSearchSpace::calculate(&trials);
///
/// // Should produce two groups: {"x", "y"} and {"a", "b"}
/// assert_eq!(groups.len(), 2);
/// ```
#[derive(Debug, Clone, Default)]
pub struct GroupDecomposedSearchSpace;

impl GroupDecomposedSearchSpace {
    /// Calculates parameter groups based on co-occurrence in completed trials.
    ///
    /// This method builds a co-occurrence graph where:
    /// - Each parameter is a node
    /// - An edge exists between two parameters if they appear in the same trial
    ///
    /// Then finds connected components to partition parameters into independent groups.
    ///
    /// # Arguments
    ///
    /// * `trials` - A slice of completed trials to analyze.
    ///
    /// # Returns
    ///
    /// A `Vec<HashSet<ParamId>>` where each `HashSet` represents an independent group
    /// of parameters that co-occur. Parameters within the same group have appeared
    /// together in at least one trial (directly or transitively). Parameters in
    /// different groups have never appeared in the same trial.
    ///
    /// Returns an empty vector if `trials` is empty.
    ///
    /// # Panics
    ///
    /// This function does not panic under normal circumstances. The internal
    /// `.expect()` calls are guarded by the algorithm's invariants.
    ///
    /// # Notes
    ///
    /// - Single-parameter groups are included in the output.
    /// - The order of groups in the output vector is not guaranteed.
    /// - This is useful for `MultivariateTpeSampler` when `group=true` to sample
    ///   independent parameter groups separately.
    #[must_use]
    pub fn calculate(trials: &[CompletedTrial]) -> Vec<HashSet<ParamId>> {
        if trials.is_empty() {
            return Vec::new();
        }

        // Collect all unique parameter ids
        let mut all_params: HashSet<ParamId> = HashSet::new();
        for trial in trials {
            for &param_id in trial.distributions.keys() {
                all_params.insert(param_id);
            }
        }

        if all_params.is_empty() {
            return Vec::new();
        }

        // Build adjacency list for co-occurrence graph
        // Two parameters are adjacent if they appear in the same trial
        let mut adjacency: HashMap<ParamId, HashSet<ParamId>> = HashMap::new();
        for &param in &all_params {
            adjacency.insert(param, HashSet::new());
        }

        for trial in trials {
            let trial_params: Vec<ParamId> = trial.distributions.keys().copied().collect();
            // Connect all pairs of parameters in this trial
            for (i, &param1) in trial_params.iter().enumerate() {
                for &param2 in trial_params.iter().skip(i + 1) {
                    adjacency
                        .get_mut(&param1)
                        .expect("param should exist in adjacency map")
                        .insert(param2);
                    adjacency
                        .get_mut(&param2)
                        .expect("param should exist in adjacency map")
                        .insert(param1);
                }
            }
        }

        // Find connected components using BFS
        let mut visited: HashSet<ParamId> = HashSet::new();
        let mut groups: Vec<HashSet<ParamId>> = Vec::new();

        for &param in &all_params {
            if visited.contains(&param) {
                continue;
            }

            // BFS to find all parameters in this component
            let mut component: HashSet<ParamId> = HashSet::new();
            let mut queue: VecDeque<ParamId> = VecDeque::new();
            queue.push_back(param);
            visited.insert(param);

            while let Some(current) = queue.pop_front() {
                component.insert(current);

                if let Some(neighbors) = adjacency.get(&current) {
                    for &neighbor in neighbors {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }

            groups.push(component);
        }

        groups
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::{CategoricalDistribution, FloatDistribution, IntDistribution};
    use crate::param::ParamValue;
    use crate::parameter::ParamId;

    fn create_trial(
        id: u64,
        params: Vec<(ParamId, ParamValue, Distribution)>,
        value: f64,
    ) -> CompletedTrial {
        let mut param_map = HashMap::new();
        let mut dist_map = HashMap::new();
        for (param_id, pv, dist) in params {
            param_map.insert(param_id, pv);
            dist_map.insert(param_id, dist);
        }
        CompletedTrial::new(id, param_map, dist_map, HashMap::new(), value)
    }

    #[test]
    fn test_empty_trials() {
        let trials: Vec<CompletedTrial> = vec![];
        let result = IntersectionSearchSpace::calculate(&trials);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_trial() {
        let x_id = ParamId::new();
        let y_id = ParamId::new();
        let dist_x = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });
        let dist_y = Distribution::Int(IntDistribution {
            low: 1,
            high: 10,
            log_scale: false,
            step: None,
        });

        let trial = create_trial(
            0,
            vec![
                (x_id, ParamValue::Float(0.5), dist_x.clone()),
                (y_id, ParamValue::Int(5), dist_y.clone()),
            ],
            1.0,
        );

        let result = IntersectionSearchSpace::calculate(&[trial]);
        assert_eq!(result.len(), 2);
        assert!(result.contains_key(&x_id));
        assert!(result.contains_key(&y_id));
        assert_eq!(result.get(&x_id), Some(&dist_x));
        assert_eq!(result.get(&y_id), Some(&dist_y));
    }

    #[test]
    fn test_all_trials_same_params() {
        let x_id = ParamId::new();
        let y_id = ParamId::new();
        let dist_x = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });
        let dist_y = Distribution::Float(FloatDistribution {
            low: -1.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        let trials: Vec<CompletedTrial> = (0..5)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let val = i as f64 * 0.1;
                create_trial(
                    i,
                    vec![
                        (x_id, ParamValue::Float(val), dist_x.clone()),
                        (y_id, ParamValue::Float(val - 0.5), dist_y.clone()),
                    ],
                    val * val,
                )
            })
            .collect();

        let result = IntersectionSearchSpace::calculate(&trials);
        assert_eq!(result.len(), 2);
        assert!(result.contains_key(&x_id));
        assert!(result.contains_key(&y_id));
    }

    #[test]
    fn test_partial_overlap() {
        let x_id = ParamId::new();
        let y_id = ParamId::new();
        let z_id = ParamId::new();
        let dist_x = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });
        let dist_y = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });
        let dist_z = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        // Trial 1: x, y
        let trial1 = create_trial(
            0,
            vec![
                (x_id, ParamValue::Float(0.5), dist_x.clone()),
                (y_id, ParamValue::Float(0.3), dist_y.clone()),
            ],
            1.0,
        );

        // Trial 2: x, z (no y)
        let trial2 = create_trial(
            1,
            vec![
                (x_id, ParamValue::Float(0.7), dist_x.clone()),
                (z_id, ParamValue::Float(0.2), dist_z.clone()),
            ],
            0.5,
        );

        // Trial 3: x, y, z (has all)
        let trial3 = create_trial(
            2,
            vec![
                (x_id, ParamValue::Float(0.6), dist_x.clone()),
                (y_id, ParamValue::Float(0.4), dist_y.clone()),
                (z_id, ParamValue::Float(0.1), dist_z.clone()),
            ],
            0.8,
        );

        let result = IntersectionSearchSpace::calculate(&[trial1, trial2, trial3]);

        // Only x appears in all three trials
        assert_eq!(result.len(), 1);
        assert!(result.contains_key(&x_id));
        assert!(!result.contains_key(&y_id));
        assert!(!result.contains_key(&z_id));
    }

    #[test]
    fn test_no_common_params() {
        let x_id = ParamId::new();
        let y_id = ParamId::new();
        let dist_x = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });
        let dist_y = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        // Trial 1: only x
        let trial1 = create_trial(0, vec![(x_id, ParamValue::Float(0.5), dist_x.clone())], 1.0);

        // Trial 2: only y
        let trial2 = create_trial(1, vec![(y_id, ParamValue::Float(0.3), dist_y.clone())], 0.5);

        let result = IntersectionSearchSpace::calculate(&[trial1, trial2]);

        // No common parameters
        assert!(result.is_empty());
    }

    #[test]
    fn test_mixed_distribution_types() {
        let lr_id = ParamId::new();
        let n_layers_id = ParamId::new();
        let optimizer_id = ParamId::new();
        let dist_float = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });
        let dist_int = Distribution::Int(IntDistribution {
            low: 1,
            high: 100,
            log_scale: false,
            step: None,
        });
        let dist_cat = Distribution::Categorical(CategoricalDistribution { n_choices: 3 });

        let trial1 = create_trial(
            0,
            vec![
                (lr_id, ParamValue::Float(0.01), dist_float.clone()),
                (n_layers_id, ParamValue::Int(3), dist_int.clone()),
                (optimizer_id, ParamValue::Categorical(0), dist_cat.clone()),
            ],
            1.0,
        );

        let trial2 = create_trial(
            1,
            vec![
                (lr_id, ParamValue::Float(0.001), dist_float.clone()),
                (n_layers_id, ParamValue::Int(5), dist_int.clone()),
                (optimizer_id, ParamValue::Categorical(1), dist_cat.clone()),
            ],
            0.8,
        );

        let result = IntersectionSearchSpace::calculate(&[trial1, trial2]);

        assert_eq!(result.len(), 3);
        assert!(matches!(result.get(&lr_id), Some(Distribution::Float(_))));
        assert!(matches!(
            result.get(&n_layers_id),
            Some(Distribution::Int(_))
        ));
        assert!(matches!(
            result.get(&optimizer_id),
            Some(Distribution::Categorical(_))
        ));
    }

    #[test]
    fn test_distribution_from_first_trial() {
        let x_id = ParamId::new();
        // Test that when distributions differ, the first trial's distribution is used
        let dist_x_v1 = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });
        let dist_x_v2 = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 10.0, // Different upper bound
            log_scale: false,
            step: None,
        });

        let trial1 = create_trial(
            0,
            vec![(x_id, ParamValue::Float(0.5), dist_x_v1.clone())],
            1.0,
        );

        let trial2 = create_trial(
            1,
            vec![(x_id, ParamValue::Float(5.0), dist_x_v2.clone())],
            0.5,
        );

        let result = IntersectionSearchSpace::calculate(&[trial1, trial2]);

        assert_eq!(result.len(), 1);
        // Should use the distribution from the first trial
        assert_eq!(result.get(&x_id), Some(&dist_x_v1));
    }

    #[test]
    fn test_many_trials_with_conditional_params() {
        let lr_id = ParamId::new();
        let use_dropout_id = ParamId::new();
        let dropout_rate_id = ParamId::new();
        // Simulate a scenario with conditional parameters
        let dist_lr = Distribution::Float(FloatDistribution {
            low: 1e-5,
            high: 1e-1,
            log_scale: true,
            step: None,
        });
        let dist_dropout = Distribution::Categorical(CategoricalDistribution { n_choices: 2 });
        let dist_dropout_rate = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 0.5,
            log_scale: false,
            step: None,
        });

        // Trial 1: use_dropout=true, has dropout_rate
        let trial1 = create_trial(
            0,
            vec![
                (lr_id, ParamValue::Float(0.01), dist_lr.clone()),
                (
                    use_dropout_id,
                    ParamValue::Categorical(1),
                    dist_dropout.clone(),
                ),
                (
                    dropout_rate_id,
                    ParamValue::Float(0.2),
                    dist_dropout_rate.clone(),
                ),
            ],
            1.0,
        );

        // Trial 2: use_dropout=false, no dropout_rate
        let trial2 = create_trial(
            1,
            vec![
                (lr_id, ParamValue::Float(0.001), dist_lr.clone()),
                (
                    use_dropout_id,
                    ParamValue::Categorical(0),
                    dist_dropout.clone(),
                ),
            ],
            0.8,
        );

        // Trial 3: use_dropout=true, has dropout_rate
        let trial3 = create_trial(
            2,
            vec![
                (lr_id, ParamValue::Float(0.005), dist_lr.clone()),
                (
                    use_dropout_id,
                    ParamValue::Categorical(1),
                    dist_dropout.clone(),
                ),
                (
                    dropout_rate_id,
                    ParamValue::Float(0.3),
                    dist_dropout_rate.clone(),
                ),
            ],
            0.9,
        );

        let result = IntersectionSearchSpace::calculate(&[trial1, trial2, trial3]);

        // Only lr and use_dropout appear in all trials
        assert_eq!(result.len(), 2);
        assert!(result.contains_key(&lr_id));
        assert!(result.contains_key(&use_dropout_id));
        assert!(!result.contains_key(&dropout_rate_id)); // Not in trial2
    }

    // ==================== GroupDecomposedSearchSpace Tests ====================

    #[test]
    fn test_group_empty_trials() {
        let trials: Vec<CompletedTrial> = vec![];
        let groups = GroupDecomposedSearchSpace::calculate(&trials);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_group_single_trial_single_param() {
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        let x_id = ParamId::new();
        let trial = create_trial(0, vec![(x_id, ParamValue::Float(0.5), dist)], 1.0);

        let groups = GroupDecomposedSearchSpace::calculate(&[trial]);

        assert_eq!(groups.len(), 1);
        assert!(groups[0].contains(&x_id));
        assert_eq!(groups[0].len(), 1);
    }

    #[test]
    fn test_group_single_trial_multiple_params() {
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        let x_id = ParamId::new();
        let y_id = ParamId::new();
        let z_id = ParamId::new();
        let trial = create_trial(
            0,
            vec![
                (x_id, ParamValue::Float(0.5), dist.clone()),
                (y_id, ParamValue::Float(0.3), dist.clone()),
                (z_id, ParamValue::Float(0.7), dist),
            ],
            1.0,
        );

        let groups = GroupDecomposedSearchSpace::calculate(&[trial]);

        // All params appear together, so one group
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
        assert!(groups[0].contains(&x_id));
        assert!(groups[0].contains(&y_id));
        assert!(groups[0].contains(&z_id));
    }

    #[test]
    fn test_group_two_independent_groups() {
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        let x_id = ParamId::new();
        let y_id = ParamId::new();
        let a_id = ParamId::new();
        let b_id = ParamId::new();

        // Trial 1: x, y (group 1)
        let trial1 = create_trial(
            0,
            vec![
                (x_id, ParamValue::Float(0.1), dist.clone()),
                (y_id, ParamValue::Float(0.2), dist.clone()),
            ],
            1.0,
        );

        // Trial 2: a, b (group 2 - never appears with x or y)
        let trial2 = create_trial(
            1,
            vec![
                (a_id, ParamValue::Float(0.3), dist.clone()),
                (b_id, ParamValue::Float(0.4), dist),
            ],
            0.5,
        );

        let groups = GroupDecomposedSearchSpace::calculate(&[trial1, trial2]);

        // Should have 2 independent groups
        assert_eq!(groups.len(), 2);

        // Find which group has x/y and which has a/b
        let group_xy = groups.iter().find(|g| g.contains(&x_id));
        let group_ab = groups.iter().find(|g| g.contains(&a_id));

        assert!(group_xy.is_some());
        assert!(group_ab.is_some());

        let group_xy = group_xy.expect("group with x should exist");
        let group_ab = group_ab.expect("group with a should exist");

        assert_eq!(group_xy.len(), 2);
        assert!(group_xy.contains(&x_id));
        assert!(group_xy.contains(&y_id));

        assert_eq!(group_ab.len(), 2);
        assert!(group_ab.contains(&a_id));
        assert!(group_ab.contains(&b_id));
    }

    #[test]
    fn test_group_transitive_connection() {
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        let x_id = ParamId::new();
        let y_id = ParamId::new();
        let z_id = ParamId::new();

        // Trial 1: x, y
        let trial1 = create_trial(
            0,
            vec![
                (x_id, ParamValue::Float(0.1), dist.clone()),
                (y_id, ParamValue::Float(0.2), dist.clone()),
            ],
            1.0,
        );

        // Trial 2: y, z (y connects x and z transitively)
        let trial2 = create_trial(
            1,
            vec![
                (y_id, ParamValue::Float(0.3), dist.clone()),
                (z_id, ParamValue::Float(0.4), dist),
            ],
            0.5,
        );

        let groups = GroupDecomposedSearchSpace::calculate(&[trial1, trial2]);

        // All should be in one group due to transitive connection via y
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
        assert!(groups[0].contains(&x_id));
        assert!(groups[0].contains(&y_id));
        assert!(groups[0].contains(&z_id));
    }

    #[test]
    fn test_group_chain_connection() {
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        let a_id = ParamId::new();
        let b_id = ParamId::new();
        let c_id = ParamId::new();
        let d_id = ParamId::new();

        // Create a chain: a-b, b-c, c-d
        let trial1 = create_trial(
            0,
            vec![
                (a_id, ParamValue::Float(0.1), dist.clone()),
                (b_id, ParamValue::Float(0.2), dist.clone()),
            ],
            1.0,
        );

        let trial2 = create_trial(
            1,
            vec![
                (b_id, ParamValue::Float(0.3), dist.clone()),
                (c_id, ParamValue::Float(0.4), dist.clone()),
            ],
            0.5,
        );

        let trial3 = create_trial(
            2,
            vec![
                (c_id, ParamValue::Float(0.5), dist.clone()),
                (d_id, ParamValue::Float(0.6), dist),
            ],
            0.3,
        );

        let groups = GroupDecomposedSearchSpace::calculate(&[trial1, trial2, trial3]);

        // All should be connected in one group
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 4);
        assert!(groups[0].contains(&a_id));
        assert!(groups[0].contains(&b_id));
        assert!(groups[0].contains(&c_id));
        assert!(groups[0].contains(&d_id));
    }

    #[test]
    fn test_group_multiple_isolated_params() {
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        let x_id = ParamId::new();
        let y_id = ParamId::new();
        let z_id = ParamId::new();

        // Each trial has exactly one parameter (all isolated)
        let trial1 = create_trial(0, vec![(x_id, ParamValue::Float(0.1), dist.clone())], 1.0);
        let trial2 = create_trial(1, vec![(y_id, ParamValue::Float(0.2), dist.clone())], 0.5);
        let trial3 = create_trial(2, vec![(z_id, ParamValue::Float(0.3), dist)], 0.3);

        let groups = GroupDecomposedSearchSpace::calculate(&[trial1, trial2, trial3]);

        // Each param should be its own group
        assert_eq!(groups.len(), 3);
        for group in &groups {
            assert_eq!(group.len(), 1);
        }

        // Verify all params are covered
        let all_params: HashSet<ParamId> = groups.iter().flatten().copied().collect();
        assert!(all_params.contains(&x_id));
        assert!(all_params.contains(&y_id));
        assert!(all_params.contains(&z_id));
    }

    #[test]
    fn test_group_complex_scenario() {
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        let x_id = ParamId::new();
        let y_id = ParamId::new();
        let z_id = ParamId::new();
        let a_id = ParamId::new();
        let b_id = ParamId::new();
        let w_id = ParamId::new();

        // Group 1: x, y, z connected
        // Group 2: a, b connected
        // w is isolated

        let trial1 = create_trial(
            0,
            vec![
                (x_id, ParamValue::Float(0.1), dist.clone()),
                (y_id, ParamValue::Float(0.2), dist.clone()),
            ],
            1.0,
        );

        let trial2 = create_trial(
            1,
            vec![
                (y_id, ParamValue::Float(0.3), dist.clone()),
                (z_id, ParamValue::Float(0.4), dist.clone()),
            ],
            0.5,
        );

        let trial3 = create_trial(
            2,
            vec![
                (a_id, ParamValue::Float(0.5), dist.clone()),
                (b_id, ParamValue::Float(0.6), dist.clone()),
            ],
            0.3,
        );

        let trial4 = create_trial(3, vec![(w_id, ParamValue::Float(0.7), dist)], 0.2);

        let groups = GroupDecomposedSearchSpace::calculate(&[trial1, trial2, trial3, trial4]);

        // Should have 3 groups: {x,y,z}, {a,b}, {w}
        assert_eq!(groups.len(), 3);

        let group_xyz = groups.iter().find(|g| g.contains(&x_id));
        let group_ab = groups.iter().find(|g| g.contains(&a_id));
        let group_w = groups.iter().find(|g| g.contains(&w_id));

        assert!(group_xyz.is_some());
        assert!(group_ab.is_some());
        assert!(group_w.is_some());

        let group_xyz = group_xyz.expect("group with x should exist");
        assert_eq!(group_xyz.len(), 3);
        assert!(group_xyz.contains(&x_id));
        assert!(group_xyz.contains(&y_id));
        assert!(group_xyz.contains(&z_id));

        let group_ab = group_ab.expect("group with a should exist");
        assert_eq!(group_ab.len(), 2);
        assert!(group_ab.contains(&a_id));
        assert!(group_ab.contains(&b_id));

        let group_w = group_w.expect("group with w should exist");
        assert_eq!(group_w.len(), 1);
        assert!(group_w.contains(&w_id));
    }

    #[test]
    fn test_group_all_params_same_trial() {
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        let a_id = ParamId::new();
        let b_id = ParamId::new();
        let c_id = ParamId::new();
        let d_id = ParamId::new();

        // All params in single trial
        let trial = create_trial(
            0,
            vec![
                (a_id, ParamValue::Float(0.1), dist.clone()),
                (b_id, ParamValue::Float(0.2), dist.clone()),
                (c_id, ParamValue::Float(0.3), dist.clone()),
                (d_id, ParamValue::Float(0.4), dist),
            ],
            1.0,
        );

        let groups = GroupDecomposedSearchSpace::calculate(&[trial]);

        // All in one group
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 4);
    }

    #[test]
    fn test_group_with_mixed_distribution_types() {
        let lr_id = ParamId::new();
        let n_layers_id = ParamId::new();
        let optimizer_id = ParamId::new();
        let dist_float = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });
        let dist_int = Distribution::Int(IntDistribution {
            low: 1,
            high: 10,
            log_scale: false,
            step: None,
        });
        let dist_cat = Distribution::Categorical(CategoricalDistribution { n_choices: 3 });

        // Group 1: learning_rate (float), n_layers (int)
        // Group 2: optimizer (categorical)
        let trial1 = create_trial(
            0,
            vec![
                (lr_id, ParamValue::Float(0.01), dist_float.clone()),
                (n_layers_id, ParamValue::Int(3), dist_int.clone()),
            ],
            1.0,
        );

        let trial2 = create_trial(
            1,
            vec![(optimizer_id, ParamValue::Categorical(1), dist_cat)],
            0.5,
        );

        let trial3 = create_trial(
            2,
            vec![
                (lr_id, ParamValue::Float(0.001), dist_float),
                (n_layers_id, ParamValue::Int(5), dist_int),
            ],
            0.8,
        );

        let groups = GroupDecomposedSearchSpace::calculate(&[trial1, trial2, trial3]);

        // Should have 2 groups: {learning_rate, n_layers} and {optimizer}
        assert_eq!(groups.len(), 2);

        let group_lr = groups.iter().find(|g| g.contains(&lr_id));
        let group_opt = groups.iter().find(|g| g.contains(&optimizer_id));

        assert!(group_lr.is_some());
        assert!(group_opt.is_some());

        let group_lr = group_lr.expect("group with learning_rate should exist");
        assert_eq!(group_lr.len(), 2);
        assert!(group_lr.contains(&lr_id));
        assert!(group_lr.contains(&n_layers_id));

        let group_opt = group_opt.expect("group with optimizer should exist");
        assert_eq!(group_opt.len(), 1);
        assert!(group_opt.contains(&optimizer_id));
    }

    #[test]
    fn test_group_star_topology() {
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        let center_id = ParamId::new();
        let a_id = ParamId::new();
        let b_id = ParamId::new();
        let c_id = ParamId::new();

        // Star topology: center connects to all others
        let trial1 = create_trial(
            0,
            vec![
                (center_id, ParamValue::Float(0.1), dist.clone()),
                (a_id, ParamValue::Float(0.2), dist.clone()),
            ],
            1.0,
        );

        let trial2 = create_trial(
            1,
            vec![
                (center_id, ParamValue::Float(0.3), dist.clone()),
                (b_id, ParamValue::Float(0.4), dist.clone()),
            ],
            0.5,
        );

        let trial3 = create_trial(
            2,
            vec![
                (center_id, ParamValue::Float(0.5), dist.clone()),
                (c_id, ParamValue::Float(0.6), dist),
            ],
            0.3,
        );

        let groups = GroupDecomposedSearchSpace::calculate(&[trial1, trial2, trial3]);

        // All connected via center
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 4);
        assert!(groups[0].contains(&center_id));
        assert!(groups[0].contains(&a_id));
        assert!(groups[0].contains(&b_id));
        assert!(groups[0].contains(&c_id));
    }
}
