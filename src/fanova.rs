//! fANOVA (functional ANOVA) parameter importance via random forest.
//!
//! fANOVA decomposes the variance of the objective function into
//! contributions from individual parameters (**main effects**) and
//! parameter pairs (**interaction effects**). This helps answer the
//! question: *"Which parameters matter most, and do any parameters
//! interact?"*
//!
//! # Algorithm
//!
//! 1. Fit a random forest to the mapping `(parameters) → objective`
//! 2. Apply functional ANOVA decomposition to the trained forest
//! 3. Compute main effects: the variance explained by each parameter alone
//! 4. Compute interaction effects: the additional variance explained by
//!    pairs of parameters beyond their individual contributions
//! 5. Normalize so all importances sum to 1.0
//!
//! # When to use
//!
//! - **After optimization**: call [`Study::fanova()`](crate::Study::fanova)
//!   or [`Study::fanova_with_config()`](crate::Study::fanova_with_config)
//!   to identify which parameters had the most impact
//! - **Interaction detection**: unlike Spearman correlation
//!   ([`Study::param_importance()`](crate::Study::param_importance)),
//!   fANOVA can detect non-linear relationships and parameter interactions
//! - **Hyperparameter tuning**: focus tuning effort on high-importance
//!   parameters and fix low-importance ones to reasonable defaults
//!
//! # Reference
//!
//! Hutter, F., Hoos, H. & Leyton-Brown, K. (2014). "An Efficient
//! Approach for Assessing Hyperparameter Importance." ICML 2014.
//!
//! # Example
//!
//! ```
//! use optimizer::prelude::*;
//!
//! let study: Study<f64> = Study::new(Direction::Minimize);
//! let x = FloatParam::new(0.0, 10.0).name("x");
//! let y = FloatParam::new(0.0, 10.0).name("y");
//!
//! study
//!     .optimize(50, |trial: &mut optimizer::Trial| {
//!         let xv = x.suggest(trial)?;
//!         let yv = y.suggest(trial)?;
//!         // x matters much more than y
//!         Ok::<_, optimizer::Error>(3.0 * xv + 0.1 * yv)
//!     })
//!     .unwrap();
//!
//! let result = study.fanova().unwrap();
//! // Main effects sorted by descending importance
//! assert_eq!(result.main_effects[0].0, "x");
//! ```

/// Result of fANOVA analysis.
///
/// All importance values are fractions of total variance and sum to 1.0
/// across main effects and interactions combined.
#[derive(Debug, Clone)]
pub struct FanovaResult {
    /// Per-parameter importance (fraction of total variance explained).
    ///
    /// Sorted by descending importance. Each entry is
    /// `(parameter_name, importance)` where importance is in `[0.0, 1.0]`.
    pub main_effects: Vec<(String, f64)>,
    /// Pairwise interaction importance (fraction of total variance explained).
    ///
    /// Sorted by descending importance. Each entry is
    /// `((param_a, param_b), importance)`. Only pairs with non-negligible
    /// interaction (> 1e-10) are included.
    pub interactions: Vec<((String, String), f64)>,
}

/// Configuration for fANOVA analysis.
///
/// Use [`Default::default()`] for reasonable settings, or customize
/// the random forest parameters for specific needs. Pass to
/// [`Study::fanova_with_config()`](crate::Study::fanova_with_config).
#[derive(Debug, Clone)]
pub struct FanovaConfig {
    /// Number of trees in the random forest (default: 64).
    pub n_trees: usize,
    /// Maximum depth of each tree. `None` for unlimited (default: `None`).
    pub max_depth: Option<usize>,
    /// Minimum samples required to split a node (default: 2).
    pub min_samples_split: usize,
    /// Minimum samples required in a leaf node (default: 1).
    pub min_samples_leaf: usize,
    /// Random seed for reproducibility (default: `Some(42)`).
    pub seed: Option<u64>,
}

impl Default for FanovaConfig {
    fn default() -> Self {
        Self {
            n_trees: 64,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            seed: Some(42),
        }
    }
}

// --- Decision Tree ---

/// A node in the regression tree (arena-allocated).
#[derive(Debug, Clone)]
enum TreeNode {
    Leaf {
        value: f64,
        n_samples: usize,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
        n_samples: usize,
    },
}

/// A regression decision tree for fANOVA.
#[derive(Debug, Clone)]
struct DecisionTree {
    nodes: Vec<TreeNode>,
}

impl DecisionTree {
    /// Build a tree from the given data using the specified bootstrap indices.
    fn build(
        data: &[Vec<f64>],
        targets: &[f64],
        indices: &[usize],
        config: &FanovaConfig,
        rng: &mut fastrand::Rng,
    ) -> Self {
        let mut tree = Self { nodes: Vec::new() };
        tree.build_node(data, targets, indices, 0, config, rng);
        tree
    }

    #[allow(clippy::cast_precision_loss)]
    fn build_node(
        &mut self,
        data: &[Vec<f64>],
        targets: &[f64],
        indices: &[usize],
        depth: usize,
        config: &FanovaConfig,
        rng: &mut fastrand::Rng,
    ) -> usize {
        let n = indices.len();
        let mean = indices.iter().map(|&i| targets[i]).sum::<f64>() / n as f64;

        // Stopping conditions
        if n < config.min_samples_split || config.max_depth.is_some_and(|d| depth >= d) {
            let idx = self.nodes.len();
            self.nodes.push(TreeNode::Leaf {
                value: mean,
                n_samples: n,
            });
            return idx;
        }

        // Pure node check (all targets identical)
        #[allow(clippy::float_cmp)]
        if indices.iter().all(|&i| targets[i] == targets[indices[0]]) {
            let idx = self.nodes.len();
            self.nodes.push(TreeNode::Leaf {
                value: mean,
                n_samples: n,
            });
            return idx;
        }

        let n_features = data[0].len();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let max_features = ((n_features as f64).sqrt().ceil() as usize)
            .max(1)
            .min(n_features);
        let candidates = partial_shuffle(n_features, max_features, rng);

        // Total variance at this node
        let total_var: f64 = indices.iter().map(|&i| (targets[i] - mean).powi(2)).sum();
        if total_var == 0.0 {
            let idx = self.nodes.len();
            self.nodes.push(TreeNode::Leaf {
                value: mean,
                n_samples: n,
            });
            return idx;
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;

        for &feat in &candidates {
            let mut values: Vec<f64> = indices.iter().map(|&i| data[i][feat]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            values.dedup();

            if values.len() < 2 {
                continue;
            }

            for w in values.windows(2) {
                let threshold = f64::midpoint(w[0], w[1]);
                let (l_sum, l_sq, l_n, r_sum, r_sq, r_n) =
                    split_stats(data, targets, indices, feat, threshold);

                if l_n < config.min_samples_leaf || r_n < config.min_samples_leaf {
                    continue;
                }

                let l_var = l_sq - l_sum * l_sum / l_n as f64;
                let r_var = r_sq - r_sum * r_sum / r_n as f64;
                let score = total_var - l_var - r_var;

                if score > best_score {
                    best_score = score;
                    best_feature = feat;
                    best_threshold = threshold;
                }
            }
        }

        if best_score <= 0.0 {
            let idx = self.nodes.len();
            self.nodes.push(TreeNode::Leaf {
                value: mean,
                n_samples: n,
            });
            return idx;
        }

        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| data[i][best_feature] <= best_threshold);

        if left_indices.is_empty() || right_indices.is_empty() {
            let idx = self.nodes.len();
            self.nodes.push(TreeNode::Leaf {
                value: mean,
                n_samples: n,
            });
            return idx;
        }

        // Reserve slot for this split node (placeholder replaced below)
        let node_idx = self.nodes.len();
        self.nodes.push(TreeNode::Leaf {
            value: 0.0,
            n_samples: 0,
        });

        let left = self.build_node(data, targets, &left_indices, depth + 1, config, rng);
        let right = self.build_node(data, targets, &right_indices, depth + 1, config, rng);

        self.nodes[node_idx] = TreeNode::Split {
            feature: best_feature,
            threshold: best_threshold,
            left,
            right,
            n_samples: n,
        };

        node_idx
    }

    /// Compute marginal prediction for a given feature subset.
    ///
    /// Features in `subset` use values from `feature_values`.
    /// Features not in `subset` are marginalized by weighting branches
    /// proportionally to their training-data fractions.
    fn marginal_predict(&self, subset: &[usize], feature_values: &[f64]) -> f64 {
        self.marginal_predict_at(0, subset, feature_values)
    }

    #[allow(clippy::cast_precision_loss)]
    fn marginal_predict_at(&self, idx: usize, subset: &[usize], vals: &[f64]) -> f64 {
        match self.nodes[idx] {
            TreeNode::Leaf { value, .. } => value,
            TreeNode::Split {
                feature,
                threshold,
                left,
                right,
                n_samples,
            } => {
                if subset.contains(&feature) {
                    if vals[feature] <= threshold {
                        self.marginal_predict_at(left, subset, vals)
                    } else {
                        self.marginal_predict_at(right, subset, vals)
                    }
                } else {
                    let l_n = self.n_samples(left) as f64;
                    let r_n = self.n_samples(right) as f64;
                    let total = n_samples as f64;
                    (l_n / total) * self.marginal_predict_at(left, subset, vals)
                        + (r_n / total) * self.marginal_predict_at(right, subset, vals)
                }
            }
        }
    }

    fn n_samples(&self, idx: usize) -> usize {
        match self.nodes[idx] {
            TreeNode::Leaf { n_samples, .. } | TreeNode::Split { n_samples, .. } => n_samples,
        }
    }
}

// --- Helper Functions ---

/// Select `k` random indices from `0..n` using partial Fisher-Yates shuffle.
fn partial_shuffle(n: usize, k: usize, rng: &mut fastrand::Rng) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    let k = k.min(n);
    for i in 0..k {
        let j = rng.usize(i..n);
        indices.swap(i, j);
    }
    indices.truncate(k);
    indices
}

/// Compute left/right split statistics for variance reduction.
#[allow(clippy::cast_precision_loss)]
fn split_stats(
    data: &[Vec<f64>],
    targets: &[f64],
    indices: &[usize],
    feature: usize,
    threshold: f64,
) -> (f64, f64, usize, f64, f64, usize) {
    let (mut l_sum, mut l_sq, mut l_n) = (0.0, 0.0, 0usize);
    let (mut r_sum, mut r_sq, mut r_n) = (0.0, 0.0, 0usize);

    for &i in indices {
        let y = targets[i];
        if data[i][feature] <= threshold {
            l_sum += y;
            l_sq += y * y;
            l_n += 1;
        } else {
            r_sum += y;
            r_sq += y * y;
            r_n += 1;
        }
    }

    (l_sum, l_sq, l_n, r_sum, r_sq, r_n)
}

/// Population variance of a slice.
#[allow(clippy::cast_precision_loss)]
fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n
}

// --- Public API ---

/// Run fANOVA analysis on pre-processed numerical data.
///
/// `data` is `n_samples` rows, each with `n_features` columns.
/// `targets` has one entry per sample.
/// `feature_names` maps feature index to human-readable name.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn compute_fanova(
    data: &[Vec<f64>],
    targets: &[f64],
    feature_names: &[String],
    config: &FanovaConfig,
) -> FanovaResult {
    let n_samples = data.len();
    let n_features = data[0].len();

    let mut rng: fastrand::Rng = config
        .seed
        .map_or_else(fastrand::Rng::new, fastrand::Rng::with_seed);

    // Build random forest with bootstrap sampling
    let trees: Vec<DecisionTree> = (0..config.n_trees)
        .map(|_| {
            let bootstrap: Vec<usize> = (0..n_samples).map(|_| rng.usize(0..n_samples)).collect();
            DecisionTree::build(data, targets, &bootstrap, config, &mut rng)
        })
        .collect();

    // Compute main effects: V_j = Var[E[f | x_j]]
    let main_var: Vec<f64> = (0..n_features)
        .map(|j| {
            let subset = [j];
            let preds: Vec<f64> = (0..n_samples)
                .map(|i| {
                    trees
                        .iter()
                        .map(|t| t.marginal_predict(&subset, &data[i]))
                        .sum::<f64>()
                        / trees.len() as f64
                })
                .collect();
            variance(&preds)
        })
        .collect();

    // Compute pairwise interaction effects: V_{j,k} - V_j - V_k
    let mut interactions: Vec<((String, String), f64)> = Vec::new();
    for j in 0..n_features {
        for k in (j + 1)..n_features {
            let subset = [j, k];
            let preds: Vec<f64> = (0..n_samples)
                .map(|i| {
                    trees
                        .iter()
                        .map(|t| t.marginal_predict(&subset, &data[i]))
                        .sum::<f64>()
                        / trees.len() as f64
                })
                .collect();
            let joint = variance(&preds);
            let interaction = (joint - main_var[j] - main_var[k]).max(0.0);
            if interaction > 1e-10 {
                interactions.push((
                    (feature_names[j].clone(), feature_names[k].clone()),
                    interaction,
                ));
            }
        }
    }

    // Normalize so all importances sum to 1.0
    let total: f64 =
        main_var.iter().sum::<f64>() + interactions.iter().map(|(_, v)| *v).sum::<f64>();

    let mut main_effects: Vec<(String, f64)> = feature_names
        .iter()
        .zip(&main_var)
        .map(|(name, &v)| (name.clone(), if total > 0.0 { v / total } else { 0.0 }))
        .collect();
    main_effects.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    if total > 0.0 {
        for entry in &mut interactions {
            entry.1 /= total;
        }
    }
    interactions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    FanovaResult {
        main_effects,
        interactions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng_util;

    #[test]
    fn single_dominant_parameter() {
        // f(x, y) = x — only x matters
        let mut rng = fastrand::Rng::with_seed(0);
        let n = 100;
        let data: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                vec![
                    rng_util::f64_range(&mut rng, 0.0, 10.0),
                    rng_util::f64_range(&mut rng, 0.0, 10.0),
                ]
            })
            .collect();
        let targets: Vec<f64> = data.iter().map(|row| row[0]).collect();

        let result = compute_fanova(
            &data,
            &targets,
            &["x".into(), "y".into()],
            &FanovaConfig::default(),
        );

        assert_eq!(result.main_effects[0].0, "x");
        assert!(
            result.main_effects[0].1 > 0.8,
            "x importance = {}",
            result.main_effects[0].1
        );
    }

    #[test]
    fn interaction_detection() {
        // f(x, y) = x * y — both matter and interact
        let mut rng = fastrand::Rng::with_seed(42);
        let n = 200;
        let data: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                vec![
                    rng_util::f64_range(&mut rng, 0.0, 10.0),
                    rng_util::f64_range(&mut rng, 0.0, 10.0),
                ]
            })
            .collect();
        let targets: Vec<f64> = data.iter().map(|row| row[0] * row[1]).collect();

        let config = FanovaConfig {
            n_trees: 128,
            ..FanovaConfig::default()
        };
        let result = compute_fanova(&data, &targets, &["x".into(), "y".into()], &config);

        assert!(
            !result.interactions.is_empty(),
            "should detect x*y interaction"
        );
        assert!(
            result.interactions[0].1 > 0.05,
            "interaction importance = {}",
            result.interactions[0].1
        );
    }

    #[test]
    fn variance_computation() {
        assert!((variance(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 2.0).abs() < 1e-10);
        assert!(variance(&[5.0, 5.0, 5.0]).abs() < 1e-10);
        assert!(variance(&[]).abs() < 1e-10);
    }

    #[test]
    fn three_params_one_dominant() {
        // f(x, y, z) = 3*x + 0.1*y + 0*z
        let mut rng = fastrand::Rng::with_seed(7);
        let n = 150;
        let data: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                vec![
                    rng_util::f64_range(&mut rng, 0.0, 10.0),
                    rng_util::f64_range(&mut rng, 0.0, 10.0),
                    rng_util::f64_range(&mut rng, 0.0, 10.0),
                ]
            })
            .collect();
        let targets: Vec<f64> = data.iter().map(|r| 3.0 * r[0] + 0.1 * r[1]).collect();

        let result = compute_fanova(
            &data,
            &targets,
            &["x".into(), "y".into(), "z".into()],
            &FanovaConfig::default(),
        );

        // x should be the most important
        assert_eq!(result.main_effects[0].0, "x");
        assert!(result.main_effects[0].1 > 0.5);

        // z should have near-zero importance
        let z_imp = result
            .main_effects
            .iter()
            .find(|(name, _)| name == "z")
            .map_or(0.0, |(_, v)| *v);
        assert!(z_imp < 0.1, "z importance = {z_imp}");
    }

    #[test]
    fn importances_sum_to_one() {
        let mut rng = fastrand::Rng::with_seed(3);
        let n = 100;
        let data: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                vec![
                    rng_util::f64_range(&mut rng, 0.0, 10.0),
                    rng_util::f64_range(&mut rng, 0.0, 10.0),
                ]
            })
            .collect();
        let targets: Vec<f64> = data.iter().map(|r| r[0] + r[1]).collect();

        let result = compute_fanova(
            &data,
            &targets,
            &["x".into(), "y".into()],
            &FanovaConfig::default(),
        );

        let total: f64 = result.main_effects.iter().map(|(_, v)| *v).sum::<f64>()
            + result.interactions.iter().map(|(_, v)| *v).sum::<f64>();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "importances should sum to 1.0, got {total}"
        );
    }
}
