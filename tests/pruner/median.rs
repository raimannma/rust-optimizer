use std::collections::HashMap;

use optimizer::Direction;
use optimizer::pruner::{MedianPruner, Pruner};
use optimizer::sampler::CompletedTrial;

/// Helper to build a completed trial with given intermediate values.
fn trial_with_values(id: u64, intermediate_values: Vec<(u64, f64)>) -> CompletedTrial {
    CompletedTrial::with_intermediate_values(
        id,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        0.0,
        intermediate_values,
        HashMap::new(),
    )
}

// --- Minimize direction ---

#[test]
fn prune_when_worse_than_median_minimize() {
    let pruner = MedianPruner::new(Direction::Minimize);
    // 3 completed trials with values at step 2: [1.0, 2.0, 3.0] => median = 2.0
    let completed = vec![
        trial_with_values(0, vec![(0, 0.5), (1, 0.8), (2, 1.0)]),
        trial_with_values(1, vec![(0, 0.6), (1, 1.5), (2, 2.0)]),
        trial_with_values(2, vec![(0, 0.7), (1, 2.0), (2, 3.0)]),
    ];
    // Current trial value at step 2 is 2.5 > median 2.0 => prune
    let current = vec![(0, 0.5), (1, 1.0), (2, 2.5)];
    assert!(pruner.should_prune(3, 2, &current, &completed));
}

#[test]
fn no_prune_when_better_than_median_minimize() {
    let pruner = MedianPruner::new(Direction::Minimize);
    let completed = vec![
        trial_with_values(0, vec![(0, 0.5), (1, 0.8), (2, 1.0)]),
        trial_with_values(1, vec![(0, 0.6), (1, 1.5), (2, 2.0)]),
        trial_with_values(2, vec![(0, 0.7), (1, 2.0), (2, 3.0)]),
    ];
    // Current trial value at step 2 is 1.5 < median 2.0 => don't prune
    let current = vec![(0, 0.5), (1, 1.0), (2, 1.5)];
    assert!(!pruner.should_prune(3, 2, &current, &completed));
}

// --- Maximize direction ---

#[test]
fn prune_when_worse_than_median_maximize() {
    let pruner = MedianPruner::new(Direction::Maximize);
    // Values at step 1: [5.0, 7.0, 9.0] => median = 7.0
    let completed = vec![
        trial_with_values(0, vec![(0, 3.0), (1, 5.0)]),
        trial_with_values(1, vec![(0, 4.0), (1, 7.0)]),
        trial_with_values(2, vec![(0, 5.0), (1, 9.0)]),
    ];
    // Current value 6.0 < median 7.0 => prune (worse for maximize)
    let current = vec![(0, 4.0), (1, 6.0)];
    assert!(pruner.should_prune(3, 1, &current, &completed));
}

#[test]
fn no_prune_when_better_than_median_maximize() {
    let pruner = MedianPruner::new(Direction::Maximize);
    let completed = vec![
        trial_with_values(0, vec![(0, 3.0), (1, 5.0)]),
        trial_with_values(1, vec![(0, 4.0), (1, 7.0)]),
        trial_with_values(2, vec![(0, 5.0), (1, 9.0)]),
    ];
    // Current value 8.0 > median 7.0 => don't prune
    let current = vec![(0, 4.0), (1, 8.0)];
    assert!(!pruner.should_prune(3, 1, &current, &completed));
}

// --- Warmup steps ---

#[test]
fn no_prune_during_warmup() {
    let pruner = MedianPruner::new(Direction::Minimize).n_warmup_steps(5);
    let completed = vec![trial_with_values(0, vec![(0, 1.0), (1, 1.0), (2, 1.0)])];
    // Step 2 < warmup 5 => never prune, even if value is terrible
    let current = vec![(0, 100.0), (1, 100.0), (2, 100.0)];
    assert!(!pruner.should_prune(1, 2, &current, &completed));
}

#[test]
fn prune_after_warmup() {
    let pruner = MedianPruner::new(Direction::Minimize).n_warmup_steps(2);
    let completed = vec![trial_with_values(0, vec![(0, 1.0), (1, 1.0), (2, 1.0)])];
    // Step 2 >= warmup 2 => pruning allowed; current 100.0 > median 1.0
    let current = vec![(0, 100.0), (1, 100.0), (2, 100.0)];
    assert!(pruner.should_prune(1, 2, &current, &completed));
}

// --- n_min_trials ---

#[test]
fn no_prune_when_fewer_than_n_min_trials() {
    let pruner = MedianPruner::new(Direction::Minimize).n_min_trials(3);
    // Only 2 completed trials — below threshold of 3
    let completed = vec![
        trial_with_values(0, vec![(0, 1.0)]),
        trial_with_values(1, vec![(0, 2.0)]),
    ];
    let current = vec![(0, 100.0)];
    assert!(!pruner.should_prune(2, 0, &current, &completed));
}

#[test]
fn prune_when_at_least_n_min_trials() {
    let pruner = MedianPruner::new(Direction::Minimize).n_min_trials(3);
    // 3 completed trials with step 0: [1.0, 2.0, 3.0] => median 2.0
    let completed = vec![
        trial_with_values(0, vec![(0, 1.0)]),
        trial_with_values(1, vec![(0, 2.0)]),
        trial_with_values(2, vec![(0, 3.0)]),
    ];
    // 5.0 > median 2.0 => prune
    let current = vec![(0, 5.0)];
    assert!(pruner.should_prune(3, 0, &current, &completed));
}

// --- No completed trials with values at step ---

#[test]
fn no_prune_when_no_completed_trials_at_step() {
    let pruner = MedianPruner::new(Direction::Minimize);
    // Completed trials only have values at step 0, not step 5
    let completed = vec![
        trial_with_values(0, vec![(0, 1.0)]),
        trial_with_values(1, vec![(0, 2.0)]),
    ];
    let current = vec![(0, 0.5), (5, 100.0)];
    assert!(!pruner.should_prune(2, 5, &current, &completed));
}

// --- Median calculation edge cases ---

#[test]
fn correct_median_with_even_number_of_trials() {
    let pruner = MedianPruner::new(Direction::Minimize);
    // 4 trials at step 0: [1.0, 2.0, 3.0, 4.0] => median = 2.5
    let completed = vec![
        trial_with_values(0, vec![(0, 1.0)]),
        trial_with_values(1, vec![(0, 2.0)]),
        trial_with_values(2, vec![(0, 3.0)]),
        trial_with_values(3, vec![(0, 4.0)]),
    ];
    // 2.6 > 2.5 => prune
    let current = vec![(0, 2.6)];
    assert!(pruner.should_prune(4, 0, &current, &completed));
    // 2.4 < 2.5 => don't prune
    let current = vec![(0, 2.4)];
    assert!(!pruner.should_prune(4, 0, &current, &completed));
}

#[test]
fn correct_median_with_odd_number_of_trials() {
    let pruner = MedianPruner::new(Direction::Minimize);
    // 5 trials at step 0: [1.0, 2.0, 3.0, 4.0, 5.0] => median = 3.0
    let completed = vec![
        trial_with_values(0, vec![(0, 1.0)]),
        trial_with_values(1, vec![(0, 2.0)]),
        trial_with_values(2, vec![(0, 3.0)]),
        trial_with_values(3, vec![(0, 4.0)]),
        trial_with_values(4, vec![(0, 5.0)]),
    ];
    // 3.5 > 3.0 => prune
    let current = vec![(0, 3.5)];
    assert!(pruner.should_prune(5, 0, &current, &completed));
    // 2.5 < 3.0 => don't prune
    let current = vec![(0, 2.5)];
    assert!(!pruner.should_prune(5, 0, &current, &completed));
}

// --- Non-contiguous step numbers ---

#[test]
fn works_with_non_contiguous_steps() {
    let pruner = MedianPruner::new(Direction::Minimize);
    // Steps are 0, 10, 100 — non-contiguous
    let completed = vec![
        trial_with_values(0, vec![(0, 1.0), (10, 2.0), (100, 3.0)]),
        trial_with_values(1, vec![(0, 1.5), (10, 2.5), (100, 4.0)]),
        trial_with_values(2, vec![(0, 2.0), (10, 3.0), (100, 5.0)]),
    ];
    // At step 100: [3.0, 4.0, 5.0] => median = 4.0
    let current = vec![(0, 1.0), (10, 2.0), (100, 4.5)];
    assert!(pruner.should_prune(3, 100, &current, &completed));

    let current = vec![(0, 1.0), (10, 2.0), (100, 3.5)];
    assert!(!pruner.should_prune(3, 100, &current, &completed));
}

// --- No intermediate values for current trial ---

#[test]
fn no_prune_when_no_intermediate_values() {
    let pruner = MedianPruner::new(Direction::Minimize);
    let completed = vec![trial_with_values(0, vec![(0, 1.0)])];
    assert!(!pruner.should_prune(1, 0, &[], &completed));
}

// --- Pruned trials are excluded from median calculation ---

#[test]
fn pruned_trials_excluded_from_median() {
    use optimizer::TrialState;

    let pruner = MedianPruner::new(Direction::Minimize);

    let mut pruned = trial_with_values(0, vec![(0, 0.1)]);
    pruned.state = TrialState::Pruned;

    // Only the completed trial (value 5.0) counts. Pruned trial (0.1) is excluded.
    let completed = vec![pruned, trial_with_values(1, vec![(0, 5.0)])];

    // 3.0 < 5.0 => don't prune (only 1 completed trial with median 5.0)
    let current = vec![(0, 3.0)];
    assert!(!pruner.should_prune(2, 0, &current, &completed));

    // 6.0 > 5.0 => prune
    let current = vec![(0, 6.0)];
    assert!(pruner.should_prune(2, 0, &current, &completed));
}
