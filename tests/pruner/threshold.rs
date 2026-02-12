use optimizer::pruner::{Pruner, ThresholdPruner};

#[test]
fn prune_when_value_exceeds_upper_threshold() {
    let pruner = ThresholdPruner::new().upper(10.0);
    let values = vec![(0, 5.0), (1, 8.0), (2, 11.0)];
    assert!(pruner.should_prune(0, 2, &values, &[]));
}

#[test]
fn prune_when_value_falls_below_lower_threshold() {
    let pruner = ThresholdPruner::new().lower(0.0);
    let values = vec![(0, 5.0), (1, 2.0), (2, -1.0)];
    assert!(pruner.should_prune(0, 2, &values, &[]));
}

#[test]
fn no_prune_when_value_within_bounds() {
    let pruner = ThresholdPruner::new().upper(10.0).lower(0.0);
    let values = vec![(0, 3.0), (1, 5.0), (2, 7.0)];
    assert!(!pruner.should_prune(0, 2, &values, &[]));
}

#[test]
fn no_prune_when_no_intermediate_values() {
    let pruner = ThresholdPruner::new().upper(10.0).lower(0.0);
    assert!(!pruner.should_prune(0, 0, &[], &[]));
}

#[test]
fn works_with_only_upper_set() {
    let pruner = ThresholdPruner::new().upper(5.0);
    let below = vec![(0, 3.0)];
    let above = vec![(0, 6.0)];
    assert!(!pruner.should_prune(0, 0, &below, &[]));
    assert!(pruner.should_prune(0, 0, &above, &[]));
}

#[test]
fn works_with_only_lower_set() {
    let pruner = ThresholdPruner::new().lower(2.0);
    let above = vec![(0, 5.0)];
    let below = vec![(0, 1.0)];
    assert!(!pruner.should_prune(0, 0, &above, &[]));
    assert!(pruner.should_prune(0, 0, &below, &[]));
}

#[test]
fn works_with_both_thresholds_set() {
    let pruner = ThresholdPruner::new().upper(10.0).lower(0.0);

    // Within bounds
    assert!(!pruner.should_prune(0, 0, &[(0, 5.0)], &[]));

    // Exceeds upper
    assert!(pruner.should_prune(0, 0, &[(0, 15.0)], &[]));

    // Below lower
    assert!(pruner.should_prune(0, 0, &[(0, -3.0)], &[]));

    // At exact boundary (not pruned — strictly greater/less)
    assert!(!pruner.should_prune(0, 0, &[(0, 10.0)], &[]));
    assert!(!pruner.should_prune(0, 0, &[(0, 0.0)], &[]));
}
