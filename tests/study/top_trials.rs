use optimizer::{Direction, Study};

#[test]
fn test_top_trials_minimize() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    // Manually complete trials with known values
    for &val in &[5.0, 1.0, 3.0, 2.0, 4.0] {
        let trial = study.create_trial();
        study.complete_trial(trial, val);
    }

    let top3 = study.top_trials(3);
    assert_eq!(top3.len(), 3);
    assert_eq!(top3[0].value, 1.0);
    assert_eq!(top3[1].value, 2.0);
    assert_eq!(top3[2].value, 3.0);
}

#[test]
fn test_top_trials_maximize() {
    let study: Study<f64> = Study::new(Direction::Maximize);

    for &val in &[5.0, 1.0, 3.0, 2.0, 4.0] {
        let trial = study.create_trial();
        study.complete_trial(trial, val);
    }

    let top3 = study.top_trials(3);
    assert_eq!(top3.len(), 3);
    assert_eq!(top3[0].value, 5.0);
    assert_eq!(top3[1].value, 4.0);
    assert_eq!(top3[2].value, 3.0);
}

#[test]
fn test_top_trials_n_greater_than_total() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    for &val in &[3.0, 1.0] {
        let trial = study.create_trial();
        study.complete_trial(trial, val);
    }

    let top = study.top_trials(10);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].value, 1.0);
    assert_eq!(top[1].value, 3.0);
}

#[test]
fn test_top_trials_empty() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    let top = study.top_trials(5);
    assert!(top.is_empty());
}

#[test]
fn test_top_trials_excludes_pruned() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    // Complete some trials
    for &val in &[5.0, 1.0, 3.0] {
        let trial = study.create_trial();
        study.complete_trial(trial, val);
    }

    // Prune a trial (it gets a default value of 0.0 but should be excluded)
    let trial = study.create_trial();
    study.prune_trial(trial);

    let top = study.top_trials(5);
    assert_eq!(top.len(), 3, "pruned trial should be excluded");
    assert_eq!(top[0].value, 1.0);
}
