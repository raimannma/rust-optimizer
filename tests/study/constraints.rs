use optimizer::{Direction, Study};

#[test]
fn test_is_feasible_all_satisfied() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let mut trial = study.create_trial();
    trial.set_constraints(vec![-1.0, 0.0, -0.5]);
    study.complete_trial(trial, 1.0);

    let completed = study.best_trial().unwrap();
    assert!(completed.is_feasible());
}

#[test]
fn test_is_feasible_one_violated() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let mut trial = study.create_trial();
    trial.set_constraints(vec![-1.0, 0.5, -0.5]);
    study.complete_trial(trial, 1.0);

    let completed = study.best_trial().unwrap();
    assert!(!completed.is_feasible());
}

#[test]
fn test_is_feasible_empty_constraints() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let trial = study.create_trial();
    study.complete_trial(trial, 1.0);

    let completed = study.best_trial().unwrap();
    assert!(completed.is_feasible());
}

#[test]
fn test_best_trial_prefers_feasible() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    // Infeasible trial with better objective
    let mut trial1 = study.create_trial();
    trial1.set_constraints(vec![1.0]);
    study.complete_trial(trial1, 0.1);

    // Feasible trial with worse objective
    let mut trial2 = study.create_trial();
    trial2.set_constraints(vec![-1.0]);
    study.complete_trial(trial2, 100.0);

    let best = study.best_trial().unwrap();
    assert_eq!(best.id, 1); // feasible trial wins
    assert_eq!(best.value, 100.0);
}

#[test]
fn test_best_trial_feasible_by_objective() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    // Feasible, worse objective
    let mut trial1 = study.create_trial();
    trial1.set_constraints(vec![-1.0]);
    study.complete_trial(trial1, 10.0);

    // Feasible, better objective
    let mut trial2 = study.create_trial();
    trial2.set_constraints(vec![-0.5]);
    study.complete_trial(trial2, 2.0);

    let best = study.best_trial().unwrap();
    assert_eq!(best.id, 1); // lower objective wins among feasible
    assert_eq!(best.value, 2.0);
}

#[test]
fn test_top_trials_ranks_feasible_above_infeasible() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    // Infeasible, low violation
    let mut t0 = study.create_trial();
    t0.set_constraints(vec![0.5]);
    study.complete_trial(t0, 1.0);

    // Feasible, worst objective among feasible
    let mut t1 = study.create_trial();
    t1.set_constraints(vec![-1.0]);
    study.complete_trial(t1, 50.0);

    // Feasible, best objective among feasible
    let mut t2 = study.create_trial();
    t2.set_constraints(vec![-0.1]);
    study.complete_trial(t2, 5.0);

    // Infeasible, high violation
    let mut t3 = study.create_trial();
    t3.set_constraints(vec![3.0]);
    study.complete_trial(t3, 0.5);

    let top = study.top_trials(4);
    let ids: Vec<u64> = top.iter().map(|t| t.id).collect();
    // Feasible sorted by objective first (5.0, 50.0), then infeasible by violation (0.5, 3.0)
    assert_eq!(ids, vec![2, 1, 0, 3]);
}
