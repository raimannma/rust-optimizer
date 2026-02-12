//! Integration tests for multi-objective optimization.

use optimizer::Direction;
use optimizer::multi_objective::MultiObjectiveStudy;
use optimizer::parameter::{CategoricalParam, FloatParam, Parameter};
use optimizer::sampler::Decomposition;
use optimizer::sampler::moead::MoeadSampler;
use optimizer::sampler::nsga2::Nsga2Sampler;
use optimizer::sampler::nsga3::Nsga3Sampler;

// ---------------------------------------------------------------------------
// Pareto utility tests (via public MultiObjectiveStudy)
// ---------------------------------------------------------------------------

#[test]
fn test_basic_two_objective_random() {
    let study = MultiObjectiveStudy::new(vec![Direction::Minimize, Direction::Minimize]);
    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(30, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(vec![xv, 1.0 - xv])
        })
        .unwrap();

    let front = study.pareto_front();
    assert!(!front.is_empty(), "Pareto front should be non-empty");

    // Verify no solution in the front dominates another
    for a in &front {
        for b in &front {
            if core::ptr::eq(a, b) {
                continue;
            }
            let a_dom_b = a.values[0] <= b.values[0]
                && a.values[1] <= b.values[1]
                && (a.values[0] < b.values[0] || a.values[1] < b.values[1]);
            assert!(
                !a_dom_b,
                "Front solution {:?} dominates {:?}",
                a.values, b.values
            );
        }
    }
}

#[test]
fn test_dimension_mismatch_error() {
    let study = MultiObjectiveStudy::new(vec![Direction::Minimize, Direction::Minimize]);
    let x = FloatParam::new(0.0, 1.0);

    let result = study.optimize(1, |trial: &mut optimizer::Trial| {
        let xv = x.suggest(trial)?;
        // Return wrong number of values
        Ok::<_, optimizer::Error>(vec![xv])
    });

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(
            err,
            optimizer::Error::ObjectiveDimensionMismatch {
                expected: 2,
                got: 1
            }
        ),
        "Expected ObjectiveDimensionMismatch, got: {err}"
    );
}

#[test]
fn test_ask_tell() {
    let study = MultiObjectiveStudy::new(vec![Direction::Minimize, Direction::Maximize]);
    let x = FloatParam::new(0.0, 10.0);

    for _ in 0..10 {
        let mut trial = study.ask();
        let xv = x.suggest(&mut trial).unwrap();
        study
            .tell(trial, Ok::<_, &str>(vec![xv, 10.0 - xv]))
            .unwrap();
    }

    assert_eq!(study.n_trials(), 10);
    let front = study.pareto_front();
    assert!(!front.is_empty());
}

#[test]
fn test_ask_tell_dimension_mismatch() {
    let study = MultiObjectiveStudy::new(vec![Direction::Minimize, Direction::Minimize]);
    let trial = study.ask();
    let result = study.tell(trial, Ok::<_, &str>(vec![1.0, 2.0, 3.0]));
    assert!(result.is_err());
}

#[test]
fn test_n_trials_counting() {
    let study = MultiObjectiveStudy::new(vec![Direction::Minimize, Direction::Minimize]);
    assert_eq!(study.n_trials(), 0);

    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(5, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(vec![xv, 1.0 - xv])
        })
        .unwrap();

    assert_eq!(study.n_trials(), 5);
}

#[test]
fn test_three_objectives() {
    let study = MultiObjectiveStudy::new(vec![
        Direction::Minimize,
        Direction::Minimize,
        Direction::Maximize,
    ]);
    let x = FloatParam::new(0.0, 1.0);
    let y = FloatParam::new(0.0, 1.0);

    study
        .optimize(30, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            let yv = y.suggest(trial)?;
            Ok::<_, optimizer::Error>(vec![xv, yv, 1.0 - xv - yv])
        })
        .unwrap();

    let front = study.pareto_front();
    assert!(!front.is_empty());
    assert_eq!(study.n_objectives(), 3);
}

#[test]
fn test_directions_accessor() {
    let dirs = vec![Direction::Minimize, Direction::Maximize];
    let study = MultiObjectiveStudy::new(dirs.clone());
    assert_eq!(study.directions(), &dirs);
    assert_eq!(study.n_objectives(), 2);
}

#[test]
fn test_trials_accessor() {
    let study = MultiObjectiveStudy::new(vec![Direction::Minimize, Direction::Minimize]);
    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(3, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(vec![xv, 1.0 - xv])
        })
        .unwrap();

    let trials = study.trials();
    assert_eq!(trials.len(), 3);
    for t in &trials {
        assert_eq!(t.values.len(), 2);
    }
}

// ---------------------------------------------------------------------------
// NSGA-II sampler tests
// ---------------------------------------------------------------------------

#[test]
fn test_nsga2_zdt1() {
    // ZDT1 benchmark: minimize both objectives
    let n_vars = 5;
    let params: Vec<FloatParam> = (0..n_vars).map(|_| FloatParam::new(0.0, 1.0)).collect();

    let sampler = Nsga2Sampler::builder().population_size(20).seed(42).build();
    let study =
        MultiObjectiveStudy::with_sampler(vec![Direction::Minimize, Direction::Minimize], sampler);

    study
        .optimize(200, |trial: &mut optimizer::Trial| {
            let xs: Vec<f64> = params
                .iter()
                .map(|p| p.suggest(trial))
                .collect::<Result<_, _>>()?;

            let f1 = xs[0];
            let g = 1.0 + 9.0 * xs[1..].iter().sum::<f64>() / (n_vars - 1) as f64;
            let f2 = g * (1.0 - (f1 / g).sqrt());
            Ok::<_, optimizer::Error>(vec![f1, f2])
        })
        .unwrap();

    let front = study.pareto_front();
    assert!(!front.is_empty(), "Pareto front should be non-empty");

    // Verify no dominated solutions in the front
    for a in &front {
        for b in &front {
            if core::ptr::eq(a, b) {
                continue;
            }
            let a_dom_b = a.values[0] <= b.values[0]
                && a.values[1] <= b.values[1]
                && (a.values[0] < b.values[0] || a.values[1] < b.values[1]);
            assert!(
                !a_dom_b,
                "Front solution {:?} dominates {:?}",
                a.values, b.values
            );
        }
    }
}

#[test]
fn test_nsga2_with_seed_reproducible() {
    let x = FloatParam::new(0.0, 1.0);
    let y = FloatParam::new(0.0, 1.0);

    let run = |seed: u64| -> Vec<Vec<f64>> {
        let sampler = Nsga2Sampler::with_seed(seed);
        let study = MultiObjectiveStudy::with_sampler(
            vec![Direction::Minimize, Direction::Minimize],
            sampler,
        );
        study
            .optimize(30, |trial: &mut optimizer::Trial| {
                let xv = x.suggest(trial)?;
                let yv = y.suggest(trial)?;
                Ok::<_, optimizer::Error>(vec![xv, yv])
            })
            .unwrap();
        study.trials().iter().map(|t| t.values.clone()).collect()
    };

    let r1 = run(123);
    let r2 = run(123);
    assert_eq!(r1, r2, "Same seed should produce same results");

    let r3 = run(456);
    assert_ne!(r1, r3, "Different seeds should produce different results");
}

#[test]
fn test_nsga2_builder() {
    let sampler = Nsga2Sampler::builder()
        .population_size(10)
        .crossover_prob(0.8)
        .crossover_eta(15.0)
        .mutation_eta(25.0)
        .seed(42)
        .build();

    let study =
        MultiObjectiveStudy::with_sampler(vec![Direction::Minimize, Direction::Minimize], sampler);
    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(30, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(vec![xv, 1.0 - xv])
        })
        .unwrap();

    assert_eq!(study.n_trials(), 30);
}

#[test]
fn test_nsga2_categorical_params() {
    let sampler = Nsga2Sampler::with_seed(42);
    let study =
        MultiObjectiveStudy::with_sampler(vec![Direction::Minimize, Direction::Minimize], sampler);

    let x = FloatParam::new(0.0, 1.0);
    let cat = CategoricalParam::new(vec!["a", "b", "c"]);

    study
        .optimize(30, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            let cv = cat.suggest(trial)?;
            let bonus = match cv {
                "a" => 0.0,
                "b" => 0.5,
                _ => 1.0,
            };
            Ok::<_, optimizer::Error>(vec![xv + bonus, 1.0 - xv])
        })
        .unwrap();

    assert_eq!(study.n_trials(), 30);
    let front = study.pareto_front();
    assert!(!front.is_empty());
}

#[test]
fn test_nsga2_constraints() {
    let sampler = Nsga2Sampler::with_seed(42);
    let study =
        MultiObjectiveStudy::with_sampler(vec![Direction::Minimize, Direction::Minimize], sampler);

    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(50, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            // Constraint: x >= 0.3 (i.e. 0.3 - x <= 0)
            trial.set_constraints(vec![0.3 - xv]);
            Ok::<_, optimizer::Error>(vec![xv, 1.0 - xv])
        })
        .unwrap();

    let front = study.pareto_front();
    assert!(!front.is_empty());

    // Check that feasible solutions exist on the front
    let feasible_count = front.iter().filter(|t| t.is_feasible()).count();
    assert!(
        feasible_count > 0,
        "Should have feasible solutions on front"
    );
}

#[test]
fn test_multi_objective_trial_get() {
    let study = MultiObjectiveStudy::new(vec![Direction::Minimize, Direction::Minimize]);
    let x = FloatParam::new(0.0, 10.0).name("x");

    study
        .optimize(5, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(vec![xv, 10.0 - xv])
        })
        .unwrap();

    let front = study.pareto_front();
    for t in &front {
        let xv: f64 = t.get(&x).unwrap();
        assert!((0.0..=10.0).contains(&xv));
    }
}

#[test]
fn test_multi_objective_trial_is_feasible() {
    let study = MultiObjectiveStudy::new(vec![Direction::Minimize, Direction::Minimize]);
    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(10, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            trial.set_constraints(vec![0.5 - xv]); // feasible if x >= 0.5
            Ok::<_, optimizer::Error>(vec![xv, 1.0 - xv])
        })
        .unwrap();

    let trials = study.trials();
    for t in &trials {
        let xv = t.values[0];
        if xv >= 0.5 {
            assert!(t.is_feasible());
        } else {
            assert!(!t.is_feasible());
        }
    }
}

#[test]
fn test_multi_objective_trial_user_attrs() {
    let study = MultiObjectiveStudy::new(vec![Direction::Minimize, Direction::Minimize]);
    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(3, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            trial.set_user_attr("iteration", 42_i64);
            Ok::<_, optimizer::Error>(vec![xv, 1.0 - xv])
        })
        .unwrap();

    let trials = study.trials();
    for t in &trials {
        assert!(t.user_attr("iteration").is_some());
    }
}

#[test]
fn test_tell_with_failure() {
    let study = MultiObjectiveStudy::new(vec![Direction::Minimize, Direction::Minimize]);

    let trial = study.ask();
    study
        .tell(trial, Err::<Vec<f64>, _>("evaluation failed"))
        .unwrap();

    // Failed trial not counted
    assert_eq!(study.n_trials(), 0);
}

// ---------------------------------------------------------------------------
// NSGA-III sampler tests
// ---------------------------------------------------------------------------

#[test]
fn test_nsga3_zdt1() {
    let n_vars = 5;
    let params: Vec<FloatParam> = (0..n_vars).map(|_| FloatParam::new(0.0, 1.0)).collect();

    let sampler = Nsga3Sampler::builder().population_size(20).seed(42).build();
    let study =
        MultiObjectiveStudy::with_sampler(vec![Direction::Minimize, Direction::Minimize], sampler);

    study
        .optimize(200, |trial: &mut optimizer::Trial| {
            let xs: Vec<f64> = params
                .iter()
                .map(|p| p.suggest(trial))
                .collect::<Result<_, _>>()?;

            let f1 = xs[0];
            let g = 1.0 + 9.0 * xs[1..].iter().sum::<f64>() / (n_vars - 1) as f64;
            let f2 = g * (1.0 - (f1 / g).sqrt());
            Ok::<_, optimizer::Error>(vec![f1, f2])
        })
        .unwrap();

    let front = study.pareto_front();
    assert!(
        !front.is_empty(),
        "NSGA-III Pareto front should be non-empty"
    );

    // Verify no dominated solutions in the front
    for a in &front {
        for b in &front {
            if core::ptr::eq(a, b) {
                continue;
            }
            let a_dom_b = a.values[0] <= b.values[0]
                && a.values[1] <= b.values[1]
                && (a.values[0] < b.values[0] || a.values[1] < b.values[1]);
            assert!(
                !a_dom_b,
                "Front solution {:?} dominates {:?}",
                a.values, b.values
            );
        }
    }
}

#[test]
fn test_nsga3_four_objectives() {
    // DTLZ2 with 4 objectives
    let n_obj = 4;
    let n_vars = n_obj + 4; // k = 5 decision variables beyond the first (n_obj-1)
    let params: Vec<FloatParam> = (0..n_vars).map(|_| FloatParam::new(0.0, 1.0)).collect();

    let sampler = Nsga3Sampler::builder().population_size(50).seed(42).build();
    let directions = vec![Direction::Minimize; n_obj];
    let study = MultiObjectiveStudy::with_sampler(directions, sampler);

    study
        .optimize(500, |trial: &mut optimizer::Trial| {
            let xs: Vec<f64> = params
                .iter()
                .map(|p| p.suggest(trial))
                .collect::<Result<_, _>>()?;

            // DTLZ2 formulation
            let g: f64 = xs[n_obj - 1..]
                .iter()
                .map(|&xi| (xi - 0.5).powi(2))
                .sum::<f64>();

            let mut objectives = vec![0.0_f64; n_obj];
            for i in 0..n_obj {
                let mut f = 1.0 + g;
                for xj in &xs[..(n_obj - 1 - i)] {
                    f *= (xj * core::f64::consts::FRAC_PI_2).cos();
                }
                if i > 0 {
                    f *= (xs[n_obj - 1 - i] * core::f64::consts::FRAC_PI_2).sin();
                }
                objectives[i] = f;
            }

            Ok::<_, optimizer::Error>(objectives)
        })
        .unwrap();

    let front = study.pareto_front();
    assert!(!front.is_empty(), "4-objective front should be non-empty");
    // All front solutions should have 4 objectives
    for t in &front {
        assert_eq!(t.values.len(), 4);
    }
}

#[test]
fn test_nsga3_reproducible() {
    let x = FloatParam::new(0.0, 1.0);
    let y = FloatParam::new(0.0, 1.0);

    let run = |seed: u64| -> Vec<Vec<f64>> {
        let sampler = Nsga3Sampler::with_seed(seed);
        let study = MultiObjectiveStudy::with_sampler(
            vec![Direction::Minimize, Direction::Minimize],
            sampler,
        );
        study
            .optimize(30, |trial: &mut optimizer::Trial| {
                let xv = x.suggest(trial)?;
                let yv = y.suggest(trial)?;
                Ok::<_, optimizer::Error>(vec![xv, yv])
            })
            .unwrap();
        study.trials().iter().map(|t| t.values.clone()).collect()
    };

    let r1 = run(123);
    let r2 = run(123);
    assert_eq!(r1, r2, "Same seed should produce same results");

    let r3 = run(456);
    assert_ne!(r1, r3, "Different seeds should produce different results");
}

#[test]
fn test_nsga3_builder() {
    let sampler = Nsga3Sampler::builder()
        .population_size(12)
        .n_divisions(4)
        .crossover_prob(0.9)
        .crossover_eta(20.0)
        .mutation_eta(20.0)
        .seed(42)
        .build();

    let study =
        MultiObjectiveStudy::with_sampler(vec![Direction::Minimize, Direction::Minimize], sampler);
    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(30, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(vec![xv, 1.0 - xv])
        })
        .unwrap();

    assert_eq!(study.n_trials(), 30);
}

#[test]
fn test_nsga3_constraints() {
    let sampler = Nsga3Sampler::with_seed(42);
    let study =
        MultiObjectiveStudy::with_sampler(vec![Direction::Minimize, Direction::Minimize], sampler);

    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(50, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            trial.set_constraints(vec![0.3 - xv]);
            Ok::<_, optimizer::Error>(vec![xv, 1.0 - xv])
        })
        .unwrap();

    let front = study.pareto_front();
    assert!(!front.is_empty());

    let feasible_count = front.iter().filter(|t| t.is_feasible()).count();
    assert!(
        feasible_count > 0,
        "Should have feasible solutions on front"
    );
}

// ---------------------------------------------------------------------------
// MOEA/D sampler tests
// ---------------------------------------------------------------------------

#[test]
fn test_moead_zdt1_tchebycheff() {
    let n_vars = 5;
    let params: Vec<FloatParam> = (0..n_vars).map(|_| FloatParam::new(0.0, 1.0)).collect();

    let sampler = MoeadSampler::builder().population_size(20).seed(42).build();
    let study =
        MultiObjectiveStudy::with_sampler(vec![Direction::Minimize, Direction::Minimize], sampler);

    study
        .optimize(200, |trial: &mut optimizer::Trial| {
            let xs: Vec<f64> = params
                .iter()
                .map(|p| p.suggest(trial))
                .collect::<Result<_, _>>()?;

            let f1 = xs[0];
            let g = 1.0 + 9.0 * xs[1..].iter().sum::<f64>() / (n_vars - 1) as f64;
            let f2 = g * (1.0 - (f1 / g).sqrt());
            Ok::<_, optimizer::Error>(vec![f1, f2])
        })
        .unwrap();

    let front = study.pareto_front();
    assert!(!front.is_empty(), "MOEA/D Pareto front should be non-empty");

    for a in &front {
        for b in &front {
            if core::ptr::eq(a, b) {
                continue;
            }
            let a_dom_b = a.values[0] <= b.values[0]
                && a.values[1] <= b.values[1]
                && (a.values[0] < b.values[0] || a.values[1] < b.values[1]);
            assert!(
                !a_dom_b,
                "Front solution {:?} dominates {:?}",
                a.values, b.values
            );
        }
    }
}

#[test]
fn test_moead_zdt1_weighted_sum() {
    let n_vars = 3;
    let params: Vec<FloatParam> = (0..n_vars).map(|_| FloatParam::new(0.0, 1.0)).collect();

    let sampler = MoeadSampler::builder()
        .population_size(20)
        .decomposition(Decomposition::WeightedSum)
        .seed(42)
        .build();
    let study =
        MultiObjectiveStudy::with_sampler(vec![Direction::Minimize, Direction::Minimize], sampler);

    study
        .optimize(200, |trial: &mut optimizer::Trial| {
            let xs: Vec<f64> = params
                .iter()
                .map(|p| p.suggest(trial))
                .collect::<Result<_, _>>()?;

            let f1 = xs[0];
            let g = 1.0 + 9.0 * xs[1..].iter().sum::<f64>() / (n_vars - 1) as f64;
            let f2 = g * (1.0 - (f1 / g).sqrt());
            Ok::<_, optimizer::Error>(vec![f1, f2])
        })
        .unwrap();

    let front = study.pareto_front();
    assert!(!front.is_empty());
}

#[test]
fn test_moead_zdt1_pbi() {
    let n_vars = 3;
    let params: Vec<FloatParam> = (0..n_vars).map(|_| FloatParam::new(0.0, 1.0)).collect();

    let sampler = MoeadSampler::builder()
        .population_size(20)
        .decomposition(Decomposition::Pbi { theta: 5.0 })
        .seed(42)
        .build();
    let study =
        MultiObjectiveStudy::with_sampler(vec![Direction::Minimize, Direction::Minimize], sampler);

    study
        .optimize(200, |trial: &mut optimizer::Trial| {
            let xs: Vec<f64> = params
                .iter()
                .map(|p| p.suggest(trial))
                .collect::<Result<_, _>>()?;

            let f1 = xs[0];
            let g = 1.0 + 9.0 * xs[1..].iter().sum::<f64>() / (n_vars - 1) as f64;
            let f2 = g * (1.0 - (f1 / g).sqrt());
            Ok::<_, optimizer::Error>(vec![f1, f2])
        })
        .unwrap();

    let front = study.pareto_front();
    assert!(!front.is_empty());
}

#[test]
fn test_moead_reproducible() {
    let x = FloatParam::new(0.0, 1.0);
    let y = FloatParam::new(0.0, 1.0);

    let run = |seed: u64| -> Vec<Vec<f64>> {
        let sampler = MoeadSampler::with_seed(seed);
        let study = MultiObjectiveStudy::with_sampler(
            vec![Direction::Minimize, Direction::Minimize],
            sampler,
        );
        study
            .optimize(30, |trial: &mut optimizer::Trial| {
                let xv = x.suggest(trial)?;
                let yv = y.suggest(trial)?;
                Ok::<_, optimizer::Error>(vec![xv, yv])
            })
            .unwrap();
        study.trials().iter().map(|t| t.values.clone()).collect()
    };

    let r1 = run(123);
    let r2 = run(123);
    assert_eq!(r1, r2, "Same seed should produce same results");

    let r3 = run(456);
    assert_ne!(r1, r3, "Different seeds should produce different results");
}

#[test]
fn test_moead_builder() {
    let sampler = MoeadSampler::builder()
        .population_size(15)
        .neighborhood_size(5)
        .decomposition(Decomposition::Tchebycheff)
        .crossover_prob(0.9)
        .crossover_eta(20.0)
        .mutation_eta(20.0)
        .seed(42)
        .build();

    let study =
        MultiObjectiveStudy::with_sampler(vec![Direction::Minimize, Direction::Minimize], sampler);
    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(30, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(vec![xv, 1.0 - xv])
        })
        .unwrap();

    assert_eq!(study.n_trials(), 30);
}
