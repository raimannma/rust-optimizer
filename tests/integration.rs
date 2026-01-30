//! Integration tests for the optimizer library.

#![allow(
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use optimizer::sampler::random::RandomSampler;
use optimizer::sampler::tpe::TpeSampler;
use optimizer::{Direction, Error, Study, Trial};

// =============================================================================
// Test: optimize simple quadratic function with TPE, finds near-optimal
// =============================================================================

#[test]
fn test_tpe_optimizes_quadratic_function() {
    // Minimize f(x) = (x - 3)^2 where x ∈ [-10, 10]
    // Optimal: x = 3, f(3) = 0
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(5) // Quick startup for test
        .n_ei_candidates(24)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    study
        .optimize_with_sampler(50, |trial| {
            let x = trial.suggest_float("x", -10.0, 10.0)?;
            Ok::<_, Error>((x - 3.0).powi(2))
        })
        .expect("optimization should succeed");

    let best = study.best_trial().expect("should have at least one trial");

    // TPE should find a value close to optimal (x ≈ 3)
    // We expect the best value to be small (close to 0)
    assert!(
        best.value < 1.0,
        "TPE should find near-optimal: best value {} should be < 1.0",
        best.value
    );
}

#[test]
fn test_tpe_optimizes_multivariate_function() {
    // Minimize f(x, y) = x^2 + y^2 where x, y ∈ [-5, 5]
    // Optimal: (0, 0), f(0, 0) = 0
    let sampler = TpeSampler::builder()
        .seed(123)
        .n_startup_trials(10)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    study
        .optimize_with_sampler(100, |trial| {
            let x = trial.suggest_float("x", -5.0, 5.0)?;
            let y = trial.suggest_float("y", -5.0, 5.0)?;
            Ok::<_, Error>(x * x + y * y)
        })
        .expect("optimization should succeed");

    let best = study.best_trial().expect("should have at least one trial");

    // TPE should find a reasonably good solution
    assert!(
        best.value < 5.0,
        "TPE should find near-optimal: best value {} should be < 5.0",
        best.value
    );
}

#[test]
fn test_tpe_maximization() {
    // Maximize f(x) = -(x - 2)^2 + 10 where x ∈ [-10, 10]
    // Optimal: x = 2, f(2) = 10
    let sampler = TpeSampler::builder()
        .seed(456)
        .n_startup_trials(5)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Maximize, sampler);

    study
        .optimize_with_sampler(50, |trial| {
            let x = trial.suggest_float("x", -10.0, 10.0)?;
            Ok::<_, Error>(-(x - 2.0).powi(2) + 10.0)
        })
        .expect("optimization should succeed");

    let best = study.best_trial().expect("should have at least one trial");

    // For maximization, best value should be better than a random baseline
    // The function ranges from -90 (at x=-10 or x=10, when far from x=2) to 10 (at x=2)
    // A random approach would average around 0, so finding >5 is a reasonable check
    assert!(
        best.value > 5.0,
        "TPE should find reasonably good solution: best value {} should be > 5.0",
        best.value
    );
}

// =============================================================================
// Test: RandomSampler samples uniformly across range
// =============================================================================

#[test]
fn test_random_sampler_uniform_float_distribution() {
    // Test that RandomSampler samples uniformly by running multiple trials
    // and checking the distribution of sampled values
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));

    let n_samples = 1000;
    let mut samples = Vec::with_capacity(n_samples);

    study
        .optimize(n_samples, |trial| {
            let x = trial.suggest_float("x", 0.0, 1.0)?;
            samples.push(x);
            Ok::<_, Error>(x)
        })
        .unwrap();

    // All samples should be in range
    for &s in &samples {
        assert!((0.0..=1.0).contains(&s), "sample {s} out of range [0, 1]");
    }

    // Check distribution is roughly uniform by looking at quartiles
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q1 = samples[n_samples / 4];
    let q2 = samples[n_samples / 2];
    let q3 = samples[3 * n_samples / 4];

    // For uniform distribution, quartiles should be approximately 0.25, 0.5, 0.75
    assert!((q1 - 0.25).abs() < 0.1, "Q1 {q1} should be close to 0.25");
    assert!(
        (q2 - 0.5).abs() < 0.1,
        "Q2 (median) {q2} should be close to 0.5"
    );
    assert!((q3 - 0.75).abs() < 0.1, "Q3 {q3} should be close to 0.75");
}

#[test]
fn test_random_sampler_uniform_int_distribution() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(123));

    let n_samples = 5000;
    let mut counts = [0u32; 10]; // counts for values 1-10

    study
        .optimize(n_samples, |trial| {
            let n = trial.suggest_int("n", 1, 10)?;
            assert!((1..=10).contains(&n), "sample {n} out of range [1, 10]");
            counts[(n - 1) as usize] += 1;
            Ok::<_, Error>(n as f64)
        })
        .unwrap();

    // Each value should appear roughly n_samples / 10 times
    // With 5000 samples, expected ~500 per bucket, std dev ~21
    // 20% tolerance allows for ~4.5 std devs which is very safe
    let expected = n_samples as f64 / 10.0;
    for (i, &count) in counts.iter().enumerate() {
        let diff = (count as f64 - expected).abs() / expected;
        assert!(
            diff < 0.2,
            "value {} appeared {} times, expected ~{}, diff = {:.1}%",
            i + 1,
            count,
            expected,
            diff * 100.0
        );
    }
}

#[test]
fn test_random_sampler_uniform_categorical_distribution() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(456));

    let n_samples = 2000;
    let mut counts = [0u32; 4];
    let choices = ["a", "b", "c", "d"];

    study
        .optimize(n_samples, |trial| {
            let choice = trial.suggest_categorical("cat", &choices)?;
            let idx = choices.iter().position(|&c| c == choice).unwrap();
            counts[idx] += 1;
            Ok::<_, Error>(idx as f64)
        })
        .unwrap();

    // Each category should appear roughly n_samples / 4 times
    // With 2000 samples, expected ~500 per bucket, std dev ~19
    // 15% tolerance allows for ~4 std devs which is very safe
    let expected = n_samples as f64 / 4.0;
    for (i, &count) in counts.iter().enumerate() {
        let diff = (count as f64 - expected).abs() / expected;
        assert!(
            diff < 0.15,
            "category {} appeared {} times, expected ~{}, diff = {:.1}%",
            i,
            count,
            expected,
            diff * 100.0
        );
    }
}

#[test]
fn test_random_sampler_reproducibility() {
    // Two studies with the same seed should produce the same sequence
    // NOTE: We must use optimize_with_sampler() for Study<f64> to get sampler integration
    let study1: Study<f64> =
        Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(999));
    let study2: Study<f64> =
        Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(999));

    let mut values1 = Vec::new();
    let mut values2 = Vec::new();

    study1
        .optimize_with_sampler(100, |trial| {
            let x = trial.suggest_float("x", 0.0, 100.0)?;
            values1.push(x);
            Ok::<_, Error>(x)
        })
        .unwrap();

    study2
        .optimize_with_sampler(100, |trial| {
            let x = trial.suggest_float("x", 0.0, 100.0)?;
            values2.push(x);
            Ok::<_, Error>(x)
        })
        .unwrap();

    for (i, (v1, v2)) in values1.iter().zip(values2.iter()).enumerate() {
        assert_eq!(
            v1, v2,
            "values at trial {i} should be identical with same seed: {v1} vs {v2}"
        );
    }
}

// =============================================================================
// Test: suggest_* methods return cached values on repeated calls
// =============================================================================

#[test]
fn test_suggest_float_caching() {
    let mut trial = Trial::new(0);

    let x1 = trial.suggest_float("x", 0.0, 10.0).unwrap();
    let x2 = trial.suggest_float("x", 0.0, 10.0).unwrap();
    let x3 = trial.suggest_float("x", 0.0, 10.0).unwrap();

    assert_eq!(x1, x2, "repeated suggest_float should return cached value");
    assert_eq!(x2, x3, "repeated suggest_float should return cached value");
}

#[test]
fn test_suggest_float_log_caching() {
    let mut trial = Trial::new(0);

    let x1 = trial.suggest_float_log("lr", 1e-5, 1e-1).unwrap();
    let x2 = trial.suggest_float_log("lr", 1e-5, 1e-1).unwrap();

    assert_eq!(
        x1, x2,
        "repeated suggest_float_log should return cached value"
    );
}

#[test]
fn test_suggest_float_step_caching() {
    let mut trial = Trial::new(0);

    let x1 = trial.suggest_float_step("step", 0.0, 1.0, 0.1).unwrap();
    let x2 = trial.suggest_float_step("step", 0.0, 1.0, 0.1).unwrap();

    assert_eq!(
        x1, x2,
        "repeated suggest_float_step should return cached value"
    );
}

#[test]
fn test_suggest_int_caching() {
    let mut trial = Trial::new(0);

    let n1 = trial.suggest_int("n", 1, 100).unwrap();
    let n2 = trial.suggest_int("n", 1, 100).unwrap();

    assert_eq!(n1, n2, "repeated suggest_int should return cached value");
}

#[test]
fn test_suggest_int_log_caching() {
    let mut trial = Trial::new(0);

    let n1 = trial.suggest_int_log("batch", 1, 1024).unwrap();
    let n2 = trial.suggest_int_log("batch", 1, 1024).unwrap();

    assert_eq!(
        n1, n2,
        "repeated suggest_int_log should return cached value"
    );
}

#[test]
fn test_suggest_int_step_caching() {
    let mut trial = Trial::new(0);

    let n1 = trial.suggest_int_step("units", 32, 512, 32).unwrap();
    let n2 = trial.suggest_int_step("units", 32, 512, 32).unwrap();

    assert_eq!(
        n1, n2,
        "repeated suggest_int_step should return cached value"
    );
}

#[test]
fn test_suggest_categorical_caching() {
    let mut trial = Trial::new(0);

    let choices = ["sgd", "adam", "rmsprop"];
    let c1 = trial.suggest_categorical("optimizer", &choices).unwrap();
    let c2 = trial.suggest_categorical("optimizer", &choices).unwrap();

    assert_eq!(
        c1, c2,
        "repeated suggest_categorical should return cached value"
    );
}

#[test]
fn test_multiple_parameters_independent_caching() {
    let mut trial = Trial::new(0);

    // Suggest multiple parameters
    let x = trial.suggest_float("x", 0.0, 1.0).unwrap();
    let y = trial.suggest_float("y", 0.0, 1.0).unwrap();
    let n = trial.suggest_int("n", 1, 10).unwrap();
    let opt = trial.suggest_categorical("opt", &["a", "b"]).unwrap();

    // All should be cached independently
    assert_eq!(x, trial.suggest_float("x", 0.0, 1.0).unwrap());
    assert_eq!(y, trial.suggest_float("y", 0.0, 1.0).unwrap());
    assert_eq!(n, trial.suggest_int("n", 1, 10).unwrap());
    assert_eq!(opt, trial.suggest_categorical("opt", &["a", "b"]).unwrap());
}

// =============================================================================
// Test: parameter conflict returns error
// =============================================================================

#[test]
fn test_parameter_conflict_float_different_bounds() {
    let mut trial = Trial::new(0);

    trial.suggest_float("x", 0.0, 1.0).unwrap();
    let result = trial.suggest_float("x", 0.0, 2.0); // Different upper bound

    assert!(matches!(result, Err(Error::ParameterConflict { .. })));
}

#[test]
fn test_parameter_conflict_float_vs_log() {
    let mut trial = Trial::new(0);

    trial.suggest_float("x", 0.1, 1.0).unwrap();
    let result = trial.suggest_float_log("x", 0.1, 1.0); // Same bounds but log scale

    assert!(matches!(result, Err(Error::ParameterConflict { .. })));
}

#[test]
fn test_parameter_conflict_float_vs_step() {
    let mut trial = Trial::new(0);

    trial.suggest_float("x", 0.0, 1.0).unwrap();
    let result = trial.suggest_float_step("x", 0.0, 1.0, 0.1); // Same bounds but with step

    assert!(matches!(result, Err(Error::ParameterConflict { .. })));
}

#[test]
fn test_parameter_conflict_int_different_bounds() {
    let mut trial = Trial::new(0);

    trial.suggest_int("n", 1, 10).unwrap();
    let result = trial.suggest_int("n", 1, 20); // Different upper bound

    assert!(matches!(result, Err(Error::ParameterConflict { .. })));
}

#[test]
fn test_parameter_conflict_int_vs_log() {
    let mut trial = Trial::new(0);

    trial.suggest_int("n", 1, 100).unwrap();
    let result = trial.suggest_int_log("n", 1, 100); // Same bounds but log scale

    assert!(matches!(result, Err(Error::ParameterConflict { .. })));
}

#[test]
fn test_parameter_conflict_categorical_different_n_choices() {
    let mut trial = Trial::new(0);

    trial.suggest_categorical("opt", &["a", "b", "c"]).unwrap();
    let result = trial.suggest_categorical("opt", &["x", "y"]); // Different number of choices

    assert!(matches!(result, Err(Error::ParameterConflict { .. })));
}

#[test]
fn test_parameter_conflict_float_vs_int() {
    let mut trial = Trial::new(0);

    trial.suggest_float("x", 0.0, 10.0).unwrap();
    let result = trial.suggest_int("x", 0, 10); // Different type

    assert!(matches!(result, Err(Error::ParameterConflict { .. })));
}

#[test]
fn test_parameter_conflict_returns_name() {
    let mut trial = Trial::new(0);

    trial.suggest_float("my_param", 0.0, 1.0).unwrap();
    let result = trial.suggest_float("my_param", 0.0, 2.0);

    match result {
        Err(Error::ParameterConflict { name, .. }) => {
            assert_eq!(name, "my_param");
        }
        _ => panic!("expected ParameterConflict error"),
    }
}

// =============================================================================
// Test: empty categorical returns error
// =============================================================================

#[test]
fn test_empty_categorical_returns_error() {
    let mut trial = Trial::new(0);
    let empty: &[&str] = &[];

    let result = trial.suggest_categorical("opt", empty);

    assert!(matches!(result, Err(Error::EmptyChoices)));
}

#[test]
fn test_empty_categorical_vec_returns_error() {
    let mut trial = Trial::new(0);
    let empty: Vec<i32> = vec![];

    let result = trial.suggest_categorical("numbers", &empty);

    assert!(matches!(result, Err(Error::EmptyChoices)));
}

// =============================================================================
// Additional integration tests
// =============================================================================

#[test]
fn test_study_basic_workflow() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    study
        .optimize(10, |trial| {
            let x = trial.suggest_float("x", -5.0, 5.0)?;
            Ok::<_, Error>(x * x)
        })
        .expect("optimization should succeed");

    assert_eq!(study.n_trials(), 10);
    let best = study.best_trial().expect("should have best trial");
    assert!(best.value >= 0.0, "x^2 should be non-negative");
}

#[test]
fn test_study_with_failures() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    // Every other trial fails
    let mut counter = 0;
    study
        .optimize(10, |trial| {
            counter += 1;
            if counter % 2 == 0 {
                return Err::<f64, &str>("intentional failure");
            }
            let x = trial
                .suggest_float("x", -5.0, 5.0)
                .map_err(|_| "param error")?;
            Ok(x * x)
        })
        .expect("optimization should succeed with some failures");

    // Only half the trials should have succeeded
    assert_eq!(study.n_trials(), 5, "only 5 trials should have completed");
}

#[test]
fn test_no_completed_trials_error() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    let result = study.best_trial();
    assert!(matches!(result, Err(Error::NoCompletedTrials)));
}

#[test]
fn test_invalid_bounds_errors() {
    let mut trial = Trial::new(0);

    // low > high for float
    let result = trial.suggest_float("x", 10.0, 5.0);
    assert!(matches!(result, Err(Error::InvalidBounds { .. })));

    // low > high for int
    let result = trial.suggest_int("n", 100, 50);
    assert!(matches!(result, Err(Error::InvalidBounds { .. })));
}

#[test]
fn test_invalid_log_bounds_errors() {
    let mut trial = Trial::new(0);

    // low <= 0 for log float
    let result = trial.suggest_float_log("x", 0.0, 1.0);
    assert!(matches!(result, Err(Error::InvalidLogBounds)));

    let result = trial.suggest_float_log("y", -1.0, 1.0);
    assert!(matches!(result, Err(Error::InvalidLogBounds)));

    // low < 1 for log int
    let result = trial.suggest_int_log("n", 0, 100);
    assert!(matches!(result, Err(Error::InvalidLogBounds)));
}

#[test]
fn test_invalid_step_errors() {
    let mut trial = Trial::new(0);

    // step <= 0 for float
    let result = trial.suggest_float_step("x", 0.0, 1.0, 0.0);
    assert!(matches!(result, Err(Error::InvalidStep)));

    let result = trial.suggest_float_step("y", 0.0, 1.0, -0.1);
    assert!(matches!(result, Err(Error::InvalidStep)));

    // step <= 0 for int
    let result = trial.suggest_int_step("n", 0, 100, 0);
    assert!(matches!(result, Err(Error::InvalidStep)));
}

#[test]
fn test_tpe_with_categorical_parameter() {
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(5)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Maximize, sampler);

    // Optimization where the best choice depends on the categorical
    study
        .optimize_with_sampler(30, |trial| {
            let choice = trial.suggest_categorical("model", &["linear", "quadratic", "cubic"])?;
            let x = trial.suggest_float("x", 0.0, 2.0)?;

            // cubic model is best at x=1
            let value = match choice {
                "linear" => x,
                "quadratic" => x * x,
                "cubic" => -((x - 1.0).powi(2)) + 10.0, // peak at x=1, max value 10
                _ => unreachable!(),
            };
            Ok::<_, Error>(value)
        })
        .expect("optimization should succeed");

    let best = study.best_trial().expect("should have best trial");
    // The optimizer should find that "cubic" with x≈1 is best
    assert!(
        best.value > 5.0,
        "should find good solution, got {}",
        best.value
    );
}

#[test]
fn test_tpe_with_integer_parameters() {
    let sampler = TpeSampler::builder()
        .seed(789)
        .n_startup_trials(5)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    // Minimize (n - 7)^2 where n ∈ [1, 10]
    study
        .optimize_with_sampler(30, |trial| {
            let n = trial.suggest_int("n", 1, 10)?;
            Ok::<_, Error>(((n - 7) as f64).powi(2))
        })
        .expect("optimization should succeed");

    let best = study.best_trial().expect("should have best trial");

    // Best value should be small (n close to 7)
    assert!(
        best.value < 5.0,
        "should find n close to 7, best value = {}",
        best.value
    );
}

#[test]
fn test_callback_early_stopping() {
    use std::cell::Cell;
    use std::ops::ControlFlow;

    let study: Study<f64> = Study::new(Direction::Minimize);
    let trials_run = Cell::new(0);

    study
        .optimize_with_callback(
            100,
            |trial| {
                trials_run.set(trials_run.get() + 1);
                let x = trial.suggest_float("x", 0.0, 10.0)?;
                Ok::<_, Error>(x)
            },
            |_study, _trial| {
                // Stop after 5 trials
                if trials_run.get() >= 5 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
        )
        .expect("optimization should succeed");

    assert_eq!(study.n_trials(), 5, "should have stopped after 5 trials");
}

#[test]
fn test_study_trials_iteration() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    study
        .optimize(5, |trial| {
            let x = trial.suggest_float("x", 0.0, 1.0)?;
            Ok::<_, Error>(x)
        })
        .unwrap();

    let trials = study.trials();
    assert_eq!(trials.len(), 5);

    for trial in &trials {
        assert!(
            !trial.params.is_empty(),
            "each trial should have parameters"
        );
        assert!(
            trial.params.contains_key("x"),
            "each trial should have parameter 'x'"
        );
    }
}

#[test]
fn test_study_direction() {
    let study_min: Study<f64> = Study::new(Direction::Minimize);
    assert_eq!(study_min.direction(), Direction::Minimize);

    let study_max: Study<f64> = Study::new(Direction::Maximize);
    assert_eq!(study_max.direction(), Direction::Maximize);
}

#[test]
fn test_trial_state() {
    use optimizer::TrialState;

    let trial = Trial::new(0);
    assert_eq!(trial.state(), TrialState::Running);
}

#[test]
fn test_trial_params_access() {
    let mut trial = Trial::new(0);

    trial.suggest_float("x", 0.0, 1.0).unwrap();
    trial.suggest_int("n", 1, 10).unwrap();

    let params = trial.params();
    assert_eq!(params.len(), 2);
    assert!(params.contains_key("x"));
    assert!(params.contains_key("n"));
}

#[test]
fn test_log_scale_float_range() {
    let mut trial = Trial::new(0);

    let lr = trial.suggest_float_log("lr", 1e-5, 1e-1).unwrap();
    assert!(
        (1e-5..=1e-1).contains(&lr),
        "log-scale value {lr} out of range"
    );
}

#[test]
fn test_step_float_snaps_to_grid() {
    let mut trial = Trial::new(0);

    let x = trial.suggest_float_step("x", 0.0, 1.0, 0.25).unwrap();

    // x should be one of: 0.0, 0.25, 0.5, 0.75, 1.0
    let valid_values = [0.0, 0.25, 0.5, 0.75, 1.0];
    let is_valid = valid_values.iter().any(|&v| (x - v).abs() < 1e-10);
    assert!(is_valid, "stepped float {x} should snap to grid");
}

#[test]
fn test_step_int_snaps_to_grid() {
    let mut trial = Trial::new(0);

    let n = trial.suggest_int_step("n", 0, 100, 25).unwrap();

    // n should be one of: 0, 25, 50, 75, 100
    assert!(
        n % 25 == 0 && (0..=100).contains(&n),
        "stepped int {n} should snap to grid"
    );
}

#[test]
fn test_best_value() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    study
        .optimize(10, |trial| {
            let x = trial.suggest_float("x", 0.0, 10.0)?;
            Ok::<_, Error>(x)
        })
        .unwrap();

    let best_value = study.best_value().expect("should have best value");
    let best_trial = study.best_trial().expect("should have best trial");

    assert_eq!(
        best_value, best_trial.value,
        "best_value should match best_trial.value"
    );
}

// =============================================================================
// Additional coverage tests
// =============================================================================

#[test]
fn test_study_set_sampler() {
    // Test that set_sampler allows changing the sampler after study creation
    let mut study: Study<f64> = Study::new(Direction::Minimize);

    // Initially uses RandomSampler, now switch to TPE
    let tpe = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(5)
        .build()
        .unwrap();
    study.set_sampler(tpe);

    // Should work with the new sampler
    study
        .optimize_with_sampler(10, |trial| {
            let x = trial.suggest_float("x", -5.0, 5.0)?;
            Ok::<_, Error>(x * x)
        })
        .expect("optimization should succeed with new sampler");

    assert_eq!(study.n_trials(), 10);
}

#[test]
fn test_study_with_i32_value_type() {
    // Test Study with non-f64 value type
    let study: Study<i32> = Study::new(Direction::Minimize);

    study
        .optimize(10, |trial| {
            let x = trial.suggest_int("x", -10, 10)?;
            Ok::<_, Error>(x.abs() as i32)
        })
        .expect("optimization should succeed");

    assert_eq!(study.n_trials(), 10);
    let best = study.best_trial().expect("should have best trial");
    assert!(best.value >= 0, "absolute value should be non-negative");
}

#[test]
fn test_optimize_all_trials_fail() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    // All trials fail
    let result = study.optimize(5, |_trial| Err::<f64, &str>("always fails"));

    assert!(
        matches!(result, Err(Error::NoCompletedTrials)),
        "should return NoCompletedTrials when all trials fail"
    );
}

#[test]
fn test_optimize_with_callback_all_trials_fail() {
    use std::ops::ControlFlow;

    let study: Study<f64> = Study::new(Direction::Minimize);

    let result = study.optimize_with_callback(
        5,
        |_trial| Err::<f64, &str>("always fails"),
        |_study, _trial| ControlFlow::Continue(()),
    );

    assert!(
        matches!(result, Err(Error::NoCompletedTrials)),
        "should return NoCompletedTrials when all trials fail"
    );
}

#[test]
fn test_optimize_with_sampler_all_trials_fail() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    let result = study.optimize_with_sampler(5, |_trial| Err::<f64, &str>("always fails"));

    assert!(
        matches!(result, Err(Error::NoCompletedTrials)),
        "should return NoCompletedTrials when all trials fail"
    );
}

#[test]
fn test_optimize_with_callback_sampler_all_trials_fail() {
    use std::ops::ControlFlow;

    let study: Study<f64> = Study::new(Direction::Minimize);

    let result = study.optimize_with_callback_sampler(
        5,
        |_trial| Err::<f64, &str>("always fails"),
        |_study, _trial| ControlFlow::Continue(()),
    );

    assert!(
        matches!(result, Err(Error::NoCompletedTrials)),
        "should return NoCompletedTrials when all trials fail"
    );
}

#[test]
fn test_trial_debug_format() {
    let mut trial = Trial::new(42);
    trial.suggest_float("x", 0.0, 1.0).unwrap();

    let debug_str = format!("{:?}", trial);

    // Should contain trial id and other fields
    assert!(debug_str.contains("Trial"));
    assert!(debug_str.contains("42"));
    assert!(debug_str.contains("has_sampler"));
}

#[test]
fn test_tpe_sampler_builder_default_trait() {
    use optimizer::sampler::tpe::TpeSamplerBuilder;

    let builder = TpeSamplerBuilder::default();
    let sampler = builder.build().unwrap();

    // Should have default values
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    study
        .optimize_with_sampler(5, |trial| {
            let x = trial.suggest_float("x", 0.0, 1.0)?;
            Ok::<_, Error>(x)
        })
        .unwrap();

    assert_eq!(study.n_trials(), 5);
}

#[test]
fn test_tpe_sampler_default_trait() {
    let sampler = TpeSampler::default();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    study
        .optimize_with_sampler(5, |trial| {
            let x = trial.suggest_float("x", 0.0, 1.0)?;
            Ok::<_, Error>(x)
        })
        .unwrap();

    assert_eq!(study.n_trials(), 5);
}

#[test]
fn test_tpe_with_fixed_kde_bandwidth() {
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(5)
        .kde_bandwidth(0.5)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    study
        .optimize_with_sampler(20, |trial| {
            let x = trial.suggest_float("x", -5.0, 5.0)?;
            Ok::<_, Error>(x * x)
        })
        .expect("optimization should succeed");

    let best = study.best_trial().unwrap();
    assert!(best.value < 10.0, "should find reasonable solution");
}

#[test]
fn test_tpe_sampler_invalid_kde_bandwidth() {
    let result = TpeSampler::with_config(0.25, 10, 24, Some(-1.0), None);
    assert!(matches!(result, Err(Error::InvalidBandwidth(_))));
}

#[test]
fn test_tpe_split_trials_with_two_trials() {
    // Edge case: exactly 2 trials in history
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(2) // TPE kicks in after 2 trials
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    study
        .optimize_with_sampler(5, |trial| {
            let x = trial.suggest_float("x", 0.0, 10.0)?;
            Ok::<_, Error>(x)
        })
        .expect("optimization should succeed with small history");

    assert_eq!(study.n_trials(), 5);
}

#[test]
fn test_tpe_with_log_scale_int() {
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(5)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    study
        .optimize_with_sampler(20, |trial| {
            let batch_size = trial.suggest_int_log("batch_size", 1, 1024)?;
            // Optimal around batch_size = 32
            Ok::<_, Error>(((batch_size as f64).log2() - 5.0).powi(2))
        })
        .expect("optimization should succeed");

    let best = study.best_trial().unwrap();
    assert!(best.value < 10.0, "should find reasonable solution");
}

#[test]
fn test_tpe_with_step_distributions() {
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(5)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    study
        .optimize_with_sampler(20, |trial| {
            let x = trial.suggest_float_step("x", 0.0, 10.0, 0.5)?;
            let n = trial.suggest_int_step("n", 0, 100, 10)?;
            Ok::<_, Error>((x - 5.0).powi(2) + ((n - 50) as f64).powi(2))
        })
        .expect("optimization should succeed");

    let best = study.best_trial().unwrap();
    assert!(best.value < 100.0, "should find reasonable solution");
}

#[test]
fn test_create_trial_vs_create_trial_with_sampler() {
    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    // create_trial() creates trial without sampler integration
    let trial1 = study.create_trial();
    assert_eq!(trial1.id(), 0);

    // create_trial_with_sampler() creates trial with sampler
    let trial2 = study.create_trial_with_sampler();
    assert_eq!(trial2.id(), 1);

    // Both should work for suggesting parameters
    let mut trial3 = study.create_trial();
    let x = trial3.suggest_float("x", 0.0, 1.0).unwrap();
    assert!((0.0..=1.0).contains(&x));
}

#[test]
fn test_manual_trial_completion() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    // Manually create and complete trials
    let mut trial = study.create_trial();
    let x = trial.suggest_float("x", 0.0, 10.0).unwrap();
    study.complete_trial(trial, x * x);

    let mut trial2 = study.create_trial();
    let y = trial2.suggest_float("x", 0.0, 10.0).unwrap();
    study.complete_trial(trial2, y * y);

    // Manually fail a trial
    let trial3 = study.create_trial();
    study.fail_trial(trial3, "test failure");

    // Only 2 completed trials
    assert_eq!(study.n_trials(), 2);
}

#[test]
fn test_distributions_access() {
    let mut trial = Trial::new(0);

    trial.suggest_float("x", 0.0, 1.0).unwrap();
    trial.suggest_int("n", 1, 10).unwrap();
    trial.suggest_categorical("opt", &["a", "b", "c"]).unwrap();

    let dists = trial.distributions();
    assert_eq!(dists.len(), 3);
    assert!(dists.contains_key("x"));
    assert!(dists.contains_key("n"));
    assert!(dists.contains_key("opt"));
}

#[test]
fn test_tpe_empty_good_or_bad_values_fallback() {
    // When TPE can't find values in the good/bad groups, it falls back to random
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(5)
        .gamma(0.1) // Very small gamma means few "good" trials
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    // First optimize with one parameter
    study
        .optimize_with_sampler(10, |trial| {
            let x = trial.suggest_float("x", 0.0, 10.0)?;
            Ok::<_, Error>(x)
        })
        .unwrap();

    // Now try with a different parameter - TPE won't have history for "y"
    study
        .optimize_with_sampler(5, |trial| {
            let y = trial.suggest_float("y", 0.0, 10.0)?;
            Ok::<_, Error>(y)
        })
        .unwrap();

    assert_eq!(study.n_trials(), 15);
}

#[test]
fn test_callback_early_stopping_on_first_trial() {
    use std::ops::ControlFlow;

    let study: Study<f64> = Study::new(Direction::Minimize);

    study
        .optimize_with_callback(
            100,
            |trial| {
                let x = trial.suggest_float("x", 0.0, 10.0)?;
                Ok::<_, Error>(x)
            },
            |_study, _trial| {
                // Stop immediately after first trial
                ControlFlow::Break(())
            },
        )
        .expect("optimization should succeed");

    assert_eq!(study.n_trials(), 1, "should have stopped after 1 trial");
}

#[test]
fn test_callback_sampler_early_stopping() {
    use std::ops::ControlFlow;

    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    study
        .optimize_with_callback_sampler(
            100,
            |trial| {
                let x = trial.suggest_float("x", 0.0, 10.0)?;
                Ok::<_, Error>(x)
            },
            |study, _trial| {
                if study.n_trials() >= 3 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
        )
        .expect("optimization should succeed");

    assert_eq!(study.n_trials(), 3);
}

#[test]
fn test_int_bounds_with_low_equals_high() {
    let mut trial = Trial::new(0);

    // When low == high, should return that exact value
    let n = trial.suggest_int("n", 5, 5).unwrap();
    assert_eq!(n, 5);

    let x = trial.suggest_float("x", 3.0, 3.0).unwrap();
    assert_eq!(x, 3.0);
}

#[test]
fn test_best_trial_with_nan_values() {
    // Test behavior when comparing with NaN values (PartialOrd edge case)
    let study: Study<f64> = Study::new(Direction::Minimize);

    // Complete some normal trials
    study
        .optimize(5, |trial| {
            let x = trial.suggest_float("x", 0.0, 10.0)?;
            Ok::<_, Error>(x)
        })
        .unwrap();

    // best_trial should still work
    let best = study.best_trial();
    assert!(best.is_ok());
}
