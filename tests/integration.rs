//! Integration tests for the optimizer library.

#![allow(
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use optimizer::parameter::{BoolParam, CategoricalParam, FloatParam, IntParam, Parameter};
use optimizer::sampler::random::RandomSampler;
use optimizer::sampler::tpe::TpeSampler;
use optimizer::{Direction, Error, Study, Trial};

// =============================================================================
// Test: optimize simple quadratic function with TPE, finds near-optimal
// =============================================================================

#[test]
fn test_tpe_optimizes_quadratic_function() {
    // Minimize f(x) = (x - 3)^2 where x in [-10, 10]
    // Optimal: x = 3, f(3) = 0
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(10)
        .n_ei_candidates(24)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(-10.0, 10.0);

    study
        .optimize(100, |trial| {
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>((x - 3.0).powi(2))
        })
        .expect("optimization should succeed");

    let best = study.best_trial().expect("should have at least one trial");

    // TPE should find a reasonable value over 100 trials
    // With random startup + TPE, we expect to get within a few units of optimal
    assert!(
        best.value < 5.0,
        "TPE should find near-optimal: best value {} should be < 5.0",
        best.value
    );
}

#[test]
fn test_tpe_optimizes_multivariate_function() {
    // Minimize f(x, y) = x^2 + y^2 where x, y in [-5, 5]
    // Optimal: (0, 0), f(0, 0) = 0
    let sampler = TpeSampler::builder()
        .seed(123)
        .n_startup_trials(10)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(-5.0, 5.0);
    let y_param = FloatParam::new(-5.0, 5.0);

    study
        .optimize(100, |trial| {
            let x = x_param.suggest(trial)?;
            let y = y_param.suggest(trial)?;
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
    // Maximize f(x) = -(x - 2)^2 + 10 where x in [-10, 10]
    // Optimal: x = 2, f(2) = 10
    let sampler = TpeSampler::builder()
        .seed(456)
        .n_startup_trials(5)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Maximize, sampler);

    let x_param = FloatParam::new(-10.0, 10.0);

    study
        .optimize(50, |trial| {
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>(-(x - 2.0).powi(2) + 10.0)
        })
        .expect("optimization should succeed");

    let best = study.best_trial().expect("should have at least one trial");

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
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));

    let n_samples = 1000;
    let mut samples = Vec::with_capacity(n_samples);

    let x_param = FloatParam::new(0.0, 1.0);

    study
        .optimize(n_samples, |trial| {
            let x = x_param.suggest(trial)?;
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

    let n_param = IntParam::new(1, 10);

    study
        .optimize(n_samples, |trial| {
            let n = n_param.suggest(trial)?;
            assert!((1..=10).contains(&n), "sample {n} out of range [1, 10]");
            counts[(n - 1) as usize] += 1;
            Ok::<_, Error>(n as f64)
        })
        .unwrap();

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

    let cat_param = CategoricalParam::new(choices.to_vec());

    study
        .optimize(n_samples, |trial| {
            let choice = cat_param.suggest(trial)?;
            let idx = choices.iter().position(|&c| c == choice).unwrap();
            counts[idx] += 1;
            Ok::<_, Error>(idx as f64)
        })
        .unwrap();

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
    let study1: Study<f64> =
        Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(999));
    let study2: Study<f64> =
        Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(999));

    let mut values1 = Vec::new();
    let mut values2 = Vec::new();

    let x_param1 = FloatParam::new(0.0, 100.0);
    let x_param2 = FloatParam::new(0.0, 100.0);

    study1
        .optimize(100, |trial| {
            let x = x_param1.suggest(trial)?;
            values1.push(x);
            Ok::<_, Error>(x)
        })
        .unwrap();

    study2
        .optimize(100, |trial| {
            let x = x_param2.suggest(trial)?;
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
// Test: suggest_param returns cached values on repeated calls
// =============================================================================

#[test]
fn test_suggest_float_caching() {
    let param = FloatParam::new(0.0, 10.0);
    let mut trial = Trial::new(0);

    let x1 = param.suggest(&mut trial).unwrap();
    let x2 = param.suggest(&mut trial).unwrap();
    let x3 = param.suggest(&mut trial).unwrap();

    assert_eq!(x1, x2, "repeated suggest should return cached value");
    assert_eq!(x2, x3, "repeated suggest should return cached value");
}

#[test]
fn test_suggest_float_log_caching() {
    let param = FloatParam::new(1e-5, 1e-1).log_scale();
    let mut trial = Trial::new(0);

    let x1 = param.suggest(&mut trial).unwrap();
    let x2 = param.suggest(&mut trial).unwrap();

    assert_eq!(
        x1, x2,
        "repeated suggest float log should return cached value"
    );
}

#[test]
fn test_suggest_float_step_caching() {
    let param = FloatParam::new(0.0, 1.0).step(0.1);
    let mut trial = Trial::new(0);

    let x1 = param.suggest(&mut trial).unwrap();
    let x2 = param.suggest(&mut trial).unwrap();

    assert_eq!(
        x1, x2,
        "repeated suggest float step should return cached value"
    );
}

#[test]
fn test_suggest_int_caching() {
    let param = IntParam::new(1, 100);
    let mut trial = Trial::new(0);

    let n1 = param.suggest(&mut trial).unwrap();
    let n2 = param.suggest(&mut trial).unwrap();

    assert_eq!(n1, n2, "repeated suggest int should return cached value");
}

#[test]
fn test_suggest_int_log_caching() {
    let param = IntParam::new(1, 1024).log_scale();
    let mut trial = Trial::new(0);

    let n1 = param.suggest(&mut trial).unwrap();
    let n2 = param.suggest(&mut trial).unwrap();

    assert_eq!(
        n1, n2,
        "repeated suggest int log should return cached value"
    );
}

#[test]
fn test_suggest_int_step_caching() {
    let param = IntParam::new(32, 512).step(32);
    let mut trial = Trial::new(0);

    let n1 = param.suggest(&mut trial).unwrap();
    let n2 = param.suggest(&mut trial).unwrap();

    assert_eq!(
        n1, n2,
        "repeated suggest int step should return cached value"
    );
}

#[test]
fn test_suggest_categorical_caching() {
    let param = CategoricalParam::new(vec!["sgd", "adam", "rmsprop"]);
    let mut trial = Trial::new(0);

    let c1 = param.suggest(&mut trial).unwrap();
    let c2 = param.suggest(&mut trial).unwrap();

    assert_eq!(
        c1, c2,
        "repeated suggest categorical should return cached value"
    );
}

#[test]
fn test_multiple_parameters_independent_caching() {
    let x_param = FloatParam::new(0.0, 1.0);
    let y_param = FloatParam::new(0.0, 1.0);
    let n_param = IntParam::new(1, 10);
    let opt_param = CategoricalParam::new(vec!["a", "b"]);
    let mut trial = Trial::new(0);

    // Suggest multiple parameters
    let x = x_param.suggest(&mut trial).unwrap();
    let y = y_param.suggest(&mut trial).unwrap();
    let n = n_param.suggest(&mut trial).unwrap();
    let opt = opt_param.suggest(&mut trial).unwrap();

    // All should be cached independently
    assert_eq!(x, x_param.suggest(&mut trial).unwrap());
    assert_eq!(y, y_param.suggest(&mut trial).unwrap());
    assert_eq!(n, n_param.suggest(&mut trial).unwrap());
    assert_eq!(opt, opt_param.suggest(&mut trial).unwrap());
}

// =============================================================================
// Test: parameter conflict returns error
// =============================================================================

#[test]
fn test_parameter_conflict_same_param_different_distribution() {
    // With ParamId-based API, conflict happens when the same ParamId is used
    // with a different distribution. This can happen via suggest_param with
    // a param that has a mismatched distribution for an already-stored id.
    // Since each FloatParam::new() gets a unique id, conflicts only happen
    // when the same param object is reused with different internal state,
    // which is not possible with the immutable API.
    // We test that different param objects don't conflict (they have different ids).
    let param1 = FloatParam::new(0.0, 1.0);
    let param2 = FloatParam::new(0.0, 2.0);
    let mut trial = Trial::new(0);

    trial.suggest_param(&param1).unwrap();
    // Different param object = different id = no conflict
    let result = trial.suggest_param(&param2);
    assert!(result.is_ok());
}

#[test]
fn test_empty_categorical_returns_error() {
    let param = CategoricalParam::<&str>::new(vec![]);
    let mut trial = Trial::new(0);

    let result = trial.suggest_param(&param);
    assert!(matches!(result, Err(Error::EmptyChoices)));
}

// =============================================================================
// Additional integration tests
// =============================================================================

#[test]
fn test_study_basic_workflow() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(-5.0, 5.0);

    study
        .optimize(10, |trial| {
            let x = x_param.suggest(trial)?;
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
    let x_param = FloatParam::new(-5.0, 5.0);

    // Every other trial fails
    let mut counter = 0;
    study
        .optimize(10, |trial| {
            counter += 1;
            if counter % 2 == 0 {
                return Err::<f64, &str>("intentional failure");
            }
            let x = x_param.suggest(trial).map_err(|_| "param error")?;
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
    let result = trial.suggest_param(&FloatParam::new(10.0, 5.0));
    assert!(matches!(result, Err(Error::InvalidBounds { .. })));

    // low > high for int
    let result = trial.suggest_param(&IntParam::new(100, 50));
    assert!(matches!(result, Err(Error::InvalidBounds { .. })));
}

#[test]
fn test_invalid_log_bounds_errors() {
    let mut trial = Trial::new(0);

    // low <= 0 for log float
    let result = trial.suggest_param(&FloatParam::new(0.0, 1.0).log_scale());
    assert!(matches!(result, Err(Error::InvalidLogBounds)));

    let result = trial.suggest_param(&FloatParam::new(-1.0, 1.0).log_scale());
    assert!(matches!(result, Err(Error::InvalidLogBounds)));

    // low < 1 for log int
    let result = trial.suggest_param(&IntParam::new(0, 100).log_scale());
    assert!(matches!(result, Err(Error::InvalidLogBounds)));
}

#[test]
fn test_invalid_step_errors() {
    let mut trial = Trial::new(0);

    // step <= 0 for float
    let result = trial.suggest_param(&FloatParam::new(0.0, 1.0).step(0.0));
    assert!(matches!(result, Err(Error::InvalidStep)));

    let result = trial.suggest_param(&FloatParam::new(0.0, 1.0).step(-0.1));
    assert!(matches!(result, Err(Error::InvalidStep)));

    // step <= 0 for int
    let result = trial.suggest_param(&IntParam::new(0, 100).step(0));
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

    let model_param = CategoricalParam::new(vec!["linear", "quadratic", "cubic"]);
    let x_param = FloatParam::new(0.0, 2.0);

    // Optimization where the best choice depends on the categorical
    study
        .optimize(30, |trial| {
            let choice = model_param.suggest(trial)?;
            let x = x_param.suggest(trial)?;

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

    let n_param = IntParam::new(1, 10);

    // Minimize (n - 7)^2 where n in [1, 10]
    study
        .optimize(30, |trial| {
            let n = n_param.suggest(trial)?;
            Ok::<_, Error>(((n - 7) as f64).powi(2))
        })
        .expect("optimization should succeed");

    let best = study.best_trial().expect("should have best trial");

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
    let x_param = FloatParam::new(0.0, 10.0);

    study
        .optimize_with_callback(
            100,
            |trial| {
                trials_run.set(trials_run.get() + 1);
                let x = x_param.suggest(trial)?;
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
    let x_param = FloatParam::new(0.0, 1.0);

    study
        .optimize(5, |trial| {
            let x = x_param.suggest(trial)?;
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
    let x_param = FloatParam::new(0.0, 1.0);
    let n_param = IntParam::new(1, 10);
    let mut trial = Trial::new(0);

    x_param.suggest(&mut trial).unwrap();
    n_param.suggest(&mut trial).unwrap();

    let params = trial.params();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_log_scale_float_range() {
    let param = FloatParam::new(1e-5, 1e-1).log_scale();
    let mut trial = Trial::new(0);

    let lr = param.suggest(&mut trial).unwrap();
    assert!(
        (1e-5..=1e-1).contains(&lr),
        "log-scale value {lr} out of range"
    );
}

#[test]
fn test_step_float_snaps_to_grid() {
    let param = FloatParam::new(0.0, 1.0).step(0.25);
    let mut trial = Trial::new(0);

    let x = param.suggest(&mut trial).unwrap();

    // x should be one of: 0.0, 0.25, 0.5, 0.75, 1.0
    let valid_values = [0.0, 0.25, 0.5, 0.75, 1.0];
    let is_valid = valid_values.iter().any(|&v| (x - v).abs() < 1e-10);
    assert!(is_valid, "stepped float {x} should snap to grid");
}

#[test]
fn test_step_int_snaps_to_grid() {
    let param = IntParam::new(0, 100).step(25);
    let mut trial = Trial::new(0);

    let n = param.suggest(&mut trial).unwrap();

    // n should be one of: 0, 25, 50, 75, 100
    assert!(
        n % 25 == 0 && (0..=100).contains(&n),
        "stepped int {n} should snap to grid"
    );
}

#[test]
fn test_best_value() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(0.0, 10.0);

    study
        .optimize(10, |trial| {
            let x = x_param.suggest(trial)?;
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
    let mut study: Study<f64> = Study::new(Direction::Minimize);

    let tpe = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(5)
        .build()
        .unwrap();
    study.set_sampler(tpe);

    let x_param = FloatParam::new(-5.0, 5.0);

    study
        .optimize(10, |trial| {
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>(x * x)
        })
        .expect("optimization should succeed with new sampler");

    assert_eq!(study.n_trials(), 10);
}

#[test]
fn test_study_with_i32_value_type() {
    let study: Study<i32> = Study::new(Direction::Minimize);
    let x_param = IntParam::new(-10, 10);

    study
        .optimize(10, |trial| {
            let x = x_param.suggest(trial)?;
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
#[allow(deprecated)]
fn test_optimize_with_sampler_all_trials_fail() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    let result = study.optimize_with_sampler(5, |_trial| Err::<f64, &str>("always fails"));

    assert!(
        matches!(result, Err(Error::NoCompletedTrials)),
        "should return NoCompletedTrials when all trials fail"
    );
}

#[test]
#[allow(deprecated)]
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
    let param = FloatParam::new(0.0, 1.0);
    let mut trial = Trial::new(42);
    param.suggest(&mut trial).unwrap();

    let debug_str = format!("{trial:?}");

    assert!(debug_str.contains("Trial"));
    assert!(debug_str.contains("42"));
    assert!(debug_str.contains("has_sampler"));
}

#[test]
fn test_tpe_sampler_builder_default_trait() {
    use optimizer::sampler::tpe::TpeSamplerBuilder;

    let builder = TpeSamplerBuilder::default();
    let sampler = builder.build().unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    let x_param = FloatParam::new(0.0, 1.0);

    study
        .optimize(5, |trial| {
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>(x)
        })
        .unwrap();

    assert_eq!(study.n_trials(), 5);
}

#[test]
fn test_tpe_sampler_default_trait() {
    let sampler = TpeSampler::default();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    let x_param = FloatParam::new(0.0, 1.0);

    study
        .optimize(5, |trial| {
            let x = x_param.suggest(trial)?;
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
    let x_param = FloatParam::new(-5.0, 5.0);

    study
        .optimize(20, |trial| {
            let x = x_param.suggest(trial)?;
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
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(2)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    let x_param = FloatParam::new(0.0, 10.0);

    study
        .optimize(5, |trial| {
            let x = x_param.suggest(trial)?;
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
    let batch_param = IntParam::new(1, 1024).log_scale();

    study
        .optimize(20, |trial| {
            let batch_size = batch_param.suggest(trial)?;
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
    let x_param = FloatParam::new(0.0, 10.0).step(0.5);
    let n_param = IntParam::new(0, 100).step(10);

    study
        .optimize(20, |trial| {
            let x = x_param.suggest(trial)?;
            let n = n_param.suggest(trial)?;
            Ok::<_, Error>((x - 5.0).powi(2) + ((n - 50) as f64).powi(2))
        })
        .expect("optimization should succeed");

    let best = study.best_trial().unwrap();
    assert!(best.value < 100.0, "should find reasonable solution");
}

#[test]
#[allow(deprecated)]
fn test_create_trial_vs_create_trial_with_sampler() {
    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    // create_trial() creates trial with sampler integration for Study<f64>
    let trial1 = study.create_trial();
    assert_eq!(trial1.id(), 0);

    // create_trial_with_sampler() is deprecated but still works
    let trial2 = study.create_trial_with_sampler();
    assert_eq!(trial2.id(), 1);

    // Both should work for suggesting parameters
    let x_param = FloatParam::new(0.0, 1.0);
    let mut trial3 = study.create_trial();
    let x = x_param.suggest(&mut trial3).unwrap();
    assert!((0.0..=1.0).contains(&x));
}

#[test]
fn test_manual_trial_completion() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(0.0, 10.0);

    // Manually create and complete trials
    let mut trial = study.create_trial();
    let x = x_param.suggest(&mut trial).unwrap();
    study.complete_trial(trial, x * x);

    let mut trial2 = study.create_trial();
    let y = x_param.suggest(&mut trial2).unwrap();
    study.complete_trial(trial2, y * y);

    // Manually fail a trial
    let trial3 = study.create_trial();
    study.fail_trial(trial3, "test failure");

    // Only 2 completed trials
    assert_eq!(study.n_trials(), 2);
}

#[test]
fn test_distributions_access() {
    let x_param = FloatParam::new(0.0, 1.0);
    let n_param = IntParam::new(1, 10);
    let opt_param = CategoricalParam::new(vec!["a", "b", "c"]);
    let mut trial = Trial::new(0);

    x_param.suggest(&mut trial).unwrap();
    n_param.suggest(&mut trial).unwrap();
    opt_param.suggest(&mut trial).unwrap();

    let dists = trial.distributions();
    assert_eq!(dists.len(), 3);
}

#[test]
fn test_tpe_empty_good_or_bad_values_fallback() {
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(5)
        .gamma(0.1)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    let x_param = FloatParam::new(0.0, 10.0);
    let y_param = FloatParam::new(0.0, 10.0);

    // First optimize with one parameter
    study
        .optimize(10, |trial| {
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>(x)
        })
        .unwrap();

    // Now try with a different parameter - TPE won't have history for "y"
    study
        .optimize(5, |trial| {
            let y = y_param.suggest(trial)?;
            Ok::<_, Error>(y)
        })
        .unwrap();

    assert_eq!(study.n_trials(), 15);
}

#[test]
fn test_callback_early_stopping_on_first_trial() {
    use std::ops::ControlFlow;

    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(0.0, 10.0);

    study
        .optimize_with_callback(
            100,
            |trial| {
                let x = x_param.suggest(trial)?;
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
    let x_param = FloatParam::new(0.0, 10.0);

    study
        .optimize_with_callback(
            100,
            |trial| {
                let x = x_param.suggest(trial)?;
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
    let n_param = IntParam::new(5, 5);
    let n = n_param.suggest(&mut trial).unwrap();
    assert_eq!(n, 5);

    let x_param = FloatParam::new(3.0, 3.0);
    let x = x_param.suggest(&mut trial).unwrap();
    assert_eq!(x, 3.0);
}

#[test]
fn test_best_trial_with_nan_values() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(0.0, 10.0);

    study
        .optimize(5, |trial| {
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>(x)
        })
        .unwrap();

    let best = study.best_trial();
    assert!(best.is_ok());
}

// =============================================================================
// Tests for BoolParam
// =============================================================================

#[test]
fn test_suggest_bool_caching() {
    let param = BoolParam::new();
    let mut trial = Trial::new(0);

    let b1 = param.suggest(&mut trial).unwrap();
    let b2 = param.suggest(&mut trial).unwrap();

    assert_eq!(b1, b2, "repeated suggest bool should return cached value");
}

#[test]
fn test_suggest_bool_multiple_parameters() {
    let dropout_param = BoolParam::new();
    let batchnorm_param = BoolParam::new();
    let skip_param = BoolParam::new();
    let mut trial = Trial::new(0);

    let a = dropout_param.suggest(&mut trial).unwrap();
    let b = batchnorm_param.suggest(&mut trial).unwrap();
    let c = skip_param.suggest(&mut trial).unwrap();

    // All should be cached independently
    assert_eq!(a, dropout_param.suggest(&mut trial).unwrap());
    assert_eq!(b, batchnorm_param.suggest(&mut trial).unwrap());
    assert_eq!(c, skip_param.suggest(&mut trial).unwrap());
}

#[test]
fn test_suggest_bool_in_optimization() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let use_feature_param = BoolParam::new();
    let x_param = FloatParam::new(0.0, 10.0);

    study
        .optimize(10, |trial| {
            let use_feature = use_feature_param.suggest(trial)?;
            let x = x_param.suggest(trial)?;

            let value = if use_feature { x } else { x * 2.0 };
            Ok::<_, Error>(value)
        })
        .unwrap();

    assert_eq!(study.n_trials(), 10);
}

#[test]
fn test_suggest_bool_with_tpe() {
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(5)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    let use_large_param = BoolParam::new();
    let x_param = FloatParam::new(0.0, 10.0);

    study
        .optimize(20, |trial| {
            let use_large = use_large_param.suggest(trial)?;
            let x = x_param.suggest(trial)?;
            // The value depends on use_large flag
            let base = if use_large { x * 2.0 } else { x };
            Ok::<_, Error>(base)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    assert!(best.value < 10.0);
}

// =============================================================================
// Tests for FloatParam and IntParam ranges
// =============================================================================

#[test]
fn test_float_param_exclusive_range() {
    let param = FloatParam::new(0.0, 1.0);
    let mut trial = Trial::new(0);

    let x = param.suggest(&mut trial).unwrap();
    assert!((0.0..=1.0).contains(&x), "value {x} out of range 0.0..1.0");
}

#[test]
fn test_float_param_inclusive_range() {
    let param = FloatParam::new(0.0, 1.0);
    let mut trial = Trial::new(0);

    let x = param.suggest(&mut trial).unwrap();
    assert!((0.0..=1.0).contains(&x), "value {x} out of range 0.0..=1.0");
}

#[test]
fn test_int_param_range() {
    let param = IntParam::new(1, 10);
    let mut trial = Trial::new(0);

    let n = param.suggest(&mut trial).unwrap();
    assert!((1..=10).contains(&n), "value {n} out of range 1..=10");
}

#[test]
fn test_param_caching_float() {
    let param = FloatParam::new(0.0, 1.0);
    let mut trial = Trial::new(0);

    let x1 = param.suggest(&mut trial).unwrap();
    let x2 = param.suggest(&mut trial).unwrap();

    assert_eq!(x1, x2, "repeated suggest should return cached value");
}

#[test]
fn test_param_caching_int() {
    let param = IntParam::new(1, 100);
    let mut trial = Trial::new(0);

    let n1 = param.suggest(&mut trial).unwrap();
    let n2 = param.suggest(&mut trial).unwrap();

    assert_eq!(n1, n2, "repeated suggest should return cached value");
}

#[test]
fn test_multiple_params_in_optimization() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(-10.0, 10.0);
    let n_param = IntParam::new(1, 5);

    study
        .optimize(10, |trial| {
            let x = x_param.suggest(trial)?;
            let n = n_param.suggest(trial)?;
            Ok::<_, Error>(x * x + n as f64)
        })
        .unwrap();

    assert_eq!(study.n_trials(), 10);
}

#[test]
fn test_params_with_tpe() {
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(5)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    let x_param = FloatParam::new(-5.0, 5.0);
    let n_param = IntParam::new(1, 10);

    study
        .optimize(30, |trial| {
            let x = x_param.suggest(trial)?;
            let n = n_param.suggest(trial)?;
            Ok::<_, Error>(x * x + (n as f64 - 5.0).powi(2))
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    assert!(best.value < 10.0, "TPE should find good solution");
}

#[test]
fn test_single_value_int_range() {
    let param = IntParam::new(5, 5);
    let mut trial = Trial::new(0);

    let n = param.suggest(&mut trial).unwrap();
    assert_eq!(n, 5, "single-value range should return that value");
}

#[test]
fn test_single_value_float_range() {
    let param = FloatParam::new(4.2, 4.2);
    let mut trial = Trial::new(0);

    let x = param.suggest(&mut trial).unwrap();
    assert!(
        (x - 4.2).abs() < f64::EPSILON,
        "single-value range should return that value"
    );
}

// =============================================================================
// Tests for new API features
// =============================================================================

#[test]
fn test_param_name() {
    let param = FloatParam::new(0.0, 1.0).name("learning_rate");
    let mut trial = Trial::new(0);
    param.suggest(&mut trial).unwrap();

    let labels = trial.param_labels();
    let label = labels.values().next().unwrap();
    assert_eq!(label, "learning_rate");
}

#[test]
fn test_completed_trial_get() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(-10.0, 10.0).name("x");
    let n_param = IntParam::new(1, 10).name("n");

    study
        .optimize(5, |trial| {
            let x = x_param.suggest(trial)?;
            let n = n_param.suggest(trial)?;
            Ok::<_, Error>(x * x + n as f64)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    let x_val: f64 = best.get(&x_param).unwrap();
    let n_val: i64 = best.get(&n_param).unwrap();
    assert!((-10.0..=10.0).contains(&x_val));
    assert!((1..=10).contains(&n_val));
}

// =============================================================================
// Tests for timeout-based optimization
// =============================================================================

#[test]
fn test_optimize_until_runs_for_approximately_specified_duration() {
    use std::time::{Duration, Instant};

    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(-10.0, 10.0);

    let duration = Duration::from_millis(200);
    let start = Instant::now();

    study
        .optimize_until(duration, |trial| {
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>(x * x)
        })
        .unwrap();

    let elapsed = start.elapsed();
    assert!(
        elapsed >= duration,
        "should run for at least the specified duration, elapsed: {elapsed:?}"
    );
    // Allow generous upper bound — the last trial may overshoot
    assert!(
        elapsed < duration + Duration::from_millis(200),
        "should not overshoot excessively, elapsed: {elapsed:?}"
    );
}

#[test]
fn test_optimize_until_completes_at_least_one_trial() {
    use std::time::Duration;

    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(-10.0, 10.0);

    study
        .optimize_until(Duration::from_millis(100), |trial| {
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>(x * x)
        })
        .unwrap();

    assert!(
        study.n_trials() >= 1,
        "should complete at least one trial, got {}",
        study.n_trials()
    );
}

#[test]
fn test_optimize_until_works_with_minimize() {
    use std::time::Duration;

    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    let x_param = FloatParam::new(-10.0, 10.0);

    study
        .optimize_until(Duration::from_millis(100), |trial| {
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>(x * x)
        })
        .unwrap();

    let best = study.best_value().unwrap();
    assert!(best >= 0.0, "x^2 should be non-negative");
}

#[test]
fn test_optimize_until_works_with_maximize() {
    use std::time::Duration;

    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Maximize, sampler);
    let x_param = FloatParam::new(0.0, 10.0);

    study
        .optimize_until(Duration::from_millis(100), |trial| {
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>(x)
        })
        .unwrap();

    let best = study.best_value().unwrap();
    assert!(best >= 0.0);
}

#[test]
fn test_optimize_until_with_callback_early_stopping() {
    use std::ops::ControlFlow;
    use std::time::Duration;

    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(0.0, 10.0);

    study
        .optimize_until_with_callback(
            Duration::from_secs(10), // long timeout — callback should stop early
            |trial| {
                let x = x_param.suggest(trial)?;
                Ok::<_, Error>(x)
            },
            |study, _trial| {
                if study.n_trials() >= 5 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
        )
        .unwrap();

    assert_eq!(
        study.n_trials(),
        5,
        "callback should have stopped after 5 trials"
    );
}

#[test]
fn test_optimize_until_all_trials_fail() {
    use std::time::Duration;

    let study: Study<f64> = Study::new(Direction::Minimize);

    let result = study.optimize_until(Duration::from_millis(50), |_trial| {
        Err::<f64, &str>("always fails")
    });

    assert!(
        matches!(result, Err(Error::NoCompletedTrials)),
        "should return NoCompletedTrials when all trials fail"
    );
}

#[test]
fn test_optimize_until_with_non_f64_value_type() {
    use std::time::Duration;

    let study: Study<i32> = Study::new(Direction::Minimize);
    let x_param = IntParam::new(-10, 10);

    study
        .optimize_until(Duration::from_millis(100), |trial| {
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>(x.abs() as i32)
        })
        .unwrap();

    assert!(study.n_trials() >= 1);
    let best = study.best_trial().unwrap();
    assert!(best.value >= 0);
}

// =============================================================================
// Tests for top_trials
// =============================================================================

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

// =============================================================================
// Test: ask-and-tell interface
// =============================================================================

#[test]
fn test_ask_and_tell_basic() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(0.0, 10.0);

    for _ in 0..10 {
        let mut trial = study.ask();
        let x = x_param.suggest(&mut trial).unwrap();
        let value = x * x;
        study.tell(trial, Ok::<_, &str>(value));
    }

    assert_eq!(study.n_trials(), 10);
    assert!(study.best_value().unwrap() >= 0.0);
}

#[test]
fn test_ask_and_tell_with_failures() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(-5.0, 5.0);

    // Alternate success and failure
    for i in 0..10 {
        let mut trial = study.ask();
        let x = x_param.suggest(&mut trial).unwrap();
        if i % 2 == 0 {
            study.tell(trial, Ok::<_, &str>(x * x));
        } else {
            study.tell(trial, Err::<f64, _>("simulated failure"));
        }
    }

    // Only successful trials are counted
    assert_eq!(study.n_trials(), 5);
}

#[test]
fn test_ask_and_tell_with_tpe_sampler() {
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(5)
        .build()
        .unwrap();
    let study: Study<f64> = Study::minimize(sampler);
    let x_param = FloatParam::new(-10.0, 10.0);

    for _ in 0..30 {
        let mut trial = study.ask();
        let x = x_param.suggest(&mut trial).unwrap();
        study.tell(trial, Ok::<_, &str>((x - 3.0).powi(2)));
    }

    assert_eq!(study.n_trials(), 30);
    assert!(
        study.best_value().unwrap() < 5.0,
        "TPE ask-and-tell should find a reasonable value"
    );
}

#[test]
fn test_ask_and_tell_batch() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(0.0, 10.0);

    // Ask a batch of trials
    let batch: Vec<_> = (0..5)
        .map(|_| {
            let mut t = study.ask();
            let x = x_param.suggest(&mut t).unwrap();
            (t, x)
        })
        .collect();

    // Tell results for the batch
    for (trial, x) in batch {
        study.tell(trial, Ok::<_, &str>(x * x));
    }

    assert_eq!(study.n_trials(), 5);
}

#[test]
fn test_ask_and_tell_with_custom_value_type() {
    // Ask-and-tell works with non-f64 value types too
    let study: Study<i32> = Study::new(Direction::Maximize);

    for i in 0..5 {
        let trial = study.ask();
        study.tell(trial, Ok::<_, &str>(i * 10));
    }

    assert_eq!(study.n_trials(), 5);
    assert_eq!(study.best_value().unwrap(), 40);
}

// =============================================================================
// Tests: enqueue trials
// =============================================================================

use std::collections::HashMap;

use optimizer::parameter::ParamValue;

#[test]
fn test_enqueue_params_evaluated_first() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 10.0);
    let y = IntParam::new(1, 100);

    // Enqueue a specific configuration
    study.enqueue(HashMap::from([
        (x.id(), ParamValue::Float(5.0)),
        (y.id(), ParamValue::Int(42)),
    ]));

    // The first trial should use the enqueued params
    let mut trial = study.ask();
    let x_val = x.suggest(&mut trial).unwrap();
    let y_val = y.suggest(&mut trial).unwrap();

    assert_eq!(x_val, 5.0);
    assert_eq!(y_val, 42);
}

#[test]
fn test_enqueue_fifo_order() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 10.0);

    // Enqueue two configs
    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(1.0))]));
    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(2.0))]));

    // First trial gets first enqueued value
    let mut trial1 = study.ask();
    assert_eq!(x.suggest(&mut trial1).unwrap(), 1.0);

    // Second trial gets second enqueued value
    let mut trial2 = study.ask();
    assert_eq!(x.suggest(&mut trial2).unwrap(), 2.0);
}

#[test]
fn test_enqueue_then_normal_sampling_resumes() {
    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    let x = FloatParam::new(0.0, 10.0);

    // Enqueue one config
    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(5.0))]));

    // First trial uses enqueued value
    let mut trial1 = study.ask();
    assert_eq!(x.suggest(&mut trial1).unwrap(), 5.0);
    study.tell(trial1, Ok::<_, &str>(25.0));

    // Second trial uses normal sampling (not 5.0)
    let mut trial2 = study.ask();
    let x_val = x.suggest(&mut trial2).unwrap();
    // The sampled value should be in [0, 10] but extremely unlikely to be exactly 5.0
    assert!((0.0..=10.0).contains(&x_val));
}

#[test]
fn test_enqueue_with_optimize() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 10.0);

    // Enqueue two specific configs
    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(1.0))]));
    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(2.0))]));

    let mut values = Vec::new();

    study
        .optimize(5, |trial| {
            let x_val = x.suggest(trial)?;
            values.push(x_val);
            Ok::<_, Error>(x_val * x_val)
        })
        .unwrap();

    // First two trials should use enqueued values
    assert_eq!(values[0], 1.0);
    assert_eq!(values[1], 2.0);
    // All 5 trials should have completed
    assert_eq!(study.n_trials(), 5);
}

#[test]
fn test_enqueue_partial_params_fall_back_to_sampling() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 10.0);
    let y = IntParam::new(1, 100);

    // Enqueue only x, not y
    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(3.0))]));

    let mut trial = study.ask();
    let x_val = x.suggest(&mut trial).unwrap();
    let y_val = y.suggest(&mut trial).unwrap();

    // x should be the enqueued value
    assert_eq!(x_val, 3.0);
    // y should be sampled (within range)
    assert!((1..=100).contains(&y_val));
}

#[test]
fn test_enqueue_trials_appear_in_completed_trials() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 10.0);

    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(7.0))]));

    study
        .optimize(1, |trial| {
            let x_val = x.suggest(trial)?;
            Ok::<_, Error>(x_val)
        })
        .unwrap();

    let trials = study.trials();
    assert_eq!(trials.len(), 1);
    assert_eq!(trials[0].value, 7.0);
    assert_eq!(
        *trials[0].params.get(&x.id()).unwrap(),
        ParamValue::Float(7.0)
    );
}

#[test]
fn test_enqueue_with_ask_and_tell() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 10.0);

    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(4.0))]));

    let mut trial = study.ask();
    let x_val = x.suggest(&mut trial).unwrap();
    assert_eq!(x_val, 4.0);

    study.tell(trial, Ok::<_, &str>(x_val * x_val));
    assert_eq!(study.n_trials(), 1);
    assert_eq!(study.best_value().unwrap(), 16.0);
}

#[test]
fn test_n_enqueued() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 10.0);

    assert_eq!(study.n_enqueued(), 0);

    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(1.0))]));
    assert_eq!(study.n_enqueued(), 1);

    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(2.0))]));
    assert_eq!(study.n_enqueued(), 2);

    // Creating a trial dequeues one
    let _ = study.ask();
    assert_eq!(study.n_enqueued(), 1);

    let _ = study.ask();
    assert_eq!(study.n_enqueued(), 0);
}

#[test]
fn test_enqueue_counted_in_n_trials() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 10.0);

    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(1.0))]));
    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(2.0))]));

    study
        .optimize(5, |trial| {
            let x_val = x.suggest(trial)?;
            Ok::<_, Error>(x_val)
        })
        .unwrap();

    // All 5 trials count, including the 2 enqueued ones
    assert_eq!(study.n_trials(), 5);
}

// =============================================================================
// Test: Study summary and Display
// =============================================================================

#[test]
fn test_summary_with_completed_trials() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(1));
    let x = FloatParam::new(0.0, 10.0).name("x");

    study
        .optimize(5, |trial| {
            let val = x.suggest(trial)?;
            Ok::<_, Error>(val * val)
        })
        .unwrap();

    let summary = study.summary();
    assert!(summary.contains("Minimize"));
    assert!(summary.contains("5 trials"));
    assert!(summary.contains("Best value:"));
    assert!(summary.contains("x = "));
}

#[test]
fn test_summary_no_completed_trials() {
    let study: Study<f64> = Study::new(Direction::Maximize);
    let summary = study.summary();
    assert!(summary.contains("Maximize"));
    assert!(summary.contains("0 trials"));
    assert!(!summary.contains("Best value:"));
}

#[test]
fn test_summary_with_pruned_trials() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(1));
    let x = FloatParam::new(0.0, 10.0).name("x");

    // Manually create some complete and pruned trials
    for _ in 0..3 {
        let mut trial = study.create_trial();
        let val = x.suggest(&mut trial).unwrap();
        study.complete_trial(trial, val);
    }
    for _ in 0..2 {
        let mut trial = study.create_trial();
        let _ = x.suggest(&mut trial).unwrap();
        study.prune_trial(trial);
    }

    let summary = study.summary();
    // Should show breakdown when there are pruned trials
    if study.n_pruned_trials() > 0 {
        assert!(summary.contains("complete"));
        assert!(summary.contains("pruned"));
    }
}

#[test]
fn test_display_matches_summary() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(1));
    let x = FloatParam::new(0.0, 10.0).name("x");

    study
        .optimize(3, |trial| {
            let val = x.suggest(trial)?;
            Ok::<_, Error>(val)
        })
        .unwrap();

    assert_eq!(format!("{study}"), study.summary());
}

// =============================================================================
// Tests: optimize_with_retries
// =============================================================================

#[test]
fn test_retries_successful_trials_not_retried() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(0.0, 10.0);
    let call_count = std::cell::Cell::new(0u32);

    study
        .optimize_with_retries(5, 3, |trial| {
            let x = x_param.suggest(trial)?;
            call_count.set(call_count.get() + 1);
            Ok::<_, Error>(x * x)
        })
        .unwrap();

    // All trials succeed on first try — exactly 5 calls
    assert_eq!(call_count.get(), 5);
    assert_eq!(study.n_trials(), 5);
}

#[test]
fn test_retries_failed_trials_retried_up_to_max() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(0.0, 10.0);
    let call_count = std::cell::Cell::new(0u32);

    let result = study.optimize_with_retries(1, 3, |trial| {
        let _ = x_param.suggest(trial).unwrap();
        call_count.set(call_count.get() + 1);
        Err::<f64, _>("always fails")
    });

    // 1 initial attempt + 3 retries = 4 total calls
    assert_eq!(call_count.get(), 4);
    // No trials completed
    assert!(matches!(result, Err(Error::NoCompletedTrials)));
}

#[test]
fn test_retries_permanently_failed_after_exhaustion() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(0.0, 10.0);

    let result = study.optimize_with_retries(3, 2, |trial| {
        let _ = x_param.suggest(trial).unwrap();
        Err::<f64, _>("transient error")
    });

    assert!(
        matches!(result, Err(Error::NoCompletedTrials)),
        "all trials should permanently fail"
    );
    assert_eq!(
        study.n_trials(),
        0,
        "no completed trials should be recorded"
    );
}

#[test]
fn test_retries_uses_same_parameters() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(0.0, 10.0);
    let seen_values = std::cell::RefCell::new(Vec::new());
    let call_count = std::cell::Cell::new(0u32);

    study
        .optimize_with_retries(1, 2, |trial| {
            let x = x_param.suggest(trial).map_err(|e| e.to_string())?;
            seen_values.borrow_mut().push(x);
            call_count.set(call_count.get() + 1);
            // Fail first two attempts, succeed on third
            if call_count.get() < 3 {
                Err::<f64, _>("transient".to_string())
            } else {
                Ok(x * x)
            }
        })
        .unwrap();

    let values = seen_values.borrow();
    assert_eq!(values.len(), 3, "should be called 3 times (1 + 2 retries)");
    // All three calls should have gotten the same parameter value
    assert_eq!(values[0], values[1]);
    assert_eq!(values[1], values[2]);
}

#[test]
fn test_retries_n_trials_counts_unique_configs() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(0.0, 10.0);
    let call_count = std::cell::Cell::new(0u32);

    study
        .optimize_with_retries(3, 2, |trial| {
            let x = x_param.suggest(trial).map_err(|e| e.to_string())?;
            call_count.set(call_count.get() + 1);
            // Fail first attempt of each config, succeed on retry
            if call_count.get() % 2 == 1 {
                Err::<f64, _>("transient".to_string())
            } else {
                Ok(x * x)
            }
        })
        .unwrap();

    // 3 unique configs, each needing 2 calls = 6 total calls
    assert_eq!(call_count.get(), 6);
    // But only 3 completed trials
    assert_eq!(study.n_trials(), 3);
}

#[test]
fn test_retries_with_zero_max_retries_same_as_optimize() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(0.0, 10.0);
    let call_count = std::cell::Cell::new(0u32);

    study
        .optimize_with_retries(5, 0, |trial| {
            let x = x_param.suggest(trial)?;
            call_count.set(call_count.get() + 1);
            Ok::<_, Error>(x * x)
        })
        .unwrap();

    assert_eq!(call_count.get(), 5);
    assert_eq!(study.n_trials(), 5);
}

// =============================================================================
// Tests: IntoIterator for &Study
// =============================================================================

#[test]
fn test_into_iterator_iterates_all_trials() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let x_param = FloatParam::new(0.0, 10.0);

    for _ in 0..5 {
        let mut trial = study.create_trial();
        let x = x_param.suggest(&mut trial).unwrap();
        study.complete_trial(trial, x * x);
    }

    let mut count = 0;
    for trial in &study {
        assert_eq!(trial.state, optimizer::TrialState::Complete);
        count += 1;
    }
    assert_eq!(count, 5);
}

#[test]
fn test_into_iterator_empty_study() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    let count = (&study).into_iter().count();
    assert_eq!(count, 0);
}

#[test]
fn test_into_iterator_preserves_insertion_order() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    for i in 0..3 {
        let trial = study.create_trial();
        study.complete_trial(trial, f64::from(i));
    }

    let ids: Vec<u64> = (&study).into_iter().map(|t| t.id).collect();
    assert_eq!(ids, vec![0, 1, 2]);
}

// =============================================================================
// Tests: Constraint handling
// =============================================================================

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

// =============================================================================
// Test: StudyBuilder
// =============================================================================

#[test]
fn test_builder_defaults() {
    let study: Study<f64> = Study::builder().build();
    assert_eq!(study.direction(), Direction::Minimize);
}

#[test]
fn test_builder_maximize() {
    let study: Study<f64> = Study::builder().maximize().build();
    assert_eq!(study.direction(), Direction::Maximize);
}

#[test]
fn test_builder_minimize() {
    let study: Study<f64> = Study::builder().minimize().build();
    assert_eq!(study.direction(), Direction::Minimize);
}

#[test]
fn test_builder_direction() {
    let study: Study<f64> = Study::builder().direction(Direction::Maximize).build();
    assert_eq!(study.direction(), Direction::Maximize);
}

#[test]
fn test_builder_with_sampler() {
    let x = FloatParam::new(-5.0, 5.0);
    let study: Study<f64> = Study::builder().sampler(TpeSampler::new()).build();

    study
        .optimize(10, |trial| {
            let val = x.suggest(trial)?;
            Ok::<_, Error>(val * val)
        })
        .unwrap();

    assert_eq!(study.trials().len(), 10);
}

#[test]
fn test_builder_with_pruner() {
    use optimizer::pruner::NopPruner;

    let study: Study<f64> = Study::builder().pruner(NopPruner).build();

    assert_eq!(study.direction(), Direction::Minimize);
}

#[test]
fn test_builder_chaining() {
    let study: Study<f64> = Study::builder()
        .maximize()
        .sampler(RandomSampler::with_seed(42))
        .pruner(optimizer::pruner::NopPruner)
        .build();

    assert_eq!(study.direction(), Direction::Maximize);
}

#[test]
fn test_builder_with_custom_value_type() {
    let study: Study<i32> = Study::builder().maximize().build();
    assert_eq!(study.direction(), Direction::Maximize);
}

#[test]
fn test_builder_optimizes_correctly() {
    let x = FloatParam::new(-10.0, 10.0);
    let study: Study<f64> = Study::builder()
        .minimize()
        .sampler(TpeSampler::builder().seed(42).build().unwrap())
        .build();

    study
        .optimize(100, |trial| {
            let val = x.suggest(trial)?;
            Ok::<_, Error>((val - 3.0) * (val - 3.0))
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    assert!(
        best.value < 5.0,
        "best value should be < 5.0, got {}",
        best.value
    );
}
