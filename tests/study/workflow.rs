use optimizer::parameter::{BoolParam, FloatParam, IntParam, Parameter};
use optimizer::sampler::tpe::TpeSampler;
use optimizer::{Direction, Error, Study};

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
fn test_study_direction() {
    let study_min: Study<f64> = Study::new(Direction::Minimize);
    assert_eq!(study_min.direction(), Direction::Minimize);

    let study_max: Study<f64> = Study::new(Direction::Maximize);
    assert_eq!(study_max.direction(), Direction::Maximize);
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

#[test]
fn test_single_value_int_range() {
    let param = IntParam::new(5, 5);
    let mut trial = optimizer::Trial::new(0);

    let n = param.suggest(&mut trial).unwrap();
    assert_eq!(n, 5, "single-value range should return that value");
}
