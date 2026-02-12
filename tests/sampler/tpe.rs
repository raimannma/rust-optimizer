use optimizer::parameter::{BoolParam, CategoricalParam, FloatParam, IntParam, Parameter};
use optimizer::sampler::tpe::TpeSampler;
use optimizer::{Direction, Error, Study};

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
