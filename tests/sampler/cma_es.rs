use optimizer::prelude::*;
use optimizer::sampler::cma_es::CmaEsSampler;

#[test]
fn sphere_function() {
    let sampler = CmaEsSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x = FloatParam::new(-5.0, 5.0).name("x");
    let y = FloatParam::new(-5.0, 5.0).name("y");

    study
        .optimize(200, |trial| {
            let xv = x.suggest(trial)?;
            let yv = y.suggest(trial)?;
            Ok::<_, Error>(xv * xv + yv * yv)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    assert!(
        best.value < 1.0,
        "sphere best value should be < 1.0, got {}",
        best.value
    );
}

#[test]
fn rosenbrock_function() {
    let sampler = CmaEsSampler::builder().population_size(20).seed(42).build();
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x = FloatParam::new(-5.0, 5.0).name("x");
    let y = FloatParam::new(-5.0, 5.0).name("y");

    study
        .optimize(300, |trial| {
            let xv = x.suggest(trial)?;
            let yv = y.suggest(trial)?;
            let val = (1.0 - xv).powi(2) + 100.0 * (yv - xv * xv).powi(2);
            Ok::<_, Error>(val)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    // Rosenbrock minimum is 0 at (1, 1); we just check reasonable convergence
    assert!(
        best.value < 50.0,
        "rosenbrock best value should be < 50.0, got {}",
        best.value
    );
}

#[test]
fn bounds_respected() {
    let sampler = CmaEsSampler::with_seed(123);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x = FloatParam::new(-2.0, 3.0).name("x");
    let y = FloatParam::new(0.0, 10.0).name("y");

    study
        .optimize(100, |trial| {
            let xv = x.suggest(trial)?;
            let yv = y.suggest(trial)?;
            Ok::<_, Error>(xv + yv)
        })
        .unwrap();

    for trial in study.trials() {
        let xv: f64 = trial.get(&x).unwrap();
        let yv: f64 = trial.get(&y).unwrap();
        assert!((-2.0..=3.0).contains(&xv), "x = {xv} out of bounds [-2, 3]");
        assert!((0.0..=10.0).contains(&yv), "y = {yv} out of bounds [0, 10]");
    }
}

#[test]
fn mixed_params_float_and_categorical() {
    let sampler = CmaEsSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x = FloatParam::new(-5.0, 5.0).name("x");
    let cat = CategoricalParam::new(vec!["a", "b", "c"]).name("cat");

    study
        .optimize(50, |trial| {
            let xv = x.suggest(trial)?;
            let cv = cat.suggest(trial)?;
            let penalty = match cv {
                "a" => 0.0,
                "b" => 1.0,
                _ => 2.0,
            };
            Ok::<_, Error>(xv * xv + penalty)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    // Should find a reasonable value
    assert!(
        best.value < 10.0,
        "best value should be < 10.0, got {}",
        best.value
    );
}

#[test]
fn seeded_reproducibility() {
    let x = FloatParam::new(-5.0, 5.0).name("x");
    let y = FloatParam::new(-5.0, 5.0).name("y");

    let run = |seed: u64| {
        let sampler = CmaEsSampler::with_seed(seed);
        let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
        study
            .optimize(50, |trial| {
                let xv = x.suggest(trial)?;
                let yv = y.suggest(trial)?;
                Ok::<_, Error>(xv * xv + yv * yv)
            })
            .unwrap();
        study.trials().iter().map(|t| t.value).collect::<Vec<_>>()
    };

    let results1 = run(42);
    let results2 = run(42);
    assert_eq!(results1, results2, "same seed should produce same results");
}

#[test]
fn different_seeds_different_results() {
    let x = FloatParam::new(-5.0, 5.0).name("x");
    let y = FloatParam::new(-5.0, 5.0).name("y");

    let run = |seed: u64| {
        let sampler = CmaEsSampler::with_seed(seed);
        let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
        study
            .optimize(20, |trial| {
                let xv = x.suggest(trial)?;
                let yv = y.suggest(trial)?;
                Ok::<_, Error>(xv * xv + yv * yv)
            })
            .unwrap();
        study.trials().iter().map(|t| t.value).collect::<Vec<_>>()
    };

    let results1 = run(42);
    let results2 = run(99);
    assert_ne!(
        results1, results2,
        "different seeds should produce different results"
    );
}

#[test]
fn single_dimension() {
    let sampler = CmaEsSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x = FloatParam::new(-10.0, 10.0).name("x");

    study
        .optimize(100, |trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, Error>((xv - 3.0).powi(2))
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    assert!(
        best.value < 1.0,
        "1-D optimization should converge, got {}",
        best.value
    );
}

#[test]
fn integer_params() {
    let sampler = CmaEsSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let n = IntParam::new(1, 20).name("n");

    study
        .optimize(100, |trial| {
            let nv = n.suggest(trial)?;
            // Minimum at n = 10
            Ok::<_, Error>(((nv - 10) * (nv - 10)) as f64)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    let best_n: i64 = best.get(&n).unwrap();
    assert!(
        (1..=20).contains(&best_n),
        "integer value {best_n} out of bounds"
    );
    assert!(
        best.value < 10.0,
        "integer optimization should converge, got {}",
        best.value
    );
}

#[test]
fn log_scale_params() {
    let sampler = CmaEsSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let lr = FloatParam::new(1e-5, 1.0).log_scale().name("lr");

    study
        .optimize(100, |trial| {
            let lrv = lr.suggest(trial)?;
            // Minimum at lr = 0.01
            Ok::<_, Error>((lrv.ln() - 0.01_f64.ln()).powi(2))
        })
        .unwrap();

    for trial in study.trials() {
        let lrv: f64 = trial.get(&lr).unwrap();
        assert!(
            (1e-5..=1.0).contains(&lrv),
            "log-scale value {lrv} out of bounds"
        );
    }
}

#[test]
fn custom_population_size_and_sigma() {
    let sampler = CmaEsSampler::builder()
        .sigma0(1.0)
        .population_size(10)
        .seed(42)
        .build();
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x = FloatParam::new(-5.0, 5.0).name("x");
    let y = FloatParam::new(-5.0, 5.0).name("y");

    study
        .optimize(100, |trial| {
            let xv = x.suggest(trial)?;
            let yv = y.suggest(trial)?;
            Ok::<_, Error>(xv * xv + yv * yv)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    assert!(
        best.value < 5.0,
        "custom config optimization should work, got {}",
        best.value
    );
}
