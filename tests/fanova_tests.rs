//! Integration tests for fANOVA parameter importance.

use optimizer::prelude::*;

#[test]
fn fanova_dominant_parameter() {
    // f(x, y) = x^2 — x should dominate
    let x = FloatParam::new(0.0, 10.0).name("x");
    let y = FloatParam::new(0.0, 10.0).name("y");

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    study
        .optimize(50, |trial| {
            let xv = x.suggest(trial)?;
            let _yv = y.suggest(trial)?;
            Ok::<_, Error>(xv * xv)
        })
        .unwrap();

    let result = study.fanova().unwrap();
    assert_eq!(result.main_effects[0].0, "x");
    assert!(
        result.main_effects[0].1 > 0.7,
        "x importance = {}",
        result.main_effects[0].1
    );
}

#[test]
fn fanova_interaction() {
    // f(x, y) = x * y — both matter and interact
    let x = FloatParam::new(0.0, 10.0).name("x");
    let y = FloatParam::new(0.0, 10.0).name("y");

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(7));
    study
        .optimize(100, |trial| {
            let xv = x.suggest(trial)?;
            let yv = y.suggest(trial)?;
            Ok::<_, Error>(xv * yv)
        })
        .unwrap();

    let config = FanovaConfig {
        n_trees: 128,
        ..FanovaConfig::default()
    };
    let result = study.fanova_with_config(&config).unwrap();

    // Should detect interaction
    assert!(
        !result.interactions.is_empty(),
        "should detect x*y interaction"
    );
}

#[test]
fn fanova_consistent_with_correlation() {
    // f(x, y) = 3*x + 0.5*y — x should rank higher in both methods
    let x = FloatParam::new(0.0, 10.0).name("x");
    let y = FloatParam::new(0.0, 10.0).name("y");

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(99));
    study
        .optimize(80, |trial| {
            let xv = x.suggest(trial)?;
            let yv = y.suggest(trial)?;
            Ok::<_, Error>(3.0 * xv + 0.5 * yv)
        })
        .unwrap();

    let corr = study.param_importance();
    let fanova = study.fanova().unwrap();

    // Both methods should rank x above y
    assert_eq!(corr[0].0, "x", "correlation should rank x first");
    assert_eq!(fanova.main_effects[0].0, "x", "fanova should rank x first");
}

#[test]
fn fanova_too_few_trials() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    let result = study.fanova();
    assert!(result.is_err(), "should error with no trials");
}
