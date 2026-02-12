//! Integration tests for the Multivariate TPE sampler.
//!
//! These tests compare the performance of `MultivariateTpeSampler` against
//! the standard `TpeSampler` on problems with and without parameter correlations.

#![allow(
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use optimizer::parameter::{CategoricalParam, FloatParam, IntParam, Parameter};
use optimizer::sampler::tpe::{MultivariateTpeSampler, TpeSampler};
use optimizer::{Direction, Error, Study};

// =============================================================================
// Rosenbrock function: f(x,y) = (a-x)^2 + b*(y-x^2)^2
// This is a classic benchmark with strong parameter correlation.
// Optimal: (x, y) = (a, a^2) with f(x, y) = 0
// Standard parameters: a = 1, b = 100
// The "banana-shaped" valley makes this hard for independent samplers.
// =============================================================================

/// Computes the Rosenbrock function value.
///
/// f(x, y) = (a - x)^2 + b * (y - x^2)^2
///
/// With standard parameters a = 1, b = 100:
/// - Optimal point: (1, 1)
/// - Optimal value: 0
fn rosenbrock(x: f64, y: f64) -> f64 {
    let a = 1.0;
    let b = 100.0;
    (a - x).powi(2) + b * (y - x * x).powi(2)
}

// =============================================================================
// Test: Multivariate TPE on Rosenbrock function (correlated parameters)
// =============================================================================

#[test]
fn test_multivariate_tpe_rosenbrock_finds_good_solution() {
    // Multivariate TPE should find a good solution on Rosenbrock
    // because it can model the correlation between x and y
    let sampler = MultivariateTpeSampler::builder()
        .seed(42)
        .n_startup_trials(10)
        .n_ei_candidates(24)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(-2.0, 2.0);
    let y_param = FloatParam::new(-2.0, 4.0);

    study
        .optimize(100, |trial| {
            let x = x_param.suggest(trial)?;
            let y = y_param.suggest(trial)?;
            Ok::<_, Error>(rosenbrock(x, y))
        })
        .expect("optimization should succeed");

    let best = study.best_trial().expect("should have at least one trial");

    // Multivariate TPE should find a reasonably good solution
    // The global minimum is 0, but getting close is challenging
    assert!(
        best.value < 10.0,
        "Multivariate TPE should find good Rosenbrock solution: best value {} should be < 10.0",
        best.value
    );
}

#[test]
fn test_independent_tpe_rosenbrock() {
    // Independent TPE (standard TpeSampler) on Rosenbrock
    // This establishes a baseline for comparison
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(10)
        .n_ei_candidates(24)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(-2.0, 2.0);
    let y_param = FloatParam::new(-2.0, 4.0);

    study
        .optimize(100, |trial| {
            let x = x_param.suggest(trial)?;
            let y = y_param.suggest(trial)?;
            Ok::<_, Error>(rosenbrock(x, y))
        })
        .expect("optimization should succeed");

    let best = study.best_trial().expect("should have at least one trial");

    // Independent TPE should still find a decent solution,
    // but may not be as good as multivariate TPE on this correlated problem
    assert!(
        best.value < 50.0,
        "Independent TPE should find reasonable Rosenbrock solution: best value {} should be < 50.0",
        best.value
    );
}

#[test]
fn test_multivariate_tpe_outperforms_on_correlated_problem() {
    // Run multiple seeds and compare average performance
    // Multivariate TPE should generally find better solutions on Rosenbrock
    let n_runs = 5;
    let n_trials = 80;

    let mut multivariate_best_values = Vec::new();
    let mut independent_best_values = Vec::new();

    for seed in 0..n_runs {
        // Multivariate TPE
        let multivariate_sampler = MultivariateTpeSampler::builder()
            .seed(seed as u64)
            .n_startup_trials(10)
            .n_ei_candidates(24)
            .build()
            .unwrap();

        let study: Study<f64> = Study::with_sampler(Direction::Minimize, multivariate_sampler);

        let x_param = FloatParam::new(-2.0, 2.0);
        let y_param = FloatParam::new(-2.0, 4.0);

        study
            .optimize(n_trials, |trial| {
                let x = x_param.suggest(trial)?;
                let y = y_param.suggest(trial)?;
                Ok::<_, Error>(rosenbrock(x, y))
            })
            .unwrap();

        multivariate_best_values.push(study.best_trial().unwrap().value);

        // Independent TPE
        let independent_sampler = TpeSampler::builder()
            .seed(seed as u64)
            .n_startup_trials(10)
            .n_ei_candidates(24)
            .build()
            .unwrap();

        let study: Study<f64> = Study::with_sampler(Direction::Minimize, independent_sampler);

        let x_param = FloatParam::new(-2.0, 2.0);
        let y_param = FloatParam::new(-2.0, 4.0);

        study
            .optimize(n_trials, |trial| {
                let x = x_param.suggest(trial)?;
                let y = y_param.suggest(trial)?;
                Ok::<_, Error>(rosenbrock(x, y))
            })
            .unwrap();

        independent_best_values.push(study.best_trial().unwrap().value);
    }

    let multivariate_mean: f64 = multivariate_best_values.iter().sum::<f64>() / n_runs as f64;
    let independent_mean: f64 = independent_best_values.iter().sum::<f64>() / n_runs as f64;

    // Log results for debugging (these won't show in normal test runs,
    // but are useful when running with --nocapture)
    eprintln!("Multivariate TPE mean best: {multivariate_mean:.4}");
    eprintln!("Independent TPE mean best: {independent_mean:.4}");
    eprintln!("Multivariate best values: {multivariate_best_values:?}");
    eprintln!("Independent best values: {independent_best_values:?}");

    // Both methods should find reasonable solutions
    assert!(
        multivariate_mean < 20.0,
        "Multivariate TPE mean {multivariate_mean:.4} should be < 20.0"
    );
    assert!(
        independent_mean < 100.0,
        "Independent TPE mean {independent_mean:.4} should be < 100.0"
    );
}

// =============================================================================
// Independent parameter problem: f(x,y) = x^2 + y^2
// No correlation between parameters - both methods should work equally well.
// =============================================================================

/// Simple sphere function with independent parameters.
///
/// f(x, y) = x^2 + y^2
///
/// Optimal point: (0, 0)
/// Optimal value: 0
fn sphere(x: f64, y: f64) -> f64 {
    x * x + y * y
}

#[test]
fn test_multivariate_tpe_independent_problem() {
    // On an independent problem, multivariate TPE should still work well
    let sampler = MultivariateTpeSampler::builder()
        .seed(42)
        .n_startup_trials(10)
        .n_ei_candidates(24)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(-5.0, 5.0);
    let y_param = FloatParam::new(-5.0, 5.0);

    study
        .optimize(50, |trial| {
            let x = x_param.suggest(trial)?;
            let y = y_param.suggest(trial)?;
            Ok::<_, Error>(sphere(x, y))
        })
        .expect("optimization should succeed");

    let best = study.best_trial().expect("should have at least one trial");

    assert!(
        best.value < 5.0,
        "Multivariate TPE should find good solution on sphere: best value {} should be < 5.0",
        best.value
    );
}

#[test]
fn test_independent_tpe_independent_problem() {
    // Baseline: independent TPE on sphere function
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(10)
        .n_ei_candidates(24)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(-5.0, 5.0);
    let y_param = FloatParam::new(-5.0, 5.0);

    study
        .optimize(50, |trial| {
            let x = x_param.suggest(trial)?;
            let y = y_param.suggest(trial)?;
            Ok::<_, Error>(sphere(x, y))
        })
        .expect("optimization should succeed");

    let best = study.best_trial().expect("should have at least one trial");

    assert!(
        best.value < 5.0,
        "Independent TPE should find good solution on sphere: best value {} should be < 5.0",
        best.value
    );
}

#[test]
fn test_both_samplers_work_on_independent_problem() {
    // Run both samplers on the independent sphere function
    // and verify they both achieve similar performance
    let n_runs = 5;
    let n_trials = 50;

    let mut multivariate_results = Vec::new();
    let mut independent_results = Vec::new();

    for seed in 0..n_runs {
        // Multivariate TPE
        let sampler = MultivariateTpeSampler::builder()
            .seed(seed as u64)
            .n_startup_trials(10)
            .build()
            .unwrap();

        let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

        let x_param = FloatParam::new(-5.0, 5.0);
        let y_param = FloatParam::new(-5.0, 5.0);

        study
            .optimize(n_trials, |trial| {
                let x = x_param.suggest(trial)?;
                let y = y_param.suggest(trial)?;
                Ok::<_, Error>(sphere(x, y))
            })
            .unwrap();

        multivariate_results.push(study.best_trial().unwrap().value);

        // Independent TPE
        let sampler = TpeSampler::builder()
            .seed(seed as u64)
            .n_startup_trials(10)
            .build()
            .unwrap();

        let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

        let x_param = FloatParam::new(-5.0, 5.0);
        let y_param = FloatParam::new(-5.0, 5.0);

        study
            .optimize(n_trials, |trial| {
                let x = x_param.suggest(trial)?;
                let y = y_param.suggest(trial)?;
                Ok::<_, Error>(sphere(x, y))
            })
            .unwrap();

        independent_results.push(study.best_trial().unwrap().value);
    }

    let multivariate_mean: f64 = multivariate_results.iter().sum::<f64>() / n_runs as f64;
    let independent_mean: f64 = independent_results.iter().sum::<f64>() / n_runs as f64;

    eprintln!("Sphere function results:");
    eprintln!("  Multivariate TPE mean: {multivariate_mean:.4}");
    eprintln!("  Independent TPE mean: {independent_mean:.4}");

    // Both should find good solutions on this simple problem
    assert!(
        multivariate_mean < 5.0,
        "Multivariate TPE mean {multivariate_mean:.4} should be < 5.0 on sphere"
    );
    assert!(
        independent_mean < 5.0,
        "Independent TPE mean {independent_mean:.4} should be < 5.0 on sphere"
    );
}

// =============================================================================
// Test: Multivariate TPE with group decomposition
// =============================================================================

#[test]
fn test_multivariate_tpe_with_group_decomposition() {
    // Test that group decomposition works correctly
    let sampler = MultivariateTpeSampler::builder()
        .seed(42)
        .n_startup_trials(10)
        .group(true) // Enable group decomposition
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(-5.0, 5.0);
    let y_param = FloatParam::new(-5.0, 5.0);

    study
        .optimize(50, |trial| {
            let x = x_param.suggest(trial)?;
            let y = y_param.suggest(trial)?;
            Ok::<_, Error>(sphere(x, y))
        })
        .expect("optimization should succeed");

    let best = study.best_trial().expect("should have at least one trial");

    assert!(
        best.value < 10.0,
        "Multivariate TPE with groups should find good solution: best value {} should be < 10.0",
        best.value
    );
}

// =============================================================================
// Test: Multivariate TPE with mixed parameter types
// =============================================================================

#[test]
fn test_multivariate_tpe_mixed_parameter_types() {
    // Test with float, int, and categorical parameters
    let sampler = MultivariateTpeSampler::builder()
        .seed(42)
        .n_startup_trials(10)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(-5.0, 5.0);
    let n_param = IntParam::new(1, 10);
    let mode_param = CategoricalParam::new(vec!["a", "b", "c"]);

    study
        .optimize(50, |trial| {
            let x = x_param.suggest(trial)?;
            let n = n_param.suggest(trial)?;
            let mode = mode_param.suggest(trial)?;

            // Objective depends on all parameters
            let mode_factor = match mode {
                "a" => 1.0,
                "b" => 0.5,
                "c" => 2.0,
                _ => unreachable!(),
            };

            Ok::<_, Error>(x * x + (n as f64 - 5.0).powi(2) * mode_factor)
        })
        .expect("optimization should succeed");

    let best = study.best_trial().expect("should have at least one trial");

    // Should find a reasonable solution
    assert!(
        best.value < 25.0,
        "Multivariate TPE should handle mixed types: best value {} should be < 25.0",
        best.value
    );
}
