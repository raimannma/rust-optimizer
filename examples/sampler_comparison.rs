//! Sampler comparison example — benchmarks Random, TPE, and Grid samplers on the same problem.
//!
//! Runs the Sphere function f(x, y) = x² + y² with each sampler and compares the best
//! value found. This shows how sampler choice affects optimization quality.
//!
//! Run with: `cargo run --example sampler_comparison`

use optimizer::prelude::*;

/// Shared objective function: Sphere function with global minimum at (0, 0).
/// Simple enough to solve well, but 2-D so samplers have room to differ.
fn sphere(x: f64, y: f64) -> f64 {
    x.powi(2) + y.powi(2)
}

/// Run an optimization study and return the best value found.
fn run_study(study: Study<f64>, n_trials: usize) -> f64 {
    // Use asymmetric ranges so the Grid sampler tracks each parameter independently.
    let x = FloatParam::new(-5.0, 5.0).name("x");
    let y = FloatParam::new(-3.0, 3.0).name("y");

    study
        .optimize(n_trials, |trial| {
            let x_val = x.suggest(trial)?;
            let y_val = y.suggest(trial)?;
            Ok::<_, Error>(sphere(x_val, y_val))
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    println!(
        "  Best trial #{:>3}: x = {:>7.4}, y = {:>7.4}, f(x,y) = {:.6}",
        best.id,
        best.get(&x).unwrap(),
        best.get(&y).unwrap(),
        best.value,
    );
    best.value
}

fn main() {
    let n_trials: usize = 100;
    println!("Comparing samplers on Sphere(x, y) = x² + y²  ({n_trials} trials each)");
    println!();

    // --- Random sampler (baseline) ---
    // Pure random search: samples uniformly at random. Fast but not guided.
    println!("1. Random sampler:");
    let random_best = run_study(Study::minimize(RandomSampler::with_seed(42)), n_trials);

    // --- TPE sampler (Bayesian) ---
    // Tree-structured Parzen Estimator: builds a probabilistic model of good vs bad
    // regions and focuses sampling where improvements are likely.
    println!("\n2. TPE sampler (Bayesian):");
    let tpe = TpeSampler::builder()
        .n_startup_trials(10) // random exploration for the first 10 trials
        .n_ei_candidates(24) // candidates evaluated per Expected Improvement step
        .gamma(0.25) // top 25% of trials define the "good" distribution
        .seed(42)
        .build()
        .unwrap();
    let tpe_best = run_study(Study::minimize(tpe), n_trials);

    // --- Grid sampler (exhaustive) ---
    // Evaluates evenly spaced grid points. Each parameter gets its own grid that
    // is sampled in order, so n_points_per_param must be >= n_trials.
    println!("\n3. Grid sampler (exhaustive):");
    let grid = GridSearchSampler::builder()
        .n_points_per_param(n_trials) // one grid point per trial per parameter
        .build();
    let grid_best = run_study(Study::minimize(grid), n_trials);

    // --- Summary ---
    println!("\n--- Summary ---");
    println!("  Random : {random_best:.6}");
    println!("  TPE    : {tpe_best:.6}");
    println!("  Grid   : {grid_best:.6}");
}
