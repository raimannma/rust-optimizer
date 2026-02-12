//! Basic optimization example — the "hello world" of the optimizer crate.
//!
//! Minimizes a simple quadratic function f(x) = (x - 3)² using the default
//! random sampler. No feature flags are required.
//!
//! Run with: `cargo run --example basic_optimization`

use optimizer::prelude::*;

fn main() {
    // Create a study that minimizes the objective function.
    // The default sampler is random; for smarter sampling, pass a TpeSampler.
    let study: Study<f64> = Study::new(Direction::Minimize);

    // Search for x in [-10, 10]. The optimizer will suggest values from this range.
    let x = FloatParam::new(-10.0, 10.0).name("x");

    // Run 50 trials, each evaluating f(x) = (x - 3)²
    study
        .optimize(50, |trial: &mut optimizer::Trial| {
            let x_val = x.suggest(trial)?;
            let value = (x_val - 3.0).powi(2);
            Ok::<_, Error>(value)
        })
        .unwrap();

    // Retrieve and display the best result
    let best = study.best_trial().unwrap();
    println!("Best trial #{}", best.id);
    println!("  x     = {:.4}", best.get(&x).unwrap());
    println!("  f(x)  = {:.4}", best.value);
}
