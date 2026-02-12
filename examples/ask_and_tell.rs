//! Ask-and-tell interface — decouple sampling from evaluation.
//!
//! Use `ask()` to get a trial with sampled parameters, evaluate it however
//! you like (workers, GPUs, external processes), then `tell()` the result.
//! This is useful for batch evaluation or custom scheduling.
//!
//! Run with: `cargo run --example ask_and_tell`

use optimizer::prelude::*;

fn main() -> optimizer::Result<()> {
    let study: Study<f64> = Study::new(Direction::Minimize);

    let x = FloatParam::new(-5.0, 5.0).name("x");
    let y = FloatParam::new(-5.0, 5.0).name("y");

    for batch in 0..3 {
        let batch_size = 5;
        let mut trials = Vec::with_capacity(batch_size);

        // ask() creates trials with sampled parameters
        for _ in 0..batch_size {
            let mut trial = study.ask();
            let xv = x.suggest(&mut trial)?;
            let yv = y.suggest(&mut trial)?;
            trials.push((trial, xv, yv));
        }

        // Evaluate the batch (could be sent to workers, GPUs, etc.)
        for (trial, xv, yv) in trials {
            let value = xv * xv + yv * yv;
            study.tell(trial, Ok::<_, &str>(value));
        }

        println!(
            "Batch {}: evaluated {batch_size} trials (total: {})",
            batch + 1,
            study.n_trials(),
        );
    }

    let best = study.best_trial()?;
    println!(
        "Best: f({:.3}, {:.3}) = {:.6}",
        best.get(&x).unwrap(),
        best.get(&y).unwrap(),
        best.value,
    );

    Ok(())
}
