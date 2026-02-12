//! Early stopping — halt an entire study once a target is reached.
//!
//! Use `optimize_with_callback` to inspect each completed trial and return
//! `ControlFlow::Break(())` when the study should stop (e.g. a quality
//! threshold is met or a time budget is exhausted).
//!
//! Run with: `cargo run --example early_stopping`

use std::ops::ControlFlow;

use optimizer::prelude::*;

fn main() -> optimizer::Result<()> {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(-10.0, 10.0).name("x");

    let target = 0.01;

    study.optimize_with_callback(
        100, // upper bound — we expect to stop much earlier
        |trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, Error>((xv - 3.0).powi(2))
        },
        |_study, completed| {
            if completed.value < target {
                println!("Target {target} reached at trial #{}", completed.id);
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        },
    )?;

    let best = study.best_trial()?;
    println!(
        "Stopped after {} trials — best f({:.4}) = {:.6}",
        study.n_trials(),
        best.get(&x).unwrap(),
        best.value,
    );

    Ok(())
}
