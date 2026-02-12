//! Async parallel optimization — evaluate multiple trials concurrently.
//!
//! Uses `optimize_parallel` with tokio to run several trials at once,
//! reducing wall-clock time when the objective involves blocking work.
//! Each sync closure is internally wrapped in `spawn_blocking`.
//!
//! Run with: `cargo run --example async_parallel --features async`

use optimizer::prelude::*;

#[tokio::main]
async fn main() -> optimizer::Result<()> {
    let study: Study<f64> = Study::minimize(TpeSampler::new());

    let x = FloatParam::new(-5.0, 5.0).name("x");
    let y = FloatParam::new(-5.0, 5.0).name("y");

    let n_trials = 30;
    let concurrency = 4;

    println!("Running {n_trials} trials with {concurrency} concurrent workers...");

    let xc = x.clone();
    let yc = y.clone();
    study
        .optimize_parallel(
            n_trials,
            concurrency,
            move |trial: &mut optimizer::Trial| {
                let xv = xc.suggest(trial)?;
                let yv = yc.suggest(trial)?;
                Ok::<_, optimizer::Error>(xv * xv + yv * yv)
            },
        )
        .await?;

    let best = study.best_trial()?;
    println!(
        "Best: f({:.3}, {:.3}) = {:.6}",
        best.get(&x).unwrap(),
        best.get(&y).unwrap(),
        best.value,
    );

    Ok(())
}
