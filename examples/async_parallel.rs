//! Async parallel optimization — evaluate multiple trials concurrently.
//!
//! Uses `optimize_parallel` with tokio to run several trials at once,
//! reducing wall-clock time when the objective involves I/O or async work.
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

    study
        .optimize_parallel(n_trials, concurrency, {
            let x = x.clone();
            let y = y.clone();
            move |mut trial| {
                let x = x.clone();
                let y = y.clone();
                async move {
                    let xv = x.suggest(&mut trial)?;
                    let yv = y.suggest(&mut trial)?;

                    // Simulate async I/O (e.g. calling an external service)
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

                    let value = xv * xv + yv * yv;
                    Ok::<_, optimizer::Error>((trial, value))
                }
            }
        })
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
