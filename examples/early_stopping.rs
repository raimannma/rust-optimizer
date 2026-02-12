//! Early stopping — halt an entire study once a target is reached.
//!
//! Implements the [`Objective`] trait on a custom struct and uses the
//! [`after_trial`](Objective::after_trial) hook to return
//! `ControlFlow::Break(())` when the best value drops below a threshold.
//!
//! Run with: `cargo run --example early_stopping`

use std::ops::ControlFlow;

use optimizer::prelude::*;

/// An objective that minimises `(x - 3)^2` and stops early once the
/// value drops below `target`.
struct EarlyStopObjective {
    x: FloatParam,
    target: f64,
}

impl Objective<f64> for EarlyStopObjective {
    type Error = Error;

    fn evaluate(&self, trial: &mut Trial) -> Result<f64> {
        let v = self.x.suggest(trial)?;
        Ok((v - 3.0).powi(2))
    }

    fn after_trial(&self, _study: &Study<f64>, trial: &CompletedTrial<f64>) -> ControlFlow<()> {
        if trial.value < self.target {
            println!("Target {} reached at trial #{}", self.target, trial.id);
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

fn main() -> optimizer::Result<()> {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(-10.0, 10.0).name("x");

    let objective = EarlyStopObjective {
        x: x.clone(),
        target: 0.01,
    };

    study.optimize(100, objective)?;

    let best = study.best_trial()?;
    println!(
        "Stopped after {} trials — best f({:.4}) = {:.6}",
        study.n_trials(),
        best.get(&x).unwrap(),
        best.value,
    );

    Ok(())
}
