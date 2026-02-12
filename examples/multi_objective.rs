//! Multi-objective optimization — optimize competing objectives simultaneously.
//!
//! `MultiObjectiveStudy` returns the Pareto front: the set of solutions where
//! no objective can be improved without worsening another.
//!
//! Run with: `cargo run --example multi_objective`

use optimizer::multi_objective::MultiObjectiveStudy;
use optimizer::prelude::*;

fn main() -> optimizer::Result<()> {
    let study = MultiObjectiveStudy::new(vec![Direction::Minimize, Direction::Minimize]);

    let x = FloatParam::new(0.0, 1.0).name("x");

    // Classic bi-objective: f1(x) = x², f2(x) = (x-1)²
    // The Pareto front is the curve where improving f1 worsens f2.
    study.optimize(50, |trial: &mut optimizer::Trial| {
        let xv = x.suggest(trial)?;
        let f1 = xv * xv;
        let f2 = (xv - 1.0) * (xv - 1.0);
        Ok::<_, optimizer::Error>(vec![f1, f2])
    })?;

    let front = study.pareto_front();
    println!(
        "Ran {} trials, Pareto front has {} solutions:",
        study.n_trials(),
        front.len(),
    );

    let mut sorted = front.clone();
    sorted.sort_by(|a, b| a.values[0].partial_cmp(&b.values[0]).unwrap());

    for (i, trial) in sorted.iter().take(5).enumerate() {
        println!(
            "  {}: x={:.3}, f1={:.4}, f2={:.4}",
            i + 1,
            trial.get(&x).unwrap(),
            trial.values[0],
            trial.values[1],
        );
    }
    if sorted.len() > 5 {
        println!("  ... and {} more", sorted.len() - 5);
    }

    Ok(())
}
