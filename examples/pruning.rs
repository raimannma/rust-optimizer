//! Trial pruning — stop unpromising trials early with `MedianPruner`.
//!
//! When your objective involves an iterative loop (e.g. training epochs),
//! the pruner compares intermediate values across trials and kills the
//! ones that fall below the median — saving compute on bad configurations.
//!
//! Run with: `cargo run --example pruning`

use optimizer::prelude::*;

fn main() -> optimizer::Result<()> {
    // MedianPruner prunes trials whose intermediate value falls below the
    // median of previously completed trials at the same step.
    let study: Study<f64> = Study::builder()
        .minimize()
        .sampler(RandomSampler::with_seed(42))
        .pruner(
            MedianPruner::new(Direction::Minimize)
                .n_warmup_steps(3) // run at least 3 epochs before pruning
                .n_min_trials(3), // need 3 completed trials before pruning kicks in
        )
        .build();

    let lr = FloatParam::new(1e-4, 1.0).name("learning_rate");
    let momentum = FloatParam::new(0.0, 0.99).name("momentum");

    let n_epochs: u64 = 20;

    study.optimize(30, |trial: &mut optimizer::Trial| {
        let lr_val = lr.suggest(trial)?;
        let mom = momentum.suggest(trial)?;

        // Simulated training loop — good hyperparameters converge to low loss,
        // bad ones plateau high, giving the pruner something to cut.
        let mut loss = 1.0;
        for epoch in 0..n_epochs {
            let lr_penalty = (lr_val.log10() - 0.01_f64.log10()).powi(2);
            let mom_penalty = (mom - 0.8).powi(2);
            let base_loss = 0.02 + 0.05 * lr_penalty + 1.5 * mom_penalty;
            let progress = (epoch as f64 + 1.0) / n_epochs as f64;
            loss = base_loss + (1.0 - base_loss) * (-3.5 * progress).exp();

            // Report intermediate value so the pruner can evaluate this trial.
            trial.report(epoch, loss);

            // Check whether the pruner recommends stopping early.
            if trial.should_prune() {
                Err(TrialPruned)?;
            }
        }

        Ok::<_, Error>(loss)
    })?;

    // --- Results ---
    let best = study.best_trial()?;
    println!(
        "Completed {} trials ({} pruned)",
        study.n_trials(),
        study.n_pruned_trials()
    );
    println!("Best trial #{}: loss = {:.6}", best.id, best.value);
    println!("  learning_rate = {:.6}", best.get(&lr).unwrap());
    println!("  momentum      = {:.4}", best.get(&momentum).unwrap());

    Ok(())
}
