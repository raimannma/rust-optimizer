//! Journal storage — persist trials to disk and resume later.
//!
//! `JournalStorage` writes every trial to a JSONL file so that a study can
//! be resumed after a crash or across separate runs.
//!
//! Run with: `cargo run --example journal_storage --features journal`

use optimizer::prelude::*;

fn main() -> optimizer::Result<()> {
    let path = std::env::temp_dir().join("optimizer_journal_example.jsonl");

    // Clean up from any previous run
    let _ = std::fs::remove_file(&path);

    let x = FloatParam::new(-5.0, 5.0).name("x");

    // --- First run: optimize 20 trials and persist to disk ---
    {
        let storage = JournalStorage::<f64>::new(&path);
        let study: Study<f64> = Study::builder()
            .minimize()
            .sampler(TpeSampler::new())
            .storage(storage)
            .build();

        study.optimize(20, |trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(xv * xv)
        })?;

        println!(
            "First run: {} trials saved to {}",
            study.n_trials(),
            path.display(),
        );
    }

    // --- Second run: resume from the journal file ---
    {
        let storage = JournalStorage::<f64>::open(&path)?;
        let study: Study<f64> = Study::builder()
            .minimize()
            .sampler(TpeSampler::new())
            .storage(storage)
            .build();

        let before = study.n_trials();
        study.optimize(10, |trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(xv * xv)
        })?;

        let best = study.best_trial()?;
        println!(
            "Resumed: {} → {} trials, best f({:.4}) = {:.6}",
            before,
            study.n_trials(),
            best.get(&x).unwrap(),
            best.value,
        );
    }

    // Clean up
    let _ = std::fs::remove_file(&path);

    Ok(())
}
