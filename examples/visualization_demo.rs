use optimizer::prelude::*;

fn main() {
    // Multi-parameter optimization with TPE sampler.
    let sampler = TpeSampler::builder().seed(42).build().unwrap();
    let mut study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    study.set_pruner(MedianPruner::new(Direction::Minimize));

    let lr = FloatParam::new(1e-5, 1e-1)
        .log_scale()
        .name("learning_rate");
    let n_layers = IntParam::new(1, 5).name("n_layers");
    let dropout = FloatParam::new(0.0, 0.5).step(0.05).name("dropout");
    let batch_size = CategoricalParam::new(vec![16, 32, 64, 128]).name("batch_size");

    study
        .optimize(80, |trial| {
            let lr_val = lr.suggest(trial)?;
            let layers = n_layers.suggest(trial)?;
            let drop = dropout.suggest(trial)?;
            let bs = batch_size.suggest(trial)?;

            // Simulate training with intermediate reporting.
            let mut loss = 1.0;
            for epoch in 0..10 {
                loss *= 0.7 + 0.3 * lr_val.ln().abs() / 12.0;
                loss += drop * 0.05;
                loss += (1.0 / bs as f64) * 0.1;
                loss -= layers as f64 * 0.02;
                trial.report(epoch, loss);
                if trial.should_prune() {
                    return Err(TrialPruned.into());
                }
            }

            Ok::<_, Error>(loss)
        })
        .unwrap();

    println!("{}", study.summary());

    let path = "optimization_report.html";
    generate_html_report(&study, path).unwrap();
    println!("\nReport saved to {path}");
}
