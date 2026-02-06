use optimizer::parameter::{
    BoolParam, CategoricalParam, EnumParam, FloatParam, IntParam, Parameter,
};
use optimizer::{Direction, Study};
use optimizer_derive::Categorical;

#[derive(Clone, Debug, Categorical)]
enum Activation {
    Relu,
    Sigmoid,
    Tanh,
}

fn main() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    // Define parameters outside the objective function
    let lr_param = FloatParam::new(1e-5, 1e-1).log_scale();
    let n_layers_param = IntParam::new(1, 5);
    let units_param = IntParam::new(32, 512).step(32);
    let optimizer_param = CategoricalParam::new(vec!["sgd", "adam", "rmsprop"]);
    let activation_param = EnumParam::<Activation>::new();
    let batch_size_param = IntParam::new(16, 256).log_scale();
    let use_dropout_param = BoolParam::new();

    study
        .optimize(20, |trial| {
            let lr = lr_param.suggest(trial)?;
            let n_layers = n_layers_param.suggest(trial)?;
            let units = units_param.suggest(trial)?;
            let optimizer = optimizer_param.suggest(trial)?;
            let use_dropout = use_dropout_param.suggest(trial)?;
            let activation = activation_param.suggest(trial)?;
            let batch_size = batch_size_param.suggest(trial)?;

            // Simulate a loss function
            let loss = lr * (n_layers as f64) + (units as f64) * 0.001
                - if use_dropout { 0.1 } else { 0.0 };

            println!(
                "Trial {}: lr={lr:.6}, layers={n_layers}, units={units}, opt={optimizer}, \
                 dropout={use_dropout}, activation={activation:?}, batch={batch_size} -> loss={loss:.4}",
                trial.id()
            );

            Ok::<_, optimizer::Error>(loss)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    println!("\nBest trial: value={:.4}", best.value);
    for (id, label) in &best.param_labels {
        println!("  {}: {:?}", label, best.params[id]);
    }
}
