//! Parameter types example — demonstrates all five parameter types and the derive feature.
//!
//! Shows `FloatParam`, `IntParam`, `CategoricalParam`, `BoolParam`, and `EnumParam`
//! with `.name()` labels, `#[derive(Categorical)]` for enums, and typed access
//! to results via `CompletedTrial::get()`.
//!
//! Run with: `cargo run --example parameter_types --features derive`

use optimizer::prelude::*;
use optimizer_derive::Categorical;

/// Activation functions — `#[derive(Categorical)]` auto-generates the
/// `Categorical` trait, mapping each variant to a sequential index.
#[derive(Clone, Debug, Categorical)]
enum Activation {
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
}

fn main() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    // --- Define one of each parameter type, each with a human-readable .name() ---

    // Float: learning rate on a log scale (common for ML hyperparameters)
    let lr = FloatParam::new(1e-5, 1e-1).log_scale().name("lr");

    // Int: number of hidden layers (stepped by 1, the default)
    let n_layers = IntParam::new(1, 5).name("n_layers");

    // Categorical: optimizer algorithm chosen from a list of strings
    let optimizer = CategoricalParam::new(vec!["sgd", "adam", "rmsprop"]).name("optimizer");

    // Bool: whether to apply dropout
    let use_dropout = BoolParam::new().name("use_dropout");

    // Enum: activation function — uses #[derive(Categorical)] above
    let activation = EnumParam::<Activation>::new().name("activation");

    // --- Run the optimization ---
    study
        .optimize(30, |trial: &mut optimizer::Trial| {
            let lr_val = lr.suggest(trial)?;
            let layers = n_layers.suggest(trial)?;
            let opt = optimizer.suggest(trial)?;
            let dropout = use_dropout.suggest(trial)?;
            let act = activation.suggest(trial)?;

            // Simulated loss that depends on all parameters
            let loss = lr_val * f64::from(layers as i32)
                + if opt == "adam" { -0.05 } else { 0.0 }
                + if dropout { -0.02 } else { 0.0 }
                + match act {
                    Activation::Gelu => -0.03,
                    Activation::Relu => -0.01,
                    _ => 0.0,
                };

            Ok::<_, Error>(loss)
        })
        .unwrap();

    // --- Retrieve best trial and read back each parameter with typed .get() ---
    let best = study.best_trial().unwrap();
    println!("Best trial #{} — loss = {:.6}", best.id, best.value);
    println!("  lr         = {:.6}", best.get(&lr).unwrap());
    println!("  n_layers   = {}", best.get(&n_layers).unwrap());
    println!("  optimizer  = {}", best.get(&optimizer).unwrap());
    println!("  use_dropout = {}", best.get(&use_dropout).unwrap());
    println!("  activation = {:?}", best.get(&activation).unwrap());
}
