//! Analyse a study with polars.
//!
//! Run with: `cargo run --example dataframe --features polars`

use optimizer::polars::prelude::*;
use optimizer::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Activation {
    Relu,
    Tanh,
}

impl Categorical for Activation {
    const N_CHOICES: usize = 2;
    fn from_index(index: usize) -> Self {
        [Self::Relu, Self::Tanh][index]
    }
    fn to_index(&self) -> usize {
        *self as usize
    }
}

// polars errors are `PolarsError`, not `optimizer::Error`, so return a boxed
// error. `std::result::Result` is spelled out because the optimizer prelude's
// `Result<T>` alias takes one argument.
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let lr = FloatParam::new(1e-4, 1e-1).log_scale().name("lr");
    let layers = IntParam::new(1, 4).name("n_layers");
    let optimizer = CategoricalParam::new(vec!["sgd", "adam", "rmsprop"]).name("optimizer");
    let batch_norm = BoolParam::new().name("batch_norm");
    let activation = EnumParam::<Activation>::new().name("activation");

    let study: Study<f64> = Study::new(Direction::Minimize);
    study.optimize(30, |trial: &mut Trial| {
        let lr_v = lr.suggest(trial)?;
        let n = layers.suggest(trial)?;
        let opt = optimizer.suggest(trial)?;
        let bn = batch_norm.suggest(trial)?;
        let act = activation.suggest(trial)?;
        trial.set_user_attr("epochs", 10_i64);
        #[allow(clippy::cast_precision_loss)]
        let loss = (lr_v.log10() + 2.0).powi(2)
            + n as f64 * 0.1
            + if opt == "adam" { 0.0 } else { 0.5 }
            + if bn { 0.0 } else { 0.2 }
            + if act == Activation::Relu { 0.0 } else { 0.1 };
        Ok::<_, Error>(loss)
    })?;

    // 1. Everything, raw: categorical params hold their choice index.
    let raw = study.to_dataframe()?;
    println!("{raw}");

    // 2. Decode categorical / bool / enum params into their actual values.
    let df = study.to_dataframe_with(&[&optimizer, &batch_norm, &activation])?;
    println!("{df}");

    // 3. Best 5 trials, decoded.
    let top = study
        .top_trials(5)
        .to_dataframe_with(&[&optimizer, &batch_norm, &activation])?;
    println!("{top}");

    // 4. Plain polars from here on: count the adam trials.
    let adam = df.column("optimizer")?.str()?.equal("adam");
    println!("adam trials: {}", df.filter(&adam)?.height());

    Ok(())
}
