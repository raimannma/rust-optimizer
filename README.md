# optimizer

A Rust library for black-box optimization using Tree-Parzen Estimator (TPE).

[![Docs](https://docs.rs/optimizer/badge.svg)](https://docs.rs/optimizer)
[![Crates.io](https://img.shields.io/crates/v/optimizer.svg)](https://crates.io/crates/optimizer)
[![codecov](https://codecov.io/gh/raimannma/rust-optimizer/graph/badge.svg?token=WOE77XJ4M6)](https://codecov.io/gh/raimannma/rust-optimizer)

## Features

- Optuna-like API for hyperparameter optimization
- Float, integer, and categorical parameter types
- Log-scale and stepped parameter sampling
- Sync and async optimization with parallel trial evaluation
- Serialization support for saving/loading study state

## Quick Start

```rust
use optimizer::{Direction, Study, TpeSampler};

let sampler = TpeSampler::builder().seed(42).build();
let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

study
    .optimize_with_sampler(20, |trial| {
        let x = trial.suggest_float("x", -10.0, 10.0)?;
        Ok::<_, optimizer::TpeError>(x * x)
    })
    .unwrap();

let best = study.best_trial().unwrap();
println!("Best value: {} at x={:?}", best.value, best.params);
```

## Feature Flags

- `serde` - Enable serialization/deserialization of studies and trials
- `async` - Enable async optimization methods (requires tokio)

## Documentation

Full API documentation is available at [docs.rs/optimizer](https://docs.rs/optimizer).

## License

MIT
