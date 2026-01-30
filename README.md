# optimizer

A Rust library for black-box optimization with multiple sampling strategies.

[![Docs](https://docs.rs/optimizer/badge.svg)](https://docs.rs/optimizer)
[![Crates.io](https://img.shields.io/crates/v/optimizer.svg)](https://crates.io/crates/optimizer)
[![codecov](https://codecov.io/gh/raimannma/rust-optimizer/graph/badge.svg?token=WOE77XJ4M6)](https://codecov.io/gh/raimannma/rust-optimizer)

## Features

- Optuna-like API for hyperparameter optimization
- Multiple sampling strategies:
  - **Random Search** - Simple random sampling for baseline comparisons
  - **TPE (Tree-Parzen Estimator)** - Bayesian optimization for efficient search
  - **Grid Search** - Exhaustive search over a specified parameter grid
- Float, integer, and categorical parameter types
- Log-scale and stepped parameter sampling
- Sync and async optimization with parallel trial evaluation

## Quick Start

```rust
use optimizer::{Direction, Study};
use optimizer::sampler::tpe::TpeSampler;

let sampler = TpeSampler::builder().seed(42).build().unwrap();
let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

study
    .optimize_with_sampler(20, |trial| {
        let x = trial.suggest_float("x", -10.0, 10.0)?;
        Ok::<_, optimizer::Error>(x * x)
    })
    .unwrap();

let best = study.best_trial().unwrap();
println!("Best value: {} at x={:?}", best.value, best.params);
```

## Samplers

### Random Search

```rust
use optimizer::{Direction, Study};
use optimizer::sampler::random::RandomSampler;

let study: Study<f64> = Study::with_sampler(
    Direction::Minimize,
    RandomSampler::with_seed(42),
);
```

### TPE (Tree-Parzen Estimator)

```rust
use optimizer::{Direction, Study};
use optimizer::sampler::tpe::TpeSampler;

let sampler = TpeSampler::builder()
    .gamma(0.15)           // Quantile for good/bad split
    .n_startup_trials(20)  // Random trials before TPE kicks in
    .n_ei_candidates(32)   // Candidates to evaluate
    .seed(42)
    .build()
    .unwrap();

let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
```

### Grid Search

```rust
use optimizer::{Direction, Study};
use optimizer::sampler::grid::GridSearchSampler;

let sampler = GridSearchSampler::builder()
    .n_points_per_param(10)  // Number of points per parameter dimension
    .build();

let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
```

## Feature Flags

- `async` - Enable async optimization methods (requires tokio)

## Documentation

Full API documentation is available at [docs.rs/optimizer](https://docs.rs/optimizer).

## License

MIT
