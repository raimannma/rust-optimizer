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
- Float, integer, categorical, boolean, and enum parameter types
- Log-scale and stepped parameter sampling
- Sync and async optimization with parallel trial evaluation
- `#[derive(Categorical)]` for enum parameters

## Quick Start

```rust
use optimizer::parameter::{FloatParam, Parameter};
use optimizer::sampler::tpe::TpeSampler;
use optimizer::{Direction, Study};

// Create a study with TPE sampler
let sampler = TpeSampler::builder().seed(42).build().unwrap();
let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

// Define parameter search space
let x_param = FloatParam::new(-10.0, 10.0);

// Optimize x^2 for 20 trials
study
    .optimize_with_sampler(20, |trial| {
        let x = x_param.suggest(trial)?;
        Ok::<_, optimizer::Error>(x * x)
    })
    .unwrap();

// Get the best result
let best = study.best_trial().unwrap();
println!("Best value: {}", best.value);
for (id, label) in &best.param_labels {
    println!("  {}: {:?}", label, best.params[id]);
}
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

#### Gamma Strategies

The gamma parameter controls what fraction of trials are considered "good" when building the TPE model. Instead of a fixed value, you can use adaptive strategies:

| Strategy | Description | Formula |
|----------|-------------|---------|
| `FixedGamma` | Constant value (default: 0.25) | `γ = constant` |
| `LinearGamma` | Linear interpolation over trials | `γ = γ_min + (γ_max - γ_min) * min(n/n_max, 1)` |
| `SqrtGamma` | Optuna-style inverse sqrt scaling | `γ = min(γ_max, factor/√n / n)` |
| `HyperoptGamma` | Hyperopt-style adaptive | `γ = min(γ_max, (base + 1) / n)` |

```rust
use optimizer::sampler::tpe::{TpeSampler, SqrtGamma, LinearGamma};

// Optuna-style gamma that decreases with more trials
let sampler = TpeSampler::builder()
    .gamma_strategy(SqrtGamma::default())
    .build()
    .unwrap();

// Linear interpolation from 0.1 to 0.3 over 100 trials
let sampler = TpeSampler::builder()
    .gamma_strategy(LinearGamma::new(0.1, 0.3, 100).unwrap())
    .build()
    .unwrap();
```

You can also implement custom strategies:

```rust
use optimizer::sampler::tpe::{TpeSampler, GammaStrategy};

#[derive(Debug, Clone)]
struct MyGamma { base: f64 }

impl GammaStrategy for MyGamma {
    fn gamma(&self, n_trials: usize) -> f64 {
        (self.base + 0.01 * n_trials as f64).min(0.5)
    }
    fn clone_box(&self) -> Box<dyn GammaStrategy> {
        Box::new(self.clone())
    }
}

let sampler = TpeSampler::builder()
    .gamma_strategy(MyGamma { base: 0.1 })
    .build()
    .unwrap();
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
- `derive` - Enable `#[derive(Categorical)]` for enum parameters

## Documentation

Full API documentation is available at [docs.rs/optimizer](https://docs.rs/optimizer).

## License

MIT
