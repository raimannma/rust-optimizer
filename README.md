# optimizer

Bayesian and population-based optimization library with an Optuna-like API
for hyperparameter tuning and black-box optimization. Supports 12 samplers,
8 pruners, multi-objective optimization, async parallelism, and persistent storage.

[![Docs](https://docs.rs/optimizer/badge.svg)](https://docs.rs/optimizer)
[![Crates.io](https://img.shields.io/crates/v/optimizer.svg)](https://crates.io/crates/optimizer)
[![codecov](https://codecov.io/gh/raimannma/rust-optimizer/graph/badge.svg?token=WOE77XJ4M6)](https://codecov.io/gh/raimannma/rust-optimizer)

## Quick Start

```rust
use optimizer::prelude::*;

let study: Study<f64> = Study::new(Direction::Minimize);
let x = FloatParam::new(-10.0, 10.0).name("x");

study.optimize(50, |trial| {
    let val = x.suggest(trial)?;
    Ok::<_, Error>((val - 3.0).powi(2))
}).unwrap();

let best = study.best_trial().unwrap();
println!("Best x = {:.4}, f(x) = {:.4}", best.get(&x).unwrap(), best.value);
```

## Features at a Glance

- **[Samplers](https://docs.rs/optimizer/latest/optimizer/sampler/)** — Random, TPE, Multivariate TPE, Grid, Sobol, CMA-ES, Gaussian Process, Differential Evolution, BOHB, NSGA-II, NSGA-III, MOEA/D
- **[Pruners](https://docs.rs/optimizer/latest/optimizer/pruner/)** — Median, Percentile, Threshold, Patient, Hyperband, Successive Halving, Wilcoxon, Nop
- **[Parameters](https://docs.rs/optimizer/latest/optimizer/parameter/)** — Float, Int, Categorical, Bool, and Enum types with `.name()` labels and typed access
- **[Multi-objective](https://docs.rs/optimizer/latest/optimizer/multi_objective/)** — Pareto front extraction with NSGA-II/III and MOEA/D
- **[Async & parallel](https://docs.rs/optimizer/latest/optimizer/struct.Study.html#method.optimize_parallel)** — Concurrent trial evaluation with Tokio
- **[Storage backends](https://docs.rs/optimizer/latest/optimizer/storage/)** — In-memory (default) or JSONL journal for persistence and resumption
- **[Visualization](https://docs.rs/optimizer/latest/optimizer/fn.generate_html_report.html)** — HTML reports with optimization history and parameter importance
- **[Analysis](https://docs.rs/optimizer/latest/optimizer/struct.Study.html#method.fanova)** — fANOVA and Spearman correlation for parameter importance

## Feature Flags

| Flag | Enables | Default |
|------|---------|---------|
| `async` | Async/parallel optimization (Tokio) | No |
| `derive` | `#[derive(Categorical)]` for enum parameters | No |
| `serde` | Serialization of trials and parameters | No |
| `journal` | JSONL storage backend (implies `serde`) | No |
| `sobol` | Sobol quasi-random sampler | No |
| `cma-es` | CMA-ES sampler (requires `nalgebra`) | No |
| `gp` | Gaussian Process sampler (requires `nalgebra`) | No |
| `tracing` | Structured logging with `tracing` | No |

## Examples

```sh
cargo run --example basic_optimization                       # Minimize a quadratic — simplest possible usage
cargo run --example parameter_types --features derive        # All 5 param types + #[derive(Categorical)]
cargo run --example sampler_comparison                       # Compare Random, TPE, and Grid on the same problem
cargo run --example pruning                                  # Trial pruning with MedianPruner
cargo run --example early_stopping                           # Halt a study when a target is reached
cargo run --example async_parallel --features async          # Evaluate trials concurrently with tokio
cargo run --example journal_storage --features journal       # Persist trials to disk and resume later
cargo run --example ask_and_tell                             # Decouple sampling from evaluation
cargo run --example multi_objective                          # Optimize competing objectives + Pareto front
```

## Learn More

- [API documentation](https://docs.rs/optimizer)
- [Changelog](CHANGELOG.md)
- [GitHub Issues](https://github.com/raimannma/rust-optimizer/issues)

## License

MIT
