use std::collections::HashMap;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use optimizer::parameter::{FloatParam, Parameter};
use optimizer::sampler::grid::GridSearchSampler;
use optimizer::sampler::random::RandomSampler;
use optimizer::sampler::tpe::TpeSampler;
use optimizer::sampler::{CompletedTrial, Sampler};

/// Build a synthetic history of `n` completed trials over `dims` float parameters.
fn build_history(n: usize, dims: usize) -> Vec<CompletedTrial<f64>> {
    let params: Vec<FloatParam> = (0..dims)
        .map(|i| FloatParam::new(-5.0, 5.0).name(format!("x{i}")))
        .collect();

    let mut history = Vec::with_capacity(n);
    let sampler = RandomSampler::with_seed(42);

    for trial_id in 0..n {
        let id = trial_id as u64;
        let mut param_values = HashMap::new();
        let mut distributions = HashMap::new();
        let mut param_labels = HashMap::new();
        for p in &params {
            let dist = p.distribution();
            let val = sampler.sample(&dist, id, &history);
            param_values.insert(p.id(), val);
            distributions.insert(p.id(), dist);
            param_labels.insert(p.id(), p.label());
        }
        // Use sphere function value as objective
        let value: f64 = param_values
            .values()
            .map(|v| {
                let optimizer::parameter::ParamValue::Float(f) = v else {
                    unreachable!()
                };
                f * f
            })
            .sum();
        history.push(CompletedTrial::new(
            id,
            param_values,
            distributions,
            param_labels,
            value,
        ));
    }
    history
}

fn bench_tpe_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("tpe_sample");
    let dist = FloatParam::new(-5.0, 5.0).distribution();
    let tpe = TpeSampler::builder().seed(42).build().unwrap();

    for history_size in [10, 100, 1000] {
        let history = build_history(history_size, 2);
        group.bench_with_input(
            BenchmarkId::new("history", history_size),
            &history,
            |b, history| {
                b.iter(|| tpe.sample(&dist, history.len() as u64, history));
            },
        );
    }
    group.finish();
}

fn bench_random_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_sample");
    let dist = FloatParam::new(-5.0, 5.0).distribution();
    let sampler = RandomSampler::with_seed(42);

    for history_size in [10, 100, 1000] {
        let history = build_history(history_size, 2);
        group.bench_with_input(
            BenchmarkId::new("history", history_size),
            &history,
            |b, history| {
                b.iter(|| sampler.sample(&dist, history.len() as u64, history));
            },
        );
    }
    group.finish();
}

fn bench_grid_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_sample");
    let dist = FloatParam::new(-5.0, 5.0).distribution();
    let history: Vec<CompletedTrial<f64>> = Vec::new();

    for grid_points in [5, 10, 50] {
        group.bench_with_input(
            BenchmarkId::new("points", grid_points),
            &grid_points,
            |b, _| {
                b.iter(|| {
                    // Fresh sampler each iteration since grid tracks used points
                    let sampler = GridSearchSampler::builder()
                        .n_points_per_param(grid_points)
                        .build();
                    sampler.sample(&dist, 0, &history)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_tpe_sample,
    bench_random_sample,
    bench_grid_sample
);
criterion_main!(benches);
