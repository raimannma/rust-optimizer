#[allow(dead_code)]
mod test_functions;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use optimizer::Study;
use optimizer::parameter::{FloatParam, Parameter};
use optimizer::sampler::random::RandomSampler;
use optimizer::sampler::tpe::TpeSampler;

fn make_params(dims: usize) -> Vec<FloatParam> {
    (0..dims)
        .map(|i| FloatParam::new(-5.0, 5.0).name(format!("x{i}")))
        .collect()
}

fn bench_tpe_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("tpe_sphere");
    group.sample_size(10);

    for dims in [2, 10, 50] {
        let params = make_params(dims);
        group.bench_with_input(BenchmarkId::new("dims", dims), &params, |b, params| {
            b.iter(|| {
                let study = Study::minimize(TpeSampler::builder().seed(42).build().unwrap());
                study
                    .optimize(100, |trial| {
                        let x: Vec<f64> = params
                            .iter()
                            .map(|p| p.suggest(trial))
                            .collect::<Result<_, _>>()
                            .unwrap();
                        Ok::<_, optimizer::Error>(test_functions::sphere(&x))
                    })
                    .unwrap();
            });
        });
    }
    group.finish();
}

fn bench_tpe_rosenbrock(c: &mut Criterion) {
    let mut group = c.benchmark_group("tpe_rosenbrock");
    group.sample_size(10);

    for dims in [2, 10] {
        let params = make_params(dims);
        group.bench_with_input(BenchmarkId::new("dims", dims), &params, |b, params| {
            b.iter(|| {
                let study = Study::minimize(TpeSampler::builder().seed(42).build().unwrap());
                study
                    .optimize(100, |trial| {
                        let x: Vec<f64> = params
                            .iter()
                            .map(|p| p.suggest(trial))
                            .collect::<Result<_, _>>()
                            .unwrap();
                        Ok::<_, optimizer::Error>(test_functions::rosenbrock(&x))
                    })
                    .unwrap();
            });
        });
    }
    group.finish();
}

fn bench_random_vs_tpe(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_vs_tpe");
    group.sample_size(10);
    let params = make_params(5);

    group.bench_function("random_5d", |b| {
        b.iter(|| {
            let study = Study::minimize(RandomSampler::with_seed(42));
            study
                .optimize(100, |trial| {
                    let x: Vec<f64> = params
                        .iter()
                        .map(|p| p.suggest(trial))
                        .collect::<Result<_, _>>()
                        .unwrap();
                    Ok::<_, optimizer::Error>(test_functions::sphere(&x))
                })
                .unwrap();
        });
    });

    group.bench_function("tpe_5d", |b| {
        b.iter(|| {
            let study = Study::minimize(TpeSampler::builder().seed(42).build().unwrap());
            study
                .optimize(100, |trial| {
                    let x: Vec<f64> = params
                        .iter()
                        .map(|p| p.suggest(trial))
                        .collect::<Result<_, _>>()
                        .unwrap();
                    Ok::<_, optimizer::Error>(test_functions::sphere(&x))
                })
                .unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tpe_sphere,
    bench_tpe_rosenbrock,
    bench_random_vs_tpe
);
criterion_main!(benches);
