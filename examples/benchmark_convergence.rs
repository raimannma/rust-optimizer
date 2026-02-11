use std::ops::ControlFlow;
use std::time::Instant;

use optimizer::parameter::Parameter;
use optimizer::sampler::random::RandomSampler;
use optimizer::sampler::tpe::TpeSampler;
use optimizer::{FloatParam, Study};

/// Standard optimization test functions.
mod functions {
    pub fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    pub fn rosenbrock(x: &[f64]) -> f64 {
        x.windows(2)
            .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
            .sum()
    }

    pub fn rastrigin(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        10.0 * n
            + x.iter()
                .map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    }
}

fn run_convergence(
    name: &str,
    sampler_name: &str,
    study: Study<f64>,
    params: &[FloatParam],
    objective: fn(&[f64]) -> f64,
    n_trials: usize,
) {
    let start = Instant::now();

    study
        .optimize_with_callback(
            n_trials,
            |trial| {
                let x: Vec<f64> = params
                    .iter()
                    .map(|p| p.suggest(trial))
                    .collect::<Result<_, _>>()
                    .unwrap();
                Ok::<_, optimizer::Error>(objective(&x))
            },
            |study, _trial| {
                let elapsed = start.elapsed().as_millis();
                let best = study.best_value().unwrap();
                let n = study.n_trials();
                println!("{n},{best},{elapsed},{sampler_name},{name}");
                ControlFlow::Continue(())
            },
        )
        .unwrap();
}

fn main() {
    println!("trial,best_value,elapsed_ms,sampler,function");

    let dims = 5;
    let params: Vec<FloatParam> = (0..dims)
        .map(|i| FloatParam::new(-5.0, 5.0).name(format!("x{i}")))
        .collect();
    let n_trials = 200;

    // Sphere: Random vs TPE
    run_convergence(
        "sphere_5d",
        "random",
        Study::minimize(RandomSampler::with_seed(1)),
        &params,
        functions::sphere,
        n_trials,
    );
    run_convergence(
        "sphere_5d",
        "tpe",
        Study::minimize(TpeSampler::builder().seed(1).build().unwrap()),
        &params,
        functions::sphere,
        n_trials,
    );

    // Rosenbrock: Random vs TPE
    run_convergence(
        "rosenbrock_5d",
        "random",
        Study::minimize(RandomSampler::with_seed(2)),
        &params,
        functions::rosenbrock,
        n_trials,
    );
    run_convergence(
        "rosenbrock_5d",
        "tpe",
        Study::minimize(TpeSampler::builder().seed(2).build().unwrap()),
        &params,
        functions::rosenbrock,
        n_trials,
    );

    // Rastrigin: Random vs TPE
    run_convergence(
        "rastrigin_5d",
        "random",
        Study::minimize(RandomSampler::with_seed(3)),
        &params,
        functions::rastrigin,
        n_trials,
    );
    run_convergence(
        "rastrigin_5d",
        "tpe",
        Study::minimize(TpeSampler::builder().seed(3).build().unwrap()),
        &params,
        functions::rastrigin,
        n_trials,
    );
}
