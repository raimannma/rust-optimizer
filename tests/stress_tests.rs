//! Stress and large-scale tests for the optimizer library.
//!
//! All tests are `#[ignore]`-gated so they don't run in normal CI.
//! Run with: `cargo test --features async -- --ignored`

use std::collections::HashSet;

use optimizer::parameter::{FloatParam, Parameter};
use optimizer::sampler::random::RandomSampler;
use optimizer::sampler::tpe::TpeSampler;
use optimizer::{Direction, Error, Study};

fn make_float_params(n: usize) -> Vec<FloatParam> {
    (0..n)
        .map(|i| FloatParam::new(-5.0, 5.0).name(format!("x{i}")))
        .collect()
}

fn sphere(trial: &mut optimizer::Trial, params: &[FloatParam]) -> Result<f64, Error> {
    let mut sum = 0.0;
    for p in params {
        let v = p.suggest(trial)?;
        sum += v * v;
    }
    Ok(sum)
}

#[test]
#[ignore]
fn stress_many_trials_random() {
    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    let params = make_float_params(5);

    study
        .optimize(10_000, |trial: &mut optimizer::Trial| {
            sphere(trial, &params)
        })
        .expect("10k trials should complete");

    assert_eq!(study.n_trials(), 10_000);
    let best = study.best_value().expect("should have a best value");
    assert!(best.is_finite(), "best value should be finite");
    assert!(best >= 0.0, "sphere function is non-negative");
}

#[test]
#[ignore]
fn stress_many_params_tpe() {
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(10)
        .build()
        .unwrap();
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    let params = make_float_params(128);

    study
        .optimize(200, |trial: &mut optimizer::Trial| sphere(trial, &params))
        .expect("200 trials with 128 params should complete");

    assert_eq!(study.n_trials(), 200);

    let best = study.best_trial().expect("should have a best trial");
    assert_eq!(best.params.len(), 128, "best trial should have 128 params");
    assert!(best.value.is_finite(), "best value should be finite");
    for v in best.params.values() {
        let f = match v {
            optimizer::param::ParamValue::Float(f) => *f,
            other => panic!("expected Float param, got {other}"),
        };
        assert!(f.is_finite(), "all param values should be finite");
    }
}

#[cfg(feature = "async")]
#[tokio::test]
#[ignore]
async fn stress_high_concurrency_parallel() {
    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    let params = make_float_params(10);

    study
        .optimize_parallel(1_000, 128, move |trial: &mut optimizer::Trial| {
            sphere(trial, &params)
        })
        .await
        .expect("1k trials with 128 workers should complete");

    assert_eq!(study.n_trials(), 1_000);

    let trials = study.trials();
    let ids: HashSet<u64> = trials.iter().map(|t| t.id).collect();
    assert_eq!(ids.len(), 1_000, "all trial IDs should be unique");

    let best = study.best_value().expect("should have a best value");
    assert!(best.is_finite(), "best value should be finite");
}

#[cfg(feature = "async")]
#[tokio::test]
#[ignore]
async fn stress_long_running_tpe_parallel() {
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(20)
        .build()
        .unwrap();
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    let params = make_float_params(20);

    study
        .optimize_parallel(5_000, 32, move |trial: &mut optimizer::Trial| {
            sphere(trial, &params)
        })
        .await
        .expect("5k trials with TPE and 32 workers should complete");

    assert_eq!(study.n_trials(), 5_000);

    let trials = study.trials();
    let ids: HashSet<u64> = trials.iter().map(|t| t.id).collect();
    assert_eq!(ids.len(), 5_000, "all trial IDs should be unique");

    let best = study.best_trial().expect("should have a best trial");
    assert!(best.value.is_finite(), "best value should be finite");
    assert_eq!(best.params.len(), 20, "best trial should have 20 params");
}
