//! Async integration tests for the optimizer library.
//!
//! These tests are only compiled when the `async` feature is enabled.

#![cfg(feature = "async")]

use optimizer::parameter::{FloatParam, Parameter};
use optimizer::sampler::random::RandomSampler;
use optimizer::sampler::tpe::TpeSampler;
use optimizer::{Direction, Error, Study};

#[tokio::test]
async fn test_optimize_async_basic() {
    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(-10.0, 10.0);

    study
        .optimize_async(10, move |mut trial| {
            let x_param = x_param.clone();
            async move {
                let x = x_param.suggest(&mut trial)?;
                Ok::<_, Error>((trial, x * x))
            }
        })
        .await
        .expect("async optimization should succeed");

    assert_eq!(study.n_trials(), 10);
    let best = study.best_trial().expect("should have best trial");
    assert!(best.value >= 0.0, "x^2 should be non-negative");
}

#[tokio::test]
async fn test_optimize_async_with_sampler() {
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(5)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(-5.0, 5.0);

    study
        .optimize_async_with_sampler(15, move |mut trial| {
            let x_param = x_param.clone();
            async move {
                let x = x_param.suggest(&mut trial)?;
                Ok::<_, Error>((trial, x * x))
            }
        })
        .await
        .expect("async optimization with sampler should succeed");

    assert_eq!(study.n_trials(), 15);
    let best = study.best_trial().expect("should have best trial");
    assert!(best.value < 10.0, "should find reasonable solution");
}

#[tokio::test]
async fn test_optimize_parallel() {
    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(-10.0, 10.0);

    study
        .optimize_parallel(20, 4, move |mut trial| {
            let x_param = x_param.clone();
            async move {
                let x = x_param.suggest(&mut trial)?;
                Ok::<_, Error>((trial, x * x))
            }
        })
        .await
        .expect("parallel optimization should succeed");

    assert_eq!(study.n_trials(), 20);
}

#[tokio::test]
async fn test_optimize_parallel_with_sampler() {
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(5)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(-5.0, 5.0);
    let y_param = FloatParam::new(-5.0, 5.0);

    study
        .optimize_parallel_with_sampler(15, 3, move |mut trial| {
            let x_param = x_param.clone();
            let y_param = y_param.clone();
            async move {
                let x = x_param.suggest(&mut trial)?;
                let y = y_param.suggest(&mut trial)?;
                Ok::<_, Error>((trial, x * x + y * y))
            }
        })
        .await
        .expect("parallel optimization with sampler should succeed");

    assert_eq!(study.n_trials(), 15);
}

#[tokio::test]
async fn test_optimize_async_all_failures() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    let result = study
        .optimize_async(5, |trial| async move {
            let _ = trial;
            Err::<(_, f64), &str>("always fails")
        })
        .await;

    assert!(
        matches!(result, Err(Error::NoCompletedTrials)),
        "should return NoCompletedTrials when all trials fail"
    );
}

#[tokio::test]
async fn test_optimize_async_with_sampler_all_failures() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    let result = study
        .optimize_async_with_sampler(5, |trial| async move {
            let _ = trial;
            Err::<(_, f64), &str>("always fails")
        })
        .await;

    assert!(
        matches!(result, Err(Error::NoCompletedTrials)),
        "should return NoCompletedTrials when all trials fail"
    );
}

#[tokio::test]
async fn test_optimize_parallel_all_failures() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    let result = study
        .optimize_parallel(5, 2, |trial| async move {
            let _ = trial;
            Err::<(_, f64), &str>("always fails")
        })
        .await;

    assert!(
        matches!(result, Err(Error::NoCompletedTrials)),
        "should return NoCompletedTrials when all trials fail"
    );
}

#[tokio::test]
async fn test_optimize_parallel_with_sampler_all_failures() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    let result = study
        .optimize_parallel_with_sampler(5, 2, |trial| async move {
            let _ = trial;
            Err::<(_, f64), &str>("always fails")
        })
        .await;

    assert!(
        matches!(result, Err(Error::NoCompletedTrials)),
        "should return NoCompletedTrials when all trials fail"
    );
}

#[tokio::test]
async fn test_optimize_async_partial_failures() {
    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let counter = std::sync::atomic::AtomicUsize::new(0);

    let x_param = FloatParam::new(0.0, 10.0);

    study
        .optimize_async(10, move |mut trial| {
            let count = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let x_param = x_param.clone();
            async move {
                if count.is_multiple_of(2) {
                    let x = x_param.suggest(&mut trial)?;
                    Ok::<_, Error>((trial, x))
                } else {
                    Err(Error::NoCompletedTrials) // Use as error type
                }
            }
        })
        .await
        .expect("should succeed with partial failures");

    // Only half should have succeeded
    assert_eq!(study.n_trials(), 5);
}

#[tokio::test]
async fn test_optimize_parallel_high_concurrency() {
    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(0.0, 10.0);

    // Run with concurrency higher than n_trials
    study
        .optimize_parallel(5, 10, move |mut trial| {
            let x_param = x_param.clone();
            async move {
                let x = x_param.suggest(&mut trial)?;
                Ok::<_, Error>((trial, x))
            }
        })
        .await
        .expect("should handle high concurrency");

    assert_eq!(study.n_trials(), 5);
}

#[tokio::test]
async fn test_optimize_parallel_single_concurrency() {
    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(0.0, 10.0);

    // Run with concurrency of 1 (sequential)
    study
        .optimize_parallel(10, 1, move |mut trial| {
            let x_param = x_param.clone();
            async move {
                let x = x_param.suggest(&mut trial)?;
                Ok::<_, Error>((trial, x))
            }
        })
        .await
        .expect("should work with single concurrency");

    assert_eq!(study.n_trials(), 10);
}
