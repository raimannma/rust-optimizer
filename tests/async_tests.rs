//! Async integration tests for the optimizer library.
//!
//! These tests are only compiled when the `async` feature is enabled.

#![cfg(feature = "async")]

use std::sync::Arc;

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
        .optimize_async(10, move |trial: &mut optimizer::Trial| {
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>(x * x)
        })
        .await
        .expect("async optimization should succeed");

    assert_eq!(study.n_trials(), 10);
    let best = study.best_trial().expect("should have best trial");
    assert!(best.value >= 0.0, "x^2 should be non-negative");
}

#[tokio::test]
async fn test_optimize_async_with_tpe() {
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(5)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(-5.0, 5.0);

    study
        .optimize_async(15, move |trial: &mut optimizer::Trial| {
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>(x * x)
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
        .optimize_parallel(20, 4, move |trial: &mut optimizer::Trial| {
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>(x * x)
        })
        .await
        .expect("parallel optimization should succeed");

    assert_eq!(study.n_trials(), 20);
}

#[tokio::test]
async fn test_optimize_parallel_with_tpe() {
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(5)
        .build()
        .unwrap();

    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(-5.0, 5.0);
    let y_param = FloatParam::new(-5.0, 5.0);

    study
        .optimize_parallel(15, 3, move |trial: &mut optimizer::Trial| {
            let x = x_param.suggest(trial)?;
            let y = y_param.suggest(trial)?;
            Ok::<_, Error>(x * x + y * y)
        })
        .await
        .expect("parallel optimization with sampler should succeed");

    assert_eq!(study.n_trials(), 15);
}

#[tokio::test]
async fn test_optimize_async_all_failures() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    let result = study
        .optimize_async(5, |_trial: &mut optimizer::Trial| {
            Err::<f64, &str>("always fails")
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
        .optimize_parallel(5, 2, |_trial: &mut optimizer::Trial| {
            Err::<f64, &str>("always fails")
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
        .optimize_async(10, move |trial: &mut optimizer::Trial| {
            let count = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if count.is_multiple_of(2) {
                let x = x_param.suggest(trial)?;
                Ok::<_, Error>(x)
            } else {
                Err(Error::NoCompletedTrials) // Use as error type
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
        .optimize_parallel(5, 10, move |trial: &mut optimizer::Trial| {
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>(x)
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
        .optimize_parallel(10, 1, move |trial: &mut optimizer::Trial| {
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>(x)
        })
        .await
        .expect("should work with single concurrency");

    assert_eq!(study.n_trials(), 10);
}

#[tokio::test]
async fn test_parallel_executes_concurrently() {
    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(0.0, 10.0);

    let start = tokio::time::Instant::now();
    study
        .optimize_parallel(4, 4, move |trial: &mut optimizer::Trial| {
            let x = x_param.suggest(trial)?;
            std::thread::sleep(std::time::Duration::from_millis(100));
            Ok::<_, Error>(x)
        })
        .await
        .expect("parallel optimization should succeed");

    let elapsed = start.elapsed();
    assert_eq!(study.n_trials(), 4);
    // Sequential would take ~400ms; parallel with concurrency=4 should be ~100ms
    assert!(
        elapsed < std::time::Duration::from_millis(350),
        "expected parallel execution under 350ms, took {elapsed:?}"
    );
}

#[tokio::test]
async fn test_parallel_max_concurrency_reached() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(0.0, 10.0);
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));

    let active_c = Arc::clone(&active);
    let max_active_c = Arc::clone(&max_active);

    study
        .optimize_parallel(8, 4, move |trial: &mut optimizer::Trial| {
            let x = x_param.suggest(trial)?;

            let current = active_c.fetch_add(1, Ordering::SeqCst) + 1;
            // Update max_active if this is the highest seen so far
            max_active_c.fetch_max(current, Ordering::SeqCst);

            std::thread::sleep(std::time::Duration::from_millis(50));

            active_c.fetch_sub(1, Ordering::SeqCst);
            Ok::<_, Error>(x)
        })
        .await
        .expect("parallel optimization should succeed");

    assert_eq!(study.n_trials(), 8);
    let max = max_active.load(Ordering::SeqCst);
    assert!(
        max >= 2,
        "expected at least 2 concurrent workers, but max was {max}"
    );
}

#[tokio::test]
async fn test_parallel_panic_returns_task_error() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    let result = study
        .optimize_parallel(3, 2, |_trial: &mut optimizer::Trial| {
            panic!("boom");
            #[allow(unreachable_code)]
            Ok::<_, Error>(0.0)
        })
        .await;

    match result {
        Err(Error::TaskError(msg)) => {
            assert!(
                msg.contains("panic"),
                "expected error message to contain 'panic', got: {msg}"
            );
        }
        other => panic!("expected TaskError, got {other:?}"),
    }
}

#[tokio::test]
async fn test_parallel_partial_failures_trial_count() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    let x_param = FloatParam::new(0.0, 10.0);
    let counter = Arc::new(AtomicUsize::new(0));

    let counter_c = Arc::clone(&counter);

    study
        .optimize_parallel(10, 3, move |trial: &mut optimizer::Trial| {
            let idx = counter_c.fetch_add(1, Ordering::SeqCst);
            if idx % 2 == 1 {
                return Err(Error::NoCompletedTrials);
            }
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>(x)
        })
        .await
        .expect("should succeed with partial failures");

    // Even indices succeed (0, 2, 4, 6, 8), odd indices fail
    assert_eq!(study.n_trials(), 5);
}
