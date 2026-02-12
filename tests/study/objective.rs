use optimizer::parameter::{FloatParam, Parameter};
use optimizer::sampler::random::RandomSampler;
use optimizer::{Direction, Error, Study, Trial};

#[test]
fn test_callback_early_stopping() {
    use std::ops::ControlFlow;

    use optimizer::Objective;
    use optimizer::sampler::CompletedTrial;

    struct EarlyStopAfter5 {
        x_param: FloatParam,
    }

    impl Objective<f64> for EarlyStopAfter5 {
        type Error = Error;
        fn evaluate(&self, trial: &mut Trial) -> Result<f64, Error> {
            let x = self.x_param.suggest(trial)?;
            Ok(x)
        }
        fn after_trial(&self, study: &Study<f64>, _trial: &CompletedTrial<f64>) -> ControlFlow<()> {
            if study.n_trials() >= 5 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }
    }

    let study: Study<f64> = Study::new(Direction::Minimize);
    study
        .optimize_with(
            100,
            EarlyStopAfter5 {
                x_param: FloatParam::new(0.0, 10.0),
            },
        )
        .expect("optimization should succeed");

    assert_eq!(study.n_trials(), 5, "should have stopped after 5 trials");
}

#[test]
fn test_callback_early_stopping_on_first_trial() {
    use std::ops::ControlFlow;

    use optimizer::Objective;
    use optimizer::sampler::CompletedTrial;

    struct StopImmediately {
        x_param: FloatParam,
    }

    impl Objective<f64> for StopImmediately {
        type Error = Error;
        fn evaluate(&self, trial: &mut Trial) -> Result<f64, Error> {
            let x = self.x_param.suggest(trial)?;
            Ok(x)
        }
        fn after_trial(
            &self,
            _study: &Study<f64>,
            _trial: &CompletedTrial<f64>,
        ) -> ControlFlow<()> {
            ControlFlow::Break(())
        }
    }

    let study: Study<f64> = Study::new(Direction::Minimize);
    study
        .optimize_with(
            100,
            StopImmediately {
                x_param: FloatParam::new(0.0, 10.0),
            },
        )
        .expect("optimization should succeed");

    assert_eq!(study.n_trials(), 1, "should have stopped after 1 trial");
}

#[test]
fn test_callback_sampler_early_stopping() {
    use std::ops::ControlFlow;

    use optimizer::Objective;
    use optimizer::sampler::CompletedTrial;

    struct StopAfter3 {
        x_param: FloatParam,
    }

    impl Objective<f64> for StopAfter3 {
        type Error = Error;
        fn evaluate(&self, trial: &mut Trial) -> Result<f64, Error> {
            let x = self.x_param.suggest(trial)?;
            Ok(x)
        }
        fn after_trial(&self, study: &Study<f64>, _trial: &CompletedTrial<f64>) -> ControlFlow<()> {
            if study.n_trials() >= 3 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }
    }

    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    study
        .optimize_with(
            100,
            StopAfter3 {
                x_param: FloatParam::new(0.0, 10.0),
            },
        )
        .expect("optimization should succeed");

    assert_eq!(study.n_trials(), 3);
}

#[test]
fn test_retries_successful_trials_not_retried() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use optimizer::Objective;

    struct SuccessObj {
        x_param: FloatParam,
        call_count: Arc<AtomicU32>,
    }

    impl Objective<f64> for SuccessObj {
        type Error = Error;
        fn evaluate(&self, trial: &mut Trial) -> Result<f64, Error> {
            let x = self.x_param.suggest(trial)?;
            self.call_count.fetch_add(1, Ordering::Relaxed);
            Ok(x * x)
        }
        fn max_retries(&self) -> usize {
            3
        }
    }

    let study: Study<f64> = Study::new(Direction::Minimize);
    let call_count = Arc::new(AtomicU32::new(0));
    let obj = SuccessObj {
        x_param: FloatParam::new(0.0, 10.0),
        call_count: Arc::clone(&call_count),
    };

    study.optimize_with(5, obj).unwrap();

    // All trials succeed on first try — exactly 5 calls
    assert_eq!(call_count.load(Ordering::Relaxed), 5);
    assert_eq!(study.n_trials(), 5);
}

#[test]
fn test_retries_failed_trials_retried_up_to_max() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use optimizer::Objective;

    struct AlwaysFailObj {
        x_param: FloatParam,
        call_count: Arc<AtomicU32>,
    }

    impl Objective<f64> for AlwaysFailObj {
        type Error = String;
        fn evaluate(&self, trial: &mut Trial) -> Result<f64, String> {
            let _ = self.x_param.suggest(trial).map_err(|e| e.to_string())?;
            self.call_count.fetch_add(1, Ordering::Relaxed);
            Err("always fails".to_string())
        }
        fn max_retries(&self) -> usize {
            3
        }
    }

    let study: Study<f64> = Study::new(Direction::Minimize);
    let call_count = Arc::new(AtomicU32::new(0));
    let obj = AlwaysFailObj {
        x_param: FloatParam::new(0.0, 10.0),
        call_count: Arc::clone(&call_count),
    };

    let result = study.optimize_with(1, obj);

    // 1 initial attempt + 3 retries = 4 total calls
    assert_eq!(call_count.load(Ordering::Relaxed), 4);
    // No trials completed
    assert!(matches!(result, Err(Error::NoCompletedTrials)));
}

#[test]
fn test_retries_permanently_failed_after_exhaustion() {
    use optimizer::Objective;

    struct AlwaysFailObj {
        x_param: FloatParam,
    }

    impl Objective<f64> for AlwaysFailObj {
        type Error = String;
        fn evaluate(&self, trial: &mut Trial) -> Result<f64, String> {
            let _ = self.x_param.suggest(trial).map_err(|e| e.to_string())?;
            Err("transient error".to_string())
        }
        fn max_retries(&self) -> usize {
            2
        }
    }

    let study: Study<f64> = Study::new(Direction::Minimize);
    let obj = AlwaysFailObj {
        x_param: FloatParam::new(0.0, 10.0),
    };

    let result = study.optimize_with(3, obj);

    assert!(
        matches!(result, Err(Error::NoCompletedTrials)),
        "all trials should permanently fail"
    );
    assert_eq!(
        study.n_trials(),
        0,
        "no completed trials should be recorded"
    );
}

#[test]
fn test_retries_uses_same_parameters() {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    use optimizer::Objective;

    struct RetryObj {
        x_param: FloatParam,
        seen_values: Arc<Mutex<Vec<f64>>>,
        call_count: Arc<AtomicU32>,
    }

    impl Objective<f64> for RetryObj {
        type Error = String;
        fn evaluate(&self, trial: &mut Trial) -> Result<f64, String> {
            let x = self.x_param.suggest(trial).map_err(|e| e.to_string())?;
            self.seen_values.lock().unwrap().push(x);
            let count = self.call_count.fetch_add(1, Ordering::Relaxed) + 1;
            // Fail first two attempts, succeed on third
            if count < 3 {
                Err("transient".to_string())
            } else {
                Ok(x * x)
            }
        }
        fn max_retries(&self) -> usize {
            2
        }
    }

    let study: Study<f64> = Study::new(Direction::Minimize);
    let seen_values = Arc::new(Mutex::new(Vec::new()));
    let call_count = Arc::new(AtomicU32::new(0));
    let obj = RetryObj {
        x_param: FloatParam::new(0.0, 10.0),
        seen_values: Arc::clone(&seen_values),
        call_count: Arc::clone(&call_count),
    };

    study.optimize_with(1, obj).unwrap();

    let values = seen_values.lock().unwrap();
    assert_eq!(values.len(), 3, "should be called 3 times (1 + 2 retries)");
    // All three calls should have gotten the same parameter value
    assert_eq!(values[0], values[1]);
    assert_eq!(values[1], values[2]);
}

#[test]
fn test_retries_n_trials_counts_unique_configs() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use optimizer::Objective;

    struct FailFirstObj {
        x_param: FloatParam,
        call_count: Arc<AtomicU32>,
    }

    impl Objective<f64> for FailFirstObj {
        type Error = String;
        fn evaluate(&self, trial: &mut Trial) -> Result<f64, String> {
            let x = self.x_param.suggest(trial).map_err(|e| e.to_string())?;
            let count = self.call_count.fetch_add(1, Ordering::Relaxed) + 1;
            // Fail first attempt of each config, succeed on retry
            if count % 2 == 1 {
                Err("transient".to_string())
            } else {
                Ok(x * x)
            }
        }
        fn max_retries(&self) -> usize {
            2
        }
    }

    let study: Study<f64> = Study::new(Direction::Minimize);
    let call_count = Arc::new(AtomicU32::new(0));
    let obj = FailFirstObj {
        x_param: FloatParam::new(0.0, 10.0),
        call_count: Arc::clone(&call_count),
    };

    study.optimize_with(3, obj).unwrap();

    // 3 unique configs, each needing 2 calls = 6 total calls
    assert_eq!(call_count.load(Ordering::Relaxed), 6);
    // But only 3 completed trials
    assert_eq!(study.n_trials(), 3);
}

#[test]
fn test_retries_with_zero_max_retries_same_as_optimize() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(0.0, 10.0);
    let call_count = std::cell::Cell::new(0u32);

    study
        .optimize(5, |trial| {
            let x = x_param.suggest(trial)?;
            call_count.set(call_count.get() + 1);
            Ok::<_, Error>(x * x)
        })
        .unwrap();

    assert_eq!(call_count.get(), 5);
    assert_eq!(study.n_trials(), 5);
}
