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
            // The current trial has not yet been pushed to storage when
            // after_trial fires, so n_trials() == 4 means this is the 5th.
            if study.n_trials() >= 4 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }
    }

    let study: Study<f64> = Study::new(Direction::Minimize);
    study
        .optimize(
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
        .optimize(
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
            // The current trial has not yet been pushed to storage when
            // after_trial fires, so n_trials() == 2 means this is the 3rd.
            if study.n_trials() >= 2 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }
    }

    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    study
        .optimize(
            100,
            StopAfter3 {
                x_param: FloatParam::new(0.0, 10.0),
            },
        )
        .expect("optimization should succeed");

    assert_eq!(study.n_trials(), 3);
}

#[test]
fn test_objective_struct_basic() {
    use optimizer::Objective;

    struct SquareObj {
        x_param: FloatParam,
    }

    impl Objective<f64> for SquareObj {
        type Error = Error;
        fn evaluate(&self, trial: &mut Trial) -> Result<f64, Error> {
            let x = self.x_param.suggest(trial)?;
            Ok(x * x)
        }
    }

    let study: Study<f64> = Study::new(Direction::Minimize);
    let obj = SquareObj {
        x_param: FloatParam::new(0.0, 10.0),
    };

    study.optimize(5, obj).unwrap();

    assert_eq!(study.n_trials(), 5);
}

#[test]
fn test_closure_and_objective_produce_same_results() {
    let x_param = FloatParam::new(0.0, 10.0);

    let study: Study<f64> = Study::new(Direction::Minimize);
    study
        .optimize(5, |trial: &mut Trial| {
            let x = x_param.suggest(trial)?;
            Ok::<_, Error>(x * x)
        })
        .unwrap();

    assert_eq!(study.n_trials(), 5);
}
