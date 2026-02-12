use optimizer::parameter::{FloatParam, Parameter};
use optimizer::sampler::tpe::TpeSampler;
use optimizer::{Direction, Study};

#[test]
fn test_ask_and_tell_basic() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(0.0, 10.0);

    for _ in 0..10 {
        let mut trial = study.ask();
        let x = x_param.suggest(&mut trial).unwrap();
        let value = x * x;
        study.tell(trial, Ok::<_, &str>(value));
    }

    assert_eq!(study.n_trials(), 10);
    assert!(study.best_value().unwrap() >= 0.0);
}

#[test]
fn test_ask_and_tell_with_failures() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(-5.0, 5.0);

    // Alternate success and failure
    for i in 0..10 {
        let mut trial = study.ask();
        let x = x_param.suggest(&mut trial).unwrap();
        if i % 2 == 0 {
            study.tell(trial, Ok::<_, &str>(x * x));
        } else {
            study.tell(trial, Err::<f64, _>("simulated failure"));
        }
    }

    // Only successful trials are counted
    assert_eq!(study.n_trials(), 5);
}

#[test]
fn test_ask_and_tell_with_tpe_sampler() {
    let sampler = TpeSampler::builder()
        .seed(42)
        .n_startup_trials(5)
        .build()
        .unwrap();
    let study: Study<f64> = Study::minimize(sampler);
    let x_param = FloatParam::new(-10.0, 10.0);

    for _ in 0..30 {
        let mut trial = study.ask();
        let x = x_param.suggest(&mut trial).unwrap();
        study.tell(trial, Ok::<_, &str>((x - 3.0).powi(2)));
    }

    assert_eq!(study.n_trials(), 30);
    assert!(
        study.best_value().unwrap() < 5.0,
        "TPE ask-and-tell should find a reasonable value"
    );
}

#[test]
fn test_ask_and_tell_batch() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x_param = FloatParam::new(0.0, 10.0);

    // Ask a batch of trials
    let batch: Vec<_> = (0..5)
        .map(|_| {
            let mut t = study.ask();
            let x = x_param.suggest(&mut t).unwrap();
            (t, x)
        })
        .collect();

    // Tell results for the batch
    for (trial, x) in batch {
        study.tell(trial, Ok::<_, &str>(x * x));
    }

    assert_eq!(study.n_trials(), 5);
}

#[test]
fn test_ask_and_tell_with_custom_value_type() {
    // Ask-and-tell works with non-f64 value types too
    let study: Study<i32> = Study::new(Direction::Maximize);

    for i in 0..5 {
        let trial = study.ask();
        study.tell(trial, Ok::<_, &str>(i * 10));
    }

    assert_eq!(study.n_trials(), 5);
    assert_eq!(study.best_value().unwrap(), 40);
}
