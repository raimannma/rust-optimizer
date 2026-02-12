use std::collections::HashMap;

use optimizer::parameter::{FloatParam, IntParam, ParamValue, Parameter};
use optimizer::sampler::random::RandomSampler;
use optimizer::{Direction, Error, Study};

#[test]
fn test_enqueue_params_evaluated_first() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 10.0);
    let y = IntParam::new(1, 100);

    // Enqueue a specific configuration
    study.enqueue(HashMap::from([
        (x.id(), ParamValue::Float(5.0)),
        (y.id(), ParamValue::Int(42)),
    ]));

    // The first trial should use the enqueued params
    let mut trial = study.ask();
    let x_val = x.suggest(&mut trial).unwrap();
    let y_val = y.suggest(&mut trial).unwrap();

    assert_eq!(x_val, 5.0);
    assert_eq!(y_val, 42);
}

#[test]
fn test_enqueue_fifo_order() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 10.0);

    // Enqueue two configs
    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(1.0))]));
    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(2.0))]));

    // First trial gets first enqueued value
    let mut trial1 = study.ask();
    assert_eq!(x.suggest(&mut trial1).unwrap(), 1.0);

    // Second trial gets second enqueued value
    let mut trial2 = study.ask();
    assert_eq!(x.suggest(&mut trial2).unwrap(), 2.0);
}

#[test]
fn test_enqueue_then_normal_sampling_resumes() {
    let sampler = RandomSampler::with_seed(42);
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
    let x = FloatParam::new(0.0, 10.0);

    // Enqueue one config
    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(5.0))]));

    // First trial uses enqueued value
    let mut trial1 = study.ask();
    assert_eq!(x.suggest(&mut trial1).unwrap(), 5.0);
    study.tell(trial1, Ok::<_, &str>(25.0));

    // Second trial uses normal sampling (not 5.0)
    let mut trial2 = study.ask();
    let x_val = x.suggest(&mut trial2).unwrap();
    // The sampled value should be in [0, 10] but extremely unlikely to be exactly 5.0
    assert!((0.0..=10.0).contains(&x_val));
}

#[test]
fn test_enqueue_with_optimize() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 10.0);

    // Enqueue two specific configs
    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(1.0))]));
    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(2.0))]));

    let mut values = Vec::new();

    study
        .optimize(5, |trial| {
            let x_val = x.suggest(trial)?;
            values.push(x_val);
            Ok::<_, Error>(x_val * x_val)
        })
        .unwrap();

    // First two trials should use enqueued values
    assert_eq!(values[0], 1.0);
    assert_eq!(values[1], 2.0);
    // All 5 trials should have completed
    assert_eq!(study.n_trials(), 5);
}

#[test]
fn test_enqueue_partial_params_fall_back_to_sampling() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 10.0);
    let y = IntParam::new(1, 100);

    // Enqueue only x, not y
    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(3.0))]));

    let mut trial = study.ask();
    let x_val = x.suggest(&mut trial).unwrap();
    let y_val = y.suggest(&mut trial).unwrap();

    // x should be the enqueued value
    assert_eq!(x_val, 3.0);
    // y should be sampled (within range)
    assert!((1..=100).contains(&y_val));
}

#[test]
fn test_enqueue_trials_appear_in_completed_trials() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 10.0);

    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(7.0))]));

    study
        .optimize(1, |trial| {
            let x_val = x.suggest(trial)?;
            Ok::<_, Error>(x_val)
        })
        .unwrap();

    let trials = study.trials();
    assert_eq!(trials.len(), 1);
    assert_eq!(trials[0].value, 7.0);
    assert_eq!(
        *trials[0].params.get(&x.id()).unwrap(),
        ParamValue::Float(7.0)
    );
}

#[test]
fn test_enqueue_with_ask_and_tell() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 10.0);

    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(4.0))]));

    let mut trial = study.ask();
    let x_val = x.suggest(&mut trial).unwrap();
    assert_eq!(x_val, 4.0);

    study.tell(trial, Ok::<_, &str>(x_val * x_val));
    assert_eq!(study.n_trials(), 1);
    assert_eq!(study.best_value().unwrap(), 16.0);
}

#[test]
fn test_n_enqueued() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 10.0);

    assert_eq!(study.n_enqueued(), 0);

    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(1.0))]));
    assert_eq!(study.n_enqueued(), 1);

    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(2.0))]));
    assert_eq!(study.n_enqueued(), 2);

    // Creating a trial dequeues one
    let _ = study.ask();
    assert_eq!(study.n_enqueued(), 1);

    let _ = study.ask();
    assert_eq!(study.n_enqueued(), 0);
}

#[test]
fn test_enqueue_counted_in_n_trials() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 10.0);

    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(1.0))]));
    study.enqueue(HashMap::from([(x.id(), ParamValue::Float(2.0))]));

    study
        .optimize(5, |trial| {
            let x_val = x.suggest(trial)?;
            Ok::<_, Error>(x_val)
        })
        .unwrap();

    // All 5 trials count, including the 2 enqueued ones
    assert_eq!(study.n_trials(), 5);
}
