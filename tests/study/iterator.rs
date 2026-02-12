use optimizer::parameter::{FloatParam, Parameter};
use optimizer::sampler::random::RandomSampler;
use optimizer::{Direction, Study};

#[test]
fn test_into_iterator_iterates_all_trials() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let x_param = FloatParam::new(0.0, 10.0);

    for _ in 0..5 {
        let mut trial = study.create_trial();
        let x = x_param.suggest(&mut trial).unwrap();
        study.complete_trial(trial, x * x);
    }

    let mut count = 0;
    for trial in &study {
        assert_eq!(trial.state, optimizer::TrialState::Complete);
        count += 1;
    }
    assert_eq!(count, 5);
}

#[test]
fn test_into_iterator_empty_study() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    let count = (&study).into_iter().count();
    assert_eq!(count, 0);
}

#[test]
fn test_into_iterator_preserves_insertion_order() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    for i in 0..3 {
        let trial = study.create_trial();
        study.complete_trial(trial, f64::from(i));
    }

    let ids: Vec<u64> = (&study).into_iter().map(|t| t.id).collect();
    assert_eq!(ids, vec![0, 1, 2]);
}
