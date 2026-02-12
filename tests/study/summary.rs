use optimizer::parameter::{FloatParam, Parameter};
use optimizer::sampler::random::RandomSampler;
use optimizer::{Direction, Error, Study};

#[test]
fn test_summary_with_completed_trials() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(1));
    let x = FloatParam::new(0.0, 10.0).name("x");

    study
        .optimize(5, |trial| {
            let val = x.suggest(trial)?;
            Ok::<_, Error>(val * val)
        })
        .unwrap();

    let summary = study.summary();
    assert!(summary.contains("Minimize"));
    assert!(summary.contains("5 trials"));
    assert!(summary.contains("Best value:"));
    assert!(summary.contains("x = "));
}

#[test]
fn test_summary_no_completed_trials() {
    let study: Study<f64> = Study::new(Direction::Maximize);
    let summary = study.summary();
    assert!(summary.contains("Maximize"));
    assert!(summary.contains("0 trials"));
    assert!(!summary.contains("Best value:"));
}

#[test]
fn test_summary_with_pruned_trials() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(1));
    let x = FloatParam::new(0.0, 10.0).name("x");

    // Manually create some complete and pruned trials
    for _ in 0..3 {
        let mut trial = study.create_trial();
        let val = x.suggest(&mut trial).unwrap();
        study.complete_trial(trial, val);
    }
    for _ in 0..2 {
        let mut trial = study.create_trial();
        let _ = x.suggest(&mut trial).unwrap();
        study.prune_trial(trial);
    }

    let summary = study.summary();
    // Should show breakdown when there are pruned trials
    if study.n_pruned_trials() > 0 {
        assert!(summary.contains("complete"));
        assert!(summary.contains("pruned"));
    }
}

#[test]
fn test_display_matches_summary() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(1));
    let x = FloatParam::new(0.0, 10.0).name("x");

    study
        .optimize(3, |trial| {
            let val = x.suggest(trial)?;
            Ok::<_, Error>(val)
        })
        .unwrap();

    assert_eq!(format!("{study}"), study.summary());
}
