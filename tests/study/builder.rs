use optimizer::parameter::{FloatParam, Parameter};
use optimizer::sampler::random::RandomSampler;
use optimizer::sampler::tpe::TpeSampler;
use optimizer::{Direction, Error, Study};

#[test]
fn test_builder_defaults() {
    let study: Study<f64> = Study::builder().build();
    assert_eq!(study.direction(), Direction::Minimize);
}

#[test]
fn test_builder_maximize() {
    let study: Study<f64> = Study::builder().maximize().build();
    assert_eq!(study.direction(), Direction::Maximize);
}

#[test]
fn test_builder_minimize() {
    let study: Study<f64> = Study::builder().minimize().build();
    assert_eq!(study.direction(), Direction::Minimize);
}

#[test]
fn test_builder_direction() {
    let study: Study<f64> = Study::builder().direction(Direction::Maximize).build();
    assert_eq!(study.direction(), Direction::Maximize);
}

#[test]
fn test_builder_with_sampler() {
    let x = FloatParam::new(-5.0, 5.0);
    let study: Study<f64> = Study::builder().sampler(TpeSampler::new()).build();

    study
        .optimize(10, |trial: &mut optimizer::Trial| {
            let val = x.suggest(trial)?;
            Ok::<_, Error>(val * val)
        })
        .unwrap();

    assert_eq!(study.trials().len(), 10);
}

#[test]
fn test_builder_with_pruner() {
    use optimizer::pruner::NopPruner;

    let study: Study<f64> = Study::builder().pruner(NopPruner).build();

    assert_eq!(study.direction(), Direction::Minimize);
}

#[test]
fn test_builder_chaining() {
    let study: Study<f64> = Study::builder()
        .maximize()
        .sampler(RandomSampler::with_seed(42))
        .pruner(optimizer::pruner::NopPruner)
        .build();

    assert_eq!(study.direction(), Direction::Maximize);
}

#[test]
fn test_builder_with_custom_value_type() {
    let study: Study<i32> = Study::builder().maximize().build();
    assert_eq!(study.direction(), Direction::Maximize);
}

#[test]
fn test_builder_optimizes_correctly() {
    let x = FloatParam::new(-10.0, 10.0);
    let study: Study<f64> = Study::builder()
        .minimize()
        .sampler(TpeSampler::builder().seed(42).build().unwrap())
        .build();

    study
        .optimize(100, |trial: &mut optimizer::Trial| {
            let val = x.suggest(trial)?;
            Ok::<_, Error>((val - 3.0) * (val - 3.0))
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    assert!(
        best.value < 5.0,
        "best value should be < 5.0, got {}",
        best.value
    );
}
