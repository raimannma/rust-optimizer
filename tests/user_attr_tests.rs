use optimizer::parameter::{FloatParam, Parameter};
use optimizer::{AttrValue, Direction, Study};

#[test]
fn set_and_get_float_attr() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(1, |trial: &mut optimizer::Trial| {
            let _ = x.suggest(trial)?;
            trial.set_user_attr("score", 42.5);
            assert_eq!(trial.user_attr("score"), Some(&AttrValue::Float(42.5)));
            Ok::<_, optimizer::Error>(1.0)
        })
        .unwrap();
}

#[test]
fn set_and_get_int_attr() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(1, |trial: &mut optimizer::Trial| {
            let _ = x.suggest(trial)?;
            trial.set_user_attr("epoch", 42_i64);
            assert_eq!(trial.user_attr("epoch"), Some(&AttrValue::Int(42)));
            Ok::<_, optimizer::Error>(1.0)
        })
        .unwrap();
}

#[test]
fn set_and_get_string_attr() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(1, |trial: &mut optimizer::Trial| {
            let _ = x.suggest(trial)?;
            trial.set_user_attr("model", "resnet50");
            assert_eq!(
                trial.user_attr("model"),
                Some(&AttrValue::String("resnet50".to_owned()))
            );
            Ok::<_, optimizer::Error>(1.0)
        })
        .unwrap();
}

#[test]
fn set_and_get_bool_attr() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(1, |trial: &mut optimizer::Trial| {
            let _ = x.suggest(trial)?;
            trial.set_user_attr("converged", true);
            assert_eq!(trial.user_attr("converged"), Some(&AttrValue::Bool(true)));
            Ok::<_, optimizer::Error>(1.0)
        })
        .unwrap();
}

#[test]
fn attrs_propagate_to_completed_trial() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(1, |trial: &mut optimizer::Trial| {
            let _ = x.suggest(trial)?;
            trial.set_user_attr("time_secs", 1.5);
            trial.set_user_attr("tag", "baseline");
            Ok::<_, optimizer::Error>(1.0)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    assert_eq!(best.user_attr("time_secs"), Some(&AttrValue::Float(1.5)));
    assert_eq!(
        best.user_attr("tag"),
        Some(&AttrValue::String("baseline".to_owned()))
    );
}

#[test]
fn overwrite_attr_replaces_value() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(1, |trial: &mut optimizer::Trial| {
            let _ = x.suggest(trial)?;
            trial.set_user_attr("key", "old");
            trial.set_user_attr("key", "new");
            assert_eq!(
                trial.user_attr("key"),
                Some(&AttrValue::String("new".to_owned()))
            );
            Ok::<_, optimizer::Error>(1.0)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    assert_eq!(
        best.user_attr("key"),
        Some(&AttrValue::String("new".to_owned()))
    );
}

#[test]
fn missing_attr_returns_none() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(1, |trial: &mut optimizer::Trial| {
            let _ = x.suggest(trial)?;
            assert_eq!(trial.user_attr("nonexistent"), None);
            Ok::<_, optimizer::Error>(1.0)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    assert_eq!(best.user_attr("nonexistent"), None);
}

#[test]
fn user_attrs_map_returns_all() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(1, |trial: &mut optimizer::Trial| {
            let _ = x.suggest(trial)?;
            trial.set_user_attr("a", 1.0);
            trial.set_user_attr("b", true);
            assert_eq!(trial.user_attrs().len(), 2);
            Ok::<_, optimizer::Error>(1.0)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    assert_eq!(best.user_attrs().len(), 2);
}
