use optimizer::parameter::{FloatParam, IntParam, Parameter};
use optimizer::sampler::random::RandomSampler;
use optimizer::{Direction, Study, generate_html_report};

#[test]
fn html_report_creates_file() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let x = FloatParam::new(0.0, 10.0).name("x");
    let y = IntParam::new(1, 5).name("y");

    study
        .optimize(10, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            let yv = y.suggest(trial)?;
            Ok::<_, optimizer::Error>(xv + yv as f64)
        })
        .unwrap();

    let path = std::env::temp_dir().join("test_report_creates_file.html");
    generate_html_report(&study, &path).unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("<!DOCTYPE html>"));
    assert!(content.contains("plotly"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn html_report_contains_all_chart_sections() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let x = FloatParam::new(0.0, 10.0).name("x");
    let y = FloatParam::new(-5.0, 5.0).name("y");

    study
        .optimize(20, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            let yv = y.suggest(trial)?;
            Ok::<_, optimizer::Error>(xv * xv + yv * yv)
        })
        .unwrap();

    let path = std::env::temp_dir().join("test_report_all_charts.html");
    generate_html_report(&study, &path).unwrap();

    let content = std::fs::read_to_string(&path).unwrap();

    // Should contain all chart divs.
    assert!(content.contains("id=\"history\""));
    assert!(content.contains("id=\"slices\""));
    assert!(content.contains("id=\"parcoords\""));
    assert!(content.contains("id=\"importance\""));
    assert!(content.contains("id=\"timeline\""));

    // Should contain chart titles.
    assert!(content.contains("Optimization History"));
    assert!(content.contains("Slice Plots"));
    assert!(content.contains("Parallel Coordinates"));
    assert!(content.contains("Parameter Importance"));
    assert!(content.contains("Trial Timeline"));

    // Should show direction and trial count.
    assert!(content.contains("Minimize"));
    assert!(content.contains("20 trials"));

    std::fs::remove_file(&path).ok();
}

#[test]
fn html_report_empty_study() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    let path = std::env::temp_dir().join("test_report_empty.html");
    generate_html_report(&study, &path).unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("<!DOCTYPE html>"));
    assert!(content.contains("0 trials"));

    std::fs::remove_file(&path).ok();
}

#[test]
fn html_report_single_param_no_parcoords() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let x = FloatParam::new(0.0, 10.0).name("x");

    study
        .optimize(5, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(xv * xv)
        })
        .unwrap();

    let path = std::env::temp_dir().join("test_report_single_param.html");
    generate_html_report(&study, &path).unwrap();

    let content = std::fs::read_to_string(&path).unwrap();

    // Should have slice plot but not parallel coordinates (needs >= 2 params).
    assert!(content.contains("id=\"slices\""));
    assert!(!content.contains("id=\"parcoords\""));

    std::fs::remove_file(&path).ok();
}

#[test]
fn html_report_maximize_direction() {
    let study: Study<f64> = Study::with_sampler(Direction::Maximize, RandomSampler::with_seed(42));
    let x = FloatParam::new(0.0, 10.0).name("x");

    study
        .optimize(5, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(xv)
        })
        .unwrap();

    let path = std::env::temp_dir().join("test_report_maximize.html");
    generate_html_report(&study, &path).unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("Maximize"));

    std::fs::remove_file(&path).ok();
}

#[test]
fn export_html_convenience_method() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let x = FloatParam::new(0.0, 10.0).name("x");

    study
        .optimize(5, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(xv * xv)
        })
        .unwrap();

    let path = std::env::temp_dir().join("test_export_html.html");
    study.export_html(&path).unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("<!DOCTYPE html>"));
    assert!(content.contains("id=\"history\""));

    std::fs::remove_file(&path).ok();
}

#[test]
fn html_report_with_intermediate_values() {
    use optimizer::pruner::MedianPruner;

    let mut study: Study<f64> =
        Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    study.set_pruner(MedianPruner::new(Direction::Minimize));
    let x = FloatParam::new(0.0, 10.0).name("x");

    study
        .optimize(10, |trial: &mut optimizer::Trial| {
            let xv = x.suggest(trial)?;
            for step in 0..5 {
                let val = xv * xv + step as f64;
                trial.report(step, val);
                if trial.should_prune() {
                    return Err(optimizer::TrialPruned.into());
                }
            }
            Ok::<_, optimizer::Error>(xv * xv)
        })
        .unwrap();

    let path = std::env::temp_dir().join("test_report_intermediate.html");
    generate_html_report(&study, &path).unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("id=\"intermediate\""));
    assert!(content.contains("Intermediate Values"));

    std::fs::remove_file(&path).ok();
}
