use optimizer::parameter::{FloatParam, IntParam, Parameter};
use optimizer::sampler::random::RandomSampler;
use optimizer::{Direction, Study};

#[test]
fn csv_empty_study_produces_header_only() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let mut buf = Vec::new();
    study.to_csv(&mut buf).unwrap();
    let csv = String::from_utf8(buf).unwrap();
    assert_eq!(csv, "trial_id,value,state\n");
}

#[test]
fn csv_includes_all_trial_data() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let x = FloatParam::new(0.0, 10.0).name("x");
    let y = IntParam::new(1, 5).name("y");

    study
        .optimize(3, |trial| {
            let xv = x.suggest(trial)?;
            let yv = y.suggest(trial)?;
            Ok::<_, optimizer::Error>(xv + yv as f64)
        })
        .unwrap();

    let mut buf = Vec::new();
    study.to_csv(&mut buf).unwrap();
    let csv = String::from_utf8(buf).unwrap();

    let lines: Vec<&str> = csv.lines().collect();
    // Header + 3 data rows.
    assert_eq!(lines.len(), 4);

    // Header should contain our parameter names.
    let header = lines[0];
    assert!(header.starts_with("trial_id,value,state"));
    assert!(header.contains("x"));
    assert!(header.contains("y"));

    // Each data row should have the right number of columns.
    let n_cols = header.split(',').count();
    for line in &lines[1..] {
        assert_eq!(line.split(',').count(), n_cols);
    }

    // All rows should have "Complete" state.
    for line in &lines[1..] {
        assert!(line.contains("Complete"));
    }
}

#[test]
fn csv_handles_pruned_trials() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let x = FloatParam::new(0.0, 10.0).name("x");

    // First trial: complete
    let mut trial = study.create_trial();
    let _ = x.suggest(&mut trial).unwrap();
    study.complete_trial(trial, 1.0);

    // Second trial: pruned
    let mut trial = study.create_trial();
    let _ = x.suggest(&mut trial).unwrap();
    study.prune_trial(trial);

    let mut buf = Vec::new();
    study.to_csv(&mut buf).unwrap();
    let csv = String::from_utf8(buf).unwrap();

    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), 3); // header + 2 data rows

    // Pruned trial should have empty value.
    let pruned_line = lines[2];
    assert!(pruned_line.contains("Pruned"));
    // The value field (second column) should be empty.
    let cols: Vec<&str> = pruned_line.split(',').collect();
    assert_eq!(cols[2], "Pruned");
    assert_eq!(cols[1], ""); // empty value for pruned
}

#[test]
fn csv_handles_different_parameter_sets() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let x = FloatParam::new(0.0, 10.0).name("x");
    let y = FloatParam::new(0.0, 10.0).name("y");

    // First trial: only x
    let mut trial = study.create_trial();
    let xv = x.suggest(&mut trial).unwrap();
    study.complete_trial(trial, xv);

    // Second trial: only y
    let mut trial = study.create_trial();
    let yv = y.suggest(&mut trial).unwrap();
    study.complete_trial(trial, yv);

    let mut buf = Vec::new();
    study.to_csv(&mut buf).unwrap();
    let csv = String::from_utf8(buf).unwrap();

    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), 3);

    // Both x and y columns should exist.
    let header = lines[0];
    assert!(header.contains("x"));
    assert!(header.contains("y"));

    // Each row has the right column count (missing params are empty).
    let n_cols = header.split(',').count();
    for line in &lines[1..] {
        assert_eq!(line.split(',').count(), n_cols);
    }
}

#[test]
fn csv_output_is_parseable() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let lr = FloatParam::new(0.001, 0.1).name("learning_rate");
    let layers = IntParam::new(1, 5).name("n_layers");

    study
        .optimize(5, |trial| {
            let l = lr.suggest(trial)?;
            let n = layers.suggest(trial)?;
            Ok::<_, optimizer::Error>(l * n as f64)
        })
        .unwrap();

    let mut buf = Vec::new();
    study.to_csv(&mut buf).unwrap();
    let csv = String::from_utf8(buf).unwrap();

    // Parse each row: every value field should be a valid f64 for complete trials.
    let lines: Vec<&str> = csv.lines().collect();
    for line in &lines[1..] {
        let cols: Vec<&str> = line.split(',').collect();
        // trial_id should be a number
        cols[0].parse::<u64>().unwrap();
        // value should be parseable as f64
        cols[1].parse::<f64>().unwrap();
        // state should be a known value
        assert!(["Complete", "Pruned", "Failed", "Running"].contains(&cols[2]));
    }
}

#[test]
fn export_csv_writes_file() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let x = FloatParam::new(0.0, 10.0).name("x");
    study
        .optimize(3, |trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(xv * xv)
        })
        .unwrap();

    let dir = std::env::temp_dir().join("optimizer_export_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_export.csv");

    study.export_csv(&path).unwrap();

    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.starts_with("trial_id,value,state"));
    assert!(contents.lines().count() == 4); // header + 3 rows

    // Clean up.
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(feature = "serde")]
#[test]
fn export_json_writes_file() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let x = FloatParam::new(0.0, 10.0).name("x");
    study
        .optimize(3, |trial| {
            let xv = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(xv * xv)
        })
        .unwrap();

    let dir = std::env::temp_dir().join("optimizer_json_export_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_export.json");

    study.export_json(&path).unwrap();

    let contents = std::fs::read_to_string(&path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
    let arr = parsed.as_array().unwrap();
    assert_eq!(arr.len(), 3);

    // Each entry should have the expected fields.
    for entry in arr {
        assert!(entry.get("id").is_some());
        assert!(entry.get("value").is_some());
        assert!(entry.get("state").is_some());
        assert!(entry.get("params").is_some());
    }

    // Clean up.
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn csv_includes_user_attributes() {
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, RandomSampler::with_seed(42));
    let x = FloatParam::new(0.0, 10.0).name("x");

    study
        .optimize(2, |trial| {
            let xv = x.suggest(trial)?;
            trial.set_user_attr("training_time_secs", 45.2);
            Ok::<_, optimizer::Error>(xv * xv)
        })
        .unwrap();

    let mut buf = Vec::new();
    study.to_csv(&mut buf).unwrap();
    let csv = String::from_utf8(buf).unwrap();

    let header = csv.lines().next().unwrap();
    assert!(header.contains("training_time_secs"));
}
