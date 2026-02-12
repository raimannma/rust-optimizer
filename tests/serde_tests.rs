#![cfg(feature = "serde")]

use std::collections::HashMap;

use optimizer::parameter::{FloatParam, IntParam, ParamValue, Parameter};
use optimizer::sampler::CompletedTrial;
use optimizer::{Direction, Study, StudySnapshot, TrialState};

#[test]
fn round_trip_save_load() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(-10.0, 10.0).name("x");
    let n = IntParam::new(1, 100).name("n");

    study
        .optimize(5, |trial| {
            let x_val = x.suggest(trial)?;
            let n_val = n.suggest(trial)?;
            Ok::<_, optimizer::Error>(x_val * x_val + n_val as f64)
        })
        .unwrap();

    let dir = tempdir();
    let path = dir.join("study.json");

    study.save(&path).unwrap();
    let loaded: Study<f64> = Study::load(&path).unwrap();

    assert_eq!(loaded.direction(), study.direction());
    assert_eq!(loaded.n_trials(), study.n_trials());

    let orig_trials = study.trials();
    let loaded_trials = loaded.trials();

    for (orig, loaded) in orig_trials.iter().zip(loaded_trials.iter()) {
        assert_eq!(orig.id, loaded.id);
        assert!((orig.value - loaded.value).abs() < 1e-10);
        assert_eq!(orig.state, loaded.state);
        assert_eq!(orig.params.len(), loaded.params.len());
        assert_eq!(orig.distributions, loaded.distributions);
        assert_eq!(orig.param_labels, loaded.param_labels);
    }

    // Clean up
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn json_output_is_human_readable() {
    let study: Study<f64> = Study::new(Direction::Maximize);
    let x = FloatParam::new(0.0, 1.0).name("x");

    study
        .optimize(2, |trial| {
            let v = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(v)
        })
        .unwrap();

    let dir = tempdir();
    let path = dir.join("study.json");
    study.save(&path).unwrap();

    let contents = std::fs::read_to_string(&path).unwrap();

    // Verify it's pretty-printed JSON with recognizable fields
    assert!(contents.contains("\"version\""));
    assert!(contents.contains("\"direction\""));
    assert!(contents.contains("\"trials\""));
    assert!(contents.contains("\"next_trial_id\""));
    assert!(contents.contains("\"Maximize\""));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn round_trip_empty_study() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    let dir = tempdir();
    let path = dir.join("empty.json");

    study.save(&path).unwrap();
    let loaded: Study<f64> = Study::load(&path).unwrap();

    assert_eq!(loaded.direction(), Direction::Minimize);
    assert_eq!(loaded.n_trials(), 0);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn snapshot_version_field_is_present() {
    let study: Study<f64> = Study::new(Direction::Minimize);

    let dir = tempdir();
    let path = dir.join("version.json");
    study.save(&path).unwrap();

    let contents = std::fs::read_to_string(&path).unwrap();
    let snapshot: StudySnapshot<f64> = serde_json::from_str(&contents).unwrap();

    assert_eq!(snapshot.version, 1);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn completed_trial_serde_round_trip() {
    let trial = CompletedTrial::new(42, HashMap::new(), HashMap::new(), HashMap::new(), 2.78);

    let json = serde_json::to_string(&trial).unwrap();
    let deserialized: CompletedTrial<f64> = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.id, 42);
    assert_eq!(deserialized.value, 2.78);
    assert_eq!(deserialized.state, TrialState::Complete);
}

#[test]
fn param_value_serde_round_trip() {
    let values = vec![
        ParamValue::Float(1.23),
        ParamValue::Int(42),
        ParamValue::Categorical(2),
    ];

    for val in &values {
        let json = serde_json::to_string(val).unwrap();
        let deserialized: ParamValue = serde_json::from_str(&json).unwrap();
        assert_eq!(&deserialized, val);
    }
}

#[test]
fn direction_serde_round_trip() {
    let min_json = serde_json::to_string(&Direction::Minimize).unwrap();
    let max_json = serde_json::to_string(&Direction::Maximize).unwrap();

    assert_eq!(
        serde_json::from_str::<Direction>(&min_json).unwrap(),
        Direction::Minimize
    );
    assert_eq!(
        serde_json::from_str::<Direction>(&max_json).unwrap(),
        Direction::Maximize
    );
}

#[test]
fn round_trip_preserves_trial_id_counter() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 1.0);

    study
        .optimize(10, |trial| {
            let v = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(v)
        })
        .unwrap();

    let dir = tempdir();
    let path = dir.join("counter.json");
    study.save(&path).unwrap();

    let loaded: Study<f64> = Study::load(&path).unwrap();

    // Creating a new trial should use an ID >= 10
    let trial = loaded.create_trial();
    assert!(trial.id() >= 10);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn checkpoint_file_created_at_interval() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(-10.0, 10.0).name("x");

    let dir = tempdir();
    let checkpoint = dir.join("checkpoint.json");

    study
        .optimize_with_checkpoint(10, 3, &checkpoint, |trial| {
            let v = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(v * v)
        })
        .unwrap();

    // Checkpoint should exist (written at trials 3, 6, 9)
    assert!(checkpoint.exists(), "checkpoint file was not created");

    // Load it and verify it's valid
    let loaded: Study<f64> = Study::load(&checkpoint).unwrap();
    // Last checkpoint was at trial 9, so it should have 9 trials
    assert_eq!(loaded.n_trials(), 9);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn checkpoint_overwrites_previous() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 1.0);

    let dir = tempdir();
    let checkpoint = dir.join("checkpoint.json");

    study
        .optimize_with_checkpoint(6, 3, &checkpoint, |trial| {
            let v = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(v)
        })
        .unwrap();

    // The checkpoint at trial 6 should overwrite the one from trial 3
    let loaded: Study<f64> = Study::load(&checkpoint).unwrap();
    assert_eq!(loaded.n_trials(), 6);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn resume_from_checkpoint_continues_trial_ids() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(-5.0, 5.0).name("x");

    let dir = tempdir();
    let checkpoint = dir.join("resume.json");

    // Run 10 trials with checkpointing
    study
        .optimize_with_checkpoint(10, 5, &checkpoint, |trial| {
            let v = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(v * v)
        })
        .unwrap();

    // Load and continue
    let loaded: Study<f64> = Study::load(&checkpoint).unwrap();
    assert_eq!(loaded.n_trials(), 10);

    let remaining = 15 - loaded.n_trials();
    loaded
        .optimize(remaining, |trial| {
            let v = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(v * v)
        })
        .unwrap();

    assert_eq!(loaded.n_trials(), 15);

    // Verify no duplicate trial IDs
    let trials = loaded.trials();
    let mut ids: Vec<u64> = trials.iter().map(|t| t.id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 15, "duplicate trial IDs found");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn atomic_write_no_temp_file_left_behind() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 1.0);

    let dir = tempdir();
    let checkpoint = dir.join("atomic.json");

    study
        .optimize_with_checkpoint(3, 3, &checkpoint, |trial| {
            let v = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(v)
        })
        .unwrap();

    // The temp file should have been renamed, not left behind
    let tmp_path = dir.join(".atomic.json.tmp");
    assert!(!tmp_path.exists(), "temp file was not cleaned up");
    assert!(checkpoint.exists(), "checkpoint file was not created");

    std::fs::remove_dir_all(&dir).ok();
}

/// Helper to create a unique temporary directory.
fn tempdir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("optimizer_serde_test_{}_{id}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
