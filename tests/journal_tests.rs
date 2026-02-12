//! Integration tests for the journal storage backend.

use std::collections::HashMap;
use std::sync::Arc;

use std::io::Write;

use optimizer::parameter::{FloatParam, Parameter};
use optimizer::sampler::CompletedTrial;
use optimizer::sampler::random::RandomSampler;
use optimizer::storage::{JournalStorage, Storage};
use optimizer::{Direction, Study};

fn temp_path() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let mut path = std::env::temp_dir();
    path.push(format!(
        "optimizer_journal_test_{}_{}.jsonl",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    path
}

fn sample_trial(id: u64, value: f64) -> CompletedTrial<f64> {
    CompletedTrial::new(id, HashMap::new(), HashMap::new(), HashMap::new(), value)
}

#[test]
fn roundtrip_single_trial() {
    let path = temp_path();
    let storage = JournalStorage::new(&path);

    storage.push(sample_trial(0, 42.0));

    let loaded = storage.trials_arc().read().clone();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].id, 0);
    assert_eq!(loaded[0].value, 42.0);

    // Also verify via a fresh open from disk
    let storage2 = JournalStorage::<f64>::open(&path).unwrap();
    let loaded2 = storage2.trials_arc().read().clone();
    assert_eq!(loaded2.len(), 1);
    assert_eq!(loaded2[0].value, 42.0);

    std::fs::remove_file(&path).ok();
}

#[test]
fn append_multiple_trials() {
    let path = temp_path();
    let storage = JournalStorage::new(&path);

    for i in 0..5 {
        storage.push(sample_trial(i, i as f64));
    }

    // Reload from disk
    let storage2 = JournalStorage::<f64>::open(&path).unwrap();
    let loaded = storage2.trials_arc().read().clone();
    assert_eq!(loaded.len(), 5);
    for (i, trial) in loaded.iter().enumerate() {
        assert_eq!(trial.id, i as u64);
        assert_eq!(trial.value, i as f64);
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn missing_file_returns_empty() {
    let path = temp_path();
    let storage = JournalStorage::<f64>::open(&path).unwrap();

    let loaded = storage.trials_arc().read().clone();
    assert!(loaded.is_empty());
}

#[test]
fn concurrent_writes() {
    let path = temp_path();
    let storage = Arc::new(JournalStorage::new(&path));

    let mut handles = Vec::new();
    for thread_id in 0..4u64 {
        let s = Arc::clone(&storage);
        handles.push(std::thread::spawn(move || {
            for i in 0..25u64 {
                let id = thread_id * 25 + i;
                s.push(sample_trial(id, id as f64));
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    // Reload from disk to verify persistence
    let storage2 = JournalStorage::<f64>::open(&path).unwrap();
    let loaded = storage2.trials_arc().read().clone();
    assert_eq!(loaded.len(), 100);

    // Verify all IDs are present (order may vary)
    let mut ids: Vec<u64> = loaded.iter().map(|t| t.id).collect();
    ids.sort();
    assert_eq!(ids, (0..100).collect::<Vec<_>>());

    std::fs::remove_file(&path).ok();
}

#[test]
fn study_with_journal_integration() {
    let path = temp_path();
    let x = FloatParam::new(-10.0, 10.0);

    // First "process": run some trials
    {
        let study =
            Study::with_journal(Direction::Minimize, RandomSampler::with_seed(1), &path).unwrap();
        study
            .optimize(5, |trial: &mut optimizer::Trial| {
                let val = x.suggest(trial)?;
                Ok::<_, optimizer::Error>(val * val)
            })
            .unwrap();
        assert_eq!(study.n_trials(), 5);
    }

    // Second "process": loads the same file, sees existing trials
    let study2 =
        Study::with_journal(Direction::Minimize, RandomSampler::with_seed(2), &path).unwrap();
    assert_eq!(study2.n_trials(), 5);

    // Continue optimizing
    study2
        .optimize(5, |trial: &mut optimizer::Trial| {
            let val = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(val * val)
        })
        .unwrap();
    assert_eq!(study2.n_trials(), 10);

    // Verify all 10 written to disk
    let storage3 = JournalStorage::<f64>::open(&path).unwrap();
    let loaded = storage3.trials_arc().read().clone();
    assert_eq!(loaded.len(), 10);

    std::fs::remove_file(&path).ok();
}

#[test]
fn ids_are_unique_after_reload() {
    let path = temp_path();

    // First batch
    {
        let study =
            Study::with_journal(Direction::Minimize, RandomSampler::with_seed(1), &path).unwrap();
        study
            .optimize(3, |trial: &mut optimizer::Trial| {
                let _ = FloatParam::new(0.0, 1.0).suggest(trial)?;
                Ok::<_, optimizer::Error>(1.0)
            })
            .unwrap();
    }

    // Second batch — IDs should continue from 3
    let study =
        Study::with_journal(Direction::Minimize, RandomSampler::with_seed(2), &path).unwrap();
    study
        .optimize(3, |trial: &mut optimizer::Trial| {
            let _ = FloatParam::new(0.0, 1.0).suggest(trial)?;
            Ok::<_, optimizer::Error>(1.0)
        })
        .unwrap();

    let all = study.trials();
    let mut ids: Vec<u64> = all.iter().map(|t| t.id).collect();
    ids.sort();
    // All 6 IDs should be unique
    ids.dedup();
    assert_eq!(ids.len(), 6);

    std::fs::remove_file(&path).ok();
}

#[test]
fn pruned_trials_are_stored() {
    let path = temp_path();
    let study =
        Study::with_journal(Direction::Minimize, RandomSampler::with_seed(1), &path).unwrap();

    // Complete one, prune one
    let x = FloatParam::new(0.0, 1.0);
    study
        .optimize(3, |trial: &mut optimizer::Trial| {
            let _ = x.suggest(trial)?;
            if trial.id() == 1 {
                Err(optimizer::TrialPruned)?;
            }
            Ok::<_, optimizer::Error>(1.0)
        })
        .unwrap();

    let storage2 = JournalStorage::<f64>::open(&path).unwrap();
    let loaded = storage2.trials_arc().read().clone();
    assert_eq!(loaded.len(), 3);
    assert!(
        loaded
            .iter()
            .any(|t| t.state == optimizer::TrialState::Pruned)
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn rejects_non_finite_values_in_journal() {
    // serde_json rejects 1e999 ("number out of range"), so non-finite
    // floats cannot sneak in through standard JSON.  Verify the overall
    // loading path catches the error regardless of which layer rejects it.
    let path = temp_path();
    std::fs::write(
        &path,
        r#"{"id":0,"params":{},"distributions":{"0":{"Float":{"low":0.0,"high":1e999,"log_scale":false,"step":null}}},"param_labels":{},"value":1.0,"intermediate_values":[],"state":"Complete","user_attrs":{},"constraints":[]}"#,
    )
    .unwrap();

    assert!(JournalStorage::<f64>::open(&path).is_err());
    std::fs::remove_file(&path).ok();
}

#[test]
fn validate_rejects_non_finite_distribution_bound() {
    use optimizer::distribution::{Distribution, FloatDistribution};

    let pid = FloatParam::new(0.0, 1.0).id();
    let mut trial = sample_trial(0, 1.0);
    trial.distributions.insert(
        pid,
        Distribution::Float(FloatDistribution {
            low: 0.0,
            high: f64::INFINITY,
            log_scale: false,
            step: None,
        }),
    );
    let err = trial.validate().unwrap_err();
    assert!(err.contains("non-finite"), "unexpected: {err}");
}

#[test]
fn validate_rejects_nan_constraint() {
    let mut trial = sample_trial(0, 1.0);
    trial.constraints.push(f64::NAN);
    let err = trial.validate().unwrap_err();
    assert!(err.contains("non-finite"), "unexpected: {err}");
}

#[test]
fn validate_rejects_non_finite_param_value() {
    use optimizer::param::ParamValue;

    let pid = FloatParam::new(0.0, 1.0).id();
    let mut trial = sample_trial(0, 1.0);
    trial
        .params
        .insert(pid, ParamValue::Float(f64::NEG_INFINITY));
    let err = trial.validate().unwrap_err();
    assert!(err.contains("non-finite"), "unexpected: {err}");
}

#[test]
fn validate_rejects_nan_intermediate_value() {
    let mut trial = sample_trial(0, 1.0);
    trial.intermediate_values.push((0, f64::NAN));
    let err = trial.validate().unwrap_err();
    assert!(err.contains("non-finite"), "unexpected: {err}");
}

#[test]
fn validate_accepts_valid_trial() {
    use optimizer::distribution::{Distribution, FloatDistribution};
    use optimizer::param::ParamValue;

    let pid = FloatParam::new(0.0, 1.0).id();
    let mut trial = sample_trial(0, 1.0);
    trial.params.insert(pid, ParamValue::Float(0.5));
    trial.distributions.insert(
        pid,
        Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        }),
    );
    trial.constraints.push(-1.0);
    trial.intermediate_values.push((0, 0.5));
    assert!(trial.validate().is_ok());
}

#[test]
fn accepts_valid_journal_with_distributions() {
    let path = temp_path();
    std::fs::write(
        &path,
        r#"{"id":0,"params":{"0":{"Float":0.5}},"distributions":{"0":{"Float":{"low":0.0,"high":1.0,"log_scale":false,"step":null}}},"param_labels":{},"value":0.25,"intermediate_values":[],"state":"Complete","user_attrs":{},"constraints":[-1.0]}"#,
    )
    .unwrap();

    let storage = JournalStorage::<f64>::open(&path).unwrap();
    let loaded = storage.trials_arc().read().clone();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].value, 0.25);

    std::fs::remove_file(&path).ok();
}

#[test]
fn refresh_skips_own_writes() {
    let path = temp_path();
    let storage = JournalStorage::new(&path);

    for i in 0..5 {
        storage.push(sample_trial(i, i as f64));
        // Our own push advanced the offset, so refresh should find nothing new.
        assert!(!storage.refresh(), "refresh returned true after push {i}");
    }

    assert_eq!(storage.trials_arc().read().len(), 5);
    std::fs::remove_file(&path).ok();
}

#[test]
fn refresh_picks_up_external_writes() {
    let path = temp_path();
    let storage = JournalStorage::new(&path);

    // Push 3 trials through the storage (advances offset).
    for i in 0..3 {
        storage.push(sample_trial(i, i as f64));
    }
    assert_eq!(storage.trials_arc().read().len(), 3);

    // Simulate an external process appending 2 more lines directly.
    {
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        for i in 3..5u64 {
            let trial = sample_trial(i, i as f64);
            let line = serde_json::to_string(&trial).unwrap();
            writeln!(file, "{line}").unwrap();
        }
        file.sync_all().unwrap();
    }

    // refresh() should pick up the 2 external trials.
    assert!(storage.refresh(), "refresh should detect external writes");
    assert_eq!(storage.trials_arc().read().len(), 5);

    // A second refresh should be a no-op.
    assert!(
        !storage.refresh(),
        "second refresh should return false (no new data)"
    );
    assert_eq!(storage.trials_arc().read().len(), 5);

    std::fs::remove_file(&path).ok();
}
