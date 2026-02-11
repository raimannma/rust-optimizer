//! Integration tests for the SQLite storage backend.

use std::collections::HashMap;
use std::sync::Arc;

use optimizer::parameter::{FloatParam, Parameter};
use optimizer::sampler::CompletedTrial;
use optimizer::sampler::random::RandomSampler;
use optimizer::storage::{SqliteStorage, Storage};
use optimizer::{Direction, Study};

fn temp_path() -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "optimizer_sqlite_test_{}.db",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    path
}

fn sample_trial(id: u64, value: f64) -> CompletedTrial<f64> {
    CompletedTrial::new(id, HashMap::new(), HashMap::new(), HashMap::new(), value)
}

#[test]
fn roundtrip_single_trial() {
    let path = temp_path();
    let storage = SqliteStorage::new(&path).unwrap();

    storage.push(sample_trial(0, 42.0));

    let loaded = storage.trials_arc().read().clone();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].id, 0);
    assert_eq!(loaded[0].value, 42.0);

    // Verify via a fresh open from disk
    let storage2 = SqliteStorage::<f64>::new(&path).unwrap();
    let loaded2 = storage2.trials_arc().read().clone();
    assert_eq!(loaded2.len(), 1);
    assert_eq!(loaded2[0].value, 42.0);

    std::fs::remove_file(&path).ok();
}

#[test]
fn append_multiple_trials() {
    let path = temp_path();
    let storage = SqliteStorage::new(&path).unwrap();

    for i in 0..5 {
        storage.push(sample_trial(i, i as f64));
    }

    // Reload from disk
    let storage2 = SqliteStorage::<f64>::new(&path).unwrap();
    let loaded = storage2.trials_arc().read().clone();
    assert_eq!(loaded.len(), 5);
    for (i, trial) in loaded.iter().enumerate() {
        assert_eq!(trial.id, i as u64);
        assert_eq!(trial.value, i as f64);
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn concurrent_writes() {
    let path = temp_path();
    let storage = Arc::new(SqliteStorage::new(&path).unwrap());

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
    let storage2 = SqliteStorage::<f64>::new(&path).unwrap();
    let loaded = storage2.trials_arc().read().clone();
    assert_eq!(loaded.len(), 100);

    // Verify all IDs are present (order may vary)
    let mut ids: Vec<u64> = loaded.iter().map(|t| t.id).collect();
    ids.sort();
    assert_eq!(ids, (0..100).collect::<Vec<_>>());

    std::fs::remove_file(&path).ok();
}

#[test]
fn refresh_picks_up_external_writes() {
    let path = temp_path();
    let storage1 = SqliteStorage::new(&path).unwrap();
    let storage2 = SqliteStorage::<f64>::new(&path).unwrap();

    // Write via storage1
    storage1.push(sample_trial(0, 1.0));
    storage1.push(sample_trial(1, 2.0));

    // storage2 doesn't see them yet in memory
    assert_eq!(storage2.trials_arc().read().len(), 0);

    // After refresh, storage2 picks them up
    assert!(storage2.refresh());
    assert_eq!(storage2.trials_arc().read().len(), 2);

    // No-op refresh returns false
    assert!(!storage2.refresh());

    std::fs::remove_file(&path).ok();
}

#[test]
fn study_with_sqlite_integration() {
    let path = temp_path();
    let x = FloatParam::new(-10.0, 10.0);

    // First "process": run some trials
    {
        let study =
            Study::with_sqlite(Direction::Minimize, RandomSampler::with_seed(1), &path).unwrap();
        study
            .optimize(5, |trial| {
                let val = x.suggest(trial)?;
                Ok::<_, optimizer::Error>(val * val)
            })
            .unwrap();
        assert_eq!(study.n_trials(), 5);
    }

    // Second "process": loads the same file, sees existing trials
    let study2 =
        Study::with_sqlite(Direction::Minimize, RandomSampler::with_seed(2), &path).unwrap();
    assert_eq!(study2.n_trials(), 5);

    // Continue optimizing
    study2
        .optimize(5, |trial| {
            let val = x.suggest(trial)?;
            Ok::<_, optimizer::Error>(val * val)
        })
        .unwrap();
    assert_eq!(study2.n_trials(), 10);

    // Verify all 10 written to disk
    let storage3 = SqliteStorage::<f64>::new(&path).unwrap();
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
            Study::with_sqlite(Direction::Minimize, RandomSampler::with_seed(1), &path).unwrap();
        study
            .optimize(3, |trial| {
                let _ = FloatParam::new(0.0, 1.0).suggest(trial)?;
                Ok::<_, optimizer::Error>(1.0)
            })
            .unwrap();
    }

    // Second batch — IDs should continue from 3
    let study =
        Study::with_sqlite(Direction::Minimize, RandomSampler::with_seed(2), &path).unwrap();
    study
        .optimize(3, |trial| {
            let _ = FloatParam::new(0.0, 1.0).suggest(trial)?;
            Ok::<_, optimizer::Error>(1.0)
        })
        .unwrap();

    let all = study.trials();
    let mut ids: Vec<u64> = all.iter().map(|t| t.id).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 6);

    std::fs::remove_file(&path).ok();
}

#[test]
fn pruned_trials_are_stored() {
    let path = temp_path();
    let study =
        Study::with_sqlite(Direction::Minimize, RandomSampler::with_seed(1), &path).unwrap();

    let x = FloatParam::new(0.0, 1.0);
    study
        .optimize(3, |trial| {
            let _ = x.suggest(trial)?;
            if trial.id() == 1 {
                Err(optimizer::TrialPruned)?;
            }
            Ok::<_, optimizer::Error>(1.0)
        })
        .unwrap();

    let storage2 = SqliteStorage::<f64>::new(&path).unwrap();
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
fn handles_many_trials() {
    let path = temp_path();
    let storage = SqliteStorage::new(&path).unwrap();

    for i in 0..1000 {
        storage.push(sample_trial(i, i as f64));
    }

    let storage2 = SqliteStorage::<f64>::new(&path).unwrap();
    let loaded = storage2.trials_arc().read().clone();
    assert_eq!(loaded.len(), 1000);

    std::fs::remove_file(&path).ok();
}
