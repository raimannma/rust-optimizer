//! JSONL-based journal storage backend.

use core::marker::PhantomData;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use fs2::FileExt;
use parking_lot::{Mutex, RwLock};
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::{MemoryStorage, Storage};
use crate::sampler::CompletedTrial;

/// A storage backend that appends completed trials as JSON lines to a file.
///
/// Trials are kept in memory for fast read access and simultaneously
/// persisted to a JSONL file.  Multiple processes can safely share
/// the same file: writes use an exclusive file lock, reads use a
/// shared file lock.
///
/// The type parameter `V` is the objective value type (typically `f64`).
/// It must be serializable so that trials can be written to disk.
///
/// # Examples
///
/// ```no_run
/// use optimizer::storage::JournalStorage;
///
/// let storage: JournalStorage<f64> = JournalStorage::new("trials.jsonl");
/// ```
pub struct JournalStorage<V = f64> {
    memory: MemoryStorage<V>,
    path: PathBuf,
    /// Serialise in-process writes so we only hold the file lock briefly.
    write_lock: Mutex<()>,
    _marker: PhantomData<V>,
}

impl<V: Serialize + DeserializeOwned + Send + Sync> JournalStorage<V> {
    /// Creates a new journal storage that writes to the given path.
    ///
    /// The file does not need to exist yet — it will be created on the
    /// first write.  Existing trials in the file are **not** loaded
    /// until [`refresh`](Storage::refresh) is called (which happens
    /// automatically at the start of each trial via the [`Study`](crate::Study)).
    ///
    /// To pre-load existing trials, use [`JournalStorage::open`].
    #[must_use]
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            memory: MemoryStorage::new(),
            path: path.as_ref().to_path_buf(),
            write_lock: Mutex::new(()),
            _marker: PhantomData,
        }
    }

    /// Opens an existing journal file and loads all stored trials.
    ///
    /// If the file does not exist, returns an empty storage (no error).
    ///
    /// # Errors
    ///
    /// Returns a [`Storage`](crate::Error::Storage) error if the file
    /// exists but cannot be read or parsed.
    pub fn open(path: impl AsRef<Path>) -> crate::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let trials = load_trials_from_file(&path)?;
        Ok(Self {
            memory: MemoryStorage::with_trials(trials),
            path,
            write_lock: Mutex::new(()),
            _marker: PhantomData,
        })
    }

    /// Append a single trial to the JSONL file (best-effort).
    fn write_to_file(&self, trial: &CompletedTrial<V>) -> crate::Result<()> {
        let _guard = self.write_lock.lock();

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| crate::Error::Storage(e.to_string()))?;

        file.lock_exclusive()
            .map_err(|e| crate::Error::Storage(e.to_string()))?;

        let line =
            serde_json::to_string(trial).map_err(|e| crate::Error::Storage(e.to_string()))?;

        writeln!(file, "{line}").map_err(|e| crate::Error::Storage(e.to_string()))?;
        file.flush()
            .map_err(|e| crate::Error::Storage(e.to_string()))?;

        file.unlock()
            .map_err(|e| crate::Error::Storage(e.to_string()))?;

        Ok(())
    }
}

impl<V: Serialize + DeserializeOwned + Send + Sync> Storage<V> for JournalStorage<V> {
    fn push(&self, trial: CompletedTrial<V>) {
        // Best-effort persist; the trial stays in memory regardless.
        let _ = self.write_to_file(&trial);
        self.memory.push(trial);
    }

    fn trials_arc(&self) -> &Arc<RwLock<Vec<CompletedTrial<V>>>> {
        self.memory.trials_arc()
    }

    fn next_trial_id(&self) -> u64 {
        self.memory.next_trial_id()
    }

    fn refresh(&self) -> bool {
        let Ok(loaded) = load_trials_from_file::<V>(&self.path) else {
            return false;
        };
        let mut guard = self.memory.trials_arc().write();
        if loaded.len() > guard.len() {
            if let Some(max_id) = loaded.iter().map(|t| t.id).max() {
                self.memory.bump_next_id(max_id + 1);
            }
            *guard = loaded;
            true
        } else {
            false
        }
    }
}

/// Read all trials from a JSONL file.  Returns an empty vec if the
/// file does not exist.
fn load_trials_from_file<V: DeserializeOwned>(
    path: &Path,
) -> crate::Result<Vec<CompletedTrial<V>>> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(crate::Error::Storage(e.to_string())),
    };

    file.lock_shared()
        .map_err(|e| crate::Error::Storage(e.to_string()))?;

    let reader = BufReader::new(&file);
    let mut trials = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| crate::Error::Storage(e.to_string()))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let trial: CompletedTrial<V> =
            serde_json::from_str(line).map_err(|e| crate::Error::Storage(e.to_string()))?;
        trials.push(trial);
    }

    file.unlock()
        .map_err(|e| crate::Error::Storage(e.to_string()))?;

    Ok(trials)
}
