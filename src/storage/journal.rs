//! JSONL-based journal storage backend.
//!
//! [`JournalStorage`] persists completed trials as one JSON object per
//! line ([JSONL / JSON Lines](https://jsonlines.org/)) while keeping a
//! full copy in memory for fast read access.
//!
//! # File format
//!
//! Each line is a self-contained JSON serialization of a
//! [`CompletedTrial<V>`](crate::sampler::CompletedTrial).  The file
//! is append-only — no existing lines are ever modified or deleted.
//!
//! ```text
//! {"id":0,"params":{...},"value":1.23,"state":"Completed",...}
//! {"id":1,"params":{...},"value":0.87,"state":"Completed",...}
//! ```
//!
//! # File locking
//!
//! Concurrent access is coordinated with `fs2` file locks:
//!
//! - **Writes** acquire an *exclusive* lock so only one process
//!   appends at a time.
//! - **Reads** ([`refresh`](super::Storage::refresh)) acquire a
//!   *shared* lock so readers never see a partially written line.
//!
//! This makes it safe for multiple processes to share the same JSONL
//! file — for example, distributed workers each running their own
//! [`Study`](crate::Study) with a `JournalStorage` pointing to a
//! shared path.
//!
//! # Resuming a study
//!
//! Use [`JournalStorage::open`] to reload previously persisted trials
//! and continue optimization from where you left off:
//!
//! ```no_run
//! use optimizer::prelude::*;
//! use optimizer::storage::JournalStorage;
//!
//! // First run — creates the file.
//! let storage = JournalStorage::<f64>::new("trials.jsonl");
//! let mut study = Study::builder().minimize().storage(storage).build();
//! study
//!     .optimize(50, |trial: &mut optimizer::Trial| {
//!         let x = FloatParam::new(-5.0, 5.0).suggest(trial)?;
//!         Ok::<_, optimizer::Error>(x * x)
//!     })
//!     .unwrap();
//!
//! // Later run — reloads previous 50 trials, then adds 50 more.
//! let storage = JournalStorage::<f64>::open("trials.jsonl").unwrap();
//! let mut study = Study::builder().minimize().storage(storage).build();
//! study
//!     .optimize(50, |trial: &mut optimizer::Trial| {
//!         let x = FloatParam::new(-5.0, 5.0).suggest(trial)?;
//!         Ok::<_, optimizer::Error>(x * x)
//!     })
//!     .unwrap();
//! ```
//!
//! # When to use
//!
//! - **Persistence** — survive process crashes or intentional restarts.
//! - **Multi-process** — several workers collaborating on a single study.
//! - **Inspection** — `cat trials.jsonl | jq .` for quick debugging.
//!
//! For pure in-memory usage without disk I/O, use
//! [`MemoryStorage`](super::MemoryStorage) instead (the default).
//!
//! # Security considerations
//!
//! Both [`JournalStorage::open`] and [`refresh`](super::Storage::refresh)
//! read the entire JSONL file into memory.  A very large file will
//! consume memory proportional to its size, which could lead to
//! out-of-memory conditions.
//!
//! If your application accepts externally-provided file paths, consider:
//!
//! - Checking the file size before calling `open`.
//! - Validating or sanitizing untrusted JSONL content.
//! - Imposing an upper bound on the number of trials or file size you
//!   are willing to load.

use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, Ordering};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read as _, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use fs2::FileExt;
use parking_lot::{Mutex, RwLock};
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::{MemoryStorage, Storage};
use crate::sampler::CompletedTrial;

/// Append-only JSONL storage backend with file locking.
///
/// Trials are kept in memory (via an inner [`MemoryStorage`]) for fast
/// read access and simultaneously appended to a JSONL file on disk.
/// Multiple processes can safely share the same file thanks to
/// `fs2` file locks — writes use an exclusive lock, reads use a
/// shared lock.
///
/// The type parameter `V` is the objective value type (typically
/// `f64`).  It must implement [`Serialize`](serde::Serialize) and
/// [`DeserializeOwned`](serde::de::DeserializeOwned) so trials can be
/// written to and read from disk.
///
/// See the [`storage`](super) module docs for file format details
/// and a resumption example.
///
/// # Example
///
/// ```no_run
/// use optimizer::prelude::*;
/// use optimizer::storage::JournalStorage;
///
/// let storage = JournalStorage::<f64>::new("trials.jsonl");
/// let mut study = Study::builder().minimize().storage(storage).build();
/// ```
///
/// # Security considerations
///
/// File contents are loaded into memory in full; see the
/// module-level docs for details and mitigations.
pub struct JournalStorage<V = f64> {
    memory: MemoryStorage<V>,
    path: PathBuf,
    /// Serialise in-process writes and refreshes so they don't race.
    io_lock: Mutex<()>,
    /// Byte offset of last-read position for incremental refresh.
    file_offset: AtomicU64,
    _marker: PhantomData<V>,
}

impl<V: Serialize + DeserializeOwned + Send + Sync> JournalStorage<V> {
    /// Create a new journal storage that writes to the given path.
    ///
    /// The file does not need to exist yet — it will be created on the
    /// first write.  Existing trials in the file are **not** loaded
    /// until [`refresh`](Storage::refresh) is called (which happens
    /// automatically at the start of each trial via the [`Study`](crate::Study)).
    ///
    /// To pre-load existing trials at construction time, use
    /// [`JournalStorage::open`] instead.
    #[must_use]
    pub fn new(path: impl AsRef<Path>) -> Self {
        let path = path
            .as_ref()
            .canonicalize()
            .unwrap_or_else(|_| path.as_ref().to_path_buf());
        Self {
            memory: MemoryStorage::new(),
            path,
            io_lock: Mutex::new(()),
            file_offset: AtomicU64::new(0),
            _marker: PhantomData,
        }
    }

    /// Open an existing journal file and load all stored trials.
    ///
    /// If the file does not exist, return an empty storage (no error).
    /// This is the primary way to **resume** a study after a restart.
    ///
    /// # Errors
    ///
    /// Return a [`Storage`](crate::Error::Storage) error if the file
    /// exists but cannot be read or parsed.
    pub fn open(path: impl AsRef<Path>) -> crate::Result<Self> {
        let path = path
            .as_ref()
            .canonicalize()
            .unwrap_or_else(|_| path.as_ref().to_path_buf());
        let (trials, offset) = load_trials_from_file(&path)?;
        Ok(Self {
            memory: MemoryStorage::with_trials(trials),
            path,
            io_lock: Mutex::new(()),
            file_offset: AtomicU64::new(offset),
            _marker: PhantomData,
        })
    }

    /// Append a single trial to the JSONL file (best-effort).
    ///
    /// Does **not** advance `file_offset` — that is left to `refresh`
    /// so that externally-written data between the old offset and our
    /// write is never skipped.
    fn write_to_file(&self, trial: &CompletedTrial<V>) -> crate::Result<()> {
        let _guard = self.io_lock.lock();

        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&self.path)
            .map_err(|e| crate::Error::Storage(e.to_string()))?;

        file.lock_exclusive()
            .map_err(|e| crate::Error::Storage(e.to_string()))?;

        file.seek(SeekFrom::End(0))
            .map_err(|e| crate::Error::Storage(e.to_string()))?;

        let line =
            serde_json::to_string(trial).map_err(|e| crate::Error::Storage(e.to_string()))?;

        writeln!(file, "{line}").map_err(|e| crate::Error::Storage(e.to_string()))?;
        file.sync_data()
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

    fn peek_next_trial_id(&self) -> u64 {
        self.memory.peek_next_trial_id()
    }

    fn refresh(&self) -> bool {
        let _guard = self.io_lock.lock();

        let Ok(file) = File::open(&self.path) else {
            return false;
        };

        if file.lock_shared().is_err() {
            return false;
        }

        let offset = self.file_offset.load(Ordering::SeqCst);

        let file_size = if let Ok(m) = file.metadata() {
            m.len()
        } else {
            let _ = file.unlock();
            return false;
        };

        if file_size <= offset {
            let _ = file.unlock();
            return false;
        }

        let mut buf = String::new();
        let mut handle = &file;
        if handle.seek(SeekFrom::Start(offset)).is_err() {
            let _ = file.unlock();
            return false;
        }
        if handle.read_to_string(&mut buf).is_err() {
            let _ = file.unlock();
            return false;
        }

        let _ = file.unlock();

        let bytes_read = buf.len() as u64;
        let new_offset = offset + bytes_read;
        let mut new_trials = Vec::new();

        for line in buf.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let trial: CompletedTrial<V> = match serde_json::from_str(line) {
                Ok(t) => t,
                Err(_) => return false,
            };
            if trial.validate().is_err() {
                return false;
            }
            new_trials.push(trial);
        }

        if new_trials.is_empty() {
            self.file_offset.fetch_max(new_offset, Ordering::SeqCst);
            return false;
        }

        let mut mem_guard = self.memory.trials_arc().write();

        // Deduplicate: only add trials whose IDs are not already in memory.
        let existing_ids: std::collections::HashSet<u64> = mem_guard.iter().map(|t| t.id).collect();
        new_trials.retain(|t| !existing_ids.contains(&t.id));

        if let Some(max_id) = new_trials.iter().map(|t| t.id).max() {
            self.memory.bump_next_id(max_id + 1);
        }
        let added = !new_trials.is_empty();
        mem_guard.extend(new_trials);
        self.file_offset.fetch_max(new_offset, Ordering::SeqCst);
        added
    }
}

/// Read all trials from a JSONL file.  Returns an empty vec (and
/// offset 0) if the file does not exist.  The returned `u64` is the
/// file size at the time of reading, suitable for initialising the
/// incremental-refresh offset.
fn load_trials_from_file<V: DeserializeOwned>(
    path: &Path,
) -> crate::Result<(Vec<CompletedTrial<V>>, u64)> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
        Err(e) => return Err(crate::Error::Storage(e.to_string())),
    };

    file.lock_shared()
        .map_err(|e| crate::Error::Storage(e.to_string()))?;

    let file_size = file
        .metadata()
        .map_err(|e| crate::Error::Storage(e.to_string()))?
        .len();

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
        trial.validate().map_err(crate::Error::Storage)?;
        trials.push(trial);
    }

    file.unlock()
        .map_err(|e| crate::Error::Storage(e.to_string()))?;

    Ok((trials, file_size))
}
