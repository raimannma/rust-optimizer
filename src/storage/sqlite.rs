//! `SQLite`-backed storage backend for multi-process optimization.

use core::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use rusqlite::Connection;
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::{MemoryStorage, Storage};
use crate::sampler::CompletedTrial;

/// A storage backend that persists completed trials in a `SQLite` database.
///
/// Uses WAL mode for concurrent readers and a single writer, making it
/// suitable for single-machine multi-process optimization.  Compared to
/// [`JournalStorage`](super::JournalStorage), `SQLite` provides proper
/// ACID transactions and better concurrent access.
///
/// The type parameter `V` is the objective value type (typically `f64`).
/// It must be serializable so that trials can be written to disk.
///
/// # Examples
///
/// ```no_run
/// use optimizer::storage::SqliteStorage;
///
/// let storage: SqliteStorage<f64> = SqliteStorage::new("trials.db").unwrap();
/// ```
pub struct SqliteStorage<V = f64> {
    memory: MemoryStorage<V>,
    conn: Mutex<Connection>,
    _marker: PhantomData<V>,
}

impl<V: Serialize + DeserializeOwned + Send + Sync> SqliteStorage<V> {
    /// Creates a new `SQLite` storage at the given path.
    ///
    /// The database file is created if it does not exist.  Any trials
    /// already stored in the database are loaded into memory.
    ///
    /// WAL mode is enabled automatically for better concurrency.
    ///
    /// # Errors
    ///
    /// Returns a [`Storage`](crate::Error::Storage) error if the
    /// database cannot be opened or the schema cannot be created.
    pub fn new(path: impl AsRef<Path>) -> crate::Result<Self> {
        let conn = Connection::open(path).map_err(|e| crate::Error::Storage(e.to_string()))?;

        // WAL mode: concurrent readers, single writer.
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| crate::Error::Storage(e.to_string()))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS trials (
                trial_id INTEGER PRIMARY KEY,
                data     TEXT NOT NULL
            );",
        )
        .map_err(|e| crate::Error::Storage(e.to_string()))?;

        let trials = load_all(&conn)?;

        Ok(Self {
            memory: MemoryStorage::with_trials(trials),
            conn: Mutex::new(conn),
            _marker: PhantomData,
        })
    }

    /// Persist a single trial into the database.
    fn write_trial(&self, trial: &CompletedTrial<V>) -> crate::Result<()> {
        let data =
            serde_json::to_string(trial).map_err(|e| crate::Error::Storage(e.to_string()))?;

        let conn = self.conn.lock();
        conn.execute(
            "INSERT OR REPLACE INTO trials (trial_id, data) VALUES (?1, ?2)",
            rusqlite::params![i64::try_from(trial.id).unwrap_or(i64::MAX), data],
        )
        .map_err(|e| crate::Error::Storage(e.to_string()))?;

        Ok(())
    }
}

impl<V: Serialize + DeserializeOwned + Send + Sync> Storage<V> for SqliteStorage<V> {
    fn push(&self, trial: CompletedTrial<V>) {
        // Best-effort persist; the trial stays in memory regardless.
        let _ = self.write_trial(&trial);
        self.memory.push(trial);
    }

    fn trials_arc(&self) -> &Arc<RwLock<Vec<CompletedTrial<V>>>> {
        self.memory.trials_arc()
    }

    fn refresh(&self) -> bool {
        let conn = self.conn.lock();
        let Ok(loaded) = load_all::<V>(&conn) else {
            return false;
        };
        let mut guard = self.memory.trials_arc().write();
        if loaded.len() > guard.len() {
            *guard = loaded;
            true
        } else {
            false
        }
    }
}

/// Load every trial from the database, ordered by id.
fn load_all<V: DeserializeOwned>(conn: &Connection) -> crate::Result<Vec<CompletedTrial<V>>> {
    let mut stmt = conn
        .prepare("SELECT data FROM trials ORDER BY trial_id")
        .map_err(|e| crate::Error::Storage(e.to_string()))?;

    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| crate::Error::Storage(e.to_string()))?;

    let mut trials = Vec::new();
    for row in rows {
        let data = row.map_err(|e| crate::Error::Storage(e.to_string()))?;
        let trial: CompletedTrial<V> =
            serde_json::from_str(&data).map_err(|e| crate::Error::Storage(e.to_string()))?;
        trials.push(trial);
    }

    Ok(trials)
}
