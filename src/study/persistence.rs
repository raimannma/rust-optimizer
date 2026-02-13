#[cfg(feature = "serde")]
use std::collections::HashMap;

#[cfg(feature = "serde")]
use crate::sampler::CompletedTrial;
#[cfg(feature = "serde")]
use crate::types::Direction;

#[cfg(feature = "serde")]
use super::Study;

/// A serializable snapshot of a study's state.
///
/// Since [`Study`] contains non-serializable fields (samplers, atomics, etc.),
/// this struct captures the essential state needed to save and restore a study.
///
/// # Schema versioning
///
/// The `version` field enables future schema evolution without breaking existing files.
/// The current version is `1`.
///
/// # Sampler state
///
/// Sampler state is **not** included in the snapshot. After loading, the study
/// uses a default `RandomSampler`. Call [`Study::set_sampler`] to restore
/// the desired sampler configuration.
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct StudySnapshot<V> {
    /// Schema version for forward compatibility.
    pub version: u32,
    /// The optimization direction.
    pub direction: Direction,
    /// All completed (and pruned) trials.
    pub trials: Vec<CompletedTrial<V>>,
    /// The next trial ID to assign.
    pub next_trial_id: u64,
    /// Optional metadata (creation timestamp, sampler description, etc.).
    pub metadata: HashMap<String, String>,
}

#[cfg(feature = "serde")]
impl<V: PartialOrd + Clone + serde::Serialize> Study<V> {
    /// Save the study state to a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be created or written.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        let trials = self.trials();
        let next_trial_id = self.storage.peek_next_trial_id();
        let snapshot = StudySnapshot {
            version: 1,
            direction: self.direction,
            trials,
            next_trial_id,
            metadata: HashMap::new(),
        };

        // Atomic write: write to a temp file in the same directory, then rename.
        // This prevents corrupt files if the process crashes mid-write.
        let parent = path.parent().unwrap_or(std::path::Path::new("."));
        let tmp_path = parent.join(format!(
            ".{}.tmp",
            path.file_name().unwrap_or_default().to_string_lossy()
        ));
        let file = std::fs::File::create(&tmp_path)?;
        serde_json::to_writer_pretty(file, &snapshot).map_err(std::io::Error::other)?;
        std::fs::rename(&tmp_path, path)
    }
}

#[cfg(feature = "serde")]
impl<V: PartialOrd + Send + Sync + Clone + serde::de::DeserializeOwned + 'static> Study<V> {
    /// Load a study from a JSON file.
    ///
    /// The loaded study uses a `RandomSampler` by default. Call
    /// [`set_sampler()`](Self::set_sampler) to restore the original sampler
    /// configuration.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be read or parsed.
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        use crate::sampler::random::RandomSampler;

        let file = std::fs::File::open(path)?;
        let snapshot: StudySnapshot<V> = serde_json::from_reader(file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let storage = crate::storage::MemoryStorage::with_trials(snapshot.trials);
        Ok(Self::with_sampler_and_storage(
            snapshot.direction,
            RandomSampler::new(),
            storage,
        ))
    }
}

#[cfg(feature = "journal")]
impl<V> Study<V>
where
    V: PartialOrd + Send + Sync + serde::Serialize + serde::de::DeserializeOwned + 'static,
{
    /// Create a study backed by a JSONL journal file.
    ///
    /// Any existing trials in the file are loaded into memory and the
    /// trial ID counter is set to one past the highest stored ID. New
    /// trials are written through to the file on completion.
    ///
    /// # Arguments
    ///
    /// * `direction` - Whether to minimize or maximize the objective function.
    /// * `sampler` - The sampler to use for parameter sampling.
    /// * `path` - Path to the JSONL journal file (created if absent).
    ///
    /// # Errors
    ///
    /// Returns a [`Storage`](crate::Error::Storage) error if loading fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use optimizer::sampler::tpe::TpeSampler;
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> =
    ///     Study::with_journal(Direction::Minimize, TpeSampler::new(), "trials.jsonl").unwrap();
    /// ```
    pub fn with_journal(
        direction: Direction,
        sampler: impl crate::sampler::Sampler + 'static,
        path: impl AsRef<std::path::Path>,
    ) -> crate::Result<Self> {
        let storage = crate::storage::JournalStorage::<V>::open(path)?;
        Ok(Self::with_sampler_and_storage(direction, sampler, storage))
    }
}
