use core::fmt;

use crate::parameter::ParamId;
use crate::types::{Direction, TrialState};

use super::Study;

impl<V> Study<V>
where
    V: PartialOrd + Clone + fmt::Display,
{
    /// Write completed trials to a writer in CSV format.
    ///
    /// Columns: `trial_id`, `value`, `state`, then one column per unique
    /// parameter label, then one column per unique user-attribute key.
    ///
    /// Parameters without labels use a generated name (`param_<id>`).
    /// Pruned trials have an empty `value` cell.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let x = FloatParam::new(0.0, 10.0).name("x");
    ///
    /// let mut trial = study.create_trial();
    /// let _ = x.suggest(&mut trial);
    /// study.complete_trial(trial, 0.42);
    ///
    /// let mut buf = Vec::new();
    /// study.to_csv(&mut buf).unwrap();
    /// let csv = String::from_utf8(buf).unwrap();
    /// assert!(csv.contains("trial_id"));
    /// ```
    pub fn to_csv(&self, mut writer: impl std::io::Write) -> std::io::Result<()> {
        use std::collections::BTreeMap;

        let trials = self.storage.trials_arc().read();

        // Collect all unique parameter labels (sorted for deterministic column order).
        let mut param_columns: BTreeMap<ParamId, String> = BTreeMap::new();
        for trial in trials.iter() {
            for &id in trial.params.keys() {
                param_columns.entry(id).or_insert_with(|| {
                    trial
                        .param_labels
                        .get(&id)
                        .cloned()
                        .unwrap_or_else(|| id.to_string())
                });
            }
        }
        // Fill in labels from other trials that might have better labels.
        for trial in trials.iter() {
            for (&id, label) in &trial.param_labels {
                param_columns.entry(id).or_insert_with(|| label.clone());
            }
        }

        // Collect all unique attribute keys (sorted).
        let mut attr_keys: Vec<String> = Vec::new();
        for trial in trials.iter() {
            for key in trial.user_attrs.keys() {
                if !attr_keys.contains(key) {
                    attr_keys.push(key.clone());
                }
            }
        }
        attr_keys.sort();

        let param_ids: Vec<ParamId> = param_columns.keys().copied().collect();

        // Write header.
        write!(writer, "trial_id,value,state")?;
        for id in &param_ids {
            write!(writer, ",{}", csv_escape(&param_columns[id]))?;
        }
        for key in &attr_keys {
            write!(writer, ",{}", csv_escape(key))?;
        }
        writeln!(writer)?;

        // Write one row per trial.
        for trial in trials.iter() {
            write!(writer, "{}", trial.id)?;

            // Value: empty for non-complete trials.
            if trial.state == TrialState::Complete {
                write!(writer, ",{}", trial.value)?;
            } else {
                write!(writer, ",")?;
            }

            write!(
                writer,
                ",{}",
                match trial.state {
                    TrialState::Complete => "Complete",
                    TrialState::Pruned => "Pruned",
                    TrialState::Failed => "Failed",
                    TrialState::Running => "Running",
                }
            )?;

            for id in &param_ids {
                if let Some(pv) = trial.params.get(id) {
                    write!(writer, ",{pv}")?;
                } else {
                    write!(writer, ",")?;
                }
            }

            for key in &attr_keys {
                if let Some(attr) = trial.user_attrs.get(key) {
                    write!(writer, ",{}", csv_escape(&format_attr(attr)))?;
                } else {
                    write!(writer, ",")?;
                }
            }

            writeln!(writer)?;
        }

        Ok(())
    }

    /// Export completed trials to a CSV file at the given path.
    ///
    /// Convenience wrapper around [`to_csv`](Self::to_csv) that creates a
    /// buffered file writer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be created or written.
    pub fn export_csv(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        self.to_csv(std::io::BufWriter::new(file))
    }

    /// Return a human-readable summary of the study.
    ///
    /// The summary includes:
    /// - Optimization direction and total trial count
    /// - Breakdown by state (complete, pruned) when applicable
    /// - Best trial value and parameters (if any completed trials exist)
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let x = FloatParam::new(0.0, 10.0).name("x");
    ///
    /// let mut trial = study.create_trial();
    /// let _ = x.suggest(&mut trial).unwrap();
    /// study.complete_trial(trial, 0.42);
    ///
    /// let summary = study.summary();
    /// assert!(summary.contains("Minimize"));
    /// assert!(summary.contains("0.42"));
    /// ```
    #[must_use]
    pub fn summary(&self) -> String {
        use fmt::Write;

        let trials = self.storage.trials_arc().read();
        let n_complete = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .count();
        let n_pruned = trials
            .iter()
            .filter(|t| t.state == TrialState::Pruned)
            .count();

        let direction_str = match self.direction {
            Direction::Minimize => "Minimize",
            Direction::Maximize => "Maximize",
        };

        let mut s = format!("Study: {direction_str} | {n} trials", n = trials.len());
        if n_pruned > 0 {
            let _ = write!(s, " ({n_complete} complete, {n_pruned} pruned)");
        }

        drop(trials);

        if let Ok(best) = self.best_trial() {
            let _ = write!(s, "\nBest value: {} (trial #{})", best.value, best.id);
            if !best.params.is_empty() {
                s.push_str("\nBest parameters:");
                let mut params: Vec<_> = best.params.iter().collect();
                params.sort_by_key(|(id, _)| *id);
                for (id, value) in params {
                    let label = best.param_labels.get(id).map_or("?", String::as_str);
                    let _ = write!(s, "\n  {label} = {value}");
                }
            }
        }

        s
    }
}

impl<V> fmt::Display for Study<V>
where
    V: PartialOrd + Clone + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary())
    }
}

#[cfg(feature = "serde")]
impl<V: PartialOrd + Clone + serde::Serialize> Study<V> {
    /// Export trials as a pretty-printed JSON array to a file.
    ///
    /// Each element in the array is a serialized [`CompletedTrial`](crate::sampler::CompletedTrial).
    /// Requires the `serde` feature.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be created or written.
    pub fn export_json(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let trials = self.trials();
        serde_json::to_writer_pretty(file, &trials).map_err(std::io::Error::other)
    }
}

impl Study<f64> {
    /// Generate an HTML report with interactive Plotly.js charts.
    ///
    /// Create a self-contained HTML file that can be opened in any browser.
    /// See [`generate_html_report`](crate::visualization::generate_html_report)
    /// for details on the included charts.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be created or written.
    pub fn export_html(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        crate::visualization::generate_html_report(self, path)
    }
}

/// Escape a string for CSV output. If the value contains a comma, quote, or
/// newline, wrap it in double-quotes and double any embedded quotes.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Format an `AttrValue` as a string for CSV cells.
fn format_attr(attr: &crate::trial::AttrValue) -> String {
    use crate::trial::AttrValue;
    match attr {
        AttrValue::Float(v) => v.to_string(),
        AttrValue::Int(v) => v.to_string(),
        AttrValue::String(v) => v.clone(),
        AttrValue::Bool(v) => v.to_string(),
    }
}
