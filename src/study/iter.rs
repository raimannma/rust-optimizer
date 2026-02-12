use crate::sampler::CompletedTrial;

use super::Study;

impl<V> Study<V>
where
    V: PartialOrd + Clone,
{
    /// Return an iterator over all completed trials.
    ///
    /// This clones the internal trial list, so it is suitable for
    /// analysis and iteration but not for hot paths.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let trial = study.create_trial();
    /// study.complete_trial(trial, 1.0);
    ///
    /// for t in study.iter() {
    ///     println!("Trial {} → {}", t.id, t.value);
    /// }
    /// ```
    #[must_use]
    pub fn iter(&self) -> std::vec::IntoIter<CompletedTrial<V>> {
        self.trials().into_iter()
    }
}

impl<V> IntoIterator for &Study<V>
where
    V: PartialOrd + Clone,
{
    type Item = CompletedTrial<V>;
    type IntoIter = std::vec::IntoIter<CompletedTrial<V>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
