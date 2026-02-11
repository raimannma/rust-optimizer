/// Generate a random `f64` in the range `[low, high)`.
#[inline]
pub(crate) fn f64_range(rng: &mut fastrand::Rng, low: f64, high: f64) -> f64 {
    low + rng.f64() * (high - low)
}
