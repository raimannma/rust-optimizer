//! Shared TPE sampling functions used by both `TpeSampler` and `MotpeSampler`.

use crate::kde::KernelDensityEstimator;
use crate::rng_util;

/// Samples using TPE for float distributions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_tpe_float(
    low: f64,
    high: f64,
    log_scale: bool,
    step: Option<f64>,
    good_values: Vec<f64>,
    bad_values: Vec<f64>,
    n_ei_candidates: usize,
    kde_bandwidth: Option<f64>,
    rng: &mut fastrand::Rng,
) -> f64 {
    // Transform to internal space (log space if needed)
    let (internal_low, internal_high, good_internal, bad_internal) = if log_scale {
        let i_low = low.ln();
        let i_high = high.ln();
        let g = {
            let mut v = good_values;
            for x in &mut v {
                *x = x.ln();
            }
            v
        };
        let b = {
            let mut v = bad_values;
            for x in &mut v {
                *x = x.ln();
            }
            v
        };
        (i_low, i_high, g, b)
    } else {
        (low, high, good_values, bad_values)
    };

    // Fit KDEs to good and bad groups
    let l_kde = match kde_bandwidth {
        Some(bw) => KernelDensityEstimator::with_bandwidth(good_internal, bw),
        None => KernelDensityEstimator::new(good_internal),
    };
    let g_kde = match kde_bandwidth {
        Some(bw) => KernelDensityEstimator::with_bandwidth(bad_internal, bw),
        None => KernelDensityEstimator::new(bad_internal),
    };

    // If KDE construction fails, fall back to uniform sampling
    let (Ok(l_kde), Ok(g_kde)) = (l_kde, g_kde) else {
        return rng_util::f64_range(rng, low, high);
    };

    // Generate candidates from l(x) and select the one with best l(x)/g(x) ratio
    let mut best_candidate = internal_low;
    let mut best_ratio = f64::NEG_INFINITY;

    for _ in 0..n_ei_candidates {
        let candidate = l_kde.sample(rng).clamp(internal_low, internal_high);

        let l_density = l_kde.pdf(candidate);
        let g_density = g_kde.pdf(candidate);

        // Compute l(x)/g(x) ratio, handling zero density
        let ratio = if g_density < f64::EPSILON {
            if l_density > f64::EPSILON {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            l_density / g_density
        };

        if ratio > best_ratio {
            best_ratio = ratio;
            best_candidate = candidate;
        }
    }

    // Transform back from internal space
    let mut value = if log_scale {
        best_candidate.exp()
    } else {
        best_candidate
    };

    // Apply step constraint if present
    if let Some(step) = step {
        let k = ((value - low) / step).round();
        value = low + k * step;
    }

    // Ensure value is within bounds
    value.clamp(low, high)
}

/// Samples using TPE for integer distributions.
#[allow(
    clippy::too_many_arguments,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
pub(crate) fn sample_tpe_int(
    low: i64,
    high: i64,
    log_scale: bool,
    step: Option<i64>,
    good_values: Vec<i64>,
    bad_values: Vec<i64>,
    n_ei_candidates: usize,
    kde_bandwidth: Option<f64>,
    rng: &mut fastrand::Rng,
) -> i64 {
    // Convert to floats for KDE
    let good_floats: Vec<f64> = good_values.into_iter().map(|v| v as f64).collect();
    let bad_floats: Vec<f64> = bad_values.into_iter().map(|v| v as f64).collect();

    // Use float TPE sampling
    let float_value = sample_tpe_float(
        low as f64,
        high as f64,
        log_scale,
        step.map(|s| s as f64),
        good_floats,
        bad_floats,
        n_ei_candidates,
        kde_bandwidth,
        rng,
    );

    // Round to nearest integer
    let int_value = float_value.round() as i64;

    // Apply step constraint if present
    let int_value = if let Some(step) = step {
        let k = ((int_value - low) as f64 / step as f64).round() as i64;
        low + k * step
    } else {
        int_value
    };

    // Ensure value is within bounds
    int_value.clamp(low, high)
}

/// Samples using TPE for categorical distributions.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn sample_tpe_categorical(
    n_choices: usize,
    good_indices: &[usize],
    bad_indices: &[usize],
    rng: &mut fastrand::Rng,
) -> usize {
    // Stack-allocate for the common case (<=32 choices), heap for rare large cases
    let mut good_buf = [0usize; 32];
    let mut bad_buf = [0usize; 32];
    let mut weight_buf = [0.0f64; 32];

    let mut good_vec;
    let mut bad_vec;
    let mut weight_vec;

    let (good_counts, bad_counts, weights): (&mut [usize], &mut [usize], &mut [f64]) =
        if n_choices <= 32 {
            (
                &mut good_buf[..n_choices],
                &mut bad_buf[..n_choices],
                &mut weight_buf[..n_choices],
            )
        } else {
            good_vec = vec![0usize; n_choices];
            bad_vec = vec![0usize; n_choices];
            weight_vec = vec![0.0f64; n_choices];
            (&mut good_vec, &mut bad_vec, &mut weight_vec)
        };

    // Count occurrences in good and bad groups
    for &idx in good_indices {
        if idx < n_choices {
            good_counts[idx] += 1;
        }
    }
    for &idx in bad_indices {
        if idx < n_choices {
            bad_counts[idx] += 1;
        }
    }

    // Add smoothing (Laplace smoothing) to avoid zero probabilities
    let good_total = good_indices.len() as f64 + n_choices as f64;
    let bad_total = bad_indices.len() as f64 + n_choices as f64;

    // Calculate l(x)/g(x) ratio for each category
    for i in 0..n_choices {
        let l_prob = (good_counts[i] as f64 + 1.0) / good_total;
        let g_prob = (bad_counts[i] as f64 + 1.0) / bad_total;
        weights[i] = l_prob / g_prob;
    }

    // Sample proportionally to weights
    let total_weight: f64 = weights.iter().sum();
    let threshold = rng.f64() * total_weight;

    let mut cumulative = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += w;
        if cumulative >= threshold {
            return i;
        }
    }

    // Fallback to last index (shouldn't happen)
    n_choices - 1
}
