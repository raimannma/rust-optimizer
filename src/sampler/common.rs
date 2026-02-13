//! Shared distribution-level utilities used across multiple samplers.

use crate::distribution::Distribution;
use crate::param::ParamValue;
use crate::rng_util;

/// Compute internal-space bounds for a distribution.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn internal_bounds(distribution: &Distribution) -> Option<(f64, f64)> {
    match distribution {
        Distribution::Float(d) => {
            if d.log_scale {
                if d.low <= 0.0 || d.high <= 0.0 {
                    return None;
                }
                Some((d.low.ln(), d.high.ln()))
            } else {
                Some((d.low, d.high))
            }
        }
        Distribution::Int(d) => {
            if d.log_scale {
                if d.low < 1 {
                    return None;
                }
                Some(((d.low as f64).ln(), (d.high as f64).ln()))
            } else {
                Some((d.low as f64, d.high as f64))
            }
        }
        Distribution::Categorical(_) => None,
    }
}

/// Convert an internal-space value back to a `ParamValue`.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(crate) fn from_internal(value: f64, distribution: &Distribution) -> ParamValue {
    match distribution {
        Distribution::Float(d) => {
            let v = if d.log_scale { value.exp() } else { value };
            let v = if let Some(step) = d.step {
                let k = ((v - d.low) / step).round();
                d.low + k * step
            } else {
                v
            };
            ParamValue::Float(v.clamp(d.low, d.high))
        }
        Distribution::Int(d) => {
            let v = if d.log_scale { value.exp() } else { value };
            let v = if let Some(step) = d.step {
                let k = ((v - d.low as f64) / step as f64).round() as i64;
                d.low.saturating_add(k.saturating_mul(step))
            } else {
                v.round() as i64
            };
            ParamValue::Int(v.clamp(d.low, d.high))
        }
        Distribution::Categorical(_) => {
            unreachable!("from_internal should not be called for categorical distributions")
        }
    }
}

/// Convert a `ParamValue` to its internal-space representation.
#[allow(clippy::cast_precision_loss, dead_code)]
pub(crate) fn to_internal(value: &ParamValue, distribution: &Distribution) -> f64 {
    match (value, distribution) {
        (ParamValue::Float(v), Distribution::Float(d)) => {
            if d.log_scale {
                v.ln()
            } else {
                *v
            }
        }
        (ParamValue::Int(v), Distribution::Int(d)) => {
            if d.log_scale {
                (*v as f64).ln()
            } else {
                *v as f64
            }
        }
        _ => 0.0,
    }
}

/// Sample a random value for any distribution.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(crate) fn sample_random(rng: &mut fastrand::Rng, distribution: &Distribution) -> ParamValue {
    match distribution {
        Distribution::Float(d) => {
            let value = if d.log_scale {
                let log_low = d.low.ln();
                let log_high = d.high.ln();
                let v = rng_util::f64_range(rng, log_low, log_high).exp();
                if let Some(step) = d.step {
                    let k = ((v - d.low) / step).round();
                    (d.low + k * step).clamp(d.low, d.high)
                } else {
                    v
                }
            } else if let Some(step) = d.step {
                let n_steps = ((d.high - d.low) / step).floor() as i64;
                let k = rng.i64(0..=n_steps);
                d.low + (k as f64) * step
            } else {
                rng_util::f64_range(rng, d.low, d.high)
            };
            ParamValue::Float(value)
        }
        Distribution::Int(d) => {
            let value = if d.log_scale {
                let log_low = (d.low as f64).ln();
                let log_high = (d.high as f64).ln();
                let v = rng_util::f64_range(rng, log_low, log_high).exp();
                let raw = if let Some(step) = d.step {
                    let k = ((v - d.low as f64) / step as f64).round() as i64;
                    d.low.saturating_add(k.saturating_mul(step))
                } else {
                    v.round() as i64
                };
                raw.clamp(d.low, d.high)
            } else if let Some(step) = d.step {
                let n_steps = (d.high - d.low) / step;
                let k = rng.i64(0..=n_steps);
                d.low + k * step
            } else {
                rng.i64(d.low..=d.high)
            };
            ParamValue::Int(value)
        }
        Distribution::Categorical(d) => ParamValue::Categorical(rng.usize(0..d.n_choices)),
    }
}
