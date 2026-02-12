//! Gaussian Process (GP) sampler with Expected Improvement acquisition.
//!
//! A classical Bayesian optimization sampler that builds a Gaussian Process
//! surrogate model with a **Matérn 5/2 kernel** (with ARD lengthscales) and
//! selects the next trial by maximizing the **Expected Improvement (EI)**
//! acquisition function. Best suited for small, expensive evaluations in
//! low-dimensional continuous spaces (d ≤ 20).
//!
//! # Algorithm overview
//!
//! 1. **Startup phase** — the first `n_startup_trials` trials are sampled
//!    uniformly at random to build an initial dataset.
//! 2. **Fit GP** — training observations are standardized (zero mean, unit
//!    variance) and a GP with Matérn 5/2 kernel is fitted via Cholesky
//!    decomposition. ARD lengthscales are set to the per-dimension standard
//!    deviation of the training inputs.
//! 3. **Maximize EI** — `n_candidates` random points are evaluated under
//!    the GP posterior and the point with the highest Expected Improvement
//!    is returned as the next trial.
//!
//! The GP uses at most 100 training points (the most recent ones) to keep
//! the O(n³) fitting cost manageable.
//!
//! # When to use
//!
//! - **Expensive objective functions** where every evaluation is costly
//!   (e.g. physical experiments, large simulations). The GP surrogate
//!   amortizes this cost by making fewer evaluations.
//! - **Low-dimensional continuous spaces** — typically d ≤ 20. Beyond that,
//!   the GP becomes unreliable and alternatives like
//!   [`CmaEsSampler`](super::cma_es::CmaEsSampler) or
//!   [`TpeSampler`](super::tpe::TpeSampler) are preferable.
//! - **Smooth, low-noise objectives** — the GP assumes smoothness through
//!   the Matérn 5/2 kernel. Very noisy objectives require increasing
//!   `noise_variance`.
//!
//! Categorical parameters are sampled uniformly at random and do not
//! participate in the GP model. If all parameters are categorical, the
//! sampler falls back to pure random sampling.
//!
//! # Configuration
//!
//! | Option | Default | Description |
//! |--------|---------|-------------|
//! | `n_startup_trials` | 10 | Random trials before GP-guided sampling begins |
//! | `n_candidates` | 1000 | Random candidates for EI maximization |
//! | `noise_variance` | 1e-6 | Observation noise added to kernel diagonal |
//! | `seed` | random | RNG seed for reproducibility |
//!
//! # Feature flag
//!
//! Requires the **`gp`** feature (adds the `nalgebra` dependency):
//!
//! ```toml
//! [dependencies]
//! optimizer = { version = "...", features = ["gp"] }
//! ```
//!
//! # Examples
//!
//! ```
//! use optimizer::sampler::gp::GpSampler;
//! use optimizer::{Direction, Study};
//!
//! // Minimize an expensive function with GP-based Bayesian optimization
//! let sampler = GpSampler::builder()
//!     .n_startup_trials(5)
//!     .n_candidates(500)
//!     .seed(42)
//!     .build();
//!
//! let mut study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
//! ```

use std::collections::HashMap;

use nalgebra::DMatrix;
use parking_lot::Mutex;

use crate::distribution::Distribution;
use crate::param::ParamValue;
use crate::rng_util;
use crate::sampler::{CompletedTrial, Sampler};

use super::common::{from_internal, internal_bounds, sample_random, to_internal};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Gaussian Process sampler for Bayesian optimization.
///
/// Uses a GP surrogate with Matérn 5/2 kernel and Expected Improvement
/// acquisition to guide sampling toward promising regions of the search
/// space. Best suited for continuous (float/int) parameters in low
/// dimensions (up to ~20).
///
/// # Examples
///
/// ```
/// use optimizer::sampler::gp::GpSampler;
/// use optimizer::{Direction, Study};
///
/// // Default configuration
/// let study: Study<f64> = Study::with_sampler(Direction::Minimize, GpSampler::new());
///
/// // With seed for reproducibility
/// let study: Study<f64> = Study::with_sampler(Direction::Minimize, GpSampler::with_seed(42));
///
/// // Custom configuration via builder
/// let sampler = GpSampler::builder()
///     .n_startup_trials(15)
///     .n_candidates(2000)
///     .noise_variance(1e-4)
///     .seed(42)
///     .build();
/// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
/// ```
pub struct GpSampler {
    state: Mutex<GpState>,
}

impl GpSampler {
    /// Creates a new GP sampler with a random seed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(GpState::new(None, None, None, None)),
        }
    }

    /// Creates a new GP sampler with a fixed seed for reproducibility.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            state: Mutex::new(GpState::new(None, None, None, Some(seed))),
        }
    }

    /// Creates a builder for configuring a `GpSampler`.
    #[must_use]
    pub fn builder() -> GpSamplerBuilder {
        GpSamplerBuilder::new()
    }
}

impl Default for GpSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for configuring a [`GpSampler`].
///
/// All options have sensible defaults:
/// - `n_startup_trials`: 10
/// - `n_candidates`: 1000
/// - `noise_variance`: 1e-6
/// - `seed`: random
///
/// # Examples
///
/// ```
/// use optimizer::sampler::gp::GpSamplerBuilder;
///
/// let sampler = GpSamplerBuilder::new()
///     .n_startup_trials(15)
///     .n_candidates(2000)
///     .noise_variance(1e-4)
///     .seed(42)
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct GpSamplerBuilder {
    n_startup_trials: Option<usize>,
    n_candidates: Option<usize>,
    noise_variance: Option<f64>,
    seed: Option<u64>,
}

impl GpSamplerBuilder {
    /// Creates a new builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of random trials before GP-guided sampling begins.
    ///
    /// Default: 10.
    #[must_use]
    pub fn n_startup_trials(mut self, n: usize) -> Self {
        self.n_startup_trials = Some(n);
        self
    }

    /// Sets the number of random candidate points for acquisition optimization.
    ///
    /// More candidates improve the quality of the acquisition maximum
    /// at the cost of more GP predictions per trial.
    ///
    /// Default: 1000.
    #[must_use]
    pub fn n_candidates(mut self, n: usize) -> Self {
        self.n_candidates = Some(n);
        self
    }

    /// Sets the observation noise variance added to the kernel diagonal.
    ///
    /// Controls the assumed noise level. Larger values make the GP smoother.
    ///
    /// Default: 1e-6 (near-noiseless).
    #[must_use]
    pub fn noise_variance(mut self, v: f64) -> Self {
        self.noise_variance = Some(v);
        self
    }

    /// Sets the random seed for reproducibility.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Builds the configured [`GpSampler`].
    #[must_use]
    pub fn build(self) -> GpSampler {
        GpSampler {
            state: Mutex::new(GpState::new(
                self.n_startup_trials,
                self.n_candidates,
                self.noise_variance,
                self.seed,
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Default number of random startup trials before GP kicks in.
const DEFAULT_N_STARTUP: usize = 10;
/// Default number of candidate points for EI optimization.
const DEFAULT_N_CANDIDATES: usize = 1000;
/// Default observation noise variance.
const DEFAULT_NOISE_VAR: f64 = 1e-6;

/// Describes how a parameter dimension maps into the GP internal vector.
#[derive(Clone, Debug)]
struct DimensionInfo {
    distribution: Distribution,
    is_continuous: bool,
    bounds: Option<(f64, f64)>,
}

/// Tracks per-trial sampling progress.
#[derive(Clone, Debug)]
struct TrialProgress {
    /// The candidate values for each dimension.
    values: Vec<ParamValue>,
    /// Next dimension to return.
    next_dim: usize,
}

/// Phase of the GP state machine.
enum GpPhase {
    /// Discovering the search space (first trial).
    Discovery,
    /// Steady-state sampling.
    Active,
}

/// A fitted GP model ready for predictions.
struct GpModel {
    /// Cholesky factor L of K + σ²I.
    cholesky: nalgebra::linalg::Cholesky<f64, nalgebra::Dyn>,
    /// α = (K + σ²I)^{-1} y.
    alpha: nalgebra::DVector<f64>,
    /// Training inputs (each row is a data point, normalized to [0, 1]).
    x_train: Vec<Vec<f64>>,
    /// ARD lengthscales per dimension.
    lengthscales: Vec<f64>,
    /// Signal variance.
    signal_var: f64,
    /// Mean of original y values (for un-standardization, unused but kept for diagnostics).
    _y_mean: f64,
    /// Std dev of original y values (unused but kept for diagnostics).
    _y_std: f64,
    /// Best observed (standardized) y.
    f_best: f64,
}

/// Top-level mutable state behind the `Mutex`.
struct GpState {
    rng: fastrand::Rng,
    n_startup_trials: usize,
    n_candidates: usize,
    noise_variance: f64,
    phase: GpPhase,
    dimensions: Vec<DimensionInfo>,
    trial_progress: HashMap<u64, TrialProgress>,
    discovery_trial_id: Option<u64>,
}

impl GpState {
    fn new(
        n_startup: Option<usize>,
        n_candidates: Option<usize>,
        noise_var: Option<f64>,
        seed: Option<u64>,
    ) -> Self {
        let rng = seed.map_or_else(fastrand::Rng::new, fastrand::Rng::with_seed);
        Self {
            rng,
            n_startup_trials: n_startup.unwrap_or(DEFAULT_N_STARTUP),
            n_candidates: n_candidates.unwrap_or(DEFAULT_N_CANDIDATES),
            noise_variance: noise_var.unwrap_or(DEFAULT_NOISE_VAR),
            phase: GpPhase::Discovery,
            dimensions: Vec::new(),
            trial_progress: HashMap::new(),
            discovery_trial_id: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Matérn 5/2 kernel
// ---------------------------------------------------------------------------

/// Matérn 5/2 kernel with ARD lengthscales.
///
/// `k(x1, x2) = σ² (1 + √5 r + 5/3 r²) exp(-√5 r)`
/// where `r = sqrt(Σ ((x1_i - x2_i) / l_i)²)`
fn matern52(x1: &[f64], x2: &[f64], lengthscales: &[f64], signal_var: f64) -> f64 {
    let mut r_sq = 0.0;
    for i in 0..x1.len() {
        let diff = (x1[i] - x2[i]) / lengthscales[i];
        r_sq += diff * diff;
    }
    let r = r_sq.sqrt();
    let sqrt5_r = SQRT_5 * r;
    signal_var * (1.0 + sqrt5_r + 5.0 / 3.0 * r_sq) * (-sqrt5_r).exp()
}

/// Build the kernel matrix `K + σ²I`.
fn kernel_matrix(
    x: &[Vec<f64>],
    lengthscales: &[f64],
    signal_var: f64,
    noise_var: f64,
) -> DMatrix<f64> {
    let n = x.len();
    DMatrix::from_fn(n, n, |i, j| {
        let k = matern52(&x[i], &x[j], lengthscales, signal_var);
        if i == j { k + noise_var } else { k }
    })
}

/// Compute the kernel vector k(x*, X) for a test point.
fn kernel_vector(
    x_star: &[f64],
    x_train: &[Vec<f64>],
    lengthscales: &[f64],
    signal_var: f64,
) -> nalgebra::DVector<f64> {
    nalgebra::DVector::from_fn(x_train.len(), |i, _| {
        matern52(x_star, &x_train[i], lengthscales, signal_var)
    })
}

/// Precomputed √5 constant.
const SQRT_5: f64 = 2.236_213_562_373_095;

// ---------------------------------------------------------------------------
// GP fitting and prediction
// ---------------------------------------------------------------------------

/// Fit a GP model to the training data.
///
/// Returns `None` if fitting fails (e.g. Cholesky decomposition failure).
#[allow(clippy::cast_precision_loss)]
fn fit_gp(x_train: &[Vec<f64>], y_train: &[f64], noise_var: f64) -> Option<GpModel> {
    let n = y_train.len();
    if n == 0 {
        return None;
    }

    // Standardize y
    let y_mean = y_train.iter().sum::<f64>() / n as f64;
    let y_var = if n > 1 {
        y_train.iter().map(|&y| (y - y_mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        1.0
    };
    let y_std = y_var.sqrt().max(1e-10);
    let y_standardized: Vec<f64> = y_train.iter().map(|&y| (y - y_mean) / y_std).collect();

    let f_best = y_standardized.iter().copied().fold(f64::INFINITY, f64::min);

    // ARD lengthscales: per-dimension std dev of training X, clamped
    let d = if x_train.is_empty() {
        0
    } else {
        x_train[0].len()
    };
    let lengthscales: Vec<f64> = (0..d)
        .map(|j| {
            let vals: Vec<f64> = x_train.iter().map(|x| x[j]).collect();
            let mean_j = vals.iter().sum::<f64>() / n as f64;
            let var_j = vals.iter().map(|&v| (v - mean_j).powi(2)).sum::<f64>() / n as f64;
            var_j.sqrt().max(0.01)
        })
        .collect();

    // Signal variance = 1.0 (data is standardized)
    let signal_var = 1.0;

    let k = kernel_matrix(x_train, &lengthscales, signal_var, noise_var);
    let cholesky = nalgebra::linalg::Cholesky::new(k)?;

    // α = (K + σ²I)^{-1} y
    let y_vec = nalgebra::DVector::from_column_slice(&y_standardized);
    let alpha = cholesky.solve(&y_vec);

    Some(GpModel {
        cholesky,
        alpha,
        x_train: x_train.to_vec(),
        lengthscales,
        signal_var,
        _y_mean: y_mean,
        _y_std: y_std,
        f_best,
    })
}

/// Predict mean and standard deviation at a test point.
fn predict(model: &GpModel, x: &[f64]) -> (f64, f64) {
    let k_star = kernel_vector(x, &model.x_train, &model.lengthscales, model.signal_var);

    // Mean: k*^T α
    let mean = k_star.dot(&model.alpha);

    // Variance: k(x*, x*) - k*^T (K + σ²I)^{-1} k*
    let k_self = model.signal_var;
    let v = model.cholesky.solve(&k_star);
    let var = (k_self - k_star.dot(&v)).max(0.0);

    (mean, var.sqrt())
}

// ---------------------------------------------------------------------------
// Normal distribution helpers (Abramowitz-Stegun approximation)
// ---------------------------------------------------------------------------

/// Standard normal PDF.
fn norm_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal CDF (Abramowitz-Stegun rational approximation).
fn norm_cdf(x: f64) -> f64 {
    // Hart approximation (higher precision than basic A&S)
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }

    let abs_x = x.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * abs_x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let poly = 0.319_381_530 * t - 0.356_563_782 * t2 + 1.781_477_937 * t3 - 1.821_255_978 * t4
        + 1.330_274_429 * t5;
    let pdf = norm_pdf(abs_x);
    let cdf = 1.0 - pdf * poly;

    if x >= 0.0 { cdf } else { 1.0 - cdf }
}

// ---------------------------------------------------------------------------
// Expected Improvement
// ---------------------------------------------------------------------------

/// Compute Expected Improvement at a point.
///
/// `EI(x) = (f_best - mean) Φ(z) + std φ(z)`
/// where `z = (f_best - mean) / std`
fn expected_improvement(mean: f64, std: f64, f_best: f64) -> f64 {
    if std < 1e-12 {
        return (f_best - mean).max(0.0);
    }
    let z = (f_best - mean) / std;
    let improvement = (f_best - mean) * norm_cdf(z) + std * norm_pdf(z);
    improvement.max(0.0)
}

// ---------------------------------------------------------------------------
// Acquisition optimization
// ---------------------------------------------------------------------------

/// Find the point in [0, 1]^d that maximizes EI via multi-start random search.
fn optimize_acquisition(
    model: &GpModel,
    n_dims: usize,
    n_candidates: usize,
    rng: &mut fastrand::Rng,
) -> Vec<f64> {
    let mut best_ei = f64::NEG_INFINITY;
    let mut best_x = vec![0.5; n_dims];

    for _ in 0..n_candidates {
        let x: Vec<f64> = (0..n_dims)
            .map(|_| rng_util::f64_range(rng, 0.0, 1.0))
            .collect();
        let (mean, std) = predict(model, &x);
        let ei = expected_improvement(mean, std, model.f_best);
        if ei > best_ei {
            best_ei = ei;
            best_x = x;
        }
    }

    best_x
}

// ---------------------------------------------------------------------------
// Data preprocessing helpers
// ---------------------------------------------------------------------------

/// Convert an internal-space value to normalized [0, 1] using bounds.
fn to_normalized(value: f64, lo: f64, hi: f64) -> f64 {
    if (hi - lo).abs() < 1e-15 {
        0.5
    } else {
        (value - lo) / (hi - lo)
    }
}

/// Convert a normalized [0, 1] value back to internal space.
fn from_normalized(value: f64, lo: f64, hi: f64) -> f64 {
    lo + value * (hi - lo)
}

// ---------------------------------------------------------------------------
// Extract training data from history
// ---------------------------------------------------------------------------

/// Maximum number of training points to use for the GP.
/// Caps computational cost at O(`MAX_TRAIN_POINTS`^3) per trial.
const MAX_TRAIN_POINTS: usize = 100;

/// Establish a deterministic mapping from dimension index to `ParamId`
/// using the first trial in history.
///
/// Matches dimensions to params by distribution equality, consuming
/// matched params to correctly handle duplicate distributions.
fn establish_param_mapping(
    trial: &CompletedTrial,
    dimensions: &[DimensionInfo],
) -> Vec<Option<crate::parameter::ParamId>> {
    use crate::parameter::ParamId;

    let mut available: Vec<(ParamId, &Distribution)> =
        trial.distributions.iter().map(|(id, d)| (*id, d)).collect();
    // Sort for deterministic matching order
    available.sort_by_key(|(id, _)| *id);

    let mut mapping = Vec::with_capacity(dimensions.len());
    for dim in dimensions {
        let pos = available.iter().position(|(_, d)| **d == dim.distribution);
        if let Some(pos) = pos {
            mapping.push(Some(available.remove(pos).0));
        } else {
            mapping.push(None);
        }
    }
    mapping
}

/// Build normalized training data from completed trials.
///
/// Returns `(x_train, y_train)` where x values are normalized to [0, 1]
/// per dimension using the bounds from `dimensions`. Only continuous
/// dimensions are included. Uses at most [`MAX_TRAIN_POINTS`] most recent
/// trials.
#[allow(clippy::cast_precision_loss)]
fn build_training_data(
    history: &[CompletedTrial],
    dimensions: &[DimensionInfo],
) -> (Vec<Vec<f64>>, Vec<f64>) {
    if history.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Use only the most recent trials to cap GP fitting cost
    let start = history.len().saturating_sub(MAX_TRAIN_POINTS);
    let recent = &history[start..];

    // Establish dimension → ParamId mapping from the first trial
    let param_mapping = establish_param_mapping(&recent[0], dimensions);

    let continuous_indices: Vec<usize> = dimensions
        .iter()
        .enumerate()
        .filter(|(_, d)| d.is_continuous)
        .map(|(i, _)| i)
        .collect();

    let mut x_train = Vec::with_capacity(recent.len());
    let mut y_train = Vec::with_capacity(recent.len());

    for trial in recent {
        let mut x_row = Vec::with_capacity(continuous_indices.len());
        let mut valid = true;

        for &dim_idx in &continuous_indices {
            let dim_info = &dimensions[dim_idx];
            if let Some(param_id) = param_mapping[dim_idx] {
                if let Some(param_val) = trial.params.get(&param_id) {
                    let internal = to_internal(param_val, &dim_info.distribution);
                    let (lo, hi) = dim_info.bounds.unwrap_or((0.0, 1.0));
                    x_row.push(to_normalized(internal, lo, hi));
                } else {
                    valid = false;
                    break;
                }
            } else {
                valid = false;
                break;
            }
        }

        if valid && x_row.len() == continuous_indices.len() {
            x_train.push(x_row);
            y_train.push(trial.value);
        }
    }

    (x_train, y_train)
}

// ---------------------------------------------------------------------------
// Sampler trait implementation
// ---------------------------------------------------------------------------

impl Sampler for GpSampler {
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn sample(
        &self,
        distribution: &Distribution,
        trial_id: u64,
        history: &[CompletedTrial],
    ) -> ParamValue {
        let mut state = self.state.lock();

        match &state.phase {
            GpPhase::Discovery => sample_discovery(&mut state, distribution, trial_id),
            GpPhase::Active => sample_active(&mut state, distribution, trial_id, history),
        }
    }
}

/// Handle sampling during the discovery phase.
fn sample_discovery(state: &mut GpState, distribution: &Distribution, trial_id: u64) -> ParamValue {
    // A new trial_id means discovery is done
    if let Some(prev_id) = state.discovery_trial_id
        && trial_id != prev_id
    {
        finalize_discovery(state);
        return sample_active(state, distribution, trial_id, &[]);
    }

    state.discovery_trial_id = Some(trial_id);

    let is_continuous = !matches!(distribution, Distribution::Categorical(_));
    let bounds = internal_bounds(distribution);
    state.dimensions.push(DimensionInfo {
        distribution: distribution.clone(),
        is_continuous,
        bounds,
    });

    sample_random(&mut state.rng, distribution)
}

/// Finalize discovery and transition to the active phase.
fn finalize_discovery(state: &mut GpState) {
    state.phase = GpPhase::Active;
    state.trial_progress.clear();
}

/// Handle sampling during the active phase.
fn sample_active(
    state: &mut GpState,
    distribution: &Distribution,
    trial_id: u64,
    history: &[CompletedTrial],
) -> ParamValue {
    // If this trial already has progress, return the next pre-computed value
    if let Some(progress) = state.trial_progress.get_mut(&trial_id) {
        let dim_idx = progress.next_dim;
        progress.next_dim += 1;
        if dim_idx < progress.values.len() {
            return progress.values[dim_idx].clone();
        }
        // Extra dimension not seen during discovery
        return sample_random(&mut state.rng, distribution);
    }

    // New trial: compute all dimension values at once
    let n_continuous = state.dimensions.iter().filter(|d| d.is_continuous).count();
    let use_gp = n_continuous > 0 && history.len() >= state.n_startup_trials;

    let values = if use_gp {
        compute_gp_candidate(state, history)
    } else {
        compute_random_candidate(state)
    };

    let first_value = values
        .first()
        .cloned()
        .unwrap_or_else(|| sample_random(&mut state.rng, distribution));

    state.trial_progress.insert(
        trial_id,
        TrialProgress {
            values,
            next_dim: 1,
        },
    );

    first_value
}

/// Compute a candidate using the GP model.
fn compute_gp_candidate(state: &mut GpState, history: &[CompletedTrial]) -> Vec<ParamValue> {
    let (x_train, y_train) = build_training_data(history, &state.dimensions);

    // Try to fit GP; fall back to random if it fails
    let model = fit_gp(&x_train, &y_train, state.noise_variance);

    let n_continuous = state.dimensions.iter().filter(|d| d.is_continuous).count();

    let normalized_candidate = if let Some(ref model) = model {
        optimize_acquisition(model, n_continuous, state.n_candidates, &mut state.rng)
    } else {
        // GP fitting failed; use random
        (0..n_continuous)
            .map(|_| rng_util::f64_range(&mut state.rng, 0.0, 1.0))
            .collect()
    };

    // Convert normalized candidate back to parameter values
    let mut values = Vec::with_capacity(state.dimensions.len());
    let mut ci = 0; // continuous dimension index

    for dim in &state.dimensions {
        if dim.is_continuous {
            let (lo, hi) = dim.bounds.unwrap_or((0.0, 1.0));
            let internal_val = from_normalized(normalized_candidate[ci], lo, hi);
            values.push(from_internal(internal_val, &dim.distribution));
            ci += 1;
        } else {
            values.push(sample_random(&mut state.rng, &dim.distribution));
        }
    }

    values
}

/// Compute a random candidate for all dimensions.
fn compute_random_candidate(state: &mut GpState) -> Vec<ParamValue> {
    state
        .dimensions
        .iter()
        .map(|dim| sample_random(&mut state.rng, &dim.distribution))
        .collect()
}
