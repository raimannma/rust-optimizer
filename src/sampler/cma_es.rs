//! CMA-ES (Covariance Matrix Adaptation Evolution Strategy) sampler.
//!
//! CMA-ES maintains a multivariate Gaussian distribution over continuous
//! parameters and adapts its mean, covariance matrix, and step-size based
//! on trial rankings. It is one of the most effective derivative-free
//! optimizers for continuous search spaces.
//!
//! Categorical parameters are sampled uniformly at random (not part of
//! the CMA-ES vector). If all parameters are categorical, the sampler
//! falls back to pure random sampling.
//!
//! Requires the `cma-es` feature flag.
//!
//! # Examples
//!
//! ```
//! use optimizer::sampler::cma_es::CmaEsSampler;
//! use optimizer::{Direction, Study};
//!
//! let sampler = CmaEsSampler::with_seed(42);
//! let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
//! ```

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::distribution::Distribution;
use crate::param::ParamValue;
use crate::sampler::{CompletedTrial, Sampler};

/// CMA-ES sampler for continuous optimization.
///
/// Adapts a multivariate Gaussian to concentrate around promising regions
/// of the search space. Best suited for continuous (float/int) parameters
/// in moderate dimensions (up to ~100).
///
/// # Examples
///
/// ```
/// use optimizer::sampler::cma_es::CmaEsSampler;
/// use optimizer::{Direction, Study};
///
/// // Default configuration
/// let study: Study<f64> = Study::with_sampler(Direction::Minimize, CmaEsSampler::new());
///
/// // With seed for reproducibility
/// let study: Study<f64> = Study::with_sampler(Direction::Minimize, CmaEsSampler::with_seed(42));
///
/// // Custom configuration via builder
/// let sampler = CmaEsSampler::builder()
///     .sigma0(0.5)
///     .population_size(20)
///     .seed(42)
///     .build();
/// let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);
/// ```
pub struct CmaEsSampler {
    state: Mutex<CmaEsState>,
}

impl CmaEsSampler {
    /// Creates a new CMA-ES sampler with a random seed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(CmaEsState::new(None, None, None)),
        }
    }

    /// Creates a new CMA-ES sampler with a fixed seed for reproducibility.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            state: Mutex::new(CmaEsState::new(None, None, Some(seed))),
        }
    }

    /// Creates a builder for configuring a `CmaEsSampler`.
    #[must_use]
    pub fn builder() -> CmaEsSamplerBuilder {
        CmaEsSamplerBuilder::new()
    }
}

impl Default for CmaEsSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for configuring a [`CmaEsSampler`].
///
/// All options have sensible defaults:
/// - `sigma0`: auto-computed as average range / 4
/// - `population_size`: `4 + floor(3 * ln(n))`
/// - `seed`: random
///
/// # Examples
///
/// ```
/// use optimizer::sampler::cma_es::CmaEsSamplerBuilder;
///
/// let sampler = CmaEsSamplerBuilder::new()
///     .sigma0(0.3)
///     .population_size(10)
///     .seed(42)
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct CmaEsSamplerBuilder {
    sigma0: Option<f64>,
    population_size: Option<usize>,
    seed: Option<u64>,
}

impl CmaEsSamplerBuilder {
    /// Creates a new builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the initial step size (sigma).
    ///
    /// Controls the initial spread of the search distribution. Larger values
    /// explore more broadly; smaller values search more locally.
    ///
    /// Default: `average_range / 4` (auto-computed from parameter bounds).
    #[must_use]
    pub fn sigma0(mut self, sigma0: f64) -> Self {
        self.sigma0 = Some(sigma0);
        self
    }

    /// Sets the population size (lambda).
    ///
    /// Number of candidate solutions evaluated per generation.
    /// Larger populations improve robustness but increase the number of
    /// trials per generation.
    ///
    /// Default: `4 + floor(3 * ln(n))` where `n` is the number of continuous dimensions.
    #[must_use]
    pub fn population_size(mut self, population_size: usize) -> Self {
        self.population_size = Some(population_size);
        self
    }

    /// Sets the random seed for reproducibility.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Builds the configured [`CmaEsSampler`].
    #[must_use]
    pub fn build(self) -> CmaEsSampler {
        CmaEsSampler {
            state: Mutex::new(CmaEsState::new(
                self.sigma0,
                self.population_size,
                self.seed,
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Describes how a parameter dimension maps into the CMA-ES internal vector.
#[derive(Clone, Debug)]
struct DimensionInfo {
    /// The distribution for this dimension (stored for decoding).
    distribution: Distribution,
    /// Whether this dimension participates in CMA-ES (Float/Int = true, Categorical = false).
    is_continuous: bool,
    /// Internal-space bounds for continuous dimensions: `(low, high)`.
    /// For log-scale parameters these are in log-space.
    bounds: Option<(f64, f64)>,
}

/// A candidate solution produced by the CMA-ES distribution.
#[derive(Clone, Debug)]
struct Candidate {
    /// Internal-space vector (only continuous dimensions).
    x: DVector<f64>,
    /// Values for categorical dimensions (index in `dimensions` → categorical index).
    categorical_values: HashMap<usize, usize>,
}

/// Tracks per-trial sampling progress.
#[derive(Clone, Debug)]
struct TrialProgress {
    /// Index of the candidate assigned to this trial.
    candidate_idx: usize,
    /// Next dimension to return for this trial.
    next_dim: usize,
}

/// The CMA-ES algorithm constants, derived from dimension count.
#[derive(Clone, Debug)]
struct CmaEsConstants {
    /// Dimension of the continuous search space.
    n: usize,
    /// Population size (lambda).
    lambda: usize,
    /// Parent count (mu = lambda / 2).
    mu: usize,
    /// Recombination weights (length mu).
    weights: Vec<f64>,
    /// Variance effective selection mass.
    mu_eff: f64,
    /// Learning rate for the cumulation of the step-size control.
    c_sigma: f64,
    /// Damping for sigma.
    d_sigma: f64,
    /// Learning rate for the cumulation of the rank-one update.
    c_c: f64,
    /// Learning rate for the rank-one update of C.
    c_1: f64,
    /// Learning rate for the rank-mu update of C.
    c_mu: f64,
    /// Expected norm of N(0, I) in n dimensions.
    chi_n: f64,
}

/// The mutable CMA-ES algorithm state.
struct CmaEsAlgorithm {
    /// Distribution mean.
    mean: DVector<f64>,
    /// Step size.
    sigma: f64,
    /// Covariance matrix.
    c: DMatrix<f64>,
    /// Evolution path for sigma.
    p_sigma: DVector<f64>,
    /// Evolution path for rank-one update.
    p_c: DVector<f64>,
    /// Eigenvectors of C (columns of B).
    b: DMatrix<f64>,
    /// Sqrt of eigenvalues of C (diagonal of D).
    d: DVector<f64>,
    /// C^{-1/2} for sigma path update.
    inv_sqrt_c: DMatrix<f64>,
    /// Generation counter (for eigendecomposition scheduling).
    generation: usize,
    /// Last generation at which eigendecomposition was performed.
    last_eigen_generation: usize,
    /// Algorithm constants.
    constants: CmaEsConstants,
}

/// Phase of the CMA-ES state machine.
enum Phase {
    /// Discovering the search space structure (first trial).
    Discovery,
    /// Steady-state sampling and updating.
    Active(Box<CmaEsAlgorithm>),
}

/// Top-level mutable state behind the `Mutex`.
struct CmaEsState {
    /// The RNG used for sampling.
    rng: StdRng,
    /// User-provided initial sigma (None = auto).
    sigma0: Option<f64>,
    /// User-provided population size (None = auto).
    user_lambda: Option<usize>,
    /// Current phase.
    phase: Phase,
    /// Discovered dimension info (populated during discovery).
    dimensions: Vec<DimensionInfo>,
    /// Current generation's candidates.
    candidates: Vec<Candidate>,
    /// Mapping from `trial_id` to its progress.
    trial_progress: HashMap<u64, TrialProgress>,
    /// Number of candidates assigned so far in the current generation.
    assigned_count: usize,
    /// Trial IDs assigned in the current generation (for tracking completion).
    generation_trial_ids: Vec<u64>,
    /// Last `trial_id` seen during discovery (to detect when first trial ends).
    discovery_trial_id: Option<u64>,
}

impl CmaEsState {
    fn new(sigma0: Option<f64>, user_lambda: Option<usize>, seed: Option<u64>) -> Self {
        let rng = seed.map_or_else(rand::make_rng, StdRng::seed_from_u64);
        Self {
            rng,
            sigma0,
            user_lambda,
            phase: Phase::Discovery,
            dimensions: Vec::new(),
            candidates: Vec::new(),
            trial_progress: HashMap::new(),
            assigned_count: 0,
            generation_trial_ids: Vec::new(),
            discovery_trial_id: None,
        }
    }
}

// ---------------------------------------------------------------------------
// CMA-ES algorithm helpers
// ---------------------------------------------------------------------------

impl CmaEsConstants {
    /// Compute all CMA-ES constants from dimension count and population size.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn new(n: usize, user_lambda: Option<usize>) -> Self {
        let n_f = n as f64;

        // Population size
        let lambda = user_lambda.unwrap_or_else(|| 4 + (3.0 * n_f.ln()).max(0.0).floor() as usize);
        let lambda = lambda.max(4); // ensure at least 4

        // Parent count
        let mu = lambda / 2;

        // Recombination weights (log-proportional)
        let log_half_lambda = f64::midpoint(lambda as f64, 1.0).ln();
        let raw_weights: Vec<f64> = (0..mu)
            .map(|i| log_half_lambda - ((i + 1) as f64).ln())
            .collect();
        let w_sum: f64 = raw_weights.iter().sum();
        let weights: Vec<f64> = raw_weights.iter().map(|w| w / w_sum).collect();

        // Variance effective selection mass
        let w_sq_sum: f64 = weights.iter().map(|w| w * w).sum();
        let mu_eff = 1.0 / w_sq_sum;

        // Learning rates
        let c_sigma = (mu_eff + 2.0) / (n_f + mu_eff + 5.0);
        let d_sigma = 1.0 + 2.0 * (((mu_eff - 1.0) / (n_f + 1.0)).sqrt() - 1.0).max(0.0) + c_sigma;
        let c_c = (4.0 + mu_eff / n_f) / (n_f + 4.0 + 2.0 * mu_eff / n_f);
        let c_1 = 2.0 / ((n_f + 1.3).powi(2) + mu_eff);
        let c_mu_raw = (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff)) / ((n_f + 2.0).powi(2) + mu_eff);
        let c_mu = c_mu_raw.min(1.0 - c_1);

        // Expected norm of N(0, I)
        let chi_n = n_f.sqrt() * (1.0 - 1.0 / (4.0 * n_f) + 1.0 / (21.0 * n_f * n_f));

        Self {
            n,
            lambda,
            mu,
            weights,
            mu_eff,
            c_sigma,
            d_sigma,
            c_c,
            c_1,
            c_mu,
            chi_n,
        }
    }
}

impl CmaEsAlgorithm {
    /// Initialize the CMA-ES algorithm state.
    #[allow(clippy::cast_precision_loss)]
    fn new(dimensions: &[DimensionInfo], sigma0: Option<f64>, user_lambda: Option<usize>) -> Self {
        let n = dimensions.iter().filter(|d| d.is_continuous).count();

        let constants = CmaEsConstants::new(n, user_lambda);

        // Compute initial mean (center of bounds) and auto sigma
        let mut mean = DVector::zeros(n);
        let mut total_range = 0.0;
        let mut ci = 0;
        for dim in dimensions {
            if dim.is_continuous {
                if let Some((lo, hi)) = dim.bounds {
                    mean[ci] = f64::midpoint(lo, hi);
                    total_range += hi - lo;
                }
                ci += 1;
            }
        }

        let sigma = sigma0.unwrap_or_else(|| {
            if n > 0 {
                (total_range / n as f64) / 4.0
            } else {
                1.0
            }
        });

        let c = DMatrix::identity(n, n);
        let p_sigma = DVector::zeros(n);
        let p_c = DVector::zeros(n);
        let b = DMatrix::identity(n, n);
        let d = DVector::from_element(n, 1.0);
        let inv_sqrt_c = DMatrix::identity(n, n);

        Self {
            mean,
            sigma,
            c,
            p_sigma,
            p_c,
            b,
            d,
            inv_sqrt_c,
            generation: 0,
            last_eigen_generation: 0,
            constants,
        }
    }

    /// Generate `lambda` candidate vectors from the current distribution.
    fn generate_candidates(
        &self,
        rng: &mut StdRng,
        dimensions: &[DimensionInfo],
    ) -> Vec<Candidate> {
        let n = self.constants.n;
        let lambda = self.constants.lambda;
        let mut candidates = Vec::with_capacity(lambda);

        for _ in 0..lambda {
            let candidate = self.generate_single_candidate(rng, dimensions, n);
            candidates.push(candidate);
        }

        candidates
    }

    /// Generate a single candidate from the current distribution.
    fn generate_single_candidate(
        &self,
        rng: &mut StdRng,
        dimensions: &[DimensionInfo],
        n: usize,
    ) -> Candidate {
        // x = mean + sigma * B * D * z where z ~ N(0, I)
        let x = self.sample_with_rejection(rng, dimensions, n);

        // Sample categorical dimensions randomly
        let mut categorical_values = HashMap::new();
        for (i, dim) in dimensions.iter().enumerate() {
            if !dim.is_continuous
                && let Distribution::Categorical(cat) = &dim.distribution
            {
                categorical_values.insert(i, rng.random_range(0..cat.n_choices));
            }
        }

        Candidate {
            x,
            categorical_values,
        }
    }

    /// Sample a candidate vector with rejection sampling for bounds.
    fn sample_with_rejection(
        &self,
        rng: &mut StdRng,
        dimensions: &[DimensionInfo],
        n: usize,
    ) -> DVector<f64> {
        let max_attempts = 100;
        for _ in 0..max_attempts {
            let z = DVector::from_fn(n, |_, _| sample_standard_normal(rng));
            let x = &self.mean + self.sigma * (&self.b * self.d.component_mul(&z));

            if is_within_bounds(&x, dimensions) {
                return x;
            }
        }

        // Fallback: clip to bounds
        let z = DVector::from_fn(n, |_, _| sample_standard_normal(rng));
        let mut x = &self.mean + self.sigma * (&self.b * self.d.component_mul(&z));
        clip_to_bounds(&mut x, dimensions);
        x
    }

    /// Run the CMA-ES update step given ranked candidates.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn update(&mut self, ranked_candidates: &[&DVector<f64>]) {
        let n = self.constants.n;
        let mu = self.constants.mu;
        let sigma = self.sigma;

        // New mean = weighted sum of top-mu candidates
        let mut new_mean = DVector::zeros(n);
        for (i, &x) in ranked_candidates.iter().take(mu).enumerate() {
            new_mean += self.constants.weights[i] * x;
        }

        // Displacement in mean
        let mean_diff = &new_mean - &self.mean;

        // Update p_sigma (cumulation for sigma control)
        let inv_sqrt_c_times_diff = &self.inv_sqrt_c * &mean_diff / sigma;
        self.p_sigma = (1.0 - self.constants.c_sigma) * &self.p_sigma
            + (self.constants.c_sigma * (2.0 - self.constants.c_sigma) * self.constants.mu_eff)
                .sqrt()
                * &inv_sqrt_c_times_diff;

        // h_sigma: stall indicator
        let p_sigma_norm = self.p_sigma.norm();
        let threshold =
            (1.0 - (1.0 - self.constants.c_sigma).powi(2 * (self.generation as i32 + 1))).sqrt()
                * (1.4 + 2.0 / (n as f64 + 1.0))
                * self.constants.chi_n;
        let h_sigma = if p_sigma_norm < threshold { 1.0 } else { 0.0 };

        // Update p_c (cumulation for rank-one update)
        self.p_c = (1.0 - self.constants.c_c) * &self.p_c
            + h_sigma
                * (self.constants.c_c * (2.0 - self.constants.c_c) * self.constants.mu_eff).sqrt()
                * &mean_diff
                / sigma;

        // Rank-one and rank-mu update of C
        let delta_h = (1.0 - h_sigma) * self.constants.c_c * (2.0 - self.constants.c_c);
        let old_c_weight =
            1.0 - self.constants.c_1 - self.constants.c_mu + self.constants.c_1 * delta_h;

        // Rank-one term
        let rank_one = self.constants.c_1 * &self.p_c * self.p_c.transpose();

        // Rank-mu term
        let mut rank_mu = DMatrix::zeros(n, n);
        for (i, &x) in ranked_candidates.iter().take(mu).enumerate() {
            let y = (x - &self.mean) / sigma;
            rank_mu += self.constants.weights[i] * &y * y.transpose();
        }
        let rank_mu = self.constants.c_mu * rank_mu;

        self.c = old_c_weight * &self.c + rank_one + rank_mu;

        // Update sigma via CSA
        self.sigma *= ((self.constants.c_sigma / self.constants.d_sigma)
            * (p_sigma_norm / self.constants.chi_n - 1.0))
            .exp();
        self.sigma = self.sigma.clamp(1e-20, 1e10);

        // Update mean
        self.mean = new_mean;
        self.generation += 1;

        // Eigendecomposition (every n/10 generations, minimum every generation for small n)
        let eigen_interval = (n / 10).max(1);
        if self.generation - self.last_eigen_generation >= eigen_interval {
            self.update_eigen();
        }
    }

    /// Perform eigendecomposition of C and update B, D, `inv_sqrt_c`.
    fn update_eigen(&mut self) {
        let n = self.constants.n;

        // Enforce symmetry
        self.c = (&self.c + self.c.transpose()) / 2.0;

        // Eigendecomposition
        let eigen = self.c.clone().symmetric_eigen();
        let eigenvalues = &eigen.eigenvalues;
        let eigenvectors = &eigen.eigenvectors;

        // Clamp eigenvalues for numerical stability
        let mut d_vec = DVector::zeros(n);
        for i in 0..n {
            d_vec[i] = eigenvalues[i].max(1e-20).sqrt();
        }

        self.b = eigenvectors.clone();
        self.d = d_vec;

        // Compute C^{-1/2} = B * D^{-1} * B^T
        let d_inv = DVector::from_fn(n, |i, _| 1.0 / self.d[i]);
        let d_inv_diag = DMatrix::from_diagonal(&d_inv);
        self.inv_sqrt_c = &self.b * d_inv_diag * self.b.transpose();

        self.last_eigen_generation = self.generation;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check whether a continuous candidate vector is within bounds.
fn is_within_bounds(x: &DVector<f64>, dimensions: &[DimensionInfo]) -> bool {
    let mut ci = 0;
    for dim in dimensions {
        if dim.is_continuous {
            if let Some((lo, hi)) = dim.bounds
                && (x[ci] < lo || x[ci] > hi)
            {
                return false;
            }
            ci += 1;
        }
    }
    true
}

/// Clip a continuous candidate vector to bounds.
fn clip_to_bounds(x: &mut DVector<f64>, dimensions: &[DimensionInfo]) {
    let mut ci = 0;
    for dim in dimensions {
        if dim.is_continuous {
            if let Some((lo, hi)) = dim.bounds {
                x[ci] = x[ci].clamp(lo, hi);
            }
            ci += 1;
        }
    }
}

/// Convert an internal-space value back to a `ParamValue`.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn from_internal(value: f64, distribution: &Distribution) -> ParamValue {
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
                d.low + k * step
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

/// Compute internal-space bounds for a distribution.
#[allow(clippy::cast_precision_loss)]
fn internal_bounds(distribution: &Distribution) -> Option<(f64, f64)> {
    match distribution {
        Distribution::Float(d) => {
            if d.log_scale {
                Some((d.low.ln(), d.high.ln()))
            } else {
                Some((d.low, d.high))
            }
        }
        Distribution::Int(d) => {
            if d.log_scale {
                Some(((d.low as f64).ln(), (d.high as f64).ln()))
            } else {
                Some((d.low as f64, d.high as f64))
            }
        }
        Distribution::Categorical(_) => None,
    }
}

/// Sample a value from the standard normal distribution using Box-Muller transform.
fn sample_standard_normal(rng: &mut StdRng) -> f64 {
    // Box-Muller transform
    let u1: f64 = rng.random_range(f64::EPSILON..=1.0);
    let u2: f64 = rng.random_range(0.0_f64..=core::f64::consts::TAU);
    (-2.0 * u1.ln()).sqrt() * u2.cos()
}

/// Sample a categorical value randomly.
fn sample_random_categorical(rng: &mut StdRng, distribution: &Distribution) -> ParamValue {
    match distribution {
        Distribution::Categorical(d) => ParamValue::Categorical(rng.random_range(0..d.n_choices)),
        _ => unreachable!("sample_random_categorical called with non-categorical distribution"),
    }
}

/// Sample a random value for any distribution (used during discovery phase).
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn sample_random(rng: &mut StdRng, distribution: &Distribution) -> ParamValue {
    match distribution {
        Distribution::Float(d) => {
            let value = if d.log_scale {
                let log_low = d.low.ln();
                let log_high = d.high.ln();
                rng.random_range(log_low..=log_high).exp()
            } else if let Some(step) = d.step {
                let n_steps = ((d.high - d.low) / step).floor() as i64;
                let k = rng.random_range(0..=n_steps);
                d.low + (k as f64) * step
            } else {
                rng.random_range(d.low..=d.high)
            };
            ParamValue::Float(value)
        }
        Distribution::Int(d) => {
            let value = if d.log_scale {
                let log_low = (d.low as f64).ln();
                let log_high = (d.high as f64).ln();
                let raw = rng.random_range(log_low..=log_high).exp().round() as i64;
                raw.clamp(d.low, d.high)
            } else if let Some(step) = d.step {
                let n_steps = (d.high - d.low) / step;
                let k = rng.random_range(0..=n_steps);
                d.low + k * step
            } else {
                rng.random_range(d.low..=d.high)
            };
            ParamValue::Int(value)
        }
        Distribution::Categorical(d) => ParamValue::Categorical(rng.random_range(0..d.n_choices)),
    }
}

// ---------------------------------------------------------------------------
// Sampler trait implementation
// ---------------------------------------------------------------------------

impl Sampler for CmaEsSampler {
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn sample(
        &self,
        distribution: &Distribution,
        trial_id: u64,
        history: &[CompletedTrial],
    ) -> ParamValue {
        let mut state = self.state.lock();

        match &state.phase {
            Phase::Discovery => sample_discovery(&mut state, distribution, trial_id),
            Phase::Active(_) => sample_active(&mut state, distribution, trial_id, history),
        }
    }
}

/// Handle sampling during the discovery phase.
fn sample_discovery(
    state: &mut CmaEsState,
    distribution: &Distribution,
    trial_id: u64,
) -> ParamValue {
    // Check if this is a new trial (discovery phase ended for previous trial)
    if let Some(prev_id) = state.discovery_trial_id
        && trial_id != prev_id
    {
        // First trial is done; we know the search space. Initialize CMA-ES.
        finalize_discovery(state);
        // Now delegate to active sampling
        return sample_active(state, distribution, trial_id, &[]);
    }

    // Record this trial_id
    state.discovery_trial_id = Some(trial_id);

    // Record this dimension
    let is_continuous = !matches!(distribution, Distribution::Categorical(_));
    let bounds = internal_bounds(distribution);
    state.dimensions.push(DimensionInfo {
        distribution: distribution.clone(),
        is_continuous,
        bounds,
    });

    // Sample randomly for the discovery trial
    sample_random(&mut state.rng, distribution)
}

/// Finalize discovery and transition to the active phase.
fn finalize_discovery(state: &mut CmaEsState) {
    let n_continuous = state.dimensions.iter().filter(|d| d.is_continuous).count();

    let algo = CmaEsAlgorithm::new(&state.dimensions, state.sigma0, state.user_lambda);

    // Generate first batch of candidates
    let candidates = if n_continuous > 0 {
        algo.generate_candidates(&mut state.rng, &state.dimensions)
    } else {
        // Pure categorical: generate random categorical candidates
        generate_pure_categorical_candidates(
            &mut state.rng,
            &state.dimensions,
            algo.constants.lambda,
        )
    };

    state.candidates = candidates;
    state.assigned_count = 0;
    state.generation_trial_ids.clear();
    state.trial_progress.clear();
    state.phase = Phase::Active(Box::new(algo));
}

/// Generate candidates that are purely categorical (no continuous dims).
fn generate_pure_categorical_candidates(
    rng: &mut StdRng,
    dimensions: &[DimensionInfo],
    lambda: usize,
) -> Vec<Candidate> {
    (0..lambda)
        .map(|_| {
            let mut categorical_values = HashMap::new();
            for (i, dim) in dimensions.iter().enumerate() {
                if let Distribution::Categorical(cat) = &dim.distribution {
                    categorical_values.insert(i, rng.random_range(0..cat.n_choices));
                }
            }
            Candidate {
                x: DVector::zeros(0),
                categorical_values,
            }
        })
        .collect()
}

/// Handle sampling during the active phase.
fn sample_active(
    state: &mut CmaEsState,
    distribution: &Distribution,
    trial_id: u64,
    history: &[CompletedTrial],
) -> ParamValue {
    // Check if we need to update (all generation candidates assigned and completed)
    maybe_update_generation(state, history);

    // Assign a candidate to this trial if not yet done
    if !state.trial_progress.contains_key(&trial_id) {
        assign_candidate(state, trial_id);
    }

    let progress = state.trial_progress.get_mut(&trial_id).unwrap();
    let dim_idx = progress.next_dim;
    progress.next_dim += 1;

    // Safety check: dim_idx should be within dimensions
    if dim_idx >= state.dimensions.len() {
        // Extra dimension not seen during discovery; sample randomly
        return sample_random(&mut state.rng, distribution);
    }

    let candidate = &state.candidates[progress.candidate_idx];
    let dim_info = &state.dimensions[dim_idx];

    if dim_info.is_continuous {
        // Map from internal continuous index to the candidate's x vector
        let ci = state.dimensions[..dim_idx]
            .iter()
            .filter(|d| d.is_continuous)
            .count();
        if ci < candidate.x.len() {
            from_internal(candidate.x[ci], &dim_info.distribution)
        } else {
            sample_random(&mut state.rng, distribution)
        }
    } else {
        // Categorical: use pre-sampled value
        if let Some(&cat_idx) = candidate.categorical_values.get(&dim_idx) {
            ParamValue::Categorical(cat_idx)
        } else {
            sample_random_categorical(&mut state.rng, distribution)
        }
    }
}

/// Assign a candidate to a trial.
fn assign_candidate(state: &mut CmaEsState, trial_id: u64) {
    let candidate_idx = if state.assigned_count < state.candidates.len() {
        let idx = state.assigned_count;
        state.assigned_count += 1;
        idx
    } else {
        // Overflow: generate an extra candidate from current distribution
        let extra = generate_overflow_candidate(state);
        state.candidates.push(extra);
        let idx = state.candidates.len() - 1;
        state.assigned_count = state.candidates.len();
        idx
    };

    state.trial_progress.insert(
        trial_id,
        TrialProgress {
            candidate_idx,
            next_dim: 0,
        },
    );
    state.generation_trial_ids.push(trial_id);
}

/// Generate an overflow candidate from the current distribution.
fn generate_overflow_candidate(state: &mut CmaEsState) -> Candidate {
    let n_continuous = state.dimensions.iter().filter(|d| d.is_continuous).count();

    if n_continuous == 0 {
        let mut categorical_values = HashMap::new();
        for (i, dim) in state.dimensions.iter().enumerate() {
            if let Distribution::Categorical(cat) = &dim.distribution {
                categorical_values.insert(i, state.rng.random_range(0..cat.n_choices));
            }
        }
        return Candidate {
            x: DVector::zeros(0),
            categorical_values,
        };
    }

    match &state.phase {
        Phase::Active(algo) => {
            algo.generate_single_candidate(&mut state.rng, &state.dimensions, n_continuous)
        }
        Phase::Discovery => {
            // Should not happen, but handle gracefully
            Candidate {
                x: DVector::zeros(n_continuous),
                categorical_values: HashMap::new(),
            }
        }
    }
}

/// Check if we should run the CMA-ES update and generate a new generation.
fn maybe_update_generation(state: &mut CmaEsState, history: &[CompletedTrial]) {
    let Phase::Active(algo) = &state.phase else {
        return;
    };

    let lambda = algo.constants.lambda;
    let n_continuous = algo.constants.n;

    // Only update when at least lambda candidates have been assigned
    if state.generation_trial_ids.len() < lambda {
        return;
    }

    // Check if the first lambda trial IDs are all completed
    let trial_ids: Vec<u64> = state
        .generation_trial_ids
        .iter()
        .take(lambda)
        .copied()
        .collect();
    let history_ids: HashMap<u64, f64> = history.iter().map(|t| (t.id, t.value)).collect();

    let all_completed = trial_ids.iter().all(|id| history_ids.contains_key(id));
    if !all_completed {
        return;
    }

    // No continuous dimensions: just regenerate random candidates
    if n_continuous == 0 {
        state.candidates =
            generate_pure_categorical_candidates(&mut state.rng, &state.dimensions, lambda);
        state.assigned_count = 0;
        state.generation_trial_ids.clear();
        state.trial_progress.clear();
        return;
    }

    // Collect (candidate_x, value) for the generation
    let mut ranked: Vec<(&DVector<f64>, f64)> = trial_ids
        .iter()
        .filter_map(|id| {
            let progress = state.trial_progress.get(id)?;
            let value = *history_ids.get(id)?;
            Some((&state.candidates[progress.candidate_idx].x, value))
        })
        .collect();

    // Sort ascending (minimize)
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

    let ranked_xs: Vec<&DVector<f64>> = ranked.iter().map(|(x, _)| *x).collect();

    // Run update
    let Phase::Active(algo) = &mut state.phase else {
        return;
    };
    algo.update(&ranked_xs);

    // Generate new candidates
    let new_candidates = algo.generate_candidates(&mut state.rng, &state.dimensions);
    state.candidates = new_candidates;
    state.assigned_count = 0;
    state.generation_trial_ids.clear();
    state.trial_progress.clear();
}
