//! Standard optimization test functions for benchmarking.

/// Sphere function: unimodal, convex. Global minimum f(0,...,0) = 0.
pub fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

/// Rosenbrock function: narrow valley. Global minimum f(1,...,1) = 0.
pub fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

/// Rastrigin function: highly multimodal. Global minimum f(0,...,0) = 0.
pub fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n
        + x.iter()
            .map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>()
}

/// Ackley function: nearly flat with a deep well. Global minimum f(0,...,0) = 0.
pub fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_cos: f64 = x
        .iter()
        .map(|xi| (2.0 * std::f64::consts::PI * xi).cos())
        .sum();
    -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + 20.0 + std::f64::consts::E
}

/// Branin function (2D only). Three global minima with f* ≈ 0.397887.
///
/// # Panics
///
/// Panics if `x` does not have exactly 2 elements.
pub fn branin(x: &[f64]) -> f64 {
    assert!(x.len() == 2, "Branin requires exactly 2 dimensions");
    let (x1, x2) = (x[0], x[1]);
    let pi = std::f64::consts::PI;
    let a = 1.0;
    let b = 5.1 / (4.0 * pi * pi);
    let c = 5.0 / pi;
    let r = 6.0;
    let s = 10.0;
    let t = 1.0 / (8.0 * pi);
    a * (x2 - b * x1 * x1 + c * x1 - r).powi(2) + s * (1.0 - t) * x1.cos() + s
}

/// Hartmann 6D function. Global minimum f* ≈ -3.3224.
///
/// # Panics
///
/// Panics if `x` does not have exactly 6 elements.
pub fn hartmann6(x: &[f64]) -> f64 {
    assert!(x.len() == 6, "Hartmann6 requires exactly 6 dimensions");

    let alpha = [1.0, 1.2, 3.0, 3.2];
    let a_matrix = [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ];
    let p_matrix = [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ];

    let mut result = 0.0;
    for i in 0..4 {
        let mut inner = 0.0;
        for (j, xj) in x.iter().enumerate() {
            inner += a_matrix[i][j] * (xj - p_matrix[i][j]).powi(2);
        }
        result -= alpha[i] * (-inner).exp();
    }
    result
}
