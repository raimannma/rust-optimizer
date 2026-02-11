#[path = "../benches/test_functions.rs"]
mod test_functions;

use test_functions::*;

const TOL: f64 = 1e-10;

#[test]
fn sphere_at_optimum() {
    assert!(sphere(&[0.0, 0.0]).abs() < TOL);
    assert!(sphere(&[0.0; 10]).abs() < TOL);
}

#[test]
fn rosenbrock_at_optimum() {
    assert!(rosenbrock(&[1.0, 1.0]).abs() < TOL);
    assert!(rosenbrock(&[1.0; 5]).abs() < TOL);
}

#[test]
fn rastrigin_at_optimum() {
    assert!(rastrigin(&[0.0, 0.0]).abs() < TOL);
    assert!(rastrigin(&[0.0; 10]).abs() < TOL);
}

#[test]
fn ackley_at_optimum() {
    assert!(ackley(&[0.0, 0.0]).abs() < 1e-8);
    assert!(ackley(&[0.0; 10]).abs() < 1e-8);
}

#[test]
fn branin_at_optimum() {
    let target = 0.397_887_357_729_738_1;
    let val = branin(&[std::f64::consts::PI, 2.275]);
    assert!((val - target).abs() < 1e-3);
}

#[test]
fn hartmann6_at_optimum() {
    let target = -3.3224;
    let x_opt = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573];
    let val = hartmann6(&x_opt);
    assert!((val - target).abs() < 0.01);
}
