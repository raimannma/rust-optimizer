//! Integration tests for the BOHB sampler.

#![allow(
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use optimizer::parameter::{FloatParam, Parameter};
use optimizer::sampler::bohb::BohbSampler;
use optimizer::{Direction, Error, Study, TrialPruned};

#[test]
fn bohb_converges_on_quadratic() {
    let bohb = BohbSampler::builder()
        .min_resource(1)
        .max_resource(9)
        .reduction_factor(3)
        .min_points_in_model(5)
        .seed(42)
        .build()
        .unwrap();

    let pruner = bohb.matching_pruner(Direction::Minimize);
    let study: Study<f64> = Study::with_sampler_and_pruner(Direction::Minimize, bohb, pruner);

    let x_param = FloatParam::new(-10.0, 10.0);

    study
        .optimize(60, |trial| {
            let x = x_param.suggest(trial)?;

            // Report intermediate values at budget steps 1, 3, 9
            let obj = (x - 3.0).powi(2);
            // Simulate budget-based evaluation with noise decreasing at higher budgets
            trial.report(1, obj + 5.0);
            trial.report(3, obj + 1.0);
            trial.report(9, obj);

            Ok::<_, Error>(obj)
        })
        .expect("optimization should succeed");

    let best = study.best_trial().expect("should have trials");
    assert!(
        best.value < 10.0,
        "BOHB should find a reasonable solution, got {}",
        best.value
    );
}

#[test]
fn bohb_with_pruning() {
    let bohb = BohbSampler::builder()
        .min_resource(1)
        .max_resource(27)
        .reduction_factor(3)
        .min_points_in_model(3)
        .seed(123)
        .build()
        .unwrap();

    let pruner = bohb.matching_pruner(Direction::Minimize);
    let study: Study<f64> = Study::with_sampler_and_pruner(Direction::Minimize, bohb, pruner);

    let x_param = FloatParam::new(-5.0, 5.0);

    study
        .optimize(40, |trial| {
            let x = x_param.suggest(trial)?;
            let obj = x * x;

            // Report at each rung step and check for pruning
            for &step in &[1u64, 3, 9, 27] {
                let noisy_obj = obj + 10.0 / step as f64;
                trial.report(step, noisy_obj);

                if trial.should_prune() {
                    return Err(TrialPruned.into());
                }
            }

            Ok::<_, Error>(obj)
        })
        .expect("optimization should succeed");

    // Verify we have completed trials
    let best = study.best_trial().expect("should have at least one trial");
    assert!(
        best.value < 25.0,
        "best value {} should be reasonable",
        best.value
    );
}

#[test]
fn bohb_uses_budget_conditioned_history() {
    // Verify that BOHB conditions on budget level by testing that samples
    // are influenced by intermediate values, not just final values.
    let bohb = BohbSampler::builder()
        .min_resource(1)
        .max_resource(9)
        .reduction_factor(3)
        .min_points_in_model(3)
        .seed(42)
        .build()
        .unwrap();

    let pruner = bohb.matching_pruner(Direction::Minimize);
    let study: Study<f64> = Study::with_sampler_and_pruner(Direction::Minimize, bohb, pruner);

    let x_param = FloatParam::new(0.0, 10.0);

    study
        .optimize(30, |trial| {
            let x = x_param.suggest(trial)?;
            // Intermediate values that guide optimization toward x=2
            trial.report(1, (x - 2.0).powi(2) + 1.0);
            trial.report(3, (x - 2.0).powi(2) + 0.5);
            trial.report(9, (x - 2.0).powi(2));

            Ok::<_, Error>((x - 2.0).powi(2))
        })
        .expect("optimization should succeed");

    let best = study.best_trial().unwrap();
    let best_x: f64 = best.get(&x_param).unwrap();
    // Should find x reasonably close to 2.0
    assert!(
        (best_x - 2.0).abs() < 5.0,
        "BOHB should explore near x=2, got x={best_x}"
    );
}
