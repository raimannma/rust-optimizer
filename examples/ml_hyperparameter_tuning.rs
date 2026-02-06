//! Machine Learning Hyperparameter Tuning Example
//!
//! This example shows how to use the optimizer library to find the best
//! hyperparameters for a machine learning model. We simulate a gradient
//! boosting model (like XGBoost or LightGBM) and search for optimal settings.
//!
//! # Key Concepts Demonstrated
//!
//! - Creating a Study with a TPE (Tree-Parzen Estimator) sampler
//! - Defining an objective function that the optimizer will minimize
//! - Using different parameter types: floats, integers, log-scale, stepped
//! - Using callbacks to monitor progress and implement early stopping
//!
//! # How It Works
//!
//! 1. Create a `Study` - this manages the optimization process
//! 2. Define an objective function that takes a `Trial` and returns a score
//! 3. Inside the objective, use `trial.suggest_*()` to sample parameters
//! 4. The optimizer runs many trials, learning which parameter regions work best
//! 5. After optimization, retrieve the best parameters found
//!
//! Run with: `cargo run --example ml_hyperparameter_tuning`

use std::ops::ControlFlow;

use optimizer::parameter::{FloatParam, IntParam, Parameter};
use optimizer::sampler::CompletedTrial;
use optimizer::sampler::tpe::TpeSampler;
use optimizer::{Direction, ParamValue, Study, Trial};

// ============================================================================
// Configuration: Hyperparameters we want to tune
// ============================================================================

/// Holds all the hyperparameters for our model.
///
/// In a real application, you would pass these to your ML framework
/// (e.g., XGBoost, LightGBM, scikit-learn).
struct ModelConfig {
    learning_rate: f64,
    max_depth: i64,
    n_estimators: i64,
    subsample: f64,
    colsample_bytree: f64,
    min_child_weight: i64,
    reg_alpha: f64,
    reg_lambda: f64,
}

// ============================================================================
// Objective Function: What we want to optimize
// ============================================================================

/// Simulates training a model and returns the validation loss.
///
/// In a real scenario, this function would:
/// 1. Create a model with the given hyperparameters
/// 2. Train it on your training data
/// 3. Evaluate it on validation data
/// 4. Return the validation metric (e.g., RMSE, log loss, accuracy)
///
/// The optimizer will try to MINIMIZE this value (we set Direction::Minimize).
#[allow(clippy::too_many_arguments)]
fn evaluate_model(config: &ModelConfig) -> f64 {
    // Simulated optimal hyperparameters:
    // learning_rate ~ 0.05, max_depth ~ 6, n_estimators ~ 200
    // subsample ~ 0.8, colsample_bytree ~ 0.8, min_child_weight ~ 3
    // reg_alpha ~ 0.1, reg_lambda ~ 1.0

    let mut loss = 0.15; // Base loss

    // Each term penalizes deviation from the optimal value
    loss += (config.learning_rate - 0.05).powi(2) * 100.0;
    loss += ((config.max_depth - 6) as f64).powi(2) * 0.01;
    loss += ((config.n_estimators - 200) as f64).powi(2) * 0.00001;
    loss += (config.subsample - 0.8).powi(2) * 10.0;
    loss += (config.colsample_bytree - 0.8).powi(2) * 10.0;
    loss += ((config.min_child_weight - 3) as f64).powi(2) * 0.05;
    loss += (config.reg_alpha - 0.1).powi(2) * 5.0;
    loss += (config.reg_lambda - 1.0).powi(2) * 2.0;

    // Add some noise to simulate real-world variability
    let noise = (config.learning_rate * 1000.0).sin() * 0.01;

    loss + noise
}

/// The objective function that the optimizer calls for each trial.
///
/// This function:
/// 1. Uses parameter definitions passed as arguments
/// 2. Builds a model configuration from the suggested values
/// 3. Evaluates the model and returns the loss
///
/// The optimizer learns from the results to suggest better parameters
/// in future trials.
#[allow(clippy::too_many_arguments)]
fn objective(
    trial: &mut Trial,
    learning_rate_param: &FloatParam,
    max_depth_param: &IntParam,
    n_estimators_param: &IntParam,
    subsample_param: &FloatParam,
    colsample_bytree_param: &FloatParam,
    min_child_weight_param: &IntParam,
    reg_alpha_param: &FloatParam,
    reg_lambda_param: &FloatParam,
) -> optimizer::Result<f64> {
    let learning_rate = learning_rate_param.suggest(trial)?;
    let max_depth = max_depth_param.suggest(trial)?;
    let n_estimators = n_estimators_param.suggest(trial)?;
    let subsample = subsample_param.suggest(trial)?;
    let colsample_bytree = colsample_bytree_param.suggest(trial)?;
    let min_child_weight = min_child_weight_param.suggest(trial)?;
    let reg_alpha = reg_alpha_param.suggest(trial)?;
    let reg_lambda = reg_lambda_param.suggest(trial)?;

    // Build configuration and evaluate
    let config = ModelConfig {
        learning_rate,
        max_depth,
        n_estimators,
        subsample,
        colsample_bytree,
        min_child_weight,
        reg_alpha,
        reg_lambda,
    };

    let loss = evaluate_model(&config);

    Ok(loss)
}

// ============================================================================
// Callback Function: Monitor progress and implement early stopping
// ============================================================================

/// Called after each successful trial completes.
///
/// Use callbacks to:
/// - Log progress to console or file
/// - Save checkpoints
/// - Implement early stopping when a good solution is found
/// - Track metrics over time
///
/// Return `ControlFlow::Continue(())` to keep optimizing.
/// Return `ControlFlow::Break(())` to stop early.
fn on_trial_complete(study: &Study<f64>, trial: &CompletedTrial<f64>) -> ControlFlow<()> {
    // Print trial number and objective value
    print!("{:>5} ", study.n_trials());
    for value in trial.params.values() {
        match value {
            ParamValue::Float(v) => print!("{v:>12.5} "),
            ParamValue::Int(v) => print!("{v:>12} "),
            ParamValue::Categorical(v) => print!("{v:>12} "),
        }
    }
    println!("{:>12.6}", trial.value);

    // Early stopping: if we find an excellent solution, stop early
    if trial.value < 0.16 {
        println!("\nEarly stopping: found excellent solution!");
        return ControlFlow::Break(());
    }

    ControlFlow::Continue(())
}

// ============================================================================
// Main: Set up and run the optimization
// ============================================================================

fn main() -> optimizer::Result<()> {
    println!("=== ML Hyperparameter Tuning Example ===\n");

    // Step 1: Create a sampler
    //
    // TPE (Tree-Parzen Estimator) is a Bayesian optimization algorithm.
    // It learns from previous trials to suggest better parameters.
    // - n_startup_trials: Number of random trials before TPE kicks in
    // - gamma: What fraction of trials are considered "good" (lower = more selective)
    // - seed: For reproducibility
    let sampler = TpeSampler::builder()
        .n_startup_trials(10)
        .gamma(0.25)
        .seed(42)
        .build()
        .expect("Failed to build TPE sampler");

    // Step 2: Create a study
    //
    // The study manages the optimization process. We want to MINIMIZE
    // the loss (lower is better). Use Direction::Maximize for metrics
    // where higher is better (like accuracy).
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    // Print header
    println!("Starting hyperparameter optimization...\n");
    println!(
        "{:>5} {:>12} (parameters...)    {:>12}",
        "Trial", "Params", "Loss"
    );
    println!("{}", "-".repeat(60));

    // Step 3: Define parameter search spaces
    let learning_rate_param = FloatParam::new(0.001, 0.3).log_scale();
    let max_depth_param = IntParam::new(3, 12);
    let n_estimators_param = IntParam::new(50, 500).step(50);
    let subsample_param = FloatParam::new(0.5, 1.0);
    let colsample_bytree_param = FloatParam::new(0.5, 1.0);
    let min_child_weight_param = IntParam::new(1, 10);
    let reg_alpha_param = FloatParam::new(1e-3, 10.0).log_scale();
    let reg_lambda_param = FloatParam::new(1e-3, 10.0).log_scale();

    // Step 4: Run optimization
    //
    // optimize_with_callback_sampler runs the objective function for up to
    // n_trials iterations. After each trial, it calls the callback.
    // The "_sampler" suffix means the TPE sampler gets access to trial
    // history for informed sampling.
    let n_trials = 50;

    study.optimize_with_callback_sampler(
        n_trials,
        |trial| {
            objective(
                trial,
                &learning_rate_param,
                &max_depth_param,
                &n_estimators_param,
                &subsample_param,
                &colsample_bytree_param,
                &min_child_weight_param,
                &reg_alpha_param,
                &reg_lambda_param,
            )
        },
        on_trial_complete,
    )?;

    // Step 4: Get the best result
    println!("\n{}", "=".repeat(110));
    println!("\nOptimization completed!");
    println!("Total trials: {}", study.n_trials());

    let best = study.best_trial()?;
    println!("\nBest trial:");
    println!("  Loss: {:.6}", best.value);
    println!("  Parameters:");

    for (id, value) in &best.params {
        let label = best
            .param_labels
            .get(id)
            .map_or_else(|| format!("{id}"), |l| l.clone());
        match value {
            ParamValue::Float(v) => println!("    {label}: {v:.6}"),
            ParamValue::Int(v) => println!("    {label}: {v}"),
            ParamValue::Categorical(v) => println!("    {label}: category {v}"),
        }
    }

    // Step 5: Use the best parameters (in a real app)
    //
    // Now you would take best.params and use them to train your final model
    // on the full dataset.

    Ok(())
}
