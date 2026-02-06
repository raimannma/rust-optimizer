use optimizer::parameter::{
    BoolParam, Categorical, CategoricalParam, EnumParam, FloatParam, IntParam, Parameter,
};
use optimizer::{Direction, Study, Trial};

#[test]
fn suggest_float_param_via_trial() {
    let param = FloatParam::new(0.0, 1.0);
    let mut trial = Trial::new(0);
    let x = trial.suggest_param(&param).unwrap();
    assert!((0.0..=1.0).contains(&x));

    // Cached
    let x2 = trial.suggest_param(&param).unwrap();
    assert_eq!(x, x2);
}

#[test]
fn suggest_float_log_param_via_trial() {
    let param = FloatParam::new(1e-5, 1e-1).log_scale();
    let mut trial = Trial::new(0);
    let lr = trial.suggest_param(&param).unwrap();
    assert!((1e-5..=1e-1).contains(&lr));
}

#[test]
fn suggest_float_step_param_via_trial() {
    let param = FloatParam::new(0.0, 1.0).step(0.25);
    let mut trial = Trial::new(0);
    let x = trial.suggest_param(&param).unwrap();
    assert!((0.0..=1.0).contains(&x));
}

#[test]
fn suggest_int_param_via_trial() {
    let param = IntParam::new(1, 10);
    let mut trial = Trial::new(0);
    let n = trial.suggest_param(&param).unwrap();
    assert!((1..=10).contains(&n));

    // Cached
    let n2 = trial.suggest_param(&param).unwrap();
    assert_eq!(n, n2);
}

#[test]
fn suggest_int_log_param_via_trial() {
    let param = IntParam::new(1, 1024).log_scale();
    let mut trial = Trial::new(0);
    let batch = trial.suggest_param(&param).unwrap();
    assert!((1..=1024).contains(&batch));
}

#[test]
fn suggest_int_step_param_via_trial() {
    let param = IntParam::new(32, 512).step(32);
    let mut trial = Trial::new(0);
    let units = trial.suggest_param(&param).unwrap();
    assert!((32..=512).contains(&units));
    assert_eq!((units - 32) % 32, 0);
}

#[test]
fn suggest_categorical_param_via_trial() {
    let choices = vec!["sgd", "adam", "rmsprop"];
    let param = CategoricalParam::new(choices.clone());
    let mut trial = Trial::new(0);
    let opt = trial.suggest_param(&param).unwrap();
    assert!(choices.contains(&opt));

    // Cached
    let opt2 = trial.suggest_param(&param).unwrap();
    assert_eq!(opt, opt2);
}

#[test]
fn suggest_bool_param_via_trial() {
    let param = BoolParam::new();
    let mut trial = Trial::new(0);
    let val = trial.suggest_param(&param).unwrap();
    let _ = val;

    // Cached
    let val2 = trial.suggest_param(&param).unwrap();
    assert_eq!(val, val2);
}

#[derive(Clone, Debug, PartialEq)]
enum Activation {
    Relu,
    Sigmoid,
    Tanh,
}

impl Categorical for Activation {
    const N_CHOICES: usize = 3;

    fn from_index(index: usize) -> Self {
        match index {
            0 => Activation::Relu,
            1 => Activation::Sigmoid,
            2 => Activation::Tanh,
            _ => panic!("invalid index"),
        }
    }

    fn to_index(&self) -> usize {
        match self {
            Activation::Relu => 0,
            Activation::Sigmoid => 1,
            Activation::Tanh => 2,
        }
    }
}

#[test]
fn suggest_enum_param_via_trial() {
    let param = EnumParam::<Activation>::new();
    let mut trial = Trial::new(0);
    let act = trial.suggest_param(&param).unwrap();
    assert!([Activation::Relu, Activation::Sigmoid, Activation::Tanh].contains(&act));

    // Cached
    let act2 = trial.suggest_param(&param).unwrap();
    assert_eq!(act, act2);
}

#[test]
fn parameter_conflict_detection() {
    let float_param = FloatParam::new(0.0, 1.0);
    let int_param = IntParam::new(0, 10);
    let mut trial = Trial::new(0);
    let _ = trial.suggest_param(&float_param).unwrap();

    // Different param type with different id - no conflict
    let result = trial.suggest_param(&int_param);
    assert!(result.is_ok());

    // Different bounds for same param type but different id - no conflict
    let float_param2 = FloatParam::new(0.0, 2.0);
    let result = trial.suggest_param(&float_param2);
    assert!(result.is_ok());
}

#[test]
fn validation_prevents_suggest() {
    let mut trial = Trial::new(0);

    assert!(trial.suggest_param(&FloatParam::new(1.0, 0.0)).is_err());
    assert!(
        trial
            .suggest_param(&FloatParam::new(-1.0, 1.0).log_scale())
            .is_err()
    );
    assert!(
        trial
            .suggest_param(&FloatParam::new(0.0, 1.0).step(-0.1))
            .is_err()
    );
    assert!(trial.suggest_param(&IntParam::new(10, 1)).is_err());
    assert!(
        trial
            .suggest_param(&IntParam::new(0, 10).log_scale())
            .is_err()
    );
    assert!(trial.suggest_param(&IntParam::new(0, 10).step(-1)).is_err());
    assert!(
        trial
            .suggest_param(&CategoricalParam::<&str>::new(vec![]))
            .is_err()
    );
}

#[test]
fn parameter_api_with_study() {
    let x_param = FloatParam::new(-5.0, 5.0);
    let n_param = IntParam::new(1, 10);
    let dropout_param = BoolParam::new();
    let opt_param = CategoricalParam::new(vec!["sgd", "adam"]);

    let study: Study<f64> = Study::new(Direction::Minimize);
    study
        .optimize(5, |trial| {
            let x = x_param.suggest(trial)?;
            let n = n_param.suggest(trial)?;
            let dropout = dropout_param.suggest(trial)?;
            let opt = opt_param.suggest(trial)?;
            let _ = (n, dropout, opt);
            Ok::<_, optimizer::Error>(x * x)
        })
        .unwrap();

    let best = study.best_trial().unwrap();
    assert!(best.value >= 0.0);
}

#[test]
fn parameter_suggest_method() {
    let param = FloatParam::new(0.0, 1.0);
    let mut trial = Trial::new(0);
    let x = param.suggest(&mut trial).unwrap();
    assert!((0.0..=1.0).contains(&x));
}

#[test]
fn existing_suggest_methods_still_work() {
    let mut trial = Trial::new(0);

    let x_param = FloatParam::new(0.0, 1.0);
    let x = x_param.suggest(&mut trial).unwrap();
    assert!((0.0..=1.0).contains(&x));

    let lr_param = FloatParam::new(1e-5, 1e-1).log_scale();
    let lr = lr_param.suggest(&mut trial).unwrap();
    assert!((1e-5..=1e-1).contains(&lr));

    let step_param = FloatParam::new(0.0, 1.0).step(0.25);
    let step = step_param.suggest(&mut trial).unwrap();
    assert!((0.0..=1.0).contains(&step));

    let n_param = IntParam::new(1, 10);
    let n = n_param.suggest(&mut trial).unwrap();
    assert!((1..=10).contains(&n));

    let batch_param = IntParam::new(1, 1024).log_scale();
    let batch = batch_param.suggest(&mut trial).unwrap();
    assert!((1..=1024).contains(&batch));

    let units_param = IntParam::new(32, 512).step(32);
    let units = units_param.suggest(&mut trial).unwrap();
    assert!((32..=512).contains(&units));

    let opt_param = CategoricalParam::new(vec!["sgd", "adam", "rmsprop"]);
    let opt = opt_param.suggest(&mut trial).unwrap();
    assert!(["sgd", "adam", "rmsprop"].contains(&opt));

    let flag_param = BoolParam::new();
    let flag = flag_param.suggest(&mut trial).unwrap();
    let _ = flag;
}
