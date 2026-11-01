use optimizer::polars::prelude::*;
use optimizer::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Act {
    Relu,
    Tanh,
}
impl Categorical for Act {
    const N_CHOICES: usize = 2;
    fn from_index(i: usize) -> Self {
        [Self::Relu, Self::Tanh][i]
    }
    fn to_index(&self) -> usize {
        *self as usize
    }
}

fn names(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect()
}

#[test]
fn empty_study_has_typed_columns() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let df = study.to_dataframe().unwrap();
    assert_eq!(df.height(), 0);
    assert_eq!(names(&df), ["trial_id", "value", "state"]);
    assert_eq!(df.column("trial_id").unwrap().dtype(), &DataType::UInt64);
    assert_eq!(df.column("value").unwrap().dtype(), &DataType::Float64);
    assert_eq!(df.column("state").unwrap().dtype(), &DataType::String);
}

#[test]
fn schema_order_and_dtypes() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 1.0).name("x");
    let n = IntParam::new(1, 4).name("n");
    let opt = CategoricalParam::new(vec!["sgd", "adam"]).name("opt");
    let bn = BoolParam::new().name("bn");
    let act = EnumParam::<Act>::new().name("act");
    let bs = CategoricalParam::new(vec![16, 32, 64]).name("bs"); // T = i32
    let mut t = study.create_trial();
    x.suggest(&mut t).unwrap();
    n.suggest(&mut t).unwrap();
    opt.suggest(&mut t).unwrap();
    bn.suggest(&mut t).unwrap();
    act.suggest(&mut t).unwrap();
    bs.suggest(&mut t).unwrap();
    t.set_user_attr("zz", 1_i64);
    t.set_user_attr("aa", "note");
    study.complete_trial(t, 0.5);

    let raw = study.to_dataframe().unwrap();
    assert_eq!(
        names(&raw),
        [
            "trial_id", "value", "state", "x", "n", "opt", "bn", "act", "bs", "aa", "zz"
        ]
    );
    assert_eq!(raw.column("opt").unwrap().dtype(), &DataType::UInt32);
    assert_eq!(raw.column("n").unwrap().dtype(), &DataType::Int64);

    let df = study
        .to_dataframe_with(&[&opt, &bn, &act, &bs, &x])
        .unwrap();
    assert_eq!(df.column("opt").unwrap().dtype(), &DataType::String);
    assert_eq!(df.column("bn").unwrap().dtype(), &DataType::Boolean);
    assert_eq!(df.column("act").unwrap().dtype(), &DataType::String);
    assert_eq!(df.column("bs").unwrap().dtype(), &DataType::Int32);
    assert_eq!(df.column("x").unwrap().dtype(), &DataType::Float64);
    let a = df
        .column("act")
        .unwrap()
        .str()
        .unwrap()
        .get(0)
        .unwrap()
        .to_string();
    assert!(a == "Relu" || a == "Tanh");
    assert_eq!(df.column("zz").unwrap().dtype(), &DataType::Int64);
    assert_eq!(df.column("aa").unwrap().dtype(), &DataType::String);
}

#[test]
fn pruned_value_null_and_missing_params_null() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let x = FloatParam::new(0.0, 1.0).name("x");
    let y = FloatParam::new(0.0, 1.0).name("y");
    let mut t = study.create_trial();
    x.suggest(&mut t).unwrap();
    study.complete_trial(t, 1.0);
    let mut t = study.create_trial();
    y.suggest(&mut t).unwrap();
    t.report(0, 3.0);
    study.prune_trial(t);
    let df = study.to_dataframe().unwrap();
    assert_eq!(df.column("value").unwrap().null_count(), 1);
    assert_eq!(df.column("x").unwrap().null_count(), 1);
    assert_eq!(df.column("y").unwrap().null_count(), 1);
    let st = df
        .column("state")
        .unwrap()
        .str()
        .unwrap()
        .get(1)
        .unwrap()
        .to_string();
    assert_eq!(st, "Pruned");

    let iv = study.intermediate_values_dataframe().unwrap();
    assert_eq!(names(&iv), ["trial_id", "step", "intermediate_value"]);
    assert_eq!(iv.height(), 1);
}

#[test]
fn recreated_params_merge_by_label_and_unnamed_fallback() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    study
        .optimize(3, |t: &mut Trial| {
            let m = FloatParam::new(0.0, 1.0).name("momentum").suggest(t)?;
            let u = FloatParam::new(0.0, 1.0).suggest(t)?;
            Ok::<_, Error>(m + u)
        })
        .unwrap();
    let df = study.to_dataframe().unwrap();
    let cols = names(&df);
    assert_eq!(cols.iter().filter(|c| *c == "momentum").count(), 1);
    assert_eq!(df.column("momentum").unwrap().null_count(), 0);
    // three unnamed params (fresh id each trial) -> param_<id> columns
    assert_eq!(cols.iter().filter(|c| c.starts_with("param_")).count(), 3);
    assert!(cols.iter().all(|c| !c.contains("ParamId(")));
}

#[test]
fn collisions_are_renamed_and_same_trial_duplicate_errors() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let v = FloatParam::new(0.0, 1.0).name("value");
    let x = FloatParam::new(0.0, 1.0).name("x");
    let mut t = study.create_trial();
    v.suggest(&mut t).unwrap();
    x.suggest(&mut t).unwrap();
    t.set_user_attr("x", true);
    t.set_user_attr("state", "s");
    t.set_constraints(vec![0.5]);
    study.complete_trial(t, 1.0);
    let df = study.to_dataframe().unwrap();
    assert_eq!(
        names(&df),
        [
            "trial_id",
            "value",
            "state",
            "param_value",
            "x",
            "attr_state",
            "attr_x",
            "constraint_0"
        ]
    );

    let study: Study<f64> = Study::new(Direction::Minimize);
    let a = FloatParam::new(0.0, 1.0).name("dup");
    let b = FloatParam::new(0.0, 1.0).name("dup");
    let mut t = study.create_trial();
    a.suggest(&mut t).unwrap();
    b.suggest(&mut t).unwrap();
    study.complete_trial(t, 1.0);
    assert!(study.to_dataframe().is_err());
}

#[test]
fn integer_objective_and_multi_objective_and_slices() {
    let study: Study<i64> = Study::new(Direction::Minimize);
    let n = IntParam::new(0, 9).name("n");
    let mut t = study.create_trial();
    let v = n.suggest(&mut t).unwrap();
    study.complete_trial(t, v);
    let df = study.to_dataframe().unwrap();
    assert_eq!(df.column("value").unwrap().dtype(), &DataType::Int64);

    let mo = MultiObjectiveStudy::new(vec![Direction::Minimize, Direction::Minimize]);
    let empty = mo.to_dataframe().unwrap();
    assert_eq!(names(&empty), ["trial_id", "values_0", "values_1", "state"]);
    let opt = CategoricalParam::new(vec!["a", "b"]).name("opt");
    mo.optimize(10, |t: &mut Trial| {
        let i = if opt.suggest(t)? == "a" { 0.0 } else { 1.0 };
        Ok::<_, Error>(vec![i, 1.0 - i])
    })
    .unwrap();
    let front = mo.pareto_front().to_dataframe_with(&[&opt]).unwrap();
    assert!(front.height() >= 1);
    assert_eq!(front.column("opt").unwrap().dtype(), &DataType::String);

    let top = study.top_trials(1).to_dataframe().unwrap();
    assert_eq!(top.height(), 1);
    // Vec autoderefs to the slice impl.
    let all = study.trials().to_dataframe().unwrap();
    assert_eq!(all.height(), 1);
}

#[test]
fn decode_with_wrong_handle_errors() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let big = CategoricalParam::new(vec!["a", "b", "c"]).name("c");
    let t = loop {
        let mut tt = study.create_trial();
        if big.suggest(&mut tt).unwrap() == "c" {
            break tt;
        }
    };
    study.complete_trial(t, 0.0);
    let small = CategoricalParam::new(vec!["a", "b"]).name("c");
    let err = study.to_dataframe_with(&[&small]).unwrap_err().to_string();
    // A mismatched handle is the caller's mistake, not a library bug.
    assert!(err.contains("choice index 2"), "{err}");
    assert!(!err.contains("internal error"), "{err}");

    let wrong_kind = FloatParam::new(0.0, 1.0).name("c");
    let err = study
        .to_dataframe_with(&[&wrong_kind])
        .unwrap_err()
        .to_string();
    assert!(err.contains("choice index 2"), "{err}");
    assert!(!err.contains("internal error"), "{err}");
}

#[test]
fn heterogeneous_vec_needs_annotation() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let opt = CategoricalParam::new(vec!["a", "b"]).name("opt");
    let bn = BoolParam::new().name("bn");
    let decoders: Vec<&dyn DecodeParam> = vec![&opt, &bn];
    let df = study.to_dataframe_with(&decoders).unwrap();
    assert_eq!(df.height(), 0);
}

#[test]
fn unnamed_categorical_can_be_decoded() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let opt = CategoricalParam::new(vec!["a", "b"]);
    let mut t = study.create_trial();
    opt.suggest(&mut t).unwrap();
    study.complete_trial(t, 0.0);
    let df = study.to_dataframe_with(&[&opt]).unwrap();
    let col = df.column(&opt.id().to_string()).unwrap();
    assert_eq!(col.dtype(), &DataType::String);
}

#[test]
fn user_attr_dtype_unification() {
    let study: Study<f64> = Study::new(Direction::Minimize);
    let mut t = study.create_trial();
    t.set_user_attr("num", 1_i64);
    t.set_user_attr("mixed", true);
    study.complete_trial(t, 0.0);
    let mut t = study.create_trial();
    t.set_user_attr("num", 2.5_f64);
    t.set_user_attr("mixed", 7_i64);
    study.complete_trial(t, 1.0);
    let mut t = study.create_trial();
    t.set_user_attr("only_bool", false);
    study.complete_trial(t, 2.0);

    let df = study.to_dataframe().unwrap();
    // Int + Float -> Float64
    let num = df.column("num").unwrap();
    assert_eq!(num.dtype(), &DataType::Float64);
    let num = num.f64().unwrap();
    assert_eq!(num.get(0), Some(1.0));
    assert_eq!(num.get(1), Some(2.5));
    assert_eq!(num.get(2), None);
    // Bool + Int -> String
    let mixed = df.column("mixed").unwrap();
    assert_eq!(mixed.dtype(), &DataType::String);
    let mixed = mixed.str().unwrap();
    assert_eq!(mixed.get(0), Some("true"));
    assert_eq!(mixed.get(1), Some("7"));
    assert_eq!(mixed.get(2), None);
    // Bool only -> Boolean
    assert_eq!(df.column("only_bool").unwrap().dtype(), &DataType::Boolean);
}

/// Renaming must never give a column a name that another column really owns,
/// and the result must not depend on definition order.
#[test]
fn collision_renaming_never_steals_real_names() {
    for value_first in [true, false] {
        let study: Study<f64> = Study::new(Direction::Minimize);
        let (v, pv) = if value_first {
            let v = FloatParam::new(0.0, 1.0).name("value");
            let pv = IntParam::new(100, 200).name("param_value");
            (v, pv)
        } else {
            let pv = IntParam::new(100, 200).name("param_value");
            let v = FloatParam::new(0.0, 1.0).name("value");
            (v, pv)
        };
        let mut t = study.create_trial();
        let fv = v.suggest(&mut t).unwrap();
        let iv = pv.suggest(&mut t).unwrap();
        // The attr "param_value" yields to the param of that name.
        t.set_user_attr("param_value", 7_i64);
        study.complete_trial(t, 0.0);
        let df = study.to_dataframe().unwrap();
        let col = df.column("param_value").unwrap();
        assert_eq!(col.dtype(), &DataType::Int64, "value_first={value_first}");
        assert_eq!(col.i64().unwrap().get(0), Some(iv));
        let col = df.column("param_param_value").unwrap();
        assert_eq!(col.f64().unwrap().get(0), Some(fv));
        assert_eq!(
            df.column("attr_param_value").unwrap().i64().unwrap().get(0),
            Some(7)
        );
    }

    let study: Study<f64> = Study::new(Direction::Minimize);
    let z = FloatParam::new(0.0, 1.0).name("Z");
    let mut t = study.create_trial();
    z.suggest(&mut t).unwrap();
    t.set_user_attr("Z", "from-key-Z");
    t.set_user_attr("attr_Z", "from-key-attr_Z");
    study.complete_trial(t, 0.0);
    let df = study.to_dataframe().unwrap();
    let get = |c: &str| {
        df.column(c)
            .unwrap()
            .str()
            .unwrap()
            .get(0)
            .unwrap()
            .to_string()
    };
    assert_eq!(get("attr_Z"), "from-key-attr_Z");
    assert_eq!(get("attr_attr_Z"), "from-key-Z");
}

/// A user parameter named `param_<n>` must not merge with the unnamed
/// parameter whose id is `n`; the unnamed one is renamed.
#[test]
fn named_param_n_does_not_merge_with_unnamed_param() {
    let unnamed = CategoricalParam::new(vec!["a", "b"]);
    let unnamed_name = unnamed.id().to_string();
    let named = FloatParam::new(0.0, 1.0).name(unnamed_name.as_str());
    let renamed = format!("param_{unnamed_name}");

    // Different trials.
    let study: Study<f64> = Study::new(Direction::Minimize);
    let mut t = study.create_trial();
    unnamed.suggest(&mut t).unwrap();
    study.complete_trial(t, 0.0);
    let mut t = study.create_trial();
    named.suggest(&mut t).unwrap();
    study.complete_trial(t, 1.0);
    let raw = study.to_dataframe().unwrap();
    assert_eq!(
        raw.column(&unnamed_name).unwrap().dtype(),
        &DataType::Float64
    );
    assert_eq!(raw.column(&renamed).unwrap().dtype(), &DataType::UInt32);
    let df = study.to_dataframe_with(&[&unnamed]).unwrap();
    assert_eq!(df.column(&renamed).unwrap().dtype(), &DataType::String);
    assert_eq!(df.column(&renamed).unwrap().null_count(), 1);

    // Same trial: exports without error.
    let study: Study<f64> = Study::new(Direction::Minimize);
    let mut t = study.create_trial();
    unnamed.suggest(&mut t).unwrap();
    named.suggest(&mut t).unwrap();
    study.complete_trial(t, 0.0);
    let df = study.to_dataframe_with(&[&unnamed, &named]).unwrap();
    assert_eq!(df.column(&unnamed_name).unwrap().null_count(), 0);
    assert_eq!(df.column(&renamed).unwrap().null_count(), 0);
}

/// A custom `Parameter` with the default `Debug` label (no `name` field)
/// falls back to `param_<id>` instead of its whole `Debug` string.
#[test]
fn custom_unnamed_param_uses_id_fallback() {
    use optimizer::distribution::{Distribution, FloatDistribution};
    use optimizer::parameter::ParamId;

    #[derive(Debug)]
    struct MyParam {
        id: ParamId,
        lo: f64,
        hi: f64,
    }
    impl Parameter for MyParam {
        type Value = f64;
        fn id(&self) -> ParamId {
            self.id
        }
        fn distribution(&self) -> Distribution {
            Distribution::Float(FloatDistribution {
                low: self.lo,
                high: self.hi,
                log_scale: false,
                step: None,
            })
        }
        fn cast_param_value(&self, pv: &ParamValue) -> optimizer::Result<f64> {
            match pv {
                ParamValue::Float(v) => Ok(*v),
                _ => Err(Error::Internal("expected float")),
            }
        }
    }

    let study: Study<f64> = Study::new(Direction::Minimize);
    study
        .optimize(3, |t: &mut Trial| {
            let p = MyParam {
                id: ParamId::new(),
                lo: 0.0,
                hi: 1.0,
            };
            p.suggest(t)
        })
        .unwrap();
    let df = study.to_dataframe().unwrap();
    let cols = names(&df);
    assert_eq!(cols.iter().filter(|c| c.starts_with("param_")).count(), 3);
    assert!(cols.iter().all(|c| !c.contains("ParamId(")), "{cols:?}");
}

#[test]
fn usize_choices_and_objective() {
    let study: Study<usize> = Study::new(Direction::Minimize);
    let batch = CategoricalParam::new(vec![16usize, 32, 64]).name("batch");
    let mut t = study.create_trial();
    let b = batch.suggest(&mut t).unwrap();
    study.complete_trial(t, b);
    let df = study.to_dataframe_with(&[&batch]).unwrap();
    assert_eq!(df.column("value").unwrap().dtype(), &DataType::UInt64);
    let col = df.column("batch").unwrap();
    assert_eq!(col.dtype(), &DataType::UInt64);
    assert_eq!(col.u64().unwrap().get(0), Some(b as u64));

    let study: Study<f64> = Study::new(Direction::Minimize);
    let off = CategoricalParam::new(vec![-1isize, 1]).name("off");
    let mut t = study.create_trial();
    let o = off.suggest(&mut t).unwrap();
    study.complete_trial(t, 0.0);
    let df = study.to_dataframe_with(&[&off]).unwrap();
    let col = df.column("off").unwrap();
    assert_eq!(col.dtype(), &DataType::Int64);
    assert_eq!(col.i64().unwrap().get(0), Some(o as i64));
}

#[test]
fn empty_pareto_slice_has_no_value_columns() {
    // Documented: a slice does not know the objective count.
    let mo = MultiObjectiveStudy::new(vec![Direction::Minimize, Direction::Minimize]);
    let front = mo.pareto_front().to_dataframe().unwrap();
    assert_eq!(names(&front), ["trial_id", "state"]);
}

#[cfg(feature = "journal")]
#[test]
fn journal_loaded_study_exports_same_frame() {
    use optimizer::sampler::random::RandomSampler;

    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "optimizer_polars_journal_test_{}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);

    let x = FloatParam::new(-1.0, 1.0).name("x");
    let n = IntParam::new(0, 9).name("n");
    let opt = CategoricalParam::new(vec!["sgd", "adam"]).name("opt");

    let original = {
        let study: Study<f64> =
            Study::with_journal(Direction::Minimize, RandomSampler::with_seed(7), &path).unwrap();
        study
            .optimize(8, |t: &mut Trial| {
                let x = x.suggest(t)?;
                let n = n.suggest(t)?;
                let o = opt.suggest(t)?;
                t.set_user_attr("tag", o);
                t.report(0, x);
                #[allow(clippy::cast_precision_loss)]
                Ok::<_, Error>(x * x + n as f64)
            })
            .unwrap();
        (
            study.to_dataframe_with(&[&opt]).unwrap(),
            study.intermediate_values_dataframe().unwrap(),
        )
    };

    let reloaded: Study<f64> =
        Study::with_journal(Direction::Minimize, RandomSampler::with_seed(8), &path).unwrap();
    let df = reloaded.to_dataframe_with(&[&opt]).unwrap();
    let iv = reloaded.intermediate_values_dataframe().unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(df.height(), 8);
    assert_frames_match(&df, &original.0);
    assert_frames_match(&iv, &original.1);
}

/// Same names, dtypes and values. `Float64` columns are compared with a tiny
/// tolerance because `serde_json` (without `float_roundtrip`) may be off by
/// one ULP when reading floats back from the journal.
#[cfg(feature = "journal")]
fn assert_frames_match(a: &DataFrame, b: &DataFrame) {
    assert_eq!(names(a), names(b));
    for (ca, cb) in a.columns().iter().zip(b.columns()) {
        assert_eq!(ca.dtype(), cb.dtype(), "column {}", ca.name());
        if ca.dtype() == &DataType::Float64 {
            for (x, y) in ca.f64().unwrap().iter().zip(cb.f64().unwrap().iter()) {
                match (x, y) {
                    (Some(x), Some(y)) => assert!(
                        (x - y).abs() <= 1e-12 * x.abs().max(1.0),
                        "column {}: {x} != {y}",
                        ca.name()
                    ),
                    (x, y) => assert_eq!(x, y, "column {}", ca.name()),
                }
            }
        } else {
            assert!(
                ca.equals_missing(cb),
                "column {}:\n{ca:?}\n{cb:?}",
                ca.name()
            );
        }
    }
}
