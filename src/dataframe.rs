//! Export trials to [polars](https://docs.rs/polars) `DataFrame`s.
//!
//! Requires the `polars` feature. The crate re-exports the exact polars
//! version it was built against as [`optimizer::polars`](crate::polars), so
//! `use optimizer::polars::prelude::*;` never causes a version mismatch.
//!
//! The entry points are [`Study::to_dataframe`], [`Study::to_dataframe_with`],
//! [`Study::intermediate_values_dataframe`], the matching methods on
//! [`MultiObjectiveStudy`], and the [`TrialsDataFrame`] extension trait for
//! slices of trials (for example `study.top_trials(5)` or
//! `mo_study.pareto_front()`).
//!
//! # Column layout
//!
//! | Column | dtype | Notes |
//! |---|---|---|
//! | `trial_id` | `UInt64` | never null |
//! | `value` (single-objective) | dtype of `V` | null unless `state == "Complete"` |
//! | `values_0` … `values_{k-1}` (multi-objective) | `Float64` | null unless `state == "Complete"` |
//! | `state` | `String` | `"Complete"`, `"Pruned"`, `"Failed"` or `"Running"` |
//! | one per parameter label | `Float64` / `Int64` / `UInt32` | categorical params hold the choice **index** unless decoded |
//! | one per user-attribute key | `Boolean` / `Int64` / `Float64` / `String` | |
//! | `constraint_0` … | `Float64` | only present when some trial has constraints |
//!
//! Parameter columns are merged by label, so name your parameters with
//! `.name()`. Unnamed parameters get a `param_<id>` column. Each
//! `FloatParam::new(..)` (and the like) gets a new id, so an unnamed parameter
//! created inside the objective closure produces a new column in every trial:
//! a study with N trials then has N mostly-null columns, and memory grows with
//! N². Give such parameters a `.name()` so they share one column. Custom
//! [`Parameter`] implementations should override
//! [`label()`](Parameter::label) to get a stable column name; a label that
//! embeds the parameter's own `ParamId` (the default `Debug`-based label) is
//! treated as unnamed.
//!
//! Only names that clash are renamed; every other column keeps its own name.
//! A parameter name that clashes with a fixed column (`trial_id`, the
//! objective columns, `state`, `constraint_i`) gets `param_` prepended, and a
//! user-attribute key that clashes with a fixed column or a parameter column
//! gets `attr_` prepended, repeatedly until the result matches no other
//! column's own name. An unnamed parameter's `param_<id>` yields to a
//! parameter that was explicitly named the same.
//!
//! # Errors
//!
//! All methods return polars' `PolarsResult`. `PolarsError` does not convert
//! into [`crate::Error`], so in a `fn main() -> optimizer::Result<()>` use
//! `.map_err(..)`, or return `Box<dyn std::error::Error>` (or `anyhow`)
//! instead. With `optimizer::prelude::*` in scope, `Result` is the
//! one-argument [`crate::Result`] alias, so spell the boxed form out as
//! `std::result::Result<(), Box<dyn std::error::Error>>`.
//!
//! # Quick start
//!
//! ```
//! use optimizer::polars::prelude::*;
//! use optimizer::prelude::*;
//!
//! let study: Study<f64> = Study::new(Direction::Minimize);
//! let lr = FloatParam::new(1e-4, 1e-1).log_scale().name("lr");
//! let opt = CategoricalParam::new(vec!["sgd", "adam"]).name("optimizer");
//!
//! study
//!     .optimize(30, |t: &mut Trial| {
//!         let lr = lr.suggest(t)?;
//!         let bonus = if opt.suggest(t)? == "adam" { 0.0 } else { 0.1 };
//!         Ok::<_, Error>((lr - 0.01).abs() + bonus)
//!     })
//!     .unwrap();
//!
//! // Raw export: `optimizer` holds the choice index (UInt32).
//! let raw = study.to_dataframe().unwrap();
//! assert_eq!(raw.column("optimizer").unwrap().dtype(), &DataType::UInt32);
//!
//! // Decoded export: `optimizer` holds "sgd" / "adam".
//! let df = study.to_dataframe_with(&[&opt]).unwrap();
//! let mask = df.column("optimizer").unwrap().str().unwrap().equal("adam");
//! let adam = df.filter(&mask).unwrap();
//! assert!(adam.height() <= 30);
//!
//! // Any slice of trials works too.
//! let top = study.top_trials(5).to_dataframe_with(&[&opt]).unwrap();
//! assert_eq!(top.height(), 5);
//! ```
//!
//! # Versioning
//!
//! polars types appear in this API. The polars dependency may move to a new
//! 0.x minor in an optimizer minor release. Use the re-exported
//! [`optimizer::polars`](crate::polars) for types. If you need extra polars
//! features such as `lazy`, add your own `polars` dependency pinned to the
//! same 0.x minor; Cargo then unifies the features (a `polars/lazy` entry in
//! optimizer's `features` list is rejected by Cargo):
//!
//! ```toml
//! polars = { version = "0.55", default-features = false, features = ["lazy"] } # must match optimizer's polars minor
//! ```
//!
//! A different minor causes `expected DataFrame, found DataFrame` errors.
//!
//! # Minimum Rust version
//!
//! The rest of the crate supports Rust 1.89, but the `polars` feature needs
//! Rust 1.94 or later (polars itself does not declare a `rust-version`, so
//! older compilers fail with an `array_windows` error inside polars).

use std::collections::{BTreeSet, HashMap, HashSet};

use polars::prelude::{Column, DataFrame, PlSmallStr, PolarsError, PolarsResult, polars_bail};

use crate::multi_objective::{MultiObjectiveStudy, MultiObjectiveTrial};
use crate::param::ParamValue;
use crate::parameter::{
    BoolParam, Categorical, CategoricalParam, EnumParam, FloatParam, IntParam, ParamId, Parameter,
};
use crate::sampler::CompletedTrial;
use crate::study::Study;
use crate::trial::AttrValue;
use crate::types::TrialState;

mod sealed {
    #[allow(unreachable_pub)]
    pub trait Sealed {}
}

/// A Rust type that maps to a native polars column dtype.
///
/// Used for the objective value `V` of a [`Study<V>`] and for the choice type
/// `T` of a [`CategoricalParam<T>`] when it is decoded.
///
/// Implemented for `f64`, `f32`, `i64`, `i32`, `u64`, `u32`, `usize`
/// (stored as `UInt64`), `isize` (stored as `Int64`), `bool`, `String` and
/// `&str`. This trait is sealed.
///
/// Other objective types (for example a newtype `struct Loss(f64)`) are not
/// supported directly. Convert the trials yourself and use the
/// [`TrialsDataFrame`] slice impl:
///
/// ```
/// use optimizer::prelude::*;
/// use optimizer::sampler::CompletedTrial;
///
/// #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
/// struct Loss(f64);
///
/// let study: Study<Loss> = Study::new(Direction::Minimize);
/// let x = FloatParam::new(0.0, 1.0).name("x");
/// let mut t = study.create_trial();
/// let v = x.suggest(&mut t).unwrap();
/// study.complete_trial(t, Loss(v));
///
/// let trials: Vec<CompletedTrial<f64>> = study
///     .trials()
///     .into_iter()
///     .map(|t| CompletedTrial {
///         value: t.value.0,
///         id: t.id,
///         params: t.params,
///         distributions: t.distributions,
///         param_labels: t.param_labels,
///         intermediate_values: t.intermediate_values,
///         state: t.state,
///         user_attrs: t.user_attrs,
///         constraints: t.constraints,
///     })
///     .collect();
/// let df = trials.to_dataframe().unwrap();
/// assert_eq!(df.height(), 1);
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be stored in a polars column",
    note = "`DataFrameValue` is sealed; it is implemented for f64, f32, i64, i32, u64, u32, usize, isize, bool, String and &str"
)]
pub trait DataFrameValue: sealed::Sealed + Sized {
    #[doc(hidden)]
    fn into_column(name: PlSmallStr, values: Vec<Option<Self>>) -> Column;
}

macro_rules! impl_dataframe_value {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl DataFrameValue for $t {
            fn into_column(name: PlSmallStr, values: Vec<Option<Self>>) -> Column {
                Column::new(name, values)
            }
        }
    )*};
}
impl_dataframe_value!(f64, f32, i64, i32, u64, u32, bool, String);

/// Types without a native polars column, widened to one that has.
macro_rules! impl_dataframe_value_via {
    ($($t:ty => $w:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl DataFrameValue for $t {
            // usize/isize are at most 64 bits on every supported target.
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_lossless)]
            fn into_column(name: PlSmallStr, values: Vec<Option<Self>>) -> Column {
                let v: Vec<Option<$w>> = values.into_iter().map(|x| x.map(|x| x as $w)).collect();
                Column::new(name, v)
            }
        }
    )*};
}
impl_dataframe_value_via!(usize => u64, isize => i64);
impl sealed::Sealed for &str {}
impl DataFrameValue for &str {
    fn into_column(name: PlSmallStr, values: Vec<Option<Self>>) -> Column {
        Column::new(name, values)
    }
}

/// A parameter whose column can be decoded from raw stored values into its
/// typed values (for example a categorical index into `"adam"`).
///
/// Implemented for every built-in parameter type. Pass parameters to
/// [`Study::to_dataframe_with`] as `&[&opt, &use_bn, &activation]`.
///
/// | Parameter | Decoded dtype |
/// |---|---|
/// | [`FloatParam`] | `Float64` (unchanged) |
/// | [`IntParam`] | `Int64` (unchanged) |
/// | [`BoolParam`] | `Boolean` |
/// | [`CategoricalParam<T>`] | dtype of `T` (`T: DataFrameValue`; `usize` → `UInt64`, `isize` → `Int64`) |
/// | [`EnumParam<T>`] | `String` with the variant's `Debug` output |
///
/// This trait is sealed.
pub trait DecodeParam: sealed::Sealed {
    /// The parameter's id and label, used to find its column.
    #[doc(hidden)]
    fn id_and_label(&self) -> (ParamId, String);
    #[doc(hidden)]
    fn decode_column(
        &self,
        name: PlSmallStr,
        cells: &[Option<(u64, &ParamValue)>],
    ) -> PolarsResult<Column>;
}

fn decode_typed<P, T>(
    param: &P,
    name: PlSmallStr,
    cells: &[Option<(u64, &ParamValue)>],
    map: impl Fn(P::Value) -> T,
) -> PolarsResult<Column>
where
    P: Parameter,
    T: DataFrameValue,
{
    let mut out = Vec::with_capacity(cells.len());
    for cell in cells {
        out.push(match cell {
            None => None,
            // A failure here means the caller passed a handle that does not
            // match the stored values, so describe the stored value instead of
            // forwarding `Error::Internal` (which reads like a library bug).
            Some((trial_id, pv)) => Some(map(param.cast_param_value(pv).map_err(|_| {
                let stored = match pv {
                    ParamValue::Categorical(i) => format!("categorical choice index {i}"),
                    ParamValue::Float(v) => format!("float value {v}"),
                    ParamValue::Int(v) => format!("integer value {v}"),
                };
                PolarsError::ComputeError(
                    format!(
                        "cannot decode column `{name}` in trial {trial_id}: it holds {stored}, \
                         which the given parameter cannot represent; pass the same parameter \
                         (same type and choices) that was used during optimization"
                    )
                    .into(),
                )
            })?)),
        });
    }
    Ok(T::into_column(name, out))
}

macro_rules! impl_decode_identity {
    ($($p:ty),*) => {$(
        impl sealed::Sealed for $p {}
        impl DecodeParam for $p {
            fn id_and_label(&self) -> (ParamId, String) {
                (self.id(), self.label())
            }
            fn decode_column(
                &self,
                name: PlSmallStr,
                cells: &[Option<(u64, &ParamValue)>],
            ) -> PolarsResult<Column> {
                decode_typed(self, name, cells, |v| v)
            }
        }
    )*};
}
impl_decode_identity!(FloatParam, IntParam, BoolParam);

impl<T: Clone + core::fmt::Debug + DataFrameValue> sealed::Sealed for CategoricalParam<T> {}
impl<T: Clone + core::fmt::Debug + DataFrameValue> DecodeParam for CategoricalParam<T> {
    fn id_and_label(&self) -> (ParamId, String) {
        (self.id(), self.label())
    }
    fn decode_column(
        &self,
        name: PlSmallStr,
        cells: &[Option<(u64, &ParamValue)>],
    ) -> PolarsResult<Column> {
        decode_typed(self, name, cells, |v| v)
    }
}

impl<T: Categorical + core::fmt::Debug> sealed::Sealed for EnumParam<T> {}
impl<T: Categorical + core::fmt::Debug> DecodeParam for EnumParam<T> {
    fn id_and_label(&self) -> (ParamId, String) {
        (self.id(), self.label())
    }
    fn decode_column(
        &self,
        name: PlSmallStr,
        cells: &[Option<(u64, &ParamValue)>],
    ) -> PolarsResult<Column> {
        decode_typed(self, name, cells, |v| format!("{v:?}"))
    }
}

/// Extension trait: build a `DataFrame` from a slice of trials.
///
/// Lets you export any subset of trials, for example
/// `study.top_trials(5).to_dataframe()` or
/// `mo_study.pareto_front().to_dataframe()`. Same columns as
/// [`Study::to_dataframe`]. This trait is sealed.
///
/// For a slice of [`MultiObjectiveTrial`]s, the number of `values_i`
/// columns comes from the trials themselves, because a slice does not know
/// the study's objective count. An empty slice (for example `pareto_front()`
/// before any trial completes) therefore has no `values_i` columns. Use
/// [`MultiObjectiveStudy::to_dataframe`] when you need a typed schema for an
/// empty study.
pub trait TrialsDataFrame: sealed::Sealed {
    /// Build a `DataFrame` with one row per trial. Categorical-backed
    /// parameters hold their choice index.
    ///
    /// # Errors
    ///
    /// See [`Study::to_dataframe`].
    fn to_dataframe(&self) -> PolarsResult<DataFrame> {
        self.to_dataframe_with(&[])
    }

    /// Like [`to_dataframe`](Self::to_dataframe), decoding the listed
    /// parameters into their typed values.
    ///
    /// # Errors
    ///
    /// See [`Study::to_dataframe_with`].
    fn to_dataframe_with(&self, params: &[&dyn DecodeParam]) -> PolarsResult<DataFrame>;
}

impl<V: DataFrameValue + Clone> sealed::Sealed for [CompletedTrial<V>] {}
impl<V: DataFrameValue + Clone> TrialsDataFrame for [CompletedTrial<V>] {
    fn to_dataframe_with(&self, params: &[&dyn DecodeParam]) -> PolarsResult<DataFrame> {
        single_objective_frame(self, params)
    }
}

impl sealed::Sealed for [MultiObjectiveTrial] {}
impl TrialsDataFrame for [MultiObjectiveTrial] {
    fn to_dataframe_with(&self, params: &[&dyn DecodeParam]) -> PolarsResult<DataFrame> {
        // A slice does not know the study's objective count; an empty slice
        // gets no `values_i` columns (documented on the trait).
        let k = self.iter().map(|t| t.values.len()).max().unwrap_or(0);
        multi_objective_frame(self, k, params)
    }
}

impl<V> Study<V>
where
    V: PartialOrd + Clone + DataFrameValue,
{
    /// Export all trials as a polars `DataFrame`, one row per trial.
    ///
    /// Columns: `trial_id`, `value`, `state`, then one column per parameter
    /// label (in definition order), then one column per user-attribute key
    /// (sorted), then `constraint_0` … `constraint_{n-1}` if any trial
    /// recorded constraints. See the [module docs](crate::dataframe) for
    /// dtypes and null rules. Categorical, bool and enum parameters hold their
    /// choice index (`UInt32`); use
    /// [`to_dataframe_with`](Self::to_dataframe_with) to see the actual
    /// choices.
    ///
    /// Unnamed parameters created inside the objective closure get a new
    /// column in every trial; give them a `.name()` (see the
    /// [module docs](crate::dataframe)).
    ///
    /// Objective types other than those implementing [`DataFrameValue`]
    /// (for example a newtype) need a manual conversion; see
    /// [`DataFrameValue`].
    ///
    /// # Errors
    ///
    /// Returns a `ComputeError` if two different parameters with the same
    /// label were suggested in the same trial.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::prelude::*;
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let x = FloatParam::new(-10.0, 10.0).name("x");
    /// study
    ///     .optimize(20, |t: &mut Trial| {
    ///         let v = x.suggest(t)?;
    ///         Ok::<_, Error>(v * v)
    ///     })
    ///     .unwrap();
    ///
    /// let df = study.to_dataframe().unwrap();
    /// assert_eq!(df.height(), 20);
    /// println!("{df}");
    /// ```
    #[doc(alias = "trials_dataframe")]
    pub fn to_dataframe(&self) -> PolarsResult<DataFrame> {
        self.to_dataframe_with(&[])
    }

    /// Export all trials, decoding the listed parameters into their typed
    /// values (`"adam"` instead of choice index `1`).
    ///
    /// Parameters are matched to columns by label, so a parameter recreated
    /// with the same `.name()` still matches. Parameters that never appear in
    /// the study are ignored.
    ///
    /// # Errors
    ///
    /// Returns a `ComputeError` if a stored value cannot be decoded by the
    /// given parameter (for example the handle has fewer choices than were
    /// used), or in the cases listed on [`to_dataframe`](Self::to_dataframe).
    pub fn to_dataframe_with(&self, params: &[&dyn DecodeParam]) -> PolarsResult<DataFrame> {
        let trials = self.storage.trials_arc().read();
        single_objective_frame(&trials, params)
    }
}

impl<V: PartialOrd> Study<V> {
    /// Export every reported intermediate value in long format: columns
    /// `trial_id` (`UInt64`), `step` (`UInt64`), `intermediate_value`
    /// (`Float64`); one row per [`Trial::report`](crate::Trial::report) call.
    ///
    /// # Errors
    ///
    /// Currently never fails; returns `PolarsResult` for consistency.
    pub fn intermediate_values_dataframe(&self) -> PolarsResult<DataFrame> {
        let trials = self.storage.trials_arc().read();
        let n: usize = trials.iter().map(|t| t.intermediate_values.len()).sum();
        let (mut ids, mut steps, mut vals) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for t in trials.iter() {
            for &(step, v) in &t.intermediate_values {
                ids.push(t.id);
                steps.push(step);
                vals.push(v);
            }
        }
        DataFrame::new(
            n,
            vec![
                Column::new("trial_id".into(), ids),
                Column::new("step".into(), steps),
                Column::new("intermediate_value".into(), vals),
            ],
        )
    }
}

impl MultiObjectiveStudy {
    /// Export all trials as a polars `DataFrame`.
    ///
    /// Same layout as [`Study::to_dataframe`], with `values_0` …
    /// `values_{k-1}` (`Float64`) instead of `value`.
    ///
    /// # Errors
    ///
    /// See [`Study::to_dataframe`].
    #[doc(alias = "trials_dataframe")]
    pub fn to_dataframe(&self) -> PolarsResult<DataFrame> {
        self.to_dataframe_with(&[])
    }

    /// Like [`to_dataframe`](Self::to_dataframe), decoding the listed
    /// parameters into their typed values.
    ///
    /// # Errors
    ///
    /// See [`Study::to_dataframe_with`].
    pub fn to_dataframe_with(&self, params: &[&dyn DecodeParam]) -> PolarsResult<DataFrame> {
        let trials = self.completed_trials.read();
        multi_objective_frame(&trials, self.n_objectives(), params)
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

struct Row<'a> {
    id: u64,
    state: TrialState,
    params: &'a HashMap<ParamId, ParamValue>,
    labels: &'a HashMap<ParamId, String>,
    attrs: &'a HashMap<String, AttrValue>,
    constraints: &'a [f64],
}

fn single_objective_frame<V: DataFrameValue + Clone>(
    trials: &[CompletedTrial<V>],
    params: &[&dyn DecodeParam],
) -> PolarsResult<DataFrame> {
    let value = V::into_column(
        "value".into(),
        trials
            .iter()
            .map(|t| (t.state == TrialState::Complete).then(|| t.value.clone()))
            .collect(),
    );
    let rows: Vec<Row<'_>> = trials
        .iter()
        .map(|t| Row {
            id: t.id,
            state: t.state,
            params: &t.params,
            labels: &t.param_labels,
            attrs: &t.user_attrs,
            constraints: &t.constraints,
        })
        .collect();
    build_frame(&rows, vec![value], params)
}

fn multi_objective_frame(
    trials: &[MultiObjectiveTrial],
    n_objectives: usize,
    params: &[&dyn DecodeParam],
) -> PolarsResult<DataFrame> {
    let objectives = (0..n_objectives)
        .map(|i| {
            let vals: Vec<Option<f64>> = trials
                .iter()
                .map(|t| {
                    (t.state == TrialState::Complete)
                        .then(|| t.values.get(i).copied())
                        .flatten()
                })
                .collect();
            Column::new(format!("values_{i}").into(), vals)
        })
        .collect();
    let rows: Vec<Row<'_>> = trials
        .iter()
        .map(|t| Row {
            id: t.id,
            state: t.state,
            params: &t.params,
            labels: &t.param_labels,
            attrs: &t.user_attrs,
            constraints: &t.constraints,
        })
        .collect();
    build_frame(&rows, objectives, params)
}

fn state_str(s: TrialState) -> &'static str {
    match s {
        TrialState::Complete => "Complete",
        TrialState::Pruned => "Pruned",
        TrialState::Failed => "Failed",
        TrialState::Running => "Running",
    }
}

/// Internal grouping key for parameter columns. Unnamed parameters live in
/// their own key space, so a user label such as `"param_3"` never merges with
/// the unnamed parameter whose id is 3.
#[derive(Clone, PartialEq, Eq, Hash)]
enum ParamKey {
    Named(String),
    Unnamed(ParamId),
}

impl ParamKey {
    /// The column name this key asks for before collision renaming.
    fn natural_name(&self) -> String {
        match self {
            Self::Named(l) => l.clone(),
            Self::Unnamed(id) => id.to_string(),
        }
    }
}

/// Grouping key for a parameter. A label that embeds this parameter's own
/// `ParamId` (the default `Debug`-based label of unnamed built-in or custom
/// parameters) counts as unnamed and becomes `param_<id>`, like the CSV
/// export.
fn param_key(id: ParamId, label: Option<&String>) -> ParamKey {
    match label {
        Some(l) if !l.contains(&format!("{id:?}")) => ParamKey::Named(l.clone()),
        _ => ParamKey::Unnamed(id),
    }
}

/// Prepend `prefix` to a colliding `name` until the result is neither an
/// emitted column (`taken`) nor any column's own name (`avoid`).
fn rename(name: &str, prefix: &str, taken: &HashSet<String>, avoid: &HashSet<&str>) -> String {
    let mut n = format!("{prefix}{name}");
    while taken.contains(&n) || avoid.contains(n.as_str()) {
        n = format!("{prefix}{n}");
    }
    n
}

/// One column's cells: `(trial id, value)` where the trial suggested the param.
type Cells<'a> = Vec<Option<(u64, &'a ParamValue)>>;
/// Per column `(smallest ParamId, key)`, plus each column's cells.
type ParamGroups<'a> = (Vec<(ParamId, ParamKey)>, Vec<Cells<'a>>);

/// Group param cells into columns, column-major: `(smallest ParamId, key)` per
/// column plus its cells. Errors if two different params share a column in
/// one trial.
fn group_params<'a>(rows: &[Row<'a>]) -> PolarsResult<ParamGroups<'a>> {
    // A ParamId's label is the same in every trial, so its key is computed
    // once per ParamId.
    let mut id_to_col: HashMap<ParamId, usize> = HashMap::new();
    let mut key_to_col: HashMap<ParamKey, usize> = HashMap::new();
    let mut groups: Vec<(ParamId, ParamKey)> = Vec::new(); // (smallest id, key)
    let mut col_cells: Vec<Cells<'a>> = Vec::new();
    for (row, r) in rows.iter().enumerate() {
        for (&id, pv) in r.params {
            let ci = if let Some(&ci) = id_to_col.get(&id) {
                ci
            } else {
                let key = param_key(id, r.labels.get(&id));
                let ci = if let Some(&ci) = key_to_col.get(&key) {
                    groups[ci].0 = groups[ci].0.min(id);
                    ci
                } else {
                    let ci = groups.len();
                    key_to_col.insert(key.clone(), ci);
                    groups.push((id, key));
                    col_cells.push(vec![None; rows.len()]);
                    ci
                };
                id_to_col.insert(id, ci);
                ci
            };
            let slot = &mut col_cells[ci][row];
            if slot.is_some() {
                polars_bail!(ComputeError:
                    "trial {} has two different parameters labelled `{}`; give them distinct `.name()`s",
                    r.id, groups[ci].1.natural_name());
            }
            *slot = Some((r.id, pv));
        }
    }
    Ok((groups, col_cells))
}

fn build_frame(
    rows: &[Row<'_>],
    objective_cols: Vec<Column>,
    decoders: &[&dyn DecodeParam],
) -> PolarsResult<DataFrame> {
    let height = rows.len();
    let n_constraints = rows.iter().map(|r| r.constraints.len()).max().unwrap_or(0);

    // Reserved (fixed) column names.
    let mut taken: HashSet<String> = HashSet::new();
    taken.insert("trial_id".into());
    taken.insert("state".into());
    for c in &objective_cols {
        taken.insert(c.name().to_string());
    }
    for i in 0..n_constraints {
        taken.insert(format!("constraint_{i}"));
    }

    let (groups, col_cells) = group_params(rows)?;
    // Definition order: smallest ParamId per column (unique per column).
    let mut order: Vec<usize> = (0..groups.len()).collect();
    order.sort_by_key(|&ci| groups[ci].0);

    let decoders: HashMap<ParamKey, &dyn DecodeParam> = decoders
        .iter()
        .map(|d| {
            let (id, label) = d.id_and_label();
            (param_key(id, Some(&label)), *d)
        })
        .collect();

    // Every column's own name. Generated names avoid all of them, so renaming
    // never hands a column a name that another column really owns.
    let naturals: Vec<String> = groups.iter().map(|(_, k)| k.natural_name()).collect();
    let attr_keys: BTreeSet<&String> = rows.iter().flat_map(|r| r.attrs.keys()).collect();
    let named: HashSet<&str> = groups
        .iter()
        .filter_map(|(_, k)| match k {
            ParamKey::Named(l) => Some(l.as_str()),
            ParamKey::Unnamed(_) => None,
        })
        .collect();
    let avoid: HashSet<&str> = naturals
        .iter()
        .map(String::as_str)
        .chain(attr_keys.iter().map(|k| k.as_str()))
        .collect();

    let mut param_cols = Vec::with_capacity(groups.len());
    for ci in order {
        let key = &groups[ci].1;
        let natural = &naturals[ci];
        // Only reserved names can clash with a named param (labels are unique
        // and generated names avoid them). An unnamed `param_<id>` also yields
        // to a param explicitly named the same.
        let collides = taken.contains(natural)
            || (matches!(key, ParamKey::Unnamed(_)) && named.contains(natural.as_str()));
        let col_name = if collides {
            rename(natural, "param_", &taken, &avoid)
        } else {
            natural.clone()
        };
        let cells = &col_cells[ci];
        let col = match decoders.get(key) {
            Some(d) => d.decode_column(col_name.as_str().into(), cells)?,
            None => raw_param_column(col_name.as_str().into(), cells)?,
        };
        taken.insert(col_name);
        param_cols.push(col);
    }

    // Attributes yield to reserved names and parameter columns.
    let mut attr_cols = Vec::with_capacity(attr_keys.len());
    for key in attr_keys {
        let col_name = if taken.contains(key) {
            rename(key, "attr_", &taken, &avoid)
        } else {
            key.clone()
        };
        let cells: Vec<Option<&AttrValue>> = rows.iter().map(|r| r.attrs.get(key)).collect();
        attr_cols.push(attr_column(col_name.as_str().into(), &cells));
        taken.insert(col_name);
    }

    let mut cols = Vec::with_capacity(
        2 + objective_cols.len() + param_cols.len() + attr_cols.len() + n_constraints,
    );
    cols.push(Column::new(
        "trial_id".into(),
        rows.iter().map(|r| r.id).collect::<Vec<_>>(),
    ));
    cols.extend(objective_cols);
    cols.push(Column::new(
        "state".into(),
        rows.iter().map(|r| state_str(r.state)).collect::<Vec<_>>(),
    ));
    cols.extend(param_cols);
    cols.extend(attr_cols);
    for i in 0..n_constraints {
        let v: Vec<Option<f64>> = rows.iter().map(|r| r.constraints.get(i).copied()).collect();
        cols.push(Column::new(format!("constraint_{i}").into(), v));
    }
    DataFrame::new(height, cols)
}

fn raw_param_column(
    name: PlSmallStr,
    cells: &[Option<(u64, &ParamValue)>],
) -> PolarsResult<Column> {
    let (mut has_f, mut has_i, mut has_c) = (false, false, false);
    for (_, pv) in cells.iter().flatten() {
        match pv {
            ParamValue::Float(_) => has_f = true,
            ParamValue::Int(_) => has_i = true,
            ParamValue::Categorical(_) => has_c = true,
        }
    }
    Ok(match (has_f, has_i, has_c) {
        (_, _, false) if has_f => Column::new(
            name,
            cells
                .iter()
                .map(|cell| {
                    cell.map(|(_, pv)| match *pv {
                        ParamValue::Float(v) => v,
                        // Mixed Int/Float under one label: widen.
                        #[allow(clippy::cast_precision_loss)]
                        ParamValue::Int(v) => v as f64,
                        ParamValue::Categorical(_) => unreachable!(),
                    })
                })
                .collect::<Vec<_>>(),
        ),
        (false, _, false) => Column::new(
            name,
            cells
                .iter()
                .map(|cell| {
                    cell.and_then(|(_, pv)| match *pv {
                        ParamValue::Int(v) => Some(v),
                        _ => None,
                    })
                })
                .collect::<Vec<_>>(),
        ),
        (false, false, true) => {
            let mut out = Vec::with_capacity(cells.len());
            for cell in cells {
                out.push(match cell {
                    Some((_, ParamValue::Categorical(idx))) => {
                        Some(u32::try_from(*idx).map_err(|_| {
                            PolarsError::ComputeError(
                                format!("choice index {idx} exceeds u32").into(),
                            )
                        })?)
                    }
                    _ => None,
                });
            }
            Column::new(name, out)
        }
        // Categorical mixed with numeric kinds under one label: fall back to text.
        _ => Column::new(
            name,
            cells
                .iter()
                .map(|cell| cell.map(|(_, pv)| pv.to_string()))
                .collect::<Vec<_>>(),
        ),
    })
}

fn attr_column(name: PlSmallStr, cells: &[Option<&AttrValue>]) -> Column {
    let (mut has_f, mut has_i, mut has_s, mut has_b) = (false, false, false, false);
    for a in cells.iter().flatten() {
        match a {
            AttrValue::Float(_) => has_f = true,
            AttrValue::Int(_) => has_i = true,
            AttrValue::String(_) => has_s = true,
            AttrValue::Bool(_) => has_b = true,
        }
    }
    match (has_f, has_i, has_s, has_b) {
        (false, false, false, true) => Column::new(
            name,
            cells
                .iter()
                .map(|c| match c {
                    Some(AttrValue::Bool(v)) => Some(*v),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ),
        (false, true, false, false) => Column::new(
            name,
            cells
                .iter()
                .map(|c| match c {
                    Some(AttrValue::Int(v)) => Some(*v),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ),
        (true, _, false, false) => Column::new(
            name,
            cells
                .iter()
                .map(|c| match c {
                    Some(AttrValue::Float(v)) => Some(*v),
                    #[allow(clippy::cast_precision_loss)]
                    Some(AttrValue::Int(v)) => Some(*v as f64),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ),
        _ => Column::new(
            name,
            cells
                .iter()
                .map(|c| {
                    c.map(|a| match a {
                        AttrValue::Float(v) => v.to_string(),
                        AttrValue::Int(v) => v.to_string(),
                        AttrValue::String(v) => v.clone(),
                        AttrValue::Bool(v) => v.to_string(),
                    })
                })
                .collect::<Vec<_>>(),
        ),
    }
}
