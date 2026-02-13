//! HTML report generation for optimization visualization.
//!
//! Generate self-contained HTML files with embedded
//! [Plotly.js](https://plotly.com/javascript/) charts for offline
//! visualization of optimization results. No feature flag is required —
//! this module is always available.
//!
//! # Charts included
//!
//! | Chart | Description |
//! |---|---|
//! | **Optimization history** | Objective value vs trial number with best-so-far line |
//! | **Slice plots** | Objective value vs each parameter (1D scatter per param) |
//! | **Parallel coordinates** | Multi-parameter relationship view (color = objective) |
//! | **Parameter importance** | Horizontal bar chart of Spearman-based importance |
//! | **Trial timeline** | Duration/index of each trial, color-coded by state |
//! | **Intermediate values** | Per-trial learning curves (if pruning data available) |
//!
//! # Usage
//!
//! Call [`Study::export_html()`](crate::Study::export_html) or
//! [`generate_html_report()`] directly:
//!
//! ```no_run
//! use optimizer::prelude::*;
//!
//! let study: Study<f64> = Study::new(Direction::Minimize);
//! # let x = FloatParam::new(0.0, 1.0);
//! # study.optimize(10, |trial: &mut optimizer::Trial| {
//! #     let v = x.suggest(trial)?;
//! #     Ok::<_, optimizer::Error>(v * v)
//! # }).unwrap();
//! study.export_html("report.html").unwrap();
//! ```
//!
//! The output is a single HTML file that can be opened in any browser.
//! An internet connection is needed on first load to fetch `Plotly.js`
//! from a CDN.

use core::fmt::Write as _;
use std::collections::BTreeMap;
use std::path::Path;

use crate::param::ParamValue;
use crate::parameter::ParamId;
use crate::sampler::CompletedTrial;
use crate::types::{Direction, TrialState};

/// Generate an HTML report with interactive Plotly.js charts.
///
/// Create a self-contained HTML file at `path` containing up to six
/// interactive charts. Charts that require data not present in the study
/// (e.g., intermediate values) are automatically omitted.
///
/// The report includes: optimization history, slice plots, parallel
/// coordinates, parameter importance, trial timeline, and intermediate
/// values (when available).
///
/// This is also available as [`Study::export_html()`](crate::Study::export_html).
///
/// # Errors
///
/// Return an I/O error if the file cannot be created or written.
pub fn generate_html_report(
    study: &crate::Study<f64>,
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    let trials = study.trials();
    let direction = study.direction();
    let importance = study.param_importance();

    let html = build_html(&trials, direction, &importance);
    std::fs::write(path, html)
}

fn build_html(
    trials: &[CompletedTrial<f64>],
    direction: Direction,
    importance: &[(String, f64)],
) -> String {
    let mut html = String::with_capacity(8192);

    let dir_label = match direction {
        Direction::Minimize => "Minimize",
        Direction::Maximize => "Maximize",
    };

    // Collect parameter metadata.
    let param_info = collect_param_info(trials);
    let has_intermediate = trials.iter().any(|t| !t.intermediate_values.is_empty());

    let _ = write!(
        html,
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Optimization Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f6fa; color: #2c3e50; padding: 24px; }}
  h1 {{ text-align: center; margin-bottom: 8px; font-size: 1.8em; }}
  .subtitle {{ text-align: center; color: #7f8c8d; margin-bottom: 24px; }}
  .chart {{ background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 24px; padding: 16px; }}
  .chart-title {{ font-size: 1.1em; font-weight: 600; margin-bottom: 8px; }}
</style>
</head>
<body>
<h1>Optimization Report</h1>
<p class="subtitle">{dir_label} &middot; {n} trials</p>
"#,
        n = trials.len(),
    );

    // Optimization history chart.
    html.push_str("<div class=\"chart\"><div class=\"chart-title\">Optimization History</div><div id=\"history\"></div></div>\n");
    write_history_chart(&mut html, trials, direction);

    // Slice plots.
    if !param_info.is_empty() {
        html.push_str("<div class=\"chart\"><div class=\"chart-title\">Slice Plots</div><div id=\"slices\"></div></div>\n");
        write_slice_charts(&mut html, trials, &param_info);
    }

    // Parallel coordinates.
    if param_info.len() >= 2 {
        html.push_str("<div class=\"chart\"><div class=\"chart-title\">Parallel Coordinates</div><div id=\"parcoords\"></div></div>\n");
        write_parallel_coordinates(&mut html, trials, &param_info, direction);
    }

    // Parameter importance.
    if !importance.is_empty() {
        html.push_str("<div class=\"chart\"><div class=\"chart-title\">Parameter Importance</div><div id=\"importance\"></div></div>\n");
        write_importance_chart(&mut html, importance);
    }

    // Trial timeline.
    html.push_str("<div class=\"chart\"><div class=\"chart-title\">Trial Timeline</div><div id=\"timeline\"></div></div>\n");
    write_timeline_chart(&mut html, trials);

    // Intermediate values.
    if has_intermediate {
        html.push_str("<div class=\"chart\"><div class=\"chart-title\">Intermediate Values</div><div id=\"intermediate\"></div></div>\n");
        write_intermediate_chart(&mut html, trials);
    }

    html.push_str("</body>\n</html>\n");
    html
}

/// Metadata about each parameter seen across trials.
struct ParamMeta {
    label: String,
}

/// Collect parameter labels and distributions across all trials.
fn collect_param_info(trials: &[CompletedTrial<f64>]) -> BTreeMap<ParamId, ParamMeta> {
    let mut info: BTreeMap<ParamId, ParamMeta> = BTreeMap::new();
    for trial in trials {
        for &id in trial.params.keys() {
            info.entry(id).or_insert_with(|| {
                let label = trial
                    .param_labels
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| id.to_string());
                ParamMeta { label }
            });
        }
    }
    info
}

// ---------------------------------------------------------------------------
// Chart generators
// ---------------------------------------------------------------------------

fn write_history_chart(html: &mut String, trials: &[CompletedTrial<f64>], direction: Direction) {
    let complete: Vec<_> = trials
        .iter()
        .filter(|t| t.state == TrialState::Complete)
        .collect();
    if complete.is_empty() {
        return;
    }

    let mut ids = Vec::with_capacity(complete.len());
    let mut vals = Vec::with_capacity(complete.len());
    let mut best_vals = Vec::with_capacity(complete.len());
    let mut best = complete[0].value;
    for t in &complete {
        ids.push(t.id);
        vals.push(t.value);
        best = match direction {
            Direction::Minimize => best.min(t.value),
            Direction::Maximize => best.max(t.value),
        };
        best_vals.push(best);
    }

    let _ = write!(
        html,
        r##"<script>
Plotly.newPlot("history", [
  {{ x: {ids:?}, y: {vals:?}, mode: "markers", name: "Objective", type: "scatter",
     marker: {{ color: "#3498db", size: 6 }} }},
  {{ x: {ids:?}, y: {best_vals:?}, mode: "lines", name: "Best so far", type: "scatter",
     line: {{ color: "#e74c3c", width: 2 }} }}
], {{ xaxis: {{ title: "Trial" }}, yaxis: {{ title: "Objective Value" }},
     margin: {{ t: 10 }}, legend: {{ x: 1, xanchor: "right", y: 1 }} }},
   {{ responsive: true }});
</script>
"##,
    );
}

fn write_slice_charts(
    html: &mut String,
    trials: &[CompletedTrial<f64>],
    param_info: &BTreeMap<ParamId, ParamMeta>,
) {
    let complete: Vec<_> = trials
        .iter()
        .filter(|t| t.state == TrialState::Complete)
        .collect();
    if complete.is_empty() {
        return;
    }

    let n_params = param_info.len();
    let cols = if n_params <= 2 { n_params } else { 2 };
    let rows = n_params.div_ceil(cols);

    // Build subplot titles and data.
    let mut subplot_titles = Vec::new();
    let mut traces = String::new();
    for (i, (id, meta)) in param_info.iter().enumerate() {
        subplot_titles.push(format!("\"{}\"", escape_js(&meta.label)));

        let mut x_vals = Vec::new();
        let mut y_vals = Vec::new();
        for t in &complete {
            if let Some(pv) = t.params.get(id) {
                x_vals.push(param_value_to_f64(pv));
                y_vals.push(t.value);
            }
        }

        let subplot_idx = i + 1;
        let xa = if subplot_idx == 1 {
            "x".to_string()
        } else {
            format!("x{subplot_idx}")
        };
        let ya = if subplot_idx == 1 {
            "y".to_string()
        } else {
            format!("y{subplot_idx}")
        };
        let _ = write!(
            traces,
            r##"{{ x: {x_vals:?}, y: {y_vals:?}, mode: "markers", type: "scatter",
               xaxis: "{xa}", yaxis: "{ya}",
               marker: {{ color: "#3498db", size: 5 }}, showlegend: false }},"##,
        );
    }

    let _ = write!(
        html,
        r#"<script>
Plotly.newPlot("slices", [{traces}],
  {{ grid: {{ rows: {rows}, columns: {cols}, pattern: "independent" }},
     annotations: [{annotations}],
     margin: {{ t: 30 }}, showlegend: false }},
  {{ responsive: true }});
</script>
"#,
        annotations = build_subplot_annotations(&subplot_titles, rows, cols),
    );
}

fn write_parallel_coordinates(
    html: &mut String,
    trials: &[CompletedTrial<f64>],
    param_info: &BTreeMap<ParamId, ParamMeta>,
    direction: Direction,
) {
    let complete: Vec<_> = trials
        .iter()
        .filter(|t| t.state == TrialState::Complete)
        .collect();
    if complete.is_empty() {
        return;
    }

    let mut dimensions = String::new();

    // Add objective value as the first dimension.
    let obj_vals: Vec<f64> = complete.iter().map(|t| t.value).collect();
    let _ = write!(
        dimensions,
        r#"{{ label: "Objective", values: {obj_vals:?} }},"#,
    );

    // Add each parameter as a dimension.
    for (id, meta) in param_info {
        let vals: Vec<f64> = complete
            .iter()
            .map(|t| t.params.get(id).map_or(f64::NAN, param_value_to_f64))
            .collect();
        let _ = write!(
            dimensions,
            r#"{{ label: "{label}", values: {vals:?} }},"#,
            label = escape_js(&meta.label),
        );
    }

    // Color by objective value: green = good, red = bad.
    let (cmin, cmax) = min_max(&obj_vals);
    let colorscale = match direction {
        Direction::Minimize => r##"[[0,"#2ecc71"],[1,"#e74c3c"]]"##,
        Direction::Maximize => r##"[[0,"#e74c3c"],[1,"#2ecc71"]]"##,
    };

    let _ = write!(
        html,
        r#"<script>
Plotly.newPlot("parcoords", [{{
  type: "parcoords",
  line: {{ color: {obj_vals:?}, colorscale: {colorscale},
           cmin: {cmin}, cmax: {cmax}, showscale: true }},
  dimensions: [{dimensions}]
}}], {{ margin: {{ t: 10 }} }}, {{ responsive: true }});
</script>
"#,
    );
}

fn write_importance_chart(html: &mut String, importance: &[(String, f64)]) {
    let names: Vec<_> = importance
        .iter()
        .map(|(n, _)| format!("\"{}\"", escape_js(n)))
        .collect();
    let values: Vec<f64> = importance.iter().map(|(_, v)| *v).collect();

    let _ = write!(
        html,
        r##"<script>
Plotly.newPlot("importance", [{{
  x: {values:?}, y: [{names}], type: "bar", orientation: "h",
  marker: {{ color: "#9b59b6" }}
}}], {{ xaxis: {{ title: "Importance (|Spearman correlation|)" }},
       yaxis: {{ automargin: true }}, margin: {{ t: 10, l: 120 }} }},
   {{ responsive: true }});
</script>
"##,
        names = names.join(","),
    );
}

fn write_timeline_chart(html: &mut String, trials: &[CompletedTrial<f64>]) {
    let mut ids = Vec::with_capacity(trials.len());
    let mut colors = Vec::with_capacity(trials.len());
    let mut labels = Vec::with_capacity(trials.len());

    for t in trials {
        ids.push(format!("\"Trial {}\"", t.id));
        let (color, label) = match t.state {
            TrialState::Complete => ("#2ecc71", "Complete"),
            TrialState::Pruned => ("#f39c12", "Pruned"),
            TrialState::Failed => ("#e74c3c", "Failed"),
            TrialState::Running => ("#3498db", "Running"),
        };
        colors.push(format!("\"{color}\""));
        labels.push(format!("\"{label}\""));
    }

    // Use trial index as a proxy for duration (no wallclock data available).
    let indices: Vec<usize> = (0..trials.len()).collect();

    let _ = write!(
        html,
        r#"<script>
Plotly.newPlot("timeline", [{{
  y: [{ids}], x: {indices:?}, type: "bar", orientation: "h",
  text: [{labels}], textposition: "auto",
  marker: {{ color: [{colors}] }}
}}], {{ xaxis: {{ title: "Trial Index" }}, yaxis: {{ automargin: true, autorange: "reversed" }},
       margin: {{ t: 10, l: 80 }}, showlegend: false }},
   {{ responsive: true }});
</script>
"#,
        ids = ids.join(","),
        colors = colors.join(","),
        labels = labels.join(","),
    );
}

fn write_intermediate_chart(html: &mut String, trials: &[CompletedTrial<f64>]) {
    let trials_with_iv: Vec<_> = trials
        .iter()
        .filter(|t| !t.intermediate_values.is_empty())
        .collect();
    if trials_with_iv.is_empty() {
        return;
    }

    let mut traces = String::new();
    for t in &trials_with_iv {
        let steps: Vec<u64> = t.intermediate_values.iter().map(|(s, _)| *s).collect();
        let values: Vec<f64> = t.intermediate_values.iter().map(|(_, v)| *v).collect();
        let color = match t.state {
            TrialState::Pruned => "#f39c12",
            _ => "#3498db",
        };
        let _ = write!(
            traces,
            r#"{{ x: {steps:?}, y: {values:?}, mode: "lines+markers", name: "Trial {id}",
               line: {{ color: "{color}", width: 1 }}, marker: {{ size: 3 }} }},"#,
            id = t.id,
        );
    }

    let _ = write!(
        html,
        r#"<script>
Plotly.newPlot("intermediate", [{traces}],
  {{ xaxis: {{ title: "Step" }}, yaxis: {{ title: "Intermediate Value" }},
     margin: {{ t: 10 }}, showlegend: true }},
  {{ responsive: true }});
</script>
"#,
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn param_value_to_f64(pv: &ParamValue) -> f64 {
    match *pv {
        ParamValue::Float(v) => v,
        ParamValue::Int(v) => v as f64,
        ParamValue::Categorical(v) => v as f64,
    }
}

fn escape_js(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn min_max(vals: &[f64]) -> (f64, f64) {
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for &v in vals {
        if v.is_nan() {
            continue;
        }
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    // If all values were NaN, return 0.0..1.0 as a safe fallback.
    if mn > mx {
        return (0.0, 1.0);
    }
    (mn, mx)
}

/// Build Plotly annotation objects to act as subplot titles.
#[allow(clippy::cast_precision_loss)]
fn build_subplot_annotations(titles: &[String], rows: usize, cols: usize) -> String {
    let mut anns = Vec::new();
    for (i, title) in titles.iter().enumerate() {
        let row = i / cols;
        let col = i % cols;
        // Compute x/y anchor in paper coordinates.
        let x = if cols == 1 {
            0.5
        } else {
            (f64::from(u32::try_from(col).unwrap_or(0))) / (cols as f64 - 1.0)
        };
        let y = 1.0 - (f64::from(u32::try_from(row).unwrap_or(0))) / (rows as f64).max(1.0) + 0.02;
        anns.push(format!(
            r#"{{ text: {title}, x: {x:.3}, y: {y:.3}, xref: "paper", yref: "paper",
               showarrow: false, font: {{ size: 12 }} }}"#,
        ));
    }
    anns.join(",")
}
