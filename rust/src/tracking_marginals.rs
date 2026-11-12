use crate::{
    error::{Error, Result},
    exact_factors::DetectionKey,
    exact_posterior::fsum,
    linalg::{array, mat},
    tracking_model::{mixture_directions, mixture_moments, JointHistory},
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashSet};
fn normalize(logs: &[f64]) -> Result<(f64, Vec<f64>)> {
    let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scaled: Vec<_> = logs.iter().map(|x| libm::exp(x - maximum)).collect();
    let total = fsum(scaled.iter().copied())?;
    Ok((
        if total > 0. {
            maximum + libm::log(total)
        } else {
            f64::NEG_INFINITY
        },
        scaled.iter().map(|x| x / total).collect(),
    ))
}

fn log_value(x: f64) -> Result<Value> {
    crate::exact_io::serializable(&x)
}

fn assigned<'a>(
    assignments: &'a [(String, Option<DetectionKey>)],
    id: &str,
) -> Option<&'a DetectionKey> {
    assignments
        .iter()
        .rev()
        .find(|(k, _)| k == id)
        .and_then(|(_, v)| v.as_ref())
}

fn associations(v: BTreeMap<DetectionKey, f64>) -> Value {
    json!(v.into_iter().collect::<Vec<_>>())
}

pub fn project_background(
    histories: &[JointHistory],
    frame: i64,
    timestamp: f64,
    diagnostics: &[Value],
    detail: &str,
) -> Result<Vec<Value>> {
    crate::contracts::finite(&[timestamp], "timestamp")?;
    let identities: BTreeSet<_> = histories
        .iter()
        .flat_map(|h| h.backgrounds.iter().map(|b| b.background_id.as_str()))
        .collect();
    let mut rows = vec![];
    for identity in identities {
        let mut parts = vec![];
        let mut assoc = BTreeMap::new();
        let mut last = vec![];
        let mut miss = 0.;
        let mut template = None;
        let source = format!("S_U:{identity}");
        for h in histories {
            if let Some(key) = assigned(&h.background_assignments, &source) {
                *assoc.entry(key.clone()).or_insert(0.) += h.weight;
            } else {
                miss += h.weight;
            }
            let Some(b) = h.backgrounds.iter().find(|b| b.background_id == identity) else {
                continue;
            };
            template = Some(b);
            if b.log_existence > f64::NEG_INFINITY && h.weight > 0. {
                parts.push((
                    h.hypothesis_id.as_str(),
                    libm::log(h.weight) + b.log_existence,
                    b.state.as_ref().unwrap().clone(),
                ));
            }
            if let Some(key) = b.support_keys.last() {
                last.push(json!([h.hypothesis_id, key, h.weight * b.existence]));
            }
        }
        let (log_r, weights) = normalize(&parts.iter().map(|p| p.1).collect::<Vec<_>>())?;
        let moments = mixture_directions(
            &weights,
            &parts.iter().map(|p| p.2.clone()).collect::<Vec<_>>(),
        )?;
        let template = template.unwrap();
        let (direction, covariance) = if let Some(m) = &moments {
            let (u, j) = template.plane.inverse(m.mean)?;
            let j = nalgebra::SMatrix::<f64, 3, 2>::from_fn(|i, k| j[i][k]);
            (
                json!(u),
                json!(array(j * mat(m.covariance) * j.transpose())),
            )
        } else {
            (Value::Null, Value::Null)
        };
        let mut visibility = vec![];
        for d in diagnostics {
            if d.get("source_id").and_then(Value::as_str) == Some(source.as_str()) {
                let parent = d.get("parent_id").ok_or_else(|| Error {
                    kind: "KeyError",
                    code: None,
                    message: "'parent_id'".into(),
                })?;
                visibility.push(json!({
                    "parent_id": parent,
                    "mass": d.get("visibility_mass_cubature")
                }));
            }
        }
        let mut row = json!({
            "Background_ID": identity,
            "Frame": frame,
            "Timestamp": timestamp,
            "Existence_Probability": libm::exp(log_r),
            "Log_Existence_Probability": log_value(log_r)?,
            "J2000_Direction": direction,
            "Direction_Covariance": covariance,
            "Direction_Covariance_Approximation": "Jacobian at mixture mean; rank-two LOS covariance",
            "Tangent_Reference": template.plane,
            "State_Mean": moments.as_ref().map(|m|m.mean),
            "State_Covariance": moments.as_ref().map(|m|m.covariance),
            "Associated_Detection_Marginals": associations(assoc),
            "Miss_Association_Probability": miss,
            "Last_Valid_Observation": last,
            "Visibility": visibility,
            "Model_Status": "STATIC_INERTIAL_Q_ZERO; CONTINUOUS_APPROXIMATE",
            "Background_Hypothesis_Status": if last.is_empty() {
                "Latent_Bernoulli"
            } else {
                "Measurement_Supported"
            }
        });
        if detail == "full" {
            row["State_Mixture"] = json!(parts
                .iter()
                .zip(weights)
                .map(|(p, w)| json!({
                    "hypothesis_id": p.0,
                    "weight": w,
                    "state": p.2
                }))
                .collect::<Vec<_>>());
        }
        rows.push(row);
    }
    Ok(rows)
}

#[derive(Debug, Clone, Copy)]
pub struct ConfirmationWindow {
    window: i64,
    minimum: i64,
}

impl ConfirmationWindow {
    pub fn new(window: i64, minimum: i64) -> Result<Self> {
        for (name, n) in [
            ("confirmation_min_frames", minimum),
            ("confirmation_window_frames", window),
        ] {
            if n < 1 {
                return Err(Error::value(format!(
                    "positive frame count required: {name}"
                )));
            }
        }
        if window < minimum {
            return Err(Error::value(
                "confirmation window must include the required support frames",
            ));
        }
        Ok(Self { window, minimum })
    }
}

pub fn project_track_marginals(
    histories: &[JointHistory],
    frame: i64,
    timestamp: f64,
    config: ConfirmationWindow,
    detail: &str,
) -> Result<Vec<Value>> {
    crate::contracts::finite(&[timestamp], "timestamp")?;
    let labels: BTreeSet<_> = histories
        .iter()
        .flat_map(|h| h.tracks.iter().map(|t| t.label.as_str()))
        .collect();
    let mut rows = vec![];
    for label in labels {
        let mut parts = vec![];
        let mut assoc = BTreeMap::new();
        let mut miss = 0.;
        let mut support = 0.;
        let mut confirmation = 0.;
        let mut first: Option<i64> = None;
        let mut template = None;
        for h in histories {
            let Some(t) = h.tracks.iter().find(|t| t.label == label) else {
                miss += h.weight;
                continue;
            };
            template = Some(t);
            if let Some(key) = assigned(&h.target_assignments, &format!("T:{label}")) {
                *assoc.entry(key.clone()).or_insert(0.) += h.weight;
            } else {
                miss += h.weight;
            }
            if !t.support_keys.is_empty() {
                support += h.weight * t.existence;
                if t.log_existence > f64::NEG_INFINITY && h.weight > 0. {
                    let f = t.support_keys.iter().map(|k| k.0).min().unwrap();
                    first = Some(first.map_or(f, |old| old.min(f)));
                }
                let start = frame as i128 - config.window as i128 + 1;
                if t.support_keys
                    .iter()
                    .filter(|k| (start..=frame as i128).contains(&(k.0 as i128)))
                    .map(|k| k.0)
                    .collect::<HashSet<_>>()
                    .len() as i128
                    >= config.minimum as i128
                {
                    confirmation += h.weight * t.existence;
                }
            }
            if t.log_existence > f64::NEG_INFINITY && h.weight > 0. {
                parts.push((
                    h.hypothesis_id.as_str(),
                    libm::log(h.weight) + t.log_existence,
                    t.state.as_ref().unwrap().clone(),
                ));
            }
        }
        let (log_r, weights) = normalize(&parts.iter().map(|p| p.1).collect::<Vec<_>>())?;
        let moments = mixture_moments(
            &weights,
            &parts.iter().map(|p| p.2.clone()).collect::<Vec<_>>(),
        )?;
        let template = template.unwrap();
        let assoc = associations(assoc);
        let mut row = json!({
                    "Track_Label": label,
                    "Frame": frame,
                    "Timestamp": timestamp,
                    "Existence_Probability": libm::exp(log_r),
                    "Log_Existence_Probability": log_value(log_r)?,
                    "State_Mean": moments.as_ref().map(|m|m.mean()),
                    "State_Covariance": moments.as_ref().map(|m|m.covariance()),
                    "Tangent_Reference_ID": template.plane.reference_id(),
                    "Tangent_Plane": template.plane,
                    "Detection_Association_Probabilities": assoc,
                    "Miss_Association_Probability": miss,
                    "Probability_Exists_And_Ever_Measured": support,
                    "First_Real_Support_Frame": first,
                    "Confirmation_Support_Probability": confirmation.clamp(0.,
        1.),
                    "Confirmation_Window_Frames": config.window,
                    "Confirmation_Min_Frames": config.minimum,
                    "Posterior_Representation": "conditional_history_Gaussian_mixture",
                    "Motion_Model": "CV_GNOMONIC",
                    "Association_Marginals": assoc,
                    "Measured_Or_Predicted": if assoc.as_array().unwrap().is_empty() {
                        "Predicted_Only"
                    } else {
                        "Measured_Update_Mixture"
                    },
                    "Approximation_Status": "CONTINUOUS_APPROXIMATE; JOINT HISTORIES RETAINED"
                });
        if detail == "full" {
            row["Conditional_Hypothesis_Metadata"] = json!(parts
                .iter()
                .map(|p| json!({
                    "hypothesis_id": p.0,
                    "existence_weight": libm::exp(p.1),
                    "log_existence_weight": p.1
                }))
                .collect::<Vec<_>>());
            row["State_Mixture"] = json!(parts
                .iter()
                .zip(weights)
                .map(|(p, w)| json!({
                    "hypothesis_id": p.0,
                    "weight": w,
                    "state": p.2
                }))
                .collect::<Vec<_>>());
        }
        rows.push(row);
    }
    Ok(rows)
}
