use crate::{
    conditional_feedback::Feedback,
    error::{Error, Result},
    exact_factors::DetectionKey,
    exact_posterior::{fsum, InferenceResult},
    factorized_groups::birth_identity,
    gnomonic::GnomonicPlane,
    hypothesis_selection::{bounded_product, select_joint_histories},
    online_factors::State,
    tracking_model::{Background, JointHistory, Track},
};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub struct BirthState {
    pub key: DetectionKey,
    pub kind: String,
    pub state: Option<State>,
    pub plane: GnomonicPlane,
}

pub struct Context<'a> {
    pub keys: &'a [DetectionKey],
    pub intensities: &'a [(DetectionKey, Vec<(String, f64)>)],
    pub births: &'a [BirthState],
    pub priors: &'a [JointHistory],
    pub clutter_density: f64,
    pub omitted: f64,
    pub max_candidates: i64,
    pub top_k: i64,
    pub preserve: bool,
}

#[derive(Debug, Serialize)]
pub struct Updated {
    pub histories: Vec<JointHistory>,
    pub birth_loss: f64,
    pub joint_loss: f64,
    pub audit: Value,
}

fn label(prefix: &str, key: &DetectionKey) -> String {
    format!("{prefix}:{}:{}:{}", key.0, key.1.chars().count(), key.1)
}

fn missing() -> Error {
    Error::value("missing joint update binding")
}

pub fn expand(result: &InferenceResult, feedback: &[Feedback], ctx: &Context) -> Result<Updated> {
    let finite: Vec<_> = result
        .hypotheses
        .iter()
        .filter(|h| h.log_unnormalized_weight.is_finite())
        .collect();
    if finite.len() != feedback.len() {
        return Err(missing());
    }
    let mut support: Vec<_> = ctx
        .keys
        .iter()
        .map(|k| (k.clone(), BTreeMap::<String, f64>::new()))
        .collect();
    let mut output = vec![];
    let mut birth_loss = 0.;
    for (h, fb) in finite.into_iter().zip(feedback) {
        if h.hypothesis_id != fb.hypothesis_id {
            return Err(missing());
        }
        let weight = h.posterior_weight * (1. - ctx.omitted);
        for (source, key) in &h.target_assignments {
            if let Some(key) = key {
                let beta = &mut support
                    .iter_mut()
                    .find(|(k, _)| k == key)
                    .ok_or_else(missing)?
                    .1;
                *beta.entry(source.clone()).or_default() += weight;
            }
        }
        let mut alternatives = vec![];
        for key in &h.clutter_detection_keys {
            let intensities = &ctx
                .intensities
                .iter()
                .rev()
                .find(|(k, _)| k == key)
                .ok_or_else(missing)?
                .1;
            let total = ctx.clutter_density + intensities.iter().fold(0., |a, (_, v)| a + v);
            if total == 0. {
                return Err(Error {
                    kind: "ZeroDivisionError",
                    code: None,
                    message: "float division by zero".into(),
                });
            }
            let beta = &mut support
                .iter_mut()
                .find(|(k, _)| k == key)
                .ok_or_else(missing)?
                .1;
            let target = intensities
                .iter()
                .rev()
                .find(|(kind, _)| kind == "T")
                .map(|(_, v)| *v)
                .unwrap_or(0.);
            *beta.entry(format!("T:{}", label("N", key))).or_default() += weight * target / total;
            let mut weights = vec![(ctx.clutter_density, ("C".to_owned(), key.clone()))];
            weights.extend(
                intensities
                    .iter()
                    .filter(|(_, v)| *v > 0.)
                    .map(|(kind, v)| (*v, (kind.clone(), key.clone()))),
            );
            let den = fsum(weights.iter().map(|(w, _)| *w))?;
            alternatives.push(
                weights
                    .into_iter()
                    .filter(|(w, _)| *w > 0.)
                    .map(|(w, event)| (w / den, event))
                    .collect(),
            );
        }
        let (possibilities, loss) = bounded_product(&alternatives, ctx.max_candidates)?;
        birth_loss += h.posterior_weight * loss;
        for (probability, events) in possibilities {
            let mut tracks = fb.tracks.clone();
            let mut backgrounds = fb.backgrounds.clone();
            let mut ta = h.target_assignments.clone();
            let mut ba = h.unknown_star_assignments.clone();
            let mut clutter = vec![];
            for (kind, key) in &events {
                if kind == "C" {
                    clutter.push(key.clone());
                    continue;
                }
                let b = ctx
                    .births
                    .iter()
                    .rev()
                    .find(|b| &b.key == key && &b.kind == kind)
                    .ok_or_else(missing)?;
                let id = label(if kind == "T" { "N" } else { "U" }, key);
                let mut v = json!({
                    "existence": 1.,
                    "state": b.state,
                    "plane": b.plane,
                    "support_keys": [key]
                });
                if kind == "T" {
                    v["label"] = json!(id);
                    tracks.push(Track::from_json(&v)?);
                    ta.push((format!("T:{id}"), Some(key.clone())));
                } else {
                    v["background_id"] = json!(id);
                    backgrounds.push(Background::from_json(&v)?);
                    ba.push((format!("S_U:{id}"), Some(key.clone())));
                }
            }
            output.push(JointHistory::from_json(&json!({
                            "hypothesis_id": birth_identity(&h.hypothesis_id,
            &events)?,
                            "weight": h.posterior_weight*probability,
                            "tracks": tracks,
                            "backgrounds": backgrounds,
                            "target_assignments": ta,
                            "background_assignments": ba,
                            "star_assignments": h.star_assignments,
                            "clutter_keys": clutter
                        }))?);
        }
    }
    let targets = ctx
        .priors
        .iter()
        .flat_map(|p| p.tracks.iter().map(|t| t.label.clone()))
        .collect::<Vec<_>>();
    let (mut histories, loss) = select_joint_histories(&output, ctx.top_k, &targets, ctx.preserve)?;
    birth_loss += loss;
    let den = fsum(histories.iter().map(|h| h.weight))?;
    let joint_loss = (1. - (1. - ctx.omitted) * den).max(0.);
    let support: Vec<_> = support
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().collect::<Vec<_>>()))
        .collect();
    let audit = json!({
        "detection_keys": ctx.keys,
        "pre_birth_association_support": support,
        "retained_mass_within_evaluated": (1.-joint_loss).max(0.),
        "selection": if ctx.preserve {
            "EXISTING_TARGET_ASSOCIATION_DIVERSITY"
        } else {
            "JOINT_TOP_K"
        },
        "scope": "CONDITIONAL_ON_EVALUATED_CANDIDATES; GATE_AND_BEAM_LOSS_UNKNOWN"
    });
    for h in &mut histories {
        h.weight /= den;
    }
    Ok(Updated {
        histories,
        birth_loss,
        joint_loss,
        audit,
    })
}
