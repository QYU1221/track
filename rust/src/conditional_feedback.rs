use crate::{
    error::{Error, Result},
    exact_factors::DetectionKey,
    exact_posterior::InferenceResult,
    online_factors::State,
    tracking_model::{Background, JointHistory, Track},
};
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct Binding {
    pub parent_id: String,
    pub source_id: String,
    pub states: Vec<(Option<DetectionKey>, Option<State>)>,
    pub pd_bar: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Feedback {
    pub hypothesis_id: String,
    pub tracks: Vec<Track>,
    pub backgrounds: Vec<Background>,
}

fn missing(name: &str) -> Error {
    Error::value(format!("missing conditional feedback binding: {name}"))
}

pub fn restore(
    result: &InferenceResult,
    priors: &[JointHistory],
    bindings: &[Binding],
    retirement_min_pd: f64,
    check: &mut impl FnMut() -> Result<()>,
) -> Result<Vec<Feedback>> {
    let mut output = vec![];
    for h in &result.hypotheses {
        check()?;
        if !h.log_unnormalized_weight.is_finite() {
            continue;
        }
        let before = priors
            .iter()
            .rev()
            .find(|p| p.hypothesis_id == h.parent_hypothesis_id)
            .ok_or_else(|| missing(&h.parent_hypothesis_id))?;
        let mut tracks = vec![];
        let mut backgrounds = vec![];
        for (kind, objects) in [
            (
                "T",
                before
                    .tracks
                    .iter()
                    .map(|t| (t.label.as_str(), serde_json::to_value(t).unwrap()))
                    .collect::<Vec<_>>(),
            ),
            (
                "S_U",
                before
                    .backgrounds
                    .iter()
                    .map(|b| (b.background_id.as_str(), serde_json::to_value(b).unwrap()))
                    .collect(),
            ),
        ] {
            for (id, mut object) in objects {
                let source = format!("{kind}:{id}");
                let key = &h
                    .target_assignments
                    .iter()
                    .chain(&h.unknown_star_assignments)
                    .rev()
                    .find(|(s, _)| s == &source)
                    .ok_or_else(|| missing(&source))?
                    .1;
                let reference = &h
                    .conditional_state_references
                    .iter()
                    .rev()
                    .find(|(s, _)| s == &source)
                    .ok_or_else(|| missing(&source))?
                    .1;
                let posterior = result
                    .conditional_states
                    .iter()
                    .rev()
                    .find(|s| &s.reference == reference)
                    .ok_or_else(|| missing(reference))?;
                if posterior.log_existence == f64::NEG_INFINITY {
                    continue;
                }
                let binding = bindings
                    .iter()
                    .rev()
                    .find(|b| b.parent_id == before.hypothesis_id && b.source_id == source)
                    .ok_or_else(|| missing(&source))?;
                let state = &binding
                    .states
                    .iter()
                    .rev()
                    .find(|(k, _)| k == key)
                    .ok_or_else(|| missing(&source))?
                    .1;
                object["state"] = serde_json::to_value(state).unwrap();
                object["existence"] = serde_json::json!(posterior.existence);
                object["log_existence"] = serde_json::json!(posterior.log_existence);
                if let Some(key) = key {
                    object["support_keys"]
                        .as_array_mut()
                        .unwrap()
                        .push(serde_json::json!(key));
                }
                if kind == "T" {
                    tracks.push(Track::from_json(&object)?);
                } else {
                    let count = if key.is_some() {
                        0
                    } else {
                        object["visible_miss_count"]
                            .as_u64()
                            .unwrap()
                            .checked_add(u64::from(binding.pd_bar >= retirement_min_pd))
                            .ok_or_else(|| Error::value("background visible miss count overflow"))?
                    };
                    object["visible_miss_count"] = serde_json::json!(count);
                    backgrounds.push(Background::from_json(&object)?);
                }
            }
        }
        output.push(Feedback {
            hypothesis_id: h.hypothesis_id.clone(),
            tracks,
            backgrounds,
        });
    }
    Ok(output)
}
