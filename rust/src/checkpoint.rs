use crate::{
    error::{Error, Result},
    frame_commit::State,
    input_evidence,
    numerical_policy::CHECKPOINT_NORMALIZATION_ATOL,
    tracking_lifecycle::ManagementState,
    tracking_model::{JointHistory, Track},
    undetected::Population,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};

pub const SCHEMA: &str = "factorized-joint-populations-4";
pub fn model_binding(v: &Value, time_scale: &str) -> Result<String> {
    use crate::{
        camera_model::CameraModel,
        contracts::KnownStar,
        efficient_config::InferenceConfig,
        exact_factors::{ClutterModel, StarSpec},
        tracking_background::BackgroundConfig,
        tracking_lifecycle::LifecycleConfig,
        tracking_model::TrackingConfig,
        undetected::PopulationOptions,
    };
    CameraModel::from_json(&v["camera"])?;
    for star in array(v, "catalog")? {
        KnownStar::from_json(star)?;
    }
    let stars = array(v, "stars")?
        .iter()
        .map(|star| {
            let parsed = StarSpec::from_json(star)?;
            let mut raw = star.clone();
            raw["measurement_covariance_px2"] = json!(parsed.measurement_covariance_px2);
            Ok(raw)
        })
        .collect::<Result<Vec<_>>>()?;
    let parsed = ClutterModel::from_json(&v["clutter"])?;
    let mut clutter = v["clutter"].clone();
    clutter["window"] = json!(parsed.window());
    let parsed = TrackingConfig::from_json(&v["tracking"])?;
    let mut tracking = v["tracking"].clone();
    tracking["measurement_covariance_px2"] = json!(parsed.measurement_covariance_px2);
    tracking["miss_update_mode"] = json!(parsed.miss_update_mode);
    let parsed = BackgroundConfig::from_json(&v["background"])?;
    let mut background = v["background"].clone();
    background["measurement_covariance_px2"] = json!(parsed.measurement_covariance_px2);
    LifecycleConfig::from_json(&v["lifecycle"])?;
    let inference = InferenceConfig::from_json(&v["inference"])?;
    PopulationOptions::from_json(&v["options"])?;
    let angle = crate::contracts::scalar(&v["angle_degrees"], "tangent angle")?
        * (std::f64::consts::PI / 180.);
    digest(&json!({
        "version": crate::point_marginals::MODEL_VERSION,
        "camera": v["camera"],
        "catalog": v["catalog"],
        "stars": stars,
        "clutter": clutter,
        "tracking": tracking,
        "background": background,
        "lifecycle": v["lifecycle"],
        "inference": inference,
        "population": v["options"],
        "time_scale": time_scale,
        "initial_time": v["initial_timestamp_s"],
        "max_tangent_angle_rad": angle,
        "numerical_policy": crate::numerical_policy::snapshot()
    }))
}

pub fn digest(value: &Value) -> Result<String> {
    let compact = input_evidence::canonical_json(value)?;
    let mut spaced = String::new();
    let (mut quoted, mut escaped) = (false, false);
    for ch in compact.chars() {
        spaced.push(ch);
        if quoted {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                quoted = false;
            }
        } else if ch == '"' {
            quoted = true;
        } else if ch == ',' || ch == ':' {
            spaced.push(' ');
        }
    }
    Ok(format!("{:x}", Sha256::digest(spaced.as_bytes())))
}

fn array<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    v[key]
        .as_array()
        .ok_or_else(|| Error::value(format!("checkpoint invalid {key}")))
}

fn integer(v: &Value, key: &str) -> Result<i64> {
    v[key]
        .as_i64()
        .ok_or_else(|| Error::value(format!("checkpoint invalid {key}")))
}

pub fn restore(
    state: &mut State,
    raw: &Value,
    model_digest: &str,
    frame_digests: &BTreeMap<i64, String>,
) -> Result<Vec<Value>> {
    if state.frame.is_some() {
        return Err(Error::value("restore only into a fresh tracker"));
    }
    if raw["schema"] != SCHEMA {
        return Err(Error::value(
            "checkpoint schema mismatch; rerun the original input with this code version",
        ));
    }
    if raw["model_digest"] != model_digest {
        return Err(Error::value("checkpoint model mismatch"));
    }
    let data = &raw["snapshot"];
    if raw["snapshot_sha256"] != digest(data)? {
        return Err(Error::value("checkpoint integrity mismatch"));
    }
    let frame = if data["frame"].is_null() {
        None
    } else {
        Some(integer(data, "frame")?)
    };
    let version = integer(data, "version")?;
    let consumed: Vec<i64> = serde_json::from_value(data["consumed_frame_ids"].clone())
        .map_err(|_| Error::value("checkpoint invalid consumed-frame history"))?;
    let evidence = data["input_evidence_sha256"]
        .as_str()
        .ok_or_else(|| Error::value("checkpoint invalid input evidence"))?;
    input_evidence::validate_consumed_input(&consumed, evidence, frame_digests, frame, version)?;
    let mut groups = vec![];
    let mut seen = HashSet::new();
    for g in array(data, "groups")? {
        let mut hs = vec![];
        for h in array(g, "histories")? {
            let h = JointHistory::from_json(h)?;
            if h.tracks
                .iter()
                .flat_map(|t| &t.support_keys)
                .chain(h.backgrounds.iter().flat_map(|b| &b.support_keys))
                .any(|k| frame.is_none_or(|f| k.0 > f))
            {
                return Err(Error::value(
                    "checkpoint contains unconsumed future evidence",
                ));
            }
            hs.push(h);
        }
        if hs.is_empty()
            || (hs.iter().fold(0., |a, h| a + h.weight) - 1.).abs() > CHECKPOINT_NORMALIZATION_ATOL
        {
            return Err(Error::value("checkpoint group normalization"));
        }
        let labels: HashSet<_> = hs
            .iter()
            .flat_map(|h| {
                h.tracks
                    .iter()
                    .map(|t| t.label.clone())
                    .chain(h.backgrounds.iter().map(|b| b.background_id.clone()))
            })
            .collect();
        if labels.iter().any(|l| seen.contains(l)) {
            return Err(Error::value(
                "checkpoint object has multiple probability owners",
            ));
        }
        seen.extend(labels);
        groups.push(hs);
    }
    let populations = array(data, "populations")?
        .iter()
        .map(|p| {
            if p["expected_count"].as_f64().is_some_and(|x| x < 0.) {
                Err(Error::value("negative population mass"))
            } else {
                Population::from_json(p)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let management = array(data, "management")?
        .iter()
        .map(ManagementState::from_json)
        .collect::<Result<Vec<_>>>()?;
    let labels: HashSet<_> = groups
        .iter()
        .flat_map(|g| {
            g.iter()
                .flat_map(|h| h.tracks.iter().map(|t| t.label.as_str()))
        })
        .collect();
    if management
        .iter()
        .map(|m| m.label.as_str())
        .collect::<HashSet<_>>()
        != labels
    {
        return Err(Error::value(
            "checkpoint management/probability owner mismatch",
        ));
    }
    let mut retired = vec![];
    for v in array(data, "retired")? {
        let m = ManagementState::from_json(&v["management"])?;
        let branches = array(v, "conditional_branches")?
            .iter()
            .map(|branch| {
                Track::from_json(&branch[2]).map(|track| json!([branch[0], branch[1], track]))
            })
            .collect::<Result<Vec<_>>>()?;
        crate::exact_factors::probability(
            crate::contracts::scalar(&v["existence_probability"], "retired candidate existence")?,
            "retired candidate existence",
        )?;
        if v["label"] != m.label {
            return Err(Error::value(
                "retired track and management labels must match",
            ));
        }
        if integer(v, "state_version")? < 1 || integer(v, "frame")? < 0 {
            return Err(Error::value("invalid retirement version/frame"));
        }
        let mut record = v.clone();
        record["management"] = json!(m);
        record["conditional_branches"] = json!(branches);
        if record.get("approximation").is_none() {
            record["approximation"] =
                json!("MARGINAL_PROJECTION; FUTURE_SOURCE_REMOVED; NOT_PHYSICAL_DEATH");
        }
        retired.push(record);
    }
    let dormant: Vec<String> = data
        .get("dormant_background_ids")
        .map(|v| {
            serde_json::from_value(v.clone())
                .map_err(|_| Error::value("checkpoint dormant background ownership mismatch"))
        })
        .transpose()?
        .unwrap_or_default();
    let bg: HashSet<_> = groups
        .iter()
        .flat_map(|g| {
            g.iter()
                .flat_map(|h| h.backgrounds.iter().map(|b| b.background_id.as_str()))
        })
        .collect();
    if dormant.iter().collect::<HashSet<_>>().len() != dormant.len()
        || dormant.iter().any(|id| !bg.contains(id.as_str()))
    {
        return Err(Error::value(
            "checkpoint dormant background ownership mismatch",
        ));
    }
    let count = data
        .get("background_retired_count")
        .map(|x| {
            x.as_u64()
                .and_then(|x| usize::try_from(x).ok())
                .ok_or_else(|| Error::value("checkpoint invalid retired background count"))
        })
        .transpose()?
        .unwrap_or(0);
    let events = data
        .get("segment_events")
        .map(|x| {
            x.as_array()
                .cloned()
                .ok_or_else(|| Error::value("checkpoint invalid segment events"))
        })
        .transpose()?
        .unwrap_or_default();
    let timestamp_s = crate::contracts::scalar(&data["timestamp_s"], "checkpoint timestamp")?;
    *state = State {
        groups,
        populations,
        management,
        retired,
        frame,
        timestamp_s,
        version,
        dormant_background_ids: dormant,
        background_retired_count: count,
        consumed_frame_ids: consumed,
        input_evidence_sha256: evidence.into(),
    };
    Ok(events)
}
