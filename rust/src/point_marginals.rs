use crate::{
    contracts::{Detection, FrameInfo},
    error::{Error, Result},
    exact_factors::{inference, DetectionKey},
    exact_posterior::fsum,
    numerical_policy::POSTERIOR_NORMALIZATION_ATOL,
    tracking_model::JointHistory,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
pub const MODEL_VERSION: &str = "v2.1-factorized-joint-ppp-8";
pub const APPROXIMATION: &str = concat!(
    "FINITE_GATES; BOUNDED_LOCAL_JOINT_HISTORIES; GAUSSIAN_PPP_MERGE; ",
    "NONLINEAR_BIRTH_QUADRATURE; AFFINE_PPP_WINDOW_THINNING; ",
    "AFFINE_EXISTING_SOURCE_SENSOR; ",
    "CONFIGURED_TARGET_MISS_GAUSSIAN_PROJECTION; ",
    "BACKGROUND_MISS_STATE_RETENTION; ",
    "CONFIGURED_BACKGROUND_VIEW_DORMANCY_AND_RETIREMENT",
);
const FIELDS: [&str; 5] = [
    "P_Known_Star",
    "P_Unknown_Star",
    "P_Existing_Target",
    "P_Birth_Target",
    "P_Clutter",
];
#[derive(Default)]
struct Point {
    sums: [Vec<f64>; 5],
    associations: BTreeMap<String, f64>,
    confirmation: BTreeMap<String, f64>,
}

pub fn project(
    frame: &FrameInfo,
    detections: &[Detection],
    accepted: &[DetectionKey],
    groups: &[Vec<JointHistory>],
    audit: &Value,
) -> Result<Vec<Value>> {
    let mut points: BTreeMap<_, _> = accepted
        .iter()
        .map(|key| (key.clone(), Point::default()))
        .collect();
    let birth_prefix = format!("T:N:{}:", frame.frame);
    for hs in groups {
        for h in hs {
            for b in &h.backgrounds {
                let source = format!("S_U:{}", b.background_id);
                let key = h
                    .background_assignments
                    .iter()
                    .rev()
                    .find(|(s, _)| s == &source)
                    .and_then(|(_, k)| k.as_ref());
                if let Some(point) = key.and_then(|k| points.get_mut(k)) {
                    let count = b
                        .support_keys
                        .iter()
                        .filter(|k| {
                            i128::from(k.0) >= i128::from(frame.frame) - 7 && k.0 <= frame.frame
                        })
                        .map(|k| k.0)
                        .collect::<HashSet<_>>()
                        .len();
                    if count >= 4 {
                        *point.confirmation.entry(source).or_default() += h.weight * b.existence;
                    }
                }
            }
            let value = serde_json::to_value(h).unwrap();
            let stars: Vec<(String, Option<DetectionKey>)> =
                serde_json::from_value(value["star_assignments"].clone())
                    .map_err(|_| Error::value("invalid star assignments"))?;
            for (kind, assignments) in [
                ("T", &h.target_assignments),
                ("S_K", &stars),
                ("S_U", &h.background_assignments),
            ] {
                for (source, key) in assignments {
                    if let Some(point) = key.as_ref().and_then(|k| points.get_mut(k)) {
                        let index = if kind == "S_K" {
                            0
                        } else if kind == "S_U" {
                            1
                        } else if source.starts_with(&birth_prefix) {
                            3
                        } else {
                            2
                        };
                        point.sums[index].push(h.weight);
                        *point.associations.entry(source.clone()).or_default() += h.weight;
                    }
                }
            }
            let clutter: Vec<DetectionKey> = serde_json::from_value(value["clutter_keys"].clone())
                .map_err(|_| Error::value("invalid clutter keys"))?;
            for key in clutter {
                if let Some(point) = points.get_mut(&key) {
                    point.sums[4].push(h.weight);
                }
            }
        }
    }
    let mut retained = BTreeMap::new();
    let mut support = BTreeMap::new();
    if let Some(items) = audit.get("joint_selection").and_then(Value::as_array) {
        for item in items {
            let keys: Vec<DetectionKey> = serde_json::from_value(item["detection_keys"].clone())
                .map_err(|_| Error::value("invalid selection keys"))?;
            for key in keys {
                retained.insert(key, item["retained_mass_within_evaluated"].clone());
            }
            if let Some(rows) = item
                .get("pre_birth_association_support")
                .and_then(Value::as_array)
            {
                for row in rows {
                    let key: DetectionKey = serde_json::from_value(row[0].clone())
                        .map_err(|_| Error::value("invalid support key"))?;
                    support.insert(key, row[1].clone());
                }
            }
        }
    }
    let mut out = vec![];
    for detection in detections {
        let key = (detection.frame, detection.detection_id.clone());
        let point = points.get(&key);
        let mut values = vec![Value::Null; 5];
        if let Some(p) = point {
            let probabilities = p
                .sums
                .iter()
                .map(|s| fsum(s.iter().copied()))
                .collect::<Result<Vec<_>>>()?;
            if (probabilities.iter().fold(0., |a, v| a + v) - 1.).abs()
                > POSTERIOR_NORMALIZATION_ATOL
            {
                return Err(inference(
                    "POSTERIOR_NORMALIZATION",
                    "local five-source marginal",
                ));
            }
            values = probabilities.into_iter().map(|v| json!(v)).collect();
        }
        let pairs =
            |v: &BTreeMap<String, f64>| v.iter().map(|(s, p)| json!([s, p])).collect::<Vec<_>>();
        let mut row = json!({
            "Frame": frame.frame,
            "Timestamp": frame.timestamp_s,
            "Detection_ID": detection.detection_id,
            "Geometry_Valid": point.is_some(),
            "Model_Status": MODEL_VERSION,
            "Inference_Mode": "FACTORIZED_JOINT_PPP",
            "Approximation_Status": APPROXIMATION,
            "Association_Marginals": point.map(|p|pairs(&p.associations)).unwrap_or_default(),
            "Association_Retained_Mass": retained.get(&key),
            "Association_PreTruncation_Support": support.get(&key),
            "Unknown_Star_Confirmation_Support": point.map(|p|pairs(&p.confirmation)).unwrap_or_default(),
            "Association_Support_Scope": concat!(
                "PRE_BIRTH_SUPPORT_WITHIN_EVALUATED_ASSOCIATIONS; ",
                "GATE_AND_BEAM_LOSS_UNKNOWN",
            )
        });
        for (name, value) in FIELDS.into_iter().zip(values) {
            row[name] = value;
        }
        out.push(row);
    }
    Ok(out)
}
