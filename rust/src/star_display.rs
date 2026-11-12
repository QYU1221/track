use crate::error::{Error, Result};
use serde_json::{json, Value};
pub fn classify_star(point: &Value) -> Result<Value> {
    let number = |key: &str| {
        point[key]
            .as_f64()
            .ok_or_else(|| Error::value(format!("missing numeric {key}")))
    };
    let mass = point["Association_Retained_Mass"]
        .as_f64()
        .unwrap_or(0.)
        .clamp(0., 1.);
    let support = (number("P_Known_Star")? + number("P_Unknown_Star")?) * mass;
    if support < 0.70 {
        return Ok(json!([
            "未知",
            "STAR_SUPPORT_BELOW_CANDIDATE_THRESHOLD",
            null,
            support
        ]));
    }
    let mut evidence = std::collections::HashMap::new();
    if let Some(rows) = point["Unknown_Star_Confirmation_Support"].as_array() {
        for row in rows {
            if let (Some(id), Some(value)) = (row[0].as_str(), row[1].as_f64()) {
                evidence.insert(id, value);
            }
        }
    }
    let mut confirmed = vec![];
    if let Some(rows) = point["Association_Marginals"].as_array() {
        for row in rows {
            let id = row[0]
                .as_str()
                .ok_or_else(|| Error::value("invalid association source"))?;
            let p = row[1]
                .as_f64()
                .ok_or_else(|| Error::value("invalid association probability"))?;
            let strength = if id.starts_with("S_K:") {
                p * mass
            } else if id.starts_with("S_U:") {
                p.min(*evidence.get(id).unwrap_or(&0.)) * mass
            } else {
                continue;
            };
            if strength >= 0.95 {
                confirmed.push((id, strength));
            }
        }
    }
    if support >= 0.95 && confirmed.len() == 1 {
        let (id, strength) = confirmed[0];
        return Ok(json!([
            "恒星",
            if id.starts_with("S_K:") {
                "CONFIRMED_CATALOG_STAR"
            } else {
                "CONFIRMED_STATIC_STAR_4_OF_8_FRAMES"
            },
            id,
            strength
        ]));
    }
    Ok(json!([
        "恒星候选",
        "STAR_IDENTITY_OR_MULTIFRAME_EVIDENCE_INSUFFICIENT",
        null,
        support
    ]))
}
