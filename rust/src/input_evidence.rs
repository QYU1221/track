use crate::{
    contracts::{Detection, FrameInfo, ObservationPacket},
    error::{Error, Result},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn empty_digest() -> String {
    hash(b"consumed-observations-1")
}

fn ascii_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (' '..='~').contains(&c) => out.push(c),
            c => {
                let mut units = [0; 2];
                for u in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
        }
    }
    out.push('"');
}

pub fn python_float(value: f64) -> Result<String> {
    if !value.is_finite() {
        return Err(Error::value(
            "Out of range float values are not JSON compliant",
        ));
    }
    if value == 0. {
        return Ok(if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .into());
    }
    let negative = value < 0.;
    let shortest = serde_json::to_string(&value.abs()).map_err(|e| Error::value(e.to_string()))?;
    let (mantissa, explicit) = shortest
        .split_once('e')
        .map(|(m, e)| (m, e.parse::<i32>().unwrap()))
        .unwrap_or((&shortest, 0));
    let point = mantissa.find('.').unwrap_or(mantissa.len());
    let raw = mantissa.replace('.', "");
    let leading = raw.bytes().take_while(|c| *c == b'0').count();
    let exponent = explicit + point as i32 - leading as i32 - 1;
    let digits = raw[leading..].trim_end_matches('0');
    let mantissa = if digits.len() == 1 {
        digits.to_owned()
    } else {
        format!("{}.{}", &digits[..1], &digits[1..])
    };
    let sign = if negative { "-" } else { "" };
    if !(-4..16).contains(&exponent) {
        return Ok(format!(
            "{sign}{mantissa}e{}{:#02}",
            if exponent >= 0 { '+' } else { '-' },
            exponent.abs()
        ));
    }
    let point = exponent + 1;
    let body = if point <= 0 {
        format!("0.{}{}", "0".repeat((-point) as usize), digits)
    } else if point as usize >= digits.len() {
        format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
    } else {
        format!(
            "{}.{}",
            &digits[..point as usize],
            &digits[point as usize..]
        )
    };
    Ok(format!("{sign}{body}"))
}

pub fn canonical_json(value: &Value) -> Result<String> {
    fn append(v: &Value, out: &mut String) -> Result<()> {
        match v {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::String(s) => ascii_string(s, out),
            Value::Number(n) => {
                let raw = n.to_string();
                if raw.contains(['.', 'e', 'E']) {
                    out.push_str(&python_float(n.as_f64().ok_or_else(|| {
                        Error::value("Out of range float values are not JSON compliant")
                    })?)?);
                } else {
                    out.push_str(&raw);
                }
            }
            Value::Array(values) => {
                out.push('[');
                for (i, x) in values.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    append(x, out)?;
                }
                out.push(']');
            }
            Value::Object(values) => {
                out.push('{');
                let mut keys: Vec<_> = values.keys().collect();
                keys.sort();
                for (i, k) in keys.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    ascii_string(k, out);
                    out.push(':');
                    append(&values[*k], out)?;
                }
                out.push('}');
            }
        }
        Ok(())
    }
    let mut out = String::new();
    append(value, &mut out)?;
    Ok(out)
}

pub fn digest(value: &Value) -> Result<String> {
    Ok(hash(canonical_json(value)?.as_bytes()))
}

pub fn frame_digest(frame: &FrameInfo, detections: &[Detection]) -> Result<String> {
    let metadata = json!({
        "frame": frame.frame,
        "timestamp_s": frame.timestamp_s,
        "input_available": frame.input_available,
        "attitude_valid": frame.attitude_valid,
        "quaternion_xyzw": frame.quaternion_xyzw
    });
    let mut ds: Vec<_> = detections.iter().collect();
    ds.sort_by_key(|d| d.key());
    let mut points = vec![];
    for d in ds {
        let mut features = d.features.clone();
        features.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.partial_cmp(&b.1).unwrap()));
        points.push(json!({
            "frame": d.frame,
            "detection_id": d.detection_id,
            "x_px": d.x_px,
            "y_px": d.y_px,
            "covariance_px2": d.covariance_px2,
            "covariance_status": d.covariance_status,
            "features": features,
            "valid": d.valid,
            "invalid_reason": d.invalid_reason
        }));
    }
    digest(&json!({
        "metadata": metadata,
        "detections": points
    }))
}

pub fn extend_digest(previous: &str, number: i64, evidence: &str) -> Result<String> {
    digest(&json!([previous, number, evidence]))
}

pub fn packet_frame_digests(packet: &ObservationPacket) -> Result<BTreeMap<i64, String>> {
    let mut by_frame: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (index, detection) in packet.detections().iter().enumerate() {
        by_frame.entry(detection.frame).or_default().push(index);
    }
    packet
        .frames()
        .iter()
        .map(|frame| {
            let detections = by_frame
                .get(&frame.frame)
                .into_iter()
                .flatten()
                .map(|index| packet.detections()[*index].clone())
                .collect::<Vec<_>>();
            Ok((frame.frame, frame_digest(frame, &detections)?))
        })
        .collect()
}

pub fn validate_consumed_input(
    frame_ids: &[i64],
    expected: &str,
    frame_digests: &BTreeMap<i64, String>,
    last_frame: Option<i64>,
    version: i64,
) -> Result<()> {
    if frame_ids.iter().any(|n| *n < 0)
        || frame_ids.windows(2).any(|w| w[1] <= w[0])
        || version < 0
        || frame_ids.len() as u64 != version as u64
        || frame_ids.last().copied() != last_frame
    {
        return Err(Error::value("checkpoint invalid consumed-frame history"));
    }
    let mut actual = empty_digest();
    for number in frame_ids {
        let evidence = frame_digests.get(number).ok_or_else(|| {
            Error::value(format!(
                "checkpoint consumed input frame is missing: {number}"
            ))
        })?;
        actual = extend_digest(&actual, *number, evidence)?;
    }
    if actual != expected {
        return Err(Error::value(
            "checkpoint consumed input differs: detections or frame metadata changed",
        ));
    }
    Ok(())
}
