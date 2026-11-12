use crate::{
    contracts::*,
    error::{Error, Result},
    geometry::adapt_quaternion,
};
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::Path,
};

pub const PUBLIC_INPUT_SCHEMA_VERSION: &str = "v2.0-stage0.1";
pub const FEATURES: [&str; 10] = [
    "Flux",
    "Peak_Intensity",
    "Area",
    "SNR",
    "Confidence",
    "Background_Mean",
    "Background_Std",
    "PSF_Width_X",
    "PSF_Width_Y",
    "Fit_Residual",
];
pub const FRAME_FIELDS: [&str; 8] = [
    "Frame",
    "Timestamp_s",
    "Input_Available",
    "Attitude_Valid",
    "q1",
    "q2",
    "q3",
    "q4_scalar",
];
pub const DETECTION_FIELDS: [&str; 20] = [
    "Frame",
    "Detection_ID",
    "x",
    "y",
    "Sigma_X",
    "Sigma_Y",
    "Cov_XY",
    "Covariance_Status",
    "Valid",
    "Invalid_Reason",
    "Flux",
    "Peak_Intensity",
    "Area",
    "SNR",
    "Confidence",
    "Background_Mean",
    "Background_Std",
    "PSF_Width_X",
    "PSF_Width_Y",
    "Fit_Residual",
];
pub const CATALOG_FIELDS: [&str; 4] = ["KnownStar_ID", "LOS_J2000_X", "LOS_J2000_Y", "LOS_J2000_Z"];
pub type Row = Map<String, Value>;
pub(crate) fn io_error(e: std::io::Error) -> Error {
    Error {
        kind: match e.kind() {
            std::io::ErrorKind::NotFound => "FileNotFoundError",
            std::io::ErrorKind::PermissionDenied => "PermissionError",
            _ => "OSError",
        },
        code: None,
        message: e.to_string(),
    }
}

pub(crate) fn key<'a>(v: &'a Value, k: &str) -> Result<&'a Value> {
    v.get(k).ok_or_else(|| Error {
        kind: "KeyError",
        code: None,
        message: format!("'{k}'"),
    })
}

pub fn py_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        _ => v.to_string(),
    }
}

pub fn py_repr(v: &Value) -> String {
    if let Value::String(s) = v {
        let quote = if s.contains('\'') && !s.contains('"') {
            '"'
        } else {
            '\''
        };
        let escaped = s
            .replace('\\', "\\\\")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
            .replace(quote, &format!("\\{quote}"));
        format!("{quote}{escaped}{quote}")
    } else {
        py_text(v)
    }
}

fn empty(v: &Value) -> bool {
    v.is_null() || py_text(v).trim().is_empty()
}

pub(crate) fn parse_float(s: &str) -> Option<f64> {
    let bytes = s.trim().as_bytes();
    if bytes.iter().enumerate().any(|(i, b)| {
        *b == b'_'
            && (i == 0
                || i + 1 == bytes.len()
                || !bytes[i - 1].is_ascii_digit()
                || !bytes[i + 1].is_ascii_digit())
    }) {
        return None;
    }
    s.trim().replace('_', "").parse().ok()
}

pub fn number(v: &Value, name: &str, required: bool) -> Result<Option<f64>> {
    if v.is_boolean() {
        return Err(Error::value(format!("{name} must be numeric, not boolean")));
    }
    if empty(v) {
        return if required {
            Err(Error::value(format!("{name} is required")))
        } else {
            Ok(None)
        };
    }
    let n = match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => parse_float(s),
        _ => None,
    }
    .ok_or_else(|| Error::value(format!("{name} must be numeric")))?;
    if !n.is_finite() {
        return Err(Error::value(format!("{name} must be finite")));
    }
    Ok(Some(n))
}

pub fn flag(v: &Value, name: &str, default: Option<bool>) -> Result<bool> {
    if empty(v) {
        return default.ok_or_else(|| Error::value(format!("{name} is required")));
    }
    match py_text(v).trim().to_lowercase().as_str() {
        "1" | "1.0" | "true" => Ok(true),
        "0" | "0.0" | "false" => Ok(false),
        _ => Err(Error::value(format!("{name} must be 0/1 or true/false"))),
    }
}

pub fn parse_frame(v: &Value) -> Result<i64> {
    let invalid = || {
        Error::value(format!(
            "Frame must be a finite non-negative integer: {}",
            py_repr(v)
        ))
    };
    let range = || {
        Error::value(format!(
            "Frame must be an integer in [0, 2147483647]: {}",
            py_repr(v)
        ))
    };
    let text = py_text(v).trim().replace('_', "");
    let mut s = text.as_str();
    let negative = s.starts_with('-');
    if s.starts_with(['-', '+']) {
        s = &s[1..];
    }
    let lower = s.to_lowercase();
    if lower == "inf"
        || lower == "infinity"
        || ["nan", "snan"].iter().any(|x| {
            lower
                .strip_prefix(x)
                .is_some_and(|suffix| suffix.bytes().all(|b| b.is_ascii_digit()))
        })
    {
        return Err(range());
    }
    let split: Vec<_> = s.split(['e', 'E']).collect();
    if split.len() > 2 {
        return Err(invalid());
    }
    let exponent = if split.len() == 2 {
        let e = split[1];
        let digits = e.trim_start_matches(['+', '-']);
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Err(invalid());
        }
        e.parse::<i64>().map_err(|_| invalid())?
    } else {
        0
    };
    let mantissa: Vec<_> = split[0].split('.').collect();
    if mantissa.len() > 2 {
        return Err(invalid());
    }
    let mut digits = mantissa.join("");
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid());
    }
    digits = digits.trim_start_matches('0').to_owned();
    if digits.is_empty() {
        return Ok(0);
    }
    if negative {
        return Err(range());
    }
    let fraction = if mantissa.len() == 2 {
        mantissa[1].len() as i64
    } else {
        0
    };
    let shift = exponent.saturating_sub(fraction);
    if shift < 0 {
        let remove = shift.unsigned_abs();
        if remove > digits.len() as u64 {
            return Err(range());
        }
        let keep = digits.len() - remove as usize;
        if !digits[keep..].bytes().all(|b| b == b'0') {
            return Err(range());
        }
        digits.truncate(keep);
    } else {
        if shift > 10 || digits.len() as i64 + shift > 10 {
            return Err(range());
        }
        digits.extend(std::iter::repeat_n('0', shift as usize));
    }
    digits
        .parse::<i64>()
        .ok()
        .filter(|n| *n <= 2147483647)
        .ok_or_else(range)
}

pub fn timestamp_seconds(v: &Value, unit: &str) -> Result<Option<f64>> {
    if empty(v) {
        return Ok(None);
    }
    if unit != "iso" {
        let factor = match unit {
            "s" => 1.,
            "ms" => 0.001,
            "ns" => 1e-9,
            _ => {
                return Err(Error {
                    kind: "KeyError",
                    code: None,
                    message: format!("'{unit}'"),
                })
            }
        };
        return Ok(number(v, "Timestamp", true)?.map(|n| n * factor));
    }
    let original = py_text(v);
    let s = original.trim();
    if !s.contains('T') && !s.contains(' ') {
        return Err(Error::value(
            "ISO Timestamp requires a date, time and timezone",
        ));
    }
    let invalid = || {
        Error::value(format!(
            "invalid timezone-aware ISO Timestamp: {}",
            py_repr(v)
        ))
    };
    if s.len() < 11 || !s.is_char_boundary(11) || !s.is_char_boundary(10) {
        return Err(invalid());
    }
    let mut normalized = s.to_owned();
    if normalized.as_bytes().get(10) == Some(&b' ') {
        normalized.replace_range(10..11, "T");
    }
    let offset = normalized
        .get(11..)
        .and_then(|t| t.find(['+', '-', 'Z']))
        .map(|i| i + 11)
        .ok_or_else(invalid)?;
    let time = &normalized[11..offset];
    if let Some((_, fraction)) = time.split_once('.') {
        if fraction.len() != 3 && fraction.len() != 6 {
            return Err(invalid());
        }
    }
    let expanded = match time.len() {
        2 => format!("{time}:00:00"),
        5 => format!("{time}:00"),
        _ => time.to_owned(),
    };
    if expanded.get(6..8) == Some("60") {
        return Err(invalid());
    }
    normalized = format!("{}{}{}", &normalized[..11], expanded, &normalized[offset..]);
    let dt = chrono::DateTime::parse_from_rfc3339(&normalized).map_err(|_| invalid())?;
    let result = dt.timestamp_micros() as f64 / 1e6;
    number(&json!(result), "Timestamp", true)
}

fn defaults_mapping() -> Vec<(&'static str, Vec<&'static str>)> {
    let mut out = vec![
        ("Frame", vec!["Frame"]),
        ("Timestamp", vec!["Timestamp", "Timestamp_s"]),
        ("Detection_ID", vec!["Detection_ID"]),
        ("x", vec!["x", "Image_X_px", "Pixel_X"]),
        ("y", vec!["y", "Image_Y_px", "Pixel_Y"]),
        ("Sigma_X", vec!["Sigma_X"]),
        ("Sigma_Y", vec!["Sigma_Y"]),
        ("Cov_XY", vec!["Cov_XY"]),
        ("Valid", vec!["Valid"]),
        ("Invalid_Reason", vec!["Invalid_Reason"]),
        ("Attitude_Valid", vec!["Attitude_Valid"]),
        ("Input_Available", vec!["Input_Available"]),
        ("q1", vec!["q1"]),
        ("q2", vec!["q2"]),
        ("q3", vec!["q3"]),
        ("q4_scalar", vec!["q4_scalar"]),
    ];
    for feature in FEATURES {
        out.push((
            feature,
            if feature == "Flux" {
                vec!["Flux", "Brightness"]
            } else {
                vec![feature]
            },
        ));
    }
    out
}

#[derive(Debug, Clone, Serialize)]
pub struct NormalizeConfig {
    pub(crate) camera: Camera,
    pub(crate) time_scale: String,
    pub(crate) timestamp_unit: String,
    pub(crate) frame_period_s: Option<f64>,
    pub(crate) frame_zero: i64,
    pub(crate) timestamp_zero_s: f64,
    pub(crate) bounds_action: String,
    pub(crate) column_mapping: Map<String, Value>,
    pub(crate) max_rows: usize,
    pub(crate) attitude_direction: String,
}

impl NormalizeConfig {
    pub fn from_json(v: &Value) -> Result<Self> {
        let camera = Camera::from_json(&v["camera"])?;
        fields(
            v,
            "NormalizeConfig",
            &[
                "camera",
                "time_scale",
                "timestamp_unit",
                "frame_period_s",
                "frame_zero",
                "timestamp_zero_s",
                "bounds_action",
                "column_mapping",
                "max_rows",
                "attitude_direction",
            ],
            &[],
        )?;
        let msg = "declare time scale and timestamp unit s/ms/ns/iso";
        let time_scale = v["time_scale"]
            .as_str()
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| Error::value(msg))?
            .to_owned();
        let timestamp_unit = v["timestamp_unit"]
            .as_str()
            .filter(|s| ["s", "ms", "ns", "iso"].contains(s))
            .ok_or_else(|| Error::value(msg))?
            .to_owned();
        let frame_period_s = if v["frame_period_s"].is_null() {
            None
        } else {
            Some(
                v["frame_period_s"]
                    .as_f64()
                    .filter(|x| x.is_finite() && *x > 0.)
                    .ok_or_else(|| {
                        Error::value("explicit frame period must be positive seconds")
                    })?,
            )
        };
        let bounds_action = v["bounds_action"]
            .as_str()
            .filter(|s| ["reject", "mark_invalid"].contains(s))
            .ok_or_else(|| Error::value("invalid normalization policy"))?
            .to_owned();
        let max_rows = v["max_rows"]
            .as_u64()
            .filter(|n| *n > 0)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Error::value("invalid normalization policy"))?;
        let frame_zero = v["frame_zero"]
            .as_i64()
            .ok_or_else(|| Error::value("invalid timestamp/frame origin"))?;
        let timestamp_zero_s = v["timestamp_zero_s"]
            .as_f64()
            .or_else(|| {
                v["timestamp_zero_s"]
                    .as_bool()
                    .map(|b| if b { 1. } else { 0. })
            })
            .filter(|x| x.is_finite())
            .ok_or_else(|| Error::value("invalid timestamp/frame origin"))?;
        let column_mapping = v["column_mapping"]
            .as_object()
            .ok_or_else(|| Error::value("column_mapping must be an object"))?
            .clone();
        let allowed: HashSet<_> = defaults_mapping().iter().map(|(k, _)| *k).collect();
        if column_mapping.keys().any(|k| !allowed.contains(k.as_str())) {
            return Err(Error::value("unknown canonical field in column mapping"));
        }
        let attitude_direction = v["attitude_direction"]
            .as_str()
            .filter(|s| ["camera_to_j2000", "j2000_to_camera"].contains(s))
            .ok_or_else(|| {
                Error::value("declare attitude_direction camera_to_j2000 or j2000_to_camera")
            })?
            .to_owned();
        Ok(Self {
            camera,
            time_scale,
            timestamp_unit,
            frame_period_s,
            frame_zero,
            timestamp_zero_s,
            bounds_action,
            column_mapping,
            max_rows,
            attitude_direction,
        })
    }
}

pub(crate) fn read_json(path: &Path) -> Result<Value> {
    let s = std::fs::read_to_string(path).map_err(io_error)?;
    serde_json::from_str(&s).map_err(|e| Error {
        kind: "JSONDecodeError",
        code: None,
        message: e.to_string(),
    })
}

pub fn mapped_rows(
    header: &[String],
    rows: &[Row],
    c: &NormalizeConfig,
    positions: bool,
) -> Result<Vec<Row>> {
    for name in header {
        if [
            "true_",
            "truth_",
            "object_id",
            "true_track",
            "canonical_track",
            "track_id",
        ]
        .iter()
        .any(|p| name.to_lowercase().starts_with(p))
        {
            return Err(Error::value(format!(
                "hidden truth/track field forbidden in inference input: {name}"
            )));
        }
    }
    let lower: HashMap<_, _> = header.iter().map(|n| (n.to_lowercase(), n)).collect();
    let mut mapping = Vec::new();
    for (canonical, default) in defaults_mapping() {
        let aliases = match c.column_mapping.get(canonical) {
            None | Some(Value::Null) => default.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| Error::type_error("column aliases must be strings"))
                })
                .collect::<Result<Vec<_>>>()?,
            _ => return Err(Error::type_error("column aliases must be strings")),
        };
        let matches: Vec<_> = aliases
            .iter()
            .filter_map(|a| lower.get(&a.to_lowercase()))
            .collect();
        if matches.len() > 1 {
            return Err(Error::value(format!("ambiguous columns for {canonical}")));
        }
        if let Some(source) = matches.first() {
            mapping.push((canonical, (*source).to_string()));
        }
    }
    let required = if positions {
        vec!["Frame", "Detection_ID", "x", "y"]
    } else {
        vec!["Frame"]
    };
    if required
        .iter()
        .any(|n| !mapping.iter().any(|(k, _)| k == n))
    {
        return Err(Error::value(format!(
            "missing mandatory columns: {}",
            required.join(", ")
        )));
    }
    rows.iter()
        .map(|r| {
            mapping
                .iter()
                .map(|(k, s)| {
                    Ok((
                        k.to_string(),
                        r.get(s).cloned().ok_or_else(|| Error {
                            kind: "KeyError",
                            code: None,
                            message: format!("'{s}'"),
                        })?,
                    ))
                })
                .collect()
        })
        .collect()
}

fn get<'a>(r: &'a Row, k: &str) -> &'a Value {
    r.get(k).unwrap_or(&Value::Null)
}

pub fn frame_from_row(row: &Row, c: &NormalizeConfig, canonical: bool) -> Result<FrameInfo> {
    let frame = parse_frame(get(row, "Frame"))?;
    let timestamp = match timestamp_seconds(get(row, "Timestamp"), &c.timestamp_unit)? {
        Some(t) => t,
        None => {
            c.timestamp_zero_s
                + (frame as f64 - c.frame_zero as f64)
                    * c.frame_period_s.ok_or_else(|| {
                        Error::value(
                        "Timestamp missing: supply an explicit frame_period_s metadata assumption",
                    )
                    })?
        }
    };
    let available = flag(get(row, "Input_Available"), "Input_Available", Some(true))?;
    let valid = flag(get(row, "Attitude_Valid"), "Attitude_Valid", Some(false))?;
    let q = if valid {
        let mut values = [0.; 4];
        for (i, k) in ["q1", "q2", "q3", "q4_scalar"].iter().enumerate() {
            values[i] = number(get(row, k), k, true)?.unwrap();
        }
        Some(if canonical {
            values
        } else {
            adapt_quaternion(&values, "xyzw", &c.attitude_direction)?
        })
    } else {
        None
    };
    FrameInfo::from_json(&json!({
        "frame": frame,
        "timestamp_s": timestamp,
        "input_available": available,
        "attitude_valid": valid,
        "quaternion_xyzw": q
    }))
}

pub fn detection_from_row(
    row: &Row,
    c: &Camera,
    bounds: &str,
    status: Option<&str>,
) -> Result<Detection> {
    detection_from_row_with_reason(row, c, bounds, status, false)
}

fn detection_from_row_with_reason(
    row: &Row,
    c: &Camera,
    bounds: &str,
    status: Option<&str>,
    allow_unspecified_reason: bool,
) -> Result<Detection> {
    let frame = parse_frame(get(row, "Frame"))?;
    let identifier = py_text(get(row, "Detection_ID")).trim().to_owned();
    if get(row, "Detection_ID").is_null() || identifier.is_empty() {
        return Err(Error::value("Detection_ID missing"));
    }
    let x = number(get(row, "x"), "x", true)?.unwrap();
    let y = number(get(row, "y"), "y", true)?.unwrap();
    let sx = number(get(row, "Sigma_X"), "Sigma_X", false)?;
    let sy = number(get(row, "Sigma_Y"), "Sigma_Y", false)?;
    let cross = number(get(row, "Cov_XY"), "Cov_XY", false)?;
    if sx.is_none() != sy.is_none() || (sx.is_none() && cross.is_some()) {
        return Err(Error::value("Sigma_X/Y must both be provided with Cov_XY"));
    }
    let (covariance, status) = if let Some(sx) = sx {
        let sy = sy.unwrap();
        if sx <= 0. || sy <= 0. {
            return Err(Error::value("Sigma_X/Y must be positive"));
        }
        let cross = cross.unwrap_or(0.);
        let c = checked_covariance([[sx * sx, cross], [cross, sy * sy]])?;
        (
            Some(c),
            status
                .filter(|s| !s.is_empty())
                .unwrap_or("reported_pending_calibration"),
        )
    } else {
        if status.is_some_and(|s| s != "missing_pending_calibration") {
            return Err(Error::value(
                "declared covariance status disagrees with missing Sigma",
            ));
        }
        (None, "missing_pending_calibration")
    };
    let mut features = Vec::new();
    for name in FEATURES {
        if let Some(value) = number(get(row, name), name, false)? {
            features.push((name, value));
        }
    }
    let valid = c.contains([x, y]);
    if !valid && bounds == "reject" {
        return Err(Error::value(format!(
            "out-of-bounds detection {identifier} in Frame {frame}"
        )));
    }
    let declared = flag(get(row, "Valid"), "Valid", Some(true))?;
    let reason_value = get(row, "Invalid_Reason");
    let mut reason = if !valid {
        "out_of_bounds".into()
    } else if reason_value.is_null()
        || reason_value == &json!(false)
        || reason_value.as_f64() == Some(0.)
    {
        String::new()
    } else {
        py_text(reason_value)
    };
    if !declared && allow_unspecified_reason && reason.trim().is_empty() {
        reason = "unspecified".into();
    }
    if !declared && reason.is_empty() {
        return Err(Error::value(
            "declared invalid detection requires Invalid_Reason",
        ));
    }
    Detection::from_json(&json!({
        "frame": frame,
        "detection_id": identifier,
        "x_px": x,
        "y_px": y,
        "covariance_px2": covariance,
        "covariance_status": status,
        "features": features,
        "valid": valid && declared,
        "invalid_reason": reason
    }))
}

pub fn normalize_table(
    path: &Path,
    c: &NormalizeConfig,
    frames_path: Option<&Path>,
) -> Result<(ObservationPacket, Value)> {
    let mut c = c.clone();
    let mut original_rows = 0usize;
    crate::table_stream::visit_batches(path, c.max_rows, |header, batch| {
        let rows = mapped_rows(header, batch, &c, true)?;
        original_rows += rows.len();
        if c.timestamp_unit != "iso"
            && rows.iter().any(|row| {
                let value = get(row, "Timestamp");
                !empty(value)
                    && !value.is_boolean()
                    && !value.is_number()
                    && parse_float(&py_text(value)).is_none()
            })
        {
            c.timestamp_unit = "iso".into();
        }
        Ok(())
    })?;
    let mut frames = BTreeMap::new();
    if let Some(path) = frames_path {
        crate::table_stream::visit_batches(path, c.max_rows, |header, batch| {
            for row in mapped_rows(header, batch, &c, false)? {
                let info = frame_from_row(&row, &c, false)?;
                if frames.insert(info.frame, info).is_some() {
                    return Err(Error::value("duplicate Frame in frames sidecar"));
                }
            }
            Ok(())
        })?;
    }
    let mut points = Vec::new();
    crate::table_stream::visit_batches(path, c.max_rows, |header, batch| {
        let rows = mapped_rows(header, batch, &c, true)?;
        for row in &rows {
            if frames_path.is_none() {
                let info = frame_from_row(row, &c, false)?;
                if frames
                    .get(&info.frame)
                    .is_some_and(|old: &FrameInfo| old != &info)
                {
                    return Err(Error::value(format!(
                        "inconsistent timestamp/attitude within Frame {}",
                        info.frame
                    )));
                }
                frames.insert(info.frame, info);
            }
            points.push(detection_from_row_with_reason(
                row,
                &c.camera,
                &c.bounds_action,
                None,
                true,
            )?);
        }
        if frames_path.is_some() {
            for row in &rows {
                let frame = parse_frame(get(row, "Frame"))?;
                let info = frames
                    .get(&frame)
                    .ok_or_else(|| Error::value("detection Frame absent from frames sidecar"))?;
                if timestamp_seconds(get(row, "Timestamp"), &c.timestamp_unit)?.is_some() {
                    let mut r = row.clone();
                    r.insert("Attitude_Valid".into(), json!(0));
                    if frame_from_row(&r, &c, false)?.timestamp_s != info.timestamp_s {
                        return Err(Error::value("timestamp conflicts with frames sidecar"));
                    }
                }
                let supplied = |name| !get(row, name).is_null() && get(row, name) != &json!("");
                if supplied("Input_Available")
                    && flag(get(row, "Input_Available"), "Input_Available", None)?
                        != info.input_available
                {
                    return Err(Error::value(
                        "input availability conflicts with frames sidecar",
                    ));
                }
                if supplied("Attitude_Valid") {
                    if flag(get(row, "Attitude_Valid"), "Attitude_Valid", None)?
                        != info.attitude_valid
                    {
                        return Err(Error::value(
                            "attitude validity conflicts with frames sidecar",
                        ));
                    }
                    if info.attitude_valid {
                        let mut q = [0.; 4];
                        for (i, k) in ["q1", "q2", "q3", "q4_scalar"].iter().enumerate() {
                            q[i] = number(get(row, k), k, true)?.unwrap();
                        }
                        if Some(adapt_quaternion(&q, "xyzw", &c.attitude_direction)?)
                            != info.quaternion_xyzw
                        {
                            return Err(Error::value("quaternion conflicts with frames sidecar"));
                        }
                    }
                }
            }
        }
        Ok(())
    })?;
    let diagnostics = json!({
           "original_rows": original_rows,
           "invalid_bounds": points
               .iter()
               .filter(|d| !c.camera.contains([d.x_px,
    d.y_px]))
               .count(),
           "invalid_detections": points.iter().filter(|d|!d.valid).count(),
           "covariance_missing": points.iter().filter(|d|d.covariance_px2.is_none()).count(),
           "frames_without_detections_unknown": frames_path.is_none(),
           "frame_period_s_assumption": c.frame_period_s
       });
    let packet = ObservationPacket {
        camera: c.camera,
        time_scale: c.time_scale,
        frames: frames.into_values().collect(),
        detections: points,
        catalog: Vec::new(),
        origin: "real_detection_table".into(),
    }
    .validated()?;
    Ok((packet, diagnostics))
}

pub fn write_packet_csvs(folder: &Path, p: &ObservationPacket) -> Result<()> {
    std::fs::create_dir_all(folder).map_err(io_error)?;
    let mut frames = Vec::new();
    for f in &p.frames {
        let mut r = json!({
            "Frame": f.frame,
            "Timestamp_s": f.timestamp_s,
            "Input_Available": i32::from(f.input_available),
            "Attitude_Valid": i32::from(f.attitude_valid)
        })
        .as_object()
        .unwrap()
        .clone();
        for (i, k) in ["q1", "q2", "q3", "q4_scalar"].iter().enumerate() {
            r.insert(
                k.to_string(),
                f.quaternion_xyzw.map_or(json!(""), |q| json!(q[i])),
            );
        }
        frames.push(r);
    }
    let mut points = Vec::new();
    for d in &p.detections {
        let mut r = json!({
            "Frame": d.frame,
            "Detection_ID": d.detection_id,
            "x": d.x_px,
            "y": d.y_px,
            "Covariance_Status": d.covariance_status,
            "Valid": i32::from(d.valid),
            "Invalid_Reason": d.invalid_reason
        })
        .as_object()
        .unwrap()
        .clone();
        for (k, x) in [
            ("Sigma_X", d.covariance_px2.map(|c| c[0][0].sqrt())),
            ("Sigma_Y", d.covariance_px2.map(|c| c[1][1].sqrt())),
            ("Cov_XY", d.covariance_px2.map(|c| c[0][1])),
        ] {
            r.insert(k.into(), x.map_or(json!(""), |x| json!(x)));
        }
        for (k, v) in &d.features {
            r.insert(k.clone(), json!(v));
        }
        points.push(r);
    }
    let catalog = p
        .catalog
        .iter()
        .map(|s| {
            json!({
                "KnownStar_ID": s.known_star_id,
                "LOS_J2000_X": s.direction_j2000[0],
                "LOS_J2000_Y": s.direction_j2000[1],
                "LOS_J2000_Z": s.direction_j2000[2]
            })
            .as_object()
            .unwrap()
            .clone()
        })
        .collect::<Vec<_>>();
    crate::table_io::write_csv(&folder.join("frames.csv"), &FRAME_FIELDS, &frames, false)?;
    crate::table_io::write_csv(
        &folder.join("detections.csv"),
        &DETECTION_FIELDS,
        &points,
        false,
    )?;
    crate::table_io::write_csv(
        &folder.join("catalog.csv"),
        &CATALOG_FIELDS,
        &catalog,
        false,
    )?;
    Ok(())
}

pub fn load_packet(folder: &Path, max_rows: usize) -> Result<ObservationPacket> {
    let meta = read_json(&folder.join("input_metadata.json"))?;
    if key(&meta, "mode")? != "ANGLE_ONLY"
        || key(&meta, "quaternion")? != "xyzw_camera_to_j2000"
        || key(&meta, "pixel_axes")? != "camera_-X_-Y"
        || key(&meta, "timestamp_unit")? != "s"
    {
        return Err(Error::value("unsupported coordinate/time convention"));
    }
    if key(&meta, "model_version")? != PUBLIC_INPUT_SCHEMA_VERSION {
        return Err(Error::value("unsupported public input schema version"));
    }
    let camera = Camera::from_json(key(&meta, "camera")?)?;
    let (_, rows) = crate::table_io::read_table(&folder.join("frames.csv"), max_rows)?;
    let c = NormalizeConfig::from_json(&json!({
        "camera": camera,
        "time_scale": key(&meta,
    "time_scale")?,
        "timestamp_unit": "s",
        "frame_period_s": null,
        "frame_zero": 0,
        "timestamp_zero_s": 0.,
        "bounds_action": "mark_invalid",
        "column_mapping": {},
        "max_rows": max_rows,
        "attitude_direction": "camera_to_j2000"
    }))?;
    let mut frames = Vec::new();
    for mut r in rows {
        r.insert("Timestamp".into(), get(&r, "Timestamp_s").clone());
        frames.push(frame_from_row(&r, &c, true)?);
    }
    let (header, rows) = crate::table_io::read_table(&folder.join("detections.csv"), max_rows)?;
    mapped_rows(&header, &[], &c, true)?;
    let mut points = Vec::new();
    for r in rows {
        let status = r.get("Covariance_Status").ok_or_else(|| Error {
            kind: "KeyError",
            code: None,
            message: "'Covariance_Status'".into(),
        })?;
        points.push(detection_from_row(
            &r,
            &camera,
            "mark_invalid",
            status.as_str(),
        )?);
    }
    let (_, rows) = crate::table_io::read_table(&folder.join("catalog.csv"), max_rows)?;
    let mut stars = Vec::new();
    for r in rows {
        let id = r.get("KnownStar_ID").ok_or_else(|| Error {
            kind: "KeyError",
            code: None,
            message: "'KnownStar_ID'".into(),
        })?;
        let mut u = [0.; 3];
        for (i, k) in CATALOG_FIELDS[1..].iter().enumerate() {
            u[i] = number(get(&r, k), k, true)?.unwrap();
        }
        stars.push(KnownStar::from_json(&json!({
            "known_star_id": id,
            "direction_j2000": u
        }))?);
    }
    ObservationPacket::from_json(&json!({
        "camera": camera,
        "time_scale": key(&meta,
    "time_scale")?,
        "frames": frames,
        "detections": points,
        "catalog": stars,
        "origin": key(&meta,
    "origin")?
    }))
}
