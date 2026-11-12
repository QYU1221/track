use crate::{
    error::{Error, Result},
    numerical_policy::*,
};
use nalgebra::DMatrix;
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

pub(crate) fn fields(v: &Value, class: &str, required: &[&str], optional: &[&str]) -> Result<()> {
    let object = v
        .as_object()
        .ok_or_else(|| Error::type_error(format!("{class} requires an object")))?;
    for key in object.keys() {
        if !required.contains(&key.as_str()) && !optional.contains(&key.as_str()) {
            return Err(Error::type_error(format!(
                "__init__() got an unexpected keyword argument '{key}'"
            )));
        }
    }
    let missing: Vec<_> = required
        .iter()
        .filter(|k| !object.contains_key(**k))
        .map(|k| format!("'{k}'"))
        .collect();
    if !missing.is_empty() {
        let names = match missing.len() {
            1 => missing[0].clone(),
            2 => missing.join(" and "),
            _ => format!(
                "{}, and {}",
                missing[..missing.len() - 1].join(", "),
                missing.last().unwrap()
            ),
        };
        return Err(Error::type_error(format!(
            "__init__() missing {} required positional argument{}: {names}",
            missing.len(),
            if missing.len() == 1 { "" } else { "s" }
        )));
    }
    Ok(())
}

pub fn finite(values: &[f64], name: &str) -> Result<()> {
    if values.iter().any(|x| !x.is_finite()) {
        Err(Error::value(format!("{name} must be finite")))
    } else {
        Ok(())
    }
}

fn has_bool(v: &Value) -> bool {
    v.is_boolean() || v.as_array().is_some_and(|a| a.iter().any(has_bool))
}

pub fn numbers(v: &Value, name: &str) -> Result<Vec<f64>> {
    if has_bool(v) {
        return Err(Error::value(format!(
            "{name} must be real numeric values, not boolean or complex"
        )));
    }
    fn walk(v: &Value, out: &mut Vec<f64>) -> Result<()> {
        match v {
            Value::Array(a) => {
                for x in a {
                    walk(x, out)?;
                }
            }
            Value::Null => out.push(f64::NAN),
            Value::Number(n) => out.push(
                n.as_f64()
                    .ok_or_else(|| Error::value("numeric value outside float64 range"))?,
            ),
            Value::String(s) => {
                out.push(s.trim().parse::<f64>().map_err(|_| {
                    Error::value(format!("could not convert string to float: '{s}'"))
                })?)
            }
            _ => return Err(Error::value("unsupported numeric value")),
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(v, &mut out)?;
    finite(&out, name)?;
    Ok(out)
}

pub fn scalar(v: &Value, name: &str) -> Result<f64> {
    let values = numbers(v, name)?;
    if v.is_array() || values.len() != 1 {
        return Err(Error::value(format!("{name} must be scalar")));
    }
    Ok(values[0])
}

pub fn vector<const N: usize>(v: &Value, name: &str, shape: &str) -> Result<[f64; N]> {
    let values = numbers(v, name)?;
    if !v
        .as_array()
        .is_some_and(|a| a.len() == N && a.iter().all(|x| !x.is_array()))
    {
        return Err(Error::value(shape));
    }
    Ok(std::array::from_fn(|i| values[i]))
}

pub fn matrix<const R: usize, const C: usize>(
    v: &Value,
    name: &str,
    shape: &str,
) -> Result<[[f64; C]; R]> {
    let values = numbers(v, name)?;
    if !v.as_array().is_some_and(|a| {
        a.len() == R
            && a.iter().all(|r| {
                r.as_array()
                    .is_some_and(|b| b.len() == C && b.iter().all(|x| !x.is_array()))
            })
    }) {
        return Err(Error::value(shape));
    }
    Ok(std::array::from_fn(|i| {
        std::array::from_fn(|j| values[i * C + j])
    }))
}

pub fn checked_covariance<const N: usize>(c: [[f64; N]; N]) -> Result<[[f64; N]; N]> {
    finite(
        &c.iter().flatten().copied().collect::<Vec<_>>(),
        "covariance",
    )?;
    let scale = c.iter().flatten().fold(0.0_f64, |s, x| s.max(x.abs()));
    let tol = SYMMETRY_EPS_MULTIPLIER * f64::EPSILON * scale;
    if (0..N).any(|i| (0..N).any(|j| (c[i][j] - c[j][i]).abs() > tol)) {
        return Err(Error::value("covariance must be symmetric"));
    }
    let m = DMatrix::from_fn(N, N, |i, j| if i >= j { c[i][j] } else { c[j][i] });
    if m.cholesky().is_none() {
        return Err(Error::value("covariance must be positive definite"));
    }
    Ok(c)
}

pub fn covariance_json(v: &Value) -> Result<[[f64; 2]; 2]> {
    checked_covariance(matrix(
        v,
        "covariance",
        "covariance has incorrect dimensions",
    )?)
}

pub(crate) fn text(v: &Value, message: &str) -> Result<String> {
    v.as_str()
        .map(str::to_owned)
        .ok_or_else(|| Error::value(message))
}

fn frame_number(v: &Value, message: &str) -> Result<i64> {
    v.as_i64()
        .filter(|i| (0..=2147483647).contains(i))
        .ok_or_else(|| Error::value(message))
}

pub(crate) fn optional<T>(v: &Value, f: impl FnOnce(&Value) -> Result<T>) -> Result<Option<T>> {
    if v.is_null() {
        Ok(None)
    } else {
        f(v).map(Some)
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Camera {
    pub(crate) width: i64,
    pub(crate) height: i64,
    pub(crate) fx_px: f64,
    pub(crate) fy_px: f64,
    pub(crate) cx_px: f64,
    pub(crate) cy_px: f64,
}

impl Camera {
    pub fn new(width: i64, height: i64, fx: f64, fy: f64, cx: f64, cy: f64) -> Result<Self> {
        if width <= 0 || height <= 0 {
            return Err(Error::value(
                "camera image dimensions must be positive integers",
            ));
        }
        finite(&[fx, fy, cx, cy], "camera")?;
        if fx <= 0. || fy <= 0. {
            return Err(Error::value("camera focal lengths must be positive"));
        }
        if !(0. <= cx && cx < width as f64 && 0. <= cy && cy < height as f64) {
            return Err(Error::value(
                "camera principal point must be inside the image",
            ));
        }
        Ok(Self {
            width,
            height,
            fx_px: fx,
            fy_px: fy,
            cx_px: cx,
            cy_px: cy,
        })
    }
    pub fn from_json(v: &Value) -> Result<Self> {
        fields(
            v,
            "Camera",
            &["width", "height", "fx_px", "fy_px", "cx_px", "cy_px"],
            &[],
        )?;
        let msg = "camera image dimensions must be positive integers";
        let w = v["width"]
            .as_i64()
            .filter(|x| *x > 0)
            .ok_or_else(|| Error::value(msg))?;
        let h = v["height"]
            .as_i64()
            .filter(|x| *x > 0)
            .ok_or_else(|| Error::value(msg))?;
        numbers(
            &serde_json::json!([v["fx_px"], v["fy_px"], v["cx_px"], v["cy_px"]]),
            "camera",
        )?;
        Self::new(
            w,
            h,
            scalar(&v["fx_px"], "camera")?,
            scalar(&v["fy_px"], "camera")?,
            scalar(&v["cx_px"], "camera")?,
            scalar(&v["cy_px"], "camera")?,
        )
    }
    pub fn contains(&self, xy: [f64; 2]) -> bool {
        0. <= xy[0] && xy[0] < self.width as f64 && 0. <= xy[1] && xy[1] < self.height as f64
    }
    pub fn dimensions(&self) -> (i64, i64) {
        (self.width, self.height)
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FrameInfo {
    pub(crate) frame: i64,
    pub(crate) timestamp_s: f64,
    pub(crate) input_available: bool,
    pub(crate) attitude_valid: bool,
    pub(crate) quaternion_xyzw: Option<[f64; 4]>,
}

impl FrameInfo {
    pub fn from_json(v: &Value) -> Result<Self> {
        fields(
            v,
            "FrameInfo",
            &[
                "frame",
                "timestamp_s",
                "input_available",
                "attitude_valid",
                "quaternion_xyzw",
            ],
            &[],
        )?;
        let frame = frame_number(&v["frame"], "Frame must be a non-negative integer")?;
        let timestamp_s = scalar(&v["timestamp_s"], "timestamp_s")?;
        let msg = "frame validity flags must be booleans";
        let input_available = v["input_available"]
            .as_bool()
            .ok_or_else(|| Error::value(msg))?;
        let attitude_valid = v["attitude_valid"]
            .as_bool()
            .ok_or_else(|| Error::value(msg))?;
        let quaternion_xyzw = if attitude_valid {
            let q = vector::<4>(
                &v["quaternion_xyzw"],
                "quaternion",
                "valid attitude requires a normalized xyzw quaternion",
            )?;
            if (q.iter().map(|x| x * x).sum::<f64>() - 1.).abs()
                > UNIT_VECTOR_EPS_MULTIPLIER * f64::EPSILON
            {
                return Err(Error::value(
                    "valid attitude requires a normalized xyzw quaternion",
                ));
            }
            Some(q)
        } else {
            if !v["quaternion_xyzw"].is_null() {
                return Err(Error::value(
                    "invalid attitude must not expose a trusted quaternion",
                ));
            }
            None
        };
        Ok(Self {
            frame,
            timestamp_s,
            input_available,
            attitude_valid,
            quaternion_xyzw,
        })
    }
    pub fn inertial_observation_available(&self) -> bool {
        self.input_available && self.attitude_valid
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Detection {
    pub(crate) frame: i64,
    pub(crate) detection_id: String,
    pub(crate) x_px: f64,
    pub(crate) y_px: f64,
    pub(crate) covariance_px2: Option<[[f64; 2]; 2]>,
    pub(crate) covariance_status: String,
    pub(crate) features: Vec<(String, f64)>,
    pub(crate) valid: bool,
    pub(crate) invalid_reason: String,
}

impl Detection {
    pub fn from_json(v: &Value) -> Result<Self> {
        fields(
            v,
            "Detection",
            &[
                "frame",
                "detection_id",
                "x_px",
                "y_px",
                "covariance_px2",
                "covariance_status",
            ],
            &["features", "valid", "invalid_reason"],
        )?;
        let frame = frame_number(
            &v["frame"],
            "detection Frame must be a non-negative integer",
        )?;
        let detection_id = text(&v["detection_id"], "Detection_ID must be non-empty")?;
        if detection_id.trim().is_empty() {
            return Err(Error::value("Detection_ID must be non-empty"));
        }
        numbers(&serde_json::json!([v["x_px"], v["y_px"]]), "pixel position")?;
        let p = [
            scalar(&v["x_px"], "pixel position")?,
            scalar(&v["y_px"], "pixel position")?,
        ];
        let covariance_px2 = optional(&v["covariance_px2"], covariance_json)?;
        let covariance_status = text(&v["covariance_status"], "unknown covariance status")?;
        if ![
            "simulation_known",
            "reported_pending_calibration",
            "missing_pending_calibration",
        ]
        .contains(&covariance_status.as_str())
        {
            return Err(Error::value("unknown covariance status"));
        }
        if covariance_px2.is_none() != (covariance_status == "missing_pending_calibration") {
            return Err(Error::value("covariance and calibration status disagree"));
        }
        let mut features = Vec::new();
        if let Some(list) = v.get("features") {
            let list = list
                .as_array()
                .ok_or_else(|| Error::value("features must be pairs"))?;
            let mut seen = HashSet::new();
            for pair in list {
                if !pair.as_array().is_some_and(|a| a.len() == 2) {
                    return Err(Error::value("features must be pairs"));
                }
                let name = text(&pair[0], "feature name must be a string")?;
                if !seen.insert(name.clone()) {
                    return Err(Error::value("duplicate feature"));
                }
            }
            for pair in list {
                let name = pair[0].as_str().unwrap();
                if ![
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
                ]
                .contains(&name)
                {
                    return Err(Error::value(format!(
                        "unsupported or hidden feature: {name}"
                    )));
                }
                let value = scalar(&pair[1], name)?;
                if name == "Confidence" && !(0. ..=1.).contains(&value) {
                    return Err(Error::value(
                        "Confidence is a front-end score in [0,1], not a posterior",
                    ));
                }
                if name != "Background_Mean" && value < 0. {
                    return Err(Error::value(format!("{name} must be non-negative")));
                }
                features.push((name.to_owned(), value));
            }
        }
        let valid = v.get("valid").map_or(Ok(true), |x| {
            x.as_bool()
                .ok_or_else(|| Error::value("valid must be boolean"))
        })?;
        let invalid_reason = v.get("invalid_reason").map_or(Ok(String::new()), |x| {
            text(x, "invalid reason must be a string")
        })?;
        if !valid && invalid_reason.is_empty() {
            return Err(Error::value("invalid detections require a reason"));
        }
        Ok(Self {
            frame,
            detection_id,
            x_px: p[0],
            y_px: p[1],
            covariance_px2,
            covariance_status,
            features,
            valid,
            invalid_reason,
        })
    }
    pub fn key(&self) -> (i64, &str) {
        (self.frame, &self.detection_id)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct KnownStar {
    pub(crate) known_star_id: String,
    pub(crate) direction_j2000: [f64; 3],
}

impl KnownStar {
    pub fn from_json(v: &Value) -> Result<Self> {
        fields(v, "KnownStar", &["known_star_id", "direction_j2000"], &[])?;
        let msg = "known star requires a unique ID and unit direction";
        let direction_j2000 = vector::<3>(&v["direction_j2000"], "known star direction", msg)?;
        let known_star_id = text(&v["known_star_id"], msg)?;
        if known_star_id.is_empty()
            || (direction_j2000.iter().map(|x| x * x).sum::<f64>() - 1.).abs()
                > UNIT_VECTOR_EPS_MULTIPLIER * f64::EPSILON
        {
            return Err(Error::value(msg));
        }
        Ok(Self {
            known_star_id,
            direction_j2000,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ObservationPacket {
    pub(crate) camera: Camera,
    pub(crate) time_scale: String,
    pub(crate) frames: Vec<FrameInfo>,
    pub(crate) detections: Vec<Detection>,
    pub(crate) catalog: Vec<KnownStar>,
    pub(crate) origin: String,
}

impl ObservationPacket {
    pub fn from_json(v: &Value) -> Result<Self> {
        let camera = Camera::from_json(&v["camera"])?;
        let frames = v["frames"]
            .as_array()
            .ok_or_else(|| Error::value("frames must be an array"))?
            .iter()
            .map(FrameInfo::from_json)
            .collect::<Result<Vec<_>>>()?;
        let detections = v["detections"]
            .as_array()
            .ok_or_else(|| Error::value("detections must be an array"))?
            .iter()
            .map(Detection::from_json)
            .collect::<Result<Vec<_>>>()?;
        let catalog = match v.get("catalog") {
            None => vec![],
            Some(a) => a
                .as_array()
                .ok_or_else(|| Error::value("catalog must be an array"))?
                .iter()
                .map(KnownStar::from_json)
                .collect::<Result<Vec<_>>>()?,
        };
        fields(
            v,
            "ObservationPacket",
            &["camera", "time_scale", "frames", "detections"],
            &["catalog", "origin"],
        )?;
        let time_scale = text(&v["time_scale"], "time scale must be explicitly declared")?;
        if time_scale.trim().is_empty() {
            return Err(Error::value("time scale must be explicitly declared"));
        }
        let origin = v
            .get("origin")
            .map_or(Ok("real_detection_table".to_owned()), |x| {
                text(x, "origin must be a string")
            })?;
        Self {
            camera,
            time_scale,
            frames,
            detections,
            catalog,
            origin,
        }
        .validated()
    }
    pub(crate) fn validated(self) -> Result<Self> {
        let Self {
            camera,
            frames,
            detections,
            catalog,
            ..
        } = &self;
        let frame_map: HashMap<_, _> = frames.iter().map(|f| (f.frame, f)).collect();
        if frame_map.len() != frames.len() {
            return Err(Error::value("duplicate Frame"));
        }
        if frames
            .windows(2)
            .any(|w| w[1].frame <= w[0].frame || w[1].timestamp_s <= w[0].timestamp_s)
        {
            return Err(Error::value(
                "frames and timestamps must be strictly increasing",
            ));
        }
        if detections
            .iter()
            .map(Detection::key)
            .collect::<HashSet<_>>()
            .len()
            != detections.len()
        {
            return Err(Error::value("duplicate (Frame, Detection_ID)"));
        }
        for d in detections {
            let f = frame_map
                .get(&d.frame)
                .ok_or_else(|| Error::value("detection refers to an absent frame"))?;
            if !f.input_available {
                return Err(Error::value(
                    "missing input frames cannot contain detections",
                ));
            }
            if d.valid && !camera.contains([d.x_px, d.y_px]) {
                return Err(Error::value("valid detection is outside the camera image"));
            }
        }
        if catalog
            .iter()
            .map(|s| &s.known_star_id)
            .collect::<HashSet<_>>()
            .len()
            != catalog.len()
        {
            return Err(Error::value("duplicate KnownStar_ID"));
        }
        Ok(self)
    }
    pub fn frames(&self) -> &[FrameInfo] {
        &self.frames
    }
    pub fn detections(&self) -> &[Detection] {
        &self.detections
    }
}
