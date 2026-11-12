use crate::{
    contracts as c,
    error::{Error, Result},
    exact_factors::{identifier, inference, number, probability, wire, DetectionKey},
    exact_numerics::GaussianState,
    existence::checked_log_existence,
    gnomonic::GnomonicPlane,
    linalg::{array, mat},
};
use nalgebra::{SMatrix, SVector};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashSet;
pub fn cv_matrices(dt: f64, q_a: f64) -> Result<([[f64; 4]; 4], [[f64; 4]; 4])> {
    c::finite(&[dt, q_a], "seconds and acceleration spectral density")?;
    if dt < 0. || q_a < 0. {
        return Err(Error::value("dt and q_a must be nonnegative"));
    }
    let d3 = libm::pow(dt, 3.);
    let d2 = libm::pow(dt, 2.);
    if !d3.is_finite() || !d2.is_finite() {
        return Err(Error {
            kind: "OverflowError",
            code: None,
            message: "(34, 'Result too large')".into(),
        });
    }
    let mut f = SMatrix::<f64, 4, 4>::identity();
    f[(0, 2)] = dt;
    f[(1, 3)] = dt;
    let block = [[d3 / 3., d2 / 2.], [d2 / 2., dt]];
    let q = std::array::from_fn(|i| {
        std::array::from_fn(|j| q_a * block[i / 2][j / 2] * if i % 2 == j % 2 { 1. } else { 0. })
    });
    if q.iter().flatten().any(|x| !x.is_finite()) {
        return Err(Error {
            kind: "FloatingPointError",
            code: None,
            message: "overflow encountered in multiply".into(),
        });
    }
    Ok((array(f), q))
}

pub fn predict_gaussian(state: &GaussianState, dt: f64, q_a: f64) -> Result<GaussianState> {
    let (f, q) = cv_matrices(dt, q_a)?;
    let f = mat(f);
    let p = f * mat(state.covariance()) * f.transpose() + mat(q);
    GaussianState::new(
        (f * SVector::<f64, 4>::from(state.mean())).into(),
        array((p + p.transpose()) / 2.),
    )
}

fn moments<const N: usize>(
    weights: &[f64],
    means: &[[f64; N]],
    covariances: &[[[f64; N]; N]],
) -> Result<Option<([f64; N], [[f64; N]; N])>> {
    c::finite(weights, "mixture weights")?;
    if weights.len() != means.len() || weights.iter().any(|w| *w < 0.) {
        return Err(Error::value("invalid mixture"));
    }
    let total = weights.iter().fold(0., |a, b| a + b);
    if weights.is_empty() || total == 0. {
        return Ok(None);
    }
    let w: Vec<_> = weights.iter().map(|w| w / total).collect();
    let mut mean = [0.; N];
    for (a, m) in w.iter().zip(means) {
        for i in 0..N {
            mean[i] += a * m[i];
        }
    }
    let mut p = [[0.; N]; N];
    for ((a, m), cov) in w.iter().zip(means).zip(covariances) {
        for i in 0..N {
            for j in 0..N {
                p[i][j] += a * (cov[i][j] + (m[i] - mean[i]) * (m[j] - mean[j]));
            }
        }
    }
    let symmetric = std::array::from_fn(|i| std::array::from_fn(|j| (p[i][j] + p[j][i]) / 2.));
    Ok(Some((mean, symmetric)))
}

pub fn mixture_moments(weights: &[f64], states: &[GaussianState]) -> Result<Option<GaussianState>> {
    moments(
        weights,
        &states.iter().map(GaussianState::mean).collect::<Vec<_>>(),
        &states
            .iter()
            .map(GaussianState::covariance)
            .collect::<Vec<_>>(),
    )?
    .map(|(m, p)| GaussianState::new(m, p))
    .transpose()
}

#[derive(Debug, Clone, Serialize)]
pub struct GaussianDirection {
    pub(crate) mean: [f64; 2],
    pub(crate) covariance: [[f64; 2]; 2],
}

impl GaussianDirection {
    pub fn new(mean: [f64; 2], covariance: [[f64; 2]; 2]) -> Result<Self> {
        c::finite(&mean, "2D gnomonic direction")?;
        Ok(Self {
            mean,
            covariance: c::checked_covariance(covariance)?,
        })
    }
    pub fn from_json(v: &Value) -> Result<Self> {
        Self::new(
            c::vector(
                &v["mean"],
                "2D gnomonic direction",
                "unknown star requires 2D direction, not CV",
            )?,
            c::matrix(
                &v["covariance"],
                "covariance",
                "covariance has incorrect dimensions",
            )?,
        )
    }
}

pub fn mixture_directions(
    weights: &[f64],
    states: &[GaussianDirection],
) -> Result<Option<GaussianDirection>> {
    moments(
        weights,
        &states.iter().map(|s| s.mean).collect::<Vec<_>>(),
        &states.iter().map(|s| s.covariance).collect::<Vec<_>>(),
    )?
    .map(|(m, p)| GaussianDirection::new(m, p))
    .transpose()
}

fn keys(v: Option<&Value>) -> Result<Vec<DetectionKey>> {
    v.map(|v| serde_json::from_value(v.clone()).map_err(|_| Error::value("invalid support keys")))
        .transpose()
        .map(|v| v.unwrap_or_default())
}

fn optional_log(v: &Value) -> Result<Option<f64>> {
    v.get("log_existence")
        .filter(|v| !v.is_null())
        .map(number)
        .transpose()
}

#[derive(Debug, Clone, Serialize)]
pub struct Track {
    pub(crate) label: String,
    pub(crate) existence: f64,
    pub(crate) state: Option<GaussianState>,
    pub(crate) plane: GnomonicPlane,
    pub(crate) support_keys: Vec<DetectionKey>,
    #[serde(serialize_with = "wire::float")]
    pub(crate) log_existence: f64,
}

impl Track {
    pub fn from_json(v: &Value) -> Result<Self> {
        c::fields(
            v,
            "Track",
            &["label", "existence", "state", "plane"],
            &["support_keys", "log_existence"],
        )?;
        let state = if v["state"].is_null() {
            None
        } else {
            Some(GaussianState::from_json(&v["state"])?)
        };
        let plane = GnomonicPlane::from_json(&v["plane"])?;
        Self::new(
            identifier(&v["label"], "track label")?,
            c::scalar(&v["existence"], "existence")?,
            state,
            plane,
            keys(v.get("support_keys"))?,
            optional_log(v)?,
        )
    }
    pub fn new(
        label: String,
        existence: f64,
        state: Option<GaussianState>,
        plane: GnomonicPlane,
        support_keys: Vec<DetectionKey>,
        log_existence: Option<f64>,
    ) -> Result<Self> {
        if label.trim().is_empty() {
            return Err(Error::value("track label must be nonempty"));
        }
        probability(existence, "existence")?;
        let log_existence = checked_log_existence(existence, log_existence)?;
        if state.is_none() != (log_existence == f64::NEG_INFINITY) {
            return Err(Error::value(
                "state density is defined iff existence is positive",
            ));
        }
        if support_keys.iter().collect::<HashSet<_>>().len() != support_keys.len() {
            return Err(Error::value("duplicate measurement in conditional history"));
        }
        Ok(Self {
            label,
            existence,
            state,
            plane,
            support_keys,
            log_existence,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Background {
    pub(crate) background_id: String,
    pub(crate) existence: f64,
    pub(crate) state: Option<GaussianDirection>,
    pub(crate) plane: GnomonicPlane,
    pub(crate) support_keys: Vec<DetectionKey>,
    pub(crate) visible_miss_count: u64,
    #[serde(serialize_with = "wire::float")]
    pub(crate) log_existence: f64,
}

impl Background {
    pub fn from_json(v: &Value) -> Result<Self> {
        c::fields(
            v,
            "Background",
            &["background_id", "existence", "state", "plane"],
            &["support_keys", "visible_miss_count", "log_existence"],
        )?;
        let state = if v["state"].is_null() {
            None
        } else {
            Some(GaussianDirection::from_json(&v["state"])?)
        };
        let plane = GnomonicPlane::from_json(&v["plane"])?;
        let background_id = identifier(&v["background_id"], "Background_ID")?;
        let existence = probability(
            c::scalar(&v["existence"], "background existence")?,
            "background existence",
        )?;
        let log_existence = checked_log_existence(existence, optional_log(v)?)?;
        if state.is_none() != (log_existence == f64::NEG_INFINITY) {
            return Err(Error::value(
                "direction density defined iff positive existence",
            ));
        }
        let support_keys = keys(v.get("support_keys"))?;
        if support_keys.iter().collect::<HashSet<_>>().len() != support_keys.len() {
            return Err(Error::value("duplicate background measurement"));
        }
        let visible_miss_count = v
            .get("visible_miss_count")
            .map(|x| {
                x.as_u64().ok_or_else(|| {
                    Error::value("background visible miss count must be a nonnegative integer")
                })
            })
            .transpose()?
            .unwrap_or(0);
        Ok(Self {
            background_id,
            existence,
            state,
            plane,
            support_keys,
            visible_miss_count,
            log_existence,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TrackingConfig {
    pub(crate) survival_probability: f64,
    pub(crate) q_a: f64,
    pub(crate) p_detection: f64,
    pub(crate) measurement_covariance_px2: [[f64; 2]; 2],
    pub(crate) parameter_provenance: String,
    pub(crate) max_dt_s: f64,
    pub(crate) survival_reference_dt_s: f64,
    pub(crate) miss_update_mode: String,
}

impl TrackingConfig {
    pub fn from_json(v: &Value) -> Result<Self> {
        c::fields(
            v,
            "TrackingConfig",
            &[
                "survival_probability",
                "q_a",
                "p_detection",
                "measurement_covariance_px2",
                "parameter_provenance",
                "max_dt_s",
                "survival_reference_dt_s",
            ],
            &["miss_update_mode"],
        )?;
        let survival_probability = probability(
            c::scalar(
                &v["survival_probability"],
                "constant per-transition survival",
            )?,
            "constant per-transition survival",
        )?;
        let p_detection = probability(
            c::scalar(&v["p_detection"], "detection probability")?,
            "detection probability",
        )?;
        let q_a = c::scalar(&v["q_a"], "dynamic parameters")?;
        let max_dt_s = c::scalar(&v["max_dt_s"], "dynamic parameters")?;
        if q_a < 0. || max_dt_s <= 0. {
            return Err(Error::value("q_a>=0 and max_dt_s>0 required"));
        }
        let survival_reference_dt_s =
            c::scalar(&v["survival_reference_dt_s"], "survival reference seconds")?;
        if survival_reference_dt_s <= 0. {
            return Err(Error::value("positive survival reference seconds required"));
        }
        let measurement_covariance_px2 = c::covariance_json(&v["measurement_covariance_px2"])?;
        let parameter_provenance = identifier(&v["parameter_provenance"], "parameter provenance")?;
        let miss_update_mode = v
            .get("miss_update_mode")
            .and_then(Value::as_str)
            .unwrap_or(if v.get("miss_update_mode").is_none() {
                "retain"
            } else {
                ""
            })
            .to_owned();
        if !["retain", "moment_matched"].contains(&miss_update_mode.as_str()) {
            return Err(Error::value(
                "miss_update_mode must be retain or moment_matched",
            ));
        }
        Ok(Self {
            survival_probability,
            q_a,
            p_detection,
            measurement_covariance_px2,
            parameter_provenance,
            max_dt_s,
            survival_reference_dt_s,
            miss_update_mode,
        })
    }
}

pub fn predict_track(track: &Track, dt: f64, config: &TrackingConfig) -> Result<Track> {
    let mut log_r = track.log_existence;
    if dt > 0. {
        log_r += (dt / config.survival_reference_dt_s)
            * checked_log_existence(config.survival_probability, None)?;
    }
    let state = if log_r > f64::NEG_INFINITY {
        Some(predict_gaussian(
            track
                .state
                .as_ref()
                .ok_or_else(|| Error::value("missing state"))?,
            dt,
            config.q_a,
        )?)
    } else {
        None
    };
    Track::new(
        track.label.clone(),
        libm::exp(log_r),
        state,
        track.plane.clone(),
        track.support_keys.clone(),
        Some(log_r),
    )
}

type Assignments = Vec<(String, Option<DetectionKey>)>;
#[derive(Debug, Clone, Serialize)]
pub struct JointHistory {
    pub(crate) hypothesis_id: String,
    pub(crate) weight: f64,
    pub(crate) tracks: Vec<Track>,
    pub(crate) target_assignments: Assignments,
    star_assignments: Assignments,
    clutter_keys: Vec<DetectionKey>,
    pub(crate) backgrounds: Vec<Background>,
    pub(crate) background_assignments: Assignments,
}

fn assignments(v: Option<&Value>) -> Result<Assignments> {
    v.map(|v| {
        serde_json::from_value(v.clone()).map_err(|_| Error::value("invalid history assignments"))
    })
    .transpose()
    .map(|v| v.unwrap_or_default())
}

impl JointHistory {
    pub fn from_json(v: &Value) -> Result<Self> {
        c::fields(
            v,
            "JointHistory",
            &["hypothesis_id", "weight", "tracks"],
            &[
                "target_assignments",
                "star_assignments",
                "clutter_keys",
                "backgrounds",
                "background_assignments",
            ],
        )?;
        let tracks = v["tracks"]
            .as_array()
            .ok_or_else(|| Error::value("tracks must be an array"))?
            .iter()
            .map(Track::from_json)
            .collect::<Result<Vec<_>>>()?;
        let backgrounds = v
            .get("backgrounds")
            .map(|v| {
                v.as_array()
                    .ok_or_else(|| Error::value("backgrounds must be an array"))?
                    .iter()
                    .map(Background::from_json)
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        let weight = probability(c::scalar(&v["weight"], "joint weight")?, "joint weight")?;
        if tracks
            .iter()
            .map(|t| &t.label)
            .collect::<HashSet<_>>()
            .len()
            != tracks.len()
        {
            return Err(Error::value("duplicate track labels"));
        }
        if backgrounds
            .iter()
            .map(|t| &t.background_id)
            .collect::<HashSet<_>>()
            .len()
            != backgrounds.len()
        {
            return Err(Error::value("duplicate Background_ID"));
        }
        let support: Vec<_> = tracks
            .iter()
            .flat_map(|t| t.support_keys.iter())
            .chain(backgrounds.iter().flat_map(|b| b.support_keys.iter()))
            .collect();
        if support.iter().collect::<HashSet<_>>().len() != support.len() {
            return Err(inference(
                "DUPLICATE_OBSERVATION_OWNER",
                "target/background histories share a detection",
            ));
        }
        Ok(Self {
            hypothesis_id: v["hypothesis_id"]
                .as_str()
                .ok_or_else(|| Error::value("hypothesis_id must be a string"))?
                .into(),
            weight,
            tracks,
            backgrounds,
            target_assignments: assignments(v.get("target_assignments"))?,
            star_assignments: assignments(v.get("star_assignments"))?,
            background_assignments: assignments(v.get("background_assignments"))?,
            clutter_keys: keys(v.get("clutter_keys"))?,
        })
    }
}
