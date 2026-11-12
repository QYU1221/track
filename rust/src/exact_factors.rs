use crate::{
    contracts as c,
    error::{Error, Result},
    exact_numerics as n, existence,
    numerical_policy::*,
};
use serde::{Serialize, Serializer};
use serde_json::Value;
use std::collections::HashSet;
pub(crate) fn inference(code: &'static str, message: &str) -> Error {
    Error {
        kind: "InferenceError",
        code: Some(code),
        message: format!("{code}: {message}"),
    }
}

pub(crate) fn identifier(v: &Value, name: &str) -> Result<String> {
    v.as_str()
        .filter(|s| !s.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| Error::value(format!("{name} must be nonempty")))
}

pub(crate) fn probability(v: f64, name: &str) -> Result<f64> {
    c::finite(&[v], name)?;
    if !(0. ..=1.).contains(&v) {
        return Err(Error::value(format!("{name} must be in [0,1]")));
    }
    Ok(v)
}

pub(crate) fn number(v: &Value) -> Result<f64> {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        .ok_or_else(|| Error::value("invalid numeric value"))
}

pub(crate) fn logs(v: &Value, message: &str) -> Result<Vec<f64>> {
    let a = v.as_array().ok_or_else(|| Error::value(message))?;
    a.iter()
        .map(|x| {
            number(x)
                .ok()
                .filter(|n| !n.is_nan() && *n != f64::INFINITY)
                .ok_or_else(|| Error::value(message))
        })
        .collect()
}

pub mod wire {
    use super::*;
    pub struct Float(pub f64);
    pub(crate) const FLOAT_MARKER: &str = "space_target_detection::wire::Float";
    struct Encoded(f64);
    impl Serialize for Encoded {
        fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
            if self.0.is_finite() {
                s.serialize_f64(self.0)
            } else {
                s.serialize_str(if self.0.is_nan() {
                    "NaN"
                } else if self.0 > 0. {
                    "Infinity"
                } else {
                    "-Infinity"
                })
            }
        }
    }
    impl Serialize for Float {
        fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
            s.serialize_newtype_struct(FLOAT_MARKER, &Encoded(self.0))
        }
    }
    pub fn float<S: Serializer>(x: &f64, s: S) -> std::result::Result<S::Ok, S::Error> {
        Float(*x).serialize(s)
    }
    pub fn floats<S: Serializer>(x: &[f64], s: S) -> std::result::Result<S::Ok, S::Error> {
        x.iter().map(|x| Float(*x)).collect::<Vec<_>>().serialize(s)
    }
    pub fn optional<S: Serializer>(x: &Option<f64>, s: S) -> std::result::Result<S::Ok, S::Error> {
        x.map(Float).serialize(s)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StarSpec {
    pub(crate) known_star_id: String,
    pub(crate) p_detection: f64,
    pub(crate) measurement_covariance_px2: [[f64; 2]; 2],
}

impl StarSpec {
    pub fn from_json(v: &Value) -> Result<Self> {
        c::fields(
            v,
            "StarSpec",
            &["known_star_id", "p_detection", "measurement_covariance_px2"],
            &[],
        )?;
        Ok(Self {
            known_star_id: identifier(&v["known_star_id"], "KnownStar_ID")?,
            p_detection: probability(
                c::scalar(&v["p_detection"], "star p_detection")?,
                "star p_detection",
            )?,
            measurement_covariance_px2: c::covariance_json(&v["measurement_covariance_px2"])?,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ClutterModel {
    expected_count: f64,
    window: n::ObservationWindow,
    parameter_provenance: String,
}

impl ClutterModel {
    pub fn expected_count(&self) -> f64 {
        self.expected_count
    }
    pub fn window(&self) -> &n::ObservationWindow {
        &self.window
    }
}

fn window(v: &Value) -> Result<n::ObservationWindow> {
    n::ObservationWindow::from_json(v.get("bounds_px").unwrap_or(v))
}

impl ClutterModel {
    pub fn from_json(v: &Value) -> Result<Self> {
        c::fields(
            v,
            "ClutterModel",
            &["expected_count", "window", "parameter_provenance"],
            &[],
        )?;
        let expected_count = c::scalar(&v["expected_count"], "Lambda_C")?;
        if expected_count < 0. {
            return Err(Error::value("expected clutter count must be nonnegative"));
        }
        Ok(Self {
            expected_count,
            window: window(&v["window"])?,
            parameter_provenance: identifier(
                &v["parameter_provenance"],
                "clutter parameter provenance",
            )?,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AffinePixelModel {
    pub(crate) mean_px: [f64; 2],
    pub(crate) noise_px2: [[f64; 2]; 2],
    pub(crate) p_detection_pre_window: f64,
    pub(crate) window: n::ObservationWindow,
    #[serde(serialize_with = "wire::float")]
    pub(crate) integration_error: f64,
}

impl AffinePixelModel {
    pub fn from_json(v: &Value) -> Result<Self> {
        c::fields(
            v,
            "AffinePixelModel",
            &["mean_px", "noise_px2", "p_detection_pre_window", "window"],
            &["integration_error"],
        )?;
        Ok(Self {
            mean_px: c::vector(
                &v["mean_px"],
                "predicted pixel mean",
                "pixel model requires a two-dimensional mean",
            )?,
            noise_px2: c::covariance_json(&v["noise_px2"])?,
            p_detection_pre_window: probability(
                c::scalar(&v["p_detection_pre_window"], "pre-window PD")?,
                "pre-window PD",
            )?,
            window: window(&v["window"])?,
            integration_error: v
                .get("integration_error")
                .map(number)
                .transpose()?
                .unwrap_or(0.),
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ProjectedKnownStar {
    pub(crate) source_id: String,
    pub(crate) model: AffinePixelModel,
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceFactors {
    pub(crate) source_id: String,
    pub(crate) kind: String,
    pub(crate) existence_prior: f64,
    pub(crate) pd_bar: f64,
    #[serde(serialize_with = "wire::floats")]
    pub(crate) log_eta: Vec<f64>,
    pub(crate) pixel_model: Option<AffinePixelModel>,
    #[serde(serialize_with = "wire::optional")]
    pub(crate) log_miss_given_exists: Option<f64>,
    #[serde(serialize_with = "wire::float")]
    pub(crate) log_existence_prior: f64,
}

impl SourceFactors {
    pub fn new(
        source_id: String,
        kind: String,
        existence_prior: f64,
        pd_bar: f64,
        log_eta: Vec<f64>,
        pixel_model: Option<AffinePixelModel>,
        log_miss_given_exists: Option<f64>,
        log_existence_prior: Option<f64>,
    ) -> Result<Self> {
        if source_id.trim().is_empty() {
            return Err(Error::value("source ID must be nonempty"));
        }
        if !["T", "S_K", "S_U"].contains(&kind.as_str()) {
            return Err(Error::value("source kind must be T, S_K or S_U"));
        }
        probability(existence_prior, "existence prior")?;
        let log_existence_prior =
            existence::checked_log_existence(existence_prior, log_existence_prior)?;
        probability(pd_bar, "PD_bar")?;
        if kind == "S_K" && existence_prior != 1. {
            return Err(Error::value("known stars have certain existence"));
        }
        if log_eta.iter().any(|x| x.is_nan() || *x == f64::INFINITY) {
            return Err(Error::value(
                "log_eta must be finite or -inf for zero density",
            ));
        }
        if let Some(l) = log_miss_given_exists {
            if l.is_nan() || l > 0. {
                return Err(Error::value(
                    "log miss probability must be <=0, including -inf",
                ));
            }
            let allowance = (PROBABILITY_EPS_MULTIPLIER * f64::EPSILON)
                .max(pixel_model.as_ref().map_or(0., |m| m.integration_error));
            if (l.exp() - (1. - pd_bar)).abs() > allowance {
                return Err(Error::value(
                    "miss log probability and PD_bar disagree beyond quadrature/roundoff",
                ));
            }
        }
        Ok(Self {
            source_id,
            kind,
            existence_prior,
            pd_bar,
            log_eta,
            pixel_model,
            log_miss_given_exists,
            log_existence_prior,
        })
    }
    pub fn from_json(v: &Value) -> Result<Self> {
        c::fields(
            v,
            "SourceFactors",
            &["source_id", "kind", "existence_prior", "pd_bar", "log_eta"],
            &[
                "pixel_model",
                "log_miss_given_exists",
                "log_existence_prior",
            ],
        )?;
        let optional = |key: &str| v.get(key).filter(|x| !x.is_null()).map(number).transpose();
        Self::new(
            identifier(&v["source_id"], "source ID")?,
            v["kind"].as_str().unwrap_or("").into(),
            c::scalar(&v["existence_prior"], "existence prior")?,
            c::scalar(&v["pd_bar"], "PD_bar")?,
            logs(
                &v["log_eta"],
                "log_eta must be finite or -inf for zero density",
            )?,
            v.get("pixel_model")
                .filter(|x| !x.is_null())
                .map(AffinePixelModel::from_json)
                .transpose()?,
            optional("log_miss_given_exists")?,
            optional("log_existence_prior")?,
        )
    }
}

pub type DetectionKey = (i64, String);
#[derive(Debug, Clone, Serialize)]
pub struct FactorTable {
    pub(crate) parent_id: String,
    pub(crate) parent_weight: f64,
    pub(crate) detection_keys: Vec<DetectionKey>,
    pub(crate) sources: Vec<SourceFactors>,
    #[serde(serialize_with = "wire::floats")]
    pub(crate) log_kappa: Vec<f64>,
}

impl FactorTable {
    pub fn from_json(v: &Value) -> Result<Self> {
        c::fields(
            v,
            "FactorTable",
            &[
                "parent_id",
                "parent_weight",
                "detection_keys",
                "sources",
                "log_kappa",
            ],
            &[],
        )?;
        let parent_id = identifier(&v["parent_id"], "parent ID")?;
        let parent_weight = probability(
            c::scalar(&v["parent_weight"], "parent weight")?,
            "parent weight",
        )?;
        let message = "Detection_Key must be (Frame:int,Detection_ID:str)";
        let keys = v["detection_keys"]
            .as_array()
            .ok_or_else(|| Error::value(message))?;
        let mut detection_keys = Vec::new();
        for k in keys {
            if k.as_array().is_none_or(|a| a.len() != 2) {
                return Err(Error::value(message));
            }
            detection_keys.push((
                k[0].as_i64().ok_or_else(|| Error::value(message))?,
                k[1].as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| Error::value(message))?
                    .into(),
            ));
        }
        if detection_keys.iter().collect::<HashSet<_>>().len() != detection_keys.len() {
            return Err(inference("DUPLICATE_DETECTION_ID", "keys must be unique"));
        }
        let sources = v["sources"]
            .as_array()
            .ok_or_else(|| Error::value("sources must be an array"))?
            .iter()
            .map(SourceFactors::from_json)
            .collect::<Result<Vec<_>>>()?;
        let log_kappa = logs(&v["log_kappa"], "log_kappa dimensions/values invalid")?;
        if log_kappa.len() != detection_keys.len() {
            return Err(Error::value("log_kappa dimensions/values invalid"));
        }
        if sources
            .iter()
            .map(|s| &s.source_id)
            .collect::<HashSet<_>>()
            .len()
            != sources.len()
            || sources
                .iter()
                .any(|s| s.log_eta.len() != detection_keys.len())
        {
            return Err(Error::value(
                "duplicate source ID or factor dimensions mismatch",
            ));
        }
        Ok(Self {
            parent_id,
            parent_weight,
            detection_keys,
            sources,
            log_kappa,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Association {
    pub hypothesis_id: String,
    pub assignments: Vec<(String, Option<usize>)>,
    pub clutter_indices: Vec<usize>,
}

pub fn build_known_star_factors(
    sources: &[ProjectedKnownStar],
    detections: &[c::Detection],
) -> Result<Vec<SourceFactors>> {
    let mut detections: Vec<_> = detections.iter().collect();
    detections.sort_by_key(|d| d.key());
    let mut output = Vec::new();
    for source in sources {
        let mut model = source.model.clone();
        let residuals: Vec<_> = detections
            .iter()
            .map(|d| [d.x_px - model.mean_px[0], d.y_px - model.mean_px[1]])
            .collect();
        let densities = n::gaussian_logpdf(&residuals, model.noise_px2)?;
        let (mass, error) = n::gaussian_window_mass(model.mean_px, model.noise_px2, &model.window)?;
        let miss = n::log_nondetection(
            model.p_detection_pre_window,
            model.mean_px,
            model.noise_px2,
            &model.window,
        )?;
        model.integration_error = error;
        let log_pd = n::log_probability(model.p_detection_pre_window);
        let log_eta = detections
            .iter()
            .zip(densities)
            .map(|(d, l)| {
                if model.window.contains([d.x_px, d.y_px]) {
                    log_pd + l
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect();
        output.push(SourceFactors::new(
            source.source_id.clone(),
            "S_K".into(),
            1.,
            model.p_detection_pre_window * mass,
            log_eta,
            Some(model),
            Some(miss),
            None,
        )?);
    }
    Ok(output)
}
