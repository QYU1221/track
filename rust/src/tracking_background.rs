use crate::{
    contracts as c,
    error::{Error, Result},
    exact_factors::{identifier, probability},
    existence::checked_log_existence,
    tracking_model::Background,
};
use serde::Serialize;
use serde_json::Value;
use std::borrow::Cow;
#[derive(Debug, Clone, Serialize)]
pub struct BackgroundConfig {
    pub(crate) persistence_probability: f64,
    pub(crate) p_detection: f64,
    pub(crate) measurement_covariance_px2: [[f64; 2]; 2],
    pub(crate) parameter_provenance: String,
    pub(crate) enable_view_dormancy: bool,
    pub(crate) dormancy_margin_deg: f64,
    pub(crate) dormancy_sigma: f64,
    pub(crate) enable_retirement: bool,
    pub(crate) retirement_existence_probability: f64,
    pub(crate) retirement_visible_misses: u64,
    pub(crate) retirement_min_pd: f64,
    pub(crate) retirement_mature_mass_fraction: f64,
}

impl BackgroundConfig {
    pub fn from_json(v: &Value) -> Result<Self> {
        c::fields(
            v,
            "BackgroundConfig",
            &[
                "persistence_probability",
                "p_detection",
                "measurement_covariance_px2",
                "parameter_provenance",
                "enable_view_dormancy",
                "dormancy_margin_deg",
                "dormancy_sigma",
                "enable_retirement",
                "retirement_existence_probability",
                "retirement_visible_misses",
                "retirement_min_pd",
                "retirement_mature_mass_fraction",
            ],
            &[],
        )?;
        let prob = |key: &str, name: &str| probability(c::scalar(&v[key], name)?, name);
        let persistence_probability = prob("persistence_probability", "background persistence")?;
        let p_detection = prob("p_detection", "unknown-star detection")?;
        let parameter_provenance = identifier(
            &v["parameter_provenance"],
            "background parameter provenance",
        )?;
        let switches = "background lifecycle switches must be boolean";
        let enable_view_dormancy = v["enable_view_dormancy"]
            .as_bool()
            .ok_or_else(|| Error::value(switches))?;
        let enable_retirement = v["enable_retirement"]
            .as_bool()
            .ok_or_else(|| Error::value(switches))?;
        let dormancy_margin_deg =
            c::scalar(&v["dormancy_margin_deg"], "background dormancy policy")?;
        let dormancy_sigma = c::scalar(&v["dormancy_sigma"], "background dormancy policy")?;
        if !(0. ..90.).contains(&dormancy_margin_deg) || dormancy_sigma < 3. {
            return Err(Error::value(
                "background dormancy requires margin in [0,90) and sigma >= 3",
            ));
        }
        let retirement_existence_probability = prob(
            "retirement_existence_probability",
            "background retirement probability",
        )?;
        let retirement_min_pd = prob(
            "retirement_min_pd",
            "background retirement minimum detection probability",
        )?;
        let retirement_mature_mass_fraction = prob(
            "retirement_mature_mass_fraction",
            "background retirement mature mass fraction",
        )?;
        if retirement_mature_mass_fraction <= 0. {
            return Err(Error::value(
                "background retirement mature mass fraction must be positive",
            ));
        }
        if retirement_min_pd <= 0. {
            return Err(Error::value(
                "background retirement minimum detection probability must be positive",
            ));
        }
        let retirement_visible_misses = v["retirement_visible_misses"]
            .as_u64()
            .filter(|x| *x >= 1)
            .ok_or_else(|| {
                Error::value("background retirement requires positive integer visible misses")
            })?;
        let measurement_covariance_px2 = c::covariance_json(&v["measurement_covariance_px2"])?;
        Ok(Self {
            persistence_probability,
            p_detection,
            measurement_covariance_px2,
            parameter_provenance,
            enable_view_dormancy,
            dormancy_margin_deg,
            dormancy_sigma,
            enable_retirement,
            retirement_existence_probability,
            retirement_visible_misses,
            retirement_min_pd,
            retirement_mature_mass_fraction,
        })
    }
}

pub fn predict_background<'a>(
    background: &'a Background,
    dt: f64,
    config: &BackgroundConfig,
) -> Result<Cow<'a, Background>> {
    c::finite(&[dt], "background time interval")?;
    if dt < 0. {
        return Err(Error::value("negative background dt"));
    }
    let log_r = background.log_existence
        + if dt > 0. {
            checked_log_existence(config.persistence_probability, None)?
        } else {
            0.
        };
    if log_r == background.log_existence {
        return Ok(Cow::Borrowed(background));
    }
    let mut out = background.clone();
    out.log_existence = log_r;
    out.existence = libm::exp(log_r);
    if log_r == f64::NEG_INFINITY {
        out.state = None;
    }
    Ok(Cow::Owned(out))
}
