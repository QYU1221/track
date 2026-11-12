use crate::{
    contracts::{checked_covariance, finite, matrix, vector},
    error::{Error, Result},
    linalg::{array, mat},
    numerical_policy::*,
    quadrature,
};
use nalgebra::{SMatrix, SVector};
use serde::Serialize;
use serde_json::Value;
const LOG_2PI: f64 = 1.8378770664093453;
fn inference(code: &'static str, message: &str) -> Error {
    Error {
        kind: "InferenceError",
        code: Some(code),
        message: format!("{code}: {message}"),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GaussianState {
    mean: [f64; 4],
    covariance: [[f64; 4]; 4],
}

impl GaussianState {
    pub fn new(mean: [f64; 4], covariance: [[f64; 4]; 4]) -> Result<Self> {
        finite(&mean, "state mean")?;
        Ok(Self {
            mean,
            covariance: checked_covariance(covariance)?,
        })
    }
    pub fn from_json(v: &Value) -> Result<Self> {
        Self::new(
            vector(
                &v["mean"],
                "state mean",
                "gnomonic CV state must be [xi,eta,v_xi,v_eta]",
            )?,
            matrix(
                &v["covariance"],
                "covariance",
                "covariance has incorrect dimensions",
            )?,
        )
    }
    pub fn mean(&self) -> [f64; 4] {
        self.mean
    }
    pub fn covariance(&self) -> [[f64; 4]; 4] {
        self.covariance
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ObservationWindow {
    bounds_px: [f64; 4],
}

impl ObservationWindow {
    pub fn bounds(&self) -> [f64; 4] {
        self.bounds_px
    }
    pub fn new(b: [f64; 4]) -> Result<Self> {
        finite(&b, "observation window")?;
        if b[1] <= b[0] || b[3] <= b[2] {
            return Err(Error::value("window must have positive width/height"));
        }
        Ok(Self { bounds_px: b })
    }
    pub fn from_json(v: &Value) -> Result<Self> {
        Self::new(vector(
            v,
            "observation window",
            "window must have positive width/height",
        )?)
    }
    pub fn area(&self) -> Result<f64> {
        let b = self.bounds_px;
        let a = (b[1] - b[0]) * (b[3] - b[2]);
        finite(&[a], "window area")?;
        Ok(a)
    }
    pub fn contains(&self, p: [f64; 2]) -> bool {
        let b = self.bounds_px;
        b[0] <= p[0] && p[0] < b[1] && b[2] <= p[1] && p[1] < b[3]
    }
}

pub fn log_probability(p: f64) -> f64 {
    if p == 0. {
        f64::NEG_INFINITY
    } else {
        libm::log(p)
    }
}

pub fn normal_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x / std::f64::consts::SQRT_2)
}

pub fn log_normal_cdf(x: f64) -> f64 {
    if x >= 0. {
        return (-normal_cdf(-x)).ln_1p();
    }
    if x > -20. {
        return normal_cdf(x).ln();
    }
    if x == f64::NEG_INFINITY {
        return x;
    }
    let inverse = 1. / x / x;
    let mut term = 1.;
    let mut sum = 1.;
    for k in 1..1000 {
        let next = term * (-((2 * k - 1) as f64)) * inverse;
        if next.abs() >= term.abs() {
            break;
        }
        sum += next;
        term = next;
        if term.abs() < f64::EPSILON * sum.abs() {
            break;
        }
    }
    (-0.5 * x) * x - (-x).ln() - 0.5 * LOG_2PI + sum.ln()
}

pub fn normal_interval(lower: f64, upper: f64) -> f64 {
    if lower >= upper {
        return 0.;
    }
    if lower > 0. {
        let a = log_normal_cdf(-lower);
        if a == f64::NEG_INFINITY {
            return 0.;
        }
        return a.exp() * (-(log_normal_cdf(-upper) - a).exp_m1());
    }
    normal_cdf(upper) - normal_cdf(lower)
}

fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + (-(a - b).abs()).exp().ln_1p()
}

pub fn gaussian_logpdf(residuals: &[[f64; 2]], covariance: [[f64; 2]; 2]) -> Result<Vec<f64>> {
    let c = checked_covariance(covariance)?;
    let l = mat(c)
        .cholesky()
        .ok_or_else(|| Error::value("covariance must be positive definite"))?
        .l();
    finite(
        &residuals.iter().flatten().copied().collect::<Vec<_>>(),
        "residuals",
    )?;
    let constant = 2. * LOG_2PI + 2. * (l[(0, 0)].ln() + l[(1, 1)].ln());
    Ok(residuals
        .iter()
        .map(|r| {
            let x = r[0] / l[(0, 0)];
            let y = (r[1] - l[(1, 0)] * x) / l[(1, 1)];
            -0.5 * (constant + x * x + y * y)
        })
        .collect())
}

fn validate(mean: [f64; 2], c: [[f64; 2]; 2]) -> Result<()> {
    finite(&mean, "pixel mean")?;
    checked_covariance(c)?;
    Ok(())
}

pub fn gaussian_window_mass(
    mean: [f64; 2],
    c: [[f64; 2]; 2],
    window: &ObservationWindow,
) -> Result<(f64, f64)> {
    validate(mean, c)?;
    let [x0, x1, y0, y1] = window.bounds_px;
    let sx = c[0][0].sqrt();
    let sy = c[1][1].sqrt();
    if [
        (mean[0] - x0) / sx,
        (x1 - mean[0]) / sx,
        (mean[1] - y0) / sy,
        (y1 - mean[1]) / sy,
    ]
    .into_iter()
    .fold(f64::INFINITY, f64::min)
        >= WINDOW_INTERIOR_SIGMA
    {
        return Ok((1., 0.));
    }
    let lower = (x0 - mean[0]) / sx;
    let upper = (x1 - mean[0]) / sx;
    if c[0][1] == 0. {
        return Ok((
            normal_interval(lower, upper)
                * normal_interval((y0 - mean[1]) / sy, (y1 - mean[1]) / sy),
            0.,
        ));
    }
    let rho = c[0][1] / (sx * sy);
    let sigma = mat(c)
        .cholesky()
        .ok_or_else(|| Error::value("covariance must be positive definite"))?
        .l()[(1, 1)];
    if !sigma.is_finite() || sigma <= 0. {
        return Err(inference(
            "DEGENERATE_COVARIANCE",
            "conditional normal variance is not positive",
        ));
    }
    let f = |t: f64| {
        let m = mean[1] + rho * sy * t;
        (-t * t / 2.).exp() / (2. * std::f64::consts::PI).sqrt()
            * normal_interval((y0 - m) / sigma, (y1 - m) / sigma)
    };
    let mut cuts = vec![lower];
    cuts.extend(
        NORMAL_INTEGRATION_CUTS
            .into_iter()
            .filter(|v| lower < *v && *v < upper),
    );
    cuts.push(upper);
    let slope = rho * sy;
    for boundary in [y0, y1] {
        for offset in [-8., 0., 8.] {
            let t = (boundary - mean[1] + offset * sigma) / slope;
            if t.is_finite() && lower < t && t < upper {
                cuts.push(t);
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    let mut value = 0.;
    let mut error = 0.;
    for pair in cuts.windows(2) {
        let (v, e) = quadrature::integrate(f, pair[0], pair[1]).ok_or_else(|| {
            inference(
                "INTEGRATION_FAILED",
                "Gaussian window integral did not converge",
            )
        })?;
        value += v;
        error += e;
    }
    let tol = PROBABILITY_EPS_MULTIPLIER * f64::EPSILON;
    if !value.is_finite() || value < -tol || value > 1. + tol || error > QUADRATURE_ERROR_LIMIT {
        return Err(inference(
            "INTEGRATION_FAILED",
            "invalid window mass or excessive quadrature error",
        ));
    }
    Ok((value.clamp(0., 1.), error))
}

pub fn gaussian_window_log_outside(
    mean: [f64; 2],
    c: [[f64; 2]; 2],
    window: &ObservationWindow,
) -> Result<f64> {
    validate(mean, c)?;
    let [x0, x1, y0, y1] = window.bounds_px;
    let sx = c[0][0].sqrt();
    let sy = c[1][1].sqrt();
    let log_y = log_add(
        log_normal_cdf((y0 - mean[1]) / sy),
        log_normal_cdf((mean[1] - y1) / sy),
    );
    if c[0][1] == 0. {
        let log_x = log_add(
            log_normal_cdf((x0 - mean[0]) / sx),
            log_normal_cdf((mean[0] - x1) / sx),
        );
        let inside = if log_y < 0. {
            log_probability(normal_interval((y0 - mean[1]) / sy, (y1 - mean[1]) / sy))
        } else {
            f64::NEG_INFINITY
        };
        return Ok(log_add(log_y, log_x + inside).min(0.));
    }
    let rho_sy = c[0][1] / sx;
    let sigma = mat(c)
        .cholesky()
        .ok_or_else(|| Error::value("covariance must be positive definite"))?
        .l()[(1, 1)];
    let tail = |threshold: f64, sign: f64| -> Result<f64> {
        let fail = || {
            inference(
                "INTEGRATION_FAILED",
                "Gaussian complement quadrature failed",
            )
        };
        let conditional = |t: f64| {
            let m = mean[1] + sign * rho_sy * t;
            normal_interval((y0 - m) / sigma, (y1 - m) / sigma)
        };
        let points: Vec<f64> = [y0, y1]
            .into_iter()
            .flat_map(|boundary| {
                [-8., 0., 8.]
                    .into_iter()
                    .map(move |offset| (boundary - mean[1] + offset * sigma) / (sign * rho_sy))
            })
            .collect();
        if threshold >= 0. {
            let scale = threshold.max(1.);
            let transformed: Vec<_> = points.iter().map(|t| (t - threshold) * scale).collect();
            let (v, _) = quadrature::positive_infinite_partition(
                |s| {
                    (-threshold * s / scale - 0.5 * (s / scale).powi(2)).exp()
                        * conditional(threshold + s / scale)
                },
                &transformed,
            )
            .ok_or_else(fail)?;
            Ok(if v == 0. {
                f64::NEG_INFINITY
            } else {
                -threshold.powi(2) / 2. - LOG_2PI / 2. - scale.ln() + v.ln()
            })
        } else {
            let f =
                |t: f64| (-t * t / 2.).exp() / (2. * std::f64::consts::PI).sqrt() * conditional(t);
            let (a, _) = quadrature::partition(f, threshold, 0., &points).ok_or_else(fail)?;
            let (b, _) = quadrature::positive_infinite_partition(f, &points).ok_or_else(fail)?;
            Ok(log_probability(a + b))
        }
    };
    let value = log_add(
        log_add(log_y, tail((mean[0] - x0) / sx, -1.)?),
        tail((x1 - mean[0]) / sx, 1.)?,
    );
    if value.is_nan() || value > BASIS_EPS_MULTIPLIER * f64::EPSILON {
        return Err(inference(
            "INTEGRATION_FAILED",
            "invalid Gaussian complement",
        ));
    }
    Ok(value.min(0.))
}

pub fn log_nondetection(
    pd: f64,
    mean: [f64; 2],
    c: [[f64; 2]; 2],
    w: &ObservationWindow,
) -> Result<f64> {
    finite(&[pd], "pre-window PD")?;
    if !(0. ..=1.).contains(&pd) {
        return Err(Error::value("pre-window PD must be in [0,1]"));
    }
    if pd == 0. {
        return Ok(0.);
    }
    Ok(log_add(
        log_probability(1. - pd),
        log_probability(pd) + gaussian_window_log_outside(mean, c, w)?,
    ))
}

pub fn gaussian_hit_updates(
    state: &GaussianState,
    h: [[f64; 4]; 2],
    r: [[f64; 2]; 2],
    residuals: &[[f64; 2]],
) -> Result<Vec<GaussianState>> {
    finite(
        &residuals.iter().flatten().copied().collect::<Vec<_>>(),
        "innovations",
    )?;
    finite(
        &h.iter().flatten().copied().collect::<Vec<_>>(),
        "observation matrix",
    )?;
    let r = mat(checked_covariance(r)?);
    let h = mat(h);
    let p = mat(state.covariance);
    let s = mat(checked_covariance(array(h * p * h.transpose() + r))?);
    let k = s
        .lu()
        .solve(&(h * p))
        .ok_or_else(|| Error::value("Singular matrix"))?
        .transpose();
    let a = SMatrix::<f64, 4, 4>::identity() - k * h;
    let covariance = a * p * a.transpose() + k * r * k.transpose();
    let covariance = array((covariance + covariance.transpose()) / 2.);
    let mean = SVector::<f64, 4>::from(state.mean);
    residuals
        .iter()
        .map(|v| GaussianState::new((mean + k * SVector::<f64, 2>::from(*v)).into(), covariance))
        .collect()
}
