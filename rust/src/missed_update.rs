use crate::{
    error::{Error, Result},
    exact_numerics::{normal_interval, GaussianState, ObservationWindow},
    linalg::{array, mat},
    numerical_policy::NORMAL_INTEGRATION_CUTS,
    quadrature,
    tracking_model::GaussianDirection,
};
use nalgebra::{SMatrix, SVector};
use std::borrow::Cow;
fn failed() -> Error {
    Error {
        kind: "InferenceError",
        code: Some("MISS_MOMENT_INTEGRATION_FAILED"),
        message:
            "MISS_MOMENT_INTEGRATION_FAILED: finite-window moments inconsistent with detection mass"
                .into(),
    }
}

fn update<const N: usize>(
    mean: [f64; N],
    p: [[f64; N]; N],
    h: [[f64; N]; 2],
    r: [[f64; 2]; 2],
    mu: [f64; 2],
    s: [[f64; 2]; 2],
    window: &ObservationWindow,
    pd: f64,
    mass: f64,
) -> Result<([f64; N], [[f64; N]; N])> {
    let sx = s[0][0].sqrt();
    let sy = s[1][1].sqrt();
    let rho = s[0][1] / (sx * sy);
    let s = mat(s);
    let chol = s.cholesky().ok_or_else(|| Error {
        kind: "LinAlgError",
        code: None,
        message: "Matrix is not positive definite".into(),
    })?;
    let sigma = chol.l()[(1, 1)] / sy;
    let [x0, x1, y0, y1] = window.bounds();
    let lo = (x0 - mu[0]) / sx;
    let hi = (x1 - mu[0]) / sx;
    let ya = (y0 - mu[1]) / sy;
    let yb = (y1 - mu[1]) / sy;
    let phi = |x: f64| libm::exp(-x * x / 2.) / (2. * std::f64::consts::PI).sqrt();
    let integrand = |t: f64| {
        let center = rho * t;
        let a = (ya - center) / sigma;
        let b = (yb - center) / sigma;
        let pa = phi(a);
        let pb = phi(b);
        let m = normal_interval(a, b);
        let first = center * m + sigma * (pa - pb);
        let second = (center * center + sigma * sigma) * m
            + 2. * center * sigma * (pa - pb)
            + sigma * sigma * (a * pa - b * pb);
        [m, t * m, first, t * t * m, t * first, second].map(|x| phi(t) * x)
    };
    let mut cuts = vec![lo, hi];
    cuts.extend(
        NORMAL_INTEGRATION_CUTS
            .into_iter()
            .filter(|x| lo < *x && *x < hi),
    );
    for bound in [ya, yb] {
        for offset in [-8., 0., 8.] {
            let t = (bound + offset * sigma) / rho;
            if t.is_finite() && lo < t && t < hi {
                cuts.push(t);
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    let mut moments = [0.; 6];
    let mut error = 0.;
    for pair in cuts.windows(2) {
        let (v, e) =
            quadrature::integrate_vector(integrand, pair[0], pair[1]).ok_or_else(failed)?;
        for i in 0..6 {
            moments[i] += v[i];
        }
        error += e;
    }
    if moments.iter().any(|x| !x.is_finite()) || error > 1e-8 || (moments[0] - mass).abs() > 1e-9 {
        return Err(failed());
    }
    let first = SVector::<f64, 2>::new(sx * moments[1], sy * moments[2]);
    let second = mat([
        [sx * sx * moments[3], sx * sy * moments[4]],
        [sx * sy * moments[4], sy * sy * moments[5]],
    ]);
    let p = mat(p);
    let h = SMatrix::<f64, 2, N>::from_fn(|i, j| h[i][j]);
    let k = s
        .lu()
        .solve(&(h * p))
        .ok_or_else(|| Error {
            kind: "LinAlgError",
            code: None,
            message: "Singular matrix".into(),
        })?
        .transpose();
    let a = SMatrix::<f64, N, N>::identity() - k * h;
    let conditional = a * p * a.transpose() + k * mat(r) * k.transpose();
    let denominator = 1. - pd * mass;
    let shift = (k * first) * (-pd) / denominator;
    let covariance = (p - (conditional * mass + k * second * k.transpose()) * pd) / denominator
        - shift * shift.transpose();
    Ok((
        (SVector::<f64, N>::from(mean) + shift).into(),
        array((covariance + covariance.transpose()) / 2.),
    ))
}

pub fn miss_conditioned_gaussian<'a>(
    state: &'a GaussianState,
    h: [[f64; 4]; 2],
    r: [[f64; 2]; 2],
    mu: [f64; 2],
    s: [[f64; 2]; 2],
    window: &ObservationWindow,
    pd: f64,
    mass: f64,
) -> Result<Cow<'a, GaussianState>> {
    if pd == 0. || mass == 0. || mass == 1. {
        return Ok(Cow::Borrowed(state));
    }
    let (m, p) = update(
        state.mean(),
        state.covariance(),
        h,
        r,
        mu,
        s,
        window,
        pd,
        mass,
    )?;
    Ok(Cow::Owned(GaussianState::new(m, p)?))
}

pub fn miss_conditioned_direction<'a>(
    state: &'a GaussianDirection,
    h: [[f64; 2]; 2],
    r: [[f64; 2]; 2],
    mu: [f64; 2],
    s: [[f64; 2]; 2],
    window: &ObservationWindow,
    pd: f64,
    mass: f64,
) -> Result<Cow<'a, GaussianDirection>> {
    if pd == 0. || mass == 0. || mass == 1. {
        return Ok(Cow::Borrowed(state));
    }
    let (m, p) = update(state.mean, state.covariance, h, r, mu, s, window, pd, mass)?;
    Ok(Cow::Owned(GaussianDirection::new(m, p)?))
}
