use crate::{
    camera_model::CameraModel,
    contracts::{self as c, Detection, FrameInfo},
    efficient_graph,
    error::{Error, Result},
    exact_factors::{DetectionKey, SourceFactors},
    exact_numerics::{self as n, GaussianState, ObservationWindow},
    exact_sources::project_direction_pixel_with_rotation,
    gnomonic::GnomonicPlane,
    linalg::{array, mat},
    missed_update,
    tracking_model::GaussianDirection,
};
use nalgebra::{SMatrix, SVector};
use serde::Serialize;
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum State {
    Target(GaussianState),
    Direction(GaussianDirection),
}

#[derive(Debug, Clone)]
pub struct Request {
    pub source_id: String,
    pub kind: String,
    pub state: Option<State>,
    pub existence: f64,
    pub plane: GnomonicPlane,
    pub pd: f64,
    pub noise: [[f64; 2]; 2],
}

pub struct Context<'a> {
    pub frame: &'a FrameInfo,
    pub camera: &'a CameraModel,
    pub window: &'a ObservationWindow,
    pub detections: &'a [Detection],
    pub gate: Option<f64>,
    pub rotation: Option<[[f64; 3]; 3]>,
    pub miss_update_mode: &'a str,
}

#[derive(Debug, Clone, Serialize)]
pub struct AffineResult {
    pub factor: SourceFactors,
    pub states: Vec<(Option<DetectionKey>, Option<State>)>,
    pub geometry: Option<([f64; 2], [[f64; 2]; 2])>,
}

struct Projection {
    mu: [f64; 2],
    h: [[f64; 2]; 2],
    s: [[f64; 2]; 2],
    mass: f64,
}

fn h4(h: [[f64; 2]; 2]) -> [[f64; 4]; 2] {
    [[h[0][0], h[0][1], 0., 0.], [h[1][0], h[1][1], 0., 0.]]
}

fn project(request: &Request, ctx: &Context) -> Result<Option<Projection>> {
    let Some(state) = &request.state else {
        if request.existence != 0. {
            return Err(Error::value(
                "positive existence requires a conditional state",
            ));
        }
        return Ok(None);
    };
    let mean = match state {
        State::Target(s) => [s.mean()[0], s.mean()[1]],
        State::Direction(s) => s.mean,
    };
    let (mu, h) = match project_direction_pixel_with_rotation(
        mean,
        &request.plane,
        ctx.frame,
        ctx.camera,
        ctx.rotation,
    ) {
        Ok(v) => v,
        Err(e)
            if matches!(
                e.code,
                Some("camera_singularity" | "distortion_domain" | "tangent_domain")
            ) =>
        {
            return Ok(None)
        }
        Err(e) => return Err(e),
    };
    let s = match state {
        State::Target(s) => {
            let hh = mat(h4(h));
            array(hh * mat(s.covariance()) * hh.transpose() + mat(request.noise))
        }
        State::Direction(s) => {
            array(mat(h) * mat(s.covariance) * mat(h).transpose() + mat(request.noise))
        }
    };
    let (mass, _) = n::gaussian_window_mass(mu, s, ctx.window)?;
    Ok(Some(Projection { mu, h, s, mass }))
}

fn invisible(r: &Request, count: usize) -> Result<AffineResult> {
    Ok(AffineResult {
        factor: SourceFactors::new(
            r.source_id.clone(),
            r.kind.clone(),
            r.existence,
            0.,
            vec![f64::NEG_INFINITY; count],
            None,
            None,
            None,
        )?,
        states: vec![(None, r.state.clone())],
        geometry: None,
    })
}

fn direction_hits(
    state: &GaussianDirection,
    h: [[f64; 2]; 2],
    r: [[f64; 2]; 2],
    residuals: &[[f64; 2]],
) -> Result<Vec<State>> {
    c::finite(
        &residuals.iter().flatten().copied().collect::<Vec<_>>(),
        "innovations",
    )?;
    c::finite(
        &h.iter().flatten().copied().collect::<Vec<_>>(),
        "observation matrix",
    )?;
    let r = mat(c::checked_covariance(r)?);
    let h = mat(h);
    let p = mat(state.covariance);
    let s = mat(c::checked_covariance(array(h * p * h.transpose() + r))?);
    let k = s
        .lu()
        .solve(&(h * p))
        .ok_or_else(|| Error::value("Singular matrix"))?
        .transpose();
    let a = SMatrix::<f64, 2, 2>::identity() - k * h;
    let cov = a * p * a.transpose() + k * r * k.transpose();
    let cov = array((cov + cov.transpose()) / 2.);
    residuals
        .iter()
        .map(|v| {
            Ok(State::Direction(GaussianDirection::new(
                (SVector::<f64, 2>::from(state.mean) + k * SVector::<f64, 2>::from(*v)).into(),
                cov,
            )?))
        })
        .collect()
}

fn distances(p: &Projection, ctx: &Context) -> Result<(Vec<[f64; 2]>, Vec<f64>, f64)> {
    let residuals: Vec<_> = ctx
        .detections
        .iter()
        .map(|d| [d.x_px - p.mu[0], d.y_px - p.mu[1]])
        .collect();
    let ds = if residuals.is_empty() {
        vec![]
    } else {
        efficient_graph::distances(&residuals, p.s)?
    };
    let lu = mat(p.s).lu();
    let u = lu.u();
    let logdet = libm::log(u[(0, 0)].abs()) + libm::log(u[(1, 1)].abs());
    Ok((residuals, ds, logdet))
}

fn result(
    r: &Request,
    p: &Projection,
    ctx: &Context,
    residuals: &[[f64; 2]],
    ds: &[f64],
    logdet: f64,
) -> Result<AffineResult> {
    let threshold = ctx
        .gate
        .map(efficient_graph::threshold)
        .transpose()?
        .unwrap_or(f64::INFINITY);
    let state = r.state.as_ref().unwrap();
    let mut miss = state.clone();
    if r.kind == "T" && ctx.miss_update_mode == "moment_matched" {
        miss = match state {
            State::Target(s) => State::Target(
                missed_update::miss_conditioned_gaussian(
                    s,
                    h4(p.h),
                    r.noise,
                    p.mu,
                    p.s,
                    ctx.window,
                    r.pd,
                    p.mass,
                )?
                .into_owned(),
            ),
            State::Direction(s) => State::Direction(
                missed_update::miss_conditioned_direction(
                    s, p.h, r.noise, p.mu, p.s, ctx.window, r.pd, p.mass,
                )?
                .into_owned(),
            ),
        };
    }
    let logs: Vec<_> = ds
        .iter()
        .map(|d| {
            if r.pd > 0. && *d <= threshold {
                libm::log(r.pd) - libm::log(2. * std::f64::consts::PI) - 0.5 * (logdet + d)
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let hits: Vec<_> = ctx
        .detections
        .iter()
        .zip(residuals)
        .zip(&logs)
        .filter(|(_, v)| v.is_finite())
        .map(|((d, res), _)| ((d.frame, d.detection_id.clone()), *res))
        .collect();
    let mut states = vec![(None, Some(miss))];
    if !hits.is_empty() {
        let residuals: Vec<_> = hits.iter().map(|x| x.1).collect();
        let updated = match state {
            State::Target(s) => n::gaussian_hit_updates(s, h4(p.h), r.noise, &residuals)?
                .into_iter()
                .map(State::Target)
                .collect(),
            State::Direction(s) => direction_hits(s, p.h, r.noise, &residuals)?,
        };
        for ((key, _), value) in hits.into_iter().zip(updated) {
            if let Some(old) = states.iter_mut().find(|(k, _)| k.as_ref() == Some(&key)) {
                old.1 = Some(value);
            } else {
                states.push((Some(key), Some(value)));
            }
        }
    }
    Ok(AffineResult {
        factor: SourceFactors::new(
            r.source_id.clone(),
            r.kind.clone(),
            r.existence,
            r.pd * p.mass,
            logs,
            None,
            None,
            None,
        )?,
        states,
        geometry: Some((p.mu, p.s)),
    })
}

pub fn affine_factors(request: &Request, ctx: &Context) -> Result<AffineResult> {
    let Some(p) = project(request, ctx)? else {
        return invisible(request, ctx.detections.len());
    };
    let (res, ds, ld) = distances(&p, ctx)?;
    result(request, &p, ctx, &res, &ds, ld)
}

pub fn affine_factors_many(
    requests: &[Request],
    ctx: &Context,
    check: &mut impl FnMut() -> Result<()>,
) -> Result<Vec<AffineResult>> {
    if requests.len() < 16 {
        return requests.iter().map(|r| affine_factors(r, ctx)).collect();
    }
    let mut out = vec![];
    for chunk in requests.chunks(128) {
        check()?;
        let projections = chunk
            .iter()
            .map(|r| project(r, ctx))
            .collect::<Result<Vec<_>>>()?;
        let computed = projections
            .iter()
            .map(|p| p.as_ref().map(|p| distances(p, ctx)).transpose())
            .collect::<Result<Vec<_>>>()?;
        for ((r, p), values) in chunk.iter().zip(projections.iter()).zip(computed) {
            out.push(if let (Some(p), Some((res, ds, ld))) = (p, values) {
                result(r, p, ctx, &res, &ds, ld)?
            } else {
                invisible(r, ctx.detections.len())?
            });
        }
    }
    Ok(out)
}
