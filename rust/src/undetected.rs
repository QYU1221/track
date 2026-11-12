use crate::{
    camera_model::CameraModel,
    contracts::{self as c, Detection, FrameInfo},
    error::{Error, Result},
    exact_factors::DetectionKey,
    exact_numerics::{GaussianState, ObservationWindow},
    geometry::quaternion_matrix,
    gnomonic::GnomonicPlane,
    linalg::{array, mat},
    online_factors::{self, Context, Request, State},
    tracking_background::BackgroundConfig,
    tracking_model::{self, GaussianDirection, TrackingConfig},
};
use nalgebra::{Matrix2, Vector2};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Serialize)]
pub struct Population {
    pub(crate) kind: String,
    pub(crate) expected_count: f64,
    pub(crate) state: State,
    pub(crate) plane: GnomonicPlane,
}

impl Population {
    pub fn new(
        kind: String,
        expected_count: f64,
        state: State,
        plane: GnomonicPlane,
    ) -> Result<Self> {
        c::finite(&[expected_count], "population expected count")?;
        if expected_count < 0. {
            return Err(Error::value(
                "population expected count must be nonnegative",
            ));
        }
        if !matches!(
            (kind.as_str(), &state),
            ("T", State::Target(_)) | ("S_U", State::Direction(_))
        ) {
            return Err(Error::value("population kind/state must be T/4D or S_U/2D"));
        }
        Ok(Self {
            kind,
            expected_count,
            state,
            plane,
        })
    }
    pub fn from_json(v: &Value) -> Result<Self> {
        let kind = v["kind"]
            .as_str()
            .ok_or_else(|| Error::value("population kind must be a string"))?;
        let state = if kind == "T" {
            State::Target(GaussianState::from_json(&v["state"])?)
        } else {
            State::Direction(GaussianDirection::from_json(&v["state"])?)
        };
        Self::new(
            kind.into(),
            c::scalar(&v["expected_count"], "population expected count")?,
            state,
            GnomonicPlane::from_json(&v["plane"])?,
        )
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PopulationOptions {
    target_initial_expected_count: f64,
    unknown_star_initial_expected_count: f64,
    target_birth_rate_per_second: f64,
    unknown_star_birth_rate_per_second: f64,
    velocity_sigma_rad_s: f64,
    parameter_provenance: String,
}

impl PopulationOptions {
    pub fn from_json(v: &Value) -> Result<Self> {
        let mut values = vec![];
        for key in [
            "target_initial_expected_count",
            "unknown_star_initial_expected_count",
            "target_birth_rate_per_second",
            "unknown_star_birth_rate_per_second",
            "velocity_sigma_rad_s",
        ] {
            let x = v.get(key).ok_or_else(|| Error {
                kind: "KeyError",
                code: None,
                message: format!("'{key}'"),
            })?;
            let x = c::scalar(x, key)
                .ok()
                .filter(|x| *x >= 0. && (key != "velocity_sigma_rad_s" || *x > 0.))
                .ok_or_else(|| {
                    Error::value(format!("invalid undetected population parameter: {key}"))
                })?;
            values.push(x);
        }
        let provenance = v["parameter_provenance"]
            .as_str()
            .filter(|x| !x.is_empty())
            .ok_or_else(|| Error::value("population parameter provenance required"))?;
        Ok(Self {
            target_initial_expected_count: values[0],
            unknown_star_initial_expected_count: values[1],
            target_birth_rate_per_second: values[2],
            unknown_star_birth_rate_per_second: values[3],
            velocity_sigma_rad_s: values[4],
            parameter_provenance: provenance.into(),
        })
    }
}

pub fn spatial_prior(
    camera: &CameraModel,
    frame: &FrameInfo,
    angle: f64,
) -> Result<(GnomonicPlane, [f64; 2], [[f64; 2]; 2])> {
    let r = quaternion_matrix(
        &frame
            .quaternion_xyzw
            .ok_or_else(|| Error::value("quaternion must be finite"))?,
    )?;
    let plane = GnomonicPlane::new(
        &format!("population-frame-{}", frame.frame),
        [r[0][2], r[1][2], r[2][2]],
        angle,
        Some(([r[0][0], r[1][0], r[2][0]], [r[0][1], r[1][1], r[2][1]])),
    )?;
    let k = mat(camera.k());
    let transform = mat(camera.image_to_camera_rotation())
        .fixed_view::<2, 2>(0, 0)
        .into_owned()
        * k.fixed_view::<2, 2>(0, 0)
            .into_owned()
            .lu()
            .try_inverse()
            .ok_or_else(|| Error::value("Singular matrix"))?;
    let (width, height) = camera.camera.dimensions();
    let (w, h) = (width as f64, height as f64);
    let mean = transform * Vector2::new(w / 2. - k[(0, 2)], h / 2. - k[(1, 2)]);
    let cov = transform
        * Matrix2::from_diagonal(&Vector2::new(w * w / 12., h * h / 12.))
        * transform.transpose();
    Ok((plane, mean.into(), array(cov)))
}

fn predicted(p: &Population, dt: f64, q: f64) -> Result<State> {
    match &p.state {
        State::Target(s) => Ok(State::Target(tracking_model::predict_gaussian(s, dt, q)?)),
        State::Direction(_) => Ok(p.state.clone()),
    }
}

pub fn prepare(
    populations: &[Population],
    frame: &FrameInfo,
    camera: &CameraModel,
    tracking: &TrackingConfig,
    background: &BackgroundConfig,
    options: &PopulationOptions,
    dt: f64,
    angle: f64,
) -> Result<Vec<Population>> {
    if !frame.inertial_observation_available() {
        return populations
            .iter()
            .map(|p| {
                Population::new(
                    p.kind.clone(),
                    p.expected_count
                        * if p.kind == "T" {
                            libm::pow(
                                tracking.survival_probability,
                                dt / tracking.survival_reference_dt_s,
                            )
                        } else {
                            background.persistence_probability
                        },
                    predicted(p, dt, tracking.q_a)?,
                    p.plane.clone(),
                )
            })
            .collect();
    }
    let (newplane, center, footprint) = spatial_prior(camera, frame, angle)?;
    let mut output = vec![];
    for kind in ["T", "S_U"] {
        let previous = populations.iter().find(|p| p.kind == kind);
        let plane = previous.map(|p| &p.plane).unwrap_or(&newplane);
        let (mean, cov) = if previous.is_some() {
            let (direction, j) = newplane.inverse(center)?;
            let (mean, a) = plane.project(direction)?;
            let (a, j) = (mat(a), mat(j));
            (
                mean,
                array(a * j * mat(footprint) * j.transpose() * a.transpose()),
            )
        } else {
            (center, footprint)
        };
        let (injected, survival, initial, rate) = if kind == "T" {
            let mut p = [[0.; 4]; 4];
            for i in 0..2 {
                for j in 0..2 {
                    p[i][j] = cov[i][j];
                }
                p[i + 2][i + 2] = options.velocity_sigma_rad_s * options.velocity_sigma_rad_s;
            }
            (
                State::Target(GaussianState::new([mean[0], mean[1], 0., 0.], p)?),
                libm::pow(
                    tracking.survival_probability,
                    dt / tracking.survival_reference_dt_s,
                ),
                options.target_initial_expected_count,
                options.target_birth_rate_per_second,
            )
        } else {
            (
                State::Direction(GaussianDirection::new(mean, cov)?),
                background.persistence_probability,
                options.unknown_star_initial_expected_count,
                options.unknown_star_birth_rate_per_second,
            )
        };
        let added = if previous.is_some() {
            rate * dt
        } else {
            initial
        };
        let old_count = previous.map(|p| p.expected_count * survival).unwrap_or(0.);
        let count = old_count + added;
        let old = previous
            .map(|p| predicted(p, dt, tracking.q_a))
            .transpose()?
            .unwrap_or_else(|| injected.clone());
        let state = if count > 0. {
            match (&old, &injected) {
                (State::Target(a), State::Target(b)) => State::Target(
                    tracking_model::mixture_moments(&[old_count, added], &[a.clone(), b.clone()])?
                        .unwrap(),
                ),
                (State::Direction(a), State::Direction(b)) => State::Direction(
                    tracking_model::mixture_directions(
                        &[old_count, added],
                        &[a.clone(), b.clone()],
                    )?
                    .unwrap(),
                ),
                _ => unreachable!(),
            }
        } else {
            injected
        };
        output.push(Population::new(kind.into(), count, state, plane.clone())?);
    }
    Ok(output)
}

#[derive(Debug, Clone, Serialize)]
pub struct Observation {
    pub remaining: Vec<Population>,
    pub intensities: Vec<(DetectionKey, Vec<(String, f64)>)>,
    pub conditionals: Vec<((DetectionKey, String), (Option<State>, GnomonicPlane))>,
    pub detected_count: f64,
}

pub fn observe(
    populations: &[Population],
    frame: &FrameInfo,
    camera: &CameraModel,
    window: &ObservationWindow,
    detections: &[Detection],
    tracking: &TrackingConfig,
    background: &BackgroundConfig,
    rotation: Option<[[f64; 3]; 3]>,
) -> Result<Observation> {
    let mut out = Observation {
        remaining: vec![],
        intensities: vec![],
        conditionals: vec![],
        detected_count: 0.,
    };
    for d in detections {
        let key = (d.frame, d.detection_id.clone());
        if !out.intensities.iter().any(|(k, _)| *k == key) {
            out.intensities.push((key, vec![]));
        }
    }
    for p in populations {
        let (pd, noise) = if p.kind == "T" {
            (tracking.p_detection, tracking.measurement_covariance_px2)
        } else {
            (
                background.p_detection,
                background.measurement_covariance_px2,
            )
        };
        let request = Request {
            source_id: format!("PPP:{}", p.kind),
            kind: p.kind.clone(),
            state: Some(p.state.clone()),
            existence: 1.,
            plane: p.plane.clone(),
            pd,
            noise,
        };
        let ctx = Context {
            frame,
            camera,
            window,
            detections: &[],
            gate: None,
            rotation,
            miss_update_mode: "retain",
        };
        let factor = online_factors::affine_factors(&request, &ctx)?.factor;
        out.detected_count += p.expected_count * factor.pd_bar;
        out.remaining.push(Population::new(
            p.kind.clone(),
            p.expected_count * (1. - factor.pd_bar),
            p.state.clone(),
            p.plane.clone(),
        )?);
        let skip = p.expected_count <= 0. || pd <= 0.;
        let updates = if skip {
            vec![
                crate::birth_update::BirthResult {
                    log_evidence: f64::NEG_INFINITY,
                    state: None
                };
                detections.len()
            ]
        } else {
            crate::birth_update::nonlinear_birth_updates(
                &p.state, &p.plane, frame, camera, noise, detections, rotation,
            )?
        };
        for (d, u) in detections.iter().zip(updates) {
            let evidence = libm::exp(u.log_evidence);
            if evidence.is_infinite() {
                return Err(Error {
                    kind: "OverflowError",
                    code: None,
                    message: "math range error".into(),
                });
            }
            let intensity = p.expected_count * pd * evidence;
            let key = (d.frame, d.detection_id.clone());
            let entries = &mut out
                .intensities
                .iter_mut()
                .find(|(k, _)| *k == key)
                .unwrap()
                .1;
            if let Some((_, value)) = entries.iter_mut().find(|(kind, _)| *kind == p.kind) {
                *value = intensity;
            } else {
                entries.push((p.kind.clone(), intensity));
            }
            if intensity > 0. {
                let ck = (key, p.kind.clone());
                let value = (u.state, p.plane.clone());
                if let Some((_, old)) = out.conditionals.iter_mut().find(|(k, _)| *k == ck) {
                    *old = value;
                } else {
                    out.conditionals.push((ck, value));
                }
            }
        }
    }
    Ok(out)
}
