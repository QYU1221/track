use crate::{
    camera_model::CameraModel,
    contracts::{Detection, FrameInfo},
    error::{Error, Result},
    exact_factors::wire,
    exact_numerics::GaussianState,
    geometry::quaternion_matrix,
    gnomonic::GnomonicPlane,
    linalg::{array, mat},
    online_factors::State,
    tracking_model::GaussianDirection,
};
use nalgebra::{Matrix2, Matrix3, SMatrix, Vector2, Vector3};
use serde::Serialize;
const NODES: [f64; 7] = [
    -2.6519613568352334,
    -1.6735516287674714,
    -0.8162878828589646,
    0.,
    0.8162878828589646,
    1.6735516287674714,
    2.6519613568352334,
];
const WEIGHTS: [f64; 7] = [
    0.0009717812450995199,
    0.05451558281912705,
    0.4256072526101278,
    0.8102646175568072,
    0.4256072526101278,
    0.05451558281912705,
    0.0009717812450995199,
];
fn singular() -> Error {
    Error {
        kind: "LinAlgError",
        code: None,
        message: "Singular matrix".into(),
    }
}

fn not_pd() -> Error {
    Error {
        kind: "LinAlgError",
        code: None,
        message: "Matrix is not positive definite".into(),
    }
}

fn logdet(m: Matrix2<f64>) -> f64 {
    let u = m.lu().u();
    libm::log(u[(0, 0)].abs()) + libm::log(u[(1, 1)].abs())
}

pub struct InversePixel {
    pub position: [f64; 2],
    pub jacobian: [[f64; 2]; 2],
    pub valid: bool,
}

pub struct InverseGeometry<'a> {
    camera: &'a CameraModel,
    plane: &'a GnomonicPlane,
    rotation: Matrix3<f64>,
    pinhole: Option<(Matrix3<f64>, Matrix3<f64>)>,
}

impl<'a> InverseGeometry<'a> {
    pub fn new(
        camera: &'a CameraModel,
        plane: &'a GnomonicPlane,
        rotation: [[f64; 3]; 3],
    ) -> Result<Self> {
        let rotation = mat(rotation);
        let pinhole = if camera.distortion_model == "none" {
            let (c, e1, e2) = plane.basis();
            let basis = Matrix3::from_rows(&[
                Vector3::from(e1).transpose(),
                Vector3::from(e2).transpose(),
                Vector3::from(c).transpose(),
            ]);
            let inverse = mat(camera.k()).lu().try_inverse().ok_or_else(singular)?;
            Some((
                basis * rotation * mat(camera.image_to_camera_rotation()) * inverse,
                inverse,
            ))
        } else {
            None
        };
        Ok(Self {
            camera,
            plane,
            rotation,
            pinhole,
        })
    }
    pub fn inverse(&self, pixels: &[[f64; 2]]) -> Result<Vec<InversePixel>> {
        pixels
            .iter()
            .map(|&pixel| {
                if let Some((transform, intrinsic_inverse)) = &self.pinhole {
                    let homogeneous = Vector3::new(pixel[0], pixel[1], 1.);
                    let ray = transform * homogeneous;
                    let normalized = intrinsic_inverse * homogeneous;
                    let p = [ray[0] / ray[2], ray[1] / ray[2]];
                    let j = std::array::from_fn(|i| {
                        std::array::from_fn(|k| {
                            (transform[(i, k)] - p[i] * transform[(2, k)]) / ray[2]
                        })
                    });
                    let valid = ray[2] > 0.
                        && p.iter().all(|x| x.is_finite())
                        && (p[0] * p[0] + p[1] * p[1]).sqrt()
                            < libm::tan(self.plane.max_angle_rad())
                        && (normalized[0] * normalized[0] + normalized[1] * normalized[1]).sqrt()
                            <= self.camera.max_normalized_radius;
                    Ok(InversePixel {
                        position: p,
                        jacobian: j,
                        valid,
                    })
                } else {
                    let compute = || -> Result<InversePixel> {
                        let ray = self.camera.pixel_to_camera(pixel)?;
                        let (p, j) = self
                            .plane
                            .project((self.rotation * Vector3::from(ray.camera_los)).into())?;
                        Ok(InversePixel {
                            position: p,
                            jacobian: array(
                                mat(j) * self.rotation * mat(ray.jacobian_camera_pixel),
                            ),
                            valid: true,
                        })
                    };
                    match compute() {
                        Err(e)
                            if matches!(
                                e.code,
                                Some("camera_singularity" | "distortion_domain" | "tangent_domain")
                            ) =>
                        {
                            Ok(InversePixel {
                                position: [0.; 2],
                                jacobian: [[0.; 2]; 2],
                                valid: false,
                            })
                        }
                        other => other,
                    }
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BirthResult {
    #[serde(serialize_with = "wire::float")]
    pub log_evidence: f64,
    pub state: Option<State>,
}

pub fn nonlinear_birth_updates(
    state: &State,
    plane: &GnomonicPlane,
    frame: &FrameInfo,
    camera: &CameraModel,
    noise: [[f64; 2]; 2],
    detections: &[Detection],
    rotation: Option<[[f64; 3]; 3]>,
) -> Result<Vec<BirthResult>> {
    let rotation = if let Some(r) = rotation {
        r
    } else {
        quaternion_matrix(
            &frame
                .quaternion_xyzw
                .ok_or_else(|| Error::value("quaternion must be finite"))?,
        )?
    };
    let inverse = InverseGeometry::new(camera, plane, rotation)?;
    let (mean, p, velocity) = match state {
        State::Direction(s) => (Vector2::from(s.mean), mat(s.covariance), None),
        State::Target(s) => {
            let mean = s.mean();
            let p = mat(s.covariance());
            let position = p.fixed_view::<2, 2>(0, 0).into_owned();
            let gain = position
                .lu()
                .solve(&p.fixed_view::<2, 2>(0, 2).into_owned())
                .ok_or_else(singular)?
                .transpose();
            let ch = p
                .cholesky()
                .ok_or_else(not_pd)?
                .l()
                .fixed_view::<2, 2>(2, 2)
                .into_owned();
            (
                Vector2::new(mean[0], mean[1]),
                position,
                Some((Vector2::new(mean[2], mean[3]), gain, ch * ch.transpose())),
            )
        }
    };
    let precision = p.lu().try_inverse().ok_or_else(singular)?;
    let noise = mat(noise);
    let noise_precision = noise.lu().try_inverse().ok_or_else(singular)?;
    let prior_logdet = logdet(p);
    let noise_logdet = logdet(noise);
    let mut results = vec![];
    for detection in detections {
        let observed = Vector2::new(detection.x_px, detection.y_px);
        let center = inverse.inverse(&[observed.into()])?.remove(0);
        if !center.valid {
            results.push(BirthResult {
                log_evidence: f64::NEG_INFINITY,
                state: None,
            });
            continue;
        }
        let j = mat(center.jacobian);
        let proposal = (noise_precision + j.transpose() * precision * j)
            .lu()
            .try_inverse()
            .ok_or_else(singular)?;
        let proposal_mean = observed
            + proposal * j.transpose() * precision * (mean - Vector2::from(center.position));
        let l = proposal.cholesky().ok_or_else(not_pd)?.l();
        let mut grid = vec![];
        let mut log_weights = vec![];
        for i in 0..7 {
            for k in 0..7 {
                grid.push(Vector2::new(
                    NODES[i] * std::f64::consts::SQRT_2,
                    NODES[k] * std::f64::consts::SQRT_2,
                ));
                log_weights.push(libm::log(WEIGHTS[i] * WEIGHTS[k] / std::f64::consts::PI));
            }
        }
        let pixels: Vec<[f64; 2]> = grid
            .iter()
            .map(|g| (proposal_mean + l * g).into())
            .collect();
        let projected = inverse.inverse(&pixels)?;
        let mut parts = vec![];
        for (i, x) in projected.iter().enumerate() {
            if !x.valid {
                continue;
            }
            let position = Vector2::from(x.position);
            let prior_residual = position - mean;
            let measurement = Vector2::from(pixels[i]) - observed;
            let value = log_weights[i]
                - libm::log(2. * std::f64::consts::PI)
                - 0.5 * (prior_logdet + noise_logdet)
                + libm::log(l[(0, 0)])
                + libm::log(l[(1, 1)])
                - 0.5 * prior_residual.dot(&(precision * prior_residual))
                - 0.5 * measurement.dot(&(noise_precision * measurement))
                + 0.5 * grid[i].dot(&grid[i])
                + logdet(mat(x.jacobian));
            parts.push((value, position));
        }
        if parts.is_empty() {
            results.push(BirthResult {
                log_evidence: f64::NEG_INFINITY,
                state: None,
            });
            continue;
        }
        let maximum = parts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let mut weights: Vec<_> = parts.iter().map(|p| libm::exp(p.0 - maximum)).collect();
        let normalizer: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= normalizer;
        }
        let mut position_mean = Vector2::zeros();
        for (p, w) in parts.iter().zip(&weights) {
            position_mean += p.1 * (*w);
        }
        let mut position_cov = Matrix2::zeros();
        for (p, w) in parts.iter().zip(&weights) {
            let centered = p.1 - position_mean;
            position_cov += centered * centered.transpose() * (*w);
        }
        let posterior = if let Some((velocity_mean, gain, velocity_cov)) = velocity {
            let v = velocity_mean + gain * (position_mean - mean);
            let cross = position_cov * gain.transpose();
            let vv = velocity_cov + gain * position_cov * gain.transpose();
            let mut covariance = SMatrix::<f64, 4, 4>::zeros();
            covariance
                .fixed_view_mut::<2, 2>(0, 0)
                .copy_from(&position_cov);
            covariance.fixed_view_mut::<2, 2>(0, 2).copy_from(&cross);
            covariance
                .fixed_view_mut::<2, 2>(2, 0)
                .copy_from(&cross.transpose());
            covariance.fixed_view_mut::<2, 2>(2, 2).copy_from(&vv);
            State::Target(GaussianState::new(
                [position_mean[0], position_mean[1], v[0], v[1]],
                array((covariance + covariance.transpose()) / 2.),
            )?)
        } else {
            State::Direction(GaussianDirection::new(
                position_mean.into(),
                array((position_cov + position_cov.transpose()) / 2.),
            )?)
        };
        results.push(BirthResult {
            log_evidence: maximum + libm::log(normalizer),
            state: Some(posterior),
        });
    }
    Ok(results)
}
