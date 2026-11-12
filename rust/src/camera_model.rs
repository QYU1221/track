use crate::{
    contracts::{fields, finite, numbers, scalar, text, vector, Camera},
    error::{Error, Result},
    geometry::unit,
    linalg::{array, mat},
    numerical_policy::*,
};
use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Serialize)]
pub struct CameraModel {
    pub(crate) camera: Camera,
    pub(crate) skew_px: f64,
    pub(crate) distortion_model: String,
    pub(crate) distortion_coefficients: [f64; 5],
    pub(crate) max_normalized_radius: f64,
    pub(crate) inverse_tolerance_px: f64,
    pub(crate) max_iterations: usize,
    pub(crate) calibration_version: String,
    pub(crate) calibration_status: String,
    pub(crate) camera_axis_rotation_deg: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CameraRay {
    pub normalized_image: [f64; 2],
    pub corrected_pixel: [f64; 2],
    pub camera_los: [f64; 3],
    pub jacobian_camera_pixel: [[f64; 2]; 3],
    pub inverse_iterations: usize,
}

fn singular_solve() -> Error {
    Error::geometry(
        "distortion_noninvertible",
        "cannot solve calibration Jacobian",
    )
}

impl CameraModel {
    pub fn from_json(v: &Value) -> Result<Self> {
        let camera = Camera::from_json(&v["camera"])?;
        fields(
            v,
            "CameraModel",
            &[
                "camera",
                "skew_px",
                "distortion_model",
                "distortion_coefficients",
                "max_normalized_radius",
                "inverse_tolerance_px",
                "max_iterations",
                "calibration_version",
                "calibration_status",
                "camera_axis_rotation_deg",
            ],
            &[],
        )?;
        let msg = "distortion supports only none or brown5 (k1,k2,p1,p2,k3)";
        let distortion_coefficients = vector::<5>(
            &v["distortion_coefficients"],
            "distortion coefficients",
            msg,
        )?;
        let distortion_model = text(&v["distortion_model"], msg)?;
        if !["none", "brown5"].contains(&distortion_model.as_str()) {
            return Err(Error::value(msg));
        }
        if distortion_model == "none" && distortion_coefficients.iter().any(|x| *x != 0.) {
            return Err(Error::value("none distortion requires zero coefficients"));
        }
        numbers(
            &serde_json::json!([
                v["skew_px"],
                v["max_normalized_radius"],
                v["inverse_tolerance_px"]
            ]),
            "camera model",
        )?;
        let skew_px = scalar(&v["skew_px"], "camera model")?;
        let max_normalized_radius = scalar(&v["max_normalized_radius"], "camera model")?;
        let inverse_tolerance_px = scalar(&v["inverse_tolerance_px"], "camera model")?;
        if max_normalized_radius <= 0. || inverse_tolerance_px <= 0. {
            return Err(Error::value(
                "camera radius and inverse pixel tolerance must be positive",
            ));
        }
        let max_iterations = v["max_iterations"]
            .as_u64()
            .filter(|x| *x > 0)
            .and_then(|x| usize::try_from(x).ok())
            .ok_or_else(|| Error::value("max_iterations must be a positive integer"))?;
        let msg = "declare calibration version and status";
        let calibration_version = text(&v["calibration_version"], msg)?;
        let calibration_status = text(&v["calibration_status"], msg)?;
        if calibration_version.is_empty()
            || ![
                "unverified",
                "reported",
                "synthetic_test_value",
                "calibrated",
            ]
            .contains(&calibration_status.as_str())
        {
            return Err(Error::value(msg));
        }
        let camera_axis_rotation_deg = v["camera_axis_rotation_deg"]
            .as_f64()
            .filter(|x| x.is_finite())
            .ok_or_else(|| {
                Error::value("camera_axis_rotation_deg must be a finite angle in degrees")
            })?;
        Ok(Self {
            camera,
            skew_px,
            distortion_model,
            distortion_coefficients,
            max_normalized_radius,
            inverse_tolerance_px,
            max_iterations,
            calibration_version,
            calibration_status,
            camera_axis_rotation_deg,
        })
    }
    pub fn image_to_camera_rotation(&self) -> [[f64; 3]; 3] {
        let angle = self.camera_axis_rotation_deg.rem_euclid(360.);
        let (cosine, sine) = if angle % 90. == 0. {
            [(1., 0.), (0., 1.), (-1., 0.), (0., -1.)][(angle / 90.) as usize % 4]
        } else {
            let radians = angle.to_radians();
            (radians.cos(), radians.sin())
        };
        [[-cosine, sine, 0.], [-sine, -cosine, 0.], [0., 0., 1.]]
    }
    pub fn k(&self) -> [[f64; 3]; 3] {
        let c = &self.camera;
        [
            [c.fx_px, self.skew_px, c.cx_px],
            [0., c.fy_px, c.cy_px],
            [0., 0., 1.],
        ]
    }
    fn a(&self) -> Matrix2<f64> {
        Matrix2::new(self.camera.fx_px, self.skew_px, 0., self.camera.fy_px)
    }
    pub fn distort(&self, p: [f64; 2]) -> Result<([f64; 2], [[f64; 2]; 2])> {
        finite(&p, "normalized image")?;
        if Vector2::from(p).norm() > self.max_normalized_radius {
            return Err(Error::geometry(
                "distortion_domain",
                "normalized image outside declared distortion domain",
            ));
        }
        if self.distortion_model == "none" {
            return Ok((p, [[1., 0.], [0., 1.]]));
        }
        let [x, y] = p;
        let [k1, k2, p1, p2, k3] = self.distortion_coefficients;
        use crate::numpy_arithmetic::N;
        let r2 = N::new(x * x + y * y);
        let x = N::new(x);
        let y = N::new(y);
        let radial =
            N::new(1.) + N::new(k1) * r2 + N::new(k2) * r2 * r2 + N::new(k3) * r2 * r2 * r2;
        let factor = N::new(2.) * (N::new(k1) + N::new(2. * k2) * r2 + N::new(3. * k3) * r2 * r2);
        let dr = [factor * x, factor * y];
        let out = [
            x * radial + N::new(2. * p1) * x * y + N::new(p2) * (r2 + N::new(2.) * x * x),
            y * radial + N::new(p1) * (r2 + N::new(2.) * y * y) + N::new(2. * p2) * x * y,
        ];
        let j = [
            [
                radial + x * dr[0] + N::new(2. * p1) * y + N::new(6. * p2) * x,
                x * dr[1] + N::new(2. * p1) * x + N::new(2. * p2) * y,
            ],
            [
                y * dr[0] + N::new(2. * p1) * x + N::new(2. * p2) * y,
                radial + y * dr[1] + N::new(6. * p1) * y + N::new(2. * p2) * x,
            ],
        ];
        if out.iter().chain(j.iter().flatten()).any(|x| x.1) {
            return Err(Error::geometry(
                "distortion_numeric",
                "distortion exceeds floating-point range",
            ));
        }
        let out = out.map(|x| x.0);
        let j = j.map(|r| r.map(|x| x.0));
        finite(&out, "distorted image")?;
        finite(
            &j.iter().flatten().copied().collect::<Vec<_>>(),
            "distortion Jacobian",
        )?;
        let m = mat(j);
        let sv = m.svd(false, false).singular_values;
        if m.determinant() <= 0.
            || sv[0].min(sv[1]) <= BASIS_EPS_MULTIPLIER * f64::EPSILON * sv[0].max(sv[1])
        {
            return Err(Error::geometry(
                "distortion_noninvertible",
                "folded or singular local distortion calibration",
            ));
        }
        Ok((out, j))
    }
    pub fn undistort_pixel(&self, pixel: [f64; 2]) -> Result<([f64; 2], [[f64; 2]; 2], usize)> {
        finite(&pixel, "pixels")?;
        let a = self.a();
        let center = Vector2::new(self.camera.cx_px, self.camera.cy_px);
        let lu = a.lu();
        let target = lu
            .solve(&(Vector2::from(pixel) - center))
            .ok_or_else(singular_solve)?;
        let mut estimate =
            if self.distortion_model == "brown5" && target.norm() > self.max_normalized_radius {
                Vector2::zeros()
            } else {
                target
            };
        for iteration in 0..self.max_iterations {
            let (forward, j) = self.distort(estimate.into())?;
            let residual = Vector2::from(forward) - target;
            let error_px = (a * residual).norm();
            let j_lu = mat(j).lu();
            if error_px <= self.inverse_tolerance_px {
                let inv_a = lu.solve(&Matrix2::identity()).ok_or_else(singular_solve)?;
                let jac = j_lu.solve(&inv_a).ok_or_else(singular_solve)?;
                return Ok((estimate.into(), array(jac), iteration));
            }
            let step = j_lu.solve(&residual).ok_or_else(singular_solve)?;
            let mut accepted = false;
            for power in 0..20 {
                let candidate = estimate - step * 0.5_f64.powi(power);
                let mapped = match self.distort(candidate.into()) {
                    Ok((mapped, _)) => Vector2::from(mapped),
                    Err(e) if e.code.is_some() => continue,
                    Err(e) => return Err(e),
                };
                if (a * (mapped - target)).norm() < error_px {
                    estimate = candidate;
                    accepted = true;
                    break;
                }
            }
            if !accepted {
                break;
            }
        }
        Err(Error::geometry(
            "distortion_nonconvergence",
            "inverse distortion did not meet declared pixel tolerance",
        ))
    }
    pub fn pixel_to_camera(&self, pixel: [f64; 2]) -> Result<CameraRay> {
        let (p, j_image, inverse_iterations) = self.undistort_pixel(pixel)?;
        let rotation = mat(self.image_to_camera_rotation());
        let a = rotation * Vector3::new(p[0], p[1], 1.);
        let norm = a.norm();
        let b = a / norm;
        let j_normalize = (Matrix3::identity() - b * b.transpose()) / norm;
        let j = j_normalize * rotation.fixed_columns::<2>(0) * mat(j_image);
        let corrected =
            self.a() * Vector2::from(p) + Vector2::new(self.camera.cx_px, self.camera.cy_px);
        Ok(CameraRay {
            normalized_image: p,
            corrected_pixel: corrected.into(),
            camera_los: b.into(),
            jacobian_camera_pixel: array(j),
            inverse_iterations,
        })
    }
    pub fn camera_to_pixel(&self, direction: [f64; 3]) -> Result<[f64; 2]> {
        let b = unit(&direction)?;
        if b[2] <= UNIT_VECTOR_EPS_MULTIPLIER * f64::EPSILON {
            return Err(Error::geometry(
                "camera_singularity",
                "camera ray is rear-facing or at projection singularity",
            ));
        }
        let rotation = mat(self.image_to_camera_rotation());
        let normalized =
            rotation.fixed_view::<2, 2>(0, 0).transpose() * Vector2::new(b[0] / b[2], b[1] / b[2]);
        let (image, _) = self.distort(normalized.into())?;
        let projected =
            self.a() * Vector2::from(image) + Vector2::new(self.camera.cx_px, self.camera.cy_px);
        let out: [f64; 2] = projected.into();
        finite(&out, "projected pixel")?;
        Ok(out)
    }
}
