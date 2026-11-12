use crate::{
    camera_model::CameraModel,
    contracts::{checked_covariance, finite, Detection, FrameInfo},
    error::{Error, Result},
    geometry::camera_to_inertial,
    gnomonic::GnomonicPlane,
    linalg::{array, mat},
    measurement_covariance::*,
    numerical_policy::*,
};
use serde::Serialize;
#[derive(Debug, Clone, Serialize)]
pub struct GeometryObservation {
    detection_key: (i64, String),
    timestamp_s: f64,
    pixel_position: [f64; 2],
    pixel_covariance: Option<[[f64; 2]; 2]>,
    camera_los: Option<[f64; 3]>,
    j2000_los: Option<[f64; 3]>,
    los_covariance: Option<[[f64; 3]; 3]>,
    tangent_position: Option<[f64; 2]>,
    tangent_covariance: Option<[[f64; 2]; 2]>,
    tangent_reference_id: String,
    attitude_valid: bool,
    geometry_valid: bool,
    covariance_valid: bool,
    covariance_source: String,
    calibration_version: String,
    calibration_status: String,
    quality_flags: Vec<String>,
    normalized_image: Option<[f64; 2]>,
    jacobian_los_pixel: Option<[[f64; 2]; 3]>,
    jacobian_tangent_pixel: Option<[[f64; 2]; 2]>,
    tangent_coordinate_type: String,
}

impl GeometryObservation {
    fn validate(&self) -> Result<()> {
        finite(&self.pixel_position, "pixel_position")?;
        for (value, name) in [
            (self.camera_los, "camera_los"),
            (self.j2000_los, "j2000_los"),
        ] {
            if let Some(v) = value {
                finite(&v, name)?;
            }
        }
        for (value, name) in [
            (self.tangent_position, "tangent_position"),
            (self.normalized_image, "normalized_image"),
        ] {
            if let Some(v) = value {
                finite(&v, name)?;
            }
        }
        for (value, name) in [
            (self.pixel_covariance, "pixel_covariance"),
            (self.tangent_covariance, "tangent_covariance"),
            (self.jacobian_tangent_pixel, "jacobian_tangent_pixel"),
        ] {
            if let Some(v) = value {
                finite(&v.iter().flatten().copied().collect::<Vec<_>>(), name)?;
            }
        }
        if let Some(v) = self.jacobian_los_pixel {
            finite(
                &v.iter().flatten().copied().collect::<Vec<_>>(),
                "jacobian_los_pixel",
            )?;
        }
        if let Some(v) = self.los_covariance {
            finite(
                &v.iter().flatten().copied().collect::<Vec<_>>(),
                "los_covariance",
            )?;
        }
        finite(&[self.timestamp_s], "geometry timestamp")?;
        for v in [self.camera_los, self.j2000_los].into_iter().flatten() {
            if (v.iter().map(|x| x * x).sum::<f64>() - 1.).abs()
                > UNIT_VECTOR_EPS_MULTIPLIER * f64::EPSILON
            {
                return Err(Error::value("geometry LOS must be a unit direction"));
            }
        }
        if !self.attitude_valid && self.j2000_los.is_some() {
            return Err(Error::value(
                "invalid attitude cannot expose trusted J2000 LOS",
            ));
        }
        if let Some(c) = self.los_covariance {
            checked_los_covariance(c)?;
            let u = self.j2000_los.ok_or_else(|| {
                Error::value("LOS covariance must be tangent to the observed unit direction")
            })?;
            let scale = c.iter().flatten().fold(0.0_f64, |a, x| a.max(x.abs()));
            if (mat(c) * nalgebra::Vector3::from(u)).norm()
                > PROBABILITY_EPS_MULTIPLIER * f64::EPSILON * scale
            {
                return Err(Error::value(
                    "LOS covariance must be tangent to the observed unit direction",
                ));
            }
        }
        if self.geometry_valid
            && (self.camera_los.is_none()
                || self.j2000_los.is_none()
                || self.tangent_position.is_none())
        {
            return Err(Error::value(
                "valid geometry requires complete directional observations",
            ));
        }
        if self.covariance_valid
            && (self.los_covariance.is_none() || self.tangent_covariance.is_none())
        {
            return Err(Error::value(
                "valid covariance requires LOS and tangent covariance",
            ));
        }
        for c in [self.pixel_covariance, self.tangent_covariance]
            .into_iter()
            .flatten()
        {
            checked_covariance(c)?;
        }
        if self.tangent_coordinate_type != "gnomonic_ratio" {
            return Err(Error::value(
                "geometry1.0 accepts only gnomonic ratio observations",
            ));
        }
        Ok(())
    }
}

pub fn observe_detection(
    d: &Detection,
    f: &FrameInfo,
    m: &CameraModel,
    p: &GnomonicPlane,
) -> Result<GeometryObservation> {
    if d.frame != f.frame {
        return Err(Error::value("detection key and frame metadata disagree"));
    }
    let mut flags = Vec::new();
    if !["calibrated", "synthetic_test_value"].contains(&m.calibration_status.as_str()) {
        flags.push("camera_calibration_unverified".into());
    }
    if d.covariance_status != "simulation_known" {
        flags.push(d.covariance_status.clone());
    }
    if !d.valid {
        flags.push(d.invalid_reason.clone());
    }
    if !f.input_available {
        flags.push("input_missing".into());
    }
    if !f.attitude_valid {
        flags.push("attitude_invalid".into());
    }
    let mut out = GeometryObservation {
        detection_key: (d.frame, d.detection_id.clone()),
        timestamp_s: f.timestamp_s,
        pixel_position: [d.x_px, d.y_px],
        pixel_covariance: d.covariance_px2,
        camera_los: None,
        j2000_los: None,
        los_covariance: None,
        tangent_position: None,
        tangent_covariance: None,
        tangent_reference_id: p.reference_id.clone(),
        attitude_valid: f.attitude_valid,
        geometry_valid: false,
        covariance_valid: false,
        covariance_source: d.covariance_status.clone(),
        calibration_version: m.calibration_version.clone(),
        calibration_status: m.calibration_status.clone(),
        quality_flags: vec![],
        normalized_image: None,
        jacobian_los_pixel: None,
        jacobian_tangent_pixel: None,
        tangent_coordinate_type: "gnomonic_ratio".into(),
    };
    if d.valid && f.input_available {
        let result = (|| -> Result<()> {
            let ray = m.pixel_to_camera(out.pixel_position)?;
            out.camera_los = Some(ray.camera_los);
            out.normalized_image = Some(ray.normalized_image);
            if f.attitude_valid {
                let q = f.quaternion_xyzw.as_ref().ok_or_else(|| {
                    Error::value("valid attitude requires a normalized xyzw quaternion")
                })?;
                let (u, j) = camera_to_inertial(ray.camera_los, ray.jacobian_camera_pixel, q)?;
                out.j2000_los = Some(u);
                out.jacobian_los_pixel = Some(j);
                if let Some(r) = d.covariance_px2 {
                    out.los_covariance = Some(propagate_los_covariance(r, j)?);
                }
                let (position, g) = p.project(u)?;
                out.tangent_position = Some(position);
                out.jacobian_tangent_pixel = Some(array(mat(g) * mat(j)));
                out.geometry_valid = true;
                if d.covariance_px2.is_some() {
                    out.tangent_covariance = Some(propagate_tangent_covariance(
                        out.los_covariance.unwrap(),
                        g,
                    )?);
                    out.covariance_valid = true;
                }
            }
            Ok(())
        })();
        if let Err(e) = result {
            if let Some(code) = e.code {
                flags.push(code.to_owned());
            } else {
                return Err(e);
            }
        }
    }
    out.quality_flags = flags;
    out.validate()?;
    Ok(out)
}
