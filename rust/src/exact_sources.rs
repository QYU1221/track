use crate::{
    camera_model::CameraModel,
    contracts::{FrameInfo, KnownStar},
    error::{Error, Result},
    exact_factors::{inference, AffinePixelModel, ProjectedKnownStar, StarSpec},
    exact_numerics::ObservationWindow,
    geometry::quaternion_matrix,
    gnomonic::GnomonicPlane,
    linalg::{array, mat},
};
use nalgebra::{SMatrix, Vector2, Vector3};
use std::collections::{HashMap, HashSet};

pub fn project_direction_pixel_with_rotation(
    mean: [f64; 2],
    plane: &GnomonicPlane,
    frame: &FrameInfo,
    camera: &CameraModel,
    rotation: Option<[[f64; 3]; 3]>,
) -> Result<([f64; 2], [[f64; 2]; 2])> {
    let (u, j) = plane.inverse(mean)?;
    let r = mat(if let Some(r) = rotation {
        r
    } else {
        let q = frame
            .quaternion_xyzw
            .ok_or_else(|| Error::value("quaternion must be finite"))?;
        quaternion_matrix(&q)?
    });
    let b = r.transpose() * Vector3::from(u);
    let pixel = camera.camera_to_pixel(b.into())?;
    let axes = mat(camera.image_to_camera_rotation())
        .fixed_view::<2, 2>(0, 0)
        .transpose();
    let image = axes * Vector2::new(b[0] / b[2], b[1] / b[2]);
    let (_, jd) = camera.distort(image.into())?;
    let jb = axes
        * SMatrix::<f64, 2, 3>::from_row_slice(&[
            1. / b[2],
            0.,
            -b[0] / b[2].powi(2),
            0.,
            1. / b[2],
            -b[1] / b[2].powi(2),
        ]);
    let h = mat(camera.k()).fixed_view::<2, 2>(0, 0) * mat(jd) * jb * r.transpose() * mat(j);
    Ok((pixel, array(h)))
}

#[derive(Default)]
pub struct VisibleStarProjector;
impl VisibleStarProjector {
    pub fn sources(
        &self,
        stars: &[StarSpec],
        catalog: &[KnownStar],
        frame: &FrameInfo,
        camera: &CameraModel,
        window: &ObservationWindow,
    ) -> Result<Vec<ProjectedKnownStar>> {
        if !frame.inertial_observation_available() {
            return Err(inference(
                "OBSERVATION_UNAVAILABLE",
                "valid input and attitude required",
            ));
        }
        let lookup: HashMap<_, _> = catalog.iter().map(|s| (&s.known_star_id, s)).collect();
        let mut ordered: Vec<_> = stars.iter().collect();
        ordered.sort_by(|a, b| a.known_star_id.cmp(&b.known_star_id));
        if ordered
            .iter()
            .map(|s| &s.known_star_id)
            .collect::<HashSet<_>>()
            .len()
            != ordered.len()
        {
            return Err(Error::value("duplicate target/star source ID"));
        }
        if ordered
            .iter()
            .any(|s| !lookup.contains_key(&s.known_star_id))
        {
            return Err(inference(
                "CATALOG_ID_MISSING",
                "star prior must reference the public catalog",
            ));
        }
        if ordered.is_empty() {
            return Ok(vec![]);
        }
        let r = mat(quaternion_matrix(&frame.quaternion_xyzw.unwrap())?);
        let mut out = Vec::new();
        for spec in ordered {
            let b = r.transpose() * Vector3::from(lookup[&spec.known_star_id].direction_j2000);
            let mean = match camera.camera_to_pixel(b.into()) {
                Ok(p) => p,
                Err(e) if matches!(e.code, Some("camera_singularity" | "distortion_domain")) => {
                    continue
                }
                Err(e)
                    if camera.distortion_model == "none"
                        && e.message == "projected pixel must be finite" =>
                {
                    continue
                }
                Err(e) => return Err(e),
            };
            if window.contains(mean) {
                out.push(ProjectedKnownStar {
                    source_id: format!("S_K:{}", spec.known_star_id),
                    model: AffinePixelModel {
                        mean_px: mean,
                        noise_px2: spec.measurement_covariance_px2,
                        p_detection_pre_window: spec.p_detection,
                        window: window.clone(),
                        integration_error: 0.,
                    },
                });
            }
        }
        Ok(out)
    }
}
