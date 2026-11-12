use crate::{
    camera_model::CameraModel,
    contracts::{KnownStar, ObservationPacket},
    error::{Error, Result},
    geometry::quaternion_matrix,
};
use serde_json::{json, Value};
use std::path::Path;
pub fn bind_profile(
    packet_json: &Value,
    raw: &Value,
    config_path: &Path,
    catalog: Option<Vec<KnownStar>>,
) -> Result<(Value, Vec<KnownStar>)> {
    let packet = ObservationPacket::from_json(packet_json)?;
    if raw["tracker_engine"] != "factorized_joint_ppp"
        || raw["bootstrap"]["schema"] != "population-bootstrap-2"
    {
        return Err(Error::value(
            "only the online population bootstrap is supported",
        ));
    }
    let mut result = raw.clone();
    let options = result.as_object_mut().unwrap().remove("bootstrap").unwrap();
    result["bootstrap_definition"] = options.clone();
    let c = &packet.camera;
    result["clutter"]["bounds_px"] = json!([0, c.width, 0, c.height]);
    result["initial_timestamp_s"] = packet_json["frames"]
        .as_array()
        .unwrap()
        .first()
        .map(|f| f["timestamp_s"].clone())
        .unwrap_or(json!(0.));
    result["bound_input"] = json!({
        "camera": packet_json["camera"],
        "time_scale": packet.time_scale
    });
    let path = Path::new(
        raw["catalog_path"]
            .as_str()
            .ok_or_else(|| Error::value("catalog_path must be a string"))?,
    );
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        config_path.parent().unwrap_or(Path::new(".")).join(path)
    };
    let path = if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .map_err(crate::data_io::io_error)?
            .join(path)
    };
    result["catalog_path"] = json!(if cfg!(windows) {
        path.to_string_lossy().replace('/', "\\")
    } else {
        path.to_string_lossy().into_owned()
    });
    let catalog = match catalog {
        Some(v) => v,
        None => crate::exact_io::public_navstar_catalog(
            &path,
            raw.get("catalog_mag_limit").and_then(Value::as_f64),
        )?,
    };
    let mut calibration = raw["calibration"].clone();
    calibration["camera"] = packet_json["camera"].clone();
    let model = CameraModel::from_json(&calibration)?;
    let mut radius = model.max_normalized_radius;
    if model.distortion_model == "none" {
        let k = model.k();
        let mut corner = 0.0_f64;
        for x in [0., c.width as f64] {
            for y in [0., c.height as f64] {
                let b = (y - k[1][2]) / k[1][1];
                let a = (x - k[0][2] - k[0][1] * b) / k[0][0];
                corner = corner.max((a * a + b * b).sqrt());
            }
        }
        radius = radius.min(corner);
    }
    let cap = 1. / (1. + radius * radius).sqrt();
    let allowance = crate::numerical_policy::PROBABILITY_EPS_MULTIPLIER * f64::EPSILON;
    let directions = catalog
        .iter()
        .map(|s| s.direction_j2000)
        .collect::<Vec<_>>();
    let tree = crate::spatial_index::Tree::new(&directions);
    let shortlist_radius = (2. - 2. * (cap - allowance) + 1024. * f64::EPSILON)
        .max(0.)
        .sqrt();
    let frames = packet
        .frames
        .iter()
        .filter(|f| f.inertial_observation_available())
        .collect::<Vec<_>>();
    let mut visible = vec![false; catalog.len()];
    for frame in &frames {
        let r = quaternion_matrix(&frame.quaternion_xyzw.unwrap())?;
        for i in tree.query(&directions, [r[0][2], r[1][2], r[2][2]], shortlist_radius) {
            let d = catalog[i].direction_j2000;
            if visible[i] || d[0] * r[0][2] + d[1] * r[1][2] + d[2] * r[2][2] < cap - allowance {
                continue;
            }
            let ray = std::array::from_fn(|j| r[0][j] * d[0] + r[1][j] * d[1] + r[2][j] * d[2]);
            match model.camera_to_pixel(ray) {
                Ok(p) => visible[i] = c.contains(p),
                Err(e) if matches!(e.code, Some("camera_singularity" | "distortion_domain")) => {}
                Err(e) => return Err(e),
            }
        }
    }
    let stars = catalog
        .iter()
        .zip(&visible)
        .filter(|(_, selected)| **selected)
        .map(|(s, _)| {
            json!({
                "known_star_id": s.known_star_id,
                "p_detection": options["star_detection_probability"],
                "measurement_covariance_px2": raw["tracking"]["measurement_covariance_px2"]
            })
        })
        .collect::<Vec<_>>();
    result["profile_binding"] = json!({
        "profile_id": raw["profile_id"],
        "public_frames": packet.frames.len(),
        "valid_attitude_frames": frames.len(),
        "catalog_selected_by_magnitude": catalog.len(),
        "catalog_visible_union": stars.len(),
        "scope": options["birth_scope"],
        "birth_representation": concat!(
            "persistent undetected PPP populations; first ",
            "detections compete every valid frame",
        )
    });
    result["stars"] = json!(stars);
    Ok((result, catalog))
}
