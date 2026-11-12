use crate::{
    camera_model::CameraModel,
    checkpoint,
    configuration::numeric_equal,
    contracts::ObservationPacket,
    efficient_config::InferenceConfig,
    error::{Error, Result},
    exact_factors::{ClutterModel, StarSpec},
    factorized_tracker::{Configuration, Tracker},
    input_evidence,
    tracking_background::BackgroundConfig,
    tracking_lifecycle::LifecycleConfig,
    tracking_model::TrackingConfig,
    undetected::PopulationOptions,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub struct Model {
    configuration: Configuration,
    initial_timestamp_s: f64,
    time_scale: String,
    input_frame_digests: BTreeMap<i64, String>,
    binding_parameters: Value,
}

impl Model {
    pub fn input_frame_digests(&self) -> &BTreeMap<i64, String> {
        &self.input_frame_digests
    }
    pub fn binding_digest(&self) -> Result<String> {
        checkpoint::model_binding(&self.binding_parameters, &self.time_scale)
    }
    pub fn tracker(&self) -> Result<Tracker> {
        Tracker::new(self.configuration.clone(), self.initial_timestamp_s)
    }
}

pub fn build_model_with_packet(
    packet_json: &Value,
    packet: &ObservationPacket,
    raw: &Value,
) -> Result<Model> {
    if raw.get("bootstrap_definition").is_some() {
        let expected = json!({
            "camera": packet_json["camera"],
            "time_scale": packet.time_scale
        });
        if !numeric_equal(&raw["bound_input"], &expected) {
            return Err(Error::value(concat!(
                "bound snapshot camera/time scale differs from input; ",
                "use an unbound profile to rebind",
            )));
        }
        if !numeric_equal(
            &raw["clutter"]["bounds_px"],
            &json!([0, packet.camera.width, 0, packet.camera.height]),
        ) {
            return Err(Error::value(
                "bound snapshot observation bounds differ from input camera",
            ));
        }
        if let Some(first) = packet.frames.first() {
            if raw["initial_timestamp_s"].as_f64() != Some(first.timestamp_s) {
                return Err(Error::value(
                    "bound snapshot initial timestamp differs from input",
                ));
            }
        }
    }
    let stars = raw["stars"]
        .as_array()
        .ok_or_else(|| Error::value("stars must be an array"))?
        .iter()
        .map(StarSpec::from_json)
        .collect::<Result<Vec<_>>>()?;
    let clutter_json = json!({
        "expected_count": raw["clutter"]["expected_count"],
        "window": raw["clutter"]["bounds_px"],
        "parameter_provenance": raw["clutter"]["parameter_provenance"]
    });
    let clutter = ClutterModel::from_json(&clutter_json)?;
    let mut camera_json = raw["calibration"].clone();
    camera_json
        .as_object_mut()
        .ok_or_else(|| Error::value("calibration must be an object"))?
        .insert("camera".into(), packet_json["camera"].clone());
    let camera = CameraModel::from_json(&camera_json)?;
    let tracking = TrackingConfig::from_json(&raw["tracking"])?;
    let background = BackgroundConfig::from_json(&raw["background"])?;
    let lifecycle = LifecycleConfig::from_json(&raw["lifecycle"])?;
    let inference = InferenceConfig::from_json(&raw["inference"])?;
    let population = PopulationOptions::from_json(&raw["undetected_population"])?;
    let initial_timestamp_s =
        crate::contracts::scalar(&raw["initial_timestamp_s"], "initial timestamp")?;
    let angle = crate::contracts::scalar(
        &raw["geometry"]["max_tangent_angle_deg"],
        "max tangent angle",
    )?;
    let binding_parameters = json!({
        "camera": camera_json,
        "tracking": raw["tracking"],
        "background": raw["background"],
        "lifecycle": raw["lifecycle"],
        "inference": raw["inference"],
        "clutter": clutter_json,
        "stars": raw["stars"],
        "catalog": packet_json.get("catalog").cloned().unwrap_or(json!([])),
        "options": raw["undetected_population"],
        "initial_timestamp_s": raw["initial_timestamp_s"],
        "angle_degrees": raw["geometry"]["max_tangent_angle_deg"]
    });
    Ok(Model {
        configuration: Configuration {
            camera,
            tracking,
            background,
            lifecycle,
            inference,
            clutter,
            stars,
            catalog: packet.catalog.clone(),
            population,
            max_tangent_angle_rad: angle * (std::f64::consts::PI / 180.),
        },
        initial_timestamp_s,
        time_scale: packet.time_scale.clone(),
        input_frame_digests: input_evidence::packet_frame_digests(&packet)?,
        binding_parameters,
    })
}
