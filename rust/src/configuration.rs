use crate::{
    data_io::{io_error, key, py_repr, read_json},
    error::{Error, Result},
    numerical_policy as n,
};
use serde_json::{json, Map, Value};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};
const SECTIONS: [&str; 10] = [
    "tracking",
    "background",
    "clutter",
    "lifecycle",
    "inference",
    "undetected_population",
    "calibration",
    "geometry",
    "reporting",
    "input_metadata",
];
fn keys(v: &Value) -> BTreeSet<String> {
    v.as_object()
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default()
}

fn list(values: BTreeSet<String>) -> String {
    values.into_iter().collect::<Vec<_>>().join(", ")
}

fn names(s: &str) -> BTreeSet<String> {
    s.split_whitespace().map(str::to_owned).collect()
}

fn merge(base: &Value, extra: &Value) -> Value {
    let mut result = base.as_object().cloned().unwrap_or_default();
    if let Some(extra) = extra.as_object() {
        result.extend(extra.clone());
    }
    Value::Object(result)
}

pub fn numerical_policy() -> Value {
    n::snapshot()
}

pub(crate) fn numeric_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| numeric_equal(a, b))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| numeric_equal(v, w)))
        }
        _ => a == b,
    }
}

pub struct ConfigurationResolver {
    default_path: PathBuf,
}

impl ConfigurationResolver {
    pub fn new(default_path: impl Into<PathBuf>) -> Self {
        Self {
            default_path: default_path.into(),
        }
    }
    fn camera_dir(&self) -> PathBuf {
        self.default_path
            .parent()
            .unwrap_or(Path::new("."))
            .join("cameras")
    }
    pub fn available_camera_profiles(&self) -> Result<Vec<String>> {
        let dir = self.camera_dir();
        if !dir.exists() {
            return Ok(vec![]);
        }
        let mut result = Vec::new();
        for entry in std::fs::read_dir(dir).map_err(io_error)? {
            let path = entry.map_err(io_error)?.path();
            if path.extension().is_some_and(|s| s == "json") {
                result.push(path.file_stem().unwrap().to_string_lossy().to_string());
            }
        }
        result.sort();
        Ok(result)
    }
    fn apply_camera(&self, defaults: &mut Value, name: &Value) -> Result<()> {
        let dir = self.camera_dir();
        let valid = name
            .as_str()
            .filter(|s| !s.is_empty() && !s.contains(['/', '\\']));
        let path = valid.map(|s| dir.join(format!("{s}.json")));
        if !path.as_ref().is_some_and(|p| p.is_file()) {
            return Err(Error::value(format!(
                "unknown camera_profile {}; available: {}",
                py_repr(name),
                self.available_camera_profiles()?.join(", ")
            )));
        }
        let path = path.unwrap();
        let profile = read_json(&path)?;
        if keys(&profile) != names("input_metadata calibration")
            || keys(&profile["input_metadata"]) != names("attitude_direction camera")
        {
            return Err(Error::value(format!(
                concat!(
                    "camera profile must contain exactly ",
                    "input_metadata.{{attitude_direction,camera}} and calibration: {}",
                ),
                path.file_name().unwrap().to_string_lossy()
            )));
        }
        defaults["input_metadata"] = merge(&defaults["input_metadata"], &profile["input_metadata"]);
        defaults["calibration"] = profile["calibration"].clone();
        Ok(())
    }
    pub fn validate_default_completeness(defaults: &Value) -> Result<()> {
        for (section, schema) in [
            (
                "tracking",
                concat!(
                    "survival_probability q_a p_detection measurement_covariance_px2 ",
                    "parameter_provenance max_dt_s survival_reference_dt_s miss_update_mode",
                ),
            ),
            (
                "background",
                concat!(
                    "persistence_probability p_detection measurement_covariance_px2 ",
                    "parameter_provenance enable_view_dormancy dormancy_margin_deg ",
                    "dormancy_sigma enable_retirement retirement_existence_probability ",
                    "retirement_visible_misses retirement_min_pd ",
                    "retirement_mature_mass_fraction",
                ),
            ),
            ("clutter", "expected_count parameter_provenance"),
            (
                "lifecycle",
                concat!(
                    "dormant_after_s reliable_association_probability ",
                    "outside_visibility_probability outside_fov_stop_s ",
                    "candidate_existence_probability confirmed_existence_probability ",
                    "rejection_existence_probability enable_dormancy confirmation_min_frames ",
                    "confirmation_window_frames candidate_deadline_s ",
                    "candidate_low_probability candidate_low_s enable_candidate_pruning ",
                    "policy_version decision_calibration",
                ),
            ),
            (
                "inference",
                concat!(
                    "gate_probability top_k max_nodes max_edges max_component_nodes ",
                    "max_local_hypotheses max_candidates max_search_states max_memory_bytes ",
                    "max_frame_seconds association_top_k preserve_target_associations",
                ),
            ),
            (
                "calibration",
                concat!(
                    "skew_px distortion_model distortion_coefficients max_normalized_radius ",
                    "inverse_tolerance_px max_iterations calibration_version ",
                    "calibration_status camera_axis_rotation_deg",
                ),
            ),
            (
                "input_metadata",
                concat!(
                    "camera time_scale timestamp_unit frame_period_s frame_zero ",
                    "timestamp_zero_s bounds_action column_mapping max_rows ",
                    "attitude_direction",
                ),
            ),
        ] {
            let required = names(schema);
            let actual = keys(&defaults[section]);
            let missing = required
                .difference(&actual)
                .cloned()
                .collect::<BTreeSet<_>>();
            if !missing.is_empty() {
                return Err(Error::value(format!(
                    "default.json must explicitly declare {section}: {}",
                    list(missing)
                )));
            }
            let extra = actual
                .difference(&required)
                .cloned()
                .collect::<BTreeSet<_>>();
            if !extra.is_empty() {
                return Err(Error::value(format!(
                    "default.json contains unused {section} fields: {}",
                    list(extra)
                )));
            }
        }
        Ok(())
    }
    pub fn resolve(&self, raw: &Value) -> Result<Value> {
        if !raw.is_object() {
            return Err(Error::value("configuration must be a JSON object"));
        }
        let mut defaults = read_json(&self.default_path)?;
        let camera = raw
            .get("camera_profile")
            .unwrap_or(key(&defaults, "camera_profile")?)
            .clone();
        self.apply_camera(&mut defaults, &camera)?;
        Self::validate_default_completeness(&defaults)?;
        if raw
            .get("tracker_engine")
            .unwrap_or(&defaults["tracker_engine"])
            != "factorized_joint_ppp"
        {
            return Err(Error::value(
                "only factorized_joint_ppp is supported by this runtime",
            ));
        }
        if raw
            .get("configuration_schema")
            .unwrap_or(&json!("online-parameters-1"))
            != "online-parameters-1"
        {
            return Err(Error::value("unsupported online configuration schema"));
        }
        let bound = raw.get("bootstrap_definition").is_some();
        let mut allowed = keys(&defaults);
        allowed.extend(names(concat!(
            "frames bootstrap_definition profile_binding numerical_policy ",
            "bound_input initial_timestamp_s stars",
        )));
        let unknown = keys(raw)
            .difference(&allowed)
            .cloned()
            .collect::<BTreeSet<_>>();
        if !unknown.is_empty() {
            return Err(Error::value(format!(
                "unsupported online configuration field: {}",
                list(unknown)
            )));
        }
        if bound && raw.get("bootstrap").is_some() {
            return Err(Error::value(
                "configuration cannot be both unbound (bootstrap) and bound (bootstrap_definition)",
            ));
        }
        if !bound {
            let mut derived = keys(raw)
                .intersection(&names("initial_timestamp_s stars"))
                .cloned()
                .collect::<BTreeSet<_>>();
            if raw["clutter"].is_object() && raw["clutter"].get("bounds_px").is_some() {
                derived.insert("clutter.bounds_px".into());
            }
            if !derived.is_empty() {
                return Err(Error::value(format!(
                    "parameters derived from input belong only in a bound snapshot: {}",
                    list(derived)
                )));
            }
        } else {
            let mut missing = names("initial_timestamp_s stars bound_input")
                .difference(&keys(raw))
                .cloned()
                .collect::<BTreeSet<_>>();
            if !raw["clutter"].is_object() || raw["clutter"].get("bounds_px").is_none() {
                missing.insert("clutter.bounds_px".into());
            }
            if !missing.is_empty() {
                return Err(Error::value(format!(
                    "bound snapshot is missing input-derived fields: {}",
                    list(missing)
                )));
            }
        }
        if let Some(bootstrap) = raw.get("bootstrap") {
            if !bootstrap.is_object() {
                return Err(Error::value("configuration object required: bootstrap"));
            }
            let unknown = keys(bootstrap)
                .difference(&keys(&defaults["bootstrap"]))
                .cloned()
                .collect::<BTreeSet<_>>();
            if !unknown.is_empty() {
                return Err(Error::value(format!(
                    "unsupported online bootstrap parameter: {}",
                    list(unknown)
                )));
            }
        }
        let mut result = merge(&defaults, raw);
        if let Some(bootstrap) = raw.get("bootstrap") {
            result["bootstrap"] = merge(&defaults["bootstrap"], bootstrap);
        } else if bound {
            result.as_object_mut().unwrap().remove("bootstrap");
        }
        if raw.get("catalog_path").is_none() {
            let path = key(&defaults, "catalog_path")?
                .as_str()
                .ok_or_else(|| Error::type_error("catalog_path must be a string"))?;
            let path = self
                .default_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(path);
            let rendered = if path.is_absolute() {
                path.to_string_lossy().into_owned()
            } else {
                std::env::current_dir()
                    .map_err(io_error)?
                    .join(path)
                    .to_string_lossy()
                    .into_owned()
            };
            result["catalog_path"] = json!(if cfg!(windows) {
                rendered.replace('/', "\\")
            } else {
                rendered
            });
        }
        let empty = Value::Object(Map::new());
        for section in SECTIONS {
            let supplied = raw.get(section).unwrap_or(&empty);
            if !supplied.is_object() {
                return Err(Error::value(format!(
                    "configuration object required: {section}"
                )));
            }
            let mut allowed = keys(&defaults[section]);
            if bound && section == "clutter" {
                allowed.insert("bounds_px".into());
            }
            let unknown = keys(supplied)
                .difference(&allowed)
                .cloned()
                .collect::<BTreeSet<_>>();
            if !unknown.is_empty() {
                return Err(Error::value(format!(
                    "unsupported online parameter: {section}.{}",
                    list(unknown)
                )));
            }
            result[section] = merge(&defaults[section], supplied);
        }
        let camera = raw
            .get("input_metadata")
            .and_then(|v| v.get("camera"))
            .unwrap_or(&empty);
        if !camera.is_object()
            || !keys(camera).is_subset(&keys(&defaults["input_metadata"]["camera"]))
        {
            return Err(Error::value("unsupported input_metadata.camera parameter"));
        }
        result["input_metadata"]["camera"] = merge(&defaults["input_metadata"]["camera"], camera);
        let angle = &result["geometry"]["max_tangent_angle_deg"];
        if !angle.is_number() && !angle.is_boolean() {
            return Err(Error::type_error(format!(
                "must be real number, not {}",
                if angle.is_null() {
                    "NoneType"
                } else if angle.is_string() {
                    "str"
                } else {
                    "list"
                }
            )));
        }
        if !angle
            .as_f64()
            .is_some_and(|a| a.is_finite() && a > 0. && a < 90.)
        {
            return Err(Error::value(
                "geometry.max_tangent_angle_deg must lie between 0 and 90",
            ));
        }
        let r = &result["reporting"];
        let alpha = r["jump_false_alarm_probability"]
            .as_f64()
            .ok_or_else(|| Error::value("jump_false_alarm_probability must lie in (0,1)"))?;
        crate::report_identity::jump_threshold_squared(alpha)?;
        let provenance = r["jump_measurement_covariance_provenance"]
            .as_str()
            .ok_or_else(|| {
                Error::value("jump_measurement_covariance_provenance must be a string")
            })?;
        if !r["jump_measurement_covariance_px2"].is_null() {
            crate::contracts::covariance_json(&r["jump_measurement_covariance_px2"])?;
            if provenance.trim().is_empty() {
                return Err(Error::value(
                    "jump measurement covariance requires an explicit provenance",
                ));
            }
        }
        if ![json!("compact"), json!("x_legacy")].contains(&r["table_format"]) {
            return Err(Error::value(
                "reporting.table_format must be compact or x_legacy",
            ));
        }
        if ![json!("compact"), json!("full")].contains(&r["detail"])
            || !r["write_frame_json"].is_boolean()
        {
            return Err(Error::value("invalid reporting detail or write_frame_json"));
        }
        if ![json!("every"), json!("interval"), json!("off")].contains(&r["track_frame_png_mode"]) {
            return Err(Error::value(
                "track_frame_png_mode must be every, interval or off",
            ));
        }
        if !r["track_frame_png_interval"]
            .as_i64()
            .is_some_and(|x| x >= 1)
        {
            return Err(Error::value(
                "positive integer track_frame_png_interval required",
            ));
        }
        if !r["track_frame_png_scale"]
            .as_f64()
            .is_some_and(|x| x.is_finite() && x > 0. && x <= 1.)
        {
            return Err(Error::value("track_frame_png_scale must lie in (0,1]"));
        }
        if !result["tracking"]["survival_reference_dt_s"]
            .as_f64()
            .is_some_and(|x| x.is_finite() && x > 0.)
        {
            return Err(Error::value(
                "online tracking requires positive survival_reference_dt_s",
            ));
        }
        result["configuration_schema"] = json!("online-parameters-1");
        let policy = numerical_policy();
        if raw
            .get("numerical_policy")
            .is_some_and(|p| !numeric_equal(p, &policy))
        {
            return Err(Error::value(concat!(
                "saved numerical policy differs from this code version; ",
                "it is not a runtime tuning section",
            )));
        }
        result["numerical_policy"] = policy;
        Ok(result)
    }
}
