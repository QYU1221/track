use crate::{
    camera_model::CameraModel,
    contracts::{Detection, FrameInfo, ObservationPacket},
    error::Result,
    geometry::quaternion_matrix,
    gnomonic::GnomonicPlane,
    numerical_policy::DISPLAY_CLASS_TIE_ATOL as TIE,
};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
fn array(v: &Value) -> &[Value] {
    v.as_array().map(Vec::as_slice).unwrap_or(&[])
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.)
}

fn text(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}

fn compact(v: &Value) -> String {
    match v {
        Value::Number(n) if n.is_f64() => {
            crate::input_evidence::python_float(n.as_f64().unwrap()).unwrap()
        }
        Value::Array(a) => format!("[{}]", a.iter().map(compact).collect::<Vec<_>>().join(",")),
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, v)| format!("{}:{}", serde_json::to_string(k).unwrap(), compact(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        _ => serde_json::to_string(v).unwrap(),
    }
}

fn event_json(v: &Value) -> String {
    let keys = [
        "frame",
        "timestamp_s",
        "label",
        "kind",
        "before",
        "after",
        "reason",
        "state_version",
        "probability_changed_by_management",
        "existence_probability",
    ];
    let mut parts = vec![];
    for key in keys {
        if let Some(value) = v.get(key) {
            parts.push(format!(
                "{}:{}",
                serde_json::to_string(key).unwrap(),
                compact(value)
            ));
        }
    }
    for (key, value) in v.as_object().unwrap() {
        if !keys.contains(&key.as_str()) {
            parts.push(format!(
                "{}:{}",
                serde_json::to_string(key).unwrap(),
                compact(value)
            ));
        }
    }
    format!("{{{}}}", parts.join(","))
}

#[derive(Debug, Clone)]
pub struct DisplayTrack {
    pub id: usize,
    pub label: String,
    pub rows: Vec<Value>,
    pub last: Value,
    pub formal_frame: Option<i64>,
    pub measured_frames: usize,
    pub last_measured_frame: Option<i64>,
    pub exit_frame: Option<i64>,
    pub exit_reason: Value,
    pub final_formal: bool,
}

pub struct Records {
    pub packet: std::sync::Arc<ObservationPacket>,
    pub configuration: Value,
    pub camera: CameraModel,
    pub frames: Vec<Value>,
    pub frame_count: usize,
    pub frame_status_counts: HashMap<String, usize>,
    pub committed_frames: std::collections::BTreeSet<i64>,
    pub history: Option<std::rc::Rc<crate::report_store::ReportStore>>,
    pub points: Vec<((i64, String), Value)>,
    pub tracks: Vec<DisplayTrack>,
    pub events: Vec<Value>,
    label_ids: HashMap<String, usize>,
    reliable: f64,
}

const DISPLAY_CLUTTER_MIN_POSTERIOR: f64 = 0.85;
fn classify(point: &Value) -> Result<(String, [Option<f64>; 3])> {
    let keys = [
        "P_Existing_Target",
        "P_Birth_Target",
        "P_Known_Star",
        "P_Unknown_Star",
        "P_Clutter",
    ];
    if keys.iter().any(|k| point[*k].is_null()) {
        return Ok(("未知".into(), [None; 3]));
    }
    let p = [
        num(&point[keys[0]]) + num(&point[keys[1]]),
        num(&point[keys[2]]) + num(&point[keys[3]]),
        num(&point[keys[4]]),
    ];
    let maximum = p.into_iter().fold(f64::NEG_INFINITY, f64::max);
    let winners = (0..3)
        .filter(|i| (p[*i] - maximum).abs() <= TIE)
        .collect::<Vec<_>>();
    let class = if winners.len() != 1
        || (winners[0] == 0 && p[0] < 0.85)
        || (winners[0] == 2 && p[2] < DISPLAY_CLUTTER_MIN_POSTERIOR)
    {
        "未知".to_owned()
    } else if winners[0] == 1 {
        crate::star_display::classify_star(point)?[0]
            .as_str()
            .unwrap()
            .to_owned()
    } else {
        ["目标", "恒星", "杂波"][winners[0]].to_owned()
    };
    Ok((class, p.map(Some)))
}

impl Records {
    fn jump_covariance(
        &self,
        detection: &Detection,
        frame: &FrameInfo,
        geometry: &Value,
    ) -> Result<(Value, String)> {
        let reporting = &self.configuration["reporting"];
        let (covariance, source) = if let Some(covariance) = detection.covariance_px2 {
            let source = format!("DETECTION:{}", detection.covariance_status);
            if !geometry["los_covariance"].is_null() {
                return Ok((geometry["los_covariance"].clone(), source));
            }
            (covariance, source)
        } else if !reporting["jump_measurement_covariance_px2"].is_null() {
            (
                crate::contracts::covariance_json(&reporting["jump_measurement_covariance_px2"])?,
                format!(
                    "REPORTING:{}",
                    reporting["jump_measurement_covariance_provenance"]
                        .as_str()
                        .unwrap_or("")
                ),
            )
        } else {
            (
                crate::contracts::covariance_json(
                    &self.configuration["tracking"]["measurement_covariance_px2"],
                )?,
                "TRACKING_MEASUREMENT_COVARIANCE:UNCALIBRATED_FALLBACK".into(),
            )
        };
        let ray = self
            .camera
            .pixel_to_camera([detection.x_px, detection.y_px])?;
        let rotation = quaternion_matrix(&frame.quaternion_xyzw.ok_or_else(|| {
            crate::error::Error::value("jump covariance requires valid attitude")
        })?)?;
        let jacobian = crate::linalg::array(
            crate::linalg::mat(rotation) * crate::linalg::mat(ray.jacobian_camera_pixel),
        );
        Ok((
            json!(crate::measurement_covariance::propagate_los_covariance(
                covariance, jacobian
            )?),
            source,
        ))
    }
    pub fn new(
        packet: impl Into<std::sync::Arc<ObservationPacket>>,
        configuration: Value,
    ) -> Result<Self> {
        let packet = packet.into();
        let mut camera = configuration["calibration"].clone();
        camera["camera"] = json!(packet.camera);
        Ok(Self {
            camera: CameraModel::from_json(&camera)?,
            reliable: num(&configuration["lifecycle"]["reliable_association_probability"]),
            packet,
            configuration,
            frames: vec![],
            frame_count: 0,
            frame_status_counts: HashMap::new(),
            committed_frames: std::collections::BTreeSet::new(),
            history: None,
            points: vec![],
            tracks: vec![],
            events: vec![],
            label_ids: HashMap::new(),
        })
    }
    fn track_index(&mut self, label: &str) -> usize {
        let next = self.label_ids.len() + 1;
        let id = *self.label_ids.entry(label.to_owned()).or_insert(next);
        if let Some(i) = self.tracks.iter().position(|t| t.label == label) {
            return i;
        }
        self.tracks.push(DisplayTrack {
            id,
            label: label.into(),
            rows: vec![],
            last: json!({}),
            formal_frame: None,
            measured_frames: 0,
            last_measured_frame: None,
            exit_frame: None,
            exit_reason: Value::Null,
            final_formal: false,
        });
        self.tracks.len() - 1
    }
    pub fn detection_row(
        &self,
        d: &Detection,
        f: &FrameInfo,
        p: &Value,
        g: &Value,
        label: Option<&str>,
    ) -> Result<Value> {
        let (class, probabilities) = classify(p)?;
        let star = if probabilities[1]
            .is_some_and(|s| s > probabilities[0].unwrap().max(probabilities[2].unwrap()) + TIE)
        {
            crate::star_display::classify_star(p)?
        } else {
            Value::Null
        };
        let valid = p["Geometry_Valid"] == true;
        let skip = if valid {
            ""
        } else if !d.valid {
            &d.invalid_reason
        } else if !f.attitude_valid {
            "invalid_attitude"
        } else {
            "not_in_observation_model"
        };
        let q = f
            .quaternion_xyzw
            .map(|q| json!(q))
            .unwrap_or(json!([null, null, null, null]));
        let los = g
            .get("camera_los")
            .filter(|v| !v.is_null())
            .cloned()
            .unwrap_or(json!([null, null, null]));
        let reason = if !star.is_null() {
            star[1].clone()
        } else if class != "未知" {
            json!("unique_maximum_source_posterior")
        } else if probabilities[0].is_some_and(|p| {
            p == probabilities[1]
                .unwrap()
                .max(probabilities[2].unwrap())
                .max(p)
                && p < 0.85
        }) {
            json!("target_posterior_below_display_threshold")
        } else if probabilities[2]
            .is_some_and(|p| p - probabilities[0].unwrap().max(probabilities[1].unwrap()) > TIE)
        {
            json!("clutter_posterior_below_display_threshold")
        } else {
            json!("unavailable_or_tied_source_posterior")
        };
        Ok(json!({
                        "Frame": d.frame,
                        "Detection_ID": d.detection_id,
                        "Attitude_Valid": i32::from(f.attitude_valid),
                        "q1": q[0],
                        "q2": q[1],
                        "q3": q[2],
                        "q4_scalar": q[3],
                        "Image_X_px": d.x_px,
                        "Image_Y_px": d.y_px,
                        "Brightness": d.features.iter().rev().find(|(k,
        _)|k=="Flux").map(|(_,
        v)|v),
                        "LOS_X": los[0],
                        "LOS_Y": los[1],
                        "LOS_Z": los[2],
                        "Class_Group": class,
                        "Track_ID": null,
                        "Track_Label": label,
                        "Class_At_Frame": class,
                        "Final_Class": class,
                        "Geometry_Valid": i32::from(g["geometry_valid"]==true),
                        "Classification_Valid": i32::from(valid),
                        "Classification_Skip_Reason": skip,
                        "Target_Posterior": probabilities[0],
                        "Star_Posterior": probabilities[1],
                        "Clutter_Posterior": probabilities[2],
                        "Existence_Probability": null,
                        "Decision_Reason": reason,
                        "Timestamp_s": f.timestamp_s,
                        "Online_Canonical_ID": null,
                        "Identity_Status": if valid{"UNRESOLVED"}else{"UNAVAILABLE"},
                        "Identity_Posterior": null,
                        "Known_Star_Probability": p["P_Known_Star"],
                        "Unknown_Star_Probability": p["P_Unknown_Star"],
                        "Star_Status": if class == "恒星" {
                            "CONFIRMED"
                        } else if class == "恒星候选" {
                            "CANDIDATE"
                        } else {
                            "UNRESOLVED"
                        },
                        "Star_Source_ID": star[2],
                        "Star_Evidence_Support": star[3],
                        "Unknown_Star_Confirmation_Support": compact(
                            p.get("Unknown_Star_Confirmation_Support")
                                .unwrap_or(&json!([])),
                        ),
                        "Existing_Target_Probability": p["P_Existing_Target"],
                        "Birth_Target_Probability": p["P_Birth_Target"],
                        "Association_Marginals": compact(
                            p.get("Association_Marginals").unwrap_or(&json!([]))
                        ),
                        "Association_Retained_Mass": p["Association_Retained_Mass"],
                        "Association_PreTruncation_Support": compact(&p["Association_PreTruncation_Support"]),
                        "Association_Support_Scope": p["Association_Support_Scope"],
                        "Model_Status": p["Model_Status"],
                        "Inference_Mode": p["Inference_Mode"],
                        "Approximation_Status": p["Approximation_Status"]
                    }))
    }
    fn prediction(&self, m: &Value, f: &FrameInfo) -> (Value, Value) {
        let project = (|| -> Result<(Value, Value)> {
            if m["State_Mean"].is_null() || m["Tangent_Plane"].is_null() {
                return Ok((Value::Null, Value::Null));
            }
            let p = GnomonicPlane::from_json(&m["Tangent_Plane"])?;
            let (u, _) = p.inverse([num(&m["State_Mean"][0]), num(&m["State_Mean"][1])])?;
            let mut pixel = Value::Null;
            if f.inertial_observation_available() {
                let r = quaternion_matrix(&f.quaternion_xyzw.unwrap())?;
                let ray = std::array::from_fn(|j| r[0][j] * u[0] + r[1][j] * u[1] + r[2][j] * u[2]);
                let p = self.camera.camera_to_pixel(ray)?;
                if self.packet.camera.contains(p) {
                    pixel = json!(p);
                }
            }
            Ok((pixel, json!(u)))
        })();
        project.unwrap_or((Value::Null, Value::Null))
    }
    pub fn consume(&mut self, result: &Value) -> Result<()> {
        self.frames.clear();
        self.points.clear();
        if self.history.is_some() {
            for track in &mut self.tracks {
                if track.rows.len() > 1 {
                    track.rows.drain(..track.rows.len() - 1);
                }
            }
        }
        self.events.clear();
        let number = result["frame"].as_i64().unwrap();
        let frame = self
            .packet
            .frames
            .iter()
            .find(|f| f.frame == number)
            .unwrap()
            .clone();
        let marginals = array(&result["track_marginals"]);
        for m in marginals {
            let label = text(&m["Track_Label"]);
            let next = self.label_ids.len() + 1;
            self.label_ids.entry(label.into()).or_insert(next);
            if m["Track_Maturity"] == "CONFIRMED"
                || num(&m["Probability_Exists_And_Ever_Measured"]) >= self.reliable
            {
                self.track_index(label);
            }
        }
        let detections = self
            .packet
            .detections
            .iter()
            .filter(|d| d.frame == number)
            .cloned()
            .collect::<Vec<_>>();
        let mut measured: HashMap<String, (Detection, Value, f64, Value)> = HashMap::new();
        let mut truncated = HashSet::new();
        for d in &detections {
            let empty = json!({});
            let point = array(&result["points"])
                .iter()
                .find(|p| p["Detection_ID"] == d.detection_id)
                .unwrap_or(&empty);
            let geometry = array(&result["geometry"])
                .iter()
                .find(|g| g["detection_key"] == json!([number, d.detection_id]))
                .unwrap_or(&empty);
            let mut reliable = vec![];
            let mass = point["Association_Retained_Mass"].as_f64().unwrap_or(1.);
            for a in array(&point["Association_Marginals"]) {
                let source = text(&a[0]);
                let p = num(&a[1]);
                let Some(label) = source.strip_prefix("T:") else {
                    continue;
                };
                let support = if point["Association_PreTruncation_Support"].is_null() {
                    p * mass
                } else {
                    p.min(
                        array(&point["Association_PreTruncation_Support"])
                            .iter()
                            .rev()
                            .find(|x| x[0] == source)
                            .map(|x| num(&x[1]))
                            .unwrap_or(0.),
                    )
                };
                if p >= self.reliable && support < self.reliable {
                    truncated.insert(label.to_owned());
                }
                if support >= self.reliable && marginals.iter().any(|m| m["Track_Label"] == label) {
                    reliable.push((label, p));
                }
            }
            let label = if reliable.len() == 1 {
                Some(reliable[0].0)
            } else {
                None
            };
            let mut item = self.detection_row(d, &frame, point, geometry, label)?;
            if let Some(label) = label {
                let i = self.track_index(label);
                let m = marginals
                    .iter()
                    .find(|m| m["Track_Label"] == label)
                    .unwrap();
                item["Track_ID"] = json!(self.tracks[i].id);
                item["Existence_Probability"] = m["Existence_Probability"].clone();
                item["Online_Canonical_ID"] = m["Canonical_Track_ID"].clone();
                item["Identity_Status"] = json!("ONLINE_ASSOCIATION_ONLY");
                item["Identity_Posterior"] = json!(reliable[0].1);
                measured.insert(
                    label.into(),
                    (d.clone(), item.clone(), reliable[0].1, geometry.clone()),
                );
            }
            if let Some(old) = self
                .points
                .iter_mut()
                .find(|(key, _)| key.0 == d.frame && key.1 == d.detection_id)
            {
                old.1 = item;
            } else {
                self.points.push(((d.frame, d.detection_id.clone()), item));
            }
        }
        for m in marginals {
            let label = text(&m["Track_Label"]);
            let Some(i) = self.tracks.iter().position(|t| t.label == label) else {
                continue;
            };
            let found = measured.get(label);
            let (pixel, direction, covariance, covariance_source) =
                if let Some((d, _, _, geometry)) = found {
                    let direction = geometry["j2000_los"].clone();
                    let (covariance, source) = if direction.is_null() {
                        (Value::Null, "UNAVAILABLE".into())
                    } else {
                        self.jump_covariance(d, &frame, geometry)?
                    };
                    (json!([d.x_px, d.y_px]), direction, covariance, source)
                } else {
                    let (pixel, direction) = self.prediction(m, &frame);
                    (pixel, direction, Value::Null, "PREDICTED_NOT_TESTED".into())
                };
            let track = &mut self.tracks[i];
            track.last = m.clone();
            if m["Track_Maturity"] == "CONFIRMED" && track.formal_frame.is_none() {
                track.formal_frame = Some(number);
            }
            let mut source = "predicted";
            if found.is_some() {
                source = "measured";
                if track
                    .last_measured_frame
                    .is_some_and(|old| number > old + 1)
                {
                    source = "recovered";
                    self.events.push(json!({
                        "frame": number,
                        "timestamp": frame.timestamp_s,
                        "track_label": label,
                        "event_type": "track_recovered",
                        "event_reason": "RELIABLE_SAME_LABEL_ASSOCIATION_AFTER_GAP",
                        "details": format!(
                            concat!(
                                "{{\"previous_measured_frame\":{},",
                                "\"scope\":\"ONLINE_SAME_LABEL_RECOVERY; ",
                                "NOT_ARCHIVE_REIDENTIFICATION\"}}",
                            ),
                            track.last_measured_frame.unwrap()
                        )
                    }));
                }
                track.last_measured_frame = Some(number);
                track.measured_frames += 1;
            }
            let item = found
                .map(|(_, item, _, _)| item.clone())
                .unwrap_or(json!({}));
            let mut row = json!({
                            "track_id": track.id,
                            "track_label": label,
                            "frame": number,
                            "timestamp_s": frame.timestamp_s,
                            "x": pixel[0],
                            "y": pixel[1],
                            "source": source,
                            "status": m["Track_Lifecycle"],
                            "classification_at_frame": if m["Track_Maturity"]=="CONFIRMED"{"目标"}else{"未确认"},
                            "target_posterior": item["Target_Posterior"],
                            "star_posterior": item["Star_Posterior"],
                            "clutter_posterior": item["Clutter_Posterior"],
                            "existence_probability": m["Existence_Probability"],
                            "association_quality": found.map(|(_,
            _,
            p,
            _)|p),
                            "geometry_valid": !pixel.is_null(),
                            "decision_reason": if truncated.contains(label) && found.is_none() {
                                json!("ASSOCIATION_UNCERTAIN_AFTER_TRUNCATION")
                            } else {
                                m["Management_Reason"].clone()
                            },
                            "measured_frames": track.measured_frames,
                            "canonical_track_id": m["Canonical_Track_ID"],
                            "posterior_update_kind": m["Measured_Or_Predicted"],
                            "miss_association_probability": m["Miss_Association_Probability"],
                            "_direction": direction,
                            "_los_covariance": covariance,
                            "jump_covariance_source": covariance_source
                        });
            if !direction.is_null() {
                row["ra_deg"] = json!(num(&direction[1]).atan2(num(&direction[0])).to_degrees());
                row["dec_deg"] = json!(num(&direction[2]).clamp(-1., 1.).asin().to_degrees());
            }
            track.rows.push(row);
        }
        for e in array(&result["lifecycle_events"]) {
            if ["POSTERIOR_TRUNCATION", "LIFECYCLE"].contains(&text(&e["kind"])) {
                if let Some(t) = self.tracks.iter_mut().find(|t| e["label"] == t.label) {
                    t.last["Track_Lifecycle"] = e["after"].clone();
                    t.last["Management_Reason"] = e["reason"].clone();
                    if ["NO_RETAINED_SUPPORT", "TERMINATED"].contains(&text(&e["after"])) {
                        t.exit_frame = Some(number);
                        t.exit_reason = e["reason"].clone();
                    }
                }
            }
            self.events.push(json!({
                "frame": e["frame"],
                "timestamp": e["timestamp_s"],
                "track_label": e["label"],
                "event_type": e["kind"],
                "event_reason": e["reason"],
                "details": event_json(e)
            }));
        }
        for e in array(&result["candidate_retirement_events"]) {
            self.events.push(json!({
                "frame": number,
                "timestamp": frame.timestamp_s,
                "track_label": e["label"],
                "event_type": "track_retired",
                "event_reason": e["reason"],
                "value": e["existence_probability"]
            }));
            if let Some(t) = self.tracks.iter_mut().find(|t| e["label"] == t.label) {
                t.last["Track_Lifecycle"] = json!("RETIRED");
                t.last["Management_Reason"] = e["reason"].clone();
                t.exit_frame = Some(number);
                t.exit_reason = e["reason"].clone();
            }
        }
        let row = self.frame_row(result, &frame);
        *self
            .frame_status_counts
            .entry(row["frame_status"].as_str().unwrap().into())
            .or_default() += 1;
        self.frame_count += 1;
        self.committed_frames.insert(number);
        self.frames.push(row);
        Ok(())
    }
    fn frame_row(&self, result: &Value, f: &FrameInfo) -> Value {
        let ms = array(&result["track_marginals"])
            .iter()
            .filter(|m| self.tracks.iter().any(|t| m["Track_Label"] == t.label))
            .collect::<Vec<_>>();
        let ps = self
            .points
            .iter()
            .filter(|((n, _), _)| *n == f.frame)
            .map(|(_, p)| p)
            .collect::<Vec<_>>();
        let count = |status: &str| ms.iter().filter(|m| m["Track_Lifecycle"] == status).count();
        let confirmed = ms
            .iter()
            .filter(|m| m["Track_Maturity"] == "CONFIRMED")
            .count();
        let q = f
            .quaternion_xyzw
            .map(|q| json!(q))
            .unwrap_or(json!([null, null, null, null]));
        let mut row = json!({
            "frame": f.frame,
            "timestamp_s": f.timestamp_s,
            "frame_status": if !f.input_available {
                "missing_input_frame"
            } else if !f.attitude_valid {
                "skipped_invalid_attitude"
            } else {
                "processed"
            },
            "n_stars": ps.iter().filter(|p|p["Class_Group"]=="恒星").count(),
            "n_candidates": ps.iter().filter(|p|p["Class_Group"]=="目标").count(),
            "n_targets": confirmed,
            "n_active_tracks": count("ACTIVE"),
            "n_lost_tracks": count("LOST"),
            "n_live_tracks": ms.len(),
            "n_unknown_class_tracks": ms.len()-confirmed,
            "n_exit_memory": count("DORMANT"),
            "n_reconnect_events": self.tracks
                .iter()
                .filter(|t| {
                    t.rows.last().is_some_and(|r| {
                        r["frame"] == f.frame && r["source"] == "recovered"
                    })
                })
                .count(),
            "attitude_source": "public_frame_metadata",
            "attitude_ok": i32::from(f.attitude_valid),
            "q_w": q[3],
            "q_x": q[0],
            "q_y": q[1],
            "q_z": q[2],
            "catalog_available": i32::from(!self.packet.catalog.is_empty()),
            "catalog_projection_available": i32::from(f.inertial_observation_available()),
            "catalog_projection_count": result["diagnostics"]["Projected_KnownStar_Count"],
            "input_detections": ps.len(),
            "classified_detections": ps
                .iter()
                .map(|p| p["Classification_Valid"].as_i64().unwrap())
                .sum::<i64>(),
            "model_target_components": array(&result["track_marginals"]).len(),
            "time_ms": num(&result["diagnostics"]["Computation_Time_s"])*1000.,
            "inference_mode": result["inference_mode"],
            "model_status": result["status"]
        });
        if let Some([x, y, z, w]) = f.quaternion_xyzw {
            row["roll_deg"] = json!((2. * (w * x + y * z))
                .atan2(1. - 2. * (x * x + y * y))
                .to_degrees());
            row["pitch_deg"] = json!((2. * (w * y - z * x)).clamp(-1., 1.).asin().to_degrees());
            row["yaw_deg"] = json!((2. * (w * z + x * y))
                .atan2(1. - 2. * (y * y + z * z))
                .to_degrees());
        }
        row
    }
    pub fn finalize_point(&self, p: &mut Value) {
        let formal = self
            .tracks
            .iter()
            .any(|t| p["Track_Label"] == t.label && t.final_formal);
        p["Final_Class"] = if formal {
            json!("目标")
        } else if p["Class_Group"] == "目标" {
            json!("未确认")
        } else {
            p["Class_Group"].clone()
        };
    }
    pub fn finalize_event(&self, e: &mut Value) {
        let track = self.tracks.iter().find(|t| e["track_label"] == t.label);
        e["track_id"] = json!(track.map(|t| t.id));
        if e["event_type"] == "track_retired" {
            e["event_type"] = json!(if track.is_some_and(|t| t.final_formal) {
                "track_stopped"
            } else {
                "track_pruned"
            });
        }
    }
    pub fn track_rows(&self, track: &DisplayTrack) -> Result<Vec<Value>> {
        match &self.history {
            Some(history) => history.track_rows(track.id),
            None => Ok(track.rows.clone()),
        }
    }
    pub fn finish_records_stream(
        &mut self,
        emit: &mut dyn FnMut(Value) -> Result<()>,
    ) -> Result<Vec<Value>> {
        let mut summaries = vec![];
        for t in &mut self.tracks {
            if let Some(history) = &self.history {
                t.rows = history.track_rows(t.id)?;
            }
            let last_view = t.rows.iter().rposition(|r| !r["x"].is_null());
            if let Some(last) = last_view {
                t.rows.truncate(last + 1);
            }
            let observed = t
                .rows
                .iter()
                .filter(|r| r["source"] != "predicted")
                .cloned()
                .collect::<Vec<_>>();
            let frames = observed
                .iter()
                .map(|r| r["frame"].as_i64().unwrap())
                .collect::<Vec<_>>();
            let coverage = frames
                .first()
                .map(|first| observed.len() as f64 / (frames.last().unwrap() - first + 1) as f64);
            let gap = frames
                .windows(2)
                .map(|w| w[1] - w[0] - 1)
                .max()
                .unwrap_or(0);
            let alpha = self.configuration["reporting"]["jump_false_alarm_probability"]
                .as_f64()
                .ok_or_else(|| {
                    crate::error::Error::value("jump_false_alarm_probability must lie in (0,1)")
                })?;
            let identity = crate::report_identity::target_identity_diagnostics(
                t.formal_frame.is_some(),
                &observed,
                alpha,
            )?;
            let jumps = &identity["jumps"];
            t.final_formal = identity["confirmed"] == true;
            for row in t.rows.iter_mut().filter(|row| row["source"] != "predicted") {
                if let Some(check) = jumps["checks"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|check| check["frame"] == row["frame"])
                {
                    row["jump_mahalanobis_squared"] = check["mahalanobis_squared"].clone();
                    row["jump_threshold_squared"] = jumps["threshold_squared"].clone();
                    row["jump_detected"] = json!(i32::from(check["jump"] == true));
                }
            }
            for row in &mut t.rows {
                row["observation_coverage"] = json!(coverage);
                row["max_observation_gap"] = json!(gap);
            }
            for row in &t.rows {
                emit(row.clone())?;
            }
            if let Some(history) = &self.history {
                history.finalize_track(t.id, &t.rows)?;
            }
            let last = &t.last;
            let mut row = json!({
                "track_id": t.id,
                "track_label": t.label,
                "status": last["Track_Lifecycle"],
                "maturity_status": last["Track_Maturity"],
                "formal_confirmed": i32::from(t.final_formal),
                "formal_confirmation_frame": t.formal_frame,
                "final_classification": if t.final_formal{"目标"}else{"未确认"},
                "existence_probability": last["Existence_Probability"],
                "target_confirmed_ever": i32::from(t.formal_frame.is_some()),
                "positions": t.rows.len(),
                "first_frame": t.rows.first().map(|r|&r["frame"]),
                "last_frame": t.rows.last().map(|r|&r["frame"]),
                "measured_frames": observed.len(),
                "observation_coverage": coverage,
                "max_observation_gap": gap,
                "formal_target": i32::from(t.final_formal&&last["Track_Lifecycle"]=="ACTIVE"),
                "formal_exit_frame": if t.formal_frame.is_some(){json!(t.exit_frame)}else{Value::Null},
                "formal_exit_reason": if t.formal_frame.is_some() {
                    t.exit_reason.clone()
                } else {
                    Value::Null
                },
                "terminal_reason": t.exit_reason,
                "last_decision_reason": last["Management_Reason"],
                "canonical_track_id": last["Canonical_Track_ID"],
                "final_identity_reason": identity["reason"],
                "jump_check_status": jumps["status"],
                "jump_check_count": jumps["check_count"],
                "jump_count": jumps["jump_count"],
                "jump_fraction": jumps["jump_fraction"],
                "jump_covariance_sources": compact(&jumps["covariance_sources"])
            });
            if let Some(last) = observed.last() {
                for key in [
                    "target_posterior",
                    "star_posterior",
                    "clutter_posterior",
                    "association_quality",
                ] {
                    row[key] = last[key].clone();
                }
            }
            summaries.push(row);
            if self.history.is_some() && t.rows.len() > 1 {
                t.rows.drain(..t.rows.len() - 1);
            }
        }
        Ok(summaries)
    }
}

#[cfg(test)]
mod clutter_display_tests {
    use super::*;

    #[test]
    fn clutter_winner_needs_minimum_posterior() {
        for (target, star, clutter, expected) in [
            (0.33, 0.32, 0.35, "未知"),
            (0.1, 0.050001, 0.849999, "未知"),
            (0.1, 0.05, 0.85, "杂波"),
            (0.05, 0.05, 0.90, "杂波"),
        ] {
            let point = json!({
                "P_Existing_Target": target,
                "P_Birth_Target": 0.0,
                "P_Known_Star": star,
                "P_Unknown_Star": 0.0,
                "P_Clutter": clutter
            });
            let (label, probabilities) = classify(&point).unwrap();
            assert_eq!(label, expected);
            assert_eq!(probabilities[2], Some(clutter));
        }
    }
}
