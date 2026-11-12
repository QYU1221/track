use crate::{
    background_lifecycle,
    camera_model::CameraModel,
    conditional_feedback::{self, Binding},
    contracts::{Detection, FrameInfo, KnownStar},
    efficient_config::{InferenceBudgetController, InferenceConfig, ResourceSampler},
    efficient_solver::GraphInferencePolicy,
    error::{Error, Result},
    exact_factors::{self, ClutterModel, StarSpec},
    exact_posterior::infer_factors,
    exact_sources::VisibleStarProjector,
    factorized_groups::{self, ConditionalHistory},
    finish_management::UpdatedGroup,
    frame_commit::{self, State},
    geometry::quaternion_matrix,
    input_evidence, joint_update,
    online_factors::{self, AffineResult, Request},
    tracking_background::{predict_background, BackgroundConfig},
    tracking_lifecycle::LifecycleConfig,
    tracking_model::{predict_track, JointHistory, TrackingConfig},
    undetected::{self, PopulationOptions},
};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
#[derive(Clone)]
pub struct Configuration {
    pub camera: CameraModel,
    pub tracking: TrackingConfig,
    pub background: BackgroundConfig,
    pub lifecycle: LifecycleConfig,
    pub inference: InferenceConfig,
    pub clutter: ClutterModel,
    pub stars: Vec<StarSpec>,
    pub catalog: Vec<KnownStar>,
    pub population: PopulationOptions,
    pub max_tangent_angle_rad: f64,
}

pub struct Tracker {
    pub state: State,
    pub configuration: Configuration,
    pub segment_events: Vec<Value>,
}

fn object_key(kind: &str, v: &Value) -> String {
    format!("{kind}:{}", v)
}

fn objects(h: &JointHistory, c: &Configuration) -> Vec<(String, Request, f64)> {
    let mut out = vec![];
    for t in &h.tracks {
        out.push((
            object_key("T", &json!(t)),
            Request {
                source_id: format!("T:{}", t.label),
                kind: "T".into(),
                state: t.state.clone().map(online_factors::State::Target),
                existence: t.existence,
                plane: t.plane.clone(),
                pd: c.tracking.p_detection,
                noise: c.tracking.measurement_covariance_px2,
            },
            t.log_existence,
        ));
    }
    for b in &h.backgrounds {
        out.push((
            object_key("S_U", &json!(b)),
            Request {
                source_id: format!("S_U:{}", b.background_id),
                kind: "S_U".into(),
                state: b.state.clone().map(online_factors::State::Direction),
                existence: b.existence,
                plane: b.plane.clone(),
                pd: c.background.p_detection,
                noise: c.background.measurement_covariance_px2,
            },
            b.log_existence,
        ));
    }
    out
}

fn clear(h: &JointHistory) -> Result<JointHistory> {
    let mut v = json!(h);
    for k in [
        "target_assignments",
        "star_assignments",
        "background_assignments",
        "clutter_keys",
    ] {
        v[k] = json!([]);
    }
    JointHistory::from_json(&v)
}

fn count(v: &Value) -> Result<i64> {
    v.as_i64()
        .ok_or_else(|| Error::value("joint limit exceeds native signed integer range"))
}

impl Tracker {
    pub fn new(configuration: Configuration, initial_timestamp_s: f64) -> Result<Self> {
        crate::contracts::finite(&[initial_timestamp_s], "initial timestamp")?;
        Ok(Self {
            configuration,
            segment_events: vec![],
            state: State {
                groups: vec![],
                populations: vec![],
                management: vec![],
                retired: vec![],
                frame: None,
                timestamp_s: initial_timestamp_s,
                version: 0,
                dormant_background_ids: vec![],
                background_retired_count: 0,
                consumed_frame_ids: vec![],
                input_evidence_sha256: input_evidence::empty_digest(),
            },
        })
    }
    pub fn step_with_segmentation<S: ResourceSampler, F: FnMut() -> S>(
        &mut self,
        frame: &FrameInfo,
        detections: &[Detection],
        detail: &str,
        mut sampler: F,
    ) -> Result<(Value, Option<Value>)> {
        match self.step(frame, detections, detail, sampler()) {
            Ok(result) => return Ok((result, None)),
            Err(e) if e.kind == "InferenceError" && e.code == Some("PREDICTION_GAP_EXCEEDED") => {}
            Err(e) => return Err(e),
        }
        let event = json!({
            "segment_id": self.segment_events.len()+2,
            "reason": "PREDICTION_GAP_EXCEEDED",
            "previous_frame": self.state.frame,
            "previous_timestamp_s": self.state.timestamp_s,
            "start_frame": frame.frame,
            "start_timestamp_s": frame.timestamp_s,
            "gap_s": frame.timestamp_s-self.state.timestamp_s,
            "max_dt_s": self.configuration.tracking.max_dt_s
        });
        let mut fresh = Self::new(self.configuration.clone(), frame.timestamp_s)?;
        fresh.state.version = self.state.version;
        fresh.state.consumed_frame_ids = self.state.consumed_frame_ids.clone();
        fresh.state.input_evidence_sha256 = self.state.input_evidence_sha256.clone();
        fresh.state.retired = self.state.retired.clone();
        fresh.state.background_retired_count = self.state.background_retired_count;
        fresh.segment_events = self.segment_events.clone();
        fresh.segment_events.push(event.clone());
        let result = fresh.step(frame, detections, detail, sampler())?;
        *self = fresh;
        Ok((result, Some(event)))
    }
    pub fn step<S: ResourceSampler>(
        &mut self,
        frame: &FrameInfo,
        detections: &[Detection],
        detail: &str,
        sampler: S,
    ) -> Result<Value> {
        if !["full", "compact"].contains(&detail) {
            return Err(Error::value("reporting detail must be compact or full"));
        }
        let c = &self.configuration;
        let dt = frame.timestamp_s - self.state.timestamp_s;
        if dt < 0.
            || self
                .state
                .frame
                .is_some_and(|old| frame.frame <= old || dt == 0.)
        {
            return Err(exact_factors::inference(
                "NONMONOTONIC_TIME",
                "strict frame/time ordering; no evidence replay",
            ));
        }
        if dt > c.tracking.max_dt_s {
            return Err(exact_factors::inference(
                "PREDICTION_GAP_EXCEEDED",
                "physical-time prediction budget",
            ));
        }
        let mut detections = detections.to_vec();
        detections.sort_by(|a, b| a.key().cmp(&b.key()));
        if detections.iter().any(|d| d.frame != frame.frame)
            || detections
                .iter()
                .map(Detection::key)
                .collect::<HashSet<_>>()
                .len()
                != detections.len()
        {
            return Err(Error::value(
                "one frame with unique detection keys required",
            ));
        }
        let mut controller = InferenceBudgetController::new(c.inference.clone(), sampler);
        let window = c.clutter.window();
        let accepted = detections
            .iter()
            .filter(|d| {
                frame.inertial_observation_available()
                    && d.valid
                    && window.contains([d.x_px, d.y_px])
            })
            .cloned()
            .collect::<Vec<_>>();
        let populations = undetected::prepare(
            &self.state.populations,
            frame,
            &c.camera,
            &c.tracking,
            &c.background,
            &c.population,
            dt,
            c.max_tangent_angle_rad,
        )?;
        let rotation = if frame.inertial_observation_available() {
            Some(quaternion_matrix(&frame.quaternion_xyzw.unwrap())?)
        } else {
            None
        };
        let active = if let Some(rotation) = rotation {
            background_lifecycle::active_groups(
                &self.state.groups,
                rotation,
                &c.camera,
                window,
                &c.background,
            )?
        } else {
            (0..self.state.groups.len()).collect()
        };
        let mut predicted = vec![];
        let mut dormant = vec![];
        let cutoff =
            i128::from(frame.frame) - i128::from(c.lifecycle.confirmation_window_frames) + 1;
        for (i, group) in self.state.groups.iter().enumerate() {
            let mut hs = vec![];
            for h in group {
                let mut h = h.clone();
                if !active.contains(&i) {
                    h.backgrounds = h
                        .backgrounds
                        .iter()
                        .map(|b| predict_background(b, dt, &c.background).map(|b| b.into_owned()))
                        .collect::<Result<Vec<_>>>()?;
                    h.backgrounds
                        .retain(|b| b.log_existence > f64::NEG_INFINITY);
                    hs.push(clear(&h)?);
                    continue;
                }
                h.tracks = h
                    .tracks
                    .iter()
                    .map(|t| predict_track(t, dt, &c.tracking))
                    .collect::<Result<Vec<_>>>()?;
                h.backgrounds = h
                    .backgrounds
                    .iter()
                    .map(|b| predict_background(b, dt, &c.background).map(|b| b.into_owned()))
                    .collect::<Result<Vec<_>>>()?;
                for keys in h
                    .tracks
                    .iter_mut()
                    .map(|t| &mut t.support_keys)
                    .chain(h.backgrounds.iter_mut().map(|b| &mut b.support_keys))
                {
                    let mut support = keys
                        .iter()
                        .find(|k| i128::from(k.0) < cutoff)
                        .cloned()
                        .into_iter()
                        .collect::<Vec<_>>();
                    support.extend(keys.iter().filter(|k| i128::from(k.0) >= cutoff).cloned());
                    *keys = support;
                }
                hs.push(h);
            }
            if active.contains(&i) {
                predicted.push(hs);
            } else if hs.iter().any(|h| !h.backgrounds.is_empty()) {
                dormant.push(hs);
            }
        }
        let (mut updated, mut audit, mut projected_count) = (
            vec![],
            json!({
                "no_observation": true
            }),
            0,
        );
        let remaining;
        if !frame.inertial_observation_available() {
            remaining = populations;
            for hs in predicted {
                updated.push(UpdatedGroup {
                    histories: hs.iter().map(clear).collect::<Result<Vec<_>>>()?,
                    priors: hs,
                    diagnostics: vec![],
                });
            }
        } else {
            let observation = undetected::observe(
                &populations,
                frame,
                &c.camera,
                window,
                &accepted,
                &c.tracking,
                &c.background,
                rotation,
            )?;
            remaining = observation.remaining;
            let mut requests = vec![];
            let mut request_map = HashMap::new();
            let mut records = vec![];
            let mut seen = HashSet::new();
            for group in &predicted {
                for h in group {
                    for (key, request, log_r) in objects(h, c) {
                        if !seen.insert(key.clone()) {
                            continue;
                        }
                        let rk =
                            json!([request.source_id, request.state, request.plane]).to_string();
                        let index = *request_map.entry(rk).or_insert_with(|| {
                            requests.push(request.clone());
                            requests.len() - 1
                        });
                        records.push((key, request.existence, log_r, index));
                    }
                }
            }
            let values = online_factors::affine_factors_many(
                &requests,
                &online_factors::Context {
                    frame,
                    camera: &c.camera,
                    window,
                    detections: &accepted,
                    gate: Some(c.inference.gate_probability()),
                    rotation,
                    miss_update_mode: &c.tracking.miss_update_mode,
                },
                &mut || controller.check(&[]),
            )?;
            let mut cache: HashMap<String, AffineResult> = HashMap::new();
            for (key, r, log_r, i) in records {
                let mut value = values[i].clone();
                value.factor.existence_prior = r;
                value.factor.log_existence_prior = log_r;
                cache.insert(key, value);
            }
            let sources =
                VisibleStarProjector.sources(&c.stars, &c.catalog, frame, &c.camera, window)?;
            let stars = exact_factors::build_known_star_factors(&sources, &accepted)?;
            projected_count = stars.len();
            let groups = predicted
                .iter()
                .map(|group| {
                    group
                        .iter()
                        .map(|h| ConditionalHistory {
                            history: h.clone(),
                            factors: objects(h, c)
                                .into_iter()
                                .map(|(key, _, _)| cache[&key].factor.clone())
                                .collect(),
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let config = json!(c.inference);
            let limit = count(&config["max_candidates"])?;
            let intensities = observation
                .intensities
                .iter()
                .map(|(_, values)| values.clone())
                .collect::<Vec<_>>();
            let density = c.clutter.expected_count() / window.area()?;
            let prepared = factorized_groups::prepare(
                &groups,
                &factorized_groups::Context {
                    detections: &accepted,
                    stars: &stars,
                    frame: frame.frame,
                    gate: Some(c.inference.gate_probability()),
                    limit,
                    target_pd: c.tracking.p_detection,
                    background_pd: c.background.p_detection,
                    clutter_density: density,
                    intensities: &intensities,
                },
                &mut |nodes| {
                    controller.check(&nodes.map(|n| vec![("nodes", n as u64)]).unwrap_or_default())
                },
            )?;
            let mut summary = json!({
                "groups": 0,
                "evaluated": 0,
                "retained": 0,
                "association_discarded_mass_sum": 0.,
                "association_discarded_mass_max": 0.,
                "evaluated_joint_discarded_mass_sum": 0.,
                "evaluated_joint_discarded_mass_max": 0.,
                "omitted_mass_status": "LOCAL_TRUNCATION_AND_FINITE_GATES; FULL_SEQUENCE_ERROR_UNKNOWN"
            });
            let mut audits = vec![];
            let mut selections = vec![];
            let mut birth_loss = 0.;
            let mut evidence = 0.;
            let births = observation
                .conditionals
                .iter()
                .map(|((key, kind), (state, plane))| joint_update::BirthState {
                    key: key.clone(),
                    kind: kind.clone(),
                    state: state.clone(),
                    plane: plane.clone(),
                })
                .collect::<Vec<_>>();
            for component in prepared.components {
                let ds = component
                    .indices
                    .iter()
                    .map(|i| accepted[*i].clone())
                    .collect::<Vec<_>>();
                let mut policy = GraphInferencePolicy::new(
                    controller,
                    ds.clone(),
                    component.gate_geometry,
                    detail,
                );
                policy.complete = false;
                let result = infer_factors(&component.tables, &mut policy)?;
                let omitted = policy.audit["discarded_mass_within_evaluated_candidates"]
                    .as_f64()
                    .unwrap();
                for (name, increment) in [
                    ("groups", 1),
                    (
                        "evaluated",
                        policy.audit["evaluated_candidates"].as_u64().unwrap(),
                    ),
                    (
                        "retained",
                        policy.audit["retained_candidates"].as_u64().unwrap(),
                    ),
                ] {
                    summary[name] = json!(summary[name].as_u64().unwrap() + increment);
                }
                summary["association_discarded_mass_sum"] =
                    json!(summary["association_discarded_mass_sum"].as_f64().unwrap() + omitted);
                summary["association_discarded_mass_max"] = json!(summary
                    ["association_discarded_mass_max"]
                    .as_f64()
                    .unwrap()
                    .max(omitted));
                if detail == "full" {
                    audits.push(policy.audit.clone());
                }
                evidence += result.log_predictive_evidence;
                let mut bindings = vec![];
                for h in &component.priors {
                    for (key, request, _) in objects(h, c) {
                        let value = &cache[&key];
                        bindings.push(Binding {
                            parent_id: h.hypothesis_id.clone(),
                            source_id: request.source_id,
                            states: value.states.clone(),
                            pd_bar: value.factor.pd_bar,
                        });
                    }
                }
                let feedback = conditional_feedback::restore(
                    &result,
                    &component.priors,
                    &bindings,
                    c.background.retirement_min_pd,
                    &mut || policy.controller.check(&[]),
                )?;
                let keys = ds
                    .iter()
                    .map(|d| (d.frame, d.detection_id.clone()))
                    .collect::<Vec<_>>();
                let expanded = joint_update::expand(
                    &result,
                    &feedback,
                    &joint_update::Context {
                        keys: &keys,
                        intensities: &observation.intensities,
                        births: &births,
                        priors: &component.priors,
                        clutter_density: density,
                        omitted,
                        max_candidates: limit,
                        top_k: count(&config["top_k"])?,
                        preserve: config["preserve_target_associations"].as_bool().unwrap(),
                    },
                )?;
                birth_loss += expanded.birth_loss;
                summary["evaluated_joint_discarded_mass_sum"] = json!(
                    summary["evaluated_joint_discarded_mass_sum"]
                        .as_f64()
                        .unwrap()
                        + expanded.joint_loss
                );
                summary["evaluated_joint_discarded_mass_max"] = json!(summary
                    ["evaluated_joint_discarded_mass_max"]
                    .as_f64()
                    .unwrap()
                    .max(expanded.joint_loss));
                selections.push(expanded.audit);
                updated.push(UpdatedGroup {
                    histories: expanded.histories,
                    priors: component.priors,
                    diagnostics: component.source_diag,
                });
                controller = policy.controller;
            }
            audit = json!({
                            "group_search_summary": summary,
                            "truncation_mass_scope": concat!(
                                "STAGE_CONDITIONAL; ",
                                "GROUP_SUMS_ARE_NOT_GLOBAL_POSTERIOR_ERROR_BOUNDS",
                            ),
                            "target_miss_update_mode": c.tracking.miss_update_mode,
                            "association_top_k": c.inference.selection_limit(),
                            "prior_product_discarded_mass_sum": prepared.prior_loss,
                            "birth_branch_discarded_mass_sum": birth_loss,
                            "joint_selection": selections,
                            "log_predictive_set_density": exact_factors::wire::Float(
                                evidence + prepared.isolated_star_log_miss
                                    - c.clutter.expected_count()
                                    - observation.detected_count,
                            ),
                            "first_detection_intensity": observation.intensities.iter().map(|(k,
            values)|(k,
            values.iter().map(|(kind,
            value)|(kind.clone(),
            json!(value))).collect::<serde_json::Map<_,
            _>>())).collect::<Vec<_>>()
                        });
            if detail == "full" {
                audit["group_audits"] = json!(audits);
            }
        }
        let ctx = frame_commit::Context {
            camera: &c.camera,
            lifecycle: &c.lifecycle,
            background: &c.background,
            moment_matched: c.tracking.miss_update_mode == "moment_matched",
            detail,
            max_tangent_angle_rad: c.max_tangent_angle_rad,
            known_star_count: c.stars.len(),
            projected_star_count: projected_count,
        };
        self.state.finish(
            frame,
            &detections,
            &accepted,
            &updated,
            &remaining,
            audit,
            &dormant,
            &ctx,
            &mut |nodes| {
                controller.check(&[("nodes", nodes as u64)])?;
                Ok(frame_commit::BudgetSnapshot {
                    elapsed_seconds: controller.sampler.elapsed_seconds(),
                    work: controller.work,
                    peak_sampled_rss: controller.peak_sampled_rss,
                })
            },
        )
    }
}
