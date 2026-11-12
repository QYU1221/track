use crate::{
    camera_model::CameraModel,
    contracts::{Detection, FrameInfo},
    error::{Error, Result},
    finish_management::{self, UpdatedGroup},
    geometry::quaternion_matrix,
    gnomonic::GnomonicPlane,
    input_evidence,
    observation_geometry::observe_detection,
    point_marginals::{self, APPROXIMATION, MODEL_VERSION},
    tracking_background::BackgroundConfig,
    tracking_lifecycle::{LifecycleConfig, ManagementState},
    tracking_model::JointHistory,
    undetected::Population,
};
use serde::Serialize;
use serde_json::{json, Value};
#[derive(Debug, Clone, Serialize)]
pub struct State {
    pub groups: Vec<Vec<JointHistory>>,
    pub populations: Vec<Population>,
    pub management: Vec<ManagementState>,
    pub retired: Vec<Value>,
    pub frame: Option<i64>,
    pub timestamp_s: f64,
    pub version: i64,
    pub dormant_background_ids: Vec<String>,
    pub background_retired_count: usize,
    pub consumed_frame_ids: Vec<i64>,
    pub input_evidence_sha256: String,
}

pub struct Context<'a> {
    pub camera: &'a CameraModel,
    pub lifecycle: &'a LifecycleConfig,
    pub background: &'a BackgroundConfig,
    pub moment_matched: bool,
    pub detail: &'a str,
    pub max_tangent_angle_rad: f64,
    pub known_star_count: usize,
    pub projected_star_count: usize,
}

pub struct BudgetSnapshot {
    pub elapsed_seconds: f64,
    pub work: u64,
    pub peak_sampled_rss: u64,
}

impl State {
    pub fn finish(
        &mut self,
        frame: &FrameInfo,
        detections: &[Detection],
        accepted: &[Detection],
        updated: &[UpdatedGroup],
        populations: &[Population],
        mut audit: Value,
        dormant: &[Vec<JointHistory>],
        ctx: &Context,
        check: &mut impl FnMut(usize) -> Result<BudgetSnapshot>,
    ) -> Result<Value> {
        let version = self
            .version
            .checked_add(1)
            .ok_or_else(|| Error::value("state version overflow"))?;
        let managed = finish_management::prepare(
            updated,
            &finish_management::Context {
                frame,
                lifecycle: ctx.lifecycle,
                background: ctx.background,
                version,
                previous_frame: self.frame,
                previous_management: &self.management,
                detail: ctx.detail,
                moment_matched: ctx.moment_matched,
                dormant,
            },
        )?;
        let histories = updated
            .iter()
            .map(|g| g.histories.clone())
            .collect::<Vec<_>>();
        let keys = accepted
            .iter()
            .map(|d| (d.frame, d.detection_id.clone()))
            .collect::<Vec<_>>();
        let points = point_marginals::project(frame, detections, &keys, &histories, &audit)?;
        let geometry = if frame.inertial_observation_available() {
            let rotation = quaternion_matrix(&frame.quaternion_xyzw.unwrap())?;
            let plane = GnomonicPlane::new(
                &format!("output-{}", frame.frame),
                [rotation[0][2], rotation[1][2], rotation[2][2]],
                ctx.max_tangent_angle_rad,
                None,
            )?;
            accepted
                .iter()
                .map(|d| observe_detection(d, frame, ctx.camera, &plane))
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![]
        };
        let nodes = managed
            .groups
            .iter()
            .map(|g| {
                g.iter()
                    .map(|h| h.tracks.len() + h.backgrounds.len())
                    .max()
                    .unwrap_or(0)
            })
            .sum();
        let budget = check(nodes)?;
        let background_retired_count = self
            .background_retired_count
            .checked_add(managed.background_events.len())
            .ok_or_else(|| Error::value("background retirement count overflow"))?;
        let population_counts = populations
            .iter()
            .map(|p| (p.kind.clone(), json!(p.expected_count)))
            .collect::<serde_json::Map<_, _>>();
        let diagnostics = json!({
                   "Computation_Time_s": budget.elapsed_seconds,
                   "Parent_Hypothesis_Count": self.groups.iter().map(Vec::len).sum::<usize>(),
                   "Retained_Hypothesis_Count": managed.groups.iter().map(Vec::len).sum::<usize>(),
                   "Active_Bernoulli_Component_Count": managed.marginals.len(),
                   "Detection_Count": accepted.len(),
                   "KnownStar_Count": ctx.known_star_count,
                   "Projected_KnownStar_Count": ctx.projected_star_count,
                   "Joint_Group_Count": managed.groups.len(),
                   "Active_Background_Count": managed.active_ids.len(),
                   "Dormant_Background_Count": managed.dormant_ids.len(),
                   "Dormant_Background_Group_Count": dormant.len(),
                   "Woken_Background_Count": if frame.inertial_observation_available() {
                       self.dormant_background_ids
                           .iter()
                           .filter(|id| managed.active_ids.contains(id))
                           .collect::<std::collections::HashSet<_>>()
                           .len()
                   } else {
                       0
                   },
                   "Retired_Background_Count": managed.background_events.len(),
                   "Total_Retired_Background_Count": background_retired_count,
                   "Largest_Joint_Group_Hypotheses": managed.groups.iter().map(Vec::len).max().unwrap_or(0),
                   "Posterior_Normalization_Error": managed
                       .groups
                       .iter()
                       .map(|g| (g.iter().fold(0.,
        |a,
        h| a + h.weight) - 1.).abs())
                       .fold(0.,
        f64::max),
                   "Retired_Candidate_Count": managed.retired.len(),
                   "Total_Retired_Candidate_Count": self.retired.len()+managed.retired.len(),
                   "Exact_Or_Approximate": APPROXIMATION,
                   "Search_States": budget.work,
                   "Undetected_Population_Counts": population_counts
               });
        if ctx.detail == "compact" && audit.get("group_search_summary").is_none() {
            audit["group_search_summary"] = json!({
                "groups": 0,
                "evaluated": 0,
                "retained": 0,
                "omitted_mass_status": "LOCAL_TRUNCATION_AND_FINITE_GATES; FULL_SEQUENCE_ERROR_UNKNOWN"
            });
        }
        let mut inference_audit = json!({
            "approximate": true,
            "model_version": MODEL_VERSION,
            "scope": APPROXIMATION
        });
        let values = audit
            .as_object()
            .ok_or_else(|| Error::value("frame audit must be an object"))?;
        for (k, v) in values {
            if inference_audit.get(k).is_some() {
                return Err(Error::value("duplicate frame audit metadata"));
            }
            inference_audit[k] = v.clone();
        }
        let result = json!({
            "frame": frame.frame,
            "timestamp_s": frame.timestamp_s,
            "dt_s": frame.timestamp_s-self.timestamp_s,
            "status": "APPROXIMATE_FACTORIZED_JOINT_POSTERIOR",
            "raw_detections": detections,
            "points": points,
            "geometry": geometry,
            "track_marginals": managed.marginals,
            "background_marginals": managed.backgrounds,
            "lifecycle_events": managed.events,
            "candidate_retirement_events": managed.retired,
            "diagnostics": diagnostics,
            "inference_mode": "FACTORIZED_JOINT_PPP",
            "background_retirement_events": managed.background_events,
            "background_population_status": {
                "dormant_background_ids": managed.dormant_ids,
                "marginal_scope": concat!(
                    "OBSERVATION_UPDATED_GROUPS_ONLY; ",
                    "DORMANT_CONDITIONAL_STATES_RETAINED_IN_CHECKPOINT",
                ),
                "dormancy_enabled": ctx.background.enable_view_dormancy,
                "retirement_enabled": ctx.background.enable_retirement
            },
            "inference_audit": inference_audit,
            "inference_performance": {
                "peak_sampled_process_rss_bytes": budget.peak_sampled_rss
            },
            "observation_provenance": managed.provenance,
            "state_version": version,
            "identity_scope": concat!(
                "PERSISTENT_LABELS_WITHIN_RETAINED_JOINT_BRANCHES; ",
                "ARCHIVE_REIDENTIFICATION_NOT_IMPLEMENTED",
            )
        });
        let evidence = input_evidence::extend_digest(
            &self.input_evidence_sha256,
            frame.frame,
            &input_evidence::frame_digest(frame, detections)?,
        )?;
        let mut retired = self.retired.clone();
        retired.extend(managed.retired.into_iter().map(|r| json!(r)));
        let mut consumed = self.consumed_frame_ids.clone();
        consumed.push(frame.frame);
        *self = Self {
            groups: managed.groups,
            populations: populations.to_vec(),
            management: managed.management,
            retired,
            frame: Some(frame.frame),
            timestamp_s: frame.timestamp_s,
            version,
            dormant_background_ids: managed.dormant_ids,
            background_retired_count,
            consumed_frame_ids: consumed,
            input_evidence_sha256: evidence,
        };
        Ok(result)
    }
}
