use crate::{
    background_lifecycle::{retire_backgrounds, RetirementEvent},
    candidates::{retire_candidates, RetiredCandidate},
    contracts::FrameInfo,
    error::Result,
    tracking_background::BackgroundConfig,
    tracking_lifecycle::{self as life, LifecycleConfig, ManagementState},
    tracking_marginals::{project_background, project_track_marginals, ConfirmationWindow},
    tracking_model::JointHistory,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashSet;
pub struct UpdatedGroup {
    pub histories: Vec<JointHistory>,
    pub priors: Vec<JointHistory>,
    pub diagnostics: Vec<Value>,
}

pub struct Context<'a> {
    pub frame: &'a FrameInfo,
    pub lifecycle: &'a LifecycleConfig,
    pub background: &'a BackgroundConfig,
    pub version: i64,
    pub previous_frame: Option<i64>,
    pub previous_management: &'a [ManagementState],
    pub detail: &'a str,
    pub moment_matched: bool,
    pub dormant: &'a [Vec<JointHistory>],
}

#[derive(Debug, Serialize)]
pub struct Managed {
    pub groups: Vec<Vec<JointHistory>>,
    pub marginals: Vec<Value>,
    pub backgrounds: Vec<Value>,
    pub management: Vec<ManagementState>,
    pub retired: Vec<RetiredCandidate>,
    pub events: Vec<Value>,
    pub provenance: Vec<Value>,
    pub background_events: Vec<RetirementEvent>,
    pub dormant_ids: Vec<String>,
    pub active_ids: Vec<String>,
}

pub fn prepare(updated: &[UpdatedGroup], ctx: &Context) -> Result<Managed> {
    let mut out = Managed {
        groups: vec![],
        marginals: vec![],
        backgrounds: vec![],
        management: vec![],
        retired: vec![],
        events: vec![],
        provenance: vec![],
        background_events: vec![],
        dormant_ids: vec![],
        active_ids: vec![],
    };
    let confirmation = ConfirmationWindow::new(
        ctx.lifecycle.confirmation_window_frames,
        ctx.lifecycle.confirmation_min_frames,
    )?;
    for group in updated {
        let hs = &group.histories;
        let mut rows = project_track_marginals(
            hs,
            ctx.frame.frame,
            ctx.frame.timestamp_s,
            confirmation,
            ctx.detail,
        )?;
        if ctx.frame.inertial_observation_available() && ctx.moment_matched {
            for row in &mut rows {
                if row["Measured_Or_Predicted"] == "Predicted_Only" {
                    row["Measured_Or_Predicted"] = json!("Miss_Conditioned_Gaussian");
                }
            }
        }
        let old = rows
            .iter()
            .filter_map(|row| {
                ctx.previous_management
                    .iter()
                    .rev()
                    .find(|m| row["Track_Label"] == m.label)
                    .cloned()
            })
            .collect::<Vec<_>>();
        let (records, events) = life::update(
            &old,
            &rows,
            &group.diagnostics,
            &group.priors,
            ctx.frame,
            ctx.lifecycle,
            ctx.version,
            ctx.previous_frame,
        )?;
        for (row, record) in rows.iter_mut().zip(&records) {
            row["Track_Lifecycle"] = json!(record.lifecycle);
            row["Track_Maturity"] = json!(record.maturity);
            row["Canonical_Track_ID"] = json!(life::canonical_id(&record.label));
            row["State_Version"] = json!(ctx.version);
            row["Management_Reason"] = json!(record.reason);
        }
        let (feedback, records, removed) =
            retire_candidates(hs, &records, ctx.frame, ctx.lifecycle, ctx.version)?;
        let mut feedback = feedback.into_owned();
        for h in &mut feedback {
            h.tracks.retain(|t| t.log_existence > f64::NEG_INFINITY);
            h.backgrounds
                .retain(|b| b.log_existence > f64::NEG_INFINITY);
        }
        let live: HashSet<_> = feedback
            .iter()
            .flat_map(|h| h.tracks.iter().map(|t| t.label.as_str()))
            .collect();
        let records = records
            .iter()
            .filter(|r| live.contains(r.label.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let (feedback, bg_removed) = retire_backgrounds(&feedback, ctx.frame, ctx.background)?;
        for row in &mut rows {
            row["Retired_After_Observation_Update"] =
                json!(removed.iter().any(|r| row["Track_Label"] == r.label));
        }
        if feedback
            .iter()
            .any(|h| !h.tracks.is_empty() || !h.backgrounds.is_empty())
        {
            out.groups.push(feedback.into_owned());
        }
        out.marginals.extend(rows);
        out.backgrounds.extend(project_background(
            hs,
            ctx.frame.frame,
            ctx.frame.timestamp_s,
            &group.diagnostics,
            ctx.detail,
        )?);
        out.management.extend(records);
        out.retired.extend(removed);
        out.events.extend(events.into_iter().map(|e| json!(e)));
        out.background_events.extend(bg_removed);
        out.provenance.push(json!(hs
            .iter()
            .map(|h| {
                let v = json!(h);
                json!({
                    "hypothesis_id": h.hypothesis_id,
                    "weight": h.weight,
                    "targets": h.target_assignments,
                    "unknown_stars": h.background_assignments,
                    "known_stars": v["star_assignments"],
                    "clutter": v["clutter_keys"]
                })
            })
            .collect::<Vec<_>>()));
    }
    out.groups.extend_from_slice(ctx.dormant);
    let mut dormant = ctx
        .dormant
        .iter()
        .flat_map(|g| {
            g.iter()
                .flat_map(|h| h.backgrounds.iter().map(|b| b.background_id.clone()))
        })
        .collect::<Vec<_>>();
    dormant.sort();
    dormant.dedup();
    out.dormant_ids = dormant;
    let mut active = updated
        .iter()
        .flat_map(|g| {
            g.histories
                .iter()
                .flat_map(|h| h.backgrounds.iter().map(|b| b.background_id.clone()))
        })
        .collect::<Vec<_>>();
    active.sort();
    active.dedup();
    out.active_ids = active;
    for record in ctx.previous_management {
        if out.management.iter().any(|r| r.label == record.label)
            || out.retired.iter().any(|r| r.label == record.label)
        {
            continue;
        }
        let predicted = updated
            .iter()
            .flat_map(|g| g.priors.iter())
            .filter(|h| h.weight > 0.)
            .flat_map(|h| h.tracks.iter())
            .filter(|t| t.label == record.label)
            .collect::<Vec<_>>();
        let zero = !predicted.is_empty()
            && predicted
                .iter()
                .all(|t| t.log_existence == f64::NEG_INFINITY);
        if zero
            && out
                .events
                .iter()
                .any(|e| e["label"] == record.label && e["after"] == "TERMINATED")
        {
            continue;
        }
        out.events.push(json!({
            "frame": ctx.frame.frame,
            "timestamp_s": ctx.frame.timestamp_s,
            "label": record.label,
            "kind": if zero{"LIFECYCLE"}else{"POSTERIOR_TRUNCATION"},
            "before": record.lifecycle,
            "after": if zero{"TERMINATED"}else{"NO_RETAINED_SUPPORT"},
            "reason": if zero {
                "ZERO_PHYSICAL_POSTERIOR"
            } else {
                "LOCAL_JOINT_BRANCH_TRUNCATION; NOT_PHYSICAL_DEATH"
            },
            "state_version": ctx.version,
            "existence_probability": record.last_existence_probability
        }));
    }
    Ok(out)
}
