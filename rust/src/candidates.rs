use crate::{
    contracts::FrameInfo,
    error::{Error, Result},
    exact_factors::probability,
    tracking_lifecycle::{LifecycleConfig, ManagementState},
    tracking_model::{JointHistory, Track},
};
use serde::Serialize;
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

pub fn low_existence_start(
    old: &ManagementState,
    first: Option<i64>,
    existence: f64,
    frame: &FrameInfo,
    previous_frame: Option<i64>,
    cfg: &LifecycleConfig,
) -> Option<f64> {
    if first.is_none() || existence >= cfg.candidate_low_probability {
        return None;
    }
    if previous_frame.and_then(|n| n.checked_add(1)) == Some(frame.frame)
        && old.low_since_time.is_some()
    {
        old.low_since_time
    } else {
        Some(frame.timestamp_s)
    }
}

pub fn retirement_reason(
    r: &ManagementState,
    frame: &FrameInfo,
    cfg: &LifecycleConfig,
) -> Option<&'static str> {
    if !cfg.enable_candidate_pruning || r.first_real_support_frame.is_none() {
        None
    } else if r
        .outside_since_time
        .is_some_and(|t| frame.timestamp_s - t >= cfg.outside_fov_stop_s)
    {
        Some("LEFT_FIELD_OF_VIEW")
    } else if r.last_existence_probability.is_some()
        && r.low_since_time
            .is_some_and(|t| frame.timestamp_s - t >= cfg.candidate_low_s)
    {
        Some("CONSECUTIVE_LOW_EXISTENCE")
    } else if r.overdue_unconfirmed {
        Some("CANDIDATE_DEADLINE")
    } else {
        None
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RetiredCandidate {
    pub label: String,
    pub frame: i64,
    pub timestamp_s: f64,
    pub existence_probability: f64,
    pub management: ManagementState,
    pub conditional_branches: Vec<(String, f64, Track)>,
    pub state_version: i64,
    pub reason: &'static str,
    pub approximation: &'static str,
}

pub type Retirement<'a> = (
    Cow<'a, [JointHistory]>,
    Cow<'a, [ManagementState]>,
    Vec<RetiredCandidate>,
);
pub fn retire_candidates<'a>(
    histories: &'a [JointHistory],
    management: &'a [ManagementState],
    frame: &FrameInfo,
    cfg: &LifecycleConfig,
    version: i64,
) -> Result<Retirement<'a>> {
    let reasons: HashMap<_, _> = management
        .iter()
        .map(|r| (r.label.as_str(), retirement_reason(r, frame, cfg)))
        .collect();
    let eligible: Vec<_> = management
        .iter()
        .filter(|r| reasons[r.label.as_str()].is_some())
        .collect();
    if eligible.is_empty() {
        return Ok((Cow::Borrowed(histories), Cow::Borrowed(management), vec![]));
    }
    let labels: HashSet<_> = eligible.iter().map(|r| r.label.as_str()).collect();
    let mut retired = vec![];
    for r in eligible {
        let existence = probability(
            r.last_existence_probability.unwrap_or(f64::NAN),
            "retired candidate existence",
        )?;
        if version < 1 || frame.frame < 0 {
            return Err(Error::value("invalid retirement version/frame"));
        }
        let branches = histories
            .iter()
            .flat_map(|h| {
                h.tracks
                    .iter()
                    .filter(|t| t.label == r.label)
                    .map(|t| (h.hypothesis_id.clone(), h.weight, t.clone()))
            })
            .collect();
        retired.push(RetiredCandidate {
            label: r.label.clone(),
            frame: frame.frame,
            timestamp_s: frame.timestamp_s,
            existence_probability: existence,
            management: r.clone(),
            conditional_branches: branches,
            state_version: version,
            reason: reasons[r.label.as_str()].unwrap(),
            approximation: "MARGINAL_PROJECTION; FUTURE_SOURCE_REMOVED; NOT_PHYSICAL_DEATH",
        });
    }
    let mut feedback = histories.to_vec();
    for h in &mut feedback {
        h.tracks.retain(|t| !labels.contains(t.label.as_str()));
    }
    Ok((
        Cow::Owned(feedback),
        Cow::Owned(
            management
                .iter()
                .filter(|r| !labels.contains(r.label.as_str()))
                .cloned()
                .collect(),
        ),
        retired,
    ))
}
