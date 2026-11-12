use crate::{
    contracts as c,
    error::{Error, Result},
    exact_factors::{number, probability},
    exact_posterior::fsum,
    tracking_model::JointHistory,
};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;

pub fn canonical_id(label: &str) -> String {
    format!("C:{}:{label}", label.chars().count())
}

#[derive(Debug, Clone, Serialize)]
pub struct LifecycleConfig {
    pub(crate) dormant_after_s: f64,
    pub(crate) reliable_association_probability: f64,
    pub(crate) outside_visibility_probability: f64,
    pub(crate) outside_fov_stop_s: f64,
    pub(crate) candidate_existence_probability: f64,
    pub(crate) confirmed_existence_probability: f64,
    pub(crate) rejection_existence_probability: f64,
    pub(crate) enable_dormancy: bool,
    pub(crate) confirmation_min_frames: i64,
    pub(crate) confirmation_window_frames: i64,
    pub(crate) candidate_deadline_s: f64,
    pub(crate) candidate_low_probability: f64,
    pub(crate) candidate_low_s: f64,
    pub(crate) enable_candidate_pruning: bool,
    pub(crate) policy_version: String,
    pub(crate) decision_calibration: String,
}

impl LifecycleConfig {
    pub fn from_json(v: &Value) -> Result<Self> {
        c::fields(
            v,
            "LifecycleConfig",
            &[
                "dormant_after_s",
                "reliable_association_probability",
                "outside_visibility_probability",
                "outside_fov_stop_s",
                "candidate_existence_probability",
                "confirmed_existence_probability",
                "rejection_existence_probability",
                "enable_dormancy",
                "confirmation_min_frames",
                "confirmation_window_frames",
                "candidate_deadline_s",
                "candidate_low_probability",
                "candidate_low_s",
                "enable_candidate_pruning",
                "policy_version",
                "decision_calibration",
            ],
            &[],
        )?;
        let p = |key: &str| probability(c::scalar(&v[key], key)?, key);
        let reliable_association_probability = p("reliable_association_probability")?;
        let outside_visibility_probability = p("outside_visibility_probability")?;
        let candidate_existence_probability = p("candidate_existence_probability")?;
        let confirmed_existence_probability = p("confirmed_existence_probability")?;
        let rejection_existence_probability = p("rejection_existence_probability")?;
        let dormant_after_s = c::scalar(&v["dormant_after_s"], "dormancy seconds")?;
        if dormant_after_s <= 0. {
            return Err(Error::value("positive dormancy seconds required"));
        }
        let outside_fov_stop_s = c::scalar(&v["outside_fov_stop_s"], "field-of-view exit seconds")?;
        if outside_fov_stop_s <= 0. {
            return Err(Error::value("positive field-of-view exit seconds required"));
        }
        if !(rejection_existence_probability < candidate_existence_probability
            && candidate_existence_probability <= confirmed_existence_probability)
        {
            return Err(Error::value("ordered maturity policy required"));
        }
        let enable_dormancy = v["enable_dormancy"]
            .as_bool()
            .ok_or_else(|| Error::value("boolean dormancy switch required"))?;
        let candidate_low_probability = probability(
            c::scalar(&v["candidate_low_probability"], "candidate low probability")?,
            "candidate low probability",
        )?;
        if !(0. < candidate_low_probability && candidate_low_probability < 1.) {
            return Err(Error::value(
                "candidate low probability must lie strictly between zero and one",
            ));
        }
        let positive = |key: &str| -> Result<f64> {
            let x = c::scalar(&v[key], key)?;
            if x <= 0. {
                return Err(Error::value(format!("positive seconds required: {key}")));
            }
            Ok(x)
        };
        let candidate_deadline_s = positive("candidate_deadline_s")?;
        let candidate_low_s = positive("candidate_low_s")?;
        let count = |key: &str| {
            v[key]
                .as_i64()
                .filter(|x| *x >= 1)
                .ok_or_else(|| Error::value(format!("positive frame count required: {key}")))
        };
        let confirmation_min_frames = count("confirmation_min_frames")?;
        let confirmation_window_frames = count("confirmation_window_frames")?;
        if confirmation_window_frames < confirmation_min_frames {
            return Err(Error::value(
                "confirmation window must include the required support frames",
            ));
        }
        let enable_candidate_pruning = v["enable_candidate_pruning"]
            .as_bool()
            .ok_or_else(|| Error::value("boolean candidate pruning switch required"))?;
        let policy_version = v["policy_version"].as_str().unwrap_or("");
        let decision_calibration = v["decision_calibration"].as_str().unwrap_or("");
        if policy_version.is_empty() || decision_calibration != "UNCALIBRATED_DECISION_COST" {
            return Err(Error::value(
                "explicit policy version and current uncalibrated cost status required",
            ));
        }
        Ok(Self {
            dormant_after_s,
            reliable_association_probability,
            outside_visibility_probability,
            outside_fov_stop_s,
            candidate_existence_probability,
            confirmed_existence_probability,
            rejection_existence_probability,
            enable_dormancy,
            confirmation_min_frames,
            confirmation_window_frames,
            candidate_deadline_s,
            candidate_low_probability,
            candidate_low_s,
            enable_candidate_pruning,
            policy_version: policy_version.into(),
            decision_calibration: decision_calibration.into(),
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ManagementState {
    pub(crate) label: String,
    pub(crate) lifecycle: String,
    pub(crate) maturity: String,
    pub(crate) first_state_time: f64,
    pub(crate) last_reliable_observation_time: Option<f64>,
    pub(crate) last_effective_observation_time: Option<f64>,
    pub(crate) visibility_probability: Option<f64>,
    pub(crate) association_probability: f64,
    pub(crate) ever_confirmed: bool,
    pub(crate) epoch: i64,
    pub(crate) version: i64,
    pub(crate) reason: String,
    pub(crate) first_real_support_frame: Option<i64>,
    pub(crate) first_real_support_time: Option<f64>,
    pub(crate) last_evaluated_frame: Option<i64>,
    pub(crate) last_existence_probability: Option<f64>,
    pub(crate) low_since_time: Option<f64>,
    pub(crate) confirmation_probability: f64,
    pub(crate) overdue_unconfirmed: bool,
    pub(crate) outside_since_time: Option<f64>,
}

impl ManagementState {
    pub fn new(label: &str, time: f64) -> Result<Self> {
        Self::from_json(&serde_json::json!({
            "label": label,
            "first_state_time": time
        }))
    }
    pub fn from_json(v: &Value) -> Result<Self> {
        c::fields(
            v,
            "ManagementState",
            &["label"],
            &[
                "lifecycle",
                "maturity",
                "first_state_time",
                "last_reliable_observation_time",
                "last_effective_observation_time",
                "visibility_probability",
                "association_probability",
                "ever_confirmed",
                "epoch",
                "version",
                "reason",
                "first_real_support_frame",
                "first_real_support_time",
                "last_evaluated_frame",
                "last_existence_probability",
                "low_since_time",
                "confirmation_probability",
                "overdue_unconfirmed",
                "outside_since_time",
            ],
        )?;
        let s = |key: &str, default: &str| -> Result<String> {
            v.get(key)
                .map(|x| {
                    x.as_str()
                        .map(String::from)
                        .ok_or_else(|| Error::value(format!("{key} must be a string")))
                })
                .unwrap_or_else(|| Ok(default.into()))
        };
        let lifecycle = s("lifecycle", "ACTIVE")?;
        let maturity = s("maturity", "PROVISIONAL")?;
        if !["ACTIVE", "LOST", "DORMANT", "TERMINATED"].contains(&lifecycle.as_str()) {
            return Err(Error::value(format!(
                "'{lifecycle}' is not a valid TrackLifecycle"
            )));
        }
        if !["PROVISIONAL", "CANDIDATE", "CONFIRMED", "REJECTED"].contains(&maturity.as_str()) {
            return Err(Error::value(format!(
                "'{maturity}' is not a valid TrackMaturity"
            )));
        }
        let n = |key: &str, name: &str| c::scalar(v.get(key).unwrap_or(&Value::from(0.)), name);
        let opt = |key: &str, name: &str| {
            v.get(key)
                .filter(|v| !v.is_null())
                .map(|v| c::scalar(v, name))
                .transpose()
        };
        let first_state_time = n("first_state_time", "management timestamp")?;
        let last_reliable_observation_time =
            opt("last_reliable_observation_time", "observation time")?;
        let last_effective_observation_time =
            opt("last_effective_observation_time", "observation time")?;
        let outside_since_time = opt("outside_since_time", "observation time")?;
        let first_real_support_time = opt("first_real_support_time", "observation time")?;
        let low_since_time = opt("low_since_time", "observation time")?;
        let visibility_probability = opt("visibility_probability", "visibility")?
            .map(|x| probability(x, "visibility"))
            .transpose()?;
        let association_probability =
            probability(n("association_probability", "association")?, "association")?;
        let confirmation_probability = probability(
            n("confirmation_probability", "confirmation support")?,
            "confirmation support",
        )?;
        let last_existence_probability = opt("last_existence_probability", "last existence")?
            .map(|x| probability(x, "last existence"))
            .transpose()?;
        let frame = |key: &str| {
            v.get(key)
                .filter(|v| !v.is_null())
                .map(|x| {
                    x.as_i64()
                        .filter(|x| *x >= 0)
                        .ok_or_else(|| Error::value("invalid management frame"))
                })
                .transpose()
        };
        let first_real_support_frame = frame("first_real_support_frame")?;
        let last_evaluated_frame = frame("last_evaluated_frame")?;
        let label = s("label", "")?;
        let integer = |key: &str| {
            v.get(key)
                .map(|x| {
                    x.as_i64()
                        .ok_or_else(|| Error::value("invalid management identity/version"))
                })
                .unwrap_or(Ok(0))
        };
        let epoch = integer("epoch")?;
        let version = integer("version")?;
        if label.is_empty() || epoch < 0 || version < 0 {
            return Err(Error::value("invalid management identity/version"));
        }
        let boolean = |key: &str| {
            v.get(key)
                .map(|x| {
                    x.as_bool()
                        .ok_or_else(|| Error::value(format!("{key} must be boolean")))
                })
                .unwrap_or(Ok(false))
        };
        Ok(Self {
            label,
            lifecycle,
            maturity,
            first_state_time,
            last_reliable_observation_time,
            last_effective_observation_time,
            visibility_probability,
            association_probability,
            ever_confirmed: boolean("ever_confirmed")?,
            epoch,
            version,
            reason: s("reason", "INITIAL_PRIOR")?,
            first_real_support_frame,
            first_real_support_time,
            last_evaluated_frame,
            last_existence_probability,
            low_since_time,
            confirmation_probability,
            overdue_unconfirmed: boolean("overdue_unconfirmed")?,
            outside_since_time,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TrackEvent {
    pub frame: i64,
    pub timestamp_s: f64,
    pub label: String,
    pub kind: &'static str,
    pub before: String,
    pub after: String,
    pub reason: String,
    pub state_version: i64,
    pub probability_changed_by_management: bool,
}

pub fn decide_maturity(
    old: &ManagementState,
    existence: f64,
    measured: f64,
    cfg: &LifecycleConfig,
    confirmation: Option<f64>,
) -> &'static str {
    if old.ever_confirmed {
        "CONFIRMED"
    } else if existence <= cfg.rejection_existence_probability {
        "REJECTED"
    } else if confirmation.is_some_and(|x| x >= cfg.confirmed_existence_probability) {
        "CONFIRMED"
    } else if measured >= cfg.candidate_existence_probability {
        "CANDIDATE"
    } else {
        "PROVISIONAL"
    }
}

fn required<'a>(v: &'a Value, key: &str) -> Result<&'a Value> {
    v.get(key).ok_or_else(|| Error {
        kind: "KeyError",
        code: None,
        message: format!("'{key}'"),
    })
}

pub fn update(
    previous: &[ManagementState],
    marginals: &[Value],
    diagnostics: &[Value],
    predicted: &[JointHistory],
    frame: &c::FrameInfo,
    cfg: &LifecycleConfig,
    version: i64,
    previous_frame: Option<i64>,
) -> Result<(Vec<ManagementState>, Vec<TrackEvent>)> {
    let records: HashMap<_, _> = previous.iter().map(|r| (r.label.as_str(), r)).collect();
    let mut index = HashMap::new();
    for d in diagnostics {
        let source = required(d, "source_id")?
            .as_str()
            .ok_or_else(|| Error::value("source_id must be a string"))?;
        index
            .entry((d.get("parent_id").and_then(Value::as_str), source))
            .or_insert(d);
    }
    let mut result = vec![];
    let mut events = vec![];
    for row in marginals {
        let label = required(row, "Track_Label")?
            .as_str()
            .ok_or_else(|| Error::value("Track_Label must be a string"))?;
        let default = ManagementState::new(label, frame.timestamp_s)?;
        let old = records.get(label).copied().unwrap_or(&default);
        let pairs = required(row, "Association_Marginals")?
            .as_array()
            .ok_or_else(|| Error::value("Association_Marginals must be an array"))?;
        let assoc = fsum(
            pairs
                .iter()
                .map(|p| number(&p[1]))
                .collect::<Result<Vec<_>>>()?,
        )?
        .min(1.);
        let source = format!("T:{label}");
        let mut parts = vec![];
        for h in predicted {
            if let (Some(t), Some(d)) = (
                h.tracks.iter().find(|t| t.label == label),
                index.get(&(Some(h.hypothesis_id.as_str()), source.as_str())),
            ) {
                if t.existence > 0. {
                    parts.push((
                        h.weight * t.existence,
                        d.get("visibility_mass_cubature")
                            .filter(|v| !v.is_null())
                            .map(number)
                            .transpose()?,
                        number(required(d, "pd_bar")?)?,
                    ));
                }
            }
        }
        let den = fsum(parts.iter().map(|p| p.0))?;
        let vden = fsum(parts.iter().filter(|p| p.1.is_some()).map(|p| p.0))?;
        let visibility = if vden > 0. {
            Some((fsum(parts.iter().filter_map(|p| p.1.map(|v| p.0 * v)))? / vden).clamp(0., 1.))
        } else {
            None
        };
        let effective = den > 0. && parts.iter().any(|p| p.0 > 0. && p.2 > 0.);
        let reliable = assoc >= cfg.reliable_association_probability && assoc > 0.;
        let now = frame.timestamp_s;
        let last = if reliable {
            Some(now)
        } else {
            old.last_reliable_observation_time
        };
        let age = now - last.unwrap_or(old.first_state_time);
        let r = number(required(row, "Existence_Probability")?)?.clamp(0., 1.);
        let outside_visibility =
            visibility.is_some_and(|v| v <= cfg.outside_visibility_probability);
        let (life, reason) = if r == 0.
            && row
                .get("Log_Existence_Probability")
                .map(number)
                .transpose()?
                .unwrap_or(f64::NEG_INFINITY)
                == f64::NEG_INFINITY
        {
            (
                "TERMINATED",
                "ZERO_PHYSICAL_POSTERIOR; software termination",
            )
        } else if reliable {
            ("ACTIVE", "JOINT_ASSOCIATION_POSTERIOR_SUPPORT")
        } else if cfg.enable_dormancy && (age >= cfg.dormant_after_s || outside_visibility) {
            (
                "DORMANT",
                if outside_visibility {
                    "OUTSIDE_SAMPLED_FOV_SUPPORT"
                } else {
                    "OBSERVATION_GAP_MANAGEMENT_POLICY"
                },
            )
        } else {
            (
                "LOST",
                if frame.inertial_observation_available() {
                    "INSUFFICIENT_ASSOCIATION_SUPPORT"
                } else {
                    "OBSERVATION_UNAVAILABLE"
                },
            )
        };
        let confirmation = row
            .get("Confirmation_Support_Probability")
            .map(number)
            .transpose()?
            .unwrap_or(0.);
        let maturity = decide_maturity(
            old,
            r,
            number(required(row, "Probability_Exists_And_Ever_Measured")?)?,
            cfg,
            Some(confirmation),
        );
        let first = match old.first_real_support_frame {
            Some(n) => Some(n),
            None => row
                .get("First_Real_Support_Frame")
                .filter(|x| !x.is_null())
                .map(|x| {
                    x.as_i64()
                        .filter(|n| *n >= 0)
                        .ok_or_else(|| Error::value("invalid management frame"))
                })
                .transpose()?,
        };
        let first_time = old.first_real_support_time.or(first.map(|_| now));
        let mut next = old.clone();
        next.lifecycle = life.into();
        next.maturity = maturity.into();
        next.last_reliable_observation_time = last;
        if effective {
            next.last_effective_observation_time = Some(now);
        }
        next.visibility_probability = visibility;
        next.association_probability = probability(assoc, "association")?;
        next.ever_confirmed = old.ever_confirmed || maturity == "CONFIRMED";
        next.epoch = old
            .epoch
            .checked_add(i64::from(life == "DORMANT" && old.lifecycle != "DORMANT"))
            .ok_or_else(|| Error::value("management epoch exceeds i64"))?;
        if version < 0 {
            return Err(Error::value("invalid management identity/version"));
        }
        next.version = version;
        next.reason = reason.into();
        next.first_real_support_frame = first;
        next.first_real_support_time = first_time;
        next.last_evaluated_frame = Some(frame.frame);
        next.last_existence_probability = Some(r);
        next.low_since_time =
            crate::candidates::low_existence_start(old, first, r, frame, previous_frame, cfg);
        next.confirmation_probability = probability(confirmation, "confirmation support")?;
        next.overdue_unconfirmed =
            !next.ever_confirmed && first_time.is_some_and(|t| now - t >= cfg.candidate_deadline_s);
        next.outside_since_time = if !reliable && outside_visibility {
            Some(old.outside_since_time.unwrap_or(now))
        } else {
            None
        };
        let why = match maturity {
            "REJECTED" => "EXISTENCE_BELOW_REJECTION",
            "CONFIRMED" => "CONFIRMATION_SUPPORT",
            "CANDIDATE" => "MEASURED_SUPPORT_ABOVE_CANDIDATE",
            _ => "MEASURED_SUPPORT_BELOW_CANDIDATE",
        };
        for (kind, before, after, why) in [
            ("LIFECYCLE", old.lifecycle.as_str(), life, reason),
            ("MATURITY", old.maturity.as_str(), maturity, why),
        ] {
            if before != after {
                events.push(TrackEvent {
                    frame: frame.frame,
                    timestamp_s: now,
                    label: label.into(),
                    kind,
                    before: before.into(),
                    after: after.into(),
                    reason: why.into(),
                    state_version: version,
                    probability_changed_by_management: false,
                });
            }
        }
        result.push(next);
    }
    Ok((result, events))
}
