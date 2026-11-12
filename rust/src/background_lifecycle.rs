use crate::{
    camera_model::CameraModel,
    contracts::FrameInfo,
    error::{Error, Result},
    exact_numerics::ObservationWindow,
    exact_posterior::fsum,
    linalg::mat,
    tracking_background::BackgroundConfig,
    tracking_model::JointHistory,
};
use nalgebra::Vector2;
use serde::Serialize;
use std::{borrow::Cow, collections::BTreeSet};
pub fn active_groups(
    groups: &[Vec<JointHistory>],
    rotation: [[f64; 3]; 3],
    camera: &CameraModel,
    window: &ObservationWindow,
    config: &BackgroundConfig,
) -> Result<BTreeSet<usize>> {
    if !config.enable_view_dormancy {
        return Ok((0..groups.len()).collect());
    }
    let k = mat(camera.k());
    let a = k.fixed_view::<2, 2>(0, 0).into_owned();
    let inverse = a
        .lu()
        .try_inverse()
        .ok_or_else(|| Error::value("Singular matrix"))?;
    let radius = if camera.distortion_model == "none" {
        let [x0, x1, y0, y1] = window.bounds();
        let mut radius = 0_f64;
        for x in [x0, x1] {
            for y in [y0, y1] {
                let p = a
                    .lu()
                    .solve(&Vector2::new(x - k[(0, 2)], y - k[(1, 2)]))
                    .ok_or_else(|| Error::value("Singular matrix"))?;
                radius = radius.max(p.norm());
            }
        }
        radius
    } else {
        camera.max_normalized_radius
    };
    let largest = |m: [[f64; 2]; 2]| mat(m).symmetric_eigen().eigenvalues.max();
    let norm = inverse.svd(false, false).singular_values.max();
    let angle = libm::atan(radius)
        + config.dormancy_margin_deg * std::f64::consts::PI / 180.
        + config.dormancy_sigma * largest(config.measurement_covariance_px2).sqrt() * norm;
    let mut active = BTreeSet::new();
    let mut entries = vec![];
    for (i, group) in groups.iter().enumerate() {
        if group.iter().any(|h| !h.tracks.is_empty()) {
            active.insert(i);
            continue;
        }
        for h in group {
            for b in &h.backgrounds {
                let Some(s) = &b.state else { continue };
                let (direction, _) = b.plane.inverse(s.mean)?;
                let chord = config.dormancy_sigma * largest(s.covariance).sqrt();
                let spread = 2. * libm::asin((chord / 2.).min(1.));
                entries.push((i, direction, angle + spread));
            }
        }
    }
    if entries.is_empty() {
        return Ok(active);
    }
    let max_span = entries
        .iter()
        .map(|e| e.2)
        .fold(f64::NEG_INFINITY, f64::max)
        .min(std::f64::consts::PI);
    let radius = 2. * libm::sin(max_span / 2.);
    let boresight = [rotation[0][2], rotation[1][2], rotation[2][2]];
    for (i, direction, span) in entries {
        let squared: f64 = direction
            .iter()
            .zip(boresight)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        if squared > radius * radius {
            continue;
        }
        let dot: f64 = direction.iter().zip(boresight).map(|(a, b)| a * b).sum();
        if dot >= libm::cos(span.min(std::f64::consts::PI)) {
            active.insert(i);
        }
    }
    Ok(active)
}

#[derive(Debug, Clone, Serialize)]
pub struct RetirementEvent {
    #[serde(rename = "Background_ID")]
    pub background_id: String,
    pub frame: i64,
    pub timestamp_s: f64,
    pub reason: &'static str,
    pub discarded_existence_mass: f64,
    pub mature_existence_mass: f64,
}

pub fn retire_backgrounds<'a>(
    histories: &'a [JointHistory],
    frame: &FrameInfo,
    config: &BackgroundConfig,
) -> Result<(Cow<'a, [JointHistory]>, Vec<RetirementEvent>)> {
    if !config.enable_retirement {
        return Ok((Cow::Borrowed(histories), vec![]));
    }
    let identities: BTreeSet<_> = histories
        .iter()
        .flat_map(|h| h.backgrounds.iter().map(|b| b.background_id.as_str()))
        .collect();
    let mut removed = BTreeSet::new();
    let mut events = vec![];
    for identity in identities {
        let parts: Vec<_> = histories
            .iter()
            .flat_map(|h| {
                h.backgrounds
                    .iter()
                    .filter(move |b| b.background_id == identity)
                    .map(move |b| (h.weight, b))
            })
            .collect();
        let mass = fsum(parts.iter().map(|(w, b)| w * b.existence))?;
        let mature_mass = fsum(
            parts
                .iter()
                .filter(|(_, b)| b.visible_miss_count >= config.retirement_visible_misses)
                .map(|(w, b)| w * b.existence),
        )?;
        let logs: Vec<_> = parts
            .iter()
            .filter(|(w, b)| *w > 0. && b.log_existence > f64::NEG_INFINITY)
            .map(|(w, b)| (libm::log(*w) + b.log_existence, *b))
            .collect();
        let maximum = logs.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let scaled = fsum(logs.iter().map(|p| libm::exp(p.0 - maximum)))?;
        let mature_scaled = fsum(
            logs.iter()
                .filter(|(_, b)| b.visible_miss_count >= config.retirement_visible_misses)
                .map(|p| libm::exp(p.0 - maximum)),
        )?;
        if mass < config.retirement_existence_probability
            && mature_scaled > 0.
            && mature_scaled >= config.retirement_mature_mass_fraction * scaled
        {
            removed.insert(identity);
            events.push(RetirementEvent {
                background_id: identity.into(),
                frame: frame.frame,
                timestamp_s: frame.timestamp_s,
                reason: "LOW_MARGINAL_EXISTENCE_AFTER_VISIBLE_MISSES",
                discarded_existence_mass: mass,
                mature_existence_mass: mature_mass,
            });
        }
    }
    if removed.is_empty() {
        return Ok((Cow::Borrowed(histories), events));
    }
    let mut result = histories.to_vec();
    for h in &mut result {
        h.backgrounds
            .retain(|b| !removed.contains(b.background_id.as_str()));
        h.background_assignments
            .retain(|(s, _)| !removed.contains(s.chars().skip(4).collect::<String>().as_str()));
    }
    Ok((Cow::Owned(result), events))
}
