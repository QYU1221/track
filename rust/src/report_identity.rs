use crate::{
    contracts,
    error::{Error, Result},
    gnomonic::GnomonicPlane,
    linalg::{array, mat},
    measurement_covariance::propagate_tangent_covariance,
};
use nalgebra::Vector2;
use serde_json::{json, Value};
use std::collections::BTreeSet;

pub const MIN_OBSERVED_POINTS: usize = 10;
pub const MIN_MEAN_TARGET_POSTERIOR: f64 = 0.85;
pub const MAX_JUMP_FRACTION: f64 = 0.1;

pub fn jump_threshold_squared(false_alarm_probability: f64) -> Result<f64> {
    if !false_alarm_probability.is_finite()
        || !(0.0 < false_alarm_probability && false_alarm_probability < 1.0)
    {
        return Err(Error::value(
            "jump_false_alarm_probability must lie in (0,1)",
        ));
    }
    Ok(-2.0 * false_alarm_probability.ln())
}

pub fn jump_diagnostics(observed: &[Value], false_alarm_probability: f64) -> Result<Value> {
    let threshold = jump_threshold_squared(false_alarm_probability)?;
    let sources = observed
        .iter()
        .map(|row| {
            row["jump_covariance_source"]
                .as_str()
                .filter(|s| !s.is_empty())
                .unwrap_or("UNSPECIFIED")
                .to_owned()
        })
        .collect::<BTreeSet<_>>();
    let mut result = json!({
        "status": "INSUFFICIENT_OBSERVATIONS",
        "check_count": 0,
        "jump_count": 0,
        "jump_fraction": null,
        "threshold_squared": threshold,
        "checks": [],
        "covariance_sources": sources,
    });
    if observed.len() < 3 {
        return Ok(result);
    }
    if observed.iter().any(|row| row["_direction"].is_null()) {
        result["status"] = json!("MISSING_DIRECTION");
        return Ok(result);
    }
    if observed.iter().any(|row| row["_los_covariance"].is_null()) {
        result["status"] = json!("MISSING_COVARIANCE");
        return Ok(result);
    }
    let times = observed
        .iter()
        .map(|row| contracts::scalar(&row["timestamp_s"], "jump timestamps"))
        .collect::<Result<Vec<_>>>();
    let times = match times {
        Ok(times) if times.windows(2).all(|pair| pair[1] > pair[0]) => times,
        _ => {
            result["status"] = json!("INVALID_TIMESTAMPS");
            return Ok(result);
        }
    };
    let checks = match covariance_checks(observed, &times, threshold) {
        Ok(checks) => checks,
        Err(_) => {
            result["status"] = json!("INVALID_GEOMETRY_OR_COVARIANCE");
            return Ok(result);
        }
    };
    let count = checks.iter().filter(|check| check["jump"] == true).count();
    result["status"] = json!("AVAILABLE");
    result["check_count"] = json!(checks.len());
    result["jump_count"] = json!(count);
    result["jump_fraction"] = json!(count as f64 / checks.len() as f64);
    result["checks"] = json!(checks);
    Ok(result)
}

fn covariance_checks(observed: &[Value], times: &[f64], threshold: f64) -> Result<Vec<Value>> {
    let directions = observed
        .iter()
        .map(|row| {
            contracts::vector::<3>(
                &row["_direction"],
                "jump directions",
                "invalid jump direction",
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let center = std::array::from_fn(|axis| {
        directions.iter().map(|u| u[axis]).sum::<f64>() / directions.len() as f64
    });
    let plane = GnomonicPlane::new("identity-quality", center, 89_f64.to_radians(), None)?;
    let mut positions = Vec::with_capacity(observed.len());
    let mut covariances = Vec::with_capacity(observed.len());
    for (row, direction) in observed.iter().zip(directions) {
        let (position, jacobian) = plane.project(direction)?;
        let covariance = contracts::matrix::<3, 3>(
            &row["_los_covariance"],
            "LOS covariance",
            "invalid LOS covariance dimensions",
        )?;
        positions.push(Vector2::from(position));
        covariances.push(mat(propagate_tangent_covariance(covariance, jacobian)?));
    }
    let mut checks = Vec::with_capacity(observed.len() - 2);
    for i in 1..observed.len() - 1 {
        let ratio = (times[i + 1] - times[i]) / (times[i] - times[i - 1]);
        let residual = positions[i + 1] - positions[i] - ratio * (positions[i] - positions[i - 1]);
        let covariance = covariances[i + 1]
            + (1.0 + ratio).powi(2) * covariances[i]
            + ratio * ratio * covariances[i - 1];
        let covariance = mat(contracts::checked_covariance(array(covariance))?);
        let lower = covariance
            .cholesky()
            .ok_or_else(|| Error::value("jump covariance must be positive definite"))?
            .l();
        let whitened = lower
            .solve_lower_triangular(&residual)
            .ok_or_else(|| Error::value("invalid jump covariance"))?;
        let distance = whitened.dot(&whitened);
        if !distance.is_finite() {
            return Err(Error::value("invalid jump distance"));
        }
        checks.push(json!({
            "frame": observed[i + 1]["frame"],
            "residual_norm": residual.norm(),
            "mahalanobis_squared": distance,
            "jump": distance > threshold,
        }));
    }
    Ok(checks)
}

pub fn target_identity_diagnostics(
    ever_confirmed: bool,
    observed: &[Value],
    false_alarm_probability: f64,
) -> Result<Value> {
    let jumps = jump_diagnostics(observed, false_alarm_probability)?;
    let posteriors = observed
        .iter()
        .filter(|row| !row["target_posterior"].is_null())
        .map(|row| row["target_posterior"].as_f64().unwrap_or(f64::NAN))
        .collect::<Vec<_>>();
    let reason = if !ever_confirmed {
        "NEVER_CONFIRMED"
    } else if observed.len() < MIN_OBSERVED_POINTS {
        "INSUFFICIENT_OBSERVATIONS"
    } else if posteriors.is_empty()
        || posteriors.iter().any(|p| !p.is_finite())
        || crate::numpy_arithmetic::pairwise_sum(&posteriors) / (posteriors.len() as f64)
            < MIN_MEAN_TARGET_POSTERIOR
    {
        "INSUFFICIENT_TARGET_POSTERIOR"
    } else if jumps["status"] != "AVAILABLE" {
        return Ok(json!({
                   "confirmed": false,
                   "reason": format!("JUMP_CHECK_{}",
        jumps["status"].as_str().unwrap()),
                   "jumps": jumps
               }));
    } else if jumps["jump_fraction"].as_f64().unwrap() > MAX_JUMP_FRACTION {
        "EXCESS_NOISE_NORMALIZED_JUMPS"
    } else {
        "CONFIRMED_TARGET_IDENTITY"
    };
    Ok(json!({
        "confirmed": reason == "CONFIRMED_TARGET_IDENTITY",
        "reason": reason,
        "jumps": jumps
    }))
}
