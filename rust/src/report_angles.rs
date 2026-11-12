use nalgebra::{Vector2, Vector3};
use serde_json::{json, Value};
pub fn angular_samples(rows: &[Value]) -> Vec<Value> {
    let valid = rows
        .iter()
        .filter(|r| r["_direction"].is_array())
        .collect::<Vec<_>>();
    let Some(first) = valid.first() else {
        return vec![];
    };
    let start = first["timestamp_s"].as_f64().unwrap();
    let mut previous = 0.;
    let mut offset = 0.;
    let mut out = vec![];
    for (i, row) in valid.iter().enumerate() {
        let u = &row["_direction"];
        let ra = u[1].as_f64().unwrap().atan2(u[0].as_f64().unwrap());
        if i > 0 {
            let delta = ra - previous;
            let wrapped = (delta + std::f64::consts::PI).rem_euclid(2. * std::f64::consts::PI)
                - std::f64::consts::PI;
            let wrapped = if wrapped == -std::f64::consts::PI && delta > 0. {
                std::f64::consts::PI
            } else {
                wrapped
            };
            if delta.abs() >= std::f64::consts::PI {
                offset += wrapped - delta;
            }
        }
        previous = ra;
        let ra = ra + offset;
        let dec = u[2].as_f64().unwrap().clamp(-1., 1.).asin();
        out.push(json!({
            "frame": row["frame"],
            "timestamp_s": row["timestamp_s"],
            "elapsed_seconds": row["timestamp_s"].as_f64().unwrap()-start,
            "source": row["source"],
            "ra_rad": ra,
            "ra_deg": ra.to_degrees(),
            "dec_rad": dec,
            "dec_deg": dec.to_degrees(),
            "inertial": u
        }));
    }
    out
}

pub fn local_offsets(samples: &[Value]) -> Vec<[f64; 2]> {
    let Some(first) = samples.first() else {
        return vec![];
    };
    let ra = first["ra_rad"].as_f64().unwrap();
    let dec = first["dec_rad"].as_f64().unwrap();
    let origin = Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());
    let east = Vector3::new(-ra.sin(), ra.cos(), 0.);
    let north = origin.cross(&east);
    samples
        .iter()
        .map(|s| {
            let ra = s["ra_rad"].as_f64().unwrap();
            let dec = s["dec_rad"].as_f64().unwrap();
            let u = Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());
            let components = Vector2::new(u.dot(&east), u.dot(&north));
            let sine = components.norm();
            let cosine = u.dot(&origin).clamp(-1., 1.);
            if sine < crate::numerical_policy::DISPLAY_DIRECTION_SINE_FLOOR {
                if cosine > 0. {
                    [0., 0.]
                } else {
                    [f64::NAN; 2]
                }
            } else {
                let v =
                    components * (sine.atan2(cosine) / sine) * (180. / std::f64::consts::PI) * 60.;
                [v[0], v[1]]
            }
        })
        .collect()
}

pub fn observed_segments(samples: &[Value]) -> Vec<Vec<Value>> {
    let mut out = vec![];
    let mut current: Vec<Value> = vec![];
    for s in samples {
        let source = s["source"].as_str().unwrap();
        if !["measured", "recovered", "reconnected"].contains(&source) {
            continue;
        }
        if let Some(last) = current.last() {
            let starts = s["frame"].as_i64().unwrap() > last["frame"].as_i64().unwrap() + 1
                || (["recovered", "reconnected"].contains(&source)
                    && !["recovered", "reconnected"].contains(&last["source"].as_str().unwrap()));
            if starts {
                out.push(std::mem::take(&mut current));
            }
        }
        current.push(s.clone());
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

pub fn gap_diagnostics(samples: &[Value], segments: &[Vec<Value>]) -> Vec<Value> {
    segments
        .windows(2)
        .map(|pair| {
            let previous = pair[0].last().unwrap();
            let next = &pair[1][0];
            let pn = previous["frame"].as_i64().unwrap();
            let nn = next["frame"].as_i64().unwrap();

            let gaps = samples
                .iter()
                .filter(|s| {
                    s["frame"].as_i64().is_some_and(|n| n > pn && n < nn)
                        && s["source"] == "predicted"
                })
                .cloned()
                .collect::<Vec<_>>();
            let mut history = pair[0].iter().rev().take(2).cloned().collect::<Vec<_>>();
            history.reverse();
            history.extend(gaps.clone());

            let (mut ra, mut dec) = (
                previous["ra_rad"].as_f64().unwrap(),
                previous["dec_rad"].as_f64().unwrap(),
            );
            if history.len() >= 2 {
                let first = &history[history.len() - 2];
                let last = &history[history.len() - 1];
                let dt =
                    last["timestamp_s"].as_f64().unwrap() - first["timestamp_s"].as_f64().unwrap();
                let remaining =
                    next["timestamp_s"].as_f64().unwrap() - last["timestamp_s"].as_f64().unwrap();
                ra = last["ra_rad"].as_f64().unwrap()
                    + (last["ra_rad"].as_f64().unwrap() - first["ra_rad"].as_f64().unwrap()) / dt
                        * remaining;
                dec = last["dec_rad"].as_f64().unwrap()
                    + (last["dec_rad"].as_f64().unwrap() - first["dec_rad"].as_f64().unwrap()) / dt
                        * remaining;
            }

            dec = dec.clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
            let u = [dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()];
            let dot = (0..3)
                .map(|i| u[i] * next["inertial"][i].as_f64().unwrap())
                .sum::<f64>()
                .clamp(-1., 1.);
            json!({
                "previous": previous,
                "observed_next": next,
                "predicted_ra_deg": ra.to_degrees(),
                "predicted_dec_deg": dec.to_degrees(),
                "residual_deg": dot.acos().to_degrees(),
                "gap_frames": (nn - pn - 1).max(0),
                "delta_seconds": next["timestamp_s"].as_f64().unwrap()
                    - previous["timestamp_s"].as_f64().unwrap(),
                "prediction_samples": gaps
            })
        })
        .collect()
}
