use crate::{
    error::{Error, Result},
    geometry::quaternion_matrix,
    report_angles,
    report_canvas::{Canvas, PALETTE},
    report_records::{DisplayTrack, Records},
};
use serde_json::Value;
use std::path::Path;
fn coordinates(rows: &[Value]) -> Vec<Option<[f64; 2]>> {
    rows.iter()
        .map(|r| Some([r["x"].as_f64()?, r["y"].as_f64()?]))
        .collect()
}

fn draw(canvas: &mut Canvas, track: &DisplayTrack, rows: &[Value], coords: &[Option<[f64; 2]>]) {
    let color = PALETTE[track.id % PALETTE.len()];
    for i in 1..coords.len() {
        let (Some(a), Some(b)) = (coords[i - 1], coords[i]) else {
            continue;
        };
        if rows[i]["frame"].as_i64().unwrap() != rows[i - 1]["frame"].as_i64().unwrap() + 1 {
            continue;
        }
        let predicted = rows[i]["source"] == "predicted" || rows[i - 1]["source"] == "predicted";
        canvas.line(
            a[0],
            a[1],
            b[0],
            b[1],
            if predicted {
                color.map(|c| (c as f64 * 0.45) as u8)
            } else {
                color
            },
            if predicted { 1 } else { 2 },
            predicted,
        );
    }
    if let Some(Some([x, y])) = coords.last() {
        let predicted = rows.last().is_some_and(|r| r["source"] == "predicted");
        let color = if predicted {
            color.map(|c| (c as f64 * 0.45) as u8)
        } else {
            color
        };
        let exited = ["RETIRED", "TERMINATED", "NO_RETAINED_SUPPORT"]
            .contains(&track.last["Track_Lifecycle"].as_str().unwrap_or(""));
        canvas.circle(
            *x,
            *y,
            if exited {
                7
            } else if coords.len() == 1 {
                4
            } else {
                6
            },
            color,
            !predicted && !exited,
        );
        canvas.text(
            *x as i32 + 8,
            *y as i32 - 6,
            &format!(
                "ID:{}{}{}",
                track.id,
                if predicted { " P" } else { "" },
                if exited { " X" } else { "" }
            ),
            color,
            1,
        );
    }
}

pub fn frame(folder: &Path, records: &Records, number: i64) -> Result<()> {
    let r = &records.configuration["reporting"];
    if r["track_frame_png_mode"] == "off"
        || (r["track_frame_png_mode"] == "interval"
            && records.frame_count as u64 % r["track_frame_png_interval"].as_u64().unwrap() != 0)
    {
        return Ok(());
    }
    let scale = r["track_frame_png_scale"].as_f64().unwrap();
    let c = &records.packet.camera;
    let width = (c.width as f64 * scale).round_ties_even() as usize;
    let height = (c.height as f64 * scale).round_ties_even() as usize;
    if width == 0 || height == 0 {
        return Err(Error::value("rendered image dimensions must be positive"));
    }
    let mut canvas = Canvas::new(width, height, [0; 3])?;
    for d in records
        .packet
        .detections
        .iter()
        .filter(|d| d.frame == number)
    {
        if c.contains([d.x_px, d.y_px]) {
            canvas.circle(d.x_px * scale, d.y_px * scale, 2, [180; 3], true);
        }
    }
    canvas.text(12, 12, "SOLID=MEASURED DASHED=PREDICTED", [180; 3], 1);
    canvas.text(12, 26, &format!("FRAME {number}"), [180; 3], 1);
    let timestamp = records
        .packet
        .frames
        .iter()
        .find(|f| f.frame == number)
        .unwrap()
        .timestamp_s;
    let low = records.configuration["lifecycle"]["candidate_low_probability"]
        .as_f64()
        .unwrap();
    let dormant = records.configuration["lifecycle"]["dormant_after_s"]
        .as_f64()
        .unwrap();
    for t in &records.tracks {
        let Some(last) = t
            .rows
            .last()
            .filter(|r| r["frame"] == number && !r["x"].is_null())
        else {
            continue;
        };
        let history = records.track_rows(t)?;
        if last["source"] == "predicted" {
            let measured = history.iter().rev().find(|r| r["source"] != "predicted");
            if measured.is_none_or(|r| timestamp - r["timestamp_s"].as_f64().unwrap() >= dormant)
                || last["existence_probability"].as_f64().unwrap() < low
            {
                continue;
            }
        }
        let rows = history
            .iter()
            .filter(|r| {
                r["source"] != "predicted"
                    || (r["existence_probability"].as_f64().unwrap() >= low && !r["x"].is_null())
            })
            .cloned()
            .collect::<Vec<_>>();
        let coords = coordinates(&rows)
            .into_iter()
            .map(|p| p.map(|p| p.map(|x| x * scale)))
            .collect::<Vec<_>>();
        draw(&mut canvas, t, &rows, &coords);
    }
    canvas.save(
        &folder
            .join("track_frames")
            .join(format!("frame_{number:04}.png")),
    )
}

fn panel(
    canvas: &mut Canvas,
    rect: [f64; 4],
    paths: &[(Vec<Value>, Vec<Option<[f64; 2]>>, usize)],
    title: &str,
    xlabel: &str,
    ylabel: &str,
) {
    let [x, y, w, h] = rect;
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for (_, points, _) in paths {
        for p in points.iter().flatten() {
            if p.iter().all(|x| x.is_finite()) {
                for j in 0..2 {
                    lo[j] = lo[j].min(p[j]);
                    hi[j] = hi[j].max(p[j]);
                }
            }
        }
    }
    let short = title
        .chars()
        .take(((w - 30.) / 6.) as usize)
        .collect::<String>();
    canvas.text(x as i32 + 12, y as i32 + 8, &short, [35; 3], 1);
    let left = x + 85.;
    let right = x + w - 25.;
    let top = y + 50.;
    let bottom = y + h - 55.;
    for i in 0..=4 {
        let t = i as f64 / 4.;
        canvas.line(
            left,
            top + t * (bottom - top),
            right,
            top + t * (bottom - top),
            [220; 3],
            1,
            false,
        );
        canvas.line(
            left + t * (right - left),
            top,
            left + t * (right - left),
            bottom,
            [220; 3],
            1,
            false,
        );
    }
    canvas.line(left, top, left, bottom, [30; 3], 1, false);
    canvas.line(left, bottom, right, bottom, [30; 3], 1, false);
    canvas.text(left as i32, bottom as i32 + 30, xlabel, [40; 3], 1);
    canvas.text(x as i32, top as i32 - 16, ylabel, [40; 3], 1);
    if !lo[0].is_finite() {
        canvas.text(
            left as i32 + 20,
            top as i32 + 40,
            "NO CONFIRMED TARGET SAMPLES",
            [80; 3],
            1,
        );
        return;
    }
    let span = [(hi[0] - lo[0]).max(1e-9), (hi[1] - lo[1]).max(1e-9)];
    let map = |p: [f64; 2]| {
        [
            left + 15. + (p[0] - lo[0]) / span[0] * (right - left - 30.),
            bottom - 15. - (p[1] - lo[1]) / span[1] * (bottom - top - 30.),
        ]
    };
    for i in 0..=4 {
        let t = i as f64 / 4.;
        canvas.text(
            (left + t * (right - left)) as i32 - 15,
            bottom as i32 + 8,
            &format!("{:.4}", lo[0] + t * span[0]),
            [60; 3],
            1,
        );
        canvas.text(
            x as i32,
            (bottom - t * (bottom - top)) as i32,
            &format!("{:.4}", lo[1] + t * span[1]),
            [60; 3],
            1,
        );
    }
    for (rows, points, id) in paths {
        let color = PALETTE[*id % PALETTE.len()];
        let timed = xlabel == "EAST OFFSET ARCMIN";
        let first = rows
            .first()
            .and_then(|r| r["elapsed_seconds"].as_f64())
            .unwrap_or(0.);
        let last = rows
            .last()
            .and_then(|r| r["elapsed_seconds"].as_f64())
            .unwrap_or(first);
        let time_color = |i: usize| {
            let t = (rows[i]["elapsed_seconds"].as_f64().unwrap_or(first) - first)
                / (last - first).max(1e-12);
            let anchors = [
                [68., 1., 84.],
                [59., 82., 139.],
                [33., 145., 140.],
                [94., 201., 98.],
                [253., 231., 37.],
            ];
            let a = (t.clamp(0., 1.) * 4.).min(3.999999);
            let j = a.floor() as usize;
            let f = a - j as f64;
            std::array::from_fn(|k| {
                (anchors[j][k] + (anchors[j + 1][k] - anchors[j][k]) * f).round() as u8
            })
        };
        for i in 1..points.len() {
            let (Some(a), Some(b)) = (points[i - 1], points[i]) else {
                continue;
            };
            if !a.iter().chain(&b).all(|x| x.is_finite())
                || rows[i]["frame"].as_i64() != rows[i - 1]["frame"].as_i64().map(|n| n + 1)
            {
                continue;
            }
            let a = map(a);
            let b = map(b);
            canvas.line(
                a[0],
                a[1],
                b[0],
                b[1],
                if timed { time_color(i - 1) } else { color },
                2,
                rows[i]["source"] == "predicted" || rows[i - 1]["source"] == "predicted",
            );
        }
        for (i, p) in points.iter().enumerate() {
            let Some(p) = p.filter(|p| p.iter().all(|x| x.is_finite())) else {
                continue;
            };
            let p = map(p);
            let predicted = rows[i]["source"] == "predicted";
            if predicted {
                canvas.line(
                    p[0] - 4.,
                    p[1] - 4.,
                    p[0] + 4.,
                    p[1] + 4.,
                    [230, 100, 27],
                    1,
                    false,
                );
                canvas.line(
                    p[0] - 4.,
                    p[1] + 4.,
                    p[0] + 4.,
                    p[1] - 4.,
                    [230, 100, 27],
                    1,
                    false,
                );
            } else {
                canvas.circle(
                    p[0],
                    p[1],
                    3,
                    if timed { time_color(i) } else { color },
                    true,
                );
            }
        }
        if let Some(Some(p)) = points.first() {
            let p = map(*p);
            canvas.circle(p[0], p[1], 6, [50, 205, 50], true);
        }
        if let Some(Some(p)) = points.last() {
            let p = map(*p);
            canvas.circle(p[0], p[1], 6, [239, 51, 64], false);
        }
    }
}

pub fn finish(folder: &Path, records: &Records) -> Result<Vec<String>> {
    let formal = records
        .tracks
        .iter()
        .filter(|t| t.final_formal)
        .collect::<Vec<_>>();
    let c = &records.packet.camera;
    let mut figures = vec![];
    for (name, observed) in [
        ("tracks_overlay.png", false),
        ("tracks_observed_overlay.png", true),
    ] {
        let mut canvas = Canvas::new(c.width as usize, c.height as usize, [0; 3])?;
        for t in &formal {
            let history = records.track_rows(t)?;
            let rows = history
                .iter()
                .filter(|r| !observed || r["source"] != "predicted")
                .cloned()
                .collect::<Vec<_>>();
            draw(&mut canvas, t, &rows, &coordinates(&rows));
        }
        if formal.is_empty() {
            canvas.text(12, 24, "NO CONFIRMED TARGET TRACKS", [180; 3], 1);
        }
        canvas.save(&folder.join(name))?;
        figures.push(name.into());
    }
    let mut image = Canvas::new(1600, 1000, [0; 3])?;
    let valid = records
        .packet
        .frames
        .iter()
        .filter(|frame| {
            records.committed_frames.contains(&frame.frame)
                && frame.inertial_observation_available()
        })
        .collect::<Vec<_>>();
    let mut paths = vec![];
    if !valid.is_empty() {
        let r = quaternion_matrix(&valid[valid.len() / 2].quaternion_xyzw.unwrap())?;
        for t in &formal {
            let history = records.track_rows(t)?;
            let coords = history
                .iter()
                .map(|row| {
                    let d = row["_direction"].as_array()?;
                    let v = std::array::from_fn::<_, 3, _>(|j| {
                        (0..3)
                            .map(|i| r[i][j] * d[i].as_f64().unwrap())
                            .sum::<f64>()
                    });
                    if v[2] > crate::numerical_policy::DISPLAY_PROJECTION_MIN_Z {
                        Some([-v[0] / v[2], -v[1] / v[2]])
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            paths.push((*t, coords));
        }
    }
    let points = paths
        .iter()
        .flat_map(|(_, p)| p.iter().flatten())
        .collect::<Vec<_>>();
    if !points.is_empty() {
        let lo = std::array::from_fn::<_, 2, _>(|j| {
            points.iter().map(|p| p[j]).fold(f64::INFINITY, f64::min)
        });
        let hi = std::array::from_fn::<_, 2, _>(|j| {
            points
                .iter()
                .map(|p| p[j])
                .fold(f64::NEG_INFINITY, f64::max)
        });
        let scale = (1480. / (hi[0] - lo[0]).max(1e-9)).min(880. / (hi[1] - lo[1]).max(1e-9));
        for (t, p) in &paths {
            let p = p
                .iter()
                .map(|p| {
                    p.map(|p| {
                        [
                            (p[0] - (lo[0] + hi[0]) / 2.) * scale + 800.,
                            (p[1] - (lo[1] + hi[1]) / 2.) * scale + 500.,
                        ]
                    })
                })
                .collect::<Vec<_>>();
            let history = records.track_rows(t)?;
            draw(&mut image, t, &history, &p);
        }
    } else {
        image.text(
            12,
            24,
            "NO CONFIRMED TRACKS WITH VALID INERTIAL SAMPLES",
            [180; 3],
            2,
        );
    }
    image.save(&folder.join("tracks_inertial_overlay.png"))?;
    figures.push("tracks_inertial_overlay.png".into());
    let low = records.configuration["lifecycle"]["candidate_low_probability"]
        .as_f64()
        .unwrap();
    let mut samples = Vec::new();
    for t in &formal {
        let history = records.track_rows(t)?;
        let rows = history
            .into_iter()
            .filter(|r| {
                r["source"] != "predicted"
                    || (r["existence_probability"].as_f64().unwrap() >= low && !r["x"].is_null())
            })
            .collect::<Vec<_>>();
        let angular = report_angles::angular_samples(&rows);
        if !angular.is_empty() {
            samples.push((*t, angular));
        }
    }
    for observed in [false, true] {
        if observed && samples.is_empty() {
            continue;
        }
        let count = samples.len().max(1);
        let columns = count.min(2);
        let rows = count.div_ceil(columns);
        let mut canvas = Canvas::new(1000 * columns, 620 * rows, [255; 3])?;
        if samples.is_empty() {
            panel(
                &mut canvas,
                [0., 0., 1000., 620.],
                &[],
                "FORMAL TRAJECTORY ANGULAR OVERVIEW",
                "EAST OFFSET ARCMIN",
                "NORTH OFFSET ARCMIN",
            );
        }
        for (i, (t, s)) in samples.iter().enumerate() {
            let s = s
                .iter()
                .filter(|s| !observed || s["source"] != "predicted")
                .cloned()
                .collect::<Vec<_>>();
            let points = report_angles::local_offsets(&s)
                .into_iter()
                .map(Some)
                .collect::<Vec<_>>();
            panel(
                &mut canvas,
                [
                    (i % columns * 1000) as f64,
                    (i / columns * 620) as f64,
                    1000.,
                    620.,
                ],
                &[(s, points, t.id)],
                &format!(
                    "T{:06} {}",
                    t.id,
                    if observed {
                        "OBSERVED"
                    } else {
                        "OBSERVED AND PREDICTED"
                    }
                ),
                "EAST OFFSET ARCMIN",
                "NORTH OFFSET ARCMIN",
            );
        }
        let name = format!(
            "formal_tracks/{}_trajectory_angular_overview.png",
            if observed { "observed" } else { "formal" }
        );
        canvas.save(&folder.join(&name))?;
        figures.push(name);
    }
    for (t, samples) in &samples {
        let segments = report_angles::observed_segments(samples);
        if segments.is_empty() {
            continue;
        }
        let gaps = report_angles::gap_diagnostics(samples, &segments);
        let mut canvas = Canvas::new(2000, 1400, [255; 3])?;
        for (index, (field, label)) in [("ra_deg", "RA DEG"), ("dec_deg", "DEC DEG")]
            .into_iter()
            .enumerate()
        {
            let mut paths = segments
                .iter()
                .map(|s| {
                    (
                        s.clone(),
                        s.iter()
                            .map(|r| {
                                Some([
                                    r["elapsed_seconds"].as_f64().unwrap(),
                                    r[field].as_f64().unwrap(),
                                ])
                            })
                            .collect::<Vec<_>>(),
                        t.id,
                    )
                })
                .collect::<Vec<_>>();
            for g in &gaps {
                let a = &g["previous"];
                let b = &g["observed_next"];
                let mut end = b.clone();
                end["source"] = Value::String("predicted".into());
                end["frame"] = serde_json::json!(a["frame"].as_i64().unwrap() + 1);
                let predicted = g[if field == "ra_deg" {
                    "predicted_ra_deg"
                } else {
                    "predicted_dec_deg"
                }]
                .as_f64()
                .unwrap();
                paths.push((
                    vec![a.clone(), end],
                    vec![
                        Some([
                            a["elapsed_seconds"].as_f64().unwrap(),
                            a[field].as_f64().unwrap(),
                        ]),
                        Some([b["elapsed_seconds"].as_f64().unwrap(), predicted]),
                    ],
                    t.id,
                ));
            }
            panel(
                &mut canvas,
                [(index * 1000) as f64, 0., 1000., 650.],
                &paths,
                &format!("T{:06} {label} OBSERVED AND GAP EXTRAPOLATION", t.id),
                "ELAPSED SECONDS",
                label,
            );
        }
        let mut paths = segments
            .iter()
            .map(|s| {
                (
                    s.clone(),
                    s.iter()
                        .map(|r| {
                            Some([
                                r["ra_deg"].as_f64().unwrap(),
                                r["dec_deg"].as_f64().unwrap(),
                            ])
                        })
                        .collect::<Vec<_>>(),
                    t.id,
                )
            })
            .collect::<Vec<_>>();
        for g in &gaps {
            let a = g["previous"].clone();
            let mut b = g["observed_next"].clone();
            b["source"] = serde_json::json!("predicted");
            b["frame"] = serde_json::json!(a["frame"].as_i64().unwrap() + 1);
            let predicted = [
                g["predicted_ra_deg"].as_f64().unwrap(),
                g["predicted_dec_deg"].as_f64().unwrap(),
            ];
            paths.push((
                vec![a.clone(), b.clone()],
                vec![
                    Some([
                        a["ra_deg"].as_f64().unwrap(),
                        a["dec_deg"].as_f64().unwrap(),
                    ]),
                    Some(predicted),
                ],
                t.id,
            ));
            let mut c = b.clone();
            c["frame"] = serde_json::json!(b["frame"].as_i64().unwrap() + 1);
            paths.push((
                vec![b, c],
                vec![
                    Some(predicted),
                    Some([
                        g["observed_next"]["ra_deg"].as_f64().unwrap(),
                        g["observed_next"]["dec_deg"].as_f64().unwrap(),
                    ]),
                ],
                t.id,
            ));
        }
        panel(
            &mut canvas,
            [0., 680., 1000., 650.],
            &paths,
            "RA DEC OBSERVED SUBSEGMENTS",
            "RA DEG",
            "DEC DEG",
        );
        canvas.text(
            1012,
            688,
            "GREAT CIRCLE EXTRAPOLATION RESIDUALS",
            [35; 3],
            1,
        );
        let max = gaps
            .iter()
            .map(|g| g["residual_deg"].as_f64().unwrap())
            .fold(0.0_f64, f64::max)
            .max(1e-6);
        canvas.line(1085., 730., 1085., 1275., [40; 3], 1, false);
        canvas.line(1085., 1275., 1975., 1275., [40; 3], 1, false);
        if gaps.is_empty() {
            canvas.text(1200, 950, "NO OBSERVATION GAPS", [80; 3], 2);
        } else {
            let width = 850. / gaps.len() as f64;
            for (i, g) in gaps.iter().enumerate() {
                let residual = g["residual_deg"].as_f64().unwrap();
                let x = 1100. + i as f64 * width;
                let height = residual / max * 440.;
                canvas.rect(
                    x as i32,
                    (1275. - height) as i32,
                    (x + width * 0.65) as i32,
                    1275,
                    [114, 151, 189],
                );
                canvas.text(
                    x as i32,
                    (1255. - height) as i32,
                    &format!("{residual:.6} DEG"),
                    [40; 3],
                    1,
                );
                canvas.text(
                    x as i32,
                    1290,
                    &format!("P{}-P{}", i + 1, i + 2),
                    [40; 3],
                    1,
                );
                canvas.text(
                    x as i32,
                    1304,
                    &format!(
                        "DT {:.2} MISS {}",
                        g["delta_seconds"].as_f64().unwrap(),
                        g["gap_frames"]
                    ),
                    [40; 3],
                    1,
                );
            }
        }
        let name = format!(
            "formal_tracks/T{:06}_complete_track_prediction_diagnostics.png",
            t.id
        );
        canvas.save(&folder.join(&name))?;
        figures.push(name);
    }
    Ok(figures)
}
