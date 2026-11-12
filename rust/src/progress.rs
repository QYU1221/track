use crate::{
    contracts::{FrameInfo, ObservationPacket},
    data_io::io_error,
    error::{Error, Result},
};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

fn counts(values: impl Iterator<Item = String>) -> String {
    let mut map = BTreeMap::new();
    for key in values {
        *map.entry(key).or_insert(0usize) += 1;
    }
    map.into_iter()
        .map(|(k, n)| format!("{k}={n}"))
        .collect::<Vec<_>>()
        .join(", ")
}

pub struct Progress {
    path: PathBuf,
    verbose: bool,
    log_every: usize,
    total: usize,
    committed: usize,
    previous: Option<(bool, usize)>,
    started: Instant,
}

impl Progress {
    pub fn new(
        folder: &Path,
        packet: &ObservationPacket,
        selected: &[i64],
        source: &str,
        verbose: bool,
        log_every: usize,
    ) -> Result<Self> {
        if log_every == 0 {
            return Err(Error::value(
                "log_every / --log-every must be a positive integer",
            ));
        }
        let path = folder.join("debug/terminal_output.log");
        std::fs::create_dir_all(path.parent().unwrap()).map_err(io_error)?;
        std::fs::write(&path, []).map_err(io_error)?;
        let p = Self {
            path,
            verbose,
            log_every,
            total: selected.len(),
            committed: 0,
            previous: None,
            started: Instant::now(),
        };
        let frames = packet
            .frames
            .iter()
            .filter(|f| selected.contains(&f.frame))
            .collect::<Vec<_>>();
        let detections = packet
            .detections
            .iter()
            .filter(|d| selected.contains(&d.frame))
            .collect::<Vec<_>>();
        p.write(
            &format!(
                "Input: {} frames, {} detections (external attitude)",
                frames.len(),
                detections.len()
            ),
            Some(&format!("开始处理：{} 帧", frames.len())),
        )?;
        p.write(&format!("Source: {source}"), None)?;
        p.write(&format!("Output: {}", folder.display()), None)?;
        p.write(
            "Engine: factorized_joint_ppp; inference=APPROXIMATE_FAST",
            None,
        )?;
        p.write(
            &format!(
                concat!(
                    "Input diagnostics: camera={}x{} valid_detections={} ",
                    "invalid_detections={} out_of_bounds={} invalid_attitude_frames={} ",
                    "missing_input_frames={}",
                ),
                packet.camera.width,
                packet.camera.height,
                detections.iter().filter(|d| d.valid).count(),
                detections.iter().filter(|d| !d.valid).count(),
                detections
                    .iter()
                    .filter(|d| !packet.camera.contains([d.x_px, d.y_px]))
                    .count(),
                frames.iter().filter(|f| !f.attitude_valid).count(),
                frames.iter().filter(|f| !f.input_available).count()
            ),
            None,
        )?;
        let reasons = counts(detections.iter().filter(|d| !d.valid).map(|d| {
            if d.invalid_reason.is_empty() {
                "unspecified".into()
            } else {
                d.invalid_reason.clone()
            }
        }));
        if !reasons.is_empty() {
            p.write(&format!("Invalid detection reasons: {reasons}"), None)?;
        }
        p.write(
            &format!(
                "Progress: console every {log_every} frames and on validity/track-count \
                    changes or invalid frames; log saves every frame. pts=input, \
                    valid=modeled, cat=known-star priors, cand=target-class points, \
                    tracks=displayed tracks, confirmed=confirmed tracks, components=target \
                    model components, bg=unknown-star components, step=tracking time, \
                    wall=step plus frame export time"
            ),
            None,
        )?;
        Ok(p)
    }
    pub fn write(&self, message: &str, console: Option<&str>) -> Result<()> {
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&self.path)
            .map_err(io_error)?;
        writeln!(f, "{message}").map_err(io_error)?;
        if self.verbose {
            if let Some(s) = console {
                eprintln!("{s}");
            }
        }
        Ok(())
    }
    pub fn frame(
        &mut self,
        f: &FrameInfo,
        row: &Value,
        result: &Value,
        points: &[&Value],
        wall: f64,
    ) -> Result<()> {
        self.committed += 1;
        let n = |k: &str| row[k].as_u64().unwrap_or(0);
        let diag = &result["diagnostics"];
        let d = |k: &str| diag[k].as_u64().unwrap_or(0);
        let state = (
            f.inertial_observation_available(),
            n("n_live_tracks") as usize,
        );
        let console = self.committed == 1
            || self.committed == self.total
            || f.frame % self.log_every as i64 == 0
            || self.previous != Some(state)
            || !f.inertial_observation_available();
        self.previous = Some(state);
        let status = if !f.input_available {
            "prediction-only: missing input frame"
        } else if !f.attitude_valid {
            "prediction-only: external attitude invalid"
        } else if n("input_detections") == 0 {
            "+att; valid empty frame"
        } else {
            "+att"
        };
        let message = format!(
            "[{:04}]  pts={}  valid={}  cat={}  stars={}  cand={}  tracks={}  \
                confirmed={}  components={}  bg={}  hyp={}  step={:.0}ms  wall={:.0}ms  \
                groups={}  bg_sleep={}  bg_retired={}  {status}  ({}/{})",
            f.frame,
            n("input_detections"),
            n("classified_detections"),
            d("KnownStar_Count"),
            n("n_stars"),
            n("n_candidates"),
            n("n_live_tracks"),
            n("n_targets"),
            n("model_target_components"),
            result["background_marginals"]
                .as_array()
                .map(Vec::len)
                .unwrap_or(0),
            d("Retained_Hypothesis_Count"),
            row["time_ms"].as_f64().unwrap_or(0.),
            wall * 1000.,
            d("Joint_Group_Count"),
            d("Dormant_Background_Count"),
            d("Retired_Background_Count"),
            self.committed,
            self.total
        );
        let validity = if !f.input_available {
            "无效（输入缺失）"
        } else if !f.attitude_valid {
            "无效（姿态无效）"
        } else if n("input_detections") == 0 {
            "有效（无检测）"
        } else {
            "有效"
        };
        let concise = format!(
            "[{:04}] {validity}  总耗时={:.0}ms  轨迹={}  恒星={}  ({}/{})",
            f.frame,
            wall * 1000.,
            n("n_live_tracks"),
            if f.inertial_observation_available() {
                n("n_stars").to_string()
            } else {
                "--".into()
            },
            self.committed,
            self.total
        );
        self.write(&message, if console { Some(&concise) } else { None })?;
        let reasons = counts(
            points
                .iter()
                .filter(|p| p["Classification_Valid"] == 0)
                .map(|p| {
                    p["Classification_Skip_Reason"]
                        .as_str()
                        .filter(|s| !s.is_empty())
                        .unwrap_or("not_in_observation_model")
                        .to_owned()
                }),
        );
        if !reasons.is_empty() {
            self.write(&format!("[{:04}] excluded: {reasons}", f.frame), None)?;
        }
        for event in result["lifecycle_events"].as_array().into_iter().flatten() {
            self.write(
                &format!(
                    "[{:04}] event: {} label={} {}->{} reason={}",
                    f.frame,
                    event["kind"].as_str().unwrap_or(""),
                    event["label"].as_str().unwrap_or(""),
                    event["before"].as_str().unwrap_or(""),
                    event["after"].as_str().unwrap_or(""),
                    event["reason"].as_str().unwrap_or("")
                ),
                None,
            )?;
        }
        for event in result["candidate_retirement_events"]
            .as_array()
            .into_iter()
            .flatten()
        {
            self.write(
                &format!(
                    "[{:04}] retired: label={} existence={:.6} reason={}",
                    f.frame,
                    event["label"].as_str().unwrap_or(""),
                    event["existence_probability"].as_f64().unwrap_or(0.),
                    event["reason"].as_str().unwrap_or("")
                ),
                None,
            )?;
        }
        Ok(())
    }
    pub fn failure(&self, failed: &Value) -> Result<()> {
        let message = format!(
            "[{:04}] FAILED: {}; previous_committed_frame={}",
            failed["frame"].as_i64().unwrap(),
            failed["error"].as_str().unwrap(),
            failed["previous_committed_frame"]
        );
        self.write(&message, Some(&message))
    }
    pub fn finish(&self, summary: &Value, report: &Value) -> Result<()> {
        let elapsed = self.started.elapsed().as_secs_f64();
        let message = format!(
            "{} | Committed frames: {}/{} | tracks={} confirmed={} | elapsed={elapsed:.2}s",
            summary["status"].as_str().unwrap(),
            summary["frames_committed"],
            summary["frames_read"],
            report["total_tracks"],
            report["formal_tracks"]
        );
        self.write(
            &message,
            Some(&format!(
                "{}：{}/{} 帧，总用时={elapsed:.2}s",
                if summary["status"] == "COMPLETED" {
                    "处理完成"
                } else {
                    "处理停止"
                },
                summary["frames_committed"],
                summary["frames_read"]
            )),
        )?;
        if let Some(path) = summary.get("checkpoint_path").filter(|p| !p.is_null()) {
            self.write(
                &format!("Checkpoint: {}", path.as_str().unwrap_or("")),
                None,
            )?;
        }
        if let Some(failed) = summary.get("failed_frame").filter(|f| !f.is_null()) {
            self.write(&format!("Stopped: {failed}"), None)?;
        }
        self.write(&format!("Debug log: {}", self.path.display()), None)
    }
}
