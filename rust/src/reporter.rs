use crate::{
    contracts::ObservationPacket, data_io::io_error, error::Result, report_files, report_plots,
    report_records::Records,
};
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

pub struct Reporter {
    pub records: Records,
    folder: PathBuf,
    source: String,
    pub export_seconds: f64,
    history: std::rc::Rc<crate::report_store::ReportStore>,
    classified_detections: i64,
}

impl Reporter {
    pub fn new(
        folder: &Path,
        packet: impl Into<std::sync::Arc<ObservationPacket>>,
        configuration: Value,
        source: &str,
    ) -> Result<Self> {
        for name in ["formal_tracks", "debug", "frame_results"] {
            std::fs::create_dir_all(folder.join(name)).map_err(io_error)?;
        }
        let history = std::rc::Rc::new(crate::report_store::ReportStore::new(
            &folder.join("debug/report_history.sqlite"),
        )?);
        let mut records = Records::new(packet, configuration)?;
        records.history = Some(history.clone());
        Ok(Self {
            records,
            folder: folder.into(),
            source: source.into(),
            export_seconds: 0.,
            history,
            classified_detections: 0,
        })
    }
    fn detection_rows(
        &self,
        number: Option<i64>,
        emit: &mut dyn FnMut(Value) -> Result<()>,
    ) -> Result<()> {
        let requested = self.records.configuration["frames"].as_array().map(|rows| {
            rows.iter()
                .filter_map(Value::as_i64)
                .collect::<std::collections::HashSet<_>>()
        });
        let mut cached_frame = None;
        let mut cached_points = Vec::new();
        for d in &self.records.packet.detections {
            if number.is_some_and(|n| n != d.frame)
                || requested
                    .as_ref()
                    .is_some_and(|frames| !frames.contains(&d.frame))
            {
                continue;
            }
            if cached_frame != Some(d.frame) {
                cached_points = self.history.points(d.frame)?;
                cached_frame = Some(d.frame);
            }
            let mut row = if let Some((_, point)) = cached_points
                .iter()
                .find(|((_, id), _)| id == &d.detection_id)
            {
                point.clone()
            } else {
                let frame = self
                    .records
                    .packet
                    .frames
                    .iter()
                    .find(|f| f.frame == d.frame)
                    .unwrap();
                let mut point =
                    self.records
                        .detection_row(d, frame, &json!({}), &json!({}), None)?;
                point["Classification_Skip_Reason"] = json!("not_processed");
                point["Model_Status"] = json!("NOT_PROCESSED");
                point
            };
            if number.is_some() {
                row["Final_Class"] = Value::Null;
            } else {
                self.records.finalize_point(&mut row);
            }
            emit(row)?;
        }
        Ok(())
    }
    pub fn consume_with_diagnostics(&mut self, result: &Value, diagnostics: &Value) -> Result<()> {
        let started = Instant::now();
        self.records.consume(result)?;
        let n = result["frame"].as_i64().unwrap();
        let tracks = self
            .records
            .tracks
            .iter()
            .filter_map(|track| {
                track
                    .rows
                    .last()
                    .filter(|row| row["frame"] == n)
                    .map(|row| (track.id, row.clone()))
            })
            .collect::<Vec<_>>();
        self.history.commit_frame(
            n,
            &self.records.points,
            &self.records.events,
            &tracks,
            self.records.frames.last().unwrap(),
            diagnostics,
        )?;
        self.classified_detections += self
            .records
            .points
            .iter()
            .map(|(_, p)| p["Classification_Valid"].as_i64().unwrap())
            .sum::<i64>();
        let format = self.records.configuration["reporting"]["table_format"]
            .as_str()
            .unwrap();
        report_files::workbook_stream(
            &self
                .folder
                .join("frame_results")
                .join(format!("result_{n:06}.xlsx")),
            format,
            &[],
            |emit| self.detection_rows(Some(n), emit),
        )?;
        report_plots::frame(&self.folder, &self.records, n)?;
        self.export_seconds += started.elapsed().as_secs_f64();
        Ok(())
    }
    pub fn finish(&mut self, summary: &Value) -> Result<Value> {
        let started = Instant::now();
        let format_owned = self.records.configuration["reporting"]["table_format"]
            .as_str()
            .unwrap()
            .to_owned();
        let format = format_owned.as_str();
        let mut trajectory = report_files::CsvWriter::new(
            &self.folder.join("trajectory_points.csv"),
            "trajectory",
            format,
        )?;
        let tracks = self
            .records
            .finish_records_stream(&mut |row| trajectory.write(&row))?;
        trajectory.finish()?;
        report_files::csv(
            &self.folder.join("track_summary.csv"),
            "track",
            format,
            &tracks,
        )?;
        let mut frames =
            report_files::CsvWriter::new(&self.folder.join("frame_summary.csv"), "frame", format)?;
        self.history
            .frame_records("statistics", |row| frames.write(&row))?;
        frames.finish()?;
        let mut events =
            report_files::CsvWriter::new(&self.folder.join("track_events.csv"), "event", format)?;
        self.history.events(|mut event| {
            self.records.finalize_event(&mut event);
            events.write(&event)
        })?;
        events.finish()?;
        let segments = summary["segment_events"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        report_files::workbook_stream(
            &self.folder.join("detection_results.xlsx"),
            format,
            &segments,
            |emit| self.detection_rows(None, emit),
        )?;
        let figures = report_plots::finish(&self.folder, &self.records)?
            .into_iter()
            .map(|p| {
                if cfg!(windows) {
                    p.replace('/', "\\")
                } else {
                    p
                }
            })
            .collect::<Vec<_>>();
        self.export_seconds += started.elapsed().as_secs_f64();
        let requested = self.records.configuration["frames"].as_array();
        let input = self
            .records
            .packet
            .detections
            .iter()
            .filter(|d| requested.is_none_or(|a| a.iter().any(|n| n.as_i64() == Some(d.frame))))
            .collect::<Vec<_>>();
        let formal = self
            .records
            .tracks
            .iter()
            .filter(|t| t.final_formal)
            .count();
        let nstatus = |s: &str| {
            self.records
                .frame_status_counts
                .get(s)
                .copied()
                .unwrap_or(0)
        };
        let mut report = json!({
                   "input_file": self.source,
                   "total_frames": self.records.frame_count,
                   "input_frames": self.records.frame_count,
                   "processed_frames": nstatus("processed"),
                   "skipped_frames": nstatus("skipped_invalid_attitude"),
                   "missing_input_frames": nstatus("missing_input_frame"),
                   "total_detections": input.len(),
                   "classified_detections": self.classified_detections,
                   "valid_detections": input.iter().filter(|d|d.valid).count(),
                   "out_of_bounds_count": input
                       .iter()
                       .filter(|d| !self.records.packet.camera.contains([d.x_px,
        d.y_px]))
                       .count(),
                   "total_tracks": self.records.tracks.len(),
                   "formal_tracks": formal,
                   "confirmed_targets": formal,
                   "likely_targets": 0,
                   "unknown_tracks": self.records.tracks.len()-formal,
                   "status": summary["status"],
                   "failed_frame": summary["failed_frame"],
                   "segment_count": summary.get("segment_count").cloned().unwrap_or(json!(1)),
                   "segment_events": segments,
                   "requested_frames": summary["frames_read"],
                   "frames_committed": summary["frames_committed"],
                   "output_style": if format=="x_legacy"{"X"}else{"V2"},
                   "table_format": format,
                   "model_version": summary["model_version"],
                   "reporting_elapsed_s": self.export_seconds,
                   "report_scope": concat!(
                       "current committed chunk; unprocessed input rows are ",
                       "explicitly marked",
                   ),
                   "track_id_scope": concat!(
                       "numeric display ID for this export; track_label and ",
                       "canonical_track_id retain V2 identity",
                   ),
                   "classification_rule": concat!(
                       "unique maximum category posterior; target winners need posterior >= ",
                       "0.85; clutter winners need posterior >= 0.85; star winners require the ",
                       "separate star classification policy; ties/unavailable are unknown; ",
                       "formal target tracks need a confirmed identity with >= 10 observed ",
                       "points, mean target posterior >= 0.85 and at most 10% ",
                       "covariance-normalized CV residual outliers; later misses or leaving the ",
                       "field of view do not revoke target identity",
                   ),
                   "star_classification_policy": {
                       "candidate_support": 0.70,
                       "confirmed_support": 0.95,
                       "known_star_evidence": "unique catalog source association",
                       "unknown_star_evidence": concat!(
                           "joint current association and >=4 distinct measured ",
                           "frames within last 8 frame numbers",
                       ),
                       "truncation_guard": concat!(
                           "posterior multiplied by retained evaluated mass; ",
                           "missing mass prevents confirmation",
                       ),
                       "calibration": "UNCALIBRATED_ENGINEERING_THRESHOLDS",
                       "error_scope": "gate and heuristic search error remains unknown"
                   },
                   "final_class_scope": concat!(
                       "final display class within this export; not truth or ",
                       "smoothed identity",
                   ),
                   "reconnect_scope": concat!(
                       "reliable same-label recovery after a measurement gap; ",
                       "not archive reidentification",
                   ),
                   "unsupported_reference_scores": if format == "x_legacy" {
                       "blank; no replacement scores are manufactured"
                   } else {
                       "omitted"
                   },
                   "reporting_source_sha256": crate::production::source_hashes(Some("report")),
                   "figures": figures
               });
        let reporting = &self.records.configuration["reporting"];
        let alpha = reporting["jump_false_alarm_probability"].as_f64().unwrap();
        report["jump_classification_policy"] = json!({
            "method": "THREE_OBSERVATION_CV_MAHALANOBIS_CHI_SQUARE_2",
            "false_alarm_probability": alpha,
            "threshold_squared": crate::report_identity::jump_threshold_squared(alpha)?,
            "covariance_rule": "S=R_next+(1+a)^2*R_current+a^2*R_previous; a=dt_next/dt_previous",
            "covariance_priority": concat!(
                "detection covariance; explicit reporting covariance; ",
                "uncalibrated tracking R fallback",
            ),
            "reporting_covariance_px2": reporting["jump_measurement_covariance_px2"],
            "reporting_covariance_provenance": reporting["jump_measurement_covariance_provenance"],
            "assumptions": concat!(
                "independent Gaussian measurement errors; locally ",
                "constant velocity; deterministic attitude",
            ),
            "scope": concat!(
                "single residual probability conditional on calibrated covariance; ",
                "overlapping triples are correlated",
            ),
            "max_jump_fraction": crate::report_identity::MAX_JUMP_FRACTION,
            "fraction_calibration": "UNCALIBRATED_ENGINEERING_QUALITY_RULE",
            "missing_covariance": "unavailable check cannot confirm identity",
        });
        Ok(report)
    }
}
