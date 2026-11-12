use crate::{
    configuration::ConfigurationResolver,
    data_io::{self, io_error},
    efficient_config::{LiveResources, ResourceSampler},
    error::{Error, Result},
    reporter::Reporter,
    sequence_session::Session,
};
use serde_json::{json, Value};

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Instant,
};

fn application_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("cannot locate running executable");
    let dir = exe.parent().expect("executable has no directory");
    if matches!(
        dir.file_name().and_then(|s| s.to_str()),
        Some("debug" | "release")
    ) {
        if let Some(target) = dir.parent().filter(|p| p.ends_with("target")) {
            if let Some(project) = target.parent().filter(|p| p.join("Cargo.toml").is_file()) {
                return project.to_owned();
            }
        }
    }
    dir.to_owned()
}

pub fn data_root() -> PathBuf {
    std::env::var_os("SPACE_TARGET_DATA_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| application_dir().join("resources"))
}

pub fn datasets_root() -> PathBuf {
    let resources = data_root().join("datasets");
    if std::env::var_os("SPACE_TARGET_DATA_ROOT").is_some() || resources.is_dir() {
        return resources;
    }
    application_dir().join("datasets")
}

pub fn source_hashes(prefix: Option<&str>) -> BTreeMap<String, String> {
    let entries: &[(&str, &str)] = include!(concat!(env!("OUT_DIR"), "/native_source_hashes.rs"));
    entries
        .iter()
        .filter(|(path, _)| prefix.is_none_or(|p| path.rsplit('/').next().unwrap().starts_with(p)))
        .map(|(path, digest)| ((*path).to_owned(), (*digest).to_owned()))
        .collect()
}

#[derive(Debug, Clone)]
pub struct Options {
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    pub config: Option<PathBuf>,
    pub metadata: Option<PathBuf>,
    pub frame_metadata: Option<PathBuf>,
    pub resume: Option<PathBuf>,
    pub checkpoint: Option<PathBuf>,
    pub frame_limit: Option<usize>,
    pub verbose: bool,
    pub log_every: usize,
}

fn output_timestamp() -> Result<String> {
    let utc = chrono::DateTime::<chrono::Utc>::from(std::time::SystemTime::now());
    #[cfg(windows)]
    {
        use chrono::{Datelike, Timelike};
        #[repr(C)]
        struct SystemTime {
            year: u16,
            month: u16,
            weekday: u16,
            day: u16,
            hour: u16,
            minute: u16,
            second: u16,
            milliseconds: u16,
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn SystemTimeToTzSpecificLocalTime(
                zone: *const std::ffi::c_void,
                utc: *const SystemTime,
                local: *mut SystemTime,
            ) -> i32;
        }
        let time = SystemTime {
            year: utc.year() as u16,
            month: utc.month() as u16,
            weekday: 0,
            day: utc.day() as u16,
            hour: utc.hour() as u16,
            minute: utc.minute() as u16,
            second: utc.second() as u16,
            milliseconds: 0,
        };
        let mut local = SystemTime {
            year: 0,
            month: 0,
            weekday: 0,
            day: 0,
            hour: 0,
            minute: 0,
            second: 0,
            milliseconds: 0,
        };
        if unsafe { SystemTimeToTzSpecificLocalTime(std::ptr::null(), &time, &mut local) } == 0 {
            return Err(io_error(std::io::Error::last_os_error()));
        }
        Ok(format!(
            "{:04}{:02}{:02}_{:02}{:02}{:02}_{:06}",
            local.year,
            local.month,
            local.day,
            local.hour,
            local.minute,
            local.second,
            utc.timestamp_subsec_micros()
        ))
    }
    #[cfg(not(windows))]
    Ok(utc.format("%Y%m%d_%H%M%S_%6f").to_string())
}

pub fn default_output_path(input: &Path) -> Result<PathBuf> {
    let name = input
        .file_name()
        .ok_or_else(|| Error::value("input filename missing"))?;
    let executable = std::env::current_exe().map_err(io_error)?;
    let directory = executable
        .parent()
        .ok_or_else(|| Error::value("executable directory missing"))?;
    let project = Path::new(env!("CARGO_MANIFEST_DIR"));
    let package = if executable.starts_with(project.join("target")) {
        project.join("dist")
    } else {
        directory.to_owned()
    };
    Ok(package.join("output").join(format!(
        "{}_{}",
        output_timestamp()?,
        name.to_string_lossy()
    )))
}

fn exists_error(message: &str) -> Error {
    Error {
        kind: "FileExistsError",
        code: None,
        message: message.into(),
    }
}

fn abs(p: &Path) -> Result<PathBuf> {
    std::path::absolute(p).map_err(io_error)
}

pub fn process_sequence(options: &Options) -> Result<Value> {
    process_with_sampler(options, LiveResources::default)
}

pub fn process_with_sampler<S: ResourceSampler, F: FnMut() -> S>(
    options: &Options,
    sampler: F,
) -> Result<Value> {
    if options.log_every == 0 {
        return Err(Error::value(
            "log_every / --log-every must be a positive integer",
        ));
    }
    if options.frame_limit == Some(0) {
        return Err(Error::value(
            "end_frame / --frames must be a positive number of input frames",
        ));
    }
    let package = data_root();
    let default = package.join("pipeline/profiles/default.json");
    let input = abs(&options.input)?;
    let automatic = options.config.is_none();
    if automatic && input.is_dir() {
        return Err(Error::value(
            "V2 公共输入目录需要 --config / config_path 指定先验 JSON",
        ));
    }
    let config_path = abs(options.config.as_deref().unwrap_or(&default))?;
    let raw = data_io::read_json(&config_path)?;
    if !raw.is_object() {
        return Err(Error::value("V2 prior configuration must be a JSON object"));
    }
    let mut raw = ConfigurationResolver::new(default).resolve(&raw)?;
    if !input.exists() {
        return Err(Error {
            kind: "FileNotFoundError",
            code: None,
            message: format!("V2 input does not exist: {}", input.display()),
        });
    }
    if options.verbose {
        eprintln!("读取输入：{}", input.display());
    }
    let mut packet = if input.is_dir() {
        if options.metadata.is_some() || options.frame_metadata.is_some() {
            return Err(Error::value(
                "normalized public input already contains metadata and frames",
            ));
        }
        data_io::load_packet(
            &input,
            raw["input_metadata"]["max_rows"].as_u64().unwrap() as usize,
        )?
    } else {
        let metadata = if let Some(path) = &options.metadata {
            data_io::read_json(path)?
        } else {
            raw["input_metadata"].clone()
        };
        let cfg = data_io::NormalizeConfig::from_json(&metadata)?;
        raw["input_metadata"] = json!(cfg);
        raw["input_metadata"]["camera"] = metadata["camera"].clone();
        data_io::normalize_table(&input, &cfg, options.frame_metadata.as_deref())?.0
    };
    let mut packet_json = json!({
        "camera": packet.camera, "time_scale": packet.time_scale,
        "frames": packet.frames, "catalog": packet.catalog,
        "origin": packet.origin, "detections": [],
    });
    if input.is_file() {
        packet_json["camera"] = raw["input_metadata"]["camera"].clone();
    }
    if input.is_dir() {
        let meta = data_io::read_json(&input.join("input_metadata.json"))?;
        packet_json["camera"] = meta["camera"].clone();
    }
    let mut effective = raw.clone();
    let bootstrap = raw.get("bootstrap").is_some();
    if bootstrap {
        let (bound, catalog) =
            crate::profile_binding::bind_profile(&packet_json, &raw, &config_path, None)?;
        effective = bound;
        packet_json["catalog"] = json!(catalog);
    }
    if let Some(path) = effective["catalog_path"].as_str().filter(|s| !s.is_empty()) {
        let path = Path::new(path);
        let path = if path.is_absolute() {
            path.to_owned()
        } else {
            config_path.parent().unwrap().join(path)
        };
        effective["catalog_path"] = json!(path.to_string_lossy());
        if !bootstrap {
            packet_json["catalog"] = json!(crate::exact_io::public_navstar_catalog(
                &path,
                effective["catalog_mag_limit"].as_f64()
            )?);
        }
    }
    packet.catalog = packet_json["catalog"]
        .as_array()
        .unwrap()
        .iter()
        .map(crate::contracts::KnownStar::from_json)
        .collect::<Result<Vec<_>>>()?;
    let packet = std::sync::Arc::new(packet);
    let resume = options
        .resume
        .as_deref()
        .map(data_io::read_json)
        .transpose()?;
    let mut selected = packet
        .frames
        .iter()
        .filter(|f| {
            raw.get("frames").is_none_or(|a| {
                a.as_array()
                    .is_some_and(|a| a.iter().any(|n| n.as_i64() == Some(f.frame)))
            })
        })
        .map(|f| f.frame)
        .collect::<Vec<_>>();
    if let Some(frames) = raw.get("frames") {
        let a = frames
            .as_array()
            .ok_or_else(|| Error::value("requested frames missing or repeated"))?;
        if a.len() != selected.len()
            || a.iter()
                .any(|n| !selected.iter().any(|f| n.as_i64() == Some(*f)))
        {
            return Err(Error::value("requested frames missing or repeated"));
        }
    }
    if selected.is_empty() {
        return Err(Error::value("no requested frames"));
    }
    if let Some(limit) = options.frame_limit {
        if let Some(r) = &resume {
            if let Some(n) = r["snapshot"]["frame"].as_i64() {
                selected.retain(|f| *f > n);
            }
        }
        if selected.is_empty() {
            return Err(Error::value("no frames after restored checkpoint"));
        }
        selected.truncate(limit);
        effective["frames"] = json!(selected);
    }
    let folder = if let Some(path) = &options.output {
        abs(path)?
    } else {
        default_output_path(&input)?
    };
    if folder.exists()
        && (!folder.is_dir()
            || std::fs::read_dir(&folder)
                .map_err(io_error)?
                .next()
                .is_some())
    {
        return Err(exists_error(&format!(
            "V2 tracking output must be an empty directory: {}",
            folder.display()
        )));
    }
    if options.checkpoint.is_some() {
        return Err(Error::value(
            "checkpoint output is disabled; JSON output has been removed",
        ));
    }
    std::fs::create_dir_all(&folder).map_err(io_error)?;
    let mut session =
        Session::new_shared(&packet_json, packet.clone(), &effective, resume.as_ref())?;
    let selected = session.selected_frames().to_vec();
    effective["bound_input"] = json!({
        "camera": packet_json["camera"],
        "time_scale": packet_json["time_scale"]
    });
    drop(packet_json);
    let mut presentation_config = effective.clone();
    presentation_config["frames"] = json!(selected);
    let mut reporter = Reporter::new(
        &folder,
        packet.clone(),
        presentation_config,
        &input.to_string_lossy(),
    )?;
    let mut progress = crate::progress::Progress::new(
        &folder,
        &packet,
        &selected,
        &input.to_string_lossy(),
        options.verbose,
        options.log_every,
    )?;
    progress.write(
        &format!(
            "Memory after initialization: {} bytes",
            crate::efficient_config::process_memory_bytes()?
        ),
        None,
    )?;
    let sources = source_hashes(None);
    let mut times = Vec::new();
    let started = Instant::now();
    let cpu = cpu_time();
    let failed = session.run(sampler, |mut result| {
        crate::report_ledger::numeric_markers(&mut result)?;
        let n = result["frame"].as_i64().unwrap();
        let f = packet.frames.iter().find(|f| f.frame == n).unwrap();
        let frame_start = Instant::now();
        let mut row = result["diagnostics"].clone();
        row["frame"] = json!(n);
        row["input_detections"] = json!(packet.detections.iter().filter(|d| d.frame == n).count());
        row["modeled_detections"] = json!(result["points"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p["Geometry_Valid"] == true)
            .count());
        row["valid_attitude"] = json!(f.attitude_valid);
        row["valid_observation"] = json!(f.inertial_observation_available());
        times.push(row["Computation_Time_s"].as_f64().unwrap());
        reporter.consume_with_diagnostics(&result, &row)?;
        if times.len() % 500 == 0 {
            progress.write(
                &format!(
                    "Memory after {} frames: {} bytes",
                    times.len(),
                    crate::efficient_config::process_memory_bytes()?
                ),
                None,
            )?;
        }
        let point_rows = reporter
            .records
            .points
            .iter()
            .filter(|((frame, _), _)| *frame == n)
            .map(|(_, p)| p)
            .collect::<Vec<_>>();
        progress.frame(
            f,
            reporter.records.frames.last().unwrap(),
            &result,
            &point_rows,
            frame_start.elapsed().as_secs_f64()
                + result["diagnostics"]["Computation_Time_s"]
                    .as_f64()
                    .unwrap(),
        )?;
        Ok(())
    })?;
    if let Some(failed) = &failed {
        progress.failure(failed)?;
    }
    let state = &session.tracker.state;
    let labels = state
        .groups
        .iter()
        .flat_map(|g| g.iter())
        .flat_map(|h| h.tracks.iter().map(|t| t.label.clone()))
        .collect::<std::collections::BTreeSet<_>>();
    times.sort_by(f64::total_cmp);
    let quantiles = if times.is_empty() {
        json!({})
    } else {
        json!({
                    "50": percentile(&times,
        0.5),
                    "95": percentile(&times,
        0.95),
                    "99": percentile(&times,
        0.99)
                })
    };
    if input.is_file() {
        data_io::write_packet_csvs(&folder.join("public_input"), &packet)?;
    }
    let mut summary = json!({
        "model_version": crate::point_marginals::MODEL_VERSION,
        "tracker_engine": "factorized_joint_ppp",
        "status": if failed.is_none() {
            "COMPLETED"
        } else {
            "STOPPED_WITH_PREVIOUS_STATE_RETAINED"
        },
        "failed_frame": failed,
        "frames_read": selected.len(),
        "frames_committed": times.len(),
        "detections_read": packet
            .detections
            .iter()
            .filter(|d| selected.contains(&d.frame))
            .count(),
        "segment_count": session.tracker.segment_events.len()+1,
        "segment_events": session.tracker.segment_events,
        "segmentation_policy": "REINITIALIZE_ON_PREDICTION_GAP; NO_CROSS_SEGMENT_IDENTITY",
        "valid_attitude_frames": packet
            .frames
            .iter()
            .filter(|f| selected.contains(&f.frame) && f.attitude_valid)
            .count(),
        "valid_empty_frames": packet
            .frames
            .iter()
            .filter(|f| {
                selected.contains(&f.frame)
                    && f.inertial_observation_available()
                    && !packet.detections.iter().any(|d| d.frame == f.frame)
            })
            .count(),
        "elapsed_s": started.elapsed().as_secs_f64(),
        "cpu_time_s": cpu_time()-cpu,
        "configuration": effective,
        "labels": labels,
        "approximation_scope": crate::point_marginals::APPROXIMATION,
        "parameter_scope": {
            "mode": "FIXED_DECLARED_PARAMETERS",
            "online_learning": "NOT_IMPLEMENTED"
        },
        "real_accuracy_status": "UNVERIFIED_WITH_AVAILABLE_TRUTH",
        "retired_candidate_count": state.retired.len(),
        "retired_background_count": state.background_retired_count,
        "dormant_background_count": state.dormant_background_ids.len(),
        "identity_scope": "PERSISTENT_RETAINED_LABELS; ARCHIVE_REIDENTIFICATION_NOT_IMPLEMENTED",
        "frame_json_written": false,
        "performance_quantiles_s": quantiles,
        "source_sha256": sources,
        "source_hash_scope": "ALL_NATIVE_RUST_SOURCE_FILES_AT_BUILD_TIME",
        "resumed_from": options.resume.as_ref().map(|p|p.to_string_lossy())
    });
    progress.write(
        &format!(
            "Memory before final export: {} bytes",
            crate::efficient_config::process_memory_bytes()?
        ),
        None,
    )?;
    summary["presentation"] = reporter.finish(&summary)?;
    progress.write(
        &format!(
            "Memory after final export: {} bytes",
            crate::efficient_config::process_memory_bytes()?
        ),
        None,
    )?;
    progress.finish(&summary, &summary["presentation"])?;
    summary["output_dir"] = json!(folder.to_string_lossy());
    Ok(summary)
}

pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    let index = (sorted.len() - 1) as f64 * q;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    let t = index - low as f64;
    sorted[low] + (sorted[high] - sorted[low]) * t
}

#[cfg(windows)]
fn cpu_time() -> f64 {
    #[repr(C)]
    struct Ft {
        low: u32,
        high: u32,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn GetProcessTimes(
            p: *mut std::ffi::c_void,
            c: *mut Ft,
            e: *mut Ft,
            k: *mut Ft,
            u: *mut Ft,
        ) -> i32;
    }
    let mut a = [
        Ft { low: 0, high: 0 },
        Ft { low: 0, high: 0 },
        Ft { low: 0, high: 0 },
        Ft { low: 0, high: 0 },
    ];
    unsafe {
        let p = a.as_mut_ptr();
        if GetProcessTimes(GetCurrentProcess(), p, p.add(1), p.add(2), p.add(3)) == 0 {
            return 0.;
        }
    }
    let value = |f: &Ft| ((f.high as u64) << 32) | f.low as u64;
    (value(&a[2]) + value(&a[3])) as f64 / 1e7
}

#[cfg(not(windows))]
fn cpu_time() -> f64 {
    extern "C" {
        fn clock() -> std::ffi::c_long;
    }
    unsafe { clock() as f64 / 1_000_000. }
}
