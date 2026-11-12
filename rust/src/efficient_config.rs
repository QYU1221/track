use crate::{
    contracts::fields,
    error::{Error, Result},
    exact_factors::inference,
};
use serde::{Serialize, Serializer};
use serde_json::Value;
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    values: Value,
}

impl Serialize for InferenceConfig {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        self.values.serialize(s)
    }
}

impl InferenceConfig {
    pub fn from_json(v: &Value) -> Result<Self> {
        let names = [
            "gate_probability",
            "top_k",
            "max_nodes",
            "max_edges",
            "max_component_nodes",
            "max_local_hypotheses",
            "max_candidates",
            "max_search_states",
            "max_memory_bytes",
            "max_frame_seconds",
        ];
        fields(
            v,
            "InferenceConfig",
            &names,
            &["association_top_k", "preserve_target_associations"],
        )?;
        let mut values = v.clone();
        if v.get("preserve_target_associations")
            .is_some_and(|x| !x.is_boolean())
        {
            return Err(Error::value("preserve_target_associations must be boolean"));
        }
        values["preserve_target_associations"] = v
            .get("preserve_target_associations")
            .cloned()
            .unwrap_or(Value::Bool(false));
        if !v["association_top_k"].is_null()
            && (v["association_top_k"].as_u64().is_none()
                || v["association_top_k"].as_f64() < v["top_k"].as_f64())
        {
            return Err(Error::value(
                "association_top_k must be null or an integer >= top_k",
            ));
        }
        values["association_top_k"] = v["association_top_k"].clone();
        if !v["gate_probability"]
            .as_f64()
            .is_some_and(|x| x > 0. && x < 1.)
        {
            return Err(Error::value(
                "gate probability must lie strictly between zero and one",
            ));
        }
        for name in &names[1..] {
            if !v[name].as_f64().is_some_and(|x| x.is_finite() && x > 0.) {
                return Err(Error::value(format!(
                    "positive finite computation budget required: {name}"
                )));
            }
            if *name != "max_frame_seconds" && v[name].as_u64().is_none() {
                return Err(Error::value(format!(
                    "integer computation budget required: {name}"
                )));
            }
        }
        if v["max_memory_bytes"].as_u64().unwrap() > 16 * 1024_u64.pow(3) {
            return Err(Error::value("runtime memory ceiling is 16 GiB"));
        }
        Ok(Self { values })
    }
    pub fn gate_probability(&self) -> f64 {
        self.values["gate_probability"].as_f64().unwrap()
    }
    pub fn limit(&self, name: &str) -> Result<u64> {
        self.values[format!("max_{name}")]
            .as_u64()
            .ok_or_else(|| Error::value("unknown integer budget"))
    }
    pub fn frame_seconds(&self) -> f64 {
        self.values["max_frame_seconds"].as_f64().unwrap()
    }
    pub fn selection_limit(&self) -> u64 {
        self.values["association_top_k"]
            .as_u64()
            .unwrap_or_else(|| self.values["top_k"].as_u64().unwrap())
    }
}

pub trait ResourceSampler {
    fn elapsed_seconds(&mut self) -> f64;
    fn working_set_bytes(&mut self) -> Result<u64>;
}

pub struct LiveResources {
    started: std::time::Instant,
}

impl Default for LiveResources {
    fn default() -> Self {
        Self {
            started: std::time::Instant::now(),
        }
    }
}

impl ResourceSampler for LiveResources {
    fn elapsed_seconds(&mut self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }
    fn working_set_bytes(&mut self) -> Result<u64> {
        crate::cancellation::check()?;
        process_memory_bytes()
    }
}

#[cfg(windows)]
pub fn process_memory_bytes() -> Result<u64> {
    #[repr(C)]
    struct Counters {
        cb: u32,
        faults: u32,
        peak: usize,
        working: usize,
        other: [usize; 6],
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(p: *mut std::ffi::c_void, c: *mut Counters, n: u32) -> i32;
    }
    let mut c = Counters {
        cb: std::mem::size_of::<Counters>() as u32,
        faults: 0,
        peak: 0,
        working: 0,
        other: [0; 6],
    };
    if unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } == 0 {
        return Err(Error {
            kind: "OSError",
            code: None,
            message: "cannot query process working set".into(),
        });
    }
    Ok(c.working as u64)
}

#[cfg(not(windows))]
pub fn process_memory_bytes() -> Result<u64> {
    Err(Error::value(
        "live memory sampling currently requires Windows",
    ))
}

pub struct InferenceBudgetController<S: ResourceSampler> {
    pub config: InferenceConfig,
    pub sampler: S,
    pub peak_sampled_rss: u64,
    pub work: u64,
}

fn python_float(x: f64) -> String {
    let s = format!("{x:?}");
    if let Some((mantissa, exp)) = s.split_once('e') {
        let exponent: i32 = exp.parse().unwrap();
        format!(
            "{mantissa}e{}{digit:02}",
            if exponent < 0 { "-" } else { "+" },
            digit = exponent.abs()
        )
    } else {
        s
    }
}

impl<S: ResourceSampler> InferenceBudgetController<S> {
    pub fn new(config: InferenceConfig, sampler: S) -> Self {
        Self {
            config,
            sampler,
            peak_sampled_rss: 0,
            work: 0,
        }
    }
    fn fail(name: &str, value: impl std::fmt::Display, limit: impl std::fmt::Display) -> Error {
        inference(
            "APPROXIMATE_BUDGET_EXCEEDED",
            &format!("{name}={value} exceeds {limit}; no silent model substitution"),
        )
    }
    pub fn check(&mut self, counts: &[(&str, u64)]) -> Result<()> {
        for &(name, value) in counts {
            let limit = self.config.limit(name)?;
            if value > limit {
                return Err(Self::fail(name, value, limit));
            }
        }
        let elapsed = self.sampler.elapsed_seconds();
        if elapsed > self.config.frame_seconds() {
            let value = &self.config.values["max_frame_seconds"];
            let limit = if value.is_i64() || value.is_u64() {
                value.to_string()
            } else {
                python_float(self.config.frame_seconds())
            };
            return Err(Self::fail("frame_seconds", python_float(elapsed), limit));
        }
        let rss = self.sampler.working_set_bytes()?;
        self.peak_sampled_rss = self.peak_sampled_rss.max(rss);
        if rss > self.config.limit("memory_bytes")? {
            return Err(Self::fail(
                "memory_bytes",
                rss,
                self.config.limit("memory_bytes")?,
            ));
        }
        Ok(())
    }
    pub fn consume(&mut self, n: u64) -> Result<()> {
        self.work = self
            .work
            .checked_add(n)
            .ok_or_else(|| inference("NUMERIC_RANGE_EXCEEDED", "work counter exceeds u64"))?;
        if self.work > self.config.limit("search_states")? {
            return Err(Self::fail(
                "search_states",
                self.work,
                self.config.limit("search_states")?,
            ));
        }
        if self.work % 128 == 0 {
            self.check(&[])?;
        }
        Ok(())
    }
}
