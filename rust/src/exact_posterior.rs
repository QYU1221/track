use crate::{
    error::{Error, Result},
    exact_factors::{inference, wire, *},
    exact_numerics::log_probability,
    existence::log_absence,
    numerical_policy::BASIS_EPS_MULTIPLIER,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    a.max(b) + libm::log1p(libm::exp(-(a - b).abs()))
}

pub(crate) fn fsum(values: impl IntoIterator<Item = f64>) -> Result<f64> {
    let mut partials: Vec<f64> = Vec::new();
    for mut x in values {
        let mut next = Vec::new();
        for mut y in partials {
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0. {
                next.push(lo);
            }
            x = hi;
        }
        if !x.is_finite() {
            return Err(inference(
                "NUMERIC_RANGE_EXCEEDED",
                "log weight exceeds numeric range",
            ));
        }
        if x != 0. {
            next.push(x);
        }
        partials = next;
    }
    let mut hi = 0.;
    let mut lo = 0.;
    while let Some(y) = partials.pop() {
        let x = hi;
        hi = x + y;
        lo = y - (hi - x);
        if lo != 0. {
            break;
        }
    }
    if partials
        .last()
        .is_some_and(|x| (*x > 0. && lo > 0.) || (*x < 0. && lo < 0.))
    {
        let y = lo * 2.;
        let x = hi + y;
        if y == x - hi {
            hi = x;
        }
    }
    Ok(hi)
}

pub fn source_log_miss_given_exists(s: &SourceFactors) -> f64 {
    s.log_miss_given_exists.unwrap_or_else(|| {
        if s.pd_bar < 1. {
            libm::log1p(-s.pd_bar)
        } else {
            f64::NEG_INFINITY
        }
    })
}

pub fn source_log_miss_factor(s: &SourceFactors) -> Result<f64> {
    Ok(log_add(
        log_absence(s.log_existence_prior)?,
        s.log_existence_prior + source_log_miss_given_exists(s),
    ))
}

fn get_source<'a>(table: &'a FactorTable, id: &str) -> Result<&'a SourceFactors> {
    table
        .sources
        .iter()
        .find(|s| s.source_id == id)
        .ok_or_else(|| Error {
            kind: "KeyError",
            code: None,
            message: format!("'{id}'"),
        })
}

fn index<T>(values: &[T], i: usize) -> Result<&T> {
    values.get(i).ok_or_else(|| Error {
        kind: "IndexError",
        code: None,
        message: "tuple index out of range".into(),
    })
}

pub fn joint_log_weight(table: &FactorTable, a: &Association) -> Result<f64> {
    let mut terms = vec![log_probability(table.parent_weight)];
    for (id, det) in &a.assignments {
        let s = get_source(table, id)?;
        terms.push(match det {
            None => source_log_miss_factor(s)?,
            Some(i) => s.log_existence_prior + index(&s.log_eta, *i)?,
        });
    }
    for i in &a.clutter_indices {
        terms.push(*index(&table.log_kappa, *i)?);
    }
    if terms.contains(&f64::NEG_INFINITY) {
        return Ok(f64::NEG_INFINITY);
    }
    let value = fsum(terms)?;
    if !value.is_finite() {
        return Err(inference("NUMERIC_RANGE_EXCEEDED", "nonfinite log weight"));
    }
    Ok(value)
}

pub fn normalize_log_weights(values: &[f64]) -> Result<(Vec<f64>, f64)> {
    if values.is_empty() || values.iter().any(|v| v.is_nan() || *v == f64::INFINITY) {
        return Err(inference(
            "INVALID_LOG_WEIGHT",
            "weights must be finite or -inf",
        ));
    }
    if values.iter().all(|v| *v == f64::NEG_INFINITY) {
        return Err(inference(
            "ZERO_EVIDENCE",
            "no hypothesis has positive model density",
        ));
    }
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let shifted: Vec<_> = values.iter().map(|v| v - max).collect();
    let normalizer = libm::log(fsum(shifted.iter().map(|v| libm::exp(*v)))?);
    let weights = shifted.iter().map(|v| libm::exp(v - normalizer)).collect();
    let evidence = max + normalizer;
    if !evidence.is_finite() {
        return Err(inference(
            "NUMERIC_RANGE_EXCEEDED",
            "evidence exceeds numeric range",
        ));
    }
    Ok((weights, evidence))
}

#[derive(Debug, Clone, Serialize)]
pub struct ConditionalExistence {
    pub(crate) reference: String,
    pub(crate) existence: f64,
    #[serde(serialize_with = "wire::float")]
    pub(crate) log_existence: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct JointPosterior {
    pub(crate) hypothesis_id: String,
    pub(crate) parent_hypothesis_id: String,
    pub(crate) target_assignments: Vec<(String, Option<DetectionKey>)>,
    pub(crate) star_assignments: Vec<(String, Option<DetectionKey>)>,
    pub(crate) unknown_star_assignments: Vec<(String, Option<DetectionKey>)>,
    pub(crate) clutter_detection_keys: Vec<DetectionKey>,
    #[serde(serialize_with = "wire::float")]
    pub(crate) log_unnormalized_weight: f64,
    pub(crate) posterior_weight: f64,
    pub(crate) conditional_state_references: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InferenceResult {
    pub(crate) hypotheses: Vec<JointPosterior>,
    pub(crate) conditional_states: Vec<ConditionalExistence>,
    pub(crate) log_predictive_evidence: f64,
}

fn state_reference(parent: &str, source: &str, key: Option<&DetectionKey>) -> String {
    let text = serde_json::to_string(&(parent, source, key)).unwrap();
    let mut ascii = String::new();
    for c in text.chars() {
        if c as u32 >= 127 {
            let mut units = [0; 2];
            for u in c.encode_utf16(&mut units) {
                ascii.push_str(&format!("\\u{u:04x}"));
            }
        } else {
            ascii.push(c);
        }
    }
    format!("STATE:{:x}", Sha256::digest(ascii.as_bytes()))
}

pub fn conditional_existence(
    table: &FactorTable,
    s: &SourceFactors,
    det: Option<usize>,
) -> Result<ConditionalExistence> {
    let key = det.map(|i| index(&table.detection_keys, i)).transpose()?;
    let reference = state_reference(&table.parent_id, &s.source_id, key);
    if det.is_some() {
        return Ok(ConditionalExistence {
            reference,
            existence: 1.,
            log_existence: 0.,
        });
    }
    let denominator = source_log_miss_factor(s)?;
    if denominator == f64::NEG_INFINITY {
        return Err(inference(
            "IMPOSSIBLE_CONDITION",
            "zero-weight miss has no conditional posterior",
        ));
    }
    let log_existence = s.log_existence_prior + source_log_miss_given_exists(s) - denominator;
    Ok(ConditionalExistence {
        reference,
        existence: libm::exp(log_existence),
        log_existence,
    })
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub table_index: usize,
    pub association: Association,
}

pub trait ComputationPolicy {
    fn set_parent_count(&mut self, count: usize);
    fn proposals(&mut self, table: &FactorTable) -> Result<Vec<Association>>;
    fn consume(&mut self) -> Result<()>;
    fn select(&mut self, plans: &[Plan], logs: &[f64]) -> Result<Vec<usize>>;
}

pub fn infer_factors(
    tables: &[FactorTable],
    policy: &mut impl ComputationPolicy,
) -> Result<InferenceResult> {
    if tables.is_empty() {
        return Err(Error::value("at least one parent required"));
    }
    let mut tables: Vec<_> = tables.iter().collect();
    tables.sort_by(|a, b| a.parent_id.cmp(&b.parent_id));
    let keys: HashSet<_> = tables[0].detection_keys.iter().collect();
    if tables
        .iter()
        .map(|t| &t.parent_id)
        .collect::<HashSet<_>>()
        .len()
        != tables.len()
        || tables
            .iter()
            .any(|t| t.detection_keys.iter().collect::<HashSet<_>>() != keys)
    {
        return Err(Error::value(
            "parents must be unique and share one detection set",
        ));
    }
    if (fsum(tables.iter().map(|t| t.parent_weight))? - 1.).abs()
        > BASIS_EPS_MULTIPLIER * f64::EPSILON
    {
        return Err(Error::value("parent prior weights must sum to 1"));
    }
    policy.set_parent_count(tables.len());
    let mut plans = Vec::new();
    for (i, t) in tables.iter().enumerate() {
        for association in policy.proposals(t)? {
            policy.consume()?;
            plans.push(Plan {
                table_index: i,
                association,
            });
        }
    }
    let logs = plans
        .iter()
        .map(|p| joint_log_weight(tables[p.table_index], &p.association))
        .collect::<Result<Vec<_>>>()?;
    let keep = policy.select(&plans, &logs)?;
    let selected = keep
        .iter()
        .map(|i| index(&logs, *i).copied())
        .collect::<Result<Vec<_>>>()?;
    let (weights, evidence) = normalize_log_weights(&selected)?;
    let mut cache: HashMap<(String, String, Option<DetectionKey>), ConditionalExistence> =
        HashMap::new();
    let mut rows = Vec::new();
    for ((i, log_weight), weight) in keep.iter().zip(selected).zip(weights) {
        let plan = index(&plans, *i)?;
        let table = tables[plan.table_index];
        let a = &plan.association;
        let (mut targets, mut stars, mut unknown, mut references) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (id, det) in &a.assignments {
            let source = get_source(table, id)?;
            let key = det
                .map(|i| index(&table.detection_keys, i).cloned())
                .transpose()?;
            if source.kind == "T" || source.kind == "S_U" {
                if source.kind == "T" {
                    targets.push((id.clone(), key.clone()));
                } else {
                    unknown.push((id.clone(), key.clone()));
                }
                if log_weight.is_finite() {
                    let cache_key = (table.parent_id.clone(), id.clone(), key);
                    if !cache.contains_key(&cache_key) {
                        cache.insert(
                            cache_key.clone(),
                            conditional_existence(table, source, *det)?,
                        );
                    }
                    references.push((id.clone(), cache[&cache_key].reference.clone()));
                }
            } else {
                stars.push((id.clone(), key));
            }
        }
        rows.push(JointPosterior {
            hypothesis_id: a.hypothesis_id.clone(),
            parent_hypothesis_id: table.parent_id.clone(),
            target_assignments: targets,
            star_assignments: stars,
            unknown_star_assignments: unknown,
            clutter_detection_keys: a
                .clutter_indices
                .iter()
                .map(|i| index(&table.detection_keys, *i).cloned())
                .collect::<Result<_>>()?,
            log_unnormalized_weight: log_weight,
            posterior_weight: weight,
            conditional_state_references: references,
        });
    }
    let ordered: BTreeMap<_, _> = cache
        .into_values()
        .map(|s| (s.reference.clone(), s))
        .collect();
    Ok(InferenceResult {
        hypotheses: rows,
        conditional_states: ordered.into_values().collect(),
        log_predictive_evidence: evidence,
    })
}
