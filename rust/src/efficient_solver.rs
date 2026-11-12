use crate::{
    contracts::Detection,
    efficient_config::{InferenceBudgetController, ResourceSampler},
    efficient_graph::{
        self as g, ConditionalAssociationComponent, SourceGeometry, UnifiedAssociationGraph,
    },
    error::{Error, Result},
    exact_factors::{Association, FactorTable},
    exact_posterior::{self as p, ComputationPolicy, Plan},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
pub type Assignments = Vec<(String, Option<usize>)>;
pub fn association(table: &FactorTable, mut assignments: Assignments) -> Result<Association> {
    assignments.sort();
    let used: HashSet<_> = assignments.iter().filter_map(|(_, j)| *j).collect();
    let mut representation = Vec::new();
    for (s, j) in &assignments {
        let key = j
            .map(|j| {
                table.detection_keys.get(j).ok_or_else(|| Error {
                    kind: "IndexError",
                    code: None,
                    message: "tuple index out of range".into(),
                })
            })
            .transpose()?;
        representation.push((s, key));
    }
    let text = serde_json::to_string(&(&table.parent_id, representation)).unwrap();
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
    let mut clutter_indices: Vec<_> = (0..table.detection_keys.len())
        .filter(|j| !used.contains(j))
        .collect();
    clutter_indices.sort_by_key(|j| &table.detection_keys[*j]);
    Ok(Association {
        hypothesis_id: format!("H:{:x}", Sha256::digest(ascii.as_bytes())),
        assignments,
        clutter_indices,
    })
}

fn allowed(
    component: &ConditionalAssociationComponent,
    graph: &UnifiedAssociationGraph,
) -> Vec<Vec<usize>> {
    component
        .sources
        .iter()
        .map(|s| {
            let mut v: Vec<_> = graph
                .edges
                .iter()
                .filter(|e| e.source_id == *s)
                .map(|e| e.detection_index)
                .collect();
            v.sort();
            v
        })
        .collect()
}

fn with_hit(used: &[usize], j: usize) -> Vec<usize> {
    let mut v = used.to_vec();
    let i = v.binary_search(&j).unwrap_err();
    v.insert(i, j);
    v
}

pub fn exact_component<S: ResourceSampler>(
    component: &ConditionalAssociationComponent,
    graph: &UnifiedAssociationGraph,
    controller: &mut InferenceBudgetController<S>,
    limit: u64,
) -> Result<(Option<Vec<Assignments>>, u128)> {
    let choices = allowed(component, graph);
    let cap = limit as u128 + 1;
    controller.consume(1)?;
    let mut lower = 1 + choices.iter().map(|x| x.len() as u128).sum::<u128>();
    if lower > limit as u128 {
        return Ok((None, cap));
    }
    if choices.len() == 1 {
        controller.consume(2 * choices[0].len() as u64 + 4)?;
        let s = &component.sources[0];
        let mut out = vec![vec![(s.clone(), None)]];
        out.extend(choices[0].iter().map(|j| vec![(s.clone(), Some(*j))]));
        return Ok((Some(out), lower));
    }
    for i in 0..choices.len() {
        for right in &choices[i + 1..] {
            controller.consume(1)?;
            let common = choices[i]
                .iter()
                .filter(|j| right.binary_search(j).is_ok())
                .count();
            lower += choices[i].len() as u128 * right.len() as u128 - common as u128;
            if lower > limit as u128 {
                return Ok((None, cap));
            }
        }
    }
    enum Task {
        Call(usize, Vec<usize>),
        Resume(usize, Vec<usize>, usize, u128),
    }
    let mut memo: HashMap<(usize, Vec<usize>), u128> = HashMap::new();
    let mut stack = vec![Task::Call(0, vec![])];
    let mut last = 0;
    while let Some(task) = stack.pop() {
        match task {
            Task::Call(i, used) => {
                controller.consume(1)?;
                if let Some(v) = memo.get(&(i, used.clone())) {
                    last = *v;
                } else if i == choices.len() {
                    last = 1;
                } else {
                    stack.push(Task::Resume(i, used.clone(), 0, 0));
                    stack.push(Task::Call(i + 1, used));
                }
            }
            Task::Resume(i, used, next, value) => {
                let value = value + last;
                let j = if next > 0 && value > limit as u128 {
                    None
                } else {
                    choices[i]
                        .iter()
                        .enumerate()
                        .skip(next)
                        .find(|(_, j)| used.binary_search(j).is_err())
                        .map(|(k, j)| (k, *j))
                };
                if let Some((k, j)) = j {
                    stack.push(Task::Resume(i, used.clone(), k + 1, value));
                    stack.push(Task::Call(i + 1, with_hit(&used, j)));
                } else {
                    last = value.min(cap);
                    memo.insert((i, used), last);
                }
            }
        }
    }
    if last > limit as u128 {
        return Ok((None, last));
    }
    let total = last;
    let mut out = vec![];
    let mut stack = vec![(0, vec![], vec![])];
    while let Some((i, used, items)) = stack.pop() {
        controller.consume(1)?;
        if i == choices.len() {
            out.push(items);
            continue;
        }
        let s = &component.sources[i];
        for &j in choices[i].iter().rev() {
            if used.binary_search(&j).is_err() {
                let mut hit = items.clone();
                hit.push((s.clone(), Some(j)));
                stack.push((i + 1, with_hit(&used, j), hit));
            }
        }
        let mut miss = items;
        miss.push((s.clone(), None));
        stack.push((i + 1, used, miss));
    }
    Ok((Some(out), total))
}

struct Scores {
    miss: HashMap<String, f64>,
    hits: HashMap<String, Vec<f64>>,
    clutter: Vec<f64>,
}

impl Scores {
    fn new(table: &FactorTable) -> Result<Self> {
        let mut miss = HashMap::new();
        let mut hits = HashMap::new();
        for s in &table.sources {
            miss.insert(s.source_id.clone(), p::source_log_miss_factor(s)?);
            hits.insert(
                s.source_id.clone(),
                s.log_eta
                    .iter()
                    .map(|v| s.log_existence_prior + v)
                    .collect(),
            );
        }
        Ok(Self {
            miss,
            hits,
            clutter: table.log_kappa.clone(),
        })
    }
    fn score(&self, items: &Assignments, indices: Option<&[usize]>) -> Result<f64> {
        let used: HashSet<_> = items.iter().filter_map(|(_, j)| *j).collect();
        let mut terms: Vec<_> = items
            .iter()
            .map(|(s, j)| match j {
                None => self.miss[s],
                Some(j) => self.hits[s][*j],
            })
            .collect();
        if let Some(indices) = indices {
            for j in indices {
                if !used.contains(j) {
                    terms.push(self.clutter[*j]);
                }
            }
        } else {
            for (j, v) in self.clutter.iter().enumerate() {
                if !used.contains(&j) {
                    terms.push(*v);
                }
            }
        }
        if terms.contains(&f64::NEG_INFINITY) {
            Ok(f64::NEG_INFINITY)
        } else {
            p::fsum(terms).map_err(|_| Error {
                kind: "OverflowError",
                code: None,
                message: "intermediate overflow in fsum".into(),
            })
        }
    }
    fn prune(
        &self,
        pool: Vec<Assignments>,
        indices: Option<&[usize]>,
        width: usize,
    ) -> Result<Vec<Assignments>> {
        let mut scored = pool
            .into_iter()
            .map(|a| Ok((self.score(&a, indices)?, a)))
            .collect::<Result<Vec<_>>>()?;
        scored.sort_by(|(x, a), (y, b)| y.partial_cmp(x).unwrap().then_with(|| a.cmp(b)));
        Ok(scored.into_iter().take(width).map(|(_, a)| a).collect())
    }
}

fn beam_component<S: ResourceSampler>(
    component: &ConditionalAssociationComponent,
    graph: &UnifiedAssociationGraph,
    scores: &Scores,
    controller: &mut InferenceBudgetController<S>,
    width: usize,
) -> Result<Vec<Assignments>> {
    let mut beam = vec![vec![]];
    for s in &component.sources {
        let choices: Vec<_> = std::iter::once(None)
            .chain(
                graph
                    .edges
                    .iter()
                    .filter(|e| e.source_id == *s)
                    .map(|e| Some(e.detection_index)),
            )
            .collect();
        let mut pool = vec![];
        for items in beam {
            for j in &choices {
                if j.is_none() || !items.iter().any(|(_, k)| k == j) {
                    controller.consume(1)?;
                    let mut a = items.clone();
                    a.push((s.clone(), *j));
                    pool.push(a);
                }
            }
        }
        beam = scores.prune(pool, Some(&component.detections), width)?;
    }
    Ok(beam)
}

fn product_count(local: &[Vec<Assignments>]) -> Value {
    let mut digits = vec![1u128];
    for values in local {
        let mut carry = 0;
        for d in &mut digits {
            let n = *d * values.len() as u128 + carry;
            *d = n % 10;
            carry = n / 10;
        }
        while carry > 0 {
            digits.push(carry % 10);
            carry /= 10;
        }
    }
    while digits.len() > 1 && digits.last() == Some(&0) {
        digits.pop();
    }
    let s: String = digits
        .into_iter()
        .rev()
        .map(|d| (b'0' + d as u8) as char)
        .collect();
    serde_json::from_str(&s).unwrap()
}

fn combine<S: ResourceSampler>(
    local: &[Vec<Assignments>],
    scores: &Scores,
    controller: &mut InferenceBudgetController<S>,
    limit: usize,
) -> Result<(Vec<Assignments>, bool, Value)> {
    let total = product_count(local);
    if total.as_u64().is_some_and(|v| v <= limit as u64) {
        let mut out = vec![];
        let mut indices = vec![0; local.len()];
        if local.iter().any(Vec::is_empty) {
            return Ok((out, true, total));
        }
        loop {
            controller.consume(1)?;
            let mut a: Assignments = local
                .iter()
                .zip(&indices)
                .flat_map(|(c, i)| c[*i].clone())
                .collect();
            a.sort();
            out.push(a);
            let mut k = indices.len();
            while k > 0 {
                k -= 1;
                indices[k] += 1;
                if indices[k] < local[k].len() {
                    break;
                }
                indices[k] = 0;
            }
            if indices.is_empty() || k == 0 && indices[0] == 0 {
                break;
            }
        }
        return Ok((out, true, total));
    }
    let mut beam = vec![vec![]];
    for choices in local {
        let mut pool = vec![];
        for a in beam {
            for b in choices {
                controller.consume(1)?;
                let mut c = a.clone();
                c.extend(b.clone());
                c.sort();
                pool.push(c);
            }
        }
        beam = scores.prune(pool, None, limit)?;
    }
    Ok((beam, false, total))
}

pub fn prune(logs: &[f64], ids: &[String], limit: usize) -> Result<(Vec<usize>, f64, f64)> {
    let (weights, evidence) = p::normalize_log_weights(logs)?;
    if ids.len() != logs.len() {
        return Err(Error::value("hypothesis ID count mismatch"));
    }
    let mut keep: Vec<_> = (0..logs.len()).collect();
    keep.sort_by(|&i, &j| {
        logs[j]
            .partial_cmp(&logs[i])
            .unwrap()
            .then_with(|| ids[i].cmp(&ids[j]))
    });
    keep.truncate(limit);
    let kept: HashSet<_> = keep.iter().copied().collect();
    let omitted = p::fsum(
        weights
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !kept.contains(i))
            .map(|(_, w)| w),
    )?;
    Ok((keep, omitted, evidence))
}

pub struct GraphInferencePolicy<S: ResourceSampler> {
    pub controller: InferenceBudgetController<S>,
    detections: Vec<Detection>,
    geometry: BTreeMap<String, SourceGeometry>,
    detail: String,
    pub records: Vec<Value>,
    pub complete: bool,
    pub audit: Value,
    parent_count: usize,
}

impl<S: ResourceSampler> GraphInferencePolicy<S> {
    pub fn new(
        controller: InferenceBudgetController<S>,
        detections: Vec<Detection>,
        geometry: BTreeMap<String, SourceGeometry>,
        detail: &str,
    ) -> Self {
        Self {
            controller,
            detections,
            geometry,
            detail: detail.into(),
            records: vec![],
            complete: true,
            audit: json!({}),
            parent_count: 1,
        }
    }
}

impl<S: ResourceSampler> ComputationPolicy for GraphInferencePolicy<S> {
    fn set_parent_count(&mut self, count: usize) {
        self.parent_count = count;
    }
    fn consume(&mut self) -> Result<()> {
        self.controller.consume(1)
    }
    fn proposals(&mut self, table: &FactorTable) -> Result<Vec<Association>> {
        let c = self.controller.config.clone();
        let geometry = self
            .geometry
            .get(&table.parent_id)
            .cloned()
            .unwrap_or_default();
        let graph = g::build(
            table,
            &c,
            &self.detections,
            &geometry,
            self.detail == "full",
        )?;
        let components = g::components(&graph);
        let max_nodes = components
            .iter()
            .map(|x| x.sources.len() + x.detections.len())
            .max()
            .unwrap_or(0);
        self.controller.check(&[
            (
                "nodes",
                (graph.source_ids.len() + graph.detection_keys.len()) as u64,
            ),
            ("edges", graph.edges.len() as u64),
            ("component_nodes", max_nodes as u64),
        ])?;
        let scores = Scores::new(table)?;
        let per_parent = (c.limit("candidates")? / self.parent_count.max(1) as u64).max(1) as usize;
        let mut local_complete = true;
        let mut counts = vec![];
        let (combos, combo_complete, total) = if graph.edges.is_empty() {
            counts = vec![1; components.len()];
            self.controller.consume(
                5 * graph.source_ids.len() as u64 + 3 * graph.detection_keys.len() as u64 + 1,
            )?;
            (
                vec![graph.source_ids.iter().map(|s| (s.clone(), None)).collect()],
                true,
                json!(1),
            )
        } else {
            let mut local = vec![];
            for comp in &components {
                let (values, count) = exact_component(
                    comp,
                    &graph,
                    &mut self.controller,
                    c.limit("local_hypotheses")?,
                )?;
                counts.push(count);
                local.push(match values {
                    Some(v) => v,
                    None => {
                        local_complete = false;
                        beam_component(comp, &graph, &scores, &mut self.controller, per_parent)?
                    }
                });
            }
            combine(&local, &scores, &mut self.controller, per_parent)?
        };
        let complete = local_complete
            && combo_complete
            && !graph
                .excluded_edges
                .iter()
                .any(|e| e.2 == "FINITE_GATE_POSSIBLE_NONZERO");
        self.complete &= complete;
        if self.detail == "full" {
            self.records.push(json!({
                "parent_id": table.parent_id,
                "nodes": graph.source_ids.len()+graph.detection_keys.len(),
                "edges": graph.edges.len(),
                "components": components.len(),
                "max_component_nodes": max_nodes,
                "local_hypothesis_counts_capped": counts,
                "candidate_product_count": total,
                "candidates": combos.len(),
                "excluded_edges": graph.excluded_edges,
                "gate_semantics": concat!(
                    "ASSOCIATION_HYPOTHESIS_TRUNCATION; ",
                    "ORIGINAL_MISS_UNCHANGED; ",
                    "NOT_A_RENORMALIZED_SENSOR_MODEL",
                ),
                "gate_probability": c.gate_probability(),
                "gate_tail_is_posterior_error_bound": false,
                "ranking": if local_complete && combo_complete {
                    "EXHAUSTIVE"
                } else {
                    "HEURISTIC_BEAM; GLOBAL_TOP_K_NOT_GUARANTEED"
                },
                "complete_relative_to_original_factors": complete
            }));
        }
        combos.into_iter().map(|a| association(table, a)).collect()
    }
    fn select(&mut self, plans: &[Plan], logs: &[f64]) -> Result<Vec<usize>> {
        let ids = plans
            .iter()
            .map(|p| p.association.hypothesis_id.clone())
            .collect::<Vec<_>>();
        let (keep, omitted, evidence) = prune(
            logs,
            &ids,
            self.controller.config.selection_limit() as usize,
        )?;
        if self.detail == "compact" {
            self.audit = json!({
                "evaluated_candidates": plans.len(),
                "retained_candidates": keep.len(),
                "discarded_mass_within_evaluated_candidates": omitted,
                "omitted_mass_status": concat!(
                    "CANDIDATE_CONDITIONAL_ONLY; ",
                    "PRE_GATED_FACTORS_AND_BEAM_LOSS_UNKNOWN",
                )
            });
            return Ok(keep);
        }
        self.audit = json!({
            "mode": "APPROXIMATE_FAST",
            "graphs": self.records,
            "evaluated_candidates": plans.len(),
            "retained_candidates": keep.len(),
            "candidate_log_evidence": evidence,
            "complete_model_log_evidence": if self.complete{Some(evidence)}else{None},
            "discarded_mass_within_evaluated_candidates": omitted,
            "omitted_mass": if self.complete{Some(omitted)}else{None},
            "omitted_mass_status": if self.complete {
                "EXACT_FOR_THIS_FRAME_GIVEN_INPUT_PRIOR"
            } else {
                "OMITTED_MASS_UNKNOWN"
            },
            "ranking_guarantee": if self.complete {
                "GLOBAL_TRUE_TOP_K_GIVEN_INPUT_PRIOR"
            } else {
                "ONLY_TRUE_WEIGHT_RANKING_WITHIN_CANDIDATES"
            },
            "approximate": !self.complete||(keep.len()<plans.len()&&omitted>0.),
            "search_states": self.controller.work,
            "continuous_status": "DECLARED_ONLINE_CONTINUOUS_APPROXIMATIONS"
        });
        Ok(keep)
    }
}
