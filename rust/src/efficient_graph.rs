use crate::{
    contracts::{checked_covariance, Detection},
    efficient_config::InferenceConfig,
    error::{Error, Result},
    exact_factors::{wire, DetectionKey, FactorTable},
    linalg::mat,
};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
pub fn threshold(p: f64) -> Result<f64> {
    if !(p > 0. && p < 1.) {
        return Err(Error::value("0 < P_G < 1 required"));
    }
    Ok(-2. * (-p).ln_1p())
}

pub fn distances(residuals: &[[f64; 2]], covariance: [[f64; 2]; 2]) -> Result<Vec<f64>> {
    let c = checked_covariance(covariance)?;
    if residuals.iter().flatten().any(|x| !x.is_finite()) {
        return Err(Error::value("finite two-dimensional innovations required"));
    }
    let l = mat(c)
        .cholesky()
        .ok_or_else(|| Error::value("covariance must be positive definite"))?
        .l();
    Ok(residuals
        .iter()
        .map(|r| {
            let x = r[0] / l[(0, 0)];
            let y = (r[1] - l[(1, 0)] * x) / l[(1, 1)];
            x * x + y * y
        })
        .collect())
}

pub type SourceGeometry = BTreeMap<String, ([f64; 2], [[f64; 2]; 2])>;

#[derive(Debug, Clone, Serialize)]
pub struct AssociationEdge {
    pub(crate) source_id: String,
    pub(crate) detection_index: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExcludedEdge(
    pub(crate) String,
    pub(crate) usize,
    pub(crate) &'static str,
    #[serde(serialize_with = "wire::optional")] pub(crate) Option<f64>,
);
#[derive(Debug, Clone, Serialize)]
pub struct UnifiedAssociationGraph {
    pub(crate) detection_keys: Vec<DetectionKey>,
    pub(crate) source_ids: Vec<String>,
    pub(crate) edges: Vec<AssociationEdge>,
    pub(crate) excluded_edges: Vec<ExcludedEdge>,
}

pub fn build(
    table: &FactorTable,
    config: &InferenceConfig,
    detections: &[Detection],
    geometry: &SourceGeometry,
    record_exclusions: bool,
) -> Result<UnifiedAssociationGraph> {
    let threshold = threshold(config.gate_probability())?;
    let mut sources: Vec<_> = table.sources.iter().collect();
    sources.sort_by(|a, b| a.source_id.cmp(&b.source_id));
    let mut graph = UnifiedAssociationGraph {
        detection_keys: table.detection_keys.clone(),
        source_ids: sources.iter().map(|s| s.source_id.clone()).collect(),
        edges: vec![],
        excluded_edges: vec![],
    };
    for s in sources {
        let ds = if let Some((mean, cov)) = geometry.get(&s.source_id) {
            let ds = if detections.is_empty() {
                vec![]
            } else {
                distances(
                    &detections
                        .iter()
                        .map(|d| [d.x_px - mean[0], d.y_px - mean[1]])
                        .collect::<Vec<_>>(),
                    *cov,
                )?
            };
            if ds.len() != table.detection_keys.len() {
                return Err(Error::value("gate detection order mismatch"));
            }
            Some(ds)
        } else {
            None
        };
        for (j, eta) in s.log_eta.iter().enumerate() {
            if !eta.is_finite() {
                if record_exclusions {
                    graph.excluded_edges.push(ExcludedEdge(
                        s.source_id.clone(),
                        j,
                        "ZERO_DENSITY_FACTOR",
                        None,
                    ));
                }
                continue;
            }
            if let Some(ds) = &ds {
                if ds[j] > threshold {
                    let reason = if s.log_existence_prior > f64::NEG_INFINITY {
                        "FINITE_GATE_POSSIBLE_NONZERO"
                    } else {
                        "ZERO_DENSITY_GATE"
                    };
                    if record_exclusions
                        || reason == "FINITE_GATE_POSSIBLE_NONZERO"
                            && graph.excluded_edges.is_empty()
                    {
                        graph.excluded_edges.push(ExcludedEdge(
                            s.source_id.clone(),
                            j,
                            reason,
                            Some(ds[j]),
                        ));
                    }
                    continue;
                }
            }
            graph.edges.push(AssociationEdge {
                source_id: s.source_id.clone(),
                detection_index: j,
            });
        }
    }
    Ok(graph)
}

#[derive(Debug, Clone, Serialize)]
pub struct ConditionalAssociationComponent {
    pub sources: Vec<String>,
    pub detections: Vec<usize>,
}

pub fn components(graph: &UnifiedAssociationGraph) -> Vec<ConditionalAssociationComponent> {
    let n = graph.source_ids.len();
    let mut adj = vec![vec![]; n + graph.detection_keys.len()];
    let indices: HashMap<_, _> = graph
        .source_ids
        .iter()
        .enumerate()
        .map(|(i, s)| (s, i))
        .collect();
    for e in &graph.edges {
        let a = indices[&e.source_id];
        let b = n + e.detection_index;
        adj[a].push(b);
        adj[b].push(a);
    }
    let mut seen = vec![false; adj.len()];
    let mut out = vec![];
    for node in 0..adj.len() {
        if seen[node] {
            continue;
        }
        seen[node] = true;
        let mut stack = vec![node];
        let mut group = ConditionalAssociationComponent {
            sources: vec![],
            detections: vec![],
        };
        while let Some(x) = stack.pop() {
            if x < n {
                group.sources.push(graph.source_ids[x].clone());
            } else {
                group.detections.push(x - n);
            }
            for &y in &adj[x] {
                if !seen[y] {
                    seen[y] = true;
                    stack.push(y);
                }
            }
        }
        group.sources.sort();
        group.detections.sort();
        out.push(group);
    }
    out
}
