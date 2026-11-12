use crate::{
    contracts::Detection,
    efficient_graph,
    error::{Error, Result},
    exact_factors::{wire, FactorTable, SourceFactors},
    exact_posterior::{fsum, source_log_miss_given_exists},
    hypothesis_selection::bounded_product,
    input_evidence,
    tracking_model::JointHistory,
};
use serde::{Serialize, Serializer};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Node {
    Detection(usize),
    Group(usize),
    Star(String),
}

impl Serialize for Node {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Detection(i) => ("D", i).serialize(s),
            Self::Group(i) => ("G", i).serialize(s),
            Self::Star(id) => ("S", id).serialize(s),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConditionalHistory {
    pub history: JointHistory,
    pub factors: Vec<SourceFactors>,
}

#[derive(Debug, Serialize)]
pub struct PreparedComponent {
    pub nodes: Vec<Node>,
    pub indices: Vec<usize>,
    pub groups: Vec<usize>,
    pub known: Vec<String>,
    pub priors: Vec<JointHistory>,
    pub tables: Vec<FactorTable>,
    pub gate_geometry: BTreeMap<String, efficient_graph::SourceGeometry>,
    pub source_diag: Vec<Value>,
    pub loss: f64,
}

#[derive(Debug, Serialize)]
pub struct PreparedGroups {
    pub node_count: usize,
    pub components: Vec<PreparedComponent>,
    #[serde(serialize_with = "wire::float")]
    pub isolated_star_log_miss: f64,
    pub prior_loss: f64,
}

struct Union {
    nodes: Vec<Node>,
    indices: HashMap<Node, usize>,
    parent: Vec<usize>,
}

impl Union {
    fn new() -> Self {
        Self {
            nodes: vec![],
            indices: HashMap::new(),
            parent: vec![],
        }
    }
    fn index(&mut self, node: Node) -> usize {
        if let Some(&i) = self.indices.get(&node) {
            i
        } else {
            let i = self.nodes.len();
            self.indices.insert(node.clone(), i);
            self.nodes.push(node);
            self.parent.push(i);
            i
        }
    }
    fn root(&mut self, mut i: usize) -> usize {
        let mut root = i;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        while self.parent[i] != i {
            let next = self.parent[i];
            self.parent[i] = root;
            i = next;
        }
        root
    }
    fn join(&mut self, a: Node, b: Node) {
        let a = self.index(a);
        let a = self.root(a);
        let b = self.index(b);
        let b = self.root(b);
        self.parent[b] = a;
    }
    fn components(&mut self) -> Vec<Vec<Node>> {
        let mut groups = vec![];
        let mut index = HashMap::new();
        for i in 0..self.nodes.len() {
            let root = self.root(i);
            let g = *index.entry(root).or_insert_with(|| {
                groups.push(vec![]);
                groups.len() - 1
            });
            groups[g].push(self.nodes[i].clone());
        }
        groups.sort_by_key(|nodes| match nodes[0] {
            Node::Detection(i) => format!("('D', {i})"),
            Node::Group(i) => format!("('G', {i})"),
            Node::Star(_) => unreachable!("stars only enter via a detection edge"),
        });
        groups
    }
}

pub fn group_identity(frame: i64, parent_ids: &[String], indices: &[usize]) -> Result<String> {
    Ok(format!(
        "G:{}",
        input_evidence::digest(&json!([frame, parent_ids, indices]))?
    ))
}

pub fn birth_identity(
    parent: &str,
    events: &[(String, crate::exact_factors::DetectionKey)],
) -> Result<String> {
    Ok(format!(
        "J:{}",
        input_evidence::digest(&json!([parent, events]))?
    ))
}

pub struct Context<'a> {
    pub detections: &'a [Detection],
    pub stars: &'a [SourceFactors],
    pub frame: i64,
    pub gate: Option<f64>,
    pub limit: i64,
    pub target_pd: f64,
    pub background_pd: f64,
    pub clutter_density: f64,
    pub intensities: &'a [Vec<(String, f64)>],
}

pub fn prepare(
    groups: &[Vec<ConditionalHistory>],
    ctx: &Context,
    check: &mut impl FnMut(Option<usize>) -> Result<()>,
) -> Result<PreparedGroups> {
    let n = ctx.detections.len();
    if ctx.intensities.len() != n {
        return Err(Error::value("PPP intensity count mismatch"));
    }
    for group in groups {
        for h in group {
            let expected: Vec<_> = h
                .history
                .tracks
                .iter()
                .map(|t| format!("T:{}", t.label))
                .chain(
                    h.history
                        .backgrounds
                        .iter()
                        .map(|b| format!("S_U:{}", b.background_id)),
                )
                .collect();
            if expected.len() != h.factors.len()
                || expected
                    .iter()
                    .zip(&h.factors)
                    .any(|(id, f)| *id != f.source_id || f.log_eta.len() != n)
            {
                return Err(Error::value(
                    "conditional history factor alignment mismatch",
                ));
            }
        }
    }
    if ctx
        .stars
        .iter()
        .any(|f| f.kind != "S_K" || f.log_eta.len() != n)
    {
        return Err(Error::value("known-star factor alignment mismatch"));
    }
    let mut graph = Union::new();
    for j in 0..n {
        graph.index(Node::Detection(j));
    }
    for (i, group) in groups.iter().enumerate() {
        graph.index(Node::Group(i));
        for h in group {
            for f in &h.factors {
                for (j, x) in f.log_eta.iter().enumerate() {
                    if x.is_finite() {
                        graph.join(Node::Group(i), Node::Detection(j));
                    }
                }
            }
        }
    }
    for f in ctx.stars {
        if !f.log_eta.iter().any(|x| x.is_finite()) {
            continue;
        }
        let distances = if !ctx.detections.is_empty() && ctx.gate.is_some() {
            f.pixel_model
                .as_ref()
                .map(|p| {
                    efficient_graph::distances(
                        &ctx.detections
                            .iter()
                            .map(|d| [d.x_px - p.mean_px[0], d.y_px - p.mean_px[1]])
                            .collect::<Vec<_>>(),
                        p.noise_px2,
                    )
                })
                .transpose()?
        } else {
            None
        };
        for (j, x) in f.log_eta.iter().enumerate() {
            if x.is_finite()
                && (distances.is_none()
                    || distances.as_ref().unwrap()[j]
                        <= efficient_graph::threshold(ctx.gate.unwrap())?)
            {
                graph.join(Node::Star(f.source_id.clone()), Node::Detection(j));
            }
        }
    }
    let isolated: Vec<_> = ctx
        .stars
        .iter()
        .filter(|s| !graph.indices.contains_key(&Node::Star(s.source_id.clone())))
        .map(source_log_miss_given_exists)
        .collect();
    let isolated_star_log_miss = if isolated.contains(&f64::NEG_INFINITY) {
        f64::NEG_INFINITY
    } else {
        fsum(isolated)?
    };
    let node_count = graph.nodes.len();
    let components = graph.components();
    check(Some(node_count))?;
    let mut output = PreparedGroups {
        node_count,
        components: vec![],
        isolated_star_log_miss,
        prior_loss: 0.,
    };
    for nodes in components {
        check(None)?;
        let mut indices: Vec<_> = nodes
            .iter()
            .filter_map(|n| {
                if let Node::Detection(i) = n {
                    Some(*i)
                } else {
                    None
                }
            })
            .collect();
        indices.sort();
        let mut selected: Vec<_> = nodes
            .iter()
            .filter_map(|n| {
                if let Node::Group(i) = n {
                    Some(*i)
                } else {
                    None
                }
            })
            .collect();
        selected.sort();
        let mut known: Vec<_> = nodes
            .iter()
            .filter_map(|n| {
                if let Node::Star(id) = n {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();
        known.sort();
        let choices: Vec<_> = selected
            .iter()
            .map(|i| {
                groups[*i]
                    .iter()
                    .enumerate()
                    .map(|(j, h)| (h.history.weight, (*i, j)))
                    .collect()
            })
            .collect();
        let (products, loss) = bounded_product(&choices, ctx.limit)?;
        output.prior_loss += loss;
        let den = fsum(products.iter().map(|(w, _)| *w))?;
        let mut component = PreparedComponent {
            nodes,
            indices: indices.clone(),
            groups: selected,
            known: known.clone(),
            priors: vec![],
            tables: vec![],
            gate_geometry: BTreeMap::new(),
            source_diag: vec![],
            loss,
        };
        for (weight, parts) in products {
            let hs: Vec<_> = parts.iter().map(|(i, j)| &groups[*i][*j]).collect();
            let hid = group_identity(
                ctx.frame,
                &hs.iter()
                    .map(|h| h.history.hypothesis_id.clone())
                    .collect::<Vec<_>>(),
                &indices,
            )?;
            let tracks: Vec<_> = hs.iter().flat_map(|h| h.history.tracks.iter()).collect();
            let backgrounds: Vec<_> = hs
                .iter()
                .flat_map(|h| h.history.backgrounds.iter())
                .collect();
            let prior = JointHistory::from_json(&json!({
                "hypothesis_id": hid,
                "weight": weight/den,
                "tracks": tracks,
                "backgrounds": backgrounds
            }))?;
            let mut factors = vec![];
            for kind in ["T", "S_U"] {
                for h in &hs {
                    for f in h.factors.iter().filter(|f| f.kind == kind) {
                        let mut f = f.clone();
                        f.log_eta = indices.iter().map(|j| f.log_eta[*j]).collect();
                        let pd = if kind == "T" {
                            ctx.target_pd
                        } else {
                            ctx.background_pd
                        };
                        component.source_diag.push(json!({
                            "parent_id": hid,
                            "source_id": f.source_id,
                            "pd_bar": f.pd_bar,
                            "visibility_mass_cubature": if pd>0.{Some(f.pd_bar/pd)}else{None}
                        }));
                        factors.push(f);
                    }
                }
            }
            let mut geometry = BTreeMap::new();
            for id in &known {
                let mut f = ctx
                    .stars
                    .iter()
                    .find(|f| f.source_id == *id)
                    .unwrap()
                    .clone();
                f.log_eta = indices.iter().map(|j| f.log_eta[*j]).collect();
                let p = f.pixel_model.as_ref().ok_or_else(|| Error {
                    kind: "AttributeError",
                    code: None,
                    message: "'NoneType' object has no attribute 'mean_px'".into(),
                })?;
                geometry.insert(id.clone(), (p.mean_px, p.noise_px2));
                factors.push(f);
            }
            let log_kappa: Vec<_> = indices
                .iter()
                .map(|j| {
                    let sum = ctx.intensities[*j].iter().fold(0., |sum, (_, v)| sum + v);
                    let k = ctx.clutter_density + sum;
                    if k == 0. {
                        Ok(wire::Float(f64::NEG_INFINITY))
                    } else if k < 0. {
                        Err(Error::value("math domain error"))
                    } else {
                        Ok(wire::Float(libm::log(k)))
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let keys: Vec<_> = indices.iter().map(|j| ctx.detections[*j].key()).collect();
            component.tables.push(FactorTable::from_json(&json!({
                "parent_id": hid,
                "parent_weight": prior.weight,
                "detection_keys": keys,
                "sources": factors,
                "log_kappa": log_kappa
            }))?);
            component.priors.push(prior);
            component.gate_geometry.insert(hid, geometry);
        }
        output.components.push(component);
    }
    Ok(output)
}
