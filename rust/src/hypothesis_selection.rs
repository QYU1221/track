use crate::{
    error::Result, exact_factors::DetectionKey, exact_posterior::fsum, tracking_model::JointHistory,
};
use std::collections::{HashMap, HashSet};
fn prefix(length: usize, limit: i64) -> usize {
    if limit >= 0 {
        length.min(limit as usize)
    } else {
        length.saturating_sub(usize::try_from(limit.unsigned_abs()).unwrap_or(usize::MAX))
    }
}

pub fn select_joint_histories(
    histories: &[JointHistory],
    limit: i64,
    existing_targets: &[String],
    preserve: bool,
) -> Result<(Vec<JointHistory>, f64)> {
    let order = |a: &&JointHistory, b: &&JointHistory| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap()
            .then_with(|| a.hypothesis_id.cmp(&b.hypothesis_id))
    };
    let mut ranked: Vec<_> = histories.iter().collect();
    ranked.sort_by(order);
    let mut kept = if !preserve || existing_targets.is_empty() {
        ranked[..prefix(ranked.len(), limit)].to_vec()
    } else {
        type Signature = Vec<(String, bool, Option<DetectionKey>)>;
        let mut labels = existing_targets.to_vec();
        labels.sort();
        let mut index: HashMap<Signature, usize> = HashMap::new();
        let mut groups: Vec<Vec<&JointHistory>> = vec![];
        for h in &ranked {
            let assignments: HashMap<_, _> = h
                .target_assignments
                .iter()
                .map(|(s, k)| (s.as_str(), k))
                .collect();
            let present: HashSet<_> = h.tracks.iter().map(|t| t.label.as_str()).collect();
            let signature = labels
                .iter()
                .map(|label| {
                    (
                        label.clone(),
                        present.contains(label.as_str()),
                        assignments
                            .get(format!("T:{label}").as_str())
                            .and_then(|k| (*k).clone()),
                    )
                })
                .collect();
            let i = *index.entry(signature).or_insert_with(|| {
                groups.push(vec![]);
                groups.len() - 1
            });
            groups[i].push(h);
        }
        let mut ordered = groups
            .into_iter()
            .map(|g| Ok((fsum(g.iter().map(|h| h.weight))?, g)))
            .collect::<Result<Vec<_>>>()?;
        ordered.sort_by(|(a, ga), (b, gb)| {
            b.partial_cmp(a)
                .unwrap()
                .then_with(|| ga[0].hypothesis_id.cmp(&gb[0].hypothesis_id))
        });
        let mut selected: Vec<_> = ordered[..prefix(ordered.len(), limit)]
            .iter()
            .map(|(_, g)| g[0])
            .collect();
        let ids: HashSet<_> = selected.iter().map(|h| h.hypothesis_id.as_str()).collect();
        for h in &ranked {
            if limit <= 0 || selected.len() >= limit as usize {
                break;
            }
            if !ids.contains(h.hypothesis_id.as_str()) {
                selected.push(h);
            }
        }
        selected.sort_by(order);
        selected
    };
    let retained = fsum(kept.iter().map(|h| h.weight))?;
    let discarded = (fsum(ranked.iter().map(|h| h.weight))? - retained).max(0.);
    Ok((kept.drain(..).cloned().collect(), discarded))
}

pub fn bounded_product<T: Clone>(
    choices: &[Vec<(f64, T)>],
    limit: i64,
) -> Result<(Vec<(f64, Vec<T>)>, f64)> {
    let mut beam = vec![(1., vec![])];
    let mut discarded = 0.;
    for options in choices {
        let mut pool = vec![];
        for (w, items) in &beam {
            for (p, item) in options {
                let weight = w * p;
                if weight > 0. {
                    let mut next = items.clone();
                    next.push(item.clone());
                    pool.push((weight, next));
                }
            }
        }
        pool.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());
        let n = prefix(pool.len(), limit);
        discarded += fsum(pool[n..].iter().map(|(w, _)| *w))?;
        pool.truncate(n);
        beam = pool;
    }
    Ok((beam, discarded))
}
