#[derive(Debug, Clone)]
struct Node {
    index: usize,
    axis: usize,
    left: Option<usize>,
    right: Option<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct Tree {
    nodes: Vec<Node>,
    root: Option<usize>,
}

impl Tree {
    pub(crate) fn new(points: &[[f64; 3]]) -> Self {
        fn build(
            tree: &mut Tree,
            indices: &mut [usize],
            points: &[[f64; 3]],
            depth: usize,
        ) -> Option<usize> {
            if indices.is_empty() {
                return None;
            }
            let axis = depth % 3;
            let mid = indices.len() / 2;
            indices.select_nth_unstable_by(mid, |a, b| {
                points[*a][axis].total_cmp(&points[*b][axis]).then(a.cmp(b))
            });
            let index = indices[mid];
            let n = tree.nodes.len();
            tree.nodes.push(Node {
                index,
                axis,
                left: None,
                right: None,
            });
            let (a, b) = indices.split_at_mut(mid);
            let left = build(tree, a, points, depth + 1);
            let right = build(tree, &mut b[1..], points, depth + 1);
            tree.nodes[n].left = left;
            tree.nodes[n].right = right;
            Some(n)
        }
        let mut tree = Self {
            nodes: vec![],
            root: None,
        };
        tree.root = build(
            &mut tree,
            &mut (0..points.len()).collect::<Vec<_>>(),
            points,
            0,
        );
        tree
    }
    pub(crate) fn query(&self, points: &[[f64; 3]], center: [f64; 3], radius: f64) -> Vec<usize> {
        fn visit(
            t: &Tree,
            n: Option<usize>,
            points: &[[f64; 3]],
            c: [f64; 3],
            r2: f64,
            out: &mut Vec<usize>,
        ) {
            let Some(i) = n else {
                return;
            };
            let n = &t.nodes[i];
            let p = points[n.index];
            if (0..3).map(|i| (p[i] - c[i]).powi(2)).sum::<f64>() <= r2 {
                out.push(n.index);
            }
            let delta = c[n.axis] - p[n.axis];
            if delta <= 0. || delta * delta <= r2 {
                visit(t, n.left, points, c, r2, out);
            }
            if delta >= 0. || delta * delta <= r2 {
                visit(t, n.right, points, c, r2, out);
            }
        }
        let mut out = vec![];
        visit(self, self.root, points, center, radius * radius, &mut out);
        out.sort_unstable();
        out
    }
}
