use crate::numerical_policy::*;
const X: [f64; 8] = [
    0.9914553711208126,
    0.9491079123427585,
    0.8648644233597691,
    0.7415311855993945,
    0.5860872354676911,
    0.4058451513773972,
    0.2077849550078985,
    0.,
];
const K: [f64; 8] = [
    0.022935322010529225,
    0.06309209262997855,
    0.10479001032225018,
    0.14065325971552592,
    0.1690047266392679,
    0.19035057806478542,
    0.20443294007529889,
    0.20948214108472783,
];
const G: [f64; 4] = [
    0.1294849661688697,
    0.27970539148927667,
    0.3818300505051189,
    0.4179591836734694,
];
fn vector_rule<const N: usize>(
    f: &impl Fn(f64) -> [f64; N],
    a: f64,
    b: f64,
) -> Option<([f64; N], f64)> {
    let half = (b - a) / 2.;
    let mid = a + half;
    let fc = f(mid);
    let mut values = [([0.; N], [0.; N]); 7];
    let mut k = fc.map(|x| K[7] * x);
    let mut g = fc.map(|x| G[3] * x);
    let mut absolute = fc.map(|x| K[7] * x.abs());
    for i in 0..7 {
        let u = f(mid - half * X[i]);
        let v = f(mid + half * X[i]);
        values[i] = (u, v);
        for j in 0..N {
            k[j] += K[i] * (u[j] + v[j]);
            absolute[j] += K[i] * (u[j].abs() + v[j].abs());
            if i % 2 == 1 {
                g[j] += G[i / 2] * (u[j] + v[j]);
            }
        }
    }
    let mut error_norm = 0.;
    for j in 0..N {
        let mean = k[j] / 2.;
        let mut asc = K[7] * (fc[j] - mean).abs();
        for i in 0..7 {
            asc += K[i] * ((values[i].0[j] - mean).abs() + (values[i].1[j] - mean).abs());
        }
        asc *= half.abs();
        absolute[j] *= half.abs();
        let mut error = ((k[j] - g[j]) * half).abs();
        if asc != 0. && error != 0. {
            error = asc * (200. * error / asc).powf(1.5).min(1.);
        }
        if absolute[j] > f64::MIN_POSITIVE / (50. * f64::EPSILON) {
            error = error.max(50. * f64::EPSILON * absolute[j]);
        }
        error_norm = f64::hypot(error_norm, error);
        k[j] *= half;
    }
    if k.iter().all(|x| x.is_finite()) && error_norm.is_finite() {
        Some((k, error_norm))
    } else {
        None
    }
}

pub(crate) fn integrate_vector<const N: usize>(
    f: impl Fn(f64) -> [f64; N],
    a: f64,
    b: f64,
) -> Option<([f64; N], f64)> {
    if a == b {
        return Some(([0.; N], 0.));
    }
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let (v, e) = vector_rule(&f, a, b)?;
    let mut pieces = vec![(a, b, v, e)];
    loop {
        let mut sum = [0.; N];
        let mut error = 0.;
        for p in &pieces {
            for j in 0..N {
                sum[j] += p.2[j];
            }
            error += p.3;
        }
        let norm = sum.iter().fold(0., |a, x| f64::hypot(a, *x));
        if error <= 1e-12_f64.max(1e-10 * norm) {
            return Some((sum, error));
        }
        if pieces.len() >= 10000 {
            return None;
        }
        let i = (0..pieces.len()).max_by(|i, j| pieces[*i].3.total_cmp(&pieces[*j].3))?;
        let (a, b, _, _) = pieces.remove(i);
        let mid = a + (b - a) / 2.;
        if mid == a || mid == b {
            return None;
        }
        let (l, le) = vector_rule(&f, a, mid)?;
        let (r, re) = vector_rule(&f, mid, b)?;
        pieces.insert(i, (mid, b, r, re));
        pieces.insert(i, (a, mid, l, le));
    }
}

fn rule(f: &impl Fn(f64) -> f64, a: f64, b: f64) -> Option<(f64, f64)> {
    let half = (b - a) / 2.;
    let mid = a + half;
    let fc = f(mid);
    let mut k = K[7] * fc;
    let mut g = G[3] * fc;
    let mut absolute = k.abs();
    let mut values = [(0., 0.); 7];
    for i in 0..7 {
        let u = f(mid - half * X[i]);
        let v = f(mid + half * X[i]);
        values[i] = (u, v);
        k += K[i] * (u + v);
        absolute += K[i] * (u.abs() + v.abs());
        if i % 2 == 1 {
            g += G[i / 2] * (u + v);
        }
    }
    let mean = k / 2.;
    let mut asc = K[7] * (fc - mean).abs();
    for i in 0..7 {
        asc += K[i] * ((values[i].0 - mean).abs() + (values[i].1 - mean).abs());
    }
    asc *= half.abs();
    absolute *= half.abs();
    let mut error = ((k - g) * half).abs();
    if asc != 0. && error != 0. {
        error = asc * (200. * error / asc).powf(1.5).min(1.);
    }
    if absolute > f64::MIN_POSITIVE / (50. * f64::EPSILON) {
        error = error.max(50. * f64::EPSILON * absolute);
    }
    let value = k * half;
    if value.is_finite() && error.is_finite() {
        Some((value, error))
    } else {
        None
    }
}

pub(crate) fn integrate(f: impl Fn(f64) -> f64, a: f64, b: f64) -> Option<(f64, f64)> {
    if a == b {
        return Some((0., 0.));
    }
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let (value, error) = rule(&f, a, b)?;
    let mut pieces = vec![(a, b, value, error)];
    for _ in 0..QUADRATURE_SUBDIVISIONS {
        let sum: f64 = pieces.iter().map(|p| p.2).sum();
        let err: f64 = pieces.iter().map(|p| p.3).sum();
        if err <= QUADRATURE_ABS_TOL.max(QUADRATURE_REL_TOL * sum.abs()) {
            return Some((sum, err));
        }
        if pieces.len() >= QUADRATURE_SUBDIVISIONS {
            return None;
        }
        let index = (0..pieces.len()).max_by(|i, j| pieces[*i].3.total_cmp(&pieces[*j].3))?;
        let (a, b, _, _) = pieces.remove(index);
        let mid = a + (b - a) / 2.;
        if mid == a || mid == b {
            return None;
        }
        let (l, le) = rule(&f, a, mid)?;
        let (r, re) = rule(&f, mid, b)?;
        pieces.insert(index, (mid, b, r, re));
        pieces.insert(index, (a, mid, l, le));
    }
    None
}

pub(crate) fn partition(
    f: impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    points: &[f64],
) -> Option<(f64, f64)> {
    let mut cuts = vec![a, b];
    cuts.extend(
        points
            .iter()
            .copied()
            .filter(|x| x.is_finite() && a < *x && *x < b),
    );
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    let (mut value, mut error) = (0., 0.);
    for pair in cuts.windows(2) {
        let (v, e) = integrate(&f, pair[0], pair[1])?;
        value += v;
        error += e;
    }
    Some((value, error))
}

pub(crate) fn positive_infinite_partition(
    f: impl Fn(f64) -> f64,
    points: &[f64],
) -> Option<(f64, f64)> {
    let mapped: Vec<_> = points
        .iter()
        .filter(|x| **x > 0. && x.is_finite())
        .map(|x| x / (1. + x))
        .collect();
    partition(
        |u| {
            let t = u / (1. - u);
            f(t) / (1. - u).powi(2)
        },
        0.,
        1.,
        &mapped,
    )
}
