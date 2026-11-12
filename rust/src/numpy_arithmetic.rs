#[derive(Clone, Copy)]
pub(crate) struct N(pub f64, pub bool);
impl N {
    pub fn new(x: f64) -> Self {
        Self(x, false)
    }
}

fn checked(a: N, b: N, result: f64) -> N {
    let overflow = result.is_infinite() && a.0.is_finite() && b.0.is_finite();
    let invalid = result.is_nan() && !a.0.is_nan() && !b.0.is_nan();
    N(result, a.1 || b.1 || overflow || invalid)
}

impl std::ops::Add for N {
    type Output = N;
    fn add(self, b: N) -> N {
        checked(self, b, self.0 + b.0)
    }
}

impl std::ops::Mul for N {
    type Output = N;
    fn mul(self, b: N) -> N {
        checked(self, b, self.0 * b.0)
    }
}

pub(crate) fn pairwise_sum(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        return values.iter().fold(-0.0, |sum, x| sum + x);
    }
    if n > 128 {
        let half = (n / 2) / 8 * 8;
        return pairwise_sum(&values[..half]) + pairwise_sum(&values[half..]);
    }
    let mut lanes = [0.; 8];
    lanes.copy_from_slice(&values[..8]);
    let end = n / 8 * 8;
    for chunk in values[8..end].chunks_exact(8) {
        for i in 0..8 {
            lanes[i] += chunk[i];
        }
    }
    let left = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    let right = (lanes[4] + lanes[5]) + (lanes[6] + lanes[7]);
    values[end..].iter().fold(left + right, |sum, x| sum + x)
}
