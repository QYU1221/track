use crate::{
    numerical_policy::{BASIS_EPS_MULTIPLIER, UNIT_VECTOR_EPS_MULTIPLIER},
    ValidationError,
};

pub type Matrix3 = [[f64; 3]; 3];

fn normalize<const N: usize>(
    v: &[f64],
    name: &'static str,
    shape_error: &'static str,
) -> Result<[f64; N], ValidationError> {
    if v.iter().any(|x| !x.is_finite()) {
        return Err(ValidationError(name));
    }
    let scale = v.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
    if v.len() != N || scale == 0. {
        return Err(ValidationError(shape_error));
    }
    let scaled: [f64; N] = std::array::from_fn(|i| v[i] / scale);
    let norm = scaled.iter().map(|x| x * x).sum::<f64>().sqrt();
    Ok(scaled.map(|x| x / norm))
}

pub fn unit(v: &[f64]) -> Result<[f64; 3], ValidationError> {
    normalize(
        v,
        "direction must be finite",
        "direction must be a non-zero 3-vector",
    )
}

pub fn normalize_quaternion(v: &[f64]) -> Result<[f64; 4], ValidationError> {
    normalize(
        v,
        "quaternion must be finite",
        "quaternion must be a non-zero 4-vector",
    )
}

pub fn quaternion_matrix(q: &[f64]) -> Result<Matrix3, ValidationError> {
    let [x, y, z, w] = normalize_quaternion(q)?;
    Ok([
        [
            1. - 2. * (y * y + z * z),
            2. * (x * y - z * w),
            2. * (x * z + y * w),
        ],
        [
            2. * (x * y + z * w),
            1. - 2. * (x * x + z * z),
            2. * (y * z - x * w),
        ],
        [
            2. * (x * z - y * w),
            2. * (y * z + x * w),
            1. - 2. * (x * x + y * y),
        ],
    ])
}

pub fn checked_rotation(r: Matrix3) -> Result<Matrix3, ValidationError> {
    if r.iter().flatten().any(|x| !x.is_finite()) {
        return Err(ValidationError("rotation must be finite"));
    }
    let tol = BASIS_EPS_MULTIPLIER * f64::EPSILON;
    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    let orthogonal = (0..3).all(|i| {
        (0..3).all(|j| {
            let dot = (0..3).map(|k| r[k][i] * r[k][j]).sum::<f64>();
            (dot - if i == j { 1. } else { 0. }).abs() <= tol
        })
    });
    if !orthogonal || (det - 1.).abs() > tol {
        return Err(ValidationError(
            "rotation must be a proper orthogonal 3x3 matrix",
        ));
    }
    Ok(r)
}

pub fn adapt_quaternion(
    q: &[f64],
    order: &str,
    direction: &str,
) -> Result<[f64; 4], ValidationError> {
    let mut q = normalize_quaternion(q)?;
    match order {
        "wxyz" => q = [q[1], q[2], q[3], q[0]],
        "xyzw" => (),
        _ => return Err(ValidationError("declare quaternion order xyzw or wxyz")),
    };
    match direction {
        "j2000_to_camera" => {
            for x in &mut q[..3] {
                *x = -*x;
            }
        }
        "camera_to_j2000" => (),
        _ => return Err(ValidationError("declare attitude rotation direction")),
    };
    Ok(q)
}

pub fn camera_to_inertial(
    b: [f64; 3],
    j: [[f64; 2]; 3],
    q: &[f64],
) -> Result<([f64; 3], [[f64; 2]; 3]), ValidationError> {
    if b.iter().any(|x| !x.is_finite()) {
        return Err(ValidationError("camera unit LOS must be finite"));
    }
    if j.iter().flatten().any(|x| !x.is_finite()) {
        return Err(ValidationError("camera Jacobian must be finite"));
    }
    if (b.iter().map(|x| x * x).sum::<f64>() - 1.).abs() > UNIT_VECTOR_EPS_MULTIPLIER * f64::EPSILON
    {
        return Err(ValidationError(
            "camera LOS must be unit (3,) and Jacobian must be (3,2)",
        ));
    }
    let r = checked_rotation(quaternion_matrix(q)?)?;
    Ok((
        std::array::from_fn(|i| (0..3).map(|k| r[i][k] * b[k]).sum()),
        std::array::from_fn(|i| std::array::from_fn(|c| (0..3).map(|k| r[i][k] * j[k][c]).sum())),
    ))
}
