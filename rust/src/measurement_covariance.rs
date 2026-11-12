use crate::{
    contracts::{checked_covariance, finite},
    error::{Error, Result},
    linalg::{array, mat},
    numerical_policy::*,
};

pub fn checked_los_covariance(c: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    finite(
        &c.iter().flatten().copied().collect::<Vec<_>>(),
        "LOS covariance",
    )?;
    let scale = c.iter().flatten().fold(0.0_f64, |s, x| s.max(x.abs()));
    let tolerance = BASIS_EPS_MULTIPLIER * f64::EPSILON * scale;
    if scale == 0. || (0..3).any(|i| (0..3).any(|j| (c[i][j] - c[j][i]).abs() > tolerance)) {
        return Err(Error::value("LOS covariance must be nonzero and symmetric"));
    }
    let m = nalgebra::Matrix3::from_fn(|i, j| {
        if i >= j {
            c[i][j] / scale
        } else {
            c[j][i] / scale
        }
    });
    let ev = m.symmetric_eigen().eigenvalues * scale;
    let mut eigenvalues = [ev[0], ev[1], ev[2]];
    eigenvalues.sort_by(f64::total_cmp);
    if eigenvalues[0] < -tolerance
        || eigenvalues[1] <= tolerance
        || eigenvalues[0].abs() > tolerance
    {
        return Err(Error::value(
            "LOS covariance must be PSD with numerical rank 2",
        ));
    }
    Ok(c)
}

pub fn propagate_los_covariance(c: [[f64; 2]; 2], j: [[f64; 2]; 3]) -> Result<[[f64; 3]; 3]> {
    let r = mat(checked_covariance(c)?);
    finite(
        &j.iter().flatten().copied().collect::<Vec<_>>(),
        "LOS Jacobian",
    )?;
    let j = mat(j);
    checked_los_covariance(array(j * r * j.transpose()))
}

pub fn propagate_tangent_covariance(c: [[f64; 3]; 3], g: [[f64; 3]; 2]) -> Result<[[f64; 2]; 2]> {
    let c = mat(checked_los_covariance(c)?);
    finite(
        &g.iter().flatten().copied().collect::<Vec<_>>(),
        "tangent Jacobian",
    )?;
    let g = mat(g);
    checked_covariance(array(g * c * g.transpose()))
}
