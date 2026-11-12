use crate::ValidationError;
pub fn checked_log_existence(
    existence: f64,
    log_existence: Option<f64>,
) -> Result<f64, ValidationError> {
    let Some(value) = log_existence else {
        if existence == 0. {
            return Ok(f64::NEG_INFINITY);
        }
        if existence < 0. {
            return Err(ValidationError("math domain error"));
        }
        return Ok(libm::log(existence));
    };
    if value.is_nan() || value > 0. {
        return Err(ValidationError("log existence must be <= 0"));
    }
    let projected = libm::exp(value);
    let close = projected == existence
        || (projected.is_finite()
            && existence.is_finite()
            && (projected - existence).abs() <= 1e-12 * projected.abs().max(existence.abs()));
    if !close {
        return Err(ValidationError("existence and log existence disagree"));
    }
    Ok(value)
}

pub fn log_absence(log_existence: f64) -> Result<f64, ValidationError> {
    if log_existence == 0. {
        return Ok(f64::NEG_INFINITY);
    }
    if log_existence < -std::f64::consts::LN_2 {
        return Ok(libm::log1p(-libm::exp(log_existence)));
    }
    if log_existence > 0. {
        return Err(ValidationError("math domain error"));
    }
    Ok(libm::log(-libm::expm1(log_existence)))
}
