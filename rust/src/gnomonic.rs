use crate::{
    geometry::{checked_rotation, unit},
    numerical_policy::UNIT_VECTOR_EPS_MULTIPLIER,
    ValidationError,
};

#[derive(Debug, Clone, serde::Serialize)]
pub struct GnomonicPlane {
    pub(crate) reference_id: String,
    pub(crate) center: [f64; 3],
    pub(crate) max_angle_rad: f64,
    pub(crate) e1: [f64; 3],
    pub(crate) e2: [f64; 3],
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}

impl GnomonicPlane {
    pub fn from_json(v: &serde_json::Value) -> crate::error::Result<Self> {
        use crate::{contracts as c, error::Error};
        c::fields(
            v,
            "GnomonicPlane",
            &["reference_id", "center", "max_angle_rad"],
            &["e1", "e2"],
        )?;
        let id = v["reference_id"]
            .as_str()
            .ok_or_else(|| Error::value("tangent reference ID must be nonempty"))?;
        if v["e1"].is_null() != v["e2"].is_null() {
            return Err(Error::value(
                "provide both tangent basis vectors or neither",
            ));
        }
        let basis = if v["e1"].is_null() {
            None
        } else {
            Some((
                c::vector(&v["e1"], "e1", "tangent basis vectors must be 3D")?,
                c::vector(&v["e2"], "e2", "tangent basis vectors must be 3D")?,
            ))
        };
        Ok(Self::new(
            id,
            c::vector(
                &v["center"],
                "tangent center",
                "direction must be a non-zero 3-vector",
            )?,
            c::scalar(&v["max_angle_rad"], "tangent angular domain")?,
            basis,
        )?)
    }
    pub fn reference_id(&self) -> &str {
        &self.reference_id
    }
    pub fn basis(&self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        (self.center, self.e1, self.e2)
    }
    pub fn max_angle_rad(&self) -> f64 {
        self.max_angle_rad
    }
    pub fn new(
        id: &str,
        center: [f64; 3],
        angle: f64,
        basis: Option<([f64; 3], [f64; 3])>,
    ) -> Result<Self, ValidationError> {
        if id.trim().is_empty() {
            return Err(ValidationError("tangent reference ID must be nonempty"));
        }
        if center.iter().any(|x| !x.is_finite()) {
            return Err(ValidationError("tangent center must be finite"));
        }
        let c = if (dot(center, center) - 1.).abs() <= UNIT_VECTOR_EPS_MULTIPLIER * f64::EPSILON {
            center
        } else {
            unit(&center)?
        };
        if !angle.is_finite() {
            return Err(ValidationError("tangent angular domain must be finite"));
        }
        if !(angle > 0. && angle < std::f64::consts::FRAC_PI_2) {
            return Err(ValidationError(
                "tangent domain must be in (0,pi/2) radians",
            ));
        }
        let (e1, e2) = match basis {
            Some(b) => b,
            None => {
                let mut index = 0;
                for i in 1..3 {
                    if c[i].abs() < c[index].abs() {
                        index = i;
                    }
                }
                let a = std::array::from_fn(|i| if i == index { 1. } else { 0. });
                let e = unit(&std::array::from_fn::<_, 3, _>(|i| a[i] - c[i] * dot(a, c)))?;
                (
                    e,
                    [
                        c[1] * e[2] - c[2] * e[1],
                        c[2] * e[0] - c[0] * e[2],
                        c[0] * e[1] - c[1] * e[0],
                    ],
                )
            }
        };
        if e1.iter().any(|x| !x.is_finite()) {
            return Err(ValidationError("e1 must be finite"));
        }
        if e2.iter().any(|x| !x.is_finite()) {
            return Err(ValidationError("e2 must be finite"));
        }
        checked_rotation(std::array::from_fn(|i| [e1[i], e2[i], c[i]]))
            .map_err(|_| ValidationError("tangent basis must be orthonormal and right handed"))?;
        Ok(Self {
            reference_id: id.to_owned(),
            center: c,
            max_angle_rad: angle,
            e1,
            e2,
        })
    }
    pub fn project(&self, raw: [f64; 3]) -> Result<([f64; 2], [[f64; 3]; 2]), ValidationError> {
        if raw.iter().any(|x| !x.is_finite()) {
            return Err(ValidationError("J2000 direction must be finite"));
        }
        let u = unit(&raw)?;
        let d = dot(self.center, u);
        if d <= self.max_angle_rad.cos() {
            return Err(ValidationError("LOS outside declared gnomonic domain"));
        }
        let p = [dot(self.e1, u) / d, dot(self.e2, u) / d];
        let scale = raw.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        let norm = raw.iter().map(|x| (x / scale).powi(2)).sum::<f64>().sqrt();
        let axes = [self.e1, self.e2];
        let j = std::array::from_fn(|i| {
            std::array::from_fn(|k| ((axes[i][k] - p[i] * self.center[k]) / d / scale) / norm)
        });
        if p.iter().any(|x| !x.is_finite()) {
            return Err(ValidationError("gnomonic position must be finite"));
        }
        if j.iter().flatten().any(|x| !x.is_finite()) {
            return Err(ValidationError("gnomonic Jacobian must be finite"));
        }
        Ok((p, j))
    }
    pub fn inverse(&self, p: [f64; 2]) -> Result<([f64; 3], [[f64; 2]; 3]), ValidationError> {
        if p.iter().any(|x| !x.is_finite()) {
            return Err(ValidationError("gnomonic position must be finite"));
        }
        if (p[0] * p[0] + p[1] * p[1]).sqrt() >= self.max_angle_rad.tan() {
            return Err(ValidationError("gnomonic position outside declared domain"));
        }
        let a = std::array::from_fn(|i| self.center[i] + p[0] * self.e1[i] + p[1] * self.e2[i]);
        let norm = dot(a, a).sqrt();
        let u = a.map(|x| x / norm);
        let axes = [self.e1, self.e2];
        let j = std::array::from_fn(|i| {
            std::array::from_fn(|k| {
                (0..3)
                    .map(|h| ((if i == h { 1. } else { 0. }) - u[i] * u[h]) * axes[k][h])
                    .sum::<f64>()
                    / norm
            })
        });
        Ok((u, j))
    }
}
