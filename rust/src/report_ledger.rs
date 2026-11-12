use crate::error::{Error, Result};
use serde_json::{json, Value};
pub(crate) fn numeric_markers(v: &mut Value) -> Result<()> {
    match v {
        Value::Array(a) => {
            for x in a {
                numeric_markers(x)?;
            }
        }
        Value::Object(m) => {
            for (k, x) in m {
                if ["log_existence", "log_predictive_set_density"].contains(&k.as_str()) {
                    if x == "-Infinity" {
                        *x = json!("-inf");
                    } else if x == "Infinity" || x == "NaN" {
                        return Err(Error::value(
                            "nonfinite positive value/NaN must not enter a report",
                        ));
                    }
                }
                numeric_markers(x)?;
            }
        }
        _ => (),
    }
    Ok(())
}
