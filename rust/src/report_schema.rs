use crate::error::{Error, Result};
use serde_json::Value;
use std::sync::OnceLock;
fn schema() -> &'static Value {
    static DATA: OnceLock<Value> = OnceLock::new();
    DATA.get_or_init(|| serde_json::from_str(include_str!("report_schema.json")).unwrap())
}

pub fn table_fields(kind: &str, format: &str) -> Result<Vec<String>> {
    if !["compact", "x_legacy"].contains(&format) {
        return Err(Error::value(format!("unknown table format: {format}")));
    }
    let data = schema();
    let fields = data["fields"][kind]
        .as_array()
        .ok_or_else(|| Error::value("unknown table kind"))?;
    Ok(fields
        .iter()
        .filter(|v| {
            format == "x_legacy"
                || (!data["unsupported"][kind].as_array().unwrap().contains(v)
                    && data["aliases"][kind].get(v.as_str().unwrap()).is_none())
        })
        .map(|v| v.as_str().unwrap().to_owned())
        .collect())
}

pub fn table_row(kind: &str, row: &Value, format: &str) -> Result<Value> {
    if format == "compact" {
        return Ok(row.clone());
    }
    if format != "x_legacy" {
        return Err(Error::value(format!("unknown table format: {format}")));
    }
    let mut out = row
        .as_object()
        .cloned()
        .ok_or_else(|| Error::value("table row must be an object"))?;
    for (alias, canonical) in schema()["aliases"][kind]
        .as_object()
        .ok_or_else(|| Error::value("unknown table kind"))?
    {
        out.insert(alias.clone(), row[canonical.as_str().unwrap()].clone());
    }
    Ok(Value::Object(out))
}
