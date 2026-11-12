use crate::{
    contracts::KnownStar,
    error::{Error, Result},
    geometry::unit,
};
use serde::{ser::*, Serialize};
use serde_json::{Map, Value};
use std::{
    io::{BufRead, BufReader},
    path::Path,
};

pub fn public_navstar_catalog(
    path: impl AsRef<Path>,
    mag_limit: Option<f64>,
) -> Result<Vec<KnownStar>> {
    check_limit(mag_limit)?;
    let bytes = std::fs::read(path).map_err(crate::data_io::io_error)?;
    let text = std::str::from_utf8(&bytes).map_err(|e| Error {
        kind: "UnicodeDecodeError",
        code: None,
        message: e.to_string(),
    })?;
    catalog_from_text(text, mag_limit)
}

fn check_limit(limit: Option<f64>) -> Result<()> {
    if limit.is_some_and(|x| !x.is_finite()) {
        Err(Error::value("magnitude selection must be finite"))
    } else {
        Ok(())
    }
}

pub fn catalog_from_text(text: &str, mag_limit: Option<f64>) -> Result<Vec<KnownStar>> {
    check_limit(mag_limit)?;
    let text = text
        .strip_prefix('\u{feff}')
        .unwrap_or(text)
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let mut out = vec![];
    for (index, line) in BufReader::new(text.as_bytes()).lines().enumerate() {
        let line = line.map_err(crate::data_io::io_error)?;
        let parts: Vec<_> = line
            .split(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
            .filter(|s| !s.is_empty())
            .take(6)
            .collect();
        if parts.is_empty() {
            continue;
        }
        let invalid = || Error::value(format!("invalid public catalog row {index}"));
        let values = parts
            .iter()
            .map(|s| {
                crate::data_io::parse_float(s)
                    .filter(|x| x.is_finite())
                    .ok_or_else(invalid)
            })
            .collect::<Result<Vec<_>>>()?;
        if values.len() != 6 || !(-90. ..=90.).contains(&values[3]) {
            return Err(invalid());
        }
        if mag_limit.is_none_or(|m| values[5] <= m) {
            out.push(KnownStar {
                known_star_id: format!("NAVSTAR:{index}"),
                direction_j2000: unit(&values[..3])?,
            });
        }
    }
    if out.is_empty() {
        return Err(Error::value("public catalog contains no selected stars"));
    }
    Ok(out)
}

pub fn serializable<T: Serialize + ?Sized>(value: &T) -> Result<Value> {
    value.serialize(ReportSerializer)
}

fn json<T: Serialize>(v: T) -> Result<Value> {
    serde_json::to_value(v).map_err(|e| Error::value(e.to_string()))
}

fn float(x: f64) -> Result<Value> {
    if x == f64::NEG_INFINITY {
        Ok(Value::String("-inf".into()))
    } else if !x.is_finite() {
        Err(Error::value(
            "nonfinite positive value/NaN must not enter a report",
        ))
    } else {
        json(x)
    }
}

impl serde::ser::Error for Error {
    fn custom<T: std::fmt::Display>(message: T) -> Self {
        Self::value(message.to_string())
    }
}

struct ReportSerializer;
struct Sequence {
    values: Vec<Value>,
    variant: Option<&'static str>,
}

struct Object {
    values: Map<String, Value>,
    key: Option<String>,
    name: Option<&'static str>,
    variant: Option<&'static str>,
}

macro_rules! scalar{($($method:ident:$ty:ty),
*)=>{$(fn $method(self,
v:$ty)->Result<Value>{json(v)})*};
}

impl serde::Serializer for ReportSerializer {
    type Ok = Value;
    type Error = Error;
    type SerializeSeq = Sequence;
    type SerializeTuple = Sequence;
    type SerializeTupleStruct = Sequence;
    type SerializeTupleVariant = Sequence;
    type SerializeMap = Object;
    type SerializeStruct = Object;
    type SerializeStructVariant = Object;
    scalar!(serialize_bool:bool,
serialize_i8:i8,
serialize_i16:i16,
serialize_i32:i32,
serialize_i64:i64,
serialize_i128:i128,
serialize_u8:u8,
serialize_u16:u16,
serialize_u32:u32,
serialize_u64:u64,
serialize_u128:u128,
serialize_char:char);
    fn serialize_f32(self, v: f32) -> Result<Value> {
        float(v as f64)
    }
    fn serialize_f64(self, v: f64) -> Result<Value> {
        float(v)
    }
    fn serialize_str(self, v: &str) -> Result<Value> {
        json(v)
    }
    fn serialize_bytes(self, v: &[u8]) -> Result<Value> {
        json(v)
    }
    fn serialize_none(self) -> Result<Value> {
        Ok(Value::Null)
    }
    fn serialize_some<T: Serialize + ?Sized>(self, v: &T) -> Result<Value> {
        serializable(v)
    }
    fn serialize_unit(self) -> Result<Value> {
        Ok(Value::Null)
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<Value> {
        Ok(Value::Null)
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<Value> {
        json(variant)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        v: &T,
    ) -> Result<Value> {
        if name == crate::exact_factors::wire::FLOAT_MARKER {
            let value = json(v)?;
            return match value.as_str() {
                Some("-Infinity") => float(f64::NEG_INFINITY),
                Some("Infinity") => float(f64::INFINITY),
                Some("NaN") => float(f64::NAN),
                _ => Ok(value),
            };
        }
        serializable(v)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        v: &T,
    ) -> Result<Value> {
        Ok(Value::Object(Map::from_iter([(
            variant.into(),
            serializable(v)?,
        )])))
    }
    fn serialize_seq(self, len: Option<usize>) -> Result<Sequence> {
        Ok(Sequence {
            values: Vec::with_capacity(len.unwrap_or(0)),
            variant: None,
        })
    }
    fn serialize_tuple(self, len: usize) -> Result<Sequence> {
        self.serialize_seq(Some(len))
    }
    fn serialize_tuple_struct(self, _: &'static str, len: usize) -> Result<Sequence> {
        self.serialize_seq(Some(len))
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Sequence> {
        Ok(Sequence {
            values: Vec::with_capacity(len),
            variant: Some(variant),
        })
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Object> {
        Ok(Object {
            values: Map::new(),
            key: None,
            name: None,
            variant: None,
        })
    }
    fn serialize_struct(self, name: &'static str, _: usize) -> Result<Object> {
        Ok(Object {
            values: Map::new(),
            key: None,
            name: Some(name),
            variant: None,
        })
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Object> {
        Ok(Object {
            values: Map::new(),
            key: None,
            name: None,
            variant: Some(variant),
        })
    }
}

impl Sequence {
    fn add<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<()> {
        self.values.push(serializable(v)?);
        Ok(())
    }
    fn finish(self) -> Result<Value> {
        let v = Value::Array(self.values);
        Ok(if let Some(tag) = self.variant {
            Value::Object(Map::from_iter([(tag.into(), v)]))
        } else {
            v
        })
    }
}

macro_rules! sequence_impl {
    ($trait:ident,
$method:ident) => {
        impl $trait for Sequence {
            type Ok = Value;
            type Error = Error;
            fn $method<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<()> {
                self.add(v)
            }
            fn end(self) -> Result<Value> {
                self.finish()
            }
        }
    };
}

sequence_impl!(SerializeSeq, serialize_element);
sequence_impl!(SerializeTuple, serialize_element);
sequence_impl!(SerializeTupleStruct, serialize_field);
sequence_impl!(SerializeTupleVariant, serialize_field);
impl Object {
    fn field<T: Serialize + ?Sized>(&mut self, key: &str, v: &T) -> Result<()> {
        self.values.insert(key.into(), serializable(v)?);
        Ok(())
    }
    fn finish(self) -> Result<Value> {
        if self.name == Some("$serde_json::private::Number") {
            let s = self
                .values
                .get("$serde_json::private::Number")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::value("invalid arbitrary precision number"))?;
            return serde_json::from_str(s).map_err(|e| Error::value(e.to_string()));
        }
        let v = Value::Object(self.values);
        Ok(if let Some(tag) = self.variant {
            Value::Object(Map::from_iter([(tag.into(), v)]))
        } else {
            v
        })
    }
}

impl SerializeMap for Object {
    type Ok = Value;
    type Error = Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<()> {
        let value = serializable(key)?;
        self.key = Some(match value {
            Value::String(s) => s,
            Value::Bool(b) => if b { "True" } else { "False" }.into(),
            Value::Null => "None".into(),
            Value::Number(n) => n.to_string(),
            _ => return Err(Error::value("report keys must be scalar")),
        });
        Ok(())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<()> {
        let key = self
            .key
            .take()
            .ok_or_else(|| Error::value("report map value without key"))?;
        self.field(&key, value)
    }
    fn end(self) -> Result<Value> {
        self.finish()
    }
}

impl SerializeStruct for Object {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, key: &'static str, v: &T) -> Result<()> {
        self.field(key, v)
    }
    fn end(self) -> Result<Value> {
        self.finish()
    }
}

impl SerializeStructVariant for Object {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, key: &'static str, v: &T) -> Result<()> {
        self.field(key, v)
    }
    fn end(self) -> Result<Value> {
        self.finish()
    }
}
