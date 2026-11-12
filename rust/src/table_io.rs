use crate::{
    data_io::{io_error, py_repr, py_text, Row},
    error::{Error, Result},
};
use calamine::{Data, Reader as ExcelReader, Xlsx};
use quick_xml::{
    events::{BytesStart, Event},
    Reader,
};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufReader, Read, Write},
    path::Path,
};

fn csv_rows(path: &Path, max_rows: usize) -> Result<Vec<Vec<Value>>> {
    let mut rows = Vec::new();
    csv_visit(path, max_rows, |row| {
        rows.push(row);
        Ok(())
    })?;
    Ok(rows)
}

pub(crate) fn csv_visit(
    path: &Path,
    max_rows: usize,
    mut emit: impl FnMut(Vec<Value>) -> Result<()>,
) -> Result<()> {
    let mut input = BufReader::new(File::open(path).map_err(io_error)?)
        .bytes()
        .peekable();
    let mut prefix = Vec::new();
    for _ in 0..3 {
        if let Some(b) = input.next() {
            prefix.push(b.map_err(io_error)?);
        }
    }
    let start = if prefix == [0xef, 0xbb, 0xbf] {
        vec![]
    } else {
        prefix
    };
    let mut bytes = start.into_iter().map(Ok).chain(input).peekable();
    let mut row_count = 0usize;
    let mut row = Vec::new();
    let mut field = Vec::new();
    let mut quoted = false;
    let mut beginning = true;
    let mut touched = false;
    fn string(bytes: &mut Vec<u8>) -> Result<Value> {
        String::from_utf8(std::mem::take(bytes))
            .map(Value::String)
            .map_err(|e| Error {
                kind: "UnicodeDecodeError",
                code: None,
                message: e.to_string(),
            })
    }
    while let Some(byte) = bytes.next() {
        let b = byte.map_err(io_error)?;
        if quoted {
            if b == b'"' {
                if matches!(bytes.peek(), Some(Ok(b'"'))) {
                    bytes.next();
                    field.push(b'"');
                } else {
                    quoted = false;
                }
            } else {
                field.push(b);
            }
            continue;
        }
        match b {
            b'"' if beginning => {
                quoted = true;
                beginning = false;
                touched = true;
            }
            b',' => {
                row.push(string(&mut field)?);
                beginning = true;
                touched = true;
            }
            b'\r' | b'\n' => {
                if b == b'\r' && matches!(bytes.peek(), Some(Ok(b'\n'))) {
                    bytes.next();
                }
                if touched || !field.is_empty() || !row.is_empty() {
                    row.push(string(&mut field)?);
                }
                if row_count > max_rows {
                    return Err(Error::value("table exceeds configured row limit"));
                }
                emit(std::mem::take(&mut row))?;
                row_count += 1;
                beginning = true;
                touched = false;
            }
            _ => {
                field.push(b);
                beginning = false;
                touched = true;
            }
        }
    }
    if touched || !field.is_empty() || !row.is_empty() {
        row.push(string(&mut field)?);
        if row_count > max_rows {
            return Err(Error::value("table exceeds configured row limit"));
        }
        emit(row)?;
    }
    Ok(())
}

fn xml_error(e: impl std::fmt::Display) -> Error {
    Error {
        kind: "ValueError",
        code: None,
        message: format!("invalid XLSX: {e}"),
    }
}

pub(crate) fn xml_entry(zip: &mut zip::ZipArchive<File>, name: &str) -> Result<String> {
    let mut text = String::new();
    zip.by_name(name)
        .map_err(xml_error)?
        .read_to_string(&mut text)
        .map_err(io_error)?;
    Ok(text)
}

pub(crate) fn attrs<R: std::io::BufRead>(
    e: &BytesStart<'_>,
    reader: &Reader<R>,
) -> Result<HashMap<String, String>> {
    e.attributes()
        .map(|a| {
            let a = a.map_err(xml_error)?;
            Ok((
                String::from_utf8_lossy(a.key.as_ref()).to_string(),
                a.decode_and_unescape_value(reader.decoder())
                    .map_err(xml_error)?
                    .into_owned(),
            ))
        })
        .collect()
}

pub(crate) fn position(s: &str) -> Option<(u32, u32)> {
    let mut col = 0u32;
    let mut row = String::new();
    for c in s.chars().filter(|c| *c != '$') {
        if c.is_ascii_alphabetic() {
            col = col
                .checked_mul(26)?
                .checked_add(c.to_ascii_uppercase() as u32 - 'A' as u32 + 1)?;
        } else {
            row.push(c);
        }
    }
    Some((
        row.parse::<u32>().ok()?.checked_sub(1)?,
        col.checked_sub(1)?,
    ))
}

pub(crate) fn active_sheet(zip: &mut zip::ZipArchive<File>) -> Result<(String, String)> {
    let workbook = xml_entry(zip, "xl/workbook.xml")?;
    let mut reader = Reader::from_str(&workbook);
    let mut active = 0usize;
    let mut sheets = Vec::new();
    loop {
        match reader.read_event().map_err(xml_error)? {
            Event::Empty(e) | Event::Start(e) => {
                let name = e.local_name();
                if name.as_ref() == b"workbookView" {
                    active = attrs(&e, &reader)?
                        .get("activeTab")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                }
                if name.as_ref() == b"sheet" {
                    let a = attrs(&e, &reader)?;
                    sheets.push((
                        a.get("name").cloned().unwrap_or_default(),
                        a.get("r:id").cloned().unwrap_or_default(),
                    ));
                }
            }
            Event::Eof => break,
            _ => (),
        }
    }
    let (sheet_name, rid) = sheets
        .get(active)
        .ok_or_else(|| xml_error("active worksheet unavailable"))?;
    let rels = xml_entry(zip, "xl/_rels/workbook.xml.rels")?;
    let mut reader = Reader::from_str(&rels);
    let mut target = None;
    loop {
        match reader.read_event().map_err(xml_error)? {
            Event::Start(e) | Event::Empty(e) => {
                if e.local_name().as_ref() == b"Relationship" {
                    let a = attrs(&e, &reader)?;
                    if a.get("Id") == Some(rid) {
                        target = a.get("Target").cloned();
                    }
                }
            }
            Event::Eof => break,
            _ => (),
        }
    }
    let target = target.ok_or_else(|| xml_error("worksheet relationship missing"))?;
    let target = if target.starts_with('/') {
        target.trim_start_matches('/').to_string()
    } else {
        format!("xl/{target}")
    };
    let mut parts = Vec::new();
    for p in target.split('/') {
        match p {
            ".." => {
                parts.pop();
            }
            "." => (),
            _ => parts.push(p),
        }
    }
    Ok((sheet_name.clone(), parts.join("/")))
}

fn xlsx_rows(path: &Path, max_rows: usize) -> Result<Vec<Vec<Value>>> {
    let mut zip = zip::ZipArchive::new(File::open(path).map_err(io_error)?).map_err(xml_error)?;
    let (sheet_name, sheet_path) = active_sheet(&mut zip)?;
    let sheet = xml_entry(&mut zip, &sheet_path)?;
    let mut reader = Reader::from_str(&sheet);
    let mut dimension = None;
    let mut max_position = (0, 0);
    let mut any_cell = false;
    let mut raw_int = HashSet::new();
    let mut current = None;
    let mut numeric = false;
    let mut in_v = false;
    let mut formulas = HashSet::new();
    let mut cached = HashSet::new();
    loop {
        match reader.read_event().map_err(xml_error)? {
            Event::Start(e) | Event::Empty(e) => {
                let local = e.local_name();
                let name = local.as_ref();
                if name == b"dimension" {
                    dimension = attrs(&e, &reader)?
                        .get("ref")
                        .and_then(|s| s.split(':').next_back())
                        .and_then(position);
                }
                if name == b"c" {
                    let a = attrs(&e, &reader)?;
                    current = a.get("r").and_then(|s| position(s));
                    numeric = a.get("t").is_none_or(|s| s == "n");
                    if let Some(p) = current {
                        any_cell = true;
                        max_position = (max_position.0.max(p.0), max_position.1.max(p.1));
                    }
                }
                if name == b"v" {
                    in_v = true;
                }
                if name == b"f" {
                    if let Some(p) = current {
                        formulas.insert(p);
                    }
                }
            }
            Event::Text(t) if in_v => {
                let s = t.decode().map_err(xml_error)?;
                if let Some(p) = current {
                    cached.insert(p);
                }
                if numeric && !s.contains(['.', 'e', 'E']) {
                    if let Some(p) = current {
                        raw_int.insert(p);
                    }
                }
            }
            Event::End(e) => {
                if e.local_name().as_ref() == b"v" {
                    in_v = false;
                }
                if e.local_name().as_ref() == b"c" {
                    current = None;
                    numeric = false;
                }
            }
            Event::Eof => break,
            _ => (),
        }
    }
    let (end_row, end_col) = dimension.unwrap_or(max_position);
    if end_row as usize > max_rows {
        return Err(Error::value("table exceeds configured row limit"));
    }
    if !any_cell {
        return Ok(vec![]);
    }
    let mut book: Xlsx<_> = calamine::open_workbook(path).map_err(xml_error)?;
    let range = book.worksheet_range(&sheet_name).map_err(xml_error)?;
    let mut rows = Vec::new();
    for r in 0..=end_row {
        let mut row = Vec::new();
        for c in 0..=end_col {
            let value = match if formulas.contains(&(r, c)) && !cached.contains(&(r, c)) {
                &Data::Empty
            } else {
                range.get_value((r, c)).unwrap_or(&Data::Empty)
            } {
                Data::Empty => Value::Null,
                Data::String(s) => json!(s),
                Data::Bool(b) => json!(b),
                Data::Int(n) => json!(n),
                Data::Float(n) => {
                    if raw_int.contains(&(r, c)) && *n >= i64::MIN as f64 && *n < i64::MAX as f64 {
                        json!(*n as i64)
                    } else {
                        json!(n)
                    }
                }
                Data::Error(e) => json!(e.to_string()),
                Data::DateTime(d) => {
                    let dt = d
                        .as_datetime()
                        .ok_or_else(|| xml_error("datetime outside supported range"))?;
                    if d.is_duration() {
                        return Err(xml_error("duration cells are not supported"));
                    }
                    let format = if d.as_f64() >= 0. && d.as_f64() < 1. {
                        "%H:%M:%S"
                    } else {
                        "%Y-%m-%d %H:%M:%S"
                    };
                    let mut s = dt.format(format).to_string();
                    use chrono::Timelike;
                    if dt.nanosecond() != 0 {
                        s.push_str(&format!(".{:06}", dt.nanosecond() / 1000));
                    }
                    json!(s)
                }
                Data::DateTimeIso(s) | Data::DurationIso(s) => json!(s),
            };
            row.push(value);
        }
        rows.push(row);
    }
    Ok(rows)
}

pub fn read_table(path: &Path, max_rows: usize) -> Result<(Vec<String>, Vec<Row>)> {
    if max_rows < 1 {
        return Err(Error::value("max_rows must be a positive integer"));
    }
    let rows = match path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase()
        .as_str()
    {
        "csv" => csv_rows(path, max_rows)?,
        "xlsx" => xlsx_rows(path, max_rows)?,
        _ => return Err(Error::value("table must be CSV or XLSX")),
    };
    let first = rows
        .first()
        .ok_or_else(|| Error::value("table must contain a header, even when empty"))?;
    let header: Vec<_> = first
        .iter()
        .map(|v| {
            if v.is_null() {
                String::new()
            } else {
                py_text(v).trim().to_owned()
            }
        })
        .collect();
    if header.iter().any(String::is_empty)
        || header
            .iter()
            .map(|s| s.to_lowercase())
            .collect::<HashSet<_>>()
            .len()
            != header.len()
    {
        return Err(Error::value("table has blank or duplicate column names"));
    }
    let mut records = Vec::new();
    for (index, row) in rows.into_iter().enumerate().skip(1) {
        if row
            .iter()
            .all(|v| v.is_null() || py_text(v).trim().is_empty())
        {
            continue;
        }
        if row.len() != header.len() {
            return Err(Error::value(format!(
                "row {} has a different column count",
                index + 1
            )));
        }
        records.push(header.iter().cloned().zip(row).collect());
        if records.len() > max_rows {
            return Err(Error::value("table exceeds configured row limit"));
        }
    }
    Ok((header, records))
}

pub fn write_csv(path: &Path, fields: &[&str], rows: &[Row], ignore_extras: bool) -> Result<()> {
    let mut output = std::io::BufWriter::new(File::create(path).map_err(io_error)?);
    write_csv_record(
        &mut output,
        &fields.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
    )?;
    for row in rows {
        let extra: Vec<_> = row
            .keys()
            .filter(|k| !fields.contains(&k.as_str()))
            .collect();
        if !ignore_extras && !extra.is_empty() {
            return Err(Error::value(format!(
                "dict contains fields not in fieldnames: {}",
                extra
                    .iter()
                    .map(|k| py_repr(&json!(k)))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        write_csv_record(
            &mut output,
            &fields
                .iter()
                .map(|k| match row.get(*k) {
                    None | Some(Value::Null) => String::new(),
                    Some(v) => py_text(v),
                })
                .collect::<Vec<_>>(),
        )?;
    }
    output.flush().map_err(io_error)
}

pub(crate) fn write_csv_record(out: &mut impl Write, values: &[String]) -> Result<()> {
    for (i, s) in values.iter().enumerate() {
        if i > 0 {
            out.write_all(b",").map_err(io_error)?;
        }
        if s.contains([',', '"', '\r', '\n']) || (values.len() == 1 && s.is_empty()) {
            out.write_all(format!("\"{}\"", s.replace('"', "\"\"")).as_bytes())
                .map_err(io_error)?;
        } else {
            out.write_all(s.as_bytes()).map_err(io_error)?;
        }
    }
    out.write_all(b"\r\n").map_err(io_error)
}
