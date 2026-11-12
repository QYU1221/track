use crate::{
    data_io::{io_error, py_text, Row},
    error::{Error, Result},
    table_io,
};
use quick_xml::{events::Event, Reader};
use serde_json::{json, Value};
use std::{collections::HashSet, fs::File, io::BufReader, path::Path};

fn error(e: impl std::fmt::Display) -> Error {
    Error::value(format!("invalid XLSX: {e}"))
}

pub fn visit_batches(
    path: &Path,
    max_rows: usize,
    mut consume: impl FnMut(&[String], &[Row]) -> Result<()>,
) -> Result<()> {
    if max_rows == 0 {
        return Err(Error::value("max_rows must be a positive integer"));
    }
    let mut header = None;
    let mut batch = Vec::with_capacity(512);
    let mut index = 0usize;
    let mut emit = |cells: Vec<Value>| {
        index += 1;
        let Some(fields) = &header else {
            let fields = cells
                .iter()
                .map(|v| {
                    if v.is_null() {
                        String::new()
                    } else {
                        py_text(v).trim().to_owned()
                    }
                })
                .collect::<Vec<_>>();
            if fields.iter().any(String::is_empty)
                || fields
                    .iter()
                    .map(|f| f.to_lowercase())
                    .collect::<HashSet<_>>()
                    .len()
                    != fields.len()
            {
                return Err(Error::value("table has blank or duplicate column names"));
            }
            header = Some(fields);
            return Ok(());
        };
        if cells
            .iter()
            .all(|v| v.is_null() || py_text(v).trim().is_empty())
        {
            return Ok(());
        }
        if cells.len() != fields.len() {
            return Err(Error::value(format!(
                "row {index} has a different column count"
            )));
        }
        batch.push(fields.iter().cloned().zip(cells).collect());
        if batch.len() == 512 {
            consume(fields, &batch)?;
            batch.clear();
        }
        Ok(())
    };
    match path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase()
        .as_str()
    {
        "csv" => table_io::csv_visit(path, max_rows, &mut emit)?,
        "xlsx" => xlsx_visit(path, max_rows, &mut emit)?,
        _ => return Err(Error::value("table must be CSV or XLSX")),
    }
    let header =
        header.ok_or_else(|| Error::value("table must contain a header, even when empty"))?;
    consume(&header, &batch)
}

fn xlsx_visit(
    path: &Path,
    max_rows: usize,
    emit: &mut dyn FnMut(Vec<Value>) -> Result<()>,
) -> Result<()> {
    let mut zip = zip::ZipArchive::new(File::open(path).map_err(io_error)?).map_err(error)?;
    let (_, sheet_path) = table_io::active_sheet(&mut zip)?;
    if let Ok(styles) = table_io::xml_entry(&mut zip, "xl/styles.xml") {
        if styles.contains("numFmtId=\"14\"")
            || styles.contains("<numFmt ")
            || (15..=22).any(|n| styles.contains(&format!("numFmtId=\"{n}\"")))
            || (45..=47).any(|n| styles.contains(&format!("numFmtId=\"{n}\"")))
        {
            let (header, rows) = table_io::read_table(path, max_rows)?;
            emit(header.iter().map(|s| json!(s)).collect())?;
            for row in rows {
                emit(header.iter().map(|f| row[f].clone()).collect())?;
            }
            return Ok(());
        }
    }
    let mut shared = Vec::new();
    if let Ok(entry) = zip.by_name("xl/sharedStrings.xml") {
        let mut reader = Reader::from_reader(BufReader::new(entry));
        let mut buffer = Vec::new();
        let mut text = String::new();
        let mut in_text = false;
        let mut phonetic = false;
        loop {
            match reader.read_event_into(&mut buffer).map_err(error)? {
                Event::Start(e) => match e.local_name().as_ref() {
                    b"si" => text.clear(),
                    b"t" => in_text = true,
                    b"rPh" => phonetic = true,
                    _ => (),
                },
                Event::End(e) => match e.local_name().as_ref() {
                    b"si" => shared.push(text.clone()),
                    b"t" => in_text = false,
                    b"rPh" => phonetic = false,
                    _ => (),
                },
                Event::Text(t) if in_text && !phonetic => text.push_str(
                    &quick_xml::escape::unescape(&t.decode().map_err(error)?).map_err(error)?,
                ),
                Event::Eof => break,
                _ => (),
            }
            buffer.clear();
        }
    }
    let mut dimension = None;
    let mut maximum = (0u32, 0u32);
    let mut any = false;
    {
        let entry = zip.by_name(&sheet_path).map_err(error)?;
        let mut reader = Reader::from_reader(BufReader::new(entry));
        let mut buffer = Vec::new();
        loop {
            match reader.read_event_into(&mut buffer).map_err(error)? {
                Event::Start(e) | Event::Empty(e) => {
                    if e.local_name().as_ref() == b"dimension" {
                        dimension = table_io::attrs(&e, &reader)?
                            .get("ref")
                            .and_then(|s| s.split(':').next_back())
                            .and_then(table_io::position);
                    }
                    if e.local_name().as_ref() == b"c" {
                        if let Some(p) = table_io::attrs(&e, &reader)?
                            .get("r")
                            .and_then(|s| table_io::position(s))
                        {
                            maximum = (maximum.0.max(p.0), maximum.1.max(p.1));
                            any = true;
                        }
                    }
                }
                Event::Eof => break,
                _ => (),
            }
            buffer.clear();
        }
    }
    let (end_row, end_col) = dimension.unwrap_or(maximum);
    if end_row as usize > max_rows {
        return Err(Error::value("table exceeds configured row limit"));
    }
    if !any {
        return Ok(());
    }
    let entry = zip.by_name(&sheet_path).map_err(error)?;
    let mut reader = Reader::from_reader(BufReader::new(entry));
    let mut buffer = Vec::new();
    let mut row = vec![Value::Null; end_col as usize + 1];
    let mut row_number = 0u32;
    let mut next_row = 0u32;
    let mut cell = None;
    let mut kind = String::new();
    let mut text = String::new();
    let mut value = String::new();
    let mut in_value = false;
    let mut in_text = false;
    let mut formula = false;
    let mut cached = false;
    loop {
        match reader.read_event_into(&mut buffer).map_err(error)? {
            Event::Start(e) => match e.local_name().as_ref() {
                b"row" => {
                    row_number = table_io::attrs(&e, &reader)?
                        .get("r")
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(next_row + 1)
                        - 1;
                    row.fill(Value::Null);
                }
                b"c" => {
                    let a = table_io::attrs(&e, &reader)?;
                    cell = a.get("r").and_then(|s| table_io::position(s));
                    kind = a.get("t").cloned().unwrap_or_default();
                    text.clear();
                    value.clear();
                    formula = false;
                    cached = false;
                }
                b"v" => {
                    in_value = true;
                    cached = true;
                }
                b"t" => in_text = true,
                b"f" => formula = true,
                _ => (),
            },
            Event::Empty(e) if e.local_name().as_ref() == b"f" => formula = true,
            Event::Text(t) => {
                let decoded = t.decode().map_err(error)?;
                let decoded = quick_xml::escape::unescape(&decoded).map_err(error)?;
                if in_value {
                    value.push_str(&decoded);
                }
                if in_text {
                    text.push_str(&decoded);
                }
            }
            Event::End(e) => match e.local_name().as_ref() {
                b"v" => in_value = false,
                b"t" => in_text = false,
                b"c" => {
                    if let Some((r, c)) =
                        cell.take().filter(|(r, c)| *r <= end_row && *c <= end_col)
                    {
                        row_number = r;
                        row[c as usize] = if formula && !cached {
                            Value::Null
                        } else {
                            match kind.as_str() {
                                "inlineStr" => json!(text),
                                "s" => json!(shared
                                    .get(value.parse::<usize>().map_err(error)?)
                                    .ok_or_else(|| error("shared string index outside table"))?),
                                "b" => json!(value == "1"),
                                "str" | "e" | "d" => json!(value),
                                _ if value.is_empty() => Value::Null,
                                _ => {
                                    if !value.contains(['.', 'e', 'E'])
                                        && value.parse::<i64>().is_ok()
                                    {
                                        json!(value.parse::<i64>().unwrap())
                                    } else {
                                        json!(value.parse::<f64>().map_err(error)?)
                                    }
                                }
                            }
                        };
                    }
                }
                b"row" => {
                    if row_number <= end_row {
                        while next_row < row_number {
                            emit(vec![Value::Null; row.len()])?;
                            next_row += 1;
                        }
                        emit(row.clone())?;
                        next_row = row_number + 1;
                    }
                }
                _ => (),
            },
            Event::Eof => break,
            _ => (),
        }
        buffer.clear();
    }
    while next_row <= end_row {
        emit(vec![Value::Null; row.len()])?;
        next_row += 1;
    }
    Ok(())
}
