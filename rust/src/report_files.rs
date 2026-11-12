use crate::{
    data_io::io_error,
    error::{Error, Result},
    report_schema,
};
use serde_json::Value;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};
use zip::{write::SimpleFileOptions, ZipWriter};
pub fn csv(path: &Path, kind: &str, format: &str, rows: &[Value]) -> Result<()> {
    let mut writer = CsvWriter::new(path, kind, format)?;
    for row in rows {
        writer.write(row)?;
    }
    writer.finish()
}

pub fn cell_text(v: &Value) -> Result<String> {
    Ok(match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) if n.is_f64() => crate::input_evidence::python_float(n.as_f64().unwrap())?,
        _ => v.to_string(),
    })
}

fn xml(s: &str) -> String {
    s.chars()
        .filter(|c| *c == '\t' || *c == '\n' || *c == '\r' || *c >= ' ')
        .collect::<String>()
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn col(mut n: usize) -> String {
    let mut s = String::new();
    while n > 0 {
        n -= 1;
        s.insert(0, (b'A' + (n % 26) as u8) as char);
        n /= 26;
    }
    s
}

fn validate_strings(rows: &[Value]) -> Result<()> {
    fn visit(v: &Value) -> Result<()> {
        match v {
            Value::String(s)
                if s.chars().take(32767).any(|c| {
                    c <= '\u{8}'
                        || matches!(c, '\u{b}' | '\u{c}')
                        || ('\u{e}'..='\u{1f}').contains(&c)
                }) =>
            {
                Err(Error {
                    kind: "IllegalCharacterError",
                    code: None,
                    message: String::new(),
                })
            }
            Value::Array(a) => {
                for x in a {
                    visit(x)?;
                }
                Ok(())
            }
            Value::Object(m) => {
                for x in m.values() {
                    visit(x)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
    for row in rows {
        visit(row)?;
    }
    Ok(())
}

fn row_xml(row: &[Value], number: usize, header_style: bool) -> String {
    let mut out = String::new();
    out.push_str(&format!("<row r=\"{}\">", number));
    for (j, v) in row.iter().enumerate() {
        let reference = format!("{}{}", col(j + 1), number);
        let style = if header_style { " s=\"1\"" } else { "" };
        match v {
            Value::Null => {}
            Value::String(s) if s.is_empty() => {}
            Value::String(s) => out.push_str(&format!(
                "<c r=\"{reference}\" t=\"inlineStr\"{style}><is><t \
                        xml:space=\"preserve\">{}</t></is></c>",
                xml(&s.chars().take(32767).collect::<String>())
            )),
            Value::Bool(b) => out.push_str(&format!(
                "<c r=\"{reference}\" t=\"b\"><v>{}</v></c>",
                i32::from(*b)
            )),
            Value::Number(n) => out.push_str(&format!("<c r=\"{reference}\"><v>{n}</v></c>")),
            _ => out.push_str(&format!(
                "<c r=\"{reference}\" t=\"inlineStr\"><is><t>{}</t></is></c>",
                xml(&v.to_string())
            )),
        }
    }
    out.push_str("</row>");
    out
}

fn sheet(fields: &[String], rows: &[Vec<Value>], freeze: bool) -> String {
    let mut out = String::from(concat!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><worksheet ",
        "xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">",
    ));
    if freeze {
        out.push_str(concat!(
            "<sheetViews><sheetView workbookViewId=\"0\"><pane xSplit=\"2\" ",
            "ySplit=\"1\" topLeftCell=\"C2\" activePane=\"bottomRight\" ",
            "state=\"frozen\"/></sheetView></sheetViews><cols>",
        ));
        for (i, f) in fields.iter().enumerate() {
            let width = if f == "Decision_Reason" {
                44
            } else {
                (f.len() + 2).clamp(12, 36)
            };
            out.push_str(&format!(
                "<col min=\"{}\" max=\"{}\" width=\"{width}\" customWidth=\"1\"/>",
                i + 1,
                i + 1
            ));
        }
        out.push_str("</cols>");
    }
    out.push_str("<sheetData>");
    let header = fields
        .iter()
        .map(|s| Value::String(s.clone()))
        .collect::<Vec<_>>();
    for (i, row) in std::iter::once(&header).chain(rows.iter()).enumerate() {
        out.push_str(&row_xml(row, i + 1, i == 0 && freeze));
    }
    out.push_str("</sheetData>");
    if freeze {
        out.push_str(&format!(
            "<autoFilter ref=\"A1:{}{}\"/>",
            col(fields.len()),
            rows.len() + 1
        ));
    }
    out.push_str("</worksheet>");
    out
}

pub fn workbook_stream(
    path: &Path,
    format: &str,
    segments: &[Value],
    produce: impl FnOnce(&mut dyn FnMut(Value) -> Result<()>) -> Result<()>,
) -> Result<()> {
    let fields = report_schema::table_fields("detection", format)?;
    let temporary = path.with_extension("sheet.tmp");
    struct Temporary(std::path::PathBuf);
    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let file = File::options()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(io_error)?;
    let _cleanup = Temporary(temporary.clone());
    let mut output = BufWriter::new(file);
    let header = sheet(&fields, &[], true);
    let prefix = header.split("</sheetData>").next().unwrap();
    output.write_all(prefix.as_bytes()).map_err(io_error)?;
    let mut count = 0usize;
    produce(&mut |row| {
        let row = report_schema::table_row("detection", &row, format)?;
        let cells = fields.iter().map(|f| row[f].clone()).collect::<Vec<_>>();
        validate_strings(&cells)?;
        count += 1;
        if count >= 1_048_576 {
            return Err(Error::value("Excel worksheet row limit exceeded"));
        }
        let line = row_xml(&cells, count + 1, false);
        output.write_all(line.as_bytes()).map_err(io_error)
    })?;
    write!(
        output,
        "</sheetData><autoFilter ref=\"A1:{}{}\"/></worksheet>",
        col(fields.len()),
        count + 1
    )
    .map_err(io_error)?;
    output.flush().map_err(io_error)?;
    drop(output);
    let segment_fields = [
        "segment_id",
        "reason",
        "previous_frame",
        "previous_timestamp_s",
        "start_frame",
        "start_timestamp_s",
        "gap_s",
        "max_dt_s",
    ]
    .map(str::to_owned)
    .to_vec();
    let segment_rows = segments
        .iter()
        .map(|e| {
            segment_fields
                .iter()
                .map(|f| e[f].clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for row in segment_rows.iter() {
        validate_strings(row)?;
    }
    let mut sheets = vec![("Sheet", String::new())];
    if !segments.is_empty() {
        sheets.push((
            "Tracking_Segments",
            sheet(&segment_fields, &segment_rows, false),
        ));
    }
    let pending = path.with_extension("xlsx.pending");
    let _pending_cleanup = Temporary(pending.clone());
    package(&pending, &sheets, &temporary)?;
    std::fs::rename(&pending, path).map_err(io_error)
}

fn package(path: &Path, sheets: &[(&str, String)], first_sheet: &Path) -> Result<()> {
    let mut zip = ZipWriter::new(File::create(path).map_err(io_error)?);
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    fn put(
        zip: &mut ZipWriter<File>,
        options: SimpleFileOptions,
        name: &str,
        content: &str,
    ) -> Result<()> {
        zip.start_file(name, options)
            .map_err(|e| Error::value(e.to_string()))?;
        zip.write_all(content.as_bytes()).map_err(io_error)
    }
    let mut types = String::from(concat!(
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-ty",
        "pes\"><Default Extension=\"rels\" ",
        "ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"",
        "/><Default Extension=\"xml\" ContentType=\"application/xml\"/><Override ",
        "PartName=\"/xl/workbook.xml\" ",
        "ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetm",
        "l.sheet.main+xml\"/><Override PartName=\"/xl/styles.xml\" ",
        "ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetm",
        "l.styles+xml\"/>",
    ));
    for i in 1..=sheets.len() {
        types.push_str(&format!(
            "<Override PartName=\"/xl/worksheets/sheet{i}.xml\" \
                ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetm\
                l.worksheet+xml\"/>"
        ));
    }
    types.push_str("</Types>");
    put(&mut zip, options, "[Content_Types].xml", &types)?;
    put(
        &mut zip,
        options,
        "_rels/.rels",
        concat!(
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/re",
            "lationships\"><Relationship Id=\"rId1\" ",
            "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationshi",
            "ps/officeDocument\" Target=\"xl/workbook.xml\"/></Relationships>",
        ),
    )?;
    let mut book = String::from(concat!(
        "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/m",
        "ain\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/re",
        "lationships\"><sheets>",
    ));
    let mut rels = String::from(
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">",
    );
    for (i, (name, content)) in sheets.iter().enumerate() {
        book.push_str(&format!(
            "<sheet name=\"{name}\" sheetId=\"{}\" r:id=\"rId{}\"/>",
            i + 1,
            i + 1
        ));
        rels.push_str(&format!(
            concat!(
                "<Relationship Id=\"rId{}\" ",
                "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationshi",
                "ps/worksheet\" Target=\"worksheets/sheet{}.xml\"/>",
            ),
            i + 1,
            i + 1
        ));
        if i == 0 {
            zip.start_file("xl/worksheets/sheet1.xml", options)
                .map_err(|e| Error::value(e.to_string()))?;
            std::io::copy(&mut File::open(first_sheet).map_err(io_error)?, &mut zip)
                .map_err(io_error)?;
        } else {
            put(
                &mut zip,
                options,
                &format!("xl/worksheets/sheet{}.xml", i + 1),
                content,
            )?;
        }
    }
    book.push_str("</sheets></workbook>");
    rels.push_str(concat!(
        "<Relationship Id=\"styles\" ",
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationshi",
        "ps/styles\" Target=\"styles.xml\"/></Relationships>",
    ));
    put(&mut zip, options, "xl/workbook.xml", &book)?;
    put(&mut zip, options, "xl/_rels/workbook.xml.rels", &rels)?;
    put(
        &mut zip,
        options,
        "xl/styles.xml",
        concat!(
            "<styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006",
            "/main\"><fonts count=\"2\"><font><sz val=\"11\"/><name val=\"Calibri\"/>",
            "</font><font><b/><sz val=\"11\"/><color rgb=\"FFFFFFFF\"/><name ",
            "val=\"Calibri\"/></font></fonts><fills count=\"3\"><fill><patternFill ",
            "patternType=\"none\"/></fill><fill><patternFill ",
            "patternType=\"gray125\"/></fill><fill><patternFill ",
            "patternType=\"solid\"><fgColor rgb=\"FF203864\"/></patternFill></fill>",
            "</fills><borders count=\"1\"><border/></borders><cellStyleXfs ",
            "count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" ",
            "borderId=\"0\"/></cellStyleXfs><cellXfs count=\"2\"><xf numFmtId=\"0\" ",
            "fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\"/><xf numFmtId=\"0\" ",
            "fontId=\"1\" fillId=\"2\" borderId=\"0\" xfId=\"0\"/></cellXfs>",
            "<cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" ",
            "builtinId=\"0\"/></cellStyles></styleSheet>",
        ),
    )?;
    zip.finish().map_err(|e| Error::value(e.to_string()))?;
    Ok(())
}

pub struct CsvWriter {
    output: BufWriter<File>,
    fields: Vec<String>,
    kind: String,
    format: String,
}
impl CsvWriter {
    pub fn new(path: &Path, kind: &str, format: &str) -> Result<Self> {
        let fields = report_schema::table_fields(kind, format)?;
        let mut result = Self {
            output: BufWriter::new(File::create(path).map_err(io_error)?),
            fields,
            kind: kind.into(),
            format: format.into(),
        };
        result.output.write_all(b"\xef\xbb\xbf").map_err(io_error)?;
        result.line(result.fields.clone())?;
        Ok(result)
    }
    fn line(&mut self, values: Vec<String>) -> Result<()> {
        crate::table_io::write_csv_record(&mut self.output, &values)
    }

    pub fn write(&mut self, row: &Value) -> Result<()> {
        let row = report_schema::table_row(&self.kind, row, &self.format)?;
        let values = self
            .fields
            .iter()
            .map(|f| cell_text(&row[f]))
            .collect::<Result<Vec<_>>>()?;
        self.line(values)
    }
    pub fn finish(mut self) -> Result<()> {
        self.output.flush().map_err(io_error)
    }
}
