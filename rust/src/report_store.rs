use crate::{
    data_io::io_error,
    error::{Error, Result},
};
use serde_json::{json, Value};
use std::{
    ffi::{c_char, c_int, c_void, CStr, CString},
    path::Path,
    ptr,
};

struct SqliteApi {
    open: unsafe extern "C" fn(*const c_char, *mut *mut c_void) -> c_int,
    close: unsafe extern "C" fn(*mut c_void) -> c_int,
    errmsg: unsafe extern "C" fn(*mut c_void) -> *const c_char,
    exec: unsafe extern "C" fn(
        *mut c_void,
        *const c_char,
        *const c_void,
        *mut c_void,
        *mut *mut c_char,
    ) -> c_int,
    prepare: unsafe extern "C" fn(
        *mut c_void,
        *const c_char,
        c_int,
        *mut *mut c_void,
        *mut *const c_char,
    ) -> c_int,
    step: unsafe extern "C" fn(*mut c_void) -> c_int,
    text: unsafe extern "C" fn(*mut c_void, c_int) -> *const u8,
    finalize: unsafe extern "C" fn(*mut c_void) -> c_int,
}
impl SqliteApi {
    fn get() -> Result<&'static Self> {
        static API: std::sync::OnceLock<Result<SqliteApi>> = std::sync::OnceLock::new();
        API.get_or_init(|| {
            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
                fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
            }
            let name: Vec<u16> = "winsqlite3.dll\0".encode_utf16().collect();
            let module = unsafe { LoadLibraryExW(name.as_ptr(), ptr::null_mut(), 0x800) };
            if module.is_null() {
                return Err(io_error(std::io::Error::last_os_error()));
            }
            macro_rules! load {
                ($name:literal) => {{
                    let address =
                        unsafe { GetProcAddress(module, concat!($name, "\0").as_ptr().cast()) };
                    if address.is_null() {
                        return Err(Error::value(concat!(
                            "missing Windows SQLite function: ",
                            $name
                        )));
                    }
                    unsafe { std::mem::transmute(address) }
                }};
            }
            Ok(Self {
                open: load!("sqlite3_open"),
                close: load!("sqlite3_close"),
                errmsg: load!("sqlite3_errmsg"),
                exec: load!("sqlite3_exec"),
                prepare: load!("sqlite3_prepare_v2"),
                step: load!("sqlite3_step"),
                text: load!("sqlite3_column_text"),
                finalize: load!("sqlite3_finalize"),
            })
        })
        .as_ref()
        .map_err(Clone::clone)
    }
}
pub struct ReportStore {
    db: *mut c_void,
    api: &'static SqliteApi,
}
struct Statement {
    raw: *mut c_void,
    api: &'static SqliteApi,
}
impl Drop for Statement {
    fn drop(&mut self) {
        unsafe {
            (self.api.finalize)(self.raw);
        }
    }
}
impl Drop for ReportStore {
    fn drop(&mut self) {
        unsafe {
            (self.api.close)(self.db);
        }
    }
}

impl ReportStore {
    pub fn new(path: &Path) -> Result<Self> {
        if path.exists() {
            return Err(Error::value("report history already exists"));
        }
        let filename = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|e| Error::value(e.to_string()))?;
        let api = SqliteApi::get()?;
        let mut db = ptr::null_mut();
        let status = unsafe { (api.open)(filename.as_ptr(), &mut db) };
        if db.is_null() {
            return Err(io_error(std::io::Error::other(
                "cannot open report history",
            )));
        }
        let store = Self { db, api };
        if status != 0 {
            return Err(store.error());
        }
        store.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA cache_size=-2048; PRAGMA temp_store=FILE; CREATE TABLE frames(frame INTEGER PRIMARY KEY, payload TEXT NOT NULL); CREATE TABLE track_rows(track INTEGER, frame INTEGER, payload TEXT NOT NULL, PRIMARY KEY(track,frame)); CREATE TABLE final_tracks(track INTEGER PRIMARY KEY, payload TEXT NOT NULL);")?;
        Ok(store)
    }
    fn error(&self) -> Error {
        Error::value(format!(
            "report history: {}",
            unsafe { CStr::from_ptr((self.api.errmsg)(self.db)) }.to_string_lossy()
        ))
    }
    fn exec(&self, sql: &str) -> Result<()> {
        let sql = CString::new(sql).map_err(|e| Error::value(e.to_string()))?;
        if unsafe {
            (self.api.exec)(
                self.db,
                sql.as_ptr(),
                ptr::null(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        } != 0
        {
            return Err(self.error());
        }
        Ok(())
    }
    pub fn commit_frame(
        &self,
        frame: i64,
        points: &[((i64, String), Value)],
        events: &[Value],
        tracks: &[(usize, Value)],
        statistics: &Value,
        diagnostics: &Value,
    ) -> Result<()> {
        let payload = serde_json::to_string(&json!({"points": points, "events": events, "statistics": statistics, "diagnostics": diagnostics}))
            .map_err(|e| Error::value(e.to_string()))?;
        self.exec("BEGIN IMMEDIATE")?;
        if let Err(error) = self
            .exec(&format!(
                "INSERT INTO frames VALUES({frame}, '{}')",
                payload.replace('\'', "''")
            ))
            .and_then(|_| {
                for (track, row) in tracks {
                    let payload =
                        serde_json::to_string(row).map_err(|e| Error::value(e.to_string()))?;
                    self.exec(&format!(
                        "INSERT INTO track_rows VALUES({track}, {frame}, '{}')",
                        payload.replace('\'', "''")
                    ))?;
                }
                self.exec("COMMIT")
            })
        {
            let _ = self.exec("ROLLBACK");
            return Err(error);
        }
        Ok(())
    }
    fn visit(&self, sql: &str, mut consume: impl FnMut(Value) -> Result<()>) -> Result<()> {
        let sql = CString::new(sql).map_err(|e| Error::value(e.to_string()))?;
        let mut raw = ptr::null_mut();
        let status =
            unsafe { (self.api.prepare)(self.db, sql.as_ptr(), -1, &mut raw, ptr::null_mut()) };
        let statement = Statement { raw, api: self.api };
        if status != 0 {
            return Err(self.error());
        }
        loop {
            match unsafe { (self.api.step)(statement.raw) } {
                100 => {
                    let text = unsafe { CStr::from_ptr((self.api.text)(statement.raw, 0).cast()) };
                    let value = serde_json::from_slice(text.to_bytes())
                        .map_err(|e| Error::value(e.to_string()))?;
                    consume(value)?;
                }
                101 => return Ok(()),
                _ => return Err(self.error()),
            }
        }
    }
    pub fn track_rows(&self, track: usize) -> Result<Vec<Value>> {
        let mut rows = Vec::new();
        self.visit(
            &format!("SELECT payload FROM final_tracks WHERE track={track}"),
            |value| {
                rows = serde_json::from_value(value).map_err(|e| Error::value(e.to_string()))?;
                Ok(())
            },
        )?;
        if rows.is_empty() {
            self.visit(
                &format!("SELECT payload FROM track_rows WHERE track={track} ORDER BY frame"),
                |value| {
                    rows.push(value);
                    Ok(())
                },
            )?;
        }
        Ok(rows)
    }
    pub fn finalize_track(&self, track: usize, rows: &[Value]) -> Result<()> {
        let payload = serde_json::to_string(rows).map_err(|e| Error::value(e.to_string()))?;
        self.exec(&format!(
            "INSERT INTO final_tracks VALUES({track}, '{}')",
            payload.replace('\'', "''")
        ))
    }
    pub fn frame_records(
        &self,
        field: &str,
        mut consume: impl FnMut(Value) -> Result<()>,
    ) -> Result<()> {
        self.visit("SELECT payload FROM frames ORDER BY frame", |mut row| {
            consume(row[field].take())
        })
    }
    pub fn points(&self, frame: i64) -> Result<Vec<((i64, String), Value)>> {
        let mut points = Vec::new();
        self.visit(
            &format!("SELECT payload FROM frames WHERE frame={frame}"),
            |mut row| {
                points = serde_json::from_value(row["points"].take())
                    .map_err(|e| Error::value(e.to_string()))?;
                Ok(())
            },
        )?;
        Ok(points)
    }
    pub fn events(&self, mut consume: impl FnMut(Value) -> Result<()>) -> Result<()> {
        self.visit("SELECT payload FROM frames ORDER BY frame", |mut row| {
            for event in row["events"].as_array_mut().unwrap().drain(..) {
                consume(event)?;
            }
            Ok(())
        })
    }
}
