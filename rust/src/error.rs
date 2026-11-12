use serde::Serialize;
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Error {
    pub kind: &'static str,
    pub code: Option<&'static str>,
    pub message: String,
}

impl Error {
    pub fn type_error(message: impl Into<String>) -> Self {
        Self {
            kind: "TypeError",
            code: None,
            message: message.into(),
        }
    }
    pub fn value(message: impl Into<String>) -> Self {
        Self {
            kind: "ValueError",
            code: None,
            message: message.into(),
        }
    }
    pub fn geometry(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind: "GeometryError",
            code: Some(code),
            message: message.into(),
        }
    }
}

impl From<crate::ValidationError> for Error {
    fn from(e: crate::ValidationError) -> Self {
        match e.code() {
            Some(code) => Self::geometry(code, e.0),
            None => Self::value(e.0),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
