use derive_more::{Display, From};

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, Display, From)]
pub enum Error {
    #[display("Shape error: {_0}")]
    ShapeError(String),

    #[display("Index out of bounds: {indices:?} for shape {shape:?}")]
    IndexOutOfBounds {
        indices: Vec<usize>,
        shape: Vec<usize>,
    },

    #[display("Invalid axes: {axes:?} for tensor with {ndim} dimensions")]
    InvalidAxes { axes: Vec<usize>, ndim: usize },

    #[display("{_0}")]
    EmptyVec(String),

    #[display("File does not exist: {filepath:?}")]
    FileDoesNotExist { filepath: String },

    #[display("Unsupported ONNX operator: {op_type}")]
    UnsupportedOp { op_type: String },

    #[display("ONNX error: {_0}")]
    OnnxParse(String),

    #[from]
    Io(std::io::Error),

    #[from]
    Decode(prost::DecodeError),
}

impl Error {
    pub fn empty_vec(msg: impl Into<String>) -> Self {
        Self::EmptyVec(msg.into())
    }
}

impl std::error::Error for Error {}
