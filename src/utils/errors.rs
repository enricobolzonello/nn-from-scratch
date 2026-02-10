use derive_more::{Display, From};

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, Display, From)]
pub enum Error {
    #[display("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch {
        expected: usize,
        actual: usize,
    },

    #[display("Index out of bounds: {indices:?} for shape {shape:?}")]
    IndexOutOfBounds {
        indices: Vec<usize>,
        shape: Vec<usize>,
    },

    #[display("Invalid axes: {axes:?} for tensor with {ndim} dimensions")]
    InvalidAxes {
        axes: Vec<usize>,
        ndim: usize,
    },

    #[display("{_0}")]
    EmptyVec(String),
}

impl Error {
    pub fn empty_vec(msg: impl Into<String>) -> Self {
        Self::EmptyVec(msg.into())
    }
}

impl std::error::Error for Error {}
