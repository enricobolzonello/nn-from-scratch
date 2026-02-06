use derive_more::{Display, From};

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, Display, From)]
#[display("{self:?}")]
pub enum Error {
    ShapeMismatch {
        expected: usize,
        actual: usize,
    },

    IndexOutOfBounds {
        indices: Vec<usize>,
        shape: Vec<usize>,
    },

    InvalidAxes {
        axes: Vec<usize>,
        ndim: usize,
    },
}

impl std::error::Error for Error {}
