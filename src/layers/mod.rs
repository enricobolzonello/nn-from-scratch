use crate::{tensor::Tensor, utils::errors::Result};

pub mod dense;

pub use dense::Dense;

pub trait Layer {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn backward(&self, input: &Tensor) -> Result<Tensor>;
}
