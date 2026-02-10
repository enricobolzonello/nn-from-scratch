use crate::{tensor::Tensor, utils::errors::Result};

pub(crate) mod gemm;
pub mod linear;

pub use linear::Linear;

pub trait Layer {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn backward(&self, input: &Tensor) -> Result<Tensor>;
}
