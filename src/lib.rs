pub mod layers;
pub mod neurons;
pub mod onnx;
pub mod tensor;
mod utils;

use std::{borrow::Cow, fs};

use layers::Layer;
use tensor::Tensor;
use utils::errors::{Error, Result};

use crate::onnx::loader::load_model;

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn builder() -> NetworkBuilder {
        NetworkBuilder::new()
    }

    pub fn from_onnx(filepath: &str) -> Result<Self> {
        if filepath.is_empty() || !fs::exists(filepath)? {
            return Err(Error::FileDoesNotExist {
                filepath: filepath.into(),
            });
        }

        let layers = load_model(filepath)?;

        Ok(Self { layers })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut current: Cow<Tensor> = Cow::Borrowed(input);

        for layer in &self.layers {
            current = Cow::Owned(layer.forward(&current)?);
        }

        Ok(current.into_owned())
    }

    // pub fn train(
    //     &mut self,
    //     data: &Tensor,
    //     labels: &Tensor,
    //     optimizer: Optimizer,
    //     epochs: usize,
    // ) -> Result<()> {
    //     todo!()
    // }
}

pub struct NetworkBuilder {
    layers: Vec<Box<dyn Layer>>,
}

impl NetworkBuilder {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn layer(mut self, layer: Box<dyn Layer>) -> Self {
        self.layers.push(layer);
        self
    }

    pub fn build(self) -> Result<Network> {
        if self.layers.is_empty() {
            return Err(Error::empty_vec("Network cannot have zero layers"));
        }
        Ok(Network {
            layers: self.layers,
        })
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}
