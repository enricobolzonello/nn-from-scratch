pub mod perceptron;

pub use perceptron::Perceptron;

use crate::tensor::Tensor;

pub struct Neuron {
    weights: Tensor,
    bias: f32,
}

impl Neuron {
    pub fn new(weights: Tensor, bias: f32) -> Self {
        Self { weights, bias }
    }

    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    pub fn bias(&self) -> f32 {
        self.bias
    }
}
