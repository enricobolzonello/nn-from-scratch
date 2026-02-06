use crate::{tensor::Tensor, utils::errors::Result};

use super::Neuron;

pub struct Sigmoid(Neuron);

impl Sigmoid {
    pub fn new(weights: Tensor, bias: f32) -> Self {
        Self(Neuron::new(weights, bias))
    }

    pub fn output(&self, inputs: &Tensor) -> Result<f32> {
        let dot = self.0.weights().dot(inputs)?;
        let z = -dot - self.0.bias();

        Ok(1.0 / (1.0 + z.exp()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_sigmoid_output_range() {
        let weights = Tensor::new([2], [1.0, 1.0]).unwrap();
        let s = Sigmoid::new(weights, 0.0);

        let inputs = Tensor::new([2], [0.5, -0.3]).unwrap();
        let output = s.output(&inputs).unwrap();

        assert!(output > 0.0 && output < 1.0);
    }

    #[test]
    fn test_sigmoid_zero_input_zero_bias() {
        let weights = Tensor::new([2], [1.0, 1.0]).unwrap();
        let s = Sigmoid::new(weights, 0.0);

        let inputs = Tensor::new([2], [0.0, 0.0]).unwrap();
        let output = s.output(&inputs).unwrap();

        // sigmoid(0) = 0.5
        assert!((output - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_sigmoid_large_positive() {
        let weights = Tensor::new([2], [1.0, 1.0]).unwrap();
        let s = Sigmoid::new(weights, 0.0);

        let inputs = Tensor::new([2], [10.0, 10.0]).unwrap();
        let output = s.output(&inputs).unwrap();

        // sigmoid(20) ≈ 1.0
        assert!(output > 0.99);
    }

    #[test]
    fn test_sigmoid_large_negative() {
        let weights = Tensor::new([2], [1.0, 1.0]).unwrap();
        let s = Sigmoid::new(weights, 0.0);

        let inputs = Tensor::new([2], [-10.0, -10.0]).unwrap();
        let output = s.output(&inputs).unwrap();

        // sigmoid(-20) ≈ 0.0
        assert!(output < 0.01);
    }

    #[test]
    fn test_sigmoid_with_bias() {
        let weights = Tensor::new([1], [0.0]).unwrap();
        let s = Sigmoid::new(weights, 2.0);

        let inputs = Tensor::new([1], [0.0]).unwrap();
        let output = s.output(&inputs).unwrap();

        // sigmoid(0 + 2) = sigmoid(2) ≈ 0.8808
        let expected = 1.0 / (1.0 + (-2.0_f32).exp());
        assert!((output - expected).abs() < EPSILON);
    }

    #[test]
    fn test_sigmoid_shape_mismatch() {
        let weights = Tensor::new([2], [1.0, 1.0]).unwrap();
        let s = Sigmoid::new(weights, 0.0);

        let bad_input = Tensor::new([3], [1.0, 2.0, 3.0]).unwrap();
        assert!(s.output(&bad_input).is_err());
    }
}
