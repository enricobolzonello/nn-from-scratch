use crate::{tensor::Tensor, utils::errors::Result};

use super::Neuron;

pub struct Perceptron(Neuron);

impl Perceptron {
    pub fn new(weights: Tensor, bias: f32) -> Self {
        Self(Neuron::new(weights, bias))
    }

    pub fn output(&self, inputs: &Tensor) -> Result<bool> {
        let dot = self.0.weights().dot(inputs)?;
        Ok(dot + self.0.bias() > 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptron_and_gate() {
        // AND gate: weights [1, 1], bias -1.5
        let weights = Tensor::new([2], [1.0, 1.0]).unwrap();
        let p = Perceptron::new(weights, -1.5);

        let input_00 = Tensor::new([2], [0.0, 0.0]).unwrap();
        let input_01 = Tensor::new([2], [0.0, 1.0]).unwrap();
        let input_10 = Tensor::new([2], [1.0, 0.0]).unwrap();
        let input_11 = Tensor::new([2], [1.0, 1.0]).unwrap();

        assert_eq!(p.output(&input_00).unwrap(), false);
        assert_eq!(p.output(&input_01).unwrap(), false);
        assert_eq!(p.output(&input_10).unwrap(), false);
        assert_eq!(p.output(&input_11).unwrap(), true);
    }

    #[test]
    fn test_perceptron_or_gate() {
        // OR gate: weights [1, 1], bias -0.5
        let weights = Tensor::new([2], [1.0, 1.0]).unwrap();
        let p = Perceptron::new(weights, -0.5);

        let input_00 = Tensor::new([2], [0.0, 0.0]).unwrap();
        let input_01 = Tensor::new([2], [0.0, 1.0]).unwrap();
        let input_10 = Tensor::new([2], [1.0, 0.0]).unwrap();
        let input_11 = Tensor::new([2], [1.0, 1.0]).unwrap();

        assert_eq!(p.output(&input_00).unwrap(), false);
        assert_eq!(p.output(&input_01).unwrap(), true);
        assert_eq!(p.output(&input_10).unwrap(), true);
        assert_eq!(p.output(&input_11).unwrap(), true);
    }

    #[test]
    fn test_perceptron_shape_mismatch() {
        let weights = Tensor::new([2], [1.0, 1.0]).unwrap();
        let p = Perceptron::new(weights, 0.0);

        let bad_input = Tensor::new([3], [1.0, 2.0, 3.0]).unwrap();
        assert!(p.output(&bad_input).is_err());
    }
}
