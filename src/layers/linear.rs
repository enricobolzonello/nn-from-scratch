use crate::{tensor::Tensor, utils::errors::Result};

use super::Layer;
use super::gemm::Gemm;

pub struct Linear {
    gemm: Gemm,
}

impl Linear {
    pub fn new(weights: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            gemm: Gemm::new(weights, bias, 1.0, 1.0, false, false),
        }
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.gemm.forward(input)
    }

    fn backward(&self, grad: &Tensor) -> Result<Tensor> {
        self.gemm.backward(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        // weights: [in=2, out=3]
        let weights = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let bias = Tensor::new([3], [0.1, 0.2, 0.3]).unwrap();
        let linear = Linear::new(weights, Some(bias));

        // Input must be 2D for matmul: [1, 2]
        let input = Tensor::new([1, 2], [1.0, 2.0]).unwrap();
        let output = linear.forward(&input).unwrap();

        // [1, 2] @ [2, 3] + [3] = [1, 3]
        // output[0,0] = 1*1 + 2*4 + 0.1 = 9.1
        // output[0,1] = 1*2 + 2*5 + 0.2 = 12.2
        // output[0,2] = 1*3 + 2*6 + 0.3 = 15.3
        assert_eq!(output.shape(), &[1, 3]);
        assert!((output.get(&[0, 0]).unwrap() - 9.1).abs() < 1e-6);
        assert!((output.get(&[0, 1]).unwrap() - 12.2).abs() < 1e-6);
        assert!((output.get(&[0, 2]).unwrap() - 15.3).abs() < 1e-6);
    }

    #[test]
    fn test_linear_forward_no_bias() {
        // Identity-like weights
        let weights = Tensor::new([2, 2], [1.0, 0.0, 0.0, 1.0]).unwrap();
        let linear = Linear::new(weights, None);

        let input = Tensor::new([1, 2], [3.0, 4.0]).unwrap();
        let output = linear.forward(&input).unwrap();

        assert!((output.get(&[0, 0]).unwrap() - 3.0).abs() < 1e-6);
        assert!((output.get(&[0, 1]).unwrap() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_batched() {
        // weights: [2, 3], batch of 2 inputs: [2, 2]
        let weights = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let linear = Linear::new(weights, None);

        let input = Tensor::new([2, 2], [1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = linear.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 3]);
        // row 0: [1,2] @ [[1,2,3],[4,5,6]] = [9, 12, 15]
        // row 1: [3,4] @ [[1,2,3],[4,5,6]] = [19, 26, 33]
        assert!((output.get(&[0, 0]).unwrap() - 9.0).abs() < 1e-6);
        assert!((output.get(&[0, 1]).unwrap() - 12.0).abs() < 1e-6);
        assert!((output.get(&[0, 2]).unwrap() - 15.0).abs() < 1e-6);
        assert!((output.get(&[1, 0]).unwrap() - 19.0).abs() < 1e-6);
        assert!((output.get(&[1, 1]).unwrap() - 26.0).abs() < 1e-6);
        assert!((output.get(&[1, 2]).unwrap() - 33.0).abs() < 1e-6);
    }
}
