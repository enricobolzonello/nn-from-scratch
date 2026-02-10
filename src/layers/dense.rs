use crate::{tensor::Tensor, utils::errors::Result};

use super::Layer;

pub struct Dense {
    weights: Tensor,
    bias: Option<Tensor>,
}

impl Dense {
    pub fn new(weights: Tensor, bias: Option<Tensor>) -> Self {
        Self { weights, bias }
    }
}

impl Layer for Dense {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut weights_t = self.weights.clone();
        weights_t.transpose();
        let mut output = weights_t.matvec(input)?;

        if let Some(bias) = &self.bias {
            for i in 0..output.size() {
                let val = output.get(&[i])? + bias.get(&[i])?;
                output.set(&[i], val)?;
            }
        }

        Ok(output)
    }

    fn backward(&self, _grad: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_forward() {
        // weights: [in=2, out=3]
        let weights = Tensor::new([2, 3], [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]).unwrap();

        let bias = Tensor::new([3], [0.1, 0.2, 0.3]).unwrap();
        let dense = Dense::new(weights, Some(bias));

        let input = Tensor::new([2], [1.0, 2.0]).unwrap();
        let output = dense.forward(&input).unwrap();

        // weights.T is [3, 2], matvec with [2] -> [3]
        // output[0] = 1*1 + 4*2 + 0.1 = 9.1
        // output[1] = 2*1 + 5*2 + 0.2 = 12.2
        // output[2] = 3*1 + 6*2 + 0.3 = 15.3
        assert_eq!(output.shape(), &[3]);
        assert!((output.get(&[0]).unwrap() - 9.1).abs() < 1e-6);
        assert!((output.get(&[1]).unwrap() - 12.2).abs() < 1e-6);
        assert!((output.get(&[2]).unwrap() - 15.3).abs() < 1e-6);
    }

    #[test]
    fn test_dense_forward_no_bias() {
        // Identity-like weights
        let weights = Tensor::new([2, 2], [1.0, 0.0, 0.0, 1.0]).unwrap();
        let dense = Dense::new(weights, None);

        let input = Tensor::new([2], [3.0, 4.0]).unwrap();
        let output = dense.forward(&input).unwrap();

        assert!((output.get(&[0]).unwrap() - 3.0).abs() < 1e-6);
        assert!((output.get(&[1]).unwrap() - 4.0).abs() < 1e-6);
    }
}
