use std::simd::prelude::*;

use crate::{layers::Layer, tensor::Tensor, utils::errors::Result};

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }
}

impl Layer for ReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut output = input.clone();
        let data = output.data_mut();
        let zero = f32x8::splat(0.0);

        let (chunks, remainder) = data.as_chunks_mut::<8>();
        for chunk in chunks {
            let v = f32x8::from_array(*chunk);
            *chunk = v.simd_max(zero).to_array();
        }
        for val in remainder {
            *val = val.max(0.0);
        }

        Ok(output)
    }

    fn backward(&self, _input: &Tensor) -> Result<Tensor> {
        unimplemented!("backward not implemented for ReLU layer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_positive() {
        let t = Tensor::new([4], [1.0, 2.0, 3.0, 4.0]).unwrap();
        let relu = ReLU::new();
        let result = relu.forward(&t).unwrap();

        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_relu_negative() {
        let t = Tensor::new([4], [-1.0, -2.0, -3.0, -4.0]).unwrap();
        let relu = ReLU::new();
        let result = relu.forward(&t).unwrap();

        assert_eq!(result.data(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_relu_mixed() {
        let t = Tensor::new([6], [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]).unwrap();
        let relu = ReLU::new();
        let result = relu.forward(&t).unwrap();

        assert_eq!(result.data(), &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_relu_larger_than_simd_lane() {
        // 10 elements: 8 processed by SIMD, 2 by remainder
        let t = Tensor::new(
            [10],
            [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, -1.0, 3.0],
        )
        .unwrap();
        let relu = ReLU::new();
        let result = relu.forward(&t).unwrap();

        assert_eq!(
            result.data(),
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0]
        );
    }

    #[test]
    fn test_relu_2d() {
        let t = Tensor::new([2, 3], [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]).unwrap();
        let relu = ReLU::new();
        let result = relu.forward(&t).unwrap();

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.data(), &[0.0, 2.0, 0.0, 4.0, 0.0, 6.0]);
    }
}
