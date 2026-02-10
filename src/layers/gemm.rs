use crate::{tensor::Tensor, utils::errors::Result};

use super::Layer;

/// General Matrix Multiplication (ONNX Gemm operator)
/// Y = alpha * A' @ B' + beta * C
/// where A' = transpose(A) if trans_a, B' = transpose(B) if trans_b
pub(crate) struct Gemm {
    b: Tensor,
    c: Option<Tensor>,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
}

impl Gemm {
    pub(crate) fn new(
        b: Tensor,
        c: Option<Tensor>,
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
    ) -> Self {
        Self {
            b,
            c,
            alpha,
            beta,
            trans_a,
            trans_b,
        }
    }
}

impl Layer for Gemm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let a = if self.trans_a {
            let mut a = input.clone();
            a.transpose();
            a
        } else {
            input.clone()
        };

        let b = if self.trans_b {
            let mut b = self.b.clone();
            b.transpose();
            b
        } else {
            self.b.clone()
        };

        let mut output = a.matmul(&b)?;

        if self.alpha != 1.0 {
            for i in 0..output.shape()[0] {
                for j in 0..output.shape()[1] {
                    let val = output.get(&[i, j])? * self.alpha;
                    output.set(&[i, j], val)?;
                }
            }
        }

        if let Some(c) = &self.c {
            let (m, n) = (output.shape()[0], output.shape()[1]);
            for i in 0..m {
                for j in 0..n {
                    // C is broadcastable to [M, N]
                    // Common cases: C is [N] (1D), [1, N], or [M, N]
                    let c_val = if c.ndim() == 1 {
                        c.get(&[j])?
                    } else if c.shape()[0] == 1 {
                        c.get(&[0, j])?
                    } else {
                        c.get(&[i, j])?
                    };
                    let val = output.get(&[i, j])? + self.beta * c_val;
                    output.set(&[i, j], val)?;
                }
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
    fn test_gemm_basic() {
        // A=[2,3], B=[3,2], no C
        // Y = 1.0 * A @ B
        let a = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::new([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let gemm = Gemm::new(b, None, 1.0, 1.0, false, false);
        let output = gemm.forward(&a).unwrap();

        assert_eq!(output.shape(), &[2, 2]);
        // [0,0]: 1*1+2*3+3*5 = 22
        // [0,1]: 1*2+2*4+3*6 = 28
        // [1,0]: 4*1+5*3+6*5 = 49
        // [1,1]: 4*2+5*4+6*6 = 64
        assert!((output.get(&[0, 0]).unwrap() - 22.0).abs() < 1e-6);
        assert!((output.get(&[0, 1]).unwrap() - 28.0).abs() < 1e-6);
        assert!((output.get(&[1, 0]).unwrap() - 49.0).abs() < 1e-6);
        assert!((output.get(&[1, 1]).unwrap() - 64.0).abs() < 1e-6);
    }

    #[test]
    fn test_gemm_with_bias_broadcast() {
        // A=[2,3], B=[3,2], C=[2] (broadcast)
        // Y = A @ B + C
        let a = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::new([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let c = Tensor::new([2], [10.0, 20.0]).unwrap();
        let gemm = Gemm::new(b, Some(c), 1.0, 1.0, false, false);
        let output = gemm.forward(&a).unwrap();

        assert!((output.get(&[0, 0]).unwrap() - 32.0).abs() < 1e-6);
        assert!((output.get(&[0, 1]).unwrap() - 48.0).abs() < 1e-6);
        assert!((output.get(&[1, 0]).unwrap() - 59.0).abs() < 1e-6);
        assert!((output.get(&[1, 1]).unwrap() - 84.0).abs() < 1e-6);
    }

    #[test]
    fn test_gemm_trans_b() {
        // A=[2,3], B=[2,3] with transB=1 -> B'=[3,2]
        // Y = A @ B.T
        let a = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let gemm = Gemm::new(b, None, 1.0, 1.0, false, true);
        let output = gemm.forward(&a).unwrap();

        assert_eq!(output.shape(), &[2, 2]);
        // A @ B.T: B.T = [[1,4],[2,5],[3,6]]
        // [0,0]: 1*1+2*2+3*3 = 14
        // [0,1]: 1*4+2*5+3*6 = 32
        // [1,0]: 4*1+5*2+6*3 = 32
        // [1,1]: 4*4+5*5+6*6 = 77
        assert!((output.get(&[0, 0]).unwrap() - 14.0).abs() < 1e-6);
        assert!((output.get(&[0, 1]).unwrap() - 32.0).abs() < 1e-6);
        assert!((output.get(&[1, 0]).unwrap() - 32.0).abs() < 1e-6);
        assert!((output.get(&[1, 1]).unwrap() - 77.0).abs() < 1e-6);
    }

    #[test]
    fn test_gemm_alpha_beta() {
        // Y = 2.0 * A @ B + 0.5 * C
        let a = Tensor::new([1, 2], [1.0, 2.0]).unwrap();
        let b = Tensor::new([2, 1], [3.0, 4.0]).unwrap();
        let c = Tensor::new([1], [10.0]).unwrap();
        let gemm = Gemm::new(b, Some(c), 2.0, 0.5, false, false);
        let output = gemm.forward(&a).unwrap();

        // A @ B = [[1*3+2*4]] = [[11]]
        // Y = 2.0 * 11 + 0.5 * 10 = 22 + 5 = 27
        assert!((output.get(&[0, 0]).unwrap() - 27.0).abs() < 1e-6);
    }
}
