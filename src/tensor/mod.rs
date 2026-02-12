mod layout;
mod shape;

pub use layout::Layout;
pub use shape::Shape;

use std::sync::Arc;

use crate::utils::errors::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    data: Arc<Vec<f32>>,
    layout: Layout,
}

impl Tensor {
    pub fn new<S, D>(shape: S, data: D) -> Result<Self>
    where
        S: Into<Shape>,
        D: Into<Vec<f32>>,
    {
        let shape = shape.into();
        let data = data.into();
        let expected_size = shape.size();

        if data.len() != expected_size {
            return Err(Error::ShapeError(format!(
                "data length {} does not match shape size {}",
                data.len(),
                expected_size
            )));
        }

        let layout = Layout::new(shape);
        Ok(Self {
            data: Arc::new(data),
            layout,
        })
    }

    pub fn zeros<S>(shape: S) -> Self
    where
        S: Into<Shape>,
    {
        let shape = shape.into();
        let size = shape.size();
        let layout = Layout::new(shape);
        Self {
            data: Arc::new(vec![0.0; size]),
            layout,
        }
    }

    pub fn ones<S>(shape: S) -> Self
    where
        S: Into<Shape>,
    {
        let shape = shape.into();
        let size = shape.size();
        let layout = Layout::new(shape);
        Self {
            data: Arc::new(vec![1.0; size]),
            layout,
        }
    }

    pub fn fill<S>(shape: S, value: f32) -> Self
    where
        S: Into<Shape>,
    {
        let shape = shape.into();
        let size = shape.size();
        let layout = Layout::new(shape);
        Self {
            data: Arc::new(vec![value; size]),
            layout,
        }
    }

    pub fn get(&self, indices: &[usize]) -> Result<f32> {
        self.layout
            .flat_index(indices)
            .map(|i| self.data[i])
            .ok_or_else(|| Error::IndexOutOfBounds {
                indices: indices.to_vec(),
                shape: self.shape().to_vec(),
            })
    }

    pub fn set(&mut self, indices: &[usize], value: f32) -> Result<()> {
        let idx = self
            .layout
            .flat_index(indices)
            .ok_or_else(|| Error::IndexOutOfBounds {
                indices: indices.to_vec(),
                shape: self.shape().to_vec(),
            })?;
        Arc::make_mut(&mut self.data)[idx] = value;
        Ok(())
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape().dims()
    }

    pub fn strides(&self) -> &[usize] {
        self.layout.strides()
    }

    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    pub fn size(&self) -> usize {
        self.layout.size()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        Arc::make_mut(&mut self.data).as_mut_slice()
    }

    pub fn flatten(&self, start_dim: usize, end_dim: usize) -> Result<Tensor> {
        Ok(Tensor {
            layout: self.layout().flatten(start_dim, end_dim)?,
            data: self.data.clone(),
        })
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Transpose the tensor in place
    pub fn transpose(&mut self) {
        self.layout.transpose();
    }

    /// Transpose with an arbitrary permutation of dimensions
    pub fn transpose_axes(&mut self, axes: &[usize]) -> Result<()> {
        self.layout = self
            .layout
            .transpose_axes(axes)
            .ok_or_else(|| Error::InvalidAxes {
                axes: axes.to_vec(),
                ndim: self.ndim(),
            })?;
        Ok(())
    }

    // TODO: optimize with SIMD
    pub fn dot(&self, other: &Tensor) -> Result<f32> {
        if self.ndim() != 1 {
            return Err(Error::ShapeError(format!(
                "dot requires 1D tensors, got {}D",
                self.ndim()
            )));
        }

        if other.ndim() != 1 {
            return Err(Error::ShapeError(format!(
                "dot requires 1D tensors, got {}D",
                other.ndim()
            )));
        }

        if self.size() != other.size() {
            return Err(Error::ShapeError(format!(
                "dot requires same length, got {} and {}",
                self.size(),
                other.size()
            )));
        }

        let result = (0..self.size())
            .map(|i| {
                let a = self.get(&[i]).unwrap();
                let b = other.get(&[i]).unwrap();
                a * b
            })
            .sum();

        Ok(result)
    }

    /// Matrix multiplication with 1D promotion:
    /// - [M, K] @ [K, N] -> [M, N]
    /// - [M, K] @ [K]    -> [M]     (second operand promoted to [K, 1], result squeezed)
    /// - [K]    @ [K, N]  -> [N]     (first operand promoted to [1, K], result squeezed)
    /// TODO: optimize with SIMD
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let squeeze_first = self.ndim() == 1;
        let squeeze_last = other.ndim() == 1;

        let a_shape = if squeeze_first {
            vec![1, self.shape()[0]]
        } else if self.ndim() == 2 {
            self.shape().to_vec()
        } else {
            return Err(Error::ShapeError(format!(
                "matmul requires 1D or 2D tensors, got {}D",
                self.ndim()
            )));
        };

        let b_shape = if squeeze_last {
            vec![other.shape()[0], 1]
        } else if other.ndim() == 2 {
            other.shape().to_vec()
        } else {
            return Err(Error::ShapeError(format!(
                "matmul requires 1D or 2D tensors, got {}D",
                other.ndim()
            )));
        };

        let m = a_shape[0];
        let k = a_shape[1];
        let k2 = b_shape[0];
        let n = b_shape[1];

        if k != k2 {
            return Err(Error::ShapeError(format!(
                "matmul inner dimensions mismatch: {} vs {}",
                k, k2
            )));
        }

        let mut result_data = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    let a_val = if squeeze_first {
                        self.get(&[p])?
                    } else {
                        self.get(&[i, p])?
                    };
                    let b_val = if squeeze_last {
                        other.get(&[p])?
                    } else {
                        other.get(&[p, j])?
                    };
                    sum += a_val * b_val;
                }
                result_data[i * n + j] = sum;
            }
        }

        if squeeze_first && squeeze_last {
            Tensor::new([1], vec![result_data[0]])
        } else if squeeze_first {
            Tensor::new(vec![n], result_data)
        } else if squeeze_last {
            Tensor::new(vec![m], result_data)
        } else {
            Tensor::new([m, n], result_data)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tensor() {
        let t = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.strides(), &[3, 1]);
        assert_eq!(t.size(), 6);
    }

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros([2, 2]);
        assert_eq!(t.data(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ones() {
        let t = Tensor::ones([3]);
        assert_eq!(t.data(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_get_set() {
        let mut t = Tensor::zeros([2, 3]);
        t.set(&[1, 2], 5.0).unwrap();
        assert_eq!(t.get(&[1, 2]).unwrap(), 5.0);
        assert_eq!(t.get(&[0, 0]).unwrap(), 0.0);
    }

    #[test]
    fn test_3d_tensor() {
        let t = Tensor::new([2, 2, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        assert_eq!(t.strides(), &[4, 2, 1]);
        assert_eq!(t.get(&[0, 0, 0]).unwrap(), 1.0);
        assert_eq!(t.get(&[0, 0, 1]).unwrap(), 2.0);
        assert_eq!(t.get(&[1, 1, 1]).unwrap(), 8.0);
    }

    #[test]
    fn test_transpose_2d() {
        let mut t = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        t.transpose();

        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(t.get(&[0, 1]).unwrap(), 4.0);
        assert_eq!(t.get(&[2, 0]).unwrap(), 3.0);
        assert_eq!(t.get(&[2, 1]).unwrap(), 6.0);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let result = Tensor::new([2, 2], [1.0, 2.0, 3.0]);
        assert!(matches!(result, Err(Error::ShapeError(_))));
    }

    #[test]
    fn test_index_error() {
        let t = Tensor::zeros([2, 3]);
        let result = t.get(&[2, 0]);
        assert!(matches!(result, Err(Error::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_dot_product() {
        let a = Tensor::new([3], [1.0, 2.0, 3.0]).unwrap();
        let b = Tensor::new([3], [4.0, 5.0, 6.0]).unwrap();
        let result = a.dot(&b).unwrap();
        assert!((result - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product_length_mismatch() {
        let a = Tensor::new([2], [1.0, 2.0]).unwrap();
        let b = Tensor::new([3], [1.0, 2.0, 3.0]).unwrap();
        assert!(matches!(a.dot(&b), Err(Error::ShapeError(_))));
    }

    #[test]
    fn test_dot_product_not_1d() {
        let a = Tensor::new([2, 2], [1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::new([4], [1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(matches!(a.dot(&b), Err(Error::ShapeError(_))));
    }

    #[test]
    fn test_matmul_mat_vec() {
        // [2, 3] @ [3] -> [2] (second operand promoted to [3,1], result squeezed)
        let mat = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let vec = Tensor::new([3], [1.0, 2.0, 3.0]).unwrap();

        let result = mat.matmul(&vec).unwrap();

        assert_eq!(result.shape(), &[2]);
        // row 0: 1*1 + 2*2 + 3*3 = 14
        // row 1: 4*1 + 5*2 + 6*3 = 32
        assert!((result.get(&[0]).unwrap() - 14.0).abs() < 1e-6);
        assert!((result.get(&[1]).unwrap() - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_vec_mat() {
        // [3] @ [3, 2] -> [2] (first operand promoted to [1,3], result squeezed)
        let vec = Tensor::new([3], [1.0, 2.0, 3.0]).unwrap();
        let mat = Tensor::new([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = vec.matmul(&mat).unwrap();

        assert_eq!(result.shape(), &[2]);
        // 1*1+2*3+3*5 = 22
        // 1*2+2*4+3*6 = 28
        assert!((result.get(&[0]).unwrap() - 22.0).abs() < 1e-6);
        assert!((result.get(&[1]).unwrap() - 28.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul() {
        // [2, 3] @ [3, 2] -> [2, 2]
        let a = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::new([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let result = a.matmul(&b).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        // [0,0]: 1*7+2*9+3*11 = 58
        // [0,1]: 1*8+2*10+3*12 = 64
        // [1,0]: 4*7+5*9+6*11 = 139
        // [1,1]: 4*8+5*10+6*12 = 154
        assert!((result.get(&[0, 0]).unwrap() - 58.0).abs() < 1e-6);
        assert!((result.get(&[0, 1]).unwrap() - 64.0).abs() < 1e-6);
        assert!((result.get(&[1, 0]).unwrap() - 139.0).abs() < 1e-6);
        assert!((result.get(&[1, 1]).unwrap() - 154.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_shape_mismatch() {
        let a = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::new([2, 2], [1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(matches!(a.matmul(&b), Err(Error::ShapeError(_))));
    }

    #[test]
    fn test_matmul_vec_shape_mismatch() {
        let mat = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let vec = Tensor::new([2], [1.0, 2.0]).unwrap();
        assert!(matches!(mat.matmul(&vec), Err(Error::ShapeError(_))));
    }
}
