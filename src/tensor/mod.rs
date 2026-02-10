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
            return Err(Error::ShapeMismatch {
                expected: expected_size,
                actual: data.len(),
            });
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

    pub fn dot(&self, other: &Tensor) -> Result<f32> {
        if self.ndim() != 1 {
            return Err(Error::ShapeMismatch {
                expected: 1,
                actual: self.ndim(),
            });
        }

        if other.ndim() != 1 {
            return Err(Error::ShapeMismatch {
                expected: 1,
                actual: other.ndim(),
            });
        }

        if self.size() != other.size() {
            return Err(Error::ShapeMismatch {
                expected: self.size(),
                actual: other.size(),
            });
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

    /// Matrix-vector multiplication along last axis
    /// [..., n] @ [n] -> [...]
    pub fn matvec(&self, vec: &Tensor) -> Result<Tensor> {
        if vec.ndim() != 1 {
            return Err(Error::ShapeMismatch {
                expected: 1,
                actual: vec.ndim(),
            });
        }

        let n = *self
            .shape()
            .last()
            .ok_or_else(|| Error::empty_vec("Empty tensor shape"))?;

        if vec.size() != n {
            return Err(Error::ShapeMismatch {
                expected: n,
                actual: vec.size(),
            });
        }

        let result_shape: Vec<usize> = self.shape()[..self.ndim() - 1].to_vec();
        let result_size: usize = result_shape.iter().product();

        if result_size == 0 {
            return Tensor::new(result_shape, Vec::<f32>::new());
        }

        let mut result_data = vec![0.0; result_size];

        for (idx, result_val) in result_data.iter_mut().enumerate() {
            let mut indices = vec![0; self.ndim()];
            let mut remaining = idx;
            for (i, &dim) in result_shape.iter().enumerate().rev() {
                indices[i] = remaining % dim;
                remaining /= dim;
            }

            let mut sum = 0.0;
            for j in 0..n {
                indices[self.ndim() - 1] = j;
                sum += self.get(&indices)? * vec.get(&[j])?;
            }
            *result_val = sum;
        }

        Tensor::new(result_shape, result_data)
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
        assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
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
        assert!(matches!(a.dot(&b), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn test_dot_product_not_1d() {
        let a = Tensor::new([2, 2], [1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::new([4], [1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(matches!(a.dot(&b), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn test_matvec_2d() {
        // [2, 3] @ [3] -> [2]
        let mat = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let vec = Tensor::new([3], [1.0, 2.0, 3.0]).unwrap();

        let result = mat.matvec(&vec).unwrap();

        assert_eq!(result.shape(), &[2]);
        // row 0: 1*1 + 2*2 + 3*3 = 14
        // row 1: 4*1 + 5*2 + 6*3 = 32
        assert!((result.get(&[0]).unwrap() - 14.0).abs() < 1e-6);
        assert!((result.get(&[1]).unwrap() - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_3d() {
        // [2, 2, 3] @ [3] -> [2, 2]
        let tensor = Tensor::new(
            [2, 2, 3],
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let vec = Tensor::new([3], [1.0, 1.0, 1.0]).unwrap();

        let result = tensor.matvec(&vec).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        // [0,0]: 1+2+3 = 6
        // [0,1]: 4+5+6 = 15
        // [1,0]: 7+8+9 = 24
        // [1,1]: 10+11+12 = 33
        assert!((result.get(&[0, 0]).unwrap() - 6.0).abs() < 1e-6);
        assert!((result.get(&[0, 1]).unwrap() - 15.0).abs() < 1e-6);
        assert!((result.get(&[1, 0]).unwrap() - 24.0).abs() < 1e-6);
        assert!((result.get(&[1, 1]).unwrap() - 33.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_shape_mismatch() {
        let mat = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let vec = Tensor::new([2], [1.0, 2.0]).unwrap();
        assert!(matches!(mat.matvec(&vec), Err(Error::ShapeMismatch { .. })));
    }
}
