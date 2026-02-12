use super::shape::Shape;

use crate::utils::errors::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    strides: Vec<usize>,
    offset: usize,
}

impl Layout {
    pub fn new(shape: Shape) -> Self {
        let strides = Self::compute_strides(&shape);
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    pub fn with_offset(shape: Shape, offset: usize) -> Self {
        let strides = Self::compute_strides(&shape);
        Self {
            shape,
            strides,
            offset,
        }
    }

    fn compute_strides(shape: &Shape) -> Vec<usize> {
        let ndim = shape.ndim();
        if ndim == 0 {
            return vec![];
        }

        let mut strides = vec![1; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    pub fn size(&self) -> usize {
        self.shape.size()
    }

    pub fn flat_index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.ndim() {
            return None;
        }

        for (idx, dim) in indices.iter().zip(self.shape.dims()) {
            if *idx >= *dim {
                return None;
            }
        }

        let index = indices
            .iter()
            .zip(self.strides.iter())
            .map(|(&i, &s)| i * s)
            .sum::<usize>();

        Some(self.offset + index)
    }

    pub fn flatten(&self, start_dim: usize, end_dim: usize) -> Result<Self> {
        let dims = self.shape.dims();
        let flat_size: usize = dims[start_dim..=end_dim].iter().product();

        let mut new_dims = Vec::with_capacity(dims.len() - (end_dim - start_dim));
        new_dims.extend_from_slice(&dims[..start_dim]);
        new_dims.push(flat_size);
        new_dims.extend_from_slice(&dims[end_dim + 1..]);

        Ok(Self::new(Shape::new(new_dims)))
    }

    pub fn transpose(&mut self) {
        let dims: Vec<usize> = self.shape.dims().iter().rev().copied().collect();
        self.strides.reverse();
        self.shape = Shape::new(dims);
    }

    pub fn transpose_axes(&self, axes: &[usize]) -> Option<Self> {
        if axes.len() != self.ndim() {
            return None;
        }

        let mut seen = vec![false; self.ndim()];
        for &axis in axes {
            if axis >= self.ndim() || seen[axis] {
                return None;
            }
            seen[axis] = true;
        }

        let dims: Vec<usize> = axes.iter().map(|&i| self.shape[i]).collect();
        let strides: Vec<usize> = axes.iter().map(|&i| self.strides[i]).collect();

        Some(Self {
            shape: Shape::new(dims),
            strides,
            offset: self.offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_strides() {
        let layout = Layout::new(Shape::from([2, 3, 4]));
        assert_eq!(layout.strides(), &[12, 4, 1]);
    }

    #[test]
    fn test_flat_index() {
        let layout = Layout::new(Shape::from([2, 3]));
        assert_eq!(layout.flat_index(&[0, 0]), Some(0));
        assert_eq!(layout.flat_index(&[0, 2]), Some(2));
        assert_eq!(layout.flat_index(&[1, 0]), Some(3));
        assert_eq!(layout.flat_index(&[1, 2]), Some(5));
    }

    #[test]
    fn test_flat_index_with_offset() {
        let layout = Layout::with_offset(Shape::from([2, 3]), 10);
        assert_eq!(layout.flat_index(&[0, 0]), Some(10));
        assert_eq!(layout.flat_index(&[1, 2]), Some(15));
    }

    #[test]
    fn test_flat_index_out_of_bounds() {
        let layout = Layout::new(Shape::from([2, 3]));
        assert_eq!(layout.flat_index(&[2, 0]), None);
        assert_eq!(layout.flat_index(&[0, 3]), None);
    }

    #[test]
    fn test_flat_index_wrong_dims() {
        let layout = Layout::new(Shape::from([2, 3]));
        assert_eq!(layout.flat_index(&[0]), None);
        assert_eq!(layout.flat_index(&[0, 0, 0]), None);
    }

    #[test]
    fn test_transpose_2d() {
        let mut layout = Layout::new(Shape::from([2, 3]));
        layout.transpose();

        assert_eq!(layout.shape().dims(), &[3, 2]);
        assert_eq!(layout.strides(), &[1, 3]);
    }

    #[test]
    fn test_transpose_3d() {
        let mut layout = Layout::new(Shape::from([2, 3, 4]));
        layout.transpose();

        assert_eq!(layout.shape().dims(), &[4, 3, 2]);
        assert_eq!(layout.strides(), &[1, 4, 12]);
    }

    #[test]
    fn test_transpose_axes() {
        let layout = Layout::new(Shape::from([2, 3, 4]));
        let transposed = layout.transpose_axes(&[1, 2, 0]).unwrap();

        assert_eq!(transposed.shape().dims(), &[3, 4, 2]);
        assert_eq!(transposed.strides(), &[4, 1, 12]);
    }

    #[test]
    fn test_transpose_axes_invalid() {
        let layout = Layout::new(Shape::from([2, 3]));
        assert!(layout.transpose_axes(&[0]).is_none());
        assert!(layout.transpose_axes(&[0, 2]).is_none());
        assert!(layout.transpose_axes(&[0, 0]).is_none());
    }
}
