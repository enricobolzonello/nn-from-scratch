use std::ops::Index;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new<T: Into<Vec<usize>>>(dims: T) -> Self {
        Self(dims.into())
    }

    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    pub fn size(&self) -> usize {
        self.0.iter().product()
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(arr: [usize; N]) -> Self {
        Self(arr.to_vec())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(vec: Vec<usize>) -> Self {
        Self(vec)
    }
}

impl From<&[usize]> for Shape {
    fn from(slice: &[usize]) -> Self {
        Self(slice.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_from_array() {
        let shape = Shape::from([2, 3, 4]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.size(), 24);
        assert_eq!(shape[0], 2);
        assert_eq!(shape[1], 3);
        assert_eq!(shape[2], 4);
    }

    #[test]
    fn test_shape_from_vec() {
        let shape = Shape::from(vec![5, 10]);
        assert_eq!(shape.ndim(), 2);
        assert_eq!(shape.size(), 50);
    }

    #[test]
    fn test_shape_dims() {
        let shape = Shape::new([2, 3]);
        assert_eq!(shape.dims(), &[2, 3]);
    }
}
