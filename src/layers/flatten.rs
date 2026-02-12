use crate::{
    layers::Layer,
    tensor::Tensor,
    utils::errors::{Error, Result},
};

pub struct Flatten {
    start_dim: usize,
    end_dim: usize,
}

impl Flatten {
    pub(crate) fn new(start_dim: usize, end_dim: usize) -> Self {
        Self { start_dim, end_dim }
    }
}

impl Layer for Flatten {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.start_dim >= input.ndim() || self.end_dim >= input.ndim() {
            return Err(Error::ShapeError(format!(
                "flatten dims [{}, {}] out of range for {}D tensor",
                self.start_dim,
                self.end_dim,
                input.ndim()
            )));
        }

        if self.start_dim > self.end_dim {
            return Err(Error::ShapeError(format!(
                "start_dim needs to be < than end_dim, got start_dim {} and end_dim {}",
                self.start_dim, self.end_dim
            )));
        }

        input.flatten(self.start_dim, self.end_dim)
    }

    fn backward(&self, _input: &Tensor) -> Result<Tensor> {
        unimplemented!("Backward does not exist for Flatten layer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten_all() {
        // [2, 2, 2] -> [8]
        let t = Tensor::new([2, 2, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let flatten = Flatten::new(0, 2);
        let result = flatten.forward(&t).unwrap();

        assert_eq!(result.shape(), &[8]);
        assert_eq!(result.data(), t.data());
    }

    #[test]
    fn test_flatten_keep_batch() {
        // [2, 2, 2] -> [2, 4] (flatten dims 1..=2)
        let t = Tensor::new([2, 2, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let flatten = Flatten::new(1, 2);
        let result = flatten.forward(&t).unwrap();

        assert_eq!(result.shape(), &[2, 4]);
        assert!((result.get(&[0, 0]).unwrap() - 1.0).abs() < 1e-6);
        assert!((result.get(&[0, 3]).unwrap() - 4.0).abs() < 1e-6);
        assert!((result.get(&[1, 0]).unwrap() - 5.0).abs() < 1e-6);
        assert!((result.get(&[1, 3]).unwrap() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_flatten_middle_dims() {
        // [2, 3, 4, 5] -> [2, 12, 5] (flatten dims 1..=2)
        let t = Tensor::fill([2, 3, 4, 5], 1.0);
        let flatten = Flatten::new(1, 2);
        let result = flatten.forward(&t).unwrap();

        assert_eq!(result.shape(), &[2, 12, 5]);
    }

    #[test]
    fn test_flatten_single_dim() {
        // [2, 3] with start_dim == end_dim -> no change
        let t = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let flatten = Flatten::new(0, 0);
        let result = flatten.forward(&t).unwrap();

        assert_eq!(result.shape(), &[2, 3]);
    }

    #[test]
    fn test_flatten_out_of_range() {
        let t = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let flatten = Flatten::new(0, 5);
        assert!(matches!(flatten.forward(&t), Err(Error::ShapeError(_))));
    }

    #[test]
    fn test_flatten_start_greater_than_end() {
        let t = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let flatten = Flatten::new(2, 0);
        assert!(matches!(flatten.forward(&t), Err(Error::ShapeError(_))));
    }
}
