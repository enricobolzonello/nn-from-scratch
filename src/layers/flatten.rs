use crate::{
    layers::Layer,
    tensor::Tensor,
    utils::errors::{Error, Result},
};

pub struct Flatten {
    start_dim: i64,
    end_dim: i64,
}

impl Flatten {
    pub(crate) fn new(start_dim: Option<i64>, end_dim: Option<i64>) -> Self {
        Self {
            start_dim: start_dim.unwrap_or(1),
            end_dim: end_dim.unwrap_or(-1),
        }
    }
}

impl Layer for Flatten {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let start = if self.start_dim < 0 {
            input.ndim() as i64 + self.start_dim
        } else {
            self.start_dim
        } as usize;
        let end = if self.end_dim < 0 {
            input.ndim() as i64 + self.end_dim
        } else {
            self.end_dim
        } as usize;

        if start >= input.ndim() || end >= input.ndim() {
            return Err(Error::ShapeError(format!(
                "flatten dims [{}, {}] out of range for {}D tensor",
                self.start_dim,
                self.end_dim,
                input.ndim()
            )));
        }

        if start > end {
            return Err(Error::ShapeError(format!(
                "start_dim needs to be < than end_dim, got start_dim {} and end_dim {}",
                self.start_dim, self.end_dim
            )));
        }

        input.flatten(start, end)
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
        let flatten = Flatten::new(Some(0), Some(2));
        let result = flatten.forward(&t).unwrap();

        assert_eq!(result.shape(), &[8]);
        assert_eq!(result.data(), t.data());
    }

    #[test]
    fn test_flatten_keep_batch() {
        // [2, 2, 2] -> [2, 4] (flatten dims 1..=2)
        let t = Tensor::new([2, 2, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let flatten = Flatten::new(Some(1), Some(2));
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
        let flatten = Flatten::new(Some(1), Some(2));
        let result = flatten.forward(&t).unwrap();

        assert_eq!(result.shape(), &[2, 12, 5]);
    }

    #[test]
    fn test_flatten_single_dim() {
        // [2, 3] with start_dim == end_dim -> no change
        let t = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let flatten = Flatten::new(Some(0), Some(0));
        let result = flatten.forward(&t).unwrap();

        assert_eq!(result.shape(), &[2, 3]);
    }

    #[test]
    fn test_flatten_negative_dims() {
        // [2, 3, 4] with default start_dim=1, end_dim=-1 -> [2, 12]
        let t = Tensor::fill([2, 3, 4], 1.0);
        let flatten = Flatten::new(None, None);
        let result = flatten.forward(&t).unwrap();

        assert_eq!(result.shape(), &[2, 12]);
    }

    #[test]
    fn test_flatten_negative_end_dim() {
        // [2, 3, 4, 5] with start_dim=1, end_dim=-2 -> [2, 12, 5]
        let t = Tensor::fill([2, 3, 4, 5], 1.0);
        let flatten = Flatten::new(Some(1), Some(-2));
        let result = flatten.forward(&t).unwrap();

        assert_eq!(result.shape(), &[2, 12, 5]);
    }

    #[test]
    fn test_flatten_out_of_range() {
        let t = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let flatten = Flatten::new(Some(0), Some(5));
        assert!(matches!(flatten.forward(&t), Err(Error::ShapeError(_))));
    }

    #[test]
    fn test_flatten_start_greater_than_end() {
        let t = Tensor::new([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let flatten = Flatten::new(Some(2), Some(0));
        assert!(matches!(flatten.forward(&t), Err(Error::ShapeError(_))));
    }
}
