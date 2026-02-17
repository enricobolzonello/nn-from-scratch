use crate::tensor::Tensor;

pub struct Sample<L> {
    pub input: Tensor,
    pub label: L,
}

pub trait Dataset<L> {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Sample<L>;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
