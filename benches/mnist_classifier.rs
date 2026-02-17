mod utils;

use criterion::{Criterion, criterion_group, criterion_main};
use neural_network::Network;
use utils::MnistDataset;

const MNIST_MODEL_PATH: &str = "./data/mnist/mnist_ffn.onnx";
const MNIST_DATASET_PATH: &str = "./data/mnist/test-00000-of-00001.parquet";

pub fn mnist_load_model(c: &mut Criterion) {
    c.bench_function("mnist_load", |b| {
        b.iter(|| Network::from_onnx(std::hint::black_box(MNIST_MODEL_PATH)));
    });
}

pub fn mnist_classfier_inference(c: &mut Criterion) {
    let network = Network::from_onnx(MNIST_MODEL_PATH).expect("failed to load model");
    let dataset = MnistDataset::from_parquet(MNIST_DATASET_PATH);
    let input = &dataset.inputs[0];

    c.bench_function("mnist_inference", |b| {
        b.iter(|| network.forward(std::hint::black_box(input)))
    });
}

criterion_group!(benches, mnist_load_model, mnist_classfier_inference);
criterion_main!(benches);
