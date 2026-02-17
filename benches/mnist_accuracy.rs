mod utils;

use neural_network::Network;
use utils::MnistDataset;

const MNIST_MODEL_PATH: &str = "./data/mnist/mnist_ffn.onnx";
const MNIST_DATASET_PATH: &str = "./data/mnist/test-00000-of-00001.parquet";

fn main() {
    let network = Network::from_onnx(MNIST_MODEL_PATH).expect("failed to load model");
    let dataset = MnistDataset::from_parquet(MNIST_DATASET_PATH);

    let mut correct = 0usize;
    let total = dataset.inputs.len();

    for i in 0..total {
        let output = network.forward(&dataset.inputs[i]).expect("forward pass failed");

        let prediction = output
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        if prediction == dataset.labels[i] as usize {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / total as f64 * 100.0;

    println!(
        r#"{{"mnist_accuracy":{{"accuracy":{{"value":{accuracy},"lower_value":93.0}}}}}}"#
    );
}
