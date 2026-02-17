use std::{env, fs::File};

use arrow_array::{cast::AsArray, types::Int64Type};
use indicatif::{ProgressBar, ProgressStyle};
use neural_network::{Dataset, Network, Sample, tensor::Tensor};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

struct MnistDataset {
    inputs: Vec<Tensor>,
    labels: Vec<u64>,
}

impl MnistDataset {
    fn from_parquet(path: &str) -> Self {
        let file = File::open(path).expect("could not open parquet file");
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(file).expect("failed to read parquet file");
        let reader = builder.build().expect("failed to build parquet reader");

        let mut inputs = Vec::new();
        let mut labels = Vec::new();

        for batch in reader {
            let batch = batch.expect("failed to read record batch");

            let image_col = batch
                .column_by_name("image")
                .expect("missing 'image' column");
            let image_struct = image_col
                .as_struct_opt()
                .expect("'image' column is not a struct");
            let bytes_col = image_struct
                .column_by_name("bytes")
                .expect("missing 'bytes' sub-column");
            let bytes_array = bytes_col
                .as_binary_opt::<i32>()
                .expect("'bytes' is not a binary array");

            let label_col = batch
                .column_by_name("label")
                .expect("missing 'label' column");
            let label_array = label_col.as_primitive::<Int64Type>();

            for i in 0..batch.num_rows() {
                let png_bytes = bytes_array.value(i);
                let img = image::load_from_memory(png_bytes)
                    .expect("failed to decode image")
                    .into_luma8();

                let pixels: Vec<f32> = img.as_raw().iter().map(|&b| b as f32 / 255.0).collect();
                let input = Tensor::new([1, 784], pixels).expect("failed to create input tensor");

                inputs.push(input);
                labels.push(label_array.value(i) as u64);
            }
        }

        Self { inputs, labels }
    }
}

impl Dataset<u64> for MnistDataset {
    fn len(&self) -> usize {
        self.inputs.len()
    }

    fn get(&self, index: usize) -> Sample<u64> {
        Sample {
            input: self.inputs[index].clone(),
            label: self.labels[index],
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: neural-network <test.parquet> <model.onnx>");
        std::process::exit(1);
    }

    let network = Network::from_onnx(&args[2]).expect("failed to load model");
    println!("Loaded model with {} layers", network.layer_count());

    let dataset = MnistDataset::from_parquet(&args[1]);
    println!("Loaded {} test samples", dataset.len());

    let mut correct = 0usize;
    let total = dataset.len();

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40} {pos}/{len} ({percent}%) {msg}")
            .unwrap(),
    );

    for i in 0..total {
        pb.inc(1);
        let sample = dataset.get(i);
        let output = network.forward(&sample.input).expect("forward pass failed");

        let prediction = output
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        if prediction == sample.label as usize {
            correct += 1;
        }
    }

    pb.finish_with_message(format!(
        "Accuracy: {correct}/{total} ({:.2}%)",
        correct as f64 / total as f64 * 100.0
    ));
}
