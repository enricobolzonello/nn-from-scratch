use std::fs::File;

use arrow_array::{cast::AsArray, types::Int64Type};
use neural_network::Tensor;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub struct MnistDataset {
    pub inputs: Vec<Tensor>,
    #[allow(dead_code)]
    pub labels: Vec<u64>,
}

impl MnistDataset {
    pub fn from_parquet(path: &str) -> Self {
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
