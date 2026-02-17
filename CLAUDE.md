# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo build          # Build the project
cargo run -- data/mnist/test-00000-of-00001.parquet data/mnist/mnist_ffn.onnx  # Run MNIST inference
cargo test           # Run all tests
cargo test tensor    # Run tests in a specific module (e.g., tensor)
```

Note: Uses Rust edition 2024, requiring nightly toolchain.

## Project Goal

Learning project to understand neural networks by building a neural network library from scratch in Rust.

## Architecture

### Tensor (`src/tensor/`)
Core data structure using `Arc<Vec<f32>>` storage with shape/strides for multi-dimensional indexing. Arc enables cheap clones (refcount bump) with copy-on-write semantics via `Arc::make_mut` on mutation.

- `Shape` - wrapper around `Vec<usize>` for dimensions
- `Layout` - holds shape, strides, and offset; handles index computation and transpose
- `Tensor` - data + layout; supports `get`, `set`, `transpose`, `dot`, `matmul`

Transpose is in-place and zero-copy (reorders strides/shape, not data).
`matmul` handles 2D@2D, 2D@1D, and 1D@2D via automatic promotion/squeezing.

### Layers (`src/layers/`)
- `Layer` trait - defines `forward` and `backward`
- `Gemm` (private) - ONNX Gemm operator: `Y = alpha * A' @ B' + beta * C` with transA/transB and C broadcasting
- `Linear` - fully connected layer, wraps Gemm with alpha=1, beta=1: `y = input @ weights + bias`

### Network (`src/lib.rs`)
- `Network` - holds `Vec<Box<dyn Layer>>`, runs sequential forward pass
- `NetworkBuilder` - builder pattern for constructing networks
- `Network::from_onnx` - loads a network from an ONNX file
- Re-exports `Dataset` and `Sample` from `utils::data`

### Data (`src/utils/data/`)
- `Sample<L>` - holds an `input: Tensor` and a `label: L`
- `Dataset<L>` trait - defines `len()`, `get(index)`, `is_empty()`

### ONNX (`src/onnx/`)
- Generated protobuf types from `bin/onnx.proto` via `prost`
- `loader` - parses ONNX ModelProto into layers

### Neurons (`src/neurons/`)
- `Neuron` - base struct with `Tensor` weights and `f32` bias
- `Perceptron` - wraps Neuron, implements step activation: `output = dot(inputs, weights) + bias > 0`

### Binary (`src/main.rs`)
- `MnistDataset` - implements `Dataset<u64>`, loads MNIST from parquet (PNG-encoded images decoded via `image` crate)
- Eval loop: loads ONNX model, runs forward pass per sample, computes argmax, reports accuracy

### Errors (`src/utils/errors.rs`)
`Result<T>` alias with `Error` enum: `ShapeMismatch`, `IndexOutOfBounds`, `InvalidAxes`, `UnsupportedOp`, `OnnxParse`.
