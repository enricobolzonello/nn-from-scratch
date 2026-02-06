# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo build          # Build the project
cargo run            # Run the binary
cargo test           # Run all tests
cargo test tensor    # Run tests in a specific module (e.g., tensor)
```

Note: Uses Rust edition 2024, requiring nightly toolchain.

## Project Goal

Learning project to understand neural networks by building a neural network library from scratch in Rust.

## Architecture

### Tensor (`src/tensor/`)
Core data structure using flat `Vec<f32>` storage with shape/strides for multi-dimensional indexing.

- `Shape` - wrapper around `Vec<usize>` for dimensions
- `Layout` - holds shape, strides, and offset; handles index computation and transpose
- `Tensor` - data + layout; supports `get`, `set`, `transpose`, `dot`

Transpose is zero-copy (reorders strides, not data).

### Neurons (`src/neurons/`)
- `Neuron` - base struct with `Tensor` weights and `f32` bias
- `Perceptron` - wraps Neuron, implements step activation: `output = dot(inputs, weights) + bias > 0`

### Errors (`src/utils/errors.rs`)
`Result<T>` alias with `Error` enum: `ShapeMismatch`, `IndexOutOfBounds`, `InvalidAxes`.
