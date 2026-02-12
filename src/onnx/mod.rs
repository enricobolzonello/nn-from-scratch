include!(concat!(env!("OUT_DIR"), "/onnx.rs"));

pub(crate) mod loader;
pub(crate) mod operators;
