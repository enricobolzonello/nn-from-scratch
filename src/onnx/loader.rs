use std::{collections::HashMap, fs};

use prost::Message;

use super::{ModelProto, operators::OnnxOperators};
use crate::{
    layers::{Layer, flatten::Flatten, gemm::Gemm, relu::ReLU},
    onnx::TensorProto,
    tensor::Tensor,
    utils::errors::{Error, Result},
};

pub fn load_model(filepath: &str) -> Result<Vec<Box<dyn Layer>>> {
    let bytes = fs::read(filepath)?;
    let model_proto = ModelProto::decode(bytes.as_slice())?;

    let graph = model_proto
        .graph
        .ok_or_else(|| Error::OnnxParse("model has no graph".into()))?;

    let initalizers: HashMap<String, Tensor> = graph
        .initializer
        .into_iter()
        .map(|tp| {
            let name = tp.name().to_string();
            Ok((name, tensor_proto_to_tensor(tp)?))
        })
        .collect::<Result<HashMap<_, _>>>()?;

    let mut layers: Vec<Box<dyn Layer>> = Vec::new();
    for node in graph.node {
        match node.op_type().parse::<OnnxOperators>()? {
            OnnxOperators::Gemm => {
                let b_name = node
                    .input
                    .get(1)
                    .ok_or_else(|| Error::OnnxParse("Gemm node missing B input".into()))?;
                let b = initalizers
                    .get(b_name)
                    .ok_or_else(|| Error::OnnxParse(format!("initializer not found: {b_name}")))?
                    .clone();

                let c = node
                    .input
                    .get(2)
                    .and_then(|name| initalizers.get(name))
                    .cloned();

                let alpha = get_float_attr(&node.attribute, "alpha").unwrap_or(1.0);
                let beta = get_float_attr(&node.attribute, "beta").unwrap_or(1.0);
                let trans_a = get_int_attr(&node.attribute, "transA").unwrap_or(0) != 0;
                let trans_b = get_int_attr(&node.attribute, "transB").unwrap_or(0) != 0;

                layers.push(Box::new(Gemm::new(b, c, alpha, beta, trans_a, trans_b)));
            }
            OnnxOperators::Flatten => {
                let axis = get_int_attr(&node.attribute, "axis")
                    .ok_or(Error::OnnxParse("attribute \"node\" not found".into()))?;

                layers.push(Box::new(Flatten::new(Some(axis), None)));
            }
            OnnxOperators::ReLU => {
                layers.push(Box::new(ReLU::new()));
            }
        }
    }

    Ok(layers)
}

fn get_float_attr(attrs: &[super::AttributeProto], name: &str) -> Option<f32> {
    attrs
        .iter()
        .find(|a| a.name.as_deref() == Some(name))
        .and_then(|a| a.f)
}

fn get_int_attr(attrs: &[super::AttributeProto], name: &str) -> Option<i64> {
    attrs
        .iter()
        .find(|a| a.name.as_deref() == Some(name))
        .and_then(|a| a.i)
}

fn tensor_proto_to_tensor(tp: TensorProto) -> Result<Tensor> {
    let shape: Vec<usize> = tp.dims.iter().map(|&d| d as usize).collect();

    let data = if !tp.float_data.is_empty() {
        tp.float_data
    } else if tp.raw_data.as_ref().is_some_and(|r| !r.is_empty()) {
        tp.raw_data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    } else {
        return Err(Error::OnnxParse(format!(
            "tensor {:?} has no float_data or raw_data",
            tp.name
        )));
    };

    Tensor::new(shape, data)
}
